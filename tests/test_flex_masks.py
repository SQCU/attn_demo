# flex_masks.py -- the generic block-mask primitives. no model, no layer, no attention.
#
# THE CONTRACT under test (flex_masks.py states it at the top):
#   full   must be a SUBSET   of truly-all-allowed blocks   <- unsafe to over-claim
#   live   must be a SUPERSET of any-allowed blocks         <- unsafe to under-claim
#   demoting full -> partial is always safe (costs speed, never correctness)
#
# every assertion here is one of those two directions, checked against a dense reference
# built IN THE TEST at tiny shapes. the production path never builds one -- that is what
# lint_mask_traffic.py enforces -- but a test is exactly where a dense reference belongs.

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flex_masks as fm  # noqa: E402
from torch.nn.attention.flex_attention import BlockMask  # noqa: E402
import torch.nn.attention.flex_attention as F  # noqa: E402

BLK = 32          # small tiles so the tests stay fast and the shapes stay readable


def dense_from_mod(mod, B, Q, KV):
    b = torch.arange(B)[:, None, None]
    q = torch.arange(Q)[None, :, None]
    kv = torch.arange(KV)[None, None, :]
    return mod(b, torch.zeros_like(b), q, kv).expand(B, Q, KV)


def true_occupancy(dense, block=BLK):
    """the EXACT block occupancy, for the test only."""
    B, Q, KV = dense.shape
    v = dense.view(B, Q // block, block, KV // block, block)
    return v.any(4).any(2), v.all(4).all(2)


def assert_contract(occ, dense, label, block=BLK):
    """the two safety directions, plus disjointness and exact reconstruction."""
    any_live, all_live = true_occupancy(dense, block)
    part = occ.partial[:, 0]
    full = occ.full[:, 0]
    assert not bool((part & full).any()), f"{label}: a block is listed partial AND full"
    # UNSAFE DIRECTION: every claimed-full block must really be all-allowed.
    over = full & ~all_live
    assert not bool(over.any()), (
        f"{label}: {int(over.sum())} block(s) claimed FULL that are not all-allowed. "
        f"flex skips mask_mod inside full blocks, so this silently attends forbidden keys.")
    # UNSAFE DIRECTION: nothing allowed may fall outside the listed blocks.
    missed = any_live & ~(part | full)
    assert not bool(missed.any()), (
        f"{label}: {int(missed.sum())} any-allowed block(s) not listed -- attention dropped")
    # exact reconstruction, which is what flex actually computes
    def expand(x):
        return x.repeat_interleave(block, 1).repeat_interleave(block, 2)
    recon = expand(full) | (expand(part) & dense)
    assert torch.equal(recon, dense), f"{label}: reconstruction != reference"


def over_listing(occ, dense, block=BLK):
    """how many blocks are listed live but hold nothing -- pure wasted kernel work,
    which the contract permits. reported so the conservatism stays visible."""
    any_live, _ = true_occupancy(dense, block)
    return int((occ.live[:, 0] & ~any_live).sum()), int(any_live.sum())


# ---------------------------------------------------------------------------------
# prefix / causal family
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_prefix_blocks_contract(seed):
    torch.manual_seed(seed)
    B, Q, KV = 3, 4 * BLK, 6 * BLK
    bound = torch.randint(0, KV, (B, Q)).sort(-1).values
    valid = torch.ones(B, KV, dtype=torch.bool)
    valid[:, 0] = False
    for b in range(B):
        valid[b, KV - 1 - seed * 3 - b * 7:] = False        # pad boundary walks
    ka, kal = fm.key_block_validity(valid, BLK)
    lo, hi = fm.qblock_bounds(bound, BLK, monotonic=True)
    occ = fm.prefix_blocks(lo, hi, KV // BLK, BLK, ka, kal)
    dense = dense_from_mod(fm.prefix_mask_mod(bound, valid), B, Q, KV)
    assert_contract(occ, dense, f"prefix seed={seed}")


def test_prefix_full_blocks_never_cross_the_pad_boundary():
    """builder v1's bug: a block straddling the padding boundary declared full, so flex
    skipped mask_mod inside it and attended padding."""
    B, Q, KV = 1, 2 * BLK, 4 * BLK
    bound = torch.full((B, Q), KV - 1)
    valid = torch.zeros(B, KV, dtype=torch.bool)
    valid[:, 1:2 * BLK + BLK // 3] = True        # padding starts mid block 2
    ka, kal = fm.key_block_validity(valid, BLK)
    lo, hi = fm.qblock_bounds(bound, BLK, monotonic=True)
    occ = fm.prefix_blocks(lo, hi, KV // BLK, BLK, ka, kal)
    assert not bool(occ.full[0, 0, 0, 2]), "block across the pad boundary claimed full"
    assert bool(occ.partial[0, 0, 0, 2])
    assert not bool(occ.full[0, 0, 0, 0]), "block 0 holds the pad at key 0; not full"
    assert bool(occ.full[0, 0, 0, 1]), "block 1 is entirely real and reachable: should be full"
    assert not bool(occ.live[0, 0, 0, 3]), "all-padding block listed"
    dense = dense_from_mod(fm.prefix_mask_mod(bound, valid), B, Q, KV)
    assert_contract(occ, dense, "pad-boundary")


def test_prefix_refuses_full_without_all_valid_information():
    """if you hand over key_any but no key_all, the builder must not guess: no full
    blocks at all. safe, and it is the direction that costs only speed."""
    B, Q, KV = 1, BLK, 2 * BLK
    bound = torch.full((B, Q), KV - 1)
    valid = torch.ones(B, KV, dtype=torch.bool)
    ka, _ = fm.key_block_validity(valid, BLK)
    occ = fm.prefix_blocks(*fm.qblock_bounds(bound, BLK, True), KV // BLK, BLK, ka, None)
    assert not bool(occ.full.any())
    assert bool(occ.partial.all())


def test_qblock_bounds_monotonic_gather_equals_reduction():
    torch.manual_seed(0)
    bound = torch.randint(0, 500, (4, 8 * BLK)).sort(-1).values
    g = fm.qblock_bounds(bound, BLK, monotonic=True)
    r = fm.qblock_bounds(bound, BLK, monotonic=False)
    assert torch.equal(g[0], r[0]) and torch.equal(g[1], r[1])


def test_non_monotonic_bounds_still_satisfy_the_contract():
    """monotonic=False must be used when the bound is unsorted; the gather form would
    under-claim. assert the reduction form is correct there."""
    torch.manual_seed(3)
    B, Q, KV = 2, 3 * BLK, 4 * BLK
    bound = torch.randint(0, KV, (B, Q))                 # deliberately unsorted
    lo, hi = fm.qblock_bounds(bound, BLK, monotonic=False)
    occ = fm.prefix_blocks(lo, hi, KV // BLK, BLK)
    assert_contract(occ, dense_from_mod(fm.prefix_mask_mod(bound), B, Q, KV), "unsorted")


# ---------------------------------------------------------------------------------
# sliding-window family
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("window", [4, BLK // 2, BLK, 3 * BLK])
def test_window_blocks_contract(window):
    torch.manual_seed(window)
    B, Q, KV = 2, 4 * BLK, 6 * BLK
    bound = torch.randint(0, KV, (B, Q)).sort(-1).values
    valid = torch.ones(B, KV, dtype=torch.bool)
    valid[:, 0] = False
    ka, _ = fm.key_block_validity(valid, BLK)
    lo, hi = fm.qblock_bounds(bound, BLK, monotonic=True)
    occ = fm.window_blocks(lo, hi, window, KV // BLK, BLK, ka)
    dense = dense_from_mod(fm.window_mask_mod(bound, window, valid), B, Q, KV)
    assert_contract(occ, dense, f"window W={window}")
    assert not bool(occ.full.any()), "windows must not claim full blocks by default"


def test_gapped_window_union_is_a_legal_superset_not_a_bug():
    """builder v2 'failed' because the per-q-block union of windows is non-contiguous
    once the bounds are spaced wider than W, so its min/max hull over-listed. under the
    contract that is CORRECT -- a superset is exactly what partial must be, and mask_mod
    removes the gaps in-kernel. assert the gaps are real, the hull covers them, and the
    reconstruction is still exact."""
    B, K, W = 1, 4, 8
    A = 8 * BLK // K
    bound = (5 + 11 * torch.arange(A)).repeat_interleave(K)[None]   # spacing 11 > W=8
    Q = bound.size(1)
    KV = 8 * BLK
    valid = torch.ones(B, KV, dtype=torch.bool)
    valid[:, 0] = False
    ka, _ = fm.key_block_validity(valid, BLK)
    lo, hi = fm.qblock_bounds(bound, BLK, monotonic=True)
    occ = fm.window_blocks(lo, hi, W, KV // BLK, BLK, ka)
    dense = dense_from_mod(fm.window_mask_mod(bound, W, valid), B, Q, KV)
    assert_contract(occ, dense, "gapped-window")
    # the union really is gapped: coverage inside the live region is sparse
    assert dense.float().mean() < 0.05
    waste, real = over_listing(occ, dense)
    print(f"\ngapped-window: {waste} over-listed blocks out of {real} truly-live "
          f"(legal: mask_mod filters them in-kernel)")


def test_wide_window_full_blocks_are_still_a_subset():
    """allow_full=True is only correct when the window really does cover a whole block
    for every query in the q block. exercise it at W >> block."""
    torch.manual_seed(1)
    B, Q, KV = 2, 2 * BLK, 6 * BLK
    bound = torch.arange(Q).repeat(B, 1) + 4 * BLK        # monotonic, far along
    valid = torch.ones(B, KV, dtype=torch.bool)
    ka, kal = fm.key_block_validity(valid, BLK)
    lo, hi = fm.qblock_bounds(bound, BLK, monotonic=True)
    occ = fm.window_blocks(lo, hi, 4 * BLK, KV // BLK, BLK, ka, allow_full=True, key_all=kal)
    dense = dense_from_mod(fm.window_mask_mod(bound, 4 * BLK, valid), B, Q, KV)
    assert_contract(occ, dense, "wide-window")
    assert bool(occ.full.any()), "a 4-block window over aligned bounds should yield full blocks"


# ---------------------------------------------------------------------------------
# block-diagonal / group family
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("g", [1, 2, 4, BLK, BLK // 4, 3, 7, 5 * BLK])
def test_group_diagonal_contract_for_any_group_size(g):
    """group_size need NOT divide the tile size. when it does not, a group straddles a
    block edge and two kv blocks get listed; that generality is the point."""
    B, Q = 2, 4 * BLK
    occ = fm.group_diagonal_blocks(Q // BLK, g, Q // BLK, BLK, batch=B)
    dense = dense_from_mod(fm.group_mask_mod(g), B, Q, Q)
    assert_contract(occ, dense, f"group g={g}")
    assert not bool(occ.full.any())


def test_group_diagonal_with_offset_key_space():
    """the group keys can live at an offset inside a concatenated key space."""
    B, Q, off = 2, 2 * BLK, 3 * BLK
    KV = off + Q
    occ_pad = fm.group_diagonal_blocks(Q // BLK, 4, off // BLK, BLK, batch=B)  # empty prefix
    occ_grp = fm.group_diagonal_blocks(Q // BLK, 4, Q // BLK, BLK, batch=B)
    occ = fm.concat_kv_blocks(fm.BlockOccupancy(torch.zeros_like(occ_pad.partial),
                                                torch.zeros_like(occ_pad.full)), occ_grp)
    dense = dense_from_mod(fm.group_mask_mod(4, kv_offset=off), B, Q, KV)
    assert_contract(occ, dense, "offset-group")


def test_group_diagonal_when_group_does_not_divide_the_tile_lists_two_blocks():
    # g=3 does not divide BLK=32, so q-block 1 (queries 32..63) covers groups 10..21 =
    # keys 30..65, which touches kv blocks 0, 1 AND 2.
    occ = fm.group_diagonal_blocks(4, 3, 4, BLK, batch=1)
    per_q = occ.live[0, 0].sum(-1)
    assert bool((per_q >= 1).all()), "every q block must list at least its own group's keys"
    assert int(per_q.max()) > 1, "a straddling group must list every kv block it touches"
    # the span is bounded: a q block spans `block` queries plus at most one group of
    # slop at each end, so ceil((block + 2*(g-1)) / block) + 1 blocks is the ceiling.
    assert int(per_q.max()) <= (BLK + 2 * (3 - 1)) // BLK + 2
    # and when g DOES divide the tile it is exactly one perfectly aligned block
    aligned = fm.group_diagonal_blocks(4, 4, 4, BLK, batch=1)
    assert torch.equal(aligned.live[0, 0].sum(-1), torch.ones(4, dtype=torch.long))
    assert bool(aligned.live[0, 0].diagonal().all())


# ---------------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------------

def test_union_and_intersection_contract():
    torch.manual_seed(0)
    B, Q, KV = 2, 3 * BLK, 5 * BLK
    bound = torch.randint(0, KV, (B, Q)).sort(-1).values
    valid = torch.ones(B, KV, dtype=torch.bool); valid[:, 0] = False
    ka, kal = fm.key_block_validity(valid, BLK)
    lo, hi = fm.qblock_bounds(bound, BLK, monotonic=True)
    p = fm.prefix_blocks(lo, hi, KV // BLK, BLK, ka, kal)
    w = fm.window_blocks(lo, hi, BLK // 2, KV // BLK, BLK, ka)
    dp = dense_from_mod(fm.prefix_mask_mod(bound, valid), B, Q, KV)
    dw = dense_from_mod(fm.window_mask_mod(bound, BLK // 2, valid), B, Q, KV)
    assert_contract(fm.union_blocks(p, w), dp | dw, "union")
    assert_contract(fm.intersect_blocks(p, w), dp & dw, "intersect")


def test_concat_kv_composition_contract():
    """a concatenated key space: [windowed history ++ block-diagonal group keys]. this is
    the shape a block-parallel decoding layer needs -- each query attends a sliding window
    of the canonical history AND, bidirectionally, its own group of in-block peers -- and
    it is expressed here entirely in terms of the two primitives, with no layer in sight."""
    torch.manual_seed(2)
    B, K, W = 2, 4, BLK // 2
    T, Q = 4 * BLK, 2 * BLK
    bound = torch.randint(0, T, (B, Q // K)).sort(-1).values.repeat_interleave(K, dim=1)
    valid = torch.ones(B, T, dtype=torch.bool); valid[:, 0] = False; valid[:, 3 * BLK:] = False
    ka, _ = fm.key_block_validity(valid, BLK)
    lo, hi = fm.qblock_bounds(bound, BLK, monotonic=True)
    occ = fm.concat_kv_blocks(
        fm.window_blocks(lo, hi, W, T // BLK, BLK, ka),
        fm.group_diagonal_blocks(Q // BLK, K, Q // BLK, BLK, batch=B))
    mod = fm.concat_kv_mask_mod((T, Q), (fm.window_mask_mod(bound, W, valid),
                                         fm.group_mask_mod(K, kv_offset=T)))
    assert_contract(occ, dense_from_mod(mod, B, Q, T + Q), "concat")


def test_concat_mask_mod_clamps_out_of_range_subspace_indices():
    """flex evaluates every sub-mod at every kv index and only then selects, so a
    sub-mod holding a tensor sized to its own sub-space must not be indexed out of it."""
    B, Q, T, G = 1, BLK, 2 * BLK, BLK
    valid = torch.ones(B, T, dtype=torch.bool)
    mod = fm.concat_kv_mask_mod((T, G), (fm.prefix_mask_mod(torch.full((B, Q), T - 1), valid),
                                         fm.group_mask_mod(4, kv_offset=T)))
    out = dense_from_mod(mod, B, Q, T + G)          # would IndexError without the clamp
    assert out.shape == (B, Q, T + G)
    assert bool(out[:, :, :T].all())


# ---------------------------------------------------------------------------------
# BlockMask construction
# ---------------------------------------------------------------------------------

def test_transpose_packing_matches_torch_transpose_ordered():
    """build_block_mask computes flex's q-orientation lists as the packing of the
    transposed occupancy instead of calling _transpose_ordered, which round-trips the
    CSR back through a block-level dense tensor. assert they are bit-identical."""
    torch.manual_seed(0)
    occ = torch.rand(3, 1, 5, 7) > 0.4
    kn, ki = fm.pack_occupancy(occ)
    tqn, tqi = F._transpose_ordered(kn, ki)
    mqn, mqi = fm.pack_occupancy(occ.transpose(-1, -2))
    assert torch.equal(tqn, mqn) and torch.equal(tqi, mqi)


def test_build_block_mask_agrees_with_from_kv_blocks():
    torch.manual_seed(1)
    part = torch.rand(2, 1, 4, 6) > 0.5
    full = (torch.rand(2, 1, 4, 6) > 0.8) & ~part
    occ = fm.BlockOccupancy(part, full)
    mine = fm.build_block_mask(occ, fm.group_mask_mod(4), BLK)
    kn, ki = fm.pack_occupancy(part)
    fn, fi = fm.pack_occupancy(full)
    theirs = BlockMask.from_kv_blocks(kn, ki, fn, fi, BLOCK_SIZE=BLK,
                                      mask_mod=fm.group_mask_mod(4))
    for a in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices",
              "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"):
        assert torch.equal(getattr(mine, a), getattr(theirs, a)), a
    assert torch.equal(mine.to_dense(), theirs.to_dense())


def test_block_mask_construction_has_zero_dynamo_graph_breaks():
    """the point of building BlockMask directly rather than via create_block_mask: the
    mask build does not need an eager island at all. create_block_mask is untraceable in
    2.5.1 (inspect.signature); direct construction traces clean.

    NOTE: this asserts DYNAMO tracing only. inductor's flex_attention lowering is
    CUDA-only in torch 2.5.1 ('Torch not compiled with CUDA enabled'), so whether the
    lowered kernel is actually produced cannot be checked on a cpu box; that requires a
    cuda run and is out of scope for this suite."""
    import torch._dynamo as dyn
    torch.manual_seed(0)
    part = torch.rand(2, 1, 4, 6) > 0.5
    full = torch.zeros_like(part)

    def build(p, f):
        return fm.build_block_mask(fm.BlockOccupancy(p, f), fm.group_mask_mod(4), BLK)

    e = dyn.explain(build)(part, full)
    assert e.graph_break_count == 0, e.break_reasons


def test_assert_block_aligned_names_the_offending_length():
    fm.assert_block_aligned(BLK, a=2 * BLK, b=3 * BLK)
    with pytest.raises(AssertionError, match="not multiples of it.*ragged"):
        fm.assert_block_aligned(BLK, ragged=BLK + 1)


def test_ragged_lengths_are_refused_not_silently_truncated():
    with pytest.raises(AssertionError, match="not a multiple"):
        fm.key_block_validity(torch.ones(1, BLK + 3, dtype=torch.bool), BLK)
    with pytest.raises(AssertionError, match="not a multiple"):
        fm.qblock_bounds(torch.zeros(1, BLK + 1, dtype=torch.long), BLK)


# ---------------------------------------------------------------------------------
# the traffic claim
# ---------------------------------------------------------------------------------

def test_mask_traffic_report_matches_the_readme_arithmetic():
    """the numbers the readme quotes, computed rather than asserted by hand."""
    r = fm.mask_traffic_report(batch=128, q_len=640, kv_len=512, block=128, heads=12)
    assert r["n_q_blocks"] == 5 and r["n_kv_blocks"] == 4
    assert r["dense_mask_elems"] == 128 * 12 * 640 * 512 == 503_316_480
    assert r["block_occupancy_elems"] == 2 * 128 * 5 * 4 == 5120
    assert r["block_csr_bytes"] == 50_176
    assert r["block_total_bytes"] == 55_296
    # 503.3 MB vs 55.3 KB. the asymptotic ratio against the occupancy alone is
    # BLOCK^2 * H / 2 = 128*128*12/2 = 98304; the CSR is what costs the rest.
    assert r["dense_mask_elems"] / r["block_occupancy_elems"] == 128 * 128 * 12 / 2
    assert round(r["ratio_dense_over_block"]) == 9102
    print(f"\nB=128 H=12 Q=640 KV=512: dense {r['dense_mask_bytes']/1e6:.1f} MB "
          f"vs block {r['block_total_bytes']/1e3:.1f} KB "
          f"({r['ratio_dense_over_block']:.0f}x)")
