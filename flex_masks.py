# flex_masks.py -- build flex_attention BlockMasks from cheap conservative bounds,
# without ever materializing a dense [B, H, Q, KV] mask.
#
# this module knows nothing about any particular model. it is about mask STRUCTURE:
# prefix bounds, sliding windows, block-diagonal groups, and their unions -- the shapes
# that show up in padded batching, packed multi-user prefill, prefix-bidirectional
# attention, sliding-window attention, and grouped/block-parallel decoding alike.
#
# =================================================================================
# THE CONTRACT (this is the whole reason the module can be cheap and still correct)
# =================================================================================
# a BlockMask carries two block lists, and they have OPPOSITE safety directions:
#
#   full_kv_num_blocks / full_kv_indices
#       must be a SUBSET of the truly-all-allowed blocks. flex SKIPS mask_mod entirely
#       inside a full block, so over-claiming FULL silently attends forbidden keys.
#       this is the unsafe direction. every historical bug in this repo's mask builders
#       was an over-claim here (a block straddling the padding boundary declared full).
#
#   kv_num_blocks / kv_indices  (the partial list)
#       must be a SUPERSET of every block containing any allowed element. flex evaluates
#       mask_mod inside these, so listing a block that turns out to be empty or sparse
#       costs only kernel time. under-claiming silently drops attention.
#
#   => DEMOTING A WOULD-BE-FULL BLOCK TO PARTIAL IS ALWAYS SAFE. it costs speed, never
#      correctness, because mask_mod runs inside it and filters exactly.
#
# so a builder does NOT need exact block occupancy. it needs a cheap LOWER bound on
# full and a cheap UPPER bound on live, and the mask_mod -- which runs in-kernel, on
# registers, and never touches HBM -- makes up the difference. that is what lets every
# function here be O(n_q_blocks * n_kv_blocks) instead of O(Q * KV).
#
# =================================================================================
# WHY THIS IS NOT OPTIONAL
# =================================================================================
# the obvious way to derive block occupancy is any()/all() over a materialized dense
# mask. exact, structurally impossible to get wrong -- and bandwidth-pessimized in
# precisely the way flex_attention exists to avoid. at a representative training shape
# (B=128, H=12, Q=640, KV=512) a [B,H,Q,KV] bool mask is 503,316,480 elements = 503.3 MB
# per layer per step; the block-level occupancy at H=1 and BLOCK=128 is 2*128*5*4 = 5,120
# bools, and the int32 CSR that flex actually consumes is 50,176 bytes, for 55.3 KB
# total -- a ratio of 9,102x. against the occupancy alone the ratio is the asymptotic
# BLOCK^2 * H / 2 = 98,304.
#
# mask_traffic_report() below computes those numbers rather than asserting them by hand,
# and tests/test_flex_masks.py pins them.
#
# masks here are built with H=1 and broadcast across heads. every pattern in this module
# is a function of (batch, query position, key position) only; if you need a genuinely
# head-dependent mask, pass n_heads explicitly and pay for it deliberately.
#
# lint_mask_traffic.py enforces the ban mechanically (both statically and with a
# TorchFunctionMode allocation auditor); tests/test_flex_masks.py pins the contract.

import dataclasses
from typing import Callable, Optional, Sequence

import torch
from torch.nn.attention.flex_attention import BlockMask

DEFAULT_BLOCK = 128


# ---------------------------------------------------------------------------------
# occupancy
# ---------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BlockOccupancy:
    """block-level occupancy for one mask, as bools [B, 1, n_q_blocks, n_kv_blocks].

    `full` is a subset of the truly-all-allowed blocks; `partial` is a superset of the
    remaining any-allowed blocks. they are disjoint. see THE CONTRACT above."""
    partial: torch.Tensor
    full: torch.Tensor

    def __post_init__(self):
        assert self.partial.shape == self.full.shape, (self.partial.shape, self.full.shape)
        assert self.partial.dim() == 4 and self.partial.dtype == torch.bool

    @property
    def live(self):
        return self.partial | self.full

    @property
    def n_q_blocks(self):
        return self.partial.size(2)

    @property
    def n_kv_blocks(self):
        return self.partial.size(3)


def _disjoin(partial, full):
    """partial and full must not overlap; full wins (it is the cheaper kernel path)."""
    return BlockOccupancy(partial=partial & ~full, full=full)


def key_block_validity(valid_keys, block=DEFAULT_BLOCK):
    """[B, KV] bool (True = a real key) -> (any_valid, all_valid) per kv block, each
    [B, n_kv_blocks].

    O(B * KV), which is the whole point: it is linear in the key axis, not quadratic.
    at B=128, KV=512 that is 65,536 bools. it makes no assumption about WHERE the
    padding is -- trailing, leading, interior -- so the "padding is a trailing suffix"
    convention is not load-bearing here."""
    B, KV = valid_keys.shape
    assert KV % block == 0, f"key length {KV} is not a multiple of block {block}"
    v = valid_keys.view(B, KV // block, block)
    return v.any(-1), v.all(-1)


def qblock_bounds(bound, block=DEFAULT_BLOCK, monotonic=False):
    """[B, Q] integer per-query bound -> (lo, hi) per q block, each [B, n_q_blocks].

    monotonic=True (bound non-decreasing along Q) makes this a STRIDED GATHER of the
    first and last element of each block -- O(B * n_q_blocks). otherwise it is an
    amin/amax over the query axis, O(B * Q), which at B=128, Q=640 is 81,920 elements:
    still four orders below a dense mask, but pay it only when you must."""
    B, Q = bound.shape
    assert Q % block == 0, f"query length {Q} is not a multiple of block {block}"
    n_qb = Q // block
    if monotonic:
        idx = torch.arange(n_qb, device=bound.device) * block
        return bound[:, idx], bound[:, idx + block - 1]
    b = bound.view(B, n_qb, block)
    return b.amin(-1), b.amax(-1)


def _kv_block_edges(n_kv_blocks, block, device):
    start = torch.arange(n_kv_blocks, device=device) * block
    return start[None, None, :], start[None, None, :] + block - 1


def prefix_blocks(lo, hi, n_kv_blocks, block=DEFAULT_BLOCK,
                  key_any=None, key_all=None):
    """mask: `kv_idx <= bound[q]` (and optionally `valid_keys[kv_idx]`).

    the causal / prefix-bounded family: ordinary causal attention (bound = q_idx),
    padded-batch causality, and any "each query sees a prefix whose length it names".

    lo/hi are the per-q-block min/max of `bound` from qblock_bounds().
        partial superset:  kv_block_start <= hi   (and the block holds a valid key)
        full subset:       kv_block_end   <= lo   (and every key in it is valid)
    both bounds are in fact EXACT for this family, which is why prefix masks get the
    full-block fast path and windows below do not."""
    ks, ke = _kv_block_edges(n_kv_blocks, block, lo.device)
    lo4, hi4 = lo[:, None, :, None], hi[:, None, :, None]
    partial = ks[None] <= hi4
    full = ke[None] <= lo4
    if key_any is not None:
        partial = partial & key_any[:, None, None, :]
    if key_all is not None:
        full = full & key_all[:, None, None, :]
    elif key_any is not None:
        # no all-valid information supplied: refuse to claim full anywhere, because a
        # block straddling the padding boundary is exactly the over-claim that bites.
        full = torch.zeros_like(full)
    return _disjoin(partial, full)


def window_blocks(lo, hi, window, n_kv_blocks, block=DEFAULT_BLOCK, key_any=None,
                  allow_full=False, key_all=None):
    """mask: `bound[q] - window < kv_idx <= bound[q]` -- the sliding-window family.

    the per-q-block union of windows is NOT contiguous once the bounds inside a block
    are spaced further apart than `window`. that used to be treated as a bug; under the
    contract it is not one. the convex hull [lo - window + 1, hi] is a legal SUPERSET,
    and mask_mod removes the gaps in-kernel:
        partial superset:  kv_block_start <= hi  AND  kv_block_end >= lo - window + 1

    full blocks are NOT claimed by default. a kv block is all-allowed only if it sits
    inside EVERY query's window, which needs window >= block + (hi - lo); with the usual
    window < block that is impossible, and claiming it wrongly is the unsafe direction.
    pass allow_full=True (with key_all) if you actually have wide windows and want the
    fast path; the bound used then is exact."""
    ks, ke = _kv_block_edges(n_kv_blocks, block, lo.device)
    lo4, hi4 = lo[:, None, :, None], hi[:, None, :, None]
    partial = (ks[None] <= hi4) & (ke[None] >= lo4 - window + 1)
    if key_any is not None:
        partial = partial & key_any[:, None, None, :]
    if allow_full and key_all is not None:
        # inside every window: block_start > hi - window and block_end <= lo
        full = (ks[None] > hi4 - window) & (ke[None] <= lo4) & key_all[:, None, None, :]
    else:
        full = torch.zeros_like(partial)
    return _disjoin(partial, full)


def group_diagonal_blocks(n_q_blocks, group_size, n_kv_blocks, block=DEFAULT_BLOCK,
                          batch=1, kv_offset=0, device=None):
    """mask: `q_idx // group_size == (kv_idx - kv_offset) // group_size` -- the
    block-diagonal / grouped family (each query attends bidirectionally within its own
    group of `group_size` consecutive positions).

    q block qb covers queries [qb*block, (qb+1)*block), i.e. groups
    [qb*block // g, ((qb+1)*block - 1) // g], i.e. key indices
    [g * (qb*block // g), g * (((qb+1)*block - 1)//g + 1)) + kv_offset.

    when g divides block this is exactly ONE perfectly aligned kv block per q block.
    when it does not -- and it need not; that generality is deliberate -- the group
    straddles a block edge and two kv blocks are listed. always partial: a whole kv
    block is all-allowed only if group_size >= block AND the alignment is exact, and
    claiming that is the unsafe direction for a saving of nothing."""
    device = device or torch.device("cpu")
    g = group_size
    qb = torch.arange(n_q_blocks, device=device)
    first_key = (qb * block // g) * g + kv_offset
    last_key = ((((qb + 1) * block - 1) // g) + 1) * g - 1 + kv_offset
    kb = torch.arange(n_kv_blocks, device=device) * block
    partial = (kb[None, :] <= last_key[:, None]) & (kb[None, :] + block - 1 >= first_key[:, None])
    partial = partial[None, None].expand(batch, 1, n_q_blocks, n_kv_blocks)
    return BlockOccupancy(partial=partial.contiguous(),
                          full=torch.zeros_like(partial))


def union_blocks(*occs):
    """occupancy of `A | B | ...`.

    full: a block all-allowed under ANY operand is all-allowed under the union, so the
    union of the full sets is still a subset of truly-full.
    partial: the union of every live block, minus the new full set."""
    full = occs[0].full.clone()
    live = occs[0].live.clone()
    for o in occs[1:]:
        full |= o.full
        live |= o.live
    return _disjoin(live, full)


def intersect_blocks(a, b):
    """occupancy of `A & B`. full only where both are full (still a subset); live only
    where both are live (still a superset of any-allowed)."""
    return _disjoin(a.live & b.live, a.full & b.full)


def concat_kv_blocks(*occs):
    """occupancy for a key space that is the CONCATENATION of several key spaces --
    e.g. [canonical history keys ++ in-block keys]. each operand's kv blocks keep their
    own semantics and are laid out end to end."""
    return BlockOccupancy(partial=torch.cat([o.partial for o in occs], dim=-1),
                          full=torch.cat([o.full for o in occs], dim=-1))


# ---------------------------------------------------------------------------------
# mask_mod primitives -- evaluated IN-KERNEL, inside partial blocks only
# ---------------------------------------------------------------------------------
# every one of these closes over tensors that are O(B*Q) or O(B*KV) at worst and never
# over anything O(Q*KV). a mask_mod that indexes a dense [B, Q, KV] tensor makes the
# kernel read the very thing this module exists not to build.

def prefix_mask_mod(bound, valid_keys=None):
    def mod(b, h, q_idx, kv_idx):
        m = kv_idx <= bound[b, q_idx]
        return m & valid_keys[b, kv_idx] if valid_keys is not None else m
    return mod


def window_mask_mod(bound, window, valid_keys=None):
    def mod(b, h, q_idx, kv_idx):
        t = bound[b, q_idx]
        m = (kv_idx <= t) & (kv_idx > t - window)
        return m & valid_keys[b, kv_idx] if valid_keys is not None else m
    return mod


def group_mask_mod(group_size, kv_offset=0):
    def mod(b, h, q_idx, kv_idx):
        return (q_idx // group_size) == ((kv_idx - kv_offset) // group_size)
    return mod


def union_mask_mod(*mods):
    def mod(b, h, q_idx, kv_idx):
        out = mods[0](b, h, q_idx, kv_idx)
        for m in mods[1:]:
            out = out | m(b, h, q_idx, kv_idx)
        return out
    return mod


def concat_kv_mask_mod(splits, mods):
    """one mask_mod over a concatenated key space. `splits` are the lengths of each
    sub-space in order; sub-mod i sees kv indices already offset into the whole space
    (so build it with the same kv_offset you passed to the occupancy builder)."""
    assert len(splits) == len(mods)
    bounds = []
    acc = 0
    for s in splits:
        acc += s
        bounds.append(acc)

    def mod(b, h, q_idx, kv_idx):
        out = None
        lo = 0
        for hi, m in zip(bounds, mods):
            sel = (kv_idx >= lo) & (kv_idx < hi)
            # flex evaluates the WHOLE expression at every kv index and only then
            # selects, so a sub-mod that indexes a tensor sized to its own sub-space
            # (a padding mask over the first T keys, say) would go out of bounds on the
            # keys belonging to a later sub-space. clamp into range; `sel` throws the
            # result away anyway. sub-mods still see GLOBAL indices, so build them with
            # the same kv_offset used for the occupancy.
            safe = kv_idx.clamp(lo, hi - 1)
            piece = sel & m(b, h, q_idx, safe)
            out = piece if out is None else (out | piece)
            lo = hi
        return out
    return mod


# ---------------------------------------------------------------------------------
# BlockMask construction
# ---------------------------------------------------------------------------------

def pack_occupancy(occ_bool):
    """bool [B, 1, n_a, n_b] -> flex's (num_blocks [B,1,n_a] int32,
    indices [B,1,n_a,n_b] int32), live ids front-packed."""
    num = occ_bool.sum(-1, dtype=torch.int32)
    idx = torch.argsort(occ_bool.to(torch.int32), dim=-1, descending=True, stable=True)
    return num.contiguous(), idx.to(torch.int32).contiguous()


def build_block_mask(occ, mask_mod, block=DEFAULT_BLOCK):
    """BlockOccupancy + mask_mod -> BlockMask, constructed directly.

    NOT via create_block_mask (it evaluates mask_mod over the whole Q*KV grid AND is not
    dynamo-traceable in torch 2.5.1 -- inspect.signature) and NOT via
    BlockMask.from_kv_blocks (which round-trips the CSR back to a block-level dense
    tensor via _ordered_to_dense just to transpose it, and runs per-call validation --
    by the same rule that keeps checks out of hot paths, validation belongs in the
    equivalence test, not here).

    the q-orientation lists flex's backward needs are the transpose of the kv ones, and
    the transpose of a packed occupancy is the packing of the transposed occupancy:
    tests/test_flex_masks.py asserts this is bit-identical to torch's _transpose_ordered.

    measured on torch 2.5.1: constructing BlockMask this way inside a torch.compile
    region produces ZERO dynamo graph breaks, so the mask build does not need an eager
    island at all."""
    kn, ki = pack_occupancy(occ.partial)
    qn, qi = pack_occupancy(occ.partial.transpose(-1, -2))
    fkn, fki = pack_occupancy(occ.full)
    fqn, fqi = pack_occupancy(occ.full.transpose(-1, -2))
    bs = (block, block)
    return BlockMask(kn, ki, fkn, fki, qn, qi, fqn, fqi, bs, mask_mod)


def assert_block_aligned(block=DEFAULT_BLOCK, **lengths):
    """construction-time guard. flex tiles the q and kv axes at `block`; a length that
    is not a multiple of it needs padding logic the callers here do not have. call this
    from your module's __init__ (free) rather than from forward (not free)."""
    bad = {k: v for k, v in lengths.items() if v % block}
    if bad:
        raise AssertionError(
            f"flex_attention tiles at {block}; these lengths are not multiples of it: "
            f"{bad}. pad the sequence, change the block size, or use a dense masked SDPA "
            f"path for this shape.")


def mask_traffic_report(batch, q_len, kv_len, block=DEFAULT_BLOCK, heads=1):
    """element/byte counts for the dense mask this module refuses to build vs the block
    tensors it does build. pure arithmetic -- allocates nothing, so it is safe to call
    anywhere, and it is what the readme quotes."""
    n_qb, n_kvb = -(-q_len // block), -(-kv_len // block)
    dense = batch * heads * q_len * kv_len
    occ = 2 * batch * 1 * n_qb * n_kvb                       # partial + full bools
    # int32 CSR, both orientations, partial + full
    csr = 4 * (2 * (batch * n_qb * n_kvb + batch * n_kvb * n_qb)
               + 2 * (batch * n_qb + batch * n_kvb))
    return {
        "n_q_blocks": n_qb, "n_kv_blocks": n_kvb,
        "dense_mask_elems": dense, "dense_mask_bytes": dense,      # bool = 1 byte
        "block_occupancy_elems": occ, "block_csr_bytes": csr,
        "block_total_bytes": occ + csr,
        "ratio_dense_over_block": dense / max(occ + csr, 1),
    }
