# the two generic lints, and their planted violations.
#
# neither of these costs the training loop anything. both protect against a mistake that
# is made in SOURCE, at edit time, so both are checked in source at interpreter time:
#   * lint_attention_dtypes -- the 16-bits-per-activation guarantee. a silent upcast on
#     one arm of a comparison turns an architecture result into a precision result.
#   * lint_mask_traffic     -- the ban on materializing a dense [B,H,Q,KV] attention
#     mask on the training path. at B=128 H=12 Q=640 KV=512 that is 503,316,480 bools
#     = 503.3 MB per layer per step, against 55.3 KB for the block-level description.
#
# a lint that finds nothing because it is broken passes every test, so each has a
# planted-violation case.

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lint_attention_dtypes as dtl          # noqa: E402
import lint_mask_traffic as mtl              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------
# dtype widening
# ---------------------------------------------------------------------------------

def test_no_unjustified_dtype_widening_in_the_attention_path():
    unjustified, _ = dtl.check(root=ROOT)
    assert not unjustified, (
        "new dtype-widening site(s) with no recorded numerical justification:\n  "
        + "\n  ".join(f"{f} in {s}: {d}" for f, s, d in unjustified) +
        "\n\nif it is necessary, add it to lint_allowlist_attention_dtypes.json with the "
        "NUMERICAL reason and put the same reason in a comment at the site. if it is not, "
        "it is a silent upcast masquerading as an architecture result.")


def test_allowlist_has_no_stale_entries():
    """an entry matching nothing is a licence nobody is using; it must go, or the list
    stops meaning anything."""
    _, stale = dtl.check(root=ROOT)
    assert not stale, f"stale allowlist entries: {stale}"


def test_allowlist_is_data_and_covers_only_this_repos_files():
    """the allowlist is json, not code, so a downstream consumer can ship an ADDITIONAL
    file and pass both paths rather than editing this one. that only works if the base
    file stands alone and names only files this repo actually ships."""
    base = dtl.load_allowlist(("lint_allowlist_attention_dtypes.json",), ROOT)
    assert base, "base allowlist is empty"
    assert dtl.load_allowlist(root=ROOT) == base, "the default allowlist set is the base"
    assert {f for f, _, _ in base} == set(dtl.DEFAULT_FILES) == {"pgptlformer.py"}
    for why in base.values():
        assert len(why) > 20, "every allowlist entry needs a real justification"
        # a justification with no number in it is an opinion.
        assert any(ch.isdigit() for ch in why), why


def test_a_missing_extension_allowlist_is_skipped_not_fatal():
    """load_allowlist must tolerate a path that does not exist, or the extension
    mechanism above is unusable."""
    got = dtl.load_allowlist(("lint_allowlist_attention_dtypes.json",
                              "lint_allowlist_that_does_not_exist.json"), ROOT)
    assert got == dtl.load_allowlist(("lint_allowlist_attention_dtypes.json",), ROOT)


def test_no_autocast_disabled_regions_survive():
    """an arm that runs QK^T, softmax and PV inside autocast(enabled=False) + .float()
    while its counterpart runs bf16 is not comparable to it, at any wall-clock."""
    disabled = [f for f in dtl.scan(root=ROOT) if f["kind"] == "autocast_disabled"]
    assert not disabled, (
        "autocast(enabled=False) region(s) found -- everything inside runs at full "
        "parameter precision, which breaks dtype parity between arms:\n  " +
        "\n  ".join(f"{f['file']}:{f['line']} {f['scope']}" for f in disabled))


def test_dtype_lint_detects_a_planted_upcast(tmp_path):
    p = tmp_path / "planted.py"
    p.write_text(
        "import torch\n"
        "class M:\n"
        "    def attn(self, q, k, v):\n"
        "        with torch.autocast(device_type='cuda', enabled=False):\n"
        "            s = torch.matmul(q.float(), k.to(torch.float32).transpose(-1, -2))\n"
        "        return s.type(torch.float64)\n")
    found = dtl.widening_sites(files=["planted.py"], root=str(tmp_path))
    assert {f["kind"] for f in found} == {"autocast_disabled", "cast_method", "cast_to"}
    assert all(f["scope"] == "M.attn" for f in found)
    unjustified, _ = dtl.check(files=["planted.py"], root=str(tmp_path), allowlists=())
    # .float(), .to(torch.float32), .type(torch.float64), autocast(enabled=False)
    assert len(unjustified) == 4, unjustified


# ---------------------------------------------------------------------------------
# dense mask materialization
# ---------------------------------------------------------------------------------

def test_no_dense_mask_builders_on_the_flex_path():
    found = mtl.scan_source(root=ROOT)
    assert not found, (
        "dense attention-mask materialization on the flex path:\n  " +
        "\n  ".join(f"{f['file']}:{f['line']} {f['scope']} -- {f['detail']}" for f in found))


def test_mask_lint_detects_planted_violations(tmp_path):
    p = tmp_path / "planted.py"
    p.write_text(
        "from torch.nn.attention.flex_attention import create_block_mask\n"
        "def build(mm, B, Q, KV):\n"
        "    return create_block_mask(mm, B, None, Q, KV)\n"
        "def make_mod(dense):\n"
        "    def mask_mod(b, h, q_idx, kv_idx):\n"
        "        return dense[b, q_idx, kv_idx]\n"
        "    return mask_mod\n")
    found = mtl.scan_source(files=["planted.py"], root=str(tmp_path))
    kinds = {f["kind"] for f in found}
    assert kinds == {"dense_mask_builder", "mask_mod_reads_dense"}, kinds
    scopes = {f["scope"] for f in found}
    assert scopes == {"build", "make_mod.mask_mod"}, scopes


def test_allocation_audit_catches_a_dense_mask_however_it_is_spelled():
    """the static half cannot see through helpers; the dynamic half does not care how
    the tensor was built."""
    B, Q, KV = 2, 256, 256

    def sneaky(bound, valid):
        # spelled with no banned identifier anywhere
        j = torch.arange(KV)[None, None, :]
        return (j <= bound[:, :, None]) & valid[:, None, :]

    bound = torch.arange(Q).repeat(B, 1)
    valid = torch.ones(B, KV, dtype=torch.bool)
    with pytest.raises(mtl.DenseMaskViolation, match="over the .* budget"):
        with mtl.AllocationAudit(max_elems=B * Q, raise_on_violation=True):
            sneaky(bound, valid)


def test_allocation_audit_passes_a_block_scale_builder():
    import flex_masks as fm
    B, Q, KV, blk = 4, 640, 512, 128
    bound = torch.arange(Q).repeat(B, 1)
    valid = torch.ones(B, KV, dtype=torch.bool)
    with mtl.AllocationAudit(max_elems=B * Q) as audit:
        ka, kal = fm.key_block_validity(valid, blk)
        lo, hi = fm.qblock_bounds(bound, blk, monotonic=True)
        occ = fm.prefix_blocks(lo, hi, KV // blk, blk, ka, kal)
        fm.build_block_mask(occ, fm.prefix_mask_mod(bound, valid), blk)
    assert audit.largest <= B * Q, audit.report()
    assert not audit.violations
    print(f"\nblock-scale builder: largest tensor {audit.worst[1]} = {audit.largest} "
          f"elements; a dense mask would be {B * Q * KV}")


def test_audit_report_names_the_offender():
    with mtl.AllocationAudit(max_elems=4) as audit:
        torch.zeros(3, 9)
    assert "27" in audit.report() and "(3, 9)" in audit.report()
