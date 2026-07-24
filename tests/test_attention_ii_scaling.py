"""the S-scaling measurement, as an assertion.

tools/measure_attention_ii.py prints the table; this pins the three conclusions the table
supports, so they cannot quietly stop being true.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))

from measure_attention_ii import make_block, rms, split_terms


def ratios(**overrides):
    block = make_block(**overrides)
    block.eval()
    with torch.no_grad():
        out = {}
        for s in (64, 128, 256, 512, 1024):
            soft, ii = split_terms(block, s)
            out[s] = rms(ii) / rms(soft)
    return out


def test_as_shipped_the_attention_ii_term_grows_without_bound_in_S():
    r = ratios()
    assert r[64] > 10        # already an order of magnitude larger at the SHORTEST length
    assert r[1024] / r[64] > 10
    # monotone, not noise
    keys = sorted(r)
    assert all(r[b] > r[a] for a, b in zip(keys, keys[1:]))


def test_the_sigmoid_gate_alone_does_not_make_it_commensurable():
    """the central negative result. a sigmoid multiplies by something in (0,1); it can
    shrink the term but it cannot cancel an S-dependence. and because the gate is applied
    UNIFORMLY -- to the softmax path as well -- at init it halves both and the ratio is
    exactly unchanged."""
    plain = ratios()
    gated = ratios(attn_gate="sigmoid")
    for s in plain:
        assert gated[s] == pytest.approx(plain[s], rel=1e-6)


def test_inv_sqrt_s_fixes_the_attention_ii_term_but_not_the_ratio():
    """1/sqrt(S) makes the attention-II magnitude S-invariant, which is the right correction
    for a sum of S zero-mean terms. the ratio still drifts, because the SOFTMAX term is
    itself shrinking like 1/sqrt(S) as it averages over more keys."""
    block = make_block(attention_deux_norm="inv_sqrt_s")
    block.eval()
    with torch.no_grad():
        mags = {s: rms(split_terms(block, s)[1]) for s in (64, 256, 1024)}
    assert max(mags.values()) / min(mags.values()) < 1.1     # term itself: flat
    r = ratios(attention_deux_norm="inv_sqrt_s")
    assert r[1024] / r[64] > 2                               # ratio: still drifting


def test_mean_row_normalization_makes_the_two_terms_commensurable():
    """dividing by the number of unmasked keys per row -- the row-exact analogue of the
    softmax denominator -- holds the ratio flat across a 16x span of sequence lengths."""
    r = ratios(attention_deux_norm="mean")
    assert 0.8 < min(r.values()) and max(r.values()) < 1.6
    assert max(r.values()) / min(r.values()) < 1.15


def test_gate_and_mean_together_leave_the_ratio_flat():
    a = ratios(attention_deux_norm="mean")
    b = ratios(attn_gate="sigmoid", attention_deux_norm="mean")
    for s in a:
        assert b[s] == pytest.approx(a[s], rel=1e-6)
