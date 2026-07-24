"""the author cares a lot about torch.compile, so every new knob is checked for graph breaks.

`fullgraph=True` is the assertion: dynamo raises if it cannot capture the whole forward in
one graph. backend='eager' because the inductor backend needs a c++ toolchain and this suite
has to run anywhere -- what is being tested is the TRACE (no graph breaks, no data-dependent
control flow reaching the graph), not codegen.
"""
import itertools

import pytest
import torch
import torch._dynamo

from conftest import tiny_config, tiny_t5_config

import pgptlformer

VARIANTS = list(itertools.product(
    ["none", "sigmoid"],                          # attn_gate
    [False, True],                                # attention_deux
    ["dynamic_shape_rmsnorm", "l2norm", "rmsnorm"],  # qknorm
))


@pytest.mark.parametrize("gate,attn_ii,qknorm", VARIANTS)
def test_ar_forward_compiles_fullgraph_and_matches_eager(gate, attn_ii, qknorm):
    torch._dynamo.reset()
    cfg = tiny_config(attn_gate=gate, attention_deux=attn_ii, qknorm=qknorm,
                      attention_deux_norm="mean" if attn_ii else "none")
    model = pgptlformer.PGPT_Lformer(cfg)
    compiled = torch.compile(model, fullgraph=True, backend="eager")

    x = torch.randint(0, 64, (2, 16))
    pm = torch.ones_like(x, dtype=torch.bool)
    got = compiled(x, targets=x, padding_mask=pm)
    ref = model(x, targets=x, padding_mask=pm)
    assert torch.allclose(got[0], ref[0], atol=1e-5)
    assert torch.allclose(got[1], ref[1], atol=1e-5)


@pytest.mark.parametrize("gate,attn_ii,qknorm", VARIANTS)
def test_t5_forward_compiles_fullgraph_and_matches_eager(gate, attn_ii, qknorm):
    torch._dynamo.reset()
    cfg = tiny_t5_config(attn_gate=gate, attention_deux=attn_ii, qknorm=qknorm,
                         attention_deux_norm="mean" if attn_ii else "none")
    model = pgptlformer.PGPT_Lformer(cfg)
    compiled = torch.compile(model, fullgraph=True, backend="eager")

    enc = torch.randint(0, 60, (2, 14))
    dec = torch.randint(0, 60, (2, 9))
    args = (enc, dec, dec, torch.ones(2, 14, dtype=torch.bool), torch.ones(2, 9, dtype=torch.bool))
    assert torch.allclose(compiled(*args)[1], model(*args)[1], atol=1e-5)


def test_gate_choice_does_not_reach_the_graph():
    """the on/off decision resolves at construction (a python attribute checked against a
    string), so it is constant-folded during tracing rather than becoming a branch."""
    torch._dynamo.reset()
    cfg = tiny_config(attn_gate="sigmoid", attention_deux=True, attention_deux_norm="mean")
    block = pgptlformer.vit22_tformer(cfg)
    compiled = torch.compile(block, fullgraph=True, backend="eager")
    x = torch.randn(2, 16, 32)
    mask = pgptlformer.create_attention_mask(torch.ones(2, 16, dtype=torch.bool), True)
    assert torch.allclose(compiled(x, attention_mask=mask), block(x, attention_mask=mask),
                          atol=1e-5)


def test_dynamic_shapes_still_work_under_compile():
    """the dynamic_shape_* norms exist because sampling changes B and T every step. compiling
    must not freeze that -- it should recompile or use dynamic shapes, never crash."""
    torch._dynamo.reset()
    model = pgptlformer.PGPT_Lformer(tiny_config(attn_gate="sigmoid"))
    compiled = torch.compile(model, backend="eager")
    for b, t in [(1, 4), (3, 9), (2, 16)]:
        x = torch.randint(0, 64, (b, t))
        out, *_ = compiled(x, padding_mask=torch.ones(b, t, dtype=torch.bool))
        assert out.shape == (b, 1, 64)
