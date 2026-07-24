"""the post-attention sigmoid gate: off is free, on is sane, and it is EVERYWHERE.

the requirement that matters most is the first one. an ablation axis that changes the
numbers when it is switched off is not an ablation axis, it is a rewrite, and every
checkpoint predating it is invalidated. so: gate absent from the config vs gate="none" must
be EXACTLY equal, bit for bit, not allclose.
"""
import copy

import pytest
import torch

from conftest import ar_batch, build, tiny_config, tiny_t5_config

import pgptlformer


def _t5_batch(vocab=60, pad=60):
    g = torch.Generator().manual_seed(4)
    enc = torch.randint(0, vocab, (2, 14), generator=g)
    dec = torch.randint(0, vocab, (2, 9), generator=g)
    tgt = torch.randint(0, vocab, (2, 9), generator=g)
    return enc, dec, tgt, torch.ones(2, 14, dtype=torch.bool), torch.ones(2, 9, dtype=torch.bool)


# --- 1. OFF IS FREE ----------------------------------------------------------------------

@pytest.mark.parametrize("attention_deux", [False, True])
@pytest.mark.parametrize("qknorm", ["dynamic_shape_rmsnorm", "l2norm", "identitynorm"])
def test_gate_none_is_bit_identical_to_gate_absent_ar(attention_deux, qknorm):
    explicit = tiny_config(attention_deux=attention_deux, qknorm=qknorm, attn_gate="none")
    # the fixture states every schema key, so absence is constructed here rather than
    # inherited from it -- the property under test is "a config that never mentions the
    # gate builds the identical model", which is what every pre-gate checkpoint relies on.
    absent = {k: v for k, v in explicit.items() if k != "attn_gate"}
    assert "attn_gate" not in absent

    x, y, mask = ar_batch()
    a = build(absent)(x, targets=y, padding_mask=mask, return_zloss=True)
    b = build(explicit)(x, targets=y, padding_mask=mask, return_zloss=True)
    for lhs, rhs in zip(a, b):
        if lhs is None:
            assert rhs is None
        else:
            assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("attention_deux", [False, True])
def test_gate_none_is_bit_identical_to_gate_absent_t5(attention_deux):
    explicit = tiny_t5_config(attention_deux=attention_deux, attn_gate="none")
    absent = {k: v for k, v in explicit.items() if k != "attn_gate"}
    enc, dec, tgt, em, dm = _t5_batch()
    a = build(absent)(enc, dec, tgt, em, dm, return_zloss=True)
    b = build(explicit)(enc, dec, tgt, em, dm, return_zloss=True)
    for lhs, rhs in zip(a, b):
        if lhs is None:
            assert rhs is None
        else:
            assert torch.equal(lhs, rhs)


def test_gate_none_adds_no_parameters():
    """existing checkpoints must still load. that means identical state_dict keys."""
    for cfg in (tiny_config(attention_deux=True), tiny_t5_config(attention_deux=True)):
        before = set(build(cfg).state_dict())
        after = set(build(dict(cfg, attn_gate="none")).state_dict())
        assert before == after
        assert not any("gate" in k for k in after)


def test_existing_checkpoint_loads_into_gate_none_model():
    cfg = tiny_t5_config(attention_deux=True)
    sd = build(cfg).state_dict()
    build(dict(cfg, attn_gate="none")).load_state_dict(sd, strict=True)


def test_gated_model_rejects_an_ungated_checkpoint_loudly():
    """turning the gate ON is a real architecture change and must not load silently."""
    cfg = tiny_t5_config()
    sd = build(cfg).state_dict()
    with pytest.raises(RuntimeError, match="Missing key"):
        build(dict(cfg, attn_gate="sigmoid")).load_state_dict(sd, strict=True)


# --- 2. ON IS SANE -----------------------------------------------------------------------

def test_gate_is_present_on_every_attention_schema():
    """the whole point: if the gate existed on the softmax path but not on attention-II, an
    A/B of attention-II would be confounded by the gate itself."""
    block = build(tiny_t5_config(attention_deux=True, attn_gate="sigmoid")).decoder[0]
    for name in ("self_attn_gate", "attention_II_gate", "cross_attn_gate"):
        gate = getattr(block, name)
        assert gate is not None, name
        assert gate.out_features == block.heads, name
        assert gate.in_features == block.dim, name

    enc_block = build(tiny_t5_config(attention_deux=True, attn_gate="sigmoid")).encoder[0]
    assert enc_block.self_attn_gate is not None
    assert enc_block.attention_II_gate is not None
    assert not hasattr(enc_block, "cross_attn_gate")  # encoders have no cross-attention


@pytest.mark.parametrize("qknorm", ["dynamic_shape_rmsnorm", "l2norm", "identitynorm"])
def test_gate_reaches_the_l2norm_path_too(qknorm):
    block = build(tiny_config(qknorm=qknorm, attn_gate="sigmoid")).lambdaformer.blocks[0]
    assert block.self_attn_gate is not None


def test_gate_is_zero_init_so_sigmoid_is_exactly_one_half():
    block = build(tiny_t5_config(attention_deux=True, attn_gate="sigmoid")).decoder[0]
    for name in ("self_attn_gate", "attention_II_gate", "cross_attn_gate"):
        gate = getattr(block, name)
        assert torch.equal(gate.weight, torch.zeros_like(gate.weight)), name
        assert torch.equal(gate.bias, torch.zeros_like(gate.bias)), name
        g = torch.sigmoid(gate(torch.randn(3, 5, block.dim)))
        assert torch.equal(g, torch.full_like(g, 0.5)), name


def test_turning_the_gate_on_does_not_perturb_the_trunk_init():
    """constructing an nn.Linear draws from the global rng, so a naively-added gate would
    shift every parameter created after it and the A/B would be confounded by init noise.
    _make_gate forks the rng. at a fixed seed the two models differ ONLY by the gates."""
    cfg = tiny_config(attention_deux=True)
    plain = build(cfg).state_dict()
    gated = build(dict(cfg, attn_gate="sigmoid")).state_dict()
    for k, v in plain.items():
        assert torch.equal(v, gated[k]), k
    assert set(gated) - set(plain) == {k for k in gated if "gate" in k}


def test_gate_at_init_halves_each_attention_branch_exactly():
    """zero-init means constant 0.5, so the gated block's attention contribution is exactly
    half the ungated one. checked on the pre-out-projection attention, which is where the
    gate is applied."""
    cfg = tiny_config(attention_deux=True)
    plain = build(cfg).lambdaformer.blocks[0]
    gated = build(dict(cfg, attn_gate="sigmoid")).lambdaformer.blocks[0]
    plain.eval(); gated.eval()
    torch.manual_seed(21)
    x = torch.randn(2, 11, cfg["dim"], dtype=torch.float32)
    mask = pgptlformer.create_attention_mask(torch.ones(2, 11, dtype=torch.bool), True)

    # attnoutproj is affine: gating y by 0.5 gives 0.5*(Wy) + b, so subtract the bias first.
    b = plain.attnoutproj.bias
    got = gated.self_attn(x, mask) - b
    expected = 0.5 * (plain.self_attn(x, mask) - b)
    assert torch.allclose(got, expected, atol=1e-6), (got - expected).abs().max()


def test_gate_receives_gradient_at_init():
    """0.5 sits at the sigmoid's maximum derivative. a gate initialized to saturate at 1.0
    would be dead on arrival; this asserts ours is not."""
    block = build(tiny_config(attention_deux=True, attn_gate="sigmoid")).lambdaformer.blocks[0]
    x = torch.randn(2, 9, 32)
    mask = pgptlformer.create_attention_mask(torch.ones(2, 9, dtype=torch.bool), True)
    block.self_attn(x, mask).pow(2).sum().backward()
    for name in ("self_attn_gate", "attention_II_gate"):
        gate = getattr(block, name)
        assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0, name
        assert gate.bias.grad is not None and gate.bias.grad.abs().sum() > 0, name


def test_gate_is_per_head_and_input_dependent_once_trained():
    block = build(tiny_config(attn_gate="sigmoid")).lambdaformer.blocks[0]
    with torch.no_grad():
        block.self_attn_gate.weight.normal_(0, 1.0)
        block.self_attn_gate.bias.normal_(0, 1.0)
    x = torch.randn(2, 7, 32)
    g = torch.sigmoid(block.self_attn_gate(x))
    assert g.shape == (2, 7, block.heads)
    assert ((g > 0) & (g < 1)).all()
    # heads differ from one another, and positions differ from one another
    assert g.std(dim=-1).min() > 0
    assert g.std(dim=1).min() > 0


def test_gate_bounds_the_attention_branch():
    """a sigmoid gate can only shrink. that is the bound it provides, and it is worth
    stating in a test because it is exactly the property the S-scaling measurement shows is
    NOT sufficient on its own for attention-II."""
    block = build(tiny_config(attn_gate="sigmoid")).lambdaformer.blocks[0]
    with torch.no_grad():
        block.self_attn_gate.weight.normal_(0, 3.0)
    y = torch.randn(2, 4, 6, 8)
    x = torch.randn(2, 6, 32)
    assert (block._apply_gate(block.self_attn_gate, y, x).abs() <= y.abs() + 1e-7).all()


def test_gate_choices_are_validated():
    from config_utils import ConfigError
    with pytest.raises(ConfigError, match="attn_gate"):
        build(tiny_config(attn_gate="sigmiod"))


# --- 3. the attention-II row normalization axis -------------------------------------------

def test_attention_ii_norm_default_is_the_legacy_unnormalized_sum():
    block = build(tiny_config(attention_deux=True)).lambdaformer.blocks[0]
    assert block.attention_II_norm == "none"


def test_attention_ii_mean_normalization_divides_by_unmasked_row_count():
    q = torch.randn(1, 2, 6, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 6, 8, dtype=torch.float64)
    v = torch.ones(1, 2, 6, 8, dtype=torch.float64)
    mask = torch.tril(torch.ones(6, 6, dtype=torch.float64))[None]

    raw = pgptlformer.scaled_dot_product_attn_bias(q, k, v, attn_mask=mask, row_norm="none")
    mean = pgptlformer.scaled_dot_product_attn_bias(q, k, v, attn_mask=mask, row_norm="mean")
    counts = mask.sum(-1)[0]                     # [6] -> 1, 2, 3, ...
    assert torch.allclose(mean, raw / counts[None, None, :, None])


def test_attention_ii_inv_sqrt_s_normalization():
    q = torch.randn(1, 2, 9, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 9, 8, dtype=torch.float64)
    v = torch.ones(1, 2, 9, 8, dtype=torch.float64)
    raw = pgptlformer.scaled_dot_product_attn_bias(q, k, v, row_norm="none")
    got = pgptlformer.scaled_dot_product_attn_bias(q, k, v, row_norm="inv_sqrt_s")
    assert torch.allclose(got, raw * 9 ** -0.5)


def test_attention_ii_norm_none_is_bit_identical_to_the_original_kernel():
    """the legacy path must not have moved."""
    q = torch.randn(2, 4, 7, 8, dtype=torch.float64)
    k = torch.randn(2, 4, 7, 8, dtype=torch.float64)
    v = torch.ones(2, 4, 7, 8, dtype=torch.float64)
    mask = torch.tril(torch.ones(7, 7, dtype=torch.float64)).expand(2, 7, 7)
    scale = 8 ** -0.5

    got = pgptlformer.scaled_dot_product_attn_bias(q, k, v, attn_mask=mask, scale=scale)
    legacy = ((q @ k.transpose(-2, -1)) * scale * mask[:, None, :, :]) @ v
    assert torch.equal(got, legacy)


def test_attention_ii_is_a_scalar_gain_along_a_learned_direction():
    """the framing, made checkable.

    V is all ones, so every row of (scores @ V) is a CONSTANT repeated across dim_head:
    the row sum. after attnoutproj that is (row_sum_h) * (sum of that head's columns of O) --
    a per-head scalar gain along a fixed direction in the residual stream. it is not a bias
    and it is not a value mixture; there is no information in the V slot at all.
    """
    q = torch.randn(1, 2, 5, 8, dtype=torch.float64)
    k = torch.randn(1, 2, 5, 8, dtype=torch.float64)
    v = torch.ones(1, 2, 5, 8, dtype=torch.float64)
    out = pgptlformer.scaled_dot_product_attn_bias(q, k, v)
    # every component of a given (head, position) vector is the same number
    assert torch.allclose(out, out[..., :1].expand_as(out))
    # and that number is exactly the (scaled) score row sum
    rows = ((q @ k.transpose(-2, -1)) * 8 ** -0.5).sum(-1)
    assert torch.allclose(out[..., 0], rows)
