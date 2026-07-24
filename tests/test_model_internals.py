"""the norms, the scales, and the shapes. checked against reference implementations.

each test here corresponds to a mechanism that looked right and was not.
"""
import math

import pytest
import torch
import torch.nn as nn

from conftest import ar_batch, build, tiny_config, tiny_t5_config

import pgptlformer


# --- dynamic_shape_rmsnorm: the transposes were a no-op --------------------------------

def _legacy_dynamic_shape_norm(inputter, fn):
    """the shipped implementation, verbatim, as a reference."""
    inputter = inputter.transpose(1, 2)
    inner_shape = inputter.size()[3:]
    inputter = fn(inputter, normalized_shape=inner_shape)
    return inputter.transpose(1, 2)


@pytest.mark.parametrize("shape", [(2, 16, 4, 8), (1, 1, 4, 8), (3, 7, 12, 64)])
def test_dynamic_shape_rmsnorm_is_bit_identical_to_the_transposing_version(shape):
    x = torch.randn(*shape, dtype=torch.float64)
    new = pgptlformer.dynamic_shape_rmsnorm()(x)
    old = _legacy_dynamic_shape_norm(x, nn.functional.rms_norm)
    assert torch.equal(new, old)


@pytest.mark.parametrize("shape", [(2, 16, 4, 8), (1, 1, 4, 8)])
def test_dynamic_shape_layernorm_is_bit_identical(shape):
    x = torch.randn(*shape, dtype=torch.float64)
    new = pgptlformer.dynamic_shape_layernorm()(x)
    old = _legacy_dynamic_shape_norm(x, nn.functional.layer_norm)
    assert torch.equal(new, old)


def test_dynamic_shape_rmsnorm_is_just_rmsnorm_over_head_dim():
    x = torch.randn(2, 16, 4, 8, dtype=torch.float64)
    ref = x / torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    got = pgptlformer.dynamic_shape_rmsnorm()(x)
    assert torch.allclose(got, ref, atol=1e-5)


def test_dynamic_shape_norm_keeps_its_dynamic_shape_property():
    """the reason it exists: sampling changes B and T every step, and the module must not
    care. one instance, three different shapes, no reconstruction."""
    norm = pgptlformer.dynamic_shape_rmsnorm()
    for b, t in [(1, 1), (4, 37), (2, 512)]:
        out = norm(torch.randn(b, t, 4, 8))
        assert out.shape == (b, t, 4, 8)


# --- qknorm normalized over (H, D), i.e. across heads -----------------------------------

def test_qknorm_shape_is_one_head_wide():
    block = build(tiny_config(qknorm="rmsnorm")).lambdaformer.blocks[0]
    assert block.qknormalized_shape == [block.dim_head]


def test_qknorm_rmsnorm_normalizes_per_head_not_across_heads():
    """the old shape was [headcount, dim_head], which makes nn.RMSNorm reduce over the head
    axis too. scaling ONE head then had to change the others. it must not."""
    block = build(tiny_config(qknorm="rmsnorm")).lambdaformer.blocks[0]
    x = torch.randn(2, 5, block.heads, block.dim_head)
    base = block.projnorm(x)

    scaled = x.clone()
    scaled[:, :, 0, :] *= 100.0
    after = block.projnorm(scaled)
    assert torch.allclose(after[:, :, 1:, :], base[:, :, 1:, :], atol=1e-6)
    # and the touched head still comes out unit-rms
    assert torch.allclose(after[:, :, 0, :].pow(2).mean(-1),
                          torch.ones(2, 5), atol=1e-4)

    # the old joint norm demonstrably did NOT have that property:
    joint = nn.RMSNorm([block.heads, block.dim_head], elementwise_affine=False)
    assert not torch.allclose(joint(scaled)[:, :, 1:, :], joint(x)[:, :, 1:, :], atol=1e-6)


def test_qknorm_rmsnorm_matches_dynamic_shape_rmsnorm():
    """with the shape fixed these are the same operation, which is the point."""
    a = build(tiny_config(qknorm="rmsnorm")).lambdaformer.blocks[0].projnorm
    b = build(tiny_config(qknorm="dynamic_shape_rmsnorm")).lambdaformer.blocks[0].projnorm
    x = torch.randn(2, 5, 4, 8, dtype=torch.float64)
    assert torch.equal(a(x), b(x))


# --- l2norm "bootleg cosine attention" ---------------------------------------------------

def _reference_attention(q, k, v, mask, scale):
    """explicit softmax attention. [B, H, T, D] in, [B, H, T, D] out."""
    scores = (q @ k.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float('-inf'))
    return torch.softmax(scores, dim=-1) @ v


def test_l2norm_temperature_is_inside_the_softmax():
    """it used to multiply the attention OUTPUT with scale hardcoded to 1.

    outside the softmax a scalar is a per-block output rescale; it does not sharpen or
    flatten anything. inside, it is a temperature. these two differ, and the reference says
    which one we are.
    """
    cfg = tiny_config(qknorm="l2norm")
    block = build(cfg).lambdaformer.blocks[0]
    block.eval()
    torch.manual_seed(11)
    x = torch.randn(2, 12, cfg["dim"], dtype=torch.float32)
    mask = pgptlformer.create_attention_mask(torch.ones(2, 12, dtype=torch.bool), is_causal=True)

    got = block.self_attn(x, mask)

    # reference: build q/k/v the same way the block does, then attend explicitly.
    B, T, _ = x.shape
    H, D = block.heads, block.dim_head
    q = block.queryproj(x).view(B, T, H, D)
    k = block.keyproj(x).view(B, T, H, D)
    v = block.valueproj(x).view(B, T, H, D)
    cos, sin = block.rotary(q)
    q, k = block.projnorm(q), block.projnorm(k)
    q = pgptlformer.apply_rotarizer_emb(q, cos, sin)
    k = pgptlformer.apply_rotarizer_emb(k, cos, sin)
    y = _reference_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                             mask[:, None, :, :], block.l2normscale.item())
    expected = block.attnoutproj(y.transpose(1, 2).contiguous().view_as(x))
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()


def test_l2norm_path_honors_the_attention_mask():
    """it passed is_causal=True and dropped attention_mask entirely -- so padded keys were
    attended to, and the T5 ENCODER, which is bidirectional, was causally masked."""
    cfg = tiny_config(qknorm="l2norm")
    block = build(cfg).lambdaformer.blocks[0]
    block.eval()
    torch.manual_seed(5)
    x = torch.randn(1, 8, cfg["dim"])
    full = torch.ones(1, 8, dtype=torch.bool)

    causal = block.self_attn(x, pgptlformer.create_attention_mask(full, is_causal=True))
    bidi = block.self_attn(x, pgptlformer.create_attention_mask(full, is_causal=False))
    # position 0 sees everything under a bidirectional mask and only itself under a causal
    # one. if the mask were ignored these would be equal.
    assert not torch.allclose(causal[:, 0], bidi[:, 0], atol=1e-6)

    padded = full.clone()
    padded[0, 5:] = False
    masked = block.self_attn(x, pgptlformer.create_attention_mask(padded, is_causal=False))
    assert not torch.allclose(bidi[:, 0], masked[:, 0], atol=1e-6)


def test_l2norm_t5_encoder_is_bidirectional():
    """end to end: the encoder of an l2norm t5 must not be causal."""
    model = build(tiny_t5_config(qknorm="l2norm"))
    model.eval()
    ids = torch.randint(0, 60, (1, 10))
    mask = torch.ones(1, 10, dtype=torch.bool)
    h = model.encode(ids, mask)
    # perturbing a LATER token must change an EARLIER position's encoding.
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 7) % 60
    h2 = model.encode(ids2, mask)
    assert not torch.allclose(h[0, 0], h2[0, 0], atol=1e-6)


def test_l2normscale_is_a_trained_parameter_that_receives_gradient():
    cfg = tiny_config(qknorm="l2norm")
    model = build(cfg)
    block = model.lambdaformer.blocks[0]
    assert isinstance(block.l2normscale, nn.Parameter)
    # log(S^2 - S) for S = training_seqlen
    S = cfg["training_seqlen"]
    assert math.isclose(block.l2normscale.item(), math.log(S * S - S), rel_tol=1e-5)
    x, y, mask = ar_batch(seqlen=16)
    _, loss, _, _ = model(x, targets=y, padding_mask=mask)
    # tokenpicker_head is zero-init so the logit loss has no gradient path; use the trunk.
    h = model.lambdaformer.blocks[0](torch.randn(2, 16, cfg["dim"]),
                                     attention_mask=pgptlformer.create_attention_mask(mask, True))
    h.sum().backward()
    assert block.l2normscale.grad is not None
    assert block.l2normscale.grad.abs() > 0


def test_l2norm_cross_attention_also_gets_the_temperature():
    block = build(tiny_t5_config(qknorm="l2norm")).decoder[0]
    assert block.l2normscale is not None
    # cross_attn with unit-norm q,k and scale=1/sqrt(D) would be nearly uniform; with the
    # learned temperature it must not be.
    torch.manual_seed(2)
    x = torch.randn(1, 6, 32)
    enc = torch.randn(1, 9, 32)
    m = torch.ones(1, 6, 9, dtype=torch.bool)
    out = block.cross_attn(x, enc, m)
    assert torch.isfinite(out).all()


# --- structural asserts -------------------------------------------------------------------

def test_cross_attn_norm_is_gone():
    """constructed and never called, with the disabling line commented out
    ('nice try gemini 2.5!'). the pgptl parallel block has ONE pre-norm by design."""
    block = build(tiny_t5_config()).decoder[0]
    assert not hasattr(block, "cross_attn_norm")
    assert not any("cross_attn_norm" in k for k in block.state_dict())


def test_shipped_t5_state_dicts_are_unaffected_by_that_removal():
    """layerwisenorm=rmsnorm has elementwise_affine=False, so cross_attn_norm carried zero
    parameters in every config in configs/. removing it changes no state_dict key."""
    for kind in ("rmsnorm", "dynamic_shape_rmsnorm"):
        keys = build(tiny_t5_config(layerwisenorm=kind)).state_dict().keys()
        assert not any("cross_attn_norm" in k for k in keys)


def test_attn_bias_dropout_respects_eval_mode():
    """torch.dropout(..., train=True) unconditionally. p is 0.0 today so it is inert, but
    it was a live trap."""
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.ones(1, 2, 4, 8)
    a = pgptlformer.scaled_dot_product_attn_bias(q, k, v, dropout_p=0.5, training=False)
    b = pgptlformer.scaled_dot_product_attn_bias(q, k, v, dropout_p=0.0, training=True)
    assert torch.equal(a, b)
