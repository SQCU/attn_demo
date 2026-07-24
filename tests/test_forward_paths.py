"""every forward path, every arity, every optional return value.

this whole file exists because at commit 868d946 the autoregressive path was dead:
forward_arg grew a fourth return value (loss_per_sequence), the samplers kept doing
`logits, _, _ = model(...)`, and loss_per_sequence was never even assigned on the
no-targets branch. nobody noticed for months because nothing ran the sampler in CI.
now something does.
"""
import pytest
import torch

from conftest import ar_batch, build, tiny_config, tiny_t5_config

import pgptlformer


ARITY = 4  # logits, loss, z_loss, loss_per_sequence


@pytest.mark.parametrize("attention_deux", [False, True])
@pytest.mark.parametrize("return_zloss", [False, True])
def test_forward_arg_with_targets(attention_deux, return_zloss):
    model = build(tiny_config(attention_deux=attention_deux))
    x, y, mask = ar_batch()
    out = model(x, targets=y, padding_mask=mask, return_zloss=return_zloss)
    assert len(out) == ARITY
    logits, loss, z_loss, loss_per_seq = out
    assert logits.shape == (2, 16, 64)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert loss_per_seq.shape == (2,)
    if return_zloss:
        assert z_loss is not None and torch.isfinite(z_loss)
    else:
        assert z_loss is None


@pytest.mark.parametrize("attention_deux", [False, True])
def test_forward_arg_without_targets(attention_deux):
    """the branch that raised UnboundLocalError: cannot access 'loss_per_sequence'."""
    model = build(tiny_config(attention_deux=attention_deux))
    x, _, mask = ar_batch()
    logits, loss, z_loss, loss_per_seq = model(x, padding_mask=mask)
    assert logits.shape == (2, 1, 64)  # time dim preserved, last position only
    assert loss is None and z_loss is None and loss_per_seq is None


def test_forward_arg_default_padding_mask_is_not_int64():
    """create_attention_mask used to hand SDPA an int64 mask and raise.

    callers pass torch.ones_like(input_ids) (int64) or nothing at all; `long & bool`
    promotes to long and SDPA rejects long masks outright.
    """
    model = build(tiny_config())
    x, y, _ = ar_batch()
    # no padding_mask at all -> internal torch.ones_like(input_ids), which is int64
    logits, loss, _, _ = model(x, targets=y)
    assert torch.isfinite(loss)
    # and explicitly int64
    logits2, loss2, _, _ = model(x, targets=y, padding_mask=torch.ones_like(x))
    assert torch.equal(logits, logits2)


def test_create_attention_mask_returns_bool_for_int_input():
    m = pgptlformer.create_attention_mask(torch.ones(2, 5, dtype=torch.long), is_causal=True)
    assert m.dtype == torch.bool
    assert m[0, 0, 1].item() is False and m[0, 4, 0].item() is True


def test_ar_config_without_pad_token_id_trains():
    """no autoregressive config in configs/ declares pad_token_id.

    config.get("pad_token_id") -> None -> nn.CrossEntropyLoss(ignore_index=None) -> TypeError
    on the very first training step. must default to torch's ignore sentinel instead.
    """
    cfg = tiny_config()
    assert "pad_token_id" not in cfg
    model = build(cfg)
    assert model.pad_token_id == pgptlformer.DEFAULT_IGNORE_INDEX
    x, y, mask = ar_batch()
    _, loss, _, lps = model(x, targets=y, padding_mask=mask)
    assert torch.isfinite(loss)
    # nothing was ignored: every one of the 16 positions counted
    assert torch.allclose(lps.mean(), loss)


def _t5_batch(batch=2, enc_len=16, dec_len=12, vocab=64, pad=60):
    g = torch.Generator().manual_seed(7)
    enc = torch.randint(0, 60, (batch, enc_len), generator=g)
    dec = torch.randint(0, 60, (batch, dec_len), generator=g)
    tgt = torch.randint(0, 60, (batch, dec_len), generator=g)
    tgt[:, -2:] = pad  # some ignored positions, like a real padded target
    return (enc, dec, tgt,
            torch.ones(batch, enc_len, dtype=torch.bool),
            torch.ones(batch, dec_len, dtype=torch.bool))


@pytest.mark.parametrize("attention_deux", [False, True])
@pytest.mark.parametrize("return_zloss", [False, True])
def test_forward_t5_with_targets(attention_deux, return_zloss):
    model = build(tiny_t5_config(attention_deux=attention_deux))
    enc, dec, tgt, enc_m, dec_m = _t5_batch()
    out = model(enc, dec, tgt, enc_m, dec_m, return_zloss=return_zloss)
    assert len(out) == ARITY
    logits, loss, z_loss, lps = out
    assert logits.shape == (2, 12, 64)
    assert torch.isfinite(loss)
    assert lps.shape == (2,)
    assert (z_loss is not None) == return_zloss


def test_forward_t5_without_targets():
    """this branch referenced an undefined `x`, copy-pasted from forward_arg."""
    model = build(tiny_t5_config())
    enc, dec, _, enc_m, dec_m = _t5_batch()
    logits, loss, z_loss, lps = model(enc, dec, None, enc_m, dec_m)
    assert logits.shape == (2, 1, 64)
    assert loss is None and z_loss is None and lps is None


def test_forward_t5_honors_return_logits():
    model = build(tiny_t5_config())
    enc, dec, tgt, enc_m, dec_m = _t5_batch()
    logits, loss, _, _ = model(enc, dec, tgt, enc_m, dec_m, return_logits=False)
    assert logits is None and torch.isfinite(loss)


def test_t5_encode_and_decode_step():
    model = build(tiny_t5_config())
    enc, dec, _, enc_m, _ = _t5_batch()
    memory = model.encode(enc, enc_m)
    assert memory.shape == (2, 16, 32)
    logits = model.decode_step(dec, memory, enc_m)
    assert logits.shape == (2, 1, 64)


def test_ar_sampling_loop_runs():
    """a real (short) sampling loop, arity-proof unpacking, from the shipped sampler."""
    from sampler_utils import ar_sample
    model = build(tiny_config())
    model.eval()
    x = torch.randint(0, 64, (3, 5))
    with torch.no_grad():
        out = ar_sample(model, x, max_new_tokens=6, max_seq=16, temperature=1.0, top_k=8)
    assert out.shape == (3, 11)
    assert torch.equal(out[:, :5], x)


def test_arity_proof_unpack_survives_a_fifth_return_value():
    """the actual root cause was fixed-arity unpacking at the call sites.

    simulate a future forward() that returns one more thing and assert the shipped
    sampler still works.
    """
    from sampler_utils import ar_sample

    class FiveTuple(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *a, **kw):
            return (*self.inner(*a, **kw), "a new research quantity")

    model = FiveTuple(build(tiny_config()))
    model.eval()
    x = torch.randint(0, 64, (2, 4))
    with torch.no_grad():
        out = ar_sample(model, x, max_new_tokens=3, max_seq=16)
    assert out.shape == (2, 7)
