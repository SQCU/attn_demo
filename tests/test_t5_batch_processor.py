"""the T5 target stream: what is in the loss, what is not, and why.

the defect: the "fix special token masking" commit overwrote sentinels AND EOS in `labels`
with pad_token_id so ignore_index would skip them. the model was therefore never trained to
emit EOS -- while sample_t5.py, sample_audio_t5.py, sample_audio_t5_skipdecode.py and
t5_service.py all terminate generation on EOS. a year of rollouts that could only ever stop
by exhausting max_new.
"""
import numpy as np
import pytest
import torch

from conftest import build, tiny_t5_config

from t5_utils import T5BatchProcessor

PAD, EOS, MASK_START, VOCAB = 60, 61, 62, 64


def make_bp(**kw):
    kw.setdefault("mask_token_start_id", MASK_START)
    kw.setdefault("pad_token_id", PAD)
    kw.setdefault("eos_token_id", EOS)
    kw.setdefault("vocab_size", VOCAB)
    return T5BatchProcessor(**kw)


def batch(seed=0, B=4, T=24):
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.randint(0, 60, (B, T))


# --- EOS ---------------------------------------------------------------------------------

def test_eos_survives_into_the_loss_labels():
    bp = make_bp()
    _, _, labels, _, _ = bp(batch(), avg_span_length=3, mask_prob=0.15)
    assert (labels == EOS).any(), "EOS was deleted from the labels again"
    # exactly one per sequence: _create_masked_sequence appends one terminator
    assert ((labels == EOS).sum(dim=1) == 1).all()


def test_eos_actually_contributes_to_the_cross_entropy():
    """not just present in the tensor -- present in the number.

    zeroing the model's confidence at the EOS positions must move the loss. if EOS were
    still being mapped to pad_token_id and skipped by ignore_index, it would not.
    """
    bp = make_bp()
    x = batch()
    _, dec_in, labels, _, _ = bp(x, avg_span_length=3, mask_prob=0.15)
    B, T = labels.shape
    logits = torch.zeros(B, T, VOCAB)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    base = loss_fn(logits.view(-1, VOCAB), labels.view(-1))
    # make the EOS CLASS look impossible wherever EOS is the target. (lowering the whole
    # row would be a uniform shift and softmax would not notice -- which is itself a nice
    # reminder of how easy it is to write a check that measures nothing.)
    logits[..., EOS] = torch.where(labels == EOS, torch.tensor(-50.0), logits[..., EOS])
    worse = loss_fn(logits.view(-1, VOCAB), labels.view(-1))
    assert worse > base + 1.0


def test_eos_is_in_the_curriculum_path_too():
    """create_curriculum_batch had a copy-pasted duplicate of the whole labels block."""
    bp = make_bp()

    class Sampler:
        def get_params_from_bucket(self, i):
            return 3, 0.15

    dist = torch.ones(8) / 8
    _, _, labels, _, _, buckets = bp.create_curriculum_batch(batch(), dist, Sampler())
    assert (labels == EOS).any()
    assert buckets.shape == (4,)
    assert buckets.device.type == "cpu"   # host tensor: no per-sequence sync downstream


# --- sentinels ---------------------------------------------------------------------------

def test_sentinel_exclusion_range_is_derived_not_hardcoded_at_100():
    """the range was `mask_token_start_id + 100` while the masker's own guardrail allows
    `vocab_size - mask_token_start_id` sentinels. for the audio configs that is 1022, so
    sentinels 100..1021 fell through and WERE trained on: 'sentinels are excluded, except
    the ones that aren't'."""
    bp = T5BatchProcessor(mask_token_start_id=1026, pad_token_id=1024,
                          eos_token_id=1025, vocab_size=2048)
    assert bp.max_spans == 2048 - 1026 == 1022

    # a target stream containing a high-numbered sentinel, well past the old +100 bound
    targets = torch.tensor([[1026, 5, 6, 1026 + 500, 7, 1025, 1024]])
    labels = bp.build_labels(targets)
    assert labels[0, 0] == 1024          # sentinel 0 excluded
    assert labels[0, 3] == 1024          # sentinel 500 ALSO excluded (was not, before)
    assert labels[0, 5] == 1025          # EOS kept
    assert labels[0, 1] == 5             # content kept


def test_no_sentinel_id_is_left_in_an_inconsistent_state():
    """every id the masker can emit must be handled the same way by the loss builder."""
    for start, vocab in ((62, 64), (131, 256), (1026, 2048)):
        bp = T5BatchProcessor(mask_token_start_id=start, pad_token_id=start - 2,
                              eos_token_id=start - 1, vocab_size=vocab)
        all_sentinels = torch.arange(start, start + bp.max_spans)[None, :]
        labels = bp.build_labels(all_sentinels)
        assert (labels == bp.pad_token_id).all(), (start, vocab)
        # and the masker cannot emit anything outside that range
        assert start + bp.max_spans == vocab


def test_train_on_sentinels_is_available_and_off_by_default():
    bp = make_bp()
    assert bp.train_on_sentinels is False
    targets = torch.tensor([[MASK_START, 5, EOS]])
    assert bp.build_labels(targets).tolist() == [[PAD, 5, EOS]]

    canonical = make_bp(train_on_sentinels=True)
    assert canonical.build_labels(targets).tolist() == [[MASK_START, 5, EOS]]


# --- the decoder BOS column ---------------------------------------------------------------

def test_decoder_bos_position_is_visible():
    """decoder_inputs[:, 0] is the start slot and is spelled with pad_token_id, so a bare
    `!= pad_token_id` marked the whole first COLUMN invisible: no decoder position could
    attend to the start of the sequence, and query row 0 could attend to nothing at all."""
    bp = make_bp()
    _, dec_in, _, _, dec_mask = bp(batch(), avg_span_length=3, mask_prob=0.15)
    assert (dec_in[:, 0] == PAD).all()
    assert dec_mask[:, 0].all()


def test_decoder_row_zero_attends_to_something():
    import pgptlformer
    bp = make_bp()
    _, dec_in, _, _, dec_mask = bp(batch(), avg_span_length=3, mask_prob=0.15)
    m = pgptlformer.create_attention_mask(dec_mask, is_causal=True)
    assert m[:, 0, :].any(dim=-1).all()

    old = (dec_in != PAD)   # the previous mask
    old_m = pgptlformer.create_attention_mask(old, is_causal=True)
    assert not old_m[:, 0, :].any(dim=-1).any()


def test_padding_is_still_excluded_from_the_loss():
    bp = make_bp()
    _, _, labels, enc_mask, _ = bp(batch(), avg_span_length=8, mask_prob=0.4)
    # ragged targets get padded; those positions must be pad_token_id in the labels
    assert (labels == PAD).any()


# --- end to end ---------------------------------------------------------------------------

def test_a_t5_forward_pass_learns_from_eos():
    """gradient flows from an EOS target into the model. the point of the whole fix."""
    model = build(tiny_t5_config())
    bp = make_bp()
    x = batch(B=2, T=20)
    enc_in, dec_in, labels, enc_m, dec_m = bp(x, avg_span_length=3, mask_prob=0.15)
    eos_only = torch.full_like(labels, PAD)
    eos_only[labels == EOS] = EOS       # loss on the EOS positions and nothing else
    # tokenpicker_head is zero-init, so give it something to differentiate.
    torch.nn.init.normal_(model.tokenpicker_head.weight, std=0.02)
    _, loss, _, _ = model(enc_in, dec_in, eos_only, enc_m, dec_m)
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert model.tokenpicker_head.weight.grad[EOS].abs().sum() > 0


def test_masker_guardrail_matches_the_exclusion_range():
    """the masker refuses to emit more than max_spans sentinels; force it to the wall."""
    bp = make_bp()   # only 2 sentinel ids available: 62, 63
    assert bp.max_spans == 2
    np.random.seed(3)
    inp, tgt = bp._create_masked_sequence(list(range(40)), avg_span_length=1, mask_prob=0.9)
    emitted = {t for t in tgt if t >= MASK_START and t != EOS}
    assert emitted <= {62, 63}
    labels = bp.build_labels(torch.tensor([tgt]))
    assert not ((labels >= MASK_START) & (labels < VOCAB) & (labels != EOS)).any()
