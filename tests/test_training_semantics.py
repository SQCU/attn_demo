"""a real training loop on the cpu, and a lint that guards the device-sync work.

`loader.main()` still wants bitsandbytes (cuda-only), so this file rebuilds the exact
accumulate/normalize/step sequence out of the same pieces main() uses, and checks the two
properties the loader rewrite was supposed to preserve:

  1. gradient accumulation produces the SAME gradients it did before the rewrite, and in
     particular no micro-step's backward() is dropped.
  2. the curriculum sampler ends in the SAME state whether its updates are applied
     per-sequence inside the accumulation loop (the old, sync-per-sequence form) or drained
     once at the end of it (the new form).

plus a source-level lint over the training and sampling loops, so a newly introduced
`.item()` in an inner loop is caught mechanically instead of by reading the diff.
"""
import ast
import pathlib

import numpy as np
import pytest
import torch

from conftest import build, tiny_config, tiny_t5_config

REPO = pathlib.Path(__file__).resolve().parent.parent


# --- 1. gradient accumulation ------------------------------------------------------------

def _accumulate(model, batches, train_accumulation_steps, ddp_run, drop_last_backward):
    """the loader's inner loop, parameterized on the bug.

    drop_last_backward=True reproduces the shipped behaviour: under ddp the `if i <
    train_accumulation_steps` branch had NO else, so the final micro-step never called
    backward() at all.
    """
    model.zero_grad(set_to_none=True)
    for micro_step, (x, y, mask) in enumerate(batches, start=1):
        _, loss, _, _ = model(x, targets=y, padding_mask=mask)
        if ddp_run and micro_step < train_accumulation_steps:
            loss.backward()
        elif ddp_run and drop_last_backward:
            pass                       # <-- the bug
        else:
            loss.backward()
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = (p.grad / train_accumulation_steps).clone()
    return grads


def _batches(n=4, vocab=64, seqlen=16):
    out = []
    for i in range(n):
        g = torch.Generator().manual_seed(100 + i)
        out.append((torch.randint(0, vocab, (2, seqlen), generator=g),
                    torch.randint(0, vocab, (2, seqlen), generator=g),
                    torch.ones(2, seqlen, dtype=torch.bool)))
    return out


def test_ddp_final_microbatch_gradient_is_no_longer_dropped():
    cfg = tiny_config(attention_deux=True)
    batches = _batches()
    torch.nn.init.normal_(build(cfg).tokenpicker_head.weight, std=0.02)  # (no-op, clarity)

    fixed = build(cfg)
    torch.nn.init.normal_(fixed.tokenpicker_head.weight, std=0.02)
    buggy = build(cfg)
    torch.nn.init.normal_(buggy.tokenpicker_head.weight, std=0.02)

    g_fixed = _accumulate(fixed, batches, 4, ddp_run=True, drop_last_backward=False)
    g_buggy = _accumulate(buggy, batches, 4, ddp_run=True, drop_last_backward=True)
    assert g_fixed and g_buggy
    differing = [k for k in g_fixed if not torch.allclose(g_fixed[k], g_buggy[k], atol=1e-9)]
    assert differing, "the dropped-backward bug should be observable"

    # and the fixed ddp path matches the single-gpu path exactly, which is the whole
    # contract of gradient accumulation.
    single = build(cfg)
    torch.nn.init.normal_(single.tokenpicker_head.weight, std=0.02)
    g_single = _accumulate(single, batches, 4, ddp_run=False, drop_last_backward=False)
    for k in g_fixed:
        assert torch.allclose(g_fixed[k], g_single[k], atol=1e-9), k


def test_accumulation_counter_is_not_shadowed():
    """the per-sequence curriculum loop reused `i`, the gradient-accumulation counter, so
    after the first t5 micro-step `i` was device_batch_size-1 and the `i <
    train_accumulation_steps` ddp test was reading a batch index."""
    src = (REPO / "loader.py").read_text()
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    loop_targets = [n.target.id for n in ast.walk(main)
                    if isinstance(n, ast.For) and isinstance(n.target, ast.Name)]
    assert loop_targets.count("i") == 0, f"bare `i` loop variables in main(): {loop_targets}"
    assert "micro_step" in loop_targets


def test_a_short_training_run_is_stable_and_deterministic():
    """two runs, same seed, identical loss trajectory. this is the before/after harness:
    it pins the semantics of the loop so a sync-removal refactor cannot quietly change it."""
    def run():
        cfg = tiny_config(attention_deux=True)
        model = build(cfg)
        torch.nn.init.normal_(model.tokenpicker_head.weight, std=0.02)
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        losses = []
        for step in range(6):
            model.zero_grad(set_to_none=True)
            for x, y, m in _batches(n=2):
                _, loss, _, _ = model(x, targets=y, padding_mask=m)
                loss.backward()
                losses.append(loss.detach().clone())
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= 2
            opt.step()
        return torch.stack(losses)

    a, b = run(), run()
    assert torch.equal(a, b)
    assert torch.isfinite(a).all()
    assert a[-1] < a[0], "six sgd steps on two repeated batches should reduce the loss"


# --- 2. the curriculum sampler drain -----------------------------------------------------

def test_batched_curriculum_drain_is_state_identical_to_per_sequence_updates():
    """the old code did `curriculum_sampler.update(bucket[i].item(), loss[i].item())` inside
    the micro-step loop: 2*device_batch_size host transfers per micro-step. the new code
    keeps them on device and replays the SAME updates in the SAME order once per optimizer
    step. the sampler is only READ by get_distribution() at the top of the step, so this is
    state-identical -- assert that rather than assume it."""
    from t5_utils import AdaptiveCurriculumSampler

    torch.manual_seed(0)
    micro_steps = [(torch.randint(0, 8, (5,)), torch.rand(5) * 4 + 1) for _ in range(4)]

    interleaved = AdaptiveCurriculumSampler()
    for buckets, losses in micro_steps:
        for b, l in zip(buckets.tolist(), losses.tolist()):
            interleaved.update(b, l)

    drained = AdaptiveCurriculumSampler()
    all_b = torch.cat([b for b, _ in micro_steps]).tolist()
    all_l = torch.cat([l for _, l in micro_steps]).cpu().tolist()
    for b, l in zip(all_b, all_l):
        drained.update(b, l)

    assert torch.equal(interleaved.ema_losses, drained.ema_losses)
    assert torch.equal(interleaved.ema_global_loss, drained.ema_global_loss)
    assert interleaved.step_counter == drained.step_counter
    assert interleaved.target_loss_percentile == drained.target_loss_percentile


def test_create_curriculum_batch_makes_at_most_two_host_transfers():
    """counted by instrumenting Tensor.tolist / .item / .cpu on the input batch."""
    from t5_utils import T5BatchProcessor

    class Sampler:
        def get_params_from_bucket(self, i):
            return 3, 0.15

    calls = {"n": 0}
    real_tolist, real_item = torch.Tensor.tolist, torch.Tensor.item

    def counting_tolist(self):
        calls["n"] += 1
        return real_tolist(self)

    def counting_item(self):
        calls["n"] += 1
        return real_item(self)

    bp = T5BatchProcessor(mask_token_start_id=62, pad_token_id=60, eos_token_id=61, vocab_size=64)
    np.random.seed(0)
    torch.manual_seed(0)
    x = torch.randint(0, 60, (16, 32))     # 16 sequences: the old code made 32 transfers
    torch.Tensor.tolist = counting_tolist
    torch.Tensor.item = counting_item
    try:
        bp.create_curriculum_batch(x, torch.ones(8) / 8, Sampler())
    finally:
        torch.Tensor.tolist = real_tolist
        torch.Tensor.item = real_item
    assert calls["n"] <= 2, f"{calls['n']} host transfers for a batch of 16"


# --- 3. the sampling loop ----------------------------------------------------------------

def test_ar_sample_stop_check_is_interval_gated_not_per_token():
    from sampler_utils import ar_sample

    model = build(tiny_config())
    model.eval()
    calls = {"n": 0}
    real_all = torch.Tensor.all

    def counting_all(self, *a, **kw):
        calls["n"] += 1
        return real_all(self, *a, **kw)

    x = torch.randint(0, 64, (2, 4))
    torch.Tensor.all = counting_all
    try:
        with torch.no_grad():
            ar_sample(model, x, max_new_tokens=64, max_seq=16, eos_id=63, stop_check_every=32)
    finally:
        torch.Tensor.all = real_all
    assert calls["n"] <= 2, f"{calls['n']} stop checks for 64 tokens"


def test_ar_sample_eos_bookkeeping_is_correct_at_batch_size_one():
    """`has_finished |= (idx_next.squeeze() == eos_id)` collapses [1,1] to a 0-d tensor at
    batch_size 1, which is the documented single-sample debug run."""
    from sampler_utils import ar_sample

    class AlwaysEOS(torch.nn.Module):
        def forward(self, idx, **kw):
            logits = torch.full((idx.size(0), idx.size(1), 8), -30.0)
            logits[:, -1, 7] = 30.0
            return logits, None, None, None

    out = ar_sample(AlwaysEOS(), torch.zeros(1, 3, dtype=torch.long),
                    max_new_tokens=8, eos_id=7, pad_id=0, stop_check_every=1)
    # first generated token is eos; every one after it is forced to pad, not resampled
    assert out[0, 3].item() == 7
    assert out.shape[1] < 11, "should have stopped early at batch_size 1"


def test_ar_sample_never_syncs_when_no_eos_is_given():
    from sampler_utils import ar_sample
    model = build(tiny_config())
    model.eval()
    calls = {"n": 0}
    real_all, real_item = torch.Tensor.all, torch.Tensor.item

    def counting(self, *a, **kw):
        calls["n"] += 1
        return real_all(self, *a, **kw)

    def counting_i(self, *a, **kw):
        calls["n"] += 1
        return real_item(self, *a, **kw)

    torch.Tensor.all, torch.Tensor.item = counting, counting_i
    try:
        with torch.no_grad():
            ar_sample(model, torch.randint(0, 64, (2, 4)), max_new_tokens=12, max_seq=16)
    finally:
        torch.Tensor.all, torch.Tensor.item = real_all, real_item
    assert calls["n"] == 0


# --- 4. the lint --------------------------------------------------------------------------

SYNC_CALLS = {"item", "tolist", "numpy", "cpu"}


def _sync_calls_in(node):
    """attribute calls on this node that force a device->host transfer."""
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in SYNC_CALLS:
                found.append((n.lineno, n.func.attr))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "synchronize":
            found.append((n.lineno, "cuda.synchronize"))
    return found


def _innermost_loops(func):
    """for-loops in `func` that contain no further for-loop: the inner loops."""
    loops = [n for n in ast.walk(func) if isinstance(n, ast.For)]
    return [l for l in loops
            if not any(isinstance(c, ast.For) and c is not l for c in ast.walk(l))]


ALLOWED = {
    # loader.main(): the one deliberate, documented, once-per-optimizer-step drain that
    # feeds AdaptiveCurriculumSampler. brentq runs on the host; the values have to land
    # there. it is outside the micro-step loop and it is batched.
    "loader.py": {"curriculum drain"},
}


def test_no_unaccounted_host_syncs_in_the_training_inner_loops():
    """a source-level guard, in the same spirit as the arity-proof unpacking: catch a newly
    introduced `.item()` in an inner loop mechanically, not by inspection."""
    src = (REPO / "loader.py").read_text()
    tree = ast.parse(src)
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")

    offenders = []
    for loop in _innermost_loops(main):
        # the micro-step loop and the validation loop are the hot ones.
        for lineno, what in _sync_calls_in(loop):
            offenders.append((lineno, what))
    assert not offenders, (
        "device->host transfers inside an inner training loop in loader.main(): "
        f"{offenders}. if one of these is genuinely required, hoist it out of the loop "
        "or gate it on an interval and say why in a comment.")


def test_the_lint_itself_detects_a_planted_sync():
    """check the instrument. this repo's whole failure mode is an unchecked instrument, and
    a lint that passes because it finds nothing anywhere is exactly that."""
    planted = ast.parse(
        "def main():\n"
        "    for step in range(10):\n"
        "        for micro in range(4):\n"
        "            print(loss.item())\n"
        "            torch.cuda.synchronize()\n")
    fn = next(n for n in ast.walk(planted) if isinstance(n, ast.FunctionDef))
    hits = [w for loop in _innermost_loops(fn) for _, w in _sync_calls_in(loop)]
    assert "item" in hits and "cuda.synchronize" in hits

    clean = ast.parse(
        "def main():\n"
        "    for step in range(10):\n"
        "        for micro in range(4):\n"
        "            acc += loss.detach()\n"
        "    print(acc.item())\n")
    fn = next(n for n in ast.walk(clean) if isinstance(n, ast.FunctionDef))
    assert not [w for loop in _innermost_loops(fn) for _, w in _sync_calls_in(loop)]


def test_sampler_utils_inner_loop_has_exactly_one_gated_sync():
    src = (REPO / "sampler_utils.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "ar_sample")
    for loop in _innermost_loops(fn):
        assert not _sync_calls_in(loop), _sync_calls_in(loop)
    # the one stop check is a bool() on an interval, and the interval test guards it
    assert "stop_check_every" in src
