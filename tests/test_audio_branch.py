"""the audio/dataset branch.

most of this branch needs pandas / scipy / torchaudio / encodec to import, which is exactly
why its defects survived: nothing could load it to check. the tests that need only torch run
for real; the ones that would need the audio stack are done at the AST level instead, which
is still a mechanical check and still catches the class of bug that was actually present
(module-global reads from inside methods, undefined names).
"""
import ast
import pathlib

import pytest
import torch

from conftest import build, tiny_t5_config

REPO = pathlib.Path(__file__).resolve().parent.parent


# --- the shared T5 decode loop -----------------------------------------------------------

class StubT5(torch.nn.Module):
    """emits `script` one token at a time, per batch row."""

    def __init__(self, script, vocab=8):
        super().__init__()
        self.script = script
        self.vocab = vocab
        self.decode_calls = 0

    def encode(self, ids, mask):
        return torch.zeros(ids.size(0), ids.size(1), 4)

    def decode_step(self, dec_ids, memory, enc_mask):
        step = dec_ids.size(1) - 1
        B = dec_ids.size(0)
        self.decode_calls += 1
        logits = torch.full((B, 1, self.vocab), -30.0)
        for b in range(B):
            row = self.script[b]
            logits[b, 0, row[min(step, len(row) - 1)]] = 30.0
        return logits


def test_t5_decode_handles_batch_size_one():
    """`has_finished |= (idx_next.squeeze() == eos_id)` collapses [1,1] to 0-d at B==1.

    the single-sample run is the documented debug path, so this is the configuration the
    bug was most likely to be hit in.
    """
    from sampler_utils import t5_decode, trim_at_eos
    EOS, PAD, START = 7, 0, 6
    model = StubT5([[1, 2, EOS]])
    tape = t5_decode(model, torch.ones(1, 5, dtype=torch.long), max_new=10,
                     pad_id=PAD, eos_id=EOS, decoder_start_id=START, stop_check_every=1)
    assert tape[0, 0].item() == START
    assert tape[0, 1:4].tolist() == [1, 2, EOS]
    assert tape.size(1) == 4, "should have stopped at eos, not run to max_new"
    assert trim_at_eos(tape, EOS)[0].tolist() == [1, 2]


def test_t5_decode_pads_finished_rows_and_keeps_going_for_the_others():
    from sampler_utils import t5_decode, trim_at_eos
    EOS, PAD, START = 7, 0, 6
    model = StubT5([[1, EOS], [1, 2, 3, 4, EOS]])
    tape = t5_decode(model, torch.ones(2, 5, dtype=torch.long), max_new=12,
                     pad_id=PAD, eos_id=EOS, decoder_start_id=START, stop_check_every=1)
    seqs = trim_at_eos(tape, EOS)
    assert seqs[0].tolist() == [1]
    assert seqs[1].tolist() == [1, 2, 3, 4]
    # the finished row got pad, not resampled tokens
    assert (tape[0, 3:] == PAD).all()


def test_t5_decode_stop_check_is_interval_gated():
    from sampler_utils import t5_decode
    EOS, PAD, START = 7, 0, 6
    calls = {"n": 0}
    real_all = torch.Tensor.all

    def counting(self, *a, **kw):
        calls["n"] += 1
        return real_all(self, *a, **kw)

    model = StubT5([[1, EOS]])
    torch.Tensor.all = counting
    try:
        t5_decode(model, torch.ones(1, 4, dtype=torch.long), max_new=64,
                  pad_id=PAD, eos_id=EOS, decoder_start_id=START, stop_check_every=32)
    finally:
        torch.Tensor.all = real_all
    assert calls["n"] <= 2, f"{calls['n']} host syncs for 64 decode steps"


def test_t5_decode_never_syncs_with_stop_check_disabled():
    from sampler_utils import t5_decode
    calls = {"n": 0}
    real_all, real_item = torch.Tensor.all, torch.Tensor.item
    torch.Tensor.all = lambda self, *a, **k: (calls.__setitem__("n", calls["n"] + 1)
                                              or real_all(self, *a, **k))
    torch.Tensor.item = lambda self, *a, **k: (calls.__setitem__("n", calls["n"] + 1)
                                               or real_item(self, *a, **k))
    try:
        t5_decode(StubT5([[1, 2, 3]]), torch.ones(1, 4, dtype=torch.long), max_new=10,
                  pad_id=0, eos_id=7, decoder_start_id=6, stop_check_every=0)
    finally:
        torch.Tensor.all, torch.Tensor.item = real_all, real_item
    assert calls["n"] == 0


def test_t5_decode_drives_a_real_model():
    from sampler_utils import t5_decode
    cfg = tiny_t5_config()
    model = build(cfg)
    model.eval()
    tape = t5_decode(model, torch.randint(0, 60, (3, 9)), max_new=7,
                     pad_id=cfg["pad_token_id"], eos_id=cfg["eos_token_id"],
                     decoder_start_id=cfg["mask_token_start_id"], top_k=5)
    assert tape.shape == (3, 8)


# --- source-level checks over the modules that need the audio stack ----------------------

def _module(path):
    return ast.parse((REPO / path).read_text())


def _code_only(path):
    """source with comments stripped.

    the text assertions below are about CODE. matching raw source would happily match the
    explanatory comment describing the bug, which would make them pass forever regardless of
    what the code does -- an unchecked instrument, in a test file about unchecked instruments.
    """
    import io
    import tokenize
    src = (REPO / path).read_text()
    return "\n".join(tok.string for tok in tokenize.generate_tokens(io.StringIO(src).readline)
                     if tok.type != tokenize.COMMENT)


def _global_reads_in_methods(tree, name="args"):
    """every read of `name` inside a function that neither defines it nor takes it as a
    parameter, excluding functions at module scope under __main__."""
    hits = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            bound = {a.arg for a in fn.args.args}
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    bound.add(n.id)
            for n in ast.walk(fn):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) \
                        and n.id == name and name not in bound:
                    hits.append((cls.name, fn.name, n.lineno))
    return hits


def test_structural_analyzer_methods_do_not_read_the_argparse_global():
    """_compute_predictive_stability and _compute_vector_novelty_activity read the
    module-level `args` from inside the class, so they worked only under __main__ and raised
    NameError on import -- and StructuralAnalyzer IS imported, via
    mformer_utils.analyze_audio_on_the_fly, which prompt_utils' OOD generator drives."""
    assert _global_reads_in_methods(_module("mformer_dataset.py")) == []


def test_analyze_defines_every_name_it_uses():
    """`analyze()` referenced `v_vecs`, which is never defined anywhere in the file."""
    tree = _module("mformer_dataset.py")
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "StructuralAnalyzer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "analyze")
    bound = {a.arg for a in fn.args.args}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        if isinstance(n, ast.comprehension) and isinstance(n.target, ast.Name):
            bound.add(n.target.id)
    module_level = {n.name for n in ast.walk(tree)
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    module_level |= {t.id for n in tree.body if isinstance(n, ast.Assign)
                     for t in n.targets if isinstance(t, ast.Name)}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            module_level |= {(a.asname or a.name).split(".")[0] for a in n.names}
    method_names = {m.name for m in cls.body if isinstance(m, ast.FunctionDef)}
    import builtins
    known = bound | module_level | method_names | set(dir(builtins)) | {"np", "pd", "self"}
    unknown = {n.id for n in ast.walk(fn)
               if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in known}
    assert unknown == set(), f"analyze() reads undefined names: {sorted(unknown)}"


def test_argparse_help_strings_match_the_implemented_modes():
    # --novelty_mode omitted 'activity', which is implemented; --stability_mode listed
    # 'activity', which is not a stability mode at all and silently fell through to 'local'.
    tree = _module("mformer_dataset.py")
    choices = {}
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call)]:
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"):
            continue
        flag = call.args[0].value if call.args else None
        for kw in call.keywords:
            if kw.arg == "choices":
                choices[flag] = {e.value for e in kw.value.elts}
    assert choices.get("--novelty_mode") == {"crossover", "velocity", "activity"}
    assert choices.get("--stability_mode") == {"local", "predictive"}


def test_tencache_path_is_a_real_directory_join():
    """f"tencache\\{stem}_{suffix}.pt" -- `\\{` is not an escape, so python kept the
    backslash and this wrote ONE file into the cwd literally named "tencache\\...pt"."""
    src = _code_only("sample_audio_t5.py")
    assert "tencache\\" not in src
    assert "makedirs" in src and "TENCACHE_DIR" in src and "join" in src


def test_the_superseded_decode_functions_are_gone():
    src = (REPO / "sample_audio_t5.py").read_text()
    assert "def t5_decode_fully_batched" not in src
    assert "def t5_infill_batched" not in src


def test_redis_decode_responses_is_consistent_across_the_services():
    """t5_service.py set decode_responses=True while its two siblings set it False WITH a
    comment explaining that decoding corrupts raw tensor bytes."""
    explicit = {}
    for name in ("t5_service.py", "encodec_service.py", "local_client.py"):
        for call in [n for n in ast.walk(_module(name)) if isinstance(n, ast.Call)]:
            for kw in call.keywords:
                if kw.arg == "decode_responses":
                    assert kw.value.value is False, f"{name} sets decode_responses=True"
                    explicit[name] = False
    # the two siblings rely on redis-py's byte-mode default and explain it in a comment;
    # t5_service was the only one that passed the kwarg, and it passed True. it still passes
    # the kwarg, now False, so the decision is visible rather than inherited.
    assert explicit.get("t5_service.py") is False


def test_skipdecode_fallback_guards_the_bpe_index_lookup():
    """the exact-match branch guards with `if fused_id in self.bpe_index and len(...) > 0`;
    the fallback did not, and would KeyError."""
    src = (REPO / "sample_audio_t5_skipdecode.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "decode")
    unguarded = [n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Attribute)
                 and n.value.attr in ("seq_to_id",)
                 and isinstance(n.ctx, ast.Load)]
    # the only remaining direct subscript is the exact-match one, which is inside an
    # `if sub_sequence in self.seq_to_id` test.
    assert len(unguarded) <= 1, f"unguarded seq_to_id lookups at lines {unguarded}"

    # and the fallback reaches both dicts through .get, not [] -- checked structurally.
    getters = {n.func.value.attr for n in ast.walk(fn)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "get" and isinstance(n.func.value, ast.Attribute)}
    assert {"seq_to_id", "bpe_index"} <= getters, getters


def test_every_python_file_at_repo_root_parses():
    """cheap, and it is how netspec.py's un-importability would have been caught in 2025."""
    for path in sorted(REPO.glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    for path in sorted((REPO / "data").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
