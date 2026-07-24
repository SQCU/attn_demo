"""every file in configs/ validates against MODEL_CONFIG_SCHEMA and builds a model.

this is the test that closes the drift class, so it is worth saying what the class IS.

three forces were converging on configs/:
  * config_utils reads every key by VALUE and raises on missing/mistyped/out-of-range,
  * loader.merge_config was made strict -- an unknown key is an error, not a warning,
  * the model grew knobs (attn_gate, attention_deux_norm) that default to legacy.

each of those is individually correct and all three together are a trap: a shipped config
that omits a key now either gets REJECTED at load or silently inherits a default that
nobody chose. and the reason it is a *class* of bug rather than an incident is that no
diff shows it. two configs in the same directory that look like siblings can declare
different key sets, and the only place that difference becomes visible is eight hours into
a training run.

so: not "fix the six configs", but "assert, for every file in the directory, forever, that
its KEYS are the schema's keys". a new config that forgets attn_gate fails here, at import
speed, before it burns a GPU-hour.

the model build uses a tiny SHAPE override -- a full-size 6-layer d=768 t5 build per config
is ~40x the wall clock for no extra coverage, and the shape keys are already range-checked
by the schema. every other key is used EXACTLY as shipped, which is the part that matters:
the question is whether the file's declarations are legal and coherent, not whether 768
still multiplies.
"""
import json
import pathlib

import pytest

import config_utils as cu
import pgptlformer
from config_utils import ConfigError, Key, MODEL_CONFIG_SCHEMA, validate_model_config

REPO = pathlib.Path(__file__).resolve().parent.parent
CONFIGS = sorted((REPO / "configs").glob("*.json"))
NAMES = [p.name for p in CONFIGS]

# shrink the build without touching a single semantic key.
TINY_SHAPE = {"num_layers": 1, "dim": 64, "dim_head": 16, "headcount": 4,
              "training_seqlen": 128}


def load(path):
    return json.loads(path.read_text())


def test_the_config_directory_is_not_empty():
    """a suite that iterates a directory and finds nothing passes vacuously."""
    assert len(CONFIGS) >= 6, NAMES


# --- the schema is the same table the model reads ------------------------------------------

def test_schema_choice_sets_match_pgptlformers_canonical_tuples():
    """config_utils must not import torch, so it restates these tuples. if they ever
    diverge, the schema accepts a value the model then rejects (or worse, the reverse)."""
    assert cu.NORM_KINDS == pgptlformer.NORM_KINDS
    assert cu.ATTN_GATE_KINDS == pgptlformer.ATTN_GATE_KINDS
    assert cu.ATTN_II_NORM_KINDS == pgptlformer.ATTN_II_NORM_KINDS


def test_schema_keys_are_exactly_the_loader_dataclass_model_config_keys():
    """merge_config rejects any model_config key the dataclass does not define, so a
    schema key the dataclass lacks is unreachable and a dataclass key the schema lacks is
    undocumented. they must be the same set."""
    import loader
    assert {k.name for k in MODEL_CONFIG_SCHEMA} == set(loader.Hyperparameters().model_config)


def test_schema_defaults_match_the_loader_dataclass_defaults():
    import loader
    defaults = loader.Hyperparameters().model_config
    for key in MODEL_CONFIG_SCHEMA:
        assert key.default == defaults[key.name], key.name


def test_the_shared_fixtures_are_themselves_legal_configs():
    """conftest's tiny_config/tiny_t5_config are what most of this suite builds from. if
    they are not schema-legal, the suite can be entirely green while every file in
    configs/ is rejected at load -- which is very nearly the situation this test exists to
    prevent."""
    from conftest import assert_schema_complete, tiny_config, tiny_t5_config
    assert_schema_complete(tiny_config())
    assert_schema_complete(tiny_t5_config())


def test_every_schema_key_is_documented():
    for key in MODEL_CONFIG_SCHEMA:
        assert len(key.doc) > 20, f"{key.name} has no usable doc"
    assert len(cu.schema_table()) == len(MODEL_CONFIG_SCHEMA)


# --- every shipped config, as shipped ------------------------------------------------------

@pytest.mark.parametrize("path", CONFIGS, ids=NAMES)
def test_shipped_config_model_section_validates_as_written(path):
    """require_present=True: the KEYS are checked against the file, not against the
    dataclass-merged result. this is the assertion that catches an omission."""
    validate_model_config(load(path)["model_config"], where=path.name)


@pytest.mark.parametrize("path", CONFIGS, ids=NAMES)
def test_shipped_config_top_level_keys_are_all_hyperparameters(path):
    import dataclasses
    import loader
    known = {f.name for f in dataclasses.fields(loader.Hyperparameters())} | {"model_config"}
    unknown = sorted(k for k in load(path) if k not in known and not cu.is_doc_key(k))
    assert not unknown, f"{path.name}: unknown top-level key(s) {unknown}"


@pytest.mark.parametrize("path", CONFIGS, ids=NAMES)
def test_shipped_config_training_seqlen_agrees_with_sequence_length(path):
    """merge_config OVERWRITES model_config['training_seqlen'] from the top-level
    sequence_length. a file where the two disagree is a file whose stated model shape is a
    lie -- it builds at the other number and says nothing."""
    raw = load(path)
    assert raw["model_config"]["training_seqlen"] == raw["sequence_length"], path.name


@pytest.mark.parametrize("path", CONFIGS, ids=NAMES)
def test_shipped_config_merges_and_survives_the_startup_validator(path):
    import loader
    merged = loader.merge_config(loader.Hyperparameters(), load(path))
    validate_model_config(merged.model_config, where=path.name, require_present=False)


@pytest.mark.parametrize("path", CONFIGS, ids=NAMES)
def test_shipped_config_builds_a_model_and_honors_its_switches(path):
    """construct for real, then assert the file's switches reached the block. a config
    that validates and then builds something other than what it says is the original
    incident in a new costume."""
    cfg = dict(load(path)["model_config"])
    cfg.update(TINY_SHAPE)
    model = pgptlformer.PGPT_Lformer(cfg)
    blocks = model.encoder if cfg["is_t5"] else model.lambdaformer.blocks
    block = blocks[0]
    assert block.attention_II is cfg["attention_deux"], path.name
    assert block.attention_II_norm == cfg["attention_deux_norm"], path.name
    assert block.attn_gate_kind == cfg["attn_gate"], path.name
    assert block.rotbase == cfg["rotary_embedding_base"], path.name
    assert block.qknorm_kind == cfg["qknorm"], path.name
    if cfg["is_t5"]:
        assert len(model.encoder) == len(model.decoder) == cfg["num_layers"], path.name
        assert model.pad_token_id == cfg["pad_token_id"], path.name


@pytest.mark.parametrize("path", CONFIGS, ids=NAMES)
def test_shipped_config_special_ids_fit_the_vocabulary(path):
    """a pad/eos/sentinel id at or past vocab_size indexes off the end of the embedding.
    range-checkable per key, but only cross-checkable here."""
    cfg = load(path)["model_config"]
    v = cfg["vocab_size"]
    for name in ("pad_token_id", "eos_token_id", "mask_token_start_id"):
        value = cfg.get(name)
        if value is not None:
            assert 0 <= value < v, f"{path.name}: {name}={value} outside vocab_size={v}"


def test_the_two_void_ablation_configs_say_what_they_do():
    """ascii_eos_model.json and ascii_eos_model_rollouttest.json both declared
    attention_deux: false and both got attention-II anyway, because the key was read by
    presence. the value stays false -- that is what the file asked for and what makes the
    ablation an ablation -- and each file now carries a _note saying so, because anyone
    comparing against an old checkpoint of the same run_name needs to know it is not
    comparable."""
    for name in ("ascii_eos_model.json", "ascii_eos_model_rollouttest.json"):
        cfg = load(REPO / "configs" / name)["model_config"]
        assert cfg["attention_deux"] is False, name
        assert "_note" in cfg, f"{name} lost its void-ablation note"
        assert "attention-II ON" in cfg["_note"], name
        built = pgptlformer.vit22_tformer({**cfg, **TINY_SHAPE})
        assert built.attention_II is False, name


# --- the validator itself ------------------------------------------------------------------
# a validator that cannot fail passes every config in the world.

MINIMAL = {k.name: k.default for k in MODEL_CONFIG_SCHEMA if k.required}


def test_the_minimal_schema_defaults_are_themselves_valid():
    validate_model_config(dict(MINIMAL))


def test_validator_rejects_an_unknown_key():
    with pytest.raises(ConfigError, match="attenton_deux"):
        validate_model_config({**MINIMAL, "attenton_deux": True})


def test_validator_rejects_a_missing_required_key():
    cfg = dict(MINIMAL)
    del cfg["attn_gate"]
    with pytest.raises(ConfigError, match="'attn_gate' is required and missing"):
        validate_model_config(cfg)


def test_validator_rejects_a_bad_choice_and_a_bad_type_and_a_bad_range():
    with pytest.raises(ConfigError, match="must be one of"):
        validate_model_config({**MINIMAL, "qknorm": "dynamic_shape_rmsnrom"})
    with pytest.raises(ConfigError, match="must be int"):
        validate_model_config({**MINIMAL, "dim": "768"})
    with pytest.raises(ConfigError, match="must be >= 1"):
        validate_model_config({**MINIMAL, "num_layers": 0})


def test_validator_rejects_a_boolean_where_an_int_belongs():
    """bool is a subclass of int, so isinstance(True, int) is True and a plain type check
    lets `"num_layers": true` through as 1."""
    with pytest.raises(ConfigError, match="boolean"):
        validate_model_config({**MINIMAL, "num_layers": True})


def test_validator_rejects_a_stringly_typed_boolean():
    with pytest.raises(ConfigError, match="'attention_deux' must be bool"):
        validate_model_config({**MINIMAL, "attention_deux": "false"})


def test_validator_enforces_the_head_dimension_invariant():
    with pytest.raises(ConfigError, match=r"dim_head\*headcount"):
        validate_model_config({**MINIMAL, "headcount": 11})


def test_validator_requires_the_t5_ids_only_when_is_t5():
    validate_model_config({**MINIMAL, "is_t5": False})            # absent: fine
    with pytest.raises(ConfigError, match="required when is_t5 is true"):
        validate_model_config({**MINIMAL, "is_t5": True})
    with pytest.raises(ConfigError, match="must not be null when is_t5 is true"):
        validate_model_config({**MINIMAL, "is_t5": True, "pad_token_id": None,
                               "eos_token_id": 1, "mask_token_start_id": 2})
    validate_model_config({**MINIMAL, "is_t5": True, "pad_token_id": 0,
                           "eos_token_id": 1, "mask_token_start_id": 2})


def test_validator_reports_every_problem_at_once():
    """one exception per run, not one per fix. a half-migrated config is worse than an
    unmigrated one, because it looks done."""
    cfg = dict(MINIMAL)
    del cfg["attn_gate"]
    del cfg["is_t5"]
    cfg["qknorm"] = "nope"
    cfg["bogus"] = 1
    with pytest.raises(ConfigError) as e:
        validate_model_config(cfg)
    msg = str(e.value)
    for expected in ("attn_gate", "is_t5", "qknorm", "bogus"):
        assert expected in msg, msg


def test_doc_keys_are_ignored_by_the_validator_and_dropped_by_the_merge():
    import loader
    validate_model_config({**MINIMAL, "_note": "why this file is like this"})
    merged = loader.merge_config(loader.Hyperparameters(), {
        "_comment": "top level documentation",
        "model_config": {"_note": "nested documentation", "num_layers": 3},
    })
    assert merged.model_config["num_layers"] == 3
    assert not [k for k in merged.model_config if cu.is_doc_key(k)]
    assert not hasattr(merged, "_comment")


def test_extend_schema_adds_and_overrides_without_editing_the_base():
    """the mechanism a fork uses instead of editing MODEL_CONFIG_SCHEMA."""
    extended = cu.extend_schema(
        Key("extra_knob", int, 7, lo=0, required=True, doc="a downstream key"),
        Key("num_layers", int, 4, lo=2, required=True, doc="tightened downstream"))
    names = [k.name for k in extended]
    assert "extra_knob" in names
    assert len(names) == len(set(names)), "extend_schema must not duplicate a key"
    assert {k.name for k in MODEL_CONFIG_SCHEMA} == set(names) - {"extra_knob"}, \
        "the base schema must be untouched"
    with pytest.raises(ConfigError, match="extra_knob"):
        validate_model_config(dict(MINIMAL), schema=extended)
    with pytest.raises(ConfigError, match="must be >= 2"):
        validate_model_config({**MINIMAL, "extra_knob": 1, "num_layers": 1},
                              schema=extended)
    validate_model_config({**MINIMAL, "extra_knob": 1}, schema=extended)
