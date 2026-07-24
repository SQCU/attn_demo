"""config is read by VALUE. every knob, every time.

the incident: pgptlformer.py:70 read

    if "attention_deux" in config.keys(): self.attention_II = True

which is a presence test. configs/ascii_eos_model.json and
configs/ascii_eos_model_rollouttest.json both say "attention_deux": false, and both got
attention-II anyway. those two ablations measured nothing.
"""
import json
import pathlib

import pytest
import torch

from conftest import ar_batch, build, tiny_config

import pgptlformer
from config_utils import ConfigError, cfg_choice, cfg_flag, cfg_get

REPO = pathlib.Path(__file__).resolve().parent.parent


# --- the headline regression -------------------------------------------------------------

def test_attention_deux_false_actually_disables_it():
    model = build(tiny_config(attention_deux=False))
    block = model.lambdaformer.blocks[0]
    assert block.attention_II is False
    assert not hasattr(block, "queryBproj")
    assert not hasattr(block, "keyBproj")


def test_attention_deux_true_enables_it():
    block = build(tiny_config(attention_deux=True)).lambdaformer.blocks[0]
    assert block.attention_II is True
    assert hasattr(block, "queryBproj") and hasattr(block, "keyBproj")


def test_attention_deux_absent_defaults_off():
    cfg = tiny_config()
    del cfg["attention_deux"]
    assert build(cfg).lambdaformer.blocks[0].attention_II is False


def test_attention_deux_changes_the_numbers():
    """belt and braces: prove the flag is not merely cosmetic.

    compare BLOCK outputs, not logits: tokenpicker_head is zero-initialized (re:
    @Grad62304977) so a fresh model's logits are identically zero no matter what the trunk
    does. an ablation checked at the logits of an untrained model measures nothing -- which
    is a small echo of the exact failure this file is about.
    """
    x, _, mask = ar_batch()
    h = torch.randn(2, 16, 32, generator=torch.Generator().manual_seed(3))
    attn_mask = pgptlformer.create_attention_mask(mask, is_causal=True)
    off = build(tiny_config(attention_deux=False)).lambdaformer.blocks[0](h, attention_mask=attn_mask)
    on = build(tiny_config(attention_deux=True)).lambdaformer.blocks[0](h, attention_mask=attn_mask)
    assert not torch.allclose(off, on)


def test_shipped_configs_that_say_false_get_false():
    """read the actual json on disk, not a fixture."""
    for name in ("ascii_eos_model.json", "ascii_eos_model_rollouttest.json"):
        cfg = json.loads((REPO / "configs" / name).read_text())["model_config"]
        assert cfg["attention_deux"] is False, name
        assert pgptlformer.vit22_tformer(cfg).attention_II is False, name


def test_every_shipped_config_builds():
    for path in sorted((REPO / "configs").glob("*.json")):
        cfg = json.loads(path.read_text())["model_config"]
        model = pgptlformer.PGPT_Lformer(cfg)
        blocks = model.encoder if cfg.get("is_t5") else model.lambdaformer.blocks
        assert blocks[0].attention_II is cfg["attention_deux"], path.name


# --- the helpers themselves --------------------------------------------------------------

def test_cfg_flag_rejects_stringly_typed_booleans():
    with pytest.raises(ConfigError, match="the string"):
        cfg_flag({"x": "false"}, "x")
    with pytest.raises(ConfigError):
        cfg_flag({"x": 1}, "x")
    assert cfg_flag({"x": False}, "x") is False
    assert cfg_flag({}, "x", default=True) is True


def test_cfg_get_missing_required_names_the_key():
    with pytest.raises(ConfigError, match="'dim'"):
        cfg_get({}, "dim", types=int)
    assert cfg_get({}, "dim", 768, types=int) == 768


def test_cfg_get_type_and_choice_enforcement():
    with pytest.raises(ConfigError, match="must be int"):
        cfg_get({"dim": "768"}, "dim", types=int)
    with pytest.raises(ConfigError, match="must be one of"):
        cfg_choice({"qknorm": "rmsnrom"}, "qknorm", pgptlformer.NORM_KINDS)


# --- fail loudly on malformed / partial model configs ------------------------------------

def test_missing_required_model_key_fails_loudly():
    cfg = tiny_config()
    del cfg["dim_head"]
    with pytest.raises(ConfigError, match="dim_head"):
        pgptlformer.PGPT_Lformer(cfg)


def test_typo_in_norm_name_fails_loudly():
    with pytest.raises(ConfigError, match="qknorm"):
        pgptlformer.PGPT_Lformer(tiny_config(qknorm="dynamic_shape_rmsnrom"))


def test_head_dim_mismatch_is_asserted():
    """attnoutproj requires dim_head*headcount == dim and nothing ever checked it."""
    with pytest.raises(ConfigError, match="dim_head\\*headcount"):
        pgptlformer.PGPT_Lformer(tiny_config(headcount=3))


def test_rotary_base_default_is_1000_and_configurable():
    """the base is 1000, not 10000. deliberate. explicit now, still 1000."""
    assert build(tiny_config()).lambdaformer.blocks[0].rotbase == 1000
    assert build(tiny_config(rotary_embedding_base=10000)
                 ).lambdaformer.blocks[0].rotbase == 10000


# --- loader.py's config merge ------------------------------------------------------------

def test_loader_merge_keeps_unmentioned_model_keys():
    """the hasattr branch used to setattr model_config wholesale from the json.

    a json model_config missing e.g. 'qknorm' therefore silently dropped it instead of
    defaulting to the dataclass value.
    """
    import loader
    merged = loader.merge_config(loader.Hyperparameters(), {
        "run_name": "unit-test",
        "model_config": {"num_layers": 9},
    })
    assert merged.run_name == "unit-test"
    assert merged.model_config["num_layers"] == 9
    assert merged.model_config["qknorm"] == "dynamic_shape_rmsnorm"  # not dropped
    assert merged.model_config["dim"] == 768


def test_loader_merge_rejects_unknown_keys():
    import loader
    with pytest.raises(ConfigError, match="attenton_deux"):
        loader.merge_config(loader.Hyperparameters(),
                            {"model_config": {"attenton_deux": True}})
    with pytest.raises(ConfigError, match="num_iteratons"):
        loader.merge_config(loader.Hyperparameters(), {"num_iteratons": 10})


def test_loader_merge_every_shipped_config():
    import loader
    for path in sorted((REPO / "configs").glob("*.json")):
        merged = loader.merge_config(loader.Hyperparameters(), json.loads(path.read_text()))
        pgptlformer.PGPT_Lformer(merged.model_config)


def test_hyperparameters_string_defaults_are_strings():
    """`input_npz: str = "",` -- that trailing comma makes the default the TUPLE ("",)."""
    import dataclasses
    import loader
    hp = loader.Hyperparameters()
    for f in dataclasses.fields(hp):
        value = getattr(hp, f.name)
        if f.type in ("str", str):
            assert isinstance(value, str), f"{f.name} default is {value!r}, not a str"
        assert not (isinstance(value, tuple) and len(value) == 1), \
            f"{f.name} default is a 1-tuple {value!r} -- stray trailing comma"


def test_hyperparameters_logs_every_field():
    """device/torch_compile/use_z_loss/z_loss_coefficient had no annotations, so they were
    plain class attributes -- settable, but invisible to asdict() and therefore missing from
    every ACTIVE HYPERPARAMETERS block ever written to a logfile."""
    import dataclasses
    import loader
    names = {f.name for f in dataclasses.fields(loader.Hyperparameters())}
    for expected in ("device", "torch_compile", "use_z_loss", "z_loss_coefficient"):
        assert expected in names, expected
