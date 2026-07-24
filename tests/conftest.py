"""shared fixtures for the attn_demo test suite.

run with:  uv run python -m pytest tests -q
(or, on a machine with no cuda at all, any cpu torch works -- nothing here needs a gpu.)

the point of this directory is the thing this repo has historically been short of: an
instrument that is itself checked. every test here runs a real forward pass and asserts on
real numbers, not on the absence of an exception.
"""
import os
import pathlib
import sys

import pytest
import torch

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# tests/test_flex_masks.py and tests/test_lints.py need nothing from this file beyond the
# path insert above -- they exercise flex_masks.py and the two lints, which import only
# torch, ast, json and os and know about no model at all. that is the point of them, and
# it is why they can be read (and run) standalone.


def tiny_config(**overrides):
    """the smallest config that exercises every branch. dim_head*headcount == dim.

    every key of MODEL_CONFIG_SCHEMA is stated, at the value the fixture was already
    getting implicitly, and assert_schema_complete() below pins that. a fixture that is
    not a legal config is a fixture that can pass while every shipped file fails."""
    cfg = {
        "vocab_size": 64,
        "num_layers": 2,
        "dim": 32,
        "dim_head": 8,
        "headcount": 4,
        "ff_mult": 4,
        "training_seqlen": 16,
        "layerwisenorm": "rmsnorm",
        "qknorm": "dynamic_shape_rmsnorm",
        "lambda": True,
        "is_t5": False,
        "attention_deux": False,
        "attention_deux_norm": "none",
        "attn_gate": "none",
        "rotary_embedding_base": 1000,
    }
    cfg.update(overrides)
    return cfg


def tiny_t5_config(**overrides):
    cfg = tiny_config(
        is_t5=True,
        pad_token_id=60,
        eos_token_id=61,
        mask_token_start_id=62,
    )
    cfg.update(overrides)
    return cfg


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(1337)
    torch.use_deterministic_algorithms(False)
    yield


def build(config, seed=1337):
    """seeded construction, so two models with the same seed have identical weights."""
    import pgptlformer
    torch.manual_seed(seed)
    return pgptlformer.PGPT_Lformer(config)


def ar_batch(batch=2, seqlen=16, vocab=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, vocab, (batch, seqlen), generator=g)
    y = torch.randint(0, vocab, (batch, seqlen), generator=g)
    mask = torch.ones(batch, seqlen, dtype=torch.bool)
    return x, y, mask


def assert_schema_complete(cfg):
    """the fixtures above must be legal shipped configs, not merely configs the model
    happens to accept. tests/test_config_schema.py calls this on both of them."""
    from config_utils import validate_model_config
    return validate_model_config(dict(cfg), where="conftest fixture")
