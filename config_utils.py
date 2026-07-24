# config_utils.py
#
# the recurring failure of this repo is not the mechanism, it is the instrument reading it.
# this file is the instrument for config dicts.
#
# the specific accident that motivated it:
#
#     if "attention_deux" in config.keys():
#         self.attention_II = True
#
# that is a *presence* test standing in for a *value* test. two configs in configs/ set
# "attention_deux": false and got attention-II anyway, which voids both ablations that were
# run against them. so: no more `in config`, no more bare `.get()` where a missing key would
# quietly become None and blow up three call frames later. read values, check types, and
# fail with a message that names the key.

__all__ = ["ConfigError", "cfg_get", "cfg_flag", "cfg_choice", "REQUIRED",
           "Key", "MODEL_CONFIG_SCHEMA", "validate_model_config", "extend_schema",
           "schema_table", "is_doc_key"]


class ConfigError(ValueError):
    """raised on a missing, mistyped, or out-of-range config entry."""


class _Required:
    def __repr__(self):
        return "REQUIRED"


REQUIRED = _Required()


def _typename(types):
    if isinstance(types, tuple):
        return " or ".join(t.__name__ for t in types)
    return types.__name__


def cfg_get(config, key, default=REQUIRED, types=None, choices=None):
    """read config[key] by VALUE.

    missing key + a default -> the default.
    missing key + no default -> ConfigError naming the key.
    present key of the wrong type or outside `choices` -> ConfigError naming the key.
    """
    if key not in config:
        if default is REQUIRED:
            raise ConfigError(
                f"config key {key!r} is required and missing. "
                f"keys present: {sorted(config.keys())}"
            )
        return default
    value = config[key]
    if types is not None and not isinstance(value, types):
        raise ConfigError(
            f"config key {key!r} must be {_typename(types)}, "
            f"got {type(value).__name__} ({value!r})"
        )
    if choices is not None and value not in choices:
        raise ConfigError(
            f"config key {key!r} must be one of {sorted(choices)}, got {value!r}"
        )
    return value


def cfg_flag(config, key, default=False):
    """read a boolean config entry by VALUE.

    json `true`/`false` only. the string "false" is truthy in python and has burned this
    codebase once already, so a non-bool here is an error rather than a coin flip.
    """
    if key not in config:
        if default is REQUIRED:
            raise ConfigError(f"config flag {key!r} is required and missing.")
        return default
    value = config[key]
    # bool is a subclass of int, so check bool first and reject bare ints explicitly.
    if not isinstance(value, bool):
        raise ConfigError(
            f"config flag {key!r} must be a json boolean (true/false), "
            f"got {type(value).__name__} ({value!r}). "
            f"note that the string \"false\" is TRUE in python -- that is the whole reason "
            f"this check exists."
        )
    return value


def cfg_choice(config, key, choices, default=REQUIRED):
    """read a string-valued config entry restricted to a known set."""
    return cfg_get(config, key, default=default, types=str, choices=choices)


# =========================================================================================
# THE MODEL CONFIG SCHEMA
# =========================================================================================
# the helpers above check ONE key at the point it is read. that is necessary and it is not
# sufficient, because it can only ever catch a key the code actually reaches. it cannot
# catch:
#
#   * a key the file declares that nothing reads (a typo, or a knob deleted three commits
#     ago that the config still sets -- the run then silently does the other thing),
#   * a key the file omits that some OTHER config declares (schema drift across the
#     directory: two configs that look like siblings are not, and no diff shows it),
#   * a key whose default changed under a file that was relying on the old one.
#
# so the schema below is a declarative table, and it is the single documented answer to
# "what is a model config". every entry states type, allowed values, default, whether the
# key must appear IN THE FILE, and what it does. tests/test_config_schema.py walks every
# file in configs/ against it and then builds the model, which is what makes the drift
# class impossible rather than merely discouraged.
#
# WHERE THIS RUNS: loader.merge_config, once, at startup. it is O(number of keys), it
# touches no tensor and no device, and it is nowhere near a forward pass. programmatic
# configs (tests, notebooks) are not forced through it -- they are checked key-by-key at
# construction by the cfg_* helpers above, which is the appropriate granularity there.
#
# `required` means "must be written in the config FILE", not "must exist at read time":
# loader's Hyperparameters dataclass supplies a default for every key, so at read time
# they all exist. a key is required-in-file when letting it default would silently change
# what the run means -- i.e. every knob that moves the numbers.
#
# EXTENDING IT: a fork that adds a module with its own keys calls extend_schema() with its
# own Key list rather than editing this table, exactly as lint_attention_dtypes.py takes an
# additional allowlist file. that keeps the two rebasable.

import dataclasses as _dc
from typing import Any, Optional, Sequence, Tuple


def is_doc_key(key):
    """keys beginning with '_' are documentation, not configuration.

    json has no comments, and a config that cannot explain itself accumulates decisions
    nobody can reconstruct. '_comment' / '_note' are carried in the file, ignored by the
    schema, and never reach the model."""
    return isinstance(key, str) and key.startswith("_")


@_dc.dataclass(frozen=True)
class Key:
    """one model-config key. `types` is what isinstance() gets; `choices` restricts the
    value set; `lo`/`hi` are inclusive numeric bounds; `required` means it must be present
    in the shipped file; `t5_only` means it is additionally required when is_t5 is true
    and may be absent or null otherwise."""
    name: str
    types: Any
    default: Any = None
    choices: Optional[Sequence] = None
    lo: Optional[float] = None
    hi: Optional[float] = None
    required: bool = False
    t5_only: bool = False
    doc: str = ""


def _norm_kinds():
    # imported lazily to keep config_utils free of a torch dependency: pgptlformer owns
    # the canonical tuples, but importing it here would drag torch into every consumer.
    return (
        "layernorm", "layernorm-nobias", "rmsnorm",
        "dynamic_shape_rmsnorm", "dynamic_shape_layernorm",
        "l2norm", "identitynorm",
    )


NORM_KINDS = _norm_kinds()
ATTN_GATE_KINDS = ("none", "sigmoid")
ATTN_II_NORM_KINDS = ("none", "mean", "inv_sqrt_s")


MODEL_CONFIG_SCHEMA: Tuple[Key, ...] = (
    # --- shape -------------------------------------------------------------------------
    Key("vocab_size", int, 50304, lo=1, required=True,
        doc="output vocabulary size. must cover every special id declared below."),
    Key("num_layers", int, 4, lo=1, required=True,
        doc="blocks per stack. a t5 config builds this many ENCODER blocks AND this many "
            "decoder blocks, i.e. 2*num_layers in total."),
    Key("dim", int, 768, lo=1, required=True,
        doc="residual stream width. must equal dim_head*headcount."),
    Key("dim_head", int, 64, lo=1, required=True,
        doc="per-head width. dim_head*headcount == dim is enforced by the validator."),
    Key("headcount", int, 12, lo=1, required=True,
        doc="attention heads. dim_head*headcount == dim is enforced."),
    Key("ff_mult", int, 4, lo=1, required=True,
        doc="feed-forward hidden width as a multiple of dim."),
    Key("training_seqlen", int, 512, lo=1, required=True,
        doc="sequence length the model is built for. loader OVERWRITES this from the "
            "top-level sequence_length, so the two must agree; the schema test checks it."),

    # --- normalization -----------------------------------------------------------------
    Key("layerwisenorm", str, "rmsnorm", choices=NORM_KINDS, required=True,
        doc="norm applied over the full embedding dim."),
    Key("qknorm", str, "dynamic_shape_rmsnorm", choices=NORM_KINDS, required=True,
        doc="norm applied to q and k before the dot product."),

    # --- architecture switches ---------------------------------------------------------
    Key("lambda", bool, True, required=True,
        doc="weighted skip connections (the 'lambdaformer' residual)."),
    Key("is_t5", bool, False, required=True,
        doc="true = encoder/decoder with cross-attention; false = decoder-only "
            "autoregressive. required in the file because it decides which forward path "
            "runs and therefore which other keys are meaningful."),
    Key("attention_deux", bool, False, required=True,
        doc="the attention-II term. REQUIRED IN THE FILE with no exceptions: this key was "
            "read by presence rather than by value, so two configs saying false got it "
            "anyway and both of their ablations measured nothing. it is never allowed to "
            "be implicit again."),
    Key("attention_deux_norm", str, "none", choices=ATTN_II_NORM_KINDS, required=True,
        doc="row normalization of the attention-II term. 'none' is the legacy behavior "
            "(unnormalized, magnitude grows ~linearly in S) and is what every existing "
            "checkpoint was trained with. 'mean' divides by the unmasked key count per "
            "row. required in the file so the choice is never inherited by accident."),
    Key("attn_gate", str, "none", choices=ATTN_GATE_KINDS, required=True,
        doc="post-attention gate. 'none' is bit-identical to the pre-gate model. required "
            "in the file for the same reason as attention_deux_norm."),
    Key("rotary_embedding_base", (int, float), 1000, lo=1, required=True,
        doc="rotary theta. 1000, not the usual 10000 -- deliberate, and stated explicitly "
            "in every file precisely because a reader will assume it is a typo."),

    # --- t5-only special ids -----------------------------------------------------------
    # absent or null on an autoregressive config: .bin token streams are unpadded, and
    # PGPT_Lformer substitutes torch's ignore_index sentinel for a null pad_token_id.
    Key("pad_token_id", (int, type(None)), None, lo=0, t5_only=True,
        doc="padding / ignore_index. t5 only."),
    Key("eos_token_id", (int, type(None)), None, lo=0, t5_only=True,
        doc="end-of-sequence id. t5 only."),
    Key("mask_token_start_id", (int, type(None)), None, lo=0, t5_only=True,
        doc="first id of the sentinel range. t5 only; the range runs from here to "
            "vocab_size."),
)


def _by_name(schema):
    return {k.name: k for k in schema}


def extend_schema(*extra, base=MODEL_CONFIG_SCHEMA):
    """base schema + a fork's own Key entries -> a new schema tuple.

    a later Key with the same name REPLACES an earlier one, so a downstream repo can both
    add keys and tighten an existing one without editing this file."""
    merged = {k.name: k for k in base}
    for k in extra:
        for key in (k if isinstance(k, (list, tuple)) else (k,)):
            merged[key.name] = key
    return tuple(merged.values())


def _check_value(key, value, errors):
    if value is None and (key.t5_only or type(None) in
                          (key.types if isinstance(key.types, tuple) else (key.types,))):
        return
    if not isinstance(value, key.types):
        errors.append(f"{key.name!r} must be {_typename(key.types)}, got "
                      f"{type(value).__name__} ({value!r})")
        return
    # bool is a subclass of int; a schema that says int must not accept True.
    if key.types is int and isinstance(value, bool):
        errors.append(f"{key.name!r} must be an int, got the boolean {value!r}")
        return
    if key.choices is not None and value not in key.choices:
        errors.append(f"{key.name!r} must be one of {list(key.choices)}, got {value!r}")
        return
    if key.lo is not None and value < key.lo:
        errors.append(f"{key.name!r} must be >= {key.lo}, got {value!r}")
    if key.hi is not None and value > key.hi:
        errors.append(f"{key.name!r} must be <= {key.hi}, got {value!r}")


def validate_model_config(config, schema=MODEL_CONFIG_SCHEMA, where="model_config",
                          require_present=True):
    """check a model config against the schema. raises ConfigError listing EVERY problem.

    every problem, not the first: fixing a config one exception at a time is how a config
    ends up half-migrated. `require_present=False` checks only the keys that are there,
    which is what you want for a dict that has already been merged onto defaults.

    startup-time only. see the note above the schema."""
    schema = tuple(schema)
    known = _by_name(schema)
    errors = []

    unknown = sorted(k for k in config if k not in known and not is_doc_key(k))
    if unknown:
        errors.append(f"unknown key(s) {unknown}. known keys: {sorted(known)}")

    is_t5 = config.get("is_t5", False) is True
    for key in schema:
        present = key.name in config
        if not present:
            if require_present and key.required:
                errors.append(f"{key.name!r} is required and missing")
            elif require_present and key.t5_only and is_t5:
                errors.append(f"{key.name!r} is required when is_t5 is true, and missing")
            continue
        value = config[key.name]
        if key.t5_only and is_t5 and value is None:
            errors.append(f"{key.name!r} must not be null when is_t5 is true")
            continue
        _check_value(key, value, errors)

    # the one cross-key invariant. attnoutproj reshapes [B, S, headcount, dim_head] into
    # [B, S, dim]; if the product disagrees the model builds and then produces garbage.
    d, dh, hc = (config.get(n) for n in ("dim", "dim_head", "headcount"))
    if all(isinstance(v, int) and not isinstance(v, bool) for v in (d, dh, hc)):
        if dh * hc != d:
            errors.append(f"dim_head*headcount must equal dim: {dh}*{hc} = {dh * hc} != {d}")

    if errors:
        raise ConfigError(f"{where} failed schema validation:\n  - " + "\n  - ".join(errors))
    return config


def schema_table(schema=MODEL_CONFIG_SCHEMA):
    """the schema as a readable table. this is what the readme section is generated from,
    and `python -m config_utils` prints it, so the documentation cannot drift from the
    table the validator actually uses."""
    rows = []
    for k in schema:
        if k.required:
            req = "yes"
        elif k.t5_only:
            req = "if is_t5"
        else:
            req = "no"
        allowed = ("|".join(map(str, k.choices)) if k.choices is not None
                   else _typename(k.types))
        if k.lo is not None or k.hi is not None:
            allowed += f" [{k.lo if k.lo is not None else '-inf'}"
            allowed += f", {k.hi if k.hi is not None else 'inf'}]"
        rows.append((k.name, allowed, repr(k.default), req, k.doc))
    return rows


def _print_schema_table(schema=MODEL_CONFIG_SCHEMA):
    rows = schema_table(schema)
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    header = ("key", "type / allowed", "default", "in file?")
    widths = [max(w, len(h)) for w, h in zip(widths, header)]
    line = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print(line)
    for name, allowed, default, req, doc in rows:
        print("  ".join(c.ljust(w) for c, w in zip((name, allowed, default, req), widths)))
        print(f"      {doc}")


if __name__ == "__main__":
    _print_schema_table()
