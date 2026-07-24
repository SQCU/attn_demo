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

__all__ = ["ConfigError", "cfg_get", "cfg_flag", "cfg_choice", "REQUIRED"]


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
