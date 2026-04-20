"""Strict nested configuration resolution for Shera systems."""

from .io import deep_merge, load_experiment_preset, load_system_preset
from .numeric import coerce_numeric_mapping, coerce_numeric_value, normalize_optimizer_kwargs
from .resolver import as_dict, resolve_config, resolve_experiment_config, resolve_system_config

__all__ = [
    "as_dict",
    "coerce_numeric_mapping",
    "coerce_numeric_value",
    "deep_merge",
    "load_system_preset",
    "load_experiment_preset",
    "normalize_optimizer_kwargs",
    "resolve_system_config",
    "resolve_experiment_config",
    "resolve_config",
]
