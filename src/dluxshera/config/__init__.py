"""Strict nested configuration resolution for Shera systems."""

from .resolver import (
    as_dict,
    deep_merge,
    load_experiment_preset,
    load_system_preset,
    resolve_config,
    resolve_experiment_config,
    resolve_system_config,
    resolved_config_to_system_config,
)

__all__ = [
    "as_dict",
    "deep_merge",
    "load_system_preset",
    "load_experiment_preset",
    "resolve_system_config",
    "resolve_experiment_config",
    "resolve_config",
    "resolved_config_to_system_config",
]
