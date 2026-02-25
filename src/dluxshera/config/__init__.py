"""Strict nested configuration resolution for Shera systems."""

from .resolver import (
    as_dict,
    deep_merge,
    load_preset,
    resolve_config,
    resolved_config_to_system_config,
)

__all__ = [
    "as_dict",
    "deep_merge",
    "load_preset",
    "resolve_config",
    "resolved_config_to_system_config",
]
