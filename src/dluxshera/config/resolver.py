"""
Resolve canonical configuration mappings into validated runtime blocks.

This module is the configuration *resolver* for canonical dLuxShera workflows.
It takes a user-provided mapping containing top-level ``system`` and/or
``experiment`` blocks and returns a normalized, schema-checked mapping.

Contract
--------
- Canonical workflows use a nested mapping
  with top-level ``system`` and/or ``experiment`` blocks.
- This module does **not** accept dataclasses or attribute-based configs.
- Presets are optional:
    - If ``<block>.preset`` is present, the corresponding preset file is loaded
      and deep-merged with the user block (user values win).
    - If ``<block>.preset`` is absent, the user block is treated as fully
      specified and validated as-is.

Separation of responsibilities
------------------------------
- ``config/io.py`` owns file I/O and deep merge.
- This module owns block resolution (preset merge), validation, and lightweight
  normalization (type coercions).
- Recipes decide which blocks are required and how to consume the resolved
  blocks (binder/spec composition, inference execution, outputs, etc.).
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .io import deep_merge, load_experiment_preset, load_system_preset


__all__ = [
    "as_dict",
    "resolve_system_config",
    "resolve_experiment_config",
    "resolve_config",
]


def as_dict(cfg: object) -> dict[str, Any]:
    """Coerce and validate the top-level config mapping.

    Parameters
    ----------
    cfg : object
        Must be a mapping containing at least one of top-level ``system`` or
        ``experiment``.

    Returns
    -------
    dict[str, Any]
        Deep-copied plain dict version of the config.

    Raises
    ------
    TypeError
        If ``cfg`` is not a mapping, or if it lacks both ``system`` and
        ``experiment``.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError(
            "Config must be a mapping with top-level 'system' and/or 'experiment'."
        )

    data = dict(cfg)

    if "system" not in data and "experiment" not in data:
        raise TypeError("Config must provide at least one of 'system' or 'experiment'.")

    if "system" in data and not isinstance(data["system"], Mapping):
        raise TypeError("Top-level 'system' must be a mapping/dict.")
    if "experiment" in data and not isinstance(data["experiment"], Mapping):
        raise TypeError("Top-level 'experiment' must be a mapping/dict.")

    return deepcopy(data)


def _required(path: str, mapping: Mapping[str, Any], key: str) -> Any:
    """Return a required key from mapping or raise a ValueError.

    Parameters
    ----------
    path : str
        Dot-path prefix used for error reporting (e.g., "system.source").
    mapping : Mapping[str, Any]
        Mapping to validate.
    key : str
        Required key name.

    Raises
    ------
    ValueError
        If the key is missing.
    """
    if key not in mapping:
        raise ValueError(f"Missing required config key: {path}.{key}")
    return mapping[key]


def _required_nonempty_string(path: str, mapping: Mapping[str, Any], key: str) -> str:
    """Return required non-empty string value from mapping.

    Raises
    ------
    ValueError
        If the key is missing or not a non-empty string.
    """
    value = _required(path, mapping, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required config key: {path}.{key}")
    return value


def _optional_nonempty_string(path: str, mapping: Mapping[str, Any], key: str) -> str | None:
    """Return an optional non-empty string value or None if the key is absent.

    Raises
    ------
    ValueError
        If the key is present but not a non-empty string.
    """
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid config value: {path}.{key} must be a non-empty string.")
    return value


def _validate_layer_list(layers: object) -> None:
    """Validate the structural shape of ``system.detector.layers``.

    Requirements
    ------------
    - Must be a list.
    - Each entry must be a mapping/dict.
    - Each layer mapping must contain a ``name`` key.

    Notes
    -----
    This function performs only structural validation. Layer-specific
    parameter validation is delegated to the detector builder.
    """
    if not isinstance(layers, list):
        raise ValueError("system.detector.layers must be a list of layer dictionaries.")
    for idx, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError(f"system.detector.layers[{idx}] must be a mapping/dict.")
        if "name" not in layer:
            raise ValueError(
                f"Missing required config key: system.detector.layers[{idx}].name"
            )


def _validate_system_schema(system: object) -> None:
    """Validate high-level structure of the ``system`` block.

    This function intentionally performs only minimal validation.
    Component-specific schema validation is delegated to builders.
    """
    # Must be a mapping
    if not isinstance(system, Mapping):
        raise ValueError("system must be a mapping/dict.")

    # Must contain "source", "optics", and "detector"
    source = _required("system", system, "source")
    optics = _required("system", system, "optics")
    detector = _required("system", system, "detector")

    # Each must be a mapping
    if not isinstance(source, Mapping):
        raise ValueError("system.source must be a mapping/dict.")
    if not isinstance(optics, Mapping):
        raise ValueError("system.optics must be a mapping/dict.")
    if not isinstance(detector, Mapping):
        raise ValueError("system.detector must be a mapping/dict.")

    # Source, Optics require "kind" discriminator
    _required_nonempty_string("system.source", source, "kind")
    _required_nonempty_string("system.optics", optics, "kind")

    # Detector requires "model" and "layers"
    _required_nonempty_string("system.detector", detector, "model")
    layers = _required("system.detector", detector, "layers")
    _validate_layer_list(layers)


def _validate_experiment_schema(experiment: object) -> None:
    """Minimal structural validation for experiment block.

    Resolver intentionally does not enforce experiment schema.
    Recipes/scripts are responsible for validating required keys.
    """
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment must be a mapping/dict.")


def resolve_system_config(
    system_cfg: Mapping[str, Any],
    *,
    presets_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve a ``system`` block from optional preset + overrides.

    Parameters
    ----------
    system_cfg : Mapping[str, Any]
        User ``system`` mapping. If ``system.preset`` is present, the preset
        is loaded and merged with ``system_cfg`` (user values win).
    presets_dir : Path | None, optional
        Override directory for system presets.

    Returns
    -------
    dict[str, Any]
        Resolved and normalized ``system`` mapping (no outer wrapper).
    """
    # Must be a mapping
    if not isinstance(system_cfg, Mapping):
        raise ValueError("system must be a mapping/dict.")

    # Extract optional preset discriminator
    preset_name = _optional_nonempty_string("system", system_cfg, "preset")
    if preset_name is not None:
        # Load preset and merge with user config
        preset_system = load_system_preset(preset_name, presets_dir=presets_dir)["system"]
        resolved = deep_merge(dict(preset_system), dict(system_cfg))
    else:
        # no preset, just copy user config
        resolved = dict(deepcopy(dict(system_cfg)))

    # Apply minimal structural validation
    _validate_system_schema(resolved)
    return resolved


def resolve_experiment_config(
    experiment_cfg: Mapping[str, Any],
    *,
    presets_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve an ``experiment`` block from optional preset + overrides.

    Parameters
    ----------
    experiment_cfg : Mapping[str, Any]
        User ``experiment`` mapping. If ``experiment.preset`` is present, the
        preset is loaded and merged with ``experiment_cfg`` (user values win).
    presets_dir : Path | None, optional
        Override directory for experiment presets.

    Returns
    -------
    dict[str, Any]
        Resolved ``experiment`` mapping (no outer wrapper).
    """
    # Must be a mapping
    if not isinstance(experiment_cfg, Mapping):
        raise ValueError("experiment must be a mapping/dict.")

    # Extract optional preset discriminator
    preset_name = _optional_nonempty_string("experiment", experiment_cfg, "preset")
    if preset_name is not None:
        # Load preset and merge with user config
        preset_experiment = load_experiment_preset(preset_name, presets_dir=presets_dir)["experiment"]
        resolved = deep_merge(dict(preset_experiment), dict(experiment_cfg))
    else:
        # No preset, just copy user config
        resolved = dict(deepcopy(dict(experiment_cfg)))

    return resolved


def resolve_config(
    user_cfg: Mapping[str, Any],
    *,
    system_presets_dir: Path | None = None,
    experiment_presets_dir: Path | None = None,
) -> dict[str, Any]:
    """Resolve top-level config into nested ``system``/``experiment`` blocks.

    Parameters
    ----------
    user_cfg : Mapping[str, Any]
        Canonical config mapping with top-level ``system`` and/or ``experiment``.
    system_presets_dir : Path | None, optional
        Override directory for system presets.
    experiment_presets_dir : Path | None, optional
        Override directory for experiment presets.

    Returns
    -------
    dict[str, Any]
        Mapping containing whichever resolved blocks were supplied.

    Notes
    -----
    - System and experiment are resolved independently.
    - No cross-block validation is performed here.
    - Recipes/scripts enforce workflow-specific requirements.
    """
    # Coerce and validate top-level structure
    # (must contain at least one of "system" or "experiment")
    cfg = as_dict(user_cfg)

    resolved: dict[str, Any] = {}
    if "system" in cfg:
        # Resolve system block
        resolved["system"] = resolve_system_config(
            cfg["system"], presets_dir=system_presets_dir
        )
    if "experiment" in cfg:
        # Resolve experiment block
        resolved["experiment"] = resolve_experiment_config(
            cfg["experiment"], presets_dir=experiment_presets_dir
        )

    # Must resolve at least one block
    if not resolved:
        raise ValueError("Config must provide at least one of 'system' or 'experiment'.")
    return resolved
