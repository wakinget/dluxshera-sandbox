"""Strict config resolver for the Phase 8A nested schema.

Required top-level schema:
  - system
  - experiment

Resolution flow:
  1) Load preset from ``src/dluxshera/data/presets`` via ``system.preset``.
  2) Deep-merge user config over preset defaults.
  3) Validate required keys and emit warnings for unknown keys.
  4) Normalize key numeric types used by builders.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
import json
import warnings


def _default_presets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "presets"


def as_dict(cfg: object) -> dict:
    """Convert supported config containers to a nested dict.

    Supported input:
      - ``dict`` / mapping
      - dataclass with ``system`` and ``experiment`` fields
      - object exposing ``system`` and ``experiment`` attributes

    Legacy flat config containers are rejected.
    """

    if isinstance(cfg, Mapping):
        return deepcopy(dict(cfg))

    if is_dataclass(cfg):
        data = asdict(cfg)
        if "system" in data and "experiment" in data:
            return data

    if hasattr(cfg, "system") and hasattr(cfg, "experiment"):
        system = getattr(cfg, "system")
        experiment = getattr(cfg, "experiment")
        if isinstance(system, Mapping) and isinstance(experiment, Mapping):
            return {
                "system": deepcopy(dict(system)),
                "experiment": deepcopy(dict(experiment)),
            }

    raise TypeError(
        "Config must be a nested mapping/object with top-level 'system' and 'experiment' blocks. "
        "Legacy flat config schemas are not supported."
    )


def load_preset(preset_name: str, *, presets_dir: Path | None = None) -> dict:
    """Load preset data for ``preset_name`` from YAML/YML/JSON files."""

    base_dir = presets_dir or _default_presets_dir()
    candidates = [
        base_dir / f"{preset_name}.yaml",
        base_dir / f"{preset_name}.yml",
        base_dir / f"{preset_name}.json",
    ]

    found = next((p for p in candidates if p.exists()), None)
    if found is None:
        raise ValueError(
            f"Preset {preset_name!r} was not found under {base_dir}. "
            "Expected one of: .yaml, .yml, .json"
        )

    if found.suffix == ".json":
        with found.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError(
                "YAML preset selected but PyYAML is not installed. "
                "Install PyYAML or provide presets as JSON."
            ) from exc

        with found.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)

    if not isinstance(loaded, Mapping):
        raise ValueError(f"Preset {preset_name!r} must deserialize to a mapping/dict.")
    return dict(loaded)


def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge ``overrides`` into ``base``.

    Merge rules:
      - dict + dict => recursive merge
      - lists => replaced wholesale by overrides
      - scalars => overrides win
    """

    merged = deepcopy(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(dict(current), dict(value))
        else:
            merged[key] = deepcopy(value)
    return merged


def _required(schema_path: str, mapping: Mapping, key: str) -> object:
    if key not in mapping:
        raise ValueError(f"Missing required config key: {schema_path}.{key}")
    return mapping[key]


def _required_nonempty_string(schema_path: str, mapping: Mapping, key: str) -> str:
    value = _required(schema_path, mapping, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required config key: {schema_path}.{key}")
    return value


def _warn_unknown_keys(mapping: Mapping, *, allowed: set[str], path: str) -> None:
    for key in mapping.keys():
        if key not in allowed:
            warnings.warn(
                f"Unknown config key: {path}.{key} (will be ignored by resolver validation)",
                UserWarning,
                stacklevel=3,
            )


def _validate_layer_list(layers: object) -> None:
    if not isinstance(layers, list):
        raise ValueError("system.detector.layers must be a list of layer dictionaries.")
    for idx, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise ValueError(f"system.detector.layers[{idx}] must be a mapping/dict.")
        if "name" not in layer:
            raise ValueError(f"Missing required config key: system.detector.layers[{idx}].name")


def _normalize_in_place(cfg: dict) -> None:
    source = cfg["system"]["source"]
    optics = cfg["system"]["optics"]

    source["n_lambda"] = int(source["n_lambda"])
    optics["psf_npix"] = int(optics["psf_npix"])
    optics["oversample"] = int(optics["oversample"])

    for key in ("wavelength_m", "bandwidth_m"):
        source[key] = float(source[key])


def _validate_schema(cfg: dict) -> None:
    allowed_top = {"system", "experiment"}
    _warn_unknown_keys(cfg, allowed=allowed_top, path="config")

    system = _required("config", cfg, "system")
    experiment = _required("config", cfg, "experiment")
    if not isinstance(system, Mapping):
        raise ValueError("config.system must be a mapping/dict.")
    if not isinstance(experiment, Mapping):
        raise ValueError("config.experiment must be a mapping/dict.")

    allowed_system = {"preset", "source", "optics", "detector"}
    _warn_unknown_keys(system, allowed=allowed_system, path="system")
    _required_nonempty_string("system", system, "preset")
    source = _required("system", system, "source")
    optics = _required("system", system, "optics")
    detector = _required("system", system, "detector")

    if not isinstance(source, Mapping):
        raise ValueError("system.source must be a mapping/dict.")
    if not isinstance(optics, Mapping):
        raise ValueError("system.optics must be a mapping/dict.")
    if not isinstance(detector, Mapping):
        raise ValueError("system.detector must be a mapping/dict.")

    source_allowed = {"kind", "wavelength_m", "bandwidth_m", "n_lambda"}
    optics_allowed = {"kind", "psf_npix", "oversample"}
    detector_allowed = {"model", "layers"}
    experiment_allowed = {"kind"}

    _warn_unknown_keys(source, allowed=source_allowed, path="system.source")
    _warn_unknown_keys(optics, allowed=optics_allowed, path="system.optics")
    _warn_unknown_keys(detector, allowed=detector_allowed, path="system.detector")
    _warn_unknown_keys(experiment, allowed=experiment_allowed, path="experiment")

    _required_nonempty_string("system.source", source, "kind")
    _required("system.source", source, "wavelength_m")
    _required("system.source", source, "bandwidth_m")
    _required("system.source", source, "n_lambda")

    _required_nonempty_string("system.optics", optics, "kind")
    _required("system.optics", optics, "psf_npix")
    _required("system.optics", optics, "oversample")

    _required_nonempty_string("system.detector", detector, "model")
    layers = _required("system.detector", detector, "layers")
    _validate_layer_list(layers)

    _required_nonempty_string("experiment", experiment, "kind")


def resolve_config(user_cfg: object, *, presets_dir: Path | None = None) -> dict:
    """Resolve strict nested config by loading preset and applying overrides."""

    user_cfg_dict = as_dict(user_cfg)
    if "system" not in user_cfg_dict:
        raise ValueError("Missing required config key: config.system")
    if not isinstance(user_cfg_dict["system"], Mapping):
        raise ValueError("config.system must be a mapping/dict.")

    preset_name = user_cfg_dict["system"].get("preset", None)
    if not isinstance(preset_name, str) or not preset_name:
        raise ValueError("Missing required config key: system.preset")

    preset_cfg = load_preset(preset_name, presets_dir=presets_dir)
    resolved = deep_merge(preset_cfg, user_cfg_dict)
    _validate_schema(resolved)
    _normalize_in_place(resolved)
    return resolved


def resolved_config_to_system_config(resolved_cfg: Mapping[str, object]):
    """Translate resolved nested config into system dataclasses used by binders."""

    from ..systems.three_plane import SheraThreePlaneConfig
    from ..systems.two_plane import SheraTwoPlaneConfig

    if not isinstance(resolved_cfg, Mapping):
        raise TypeError("resolved_cfg must be a mapping/dict.")

    system = resolved_cfg.get("system")
    if not isinstance(system, Mapping):
        raise TypeError("resolved_cfg.system must be a mapping/dict.")

    source = system.get("source")
    optics = system.get("optics")
    detector = system.get("detector")
    if not isinstance(source, Mapping) or not isinstance(optics, Mapping) or not isinstance(detector, Mapping):
        raise TypeError("resolved_cfg.system.{source,optics,detector} must be mappings/dicts.")

    optics_kind = str(optics.get("kind", ""))
    if optics_kind == "three_plane":
        cls = SheraThreePlaneConfig
    elif optics_kind == "two_plane":
        cls = SheraTwoPlaneConfig
    else:
        raise ValueError(
            f"Unsupported system.optics.kind {optics_kind!r}; expected 'three_plane' or 'two_plane'."
        )

    kwargs = {
        "system": dict(system),
        "wavelength_m": source.get("wavelength_m"),
        "bandwidth_m": source.get("bandwidth_m"),
        "n_lambda": source.get("n_lambda"),
        "psf_npix": optics.get("psf_npix"),
        "oversample": optics.get("oversample"),
    }
    if "pupil_npix" in optics:
        kwargs["pupil_npix"] = optics["pupil_npix"]
    if "detector_model" in {f.name for f in fields(cls)}:
        kwargs["detector_model"] = detector.get("model")

    # Pass through any known dataclass fields from optics/source blocks.
    known_fields = {f.name for f in fields(cls)}
    for block in (source, optics):
        for key, val in block.items():
            if key in known_fields and key not in kwargs:
                kwargs[key] = val

    return cls(**kwargs)


__all__ = [
    "as_dict",
    "deep_merge",
    "load_preset",
    "resolve_config",
    "resolved_config_to_system_config",
]
