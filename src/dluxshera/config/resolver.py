"""Strict nested configuration resolvers for system and experiment blocks."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
import json
import warnings


def _default_system_presets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "system_presets"


def _default_experiment_presets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "experiment_presets"


def as_dict(cfg: object) -> dict:
    """Convert supported config containers to a nested dict."""

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


def _load_preset_file(preset_name: str, base_dir: Path) -> dict:
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


def load_system_preset(name: str, *, presets_dir: Path | None = None) -> dict:
    """Load and return a system preset mapping containing only a ``system`` block."""

    loaded = _load_preset_file(name, presets_dir or _default_system_presets_dir())
    if "system" not in loaded:
        raise ValueError(f"System preset {name!r} must contain top-level 'system'.")
    if "experiment" in loaded:
        raise ValueError(f"System preset {name!r} must not contain top-level 'experiment'.")
    system = loaded["system"]
    if not isinstance(system, Mapping):
        raise ValueError(f"System preset {name!r} key 'system' must be a mapping/dict.")
    return {"system": dict(system)}


def load_experiment_preset(name: str, *, presets_dir: Path | None = None) -> dict:
    """Load and return an experiment preset mapping containing only ``experiment``."""

    loaded = _load_preset_file(name, presets_dir or _default_experiment_presets_dir())
    if "experiment" not in loaded:
        raise ValueError(f"Experiment preset {name!r} must contain top-level 'experiment'.")
    if "system" in loaded:
        raise ValueError(f"Experiment preset {name!r} must not contain top-level 'system'.")
    experiment = loaded["experiment"]
    if not isinstance(experiment, Mapping):
        raise ValueError(f"Experiment preset {name!r} key 'experiment' must be a mapping/dict.")
    return {"experiment": dict(experiment)}


def deep_merge(base: dict, overrides: dict) -> dict:
    """Deep-merge ``overrides`` into ``base``."""

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


def _normalize_system_in_place(system_cfg: dict) -> None:
    source = system_cfg["source"]
    optics = system_cfg["optics"]

    source["n_lambda"] = int(source["n_lambda"])
    optics["psf_npix"] = int(optics["psf_npix"])
    optics["oversample"] = int(optics["oversample"])

    for key in ("wavelength_m", "bandwidth_m"):
        source[key] = float(source[key])


def _validate_system_schema(system: object) -> None:
    if not isinstance(system, Mapping):
        raise ValueError("system must be a mapping/dict.")

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

    _warn_unknown_keys(source, allowed=source_allowed, path="system.source")
    _warn_unknown_keys(optics, allowed=optics_allowed, path="system.optics")
    _warn_unknown_keys(detector, allowed=detector_allowed, path="system.detector")

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


def _validate_experiment_schema(experiment: object) -> None:
    if not isinstance(experiment, Mapping):
        raise ValueError("experiment must be a mapping/dict.")

    allowed_experiment = {
        "preset",
        "kind",
        "seed",
        "optimizer",
        "init",
        "infer_keys",
        "priors",
        "outputs",
        "add_noise",
    }
    _warn_unknown_keys(experiment, allowed=allowed_experiment, path="experiment")
    _required_nonempty_string("experiment", experiment, "kind")


def resolve_system_config(system_cfg: dict, *, presets_dir: Path | None = None) -> dict:
    """Resolve ``system`` config from preset + overrides and validate strictly."""

    if not isinstance(system_cfg, Mapping):
        raise ValueError("system must be a mapping/dict.")
    preset_name = system_cfg.get("preset")
    if not isinstance(preset_name, str) or not preset_name:
        raise ValueError("Missing required config key: system.preset")

    preset_system = load_system_preset(preset_name, presets_dir=presets_dir)["system"]
    resolved = deep_merge(dict(preset_system), dict(system_cfg))
    _validate_system_schema(resolved)
    _normalize_system_in_place(resolved)
    return resolved


def resolve_experiment_config(experiment_cfg: dict, *, presets_dir: Path | None = None) -> dict:
    """Resolve ``experiment`` config from preset + overrides and validate strictly."""

    if not isinstance(experiment_cfg, Mapping):
        raise ValueError("experiment must be a mapping/dict.")
    preset_name = experiment_cfg.get("preset")
    if not isinstance(preset_name, str) or not preset_name:
        raise ValueError("Missing required config key: experiment.preset")

    preset_experiment = load_experiment_preset(preset_name, presets_dir=presets_dir)["experiment"]
    resolved = deep_merge(dict(preset_experiment), dict(experiment_cfg))
    _validate_experiment_schema(resolved)
    return resolved


def resolve_config(
    user_cfg: object,
    *,
    system_presets_dir: Path | None = None,
    experiment_presets_dir: Path | None = None,
) -> dict:
    """Resolve strict nested config via dedicated system and experiment resolvers."""

    cfg = as_dict(user_cfg)
    if "system" not in cfg:
        raise ValueError("Missing required config key: config.system")
    if "experiment" not in cfg:
        raise ValueError("Missing required config key: config.experiment")

    resolved_system = resolve_system_config(cfg["system"], presets_dir=system_presets_dir)
    resolved_experiment = resolve_experiment_config(
        cfg["experiment"],
        presets_dir=experiment_presets_dir,
    )
    return {"system": resolved_system, "experiment": resolved_experiment}


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

    known_fields = {f.name for f in fields(cls)}
    for block in (source, optics):
        for key, val in block.items():
            if key in known_fields and key not in kwargs:
                kwargs[key] = val

    return cls(**kwargs)


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
