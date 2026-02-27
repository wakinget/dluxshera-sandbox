"""Resolve preset-driven configuration blocks into validated runtime mappings.

This module is the configuration *resolver* for canonical dLuxShera workflows.
It combines three sources of information into a normalized, schema-checked
mapping:

1. Built-in preset files under ``src/dluxshera/data/system_presets`` and
   ``src/dluxshera/data/experiment_presets``.
2. User overrides loaded upstream from YAML/JSON (or provided directly as
   dictionaries/dataclasses).
3. Lightweight post-validation normalization (for example, coercing numeric
   fields to ``int``/``float`` so downstream builders do not need to repeat
   parsing logic).

Typical resolver flow:

1. Load a preset by name from disk.
2. Deep-merge user overrides into the preset (override values win).
3. Validate required keys and warn on unknown keys.
4. Normalize block fields where needed.
5. Return a resolved nested mapping.

The resolved shape is a dictionary containing ``system`` and/or ``experiment``
top-level blocks, depending on which blocks the caller supplied.

Separation of responsibilities:

* This module resolves and validates config *blocks*.
* Recipes/scripts (for example
  ``examples/recipes/canonical_astrometry.py``) decide which blocks are
  required for a workflow and how to consume the resolved values (binder/spec
  composition, inference execution, output handling, and so on).
"""

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

def _default_pupils_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "pupils"

def _default_offsets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "pixel_offsets"

def _default_prf_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "pixel_response"

def as_dict(cfg: object) -> dict:
    """Normalize supported config containers into a plain nested dictionary.

    Parameters
    ----------
    cfg : object
        Input configuration container. Supported forms include:

        - ``Mapping`` with ``system`` and/or ``experiment`` keys
        - Dataclass instances whose fields include ``system`` and/or
          ``experiment``
        - Arbitrary objects exposing ``.system`` and/or ``.experiment``
          attributes that are mappings

    Returns
    -------
    dict
        Deep-copied nested mapping containing one or both top-level blocks:
        ``{"system": {...}}``, ``{"experiment": {...}}``, or both.

    Notes
    -----
    This is the first normalization step in the resolver pipeline so later
    stages can operate on a single in-memory representation.

    Accepted top-level blocks are permissive: configs may provide ``system`` and/or
    ``experiment``. A :class:`TypeError` is raised only when neither block exists.
    """

    if isinstance(cfg, Mapping):
        data = deepcopy(dict(cfg))
        if "system" in data or "experiment" in data:
            return data

    if is_dataclass(cfg):
        data = asdict(cfg)
        if "system" in data or "experiment" in data:
            return data

    out: dict[str, dict] = {}
    if hasattr(cfg, "system"):
        system = getattr(cfg, "system")
        if isinstance(system, Mapping):
            out["system"] = deepcopy(dict(system))
    if hasattr(cfg, "experiment"):
        experiment = getattr(cfg, "experiment")
        if isinstance(experiment, Mapping):
            out["experiment"] = deepcopy(dict(experiment))
    if out:
        return out

    raise TypeError(
        "Config must provide at least one of 'system' or 'experiment'. "
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
    """Load a system preset file and return a single-block mapping.

    Parameters
    ----------
    name : str
        Preset name (file stem) to load from the system preset directory.
    presets_dir : Path | None, optional
        Override directory containing ``.yaml``, ``.yml``, or ``.json`` preset
        files. When omitted, the built-in system preset directory is used.

    Returns
    -------
    dict
        Mapping with exactly one top-level block:
        ``{"system": <system-mapping>}``.

    Notes
    -----
    System presets are strict by contract: they must define top-level
    ``system`` and must not include top-level ``experiment``.
    """

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
    """Load an experiment preset file and return a single-block mapping.

    Parameters
    ----------
    name : str
        Preset name (file stem) to load from the experiment preset directory.
    presets_dir : Path | None, optional
        Override directory containing ``.yaml``, ``.yml``, or ``.json`` preset
        files. When omitted, the built-in experiment preset directory is used.

    Returns
    -------
    dict
        Mapping with exactly one top-level block:
        ``{"experiment": <experiment-mapping>}``.

    Notes
    -----
    Experiment presets are strict by contract: they must define top-level
    ``experiment`` and must not include top-level ``system``.
    """

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
    """Recursively merge dictionaries using ``overrides``-wins semantics.

    Parameters
    ----------
    base : dict
        Baseline mapping, typically loaded from a preset.
    overrides : dict
        User-provided values to apply on top of ``base``.

    Returns
    -------
    dict
        New merged mapping. Inputs are not mutated.

    Notes
    -----
    If a key exists in both mappings and both values are mappings, merge
    recursively. Otherwise the value from ``overrides`` replaces ``base``.
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


def _normalize_system_in_place(system_cfg: dict) -> None:
    """Normalize validated ``system`` fields to stable numeric runtime types.

    This coercion happens after schema validation so downstream system builders
    can assume consistent ``int``/``float`` representations.
    """

    source = system_cfg["source"]
    optics = system_cfg["optics"]

    source["n_lambda"] = int(source["n_lambda"])
    optics["psf_npix"] = int(optics["psf_npix"])
    optics["oversample"] = int(optics["oversample"])

    for key in ("wavelength_m", "bandwidth_m"):
        source[key] = float(source[key])


def _validate_system_schema(system: object) -> None:
    """Apply strict validation for the ``system`` block.

    Required keys are enforced, and unknown keys emit warnings via allowlists.
    """

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
    optics_allowed = {
        "kind",
        "psf_npix",
        "oversample",
        "pupil_npix",
        "m1_diameter_m",
        "m2_diameter_m",
        "m1_focal_length_m",
        "m2_focal_length_m",
        "m1_m2_separation_m",
        "pixel_pitch_m",
        "n_struts",
        "strut_width_m",
        "strut_rotation_deg",
        "primary_noll_indices",
        "secondary_noll_indices",
        "dp_path",
        "dp_design_wavelength_m",
        "plate_scale_as_per_pix",
    }
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
    """Apply strict validation for the ``experiment`` block.

    Required keys are enforced, and unknown keys emit warnings via allowlists.
    """

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
    """Resolve a ``system`` block from preset + overrides.

    Parameters
    ----------
    system_cfg : dict
        User-provided ``system`` mapping. Must include ``preset`` and may add
        nested overrides.
    presets_dir : Path | None, optional
        Directory for system presets. Defaults to built-in preset data.

    Returns
    -------
    dict
        Resolved and normalized ``system`` mapping (without an outer wrapper).

    Notes
    -----
    Resolution steps are: load preset by ``system.preset``, deep-merge
    overrides, validate schema, then normalize numeric fields.
    """

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
    """Resolve an ``experiment`` block from preset + overrides.

    Parameters
    ----------
    experiment_cfg : dict
        User-provided ``experiment`` mapping. Must include ``preset`` and may
        add nested overrides.
    presets_dir : Path | None, optional
        Directory for experiment presets. Defaults to built-in preset data.

    Returns
    -------
    dict
        Resolved ``experiment`` mapping (without an outer wrapper).

    Notes
    -----
    Resolution steps are: load preset by ``experiment.preset``, deep-merge
    overrides, and validate schema.
    """

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
    """Resolve top-level config into nested ``system``/``experiment`` blocks.

    Parameters
    ----------
    user_cfg : object
        Config container accepted by :func:`as_dict`.
    system_presets_dir : Path | None, optional
        Override directory for system presets.
    experiment_presets_dir : Path | None, optional
        Override directory for experiment presets.

    Returns
    -------
    dict
        Nested mapping containing whichever resolved blocks were supplied by
        the user (``system`` and/or ``experiment``).

    Notes
    -----
    This orchestration layer resolves each block independently and does not
    impose workflow policy. Canonical recipes decide whether a block is
    required and how to interpret resolved values.

    This combiner is permissive at the top level and resolves whichever of
    ``system`` and ``experiment`` are present.
    """

    cfg = as_dict(user_cfg)
    resolved: dict[str, dict] = {}
    if "system" in cfg:
        resolved["system"] = resolve_system_config(cfg["system"], presets_dir=system_presets_dir)
    if "experiment" in cfg:
        resolved["experiment"] = resolve_experiment_config(
            cfg["experiment"],
            presets_dir=experiment_presets_dir,
        )
    if not resolved:
        raise ValueError("Config must provide at least one of 'system' or 'experiment'.")
    return resolved


def resolved_config_to_system_config(resolved_cfg: Mapping[str, object]):
    """Legacy bridge from resolved mappings to older system dataclass configs.

    Parameters
    ----------
    resolved_cfg : Mapping[str, object]
        Resolved nested configuration that includes a ``system`` block.

    Returns
    -------
    object
        Instance of a legacy system config dataclass selected from
        ``system.optics.kind``.

    Notes
    -----
    This helper is kept for older APIs and is not used by canonical,
    preset-driven workflows. Prefer passing resolved mappings directly into
    binder/spec composition in recipes/scripts. This bridge is intended for
    eventual deprecation.

    Deprecated: this translation helper exists for older APIs and is not used by
    canonical workflows.
    """

    warnings.warn(
        "resolved_config_to_system_config is deprecated and kept only as a legacy bridge.",
        DeprecationWarning,
        stacklevel=2,
    )

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
