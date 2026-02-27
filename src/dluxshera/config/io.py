from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

__all__ = [
    "deep_merge",
    "load_config_file",
    "load_user_config",
    "load_system_preset",
    "load_experiment_preset",
]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge two mappings and return a new dict.

    Rules
    -----
    - Mappings are merged recursively.
    - Non-mapping values replace the base value.
    - Inputs are not mutated.
    """
    out: dict[str, Any] = deepcopy(dict(base))
    for key, val in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(val, Mapping):
            out[key] = deep_merge(out[key], val)  # type: ignore[arg-type]
        else:
            out[key] = deepcopy(val)
    return out


def _validate_top_level_blocks(cfg: Mapping[str, Any], *, context: str) -> None:
    """Validate the canonical top-level contract for config mappings."""
    has_system = "system" in cfg
    has_experiment = "experiment" in cfg
    if not (has_system or has_experiment):
        raise ValueError(
            f"{context}: config must contain at least one top-level block: "
            "'system' and/or 'experiment'."
        )
    if has_system and not isinstance(cfg["system"], Mapping):
        raise ValueError(f"{context}: top-level 'system' must be a mapping/dict.")
    if has_experiment and not isinstance(cfg["experiment"], Mapping):
        raise ValueError(f"{context}: top-level 'experiment' must be a mapping/dict.")


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a YAML/JSON config file as a mapping.

    Parameters
    ----------
    path : Path
        Path to a ``.yaml``, ``.yml``, or ``.json`` file.

    Returns
    -------
    dict[str, Any]
        Parsed mapping. The top-level must deserialize to a dict-like mapping.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is unsupported or the parsed payload is not a mapping.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            loaded = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ValueError(
                    "YAML config selected but PyYAML is not installed. "
                    "Install PyYAML or provide configs as JSON."
                ) from exc
            loaded = yaml.safe_load(handle)
        else:
            raise ValueError("Config file must be YAML (.yaml/.yml) or JSON (.json).")

    if not isinstance(loaded, Mapping):
        raise ValueError("Config file must deserialize to a mapping/dict.")
    return dict(loaded)


def load_user_config(
    *,
    config_path: Path | None,
    system_preset: str | None,
    experiment_preset: str | None,
) -> dict[str, Any]:
    """Build a user config mapping suitable for ``resolve_config()``.

    This helper **enforces the nested schema** used by canonical workflows and
    intentionally does **not** support legacy flat configs.

    Precedence
    ----------
    - Preset selections seed the returned mapping as:
        - ``{"system": {"preset": <name>}}``
        - ``{"experiment": {"preset": <name>}}``
    - If ``config_path`` is provided, it is deep-merged over the preset seeds.

    The merged result must contain at least one of top-level ``system`` or
    ``experiment`` blocks.
    """
    base: dict[str, Any] = {}

    if system_preset is not None:
        base["system"] = {"preset": system_preset}

    if experiment_preset is not None:
        base["experiment"] = {"preset": experiment_preset}

    if config_path is None:
        if base:
            _validate_top_level_blocks(base, context="load_user_config")
        return base

    loaded = load_config_file(config_path)
    merged = deep_merge(base, loaded)

    _validate_top_level_blocks(merged, context=f"load_user_config({config_path})")
    return merged


def _default_presets_root() -> Path:
    """Default root directory for built-in presets."""
    return Path(__file__).resolve().parent / "presets"


def _default_system_presets_dir() -> Path:
    return _default_presets_root() / "system"


def _default_experiment_presets_dir() -> Path:
    return _default_presets_root() / "experiment"


def _load_preset_file(preset_name: str, base_dir: Path) -> dict[str, Any]:
    """Load a preset mapping from ``base_dir`` by trying YAML/JSON extensions."""
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

    loaded = load_config_file(found)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Preset {preset_name!r} must deserialize to a mapping/dict.")
    return dict(loaded)


def load_system_preset(name: str, *, presets_dir: Path | None = None) -> dict[str, Any]:
    """Load a system preset and return a single-block mapping: ``{'system': ...}``.

    Contract
    --------
    - Preset file must contain top-level ``system``.
    - Preset file must **not** contain top-level ``experiment``.
    """
    base_dir = presets_dir or _default_system_presets_dir()
    loaded = _load_preset_file(name, base_dir)

    if "system" not in loaded:
        raise ValueError(f"System preset {name!r} must contain top-level 'system'.")
    if "experiment" in loaded:
        raise ValueError(
            f"System preset {name!r} must not contain top-level 'experiment'."
        )

    system = loaded["system"]
    if not isinstance(system, Mapping):
        raise ValueError(f"System preset {name!r} key 'system' must be a mapping/dict.")

    return {"system": dict(system)}


def load_experiment_preset(
    name: str, *, presets_dir: Path | None = None
) -> dict[str, Any]:
    """Load an experiment preset and return a single-block mapping: ``{'experiment': ...}``.

    Contract
    --------
    - Preset file must contain top-level ``experiment``.
    - Preset file must **not** contain top-level ``system``.
    """
    base_dir = presets_dir or _default_experiment_presets_dir()
    loaded = _load_preset_file(name, base_dir)

    if "experiment" not in loaded:
        raise ValueError(
            f"Experiment preset {name!r} must contain top-level 'experiment'."
        )
    if "system" in loaded:
        raise ValueError(
            f"Experiment preset {name!r} must not contain top-level 'system'."
        )

    experiment = loaded["experiment"]
    if not isinstance(experiment, Mapping):
        raise ValueError(
            f"Experiment preset {name!r} key 'experiment' must be a mapping/dict."
        )

    return {"experiment": dict(experiment)}