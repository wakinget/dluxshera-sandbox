"""
Generate a Fisher-scaled one-parameter-at-a-time ML training dataset (V2).

This is the config-migrated version of the original V2 dataset generator.
It preserves the core dataset logic:

- emit the nominal sample once at run start,
- perturb exactly one parameter/component at a time,
- use mirrored nonzero sigma offsets,
- scale deltas by Fisher-diagonal parameter sigmas.

Prescription structure
----------------------
The preferred input is a YAML/JSON prescription with top-level `system` and
`experiment` blocks, following the same config pattern used by the canonical
recipes:

- `system`: preset selection plus any system overrides.
- `experiment`: dataset-specific controls such as sweep keys, sweep ranges,
  nominal parameter overrides, noise toggle, and outputs.

Usage examples
--------------
Dry-run the bundled YAML recipe:

    python work/experiments/generate_training_dataset_v2.py \
        --prescription work/experiments/generate_training_dataset_v2_template.yaml \
        --dry-run

Generate a dataset with the bundled YAML recipe:

    python work/experiments/generate_training_dataset_v2.py \
        --prescription work/experiments/generate_training_dataset_v2_template.yaml

Override the system preset from the CLI:

    python work/experiments/generate_training_dataset_v2.py \
        --prescription work/experiments/generate_training_dataset_v2_template.yaml \
        --system-preset SHERA_FLIGHT_3P \
        --dry-run

Run from the built-in presets without a prescription file:

    python work/experiments/generate_training_dataset_v2.py \
        --system-preset SHERA_FLIGHT_3P \
        --experiment-preset ML_TRAINING_DATA_V2 \
        --dry-run

Migration notes
---------------
- The YAML prescription supersedes the old JSON-only sweep override file.
- Config inputs are validated against the canonical `system` / `experiment`
  schema used by the current recipes; legacy field aliases are not translated.
- The output dataset layout remains the same: `manifest.json`,
  `samples.jsonl`, and per-sample FITS + JSON sidecars under `images/`.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import time
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from astropy.io import fits

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.optimization import (
    fim_theta,
    generate_fim_labels,
    make_binder_nll_fn,
)
from dluxshera.params.packing import build_index_map
from dluxshera.params.spec import ParamSpec
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec

JAX_ENABLE_X64 = True
PRINT_EVERY = 10
SCRIPT_VERSION = "v2"

DEFAULT_SYSTEM_PRESET = "SHERA_FLIGHT_3P"
DEFAULT_EXPERIMENT_PRESET = "ML_TRAINING_DATA_V2"

DEFAULT_SWEEP_KEYS = (
    "source.separation_as",
    "source.position_angle_deg",
    "source.x_position_as",
    "source.y_position_as",
    "source.log_flux_total",
    "source.contrast",
    "optics.plate_scale_as_per_pix",
    "optics.primary.zernike_coeffs_nm",
    "optics.secondary.zernike_coeffs_nm",
)

@dataclass(frozen=True)
class SweepConfig:
    min_sigma: float = 0.1
    max_sigma: float = 10.0
    n_magnitudes: int = 8
    spacing: str = "log"


DEFAULT_SWEEP_CONFIG = SweepConfig()


def _log(msg: str) -> None:
    print(f"[generate_training_dataset_v2] {msg}")


def _log_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _timestamp_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return value


def _strip_private_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            key: _strip_private_keys(value)
            for key, value in obj.items()
            if not str(key).startswith("_")
        }
    if isinstance(obj, list):
        return [_strip_private_keys(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_private_keys(item) for item in obj)
    return obj


def _normalize_optional_preset(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.lower() in {"", "none", "null"}:
        return None
    return stripped


def _resolve_preset_seed(raw_value: str | None, *, default_value: str | None) -> str | None:
    if raw_value is None:
        return default_value
    return _normalize_optional_preset(raw_value)


def _normalize_param_key_list(values: Any, *, field_name: str) -> list[str]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{field_name} must be a list of parameter keys.")

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_key in values:
        key = str(raw_key)
        if key in seen:
            raise ValueError(f"{field_name} contains duplicate key {key!r}.")
        normalized.append(key)
        seen.add(key)

    if not normalized:
        raise ValueError(f"{field_name} must contain at least one parameter key.")
    return normalized


def _normalize_keyed_mapping(values: Any, *, field_name: str) -> dict[str, Any]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError(f"{field_name} must be a mapping/dict.")
    return {str(key): copy.deepcopy(val) for key, val in values.items()}


def _resolve_path_relative_to_prescription(
    value: str | Path | None,
    *,
    prescription_path: Path | None,
    field_name: str,
) -> Path | None:
    if value is None:
        return None

    if isinstance(value, Path):
        path_value = value
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        path_value = Path(stripped)
    else:
        raise ValueError(f"{field_name} must be a path string or null.")

    if path_value.is_absolute():
        return path_value
    if prescription_path is None:
        return path_value.resolve()
    return (prescription_path.parent / path_value).resolve()


def _resolve_legacy_style_run_dir(outdir: str | Path | None, run_name: str | None) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    prefix = "ml_training_dataset_v2_"
    if outdir is None:
        suffix = run_name or _timestamp_tag()
        return repo_root / "Results" / f"{prefix}{suffix}"
    base = Path(outdir).expanduser().resolve()
    if run_name is None:
        return base / f"{prefix}{_timestamp_tag()}"
    return base / run_name


def _resolve_run_dir(
    *,
    cli_outdir: str | None,
    run_name: str | None,
    experiment_cfg: dict[str, Any],
    prescription_path: Path | None,
) -> tuple[Path, str]:
    if cli_outdir and cli_outdir.strip():
        return _resolve_legacy_style_run_dir(cli_outdir, run_name), "CLI --outdir"

    outputs_cfg = experiment_cfg.get("outputs", {}) or {}
    if not isinstance(outputs_cfg, Mapping):
        raise ValueError("experiment.outputs must be a mapping/dict when provided.")

    cfg_outdir = _resolve_path_relative_to_prescription(
        outputs_cfg.get("outdir"),
        prescription_path=prescription_path,
        field_name="experiment.outputs.outdir",
    )
    if cfg_outdir is not None:
        return _resolve_legacy_style_run_dir(cfg_outdir, run_name), "experiment.outputs.outdir"

    return _resolve_legacy_style_run_dir(None, run_name), "auto default"


def _git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _noise_model(
    rng_key: jax.Array,
    data: jax.Array,
    *,
    add_noise: bool,
) -> tuple[jax.Array, str, int | None]:
    if not add_noise:
        return data, "none", None
    rng_key, split_key = jr.split(rng_key)
    if np.min(np.asarray(data)) > 100:
        noisy = np.sqrt(np.asarray(data)) * jr.normal(split_key, data.shape) + data
        return noisy, "gaussian-approx", int(np.asarray(split_key)[0])
    noisy = jr.poisson(split_key, data)
    return noisy, "poisson", int(np.asarray(split_key)[0])


def _validate_fim_diag(fim_diag: np.ndarray, *, labels: list[str]) -> None:
    for idx, val in enumerate(fim_diag):
        if not np.isfinite(val) or val <= 0:
            label = labels[idx] if idx < len(labels) else f"index {idx}"
            warnings.warn(f"Invalid FIM diagonal entry for {label}: {val}.", RuntimeWarning)
            raise ValueError(
                f"FIM diagonal entry for {label} is invalid ({val}); cannot compute sigma scaling."
            )


def _write_fits(*, output_path: Path, image: np.ndarray, header_data: dict[str, Any]) -> None:
    header = fits.Header()
    for key, value in header_data.items():
        if value is None:
            continue
        if isinstance(value, tuple) and len(value) == 2:
            card_value, comment = value
            header.set(str(key).upper(), card_value, comment=str(comment))
        else:
            header.set(str(key).upper(), value)
    fits.PrimaryHDU(data=image, header=header).writeto(output_path, overwrite=True)


def generate_mirrored_sigma_offsets(
    *,
    min_sigma: float,
    max_sigma: float,
    n_magnitudes: int,
    spacing: str,
) -> list[float]:
    """Generate deterministic nonzero sigma offsets.

    Ordering is deterministic:
      negatives from largest magnitude to smallest, then
      positives from smallest magnitude to largest.
    """
    if n_magnitudes < 1:
        raise ValueError("n_magnitudes must be >= 1.")
    if min_sigma <= 0:
        raise ValueError("min_sigma must be > 0 for log spacing.")
    if max_sigma <= 0:
        raise ValueError("max_sigma must be > 0 for log spacing.")
    if min_sigma >= max_sigma:
        raise ValueError("min_sigma must be < max_sigma.")
    if spacing != "log":
        raise ValueError(f"Unsupported spacing {spacing!r}. Currently only 'log' is supported.")

    magnitudes = np.geomspace(min_sigma, max_sigma, num=n_magnitudes)
    negatives = [-float(v) for v in magnitudes[::-1]]
    positives = [float(v) for v in magnitudes]
    return negatives + positives


def _coerce_sweep_config(raw_cfg: Any, *, fallback: SweepConfig) -> SweepConfig:
    if raw_cfg is None:
        payload: dict[str, Any] = {}
    elif isinstance(raw_cfg, Mapping):
        payload = dict(raw_cfg)
    else:
        raise ValueError("Sweep config entries must be mappings/dicts.")

    cfg = SweepConfig(
        min_sigma=float(payload.get("min_sigma", fallback.min_sigma)),
        max_sigma=float(payload.get("max_sigma", fallback.max_sigma)),
        n_magnitudes=int(payload.get("n_magnitudes", fallback.n_magnitudes)),
        spacing=str(payload.get("spacing", fallback.spacing)),
    )
    _ = generate_mirrored_sigma_offsets(
        min_sigma=cfg.min_sigma,
        max_sigma=cfg.max_sigma,
        n_magnitudes=cfg.n_magnitudes,
        spacing=cfg.spacing,
    )
    return cfg


def _normalize_sweep_configs(
    *,
    sweep_keys: list[str] | None = None,
    infer_keys: list[str] | None = None,
    default_cfg: SweepConfig,
    overrides: dict[str, dict[str, Any]],
) -> dict[str, SweepConfig]:
    if sweep_keys is None:
        if infer_keys is None:
            raise ValueError("Either sweep_keys or infer_keys must be provided.")
        sweep_keys = infer_keys
    elif infer_keys is not None and list(sweep_keys) != list(infer_keys):
        raise ValueError("sweep_keys and infer_keys were both provided but differ.")
    normalized: dict[str, SweepConfig] = {}
    for key in sweep_keys:
        normalized[key] = _coerce_sweep_config(overrides.get(key), fallback=default_cfg)
    return normalized


def _build_sigma_summary(
    *,
    parameter_name: str,
    nominal_value: float,
    parameter_sigma: float,
    sweep_cfg: SweepConfig,
) -> dict[str, Any]:
    min_abs_delta = sweep_cfg.min_sigma * parameter_sigma
    max_abs_delta = sweep_cfg.max_sigma * parameter_sigma
    return {
        "parameter_name": parameter_name,
        "nominal_value": nominal_value,
        "parameter_sigma": parameter_sigma,
        "min_sigma": sweep_cfg.min_sigma,
        "max_sigma": sweep_cfg.max_sigma,
        "spacing": sweep_cfg.spacing,
        "n_magnitudes": sweep_cfg.n_magnitudes,
        "total_nonzero_samples": 2 * sweep_cfg.n_magnitudes,
        "min_abs_delta": min_abs_delta,
        "max_abs_delta": max_abs_delta,
    }


def compute_preview_counts(
    *,
    per_parameter_cfg: dict[str, SweepConfig],
    scalar_keys: list[str],
    zernike_component_counts: dict[str, int],
) -> dict[str, int]:
    perturbed = 0
    for key in scalar_keys:
        perturbed += 2 * per_parameter_cfg[key].n_magnitudes
    for key, n_components in zernike_component_counts.items():
        perturbed += n_components * (2 * per_parameter_cfg[key].n_magnitudes)
    return {"nominal": 1, "perturbed": perturbed, "total": 1 + perturbed}


def compute_expected_sample_counts(
    *,
    n_swept_components: int,
    n_magnitudes: int,
) -> dict[str, int]:
    """Return V2 sample counts for a uniform one-parameter sweep."""
    perturbed = int(n_swept_components) * (2 * int(n_magnitudes))
    return {"nominal": 1, "perturbed": perturbed, "total": 1 + perturbed}


def _select_first_text(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _validate_experiment_config(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(experiment_cfg, Mapping):
        raise ValueError("experiment must be a mapping/dict.")

    kind = str(experiment_cfg.get("kind", "ml_training_dataset_v2")).strip()
    if kind != "ml_training_dataset_v2":
        raise ValueError(
            "generate_training_dataset_v2.py requires experiment.kind = 'ml_training_dataset_v2'."
        )

    sweep_keys = _normalize_param_key_list(
        experiment_cfg.get("sweep_keys", DEFAULT_SWEEP_KEYS),
        field_name="experiment.sweep_keys",
    )

    outputs_cfg = copy.deepcopy(experiment_cfg.get("outputs", {}) or {})
    if not isinstance(outputs_cfg, dict):
        raise ValueError("experiment.outputs must be a mapping/dict when provided.")

    noise_cfg = copy.deepcopy(experiment_cfg.get("noise", {}) or {})
    if not isinstance(noise_cfg, dict):
        raise ValueError("experiment.noise must be a mapping/dict when provided.")
    if "enabled" in noise_cfg and "add_noise" not in noise_cfg:
        noise_cfg["add_noise"] = noise_cfg["enabled"]

    sweeps_cfg = copy.deepcopy(experiment_cfg.get("sweeps", {}) or {})
    if not isinstance(sweeps_cfg, dict):
        raise ValueError("experiment.sweeps must be a mapping/dict when provided.")

    default_sweep = _coerce_sweep_config(
        sweeps_cfg.get("default", {}),
        fallback=DEFAULT_SWEEP_CONFIG,
    )

    override_payload: dict[str, dict[str, Any]] = {}
    for raw_key, value in sweeps_cfg.items():
        if raw_key == "default":
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"experiment.sweeps.{raw_key} must be a mapping/dict.")
        override_payload[str(raw_key)] = dict(value)

    extras = sorted(set(override_payload) - set(sweep_keys))
    if extras:
        joined = ", ".join(extras)
        raise ValueError(
            "experiment.sweeps contains keys that are not present in experiment.sweep_keys: "
            f"{joined}"
        )

    nominal_values = _normalize_keyed_mapping(
        experiment_cfg.get("nominal_values", {}),
        field_name="experiment.nominal_values",
    )

    return {
        "kind": kind,
        "seed": int(experiment_cfg.get("seed", 0)),
        "notes": _select_first_text(experiment_cfg, ("notes", "note", "comment", "comments")),
        "sweep_keys": sweep_keys,
        "outputs": outputs_cfg,
        "noise": noise_cfg,
        "add_noise": bool(noise_cfg.get("add_noise", False)),
        "default_sweep": default_sweep,
        "sweep_overrides": override_payload,
        "nominal_values": nominal_values,
        "resolved_raw": copy.deepcopy(dict(experiment_cfg)),
    }


def _apply_cli_overrides(
    *,
    system_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    seed_override: int | None,
    add_noise_override: bool,
    exclude_secondary_zernikes: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system_copy = copy.deepcopy(system_cfg)
    experiment_copy = copy.deepcopy(experiment_cfg)

    if seed_override is not None:
        experiment_copy["seed"] = int(seed_override)

    if add_noise_override:
        experiment_copy["add_noise"] = True
        experiment_copy["noise"] = dict(experiment_copy.get("noise", {}))
        experiment_copy["noise"]["enabled"] = True
        experiment_copy["noise"]["add_noise"] = True

    if exclude_secondary_zernikes:
        optics_cfg = dict(system_copy.get("optics", {}) or {})
        optics_cfg["secondary_noll_indices"] = []
        system_copy["optics"] = optics_cfg

        experiment_copy["sweep_keys"] = [
            key for key in experiment_copy["sweep_keys"] if key != "optics.secondary.zernike_coeffs_nm"
        ]
        experiment_copy["sweep_overrides"] = {
            key: value
            for key, value in experiment_copy["sweep_overrides"].items()
            if key != "optics.secondary.zernike_coeffs_nm"
        }
        if "optics.secondary.zernike_coeffs_nm" in experiment_copy["nominal_values"]:
            experiment_copy["nominal_values"].pop("optics.secondary.zernike_coeffs_nm")

    return system_copy, experiment_copy


def _refresh_preserving_derived_keys(
    store: ParameterStore,
    *,
    preserved_keys: Iterable[str],
    spec: ParamSpec,
) -> ParameterStore:
    preserved_values: dict[str, Any] = {}
    for key in preserved_keys:
        if key not in spec:
            continue
        if spec.get(key).kind != "derived":
            continue
        try:
            preserved_values[key] = store.get(key)
        except KeyError:
            continue

    refreshed = store.refresh_derived(spec)
    if preserved_values:
        refreshed = refreshed.replace(preserved_values)
    return refreshed


def _infer_values(store: ParameterStore, keys: list[str]) -> dict[str, Any]:
    return {key: _serialize_value(store.get(key)) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Fisher-scaled forward-model training dataset sweep (V2)."
    )
    parser.add_argument(
        "--prescription",
        "--config",
        dest="prescription",
        type=Path,
        default=None,
        help="YAML/JSON prescription with top-level system and experiment blocks.",
    )
    parser.add_argument(
        "--system-preset",
        type=str,
        default=None,
        help=(
            "Optional system preset seed. When --prescription is omitted, "
            f"defaults to {DEFAULT_SYSTEM_PRESET!r}. Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--experiment-preset",
        type=str,
        default=None,
        help=(
            "Optional experiment preset seed. When --prescription is omitted, "
            f"defaults to {DEFAULT_EXPERIMENT_PRESET!r}. Use 'none' to disable."
        ),
    )
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional CLI override for experiment.seed.",
    )
    parser.add_argument(
        "--add-noise",
        action="store_true",
        help="Optional CLI override forcing experiment.noise.enabled = true.",
    )
    parser.add_argument(
        "--exclude-secondary-zernikes",
        action="store_true",
        help="Convenience override: remove secondary Zernikes from both system + sweep config.",
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    prescription_path = args.prescription.resolve() if args.prescription is not None else None
    system_preset = _resolve_preset_seed(
        args.system_preset,
        default_value=DEFAULT_SYSTEM_PRESET if prescription_path is None else None,
    )
    experiment_preset = _resolve_preset_seed(
        args.experiment_preset,
        default_value=DEFAULT_EXPERIMENT_PRESET if prescription_path is None else None,
    )

    user_cfg = load_user_config(
        config_path=prescription_path,
        system_preset=system_preset,
        experiment_preset=experiment_preset,
    )
    user_cfg = _strip_private_keys(user_cfg)
    resolved_cfg = resolve_config(user_cfg)

    system_cfg = resolved_cfg.get("system")
    experiment_raw = resolved_cfg.get("experiment")
    if system_cfg is None:
        raise ValueError("generate_training_dataset_v2 requires a resolved top-level 'system' block.")
    if experiment_raw is None:
        raise ValueError("generate_training_dataset_v2 requires a resolved top-level 'experiment' block.")

    experiment_cfg = _validate_experiment_config(experiment_raw)
    system_cfg, experiment_cfg = _apply_cli_overrides(
        system_cfg=system_cfg,
        experiment_cfg=experiment_cfg,
        seed_override=args.seed,
        add_noise_override=args.add_noise,
        exclude_secondary_zernikes=args.exclude_secondary_zernikes,
    )

    run_dir, run_dir_source = _resolve_run_dir(
        cli_outdir=args.outdir,
        run_name=args.run_name,
        experiment_cfg=experiment_cfg,
        prescription_path=prescription_path,
    )

    forward_spec = compose_forward_spec(system_cfg)
    sweep_keys = list(experiment_cfg["sweep_keys"])

    missing_keys = [key for key in sweep_keys if key not in forward_spec]
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(
            "The resolved system does not expose all requested sweep keys. "
            f"Missing from forward spec: {joined}"
        )

    nominal_values = dict(experiment_cfg["nominal_values"])
    invalid_nominal = [key for key in nominal_values if key not in forward_spec]
    if invalid_nominal:
        joined = ", ".join(invalid_nominal)
        raise ValueError(
            "experiment.nominal_values contains keys that are not present in the resolved system: "
            f"{joined}"
        )

    structural_nominal = sorted(forward_spec.structural_keys() & set(nominal_values))
    if structural_nominal:
        joined = ", ".join(structural_nominal)
        raise ValueError(
            "experiment.nominal_values may not override structural keys. "
            f"Move these into the top-level system block instead: {joined}"
        )

    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = base_store.refresh_derived(forward_spec)
    if nominal_values:
        base_store = base_store.replace(nominal_values)
        base_store = _refresh_preserving_derived_keys(
            base_store,
            preserved_keys=set(sweep_keys) | set(nominal_values),
            spec=forward_spec,
        )

    binder = SheraBinder(system_cfg, forward_spec, base_store)

    rng_key = jr.PRNGKey(experiment_cfg["seed"])
    data = binder.model()
    data, noise_mode, noise_seed = _noise_model(rng_key, data, add_noise=experiment_cfg["add_noise"])

    _log("Calculating Fisher information matrix; this can take ~30 seconds or more.")
    nll_loss_fn, theta_ref = make_binder_nll_fn(
        binder=binder,
        infer_keys=sweep_keys,
        data=data,
        var=data,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=base_store,
    )

    F = fim_theta(nll_loss_fn, theta_ref)
    fim_diag = np.asarray(jnp.diag(F))
    fim_labels = generate_fim_labels(sweep_keys, cfg=system_cfg, store=base_store)
    _validate_fim_diag(fim_diag, labels=fim_labels)

    inference_subspec = forward_spec.subset(sweep_keys)
    index_map = build_index_map(inference_subspec, base_store, theta=theta_ref)

    def sigma_for_key_component(param_key: str, component_index: int | None) -> float:
        for entry in index_map["entries"]:
            if entry["name"] != param_key:
                continue
            start = int(entry["start"])
            idx = start if component_index is None else start + component_index
            return float(1.0 / math.sqrt(fim_diag[idx]))
        raise KeyError(f"Missing FIM mapping for key {param_key}.")

    default_cfg = experiment_cfg["default_sweep"]
    per_parameter_cfg = _normalize_sweep_configs(
        sweep_keys=sweep_keys,
        default_cfg=default_cfg,
        overrides=experiment_cfg["sweep_overrides"],
    )

    zernike_keys = [
        key
        for key in sweep_keys
        if key in ("optics.primary.zernike_coeffs_nm", "optics.secondary.zernike_coeffs_nm")
    ]
    scalar_keys = [key for key in sweep_keys if key not in zernike_keys]
    optics_cfg = system_cfg.get("optics", {}) if isinstance(system_cfg, Mapping) else {}
    zernike_map = {
        "optics.primary.zernike_coeffs_nm": tuple(optics_cfg.get("primary_noll_indices") or ()),
        "optics.secondary.zernike_coeffs_nm": tuple(optics_cfg.get("secondary_noll_indices") or ()),
    }

    sigma_summaries: list[dict[str, Any]] = []
    _log_section("Resolved configuration")
    _log(f"Prescription path: {prescription_path if prescription_path is not None else '<presets only>'}")
    _log(f"System preset seed: {system_preset or '<none>'}")
    _log(f"Experiment preset seed: {experiment_preset or '<none>'}")
    _log(f"Resolved run directory: {run_dir} ({run_dir_source})")

    _log_section("Fisher sigma summary")
    _log(
        "parameter/component | nominal | parameter_sigma | sigma_range | delta_range | nonzero_samples"
    )
    for key in scalar_keys:
        sigma = sigma_for_key_component(key, None)
        nominal = float(base_store.get(key))
        sweep_cfg = per_parameter_cfg[key]
        entry = _build_sigma_summary(
            parameter_name=key,
            nominal_value=nominal,
            parameter_sigma=sigma,
            sweep_cfg=sweep_cfg,
        )
        sigma_summaries.append(entry)
        _log(
            f"{key} | {nominal:.6g} | {sigma:.6g} | "
            f"[{sweep_cfg.min_sigma:.6g},{sweep_cfg.max_sigma:.6g}] | "
            f"[{entry['min_abs_delta']:.6g},{entry['max_abs_delta']:.6g}] | "
            f"{entry['total_nonzero_samples']}"
        )

    for key in zernike_keys:
        coeffs = np.asarray(base_store.get(key))
        sweep_cfg = per_parameter_cfg[key]
        for idx in range(coeffs.size):
            sigma = sigma_for_key_component(key, idx)
            nominal = float(coeffs[idx])
            noll_idx = int(zernike_map[key][idx]) if idx < len(zernike_map[key]) else None
            name = f"{key}[{idx}]" if noll_idx is None else f"{key}[{idx}]_noll{noll_idx}"
            entry = _build_sigma_summary(
                parameter_name=name,
                nominal_value=nominal,
                parameter_sigma=sigma,
                sweep_cfg=sweep_cfg,
            )
            entry["base_parameter_key"] = key
            entry["component_index"] = idx
            entry["noll_index"] = noll_idx
            sigma_summaries.append(entry)
            _log(
                f"{name} | {nominal:.6g} | {sigma:.6g} | "
                f"[{sweep_cfg.min_sigma:.6g},{sweep_cfg.max_sigma:.6g}] | "
                f"[{entry['min_abs_delta']:.6g},{entry['max_abs_delta']:.6g}] | "
                f"{entry['total_nonzero_samples']}"
            )

    preview_counts = compute_preview_counts(
        per_parameter_cfg=per_parameter_cfg,
        scalar_keys=scalar_keys,
        zernike_component_counts={key: int(np.asarray(base_store.get(key)).size) for key in zernike_keys},
    )
    _log_section("Sweep preview")
    _log(
        "Preview counts: "
        f"nominal={preview_counts['nominal']} "
        f"perturbed={preview_counts['perturbed']} "
        f"total={preview_counts['total']}"
    )

    if args.dry_run:
        _log("Dry run enabled; exiting before image/output generation.")
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    runtime_resolved_cfg = {
        "system": copy.deepcopy(system_cfg),
        "experiment": {
            "kind": experiment_cfg["kind"],
            "seed": experiment_cfg["seed"],
            "notes": experiment_cfg["notes"],
            "sweep_keys": list(sweep_keys),
            "noise": copy.deepcopy(experiment_cfg["noise"]),
            "outputs": copy.deepcopy(experiment_cfg["outputs"]),
            "nominal_values": copy.deepcopy(nominal_values),
            "sweeps": {
                "default": asdict(default_cfg),
                **{key: asdict(value) for key, value in per_parameter_cfg.items()},
            },
        },
    }

    (run_dir / "prescription_input.json").write_text(
        json.dumps(_serialize_value(user_cfg), indent=2),
        encoding="utf-8",
    )
    (run_dir / "prescription_resolved.json").write_text(
        json.dumps(_serialize_value(runtime_resolved_cfg), indent=2),
        encoding="utf-8",
    )

    baseline_infer = _infer_values(base_store, sweep_keys)
    manifest = {
        "script": "generate_training_dataset_v2.py",
        "version": SCRIPT_VERSION,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "prescription_path": str(prescription_path) if prescription_path is not None else None,
        "requested_system_preset": system_preset,
        "requested_experiment_preset": experiment_preset,
        "resolved_system_preset": system_cfg.get("preset") if isinstance(system_cfg, Mapping) else None,
        "resolved_experiment_preset": (
            experiment_raw.get("preset") if isinstance(experiment_raw, Mapping) else None
        ),
        "config_id": system_cfg.get("preset", "custom_system") if isinstance(system_cfg, Mapping) else "custom_system",
        "git_commit": _git_commit(),
        "notes": experiment_cfg["notes"],
        "parameters": sweep_keys,
        "nominal_values": baseline_infer,
        "nominal_overrides": _serialize_value(nominal_values),
        "sweep_configuration": {
            "defaults": asdict(default_cfg),
            "per_parameter": {key: asdict(value) for key, value in per_parameter_cfg.items()},
            "sigma_ordering": "negative largest->smallest then positive smallest->largest",
            "nonzero_only": True,
            "nominal_sample_generated_once": True,
        },
        "fisher_sigma_summary": sigma_summaries,
        "noise": {
            "enabled": bool(experiment_cfg["add_noise"]),
            "mode": noise_mode,
            "seed": experiment_cfg["seed"],
            "realization_seed": noise_seed,
            "config": _serialize_value(experiment_cfg["noise"]),
        },
    }

    samples_path = run_dir / "samples.jsonl"
    samples_path.write_text("", encoding="utf-8")

    sample_id = 0
    nominal_count = 0
    perturbed_count = 0

    def emit_sample(
        *,
        is_nominal: bool,
        sweep_parameter: str | None,
        component_index: int | None,
        noll_index: int | None,
        sweep_index: int | None,
        sigma_offset: float | None,
        abs_sigma_offset: float | None,
        parameter_sigma: float | None,
        delta_value: float,
        nominal_value: float | None,
        parameter_value: float | None,
        spacing_kind: str | None,
        applied_store: ParameterStore,
    ) -> None:
        nonlocal nominal_count, perturbed_count, sample_id

        sample_id += 1
        sample_tag = f"sample_{sample_id:06d}"
        fits_path = images_dir / f"{sample_tag}.fits"
        meta_path = images_dir / f"{sample_tag}.json"

        model = binder.model(
            binder.strip_structural(applied_store)
        )
        model_np = np.asarray(model)

        _write_fits(
            output_path=fits_path,
            image=model_np,
            header_data={
                "SAMPLEID": (sample_id, "Training dataset sample id"),
                "ISNOM": (bool(is_nominal), "Nominal baseline sample"),
                "SWEEPKEY": (sweep_parameter, "Swept parameter key"),
                "COMPIDX": (component_index, "Vector component index"),
                "NOLL": (noll_index, "Zernike Noll index"),
                "SWPIDX": (sweep_index, "Per-parameter sweep index"),
                "DELSIG": (sigma_offset, "Delta in sigma units"),
                "ABSSIG": (abs_sigma_offset, "Absolute sigma offset"),
                "PRMSIG": (parameter_sigma, "Fisher-derived parameter sigma"),
                "DELVAL": (delta_value, "Additive delta applied to nominal"),
                "APPLVAL": (parameter_value, "Nominal + delta"),
                "NOISE": (bool(experiment_cfg["add_noise"]), "Noise added to image"),
                "SEED": (experiment_cfg["seed"], "Base RNG seed"),
            },
        )

        sample_meta = {
            "sample_id": sample_id,
            "sample_tag": sample_tag,
            "is_nominal": is_nominal,
            "sweep_parameter": sweep_parameter,
            "component_index": component_index,
            "noll_index": noll_index,
            "sweep_index": sweep_index,
            "sigma_offset": sigma_offset,
            "abs_sigma_offset": abs_sigma_offset,
            "parameter_sigma": parameter_sigma,
            "delta_value": delta_value,
            "parameter_value": parameter_value,
            "nominal_value": nominal_value,
            "spacing_kind": spacing_kind,
            "values": _infer_values(applied_store, sweep_keys),
        }
        meta_path.write_text(json.dumps(_serialize_value(sample_meta), indent=2), encoding="utf-8")
        with samples_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _serialize_value(
                        {
                            "sample_id": sample_id,
                            "sample_tag": sample_tag,
                            "fits_path": str(fits_path.relative_to(run_dir)),
                            "metadata_path": str(meta_path.relative_to(run_dir)),
                            **sample_meta,
                        }
                    )
                )
                + "\n"
            )

        if is_nominal:
            nominal_count += 1
        else:
            perturbed_count += 1

    _log_section("Dataset generation")
    t0 = time.perf_counter()
    emit_sample(
        is_nominal=True,
        sweep_parameter=None,
        component_index=None,
        noll_index=None,
        sweep_index=None,
        sigma_offset=None,
        abs_sigma_offset=None,
        parameter_sigma=None,
        delta_value=0.0,
        nominal_value=None,
        parameter_value=None,
        spacing_kind=None,
        applied_store=base_store,
    )

    def maybe_log_progress(current_key: str, sweep_index: int, applied_value: float) -> None:
        if sample_id % PRINT_EVERY != 0:
            return
        elapsed = time.perf_counter() - t0
        rate = sample_id / elapsed if elapsed > 0 else 0.0
        _log(
            f"Progress: sample={sample_id} key={current_key} idx={sweep_index} "
            f"applied={applied_value:.6g} ({rate:.2f} samples/s)"
        )

    for key in scalar_keys:
        nominal = float(base_store.get(key))
        parameter_sigma = sigma_for_key_component(key, None)
        sweep_cfg = per_parameter_cfg[key]
        offsets = generate_mirrored_sigma_offsets(
            min_sigma=sweep_cfg.min_sigma,
            max_sigma=sweep_cfg.max_sigma,
            n_magnitudes=sweep_cfg.n_magnitudes,
            spacing=sweep_cfg.spacing,
        )
        for sweep_index, sigma_offset in enumerate(offsets):
            delta_value = float(sigma_offset) * parameter_sigma
            parameter_value = nominal + delta_value
            store_delta = base_store.replace({key: parameter_value})
            store_delta = _refresh_preserving_derived_keys(
                store_delta,
                preserved_keys=sweep_keys,
                spec=forward_spec,
            )
            emit_sample(
                is_nominal=False,
                sweep_parameter=key,
                component_index=None,
                noll_index=None,
                sweep_index=sweep_index,
                sigma_offset=float(sigma_offset),
                abs_sigma_offset=abs(float(sigma_offset)),
                parameter_sigma=parameter_sigma,
                delta_value=delta_value,
                nominal_value=nominal,
                parameter_value=parameter_value,
                spacing_kind=sweep_cfg.spacing,
                applied_store=store_delta,
            )
            maybe_log_progress(key, sweep_index, parameter_value)

    for key in zernike_keys:
        coeffs = np.asarray(base_store.get(key))
        sweep_cfg = per_parameter_cfg[key]
        offsets = generate_mirrored_sigma_offsets(
            min_sigma=sweep_cfg.min_sigma,
            max_sigma=sweep_cfg.max_sigma,
            n_magnitudes=sweep_cfg.n_magnitudes,
            spacing=sweep_cfg.spacing,
        )
        for idx in range(coeffs.size):
            nominal = float(coeffs[idx])
            parameter_sigma = sigma_for_key_component(key, idx)
            noll_index = int(zernike_map[key][idx]) if idx < len(zernike_map[key]) else None
            for sweep_index, sigma_offset in enumerate(offsets):
                delta_value = float(sigma_offset) * parameter_sigma
                parameter_value = nominal + delta_value
                updated = coeffs.copy()
                updated[idx] = parameter_value
                store_delta = base_store.replace({key: updated})
                store_delta = _refresh_preserving_derived_keys(
                    store_delta,
                    preserved_keys=sweep_keys,
                    spec=forward_spec,
                )
                emit_sample(
                    is_nominal=False,
                    sweep_parameter=key,
                    component_index=idx,
                    noll_index=noll_index,
                    sweep_index=sweep_index,
                    sigma_offset=float(sigma_offset),
                    abs_sigma_offset=abs(float(sigma_offset)),
                    parameter_sigma=parameter_sigma,
                    delta_value=delta_value,
                    nominal_value=nominal,
                    parameter_value=parameter_value,
                    spacing_kind=sweep_cfg.spacing,
                    applied_store=store_delta,
                )
                maybe_log_progress(f"{key}[{idx}]", sweep_index, parameter_value)

    manifest["counts"] = {
        "nominal_samples": nominal_count,
        "perturbed_samples": perturbed_count,
        "total_samples": sample_id,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(_serialize_value(manifest), indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0
    _log_section("Run summary")
    _log(f"Manifest: {run_dir / 'manifest.json'}")
    _log(f"Samples index: {samples_path}")
    _log(f"Nominal samples: {nominal_count}")
    _log(f"Perturbed samples: {perturbed_count}")
    _log(f"Total samples: {sample_id}")
    _log(f"Elapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
