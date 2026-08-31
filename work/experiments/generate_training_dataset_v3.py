"""
Generate a plan-first Fisher-scaled ML training dataset (V3).

V3 extends the V2 one-parameter atlas into two dataset families:

- all-pairs 2D perturbation grids over scalarized source/optics parameters,
- sparse random mixtures intended for held-out evaluation.

The script emphasizes deterministic plan generation, dry-run inspection, and
stable metadata.  Dry runs resolve the system, compute Fisher-diagonal scales,
write plan artifacts, and stop before FITS rendering.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import time
import warnings
from collections.abc import Iterable, Mapping, Sequence
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
from dluxshera.params.packing import build_index_map
from dluxshera.params.spec import ParamSpec
from dluxshera.params.store import ParameterStore

JAX_ENABLE_X64 = True
SCRIPT_VERSION = "v3.0-plan-first"
DEFAULT_SYSTEM_PRESET = "SHERA_FLIGHT_3P_SIMPLE"
DEFAULT_EXPERIMENT_PRESET = None
DEFAULT_RUN_NAME = "ml_training_v3_50ms_pairgrid_sparse"

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

REGISTRATION_NUISANCE_KEYS = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)


@dataclass(frozen=True)
class SweepConfig:
    min_sigma: float = 1.0
    max_sigma: float = 1_000.0
    n_magnitudes: int = 10
    spacing: str = "log"


@dataclass(frozen=True)
class ScalarParameter:
    label: str
    base_key: str
    component_index: int | None
    nominal_value: float
    parameter_sigma: float
    sweep_source_key: str
    sweep_config: SweepConfig
    min_abs_delta: float
    max_abs_delta: float
    units: str | None = None
    display_label: str | None = None
    group: str | None = None
    noll_index: int | None = None


@dataclass(frozen=True)
class ResumeState:
    start_sample_index: int
    retained_rows: tuple[dict[str, Any], ...]
    valid_prefix_count: int
    cleanup_group_key: str | None
    cleanup_group_start_index: int
    cleanup_sample_count: int
    reason: str


DEFAULT_SWEEP_CONFIG = SweepConfig()


def _log(msg: str) -> None:
    print(f"[generate_training_dataset_v3] {msg}")


def _log_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _timestamp_tag() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_value(val) for key, val in value.items()}
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


def _resolve_run_dir(
    *,
    cli_outdir: str | None,
    run_name: str | None,
    experiment_cfg: Mapping[str, Any],
    prescription_path: Path | None,
) -> tuple[Path, str]:
    chosen_run_name = run_name or str(
        ((experiment_cfg.get("outputs", {}) or {}).get("run_name") if isinstance(experiment_cfg.get("outputs", {}), Mapping) else "")
        or DEFAULT_RUN_NAME
    )
    if cli_outdir and cli_outdir.strip():
        return Path(cli_outdir).expanduser().resolve() / chosen_run_name, "CLI --outdir"

    outputs_cfg = experiment_cfg.get("outputs", {}) or {}
    if not isinstance(outputs_cfg, Mapping):
        raise ValueError("experiment.outputs must be a mapping/dict when provided.")
    cfg_outdir = _resolve_path_relative_to_prescription(
        outputs_cfg.get("outdir"),
        prescription_path=prescription_path,
        field_name="experiment.outputs.outdir",
    )
    if cfg_outdir is not None:
        return cfg_outdir / chosen_run_name, "experiment.outputs.outdir"
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "Results" / "ML Training Datasets" / f"{chosen_run_name}_{_timestamp_tag()}", "auto default"


def _git_info() -> dict[str, Any]:
    import subprocess

    info: dict[str, Any] = {}
    for key, cmd in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    }.items():
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            info[key] = None
        else:
            info[key] = result.stdout.strip() or None
    return info


def _validate_fim_diag(fim_diag: np.ndarray, *, labels: list[str]) -> None:
    for idx, val in enumerate(fim_diag):
        if not np.isfinite(val) or val <= 0:
            label = labels[idx] if idx < len(labels) else f"index {idx}"
            warnings.warn(f"Invalid FIM diagonal entry for {label}: {val}.", RuntimeWarning)
            raise ValueError(
                f"FIM diagonal entry for {label} is invalid ({val}); cannot compute sigma scaling."
            )


def _write_fits(*, output_path: Path, image: np.ndarray, header_data: Mapping[str, Any]) -> None:
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
    """Generate V2-compatible mirrored nonzero sigma offsets."""
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
    return [-float(v) for v in magnitudes[::-1]] + [float(v) for v in magnitudes]


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
    sweep_keys: Sequence[str],
    default_cfg: SweepConfig,
    overrides: Mapping[str, Any],
) -> dict[str, SweepConfig]:
    """Return per-base-key V2 sweep configs with overrides applied."""
    return {
        key: _coerce_sweep_config(overrides.get(key), fallback=default_cfg)
        for key in sweep_keys
    }


def _dedupe_preserve_order(keys: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw_key in keys:
        key = str(raw_key)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _nuisance_uniform_sampling_keys(datasets_cfg: Mapping[str, Any]) -> list[str]:
    nuisance_cfg = datasets_cfg.get("nuisance_replicates", {}) or {}
    sampling_cfg = nuisance_cfg.get("sampling", {}) or {}
    mode = str(sampling_cfg.get("mode", "uniform_from_sweeps"))
    if mode != "uniform_from_sweeps":
        return []
    return _dedupe_preserve_order(nuisance_cfg.get("keys", REGISTRATION_NUISANCE_KEYS))


def _resolve_sweep_for_label(
    label: str,
    *,
    base_key: str,
    per_parameter_cfg: Mapping[str, SweepConfig],
    default_cfg: SweepConfig,
) -> SweepConfig:
    """Resolve the V2-style sweep config controlling a scalarized V3 label."""
    if label in per_parameter_cfg:
        return per_parameter_cfg[label]
    if base_key in per_parameter_cfg:
        return per_parameter_cfg[base_key]
    return default_cfg


def _validate_experiment_config(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(experiment_cfg, Mapping):
        raise ValueError("experiment must be a mapping/dict.")
    kind = str(experiment_cfg.get("kind", "ml_training_dataset_v3")).strip()
    if kind != "ml_training_dataset_v3":
        raise ValueError(
            "generate_training_dataset_v3.py requires experiment.kind = 'ml_training_dataset_v3'."
        )
    sweep_keys = _normalize_param_key_list(
        experiment_cfg.get("sweep_keys", DEFAULT_SWEEP_KEYS),
        field_name="experiment.sweep_keys",
    )
    sweeps_cfg = copy.deepcopy(experiment_cfg.get("sweeps", {}) or {})
    if not isinstance(sweeps_cfg, dict):
        raise ValueError("experiment.sweeps must be a mapping/dict when provided.")
    default_sweep = _coerce_sweep_config(sweeps_cfg.get("default", {}), fallback=DEFAULT_SWEEP_CONFIG)
    sweep_overrides: dict[str, dict[str, Any]] = {}
    for raw_key, value in sweeps_cfg.items():
        if raw_key == "default":
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"experiment.sweeps.{raw_key} must be a mapping/dict.")
        sweep_overrides[str(raw_key)] = dict(value)
    noise_cfg = copy.deepcopy(experiment_cfg.get("noise", {}) or {})
    if not isinstance(noise_cfg, dict):
        raise ValueError("experiment.noise must be a mapping/dict when provided.")
    if "enabled" in noise_cfg and "add_noise" not in noise_cfg:
        noise_cfg["add_noise"] = noise_cfg["enabled"]

    datasets_cfg = copy.deepcopy(experiment_cfg.get("datasets", {}) or {})
    if not isinstance(datasets_cfg, dict):
        raise ValueError("experiment.datasets must be a mapping/dict when provided.")
    datasets_cfg.setdefault("pair_grid", {})
    datasets_cfg.setdefault("nuisance_replicates", {})
    datasets_cfg.setdefault("sparse_mixture", {})

    pair_cfg = datasets_cfg["pair_grid"] or {}
    pair_cfg.setdefault("enabled", True)
    pair_cfg.setdefault("include_all_pairs", True)
    pair_cfg.setdefault("level_mode", "symmetric_grid_from_sweeps")
    pair_cfg.setdefault("grid_size", 11)
    pair_cfg.setdefault("include_zero", True)
    pair_cfg.setdefault("pair_order", "upper_triangle")
    pair_cfg.setdefault("amplitude_scale", "fisher_sigma")
    pair_cfg.setdefault("include_self_pairs", False)

    nuisance_cfg = datasets_cfg["nuisance_replicates"] or {}
    nuisance_cfg.setdefault("enabled", True)
    nuisance_cfg.setdefault("include_nominal", True)
    nuisance_cfg.setdefault("n_random", 3)
    nuisance_cfg.setdefault("keys", list(REGISTRATION_NUISANCE_KEYS))
    nuisance_cfg.setdefault("sampling", {"mode": "uniform_from_sweeps"})
    nuisance_cfg.setdefault("collision_policy", "skip_if_key_is_controlled_axis")

    sparse_cfg = datasets_cfg["sparse_mixture"] or {}
    sparse_cfg.setdefault("enabled", True)
    sparse_cfg.setdefault("split", "test")
    sparse_cfg.setdefault("n_samples", 1000)
    sparse_cfg.setdefault("active_count_probs", {1: 0.25, 2: 0.50, 3: 0.25})
    sparse_cfg.setdefault("amplitude_sampling", {"mode": "uniform_from_sweeps", "signed": True})
    sparse_cfg.setdefault("nuisance", {"enabled": True})
    sparse_cfg.setdefault("noise", {"enabled": False})

    datasets_cfg["pair_grid"] = pair_cfg
    datasets_cfg["nuisance_replicates"] = nuisance_cfg
    datasets_cfg["sparse_mixture"] = sparse_cfg

    nuisance_sweep_keys = _nuisance_uniform_sampling_keys(datasets_cfg)
    allowed_sweep_keys = set(sweep_keys) | set(nuisance_sweep_keys)
    extras = sorted(set(sweep_overrides) - allowed_sweep_keys)
    if extras:
        raise ValueError(
            "experiment.sweeps contains keys that are not used by experiment.sweep_keys or "
            "datasets.nuisance_replicates sampling: " + ", ".join(extras)
        )

    return {
        "kind": kind,
        "seed": int(experiment_cfg.get("seed", 0)),
        "notes": str(experiment_cfg.get("notes", "") or ""),
        "sweep_keys": sweep_keys,
        "nuisance_sweep_keys": nuisance_sweep_keys,
        "outputs": copy.deepcopy(experiment_cfg.get("outputs", {}) or {}),
        "noise": noise_cfg,
        "add_noise": bool(noise_cfg.get("add_noise", False)),
        "default_sweep": default_sweep,
        "sweep_overrides": sweep_overrides,
        "nominal_values": _normalize_keyed_mapping(
            experiment_cfg.get("nominal_values", {}),
            field_name="experiment.nominal_values",
        ),
        "datasets": datasets_cfg,
        "resolved_raw": copy.deepcopy(dict(experiment_cfg)),
    }


def _load_and_resolve_prescription(
    *,
    prescription_path: Path | None,
    system_preset: str | None,
    experiment_preset: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a user prescription and return input, system, and experiment config blocks."""
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
        raise ValueError("generate_training_dataset_v3 requires a resolved top-level 'system' block.")
    if experiment_raw is None:
        raise ValueError("generate_training_dataset_v3 requires a resolved top-level 'experiment' block.")
    return user_cfg, dict(system_cfg), dict(experiment_raw)


def _build_nominal_store(
    *,
    system_cfg: Mapping[str, Any],
    experiment_cfg: Mapping[str, Any],
) -> tuple[ParamSpec, ParameterStore, SheraBinder]:
    """Build the forward spec, nominal store, and Shera binder for a V3 run."""
    from dluxshera.systems import SheraBinder
    from dluxshera.systems.base import compose_forward_spec

    forward_spec = compose_forward_spec(system_cfg)
    required_keys = _dedupe_preserve_order(
        list(experiment_cfg["sweep_keys"]) + list(experiment_cfg.get("nuisance_sweep_keys", []))
    )
    missing_keys = [key for key in required_keys if key not in forward_spec]
    if missing_keys:
        raise ValueError(
            "The resolved system does not expose all requested sweep-backed keys. Missing from forward spec: "
            + ", ".join(missing_keys)
        )
    sweep_keys = list(experiment_cfg["sweep_keys"])
    nominal_values = dict(experiment_cfg["nominal_values"])
    invalid_nominal = [key for key in nominal_values if key not in forward_spec]
    if invalid_nominal:
        raise ValueError(
            "experiment.nominal_values contains keys that are not present in the resolved system: "
            + ", ".join(invalid_nominal)
        )
    structural_nominal = sorted(forward_spec.structural_keys() & set(nominal_values))
    if structural_nominal:
        raise ValueError(
            "experiment.nominal_values may not override structural keys. Move these into system: "
            + ", ".join(structural_nominal)
        )
    base_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    if nominal_values:
        base_store = base_store.replace(nominal_values)
        base_store = _refresh_preserving_derived_keys(
            base_store,
            preserved_keys=set(sweep_keys) | set(nominal_values),
            spec=forward_spec,
        )
    return forward_spec, base_store, SheraBinder(system_cfg, forward_spec, base_store)


def _refresh_preserving_derived_keys(
    store: ParameterStore,
    *,
    preserved_keys: Iterable[str],
    spec: ParamSpec,
) -> ParameterStore:
    preserved_values: dict[str, Any] = {}
    for key in preserved_keys:
        if key not in spec or spec.get(key).kind != "derived":
            continue
        try:
            preserved_values[key] = store.get(key)
        except KeyError:
            continue
    refreshed = store.refresh_derived(spec)
    if preserved_values:
        refreshed = refreshed.replace(preserved_values)
    return refreshed


def _compute_fisher_sigmas(
    *,
    binder: SheraBinder,
    system_cfg: Mapping[str, Any],
    forward_spec: ParamSpec,
    base_store: ParameterStore,
    sweep_keys: Sequence[str],
    seed: int,
    add_noise: bool,
) -> tuple[dict[tuple[str, int | None], float], tuple[int, ...]]:
    """Compute Fisher-diagonal parameter sigmas for each packed component."""
    rng_key = jr.PRNGKey(seed)
    data = binder.model()
    if add_noise:
        rng_key, split_key = jr.split(rng_key)
        data = jr.poisson(split_key, data)
    image_shape = tuple(int(v) for v in np.asarray(data).shape)
    from dluxshera.inference.optimization import fim_theta, generate_fim_labels, make_binder_nll_fn

    nll_loss_fn, theta_ref = make_binder_nll_fn(
        binder=binder,
        infer_keys=list(sweep_keys),
        data=data,
        var=data,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=base_store,
    )
    F = fim_theta(nll_loss_fn, theta_ref)
    fim_diag = np.asarray(jnp.diag(F))
    fim_labels = generate_fim_labels(list(sweep_keys), cfg=system_cfg, store=base_store)
    _validate_fim_diag(fim_diag, labels=fim_labels)
    index_map = build_index_map(forward_spec.subset(list(sweep_keys)), base_store, theta=theta_ref)
    sigmas: dict[tuple[str, int | None], float] = {}
    for entry in index_map["entries"]:
        key = str(entry["name"])
        start = int(entry["start"])
        stop = int(entry["stop"])
        size = stop - start
        if size == 1:
            sigmas[(key, None)] = float(1.0 / math.sqrt(fim_diag[start]))
        else:
            for idx in range(size):
                sigmas[(key, idx)] = float(1.0 / math.sqrt(fim_diag[start + idx]))
    return sigmas, image_shape


def _scalar_label(base_key: str, component_index: int | None) -> str:
    return base_key if component_index is None else f"{base_key}[{component_index}]"


def _group_for_key(base_key: str) -> str:
    if "." in base_key:
        return base_key.split(".", 1)[0]
    return base_key


def _scalarize_parameter_space(
    *,
    sweep_keys: Sequence[str],
    base_store: ParameterStore,
    system_cfg: Mapping[str, Any],
    per_parameter_cfg: Mapping[str, SweepConfig],
    default_cfg: SweepConfig,
    fisher_sigmas: Mapping[tuple[str, int | None], float],
) -> list[ScalarParameter]:
    """Expand V2 base sweep keys into scalar V3 labels with Fisher ranges."""
    optics_cfg = system_cfg.get("optics", {}) if isinstance(system_cfg, Mapping) else {}
    noll_map = {
        "optics.primary.zernike_coeffs_nm": tuple(optics_cfg.get("primary_noll_indices") or ()),
        "optics.secondary.zernike_coeffs_nm": tuple(optics_cfg.get("secondary_noll_indices") or ()),
    }
    out: list[ScalarParameter] = []
    for key in sweep_keys:
        raw_value = np.asarray(base_store.get(key))
        if raw_value.shape == ():
            component_indices: list[int | None] = [None]
        else:
            component_indices = list(range(int(raw_value.size)))
        for component_index in component_indices:
            label = _scalar_label(key, component_index)
            nominal = float(raw_value.reshape(-1)[0 if component_index is None else component_index])
            sweep_cfg = _resolve_sweep_for_label(
                label,
                base_key=key,
                per_parameter_cfg=per_parameter_cfg,
                default_cfg=default_cfg,
            )
            sigma = float(fisher_sigmas[(key, component_index)])
            noll_index = None
            if component_index is not None and key in noll_map and component_index < len(noll_map[key]):
                noll_index = int(noll_map[key][component_index])
            out.append(
                ScalarParameter(
                    label=label,
                    base_key=key,
                    component_index=component_index,
                    nominal_value=nominal,
                    parameter_sigma=sigma,
                    sweep_source_key=key,
                    sweep_config=sweep_cfg,
                    min_abs_delta=sweep_cfg.min_sigma * sigma,
                    max_abs_delta=sweep_cfg.max_sigma * sigma,
                    display_label=label if noll_index is None else f"{label} (Noll {noll_index})",
                    group=_group_for_key(key),
                    noll_index=noll_index,
                )
            )
    return out


def _parameter_space_records(parameters: Sequence[ScalarParameter]) -> list[dict[str, Any]]:
    return [
        {
            "label": p.label,
            "base_key": p.base_key,
            "component_index": p.component_index,
            "nominal_value": p.nominal_value,
            "parameter_sigma": p.parameter_sigma,
            "sweep_source_key": p.sweep_source_key,
            "sweep_config": asdict(p.sweep_config),
            "min_sigma": p.sweep_config.min_sigma,
            "max_sigma": p.sweep_config.max_sigma,
            "n_magnitudes": p.sweep_config.n_magnitudes,
            "spacing": p.sweep_config.spacing,
            "min_abs_delta": p.min_abs_delta,
            "max_abs_delta": p.max_abs_delta,
            "units": p.units,
            "display_label": p.display_label,
            "group": p.group,
            "noll_index": p.noll_index,
        }
        for p in parameters
    ]


def _build_pair_grid_levels(parameter: ScalarParameter, *, pair_cfg: Mapping[str, Any]) -> list[dict[str, float | int]]:
    """Build sigma/delta levels for one pair-grid axis."""
    mode = str(pair_cfg.get("level_mode", "symmetric_grid_from_sweeps"))
    if mode == "mirrored_log_from_sweeps":
        sigmas = generate_mirrored_sigma_offsets(
            min_sigma=parameter.sweep_config.min_sigma,
            max_sigma=parameter.sweep_config.max_sigma,
            n_magnitudes=parameter.sweep_config.n_magnitudes,
            spacing=parameter.sweep_config.spacing,
        )
        if bool(pair_cfg.get("include_zero", False)):
            sigmas = [0.0] + sigmas
    elif mode == "symmetric_grid_from_sweeps":
        grid_size = int(pair_cfg.get("grid_size", parameter.sweep_config.n_magnitudes))
        if grid_size < 2:
            raise ValueError("pair_grid.grid_size must be >= 2.")
        max_sigma = float(parameter.sweep_config.max_sigma)
        sigmas = [float(v) for v in np.linspace(-max_sigma, max_sigma, num=grid_size)]
        if bool(pair_cfg.get("include_zero", True)) and not any(abs(v) < 1e-12 for v in sigmas):
            sigmas[grid_size // 2] = 0.0
    else:
        raise ValueError(f"Unsupported pair_grid.level_mode {mode!r}.")
    return [
        {"index": idx, "sigma": float(sigma), "delta": float(sigma) * parameter.parameter_sigma}
        for idx, sigma in enumerate(sigmas)
    ]


def _make_subseed(seed: int, *parts: Any) -> int:
    payload = json.dumps([int(seed), *[str(part) for part in parts]], separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _draw_uniform_sigma(rng: np.random.Generator, sweep_cfg: SweepConfig, *, signed: bool) -> float:
    magnitude = float(rng.uniform(sweep_cfg.min_sigma, sweep_cfg.max_sigma))
    if signed and rng.random() < 0.5:
        return -magnitude
    return magnitude


def _build_nuisance_draws(
    *,
    parameters_by_label: Mapping[str, ScalarParameter],
    nuisance_cfg: Mapping[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Build deterministic nominal/random registration-nuisance draw records."""
    if not bool(nuisance_cfg.get("enabled", True)):
        return [{"nuisance_id": 0, "values": {}, "sigma_values": {}, "sample_role_suffix": "no_nuisance"}]
    keys = [str(key) for key in nuisance_cfg.get("keys", REGISTRATION_NUISANCE_KEYS)]
    draws: list[dict[str, Any]] = []
    if bool(nuisance_cfg.get("include_nominal", True)):
        draws.append(
            {
                "nuisance_id": 0,
                "values": {key: 0.0 for key in keys},
                "sigma_values": {key: 0.0 for key in keys},
                "sample_role_suffix": "nominal_registration",
            }
        )
    rng = np.random.default_rng(_make_subseed(seed, "nuisance_replicates"))
    next_nuisance_id = 1 if draws else 1
    for random_idx in range(int(nuisance_cfg.get("n_random", 0))):
        values: dict[str, float] = {}
        sigma_values: dict[str, float] = {}
        for key in keys:
            param = parameters_by_label.get(key)
            if param is None:
                continue
            sigma_offset = _draw_uniform_sigma(rng, param.sweep_config, signed=True)
            sigma_values[key] = sigma_offset
            values[key] = sigma_offset * param.parameter_sigma
        nuisance_id = next_nuisance_id
        next_nuisance_id += 1
        draws.append(
            {
                "nuisance_id": nuisance_id,
                "values": values,
                "sigma_values": sigma_values,
                "sample_role_suffix": f"random_registration_{random_idx:03d}",
            }
        )
    return draws


def _nuisance_for_controlled_axes(
    *,
    draw: Mapping[str, Any],
    controlled_base_keys: set[str],
    collision_policy: str,
) -> tuple[dict[str, float], dict[str, float], list[str]]:
    values = dict(draw.get("values", {}) or {})
    sigma_values = dict(draw.get("sigma_values", {}) or {})
    skipped: list[str] = []
    if collision_policy == "skip_if_key_is_controlled_axis":
        for key in list(values):
            if key in controlled_base_keys:
                values.pop(key, None)
                sigma_values.pop(key, None)
                skipped.append(key)
    return values, sigma_values, skipped


def _build_pair_grid_plan(
    *,
    parameters: Sequence[ScalarParameter],
    nuisance_parameters_by_label: Mapping[str, ScalarParameter] | None = None,
    pair_cfg: Mapping[str, Any],
    nuisance_cfg: Mapping[str, Any],
    seed: int,
) -> list[dict[str, Any]]:
    """Build the all-pairs 2D grid plan rows."""
    if not bool(pair_cfg.get("enabled", True)):
        return []
    include_self = bool(pair_cfg.get("include_self_pairs", False))
    if nuisance_parameters_by_label is None:
        nuisance_parameters_by_label = {param.label: param for param in parameters}
        if bool(nuisance_cfg.get("enabled", True)):
            missing_nuisance = [
                str(key)
                for key in nuisance_cfg.get("keys", REGISTRATION_NUISANCE_KEYS)
                if str(key) not in nuisance_parameters_by_label
            ]
            if missing_nuisance:
                raise ValueError(
                    "nuisance_parameters_by_label was omitted, but requested nuisance keys "
                    "cannot be resolved from parameters: " + ", ".join(missing_nuisance)
                )
    nuisance_draws = _build_nuisance_draws(
        parameters_by_label=nuisance_parameters_by_label,
        nuisance_cfg=nuisance_cfg,
        seed=seed,
    )
    plan: list[dict[str, Any]] = []
    sample_idx = 0
    for i, param_i in enumerate(parameters):
        start_j = i if include_self else i + 1
        for j in range(start_j, len(parameters)):
            param_j = parameters[j]
            pair_id = f"pair_{i:03d}_{j:03d}"
            levels_i = _build_pair_grid_levels(param_i, pair_cfg=pair_cfg)
            levels_j = _build_pair_grid_levels(param_j, pair_cfg=pair_cfg)
            controlled_base_keys = {param_i.base_key, param_j.base_key}
            for level_i in levels_i:
                for level_j in levels_j:
                    for draw in nuisance_draws:
                        nuisance_values, nuisance_sigmas, skipped = _nuisance_for_controlled_axes(
                            draw=draw,
                            controlled_base_keys=controlled_base_keys,
                            collision_policy=str(nuisance_cfg.get("collision_policy", "skip_if_key_is_controlled_axis")),
                        )
                        sample_id = f"sample_{sample_idx:06d}"
                        plan.append(
                            {
                                "dataset_family": "pair_grid",
                                "sample_role": "pair_grid",
                                "sample_id": sample_id,
                                "sample_index": sample_idx,
                                "pair_id": pair_id,
                                "pair_label_i": param_i.label,
                                "pair_label_j": param_j.label,
                                "grid_i_index": int(level_i["index"]),
                                "grid_j_index": int(level_j["index"]),
                                "grid_i_sigma": float(level_i["sigma"]),
                                "grid_j_sigma": float(level_j["sigma"]),
                                "delta_i": float(level_i["delta"]),
                                "delta_j": float(level_j["delta"]),
                                "delta_units": "parameter_units",
                                "controlled_labels": [param_i.label, param_j.label],
                                "theta_delta": {param_i.label: float(level_i["delta"]), param_j.label: float(level_j["delta"])},
                                "registration_nuisance_values": nuisance_values,
                                "registration_nuisance_sigma_values": nuisance_sigmas,
                                "skipped_nuisance_keys": skipped,
                                "nuisance_id": int(draw["nuisance_id"]),
                                "seed": _make_subseed(seed, "pair_grid", pair_id, level_i["index"], level_j["index"], draw["nuisance_id"]),
                                "fits_path": f"images/{sample_id}.fits",
                                "metadata_path": f"images/{sample_id}.json",
                            }
                        )
                        sample_idx += 1
    return plan


def _normalize_active_count_probs(raw_probs: Mapping[Any, Any]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([int(key) for key in raw_probs.keys()], dtype=int)
    probs = np.asarray([float(value) for value in raw_probs.values()], dtype=float)
    if np.any(counts < 1):
        raise ValueError("sparse_mixture.active_count_probs keys must be >= 1.")
    if not np.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("sparse_mixture.active_count_probs must have positive finite probabilities.")
    probs = probs / probs.sum()
    return counts, probs


def _build_sparse_mixture_plan(
    *,
    parameters: Sequence[ScalarParameter],
    nuisance_parameters_by_label: Mapping[str, ScalarParameter] | None = None,
    sparse_cfg: Mapping[str, Any],
    nuisance_cfg: Mapping[str, Any],
    seed: int,
    start_sample_index: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic sparse random mixture plan rows."""
    if not bool(sparse_cfg.get("enabled", True)):
        return []
    rng = np.random.default_rng(_make_subseed(seed, "sparse_mixture"))
    labels = [param.label for param in parameters]
    counts, probs = _normalize_active_count_probs(sparse_cfg.get("active_count_probs", {1: 1.0}))
    n_samples = int(sparse_cfg.get("n_samples", 0))
    signed = bool((sparse_cfg.get("amplitude_sampling", {}) or {}).get("signed", True))
    sparse_nuisance_enabled = bool((sparse_cfg.get("nuisance", {}) or {}).get("enabled", True))
    if nuisance_parameters_by_label is None:
        nuisance_parameters_by_label = {param.label: param for param in parameters}
        if sparse_nuisance_enabled and bool(nuisance_cfg.get("enabled", True)):
            missing_nuisance = [
                str(key)
                for key in nuisance_cfg.get("keys", REGISTRATION_NUISANCE_KEYS)
                if str(key) not in nuisance_parameters_by_label
            ]
            if missing_nuisance:
                raise ValueError(
                    "nuisance_parameters_by_label was omitted, but requested nuisance keys "
                    "cannot be resolved from parameters: " + ", ".join(missing_nuisance)
                )
    nuisance_draws = _build_nuisance_draws(
        parameters_by_label=nuisance_parameters_by_label,
        nuisance_cfg={**dict(nuisance_cfg), "n_random": max(1, int(nuisance_cfg.get("n_random", 1))), "include_nominal": False},
        seed=_make_subseed(seed, "sparse_nuisance"),
    )
    plan: list[dict[str, Any]] = []
    for idx in range(n_samples):
        active_count = int(rng.choice(counts, p=probs))
        active_count = min(active_count, len(parameters))
        active_indices = sorted(int(v) for v in rng.choice(len(parameters), size=active_count, replace=False))
        theta_delta: dict[str, float] = {}
        theta_sigma: dict[str, float] = {}
        for active_idx in active_indices:
            param = parameters[active_idx]
            sigma_offset = _draw_uniform_sigma(rng, param.sweep_config, signed=signed)
            theta_sigma[param.label] = sigma_offset
            theta_delta[param.label] = sigma_offset * param.parameter_sigma
        if sparse_nuisance_enabled and nuisance_draws:
            draw = nuisance_draws[int(rng.integers(0, len(nuisance_draws)))]
            controlled_base_keys = {parameters[active_idx].base_key for active_idx in active_indices}
            nuisance_values, nuisance_sigmas, skipped = _nuisance_for_controlled_axes(
                draw=draw,
                controlled_base_keys=controlled_base_keys,
                collision_policy=str(nuisance_cfg.get("collision_policy", "skip_if_key_is_controlled_axis")),
            )
            nuisance_id = int(draw["nuisance_id"])
        else:
            nuisance_values, nuisance_sigmas, skipped, nuisance_id = {}, {}, [], 0
        sample_index = start_sample_index + idx
        sample_id = f"sample_{sample_index:06d}"
        active_labels = [labels[active_idx] for active_idx in active_indices]
        active_mask = [1 if label in active_labels else 0 for label in labels]
        theta_nominal = {param.label: param.nominal_value for param in parameters}
        theta_applied = {
            param.label: param.nominal_value + theta_delta.get(param.label, 0.0)
            for param in parameters
        }
        plan.append(
            {
                "dataset_family": "sparse_mixture",
                "sample_role": "sparse_random",
                "sample_id": sample_id,
                "sample_index": sample_index,
                "split": str(sparse_cfg.get("split", "test")),
                "active_labels": active_labels,
                "active_mask": active_mask,
                "active_count": active_count,
                "theta_nominal": theta_nominal,
                "theta_delta": theta_delta,
                "theta_sigma": theta_sigma,
                "theta_applied": theta_applied,
                "registration_nuisance_values": nuisance_values,
                "registration_nuisance_sigma_values": nuisance_sigmas,
                "skipped_nuisance_keys": skipped,
                "nuisance_id": nuisance_id,
                "seed": _make_subseed(seed, "sparse_mixture", idx),
                "fits_path": f"images/{sample_id}.fits",
                "metadata_path": f"images/{sample_id}.json",
            }
        )
    return plan


def _set_scalar_label(store: ParameterStore, param: ScalarParameter, value: float) -> ParameterStore:
    if param.component_index is None:
        return store.replace({param.base_key: value})
    current = np.asarray(store.get(param.base_key), dtype=float).copy().reshape(-1)
    current[param.component_index] = value
    original_shape = np.asarray(store.get(param.base_key)).shape
    return store.replace({param.base_key: current.reshape(original_shape)})


def _apply_sample_to_store(
    *,
    base_store: ParameterStore,
    sample: Mapping[str, Any],
    parameters_by_label: Mapping[str, ScalarParameter],
    forward_spec: ParamSpec,
) -> ParameterStore:
    """Apply controlled and registration deltas from one plan row to a store."""
    store = base_store
    theta_delta = dict(sample.get("theta_delta", {}) or {})
    for label, delta in theta_delta.items():
        param = parameters_by_label[label]
        store = _set_scalar_label(store, param, param.nominal_value + float(delta))
    for key, delta in dict(sample.get("registration_nuisance_values", {}) or {}).items():
        if key not in forward_spec:
            continue
        current = float(np.asarray(store.get(key)))
        store = store.replace({key: current + float(delta)})
    preserve = {param.base_key for param in parameters_by_label.values()} | set(REGISTRATION_NUISANCE_KEYS)
    return _refresh_preserving_derived_keys(store, preserved_keys=preserve, spec=forward_spec)


def _noise_model(rng_key: jax.Array, data: jax.Array, *, add_noise: bool) -> tuple[jax.Array, str, int | None]:
    if not add_noise:
        return data, "none", None
    rng_key, split_key = jr.split(rng_key)
    if np.min(np.asarray(data)) > 100:
        return np.sqrt(np.asarray(data)) * jr.normal(split_key, data.shape) + data, "gaussian-approx", int(np.asarray(split_key)[0])
    return jr.poisson(split_key, data), "poisson", int(np.asarray(split_key)[0])


def _render_sample(
    *,
    binder: SheraBinder,
    applied_store: ParameterStore,
    sample: Mapping[str, Any],
    images_dir: Path,
    run_dir: Path,
    add_noise: bool,
) -> dict[str, Any]:
    model = binder.model(binder.strip_structural(applied_store))
    image, noise_mode, noise_seed = _noise_model(jr.PRNGKey(int(sample["seed"])), model, add_noise=add_noise)
    image_np = np.asarray(image)
    fits_path = run_dir / str(sample["fits_path"])
    meta_path = run_dir / str(sample["metadata_path"])
    fits_path.parent.mkdir(parents=True, exist_ok=True)
    _write_fits(
        output_path=fits_path,
        image=image_np,
        header_data={
            "SAMPLEID": (str(sample["sample_id"]), "Training dataset sample id"),
            "FAMILY": (str(sample["dataset_family"]), "Dataset family"),
            "ROLE": (str(sample["sample_role"]), "Sample role"),
            "NUISID": (int(sample.get("nuisance_id", 0)), "Registration nuisance id"),
            "NOISE": (bool(add_noise), "Observation noise added"),
            "SEED": (int(sample["seed"]), "Per-sample seed"),
        },
    )
    sample_meta = {**dict(sample), "noise_mode": noise_mode, "noise_seed": noise_seed, "image_shape": list(image_np.shape)}
    meta_path.write_text(json.dumps(_serialize_value(sample_meta), indent=2), encoding="utf-8")
    return {**sample_meta, "fits_path": str(fits_path.relative_to(run_dir)), "metadata_path": str(meta_path.relative_to(run_dir))}


def _csv_ready(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_serialize_value(value), sort_keys=True, separators=(",", ":"))
    return _serialize_value(value)


def _write_plan_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a stable CSV plan artifact, JSON-encoding nested cell values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_ready(row.get(key, "")) for key in fieldnames})


def _append_sample_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_serialize_value(dict(row)), sort_keys=True) + "\n")


def _write_samples_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_serialize_value(dict(row)), sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialize_value(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_manifest(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    _write_json(run_dir / "manifest.json", manifest)


def _dataset_family_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("dataset_family", "unknown"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _read_json_if_exists(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_samples_jsonl(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not path.exists():
        return [], None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                return rows, f"samples.jsonl line {line_number} is not valid JSON: {exc}"
            if not isinstance(payload, Mapping):
                return rows, f"samples.jsonl line {line_number} is not a JSON object."
            rows.append(dict(payload))
    return rows, None


def _resume_group_key(sample: Mapping[str, Any]) -> str:
    family = str(sample.get("dataset_family", "unknown"))
    if family == "pair_grid" and sample.get("pair_id") is not None:
        return f"pair_grid:{sample.get('pair_id')}"
    return f"{family}:{sample.get('sample_id')}"


def _resume_group_start_index(plan: Sequence[Mapping[str, Any]], sample_index: int) -> int:
    if sample_index <= 0:
        return 0
    if sample_index >= len(plan):
        return len(plan)
    group_key = _resume_group_key(plan[sample_index])
    cursor = sample_index
    while cursor > 0 and _resume_group_key(plan[cursor - 1]) == group_key:
        cursor -= 1
    return cursor


def _sample_artifact_paths(run_dir: Path, row: Mapping[str, Any]) -> list[Path]:
    out: list[Path] = []
    for key in ("fits_path", "metadata_path"):
        value = row.get(key)
        if value in (None, ""):
            continue
        out.append(run_dir / str(value))
    return out


def _delete_paths(paths: Iterable[Path]) -> int:
    removed = 0
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def _validate_resume_compatibility(
    *,
    run_dir: Path,
    runtime_resolved_cfg: Mapping[str, Any],
    parameter_space_records: Sequence[Mapping[str, Any]],
) -> None:
    if not run_dir.exists():
        raise ValueError(f"--resume requires an existing run directory: {run_dir}")
    current_resolved = _serialize_value(runtime_resolved_cfg)
    existing_resolved = _read_json_if_exists(run_dir / "prescription_resolved.json")
    if existing_resolved is not None and existing_resolved != current_resolved:
        raise ValueError(
            "--resume refused because the existing run directory was generated from a different resolved prescription."
        )
    current_parameter_space = _serialize_value({"parameters": parameter_space_records})
    existing_parameter_space = _read_json_if_exists(run_dir / "parameter_space.json")
    if existing_parameter_space is not None and existing_parameter_space != current_parameter_space:
        raise ValueError(
            "--resume refused because the existing run directory has a different parameter space."
        )


def _prepare_resume_state(
    *,
    run_dir: Path,
    planned_samples: Sequence[Mapping[str, Any]],
    runtime_resolved_cfg: Mapping[str, Any],
    parameter_space_records: Sequence[Mapping[str, Any]],
) -> ResumeState:
    _validate_resume_compatibility(
        run_dir=run_dir,
        runtime_resolved_cfg=runtime_resolved_cfg,
        parameter_space_records=parameter_space_records,
    )
    samples_path = run_dir / "samples.jsonl"
    rows, read_issue = _read_samples_jsonl(samples_path)
    if len(rows) > len(planned_samples):
        raise ValueError(
            "--resume refused because the existing samples.jsonl contains more rows than the current requested sample set."
        )

    valid_prefix_count = 0
    reason = read_issue or "existing samples.jsonl ended before the requested sample set was complete."
    for idx, row in enumerate(rows):
        expected = planned_samples[idx]
        if str(row.get("sample_id", "")) != str(expected["sample_id"]):
            reason = (
                f"samples.jsonl row {idx + 1} has sample_id={row.get('sample_id')!r}, "
                f"expected {expected['sample_id']!r}."
            )
            break
        expected_paths = _sample_artifact_paths(run_dir, expected)
        if not all(path.exists() for path in expected_paths):
            reason = f"sample_id={expected['sample_id']} is missing one or more artifacts on disk."
            break
        valid_prefix_count += 1
    else:
        if read_issue is None:
            if valid_prefix_count == len(planned_samples):
                return ResumeState(
                    start_sample_index=len(planned_samples),
                    retained_rows=tuple(rows),
                    valid_prefix_count=valid_prefix_count,
                    cleanup_group_key=None,
                    cleanup_group_start_index=len(planned_samples),
                    cleanup_sample_count=0,
                    reason="requested sample set is already complete.",
                )
            reason = "existing samples form a valid prefix but stop before the requested sample set is complete."

    cleanup_start_index = _resume_group_start_index(planned_samples, valid_prefix_count)
    cleanup_group_key = (
        None if cleanup_start_index >= len(planned_samples) else _resume_group_key(planned_samples[cleanup_start_index])
    )
    retained_rows = rows[:cleanup_start_index]
    cleanup_paths: list[Path] = []
    for sample in planned_samples[cleanup_start_index:]:
        cleanup_paths.extend(_sample_artifact_paths(run_dir, sample))
    for row in rows[cleanup_start_index:]:
        cleanup_paths.extend(_sample_artifact_paths(run_dir, row))
    _delete_paths(cleanup_paths)
    return ResumeState(
        start_sample_index=cleanup_start_index,
        retained_rows=tuple(retained_rows),
        valid_prefix_count=valid_prefix_count,
        cleanup_group_key=cleanup_group_key,
        cleanup_group_start_index=cleanup_start_index,
        cleanup_sample_count=len(planned_samples) - cleanup_start_index,
        reason=reason,
    )


def _system_summary(system_cfg: Mapping[str, Any], image_shape: Sequence[int]) -> dict[str, Any]:
    optics = system_cfg.get("optics", {}) if isinstance(system_cfg, Mapping) else {}
    source = system_cfg.get("source", {}) if isinstance(system_cfg, Mapping) else {}
    return {
        "preset": system_cfg.get("preset") if isinstance(system_cfg, Mapping) else None,
        "source_kind": source.get("kind") if isinstance(source, Mapping) else None,
        "target": source.get("target") if isinstance(source, Mapping) else None,
        "exposure_time_s": source.get("exposure_time_s") if isinstance(source, Mapping) else None,
        "optics_psf_npix": optics.get("psf_npix") if isinstance(optics, Mapping) else None,
        "optics_pupil_npix": optics.get("pupil_npix") if isinstance(optics, Mapping) else None,
        "estimated_image_shape": list(image_shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plan-first ML training dataset V3 artifacts.")
    parser.add_argument("--prescription", "--config", dest="prescription", type=Path, default=None)
    parser.add_argument("--system-preset", type=str, default=None)
    parser.add_argument("--experiment-preset", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--max-samples", type=int, default=None)
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

    user_cfg, system_cfg, experiment_raw = _load_and_resolve_prescription(
        prescription_path=prescription_path,
        system_preset=system_preset,
        experiment_preset=experiment_preset,
    )
    experiment_cfg = _validate_experiment_config(experiment_raw)
    run_dir, run_dir_source = _resolve_run_dir(
        cli_outdir=args.outdir,
        run_name=args.run_name,
        experiment_cfg=experiment_cfg,
        prescription_path=prescription_path,
    )

    _log_section("Resolved configuration")
    _log(f"Prescription path: {prescription_path if prescription_path is not None else '<presets only>'}")
    _log(f"System preset seed: {system_preset or '<none>'}")
    _log(f"Experiment preset seed: {experiment_preset or '<none>'}")
    _log(f"Resolved run directory: {run_dir} ({run_dir_source})")

    forward_spec, base_store, binder = _build_nominal_store(system_cfg=system_cfg, experiment_cfg=experiment_cfg)
    sweep_keys = list(experiment_cfg["sweep_keys"])
    nuisance_sweep_keys = list(experiment_cfg.get("nuisance_sweep_keys", []))
    analysis_keys = _dedupe_preserve_order(sweep_keys + nuisance_sweep_keys)
    _log("Calculating Fisher information matrix for V3 scalarized parameter scales.")
    fisher_sigmas, image_shape = _compute_fisher_sigmas(
        binder=binder,
        system_cfg=system_cfg,
        forward_spec=forward_spec,
        base_store=base_store,
        sweep_keys=analysis_keys,
        seed=int(experiment_cfg["seed"]),
        add_noise=bool(experiment_cfg["add_noise"]),
    )

    per_parameter_cfg = _normalize_sweep_configs(
        sweep_keys=analysis_keys,
        default_cfg=experiment_cfg["default_sweep"],
        overrides=experiment_cfg["sweep_overrides"],
    )
    parameters = _scalarize_parameter_space(
        sweep_keys=sweep_keys,
        base_store=base_store,
        system_cfg=system_cfg,
        per_parameter_cfg=per_parameter_cfg,
        default_cfg=experiment_cfg["default_sweep"],
        fisher_sigmas=fisher_sigmas,
    )
    parameters_by_label = {param.label: param for param in parameters}
    nuisance_only_keys = [key for key in nuisance_sweep_keys if key not in set(sweep_keys)]
    nuisance_parameters = _scalarize_parameter_space(
        sweep_keys=nuisance_only_keys,
        base_store=base_store,
        system_cfg=system_cfg,
        per_parameter_cfg=per_parameter_cfg,
        default_cfg=experiment_cfg["default_sweep"],
        fisher_sigmas=fisher_sigmas,
    )
    nuisance_parameters_by_label = {param.label: param for param in parameters + nuisance_parameters}

    datasets_cfg = experiment_cfg["datasets"]
    pair_cfg = datasets_cfg["pair_grid"]
    nuisance_cfg = datasets_cfg["nuisance_replicates"]
    sparse_cfg = datasets_cfg["sparse_mixture"]
    pair_plan = _build_pair_grid_plan(
        parameters=parameters,
        nuisance_parameters_by_label=nuisance_parameters_by_label,
        pair_cfg=pair_cfg,
        nuisance_cfg=nuisance_cfg,
        seed=int(experiment_cfg["seed"]),
    )
    sparse_plan = _build_sparse_mixture_plan(
        parameters=parameters,
        nuisance_parameters_by_label=nuisance_parameters_by_label,
        sparse_cfg=sparse_cfg,
        nuisance_cfg=nuisance_cfg,
        seed=int(experiment_cfg["seed"]),
        start_sample_index=len(pair_plan),
    )
    full_plan = pair_plan + sparse_plan

    n_params = len(parameters)
    unordered_pairs = n_params * (n_params - 1) // 2
    nuisance_draws = _build_nuisance_draws(
        parameters_by_label=nuisance_parameters_by_label,
        nuisance_cfg=nuisance_cfg,
        seed=int(experiment_cfg["seed"]),
    )
    random_nuisance = sum(1 for draw in nuisance_draws if int(draw.get("nuisance_id", 0)) != 0)
    nominal_nuisance = sum(1 for draw in nuisance_draws if int(draw.get("nuisance_id", 0)) == 0)

    _log_section("Dry-run/plan counts")
    _log(f"scalarized_parameter_count={n_params}")
    _log(f"unordered_pair_count={unordered_pairs}")
    _log(f"grid_size={pair_cfg.get('grid_size')}")
    _log(f"nominal_nuisance_replicates={nominal_nuisance}")
    _log(f"random_nuisance_replicates={random_nuisance}")
    _log(f"pair_grid_sample_count={len(pair_plan)}")
    _log(f"sparse_mixture_sample_count={len(sparse_plan)}")
    _log(f"total_planned_sample_count={len(full_plan)}")
    _log(f"estimated_image_shape={image_shape}")
    _log(f"output_directory={run_dir}")

    runtime_resolved_cfg = {
        "system": copy.deepcopy(system_cfg),
        "experiment": {
            **copy.deepcopy(experiment_cfg["resolved_raw"]),
            "sweeps": {
                "default": asdict(experiment_cfg["default_sweep"]),
                **{key: asdict(value) for key, value in per_parameter_cfg.items()},
            },
            "datasets": copy.deepcopy(datasets_cfg),
        },
    }
    parameter_space_records = _parameter_space_records(parameters)
    max_samples = len(full_plan) if args.max_samples is None else min(int(args.max_samples), len(full_plan))
    planned_samples = full_plan[:max_samples]
    plan_summary = {
        "scalarized_parameter_count": n_params,
        "unordered_pair_count": unordered_pairs,
        "grid_size": pair_cfg.get("grid_size"),
        "level_mode": pair_cfg.get("level_mode"),
        "nominal_nuisance_replicates": nominal_nuisance,
        "random_nuisance_replicates": random_nuisance,
        "pair_grid_sample_count": len(pair_plan),
        "sparse_mixture_sample_count": len(sparse_plan),
        "total_planned_sample_count": len(full_plan),
        "requested_render_sample_count": max_samples,
        "estimated_image_shape": list(image_shape),
    }
    resume_state = ResumeState(
        start_sample_index=0,
        retained_rows=(),
        valid_prefix_count=0,
        cleanup_group_key=None,
        cleanup_group_start_index=0,
        cleanup_sample_count=0,
        reason="fresh render requested.",
    )
    if args.resume:
        resume_state = _prepare_resume_state(
            run_dir=run_dir,
            planned_samples=planned_samples,
            runtime_resolved_cfg=runtime_resolved_cfg,
            parameter_space_records=parameter_space_records,
        )
        _log_section("Resume analysis")
        _log(f"resume_reason={resume_state.reason}")
        _log(f"resume_valid_prefix_count={resume_state.valid_prefix_count}")
        _log(f"resume_restart_sample_index={resume_state.start_sample_index}")
        if resume_state.cleanup_group_key is not None:
            _log(f"resume_cleanup_group={resume_state.cleanup_group_key}")

    manifest = {
        "schema_version": "ml_training_dataset_v3_manifest/2",
        "generator": "work/experiments/generate_training_dataset_v3.py",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "prescription_path": str(prescription_path) if prescription_path is not None else None,
        "system_preset": system_cfg.get("preset") if isinstance(system_cfg, Mapping) else system_preset,
        "experiment_preset": experiment_raw.get("preset") if isinstance(experiment_raw, Mapping) else experiment_preset,
        "seed": int(experiment_cfg["seed"]),
        "resolved_system_summary": _system_summary(system_cfg, image_shape),
        "parameter_space_summary": {
            "count": n_params,
            "labels": [param.label for param in parameters],
            "range_source": "experiment.sweeps",
        },
        "dataset_family_counts": _dataset_family_counts(full_plan),
        "output_paths": {
            "run_dir": str(run_dir),
            "manifest": "manifest.json",
            "parameter_space": "parameter_space.json",
            "pair_plan": "pair_plan.csv",
            "sparse_mixture_plan": "sparse_mixture_plan.csv",
            "samples": "samples.jsonl",
        },
        "noise_config": copy.deepcopy(experiment_cfg["noise"]),
        "nuisance_config": copy.deepcopy(nuisance_cfg),
        "sweeps_config": {
            "default": asdict(experiment_cfg["default_sweep"]),
            "per_parameter": {key: asdict(value) for key, value in per_parameter_cfg.items()},
        },
        "sweep_resolution_policy": {
            "sweep_keys": "canonical eligible base parameter keys",
            "vector_keys": "scalarized into key[index] labels",
            "range_source": "experiment.sweeps base key, falling back to experiment.sweeps.default",
            "pair_grid_level_mode": pair_cfg.get("level_mode"),
            "pair_grid_grid_size": pair_cfg.get("grid_size"),
        },
        "git_info": _git_info(),
        "plan_summary": plan_summary,
        "resume_mode": bool(args.resume),
        "resume_state": {
            "reason": resume_state.reason,
            "valid_prefix_count": resume_state.valid_prefix_count,
            "restart_sample_index": resume_state.start_sample_index,
            "cleanup_group_key": resume_state.cleanup_group_key,
            "cleanup_group_start_index": resume_state.cleanup_group_start_index,
            "cleanup_sample_count": resume_state.cleanup_sample_count,
        },
        "render_target_sample_count": max_samples,
        "rendered_sample_count": len(resume_state.retained_rows),
        "next_sample_index": resume_state.start_sample_index,
        "render_complete": False,
        "last_rendered_sample_id": (
            None if not resume_state.retained_rows else resume_state.retained_rows[-1].get("sample_id")
        ),
        "last_rendered_pair_id": (
            None if not resume_state.retained_rows else resume_state.retained_rows[-1].get("pair_id")
        ),
        "dry_run": bool(args.dry_run),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "quicklook").mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "prescription_input.json", user_cfg)
    _write_json(run_dir / "prescription_resolved.json", runtime_resolved_cfg)
    _write_json(run_dir / "parameter_space.json", {"parameters": parameter_space_records})
    _write_plan_csv(run_dir / "pair_plan.csv", pair_plan)
    _write_plan_csv(run_dir / "sparse_mixture_plan.csv", sparse_plan)
    _write_json(run_dir / "quicklook" / "plan_summary.json", plan_summary)
    _write_plan_csv(run_dir / "quicklook" / "parameter_space_summary.csv", parameter_space_records)
    _write_manifest(run_dir=run_dir, manifest=manifest)

    if args.dry_run:
        _log("Dry run enabled; wrote plan artifacts and skipped FITS rendering.")
        return

    samples_path = run_dir / "samples.jsonl"
    if args.resume:
        _write_samples_jsonl(samples_path, resume_state.retained_rows)
    else:
        _write_samples_jsonl(samples_path, [])
    _log_section("Dataset rendering")
    _log(f"Rendering {max_samples} of {len(full_plan)} planned samples.")
    if args.resume:
        _log(f"Resuming from sample index {resume_state.start_sample_index}.")
    t0 = time.perf_counter()
    rendered = len(resume_state.retained_rows)
    rendered_this_run = 0
    for sample in planned_samples[resume_state.start_sample_index:]:
        applied_store = _apply_sample_to_store(
            base_store=base_store,
            sample=sample,
            parameters_by_label=parameters_by_label,
            forward_spec=forward_spec,
        )
        row = _render_sample(
            binder=binder,
            applied_store=applied_store,
            sample=sample,
            images_dir=run_dir / "images",
            run_dir=run_dir,
            add_noise=bool(experiment_cfg["add_noise"]),
        )
        _append_sample_jsonl(samples_path, row)
        rendered += 1
        rendered_this_run += 1
        manifest = dict(manifest)
        manifest["rendered_sample_count"] = rendered
        manifest["next_sample_index"] = rendered
        manifest["render_complete"] = rendered >= max_samples
        manifest["last_rendered_sample_id"] = row.get("sample_id")
        manifest["last_rendered_pair_id"] = row.get("pair_id")
        _write_manifest(run_dir=run_dir, manifest=manifest)
        if rendered_this_run % 10 == 0:
            elapsed = time.perf_counter() - t0
            _log(f"Progress: rendered={rendered} resumed={rendered_this_run} rate={rendered_this_run / elapsed:.2f} samples/s")
    manifest = dict(manifest)
    manifest["rendered_sample_count"] = rendered
    manifest["next_sample_index"] = rendered
    manifest["render_complete"] = rendered >= max_samples
    manifest["dry_run"] = False
    _write_manifest(run_dir=run_dir, manifest=manifest)
    _log(f"Rendered samples: {rendered}")
    _log(f"Manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
