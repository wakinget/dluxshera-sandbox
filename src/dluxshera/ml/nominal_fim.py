from __future__ import annotations

import copy
import hashlib
import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.datasets import rendering as shared_rendering
from dluxshera.params.packing import build_index_map
from dluxshera.params.spec import ParamSpec
from dluxshera.params.store import ParameterStore

JAX_ENABLE_X64 = True
SCRIPT_VERSION = "v3.0-plan-first"
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


DEFAULT_SWEEP_CONFIG = SweepConfig()


def ensure_jax_x64_enabled() -> None:
    """Enable the V3/S10 nominal-FIM numerical convention for direct callers."""
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)


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


def _validate_fim_diag(fim_diag: np.ndarray, *, labels: list[str]) -> None:
    for idx, val in enumerate(fim_diag):
        if not np.isfinite(val) or val <= 0:
            label = labels[idx] if idx < len(labels) else f"index {idx}"
            warnings.warn(f"Invalid FIM diagonal entry for {label}: {val}.", RuntimeWarning)
            raise ValueError(
                f"FIM diagonal entry for {label} is invalid ({val}); cannot compute sigma scaling."
            )


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
):
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
    return shared_rendering.refresh_preserving_derived_keys(
        store,
        preserved_keys=preserved_keys,
        spec=spec,
    )


def _compute_fisher_sigmas(
    *,
    binder,
    system_cfg: Mapping[str, Any],
    forward_spec: ParamSpec,
    base_store: ParameterStore,
    sweep_keys: Sequence[str],
    seed: int,
    add_noise: bool,
) -> tuple[dict[tuple[str, int | None], float], tuple[int, ...]]:
    """Compute Fisher-diagonal parameter sigmas for each packed component."""
    result = compute_nominal_fisher_matrix(
        binder=binder,
        system_cfg=system_cfg,
        forward_spec=forward_spec,
        base_store=base_store,
        sweep_keys=sweep_keys,
        seed=seed,
        add_noise=add_noise,
    )
    fim_diag = np.diag(np.asarray(result["fim_theta"], dtype=np.float64))
    index_map = result["index_map"]
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
    return sigmas, tuple(int(v) for v in result["image_shape"])


def _index_map_scalar_labels(index_map: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for entry in index_map.get("entries", []):
        key = str(entry["name"])
        start = int(entry["start"])
        stop = int(entry["stop"])
        size = stop - start
        if size == 1:
            labels.append(key)
        else:
            labels.extend([f"{key}[{idx}]" for idx in range(size)])
    return labels


def compute_nominal_fisher_matrix(
    *,
    binder,
    system_cfg: Mapping[str, Any],
    forward_spec: ParamSpec,
    base_store: ParameterStore,
    sweep_keys: Sequence[str],
    seed: int,
    add_noise: bool,
) -> dict[str, Any]:
    """Compute the full nominal V3 physical-theta FIM using the canonical loss path."""
    ensure_jax_x64_enabled()
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
    return {
        "fim_theta": np.asarray(F, dtype=np.float64),
        "theta_ref": np.asarray(theta_ref, dtype=np.float64),
        "image_shape": image_shape,
        "sweep_keys": list(sweep_keys),
        "index_map": index_map,
        "parameter_labels": _index_map_scalar_labels(index_map),
        "fim_display_labels": fim_labels,
        "loss_convention": {
            "maker": "dluxshera.inference.optimization.make_binder_nll_fn",
            "fim": "dluxshera.inference.optimization.fim_theta",
            "data": "binder.model() nominal SheraBinder image",
            "var": "data",
            "noise_model": "gaussian",
            "reduce": "sum",
            "theta0_store": "base_store",
            "add_noise": bool(add_noise),
            "seed": int(seed),
        },
    }


def compute_s10_nominal_science_fim_source_inputs(
    *,
    catalog_labels: Sequence[str],
    prescription_path: Path | None = None,
) -> dict[str, Any]:
    """Reconstruct the V4 Fisher-scale nominal system and compute its full science FIM."""
    ensure_jax_x64_enabled()
    path = (
        Path(__file__).resolve().parents[3] / "work" / "experiments" / "ml_dataset_v3_template.yaml"
        if prescription_path is None
        else Path(prescription_path)
    )
    user_cfg, system_cfg, experiment_raw = _load_and_resolve_prescription(
        prescription_path=path,
        system_preset=None,
        experiment_preset=None,
    )
    experiment_cfg = _validate_experiment_config(experiment_raw)
    forward_spec, base_store, binder = _build_nominal_store(
        system_cfg=system_cfg,
        experiment_cfg=experiment_cfg,
    )
    sweep_keys = list(experiment_cfg["sweep_keys"])
    result = compute_nominal_fisher_matrix(
        binder=binder,
        system_cfg=system_cfg,
        forward_spec=forward_spec,
        base_store=base_store,
        sweep_keys=sweep_keys,
        seed=int(experiment_cfg["seed"]),
        add_noise=bool(experiment_cfg["add_noise"]),
    )
    labels = tuple(str(v) for v in result["parameter_labels"])
    expected = tuple(str(v) for v in catalog_labels)
    if labels != expected:
        raise ValueError(
            "S10 nominal FIM packed parameter labels do not match the prepared "
            f"catalog science ordering ({labels} != {expected})."
        )
    return {
        **result,
        "system_cfg": system_cfg,
        "experiment_cfg": experiment_cfg,
        "user_cfg": user_cfg,
        "prescription_path": str(path),
        "prescription_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "script_version": SCRIPT_VERSION,
        "source_implementation": "dluxshera.ml.nominal_fim.compute_s10_nominal_science_fim_source_inputs",
        "git_info": _git_info(),
    }


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
