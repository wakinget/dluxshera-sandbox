"""
Generate a Fisher-scaled one-parameter-at-a-time ML training dataset (V2).

V2 behavior highlights
----------------------
- Preserves the one-parameter-at-a-time sweep philosophy from V1.
- Generates the nominal (unperturbed) sample once at run start.
- Generates nonzero perturbations only, mirrored across sign.
- Uses per-parameter Fisher-derived sigma values for delta scaling.
- Uses per-parameter log-spaced sigma-offset configuration.

Sigma-space semantics
---------------------
Log spacing is defined in sigma units (not raw parameter units):
1) generate sigma offsets (e.g., ±0.1, ±0.316, ±1.0),
2) map to parameter deltas with ``delta_value = sigma_offset * parameter_sigma``,
3) apply ``parameter_value = nominal_value + delta_value``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from astropy.io import fits

from dluxshera.inference.optimization import fim_theta, generate_fim_labels
from dluxshera.inference.optimization import make_binder_nll_fn
from dluxshera.params.packing import build_index_map, pack_params
from dluxshera.params.spec import build_inference_spec_basic, make_inference_subspec
from dluxshera.params.store import ParameterStore, strip_structural
from dluxshera.systems.three_plane import (
    SHERA_TESTBED_CONFIG,
    SheraThreePlaneBinder,
    SheraThreePlaneConfig,
    build_forward_spec_from_config,
)

JAX_ENABLE_X64 = True
PRINT_EVERY = 10
SCRIPT_VERSION = "v2"

INFER_KEYS = (
    "binary.separation_as",
    "binary.position_angle_deg",
    "binary.x_position_as",
    "binary.y_position_as",
    "binary.log_flux_total",
    "binary.contrast",
    "system.plate_scale_as_per_pix",
    "primary.zernike_coeffs_nm",
    "secondary.zernike_coeffs_nm",
)


@dataclass(frozen=True)
class SweepConfig:
    min_sigma: float = 0.1
    max_sigma: float = 10.0
    n_magnitudes: int = 8
    spacing: str = "log"


def _log(msg: str) -> None:
    print(f"[generate_training_dataset_v2] {msg}")


def _log_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_value(val) for key, val in value.items()}
    return value


def _resolve_run_dir(outdir: str | None, run_name: str | None) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    prefix = "ml_training_dataset_v2_"
    if outdir is None:
        suffix = run_name or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return repo_root / "Results" / f"{prefix}{suffix}"
    base = Path(outdir).expanduser().resolve()
    if run_name is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return base / f"{prefix}{timestamp}"
    return base / run_name


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
        raise ValueError(f"Unsupported spacing '{spacing}'. Currently only 'log' is supported.")

    magnitudes = np.geomspace(min_sigma, max_sigma, num=n_magnitudes)
    negatives = [-float(v) for v in magnitudes[::-1]]
    positives = [float(v) for v in magnitudes]
    return negatives + positives


def _normalize_sweep_configs(
    *,
    infer_keys: list[str],
    default_cfg: SweepConfig,
    overrides: dict[str, dict[str, Any]],
) -> dict[str, SweepConfig]:
    normalized: dict[str, SweepConfig] = {}
    for key in infer_keys:
        payload = dict(overrides.get(key, {}))
        cfg = SweepConfig(
            min_sigma=float(payload.get("min_sigma", default_cfg.min_sigma)),
            max_sigma=float(payload.get("max_sigma", default_cfg.max_sigma)),
            n_magnitudes=int(payload.get("n_magnitudes", default_cfg.n_magnitudes)),
            spacing=str(payload.get("spacing", default_cfg.spacing)),
        )
        _ = generate_mirrored_sigma_offsets(
            min_sigma=cfg.min_sigma,
            max_sigma=cfg.max_sigma,
            n_magnitudes=cfg.n_magnitudes,
            spacing=cfg.spacing,
        )
        normalized[key] = cfg
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




def compute_expected_sample_counts(*, n_swept_components: int, n_magnitudes: int) -> dict[str, int]:
    """Return expected nominal/perturbed totals for V2 one-parameter sweeps."""
    if n_swept_components < 0:
        raise ValueError("n_swept_components must be >= 0.")
    if n_magnitudes < 1:
        raise ValueError("n_magnitudes must be >= 1.")
    perturbed = n_swept_components * (2 * n_magnitudes)
    return {"nominal": 1, "perturbed": perturbed, "total": 1 + perturbed}


def compute_preview_counts(
    *,
    per_parameter_cfg: dict[str, SweepConfig],
    scalar_keys: list[str],
    zernike_component_counts: dict[str, int],
) -> dict[str, int]:
    """Compute exact expected totals honoring per-parameter n_magnitudes."""
    perturbed = 0
    for key in scalar_keys:
        perturbed += 2 * per_parameter_cfg[key].n_magnitudes
    for key, n_components in zernike_component_counts.items():
        perturbed += n_components * (2 * per_parameter_cfg[key].n_magnitudes)
    return {"nominal": 1, "perturbed": perturbed, "total": 1 + perturbed}

def _parse_sweep_overrides(raw: str | None) -> dict[str, dict[str, Any]]:
    if raw is None:
        return {}
    path = Path(raw)
    payload = json.loads(path.read_text()) if path.exists() else json.loads(raw)
    if isinstance(payload, dict) and "parameters" in payload:
        payload = payload["parameters"]
    if not isinstance(payload, dict):
        raise ValueError("sweep-config-json must decode to a dict of parameter overrides.")
    parsed: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"Sweep override for {key} must be an object.")
        parsed[str(key)] = dict(value)
    return parsed


def _infer_values(store: ParameterStore, keys: list[str]) -> dict[str, Any]:
    return {key: _serialize_value(store.get(key)) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a forward-model training dataset sweep (V2).")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--exclude-secondary-zernikes", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Compute Fisher sigmas and sweep preview only; "
            "exit before emitting images or output files."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--add-noise", action="store_true")
    parser.add_argument("--min-sigma", type=float, default=0.1)
    parser.add_argument("--max-sigma", type=float, default=10.0)
    parser.add_argument("--n-magnitudes", type=int, default=8)
    parser.add_argument("--spacing", choices=("log",), default="log")
    parser.add_argument(
        "--sweep-config-json",
        type=str,
        default=None,
        help=(
            "JSON string or path providing per-parameter overrides keyed by infer key. "
            "Each override supports min_sigma, max_sigma, n_magnitudes, spacing."
        ),
    )
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    cfg: SheraThreePlaneConfig = SHERA_TESTBED_CONFIG
    cfg = cfg.replace(primary_noll_indices=tuple(range(4, 12)))
    cfg = cfg.replace(secondary_noll_indices=tuple(range(4, 12)))
    if args.exclude_secondary_zernikes:
        cfg = cfg.replace(secondary_noll_indices=None)

    forward_spec = build_forward_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic(cfg)

    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = base_store.refresh_derived(forward_spec)

    binder = SheraThreePlaneBinder(cfg, forward_spec, base_store)

    infer_keys = list(INFER_KEYS)
    if args.exclude_secondary_zernikes:
        infer_keys = [key for key in infer_keys if key != "secondary.zernike_coeffs_nm"]

    inference_subspec = make_inference_subspec(base_spec=inference_spec, infer_keys=infer_keys, cfg=cfg)

    rng_key = jr.PRNGKey(args.seed)
    data = binder.model()
    data, noise_mode, noise_seed = _noise_model(rng_key, data, add_noise=args.add_noise)

    nll_loss_fn, _theta0 = make_binder_nll_fn(
        binder=binder,
        infer_keys=infer_keys,
        data=data,
        var=data,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=base_store,
    )
    theta_ref = pack_params(inference_subspec, base_store)
    F = fim_theta(nll_loss_fn, theta_ref)
    fim_diag = np.asarray(jnp.diag(F))
    fim_labels = generate_fim_labels(infer_keys, cfg=cfg, store=base_store)
    _validate_fim_diag(fim_diag, labels=fim_labels)
    index_map = build_index_map(inference_subspec, base_store, theta=theta_ref)

    def sigma_for_key_component(param_key: str, component_index: int | None) -> float:
        for entry in index_map["entries"]:
            if entry["name"] != param_key:
                continue
            start = int(entry["start"])
            idx = start if component_index is None else start + component_index
            return float(1.0 / math.sqrt(fim_diag[idx]))
        raise KeyError(f"Missing FIM mapping for key {param_key}.")

    default_cfg = SweepConfig(
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        n_magnitudes=args.n_magnitudes,
        spacing=args.spacing,
    )
    per_parameter_cfg = _normalize_sweep_configs(
        infer_keys=infer_keys,
        default_cfg=default_cfg,
        overrides=_parse_sweep_overrides(args.sweep_config_json),
    )

    scalar_keys = [
        key
        for key in infer_keys
        if key not in ("primary.zernike_coeffs_nm", "secondary.zernike_coeffs_nm")
    ]
    zernike_keys = [
        key
        for key in infer_keys
        if key in ("primary.zernike_coeffs_nm", "secondary.zernike_coeffs_nm")
    ]
    zernike_map = {
        "primary.zernike_coeffs_nm": tuple(cfg.primary_noll_indices),
        "secondary.zernike_coeffs_nm": tuple(cfg.secondary_noll_indices),
    }

    sigma_summaries: list[dict[str, Any]] = []
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
        zernike_component_counts={
            key: int(np.asarray(base_store.get(key)).size) for key in zernike_keys
        },
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

    run_dir = _resolve_run_dir(args.outdir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=False)
    images_dir = run_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    baseline_infer = _infer_values(base_store, infer_keys)
    manifest = {
        "script": "generate_training_dataset_v2.py",
        "version": SCRIPT_VERSION,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "config_id": cfg.design_name,
        "git_commit": _git_commit(),
        "parameters": infer_keys,
        "nominal_values": baseline_infer,
        "sweep_configuration": {
            "defaults": asdict(default_cfg),
            "per_parameter": {k: asdict(v) for k, v in per_parameter_cfg.items()},
            "sigma_ordering": "negative largest->smallest then positive smallest->largest",
            "nonzero_only": True,
            "nominal_sample_generated_once": True,
        },
        "fisher_sigma_summary": sigma_summaries,
        "noise": {
            "enabled": bool(args.add_noise),
            "mode": noise_mode,
            "seed": args.seed,
            "realization_seed": noise_seed,
        },
    }

    samples_path = run_dir / "samples.jsonl"
    samples_path.write_text("")

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
        nonlocal sample_id, nominal_count, perturbed_count
        sample_id += 1
        sample_tag = f"sample_{sample_id:06d}"
        fits_path = images_dir / f"{sample_tag}.fits"
        meta_path = images_dir / f"{sample_tag}.json"

        model = binder.model(strip_structural(applied_store, structural_keys=binder.structural_store_keys()))
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
                "NOISE": (bool(args.add_noise), "Noise added to image"),
                "SEED": (args.seed, "Base RNG seed"),
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
            "values": _infer_values(applied_store, infer_keys),
        }
        meta_path.write_text(json.dumps(_serialize_value(sample_meta), indent=2))
        with samples_path.open("a") as handle:
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
        _log(f"Progress: sample={sample_id} key={current_key} idx={sweep_index} applied={applied_value:.6g} ({rate:.2f} samples/s)")

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
            store_delta = base_store.replace({key: parameter_value}).refresh_derived(forward_spec)
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
                store_delta = base_store.replace({key: updated}).refresh_derived(forward_spec)
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
    (run_dir / "manifest.json").write_text(json.dumps(_serialize_value(manifest), indent=2))

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
