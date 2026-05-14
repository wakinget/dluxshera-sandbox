"""Run a focused temporal-registration comparison study.

Smoke commands
--------------
Dry-run default plan:

```
PYTHONPATH=src python examples/scripts/run_temporal_registration_comparison.py \
  --results-root Results/temporal_registration_comparison \
  --run-name smoke_default \
  --dry-run
```

Run a tiny 3-frame noiseless solver smoke:

```
PYTHONPATH=src python examples/scripts/run_temporal_registration_comparison.py \
  --results-root Results/temporal_registration_comparison \
  --run-name smoke_3f_noiseless \
  --n-frames 3 \
  --noise disabled \
  --init-mode truth_plus_offset \
  --case-filter drift_truth__linear_drift_fit
```

Run the focused 20-frame shot-noise comparison:

```
PYTHONPATH=src python examples/scripts/run_temporal_registration_comparison.py \
  --results-root Results/temporal_registration_comparison \
  --run-name focused_20f_shotnoise \
  --n-frames 20 \
  --noise enabled \
  --init-mode truth \
  --reference-diagnostics-profile basic
```

Dry-run the full matrix including residual-prior fits:

```
PYTHONPATH=src python examples/scripts/run_temporal_registration_comparison.py \
  --results-root Results/temporal_registration_comparison \
  --run-name residual_prior_full_dryrun \
  --noise disabled \
  --init-mode truth \
  --full-default-matrix \
  --dry-run
```

Run a tiny residual-prior solver smoke:

```
PYTHONPATH=src python examples/scripts/run_temporal_registration_comparison.py \
  --results-root Results/temporal_registration_comparison \
  --run-name residual_prior_3f_noiseless \
  --n-frames 3 \
  --noise disabled \
  --init-mode truth_plus_offset \
  --case-filter drift_resid10mas_truth__residual_prior_fit \
  --full-default-matrix
```

Run a focused 20-frame shot-noise residual-prior comparison:

```
PYTHONPATH=src python examples/scripts/run_temporal_registration_comparison.py \
  --results-root Results/temporal_registration_comparison \
  --run-name residual_prior_20f_shotnoise \
  --n-frames 20 \
  --noise enabled \
  --init-mode truth \
  --full-default-matrix \
  --reference-diagnostics-profile basic
```

Routing note:
- `independent` cases use `schur_curvature_method=auto` with effective
  `max_dense_dim=40`, so 20-frame four-scalar runs stay on the structured
  independent-frame Schur path.
- `linear_drift` cases use dense Schur (compact nuisance state).
- `linear_drift_residual_jitter_prior` cases use dense Schur with effective
  dense guard at least `80` because the profiled temporal prior couples frames.
- All cases initialize from the generated truth trace unless
  `--init-mode truth_plus_offset` is selected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dluxshera.config.io import load_user_config
from dluxshera.inference.observation_summary import load_subblock_summary
from dluxshera.utils.obs_subblock_io import write_obs_subblock_truth_csv

from examples.recipes import observation_subblock
from examples.recipes import observation_subblock_inference
from examples.scripts import run_obs_subblock_study


TRACE_KEYS = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)
THETA_KEYS_DEFAULT = (
    "source.separation_as",
    "source.log_flux_total",
    "source.contrast",
    "optics.plate_scale_as_per_pix",
)
TRACE_COLUMNS = ("frame_index", "time_s", *TRACE_KEYS)


@dataclass(frozen=True)
class CaseSpec:
    case_name: str
    truth_model: str
    fit_model: str
    noise_mode: str = "disabled"
    init_mode: str = "truth"
    iid_x_sigma_as: float = 0.050
    iid_y_sigma_as: float = 0.050
    iid_pa_sigma_deg: float = 2.0e-3
    x_rate_sigma_as_per_s: float = 1.0
    y_rate_sigma_as_per_s: float = 1.0
    pa_rate_sigma_deg_per_s: float = 1.0e-2
    x_rate_as_per_s: float | None = None
    y_rate_as_per_s: float | None = None
    pa_rate_deg_per_s: float | None = None
    residual_x_sigma_as: float = 0.0
    residual_y_sigma_as: float = 0.0
    residual_pa_sigma_deg: float = 0.0
    fit_residual_x_sigma_as: float | None = None
    fit_residual_y_sigma_as: float | None = None
    fit_residual_pa_sigma_deg: float | None = None
    trace_seed: int = 0
    render_seed: int = 0


def _stable_seed(seed: int, *parts: object) -> int:
    payload = json.dumps([int(seed), *map(str, parts)], sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _frame_times(n_frames: int, exposure_time_s: float) -> np.ndarray:
    return (np.arange(n_frames, dtype=float) + 0.5) * float(exposure_time_s)


def _trace_stats(values: np.ndarray, times: np.ndarray) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    centered = times - float(np.mean(times))
    denom = float(np.sum(centered * centered))
    for idx, key in enumerate(TRACE_KEYS):
        series = np.asarray(values[:, idx], dtype=float)
        slope = 0.0 if denom == 0.0 else float(np.sum((series - np.mean(series)) * centered) / denom)
        intercept = float(np.mean(series))
        residual = series - (intercept + slope * centered)
        stats[key] = {
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            "rms": float(np.sqrt(np.mean(series * series))),
            "min": float(np.min(series)),
            "max": float(np.max(series)),
            "linear_slope": slope,
            "linear_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        }
    return stats


def build_iid_registration_trace(
    *,
    n_frames: int,
    exposure_time_s: float,
    seed: int,
    x_sigma_as: float = 0.050,
    y_sigma_as: float = 0.050,
    pa_sigma_deg: float = 2.0e-3,
    anchor: Sequence[float] = (0.0, 0.0, 90.0),
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    times = _frame_times(n_frames, exposure_time_s)
    anchor_arr = np.asarray(anchor, dtype=float)
    sigmas = np.asarray([x_sigma_as, y_sigma_as, pa_sigma_deg], dtype=float)
    values = anchor_arr[None, :] + rng.normal(0.0, sigmas[None, :], size=(n_frames, 3))
    return _rows_and_manifest(
        values=values,
        times=times,
        truth_model="iid_registration",
        seed=seed,
        anchor=anchor_arr,
        iid_sigmas=sigmas,
    )


def build_linear_drift_trace(
    *,
    n_frames: int,
    exposure_time_s: float,
    seed: int,
    x_rate_sigma_as_per_s: float = 1.0,
    y_rate_sigma_as_per_s: float = 1.0,
    pa_rate_sigma_deg_per_s: float = 1.0e-2,
    x_rate_as_per_s: float | None = None,
    y_rate_as_per_s: float | None = None,
    pa_rate_deg_per_s: float | None = None,
    anchor: Sequence[float] = (0.0, 0.0, 90.0),
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rates = np.asarray(
        [
            rng.normal(0.0, x_rate_sigma_as_per_s) if x_rate_as_per_s is None else x_rate_as_per_s,
            rng.normal(0.0, y_rate_sigma_as_per_s) if y_rate_as_per_s is None else y_rate_as_per_s,
            rng.normal(0.0, pa_rate_sigma_deg_per_s) if pa_rate_deg_per_s is None else pa_rate_deg_per_s,
        ],
        dtype=float,
    )
    times = _frame_times(n_frames, exposure_time_s)
    centered = times - float(np.mean(times))
    anchor_arr = np.asarray(anchor, dtype=float)
    values = anchor_arr[None, :] + centered[:, None] * rates[None, :]
    rows, manifest = _rows_and_manifest(
        values=values,
        times=times,
        truth_model="linear_drift",
        seed=seed,
        anchor=anchor_arr,
        rates=rates,
    )
    manifest["rate_sampling"] = {
        "x_rate_sigma_as_per_s": float(x_rate_sigma_as_per_s),
        "y_rate_sigma_as_per_s": float(y_rate_sigma_as_per_s),
        "pa_rate_sigma_deg_per_s": float(pa_rate_sigma_deg_per_s),
        "explicit_rates_provided": {
            "x": x_rate_as_per_s is not None,
            "y": y_rate_as_per_s is not None,
            "pa": pa_rate_deg_per_s is not None,
        },
    }
    return rows, manifest


def build_linear_drift_residual_jitter_trace(
    *,
    n_frames: int,
    exposure_time_s: float,
    seed: int,
    residual_x_sigma_as: float = 0.010,
    residual_y_sigma_as: float = 0.010,
    residual_pa_sigma_deg: float = 1.0e-4,
    **drift_kwargs: Any,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    rows, manifest = build_linear_drift_trace(
        n_frames=n_frames,
        exposure_time_s=exposure_time_s,
        seed=seed,
        **drift_kwargs,
    )
    rng = np.random.default_rng(_stable_seed(seed, "residual"))
    values = np.asarray([[row[key] for key in TRACE_KEYS] for row in rows], dtype=float)
    residual_sigmas = np.asarray(
        [residual_x_sigma_as, residual_y_sigma_as, residual_pa_sigma_deg],
        dtype=float,
    )
    values = values + rng.normal(0.0, residual_sigmas[None, :], size=values.shape)
    times = np.asarray([row["time_s"] for row in rows], dtype=float)
    rows, residual_manifest = _rows_and_manifest(
        values=values,
        times=times,
        truth_model="linear_drift_residual_jitter",
        seed=seed,
        anchor=np.asarray(manifest["anchor_values"], dtype=float),
        rates=np.asarray(manifest["drift_rates"], dtype=float),
        residual_sigmas=residual_sigmas,
    )
    residual_manifest["rate_sampling"] = manifest.get("rate_sampling", {})
    return rows, residual_manifest


def _rows_and_manifest(
    *,
    values: np.ndarray,
    times: np.ndarray,
    truth_model: str,
    seed: int,
    anchor: np.ndarray,
    rates: np.ndarray | None = None,
    iid_sigmas: np.ndarray | None = None,
    residual_sigmas: np.ndarray | None = None,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    rows: list[dict[str, float]] = []
    for idx, time_s in enumerate(times):
        row = {"frame_index": int(idx), "time_s": float(time_s)}
        row.update({key: float(values[idx, key_idx]) for key_idx, key in enumerate(TRACE_KEYS)})
        rows.append(row)
    manifest = {
        "schema_version": "temporal_trace_truth_manifest.v1",
        "truth_model": truth_model,
        "frame_count": int(values.shape[0]),
        "exposure_time_s": float(times[0] * 2.0) if len(times) else None,
        "time_convention": "local_subblock_frame_centers_seconds",
        "anchor_values": [float(v) for v in anchor],
        "drift_rates": None if rates is None else [float(v) for v in rates],
        "iid_sigmas": None if iid_sigmas is None else [float(v) for v in iid_sigmas],
        "residual_jitter_sigmas": None if residual_sigmas is None else [float(v) for v in residual_sigmas],
        "seed": int(seed),
        "units": {
            "source.x_position_as": "arcsec",
            "source.y_position_as": "arcsec",
            "source.position_angle_deg": "degree",
            "time_s": "second",
        },
        "trace_statistics": _trace_stats(values, times),
    }
    return rows, manifest


def write_temporal_trace_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    write_obs_subblock_truth_csv(output_path=path, rows=rows, fieldnames=TRACE_COLUMNS)


def write_temporal_trace_manifest(path: Path, manifest: Mapping[str, Any], csv_path: Path) -> None:
    payload = dict(manifest)
    payload["generated_csv_path"] = str(csv_path.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _default_fit_residual_sigmas(*, truth_model: str, residual_xy: float) -> tuple[float, float, float]:
    if truth_model == "linear_drift_residual_jitter" and residual_xy > 0.0:
        return residual_xy, residual_xy, 1.0e-4
    return 0.010, 0.010, 1.0e-4


def default_case_specs(
    *,
    seed: int,
    noise_mode: str,
    init_mode: str,
    full: bool = False,
    residual_prior_sweep: bool = False,
) -> list[CaseSpec]:
    base = [
        ("iid50_truth__independent_fit", "iid_registration", "independent", 0.0),
        ("iid50_truth__linear_drift_fit", "iid_registration", "linear_drift", 0.0),
        ("drift_truth__independent_fit", "linear_drift", "independent", 0.0),
        ("drift_truth__linear_drift_fit", "linear_drift", "linear_drift", 0.0),
    ]
    if full:
        base.extend(
            [
                ("iid50_truth__residual_prior_fit", "iid_registration", "linear_drift_residual_jitter_prior", 0.0),
                ("drift_truth__residual_prior_fit", "linear_drift", "linear_drift_residual_jitter_prior", 0.0),
                ("drift_resid10mas_truth__independent_fit", "linear_drift_residual_jitter", "independent", 0.010),
                ("drift_resid10mas_truth__linear_drift_fit", "linear_drift_residual_jitter", "linear_drift", 0.010),
                ("drift_resid10mas_truth__residual_prior_fit", "linear_drift_residual_jitter", "linear_drift_residual_jitter_prior", 0.010),
                ("drift_resid50mas_truth__independent_fit", "linear_drift_residual_jitter", "independent", 0.050),
                ("drift_resid50mas_truth__linear_drift_fit", "linear_drift_residual_jitter", "linear_drift", 0.050),
                ("drift_resid50mas_truth__residual_prior_fit", "linear_drift_residual_jitter", "linear_drift_residual_jitter_prior", 0.050),
            ]
        )
    if residual_prior_sweep:
        base.extend(
            [
                ("drift_resid10mas_truth__residual_prior5mas_fit", "linear_drift_residual_jitter", "linear_drift_residual_jitter_prior", 0.010),
                ("drift_resid10mas_truth__residual_prior10mas_fit", "linear_drift_residual_jitter", "linear_drift_residual_jitter_prior", 0.010),
                ("drift_resid10mas_truth__residual_prior25mas_fit", "linear_drift_residual_jitter", "linear_drift_residual_jitter_prior", 0.010),
                ("drift_resid50mas_truth__residual_prior10mas_fit", "linear_drift_residual_jitter", "linear_drift_residual_jitter_prior", 0.050),
                ("drift_resid50mas_truth__residual_prior25mas_fit", "linear_drift_residual_jitter", "linear_drift_residual_jitter_prior", 0.050),
                ("drift_resid50mas_truth__residual_prior50mas_fit", "linear_drift_residual_jitter", "linear_drift_residual_jitter_prior", 0.050),
            ]
        )
    specs: list[CaseSpec] = []
    for name, truth, fit, residual in base:
        fit_sigmas = (None, None, None)
        if fit == "linear_drift_residual_jitter_prior":
            fit_sigmas = _default_fit_residual_sigmas(
                truth_model=truth,
                residual_xy=residual,
            )
            if "prior5mas" in name:
                fit_sigmas = (0.005, 0.005, 1.0e-4)
            elif "prior10mas" in name:
                fit_sigmas = (0.010, 0.010, 1.0e-4)
            elif "prior25mas" in name:
                fit_sigmas = (0.025, 0.025, 1.0e-4)
            elif "prior50mas" in name:
                fit_sigmas = (0.050, 0.050, 1.0e-4)
        specs.append(
            CaseSpec(
                case_name=name,
                truth_model=truth,
                fit_model=fit,
                noise_mode=noise_mode,
                init_mode=init_mode,
                residual_x_sigma_as=residual,
                residual_y_sigma_as=residual,
                residual_pa_sigma_deg=1.0e-4 if residual > 0 else 0.0,
                fit_residual_x_sigma_as=fit_sigmas[0],
                fit_residual_y_sigma_as=fit_sigmas[1],
                fit_residual_pa_sigma_deg=fit_sigmas[2],
                trace_seed=_stable_seed(seed, name, "trace"),
                render_seed=_stable_seed(seed, name, "render"),
            )
        )
    return specs


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = load_user_config(config_path=path)
    return payload if isinstance(payload, dict) else {}


def _write_plan(run_root: Path, cases: Sequence[CaseSpec], manifest: Mapping[str, Any]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = [asdict(case) for case in cases]
    (run_root / "comparison_plan.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (run_root / "comparison_plan.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["case_name"])
        writer.writeheader()
        writer.writerows(rows)


def _trace_for_case(case: CaseSpec, *, n_frames: int, exposure_time_s: float) -> tuple[list[dict[str, float]], dict[str, Any]]:
    common = {
        "n_frames": n_frames,
        "exposure_time_s": exposure_time_s,
        "seed": case.trace_seed,
    }
    if case.truth_model == "iid_registration":
        return build_iid_registration_trace(
            **common,
            x_sigma_as=case.iid_x_sigma_as,
            y_sigma_as=case.iid_y_sigma_as,
            pa_sigma_deg=case.iid_pa_sigma_deg,
        )
    drift = {
        "x_rate_sigma_as_per_s": case.x_rate_sigma_as_per_s,
        "y_rate_sigma_as_per_s": case.y_rate_sigma_as_per_s,
        "pa_rate_sigma_deg_per_s": case.pa_rate_sigma_deg_per_s,
        "x_rate_as_per_s": case.x_rate_as_per_s,
        "y_rate_as_per_s": case.y_rate_as_per_s,
        "pa_rate_deg_per_s": case.pa_rate_deg_per_s,
    }
    if case.truth_model == "linear_drift":
        return build_linear_drift_trace(**common, **drift)
    if case.truth_model == "linear_drift_residual_jitter":
        return build_linear_drift_residual_jitter_trace(
            **common,
            **drift,
            residual_x_sigma_as=case.residual_x_sigma_as,
            residual_y_sigma_as=case.residual_y_sigma_as,
            residual_pa_sigma_deg=case.residual_pa_sigma_deg,
        )
    raise ValueError(f"Unsupported truth model: {case.truth_model!r}.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_render_variance(case_root: Path) -> tuple[str, str, str]:
    manifest_path = case_root / "render" / "manifest.json"
    manifest = _read_json(manifest_path)
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest.get("artifacts"), Mapping) else {}
    for key in ("variance_fits", "variance_cube", "variance"):
        value = artifacts.get(key)
        if isinstance(value, str) and value.strip():
            path = Path(value)
            if not path.is_absolute():
                path = manifest_path.parent / path
            if path.exists():
                return str(path.resolve()), "found", ""
    return "", "missing", "Renderer variance FITS sidecar was not found; using configured fallback variance model."


def _patch_inference_variance_config(
    inference_path: Path,
    *,
    case: CaseSpec,
    case_root: Path,
) -> dict[str, str]:
    cfg = _read_json(inference_path)
    noise_model = cfg["experiment"]["inference"]["objective"]["noise_model"]
    status = {"variance_model_used": str(noise_model.get("variance_model", "")), "variance_fits": "", "variance_source_status": "not_requested", "variance_warning": ""}
    if case.noise_mode != "enabled":
        status["variance_source_status"] = "noise_disabled"
        return status
    variance_fits, source_status, warning = _find_render_variance(case_root)
    status.update(
        {
            "variance_fits": variance_fits,
            "variance_source_status": source_status,
            "variance_warning": warning,
        }
    )
    if variance_fits:
        noise_model["variance_model"] = "provided_cube"
        noise_model["path"] = variance_fits
        noise_model.pop("scalar", None)
        status["variance_model_used"] = "provided_cube"
        _write_json(inference_path, cfg)
    else:
        status["variance_model_used"] = str(noise_model.get("variance_model", "data"))
    return status


def _schur_settings_for_case(
    case: CaseSpec,
    *,
    requested_max_dense_dim: int,
) -> dict[str, Any]:
    """Return per-case Schur routing settings and rationale."""

    if case.fit_model == "independent":
        return {
            "schur_curvature_method": "auto",
            "max_dense_dim": min(int(requested_max_dense_dim), 40),
            "reason": "independent frame layout should prefer structured Schur at 20 frames",
        }
    if case.fit_model == "linear_drift":
        return {
            "schur_curvature_method": "dense",
            "max_dense_dim": int(requested_max_dense_dim),
            "reason": "hard linear drift uses compact nuisance dimension; dense Schur is explicit",
        }
    if case.fit_model == "linear_drift_residual_jitter_prior":
        return {
            "schur_curvature_method": "dense",
            "max_dense_dim": max(int(requested_max_dense_dim), 80),
            "reason": "profiled residual prior couples frames and requires dense Schur",
        }
    raise ValueError(f"Unsupported fit model for Schur routing: {case.fit_model!r}.")


def _case_configs(
    *,
    case: CaseSpec,
    case_root: Path,
    trace_csv: Path,
    n_frames: int,
    exposure_time_s: float,
    system_preset: str,
) -> tuple[Path, Path]:
    render_cfg = {
        "system": {"preset": system_preset, "source": {"target": "ALPHA_CEN", "exposure_time_s": exposure_time_s}},
        "experiment": {
            "kind": "subblock_generation",
            "seed": int(case.render_seed),
            "subblock": {
                "varying_keys": list(TRACE_KEYS),
                "trace": {"format": "csv", "path": str(trace_csv)},
                "validate": {"require_contiguous_frame_index": True, "require_monotonic_time": True},
            },
            "noise": {"enabled": case.noise_mode == "enabled", "photon_noise": True, "read_noise": False, "dark_current": False},
            "outputs": {"outdir": str(case_root), "file_prefix": "obs_subblock", "frame_truth_format": "csv"},
        },
    }
    offsets = {}
    if case.init_mode == "truth_plus_offset":
        offsets = {
            "source.x_position_as": 1.0e-3,
            "source.y_position_as": 1.0e-3,
            "source.position_angle_deg": 1.0e-5,
        }
    frame_model: dict[str, Any] = {"kind": case.fit_model}
    if case.fit_model == "linear_drift_residual_jitter_prior":
        fx = 0.010 if case.fit_residual_x_sigma_as is None else case.fit_residual_x_sigma_as
        fy = 0.010 if case.fit_residual_y_sigma_as is None else case.fit_residual_y_sigma_as
        fpa = 1.0e-4 if case.fit_residual_pa_sigma_deg is None else case.fit_residual_pa_sigma_deg
        frame_model = {
            "kind": "linear_drift_residual_jitter_prior",
            "residual_prior": {
                "source.x_position_as": {"sigma": fx},
                "source.y_position_as": {"sigma": fy},
                "source.position_angle_deg": {"sigma": fpa},
            },
            "reduce": "match_subblock_reduce",
        }
    optimizer_base_lr = 0.9
    optimizer_n_iter = 40 if n_frames <= 3 else 100
    if case.fit_model == "linear_drift_residual_jitter_prior":
        optimizer_base_lr = 0.05
        optimizer_n_iter = 50
    inference_cfg = {
        "system": {"preset": system_preset, "source": {"target": "ALPHA_CEN", "exposure_time_s": exposure_time_s}},
        "experiment": {
            "kind": "subblock_inference",
            "inference": {
                "data": {
                    "cube": str(case_root / "render" / "obs_subblock_cube.fits"),
                    "truth_trace": str(trace_csv),
                    "manifest": str(case_root / "render" / "manifest.json"),
                },
                "validate": {"require_contiguous_frame_index": True, "require_monotonic_time": True},
                "active": {"frame_keys": list(TRACE_KEYS), "shared_keys": []},
                "init": {"frame": {"mode": "from_truth_trace", "offsets": offsets}, "shared": {}},
                "priors": {"frame": {}, "shared": {}},
                "temporal": {"frame_model": frame_model},
                "objective": {
                    "kind": "nll",
                    "frame_reduce": "sum",
                    "subblock_reduce": "mean",
                    "noise_model": {"kind": "gaussian", "variance_model": "data", "variance_floor": 1.0},
                },
                "optimizer": {
                    "kind": "sgd",
                    "base_lr": optimizer_base_lr,
                    "n_iter": optimizer_n_iter,
                    "preconditioning": {"enabled": True, "method": "auto", "reference": "initial"},
                },
                "diagnostics": {"plots": False, "compare_to_truth_when_available": True},
            },
            "outputs": {"outdir": str(case_root), "file_prefix": "obs_subblock_inference"},
        },
    }
    render_path = case_root / "render_config.json"
    inference_path = case_root / "inference_config.json"
    _write_json(render_path, render_cfg)
    _write_json(inference_path, inference_cfg)
    return render_path, inference_path


def _deterministic_latest(paths: Sequence[Path]) -> Path | None:
    candidates = [path for path in paths if path.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path)))[-1]


def _copy_latest_artifacts(
    render_dir: Path,
    *,
    render_result: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    selected: dict[str, str] = {}
    artifacts = dict((render_result or {}).get("artifacts", {}))
    cube = Path(artifacts["cube_fits"]) if artifacts.get("cube_fits") else _deterministic_latest(tuple(render_dir.glob("obs_subblock_*_cube.fits")))
    truth = Path(artifacts["frame_truth_csv"]) if artifacts.get("frame_truth_csv") else _deterministic_latest(tuple(render_dir.glob("obs_subblock_*_frame_truth.csv")))
    if cube is not None and cube.exists():
        shutil.copy2(cube, render_dir / "obs_subblock_cube.fits")
        selected["cube_fits"] = str(cube.resolve())
    if truth is not None and truth.exists():
        shutil.copy2(truth, render_dir / "obs_subblock_frame_truth.csv")
        selected["frame_truth_csv"] = str(truth.resolve())
    return selected


def _read_recovered_errors(truth_csv: Path, recovered_csv: Path) -> dict[str, float]:
    with truth_csv.open("r", encoding="utf-8", newline="") as handle:
        truth_rows = list(csv.DictReader(handle))
    with recovered_csv.open("r", encoding="utf-8", newline="") as handle:
        recovered_rows = list(csv.DictReader(handle))
    result: dict[str, float] = {}
    for key, suffix in zip(TRACE_KEYS, ("x", "y", "pa")):
        diff = np.asarray(
            [float(rec[key]) - float(truth[key]) for truth, rec in zip(truth_rows, recovered_rows)],
            dtype=float,
        )
        unit = "deg" if suffix == "pa" else "as"
        result[f"rms_{suffix}_error_{unit}"] = float(np.sqrt(np.mean(diff * diff)))
        result[f"max_abs_{suffix}_error_{unit}"] = float(np.max(np.abs(diff)))
    return result


def _schur_metrics(summary_json: Path, theta_keys: Sequence[str]) -> dict[str, Any]:
    if not summary_json.exists():
        return {}
    summary = load_subblock_summary(summary_json)
    labels = list(summary.theta_labels)
    info = np.asarray(summary.reduced_information, dtype=float)
    if "source.separation_as" not in labels:
        return {}
    idx = labels.index("source.separation_as")
    sep_info = float(info[idx, idx])
    metrics: dict[str, Any] = {"separation_schur_information": sep_info}
    if math.isfinite(sep_info) and sep_info > 0.0:
        sigma_uas = math.sqrt(1.0 / sep_info) * 1.0e6
        metrics["separation_sigma_single_subblock_uas"] = sigma_uas
        metrics["forecast_sigma_1800_subblocks_uas"] = sigma_uas / math.sqrt(1800.0)
    try:
        cov = np.linalg.inv(info)
        cov_diag = float(cov[idx, idx])
        if cov_diag > 0.0:
            metrics["separation_sigma_single_subblock_cov_uas"] = math.sqrt(cov_diag) * 1.0e6
    except np.linalg.LinAlgError:
        pass
    return metrics


def _run_case(
    *,
    case: CaseSpec,
    run_root: Path,
    n_frames: int,
    exposure_time_s: float,
    system_preset: str,
    dry_run: bool,
    resume: bool,
    theta_keys: Sequence[str],
    reference_diagnostics_profile: str,
    max_dense_dim: int,
) -> dict[str, Any]:
    case_root = run_root / "cases" / case.case_name
    trace_dir = case_root / "trace"
    trace_csv = trace_dir / "frame_truth.csv"
    trace_manifest = trace_dir / "trace_truth_manifest.json"
    if resume and trace_csv.exists() and trace_manifest.exists():
        manifest = _read_json(trace_manifest)
    else:
        rows, manifest = _trace_for_case(case, n_frames=n_frames, exposure_time_s=exposure_time_s)
        trace_dir.mkdir(parents=True, exist_ok=True)
        write_temporal_trace_csv(trace_csv, rows)
        write_temporal_trace_manifest(trace_manifest, manifest, trace_csv)
    render_cfg, inference_cfg = _case_configs(
        case=case,
        case_root=case_root,
        trace_csv=trace_csv,
        n_frames=n_frames,
        exposure_time_s=exposure_time_s,
        system_preset=system_preset,
    )
    row = {
        **asdict(case),
        "n_frames": int(n_frames),
        "exposure_time_s": float(exposure_time_s),
        "truth_trace_csv": str(trace_csv.resolve()),
        "temporal_prior_kind": case.fit_model if case.fit_model == "linear_drift_residual_jitter_prior" else "",
        "temporal_prior_reduce": "match_subblock_reduce" if case.fit_model == "linear_drift_residual_jitter_prior" else "",
        "schur_curvature_method_requested": "",
        "schur_max_dense_dim_effective": "",
        "schur_routing_reason": "",
        "status": "planned" if dry_run else "started",
    }
    schur_settings = _schur_settings_for_case(
        case,
        requested_max_dense_dim=max_dense_dim,
    )
    row["schur_curvature_method_requested"] = str(schur_settings["schur_curvature_method"])
    row["schur_max_dense_dim_effective"] = int(schur_settings["max_dense_dim"])
    row["schur_routing_reason"] = str(schur_settings["reason"])
    if manifest.get("drift_rates") is not None:
        row.update(
            {
                "x_rate_as_per_s": manifest["drift_rates"][0],
                "y_rate_as_per_s": manifest["drift_rates"][1],
                "pa_rate_deg_per_s": manifest["drift_rates"][2],
            }
        )
    if dry_run:
        row["status"] = "dry_run"
        return row
    render_manifest = case_root / "render" / "manifest.json"
    render_cube = case_root / "render" / "obs_subblock_cube.fits"
    if not (resume and render_manifest.exists() and render_cube.exists()):
        render_result = observation_subblock.generate_obs_subblock(
            config_path=render_cfg,
            results_dir=case_root,
            run_name="render",
            dry_run=False,
            show_progress=False,
        )
        row.update({f"selected_render_{key}": value for key, value in _copy_latest_artifacts(case_root / "render", render_result=render_result).items()})
    variance_status = _patch_inference_variance_config(
        inference_cfg,
        case=case,
        case_root=case_root,
    )
    row.update(variance_status)
    inference_manifest = case_root / "inference" / "manifest.json"
    recovered_csv = _deterministic_latest(tuple((case_root / "inference").glob("*_recovered_trace.csv")))
    if resume and inference_manifest.exists() and recovered_csv is not None and recovered_csv.exists():
        inference_result = {
            "artifacts": {
                "recovered_trace_csv": str(recovered_csv),
                "manifest_json": str(inference_manifest),
            },
            "final_loss": _read_json(inference_manifest).get("metrics", {}).get("final_loss"),
            "chi2": _read_json(inference_manifest).get("metrics", {}).get("chi2", {}),
        }
    else:
        inference_result = observation_subblock_inference.main(
            ["--config", str(inference_cfg), "--run-name", "inference", "--no-progress"]
        )
        recovered_csv = Path(inference_result["artifacts"]["recovered_trace_csv"])
    manifest_json = Path(inference_result["artifacts"]["manifest_json"])
    inference_manifest_payload = _read_json(manifest_json)
    row.update(_read_recovered_errors(trace_csv, recovered_csv))
    if inference_result.get("final_loss") is not None:
        row["final_loss"] = float(inference_result["final_loss"])
    metrics = inference_manifest_payload.get("metrics", {})
    row["initial_temporal_term"] = metrics.get("initial_temporal_term")
    row["final_temporal_term"] = metrics.get("final_temporal_term")
    initial_chi2 = inference_result.get("chi2", {}).get("initial_model", {})
    final_chi2 = inference_result.get("chi2", {}).get("final_model", {})
    row["initial_block_reduced_chi2"] = initial_chi2.get("block_reduced_chi2")
    row["final_block_reduced_chi2"] = final_chi2.get("block_reduced_chi2")
    final_frames = final_chi2.get("per_frame_reduced_chi2", [])
    if isinstance(final_frames, list) and final_frames:
        reduced = np.asarray([float(value) for value in final_frames], dtype=float)
        row["max_final_frame_reduced_chi2"] = float(np.nanmax(reduced))
        row["median_final_frame_reduced_chi2"] = float(np.nanmedian(reduced))
        row["frame_quality_good_frame_count"] = int(np.count_nonzero(reduced <= 5.0))
        row["frame_quality_bad_frame_count"] = int(np.count_nonzero(reduced > 5.0))
    row["recovered_trace_csv"] = str(recovered_csv.resolve())
    row["summary_json"] = ""
    try:
        summary_json = case_root / "study" / "schur_summary" / "subblock_summary.json"
        summary_npz = case_root / "study" / "schur_summary" / "subblock_summary_matrices.npz"
        if not (resume and summary_json.exists() and summary_npz.exists()):
            schur = run_obs_subblock_study.run_obs_subblock_study(
                mode="schur_summary",
                case_root=case_root,
                case_stages=(),
                inference_template=inference_cfg,
                theta_keys=theta_keys,
                phi_ref="recovered",
                schur_curvature_method=str(schur_settings["schur_curvature_method"]),
                max_dense_dim=int(schur_settings["max_dense_dim"]),
                schur_damping=1.0e-8,
                schur_frame_quality_policy="mask",
                schur_frame_chi2_threshold=5.0,
                schur_frame_mask_denominator="original",
                reference_diagnostics_profile=reference_diagnostics_profile,
                reuse_reference_inference=manifest_json.parent,
                dry_run=False,
            )
            summary_json = Path(schur.get("schur_summary", {}).get("summary_json_path", ""))
        row["summary_json"] = str(summary_json.resolve()) if summary_json.exists() else ""
        row.update(_schur_metrics(summary_json, theta_keys))
    except Exception as exc:  # Schur export is expensive; preserve case diagnostics.
        row["schur_error"] = str(exc)
    row["status"] = "completed"
    return row


def _write_aggregate(run_root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    aggregate = run_root / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (aggregate / "case_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        aggregate / "comparison_summary.json",
        {
            "case_count": len(rows),
            "completed_count": sum(1 for row in rows if row.get("status") == "completed"),
            "generated_at_unix": time.time(),
        },
    )
    sep_rows = [
        {
            "case_name": row.get("case_name"),
            "fit_model": row.get("fit_model"),
            "truth_model": row.get("truth_model"),
            "separation_schur_information": row.get("separation_schur_information"),
            "separation_sigma_single_subblock_uas": row.get("separation_sigma_single_subblock_uas"),
            "forecast_sigma_1800_subblocks_uas": row.get("forecast_sigma_1800_subblocks_uas"),
        }
        for row in rows
    ]
    with (aggregate / "separation_information.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sep_rows[0].keys()) if sep_rows else ["case_name"])
        writer.writeheader()
        writer.writerows(sep_rows)
    _write_plots(aggregate / "plots", rows)


def _aggregate_existing_cases(run_root: Path) -> list[dict[str, Any]]:
    plan_rows = _read_json(run_root / "comparison_plan.json")
    specs = {
        str(row.get("case_name")): dict(row)
        for row in plan_rows
        if isinstance(row, Mapping) and row.get("case_name")
    } if isinstance(plan_rows, list) else {}
    rows: list[dict[str, Any]] = []
    cases_root = run_root / "cases"
    for case_dir in sorted(cases_root.glob("*")):
        if not case_dir.is_dir():
            continue
        case_name = case_dir.name
        row = dict(specs.get(case_name, {"case_name": case_name}))
        trace_csv = case_dir / "trace" / "frame_truth.csv"
        recovered_csv = _deterministic_latest(tuple((case_dir / "inference").glob("*_recovered_trace.csv")))
        inference_manifest = case_dir / "inference" / "manifest.json"
        summary_json = case_dir / "study" / "schur_summary" / "subblock_summary.json"
        if trace_csv.exists():
            row["truth_trace_csv"] = str(trace_csv.resolve())
        if recovered_csv is not None and recovered_csv.exists():
            row["recovered_trace_csv"] = str(recovered_csv.resolve())
            if trace_csv.exists():
                row.update(_read_recovered_errors(trace_csv, recovered_csv))
        manifest = _read_json(inference_manifest)
        metrics = manifest.get("metrics", {}) if isinstance(manifest, Mapping) else {}
        row["final_loss"] = metrics.get("final_loss")
        row["initial_temporal_term"] = metrics.get("initial_temporal_term")
        row["final_temporal_term"] = metrics.get("final_temporal_term")
        chi2 = metrics.get("chi2", {}) if isinstance(metrics.get("chi2"), Mapping) else {}
        row["initial_block_reduced_chi2"] = (chi2.get("initial_model") or {}).get("block_reduced_chi2") if isinstance(chi2.get("initial_model"), Mapping) else None
        row["final_block_reduced_chi2"] = (chi2.get("final_model") or {}).get("block_reduced_chi2") if isinstance(chi2.get("final_model"), Mapping) else None
        if summary_json.exists():
            row["summary_json"] = str(summary_json.resolve())
            row.update(_schur_metrics(summary_json, THETA_KEYS_DEFAULT))
        row["status"] = "completed" if recovered_csv is not None and recovered_csv.exists() else "incomplete"
        rows.append(row)
    return rows


def _write_plots(plot_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row.get("case_name")) for row in rows]
    specs = [
        ("separation_sigma_by_case.png", "separation_sigma_single_subblock_uas", "single subblock sigma [uas]"),
        ("forecast_sigma_1800_by_case.png", "forecast_sigma_1800_subblocks_uas", "1800-subblock forecast sigma [uas]"),
        ("fit_quality_by_case.png", "final_loss", "final loss"),
        ("registration_rms_error_by_case.png", "rms_x_error_as", "RMS x error [as]"),
    ]
    for filename, key, ylabel in specs:
        values = [float(row[key]) if row.get(key) not in (None, "") else np.nan for row in rows]
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(labels)), 4))
        ax.bar(np.arange(len(labels)), values)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--results-root", type=Path, default=Path("Results/temporal_registration_comparison"))
    parser.add_argument("--run-name", default="temporal_registration_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-preset", default="SHERA_FLIGHT_3P")
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--exposure-time-s", type=float, default=0.05)
    parser.add_argument("--noise", choices=("disabled", "enabled"), default="disabled")
    parser.add_argument("--init-mode", choices=("truth", "truth_plus_offset"), default="truth")
    parser.add_argument("--case-filter", action="append", default=[])
    parser.add_argument("--full-default-matrix", action="store_true")
    parser.add_argument("--residual-prior-sweep", action="store_true")
    parser.add_argument("--max-dense-dim", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--reference-diagnostics-profile", default="basic")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    exp_cfg = cfg.get("experiment", {}) if isinstance(cfg.get("experiment"), dict) else {}
    run_name = str(exp_cfg.get("run_name", args.run_name))
    run_root = args.results_root / run_name
    cases = default_case_specs(
        seed=int(exp_cfg.get("seed", args.seed)),
        noise_mode=args.noise,
        init_mode=args.init_mode,
        full=bool(args.full_default_matrix),
        residual_prior_sweep=bool(args.residual_prior_sweep),
    )
    if args.case_filter:
        allowed = set(args.case_filter)
        cases = [case for case in cases if case.case_name in allowed]
    names = [case.case_name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("Comparison case names must be unique.")
    manifest = {
        "schema_version": "temporal_registration_comparison_manifest.v1",
        "run_name": run_name,
        "seed": int(exp_cfg.get("seed", args.seed)),
        "n_frames": int(args.n_frames),
        "exposure_time_s": float(args.exposure_time_s),
        "subblock_duration_s": float(args.n_frames * args.exposure_time_s),
        "case_count": len(cases),
        "generator": "examples/scripts/run_temporal_registration_comparison.py",
    }
    if args.aggregate_only:
        rows = _aggregate_existing_cases(run_root)
        _write_aggregate(run_root, rows)
        return {"run_root": str(run_root), "aggregate_only": True, "case_count": len(rows)}
    _write_plan(run_root, cases, manifest)
    rows = [
        _run_case(
            case=case,
            run_root=run_root,
            n_frames=int(args.n_frames),
            exposure_time_s=float(args.exposure_time_s),
            system_preset=str(args.system_preset),
            dry_run=bool(args.dry_run),
            resume=bool(args.resume),
            theta_keys=THETA_KEYS_DEFAULT,
            reference_diagnostics_profile=str(args.reference_diagnostics_profile),
            max_dense_dim=int(args.max_dense_dim),
        )
        for case in cases
    ]
    _write_aggregate(run_root, rows)
    return {"run_root": str(run_root), "case_count": len(cases), "dry_run": bool(args.dry_run)}


if __name__ == "__main__":
    main()
