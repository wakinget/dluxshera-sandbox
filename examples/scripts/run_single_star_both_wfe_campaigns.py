#!/usr/bin/env python3
"""Plan and run simultaneous M1/M2 single-star WFE campaigns.

The default validation path is plan-only: build configs, manifests, command
files, validation reports, and Slurm array scripts without running image-backed
child solves. Real execution is opt-in through ``--execute-shard``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dluxshera-matplotlib"),
)

import jax
import numpy as np

from dluxshera.config.io import load_system_preset
from dluxshera.inference.observation_belief import build_system_observation_theta_layout
from dluxshera.inference.observation_forecast import build_prior_mean_from_store
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_io import now_iso_local_ms, timestamp_tag
from dluxshera.utils.single_star_calibration import (
    ALPHA_CEN_A_PLACEHOLDER_NOTE,
    prepare_alpha_cen_a_single_star_system_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CAL_DEMO_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_single_star_calibration_demo.py"
DEFAULT_RESULTS_ROOT = Path(f"/scratch/shera_hpc/{os.environ.get('USER', 'unknown')}/dluxshera")
DEFAULT_SYSTEM_PRESET = "SHERA_FLIGHT_3P"

CAMPAIGN_A = "single_star_both_wfe_knowledge_capture_v2"
CAMPAIGN_B = "single_star_both_wfe_iterative_update_v1"
CAMPAIGNS = (CAMPAIGN_A, CAMPAIGN_B)

ACTIVE_FRAME_KEYS = ("source.x_position_as", "source.y_position_as")
NOLL_RANGE = tuple(range(4, 12))
WFE_PAIRINGS = ("independent", "matched", "differential")
SCALAR_TOLERANCES = {
    "source.log_flux_total": 1.0e-10,
    "optics.plate_scale_as_per_pix": 1.0e-12,
}


@dataclass(frozen=True)
class CaseSpec:
    row_index: int
    case_name: str
    condition_name: str
    condition_kind: str
    draw_index: int
    amplitude_value: float
    offsets: dict[str, float]
    baseline_offsets: dict[str, float]
    wfe_pairing: str


@dataclass(frozen=True)
class PlanContext:
    labels: tuple[str, ...]
    truth_by_label: dict[str, float]
    system_cfg: dict[str, Any]
    x64_enabled: bool


@contextmanager
def _jax_x64_context():
    previous = bool(jax.config.jax_enable_x64)
    if not previous:
        jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        if not previous:
            jax.config.update("jax_enable_x64", False)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _hash_seed(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def _rng_for(*parts: object) -> np.random.Generator:
    return np.random.default_rng(_hash_seed(*parts))


def _safe_fraction(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or not math.isfinite(numerator) or not math.isfinite(denominator):
        return float("nan")
    return float(numerator / denominator)


def _projection_gain(update: np.ndarray, ideal: np.ndarray) -> float:
    denom = float(np.dot(ideal, ideal))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(update, ideal) / denom)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _system_context(*, system_preset: str, exposure_time_s: float, n_lambda: int | None) -> PlanContext:
    with _jax_x64_context():
        system_seed = load_system_preset(system_preset)["system"]
        system_cfg = prepare_alpha_cen_a_single_star_system_config(
            system_seed,
            exposure_time_s=exposure_time_s,
            n_lambda=n_lambda,
        )
        spec = compose_forward_spec(system_cfg)
        store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
        layout_cfg = {
            "source": {
                "log_flux_total": True,
                "separation_as": False,
                "contrast": False,
                "position_angle_deg": False,
            },
            "optics": {
                "plate_scale_as_per_pix": True,
                "primary_zernikes": {"enabled": True, "indices": "from_system", "include": None, "exclude": []},
                "secondary_zernikes": {"enabled": True, "indices": "from_system", "include": None, "exclude": []},
            },
        }
        layout, _metadata = build_system_observation_theta_layout(store, config=layout_cfg)
        truth = np.asarray(build_prior_mean_from_store(layout.labels, store=store), dtype=float)
        truth_by_label = {label: float(truth[i]) for i, label in enumerate(layout.labels)}
        return PlanContext(
            labels=tuple(layout.labels),
            truth_by_label=truth_by_label,
            system_cfg=system_cfg,
            x64_enabled=bool(jax.config.jax_enable_x64),
        )


def _condition_grids(campaign: str) -> tuple[list[float], list[float], list[float]]:
    if campaign == CAMPAIGN_A:
        return (
            [0.1, 0.3, 1.0],
            [0.3, 1.0, 3.0, 10.0],
            [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00],
        )
    if campaign == CAMPAIGN_B:
        return ([], [], [0.10, 0.30, 0.50, 0.75, 1.00])
    raise ValueError(f"Unsupported campaign: {campaign}")


def _fmt_percent(value: float) -> str:
    return {0.1: "0p1", 0.3: "0p3", 1.0: "1p0"}.get(float(value), f"{value:.1f}".replace(".", "p"))


def _fmt_ppm(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}".replace(".", "p")


def _fmt_nm(value: float) -> str:
    mapping = {
        0.03: "0p03",
        0.05: "0p05",
        0.075: "0p075",
        0.10: "0p10",
        0.15: "0p15",
        0.20: "0p20",
        0.30: "0p30",
        0.50: "0p50",
        0.75: "0p75",
        1.00: "1p00",
    }
    for key, token in mapping.items():
        if math.isclose(value, key, rel_tol=0.0, abs_tol=1.0e-12):
            return token
    return f"{value:.2f}".replace(".", "p")


def _condition_name(kind: str, amplitude: float) -> str:
    if kind == "logflux":
        return f"logflux_{_fmt_percent(amplitude)}pct"
    if kind == "platescale":
        return f"platescale_{_fmt_ppm(amplitude)}ppm"
    if kind == "both_wfe":
        return f"both_wfe_{_fmt_nm(amplitude)}nm"
    return kind


def _baseline_scalar_offsets(truth_by_label: Mapping[str, float]) -> dict[str, float]:
    return {
        "source.log_flux_total": float(math.log10(1.0 + 0.003)),
        "optics.plate_scale_as_per_pix": float(truth_by_label["optics.plate_scale_as_per_pix"]) * 1.0e-6,
    }


def _scalar_offset(label: str, amplitude: float, truth_by_label: Mapping[str, float], *, condition: str, draw_index: int) -> float:
    rng = _rng_for("scalar", condition, label, draw_index)
    sign = 1.0 if rng.random() >= 0.5 else -1.0
    if label == "source.log_flux_total":
        return float(math.log10(1.0 + sign * amplitude / 100.0))
    if label == "optics.plate_scale_as_per_pix":
        return float(truth_by_label[label]) * sign * amplitude * 1.0e-6
    raise ValueError(f"Unsupported scalar label: {label}")


def _unit_rms_vector(*, condition: str, draw_index: int, mirror: str) -> np.ndarray:
    rng = _rng_for("wfe", condition, draw_index, mirror)
    vec = rng.normal(size=len(NOLL_RANGE))
    rms = float(np.sqrt(np.mean(np.square(vec))))
    if rms == 0.0:
        vec = np.ones(len(NOLL_RANGE), dtype=float)
        rms = 1.0
    return vec / rms


def _wfe_vectors(*, amplitude_nm: float, condition: str, draw_index: int, wfe_pairing: str) -> tuple[np.ndarray, np.ndarray]:
    if wfe_pairing not in WFE_PAIRINGS:
        raise ValueError(f"wfe_pairing must be one of {', '.join(WFE_PAIRINGS)}.")
    primary = _unit_rms_vector(condition=condition, draw_index=draw_index, mirror="primary")
    if wfe_pairing == "independent":
        secondary = _unit_rms_vector(condition=condition, draw_index=draw_index, mirror="secondary")
    elif wfe_pairing == "matched":
        secondary = primary.copy()
    else:
        secondary = -primary
    return float(amplitude_nm) * primary, float(amplitude_nm) * secondary


def _case_offsets(
    *,
    condition_kind: str,
    amplitude_value: float,
    draw_index: int,
    truth_by_label: Mapping[str, float],
    wfe_pairing: str,
) -> tuple[dict[str, float], dict[str, float]]:
    condition = _condition_name(condition_kind, amplitude_value)
    offsets: dict[str, float] = {}
    baseline: dict[str, float] = {}
    if condition_kind == "logflux":
        offsets["source.log_flux_total"] = _scalar_offset(
            "source.log_flux_total", amplitude_value, truth_by_label, condition=condition, draw_index=draw_index
        )
    elif condition_kind == "platescale":
        offsets["optics.plate_scale_as_per_pix"] = _scalar_offset(
            "optics.plate_scale_as_per_pix", amplitude_value, truth_by_label, condition=condition, draw_index=draw_index
        )
    elif condition_kind == "both_wfe":
        primary, secondary = _wfe_vectors(
            amplitude_nm=amplitude_value,
            condition=condition,
            draw_index=draw_index,
            wfe_pairing=wfe_pairing,
        )
        for i, _noll in enumerate(NOLL_RANGE):
            offsets[f"optics.primary.zernike_coeffs_nm[{i}]"] = float(primary[i])
            offsets[f"optics.secondary.zernike_coeffs_nm[{i}]"] = float(secondary[i])
        baseline = _baseline_scalar_offsets(truth_by_label)
        offsets.update(baseline)
    else:
        raise ValueError(f"Unsupported condition kind: {condition_kind}")
    return offsets, baseline


def _build_cases(
    *,
    campaign: str,
    truth_by_label: Mapping[str, float],
    wfe_pairing: str,
    n_draws: int,
    include_zero_bias_control: bool = True,
) -> list[CaseSpec]:
    logflux_grid, platescale_grid, wfe_grid = _condition_grids(campaign)
    cases: list[CaseSpec] = []
    row_index = 0
    if include_zero_bias_control:
        cases.append(
            CaseSpec(
                row_index=row_index,
                case_name="zero_bias_control",
                condition_name="zero_bias_control",
                condition_kind="control",
                draw_index=0,
                amplitude_value=0.0,
                offsets={},
                baseline_offsets={},
                wfe_pairing=wfe_pairing,
            )
        )
        row_index += 1

    for condition_kind, grid in (
        ("logflux", logflux_grid),
        ("platescale", platescale_grid),
        ("both_wfe", wfe_grid),
    ):
        for amplitude in grid:
            condition = _condition_name(condition_kind, amplitude)
            for draw_index in range(int(n_draws)):
                offsets, baseline = _case_offsets(
                    condition_kind=condition_kind,
                    amplitude_value=amplitude,
                    draw_index=draw_index,
                    truth_by_label=truth_by_label,
                    wfe_pairing=wfe_pairing,
                )
                cases.append(
                    CaseSpec(
                        row_index=row_index,
                        case_name=f"{condition}_draw_{draw_index:03d}",
                        condition_name=condition,
                        condition_kind=condition_kind,
                        draw_index=draw_index,
                        amplitude_value=float(amplitude),
                        offsets=offsets,
                        baseline_offsets=baseline,
                        wfe_pairing=wfe_pairing,
                    )
                )
                row_index += 1
    return cases


def _case_rows(cases: Sequence[CaseSpec], *, campaign: str, n_subblocks: int, n_windows: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        if campaign == CAMPAIGN_B:
            for window in range(n_windows):
                rows.append(_case_row(case, n_subblocks=n_subblocks, window_index=window))
        else:
            rows.append(_case_row(case, n_subblocks=n_subblocks, window_index=None))
    return rows


def _case_row(case: CaseSpec, *, n_subblocks: int, window_index: int | None) -> dict[str, Any]:
    m1 = np.asarray([case.offsets.get(f"optics.primary.zernike_coeffs_nm[{i}]", 0.0) for i in range(len(NOLL_RANGE))])
    m2 = np.asarray([case.offsets.get(f"optics.secondary.zernike_coeffs_nm[{i}]", 0.0) for i in range(len(NOLL_RANGE))])
    matched = (m1 + m2) / math.sqrt(2.0)
    differential = (m1 - m2) / math.sqrt(2.0)
    return {
        "row_index": int(case.row_index if window_index is None else case.row_index * 100 + int(window_index)),
        "case_group_id": int(case.row_index),
        "case_name": case.case_name,
        "window_index": "" if window_index is None else int(window_index),
        "condition_name": case.condition_name,
        "condition_kind": case.condition_kind,
        "draw_index": int(case.draw_index),
        "amplitude_value": float(case.amplitude_value),
        "wfe_pairing": case.wfe_pairing,
        "n_subblocks": int(n_subblocks),
        "matched_component_norm_nm": float(np.linalg.norm(matched)),
        "differential_component_norm_nm": float(np.linalg.norm(differential)),
        "m1_rms_bias_nm": float(np.sqrt(np.mean(np.square(m1)))),
        "m2_rms_bias_nm": float(np.sqrt(np.mean(np.square(m2)))),
        "baseline_logflux_offset": case.baseline_offsets.get("source.log_flux_total", ""),
        "baseline_platescale_offset": case.baseline_offsets.get("optics.plate_scale_as_per_pix", ""),
    }


def _shard_rows(case_rows: Sequence[Mapping[str, Any]], *, num_shards: int, campaign: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in case_rows:
        shard_key = int(row["case_group_id"]) if campaign == CAMPAIGN_B else int(row["row_index"])
        out.append({**dict(row), "shard_index": shard_key % int(num_shards), "num_shards": int(num_shards)})
    return out


def _child_run_name(*, campaign: str, run_name: str, shard_index: int, window_index: int | None) -> str:
    if campaign == CAMPAIGN_B and window_index is not None:
        return f"{run_name}_window_{int(window_index):02d}_shard_{int(shard_index):04d}"
    return f"{run_name}_shard_{int(shard_index):04d}"


def _attach_output_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_root: Path,
    campaign: str,
    run_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        shard_index = int(row["shard_index"])
        raw_window = row.get("window_index", "")
        window_index = None if raw_window == "" else int(raw_window)
        child_results_root = run_root / "shards"
        child_run_name = _child_run_name(
            campaign=campaign,
            run_name=run_name,
            shard_index=shard_index,
            window_index=window_index,
        )
        child_run_root = child_results_root / child_run_name
        case_root = child_run_root / "cases" / str(row["case_name"])
        out.append(
            {
                **dict(row),
                "child_results_root": str(child_results_root),
                "child_run_name": child_run_name,
                "child_run_root": str(child_run_root),
                "case_root": str(case_root),
                "posterior_by_parameter_csv": str(case_root / "posterior_by_parameter.csv"),
                "posterior_history_csv": str(case_root / "posterior_history.csv"),
                "case_summary_json": str(case_root / "case_summary.json"),
                "campaign_summary_json": str(child_run_root / "campaign_summary.json"),
            }
        )
    return out


def _runner_config(
    *,
    args: argparse.Namespace,
    cases: Sequence[CaseSpec],
    run_name: str,
    n_subblocks: int,
    update_gain: float | None = None,
    window_index: int | None = None,
) -> dict[str, Any]:
    source_cfg = {
        "mode": "alpha_cen_a_placeholder",
        "source_kind": "single_star",
        "x_position_as": 0.0,
        "y_position_as": 0.0,
        "position_angle_deg": 0.0,
        "n_lambda": None if args.n_lambda is None else int(args.n_lambda),
        "photometry_note": ALPHA_CEN_A_PLACEHOLDER_NOTE,
    }
    experiment: dict[str, Any] = {
        "kind": "single_star_calibration_demo",
        "run_name": run_name,
        "calibration_source": source_cfg,
        "subblocks": {
            "n_subblocks": int(n_subblocks),
            "n_frames": int(args.n_frames),
            "noise": "enabled",
            "phi_ref": "recovered",
            "schur_curvature_method": "structured_independent_frames",
            "max_dense_dim": 40,
            "schur_damping": 1.0e-8,
            "summary_information_scale": "summed_likelihood",
            "use_render_variance": "auto",
            "exposure_time_s": float(args.exposure_time_s),
            "reference_diagnostics_profile": "none",
        },
        "seeding": {"seed_policy": "different_jitter_different_noise", "base_seed": 42},
        "local_eliminated_keys": list(ACTIVE_FRAME_KEYS),
        "observation_theta": {
            "source": {"log_flux_total": True, "separation_as": False, "contrast": False},
            "optics": {
                "plate_scale_as_per_pix": True,
                "primary_zernikes": {"enabled": True, "indices": "from_system", "include": None, "exclude": []},
                "secondary_zernikes": {"enabled": True, "indices": "from_system", "include": None, "exclude": []},
            },
        },
        "prior": {
            "sigma": {
                "source.log_flux_total": {"kind": "absolute", "sigma": 1.0e-5},
                "optics.plate_scale_as_per_pix": {"kind": "fractional", "sigma": 1.0e-5},
                "optics.primary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 1.0, "unit": "nm"},
                "optics.secondary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 1.0, "unit": "nm"},
            }
        },
        "case_generation": {
            "mode": "explicit",
            "cases": [{"case_name": case.case_name, "theta_reference_offsets": dict(case.offsets)} for case in cases],
        },
        "forecast": {"enabled": False, "subblock_duration_s": 1.0},
        "eigenbasis": {
            "enabled": True,
            "sources": ["accumulated_information", "posterior_precision"],
            "whiten": True,
            "eig_floor_abs": 0.0,
            "eig_floor_rel": 1.0e-12,
            "top_k_contributors": 8,
        },
        "wrapper_metadata": {
            "campaign": args.campaign,
            "system_preset": args.system_preset,
            "exposure_time_s": float(args.exposure_time_s),
            "n_lambda": None if args.n_lambda is None else int(args.n_lambda),
            "wfe_pairing": args.wfe_pairing,
            "update_gain": update_gain,
            "window_index": window_index,
            "reference_update_policy": (
                "Campaign B damps the next-window reference/linearization state. "
                "The next-window prior mean follows that damped reference in this wrapper."
            ),
        },
    }
    if update_gain is not None:
        experiment["update_gain"] = float(update_gain)
    return {"system": {"preset": args.system_preset}, "experiment": experiment}


def _child_context_from_runner_config(config: Mapping[str, Any]) -> PlanContext:
    experiment = config["experiment"]
    source_cfg = experiment["calibration_source"]
    subblock_cfg = experiment["subblocks"]
    n_lambda = source_cfg.get("n_lambda")
    return _system_context(
        system_preset=str(config["system"]["preset"]),
        exposure_time_s=float(subblock_cfg["exposure_time_s"]),
        n_lambda=None if n_lambda is None or int(n_lambda) <= 0 else int(n_lambda),
    )


def _scalar_consistency_rows(
    parent_context: PlanContext,
    *,
    child_config: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    child_context = _child_context_from_runner_config(child_config)
    child_cases = child_config["experiment"]["case_generation"]["cases"]
    zero_case = next((case for case in child_cases if case["case_name"] == "zero_bias_control"), child_cases[0])
    zero_offsets = zero_case.get("theta_reference_offsets", {})
    rows: list[dict[str, Any]] = []
    for label, tolerance in SCALAR_TOLERANCES.items():
        parent_truth = float(parent_context.truth_by_label[label])
        parent_reference = parent_truth
        child_truth = float(child_context.truth_by_label[label])
        child_offset = float(zero_offsets.get(label, 0.0))
        child_reference = child_truth + child_offset
        parent_child_truth_mismatch = abs(child_truth - parent_truth)
        child_zero_reference_mismatch = abs(child_reference - child_truth)
        mismatch = max(parent_child_truth_mismatch, child_zero_reference_mismatch)
        rows.append(
            {
                "theta_label": label,
                "parent_truth_value": parent_truth,
                "parent_reference_value": parent_reference,
                "child_truth_value": child_truth,
                "child_reference_value": child_reference,
                "child_zero_bias_offset": child_offset,
                "parent_child_truth_abs_mismatch": parent_child_truth_mismatch,
                "child_zero_reference_abs_mismatch": child_zero_reference_mismatch,
                "abs_mismatch": mismatch,
                "tolerance": float(tolerance),
                "passed": bool(mismatch <= tolerance),
                "parent_x64_enabled": bool(parent_context.x64_enabled),
                "child_x64_enabled": bool(child_context.x64_enabled),
                "system_preset": args.system_preset,
                "exposure_time_s": float(args.exposure_time_s),
                "n_lambda": None if int(args.n_lambda) <= 0 else int(args.n_lambda),
                "calibration_source_mode": child_config["experiment"]["calibration_source"]["mode"],
                "use_render_variance": child_config["experiment"]["subblocks"]["use_render_variance"],
            }
        )
    return rows


def _expected_output_rows(planned_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "row_index",
        "shard_index",
        "num_shards",
        "case_name",
        "condition_name",
        "condition_kind",
        "draw_index",
        "window_index",
        "child_results_root",
        "child_run_name",
        "child_run_root",
        "case_root",
        "posterior_by_parameter_csv",
        "posterior_history_csv",
        "case_summary_json",
        "campaign_summary_json",
    )
    return [{field: row.get(field, "") for field in fields} for row in planned_rows]


def _validate_output_contract(planned_rows: Sequence[Mapping[str, Any]], expected_rows: Sequence[Mapping[str, Any]]) -> None:
    by_row = {str(row["row_index"]): row for row in planned_rows}
    for expected in expected_rows:
        planned = by_row.get(str(expected["row_index"]))
        if planned is None:
            raise RuntimeError(f"Expected output row has no matching planned row: {expected['row_index']}")
        for field in (
            "child_results_root",
            "child_run_name",
            "child_run_root",
            "case_root",
            "posterior_by_parameter_csv",
            "posterior_history_csv",
            "case_summary_json",
            "campaign_summary_json",
        ):
            if str(planned.get(field, "")) != str(expected.get(field, "")):
                raise RuntimeError(
                    f"Output contract mismatch for row {expected['row_index']} field {field}: "
                    f"planned={planned.get(field, '')!r} expected={expected.get(field, '')!r}"
                )


def _shape_args(args: argparse.Namespace, *, include_shard_index: bool) -> list[str]:
    pairs: list[tuple[str, Any]] = [
        ("--campaign", args.campaign),
        ("--results-root", "$DLUX_RESULTS" if getattr(args, "_sbatch_results_root", False) else str(Path(args.results_root).resolve())),
        ("--run-name", args.run_name),
        ("--n-draws", int(args.n_draws)),
        ("--n-subblocks", int(args.n_subblocks)),
        ("--n-frames", int(args.n_frames)),
        ("--windows-per-draw", int(args.windows_per_draw)),
        ("--num-shards", int(args.num_shards)),
        ("--max-workers", int(args.max_workers)),
        ("--cpus-per-task", int(args.cpus_per_task)),
        ("--array-throttle", int(args.array_throttle)),
        ("--wfe-pairing", args.wfe_pairing),
        ("--update-gain", float(args.update_gain)),
        ("--system-preset", args.system_preset),
        ("--exposure-time-s", float(args.exposure_time_s)),
        ("--n-lambda", int(args.n_lambda) if args.n_lambda is not None else 0),
        ("--slurm-mem", args.slurm_mem),
        ("--slurm-time", args.slurm_time),
        ("--slurm-partition", args.slurm_partition),
        ("--slurm-account", args.slurm_account),
    ]
    if args.slurm_job_name:
        pairs.append(("--slurm-job-name", args.slurm_job_name))
    out: list[str] = []
    for flag, value in pairs:
        out.extend([flag, str(value)])
    if include_shard_index:
        out.extend(
            [
                "--shard-index",
                '"$SLURM_ARRAY_TASK_ID"' if getattr(args, "_sbatch_results_root", False) else str(args.shard_index),
            ]
        )
    if not bool(args.resource_time):
        out.append("--no-resource-time")
    return out


def _validate_shape_command(command_text: str, args: argparse.Namespace, *, expect_shard_index: bool) -> None:
    required = {
        "--campaign": str(args.campaign),
        "--run-name": str(args.run_name),
        "--n-draws": str(int(args.n_draws)),
        "--n-subblocks": str(int(args.n_subblocks)),
        "--n-frames": str(int(args.n_frames)),
        "--windows-per-draw": str(int(args.windows_per_draw)),
        "--num-shards": str(int(args.num_shards)),
        "--max-workers": str(int(args.max_workers)),
        "--cpus-per-task": str(int(args.cpus_per_task)),
        "--array-throttle": str(int(args.array_throttle)),
        "--wfe-pairing": str(args.wfe_pairing),
        "--update-gain": str(float(args.update_gain)),
        "--system-preset": str(args.system_preset),
        "--exposure-time-s": str(float(args.exposure_time_s)),
        "--n-lambda": str(int(args.n_lambda) if args.n_lambda is not None else 0),
        "--slurm-mem": str(args.slurm_mem),
        "--slurm-time": str(args.slurm_time),
        "--slurm-partition": str(args.slurm_partition),
        "--slurm-account": str(args.slurm_account),
    }
    if args.slurm_job_name:
        required["--slurm-job-name"] = str(args.slurm_job_name)
    if expect_shard_index:
        required["--shard-index"] = ""
    for flag, value in required.items():
        if flag not in command_text:
            raise RuntimeError(f"Generated command is missing shape flag {flag}.")
        if value and value not in command_text:
            raise RuntimeError(f"Generated command is missing shape value {value!r} for {flag}.")


def _format_command(prefix: Sequence[str], args_list: Sequence[str], *, line_continuation: bool) -> str:
    tokens = [*prefix, *args_list]
    if not line_continuation:
        return " ".join(tokens)
    lines = [tokens[0]]
    i = 1
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--") and i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            rendered = f"{token} {tokens[i + 1]}"
            i += 2
        else:
            rendered = token
            i += 1
        lines[-1] += " \\"
        lines.append(f"  {rendered}")
    return "\n".join(lines)


def _write_child_commands(
    *,
    run_root: Path,
    args: argparse.Namespace,
    shard_rows: Sequence[Mapping[str, Any]],
) -> None:
    command_root = run_root / "commands"
    _ensure_dir(command_root)
    for shard in sorted({int(row["shard_index"]) for row in shard_rows}):
        shard_args = argparse.Namespace(**vars(args))
        shard_args.shard_index = shard
        command = [
            "PYTHONPATH=src",
            sys.executable,
            "examples/scripts/run_single_star_both_wfe_campaigns.py",
            "--execute-shard",
            *_shape_args(shard_args, include_shard_index=True),
        ]
        (command_root / f"shard_{shard:04d}.sh").write_text(" ".join(command) + "\n", encoding="utf-8")


def _write_sbatch_scripts(run_root: Path, *, args: argparse.Namespace) -> None:
    slurm_root = run_root / "sbatch"
    _ensure_dir(slurm_root)
    output = "/scratch/shera_hpc/%u/dluxshera/slurm_logs/%x-%A_%a.out"
    error = "/scratch/shera_hpc/%u/dluxshera/slurm_logs/%x-%A_%a.err"
    def header(*, include_array: bool) -> str:
        array_line = f"#SBATCH --array=0-{int(args.num_shards) - 1}%{int(args.array_throttle)}\n" if include_array else ""
        job_name = args.slurm_job_name or args.campaign
        return f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --partition={args.slurm_partition}
#SBATCH --account={args.slurm_account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={int(args.cpus_per_task)}
#SBATCH --mem={args.slurm_mem}
#SBATCH --time={args.slurm_time}
{array_line}#SBATCH --output={output}
#SBATCH --error={error}
"""

    body = """
set -euo pipefail

source /cm/shared/apps/miniforge/etc/profile.d/conda.sh
conda activate dluxshera-py311
cd ~/dluxshera-sandbox

export DLUX_RESULTS=/scratch/shera_hpc/$USER/dluxshera
mkdir -p "$DLUX_RESULTS/slurm_logs"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

export JAX_COMPILATION_CACHE_DIR=/scratch/shera_hpc/$USER/jax_cache
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
"""
    sbatch_args = argparse.Namespace(**vars(args))
    sbatch_args._sbatch_results_root = True
    execute_tokens = ["PYTHONPATH=src python examples/scripts/run_single_star_both_wfe_campaigns.py", "--execute-shard"]
    aggregate_tokens = ["PYTHONPATH=src python examples/scripts/run_single_star_both_wfe_campaigns.py", "--aggregate-only"]
    array_script = (
        header(include_array=True)
        + body
        + "\n"
        + _format_command(execute_tokens, _shape_args(sbatch_args, include_shard_index=True), line_continuation=True)
        + "\n"
    )
    aggregate_script = (
        header(include_array=False)
        + body
        + "\n"
        + _format_command(aggregate_tokens, _shape_args(sbatch_args, include_shard_index=False), line_continuation=True)
        + "\n"
    )
    (slurm_root / f"{args.campaign}.sbatch").write_text(array_script, encoding="utf-8")
    (slurm_root / f"{args.campaign}_aggregate.sbatch").write_text(aggregate_script, encoding="utf-8")


def _write_plan(args: argparse.Namespace) -> dict[str, Any]:
    context = _system_context(
        system_preset=args.system_preset,
        exposure_time_s=float(args.exposure_time_s),
        n_lambda=None if int(args.n_lambda) <= 0 else int(args.n_lambda),
    )
    n_draws = int(args.n_draws)
    n_subblocks = int(args.n_subblocks)
    n_windows = int(args.windows_per_draw) if args.campaign == CAMPAIGN_B else 0
    cases = _build_cases(
        campaign=args.campaign,
        truth_by_label=context.truth_by_label,
        wfe_pairing=args.wfe_pairing,
        n_draws=n_draws,
        include_zero_bias_control=True,
    )
    run_root = Path(args.results_root).resolve() / str(args.run_name)
    _ensure_dir(run_root)

    case_rows = _case_rows(cases, campaign=args.campaign, n_subblocks=n_subblocks, n_windows=n_windows)
    shard_plan = _attach_output_contract(
        _shard_rows(case_rows, num_shards=int(args.num_shards), campaign=args.campaign),
        run_root=run_root,
        campaign=args.campaign,
        run_name=args.run_name,
    )
    zero_child_config = _runner_config(
        args=args,
        cases=[case for case in cases if case.case_name == "zero_bias_control"],
        run_name=_child_run_name(campaign=args.campaign, run_name=args.run_name, shard_index=0, window_index=None),
        n_subblocks=n_subblocks,
    )
    scalar_rows = _scalar_consistency_rows(context, child_config=zero_child_config, args=args)
    if not all(bool(row["passed"]) for row in scalar_rows):
        _write_csv(run_root / "scalar_consistency_check.csv", scalar_rows)
        raise RuntimeError("Zero-bias scalar consistency check failed; refusing to write production plan.")

    if args.campaign == CAMPAIGN_A:
        _write_json(run_root / "config.json", _runner_config(args=args, cases=cases, run_name=args.run_name, n_subblocks=n_subblocks))
    else:
        for window in range(n_windows):
            _write_json(
                run_root / f"window_{window:02d}" / "config.json",
                _runner_config(
                    args=args,
                    cases=cases,
                    run_name=f"{args.run_name}_window_{window:02d}",
                    n_subblocks=n_subblocks,
                    update_gain=float(args.update_gain),
                    window_index=window,
                ),
            )

    _write_csv(run_root / "campaign_case_plan.csv", shard_plan)
    _write_csv(run_root / "campaign_shard_plan.csv", shard_plan)
    expected_rows = _expected_output_rows(shard_plan)
    _validate_output_contract(shard_plan, expected_rows)
    _write_csv(run_root / "expected_outputs.csv", expected_rows)
    _write_csv(run_root / "scalar_consistency_check.csv", scalar_rows)
    _write_child_commands(run_root=run_root, args=args, shard_rows=shard_plan)
    _write_sbatch_scripts(run_root, args=args)
    first_command = next((run_root / "commands").glob("shard_*.sh")).read_text(encoding="utf-8")
    _validate_shape_command(first_command, args, expect_shard_index=True)
    _validate_shape_command((run_root / "sbatch" / f"{args.campaign}.sbatch").read_text(encoding="utf-8"), args, expect_shard_index=True)

    logflux_grid, platescale_grid, wfe_grid = _condition_grids(args.campaign)
    expected_total_subblock_solves = sum(int(row["n_subblocks"]) for row in case_rows)
    validation = {
        "schema_version": "single_star_both_wfe_campaign_plan.v2",
        "created_at": now_iso_local_ms(),
        "campaign": args.campaign,
        "run_name": args.run_name,
        "results_root": str(Path(args.results_root).resolve()),
        "run_root": str(run_root),
        "system_preset": args.system_preset,
        "exposure_time_s": float(args.exposure_time_s),
        "n_lambda": None if int(args.n_lambda) <= 0 else int(args.n_lambda),
        "n_conditions": len(logflux_grid) + len(platescale_grid) + len(wfe_grid),
        "n_prior_draws": n_draws,
        "n_subblocks": n_subblocks,
        "n_windows": n_windows,
        "num_shards": int(args.num_shards),
        "array_throttle": int(args.array_throttle),
        "max_workers": int(args.max_workers),
        "slurm": {
            "mem": args.slurm_mem,
            "time": args.slurm_time,
            "partition": args.slurm_partition,
            "account": args.slurm_account,
            "job_name": args.slurm_job_name or args.campaign,
        },
        "expected_child_commands": int(len({int(row["shard_index"]) for row in shard_plan})),
        "expected_total_subblock_solves": int(expected_total_subblock_solves),
        "scalar_baseline_settings": {"log_flux_percent": 0.3, "plate_scale_ppm": 1.0},
        "wfe_pairing": args.wfe_pairing,
        "wfe_amplitude_grid_nm": wfe_grid,
        "log_flux_amplitude_grid_percent": logflux_grid,
        "plate_scale_amplitude_grid_ppm": platescale_grid,
        "summary_information_scale": "summed_likelihood",
        "forecast_enabled": False,
        "eigenbasis_diagnostics_enabled": True,
        "x64_enabled_during_parent_plan": bool(context.x64_enabled),
        "zero_bias_scalar_consistency_passed": all(bool(row["passed"]) for row in scalar_rows),
        "output_contract_consistency_passed": True,
        "command_shape_propagation_passed": True,
        "path_contract": {
            "child_results_root": str(run_root / "shards"),
            "child_run_name_pattern": (
                f"{args.run_name}_shard_<shard_index>"
                if args.campaign == CAMPAIGN_A
                else f"{args.run_name}_window_<window_index>_shard_<shard_index>"
            ),
            "case_root_pattern": "<child_run_root>/cases/<case_name>",
        },
        "campaign_b_production_executable": True,
        "campaign_b_execution_guard": "",
        "plan_only": bool(args.plan_only),
    }
    _write_json(run_root / "campaign_plan_validation.json", validation)
    _write_json(run_root / "resolved_config.json", {"experiment": validation, "system": context.system_cfg})
    return validation


def _selected_cases_for_shard(args: argparse.Namespace, context: PlanContext) -> list[CaseSpec]:
    cases = _build_cases(
        campaign=args.campaign,
        truth_by_label=context.truth_by_label,
        wfe_pairing=args.wfe_pairing,
        n_draws=int(args.n_draws),
        include_zero_bias_control=True,
    )
    selected: list[CaseSpec] = []
    for case in cases:
        if case.row_index % int(args.num_shards) == int(args.shard_index):
            selected.append(case)
    return selected


def _run_child_config(config_path: Path, results_root: Path, run_name: str, args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(CAL_DEMO_SCRIPT),
        "--config",
        str(config_path),
        "--results-root",
        str(results_root),
        "--run-name",
        run_name,
        "--max-workers",
        str(int(args.max_workers)),
        "--resume",
        "--quiet",
    ]
    if not bool(args.resource_time):
        command.append("--no-resource-time")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Child calibration runner failed with return code {completed.returncode}: {' '.join(command)}")


def _wfe_labels() -> tuple[str, ...]:
    labels: list[str] = []
    for i in range(len(NOLL_RANGE)):
        labels.append(f"optics.primary.zernike_coeffs_nm[{i}]")
        labels.append(f"optics.secondary.zernike_coeffs_nm[{i}]")
    return tuple(labels)


def _offset_vector(offsets: Mapping[str, float], labels: Sequence[str]) -> np.ndarray:
    return np.asarray([float(offsets.get(label, 0.0)) for label in labels], dtype=float)


def _posterior_by_label(path: Path) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    return {_posterior_label(row): row for row in rows if _posterior_label(row)}


def _next_offsets_from_posterior(
    *,
    current_offsets: Mapping[str, float],
    posterior_rows_by_label: Mapping[str, Mapping[str, Any]],
    truth_by_label: Mapping[str, float],
    update_gain: float,
) -> dict[str, float]:
    next_offsets = dict(current_offsets)
    for label, posterior_row in posterior_rows_by_label.items():
        if label not in truth_by_label:
            continue
        posterior_mean = _posterior_float(posterior_row, ("posterior_mean", "mean", "posterior", "value"))
        if not math.isfinite(posterior_mean):
            continue
        theta_ref_current = float(truth_by_label[label]) + float(current_offsets.get(label, 0.0))
        theta_ref_next = theta_ref_current + float(update_gain) * (posterior_mean - theta_ref_current)
        next_offsets[label] = float(theta_ref_next - float(truth_by_label[label]))
    return next_offsets


def _matched_differential_norms(offsets: Mapping[str, float]) -> tuple[float, float]:
    m1 = np.asarray([float(offsets.get(f"optics.primary.zernike_coeffs_nm[{i}]", 0.0)) for i in range(len(NOLL_RANGE))])
    m2 = np.asarray([float(offsets.get(f"optics.secondary.zernike_coeffs_nm[{i}]", 0.0)) for i in range(len(NOLL_RANGE))])
    matched = (m1 + m2) / math.sqrt(2.0)
    differential = (m1 - m2) / math.sqrt(2.0)
    return float(np.linalg.norm(matched)), float(np.linalg.norm(differential))


def _iterative_diagnostic_row(
    *,
    args: argparse.Namespace,
    case: CaseSpec,
    window_index: int,
    child_run_root: Path,
    case_root: Path,
    current_offsets: Mapping[str, float],
    next_offsets: Mapping[str, float],
    posterior_rows_by_label: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    labels = _wfe_labels()
    reference = _offset_vector(current_offsets, labels)
    posterior_offsets: dict[str, float] = {}
    error_over_sigma: list[float] = []
    residual_by_noll: dict[int, float] = {}
    for label in labels:
        row = posterior_rows_by_label.get(label)
        if row is None:
            posterior_offsets[label] = float(current_offsets.get(label, 0.0))
            continue
        posterior_mean = _posterior_float(row, ("posterior_mean", "mean", "posterior", "value"))
        truth = _posterior_float(row, ("truth_value", "theta_truth", "truth"))
        if not math.isfinite(truth):
            truth = 0.0
        posterior_offsets[label] = float(posterior_mean - truth) if math.isfinite(posterior_mean) else float(current_offsets.get(label, 0.0))
        sigma = _posterior_float(row, ("posterior_sigma", "sigma", "std"))
        if math.isfinite(sigma) and sigma != 0.0 and math.isfinite(posterior_offsets[label]):
            error_over_sigma.append(abs(posterior_offsets[label]) / abs(sigma))
    posterior = _offset_vector(posterior_offsets, labels)
    ideal_update = -reference
    actual_update = posterior - reference
    residual = posterior
    for i, noll in enumerate(NOLL_RANGE):
        m1 = abs(float(residual[2 * i]))
        m2 = abs(float(residual[2 * i + 1]))
        residual_by_noll[noll] = max(m1, m2)
    worst_noll = max(residual_by_noll, key=residual_by_noll.get) if residual_by_noll else ""
    matched_before, differential_before = _matched_differential_norms(current_offsets)
    matched_after, differential_after = _matched_differential_norms(posterior_offsets)
    previous = getattr(args, "_previous_residual_norms", {})
    key = case.case_name
    residual_norm = float(np.linalg.norm(residual))
    previous_norm = previous.get(key)
    previous[key] = residual_norm
    args._previous_residual_norms = previous
    bias_norm = float(np.linalg.norm(reference))
    return {
        "campaign": args.campaign,
        "condition_name": case.condition_name,
        "case_name": case.case_name,
        "draw_index": int(case.draw_index),
        "window_index": int(window_index),
        "update_gain": float(args.update_gain),
        "child_run_root": str(child_run_root),
        "case_root": str(case_root),
        "reference_error_norm_before": bias_norm,
        "posterior_error_norm_after": residual_norm,
        "update_norm": float(np.linalg.norm(actual_update)),
        "update_cosine_with_ideal": _cosine(actual_update, ideal_update),
        "vector_gain": _projection_gain(actual_update, ideal_update),
        "residual_norm_over_bias_norm": _safe_fraction(residual_norm, bias_norm),
        "residual_norm_decreased_from_previous_window": "" if previous_norm is None else bool(residual_norm < float(previous_norm)),
        "worst_noll": worst_noll,
        "worst_noll_residual_nm": residual_by_noll.get(worst_noll, float("nan")) if worst_noll != "" else float("nan"),
        "worst_error_over_sigma": max(error_over_sigma) if error_over_sigma else float("nan"),
        "matched_component_norm_before": matched_before,
        "differential_component_norm_before": differential_before,
        "matched_component_norm_after": matched_after,
        "differential_component_norm_after": differential_after,
        "next_reference_error_norm": float(np.linalg.norm(_offset_vector(next_offsets, labels))),
    }


def _execute_shard(args: argparse.Namespace) -> dict[str, Any]:
    context = _system_context(
        system_preset=args.system_preset,
        exposure_time_s=float(args.exposure_time_s),
        n_lambda=None if int(args.n_lambda) <= 0 else int(args.n_lambda),
    )
    run_root = Path(args.results_root).resolve() / str(args.run_name)
    expected_rows = _read_csv(run_root / "expected_outputs.csv")
    if not expected_rows:
        raise RuntimeError(f"Missing stored plan: {run_root / 'expected_outputs.csv'}")
    selected_rows = [row for row in expected_rows if int(row["shard_index"]) == int(args.shard_index)]
    cases_by_name = {
        case.case_name: case
        for case in _build_cases(
            campaign=args.campaign,
            truth_by_label=context.truth_by_label,
            wfe_pairing=args.wfe_pairing,
            n_draws=int(args.n_draws),
            include_zero_bias_control=True,
        )
    }
    if args.campaign == CAMPAIGN_B:
        return _execute_iterative_shard(args=args, selected_rows=selected_rows, cases_by_name=cases_by_name, truth_by_label=context.truth_by_label, run_root=run_root)
    selected = [cases_by_name[row["case_name"]] for row in selected_rows]
    if not selected_rows:
        return {"campaign": args.campaign, "shard_index": int(args.shard_index), "selected_cases": 0}
    child_results_roots = {row["child_results_root"] for row in selected_rows}
    child_run_names = {row["child_run_name"] for row in selected_rows}
    if len(child_results_roots) != 1 or len(child_run_names) != 1:
        raise RuntimeError("Shard rows must map to exactly one child results root and child run name for Campaign A.")
    child_results_root = Path(next(iter(child_results_roots)))
    child_run_name = next(iter(child_run_names))
    config_path = run_root / "shards" / f"shard_{int(args.shard_index):04d}.json"
    _write_json(
        config_path,
        _runner_config(args=args, cases=selected, run_name=child_run_name, n_subblocks=int(args.n_subblocks)),
    )
    _run_child_config(config_path, child_results_root, child_run_name, args)
    return {"campaign": args.campaign, "shard_index": int(args.shard_index), "selected_cases": len(selected)}


def _execute_iterative_shard(
    *,
    args: argparse.Namespace,
    selected_rows: Sequence[Mapping[str, str]],
    cases_by_name: Mapping[str, CaseSpec],
    truth_by_label: Mapping[str, float],
    run_root: Path,
) -> dict[str, Any]:
    if not selected_rows:
        return {"campaign": args.campaign, "shard_index": int(args.shard_index), "selected_cases": 0, "windows": 0}
    case_names = sorted({row["case_name"] for row in selected_rows})
    current_offsets_by_case = {name: dict(cases_by_name[name].offsets) for name in case_names}
    rows_by_window: dict[int, list[Mapping[str, str]]] = {}
    for row in selected_rows:
        rows_by_window.setdefault(int(row["window_index"]), []).append(row)
    diagnostics: list[dict[str, Any]] = []
    args._previous_residual_norms = {}
    for window_index in sorted(rows_by_window):
        window_rows = rows_by_window[window_index]
        selected_cases = [
            replace(cases_by_name[row["case_name"]], offsets=dict(current_offsets_by_case[row["case_name"]]))
            for row in window_rows
        ]
        child_run_names = {row["child_run_name"] for row in window_rows}
        child_results_roots = {row["child_results_root"] for row in window_rows}
        if len(child_run_names) != 1 or len(child_results_roots) != 1:
            raise RuntimeError(f"Campaign B window {window_index} shard rows must map to one child run.")
        child_run_name = next(iter(child_run_names))
        child_results_root = Path(next(iter(child_results_roots)))
        config_path = run_root / "shards" / f"window_{window_index:02d}_shard_{int(args.shard_index):04d}.json"
        _write_json(
            config_path,
            _runner_config(
                args=args,
                cases=selected_cases,
                run_name=child_run_name,
                n_subblocks=int(args.n_subblocks),
                update_gain=float(args.update_gain),
                window_index=window_index,
            ),
        )
        missing = [row for row in window_rows if not Path(row["posterior_by_parameter_csv"]).exists()]
        if missing:
            _run_child_config(config_path, child_results_root, child_run_name, args)
        for row in window_rows:
            posterior_path = Path(row["posterior_by_parameter_csv"])
            if not posterior_path.exists():
                raise RuntimeError(f"Missing posterior after Campaign B window {window_index}: {posterior_path}")
            case = cases_by_name[row["case_name"]]
            posterior_rows = _posterior_by_label(posterior_path)
            next_offsets = _next_offsets_from_posterior(
                current_offsets=current_offsets_by_case[case.case_name],
                posterior_rows_by_label=posterior_rows,
                truth_by_label=truth_by_label,
                update_gain=float(args.update_gain),
            )
            diagnostics.append(
                _iterative_diagnostic_row(
                    args=args,
                    case=case,
                    window_index=window_index,
                    child_run_root=Path(row["child_run_root"]),
                    case_root=Path(row["case_root"]),
                    current_offsets=current_offsets_by_case[case.case_name],
                    next_offsets=next_offsets,
                    posterior_rows_by_label=posterior_rows,
                )
            )
            current_offsets_by_case[case.case_name] = next_offsets
    diagnostics_path = (
        run_root
        / "analysis"
        / "shard_diagnostics"
        / f"iterative_window_diagnostics_shard_{int(args.shard_index):04d}.csv"
    )
    _write_csv(diagnostics_path, diagnostics)
    return {
        "campaign": args.campaign,
        "shard_index": int(args.shard_index),
        "selected_cases": len(case_names),
        "windows": len(rows_by_window),
        "iterative_diagnostics_path": str(diagnostics_path),
        "iterative_diagnostics_count": len(diagnostics),
    }


def _aggregate_only(args: argparse.Namespace) -> dict[str, Any]:
    run_root = Path(args.results_root).resolve() / str(args.run_name)
    expected_rows = _read_csv(run_root / "expected_outputs.csv")
    case_plan = _read_csv(run_root / "campaign_case_plan.csv")
    shard_plan = _read_csv(run_root / "campaign_shard_plan.csv")
    analysis_root = run_root / "analysis"
    shard_diagnostic_paths = sorted((analysis_root / "shard_diagnostics").glob("iterative_window_diagnostics_shard_*.csv"))
    inventory: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    existing = 0
    for row in expected_rows:
        posterior_path = Path(row["posterior_by_parameter_csv"])
        history_path = Path(row["posterior_history_csv"])
        case_summary_path = Path(row["case_summary_json"])
        campaign_summary_path = Path(row["campaign_summary_json"])
        status = {
            **row,
            "posterior_by_parameter_exists": posterior_path.exists(),
            "posterior_history_exists": history_path.exists(),
            "case_summary_exists": case_summary_path.exists(),
            "campaign_summary_exists": campaign_summary_path.exists(),
            "wrapper_analysis_summary_json": str(analysis_root / "aggregate_status.json"),
            "shard_iterative_diagnostics_available": bool(shard_diagnostic_paths),
            "shard_iterative_diagnostics_count": len(shard_diagnostic_paths),
        }
        inventory.append(status)
        if posterior_path.exists():
            existing += 1
            for posterior_row in _read_csv(posterior_path):
                posterior_rows.append(
                    {
                        **{key: row.get(key, "") for key in ("row_index", "shard_index", "case_name", "condition_name", "condition_kind", "draw_index", "window_index")},
                        **posterior_row,
                    }
                )
        else:
            missing_rows.append(status)

    _write_csv(analysis_root / "output_inventory.csv", inventory)
    _write_csv(analysis_root / "missing_outputs.csv", missing_rows)
    if posterior_rows:
        _write_csv(analysis_root / "posterior_by_parameter_all_cases.csv", posterior_rows)
        wfe_rows = _wfe_diagnostic_rows(posterior_rows)
        if wfe_rows:
            _write_csv(analysis_root / "wfe_vector_diagnostics.csv", wfe_rows)
            _write_csv(analysis_root / "wfe_pair_level_rows.csv", _wfe_pair_rows(wfe_rows))
    iterative_rows: list[dict[str, Any]] = []
    if args.campaign == CAMPAIGN_B:
        iterative_rows = _aggregate_iterative_diagnostics(args=args, expected_rows=expected_rows)
        if iterative_rows:
            _write_csv(analysis_root / "iterative_window_diagnostics.csv", iterative_rows)
    summary = {
        "schema_version": "single_star_both_wfe_aggregate_status.v1",
        "created_at": now_iso_local_ms(),
        "campaign": args.campaign,
        "run_root": str(run_root),
        "expected_outputs": len(expected_rows),
        "existing_posterior_tables": existing,
        "missing_posterior_tables": len(missing_rows),
        "case_plan_rows": len(case_plan),
        "shard_plan_rows": len(shard_plan),
        "posterior_rows": len(posterior_rows),
        "iterative_window_diagnostic_rows": len(iterative_rows),
        "shard_iterative_diagnostics_files": [str(path) for path in shard_diagnostic_paths],
        "shard_iterative_diagnostics_file_count": len(shard_diagnostic_paths),
        "forecast_enabled": False,
        "used_stored_plan": True,
        "reran_child_solves": False,
    }
    _write_json(analysis_root / "aggregate_status.json", summary)
    return summary


def _aggregate_iterative_diagnostics(*, args: argparse.Namespace, expected_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    context = _system_context(
        system_preset=args.system_preset,
        exposure_time_s=float(args.exposure_time_s),
        n_lambda=None if int(args.n_lambda) <= 0 else int(args.n_lambda),
    )
    cases_by_name = {
        case.case_name: case
        for case in _build_cases(
            campaign=args.campaign,
            truth_by_label=context.truth_by_label,
            wfe_pairing=args.wfe_pairing,
            n_draws=int(args.n_draws),
            include_zero_bias_control=True,
        )
    }
    current_offsets_by_case = {name: dict(case.offsets) for name, case in cases_by_name.items()}
    rows: list[dict[str, Any]] = []
    args._previous_residual_norms = {}
    for window_index in sorted({int(row["window_index"]) for row in expected_rows if row.get("window_index") not in ("", None)}):
        for row in [r for r in expected_rows if r.get("window_index") not in ("", None) and int(r["window_index"]) == window_index]:
            posterior_path = Path(row["posterior_by_parameter_csv"])
            if not posterior_path.exists() or row["case_name"] not in cases_by_name:
                continue
            case = cases_by_name[row["case_name"]]
            posterior_rows = _posterior_by_label(posterior_path)
            current = current_offsets_by_case[case.case_name]
            next_offsets = _next_offsets_from_posterior(
                current_offsets=current,
                posterior_rows_by_label=posterior_rows,
                truth_by_label=context.truth_by_label,
                update_gain=float(args.update_gain),
            )
            rows.append(
                _iterative_diagnostic_row(
                    args=args,
                    case=case,
                    window_index=window_index,
                    child_run_root=Path(row["child_run_root"]),
                    case_root=Path(row["case_root"]),
                    current_offsets=current,
                    next_offsets=next_offsets,
                    posterior_rows_by_label=posterior_rows,
                )
            )
            current_offsets_by_case[case.case_name] = next_offsets
    return rows


def _posterior_label(row: Mapping[str, Any]) -> str:
    for key in ("theta_label", "parameter", "label", "name"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _posterior_float(row: Mapping[str, Any], candidates: Sequence[str]) -> float:
    for key in candidates:
        value = row.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except ValueError:
                continue
    return float("nan")


def _wfe_diagnostic_rows(posterior_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in posterior_rows:
        label = _posterior_label(row)
        mirror = ""
        if "primary.zernike_coeffs_nm" in label:
            mirror = "M1"
        elif "secondary.zernike_coeffs_nm" in label:
            mirror = "M2"
        else:
            continue
        try:
            idx = int(label.rsplit("[", 1)[1].split("]", 1)[0])
        except (IndexError, ValueError):
            continue
        if idx < 0 or idx >= len(NOLL_RANGE):
            continue
        posterior_mean = _posterior_float(row, ("posterior_mean", "mean", "posterior", "value"))
        reference = _posterior_float(row, ("reference_value", "theta_reference", "reference", "prior_mean"))
        truth = _posterior_float(row, ("truth_value", "theta_truth", "truth"))
        sigma = _posterior_float(row, ("posterior_sigma", "sigma", "std"))
        bias = reference - truth if math.isfinite(reference) and math.isfinite(truth) else float("nan")
        residual = posterior_mean - truth if math.isfinite(posterior_mean) and math.isfinite(truth) else float("nan")
        shift = posterior_mean - reference if math.isfinite(posterior_mean) and math.isfinite(reference) else float("nan")
        ideal = truth - reference if math.isfinite(reference) and math.isfinite(truth) else float("nan")
        out.append(
            {
                **{key: row.get(key, "") for key in ("row_index", "shard_index", "case_name", "condition_name", "condition_kind", "draw_index", "window_index")},
                "theta_label": label,
                "mirror": mirror,
                "noll_index": NOLL_RANGE[idx],
                "bias_nm": bias,
                "shift_nm": shift,
                "residual_nm": residual,
                "correction_fraction": _safe_fraction(shift, ideal),
                "residual_fraction": _safe_fraction(residual, bias),
                "error_over_sigma": _safe_fraction(residual, sigma),
            }
        )
    return out


def _wfe_pair_rows(wfe_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = {}
    for row in wfe_rows:
        key = (
            str(row.get("case_name", "")),
            str(row.get("window_index", "")),
            str(row.get("draw_index", "")),
            str(row.get("noll_index", "")),
        )
        grouped.setdefault(key, {})[str(row.get("mirror", ""))] = row
    out: list[dict[str, Any]] = []
    for (case_name, window_index, draw_index, noll_index), mirrors in grouped.items():
        m1 = mirrors.get("M1", {})
        m2 = mirrors.get("M2", {})
        m1_residual = _posterior_float(m1, ("residual_nm",))
        m2_residual = _posterior_float(m2, ("residual_nm",))
        m1_shift = _posterior_float(m1, ("shift_nm",))
        m2_shift = _posterior_float(m2, ("shift_nm",))
        out.append(
            {
                "case_name": case_name,
                "window_index": window_index,
                "draw_index": draw_index,
                "noll_index": noll_index,
                "m1_shift_nm": m1_shift,
                "m2_shift_nm": m2_shift,
                "m1_residual_nm": m1_residual,
                "m2_residual_nm": m2_residual,
                "matched_residual_nm": (m1_residual + m2_residual) / math.sqrt(2.0)
                if math.isfinite(m1_residual) and math.isfinite(m2_residual)
                else float("nan"),
                "differential_residual_nm": (m1_residual - m2_residual) / math.sqrt(2.0)
                if math.isfinite(m1_residual) and math.isfinite(m2_residual)
                else float("nan"),
            }
        )
    return out


def _synthetic_output_test(args: argparse.Namespace) -> dict[str, Any]:
    validation = _write_plan(args)
    run_root = Path(validation["run_root"])
    _write_json(
        run_root / "analysis" / "campaign_summary.json",
        {
            "schema_version": "single_star_both_wfe_synthetic_output_test.v1",
            "synthetic_output_test": True,
            "not_scientific": True,
            "campaign": args.campaign,
        },
    )
    return {"synthetic_output_test": True, "not_scientific": True, "run_root": str(run_root)}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--system-preset", default=DEFAULT_SYSTEM_PRESET)
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--n-subblocks", type=int, default=5)
    parser.add_argument("--n-draws", type=int, default=64)
    parser.add_argument("--windows-per-draw", type=int, default=4)
    parser.add_argument("--exposure-time-s", type=float, default=0.05)
    parser.add_argument("--n-lambda", type=int, default=3)
    parser.add_argument("--wfe-pairing", choices=WFE_PAIRINGS, default="independent")
    parser.add_argument("--update-gain", type=float, default=1.0)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--array-throttle", type=int, default=8)
    parser.add_argument("--cpus-per-task", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--slurm-mem", default="64G")
    parser.add_argument("--slurm-time", default="12:00:00")
    parser.add_argument("--slurm-partition", default="compute")
    parser.add_argument("--slurm-account", default="shera_hpc")
    parser.add_argument("--slurm-job-name", default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute-shard", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--write-sbatch", action="store_true")
    parser.add_argument("--synthetic-output-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Deprecated alias for --plan-only.")
    parser.add_argument("--resource-time", dest="resource_time", action="store_true", default=False)
    parser.add_argument("--no-resource-time", dest="resource_time", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    if args.run_name is None:
        args.run_name = args.campaign
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if args.execute_shard and not (0 <= int(args.shard_index) < int(args.num_shards)):
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards.")
    if args.dry_run:
        args.plan_only = True
    if not (args.plan_only or args.execute_shard or args.aggregate_only or args.synthetic_output_test):
        args.plan_only = True

    if args.synthetic_output_test:
        result = _synthetic_output_test(args)
    elif args.execute_shard:
        result = _execute_shard(args)
    elif args.aggregate_only:
        result = _aggregate_only(args)
    else:
        result = _write_plan(args)
    if not args.quiet:
        print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
