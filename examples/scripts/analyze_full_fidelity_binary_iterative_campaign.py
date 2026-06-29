#!/usr/bin/env python3
"""Build a compact review bundle for full-fidelity binary iterative campaigns."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - exercised only on minimal installs
    plt = None


REQUIRED_ARTIFACTS = [
    "campaign_plan.json",
    "campaign_summary.json",
    "subblock_status_iterative.csv",
    "analysis/iterative_window_diagnostics.csv",
    "trajectory/smear_summary.csv",
    "model_split/model_split.json",
    "model_split/model_split_summary.json",
    "noise/noise_request_normalized.json",
    "noise/noise_render_provenance.json",
    "noise/noise_inference_provenance.json",
]

DASHBOARD_COMPONENTS = [
    "source target / components",
    "spectral grid",
    "component SEDs / weights",
    "QE",
    "M2 filter",
    "flux parameter preservation",
    "high-order WFE maps",
    "high-order WFE knowledge error",
    "low-order Zernike coefficients",
    "optics preset",
    "detector layers",
    "detector calibration maps",
    "detector noise",
    "trajectory source",
    "high-pass filter",
    "trace jitter / registration nuisance",
    "smear",
    "observation theta",
    "local eliminated phi",
    "prior draws / slow-state bias",
    "iterative update settings",
    "subblock settings",
    "optimizer settings",
    "Schur/FIM settings",
    "aggregation settings",
]

MISMATCH_COMPONENTS = [
    "spectral",
    "component_specific_seds",
    "qe",
    "m2_filter",
    "high_order_wfe",
    "high_order_wfe_knowledge_error",
    "low_order_zernike_mapping",
    "detector_layer_stack",
    "detector_calibration_maps",
    "detector_noise_model",
    "trajectory_truth_model",
    "smear_truth_model",
    "slow_state_prior_draw",
    "local_registration_policy",
]


def rel(path: Any, root: Path) -> str:
    if path in (None, "") or (isinstance(path, float) and math.isnan(path)):
        return ""
    p = Path(str(path))
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open() as f:
        return json.load(f)


def _has_non_whitespace_content(path: Path) -> bool:
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            if chunk.strip():
                return True
    return False


def read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    # Partial/failed HPC post-processing can leave zero-byte optional sidecars;
    # the analyzer treats those like missing optional tables.
    if path.stat().st_size == 0 or not _has_non_whitespace_content(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def finite_numeric_array(values: Any) -> np.ndarray:
    """Return finite values as a flat float array for plotting helpers."""

    try:
        arr = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _append_plot_warning(
    warnings_list: list[dict[str, Any]] | None,
    message: str,
    *,
    context: str | None = None,
) -> None:
    if warnings_list is None:
        return
    warning = {"message": message}
    if context:
        warning["context"] = context
    warnings_list.append(warning)


def _placeholder_plot(
    ax: Any,
    message: str,
    *,
    warnings_list: list[dict[str, Any]] | None = None,
    context: str | None = None,
) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    _append_plot_warning(warnings_list, message, context=context)


def safe_histogram_bins(
    values: Any,
    requested_bins: int,
    min_bins: int = 1,
) -> tuple[int, tuple[float, float] | None]:
    finite = finite_numeric_array(values)
    if finite.size == 0:
        return 0, None
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return 0, None
    if finite.size == 1 or np.isclose(lo, hi, rtol=1e-12, atol=0.0):
        center = float(finite[0])
        pad = max(abs(center) * 1e-6, 1e-12)
        return 1, (center - pad, center + pad)

    unique = np.unique(finite)
    max_bins = max(1, min(int(requested_bins), int(finite.size), int(unique.size)))
    bins = max(int(min_bins), max_bins)
    bins = min(bins, max_bins)
    if bins <= 0:
        bins = 1
    if np.isclose(lo, hi, rtol=1e-12, atol=0.0):
        pad = max(abs(lo) * 1e-6, 1e-12)
        return bins, (lo - pad, hi + pad)
    return bins, (lo, hi)


def safe_hist(
    ax: Any,
    values: Any,
    *,
    requested_bins: int = 20,
    min_bins: int = 3,
    label: str | None = None,
    color: str | None = None,
    alpha: float = 0.8,
    warnings_list: list[dict[str, Any]] | None = None,
    context: str | None = None,
) -> None:
    finite = finite_numeric_array(values)
    if finite.size == 0:
        _placeholder_plot(ax, "no finite data", warnings_list=warnings_list, context=context)
        return

    bins, value_range = safe_histogram_bins(
        finite,
        requested_bins=requested_bins,
        min_bins=min_bins,
    )
    if bins <= 0 or value_range is None:
        _placeholder_plot(ax, "no finite data", warnings_list=warnings_list, context=context)
        return

    try:
        if bins == 1:
            center = float(finite[0])
            ax.axvline(center, color=color or "#4C78A8", linewidth=2.0, label=label)
            ax.set_xlim(value_range)
            _append_plot_warning(
                warnings_list,
                "histogram data are constant or nearly constant; drew a marker instead of binned histogram",
                context=context,
            )
        else:
            # Avoid Matplotlib's auto edge path for degenerate precision cases:
            # "Too many bins for data range. Cannot create 3 finite-sized bins."
            ax.hist(
                finite,
                bins=bins,
                range=value_range,
                label=label,
                color=color,
                alpha=alpha,
            )
    except ValueError as exc:
        _placeholder_plot(
            ax,
            f"histogram unavailable: {exc}",
            warnings_list=warnings_list,
            context=context,
        )


def scalar(data: dict[str, Any], dotted: str, default: Any = "") -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def compact(value: Any, max_len: int = 180) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def parse_window_path(path: Path) -> tuple[str, int]:
    parts = path.parts
    if "cases" in parts and "windows" in parts:
        ci = parts.index("cases")
        wi = parts.index("windows")
        case_name = parts[ci + 1]
        window = parts[wi + 1]
        return case_name, int(window.replace("window_", ""))
    return "", -1


def label_group(label: str) -> str:
    if label.startswith("source."):
        return "source"
    if label == "optics.plate_scale_as_per_pix":
        return "optics.plate_scale"
    if "primary.zernike" in label:
        return "optics.primary_zernikes"
    if "secondary.zernike" in label:
        return "optics.secondary_zernikes"
    return label.split(".")[0]


def label_units(label: str) -> str:
    if label == "source.separation_as":
        return "as"
    if label == "source.log_flux_total":
        return "dex"
    if label == "source.contrast":
        return "scalar"
    if label == "optics.plate_scale_as_per_pix":
        return "as/pix"
    if "zernike_coeffs_nm" in label:
        return "nm"
    return ""


def display_offset(label: str, value: float) -> str:
    if pd.isna(value):
        return ""
    if label == "source.separation_as":
        return f"{value * 1e6:.3g} microas"
    return f"{value:.3g} {label_units(label)}".strip()


def validate_required(run_root: Path, strict: bool) -> list[str]:
    missing = [p for p in REQUIRED_ARTIFACTS if not (run_root / p).exists()]
    summary = read_json(run_root / "campaign_summary.json", {})
    update_mode = str(summary.get("update_mode", "physical_full"))
    if strict and update_mode.startswith("eigen_"):
        eigen_paths = list(
            run_root.glob(
                "cases/*/windows/window_*/eigen_update_diagnostics.json"
            )
        )
        if not eigen_paths:
            missing.append(
                "cases/*/windows/window_*/eigen_update_diagnostics.json"
            )
    if strict and missing:
        raise FileNotFoundError("Missing required campaign artifacts: " + ", ".join(missing))
    return missing


def combine_window_tables(run_root: Path, name: str) -> pd.DataFrame:
    rows = []
    for path in sorted(run_root.glob(f"cases/*/windows/window_*/{name}")):
        df = read_csv(path)
        if df.empty:
            continue
        case_name, window_index = parse_window_path(path)
        df.insert(0, "source_path", rel(path, run_root))
        df.insert(0, "window_index", window_index)
        df.insert(0, "case_name_root", case_name)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def iterative_window_progress(run_root: Path) -> pd.DataFrame:
    top = read_csv(run_root / "analysis/iterative_window_diagnostics.csv")
    if top.empty:
        frames = []
        for path in sorted(run_root.glob("cases/*/windows/window_*/iterative_window_diagnostics.csv")):
            df = read_csv(path)
            if not df.empty:
                frames.append(df)
        top = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    wanted = [
        "case_name",
        "window_index",
        "n_subblocks",
        "update_gain",
        "update_mode",
        "eigen_n_modes_kept",
        "eigen_n_modes_total",
        "eigen_min_kept_eigenvalue_rel",
        "eigen_max_rejected_eigenvalue_rel",
        "reference_error_norm_before",
        "posterior_error_norm_after",
        "next_reference_error_norm",
        "residual_norm_over_bias_norm",
        "update_cosine_with_ideal",
        "vector_gain",
        "applied_vector_gain",
        "separation_reference_error_before_microas",
        "separation_posterior_error_after_microas",
        "separation_next_reference_error_microas",
        "separation_update_sign_toward_truth",
        "separation_next_reference_improved",
        "posterior_sigma_separation_microas",
        "source_scalar_reference_error_norm_before",
        "source_scalar_posterior_error_norm_after",
        "plate_scale_reference_error_norm_before",
        "plate_scale_posterior_error_norm_after",
        "m1_zernike_reference_error_norm_before",
        "m1_zernike_posterior_error_norm_after",
        "m2_zernike_reference_error_norm_before",
        "m2_zernike_posterior_error_norm_after",
    ]
    for col in wanted:
        if col not in top.columns:
            top[col] = np.nan
    return top[wanted].sort_values(["case_name", "window_index"], na_position="last")


def eigen_update_tables(
    run_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    contribution_rows: list[dict[str, Any]] = []
    for path in sorted(
        run_root.glob("cases/*/windows/window_*/eigen_update_diagnostics.json")
    ):
        data = read_json(path, {})
        case_name, window_index = parse_window_path(path)
        relative = np.asarray(data.get("eigenvalue_relative", []), dtype=float)
        kept = np.asarray(data.get("kept_mode_mask", []), dtype=bool)
        summary_rows.append(
            {
                "case_name": case_name,
                "window_index": window_index,
                "update_mode": data.get("update_mode", ""),
                "basis_source": data.get("basis_source", ""),
                "gate_source": data.get("gate_source", ""),
                "whiten": data.get("whiten", False),
                "n_modes_total": data.get("n_modes_total", 0),
                "n_modes_kept": data.get("n_modes_kept", 0),
                "min_kept_relative_eigenvalue": (
                    float(np.min(relative[kept]))
                    if relative.size and kept.size == relative.size and np.any(kept)
                    else np.nan
                ),
                "max_rejected_relative_eigenvalue": (
                    float(np.max(relative[~kept]))
                    if relative.size
                    and kept.size == relative.size
                    and np.any(~kept)
                    else np.nan
                ),
                "physical_update_norm_full": float(
                    np.linalg.norm(data.get("physical_update_full", []))
                ),
                "physical_update_norm_applied": float(
                    np.linalg.norm(data.get("physical_update_applied", []))
                ),
                "eigen_update_norm_full": float(
                    np.linalg.norm(data.get("eigen_update_full", []))
                ),
                "eigen_update_norm_applied": float(
                    np.linalg.norm(data.get("eigen_update_applied", []))
                ),
                "source_path": rel(path, run_root),
            }
        )
        modes_path = path.with_name("eigen_update_modes.csv")
        modes = read_csv(modes_path)
        if modes.empty:
            modes = pd.DataFrame(data.get("mode_rows", []))
        if modes.empty:
            continue
        modes.insert(0, "source_path", rel(modes_path, run_root))
        modes.insert(0, "window_index", window_index)
        modes.insert(0, "case_name", case_name)
        mode_frames.append(modes)
        for _, row in modes.iterrows():
            contribution_rows.append(
                {
                    "case_name": case_name,
                    "window_index": window_index,
                    "mode_index": row.get("mode_index", np.nan),
                    "kept": row.get("kept", np.nan),
                    "dominant_labels": row.get("top_contributors", ""),
                    "source_group_norm": row.get("group_norm_source", np.nan),
                    "plate_scale_group_norm": row.get(
                        "group_norm_optics.plate_scale",
                        np.nan,
                    ),
                    "m1_group_norm": row.get(
                        "group_norm_optics.primary_zernikes",
                        np.nan,
                    ),
                    "m2_group_norm": row.get(
                        "group_norm_optics.secondary_zernikes",
                        np.nan,
                    ),
                }
            )
    combined_modes = (
        pd.concat(mode_frames, ignore_index=True) if mode_frames else pd.DataFrame()
    )
    return (
        combined_modes,
        pd.DataFrame(summary_rows),
        pd.DataFrame(contribution_rows),
    )


def iterative_parameter_progress(run_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_root.glob("cases/*/windows/window_*/iterative_reference_update.json")):
        data = read_json(path, {})
        case_name = data.get("case_name") or parse_window_path(path)[0]
        window_index = data.get("window_index", parse_window_path(path)[1])
        current = data.get("current_offsets", {})
        post = data.get("posterior_offsets", {})
        next_offsets = data.get("next_offsets", {})
        truth = data.get("truth_by_label", {})
        posterior_sigmas: dict[str, float] = {}
        post_path = Path(data.get("posterior_table_path", ""))
        if not post_path.is_absolute():
            post_path = path.parent / post_path
        post_df = read_csv(post_path)
        if not post_df.empty and {"theta_label", "posterior_sigma"}.issubset(post_df.columns):
            posterior_sigmas = dict(zip(post_df["theta_label"], post_df["posterior_sigma"]))
        for label in sorted(set(current) | set(post) | set(next_offsets) | set(truth)):
            t = float(truth.get(label, np.nan))
            co = float(current.get(label, np.nan))
            po = float(post.get(label, np.nan))
            no = float(next_offsets.get(label, np.nan))
            rows.append(
                {
                    "case_name": case_name,
                    "window_index": window_index,
                    "label": label,
                    "truth_value": t,
                    "current_offset": co,
                    "posterior_offset": po,
                    "next_offset": no,
                    "current_value": t + co if not pd.isna(t) and not pd.isna(co) else np.nan,
                    "posterior_value": t + po if not pd.isna(t) and not pd.isna(po) else np.nan,
                    "next_value": t + no if not pd.isna(t) and not pd.isna(no) else np.nan,
                    "posterior_minus_truth": po,
                    "next_minus_truth": no,
                    "applied_delta": no - co if not pd.isna(no) and not pd.isna(co) else np.nan,
                    "posterior_sigma": posterior_sigmas.get(label, np.nan),
                    "units": label_units(label),
                    "group": label_group(label),
                    "improved_current_to_next": abs(no) < abs(co) if not pd.isna(no) and not pd.isna(co) else np.nan,
                    "improved_current_to_posterior": abs(po) < abs(co) if not pd.isna(po) and not pd.isna(co) else np.nan,
                    "source_path": rel(path, run_root),
                }
            )
    return pd.DataFrame(rows)


def slow_state_tables(param_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if param_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    for case_name, case_df in param_df.groupby("case_name"):
        first_w = int(case_df["window_index"].min())
        for _, r in case_df[case_df["window_index"] == first_w].iterrows():
            rows.append(state_row("initial_reference", r, r["truth_value"], r["current_value"], r["current_offset"]))
        for window_index, wdf in case_df.groupby("window_index"):
            for _, r in wdf.iterrows():
                rows.append(state_row(f"posterior_window_{int(window_index):03d}", r, r["truth_value"], r["posterior_value"], r["posterior_offset"]))
                label = "final_reference" if window_index == case_df["window_index"].max() else "next_reference"
                rows.append(state_row(f"{label}_window_{int(window_index):03d}", r, r["truth_value"], r["next_value"], r["next_offset"]))
    evo = pd.DataFrame(rows).drop_duplicates(["state", "parameter_label"])
    final_rows = []
    for label, ldf in evo.groupby("parameter_label"):
        init = ldf[ldf["state"] == "initial_reference"]
        final = ldf[ldf["state"].str.startswith("final_reference")]
        if init.empty or final.empty:
            continue
        i = float(init.iloc[0]["offset"])
        f = float(final.iloc[-1]["offset"])
        sigma = final.iloc[-1].get("sigma_if_available", np.nan)
        final_rows.append(
            {
                "parameter_label": label,
                "initial_offset": i,
                "final_offset": f,
                "absolute_improvement": abs(i) - abs(f),
                "fractional_improvement": (abs(i) - abs(f)) / abs(i) if abs(i) > 0 else np.nan,
                "improved": abs(f) < abs(i),
                "final_error_over_sigma": abs(f) / sigma if pd.notna(sigma) and sigma else np.nan,
                "group": final.iloc[-1]["group"],
            }
        )
    return evo, pd.DataFrame(final_rows)


def state_row(state: str, r: pd.Series, truth: float, value: float, offset: float) -> dict[str, Any]:
    return {
        "state": state,
        "parameter_label": r["label"],
        "truth": truth,
        "value": value,
        "offset": offset,
        "offset_display": display_offset(str(r["label"]), offset),
        "sigma_if_available": r.get("posterior_sigma", np.nan),
        "group": r.get("group", ""),
    }


def campaign_dashboard(run_root: Path, artifacts: dict[str, Any]) -> pd.DataFrame:
    plan = artifacts["campaign_plan"]
    summary = artifacts["campaign_summary"]
    split = artifacts["model_split"]
    split_summary = artifacts["model_split_summary"]
    noise = artifacts["noise_request"]
    system = scalar(plan, "layout_metadata.system", {})
    comps = split.get("components", {})
    paths = split_summary.get("artifact_paths", split.get("artifact_paths", {}))
    detector_layers = system.get("detector_layer_stack_after_global_overrides") or system.get("detector_layer_stack_from_preset", [])
    rows = []

    def add(component: str, status: str, truth: Any, inf: Any, diff: Any, action: str, artifact: str) -> None:
        rows.append(
            {
                "Component": component,
                "Status": status,
                "Truth setting": compact(truth),
                "Inference setting": compact(inf),
                "Difference / mismatch": compact(diff),
                "Reviewer action": action,
                "Artifact path": artifact,
            }
        )

    add("source target / components", "configured", system.get("source_target"), system.get("source_kind"), "binary components from source config", "confirm target and binary component assumptions", "campaign_plan.json")
    add("spectral grid", "configured", scalar(plan, "layout_metadata.system.source_wavelengths_m", ""), scalar(plan, "layout_metadata.system.n_lambda", ""), "see spectral deck", "inspect weights/moments if chromatic bias matters", rel(paths.get("spectral_spectral_deck_manifest", "model_split/spectral/spectral_deck_manifest.json"), run_root))
    spectral = comps.get("spectral_model", {})
    add("component SEDs / weights", "matched" if spectral.get("matched") else "intentionally_mismatched", "truth_weights.csv", "inference_weights.csv", f"component matched={spectral.get('matched')}", "review component weights and flux preservation", rel(paths.get("spectral_spectral_comparison", "model_split/spectral/spectral_comparison.json"), run_root))
    add("QE", "configured", system.get("qe_kind", "from preset/config"), system.get("qe_kind", "from preset/config"), "not separately varied by campaign artifacts", "review model split if QE becomes active mismatch", "campaign_plan.json")
    add("M2 filter", "configured", system.get("m2_filter", "from preset/config"), system.get("m2_filter", "from preset/config"), "not separately varied by campaign artifacts", "review optics preset", "campaign_plan.json")
    add("flux parameter preservation", "preserved", "band-integrated flux", "band-integrated flux", "spectral provenance reports preservation", "ok", rel(paths.get("spectral_spectral_moments", "model_split/spectral/spectral_moments.json"), run_root))
    wfe = comps.get("high_order_wfe", {})
    add("high-order WFE maps", "intentionally_mismatched" if not wfe.get("matched", True) else "matched", wfe.get("truth_label"), wfe.get("inference_label"), f"matched={wfe.get('matched')}", "inspect RMS and map paths", rel(paths.get("high_order_wfe_primary_high_order_truth_opd_nm.fits", "model_split/high_order_wfe/high_order_wfe_summary.json"), run_root))
    add("high-order WFE knowledge error", "enabled" if wfe.get("enabled") else "not_applicable", "truth high-order maps", "knowledge-error maps", "small knowledge error if configured", "confirm intended WFE mismatch scale", "model_split/high_order_wfe/high_order_wfe_summary.json")
    add("low-order Zernike coefficients", "configured", "truth low-order CSV", "knowledge low-order CSV", "active theta labels include selected M1/M2 terms", "review matched M1/M2 behavior", rel(paths.get("high_order_wfe_low_order_zernike_errors.csv", "model_split/high_order_wfe/maps/low_order_zernike_errors.csv"), run_root))
    add("optics preset", "configured", system.get("system_preset"), system.get("optics_kind"), "preset resolved before run", "ok", "campaign_plan.json")
    add("detector layers", "enabled", f"{len(detector_layers)} truth/render layers", f"{len(detector_layers)} inference layers", [layer.get("name") for layer in detector_layers], "confirm smear/jitter/calibration layers", "campaign_plan.json")
    detector_ke = comps.get("detector_calibration_knowledge_error", {})
    detector_ke_path = paths.get(
        "detector_knowledge_error_provenance_json",
        "model_split/detector_knowledge_error/detector_knowledge_error_provenance.json",
    )
    add(
        "detector calibration maps",
        "intentionally_mismatched" if detector_ke.get("enabled") and not detector_ke.get("matched", True) else "matched_or_disabled",
        detector_ke.get("truth_label", "nominal"),
        detector_ke.get("inference_label", "nominal"),
        f"apply_to={detector_ke.get('apply_to', '')}; patched={compact(detector_ke.get('patched_layers', {}))}",
        "inspect seeds, RMS, hashes, and map summary stats",
        rel(detector_ke_path, run_root),
    )
    add("detector noise", "enabled" if noise.get("enabled") else "disabled", artifacts["noise_render"].get("mode", noise.get("legacy_noise_mode", "")), artifacts["noise_inference"].get("mode", noise.get("legacy_noise_mode", "")), f"variance_floor={noise.get('variance_floor')}", "confirm variance model and seed policy", "noise/noise_request_normalized.json")
    add("trajectory source", "configured", "trajectory/frame_truth or subblock frame_truth", "starting_guess_prediction", "external trajectory CSVs supplied to subblocks", "review residual solve demand", "trajectory/")
    add("high-pass filter", "enabled" if (run_root / "trajectory/trajectory_filter_summary.csv").exists() else "missing", "trajectory_raw.csv", "trajectory_filtered.csv", "filter summary available", "inspect removed RMS", "trajectory/trajectory_filter_summary.csv")
    add("trace jitter / registration nuisance", "configured", "trace jitter in realized command", "local phi eliminated", "see --trace-jitter and --phi-ref", "check local solve interpretation", "subblock_status_iterative.csv")
    smear = comps.get("trajectory_smear", {})
    add("smear", "enabled" if smear.get("enabled") else "missing", smear.get("mode"), smear.get("target_layer"), "per-subblock render/inference match in smear summary", "verify kernel lengths and angles", "trajectory/smear_summary.csv")
    add("observation theta", "configured", scalar(plan, "theta_layout.labels", []), scalar(plan, "theta_layout.size", ""), "slow-state vector labels", "ok", "campaign_plan.json")
    add("local eliminated phi", "configured", "trajectory/registration nuisance", "Schur local phi", "policy read from subblock commands", "confirm whether truth phi was used", "subblock_runs/**/subprocess_diagnostics.json")
    add("prior draws / slow-state bias", "configured", "truth_realization_by_label.csv", "prior_draws.csv", "initial reference offsets", "review slow_state_final_summary.csv", "prior_draws.csv")
    add("iterative update settings", "configured", f"windows={summary.get('windows_per_draw')}", f"gain={summary.get('update_gain')}, mode={summary.get('update_mode')}", f"completed_subblocks={summary.get('completed_subblocks')}", "review update alignment", "analysis/iterative_window_diagnostics.csv")
    add("subblock settings", "ok" if summary.get("failed_subblocks", 0) == 0 else "review", f"subblocks/window={summary.get('subblocks_per_window')}", "realized commands", f"failed={summary.get('failed_subblocks')}", "review failures/log tails", "subblock_status_iterative.csv")
    add("optimizer settings", "configured", "reference optimizer in commands", "reference optimizer in commands", "parsed per subblock when available", "review convergence/loss metrics", "subblock_runs/**/subprocess_diagnostics.json")
    add("Schur/FIM settings", "configured", "Schur summary matrices", "reduced information", "condition/eigen diagnostics if matrices exist", "review sloppiness tables", "subblock_runs/**/subblock_summary_matrices.npz")
    add("aggregation settings", "ok", "aggregate campaign summary", "combined window outputs", summary.get("existing_outputs_by_kind", {}), "ok", "campaign_summary.json")
    return pd.DataFrame(rows)


def mismatch_dashboard(run_root: Path, artifacts: dict[str, Any], local_policy: str) -> pd.DataFrame:
    split = artifacts["model_split"]
    comps = split.get("components", {})
    paths = split.get("artifact_paths", {})
    noise = artifacts["noise_request"]
    rows = []

    def add(component: str, truth: Any, inf: Any, status: str, effect: str, artifact: str, note: str) -> None:
        rows.append(
            {
                "component": component,
                "truth_setting": compact(truth),
                "inference_setting": compact(inf),
                "mismatch_status": status,
                "expected_effect": effect,
                "artifact_path": artifact,
                "reviewer_note": note,
            }
        )

    spectral = comps.get("spectral_model", {})
    add("spectral", "truth spectral deck", "inference spectral deck", "matched_shape" if spectral.get("matched") else "intentional_or_component_mismatch", "chromatic centroid/flux weighting bias if shapes differ", rel(paths.get("spectral_spectral_comparison", "model_split/spectral/spectral_comparison.json"), run_root), "Component weights can differ by component while truth/inference decks may still be intentional.")
    add("component_specific_seds", "primary/secondary truth weights", "primary/secondary inference weights", "review", "component color differences affect PSF weighting", "model_split/spectral", "Compare truth_weights.csv and inference_weights.csv.")
    add("qe", "resolved system QE", "resolved system QE", "not_separately_varied", "QE mismatch would scale chromatic/flux response", "campaign_plan.json", "No separate QE split artifact was found.")
    add("m2_filter", "resolved optics/filter", "resolved optics/filter", "not_separately_varied", "M2 filter mismatch could mimic chromatic or WFE effects", "campaign_plan.json", "No separate M2 filter split artifact was found.")
    wfe = comps.get("high_order_wfe", {})
    add("high_order_wfe", wfe.get("truth_label"), wfe.get("inference_label"), "intentional_small_mismatch" if wfe.get("enabled") and not wfe.get("matched", True) else "matched_or_disabled", "unmodeled high-order WFE can leak into low-order/Zernike estimates", "model_split/high_order_wfe/high_order_wfe_summary.json", "Review RMS fields in the WFE summary.")
    add("high_order_wfe_knowledge_error", "truth maps", "knowledge + error maps", "intentional_small_mismatch" if wfe.get("enabled") else "not_applicable", "tests robustness to small map knowledge error", "model_split/high_order_wfe/config_maps", "Primary/secondary error arrays are indexed in model_split.")
    add("low_order_zernike_mapping", "truth low-order CSV", "knowledge low-order CSV", "review", "M1/M2 degeneracy can appear as correlated or opposite updates", "model_split/high_order_wfe/maps/low_order_zernike_errors.csv", "See zernike_m1_m2_offsets.png.")
    add("detector_layer_stack", "render detector layers", "inference detector layers", "configured", "layer mismatch can bias astrometry and WFE", "campaign_plan.json", "Review detector layer stack rows.")
    detector_ke = comps.get("detector_calibration_knowledge_error", {})
    detector_ke_path = paths.get(
        "detector_knowledge_error_provenance_json",
        "model_split/detector_knowledge_error/detector_knowledge_error_provenance.json",
    )
    add(
        "detector_calibration_maps",
        detector_ke.get("truth_label", "nominal truth maps"),
        detector_ke.get("inference_label", "nominal inference maps"),
        "intentional_small_mismatch" if detector_ke.get("enabled") and not detector_ke.get("matched", True) else "matched_or_disabled",
        "PRF/pixel offset mismatch can couple to trace",
        rel(detector_ke_path, run_root),
        "Detector KE provenance records seeds, RMS requests, hashes, and summary stats.",
    )
    add("detector_noise_model", noise.get("legacy_noise_mode", ""), noise.get("use_render_variance_resolved", ""), "matched_or_inherited" if noise.get("enabled") else "disabled", "noise affects posterior scale and convergence", "noise/noise_request_normalized.json", f"variance_floor={noise.get('variance_floor')}")
    traj = comps.get("trajectory_smear", {})
    add("trajectory_truth_model", "frame_truth.csv", "starting_guess_prediction.csv", "intentional_residual", "local phi/registration solve must absorb trajectory residuals", "trajectory/", "Review trajectory_residual_summary.csv.")
    add("smear_truth_model", traj.get("mode"), traj.get("target_layer"), "matched_subblock_constant" if traj.get("enabled") else "missing", "smear mismatch can broaden residuals and alter Fisher scale", "trajectory/smear_summary.csv", "Render/inference match flags are in smear_summary_review.csv.")
    add("slow_state_prior_draw", "truth_realization_by_label.csv", "prior_draws.csv", "intentional_initial_bias", "iterative update should reduce slow-state offsets", "prior_draws.csv", "Review slow_state_final_summary.csv.")
    add("local_registration_policy", "truth trace available", local_policy, "diagnostic_only" if "truth_when_available" in local_policy else "configured", "truth phi policy changes interpretation of local solve", "subblock_runs/**/subprocess_diagnostics.json", "This is parsed from realized commands.")
    return pd.DataFrame(rows)


def execution_status(run_root: Path, campaign_summary: dict[str, Any]) -> pd.DataFrame:
    rows = [{"metric": k, "value": compact(v)} for k, v in campaign_summary.items() if k in {
        "expected_output_rows",
        "missing_output_rows",
        "completed_subblocks",
        "failed_subblocks",
        "incomplete_windows",
        "first_failure",
        "existing_outputs_by_kind",
        "missing_posterior_tables",
        "missing_summaries",
        "iterative_window_diagnostic_rows",
        "windows_per_draw",
        "subblocks_per_window",
        "update_gain",
        "update_mode",
    }]
    status = read_csv(run_root / "subblock_status_iterative.csv")
    if not status.empty:
        rows += [
            {"metric": "runtime_seconds_min", "value": status.get("elapsed_seconds", pd.Series(dtype=float)).min()},
            {"metric": "runtime_seconds_median", "value": status.get("elapsed_seconds", pd.Series(dtype=float)).median()},
            {"metric": "runtime_seconds_max", "value": status.get("elapsed_seconds", pd.Series(dtype=float)).max()},
            {"metric": "return_codes", "value": compact(dict(Counter(status.get("return_code", []))))},
        ]
    return pd.DataFrame(rows)


def subblock_metric_summary(run_root: Path) -> tuple[pd.DataFrame, str]:
    status = read_csv(run_root / "subblock_status_iterative.csv")
    rows = []
    policies = []
    for _, s in status.iterrows() if not status.empty else []:
        summary_path = Path(str(s.get("summary_path", "")))
        diag_path = Path(str(s.get("subprocess_diagnostics_path", "")))
        summary = read_json(summary_path, {})
        diag = read_json(diag_path, {})
        command = diag.get("command", [])
        phi_ref = command_arg(command, "--phi-ref")
        if phi_ref:
            policies.append(phi_ref)
        row = {
            "case_name": s.get("case_name", ""),
            "window_index": s.get("window_index", np.nan),
            "subblock_index": s.get("window_subblock_index", np.nan),
            "global_subblock_index": s.get("global_subblock_index", np.nan),
            "status": s.get("status", ""),
            "return_code": s.get("return_code", np.nan),
            "elapsed_seconds": s.get("elapsed_seconds", np.nan),
            "n_frames": scalar(summary, "information_accounting.n_frames_total", command_arg(command, "--n-frames")),
            "exposure_time_s": command_arg(command, "--exposure-time-s"),
            "optimizer_n_iter": command_arg(command, "--reference-n-iter"),
            "optimizer_converged": scalar(summary, "reference_optimizer.converged", ""),
            "early_stopping_triggered": scalar(summary, "reference_optimizer.early_stopping_triggered", ""),
            "final_loss": scalar(summary, "reference_optimizer.final_loss", ""),
            "initial_loss": scalar(summary, "reference_optimizer.initial_loss", ""),
            "loss_delta": "",
            "chi2": scalar(summary, "fit_diagnostics.chi2", ""),
            "reduced_chi2": scalar(summary, "fit_diagnostics.reduced_chi2", ""),
            "information_scale": scalar(summary, "information_accounting.summary_information_scale", ""),
            "theta_dim": len(scalar(summary, "theta.labels", [])) or len(command_arg(command, "--theta-keys", "").split(",")) if command_arg(command, "--theta-keys", "") else "",
            "phi_dim": scalar(summary, "dimensions.phi_dim", ""),
            "local_phi_strategy": scalar(summary, "objective.temporal_kind", ""),
            "phi_ref_policy": phi_ref,
            "used_truth_phi": "truth" in phi_ref,
            "posterior_label_count": "",
            "warnings": compact(scalar(summary, "information_accounting.warnings", [])),
            "summary_path": rel(summary_path, run_root),
        }
        try:
            row["loss_delta"] = float(row["final_loss"]) - float(row["initial_loss"])
        except Exception:
            pass
        rows.append(row)
    policy_text = ", ".join(f"{k} ({v})" for k, v in Counter(policies).items()) if policies else "unknown"
    return pd.DataFrame(rows), policy_text


def command_arg(command: list[Any], flag: str, default: str = "") -> str:
    vals = [str(x) for x in command]
    if flag not in vals:
        return default
    idx = vals.index(flag)
    if idx + 1 >= len(vals) or vals[idx + 1].startswith("--"):
        return "true"
    return vals[idx + 1]


def trajectory_tables(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    traj_rows = []
    for path in sorted(run_root.glob("trajectory/subblock_*/frame_truth.csv")):
        df = read_csv(path)
        if df.empty:
            continue
        subblock = int(path.parent.name.replace("subblock_", ""))
        numeric = df.select_dtypes(include=[np.number])
        row = {"subblock_index": subblock, "source_path": rel(path, run_root), "n_frames": len(df)}
        for col in numeric.columns:
            row[f"{col}_min"] = numeric[col].min()
            row[f"{col}_max"] = numeric[col].max()
            row[f"{col}_mean"] = numeric[col].mean()
        traj_rows.append(row)
    residual_rows = []
    for truth_path in sorted(run_root.glob("trajectory/subblock_*/frame_truth.csv")):
        guess_path = truth_path.parent / "starting_guess_prediction.csv"
        truth = read_csv(truth_path)
        guess = read_csv(guess_path)
        if truth.empty or guess.empty:
            continue
        subblock = int(truth_path.parent.name.replace("subblock_", ""))
        row = {"subblock_index": subblock, "truth_path": rel(truth_path, run_root), "model_path": rel(guess_path, run_root)}
        for col in set(truth.columns) & set(guess.columns):
            if pd.api.types.is_numeric_dtype(truth[col]) and pd.api.types.is_numeric_dtype(guess[col]):
                diff = guess[col].to_numpy()[: min(len(truth), len(guess))] - truth[col].to_numpy()[: min(len(truth), len(guess))]
                row[f"{col}_residual_mean"] = float(np.mean(diff))
                row[f"{col}_residual_rms"] = float(np.sqrt(np.mean(diff**2)))
                row[f"{col}_residual_max_abs"] = float(np.max(np.abs(diff)))
        residual_rows.append(row)
    smear = read_csv(run_root / "trajectory/smear_summary.csv")
    return pd.DataFrame(traj_rows), pd.DataFrame(residual_rows), smear


def sloppiness_tables(run_root: Path, param_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    corr_rows = []
    eig_rows = []
    labels = []
    plan = read_json(run_root / "campaign_plan.json", {})
    labels = scalar(plan, "theta_layout.labels", []) or []
    matrices = sorted(run_root.glob("subblock_runs/**/study/schur_summary/subblock_summary_matrices.npz"))
    if not matrices:
        reason = "No subblock_summary_matrices.npz files found."
        return pd.DataFrame([{"covariance_available": False, "reason": reason}]), pd.DataFrame([{"covariance_available": False, "reason": reason}]), pd.DataFrame([{"covariance_available": False, "reason": reason}])
    for path in matrices:
        try:
            z = np.load(path, allow_pickle=True)
            info = np.asarray(z["reduced_information"], dtype=float)
            cov = np.linalg.pinv(info)
            diag = np.diag(cov)
            sigma = np.sqrt(np.where(diag >= 0, diag, np.nan))
            cond = float(np.linalg.cond(info))
            vals = np.linalg.eigvalsh(info)
            stem = rel(path, run_root)
            for i, sig in enumerate(sigma):
                rows.append({"matrix_path": stem, "label": labels[i] if i < len(labels) else f"theta[{i}]", "posterior_sigma_from_information": sig, "condition_number": cond, "covariance_available": True})
            denom = np.outer(sigma, sigma)
            corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom != 0)
            pairs = []
            for i in range(corr.shape[0]):
                for j in range(i + 1, corr.shape[1]):
                    pairs.append((abs(corr[i, j]), corr[i, j], i, j))
            for _, c, i, j in sorted(pairs, reverse=True)[:10]:
                corr_rows.append({"matrix_path": stem, "label_i": labels[i] if i < len(labels) else f"theta[{i}]", "label_j": labels[j] if j < len(labels) else f"theta[{j}]", "correlation": c, "abs_correlation": abs(c), "covariance_available": True})
            eig_rows.append({"matrix_path": stem, "condition_number": cond, "eigenvalue_min": float(np.min(vals)), "eigenvalue_max": float(np.max(vals)), "eigenvalue_spread": float(np.max(vals) / np.min(vals)) if np.min(vals) > 0 else np.inf, "covariance_available": True})
        except Exception as exc:
            rows.append({"matrix_path": rel(path, run_root), "covariance_available": False, "reason": str(exc)})
    if not param_df.empty:
        latest = param_df.sort_values(["case_name", "window_index"]).groupby("label").tail(1)
        for _, r in latest.iterrows():
            sig = r.get("posterior_sigma", np.nan)
            rows.append(
                {
                    "matrix_path": "iterative_reference_update",
                    "label": r["label"],
                    "posterior_sigma": sig,
                    "normalized_update": abs(r.get("applied_delta", np.nan)) / sig if pd.notna(sig) and sig else np.nan,
                    "posterior_error_over_sigma": abs(r.get("next_minus_truth", np.nan)) / sig if pd.notna(sig) and sig else np.nan,
                    "covariance_available": bool(matrices),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(corr_rows), pd.DataFrame(eig_rows)


def artifact_index(run_root: Path) -> pd.DataFrame:
    patterns = [
        "campaign_plan.json",
        "campaign_summary.json",
        "model_split/**/*.json",
        "noise/*.json",
        "trajectory/**/*",
        "cases/*/windows/window_*/*",
        "subblock_runs/**/study/schur_summary/subblock_summary.json",
        "subblock_runs/**/study/subprocess_diagnostics.json",
        "subblock_runs/**/study/subprocess.*.log",
    ]
    seen = set()
    rows = []
    for pattern in patterns:
        for p in sorted(run_root.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                rows.append({"kind": pattern, "path": rel(p, run_root), "size_bytes": p.stat().st_size})
    return pd.DataFrame(rows)


def image_comparisons(run_root: Path, outdir: Path, max_examples: int, mode: str, include_subblock_images: bool, include_frame_images: bool) -> list[str]:
    status_path = outdir / "representative_image_comparison_status.json"
    if plt is None or max_examples <= 0:
        status_path.write_text(json.dumps({"available": False, "reason": "plots disabled or matplotlib unavailable"}, indent=2))
        return []
    candidates = []
    patterns = ["*.png", "*.npy", "*.npz", "*.fits"]
    for base in sorted(run_root.glob("subblock_runs/*/window_*/subblock_*")):
        files = []
        for pat in patterns:
            files.extend(base.rglob(pat))
        if files:
            candidates.append((base, files))
    if not candidates:
        status_path.write_text(json.dumps({"available": False, "searched_patterns": patterns, "note": "No image comparison artifacts found. Consider enabling image snapshot output in subblock runner."}, indent=2))
        return []
    if mode == "first,median,last" and len(candidates) > 2:
        idxs = sorted(set([0, len(candidates) // 2, len(candidates) - 1]))[:max_examples]
        selected = [candidates[i] for i in idxs]
    else:
        selected = candidates[:max_examples]
    written = []
    for base, files in selected[:max_examples]:
        arrays = []
        labels = []
        for f in files:
            if f.suffix.lower() == ".npy":
                try:
                    arr = np.load(f)
                    arrays.append(np.squeeze(arr)[0] if arr.ndim > 2 else np.squeeze(arr))
                    labels.append(f.name)
                except Exception:
                    pass
            elif f.suffix.lower() == ".npz":
                try:
                    z = np.load(f)
                    for k in z.files[:2]:
                        arr = np.squeeze(z[k])
                        if arr.ndim >= 2:
                            arrays.append(arr[0] if arr.ndim > 2 else arr)
                            labels.append(f"{f.name}:{k}")
                except Exception:
                    pass
            elif f.suffix.lower() == ".fits":
                try:
                    from astropy.io import fits

                    arr = np.asarray(fits.getdata(f), dtype=float)
                    arr = np.squeeze(arr)
                    if arr.ndim >= 2:
                        arrays.append(arr[0] if arr.ndim > 2 else arr)
                        labels.append(f.name)
                except Exception:
                    pass
        if not arrays:
            continue
        fig, axes = plt.subplots(1, min(4, len(arrays)), figsize=(4 * min(4, len(arrays)), 4))
        axes = np.atleast_1d(axes)
        for ax, arr, label in zip(axes, arrays[:4], labels[:4]):
            ax.imshow(arr, origin="lower", cmap="viridis")
            ax.set_title(label, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        safe = "_".join(base.parts[-3:])
        out = outdir / "plots" / f"representative_image_comparison_{safe}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(rel(out, outdir))
    if not written:
        status_path.write_text(json.dumps({"available": False, "searched_patterns": patterns, "note": "No plottable image arrays found. FITS files were present but FITS plotting is not required by this lightweight review script."}, indent=2))
    else:
        status_path.write_text(json.dumps({"available": True, "plots": written}, indent=2))
    return written


def plot_outputs(
    outdir: Path,
    win: pd.DataFrame,
    param: pd.DataFrame,
    smear: pd.DataFrame,
    traj: pd.DataFrame,
    resid: pd.DataFrame,
    subblocks: pd.DataFrame,
    warnings_list: list[dict[str, Any]] | None = None,
) -> None:
    if plt is None:
        return
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    if not win.empty:
        line_plot(win, "window_index", ["reference_error_norm_before", "posterior_error_norm_after", "next_reference_error_norm"], plotdir / "iterative_error_norm.png", "Window", "error norm")
        line_plot(win, "window_index", ["update_cosine_with_ideal", "applied_vector_gain"], plotdir / "update_alignment_by_window.png", "Window", "alignment/gain")
        line_plot(win, "window_index", ["separation_reference_error_before_microas", "separation_posterior_error_after_microas", "separation_next_reference_error_microas"], plotdir / "separation_error_by_window.png", "Window", "separation error (microas)")
    if not param.empty:
        pivot_plot(param, "parameter_offsets_by_window.png", plotdir, value="next_offset", ylabel="next offset")
        pivot_plot(param, "posterior_sigma_by_parameter.png", plotdir, value="posterior_sigma", ylabel="posterior sigma", bar=True)
        z = param[param["label"].str.contains("zernike", na=False)]
        if not z.empty:
            pivot_plot(z, "zernike_m1_m2_offsets.png", plotdir, value="next_offset", ylabel="Zernike next offset (nm)")
    if not traj.empty:
        numeric_cols = [c for c in traj.columns if c.endswith("_mean")][:6]
        if numeric_cols:
            line_plot(traj, "subblock_index", numeric_cols, plotdir / "trajectory_xy_pa_timeseries.png", "Subblock", "mean value")
    if not resid.empty:
        numeric_cols = [c for c in resid.columns if c.endswith("_residual_rms")][:6]
        if numeric_cols:
            line_plot(resid, "subblock_index", numeric_cols, plotdir / "trajectory_residuals_by_frame.png", "Subblock", "residual RMS")
    if not smear.empty and "smear_length_pix" in smear.columns:
        line_plot(smear, "subblock_index", ["smear_length_pix"], plotdir / "smear_length_by_subblock.png", "Subblock", "smear length (pix)")
    if not subblocks.empty and "elapsed_seconds" in subblocks.columns:
        line_plot(subblocks, "global_subblock_index", ["elapsed_seconds"], plotdir / "subblock_convergence_summary.png", "Subblock", "elapsed seconds")


def line_plot(df: pd.DataFrame, x: str, cols: list[str], path: Path, xlabel: str, ylabel: str) -> None:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for col in cols:
        ax.plot(df[x], pd.to_numeric(df[col], errors="coerce"), marker="o", label=col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pivot_plot(df: pd.DataFrame, filename: str, plotdir: Path, value: str, ylabel: str, bar: bool = False) -> None:
    if value not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if bar:
        latest = df.sort_values("window_index").groupby("label").tail(1)
        ax.bar(latest["label"], pd.to_numeric(latest[value], errors="coerce"))
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    else:
        for label, ldf in df.groupby("label"):
            ax.plot(ldf["window_index"], pd.to_numeric(ldf[value], errors="coerce"), marker="o", label=label)
        ax.legend(fontsize=7, ncol=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plotdir / filename, dpi=150)
    plt.close(fig)


def forecast_tables(run_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = read_csv(run_root / "analysis/final_observation_summary.csv")
    forecast = read_csv(run_root / "analysis/projected_observation_forecast.csv")
    evolution = read_csv(run_root / "analysis/window_evolution_actual_and_projected.csv")
    return final, forecast, evolution


def plot_forecast_outputs(
    outdir: Path,
    final_forecast: pd.DataFrame,
    evolution: pd.DataFrame,
    warnings_list: list[dict[str, Any]] | None = None,
) -> None:
    if plt is None:
        return
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    if not final_forecast.empty:
        if "projected_final_separation_error_microas" in final_forecast.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            values = pd.to_numeric(final_forecast["projected_final_separation_error_microas"], errors="coerce")
            safe_hist(
                ax,
                values,
                requested_bins=12,
                min_bins=3,
                color="#4C78A8",
                alpha=0.8,
                warnings_list=warnings_list,
                context="projected_30min_separation_residual_distribution",
            )
            ax.axvline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel("Projected final separation error (microas)")
            ax.set_ylabel("Cases")
            fig.tight_layout()
            fig.savefig(plotdir / "projected_30min_separation_residual_distribution.png", dpi=150)
            plt.close(fig)
        if "projected_final_posterior_sigma_separation_microas" in final_forecast.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            values = pd.to_numeric(final_forecast["projected_final_posterior_sigma_separation_microas"], errors="coerce")
            safe_hist(
                ax,
                values,
                requested_bins=12,
                min_bins=3,
                color="#59A14F",
                alpha=0.8,
                warnings_list=warnings_list,
                context="projected_30min_posterior_sigma_distribution",
            )
            ax.set_xlabel("Projected final posterior sigma (microas)")
            ax.set_ylabel("Cases")
            fig.tight_layout()
            fig.savefig(plotdir / "projected_30min_posterior_sigma_distribution.png", dpi=150)
            plt.close(fig)
    if not evolution.empty and {"case_name", "window_index", "separation_error_microas"}.issubset(evolution.columns):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for case_name, group in evolution.groupby("case_name"):
            ordered = group.sort_values("window_index")
            ax.plot(
                pd.to_numeric(ordered["window_index"], errors="coerce") + 1,
                pd.to_numeric(ordered["separation_error_microas"], errors="coerce"),
                marker="o",
                linewidth=1.2,
                label=str(case_name),
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Window")
        ax.set_ylabel("Separation error (microas)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(plotdir / "actual_vs_projected_window_evolution.png", dpi=150)
        plt.close(fig)


def write_report(outdir: Path, run_root: Path, summary: dict[str, Any], win: pd.DataFrame, final: pd.DataFrame, policy: str, image_plots: list[str], final_forecast: pd.DataFrame) -> None:
    completed = summary.get("completed_subblocks", "")
    failed = summary.get("failed_subblocks", "")
    missing = summary.get("missing_output_rows", "")
    windows = len(win)
    moved = ""
    if not win.empty:
        first = win.iloc[0]["reference_error_norm_before"]
        last = win.iloc[-1]["next_reference_error_norm"]
        moved = "decreased" if pd.notna(first) and pd.notna(last) and last < first else "did not consistently decrease"
    improved_count = int(final.get("improved", pd.Series(dtype=bool)).sum()) if not final.empty else 0
    total_params = len(final)
    caveat = "This smoke uses only a small number of subblocks and frames per subblock; do not overinterpret parameter-level performance."
    local = (
        "This run used `truth_when_available` for phi reference, so local registration interpretation is diagnostic and may not fully exercise an unconstrained local registration solve."
        if "truth_when_available" in policy
        else f"This run used phi reference policy `{policy}`."
    )
    lines = [
        "# Full-fidelity binary iterative campaign review",
        "",
        "## Executive summary",
        f"- Execution status: {completed} completed subblocks, {failed} failed subblocks, {missing} missing output rows, {windows} iterative windows.",
        f"- Full vector error {moved} across the reviewed windows.",
        f"- Slow-state final summary: {improved_count}/{total_params} parameters improved in absolute offset.",
        f"- Major caveat: {caveat}",
        "",
        "## Campaign settings dashboard",
        "See [campaign_dashboard.csv](campaign_dashboard.csv) and [campaign_dashboard.json](campaign_dashboard.json). Key settings are tied to the actual campaign plan, model split, noise, trajectory, and subblock artifacts.",
        "",
        "## Execution health",
        "See [execution_status.csv](execution_status.csv), [subblock_status_summary.csv](subblock_status_summary.csv), and [subblock_metric_summary.csv](subblock_metric_summary.csv).",
        "",
        "## Iterative progress",
        "See [iterative_window_progress.csv](iterative_window_progress.csv) and plots [iterative_error_norm.png](plots/iterative_error_norm.png), [update_alignment_by_window.png](plots/update_alignment_by_window.png), and [separation_error_by_window.png](plots/separation_error_by_window.png).",
        "",
        "## Projected 30-minute observation forecast",
        "See [final_observation_summary.csv](final_observation_summary.csv), [projected_observation_forecast.csv](projected_observation_forecast.csv), and [window_evolution_actual_and_projected.csv](window_evolution_actual_and_projected.csv). These results are projected from the realized actual windows, not from a fully rendered 60-window observation.",
        "",
        "## Slow-state evolution",
        "See [slow_state_evolution.csv](slow_state_evolution.csv) and [slow_state_final_summary.csv](slow_state_final_summary.csv).",
        "",
        "## Parameter constraints and degeneracies",
        "See [iterative_parameter_progress.csv](iterative_parameter_progress.csv), [parameter_sloppiness_summary.csv](parameter_sloppiness_summary.csv), [correlation_top_pairs.csv](correlation_top_pairs.csv), and [eigenmode_summary.csv](eigenmode_summary.csv). Zernike behavior is plotted in [zernike_m1_m2_offsets.png](plots/zernike_m1_m2_offsets.png).",
        "",
        "## Subblock behavior",
        f"Subblock local solve interpretation: {local}",
        "",
        "## Truth/inference mismatches",
        "See [mismatch_dashboard.csv](mismatch_dashboard.csv).",
        "",
        "## Trajectory and smear",
        "See [trajectory_summary.csv](trajectory_summary.csv), [trajectory_residual_summary.csv](trajectory_residual_summary.csv), and [smear_summary_review.csv](smear_summary_review.csv), plus [trajectory_xy_pa_timeseries.png](plots/trajectory_xy_pa_timeseries.png), [trajectory_residuals_by_frame.png](plots/trajectory_residuals_by_frame.png), and [smear_length_by_subblock.png](plots/smear_length_by_subblock.png).",
        "",
        "## Representative image comparisons",
    ]
    if image_plots:
        lines += [f"- [{p}]({p})" for p in image_plots]
    else:
        lines.append("No representative image array comparisons were generated. See `representative_image_comparison_status.json`.")
    lines += [
        "",
        "## Recommended next step",
        "For this smoke, scale modestly with more frames per subblock, more subblocks per window, and another iterative window before drawing strong conclusions.",
        "",
        "## Artifact index",
        "See [artifact_index.csv](artifact_index.csv).",
        "",
        "## Review warnings",
        "Plotting and optional review warnings are recorded in [review_warnings.json](review_warnings.json). These warnings do not invalidate a campaign with complete required science artifacts.",
    ]
    (outdir / "review_summary.md").write_text("\n".join(lines) + "\n")


def write_index(outdir: Path) -> None:
    md = (outdir / "review_summary.md").read_text() if (outdir / "review_summary.md").exists() else ""
    body = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    (outdir / "index.html").write_text(f"<html><body><pre>{body}</pre></body></html>\n")


def write_review_warnings(outdir: Path, warnings_list: list[dict[str, Any]]) -> None:
    payload = {
        "warning_count": len(warnings_list),
        "warnings": warnings_list,
    }
    (outdir / "review_warnings.json").write_text(json.dumps(payload, indent=2))


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else run_root / "analysis" / "full_fidelity_review"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "plots").mkdir(exist_ok=True)
    missing = validate_required(run_root, args.strict)

    artifacts = {
        "campaign_plan": read_json(run_root / "campaign_plan.json", {}),
        "campaign_summary": read_json(run_root / "campaign_summary.json", {}),
        "model_split": read_json(run_root / "model_split/model_split.json", {}),
        "model_split_summary": read_json(run_root / "model_split/model_split_summary.json", {}),
        "noise_request": read_json(run_root / "noise/noise_request_normalized.json", {}),
        "noise_render": read_json(run_root / "noise/noise_render_provenance.json", {}),
        "noise_inference": read_json(run_root / "noise/noise_inference_provenance.json", {}),
    }

    win = iterative_window_progress(run_root)
    eigen_update_modes, eigen_update_summary, eigen_update_contributions = (
        eigen_update_tables(run_root)
    )
    param = iterative_parameter_progress(run_root)
    posterior = combine_window_tables(run_root, "posterior_by_label.csv")
    science = combine_window_tables(run_root, "science_summary.csv")
    subblock_metrics, local_policy = subblock_metric_summary(run_root)
    traj, resid, smear = trajectory_tables(run_root)
    slow_evo, slow_final = slow_state_tables(param)
    sloppy, corr, eig = sloppiness_tables(run_root, param)
    final_forecast, projected_forecast, forecast_evolution = forecast_tables(run_root)
    dashboard = campaign_dashboard(run_root, artifacts)
    mismatch = mismatch_dashboard(run_root, artifacts, local_policy)
    exec_status = execution_status(run_root, artifacts["campaign_summary"])
    status_summary = read_csv(run_root / "subblock_status_iterative.csv")
    if missing:
        exec_status = pd.concat([exec_status, pd.DataFrame([{"metric": "missing_required_artifacts_non_strict", "value": ";".join(missing)}])], ignore_index=True)

    outputs = {
        "campaign_dashboard.csv": dashboard,
        "execution_status.csv": exec_status,
        "iterative_window_progress.csv": win,
        "eigen_update_modes.csv": eigen_update_modes,
        "eigen_update_window_summary.csv": eigen_update_summary,
        "eigen_update_mode_contributions.csv": eigen_update_contributions,
        "iterative_parameter_progress.csv": param,
        "posterior_by_label_combined.csv": posterior,
        "science_summary_combined.csv": science,
        "subblock_status_summary.csv": status_summary,
        "subblock_metric_summary.csv": subblock_metrics,
        "mismatch_dashboard.csv": mismatch,
        "trajectory_summary.csv": traj,
        "trajectory_residual_summary.csv": resid,
        "smear_summary_review.csv": smear,
        "slow_state_evolution.csv": slow_evo,
        "slow_state_final_summary.csv": slow_final,
        "final_observation_summary.csv": final_forecast,
        "projected_observation_forecast.csv": projected_forecast,
        "window_evolution_actual_and_projected.csv": forecast_evolution,
        "parameter_sloppiness_summary.csv": sloppy,
        "correlation_top_pairs.csv": corr,
        "eigenmode_summary.csv": eig,
        "artifact_index.csv": artifact_index(run_root),
    }
    for name, df in outputs.items():
        write_csv(df, outdir / name)
    (outdir / "campaign_dashboard.json").write_text(json.dumps(dashboard.to_dict(orient="records"), indent=2))

    review_warnings: list[dict[str, Any]] = []
    if not args.no_plots:
        for context, func in (
            (
                "campaign_review_plots",
                lambda: plot_outputs(
                    outdir,
                    win,
                    param,
                    smear,
                    traj,
                    resid,
                    subblock_metrics,
                    review_warnings,
                ),
            ),
            (
                "campaign_forecast_plots",
                lambda: plot_forecast_outputs(
                    outdir,
                    final_forecast,
                    forecast_evolution,
                    review_warnings,
                ),
            ),
        ):
            try:
                func()
            except Exception as exc:
                _append_plot_warning(
                    review_warnings,
                    f"optional plot generation failed: {exc}",
                    context=context,
                )
    image_plots = []
    if not args.no_plots:
        try:
            image_plots = image_comparisons(run_root, outdir, args.max_image_examples, args.image_examples, args.include_subblock_images, args.include_frame_images)
        except Exception as exc:
            _append_plot_warning(
                review_warnings,
                f"optional image comparison generation failed: {exc}",
                context="representative_image_comparisons",
            )
    write_report(outdir, run_root, artifacts["campaign_summary"], win, slow_final, local_policy, image_plots, final_forecast)
    write_index(outdir)
    write_review_warnings(outdir, review_warnings)
    if args.open_report:
        webbrowser.open((outdir / "index.html").as_uri())
    print(f"Wrote full-fidelity review bundle to {outdir}")
    if review_warnings:
        print(f"Review completed with {len(review_warnings)} warning(s); see {outdir / 'review_warnings.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", required=True)
    p.add_argument("--outdir")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--image-examples", default="first,median,last")
    p.add_argument("--include-subblock-images", action="store_true")
    p.add_argument("--include-frame-images", action="store_true")
    p.add_argument("--max-image-examples", type=int, default=4)
    p.add_argument("--open-report", action="store_true")
    return p


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
