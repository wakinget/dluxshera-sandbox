#!/usr/bin/env python3
"""Aggregate full-fidelity per-root review bundles into family-level products."""

from __future__ import annotations

import argparse
import glob as globlib
import math
import os
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


COMMON_GROUP_COLUMNS = [
    "campaign_label",
    "target",
    "condition",
    "ho_ke_nm",
    "low_order_regime",
    "pixel_offsets_sigma_pix",
    "pixel_response_sigma_fractional",
    "detector_ke_family",
]

SUMMARY_COLUMNS_UAS = [
    "N",
    "mean_final_sep_err_uas",
    "std_final_sep_err_uas",
    "sem_final_sep_err_uas",
    "median_final_sep_err_uas",
    "mad_final_sep_err_uas",
    "mean_abs_final_sep_err_uas",
    "median_abs_final_sep_err_uas",
    "best_abs_final_sep_err_uas",
    "worst_abs_final_sep_err_uas",
    "mean_final_posterior_sigma_sep_uas",
    "median_final_posterior_sigma_sep_uas",
]

SLOW_SUMMARY_COLUMNS = [
    "N",
    "mean_final_err",
    "std_final_err",
    "sem_final_err",
    "median_final_err",
    "mad_final_err",
    "mean_abs_final_err",
    "median_abs_final_err",
    "best_abs_final_err",
    "worst_abs_final_err",
    "mean_final_posterior_sigma",
    "median_final_posterior_sigma",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-list", type=Path, action="append", default=[])
    parser.add_argument("--run-root", type=Path, action="append", default=[])
    parser.add_argument("--run-root-glob", action="append", default=[])
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--group-cols",
        default="",
        help="Comma-separated grouping columns. Defaults to inferred common science axes.",
    )
    parser.add_argument(
        "--include-pooled",
        action="store_true",
        help="Also write explicit pooled rows across inferred group axes.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--legend-mode",
        choices=["auto", "compact", "full", "none"],
        default="auto",
        help="Legend labels for evolution plots. Auto uses compact labels up to --max-legend-items, then suppresses legends.",
    )
    parser.add_argument(
        "--plot-slow-parameters",
        dest="plot_slow_parameters",
        action="store_true",
        default=True,
        help="Generate standard slow-parameter plots when source data exist.",
    )
    parser.add_argument(
        "--no-slow-parameter-plots",
        dest="plot_slow_parameters",
        action="store_false",
        help="Skip slow-parameter plots while still writing slow-parameter CSVs.",
    )
    parser.add_argument(
        "--max-legend-items",
        type=int,
        default=12,
        help="Maximum series count for legends when --legend-mode=auto.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_float(value: Any) -> float:
    if value in (None, ""):
        return np.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if math.isfinite(out) else np.nan


def slug(value: Any) -> str:
    text = str(value if value not in (None, "") else "pooled")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text or "pooled"


def parse_number_token(token: str) -> float:
    text = token.replace("p", ".")
    text = re.sub(r"(\d+)em(\d+)", r"\1e-\2", text)
    return safe_float(text)


def condition_from_name(name: str) -> str:
    m = re.search(r"m1_([0-9]+p[0-9]+)nm_m2_([0-9]+p[0-9]+)nm", name)
    if not m:
        return ""
    m1 = parse_number_token(m.group(1))
    m2 = parse_number_token(m.group(2))
    if not np.isfinite(m1) or not np.isfinite(m2):
        return ""
    return f"m1 {m1:g} nm, m2 {m2:g} nm"


def draw_index_from_name(name: str) -> Any:
    for pattern in (r"(?:^|[_-])draw[_-]?(\d+)", r"(?:^|[_-])d(\d{3})(?:[_-]|$)"):
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))
    return ""


def campaign_label_from_name(name: str) -> str:
    label = re.sub(r"([_-])draw[_-]?\d+.*$", "", name)
    label = re.sub(r"([_-])d\d{3}([_-].*)?$", "", label)
    return label


def target_from_name(name: str) -> str:
    targets = {
        "ALPHA_CEN": ["alpha_cen", "alphacen"],
        "61_CYG": ["61_cyg", "61cyg"],
        "70_OPH": ["70_oph", "70oph"],
        "36_OPH": ["36_oph", "36oph"],
        "XI_BOO": ["xi_boo", "xiboo"],
        "P_ERI": ["p_eri", "peri"],
        "HR_2667_2668": ["hr_2667_2668", "hr2667", "hr_2667"],
    }
    lower = name.lower()
    for label, tokens in targets.items():
        if any(token in lower for token in tokens):
            return label
    return ""


def infer_metadata(root: Path) -> dict[str, Any]:
    name = root.name if root.name != "full_fidelity_review" else root.parent.parent.name
    lower = name.lower()
    condition = condition_from_name(name)
    ho_ke_nm = ""
    for pattern in (r"hoche_([0-9p]+)nm", r"hoke_([0-9p]+)nm"):
        m = re.search(pattern, lower)
        if m:
            ho_ke_nm = parse_number_token(m.group(1))
            break
    pixel_offsets = ""
    m = re.search(r"pixelposke_([0-9]+em[0-9]+)pix", lower)
    if m:
        pixel_offsets = parse_number_token(m.group(1))
    pixel_response = ""
    for pattern in (r"pixelresponseke_([0-9]+em[0-9]+)", r"response_([0-9]+em[0-9]+)"):
        m = re.search(pattern, lower)
        if m:
            pixel_response = parse_number_token(m.group(1))
            break
    detector_family = ""
    if "pixelposke" in lower:
        detector_family = "pixel_position_ke"
    elif "pixelresponseke" in lower:
        detector_family = "pixel_response_ke"
    elif "detector_ke" in lower:
        detector_family = "combined_detector_ke"
        if pixel_offsets == "":
            pixel_offsets = 1e-3
        if pixel_response == "":
            pixel_response = 1e-3
    return {
        "run_root": str(root),
        "campaign_label": campaign_label_from_name(name),
        "target": target_from_name(name),
        "condition": condition,
        "draw_index": draw_index_from_name(name),
        "ho_ke_nm": ho_ke_nm,
        "low_order_regime": condition,
        "pixel_offsets_sigma_pix": pixel_offsets,
        "pixel_response_sigma_fractional": pixel_response,
        "detector_ke_family": detector_family,
    }


def review_dir_for_root(root: Path) -> Path:
    if root.name == "full_fidelity_review":
        return root
    candidate = root / "analysis" / "full_fidelity_review"
    if candidate.exists():
        return candidate
    return root


def roots_from_args(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    for path in args.root_list:
        if not path.exists():
            raise FileNotFoundError(f"root-list path does not exist: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                roots.append(Path(text).expanduser())
    roots.extend(Path(p).expanduser() for p in args.run_root)
    for pattern in args.run_root_glob:
        roots.extend(Path(p).expanduser() for p in sorted(globlib.glob(pattern)))
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def separation_from_legacy_progress(review_dir: Path) -> pd.DataFrame:
    win = read_csv(review_dir / "iterative_window_progress.csv")
    if win.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for case_name, group in win.sort_values(["case_name", "window_index"], na_position="last").groupby("case_name", dropna=False):
        group = group.sort_values("window_index")
        first = group.iloc[0]
        last = group.iloc[-1]
        initial = safe_float(first.get("separation_reference_error_before_microas"))
        final = safe_float(last.get("separation_next_reference_error_microas"))
        sigma = safe_float(last.get("posterior_sigma_separation_microas"))
        rows.append(
            {
                "case_name": case_name,
                "initial_sep_err_uas": initial,
                "final_sep_err_uas": final,
                "initial_abs_sep_err_uas": abs(initial) if np.isfinite(initial) else np.nan,
                "final_abs_sep_err_uas": abs(final) if np.isfinite(final) else np.nan,
                "final_posterior_sigma_sep_uas": sigma,
                "signed_improvement_uas": initial - final if np.isfinite(initial) and np.isfinite(final) else np.nan,
                "abs_improvement_uas": abs(initial) - abs(final) if np.isfinite(initial) and np.isfinite(final) else np.nan,
                "final_sep_err_over_sigma": final / sigma if np.isfinite(final) and np.isfinite(sigma) and sigma else np.nan,
                "final_abs_sep_err_over_sigma": abs(final) / sigma if np.isfinite(final) and np.isfinite(sigma) and sigma else np.nan,
            }
        )
    return pd.DataFrame(rows)


def classify_missing(root: Path) -> str:
    lower = str(root).lower()
    if "detector" in lower and any(token in lower for token in ("dry", "smoke", "concurrency")):
        return "missing_non_science_dry_run_or_smoke_candidate"
    return "missing_review_bundle"


def load_separation_rows(root: Path) -> tuple[pd.DataFrame, str]:
    if not root.exists():
        return pd.DataFrame(), classify_missing(root)
    review_dir = review_dir_for_root(root)
    df = read_csv(review_dir / "separation_error_summary.csv")
    if df.empty:
        df = separation_from_legacy_progress(review_dir)
    if df.empty:
        return df, "missing_separation_metrics"
    return df, "ok"


def load_slow_rows(root: Path) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    review_dir = review_dir_for_root(root)
    df = read_csv(review_dir / "slow_parameter_error_summary.csv")
    if not df.empty:
        return df
    progress = read_csv(review_dir / "iterative_parameter_progress.csv")
    if not progress.empty:
        rows: list[dict[str, Any]] = []
        ordered = progress.sort_values(["case_name", "label", "window_index"], na_position="last")
        for (case_name, label), group in ordered.groupby(["case_name", "label"], dropna=False):
            group = group.sort_values("window_index")
            first = group.iloc[0]
            last = group.iloc[-1]
            label = str(label)
            truth = safe_float(last.get("truth_value", first.get("truth_value", np.nan)))
            unit, scale, value_offset = standard_parameter_scale(label, truth)

            def convert_value(value: Any) -> float:
                f = safe_float(value)
                return (f - value_offset) * scale if np.isfinite(f) else np.nan

            def convert_offset(value: Any) -> float:
                f = safe_float(value)
                return f * scale if np.isfinite(f) else np.nan

            initial_err = convert_offset(first.get("current_offset", np.nan))
            final_err = convert_offset(last.get("next_offset", last.get("posterior_offset", np.nan)))
            sigma = convert_offset(last.get("posterior_sigma", np.nan))
            initial_abs = abs(initial_err) if np.isfinite(initial_err) else np.nan
            final_abs = abs(final_err) if np.isfinite(final_err) else np.nan
            rows.append(
                {
                    "case_name": case_name,
                    "parameter_label": label,
                    "parameter_group": standard_parameter_group(label),
                    "unit": unit,
                    "initial_value": convert_value(first.get("current_value", np.nan)),
                    "truth_value": convert_value(truth),
                    "final_value": convert_value(last.get("next_value", last.get("posterior_value", np.nan))),
                    "initial_err": initial_err,
                    "final_err": final_err,
                    "initial_abs_err": initial_abs,
                    "final_abs_err": final_abs,
                    "abs_improvement": initial_abs - final_abs if np.isfinite(initial_abs) and np.isfinite(final_abs) else np.nan,
                    "fractional_abs_improvement": (initial_abs - final_abs) / initial_abs if np.isfinite(initial_abs) and initial_abs > 0 and np.isfinite(final_abs) else np.nan,
                    "final_posterior_sigma": sigma,
                    "final_err_over_sigma": final_err / sigma if np.isfinite(final_err) and np.isfinite(sigma) and sigma else np.nan,
                    "final_abs_err_over_sigma": final_abs / sigma if np.isfinite(final_abs) and np.isfinite(sigma) and sigma else np.nan,
                }
            )
        return pd.DataFrame(rows)
    legacy = read_csv(review_dir / "slow_state_final_summary.csv")
    if legacy.empty:
        return pd.DataFrame()
    rows = []
    for _, r in legacy.iterrows():
        final = safe_float(r.get("final_offset"))
        initial = safe_float(r.get("initial_offset"))
        sigma = safe_float(r.get("final_posterior_sigma", r.get("sigma_if_available", np.nan)))
        rows.append(
            {
                "case_name": r.get("case_name", ""),
                "parameter_label": r.get("parameter_label", ""),
                "parameter_group": r.get("group", ""),
                "unit": "",
                "initial_err": initial,
                "final_err": final,
                "initial_abs_err": abs(initial) if np.isfinite(initial) else np.nan,
                "final_abs_err": abs(final) if np.isfinite(final) else np.nan,
                "final_posterior_sigma": sigma,
                "final_err_over_sigma": final / sigma if np.isfinite(final) and np.isfinite(sigma) and sigma else np.nan,
                "final_abs_err_over_sigma": abs(final) / sigma if np.isfinite(final) and np.isfinite(sigma) and sigma else np.nan,
            }
        )
    return pd.DataFrame(rows)


def standard_parameter_group(label: str) -> str:
    if label.startswith("source."):
        return "source"
    if label == "optics.plate_scale_as_per_pix":
        return "plate_scale"
    if "primary.zernike_coeffs_nm" in label:
        return "m1_zernike"
    if "secondary.zernike_coeffs_nm" in label:
        return "m2_zernike"
    return "other"


def standard_parameter_scale(label: str, truth_value: float) -> tuple[str, float, float]:
    if label in {"source.separation_as", "source.x_position_as", "source.y_position_as", "source.x_as", "source.y_as"}:
        return "uas", 1e6, 0.0
    if label == "source.position_angle_deg":
        return "deg", 1.0, 0.0
    if label == "source.log_flux_total":
        return "dex", 1.0, 0.0
    if label == "source.contrast":
        return "dimensionless", 1.0, 0.0
    if label == "optics.plate_scale_as_per_pix":
        if np.isfinite(truth_value) and truth_value != 0:
            return "ppm", 1e6 / truth_value, truth_value
        return "ppm", 1.0, 0.0
    if "zernike_coeffs_nm" in label:
        return "nm", 1.0, 0.0
    return "", 1.0, 0.0


def load_slow_evolution_rows(root: Path) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    review_dir = review_dir_for_root(root)
    progress = read_csv(review_dir / "iterative_parameter_progress.csv")
    if not progress.empty:
        rows: list[dict[str, Any]] = []
        for _, r in progress.iterrows():
            label = str(r.get("label", r.get("parameter_label", "")))
            truth = safe_float(r.get("truth_value", np.nan))
            unit, scale, value_offset = standard_parameter_scale(label, truth)
            window_index = safe_float(r.get("window_index", np.nan))
            current = safe_float(r.get("current_offset", np.nan))
            next_offset = safe_float(r.get("next_offset", r.get("posterior_offset", np.nan)))
            sigma = safe_float(r.get("posterior_sigma", np.nan))
            if np.isfinite(window_index) and np.isfinite(current):
                rows.append(
                    {
                        "case_name": r.get("case_name", ""),
                        "evolution_step": window_index,
                        "parameter_label": label,
                        "parameter_group": standard_parameter_group(label),
                        "unit": unit,
                        "err": current * scale,
                        "posterior_sigma": sigma * scale if np.isfinite(sigma) else np.nan,
                    }
                )
            if np.isfinite(window_index) and np.isfinite(next_offset):
                rows.append(
                    {
                        "case_name": r.get("case_name", ""),
                        "evolution_step": window_index + 1.0,
                        "parameter_label": label,
                        "parameter_group": standard_parameter_group(label),
                        "unit": unit,
                        "err": next_offset * scale,
                        "posterior_sigma": sigma * scale if np.isfinite(sigma) else np.nan,
                    }
                )
        if rows:
            return pd.DataFrame(rows).drop_duplicates(
                ["case_name", "evolution_step", "parameter_label"], keep="last"
            )
    evo = read_csv(review_dir / "slow_state_evolution.csv")
    if evo.empty:
        return pd.DataFrame()
    rows = []
    state_step = {"initial_reference": 0.0}
    pattern = re.compile(r"(?:posterior|next_reference|final_reference)_window_(\d+)")
    for _, r in evo.iterrows():
        state = str(r.get("state", ""))
        if state in state_step:
            step = state_step[state]
        else:
            m = pattern.search(state)
            if not m:
                continue
            step = float(int(m.group(1)) + 1)
        label = str(r.get("parameter_label", ""))
        truth = safe_float(r.get("truth", np.nan))
        unit, scale, _ = standard_parameter_scale(label, truth)
        err = safe_float(r.get("offset", np.nan))
        sigma = safe_float(r.get("sigma_if_available", np.nan))
        rows.append(
            {
                "case_name": r.get("case_name", ""),
                "evolution_step": step,
                "parameter_label": label,
                "parameter_group": standard_parameter_group(label),
                "unit": unit,
                "err": err * scale if np.isfinite(err) else np.nan,
                "posterior_sigma": sigma * scale if np.isfinite(sigma) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_evolution_rows(root: Path) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    review_dir = review_dir_for_root(root)
    win = read_csv(review_dir / "iterative_window_progress.csv")
    if win.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for case_name, group in win.sort_values(["case_name", "window_index"], na_position="last").groupby("case_name", dropna=False):
        group = group.sort_values("window_index")
        if group.empty:
            continue
        initial = safe_float(group.iloc[0].get("separation_reference_error_before_microas"))
        rows.append({"case_name": case_name, "evolution_step": 0.0, "sep_err_uas": initial})
        for _, r in group.iterrows():
            window = safe_float(r.get("window_index"))
            err = safe_float(r.get("separation_next_reference_error_microas"))
            rows.append(
                {
                    "case_name": case_name,
                    "evolution_step": window + 1.0 if np.isfinite(window) else np.nan,
                    "sep_err_uas": err,
                }
            )
    return pd.DataFrame(rows)


def numeric_summary(values: pd.Series, sigma: pd.Series | None = None, prefix: str = "") -> dict[str, Any]:
    signed = pd.to_numeric(values, errors="coerce").dropna()
    absolute = signed.abs()
    n = int(len(signed))
    median = float(signed.median()) if n else np.nan
    std = float(signed.std(ddof=1)) if n > 1 else np.nan
    sem = std / math.sqrt(n) if n > 1 else np.nan
    out = {
        "N": n,
        f"mean_{prefix}final_err": float(signed.mean()) if n else np.nan,
        f"std_{prefix}final_err": std,
        f"sem_{prefix}final_err": sem,
        f"median_{prefix}final_err": median,
        f"mad_{prefix}final_err": float((signed - median).abs().median()) if n else np.nan,
        f"mean_abs_{prefix}final_err": float(absolute.mean()) if n else np.nan,
        f"median_abs_{prefix}final_err": float(absolute.median()) if n else np.nan,
        f"best_abs_{prefix}final_err": float(absolute.min()) if n else np.nan,
        f"worst_abs_{prefix}final_err": float(absolute.max()) if n else np.nan,
    }
    if sigma is not None:
        sig = pd.to_numeric(sigma, errors="coerce").dropna()
        out[f"mean_{prefix}final_posterior_sigma"] = float(sig.mean()) if len(sig) else np.nan
        out[f"median_{prefix}final_posterior_sigma"] = float(sig.median()) if len(sig) else np.nan
    return out


def separation_group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + SUMMARY_COLUMNS_UAS)
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        stats = numeric_summary(
            group["final_sep_err_uas"],
            group.get("final_posterior_sigma_sep_uas"),
            prefix="sep_err_uas_",
        )
        row.update(
            {
                "N": stats["N"],
                "mean_final_sep_err_uas": stats["mean_sep_err_uas_final_err"],
                "std_final_sep_err_uas": stats["std_sep_err_uas_final_err"],
                "sem_final_sep_err_uas": stats["sem_sep_err_uas_final_err"],
                "median_final_sep_err_uas": stats["median_sep_err_uas_final_err"],
                "mad_final_sep_err_uas": stats["mad_sep_err_uas_final_err"],
                "mean_abs_final_sep_err_uas": stats["mean_abs_sep_err_uas_final_err"],
                "median_abs_final_sep_err_uas": stats["median_abs_sep_err_uas_final_err"],
                "best_abs_final_sep_err_uas": stats["best_abs_sep_err_uas_final_err"],
                "worst_abs_final_sep_err_uas": stats["worst_abs_sep_err_uas_final_err"],
                "mean_final_posterior_sigma_sep_uas": stats["mean_sep_err_uas_final_posterior_sigma"],
                "median_final_posterior_sigma_sep_uas": stats["median_sep_err_uas_final_posterior_sigma"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=group_cols + SUMMARY_COLUMNS_UAS)


def slow_group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = group_cols + ["parameter_label", "parameter_group", "unit"]
    if df.empty:
        return pd.DataFrame(columns=cols + SLOW_SUMMARY_COLUMNS)
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(cols, dropna=False):
        row = dict(zip(cols, key if isinstance(key, tuple) else (key,)))
        stats = numeric_summary(group["final_err"], group.get("final_posterior_sigma"))
        row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols + SLOW_SUMMARY_COLUMNS)


def infer_group_cols(df: pd.DataFrame, explicit: str) -> list[str]:
    if explicit.strip():
        return [c.strip() for c in explicit.split(",") if c.strip()]
    cols = []
    for col in COMMON_GROUP_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip()
        if len(values[values != ""]):
            cols.append(col)
    return cols or ["campaign_label"]


def add_pooled_rows(summary: pd.DataFrame, by_root: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not group_cols or by_root.empty:
        return summary
    pooled = separation_group_summary(by_root.assign(pooled_summary="pooled"), ["pooled_summary"])
    for col in group_cols:
        pooled[col] = "pooled"
    pooled = pooled[group_cols + SUMMARY_COLUMNS_UAS]
    return pd.concat([summary, pooled], ignore_index=True)


def apply_standard_grid(ax: Any) -> None:
    ax.grid(True, which="major", alpha=0.30)
    ax.grid(True, which="minor", alpha=0.15)


def make_group_labels(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if not group_cols:
        return pd.Series(["all"] * len(df), index=df.index)
    existing = [col for col in group_cols if col in df.columns]
    if not existing:
        return pd.Series(["all"] * len(df), index=df.index)
    return df[existing].apply(lambda row: " | ".join(str(row[col]) for col in existing), axis=1)


def group_display_label(group: pd.DataFrame, group_cols: list[str]) -> str:
    if not group_cols:
        return "all"
    parts = []
    first = group.iloc[0]
    for col in group_cols:
        value = first.get(col, "")
        if pd.isna(value) or str(value) == "":
            continue
        parts.append(f"{col}={value}")
    return ", ".join(parts) or "all"


def set_wrapped_title(ax: Any, title: str, context: str = "") -> None:
    lines = [title]
    if context:
        lines.append(textwrap.fill(context, width=100))
    ax.set_title("\n".join(lines), fontsize=10)


def compact_series_label(line: pd.DataFrame) -> str:
    draw = line.get("draw_index", pd.Series(dtype=object)).dropna()
    if len(draw):
        value = safe_float(draw.iloc[0])
        if np.isfinite(value):
            return f"d{int(value):03d}"
    case = str(line.get("case_name", pd.Series([""])).iloc[0])
    draw = draw_index_from_name(case)
    if draw != "":
        return f"d{int(draw):03d}"
    return f"d{abs(hash((case, str(line.get('run_root', pd.Series([''])).iloc[0])))) % 1000:03d}"


def full_series_label(line: pd.DataFrame) -> str:
    run_root = str(line.get("run_root", pd.Series([""])).iloc[0])
    case_name = str(line.get("case_name", pd.Series([""])).iloc[0])
    return f"{Path(run_root).name}:{case_name}" if run_root else case_name


def legend_for_mode(mode: str, series_count: int, max_legend_items: int) -> bool:
    if mode == "none":
        return False
    if mode == "auto":
        return series_count <= max_legend_items
    return True


def series_plot_label(line: pd.DataFrame, mode: str) -> str:
    return full_series_label(line) if mode == "full" else compact_series_label(line)


def add_series_label_map_row(
    rows: list[dict[str, Any]],
    plot_filename: str,
    plot_label: str,
    line: pd.DataFrame,
    group_key: str,
) -> None:
    first = line.iloc[0]
    rows.append(
        {
            "plot_filename": plot_filename,
            "plot_label": plot_label,
            "run_name": Path(str(first.get("run_root", ""))).name,
            "case_name": first.get("case_name", ""),
            "run_root": first.get("run_root", ""),
            "draw_index": first.get("draw_index", ""),
            "group_key": group_key,
        }
    )


def finish_legend_and_save(
    fig: Any,
    ax: Any,
    save_path: Path,
    show_legend: bool,
    *,
    legend_outside: bool = False,
) -> None:
    if show_legend:
        if legend_outside:
            ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.legend(fontsize=7, ncol=2)
    try:
        fig.tight_layout()
    except UserWarning:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def zernike_index(label: Any) -> int:
    m = re.search(r"\[(\d+)\]", str(label))
    return int(m.group(1)) if m else -1


def symlog_linthresh(unit: str, parameter_label: str = "") -> float:
    if unit == "uas":
        return 1.0
    if unit == "deg":
        return 1e-6
    if unit == "dex":
        return 1e-6
    if unit == "dimensionless":
        return 1e-6
    if unit == "ppm":
        return 0.1
    if unit == "nm" or "zernike_coeffs_nm" in parameter_label:
        return 0.01
    return 1e-3


def plot_distribution_by_group(
    df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    save_path: Path,
    group_cols: list[str],
    *,
    fig: Any | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.8))
    if df.empty:
        return fig, ax
    work = df.copy()
    work["_group_label"] = make_group_labels(work, group_cols)
    groups = [(str(label), pd.to_numeric(group[value_col], errors="coerce").dropna()) for label, group in work.groupby("_group_label", dropna=False)]
    groups = [(label, vals) for label, vals in groups if len(vals)]
    if not groups:
        return fig, ax
    try:
        ax.boxplot([vals.to_numpy() for _, vals in groups], tick_labels=[label for label, _ in groups], showmeans=True)
    except TypeError:
        ax.boxplot([vals.to_numpy() for _, vals in groups], labels=[label for label, _ in groups], showmeans=True)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    apply_standard_grid(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_family_outputs(
    outdir: Path,
    by_root: pd.DataFrame,
    evolution: pd.DataFrame,
    group_cols: list[str],
    *,
    slow_by_root: pd.DataFrame | None = None,
    slow_evolution: pd.DataFrame | None = None,
    plot_slow_parameters: bool = True,
    legend_mode: str = "auto",
    max_legend_items: int = 12,
) -> None:
    if plt is None or by_root.empty:
        return
    plotdir = outdir / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    label_map_rows: list[dict[str, Any]] = []
    by_root = by_root.copy()
    by_root["_group_label"] = make_group_labels(by_root, group_cols)

    for col, filename, xlabel in (
        ("final_sep_err_uas", "final_signed_sep_err_histograms.png", "Signed final separation error (uas)"),
        ("final_abs_sep_err_uas", "final_abs_sep_err_histograms.png", "Absolute final separation error (uas)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for label, group in by_root.groupby("_group_label", dropna=False):
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            if len(vals):
                ax.hist(vals, bins=min(20, max(1, len(vals))), alpha=0.45, label=str(label))
        if col == "final_sep_err_uas":
            ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Runs")
        apply_standard_grid(ax)
        if by_root["_group_label"].nunique() <= 8:
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(plotdir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    for col, filename, ylabel in (
        ("final_sep_err_uas", "final_signed_sep_err_boxplot.png", "Signed final separation error (uas)"),
        ("final_abs_sep_err_uas", "final_abs_sep_err_boxplot.png", "Absolute final separation error (uas)"),
    ):
        groups = [(str(label), pd.to_numeric(group[col], errors="coerce").dropna()) for label, group in by_root.groupby("_group_label", dropna=False)]
        groups = [(label, vals) for label, vals in groups if len(vals)]
        if groups:
            fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(groups)), 4.8))
            values = [vals.to_numpy() for _, vals in groups]
            labels = [label for label, _ in groups]
            try:
                ax.boxplot(values, tick_labels=labels, showmeans=True)
            except TypeError:
                ax.boxplot(values, labels=labels, showmeans=True)
            if col == "final_sep_err_uas":
                ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            apply_standard_grid(ax)
            fig.tight_layout()
            fig.savefig(plotdir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

    if {"initial_abs_sep_err_uas", "final_abs_sep_err_uas"}.issubset(by_root.columns):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(by_root["initial_abs_sep_err_uas"], by_root["final_abs_sep_err_uas"], s=30)
        ax.set_xlabel("Initial absolute separation error (uas)")
        ax.set_ylabel("Final absolute separation error (uas)")
        apply_standard_grid(ax)
        fig.tight_layout()
        fig.savefig(plotdir / "initial_vs_final_abs_sep_err.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if {"final_abs_sep_err_uas", "final_posterior_sigma_sep_uas"}.issubset(by_root.columns):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(by_root["final_posterior_sigma_sep_uas"], by_root["final_abs_sep_err_uas"], s=30)
        ax.set_xlabel("Final posterior sigma separation (uas)")
        ax.set_ylabel("Final absolute separation error (uas)")
        apply_standard_grid(ax)
        fig.tight_layout()
        fig.savefig(plotdir / "final_abs_sep_err_vs_posterior_sigma.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if not evolution.empty and {"evolution_step", "sep_err_uas"}.issubset(evolution.columns):
        evolution = evolution.copy()
        evolution["_group_label"] = make_group_labels(evolution, group_cols)
        for group_label, group in evolution.groupby("_group_label", dropna=False):
            group_slug = slug(group_label)
            context = group_display_label(group, group_cols)
            series_count = group[["run_root", "case_name"]].drop_duplicates().shape[0]
            show_legend = legend_for_mode(legend_mode, series_count, max_legend_items)
            fig, ax = plt.subplots(figsize=(8, 4.5))
            any_values = False
            filename = f"separation_error_evolution_abs_log_{group_slug}.png"
            for (run_root, case_name), line in group.groupby(["run_root", "case_name"], dropna=False):
                line = line.sort_values("evolution_step")
                y = pd.to_numeric(line["sep_err_uas"], errors="coerce").abs()
                y = y.where(y > 0)
                if y.notna().any():
                    any_values = True
                label = series_plot_label(line, legend_mode)
                add_series_label_map_row(label_map_rows, filename, label, line, str(group_label))
                ax.plot(pd.to_numeric(line["evolution_step"], errors="coerce"), y, marker="o", linewidth=1.0, label=label)
            if any_values:
                ax.set_yscale("log")
            ax.set_xlabel("Window step")
            ax.set_ylabel("Absolute separation error (uas)")
            set_wrapped_title(ax, "Absolute separation error evolution", context)
            apply_standard_grid(ax)
            finish_legend_and_save(fig, ax, plotdir / filename, show_legend, legend_outside=series_count > 6)

            fig, ax = plt.subplots(figsize=(8, 4.5))
            filename = f"separation_error_evolution_signed_symlog_{group_slug}.png"
            for (run_root, case_name), line in group.groupby(["run_root", "case_name"], dropna=False):
                line = line.sort_values("evolution_step")
                label = series_plot_label(line, legend_mode)
                add_series_label_map_row(label_map_rows, filename, label, line, str(group_label))
                ax.plot(
                    pd.to_numeric(line["evolution_step"], errors="coerce"),
                    pd.to_numeric(line["sep_err_uas"], errors="coerce"),
                    marker="o",
                    linewidth=1.0,
                    label=label,
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_xlabel("Window step")
            ax.set_ylabel("Signed separation error (uas)")
            set_wrapped_title(ax, "Signed separation error evolution", context)
            apply_standard_grid(ax)
            finish_legend_and_save(fig, ax, plotdir / filename, show_legend, legend_outside=series_count > 6)

    if plot_slow_parameters and slow_by_root is not None and not slow_by_root.empty:
        plot_slow_parameter_outputs(
            plotdir,
            slow_by_root,
            slow_evolution if slow_evolution is not None else pd.DataFrame(),
            group_cols,
            legend_mode=legend_mode,
            max_legend_items=max_legend_items,
            label_map_rows=label_map_rows,
        )

    if label_map_rows:
        pd.DataFrame(label_map_rows).to_csv(outdir / "plot_series_label_map.csv", index=False)


def add_group_label(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["_group_label"] = make_group_labels(out, group_cols)
    return out


def plot_slow_summary_boxplots(
    slow_by_root: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    ylabel: str,
    save_path: Path,
) -> tuple[Any, Any]:
    families = []
    for (param_group, unit), group in slow_by_root.groupby(["parameter_group", "unit"], dropna=False):
        vals = pd.to_numeric(group[value_col], errors="coerce")
        if vals.notna().any():
            families.append((str(param_group), str(unit), group.copy()))
    n = len(families)
    fig, axes = plt.subplots(max(1, n), 1, figsize=(9, max(4.2, 2.8 * max(1, n))), squeeze=False)
    if not families:
        return fig, axes
    for ax, (param_group, unit, group) in zip(axes[:, 0], families):
        work = add_group_label(group, group_cols)
        groups = [
            (str(label), pd.to_numeric(g[value_col], errors="coerce").dropna())
            for label, g in work.groupby("_group_label", dropna=False)
        ]
        groups = [(label, vals) for label, vals in groups if len(vals)]
        if groups:
            try:
                ax.boxplot(
                    [vals.to_numpy() for _, vals in groups],
                    tick_labels=[label for label, _ in groups],
                    showmeans=True,
                )
            except TypeError:
                ax.boxplot(
                    [vals.to_numpy() for _, vals in groups],
                    labels=[label for label, _ in groups],
                    showmeans=True,
                )
        if "signed" in ylabel.lower() or "over sigma" in ylabel.lower():
            ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel if not unit or unit == "nan" else f"{ylabel} ({unit})")
        ax.set_title(f"{param_group} / {unit}")
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        apply_standard_grid(ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig, axes


def plot_source_slow_evolution(
    plotdir: Path,
    group: pd.DataFrame,
    group_slug: str,
    group_key: str,
    group_cols: list[str],
    legend_mode: str,
    max_legend_items: int,
    label_map_rows: list[dict[str, Any]],
) -> None:
    source = group[group["parameter_group"].eq("source")].copy()
    if source.empty:
        return
    families = []
    for (label, unit), unit_group in source.groupby(["parameter_label", "unit"], dropna=False):
        if pd.to_numeric(unit_group["err"], errors="coerce").notna().any():
            families.append((str(label), str(unit), unit_group.copy()))
    if not families:
        return
    fig, axes = plt.subplots(len(families), 1, figsize=(8, max(4.0, 2.5 * len(families))), squeeze=False)
    filename = f"slow_parameter_evolution_source_signed_symlog_{group_slug}.png"
    series_count = source[["run_root", "case_name"]].drop_duplicates().shape[0]
    show_legend = legend_for_mode(legend_mode, series_count, max_legend_items)
    for ax, (label, unit, unit_group) in zip(axes[:, 0], families):
        for _, line in unit_group.groupby(["run_root", "case_name"], dropna=False):
            line = line.sort_values("evolution_step")
            plot_label = series_plot_label(line, legend_mode)
            add_series_label_map_row(label_map_rows, filename, plot_label, line, group_key)
            ax.plot(
                pd.to_numeric(line["evolution_step"], errors="coerce"),
                pd.to_numeric(line["err"], errors="coerce"),
                marker="o",
                linewidth=1.0,
                label=plot_label,
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_yscale("symlog", linthresh=symlog_linthresh(unit, label))
        ax.set_ylabel(f"{label}\n({unit})")
        apply_standard_grid(ax)
    axes[-1, 0].set_xlabel("Window step")
    set_wrapped_title(axes[0, 0], "Source slow-parameter signed error evolution", group_display_label(group, group_cols))
    finish_legend_and_save(fig, axes[0, 0], plotdir / filename, show_legend, legend_outside=series_count > 6)


def plot_zernike_final_error(
    plotdir: Path,
    group: pd.DataFrame,
    mirror_group: str,
    group_slug: str,
    group_key: str,
    group_cols: list[str],
    legend_mode: str,
    max_legend_items: int,
    label_map_rows: list[dict[str, Any]],
) -> None:
    z = group[group["parameter_group"].eq(mirror_group)].copy()
    if z.empty:
        return
    z["_zernike_index"] = z["parameter_label"].map(zernike_index)
    z = z.sort_values(["run_root", "case_name", "_zernike_index"])
    mirror = "m1" if mirror_group == "m1_zernike" else "m2"
    filename = f"zernike_final_error_{mirror}_{group_slug}.png"
    series_count = z[["run_root", "case_name"]].drop_duplicates().shape[0]
    show_legend = legend_for_mode(legend_mode, series_count, max_legend_items)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for _, line in z.groupby(["run_root", "case_name"], dropna=False):
        line = line.sort_values("_zernike_index")
        label = series_plot_label(line, legend_mode)
        add_series_label_map_row(label_map_rows, filename, label, line, group_key)
        ax.plot(
            pd.to_numeric(line["_zernike_index"], errors="coerce"),
            pd.to_numeric(line["final_err"], errors="coerce"),
            marker="o",
            linewidth=1.0,
            alpha=0.75,
            label=label,
        )
    mean = z.groupby("_zernike_index", dropna=False)["final_err"].mean(numeric_only=True)
    if len(mean):
        ax.plot(mean.index, mean.values, color="black", linewidth=2.0, marker="s", label="mean")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Zernike index")
    ax.set_ylabel("Final signed Zernike error (nm)")
    set_wrapped_title(ax, f"{mirror.upper()} final Zernike error", group_display_label(group, group_cols))
    apply_standard_grid(ax)
    finish_legend_and_save(fig, ax, plotdir / filename, show_legend, legend_outside=series_count > 6)


def plot_zernike_rms_evolution(
    plotdir: Path,
    group: pd.DataFrame,
    group_slug: str,
    group_key: str,
    group_cols: list[str],
    legend_mode: str,
    max_legend_items: int,
    label_map_rows: list[dict[str, Any]],
) -> None:
    z = group[group["parameter_group"].isin(["m1_zernike", "m2_zernike"])].copy()
    if z.empty:
        return
    rms = (
        z.assign(abs_sq=pd.to_numeric(z["err"], errors="coerce") ** 2)
        .groupby(["parameter_group", "run_root", "case_name", "evolution_step"], dropna=False)["abs_sq"]
        .mean()
        .reset_index()
    )
    rms["rms_err_nm"] = np.sqrt(rms["abs_sq"])
    filename = f"zernike_rms_error_evolution_{group_slug}.png"
    series_count = rms[["run_root", "case_name"]].drop_duplicates().shape[0]
    show_legend = legend_for_mode(legend_mode, series_count, max_legend_items)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (mirror_group, run_root, case_name), line in rms.groupby(["parameter_group", "run_root", "case_name"], dropna=False):
        line = line.sort_values("evolution_step")
        line_for_label = group[(group["run_root"].eq(run_root)) & (group["case_name"].eq(case_name))]
        prefix = "M1" if mirror_group == "m1_zernike" else "M2"
        label = f"{prefix} {series_plot_label(line_for_label if not line_for_label.empty else line, legend_mode)}"
        add_series_label_map_row(label_map_rows, filename, label, line_for_label if not line_for_label.empty else line, group_key)
        y = pd.to_numeric(line["rms_err_nm"], errors="coerce").where(lambda s: s > 0)
        ax.plot(pd.to_numeric(line["evolution_step"], errors="coerce"), y, marker="o", linewidth=1.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Window step")
    ax.set_ylabel("RMS Zernike error (nm)")
    set_wrapped_title(ax, "Zernike RMS error evolution", group_display_label(group, group_cols))
    apply_standard_grid(ax)
    finish_legend_and_save(fig, ax, plotdir / filename, show_legend, legend_outside=series_count > 6)


def plot_slow_parameter_outputs(
    plotdir: Path,
    slow_by_root: pd.DataFrame,
    slow_evolution: pd.DataFrame,
    group_cols: list[str],
    *,
    legend_mode: str,
    max_legend_items: int,
    label_map_rows: list[dict[str, Any]],
) -> None:
    required = {"parameter_group", "unit", "final_err", "final_abs_err", "final_err_over_sigma"}
    if not required.issubset(slow_by_root.columns):
        return
    plot_slow_summary_boxplots(
        slow_by_root,
        group_cols,
        "final_abs_err",
        "Final absolute error",
        plotdir / "slow_parameter_final_abs_error_by_group.png",
    )
    plot_slow_summary_boxplots(
        slow_by_root,
        group_cols,
        "final_err",
        "Final signed error",
        plotdir / "slow_parameter_final_signed_error_by_group.png",
    )
    plot_slow_summary_boxplots(
        slow_by_root,
        group_cols,
        "final_err_over_sigma",
        "Final signed error over sigma",
        plotdir / "slow_parameter_final_error_over_sigma_by_group.png",
    )
    slow_by_root = add_group_label(slow_by_root, group_cols)
    if not slow_evolution.empty and {"parameter_group", "unit", "err", "evolution_step"}.issubset(slow_evolution.columns):
        slow_evolution = add_group_label(slow_evolution, group_cols)
        for group_key, group in slow_evolution.groupby("_group_label", dropna=False):
            group_slug = slug(group_key)
            plot_source_slow_evolution(
                plotdir,
                group,
                group_slug,
                str(group_key),
                group_cols,
                legend_mode,
                max_legend_items,
                label_map_rows,
            )
            plot_zernike_rms_evolution(
                plotdir,
                group,
                group_slug,
                str(group_key),
                group_cols,
                legend_mode,
                max_legend_items,
                label_map_rows,
            )
    for group_key, group in slow_by_root.groupby("_group_label", dropna=False):
        group_slug = slug(group_key)
        for mirror_group in ("m1_zernike", "m2_zernike"):
            plot_zernike_final_error(
                plotdir,
                group,
                mirror_group,
                group_slug,
                str(group_key),
                group_cols,
                legend_mode,
                max_legend_items,
                label_map_rows,
            )


def write_markdown(outdir: Path, by_root: pd.DataFrame, by_group: pd.DataFrame, group_cols: list[str]) -> None:
    status_counts = dict(Counter(by_root.get("metric_status", []))) if not by_root.empty else {}
    lines = [
        "# Full-fidelity campaign-family actual-final summary",
        "",
        "Headline metrics use actual realized final separation error from existing review artifacts, not projected 30-minute endpoints.",
        "",
        "Signed final separation columns estimate repeatable bias; absolute final separation columns summarize achieved error magnitude. STD, SEM, and MAD are computed from signed final separation errors.",
        "",
        f"Grouping columns: {', '.join(group_cols) if group_cols else 'pooled'}",
        f"Root rows: {len(by_root)}",
        f"Metric status counts: {status_counts}",
        "",
        "Key outputs:",
        "- `family_actual_final_by_root.csv`",
        "- `family_actual_final_by_group.csv`",
        "- `family_actual_final_by_draw_campaign_note.csv`",
        "- `family_metric_status_counts.csv`",
        "- `family_slow_parameter_by_root.csv`",
        "- `family_slow_parameter_by_group.csv`",
        "- `family_slow_parameter_evolution_by_root.csv`",
        "- `plot_series_label_map.csv`",
        "- `plots/`",
        "",
        "Posterior-sigma comparisons are diagnostics for local statistical information; they can miss systematic model-mismatch bias.",
    ]
    if not by_group.empty:
        try:
            table = by_group.to_markdown(index=False)
        except ImportError:
            table = by_group.to_string(index=False)
        lines += ["", "## Group Summary", "", table]
    (outdir / "family_actual_final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    roots = roots_from_args(args)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sep_rows: list[pd.DataFrame] = []
    slow_rows: list[pd.DataFrame] = []
    slow_evolution_rows: list[pd.DataFrame] = []
    evolution_rows: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    for root in roots:
        root = root.expanduser()
        metadata = infer_metadata(root)
        sep, status = load_separation_rows(root)
        status_rows.append({**metadata, "metric_status": status})
        if sep.empty:
            sep_rows.append(pd.DataFrame([{**metadata, "case_name": "", "metric_status": status}]))
        else:
            sep = sep.copy()
            for key, value in metadata.items():
                sep[key] = value
            sep["metric_status"] = status
            sep_rows.append(sep)
        slow = load_slow_rows(root)
        if not slow.empty:
            slow = slow.copy()
            for key, value in metadata.items():
                slow[key] = value
            slow_rows.append(slow)
        slow_evolution = load_slow_evolution_rows(root)
        if not slow_evolution.empty:
            slow_evolution = slow_evolution.copy()
            for key, value in metadata.items():
                slow_evolution[key] = value
            slow_evolution_rows.append(slow_evolution)
        evolution = load_evolution_rows(root)
        if not evolution.empty:
            evolution = evolution.copy()
            for key, value in metadata.items():
                evolution[key] = value
            evolution_rows.append(evolution)

    by_root = pd.concat(sep_rows, ignore_index=True) if sep_rows else pd.DataFrame()
    group_cols = infer_group_cols(by_root, args.group_cols)
    ok_mask = by_root["metric_status"].eq("ok") if "metric_status" in by_root.columns else pd.Series(False, index=by_root.index)
    ok_by_root = by_root[ok_mask].copy()
    for col in group_cols:
        if col not in ok_by_root.columns:
            ok_by_root[col] = ""
    by_group = separation_group_summary(ok_by_root, group_cols)
    if args.include_pooled:
        by_group = add_pooled_rows(by_group, ok_by_root, group_cols)
    note_cols = [
        c
        for c in [
            *group_cols,
            "draw_index",
            "case_name",
            "initial_sep_err_uas",
            "final_sep_err_uas",
            "initial_abs_sep_err_uas",
            "final_abs_sep_err_uas",
            "abs_improvement_uas",
            "final_posterior_sigma_sep_uas",
            "final_abs_sep_err_over_sigma",
            "metric_status",
            "run_root",
        ]
        if c in by_root.columns
    ]
    note = by_root[note_cols].sort_values(note_cols[: min(3, len(note_cols))]) if note_cols else by_root
    status_counts = pd.DataFrame(
        [{"metric_status": key, "count": value} for key, value in Counter(by_root.get("metric_status", [])).items()]
    )
    slow_by_root = pd.concat(slow_rows, ignore_index=True) if slow_rows else pd.DataFrame()
    slow_by_group = slow_group_summary(slow_by_root, group_cols) if not slow_by_root.empty else pd.DataFrame()
    slow_evolution_by_root = pd.concat(slow_evolution_rows, ignore_index=True) if slow_evolution_rows else pd.DataFrame()
    evolution_by_root = pd.concat(evolution_rows, ignore_index=True) if evolution_rows else pd.DataFrame()

    by_root.to_csv(outdir / "family_actual_final_by_root.csv", index=False)
    by_group.to_csv(outdir / "family_actual_final_by_group.csv", index=False)
    note.to_csv(outdir / "family_actual_final_by_draw_campaign_note.csv", index=False)
    status_counts.to_csv(outdir / "family_metric_status_counts.csv", index=False)
    slow_by_root.to_csv(outdir / "family_slow_parameter_by_root.csv", index=False)
    slow_by_group.to_csv(outdir / "family_slow_parameter_by_group.csv", index=False)
    slow_evolution_by_root.to_csv(outdir / "family_slow_parameter_evolution_by_root.csv", index=False)
    evolution_by_root.to_csv(outdir / "family_separation_evolution_by_root.csv", index=False)
    if not args.no_plots:
        plot_family_outputs(
            outdir,
            ok_by_root,
            evolution_by_root,
            group_cols,
            slow_by_root=slow_by_root,
            slow_evolution=slow_evolution_by_root,
            plot_slow_parameters=args.plot_slow_parameters,
            legend_mode=args.legend_mode,
            max_legend_items=args.max_legend_items,
        )
    write_markdown(outdir, by_root, by_group, group_cols)
    print(f"Wrote family review bundle to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
