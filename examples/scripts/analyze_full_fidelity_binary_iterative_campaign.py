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
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from dluxshera.inference.observation_belief import (
    ObservationBeliefState,
    ObservationLikelihoodState,
)
from dluxshera.inference.observation_summary import (
    get_summary_information_scale,
    load_subblock_summary,
    load_subblock_summary_artifact_payload,
)

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

CUMULATIVE_SCHEMA_VERSION = "full_fidelity_cumulative_information_review.v1"
CUMULATIVE_VARIANTS = (
    ("all_windows", 0),
    ("exclude_first_window", 1),
    ("exclude_first_two_windows", 2),
)


def rel(path: Any, root: Path) -> str:
    if path in (None, "") or (isinstance(path, float) and math.isnan(path)):
        return ""
    p = Path(str(path))
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists() or not path.is_file():
        return default
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


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


def safe_float(value: Any, default: float = np.nan) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: float = np.nan) -> int | float:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    continue
                rows.append(item if isinstance(item, dict) else {"value": item})
    except OSError:
        return []
    return rows


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def first_nonblank(*values: Any, default: Any = "") -> Any:
    for value in values:
        if not _is_blank(value):
            return value
    return default


def _as_bool(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if value in (None, "") or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return value


def _nested_values(data: Any, keys: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key) in keys:
                found.append(value)
            found.extend(_nested_values(value, keys))
    elif isinstance(data, list):
        for item in data:
            found.extend(_nested_values(item, keys))
    return found


def _first_nested_value(data: Any, keys: set[str]) -> Any:
    for value in _nested_values(data, keys):
        if not _is_blank(value):
            return value
    return ""


def resolve_artifact_path(
    value: Any,
    run_root: Path,
    context_dirs: list[Path] | tuple[Path, ...] = (),
) -> Path | None:
    if _is_blank(value):
        return None
    raw = str(value)
    p = Path(raw).expanduser()
    candidates = [p] if p.is_absolute() else [run_root / p, *[d / p for d in context_dirs]]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return p if p.is_absolute() else run_root / p


def find_existing_candidate_path(candidates: list[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return None


def _artifact_text(path: Path | None, run_root: Path, fallback: Any = "") -> str:
    if path is None:
        return "" if _is_blank(fallback) else str(fallback)
    return rel(path, run_root) if path.exists() else ("" if _is_blank(fallback) else str(fallback))


def _path_parent_candidates(*paths: Path | None) -> list[Path]:
    dirs: list[Path] = []
    for path in paths:
        if path is None:
            continue
        base = path if path.is_dir() else path.parent
        for item in [base, base.parent, base.parent.parent]:
            if item not in dirs:
                dirs.append(item)
    return dirs


def _subblock_study_root(summary_path: Path | None, diag_path: Path | None) -> Path | None:
    if summary_path is not None and summary_path.exists():
        if summary_path.name in {"summary.json", "subblock_summary.json"}:
            return summary_path.parent
        if summary_path.parent.name == "schur_summary":
            return summary_path.parent
    if diag_path is not None:
        candidate = diag_path.parent / "study" / "schur_summary"
        if candidate.exists():
            return candidate
    return None


def _parse_timestamp(value: Any) -> datetime | None:
    if _is_blank(value):
        return None
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _seconds_span(start: Any, finish: Any) -> float:
    a = _parse_timestamp(start)
    b = _parse_timestamp(finish)
    if a is None or b is None:
        return np.nan
    return max(0.0, (b - a).total_seconds())


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
    if label in {"source.x_as", "source.y_as"}:
        return "as"
    if label == "source.position_angle_deg":
        return "deg"
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
    if label in {"source.separation_as", "source.x_as", "source.y_as"}:
        return f"{value * 1e6:.3g} microas"
    return f"{value:.3g} {label_units(label)}".strip()


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
    """Return display unit, offset scale, and value offset for note-ready summaries."""

    if label in {"source.separation_as", "source.x_as", "source.y_as"}:
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
        return "as_per_pix", 1.0, 0.0
    if "zernike_coeffs_nm" in label:
        return "nm", 1.0, 0.0
    return label_units(label), 1.0, 0.0


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, np.ndarray):
        return [_json_scalar(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_scalar(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _matrix_diagnostics_row(matrix: np.ndarray, prefix: str = "") -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return {
            f"{prefix}rank": 0,
            f"{prefix}min_eigenvalue": np.nan,
            f"{prefix}max_eigenvalue": np.nan,
            f"{prefix}trace": np.nan,
            f"{prefix}frobenius_norm": np.nan,
            f"{prefix}condition_number": np.nan,
            f"{prefix}finite": False,
        }
    finite = bool(np.all(np.isfinite(matrix)))
    if not finite:
        return {
            f"{prefix}rank": 0,
            f"{prefix}min_eigenvalue": np.nan,
            f"{prefix}max_eigenvalue": np.nan,
            f"{prefix}trace": np.nan,
            f"{prefix}frobenius_norm": np.nan,
            f"{prefix}condition_number": np.nan,
            f"{prefix}finite": False,
        }
    sym = 0.5 * (matrix + matrix.T)
    eig = np.linalg.eigvalsh(sym)
    tol = np.finfo(float).eps * max(sym.shape) * max(float(np.max(np.abs(eig))), 1.0)
    active = eig > tol
    pos = eig[active]
    condition = float(np.max(pos) / np.min(pos)) if pos.size else float("inf")
    return {
        f"{prefix}rank": int(np.count_nonzero(active)),
        f"{prefix}min_eigenvalue": float(np.min(eig)),
        f"{prefix}max_eigenvalue": float(np.max(eig)),
        f"{prefix}trace": float(np.trace(sym)),
        f"{prefix}frobenius_norm": float(np.linalg.norm(sym)),
        f"{prefix}condition_number": condition,
        f"{prefix}finite": True,
    }


def _path_after_run_name(path: Path, run_root: Path) -> Path | None:
    parts = path.parts
    run_name = run_root.name
    if run_name not in parts:
        return None
    index = len(parts) - 1 - list(reversed(parts)).index(run_name)
    suffix = Path(*parts[index + 1 :])
    return run_root / suffix


def resolve_portable_artifact_path(
    value: Any,
    run_root: Path,
    *,
    context_dirs: Sequence[Path] = (),
    plan_run_root: Any = "",
) -> tuple[Path | None, str]:
    if _is_blank(value):
        return None, "blank"
    raw = str(value)
    path = Path(raw).expanduser()
    candidates: list[tuple[Path, str]] = []
    if path.is_absolute():
        candidates.append((path, "recorded_absolute"))
        plan_root = Path(str(plan_run_root)).expanduser() if not _is_blank(plan_run_root) else None
        if plan_root and str(path).startswith(str(plan_root)):
            try:
                candidates.append((run_root / path.relative_to(plan_root), "plan_run_root_relative"))
            except ValueError:
                pass
        run_name_candidate = _path_after_run_name(path, run_root)
        if run_name_candidate is not None:
            candidates.append((run_name_candidate, "run_name_suffix"))
    else:
        candidates.append((run_root / path, "run_root_relative"))
        for directory in context_dirs:
            candidates.append((directory / path, "context_relative"))

    seen: set[str] = set()
    for candidate, method in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve(), method
    fallback = candidates[0][0] if candidates else path
    return fallback, "unresolved"


def _extract_window_subblock_from_path(value: Any) -> tuple[int | None, int | None]:
    text = str(value or "")
    window_match = re.search(r"window[_/](\d+)|window_(\d+)", text)
    subblock_match = re.search(r"subblock[_/](\d+)|subblock_(\d+)", text)

    def match_int(match: re.Match[str] | None) -> int | None:
        if match is None:
            return None
        for item in match.groups():
            if item is not None:
                return int(item)
        return None

    return match_int(window_match), match_int(subblock_match)


def _realized_case_name(base_case: Any, window_index: Any) -> str:
    if _is_blank(base_case) or pd.isna(window_index):
        return ""
    return f"{base_case}/windows/window_{int(window_index):03d}"


def _stable_summary_id(case_name: Any, window_index: Any, subblock_index: Any, path: Any = "") -> str:
    c = str(case_name)
    w = int(window_index) if not pd.isna(window_index) else -1
    s = int(subblock_index) if not pd.isna(subblock_index) else -1
    if w >= 0 and s >= 0:
        return f"{c}/window_{w:03d}/subblock_{s:03d}"
    return f"{c}/{path}"


def discover_cumulative_summary_inventory(run_root: Path, plan: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    plan_run_root = plan.get("run_root", "")
    status = read_csv(run_root / "subblock_status_iterative.csv")
    if not status.empty and {"case_name", "summary_path", "status"}.issubset(status.columns):
        for _, row in status.iterrows():
            window_index = safe_int(row.get("window_index", np.nan))
            subblock_index = safe_int(row.get("window_subblock_index", row.get("subblock_index", np.nan)))
            if pd.isna(window_index) or pd.isna(subblock_index):
                inferred_w, inferred_s = _extract_window_subblock_from_path(row.get("summary_path", ""))
                window_index = inferred_w if inferred_w is not None else window_index
                subblock_index = inferred_s if inferred_s is not None else subblock_index
            case_name = str(row.get("case_name", ""))
            realized = first_nonblank(row.get("window_case_name", ""), _realized_case_name(case_name, window_index))
            path, method = resolve_portable_artifact_path(
                row.get("summary_path", ""),
                run_root,
                plan_run_root=plan_run_root,
            )
            accepted = str(row.get("status", "")).strip().lower() == "ok" and path is not None and path.exists()
            exclusion = "" if accepted else ("missing_summary_artifact" if str(row.get("status", "")).strip().lower() == "ok" else "status_not_ok")
            rows.append(
                {
                    "case_name": case_name,
                    "realized_case_name": realized,
                    "window_index": window_index,
                    "subblock_index": subblock_index,
                    "stable_summary_id": _stable_summary_id(case_name, window_index, subblock_index, row.get("summary_path", "")),
                    "recorded_summary_path": str(row.get("summary_path", "")),
                    "resolved_summary_path": "" if path is None else str(path),
                    "path_resolution_method": method,
                    "status_source": "subblock_status_iterative.csv",
                    "status_value": row.get("status", ""),
                    "summary_id": "",
                    "summary_kind": "",
                    "theta_size": np.nan,
                    "summary_information_scale": "",
                    "accepted_for_cumulative": bool(accepted),
                    "exclusion_reason": exclusion,
                    "load_status": "not_loaded",
                }
            )
        return pd.DataFrame(rows).sort_values(["case_name", "window_index", "subblock_index"], na_position="last")

    for item in plan.get("expected_outputs", []) or []:
        if not isinstance(item, dict):
            continue
        case_name = str(item.get("case_name", ""))
        window_index = safe_int(item.get("window_index", np.nan))
        subblock_index = safe_int(item.get("subblock_index", np.nan))
        path, method = resolve_portable_artifact_path(item.get("summary_path", ""), run_root, plan_run_root=plan_run_root)
        rows.append(
            {
                "case_name": case_name,
                "realized_case_name": first_nonblank(item.get("window_case_name", ""), _realized_case_name(case_name, window_index)),
                "window_index": window_index,
                "subblock_index": subblock_index,
                "stable_summary_id": _stable_summary_id(case_name, window_index, subblock_index, item.get("summary_path", "")),
                "recorded_summary_path": str(item.get("summary_path", "")),
                "resolved_summary_path": "" if path is None else str(path),
                "path_resolution_method": method,
                "status_source": "campaign_plan.expected_outputs",
                "status_value": "expected",
                "summary_id": "",
                "summary_kind": "",
                "theta_size": np.nan,
                "summary_information_scale": "",
                "accepted_for_cumulative": bool(path is not None and path.exists()),
                "exclusion_reason": "" if path is not None and path.exists() else "missing_summary_artifact",
                "load_status": "not_loaded",
            }
        )
    if rows:
        return pd.DataFrame(rows).sort_values(["case_name", "window_index", "subblock_index"], na_position="last")

    for path in sorted(run_root.glob("cases/*/windows/window_*/summary_paths.csv")):
        window_case, window_index = parse_window_path(path)
        table = read_csv(path)
        for idx, row in table.iterrows():
            recorded = row.get("summary_path", "")
            inferred_w, inferred_s = _extract_window_subblock_from_path(recorded)
            subblock_index = inferred_s if inferred_s is not None else idx
            base_case = window_case
            if not _is_blank(row.get("case_name", "")):
                base_case = str(row.get("case_name", "")).split("/windows/")[0]
            resolved, method = resolve_portable_artifact_path(
                recorded,
                run_root,
                context_dirs=(path.parent,),
                plan_run_root=plan_run_root,
            )
            rows.append(
                {
                    "case_name": base_case,
                    "realized_case_name": first_nonblank(row.get("case_name", ""), _realized_case_name(base_case, window_index)),
                    "window_index": window_index if inferred_w is None else inferred_w,
                    "subblock_index": subblock_index,
                    "stable_summary_id": _stable_summary_id(base_case, window_index, subblock_index, recorded),
                    "recorded_summary_path": str(recorded),
                    "resolved_summary_path": "" if resolved is None else str(resolved),
                    "path_resolution_method": method,
                    "status_source": "summary_paths.csv",
                    "status_value": "listed",
                    "summary_id": "",
                    "summary_kind": "",
                    "theta_size": np.nan,
                    "summary_information_scale": "",
                    "accepted_for_cumulative": bool(resolved is not None and resolved.exists()),
                    "exclusion_reason": "" if resolved is not None and resolved.exists() else "missing_summary_artifact",
                    "load_status": "not_loaded",
                }
            )
    if rows:
        return pd.DataFrame(rows).sort_values(["case_name", "window_index", "subblock_index"], na_position="last")

    return pd.DataFrame(columns=[
        "case_name", "realized_case_name", "window_index", "subblock_index",
        "stable_summary_id", "recorded_summary_path", "resolved_summary_path",
        "path_resolution_method", "status_source", "status_value", "summary_id",
        "summary_kind", "theta_size", "summary_information_scale",
        "accepted_for_cumulative", "exclusion_reason", "load_status",
    ])


def reconstruct_initial_observation_prior(
    run_root: Path,
    plan: dict[str, Any],
    case_name: str,
    labels: Sequence[str],
) -> tuple[ObservationBeliefState | None, dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    labels = tuple(str(label) for label in labels)
    top_prior = read_csv(run_root / "prior_draws.csv")
    source = "prior_draws.csv"
    if top_prior.empty:
        source = "campaign_plan.prior_draw_rows_by_case"
        rows = plan.get("prior_draw_rows_by_case", {}).get(case_name, [])
        top_prior = pd.DataFrame(rows)
    if not top_prior.empty and "case_name" in top_prior.columns:
        filtered = top_prior[top_prior["case_name"].astype(str) == str(case_name)]
        if not filtered.empty:
            top_prior = filtered
    if top_prior.empty or not {"theta_label", "prior_mean", "prior_sigma"}.issubset(top_prior.columns):
        return None, {
            "status": "missing_prior",
            "source": source,
            "reconstructed": False,
            "reason": "No complete initial prior table with theta_label/prior_mean/prior_sigma was found.",
        }, [{"status": "missing_prior", "case": case_name, "message": "Cumulative analysis requires initial prior_draws.csv or equivalent campaign-plan prior rows."}]

    by_label = {str(row["theta_label"]): row for _, row in top_prior.iterrows()}
    missing = [label for label in labels if label not in by_label]
    if missing:
        return None, {
            "status": "missing_prior",
            "source": source,
            "reconstructed": False,
            "missing_labels": missing,
        }, [{"status": "missing_prior", "case": case_name, "message": "Initial prior table is missing labels: " + ", ".join(missing)}]
    table_mean = np.asarray([safe_float(by_label[label].get("prior_mean")) for label in labels], dtype=float)
    sigma = np.asarray([safe_float(by_label[label].get("prior_sigma")) for label in labels], dtype=float)
    if not np.all(np.isfinite(table_mean)) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        return None, {
            "status": "missing_prior",
            "source": source,
            "reconstructed": False,
            "reason": "Initial prior mean/sigma contains non-finite or non-positive values.",
        }, [{"status": "missing_prior", "case": case_name, "message": "Initial prior mean/sigma contains invalid values."}]

    mean = table_mean.copy()
    mean_source = source
    first_update_paths = sorted((run_root / "cases" / case_name / "windows").glob("window_*/iterative_reference_update.json"))
    if first_update_paths:
        first_update = read_json(first_update_paths[0], {})
        truth = first_update.get("truth_by_label", {}) if isinstance(first_update.get("truth_by_label"), dict) else {}
        current_offsets = first_update.get("current_offsets", {}) if isinstance(first_update.get("current_offsets"), dict) else {}
        candidate = np.asarray(
            [
                safe_float(truth.get(label, np.nan)) + safe_float(current_offsets.get(label, np.nan))
                for label in labels
            ],
            dtype=float,
        )
        if np.all(np.isfinite(candidate)):
            mean_source = rel(first_update_paths[0], run_root) + ":truth_by_label+current_offsets"
            if not np.allclose(candidate, table_mean, rtol=1e-10, atol=1e-12):
                warnings.append(
                    {
                        "status": "initial_prior_mean_context_differs",
                        "case": case_name,
                        "message": "Top-level prior_draws.csv mean differs from the realized window-0 observation-context reference; cumulative analysis uses the realized initial reference as the prior mean and prior_draws.csv for sigma.",
                        "table_prior_mean_source": source,
                        "realized_initial_reference_source": mean_source,
                        "max_abs_difference": float(np.max(np.abs(candidate - table_mean))),
                    }
                )
            mean = candidate

    window_prior_paths = sorted((run_root / "cases" / case_name / "windows").glob("window_*/prior_draws.csv"))
    differing_windows: list[str] = []
    for path in window_prior_paths:
        frame = read_csv(path)
        if frame.empty or "reference_value" not in frame.columns:
            continue
        ref_by_label = {str(row["theta_label"]): safe_float(row.get("reference_value")) for _, row in frame.iterrows()}
        ref = np.asarray([ref_by_label.get(label, np.nan) for label in labels], dtype=float)
        if np.all(np.isfinite(ref)) and not np.allclose(ref, mean, rtol=1e-10, atol=1e-12):
            differing_windows.append(rel(path, run_root))
    if differing_windows:
        warnings.append(
            {
                "status": "historical_window_prior_differs",
                "case": case_name,
                "message": "Per-window reference/prior tables follow the historical moving reference and differ from the initial prior; cumulative analysis uses the initial prior once.",
                "paths": differing_windows[:5],
                "additional_count": max(0, len(differing_windows) - 5),
            }
        )

    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=mean,
        sigma=sigma,
        metadata={
            "prior_source": source,
            "prior_mean_source": mean_source,
            "prior_counted_once": True,
            "case_name": case_name,
            "reconstructed": True,
        },
    )
    provenance = {
        "status": "ok",
        "source": source,
        "prior_mean_source": mean_source,
        "prior_sigma_source": source,
        "reconstructed": True,
        "prior_counted_once": True,
        "labels": list(labels),
        "mean": mean.tolist(),
        "table_prior_mean": table_mean.tolist(),
        "sigma": sigma.tolist(),
        "window_local_prior_differs": bool(differing_windows),
        "window_local_prior_difference_count": len(differing_windows),
    }
    return prior, provenance, warnings


def recover_truth_by_label(
    run_root: Path,
    plan: dict[str, Any],
    case_name: str,
    labels: Sequence[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    labels = tuple(str(label) for label in labels)
    candidates: list[tuple[str, dict[str, Any]]] = []
    top_truth = read_csv(run_root / "truth_realization_by_label.csv")
    if not top_truth.empty and {"theta_label", "truth_value"}.issubset(top_truth.columns):
        frame = top_truth
        if "case_name" in frame.columns:
            filtered = frame[frame["case_name"].astype(str) == str(case_name)]
            if not filtered.empty:
                frame = filtered
        candidates.append(("truth_realization_by_label.csv", dict(zip(frame["theta_label"].astype(str), frame["truth_value"]))))
    if isinstance(plan.get("prior_truth_by_label"), dict):
        candidates.append(("campaign_plan.prior_truth_by_label", plan["prior_truth_by_label"]))
    for path in sorted((run_root / "cases" / case_name / "windows").glob("window_*/iterative_reference_update.json")):
        payload = read_json(path, {})
        if isinstance(payload.get("truth_by_label"), dict):
            candidates.append((rel(path, run_root) + ":truth_by_label", payload["truth_by_label"]))
            break
    for path in sorted((run_root / "cases" / case_name / "windows").glob("window_*/posterior_by_label.csv")):
        frame = read_csv(path)
        if not frame.empty and {"theta_label", "truth_value"}.issubset(frame.columns):
            candidates.append((rel(path, run_root), dict(zip(frame["theta_label"].astype(str), frame["truth_value"]))))
            break
    for source, values in candidates:
        truth = {label: safe_float(values.get(label, np.nan)) for label in labels}
        if all(np.isfinite(truth[label]) for label in labels):
            return truth, {"source": source, "status": "ok"}
    return {label: np.nan for label in labels}, {"source": "", "status": "missing_truth"}


def _load_cumulative_summaries(
    inventory: pd.DataFrame,
    labels: Sequence[str],
) -> tuple[list[Any], pd.DataFrame, list[dict[str, Any]]]:
    rows = inventory.to_dict(orient="records")
    loaded: list[Any] = []
    warnings: list[dict[str, Any]] = []
    seen_stable: set[str] = set()
    expected_labels = tuple(str(label) for label in labels)
    for row in rows:
        stable = str(row.get("stable_summary_id", ""))
        if stable in seen_stable:
            row["accepted_for_cumulative"] = False
            row["exclusion_reason"] = "duplicate_stable_summary_id"
            row["load_status"] = "not_loaded"
            warnings.append({"status": "duplicate_stable_summary_id", "summary_identity": stable, "message": "Duplicate cumulative summary identity."})
            continue
        seen_stable.add(stable)
        if not bool(row.get("accepted_for_cumulative")):
            row["load_status"] = "not_loaded"
            continue
        path = Path(str(row.get("resolved_summary_path", "")))
        try:
            payload = load_subblock_summary_artifact_payload(path)
            summary = load_subblock_summary(path)
            row["summary_id"] = summary.subblock_id
            row["summary_kind"] = summary.summary_kind
            row["theta_size"] = summary.theta_size
            row["summary_information_scale"] = get_summary_information_scale(payload) or ""
            if tuple(summary.theta_labels) != expected_labels:
                row["accepted_for_cumulative"] = False
                row["exclusion_reason"] = "label_mismatch"
                row["load_status"] = "label_mismatch"
                warnings.append({"status": "label_mismatch", "summary_identity": stable, "path": str(path), "message": "Summary theta labels do not match campaign layout."})
                continue
            if not np.all(np.isfinite(summary.theta_ref)) or not np.all(np.isfinite(summary.reduced_score)) or not np.all(np.isfinite(summary.reduced_information)):
                row["accepted_for_cumulative"] = False
                row["exclusion_reason"] = "invalid_matrix"
                row["load_status"] = "invalid_matrix"
                warnings.append({"status": "invalid_matrix", "summary_identity": stable, "path": str(path), "message": "Summary contains non-finite arrays."})
                continue
            row["load_status"] = "loaded"
            loaded.append(summary)
        except Exception as exc:
            row["accepted_for_cumulative"] = False
            row["exclusion_reason"] = "summary_load_error"
            row["load_status"] = f"error: {exc}"
            warnings.append({"status": "summary_load_error", "summary_identity": stable, "path": str(path), "message": str(exc)})
    scales = sorted({str(row.get("summary_information_scale", "")) for row in rows if row.get("accepted_for_cumulative") and str(row.get("summary_information_scale", "")).strip()})
    if len(scales) > 1:
        for row in rows:
            if row.get("accepted_for_cumulative"):
                row["accepted_for_cumulative"] = False
                row["exclusion_reason"] = "information_scale_mismatch"
        warnings.append({"status": "information_scale_mismatch", "message": "Accepted summaries have mixed information-scale provenance.", "scales": scales})
        loaded = []
    return loaded, pd.DataFrame(rows), warnings


def _vector_from_label_values(labels: Sequence[str], values: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([safe_float(values.get(label, np.nan)) for label in labels], dtype=float)


def _historical_vectors(run_root: Path, case_name: str, labels: Sequence[str]) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    labels = tuple(str(label) for label in labels)
    for path in sorted((run_root / "cases" / case_name / "windows").glob("window_*/iterative_reference_update.json")):
        data = read_json(path, {})
        window_index = safe_int(data.get("window_index", parse_window_path(path)[1]))
        truth = data.get("truth_by_label", {}) if isinstance(data.get("truth_by_label"), dict) else {}
        current_offsets = data.get("current_offsets", {}) if isinstance(data.get("current_offsets"), dict) else {}
        posterior_offsets = data.get("posterior_offsets", {}) if isinstance(data.get("posterior_offsets"), dict) else {}
        next_offsets = data.get("next_offsets", {}) if isinstance(data.get("next_offsets"), dict) else {}
        truth_vec = _vector_from_label_values(labels, truth)
        result[int(window_index)] = {
            "truth": truth_vec,
            "historical_reference_before": truth_vec + _vector_from_label_values(labels, current_offsets),
            "window_local_posterior_mean": truth_vec + _vector_from_label_values(labels, posterior_offsets),
            "historical_next_reference": truth_vec + _vector_from_label_values(labels, next_offsets),
        }
    return result


def _posterior_table_maps(run_root: Path, case_name: str) -> dict[int, dict[str, dict[str, Any]]]:
    out: dict[int, dict[str, dict[str, Any]]] = {}
    for path in sorted((run_root / "cases" / case_name / "windows").glob("window_*/posterior_by_label.csv")):
        window_index = parse_window_path(path)[1]
        frame = read_csv(path)
        if frame.empty or "theta_label" not in frame.columns:
            continue
        out[window_index] = {str(row["theta_label"]): row.to_dict() for _, row in frame.iterrows()}
    return out


def _separation_index(labels: Sequence[str]) -> int | None:
    try:
        return tuple(labels).index("source.separation_as")
    except ValueError:
        return None


def _window_duration_s(inventory: pd.DataFrame, plan: dict[str, Any]) -> float:
    subblock_duration = safe_float(scalar(plan, "trace_source.subblock_duration_s", np.nan))
    if np.isfinite(subblock_duration):
        return subblock_duration
    exposure = safe_float(scalar(plan, "subblock_command_options.exposure_time_s", np.nan))
    frames = safe_float(scalar(plan, "trace_source.n_frames_per_subblock", np.nan))
    n_per_window = float(pd.to_numeric(inventory.get("subblock_index", pd.Series(dtype=float)), errors="coerce").groupby(inventory.get("window_index", pd.Series(dtype=float))).count().median()) if not inventory.empty else np.nan
    if np.isfinite(exposure) and np.isfinite(frames) and np.isfinite(n_per_window):
        return exposure * frames
    return np.nan


def build_cumulative_case_products(
    run_root: Path,
    plan: dict[str, Any],
    case_name: str,
    case_inventory: pd.DataFrame,
    prior: ObservationBeliefState,
    truth_by_label: dict[str, float],
) -> dict[str, Any]:
    labels = tuple(prior.theta_labels)
    loaded_summaries, loaded_inventory, load_warnings = _load_cumulative_summaries(case_inventory, labels)
    exclusion_reasons = set(loaded_inventory.get("exclusion_reason", pd.Series(dtype=str)).astype(str))
    if "information_scale_mismatch" in exclusion_reasons:
        raise ValueError("information_scale_mismatch: accepted summaries have mixed information-scale provenance.")
    if "label_mismatch" in exclusion_reasons:
        raise ValueError("label_mismatch: one or more summaries have incompatible theta labels.")
    accepted_inventory = loaded_inventory[loaded_inventory["accepted_for_cumulative"].astype(bool)].copy()
    if len(loaded_summaries) != len(accepted_inventory):
        raise RuntimeError("Internal cumulative inventory/load count mismatch.")
    if accepted_inventory.empty:
        raise ValueError("No accepted summaries are available for cumulative analysis.")
    windows = sorted(int(w) for w in accepted_inventory["window_index"].dropna().unique())
    grouped_indices: dict[int, list[int]] = {}
    for pos, (_, row) in enumerate(accepted_inventory.sort_values(["window_index", "subblock_index"]).iterrows()):
        grouped_indices.setdefault(int(row["window_index"]), []).append(pos)
    ordered_summaries = [loaded_summaries[pos] for window in windows for pos in grouped_indices[window]]
    ordered_inventory = accepted_inventory.sort_values(["window_index", "subblock_index"]).reset_index(drop=True)

    historical = _historical_vectors(run_root, case_name, labels)
    posterior_maps = _posterior_table_maps(run_root, case_name)
    truth_vec = np.asarray([truth_by_label[label] for label in labels], dtype=float)
    sep_idx = _separation_index(labels)
    state = ObservationLikelihoodState.empty(labels, metadata={"case_name": case_name, "variant": "all_windows"})
    window_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    first_sigma = np.nan
    running_count = 0
    duration_per_subblock = _window_duration_s(accepted_inventory, plan)
    for window in windows:
        summaries = [ordered_summaries[int(i)] for i in grouped_indices[window]]
        state = state.add_summaries(summaries, reject_duplicate_ids=False)
        running_count += len(summaries)
        update = state.combine_with_prior(prior)
        posterior = update.posterior
        sigma = posterior.sigma()
        cumulative_mean = posterior.mean
        cumulative_error = cumulative_mean - truth_vec
        h = historical.get(window, {})
        local_map = posterior_maps.get(window, {})
        hist_ref = h.get("historical_reference_before", np.full(len(labels), np.nan))
        hist_post = h.get("window_local_posterior_mean", np.full(len(labels), np.nan))
        hist_next = h.get("historical_next_reference", np.full(len(labels), np.nan))
        if sep_idx is not None:
            cumulative_sigma_sep_uas = sigma[sep_idx] * 1e6
            if not np.isfinite(first_sigma):
                first_sigma = cumulative_sigma_sep_uas
            expected = first_sigma / math.sqrt(len([w for w in windows if w <= window])) if np.isfinite(first_sigma) else np.nan
            local_row = local_map.get(labels[sep_idx], {})
            local_mean_sep = safe_float(local_row.get("posterior_mean", hist_post[sep_idx] if hist_post.size else np.nan))
            local_sigma_sep_uas = safe_float(local_row.get("posterior_sigma", np.nan)) * 1e6
            cumulative_sep_err_uas = cumulative_error[sep_idx] * 1e6
            row = {
                "case_name": case_name,
                "window_index": window,
                "n_windows_cumulative": len([w for w in windows if w <= window]),
                "n_subblocks_window": len(summaries),
                "n_subblocks_cumulative": running_count,
                "cumulative_duration_s": running_count * duration_per_subblock if np.isfinite(duration_per_subblock) else np.nan,
                "window_local_separation_mean_as": local_mean_sep,
                "window_local_separation_error_uas": (local_mean_sep - truth_vec[sep_idx]) * 1e6 if np.isfinite(local_mean_sep) else np.nan,
                "window_local_separation_sigma_uas": local_sigma_sep_uas,
                "cumulative_separation_mean_as": cumulative_mean[sep_idx],
                "cumulative_separation_error_uas": cumulative_sep_err_uas,
                "cumulative_separation_sigma_uas": cumulative_sigma_sep_uas,
                "cumulative_error_over_sigma": cumulative_sep_err_uas / cumulative_sigma_sep_uas if cumulative_sigma_sep_uas else np.nan,
                "historical_reference_before_as": hist_ref[sep_idx] if hist_ref.size else np.nan,
                "historical_reference_before_error_uas": (hist_ref[sep_idx] - truth_vec[sep_idx]) * 1e6 if hist_ref.size and np.isfinite(hist_ref[sep_idx]) else np.nan,
                "historical_next_reference_as": hist_next[sep_idx] if hist_next.size else np.nan,
                "historical_next_reference_error_uas": (hist_next[sep_idx] - truth_vec[sep_idx]) * 1e6 if hist_next.size and np.isfinite(hist_next[sep_idx]) else np.nan,
                "cumulative_minus_window_local_uas": (cumulative_mean[sep_idx] - local_mean_sep) * 1e6 if np.isfinite(local_mean_sep) else np.nan,
                "cumulative_minus_next_reference_uas": (cumulative_mean[sep_idx] - hist_next[sep_idx]) * 1e6 if hist_next.size and np.isfinite(hist_next[sep_idx]) else np.nan,
                "expected_sqrt_n_sigma_uas": expected,
                "sigma_ratio_to_first_window": cumulative_sigma_sep_uas / first_sigma if first_sigma else np.nan,
                "sigma_ratio_to_sqrt_n_expectation": cumulative_sigma_sep_uas / expected if expected else np.nan,
                "solve_method": update.metadata.get("solve_method", ""),
                "cumulative_status": "ok",
            }
            info_diag = _matrix_diagnostics_row(state.information)
            post_diag = _matrix_diagnostics_row(posterior.precision)
            row.update(
                {
                    "information_rank": info_diag["rank"],
                    "information_min_eigenvalue": info_diag["min_eigenvalue"],
                    "information_max_eigenvalue": info_diag["max_eigenvalue"],
                    "information_condition_number": info_diag["condition_number"],
                    "posterior_rank": post_diag["rank"],
                    "posterior_condition_number": post_diag["condition_number"],
                }
            )
            window_rows.append(row)
        diagnostics_row = {
            "case_name": case_name,
            "window_index": window,
            "n_windows_cumulative": len([w for w in windows if w <= window]),
            "n_subblocks_cumulative": running_count,
            "solve_method": update.metadata.get("solve_method", ""),
            "damping": safe_float(update.metadata.get("damping", 0.0)),
        }
        diagnostics_row.update(_matrix_diagnostics_row(state.information, "information_"))
        diagnostics_row.update(_matrix_diagnostics_row(posterior.precision, "posterior_precision_"))
        diagnostics_rows.append(diagnostics_row)
        for index, label in enumerate(labels):
            local_row = local_map.get(label, {})
            local_mean = safe_float(local_row.get("posterior_mean", hist_post[index] if hist_post.size else np.nan))
            local_sigma = safe_float(local_row.get("posterior_sigma", np.nan))
            posterior_rows.append(
                {
                    "case_name": case_name,
                    "window_index": window,
                    "n_windows_cumulative": len([w for w in windows if w <= window]),
                    "n_subblocks_cumulative": running_count,
                    "theta_label": label,
                    "truth_value": truth_vec[index],
                    "initial_prior_mean": prior.mean[index],
                    "initial_prior_sigma": prior.sigma()[index],
                    "cumulative_posterior_mean": cumulative_mean[index],
                    "cumulative_posterior_sigma": sigma[index],
                    "cumulative_posterior_error": cumulative_error[index],
                    "cumulative_error_over_sigma": cumulative_error[index] / sigma[index] if sigma[index] else np.nan,
                    "window_local_posterior_mean": local_mean,
                    "window_local_posterior_sigma": local_sigma,
                    "historical_reference_before": hist_ref[index] if hist_ref.size else np.nan,
                    "historical_next_reference": hist_next[index] if hist_next.size else np.nan,
                    "parameter_unit": local_row.get("unit", label_units(label)),
                    "parameter_group": local_row.get("label_group", label_group(label)),
                }
            )
        reference_rows.append(
            {
                "case_name": case_name,
                "window_index": window,
                "n_subblocks_cumulative": running_count,
                "reference_to_cumulative_norm": float(np.linalg.norm(hist_ref - cumulative_mean)) if hist_ref.size and np.all(np.isfinite(hist_ref)) else np.nan,
                "window_posterior_to_cumulative_norm": float(np.linalg.norm(hist_post - cumulative_mean)) if hist_post.size and np.all(np.isfinite(hist_post)) else np.nan,
                "next_reference_to_cumulative_norm": float(np.linalg.norm(hist_next - cumulative_mean)) if hist_next.size and np.all(np.isfinite(hist_next)) else np.nan,
                "historical_reference_before_minus_cumulative_sep_uas": (hist_ref[sep_idx] - cumulative_mean[sep_idx]) * 1e6 if sep_idx is not None and hist_ref.size and np.isfinite(hist_ref[sep_idx]) else np.nan,
                "window_local_posterior_minus_cumulative_sep_uas": (hist_post[sep_idx] - cumulative_mean[sep_idx]) * 1e6 if sep_idx is not None and hist_post.size and np.isfinite(hist_post[sep_idx]) else np.nan,
                "historical_next_reference_minus_cumulative_sep_uas": (hist_next[sep_idx] - cumulative_mean[sep_idx]) * 1e6 if sep_idx is not None and hist_next.size and np.isfinite(hist_next[sep_idx]) else np.nan,
            }
        )

    final_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for variant, start_window in CUMULATIVE_VARIANTS:
        variant_windows = [w for w in windows if w >= start_window]
        if not variant_windows:
            continue
        variant_summaries = [ordered_summaries[int(i)] for w in variant_windows for i in grouped_indices[w]]
        variant_state = ObservationLikelihoodState.from_summaries(
            theta_labels=labels,
            summaries=variant_summaries,
            metadata={"case_name": case_name, "variant": variant},
            reject_duplicate_ids=False,
        )
        variant_update = variant_state.combine_with_prior(prior)
        variant_post = variant_update.posterior
        variant_sigma = variant_post.sigma()
        last_window = variant_windows[-1]
        hlast = historical.get(last_window, {})
        local_map = posterior_maps.get(last_window, {})
        if sep_idx is not None:
            local_sep_row = local_map.get(labels[sep_idx], {})
            local_mean = safe_float(local_sep_row.get("posterior_mean", np.nan))
            local_sigma = safe_float(local_sep_row.get("posterior_sigma", np.nan)) * 1e6
            hist_next = hlast.get("historical_next_reference", np.full(len(labels), np.nan))
            err_uas = (variant_post.mean[sep_idx] - truth_vec[sep_idx]) * 1e6
            sigma_uas = variant_sigma[sep_idx] * 1e6
            out = {
                "case_name": case_name,
                "cumulative_variant": variant,
                "cumulative_final_sep_err_uas": err_uas,
                "cumulative_final_abs_sep_err_uas": abs(err_uas),
                "cumulative_final_posterior_sigma_sep_uas": sigma_uas,
                "cumulative_final_sep_err_over_sigma": err_uas / sigma_uas if sigma_uas else np.nan,
                "window_local_final_sep_err_uas": (local_mean - truth_vec[sep_idx]) * 1e6 if np.isfinite(local_mean) else np.nan,
                "window_local_final_posterior_sigma_sep_uas": local_sigma,
                "historical_next_reference_final_sep_err_uas": (hist_next[sep_idx] - truth_vec[sep_idx]) * 1e6 if hist_next.size and np.isfinite(hist_next[sep_idx]) else np.nan,
                "n_windows": len(variant_windows),
                "n_subblocks": len(variant_summaries),
                "total_duration_s": len(variant_summaries) * duration_per_subblock if np.isfinite(duration_per_subblock) else np.nan,
                "sigma_improvement_factor": first_sigma / sigma_uas if sigma_uas and np.isfinite(first_sigma) else np.nan,
                "sigma_ratio_to_sqrt_n_expectation": sigma_uas / (first_sigma / math.sqrt(len(variant_windows))) if np.isfinite(first_sigma) and len(variant_windows) else np.nan,
                "information_rank": _matrix_diagnostics_row(variant_state.information)["rank"],
                "information_condition_number": _matrix_diagnostics_row(variant_state.information)["condition_number"],
                "posterior_rank": _matrix_diagnostics_row(variant_post.precision)["rank"],
                "posterior_condition_number": _matrix_diagnostics_row(variant_post.precision)["condition_number"],
                "metric_status": "ok",
            }
            variant_rows.append(out)
            if variant == "all_windows":
                final_rows.append({k: v for k, v in out.items() if k != "cumulative_variant"})
    return {
        "inventory": loaded_inventory,
        "window_summary": pd.DataFrame(window_rows),
        "posterior_by_label": pd.DataFrame(posterior_rows),
        "final_summary": pd.DataFrame(final_rows),
        "variant_summary": pd.DataFrame(variant_rows),
        "diagnostics": pd.DataFrame(diagnostics_rows),
        "reference_audit": pd.DataFrame(reference_rows),
        "likelihood_state": state,
        "warnings": load_warnings,
        "accepted_count": int(len(accepted_inventory)),
        "expected_count": int(len(case_inventory)),
        "windows": windows,
    }


def _cumulative_status_payload(
    *,
    run_root: Path,
    outdir: Path,
    mode: str,
    status: str,
    warnings: list[dict[str, Any]],
    outputs: Mapping[str, Any] | None = None,
    cases_processed: Sequence[str] = (),
    final_metrics: Sequence[Mapping[str, Any]] = (),
    prior_provenance: Mapping[str, Any] | None = None,
    inventory: pd.DataFrame | None = None,
) -> dict[str, Any]:
    path_counts = {}
    if inventory is not None and not inventory.empty and "path_resolution_method" in inventory.columns:
        path_counts = dict(Counter(inventory["path_resolution_method"].astype(str)))
    return {
        "schema_version": CUMULATIVE_SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "run_root": str(run_root),
        "review_output_root": str(outdir),
        "cumulative_mode": mode,
        "status": status,
        "cases_processed": list(cases_processed),
        "expected_summary_count": int(len(inventory)) if inventory is not None else 0,
        "accepted_summary_count": int(inventory["accepted_for_cumulative"].astype(bool).sum()) if inventory is not None and not inventory.empty else 0,
        "initial_prior_provenance": _json_scalar(prior_provenance or {}),
        "label_layout": read_json(run_root / "campaign_plan.json", {}).get("theta_layout", {}),
        "information_scale_provenance": sorted({str(v) for v in inventory.get("summary_information_scale", pd.Series(dtype=str)).tolist() if str(v).strip()}) if inventory is not None and not inventory.empty else [],
        "path_resolution_counts": path_counts,
        "warnings": _json_scalar(warnings),
        "output_paths": _json_scalar(dict(outputs or {})),
        "final_cumulative_metrics": _json_scalar(list(final_metrics)),
        "software_provenance": {
            "observation_likelihood_state_api": "ObservationLikelihoodState",
            "reviewer_script": "examples/scripts/analyze_full_fidelity_binary_iterative_campaign.py",
        },
    }


def _write_cumulative_disabled(outdir: Path, run_root: Path, mode: str) -> dict[str, Any]:
    cdir = outdir / "cumulative_information"
    cdir.mkdir(parents=True, exist_ok=True)
    payload = _cumulative_status_payload(
        run_root=run_root,
        outdir=outdir,
        mode=mode,
        status="disabled",
        warnings=[],
        outputs={"cumulative_summary_json": "cumulative_information/cumulative_summary.json"},
    )
    (cdir / "cumulative_summary.json").write_text(json.dumps(_json_scalar(payload), indent=2))
    return payload


def plot_cumulative_information_outputs(
    outdir: Path,
    cumulative_window_summary: pd.DataFrame,
    warnings_list: list[dict[str, Any]],
) -> list[str]:
    plot_paths: list[str] = []
    if plt is None or cumulative_window_summary.empty:
        return plot_paths
    plot_dir = outdir / "plots"
    plot_dir.mkdir(exist_ok=True)
    for case_name, group in cumulative_window_summary.groupby("case_name", dropna=False):
        group = group.sort_values("window_index")
        safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case_name))
        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            ax.plot(group["window_index"], group["historical_reference_before_error_uas"], marker="o", label="historical reference before")
            ax.plot(group["window_index"], group["window_local_separation_error_uas"], marker="o", label="window-local posterior")
            ax.plot(group["window_index"], group["historical_next_reference_error_uas"], marker="o", label="historical next reference")
            ax.plot(group["window_index"], group["cumulative_separation_error_uas"], marker="o", label="cumulative posterior")
            ax.set_xlabel("window index")
            ax.set_ylabel("separation error (uas)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = plot_dir / f"cumulative_separation_estimate_{safe_case}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plot_paths.append(rel(path, outdir))
        except Exception as exc:
            _append_plot_warning(warnings_list, f"cumulative separation estimate plot failed: {exc}", context="cumulative_information")
        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(group["n_windows_cumulative"], group["window_local_separation_sigma_uas"], marker="o", label="window-local sigma")
            ax.plot(group["n_windows_cumulative"], group["cumulative_separation_sigma_uas"], marker="o", label="cumulative sigma")
            ax.plot(group["n_windows_cumulative"], group["expected_sqrt_n_sigma_uas"], linestyle="--", label="first/sqrt(N)")
            ax.set_xlabel("cumulative windows")
            ax.set_ylabel("separation sigma (uas)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = plot_dir / f"cumulative_separation_sigma_{safe_case}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plot_paths.append(rel(path, outdir))
        except Exception as exc:
            _append_plot_warning(warnings_list, f"cumulative sigma plot failed: {exc}", context="cumulative_information")
        try:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            ax.plot(group["n_windows_cumulative"], group["cumulative_error_over_sigma"], marker="o")
            ax.set_xlabel("cumulative windows")
            ax.set_ylabel("cumulative separation error / sigma")
            fig.tight_layout()
            path = plot_dir / f"cumulative_error_over_sigma_{safe_case}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            plot_paths.append(rel(path, outdir))
        except Exception as exc:
            _append_plot_warning(warnings_list, f"cumulative error/sigma plot failed: {exc}", context="cumulative_information")
    return plot_paths


def run_cumulative_information_review(
    run_root: Path,
    outdir: Path,
    *,
    mode: str,
    no_plots: bool,
    review_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if mode == "off":
        return _write_cumulative_disabled(outdir, run_root, mode)
    cdir = outdir / "cumulative_information"
    cdir.mkdir(parents=True, exist_ok=True)
    plan = read_json(run_root / "campaign_plan.json", {})
    labels = tuple(str(label) for label in scalar(plan, "theta_layout.labels", []) or [])
    warnings: list[dict[str, Any]] = []
    outputs = {
        "cumulative_input_inventory_csv": "cumulative_information/cumulative_input_inventory.csv",
        "cumulative_window_summary_csv": "cumulative_information/cumulative_window_summary.csv",
        "cumulative_posterior_by_label_csv": "cumulative_information/cumulative_posterior_by_label.csv",
        "cumulative_final_summary_csv": "cumulative_information/cumulative_final_summary.csv",
        "cumulative_information_diagnostics_csv": "cumulative_information/cumulative_information_diagnostics.csv",
        "cumulative_reference_audit_csv": "cumulative_information/cumulative_reference_audit.csv",
        "cumulative_variant_summary_csv": "cumulative_information/cumulative_variant_summary.csv",
        "cumulative_summary_json": "cumulative_information/cumulative_summary.json",
        "cumulative_likelihood_state_json": "cumulative_information/cumulative_likelihood_state.json",
    }
    if not labels:
        warning = {"status": "missing_prior", "message": "campaign_plan.json does not contain theta_layout.labels."}
        warnings.append(warning)
        if mode == "on":
            raise RuntimeError(warning["message"])
        payload = _cumulative_status_payload(run_root=run_root, outdir=outdir, mode=mode, status="missing_prior", warnings=warnings, outputs=outputs)
        (cdir / "cumulative_summary.json").write_text(json.dumps(_json_scalar(payload), indent=2))
        review_warnings.extend(warnings)
        return payload

    inventory = discover_cumulative_summary_inventory(run_root, plan)
    if inventory.empty:
        warning = {"status": "missing_summary_inventory", "message": "No cumulative summary inventory could be discovered from status, campaign plan, or per-window summary_paths.csv."}
        warnings.append(warning)
        write_csv(inventory, cdir / "cumulative_input_inventory.csv")
        if mode == "on":
            raise RuntimeError(warning["message"])
        payload = _cumulative_status_payload(run_root=run_root, outdir=outdir, mode=mode, status="missing_summary_inventory", warnings=warnings, outputs=outputs, inventory=inventory)
        (cdir / "cumulative_summary.json").write_text(json.dumps(_json_scalar(payload), indent=2))
        review_warnings.extend(warnings)
        return payload

    duplicate_ids = inventory["stable_summary_id"].duplicated(keep=False) if "stable_summary_id" in inventory.columns else pd.Series(dtype=bool)
    if bool(duplicate_ids.any()):
        warnings.append({"status": "duplicate_stable_summary_id", "message": "Duplicate case/window/subblock cumulative summary identities were found.", "count": int(duplicate_ids.sum())})

    all_inventory: list[pd.DataFrame] = []
    window_frames: list[pd.DataFrame] = []
    posterior_frames: list[pd.DataFrame] = []
    final_frames: list[pd.DataFrame] = []
    variant_frames: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []
    reference_frames: list[pd.DataFrame] = []
    likelihood_payload: dict[str, Any] = {
        "schema_version": CUMULATIVE_SCHEMA_VERSION,
        "states_by_case": {},
    }
    prior_provenance_by_case: dict[str, Any] = {}
    cases_processed: list[str] = []
    status = "ok"
    for case_name, case_inventory in inventory.groupby("case_name", dropna=False):
        case_name = str(case_name)
        prior, prior_prov, prior_warnings = reconstruct_initial_observation_prior(run_root, plan, case_name, labels)
        prior_provenance_by_case[case_name] = prior_prov
        warnings.extend(prior_warnings)
        if prior is None:
            status = "missing_prior"
            all_inventory.append(case_inventory)
            if mode == "on":
                raise RuntimeError(f"Cumulative analysis could not reconstruct the initial prior for {case_name}: {prior_prov.get('reason', prior_prov.get('status'))}")
            continue
        truth, truth_prov = recover_truth_by_label(run_root, plan, case_name, labels)
        if not all(np.isfinite(v) for v in truth.values()):
            warnings.append({"status": "missing_truth", "case": case_name, "message": "Truth values could not be recovered for all labels.", "truth_provenance": truth_prov})
        try:
            products = build_cumulative_case_products(run_root, plan, case_name, case_inventory, prior, truth)
        except Exception as exc:
            err_text = str(exc)
            mapped_status = "posterior_solve_error"
            if "label" in err_text.lower():
                mapped_status = "label_mismatch"
            elif "information-scale" in err_text.lower() or "scale" in err_text.lower():
                mapped_status = "information_scale_mismatch"
            elif "No accepted summaries" in err_text:
                mapped_status = "missing_summary_artifact"
            status = mapped_status
            warnings.append({"status": mapped_status, "case": case_name, "message": err_text})
            if mode == "on":
                raise RuntimeError(f"Cumulative analysis failed for {case_name}: {err_text}") from exc
            all_inventory.append(case_inventory)
            continue
        warnings.extend(products["warnings"])
        all_inventory.append(products["inventory"])
        window_frames.append(products["window_summary"])
        posterior_frames.append(products["posterior_by_label"])
        final_frames.append(products["final_summary"])
        variant_frames.append(products["variant_summary"])
        diagnostics_frames.append(products["diagnostics"])
        reference_frames.append(products["reference_audit"])
        likelihood_payload["states_by_case"][case_name] = products["likelihood_state"].to_dict()
        cases_processed.append(case_name)
        if products["accepted_count"] < products["expected_count"]:
            status = "incomplete_window"
    combined_inventory = pd.concat(all_inventory, ignore_index=True) if all_inventory else inventory
    cumulative_window = pd.concat(window_frames, ignore_index=True) if window_frames else pd.DataFrame()
    cumulative_posterior = pd.concat(posterior_frames, ignore_index=True) if posterior_frames else pd.DataFrame()
    cumulative_final = pd.concat(final_frames, ignore_index=True) if final_frames else pd.DataFrame()
    cumulative_variant = pd.concat(variant_frames, ignore_index=True) if variant_frames else pd.DataFrame()
    cumulative_diagnostics = pd.concat(diagnostics_frames, ignore_index=True) if diagnostics_frames else pd.DataFrame()
    cumulative_reference = pd.concat(reference_frames, ignore_index=True) if reference_frames else pd.DataFrame()

    write_csv(combined_inventory, cdir / "cumulative_input_inventory.csv")
    write_csv(cumulative_window, cdir / "cumulative_window_summary.csv")
    write_csv(cumulative_posterior, cdir / "cumulative_posterior_by_label.csv")
    write_csv(cumulative_final, cdir / "cumulative_final_summary.csv")
    write_csv(cumulative_diagnostics, cdir / "cumulative_information_diagnostics.csv")
    write_csv(cumulative_reference, cdir / "cumulative_reference_audit.csv")
    write_csv(cumulative_variant, cdir / "cumulative_variant_summary.csv")
    (cdir / "cumulative_likelihood_state.json").write_text(json.dumps(_json_scalar(likelihood_payload), indent=2))
    plot_paths: list[str] = []
    if not no_plots:
        plot_paths = plot_cumulative_information_outputs(outdir, cumulative_window, review_warnings)
    if not cases_processed and status == "ok":
        status = "missing_summary_artifact"
    payload = _cumulative_status_payload(
        run_root=run_root,
        outdir=outdir,
        mode=mode,
        status=status,
        warnings=warnings,
        outputs={**outputs, "plots": plot_paths},
        cases_processed=cases_processed,
        final_metrics=cumulative_final.to_dict(orient="records") if not cumulative_final.empty else [],
        prior_provenance=prior_provenance_by_case,
        inventory=combined_inventory,
    )
    (cdir / "cumulative_summary.json").write_text(json.dumps(_json_scalar(payload), indent=2))
    review_warnings.extend(warnings)
    if mode == "on" and status != "ok":
        raise RuntimeError(f"Cumulative analysis status is {status}; see {cdir / 'cumulative_summary.json'}.")
    return payload


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


def separation_error_summary(win: pd.DataFrame) -> pd.DataFrame:
    headers = [
        "case_name",
        "initial_sep_err_uas",
        "final_sep_err_uas",
        "initial_abs_sep_err_uas",
        "final_abs_sep_err_uas",
        "final_posterior_sigma_sep_uas",
        "signed_improvement_uas",
        "abs_improvement_uas",
        "final_sep_err_over_sigma",
        "final_abs_sep_err_over_sigma",
    ]
    if win.empty:
        return pd.DataFrame(columns=headers)
    rows: list[dict[str, Any]] = []
    ordered = win.sort_values(["case_name", "window_index"], na_position="last")
    for case_name, group in ordered.groupby("case_name", dropna=False):
        group = group.sort_values("window_index")
        first = group.iloc[0]
        last = group.iloc[-1]
        initial = safe_float(first.get("separation_reference_error_before_microas", np.nan))
        final = safe_float(last.get("separation_next_reference_error_microas", np.nan))
        sigma = safe_float(last.get("posterior_sigma_separation_microas", np.nan))
        initial_abs = abs(initial) if np.isfinite(initial) else np.nan
        final_abs = abs(final) if np.isfinite(final) else np.nan
        rows.append(
            {
                "case_name": case_name,
                "initial_sep_err_uas": initial,
                "final_sep_err_uas": final,
                "initial_abs_sep_err_uas": initial_abs,
                "final_abs_sep_err_uas": final_abs,
                "final_posterior_sigma_sep_uas": sigma,
                "signed_improvement_uas": initial - final if np.isfinite(initial) and np.isfinite(final) else np.nan,
                "abs_improvement_uas": initial_abs - final_abs if np.isfinite(initial_abs) and np.isfinite(final_abs) else np.nan,
                "final_sep_err_over_sigma": final / sigma if np.isfinite(final) and np.isfinite(sigma) and sigma != 0 else np.nan,
                "final_abs_sep_err_over_sigma": final_abs / sigma if np.isfinite(final_abs) and np.isfinite(sigma) and sigma != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=headers)


def slow_parameter_error_summary(param_df: pd.DataFrame) -> pd.DataFrame:
    headers = [
        "case_name",
        "parameter_label",
        "parameter_group",
        "unit",
        "initial_value",
        "truth_value",
        "final_value",
        "initial_err",
        "final_err",
        "initial_abs_err",
        "final_abs_err",
        "abs_improvement",
        "fractional_abs_improvement",
        "final_posterior_sigma",
        "final_err_over_sigma",
        "final_abs_err_over_sigma",
    ]
    if param_df.empty:
        return pd.DataFrame(columns=headers)
    rows: list[dict[str, Any]] = []
    ordered = param_df.sort_values(["case_name", "label", "window_index"], na_position="last")
    for (case_name, label), group in ordered.groupby(["case_name", "label"], dropna=False):
        group = group.sort_values("window_index")
        first = group.iloc[0]
        last = group.iloc[-1]
        truth = safe_float(last.get("truth_value", first.get("truth_value", np.nan)))
        unit, scale, value_offset = standard_parameter_scale(str(label), truth)

        def convert_value(value: Any) -> float:
            f = safe_float(value)
            if not np.isfinite(f):
                return np.nan
            return (f - value_offset) * scale

        def convert_offset(value: Any) -> float:
            f = safe_float(value)
            return f * scale if np.isfinite(f) else np.nan

        initial_value = convert_value(first.get("current_value", np.nan))
        truth_value = convert_value(truth)
        final_value = convert_value(last.get("next_value", np.nan))
        initial_err = convert_offset(first.get("current_offset", np.nan))
        final_err = convert_offset(last.get("next_offset", np.nan))
        sigma = convert_offset(last.get("posterior_sigma", np.nan))
        initial_abs = abs(initial_err) if np.isfinite(initial_err) else np.nan
        final_abs = abs(final_err) if np.isfinite(final_err) else np.nan
        rows.append(
            {
                "case_name": case_name,
                "parameter_label": label,
                "parameter_group": standard_parameter_group(str(label)),
                "unit": unit,
                "initial_value": initial_value,
                "truth_value": truth_value,
                "final_value": final_value,
                "initial_err": initial_err,
                "final_err": final_err,
                "initial_abs_err": initial_abs,
                "final_abs_err": final_abs,
                "abs_improvement": initial_abs - final_abs if np.isfinite(initial_abs) and np.isfinite(final_abs) else np.nan,
                "fractional_abs_improvement": (initial_abs - final_abs) / initial_abs if np.isfinite(initial_abs) and initial_abs > 0 and np.isfinite(final_abs) else np.nan,
                "final_posterior_sigma": sigma,
                "final_err_over_sigma": final_err / sigma if np.isfinite(final_err) and np.isfinite(sigma) and sigma != 0 else np.nan,
                "final_abs_err_over_sigma": final_abs / sigma if np.isfinite(final_abs) and np.isfinite(sigma) and sigma != 0 else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=headers)


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


def classify_local_registration_policy(phi_ref: Any) -> str:
    text = str(phi_ref or "").strip().lower()
    if "truth_when_available" in text or "truth" in text:
        return "truth_phi_or_solve_skipped"
    if text in {"recovered", "starting_guess_csv"} or any(
        token in text for token in ("recovered", "starting_guess", "fit", "solve")
    ):
        return "recovered_registration_solve"
    return "unknown"


def _summary_sidecar(summary_path: Path | None, summary: dict[str, Any], run_root: Path) -> dict[str, Any]:
    explicit = scalar(summary, "schur_summary.artifacts.subblock_summary_json", "")
    context_dirs = _path_parent_candidates(summary_path)
    path = resolve_artifact_path(explicit, run_root, context_dirs) if explicit else None
    if path is None or not path.exists():
        study_root = _subblock_study_root(summary_path, None)
        path = find_existing_candidate_path(
            [
                study_root / "subblock_summary.json" if study_root else None,
                summary_path.parent / "subblock_summary.json" if summary_path else None,
            ]
        )
    data = read_json(path, {}) if path is not None else {}
    return data if isinstance(data, dict) else {}


def _candidate_artifacts(
    summary_path: Path | None,
    diag_path: Path | None,
    summary: dict[str, Any],
    diag: dict[str, Any],
    run_root: Path,
) -> dict[str, Path | None]:
    context_dirs = _path_parent_candidates(summary_path, diag_path)
    study_root = _subblock_study_root(summary_path, diag_path)

    def explicit(*keys: str) -> Path | None:
        for key in keys:
            value = scalar(summary, key, "")
            if _is_blank(value):
                value = scalar(diag, key, "")
            if not _is_blank(value):
                return resolve_artifact_path(value, run_root, context_dirs)
        return None

    runtime_summary = explicit("runtime_profile.summary_json", "runtime_profile_summary_path")
    runtime_timeline = explicit("runtime_profile.timeline_jsonl", "runtime_profile_timeline_path")
    memory_timeline = explicit("memory_diagnostics.timeline_jsonl", "memory_diagnostics_path")
    memory_audit = explicit("memory_diagnostics.audit_json", "memory_audit_path")

    conventional = {
        "runtime_profile_summary_path": [
            runtime_summary,
            study_root / "runtime_profile_summary.json" if study_root else None,
        ],
        "runtime_profile_timeline_path": [
            runtime_timeline,
            study_root / "runtime_profile_timeline.jsonl" if study_root else None,
        ],
        "memory_diagnostics_path": [
            memory_timeline,
            study_root / "schur_summary_memory_timeline.jsonl" if study_root else None,
        ],
        "memory_audit_path": [
            memory_audit,
            study_root / "schur_summary_memory_audit.json" if study_root else None,
        ],
    }
    return {key: find_existing_candidate_path(paths) or paths[0] for key, paths in conventional.items()}


def _peak_memory_from_sources(
    status_row: pd.Series,
    diag: dict[str, Any],
    memory_audit: dict[str, Any],
    memory_rows: list[dict[str, Any]],
) -> tuple[float, str]:
    candidates: list[tuple[float, str]] = []
    key_sources = [
        (status_row.get("resource_time_maximum_resident_set_mb", ""), "status.resource_time"),
        (scalar(diag, "resource_time.maximum_resident_set_mb", ""), "diagnostics.resource_time"),
        (scalar(diag, "memory_sampler.peak_total_rss_mb", ""), "diagnostics.memory_sampler"),
        (scalar(diag, "memory_sampler.peak_descendant_tree_rss_mb", ""), "diagnostics.memory_sampler"),
        (scalar(diag, "memory_sampler.peak_direct_child_rss_mb", ""), "diagnostics.memory_sampler"),
        (memory_audit.get("peak_rss_mb_observed", ""), "memory_audit"),
    ]
    for value, source in key_sources:
        f = safe_float(value)
        if np.isfinite(f):
            candidates.append((f, source))
    for row in memory_rows:
        for key in ("peak_rss_mb", "rss_mb", "last_memory_peak_rss_mb", "last_memory_rss_mb"):
            f = safe_float(row.get(key, ""))
            if np.isfinite(f):
                candidates.append((f, "memory_timeline"))
    if not candidates:
        return np.nan, ""
    return max(candidates, key=lambda item: item[0])


def _resource_time_available(status_row: pd.Series, diag: dict[str, Any]) -> bool:
    if np.isfinite(safe_float(status_row.get("resource_time_maximum_resident_set_mb", ""))):
        return True
    return bool(scalar(diag, "resource_time.available", False)) or np.isfinite(
        safe_float(scalar(diag, "resource_time.maximum_resident_set_mb", ""))
    )


def build_subblock_runtime_ledger(run_root: Path) -> pd.DataFrame:
    status = read_csv(run_root / "subblock_status_iterative.csv")
    rows: list[dict[str, Any]] = []
    for _, s in status.iterrows() if not status.empty else []:
        raw_summary_path = s.get("summary_path", "")
        raw_diag_path = s.get("subprocess_diagnostics_path", "")
        summary_path = resolve_artifact_path(raw_summary_path, run_root)
        context_dirs = _path_parent_candidates(summary_path)
        diag_path = resolve_artifact_path(raw_diag_path, run_root, context_dirs)
        summary = read_json(summary_path, {}) if summary_path is not None else {}
        diag = read_json(diag_path, {}) if diag_path is not None else {}
        summary = summary if isinstance(summary, dict) else {}
        diag = diag if isinstance(diag, dict) else {}
        sidecar = _summary_sidecar(summary_path, summary, run_root)
        artifacts = _candidate_artifacts(summary_path, diag_path, summary, diag, run_root)
        memory_rows = read_jsonl(artifacts.get("memory_diagnostics_path") or Path(""))
        memory_audit_data = read_json(artifacts.get("memory_audit_path") or Path(""), {})
        memory_audit = memory_audit_data if isinstance(memory_audit_data, dict) else {}
        command = diag.get("command", [])
        command = command if isinstance(command, list) else []
        schur = summary.get("schur_summary", {}) if isinstance(summary.get("schur_summary"), dict) else {}
        requested = summary.get("schur_summary_requested", {}) if isinstance(summary.get("schur_summary_requested"), dict) else {}
        sidecar_diag = sidecar.get("diagnostics", {}) if isinstance(sidecar.get("diagnostics"), dict) else {}
        sidecar_meta = sidecar.get("metadata", {}) if isinstance(sidecar.get("metadata"), dict) else {}
        sidecar_curvature = sidecar_meta.get("curvature", {}) if isinstance(sidecar_meta.get("curvature"), dict) else {}
        sidecar_frame_quality = sidecar_meta.get("frame_quality", {}) if isinstance(sidecar_meta.get("frame_quality"), dict) else {}

        elapsed = safe_float(first_nonblank(s.get("elapsed_seconds", ""), diag.get("elapsed_seconds", "")))
        phi_ref = first_nonblank(
            command_arg(command, "--phi-ref"),
            requested.get("phi_ref", ""),
            schur.get("phi_ref_mode", ""),
            sidecar_meta.get("phi_ref_mode", ""),
        )
        initial_loss = first_nonblank(
            scalar(summary, "reference_optimizer.initial_loss", ""),
            _first_nested_value(summary, {"initial_loss"}),
        )
        final_loss = first_nonblank(
            scalar(summary, "reference_optimizer.final_loss", ""),
            _first_nested_value(summary, {"final_loss"}),
        )
        loss_delta = np.nan
        if np.isfinite(safe_float(initial_loss)) and np.isfinite(safe_float(final_loss)):
            loss_delta = safe_float(final_loss) - safe_float(initial_loss)
        peak_mb, peak_source = _peak_memory_from_sources(s, diag, memory_audit, memory_rows)
        resource_rss = first_nonblank(
            s.get("resource_time_maximum_resident_set_mb", ""),
            scalar(diag, "resource_time.maximum_resident_set_mb", ""),
        )
        stdout_path = resolve_artifact_path(first_nonblank(s.get("stdout_log", ""), diag.get("stdout_log", "")), run_root, _path_parent_candidates(diag_path))
        stderr_path = resolve_artifact_path(first_nonblank(s.get("stderr_log", ""), diag.get("stderr_log", "")), run_root, _path_parent_candidates(diag_path))
        theta_dim = first_nonblank(
            schur.get("n_theta", ""),
            sidecar_diag.get("theta_dim", ""),
            len(scalar(sidecar, "theta_labels", [])) if scalar(sidecar, "theta_labels", []) else "",
            len(command_arg(command, "--theta-keys", "").split(",")) if command_arg(command, "--theta-keys", "") else "",
        )
        phi_dim = first_nonblank(schur.get("n_phi", ""), sidecar_diag.get("n_phi", ""), scalar(sidecar_meta, "curvature.n_phi", ""))
        row = {
            "case_name": s.get("case_name", ""),
            "window_index": s.get("window_index", np.nan),
            "window_subblock_index": s.get("window_subblock_index", np.nan),
            "global_subblock_index": s.get("global_subblock_index", np.nan),
            "status": s.get("status", ""),
            "return_code": s.get("return_code", np.nan),
            "summary_path": _artifact_text(summary_path, run_root, raw_summary_path),
            "subprocess_diagnostics_path": _artifact_text(diag_path, run_root, raw_diag_path),
            "stdout_log": _artifact_text(stdout_path, run_root, s.get("stdout_log", "")),
            "stderr_log": _artifact_text(stderr_path, run_root, s.get("stderr_log", "")),
            "runtime_profile_summary_path": _artifact_text(artifacts.get("runtime_profile_summary_path"), run_root),
            "runtime_profile_timeline_path": _artifact_text(artifacts.get("runtime_profile_timeline_path"), run_root),
            "memory_diagnostics_path": _artifact_text(artifacts.get("memory_diagnostics_path"), run_root),
            "memory_audit_path": _artifact_text(artifacts.get("memory_audit_path"), run_root),
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60.0 if np.isfinite(elapsed) else np.nan,
            "elapsed_hours": elapsed / 3600.0 if np.isfinite(elapsed) else np.nan,
            "started_at": diag.get("started_at", ""),
            "finished_at": diag.get("finished_at", ""),
            "subprocess_elapsed_seconds": diag.get("elapsed_seconds", ""),
            "resource_time_elapsed_wall_clock": scalar(diag, "resource_time.elapsed_wall_clock", ""),
            "resource_time_user_cpu_seconds": first_nonblank(scalar(diag, "resource_time.user_cpu_seconds", ""), scalar(diag, "resource_time.user_seconds", "")),
            "resource_time_sys_cpu_seconds": first_nonblank(scalar(diag, "resource_time.system_cpu_seconds", ""), scalar(diag, "resource_time.sys_seconds", "")),
            "resource_time_maximum_resident_set_mb": resource_rss,
            "maximum_resident_set_mb": first_nonblank(s.get("maximum_resident_set_mb", ""), scalar(diag, "maximum_resident_set_mb", "")),
            "max_rss_mb": first_nonblank(s.get("max_rss_mb", ""), scalar(diag, "max_rss_mb", "")),
            "peak_rss_mb": peak_mb,
            "last_memory_rss_mb": first_nonblank(s.get("last_memory_rss_mb", ""), memory_audit.get("last_memory_rss_mb", "")),
            "last_memory_peak_rss_mb": first_nonblank(s.get("last_memory_peak_rss_mb", ""), memory_audit.get("last_memory_peak_rss_mb", "")),
            "memory_peak_source": peak_source,
            "n_frames": first_nonblank(schur.get("n_frames", ""), scalar(schur, "information_accounting.n_frames_total", ""), summary.get("n_frames_requested", ""), command_arg(command, "--n-frames")),
            "exposure_time_s": first_nonblank(summary.get("exposure_time_s_requested", ""), command_arg(command, "--exposure-time-s")),
            "phi_ref": phi_ref,
            "schur_curvature_method_requested": first_nonblank(schur.get("schur_curvature_method_requested", ""), sidecar_diag.get("schur_curvature_method_requested", ""), sidecar_curvature.get("schur_curvature_method_requested", ""), requested.get("schur_curvature_method", ""), command_arg(command, "--schur-curvature-method")),
            "schur_curvature_method_used": first_nonblank(schur.get("schur_curvature_method_used", ""), sidecar_diag.get("schur_curvature_method_used", ""), sidecar_curvature.get("schur_curvature_method_used", "")),
            "summary_information_scale": first_nonblank(scalar(schur, "information_accounting.summary_information_scale", ""), scalar(sidecar_diag, "information_accounting.summary_information_scale", ""), requested.get("summary_information_scale", "")),
            "theta_dim": theta_dim,
            "phi_dim": phi_dim,
            "combined_dim": first_nonblank(schur.get("combined_dim", ""), sidecar.get("combined_dim", ""), sidecar_curvature.get("combined_dim", "")),
            "dense_global_hessian_materialized": _as_bool(first_nonblank(schur.get("dense_global_hessian_materialized", ""), sidecar_diag.get("dense_global_hessian_materialized", ""), sidecar_curvature.get("dense_global_hessian_materialized", ""))),
            "structured_curvature_used": _as_bool(first_nonblank(schur.get("structured_curvature_used", ""), sidecar_diag.get("structured_curvature_used", ""), sidecar_curvature.get("structured_curvature_used", ""))),
            "reference_optimizer_kind": first_nonblank(scalar(requested, "reference_inference_cli_overrides.reference_optimizer_kind", ""), scalar(summary, "schur_summary_plan.reference_inference_config_if_run.optimizer_kind", ""), command_arg(command, "--reference-optimizer-kind")),
            "reference_optimizer_n_iter_requested": first_nonblank(scalar(requested, "reference_inference_cli_overrides.reference_n_iter", ""), scalar(summary, "schur_summary_plan.reference_inference_config_if_run.n_iter", ""), command_arg(command, "--reference-n-iter")),
            "reference_optimizer_n_iter_actual": first_nonblank(scalar(summary, "reference_optimizer.n_iter", ""), _first_nested_value(summary, {"n_iter_actual", "iterations_completed"})),
            "early_stopping_triggered": scalar(summary, "reference_optimizer.early_stopping_triggered", ""),
            "reference_optimizer_converged": scalar(summary, "reference_optimizer.converged", ""),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_delta": loss_delta,
            "chi2": first_nonblank(scalar(summary, "fit_diagnostics.chi2", ""), _first_nested_value(summary, {"chi2"})),
            "reduced_chi2": first_nonblank(scalar(summary, "fit_diagnostics.reduced_chi2", ""), _first_nested_value(summary, {"reduced_chi2"})),
            "included_frame_count": first_nonblank(scalar(schur, "information_accounting.included_frame_count", ""), sidecar_frame_quality.get("included_frame_count", ""), sidecar_diag.get("frame_quality_good_frame_count", "")),
            "bad_frame_count": first_nonblank(schur.get("frame_quality_bad_frame_count", ""), sidecar_frame_quality.get("bad_frame_count", ""), sidecar_diag.get("frame_quality_bad_frame_count", "")),
            "frame_quality_policy": first_nonblank(schur.get("frame_quality_policy", ""), sidecar_frame_quality.get("policy", ""), sidecar_diag.get("frame_quality_policy", ""), requested.get("schur_frame_quality_policy", "")),
            "local_registration_policy_class": classify_local_registration_policy(phi_ref),
            "profile_available": bool((artifacts.get("runtime_profile_summary_path") and artifacts["runtime_profile_summary_path"].exists()) or (artifacts.get("runtime_profile_timeline_path") and artifacts["runtime_profile_timeline_path"].exists())),
            "memory_diagnostics_available": bool((artifacts.get("memory_diagnostics_path") and artifacts["memory_diagnostics_path"].exists()) or (artifacts.get("memory_audit_path") and artifacts["memory_audit_path"].exists())),
            "resource_time_available": _resource_time_available(s, diag),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _join_modes(series: pd.Series) -> str:
    vals = sorted({str(v) for v in series.dropna().tolist() if str(v).strip()})
    return ";".join(vals)


def build_window_runtime_summary(run_root: Path, ledger: pd.DataFrame, campaign_summary: dict[str, Any]) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=[
            "case_name", "window_index", "n_subblocks_planned", "n_subblocks_with_status",
            "n_subblocks_completed", "n_subblocks_failed", "n_subblocks_missing",
            "subblock_elapsed_seconds_sum", "subblock_elapsed_seconds_min",
            "subblock_elapsed_seconds_median", "subblock_elapsed_seconds_max",
            "subblock_elapsed_seconds_mean", "approx_parallel_wall_seconds_lower_bound",
            "approx_parallel_wall_seconds_observed_span", "resource_time_peak_rss_mb_max",
            "memory_peak_rss_mb_max", "theta_dim_median", "phi_dim_median",
            "n_frames_median", "phi_ref_modes", "schur_methods_used",
            "reference_solve_classes", "posterior_table_exists",
            "iterative_reference_update_exists", "window_diagnostics_exists",
        ])
    max_workers = _first_nested_value(campaign_summary, {"max_workers", "MAX_WORKERS"})
    rows: list[dict[str, Any]] = []
    planned_default = safe_int(campaign_summary.get("subblocks_per_window", ""))
    for (case_name, window_index), group in ledger.groupby(["case_name", "window_index"], dropna=False):
        elapsed = pd.to_numeric(group.get("elapsed_seconds", pd.Series(dtype=float)), errors="coerce")
        starts = [t for t in group.get("started_at", pd.Series(dtype=str)).tolist() if _parse_timestamp(t)]
        finishes = [t for t in group.get("finished_at", pd.Series(dtype=str)).tolist() if _parse_timestamp(t)]
        observed_span = np.nan
        if starts and finishes:
            observed_span = _seconds_span(min(starts, key=lambda x: _parse_timestamp(x) or datetime.max), max(finishes, key=lambda x: _parse_timestamp(x) or datetime.min))
        completed = int((group.get("status", pd.Series(dtype=str)).astype(str) == "ok").sum())
        failed = int((group.get("status", pd.Series(dtype=str)).astype(str) == "failed").sum())
        planned = planned_default if np.isfinite(safe_float(planned_default)) else len(group)
        median_elapsed = float(elapsed.median()) if elapsed.notna().any() else np.nan
        parallel_est = np.nan
        mw = safe_int(max_workers)
        if np.isfinite(safe_float(mw)) and int(mw) > 0 and np.isfinite(median_elapsed):
            parallel_est = math.ceil(len(group) / int(mw)) * median_elapsed
        window_root = run_root / "cases" / str(case_name) / "windows" / f"window_{int(window_index):03d}" if not pd.isna(window_index) else Path("")
        if not window_root.exists() and not pd.isna(window_index):
            window_root = run_root / "cases" / str(case_name) / "windows" / f"window_{int(window_index)}"
        rows.append(
            {
                "case_name": case_name,
                "window_index": window_index,
                "n_subblocks_planned": planned,
                "n_subblocks_with_status": int(len(group)),
                "n_subblocks_completed": completed,
                "n_subblocks_failed": failed,
                "n_subblocks_missing": max(0, int(planned) - int(len(group))) if np.isfinite(safe_float(planned)) else np.nan,
                "subblock_elapsed_seconds_sum": float(elapsed.sum(skipna=True)) if elapsed.notna().any() else np.nan,
                "subblock_elapsed_seconds_min": float(elapsed.min()) if elapsed.notna().any() else np.nan,
                "subblock_elapsed_seconds_median": median_elapsed,
                "subblock_elapsed_seconds_max": float(elapsed.max()) if elapsed.notna().any() else np.nan,
                "subblock_elapsed_seconds_mean": float(elapsed.mean()) if elapsed.notna().any() else np.nan,
                "approx_parallel_wall_seconds_lower_bound": float(elapsed.max()) if elapsed.notna().any() else np.nan,
                "approx_parallel_wall_seconds_observed_span": observed_span,
                "approx_parallel_wall_seconds_estimated_from_max_workers": parallel_est,
                "max_workers_for_estimate": max_workers,
                "resource_time_peak_rss_mb_max": float(pd.to_numeric(group.get("resource_time_maximum_resident_set_mb", pd.Series(dtype=float)), errors="coerce").max()) if len(group) else np.nan,
                "memory_peak_rss_mb_max": float(pd.to_numeric(group.get("peak_rss_mb", pd.Series(dtype=float)), errors="coerce").max()) if len(group) else np.nan,
                "theta_dim_median": float(pd.to_numeric(group.get("theta_dim", pd.Series(dtype=float)), errors="coerce").median()) if len(group) else np.nan,
                "phi_dim_median": float(pd.to_numeric(group.get("phi_dim", pd.Series(dtype=float)), errors="coerce").median()) if len(group) else np.nan,
                "n_frames_median": float(pd.to_numeric(group.get("n_frames", pd.Series(dtype=float)), errors="coerce").median()) if len(group) else np.nan,
                "phi_ref_modes": _join_modes(group.get("phi_ref", pd.Series(dtype=str))),
                "schur_methods_used": _join_modes(group.get("schur_curvature_method_used", pd.Series(dtype=str))),
                "reference_solve_classes": _join_modes(group.get("local_registration_policy_class", pd.Series(dtype=str))),
                "posterior_table_exists": bool((window_root / "posterior_by_label.csv").exists()) if window_root else False,
                "iterative_reference_update_exists": bool((window_root / "iterative_reference_update.json").exists()) if window_root else False,
                "window_diagnostics_exists": bool((window_root / "iterative_window_diagnostics.csv").exists()) if window_root else False,
            }
        )
    return pd.DataFrame(rows).sort_values(["case_name", "window_index"], na_position="last")


def build_campaign_runtime_summary(
    ledger: pd.DataFrame,
    window_summary: pd.DataFrame,
    campaign_summary: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(metric: str, value: Any) -> None:
        rows.append({"metric": metric, "value": compact(value, max_len=500)})

    n = len(ledger)
    elapsed = pd.to_numeric(ledger.get("elapsed_seconds", pd.Series(dtype=float)), errors="coerce") if not ledger.empty else pd.Series(dtype=float)
    add("total_completed_subblocks", int((ledger.get("status", pd.Series(dtype=str)).astype(str) == "ok").sum()) if not ledger.empty else 0)
    add("total_failed_subblocks", int((ledger.get("status", pd.Series(dtype=str)).astype(str) == "failed").sum()) if not ledger.empty else 0)
    add("total_missing_subblocks", campaign_summary.get("missing_output_rows", ""))
    add("total_windows", len(window_summary))
    add("total_subblock_elapsed_seconds", float(elapsed.sum(skipna=True)) if elapsed.notna().any() else "")
    add("total_subblock_elapsed_minutes", float(elapsed.sum(skipna=True) / 60.0) if elapsed.notna().any() else "")
    add("total_subblock_elapsed_hours", float(elapsed.sum(skipna=True) / 3600.0) if elapsed.notna().any() else "")
    for label, divisor in (("seconds", 1.0), ("minutes", 60.0), ("hours", 3600.0)):
        add(f"median_subblock_elapsed_{label}", float(elapsed.median() / divisor) if elapsed.notna().any() else "")
        add(f"max_subblock_elapsed_{label}", float(elapsed.max() / divisor) if elapsed.notna().any() else "")
    peak = pd.to_numeric(ledger.get("peak_rss_mb", pd.Series(dtype=float)), errors="coerce") if not ledger.empty else pd.Series(dtype=float)
    add("max_observed_memory_rss_mb", float(peak.max()) if peak.notna().any() else "")
    for col, label in (
        ("resource_time_available", "rows_with_resource_time_memory"),
        ("profile_available", "rows_with_runtime_profiles"),
        ("memory_diagnostics_available", "rows_with_memory_diagnostics"),
    ):
        count = int(ledger.get(col, pd.Series(dtype=bool)).astype(bool).sum()) if not ledger.empty else 0
        add(label, count)
        add(f"{label}_fraction", count / n if n else "")
    add("breakdown_by_local_registration_policy_class", dict(Counter(ledger.get("local_registration_policy_class", []))) if not ledger.empty else {})
    add("breakdown_by_schur_method", dict(Counter(ledger.get("schur_curvature_method_used", []))) if not ledger.empty else {})
    status_rc = Counter(
        f"{row.get('status', '')}:{row.get('return_code', '')}"
        for _, row in ledger.iterrows()
    ) if not ledger.empty else {}
    add("breakdown_by_status_return_code", dict(status_rc))
    add("windows_per_draw", first_nonblank(campaign_summary.get("windows_per_draw", ""), len(window_summary) if len(window_summary) else ""))
    add("subblocks_per_window", first_nonblank(campaign_summary.get("subblocks_per_window", ""), float(pd.to_numeric(window_summary.get("n_subblocks_with_status", pd.Series(dtype=float)), errors="coerce").median()) if not window_summary.empty else ""))
    add("n_frames_per_subblock", float(pd.to_numeric(ledger.get("n_frames", pd.Series(dtype=float)), errors="coerce").median()) if not ledger.empty else "")
    add("theta_dim", float(pd.to_numeric(ledger.get("theta_dim", pd.Series(dtype=float)), errors="coerce").median()) if not ledger.empty else "")
    add("phi_dim", float(pd.to_numeric(ledger.get("phi_dim", pd.Series(dtype=float)), errors="coerce").median()) if not ledger.empty else "")
    serial_by_window = pd.to_numeric(window_summary.get("subblock_elapsed_seconds_sum", pd.Series(dtype=float)), errors="coerce") if not window_summary.empty else pd.Series(dtype=float)
    add("estimated_serial_subblock_work_per_window_seconds", float(serial_by_window.median()) if serial_by_window.notna().any() else "")
    add("estimated_serial_subblock_work_full_campaign_seconds", float(elapsed.sum(skipna=True)) if elapsed.notna().any() else "")
    parallel = pd.to_numeric(window_summary.get("approx_parallel_wall_seconds_estimated_from_max_workers", pd.Series(dtype=float)), errors="coerce") if not window_summary.empty else pd.Series(dtype=float)
    add("estimated_parallel_wall_per_window_seconds_from_max_workers", float(parallel.median()) if parallel.notna().any() else "")
    caveats = []
    if n and int(ledger.get("profile_available", pd.Series(dtype=bool)).astype(bool).sum()) < n:
        caveats.append("Some subblocks lack runtime profile artifacts; stage-level timing is incomplete.")
    if n and int(ledger.get("memory_diagnostics_available", pd.Series(dtype=bool)).astype(bool).sum()) < n:
        caveats.append("Some subblocks lack memory diagnostics artifacts; peak memory uses available resource-time/status fields where possible.")
    if n and int(ledger.get("resource_time_available", pd.Series(dtype=bool)).astype(bool).sum()) < n:
        caveats.append("Some subblocks lack external resource-time memory fields.")
    add("caveats", caveats)
    return pd.DataFrame(rows)


def _profile_records_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x if isinstance(x, dict) else {"value": x} for x in payload]
    if not isinstance(payload, dict):
        return []
    if isinstance(payload.get("events"), list):
        return [x if isinstance(x, dict) else {"value": x} for x in payload["events"]]
    if isinstance(payload.get("stages"), list):
        return [x if isinstance(x, dict) else {"value": x} for x in payload["stages"]]
    if isinstance(payload.get("stages"), dict):
        return [{"stage": k, **(v if isinstance(v, dict) else {"elapsed_seconds": v})} for k, v in payload["stages"].items()]
    if isinstance(payload.get("stage_totals"), dict):
        return [{"stage": k, "elapsed_seconds": v} for k, v in payload["stage_totals"].items()]
    if payload.get("stage") or payload.get("elapsed_seconds") or payload.get("duration_s"):
        return [payload]
    return []


def _stage_row(base: dict[str, Any], record: dict[str, Any], source_path: Path, run_root: Path, min_start: float | None) -> dict[str, Any]:
    started = safe_float(record.get("started_at_unix", ""))
    finished = safe_float(record.get("finished_at_unix", ""))
    raw = {k: v for k, v in record.items() if k not in {"stage", "duration_s", "elapsed_seconds", "started_at_unix", "finished_at_unix", "details", "category", "cacheability"}}
    details = record.get("details", {}) if isinstance(record.get("details"), dict) else {}
    return {
        **base,
        "stage": first_nonblank(record.get("stage", ""), record.get("name", ""), record.get("category", ""), "unknown"),
        "elapsed_seconds": first_nonblank(record.get("elapsed_seconds", ""), record.get("duration_s", ""), record.get("duration_seconds", "")),
        "start_offset_seconds": (started - min_start) if min_start is not None and np.isfinite(started) else "",
        "end_offset_seconds": (finished - min_start) if min_start is not None and np.isfinite(finished) else "",
        "detail_level": first_nonblank(record.get("detail_level", ""), details.get("detail_level", ""), details.get("profile_runtime_detail", "")),
        "source_path": rel(source_path, run_root),
        "raw_profile_compact": compact(raw, max_len=500),
    }


def build_stage_profile_summary(run_root: Path, ledger: pd.DataFrame) -> pd.DataFrame:
    headers = ["case_name", "window_index", "window_subblock_index", "global_subblock_index", "stage", "elapsed_seconds", "start_offset_seconds", "end_offset_seconds", "detail_level", "source_path", "raw_profile_compact"]
    rows: list[dict[str, Any]] = []
    for _, item in ledger.iterrows() if not ledger.empty else []:
        base = {k: item.get(k, "") for k in ["case_name", "window_index", "window_subblock_index", "global_subblock_index"]}
        for path_text in [item.get("runtime_profile_summary_path", ""), item.get("runtime_profile_timeline_path", "")]:
            path = resolve_artifact_path(path_text, run_root)
            if path is None or not path.exists():
                continue
            payload: Any = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path, {})
            records = _profile_records_from_payload(payload)
            starts = [safe_float(r.get("started_at_unix", "")) for r in records]
            starts = [x for x in starts if np.isfinite(x)]
            min_start = min(starts) if starts else None
            if records:
                rows.extend(_stage_row(base, record, path, run_root, min_start) for record in records)
            else:
                rows.append({**base, "stage": "unparsed_profile", "elapsed_seconds": "", "start_offset_seconds": "", "end_offset_seconds": "", "detail_level": "", "source_path": rel(path, run_root), "raw_profile_compact": compact(payload, max_len=500)})
    return pd.DataFrame(rows, columns=headers)


def _array_bytes_total(value: Any) -> float:
    total = 0.0
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in {"bytes", "nbytes", "size_bytes", "array_bytes", "array_bytes_total"}:
                f = safe_float(item)
                if np.isfinite(f):
                    total += f
            else:
                total += _array_bytes_total(item)
    elif isinstance(value, list):
        for item in value:
            total += _array_bytes_total(item)
    return total


def build_memory_timeline_summary(run_root: Path, ledger: pd.DataFrame) -> pd.DataFrame:
    headers = ["case_name", "window_index", "window_subblock_index", "global_subblock_index", "stage", "rss_mb", "peak_rss_mb", "tracemalloc_current_mb", "tracemalloc_peak_mb", "array_bytes_total", "array_mb_total", "source_path"]
    rows: list[dict[str, Any]] = []
    for _, item in ledger.iterrows() if not ledger.empty else []:
        base = {k: item.get(k, "") for k in ["case_name", "window_index", "window_subblock_index", "global_subblock_index"]}
        for path_text in [item.get("memory_diagnostics_path", ""), item.get("memory_audit_path", "")]:
            path = resolve_artifact_path(path_text, run_root)
            if path is None or not path.exists():
                continue
            records = read_jsonl(path) if path.suffix == ".jsonl" else [read_json(path, {})]
            for record in records:
                if not isinstance(record, dict):
                    continue
                array_bytes = first_nonblank(record.get("array_bytes_total", ""), _array_bytes_total(record.get("arrays", {})))
                f_array_bytes = safe_float(array_bytes)
                rows.append(
                    {
                        **base,
                        "stage": first_nonblank(record.get("stage", ""), record.get("last_stage", ""), "audit" if path.suffix == ".json" else ""),
                        "rss_mb": first_nonblank(record.get("rss_mb", ""), record.get("last_memory_rss_mb", "")),
                        "peak_rss_mb": first_nonblank(record.get("peak_rss_mb", ""), record.get("peak_rss_mb_observed", ""), record.get("last_memory_peak_rss_mb", "")),
                        "tracemalloc_current_mb": record.get("tracemalloc_current_mb", ""),
                        "tracemalloc_peak_mb": record.get("tracemalloc_peak_mb", ""),
                        "array_bytes_total": array_bytes,
                        "array_mb_total": f_array_bytes / (1024.0 * 1024.0) if np.isfinite(f_array_bytes) else "",
                        "source_path": rel(path, run_root),
                    }
                )
    return pd.DataFrame(rows, columns=headers)


def write_runtime_accounting_outputs(
    run_root: Path,
    outdir: Path,
    campaign_summary: dict[str, Any],
) -> dict[str, Any]:
    runtime_dir = outdir / "runtime_accounting"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    ledger = build_subblock_runtime_ledger(run_root)
    window_summary = build_window_runtime_summary(run_root, ledger, campaign_summary)
    campaign_runtime = build_campaign_runtime_summary(ledger, window_summary, campaign_summary)
    stage_profile = build_stage_profile_summary(run_root, ledger)
    memory_timeline = build_memory_timeline_summary(run_root, ledger)
    outputs = {
        "subblock_runtime_ledger.csv": ledger,
        "window_runtime_summary.csv": window_summary,
        "campaign_runtime_summary.csv": campaign_runtime,
        "stage_profile_summary.csv": stage_profile,
        "memory_timeline_summary.csv": memory_timeline,
    }
    for name, df in outputs.items():
        write_csv(df, runtime_dir / name)
    n = len(ledger)
    summary_payload = {
        "paths": {name: rel(runtime_dir / name, outdir) for name in outputs},
        "overall_metrics": {
            "subblock_rows": int(n),
            "window_rows": int(len(window_summary)),
            "stage_profile_rows": int(len(stage_profile)),
            "memory_timeline_rows": int(len(memory_timeline)),
            "total_subblock_elapsed_seconds": safe_float(pd.to_numeric(ledger.get("elapsed_seconds", pd.Series(dtype=float)), errors="coerce").sum()) if n else 0.0,
        },
        "missing_artifact_counts": {
            "runtime_profile": int(n - ledger.get("profile_available", pd.Series(dtype=bool)).astype(bool).sum()) if n else 0,
            "memory_diagnostics": int(n - ledger.get("memory_diagnostics_available", pd.Series(dtype=bool)).astype(bool).sum()) if n else 0,
            "resource_time": int(n - ledger.get("resource_time_available", pd.Series(dtype=bool)).astype(bool).sum()) if n else 0,
        },
        "source_schema_notes": [
            "Subprocess timing and resource-time fields are read from subprocess_diagnostics.json when present.",
            "Stage profile parsing supports runtime profile summary JSON, timeline JSONL, events, stages, and stage_totals.",
            "Memory parsing supports Schur summary memory timeline JSONL and audit JSON with nested array byte totals.",
        ],
        "caveats": [
            "Existing campaigns may lack stage-level runtime and memory artifacts unless profiling, memory diagnostics, and/or resource-time were enabled.",
            "Parallel wall estimates use observed timestamps when available and otherwise remain blank unless max_workers is discoverable.",
        ],
    }
    summary_path = runtime_dir / "runtime_accounting_summary.json"
    summary_payload["paths"]["runtime_accounting_summary.json"] = rel(summary_path, outdir)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    return summary_payload


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
        plot_separation_evolution_standard(win, plotdir, warnings_list)
    if not param.empty:
        pivot_plot(param, "parameter_offsets_by_window.png", plotdir, value="next_offset", ylabel="next offset")
        pivot_plot(param, "posterior_sigma_by_parameter.png", plotdir, value="posterior_sigma", ylabel="posterior sigma", bar=True)
        z = param[param["label"].str.contains("zernike", na=False)]
        if not z.empty:
            pivot_plot(z, "zernike_m1_m2_offsets.png", plotdir, value="next_offset", ylabel="Zernike next offset (nm)")
        plot_slow_parameter_standard(param, plotdir, warnings_list)
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


def separation_evolution_points(win: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if win.empty:
        return pd.DataFrame(columns=["case_name", "step", "sep_err_uas"])
    for case_name, group in win.sort_values(["case_name", "window_index"], na_position="last").groupby("case_name", dropna=False):
        group = group.sort_values("window_index")
        if group.empty:
            continue
        initial = safe_float(group.iloc[0].get("separation_reference_error_before_microas", np.nan))
        rows.append({"case_name": case_name, "step": 0.0, "sep_err_uas": initial})
        for _, r in group.iterrows():
            window = safe_float(r.get("window_index", np.nan))
            err = first_nonblank(
                r.get("separation_next_reference_error_microas", np.nan),
                r.get("separation_posterior_error_after_microas", np.nan),
            )
            step = window + 1.0 if np.isfinite(window) else np.nan
            rows.append({"case_name": case_name, "step": step, "sep_err_uas": safe_float(err)})
    return pd.DataFrame(rows)


def plot_separation_evolution_standard(
    win: pd.DataFrame,
    plotdir: Path,
    warnings_list: list[dict[str, Any]] | None = None,
) -> None:
    points = separation_evolution_points(win)
    if points.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_values = False
    for case_name, group in points.groupby("case_name", dropna=False):
        ordered = group.sort_values("step")
        y = np.abs(pd.to_numeric(ordered["sep_err_uas"], errors="coerce"))
        y = y.where(y > 0)
        if y.notna().any():
            any_values = True
        ax.plot(pd.to_numeric(ordered["step"], errors="coerce"), y, marker="o", linewidth=1.2, label=str(case_name))
    if not any_values:
        _placeholder_plot(ax, "no positive finite separation errors", warnings_list=warnings_list, context="separation_error_evolution_abs_log")
    ax.set_yscale("log")
    ax.set_xlabel("Window step")
    ax.set_ylabel("Absolute separation error (uas)")
    ax.grid(True, alpha=0.25, which="both")
    if points["case_name"].nunique(dropna=False) <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plotdir / "separation_error_evolution_abs_log.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for case_name, group in points.groupby("case_name", dropna=False):
        ordered = group.sort_values("step")
        ax.plot(
            pd.to_numeric(ordered["step"], errors="coerce"),
            pd.to_numeric(ordered["sep_err_uas"], errors="coerce"),
            marker="o",
            linewidth=1.2,
            label=str(case_name),
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("Window step")
    ax.set_ylabel("Signed separation error (uas)")
    ax.grid(True, alpha=0.25, which="both")
    if points["case_name"].nunique(dropna=False) <= 12:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plotdir / "separation_error_evolution_signed_symlog.png", dpi=150)
    plt.close(fig)


def _symlog_linthresh(unit: str) -> float:
    if unit == "uas":
        return 1.0
    if unit == "nm":
        return 0.01
    if unit == "ppm":
        return 0.1
    return 1e-3


def standard_parameter_progress(param: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if param.empty:
        return pd.DataFrame()
    for _, r in param.iterrows():
        label = str(r.get("label", ""))
        truth = safe_float(r.get("truth_value", np.nan))
        unit, scale, _ = standard_parameter_scale(label, truth)
        next_offset = safe_float(r.get("next_offset", np.nan))
        sigma = safe_float(r.get("posterior_sigma", np.nan))
        rows.append(
            {
                "case_name": r.get("case_name", ""),
                "window_index": r.get("window_index", np.nan),
                "parameter_label": label,
                "parameter_group": standard_parameter_group(label),
                "unit": unit,
                "final_err": next_offset * scale if np.isfinite(next_offset) else np.nan,
                "final_posterior_sigma": sigma * scale if np.isfinite(sigma) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_slow_parameter_standard(
    param: pd.DataFrame,
    plotdir: Path,
    warnings_list: list[dict[str, Any]] | None = None,
) -> None:
    summary = slow_parameter_error_summary(param)
    progress = standard_parameter_progress(param)
    if summary.empty:
        return
    latest = summary.sort_values(["parameter_group", "parameter_label"])
    width = min(16, max(8, 0.35 * len(latest)))
    fig, ax = plt.subplots(figsize=(width, 5))
    labels = latest["parameter_label"].astype(str).tolist()
    ax.bar(labels, pd.to_numeric(latest["final_err"], errors="coerce"))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Final signed error (natural unit)")
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plotdir / "slow_parameter_final_error_bar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(labels, pd.to_numeric(latest["final_err_over_sigma"], errors="coerce"))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Final signed error / posterior sigma")
    ax.tick_params(axis="x", rotation=70, labelsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plotdir / "slow_parameter_final_error_over_sigma_bar.png", dpi=150)
    plt.close(fig)

    if not progress.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, ldf in progress.groupby("parameter_label", dropna=False):
            ax.plot(
                pd.to_numeric(ldf["window_index"], errors="coerce"),
                pd.to_numeric(ldf["final_err"], errors="coerce"),
                marker="o",
                linewidth=1.0,
                label=str(label),
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("Window")
        ax.set_ylabel("Signed error (natural unit)")
        ax.grid(True, alpha=0.25, which="both")
        if progress["parameter_label"].nunique(dropna=False) <= 14:
            ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(plotdir / "slow_parameter_evolution_signed_symlog.png", dpi=150)
        plt.close(fig)

        for group_name, group_df in progress.groupby("parameter_group", dropna=False):
            fig, ax = plt.subplots(figsize=(9, 4.8))
            units = [u for u in group_df.get("unit", pd.Series(dtype=str)).dropna().astype(str).unique() if u]
            linthresh = _symlog_linthresh(units[0] if len(units) == 1 else "")
            for label, ldf in group_df.groupby("parameter_label", dropna=False):
                ax.plot(
                    pd.to_numeric(ldf["window_index"], errors="coerce"),
                    pd.to_numeric(ldf["final_err"], errors="coerce"),
                    marker="o",
                    linewidth=1.1,
                    label=str(label),
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_yscale("symlog", linthresh=linthresh)
            ax.set_xlabel("Window")
            ylabel_unit = units[0] if len(units) == 1 else "natural unit"
            ax.set_ylabel(f"Signed error ({ylabel_unit})")
            ax.grid(True, alpha=0.25, which="both")
            if group_df["parameter_label"].nunique(dropna=False) <= 12:
                ax.legend(fontsize=7, ncol=2)
            fig.tight_layout()
            safe_group = str(group_name).replace(".", "_").replace("/", "_")
            fig.savefig(plotdir / f"slow_parameter_evolution_{safe_group}_signed_symlog.png", dpi=150)
            plt.close(fig)

    z = summary[summary["parameter_group"].isin(["m1_zernike", "m2_zernike"])]
    if not z.empty:
        fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(z)), 4.8))
        z = z.sort_values(["parameter_group", "parameter_label"])
        z_labels = [
            str(row.parameter_label).replace("optics.primary.", "m1.").replace("optics.secondary.", "m2.")
            for row in z.itertuples()
        ]
        ax.bar(z_labels, pd.to_numeric(z["final_err"], errors="coerce"))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Final signed Zernike error (nm)")
        ax.tick_params(axis="x", rotation=70, labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plotdir / "zernike_final_error_by_mirror.png", dpi=150)
        plt.close(fig)
    if not progress.empty:
        zp = progress[progress["parameter_group"].isin(["m1_zernike", "m2_zernike"])]
        if not zp.empty:
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for label, ldf in zp.groupby("parameter_label", dropna=False):
                ax.plot(
                    pd.to_numeric(ldf["window_index"], errors="coerce"),
                    pd.to_numeric(ldf["final_err"], errors="coerce"),
                    marker="o",
                    linewidth=1.0,
                    label=str(label).replace("optics.primary.", "m1.").replace("optics.secondary.", "m2."),
                )
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_yscale("symlog", linthresh=0.01)
            ax.set_xlabel("Window")
            ax.set_ylabel("Signed Zernike error (nm)")
            ax.grid(True, alpha=0.25, which="both")
            if zp["parameter_label"].nunique(dropna=False) <= 16:
                ax.legend(fontsize=6, ncol=2)
            fig.tight_layout()
            fig.savefig(plotdir / "zernike_error_evolution_signed_symlog.png", dpi=150)
            plt.close(fig)


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
            ax.grid(True, alpha=0.25)
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
            ax.grid(True, alpha=0.25)
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


def write_report(
    outdir: Path,
    run_root: Path,
    summary: dict[str, Any],
    win: pd.DataFrame,
    final: pd.DataFrame,
    policy: str,
    image_plots: list[str],
    final_forecast: pd.DataFrame,
    cumulative_summary: dict[str, Any] | None = None,
) -> None:
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
        "## Runtime and memory accounting",
        "See [campaign_runtime_summary.csv](runtime_accounting/campaign_runtime_summary.csv), [window_runtime_summary.csv](runtime_accounting/window_runtime_summary.csv), [subblock_runtime_ledger.csv](runtime_accounting/subblock_runtime_ledger.csv), [stage_profile_summary.csv](runtime_accounting/stage_profile_summary.csv), and [memory_timeline_summary.csv](runtime_accounting/memory_timeline_summary.csv).",
        "Existing campaigns may lack stage-level runtime and memory artifacts unless they were run with runtime profiling, memory diagnostics, and/or resource-time enabled.",
        "",
        "## Iterative progress",
        "See [iterative_window_progress.csv](iterative_window_progress.csv), [separation_error_summary.csv](separation_error_summary.csv), and plots [separation_error_evolution_abs_log.png](plots/separation_error_evolution_abs_log.png), [separation_error_evolution_signed_symlog.png](plots/separation_error_evolution_signed_symlog.png), [iterative_error_norm.png](plots/iterative_error_norm.png), [update_alignment_by_window.png](plots/update_alignment_by_window.png), and [separation_error_by_window.png](plots/separation_error_by_window.png).",
        "",
        "## Projected 30-minute observation forecast",
        "See [final_observation_summary.csv](final_observation_summary.csv), [projected_observation_forecast.csv](projected_observation_forecast.csv), and [window_evolution_actual_and_projected.csv](window_evolution_actual_and_projected.csv). These results are projected from the realized actual windows, not from a fully rendered 60-window observation.",
        "",
        "## Cumulative information",
    ]
    cumulative_summary = cumulative_summary or {}
    cumulative_status = cumulative_summary.get("status", "not_run")
    if cumulative_status == "ok" and cumulative_summary.get("final_cumulative_metrics"):
        metric = cumulative_summary["final_cumulative_metrics"][0]
        lines += [
            "The cumulative posterior combines preserved Schur likelihood summaries from all accepted windows with the initial observation prior counted once. It is a retrospective science estimate and is distinct from the historical damped next-reference state.",
            f"- Status: {cumulative_status}; accepted {cumulative_summary.get('accepted_summary_count', 0)} of {cumulative_summary.get('expected_summary_count', 0)} discovered summaries.",
            f"- Prior source: {compact(cumulative_summary.get('initial_prior_provenance', {}), max_len=300)}",
            f"- Final cumulative separation error: {safe_float(metric.get('cumulative_final_sep_err_uas')):.6g} uas.",
            f"- Final cumulative separation sigma: {safe_float(metric.get('cumulative_final_posterior_sigma_sep_uas')):.6g} uas.",
            f"- Final cumulative error/sigma: {safe_float(metric.get('cumulative_final_sep_err_over_sigma')):.6g}.",
            f"- Final window-local separation error/sigma: {safe_float(metric.get('window_local_final_sep_err_uas')):.6g} uas / {safe_float(metric.get('window_local_final_posterior_sigma_sep_uas')):.6g} uas.",
            f"- Historical final next-reference separation error: {safe_float(metric.get('historical_next_reference_final_sep_err_uas')):.6g} uas.",
            f"- Sigma improvement factor: {safe_float(metric.get('sigma_improvement_factor')):.6g}; ratio to first-window/sqrt(N) expectation: {safe_float(metric.get('sigma_ratio_to_sqrt_n_expectation')):.6g}.",
            "See [cumulative_information/](cumulative_information/) for the inventory, cumulative prefix tables, diagnostics, variants, and serialized likelihood state.",
            "",
        ]
    elif cumulative_status == "disabled":
        lines += [
            "Cumulative-information analysis was disabled for this review.",
            "",
        ]
    else:
        lines += [
            f"Cumulative-information analysis status: `{cumulative_status}`. Existing window-local review products were still generated.",
            "See [cumulative_information/cumulative_summary.json](cumulative_information/cumulative_summary.json) and [review_warnings.json](review_warnings.json) for structured warnings when available.",
            "",
        ]
    lines += [
        "## Slow-state evolution",
        "See [slow_parameter_error_summary.csv](slow_parameter_error_summary.csv), [slow_state_evolution.csv](slow_state_evolution.csv), and [slow_state_final_summary.csv](slow_state_final_summary.csv). Standard slow-parameter plots include [slow_parameter_final_error_bar.png](plots/slow_parameter_final_error_bar.png), [slow_parameter_final_error_over_sigma_bar.png](plots/slow_parameter_final_error_over_sigma_bar.png), and [slow_parameter_evolution_signed_symlog.png](plots/slow_parameter_evolution_signed_symlog.png).",
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


def parse_bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


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
    sep_summary = separation_error_summary(win)
    slow_param_summary = slow_parameter_error_summary(param)
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
        "separation_error_summary.csv": sep_summary,
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
        "slow_parameter_error_summary.csv": slow_param_summary,
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
    write_runtime_accounting_outputs(
        run_root,
        outdir,
        artifacts["campaign_summary"],
    )

    review_warnings: list[dict[str, Any]] = []
    cumulative_summary: dict[str, Any] | None = None
    try:
        cumulative_summary = run_cumulative_information_review(
            run_root,
            outdir,
            mode=args.cumulative_information,
            no_plots=args.no_plots,
            review_warnings=review_warnings,
        )
    except Exception as exc:
        if args.cumulative_information == "on":
            raise
        warning = {
            "status": "cumulative_information_error",
            "context": "cumulative_information",
            "message": str(exc),
        }
        review_warnings.append(warning)
        cumulative_summary = _cumulative_status_payload(
            run_root=run_root,
            outdir=outdir,
            mode=args.cumulative_information,
            status="summary_load_error",
            warnings=[warning],
            outputs={"cumulative_summary_json": "cumulative_information/cumulative_summary.json"},
        )
        cdir = outdir / "cumulative_information"
        cdir.mkdir(parents=True, exist_ok=True)
        (cdir / "cumulative_summary.json").write_text(json.dumps(_json_scalar(cumulative_summary), indent=2))
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
    write_report(
        outdir,
        run_root,
        artifacts["campaign_summary"],
        win,
        slow_final,
        local_policy,
        image_plots,
        final_forecast,
        cumulative_summary,
    )
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
    p.add_argument("--strict", nargs="?", const=True, default=False, type=parse_bool_arg)
    p.add_argument(
        "--cumulative-information",
        choices=("auto", "on", "off"),
        default="auto",
        help="Run retrospective cumulative-information analysis from preserved Schur summaries.",
    )
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
