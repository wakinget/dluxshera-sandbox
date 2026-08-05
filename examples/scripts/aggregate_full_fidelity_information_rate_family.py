#!/usr/bin/env python3
"""Aggregate full-fidelity information-rate review bundles by M2 family."""

from __future__ import annotations

import argparse
import glob as globlib
import json
import math
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "matplotlib"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SCHEMA_VERSION = "full_fidelity_information_rate_family.v1"
SCRIPT_PATH = Path(__file__)
ARCSEC_TO_UAS = 1.0e6
HEADLINE_POLICIES = ("astrometric_core", "high_information_calibration", "source_core")
SEQUENTIAL_SCOPES = ("observation_carry_window_bounded", "window_restart")
FLOAT_TOL_ATOL = 1.0e-9
FLOAT_TOL_RTOL = 1.0e-7

TOP_LEVEL_REQUIRED = (
    ".information_rate_complete",
    "review_warnings.json",
    "separation_error_summary.csv",
    "slow_parameter_error_summary.csv",
)
INFORMATION_REQUIRED = (
    "information_rate/information_rate_summary.json",
    "information_rate/information_rate_input_inventory.csv",
    "information_rate/information_rate_by_mode.csv",
    "information_rate/information_rate_by_window_mode.csv",
    "information_rate/information_mode_loadings.csv",
    "information_rate/information_by_physical_label.csv",
    "information_rate/mode_overlap.csv",
    "information_rate/adaptive_mode_set_resolution.csv",
    "information_rate/adaptive_cadence_sequential_updates.csv",
    "information_rate/adaptive_cadence_sequential_mode_gains.csv",
    "information_rate/adaptive_cadence_sequential_summary.csv",
    "information_rate/adaptive_cadence_candidates.csv",
    "information_rate/adaptive_cadence_prefix_diagnostics.csv",
)
OPTIONAL_INPUTS = (
    "information_rate/degenerate_subspace_summary.csv",
    "information_rate/quasi_degenerate_subspace_summary.csv",
)
REQUIRED_COLUMNS = {
    "information_rate/information_rate_input_inventory.csv": (
        "accepted_status",
    ),
    "information_rate/information_rate_by_mode.csv": (
        "canonical_mode_id",
        "canonical_eigenvalue_rate",
    ),
    "information_rate/information_rate_by_window_mode.csv": (
        "window_index",
        "canonical_mode_id",
        "information_rate",
    ),
    "information_rate/adaptive_mode_set_resolution.csv": (
        "mode_set_name",
        "requested_physical_label_or_group",
        "canonical_mode_id",
        "selected_mode_ids",
    ),
    "information_rate/adaptive_cadence_sequential_updates.csv": (
        "sequence_scope",
        "policy_mode_set_name",
        "gain_threshold",
        "update_index",
        "block_length",
        "closure_reason",
    ),
    "information_rate/adaptive_cadence_sequential_mode_gains.csv": (
        "sequence_scope",
        "policy_mode_set_name",
        "gain_threshold",
        "update_index",
        "canonical_mode_id",
        "controlling_mode",
    ),
    "information_rate/adaptive_cadence_sequential_summary.csv": (
        "sequence_scope",
        "policy_mode_set_name",
        "gain_threshold",
        "selected_mode_ids",
        "update_count",
        "final_information_invariance_status",
    ),
    "information_rate/adaptive_cadence_candidates.csv": (
        "window_index",
        "gain_threshold",
        "required_top_mode_count",
        "resolved_candidate_block_length",
    ),
    "information_rate/adaptive_cadence_prefix_diagnostics.csv": (
        "window_index",
        "prefix_index",
        "gain_threshold",
    ),
    "information_rate/information_by_physical_label.csv": (
        "analysis_scope",
        "theta_label",
        "posterior_marginal_sigma",
    ),
    "information_rate/mode_overlap.csv": (
        "comparison_scope",
    ),
}
PLOT_NAMES = (
    "physical_core_information_rate_vs_m2_ke.png",
    "physical_assignment_loading_vs_m2_ke.png",
    "gain3_acquisition_block_lengths.png",
    "gain3_trigger_and_latency_summary.png",
    "source_core_controlling_mode_fraction.png",
    "formal_sigma_and_actual_error_vs_m2_ke.png",
    "actual_error_vs_formal_sigma.png",
    "quasi_subspace_stability_early_late.png",
    "psd_projection_fraction_vs_m2_ke.png",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-root-dir", type=Path, help="Directory containing one completed review directory per run.")
    parser.add_argument("--root-list", type=Path, default=None, help="Optional original campaign-root list.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--expected-commit", default="")
    parser.add_argument("--expected-root-count", type=int, default=None)
    parser.add_argument("--expected-draws-per-amplitude", type=int, default=None)
    parser.add_argument("--expected-amplitudes", default="")
    parser.add_argument("--headline-gain-threshold", type=float, default=3.0)
    parser.add_argument("--strict", type=parse_bool, default=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--review-root", type=Path, action="append", default=[])
    parser.add_argument("--review-glob", action="append", default=[])
    return parser.parse_args(argv)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def repository_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def json_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return f if np.isfinite(f) else None
    if isinstance(value, np.ndarray):
        return [json_scalar(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_scalar(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_scalar(dict(payload)), indent=2, sort_keys=True), encoding="utf-8")


def has_content(path: Path) -> bool:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            if chunk.strip():
                return True
    return False


def read_json(path: Path, default: Any = None) -> Any:
    if not has_content(path):
        return default
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def read_csv(path: Path) -> pd.DataFrame:
    if not has_content(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if columns is not None:
        for col in columns:
            if col not in out:
                out[col] = np.nan
        rest = [col for col in out.columns if col not in columns]
        out = out[list(columns) + rest]
    out.to_csv(path, index=False)


def safe_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return default
    if isinstance(value, str) and value.strip() == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int | float = np.nan) -> int | float:
    f = safe_float(value)
    if not np.isfinite(f):
        return default
    return int(f)


def numeric_series(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def bool_series(df: pd.DataFrame, column: str, default: bool = False) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype=bool)
    raw = df[column]
    if raw.dtype == bool:
        return raw.fillna(default).astype(bool)
    return raw.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def semicolon_join(values: Iterable[Any]) -> str:
    out = []
    for value in values:
        text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
        if text and text not in out:
            out.append(text)
    return ";".join(out)


def parse_semicolon_ints(value: Any) -> tuple[int, ...]:
    text = "" if value is None else str(value)
    out: list[int] = []
    for token in re.split(r"[;, ]+", text):
        if not token:
            continue
        try:
            out.append(int(float(token)))
        except ValueError:
            continue
    return tuple(out)


def parse_scalar_mode_id(value: Any) -> int | None:
    modes = parse_semicolon_ints(value)
    return modes[0] if len(modes) == 1 else None


def parse_amplitude_token(token: str) -> float:
    return float(token.lower().replace("p", ".").replace("nm", ""))


def parse_run_metadata(name_or_path: Any) -> dict[str, Any]:
    name = Path(str(name_or_path)).name
    match = re.search(
        r"(?P<campaign>ff_.*?m2_hoke_(?P<amp>[0-9]+p[0-9]+|[0-9]+p[0-9]*|[0-9]+)nm)"
        r"_xp(?P<x>[-0-9p]+)_yp(?P<y>[-0-9p]+)_w(?P<windows>\d+)x(?P<subblocks>\d+)_draw_(?P<draw>\d+)",
        name,
    )
    if match is None:
        match = re.search(
            r"m2_hoke_(?P<amp>[0-9p]+)nm.*?xp(?P<x>[-0-9p]+)_yp(?P<y>[-0-9p]+)_w(?P<windows>\d+)x(?P<subblocks>\d+)_draw_(?P<draw>\d+)",
            name,
        )
    if match is None:
        raise ValueError(f"Could not parse M2 family metadata from run name: {name}")
    campaign = match.groupdict().get("campaign") or name.split("_xp", maxsplit=1)[0]
    return {
        "run_name": name,
        "campaign_label": campaign,
        "m2_ke_nm": parse_amplitude_token(match.group("amp")),
        "draw_index": int(match.group("draw")),
        "field_x": parse_amplitude_token(match.group("x")),
        "field_y": parse_amplitude_token(match.group("y")),
        "window_count": int(match.group("windows")),
        "subblocks_per_window": int(match.group("subblocks")),
    }


def normalize_concept(label: Any) -> str:
    text = str(label)
    mapping = {
        "source.separation_as": "source separation",
        "optics.plate_scale_as_per_pix": "plate scale",
        "source.log_flux_total": "total/log flux",
        "source.contrast": "contrast",
        "wfe_dominated_top": "selected high-information WFE modes",
        "initial_trackability": "all-trackable threshold set",
    }
    for token, concept in mapping.items():
        if token in text:
            return concept
    if "wfe" in text.lower():
        return "selected high-information WFE modes"
    return text


def read_root_list(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"root-list path does not exist: {path}")
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        out[Path(text).name] = text
    return out


def discover_review_roots(args: argparse.Namespace) -> list[Path]:
    roots: list[Path] = []
    if args.per_root_dir is not None:
        if not args.per_root_dir.exists():
            raise FileNotFoundError(f"per-root directory does not exist: {args.per_root_dir}")
        roots.extend(path for path in sorted(args.per_root_dir.iterdir()) if path.is_dir())
    roots.extend(path.expanduser() for path in args.review_root)
    for pattern in args.review_glob:
        roots.extend(Path(p).expanduser() for p in sorted(globlib.glob(pattern)))
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def extract_commit_from_sentinel(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    payload = read_json(path, None)
    if isinstance(payload, dict):
        stack = [payload]
        while stack:
            item = stack.pop()
            for key, value in item.items():
                if str(key).lower() in {"commit", "git_commit", "repository_commit", "script_git_commit"}:
                    return str(value)
                if isinstance(value, dict):
                    stack.append(value)
    match = re.search(r"\b[0-9a-f]{7,40}\b", text)
    return match.group(0) if match else text


def warning_statuses(review_root: Path) -> tuple[int, str]:
    payload = read_json(review_root / "review_warnings.json", {})
    items = payload.get("warnings", payload if isinstance(payload, list) else [])
    if not isinstance(items, list):
        items = []
    statuses = [str(item.get("status", "")) for item in items if isinstance(item, dict) and item.get("status", "")]
    return len(statuses), ";".join(sorted(Counter(statuses)))


def check_required_columns(df: pd.DataFrame, rel_path: str) -> list[str]:
    required = REQUIRED_COLUMNS.get(rel_path, ())
    return [col for col in required if col not in df.columns]


def load_root_tables(review_root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for rel_path in INFORMATION_REQUIRED:
        if rel_path.endswith(".csv"):
            tables[rel_path] = read_csv(review_root / rel_path)
    for rel_path in OPTIONAL_INPUTS:
        tables[rel_path] = read_csv(review_root / rel_path)
    tables["separation_error_summary.csv"] = read_csv(review_root / "separation_error_summary.csv")
    tables["slow_parameter_error_summary.csv"] = read_csv(review_root / "slow_parameter_error_summary.csv")
    return tables


def psd_root_summary(inventory: pd.DataFrame) -> dict[str, Any]:
    if inventory.empty:
        return {
            "accepted_matrix_count": 0,
            "projected_matrix_count": 0,
            "projected_matrix_fraction": 0.0,
            "clipped_eigenvalue_count": 0,
            "maximum_raw_negative_magnitude": 0.0,
            "maximum_relative_projection_correction": 0.0,
            "maximum_absolute_correction": 0.0,
            "projection_status_counts": "",
            "materially_indefinite_count": 0,
        }
    accepted = inventory[bool_series(inventory, "accepted_status", True)].copy()
    if accepted.empty:
        accepted = inventory.copy()
    projected = bool_series(accepted, "psd_projection_applied", False)
    clipped = numeric_series(accepted, "psd_projection_clipped_eigenvalue_count", 0.0).fillna(0.0)
    raw_min = numeric_series(accepted, "raw_minimum_eigenvalue", np.nan)
    rel = numeric_series(accepted, "projection_relative_frobenius_delta", 0.0).fillna(0.0)
    abs_delta = numeric_series(accepted, "projection_max_abs_delta", 0.0).fillna(0.0)
    status_col = "projection_status" if "projection_status" in accepted else "clipping_status"
    status_counts = Counter(accepted.get(status_col, pd.Series("", index=accepted.index)).astype(str))
    material = int((inventory.get("clipping_status", pd.Series("", index=inventory.index)).astype(str) == "materially_indefinite").sum())
    return {
        "accepted_matrix_count": int(len(accepted)),
        "projected_matrix_count": int(projected.sum()),
        "projected_matrix_fraction": float(projected.sum() / len(accepted)) if len(accepted) else 0.0,
        "clipped_eigenvalue_count": int(clipped.sum()),
        "maximum_raw_negative_magnitude": float(np.nanmax(np.maximum(-raw_min.to_numpy(dtype=float), 0.0))) if len(raw_min) else 0.0,
        "maximum_relative_projection_correction": float(np.nanmax(rel.to_numpy(dtype=float))) if len(rel) else 0.0,
        "maximum_absolute_correction": float(np.nanmax(abs_delta.to_numpy(dtype=float))) if len(abs_delta) else 0.0,
        "projection_status_counts": ";".join(f"{k}:{v}" for k, v in sorted(status_counts.items()) if k),
        "materially_indefinite_count": material,
    }


def settings_signature(summary: Mapping[str, Any]) -> dict[str, Any]:
    settings = summary.get("settings", {}) if isinstance(summary, dict) else {}
    seq = summary.get("adaptive_sequential_settings", {}) if isinstance(summary, dict) else {}
    quasi = summary.get("quasi_degeneracy_settings", {}) if isinstance(summary, dict) else {}
    return {
        "tail_windows": settings.get("tail_windows"),
        "gain_thresholds": list(settings.get("thresholds", [])),
        "cadence_mode_sets": list(seq.get("mode_sets", settings.get("adaptive_cadence_mode_sets", []))),
        "cadence_gain_thresholds": list(seq.get("gain_thresholds", settings.get("adaptive_cadence_gain_thresholds", []))),
        "minimum_subblocks": settings.get("adaptive_cadence_min_subblocks"),
        "maximum_subblocks": settings.get("adaptive_cadence_max_subblocks"),
        "high_information_wfe_count": seq.get("high_information_wfe_count", settings.get("adaptive_cadence_high_information_wfe_count")),
        "quasi_degeneracy_tolerance": quasi.get("quasi_degeneracy_rtol", settings.get("quasi_degeneracy_rtol")),
    }


def build_input_inventory(
    review_roots: Sequence[Path],
    original_roots: Mapping[str, str],
    *,
    expected_commit: str,
    strict: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tables_by_run: dict[str, dict[str, pd.DataFrame]] = {}
    missing_files: dict[str, list[str]] = {}
    settings_by_run: dict[str, dict[str, Any]] = {}
    duplicate_names = [name for name, count in Counter(root.name for root in review_roots).items() if count > 1]
    for review_root in sorted(review_roots, key=lambda p: p.name):
        errors: list[str] = []
        try:
            meta = parse_run_metadata(review_root.name)
        except ValueError as exc:
            meta = {
                "run_name": review_root.name,
                "campaign_label": "",
                "m2_ke_nm": np.nan,
                "draw_index": np.nan,
                "field_x": np.nan,
                "field_y": np.nan,
                "window_count": np.nan,
                "subblocks_per_window": np.nan,
            }
            errors.append(str(exc))
        run_name = str(meta["run_name"])
        completion = review_root / ".information_rate_complete"
        failure = review_root / ".information_rate_failed"
        recorded_commit = extract_commit_from_sentinel(completion)
        ir_summary = read_json(review_root / "information_rate/information_rate_summary.json", {})
        tables = load_root_tables(review_root)
        inv = tables["information_rate/information_rate_input_inventory.csv"]
        seq = tables["information_rate/adaptive_cadence_sequential_summary.csv"]
        psd = psd_root_summary(inv)
        warning_count, warning_classes = warning_statuses(review_root)
        required_missing = [rel for rel in TOP_LEVEL_REQUIRED + INFORMATION_REQUIRED if not (review_root / rel).exists()]
        missing_files[run_name] = required_missing
        errors.extend(f"missing:{rel}" for rel in required_missing)
        for rel, df in tables.items():
            errors.extend(f"missing_column:{rel}:{col}" for col in check_required_columns(df, rel))
        discovered = int(ir_summary.get("summary_inventory_counts", {}).get("discovered", len(inv)))
        accepted = int(ir_summary.get("summary_inventory_counts", {}).get("accepted", bool_series(inv, "accepted_status", False).sum() if not inv.empty else 0))
        status = str(ir_summary.get("status", "missing_information_rate_summary"))
        invariance_all_pass = bool((seq.get("final_information_invariance_status", pd.Series(dtype=str)).astype(str) == "pass").all()) if not seq.empty else False
        if completion.exists() is False:
            errors.append("missing_completion_sentinel")
        if failure.exists():
            errors.append("failure_sentinel_present")
        if expected_commit and recorded_commit != expected_commit:
            errors.append("commit_mismatch")
        if status != "ok":
            errors.append(f"information_rate_status:{status}")
        if discovered != 300:
            errors.append(f"discovered_summary_count:{discovered}")
        if accepted != 300:
            errors.append(f"accepted_summary_count:{accepted}")
        if len(seq) != 40:
            errors.append(f"sequential_summary_row_count:{len(seq)}")
        if not invariance_all_pass:
            errors.append("final_information_invariance_not_all_pass")
        if psd["materially_indefinite_count"]:
            errors.append("materially_indefinite_information")
        if run_name in duplicate_names:
            errors.append("duplicate_review_root_name")
        if not np.isfinite(safe_float(meta["m2_ke_nm"])) or not np.isfinite(safe_float(meta["draw_index"])):
            errors.append("metadata_parse_failed")
        settings_by_run[run_name] = settings_signature(ir_summary)
        row = {
            **meta,
            "original_run_root": original_roots.get(run_name, ""),
            "review_root": str(review_root),
            "status": status,
            "discovered_summary_count": discovered,
            "accepted_summary_count": accepted,
            "sequential_summary_row_count": int(len(seq)),
            "invariance_all_pass": invariance_all_pass,
            "completion_sentinel_present": completion.exists(),
            "failure_sentinel_present": failure.exists(),
            "recorded_commit": recorded_commit,
            "commit_matches_expected": bool(recorded_commit == expected_commit) if expected_commit else "",
            "warning_count": warning_count,
            "warning_statuses": warning_classes,
            "projected_matrix_count": psd["projected_matrix_count"],
            "projected_matrix_fraction": psd["projected_matrix_fraction"],
            "clipped_eigenvalue_count": psd["clipped_eigenvalue_count"],
            "maximum_relative_projection_correction": psd["maximum_relative_projection_correction"],
            "inclusion_status": "included" if not errors else "excluded",
            "exclusion_reason": ";".join(errors),
        }
        rows.append(row)
        tables_by_run[run_name] = tables
    inventory = pd.DataFrame(rows)
    if not inventory.empty:
        inventory = inventory.sort_values(["m2_ke_nm", "draw_index", "run_name"], kind="mergesort").reset_index(drop=True)
    discovered_names = set(inventory["run_name"].astype(str)) if not inventory.empty else set()
    expected_names = set(original_roots)
    missing_expected_run_names = sorted(expected_names - discovered_names)
    unexpected_review_run_names = sorted(discovered_names - expected_names) if expected_names else []
    sig_values = {json.dumps(v, sort_keys=True) for v in settings_by_run.values()}
    settings_consistent = len(sig_values) <= 1
    if not settings_consistent:
        inventory.loc[inventory["inclusion_status"] == "included", "inclusion_status"] = "excluded"
        inventory.loc[:, "exclusion_reason"] = inventory["exclusion_reason"].astype(str).mask(
            inventory["exclusion_reason"].astype(str) == "", "information_rate_settings_mismatch"
        )
    if unexpected_review_run_names and not inventory.empty:
        mask = inventory["run_name"].astype(str).isin(unexpected_review_run_names)
        inventory.loc[mask, "inclusion_status"] = "excluded"
        existing = inventory.loc[mask, "exclusion_reason"].astype(str)
        inventory.loc[mask, "exclusion_reason"] = existing.where(existing != "", "unexpected_review_root_not_in_root_list")
    diagnostics = {
        "missing_files": missing_files,
        "settings_by_run": settings_by_run,
        "information_rate_settings": next(iter(settings_by_run.values()), {}),
        "settings_consistent": settings_consistent,
        "missing_expected_run_names": missing_expected_run_names,
        "unexpected_review_run_names": unexpected_review_run_names,
    }
    if strict and missing_expected_run_names:
        raise RuntimeError(f"Strict family validation failed: missing root-list run names: {missing_expected_run_names[:10]}")
    if strict and (inventory.empty or (inventory["inclusion_status"] != "included").any()):
        bad = inventory[inventory["inclusion_status"] != "included"][["run_name", "exclusion_reason"]].to_dict(orient="records")
        raise RuntimeError(f"Strict family validation failed: {bad[:10]}")
    return inventory, tables_by_run, diagnostics


def validate_family_expectations(
    inventory: pd.DataFrame,
    *,
    expected_root_count: int | None,
    expected_draws_per_amplitude: int | None,
    expected_amplitudes: Sequence[float],
    strict: bool,
) -> list[str]:
    issues: list[str] = []
    if expected_root_count is not None and len(inventory) != expected_root_count:
        issues.append(f"expected_root_count:{expected_root_count}:observed:{len(inventory)}")
    included = inventory[inventory["inclusion_status"] == "included"]
    if expected_amplitudes:
        observed = sorted(float(v) for v in included["m2_ke_nm"].dropna().unique())
        missing = [amp for amp in expected_amplitudes if not any(math.isclose(amp, obs, rel_tol=0.0, abs_tol=1e-12) for obs in observed)]
        extra = [obs for obs in observed if not any(math.isclose(obs, amp, rel_tol=0.0, abs_tol=1e-12) for amp in expected_amplitudes)]
        if missing:
            issues.append("missing_expected_amplitudes:" + ",".join(f"{v:g}" for v in missing))
        if extra:
            issues.append("unexpected_amplitudes:" + ",".join(f"{v:g}" for v in extra))
    if expected_draws_per_amplitude is not None and not included.empty:
        counts = included.groupby("m2_ke_nm", dropna=False)["draw_index"].nunique()
        bad = {float(k): int(v) for k, v in counts.items() if int(v) != expected_draws_per_amplitude}
        if bad:
            issues.append("expected_draws_per_amplitude_mismatch:" + json.dumps(bad, sort_keys=True))
    if strict and issues:
        raise RuntimeError("Strict family expectation validation failed: " + ";".join(issues))
    return issues


def add_root_columns(df: pd.DataFrame, meta: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col in ("run_name", "campaign_label", "original_run_root", "review_root", "m2_ke_nm", "draw_index"):
        out[col] = meta.get(col, "")
    return out


def included_metadata(inventory: pd.DataFrame) -> list[dict[str, Any]]:
    return inventory[inventory["inclusion_status"] == "included"].to_dict(orient="records")


def collect_table(inventory: pd.DataFrame, tables_by_run: Mapping[str, Mapping[str, pd.DataFrame]], rel_path: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for meta in included_metadata(inventory):
        df = tables_by_run[str(meta["run_name"])].get(rel_path, pd.DataFrame())
        if not df.empty:
            frames.append(add_root_columns(df, meta))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_numeric(values: pd.Series, prefix: str = "") -> dict[str, Any]:
    data = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(data))
    median = float(data.median()) if n else np.nan
    std = float(data.std(ddof=1)) if n > 1 else np.nan
    return {
        f"{prefix}N": n,
        f"{prefix}mean": float(data.mean()) if n else np.nan,
        f"{prefix}std": std,
        f"{prefix}sem": std / math.sqrt(n) if n > 1 else np.nan,
        f"{prefix}median": median,
        f"{prefix}mad": float((data - median).abs().median()) if n else np.nan,
        f"{prefix}min": float(data.min()) if n else np.nan,
        f"{prefix}max": float(data.max()) if n else np.nan,
        f"{prefix}coefficient_of_variation": float(std / abs(data.mean())) if n > 1 and data.mean() else np.nan,
    }


def flatten_numeric_summary(df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=list(group_cols))
    for key, group in df.groupby(list(group_cols), dropna=False, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values))
        row["N"] = int(len(group))
        for col in metric_cols:
            if col in group:
                stats = summarize_numeric(group[col])
                for stat_name, value in stats.items():
                    if stat_name == "N":
                        continue
                    row[f"{col}_{stat_name}"] = value
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols), kind="mergesort").reset_index(drop=True)


def build_physical_assignments(resolution: pd.DataFrame, rates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if resolution.empty:
        return pd.DataFrame(), pd.DataFrame()
    rate_lookup = {}
    if not rates.empty:
        for _, row in rates.iterrows():
            mode = parse_scalar_mode_id(row.get("canonical_mode_id"))
            if mode is not None:
                rate_lookup[(row["run_name"], mode)] = row
    rows: list[dict[str, Any]] = []
    for _, r in resolution.iterrows():
        mode_id = parse_scalar_mode_id(r.get("canonical_mode_id"))
        selected = str(r.get("selected_mode_ids", ""))
        concept = normalize_concept(r.get("requested_physical_label_or_group", ""))
        rate_row = rate_lookup.get((r["run_name"], mode_id), {}) if mode_id is not None else {}
        next_loading = safe_float(r.get("next_best_loading"))
        loading = safe_float(r.get("squared_loading_used_for_assignment"))
        rows.append(
            {
                "run_name": r["run_name"],
                "m2_ke_nm": r["m2_ke_nm"],
                "draw_index": r["draw_index"],
                "physical_concept": concept,
                "requested_physical_label_or_group": r.get("requested_physical_label_or_group", ""),
                "mode_set_source": r.get("mode_set_name", ""),
                "canonical_mode_id": mode_id if mode_id is not None else r.get("canonical_mode_id", ""),
                "canonical_rate": safe_float(r.get("canonical_rate", rate_row.get("canonical_eigenvalue_rate", np.nan))),
                "squared_loading_used_for_assignment": loading,
                "assignment_rank": safe_int(r.get("assignment_rank")),
                "next_best_mode": r.get("next_best_mode", ""),
                "next_best_loading": next_loading,
                "loading_margin": loading - next_loading if np.isfinite(loading) and np.isfinite(next_loading) else np.nan,
                "assignment_status": r.get("assignment_status", ""),
                "threshold_dependency": r.get("threshold_dependency", ""),
                "selected_mode_ids": selected,
                "quasi_degenerate": bool(rate_row.get("quasi_degenerate", False)) if isinstance(rate_row, pd.Series) else "",
                "mode_identity_caution": rate_row.get("mode_identity_caution", "") if isinstance(rate_row, pd.Series) else "",
            }
        )
    root = pd.DataFrame(rows)
    dedup_cols = ["run_name", "physical_concept", "canonical_mode_id", "threshold_dependency", "selected_mode_ids"]
    root = (
        root.groupby(dedup_cols, dropna=False, sort=True)
        .agg(
            {
                "m2_ke_nm": "first",
                "draw_index": "first",
                "requested_physical_label_or_group": "first",
                "mode_set_source": semicolon_join,
                "canonical_rate": "first",
                "squared_loading_used_for_assignment": "first",
                "assignment_rank": "first",
                "next_best_mode": "first",
                "next_best_loading": "first",
                "loading_margin": "first",
                "assignment_status": semicolon_join,
                "quasi_degenerate": "first",
                "mode_identity_caution": "first",
            }
        )
        .reset_index()
    )
    amp_rows: list[dict[str, Any]] = []
    for key, group in root.groupby(["m2_ke_nm", "physical_concept"], dropna=False, sort=True):
        amp, concept = key
        successful = group["assignment_status"].astype(str).str.contains("ok|unique", case=False, regex=True)
        unique = group["assignment_status"].astype(str).str.contains("unique|ok", case=False, regex=True) & ~group["assignment_status"].astype(str).str.contains("ambiguous|weak", case=False, regex=True)
        amp_rows.append(
            {
                "m2_ke_nm": amp,
                "physical_concept": concept,
                "N_roots": int(group["run_name"].nunique()),
                "successful_assignment_count": int(successful.sum()),
                "successful_assignment_fraction": float(successful.mean()) if len(group) else np.nan,
                "unique_assignment_fraction": float(unique.mean()) if len(group) else np.nan,
                "median_assigned_squared_loading": float(pd.to_numeric(group["squared_loading_used_for_assignment"], errors="coerce").median()),
                "minimum_assigned_squared_loading": float(pd.to_numeric(group["squared_loading_used_for_assignment"], errors="coerce").min()),
                "median_next_best_loading": float(pd.to_numeric(group["next_best_loading"], errors="coerce").median()),
                "median_loading_margin": float(pd.to_numeric(group["loading_margin"], errors="coerce").median()),
                "minimum_loading_margin": float(pd.to_numeric(group["loading_margin"], errors="coerce").min()),
                "assignment_status_counts": ";".join(f"{k}:{v}" for k, v in sorted(Counter(group["assignment_status"].astype(str)).items())),
                "assigned_mode_id_distribution": ";".join(f"{k}:{v}" for k, v in sorted(Counter(group["canonical_mode_id"].astype(str)).items())),
            }
        )
    amp_df = pd.DataFrame(amp_rows)
    return root.sort_values(["m2_ke_nm", "draw_index", "physical_concept"], kind="mergesort").reset_index(drop=True), amp_df


def build_physical_rates(assignments: pd.DataFrame, rates: pd.DataFrame, window_rates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if assignments.empty or rates.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    rows: list[dict[str, Any]] = []
    for _, a in assignments.iterrows():
        mode = parse_scalar_mode_id(a.get("canonical_mode_id"))
        if mode is None:
            continue
        hit = rates[(rates["run_name"] == a["run_name"]) & (numeric_series(rates, "canonical_mode_id") == mode)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        row = {
            "run_name": a["run_name"],
            "m2_ke_nm": a["m2_ke_nm"],
            "draw_index": a["draw_index"],
            "physical_concept": a["physical_concept"],
            "canonical_mode_id": mode,
        }
        for col in (
            "canonical_eigenvalue_rate",
            "information_replacement_timescale_s",
            "gain_at_1s",
            "gain_at_5s",
            "gain_at_10s",
            "gain_at_30s",
            "gain_at_300s",
            "gain_at_1800s_projected",
            "first_window_rate",
            "final_window_rate",
            "median_window_rate",
            "late_tail_rate",
            "late_tail_median_window_rate",
            "mean_window_rate",
            "std_window_rate",
            "window_rate_coefficient_of_variation",
            "minimum_window_overlap",
            "median_window_overlap",
            "quasi_degenerate",
            "dominant_physical_group",
            "dominant_labels",
            "participation_ratio",
        ):
            row[col] = r.get(col, np.nan)
        rows.append(row)
    by_root = pd.DataFrame(rows).sort_values(["m2_ke_nm", "draw_index", "physical_concept"], kind="mergesort").reset_index(drop=True)
    metric_cols = [
        "canonical_eigenvalue_rate",
        "first_window_rate",
        "final_window_rate",
        "early_to_late_rate_ratio",
        "minimum_window_overlap",
        "median_window_overlap",
        "information_replacement_timescale_s",
    ]
    if not by_root.empty:
        by_root["early_to_late_rate_ratio"] = numeric_series(by_root, "final_window_rate") / numeric_series(by_root, "first_window_rate")
    by_amp = flatten_numeric_summary(by_root, ["m2_ke_nm", "physical_concept"], metric_cols)
    stability_root = build_rate_stability(assignments, window_rates)
    stability_amp = flatten_numeric_summary(
        stability_root,
        ["m2_ke_nm", "physical_concept"],
        ["early_median_rate", "late_median_rate", "late_to_early_ratio", "early_coefficient_of_variation", "late_coefficient_of_variation", "minimum_canonical_overlap", "median_canonical_overlap"],
    )
    return by_root, by_amp, stability_root, stability_amp


def build_rate_stability(assignments: pd.DataFrame, window_rates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if assignments.empty or window_rates.empty:
        return pd.DataFrame()
    for _, a in assignments.iterrows():
        mode = parse_scalar_mode_id(a.get("canonical_mode_id"))
        if mode is None:
            continue
        group = window_rates[(window_rates["run_name"] == a["run_name"]) & (numeric_series(window_rates, "canonical_mode_id") == mode)].copy()
        if group.empty:
            continue
        windows = sorted(pd.to_numeric(group["window_index"], errors="coerce").dropna().unique())
        split = len(windows) // 2
        early_windows = set(windows[:split])
        late_windows = set(windows[split:])
        early = group[group["window_index"].isin(early_windows)]
        late = group[group["window_index"].isin(late_windows)]
        er = numeric_series(early, "information_rate").dropna()
        lr = numeric_series(late, "information_rate").dropna()
        overlap = numeric_series(group, "overlap_with_canonical_mode").dropna()
        rows.append(
            {
                "run_name": a["run_name"],
                "m2_ke_nm": a["m2_ke_nm"],
                "draw_index": a["draw_index"],
                "physical_concept": a["physical_concept"],
                "canonical_mode_id": mode,
                "early_median_rate": float(er.median()) if len(er) else np.nan,
                "late_median_rate": float(lr.median()) if len(lr) else np.nan,
                "late_to_early_ratio": float(lr.median() / er.median()) if len(er) and len(lr) and er.median() else np.nan,
                "early_coefficient_of_variation": float(er.std(ddof=1) / abs(er.mean())) if len(er) > 1 and er.mean() else np.nan,
                "late_coefficient_of_variation": float(lr.std(ddof=1) / abs(lr.mean())) if len(lr) > 1 and lr.mean() else np.nan,
                "minimum_canonical_overlap": float(overlap.min()) if len(overlap) else np.nan,
                "median_canonical_overlap": float(overlap.median()) if len(overlap) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["m2_ke_nm", "draw_index", "physical_concept"], kind="mergesort").reset_index(drop=True)


def build_sequential(seq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if seq.empty:
        return pd.DataFrame(), pd.DataFrame()
    root = seq.copy()
    root["final_separation_sigma_uas"] = numeric_series(root, "final_separation_sigma") * ARCSEC_TO_UAS
    root["final_plate_scale_sigma_uas_per_pix"] = numeric_series(root, "final_plate_scale_sigma") * ARCSEC_TO_UAS
    root = root.sort_values(["m2_ke_nm", "draw_index", "sequence_scope", "policy_mode_set_name", "gain_threshold", "selected_mode_ids"], kind="mergesort").reset_index(drop=True)
    by_amp = flatten_numeric_summary(
        root,
        ["m2_ke_nm", "sequence_scope", "policy_mode_set_name", "gain_threshold", "selected_mode_ids"],
        [
            "update_count",
            "natural_trigger_count",
            "maximum_latency_count",
            "maximum_latency_fraction",
            "historical_boundary_flush_count",
            "first_block_length",
            "median_block_length",
            "minimum_block_length",
            "maximum_block_length",
            "final_separation_sigma_uas",
            "final_plate_scale_sigma_uas_per_pix",
        ],
    )
    return root, by_amp


def build_gain3(updates: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if updates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    subset = updates[np.isclose(numeric_series(updates, "gain_threshold"), threshold)].copy()
    if subset.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    key_cols = ["run_name", "m2_ke_nm", "draw_index", "sequence_scope", "policy_mode_set_name", "gain_threshold", "selected_mode_ids"]
    for key, group in subset.groupby(key_cols, dropna=False, sort=True):
        group = group.sort_values(["update_index"], kind="mergesort")
        natural = group[bool_series(group, "triggered_naturally", False)]
        first = natural.iloc[0] if len(natural) >= 1 else {}
        second = natural.iloc[1] if len(natural) >= 2 else {}
        rows.append(
            {
                **dict(zip(key_cols, key)),
                "natural_trigger_count": int(len(natural)),
                "first_natural_block_length": safe_int(first.get("block_length")) if isinstance(first, pd.Series) else np.nan,
                "second_natural_block_length": safe_int(second.get("block_length")) if isinstance(second, pd.Series) else np.nan,
                "first_natural_cumulative_closure_time_s": safe_float(first.get("cumulative_elapsed_time_s")) if isinstance(first, pd.Series) else np.nan,
                "second_natural_cumulative_closure_time_s": safe_float(second.get("cumulative_elapsed_time_s")) if isinstance(second, pd.Series) else np.nan,
                "first_controlling_mode": first.get("controlling_mode_id", "") if isinstance(first, pd.Series) else "",
                "second_controlling_mode": second.get("controlling_mode_id", "") if isinstance(second, pd.Series) else "",
                "maximum_latency_fraction": float(bool_series(group, "maximum_latency_reached", False).mean()) if len(group) else np.nan,
                "total_update_count": int(len(group)),
                "at_least_two_natural_triggers": bool(len(natural) >= 2),
            }
        )
    wr = subset[subset["sequence_scope"] == "window_restart"]
    if not wr.empty:
        for key, group in wr.groupby(["run_name", "m2_ke_nm", "draw_index", "policy_mode_set_name", "gain_threshold", "selected_mode_ids", "historical_window_index"], dropna=False, sort=True):
            natural = group[bool_series(group, "triggered_naturally", False)].sort_values("update_index")
            first = natural.iloc[0] if len(natural) >= 1 else {}
            second = natural.iloc[1] if len(natural) >= 2 else {}
            boundary = group[group.get("closure_reason", pd.Series("", index=group.index)).astype(str).isin({"historical_window_boundary", "end_of_scope"})]
            window_rows.append(
                {
                    **dict(zip(["run_name", "m2_ke_nm", "draw_index", "policy_mode_set_name", "gain_threshold", "selected_mode_ids", "historical_window_index"], key)),
                    "first_natural_block_length": safe_int(first.get("block_length")) if isinstance(first, pd.Series) else np.nan,
                    "second_natural_block_length": safe_int(second.get("block_length")) if isinstance(second, pd.Series) else np.nan,
                    "natural_trigger_count": int(len(natural)),
                    "remaining_boundary_flush_length": safe_int(boundary.iloc[-1].get("block_length")) if not boundary.empty else np.nan,
                    "maximum_latency_status": bool(bool_series(group, "maximum_latency_reached", False).any()),
                }
            )
    root = pd.DataFrame(rows).sort_values(key_cols, kind="mergesort").reset_index(drop=True)
    amp = flatten_numeric_summary(
        root,
        ["m2_ke_nm", "sequence_scope", "policy_mode_set_name", "gain_threshold", "selected_mode_ids"],
        ["first_natural_block_length", "second_natural_block_length", "second_natural_cumulative_closure_time_s", "maximum_latency_fraction"],
    )
    if not root.empty:
        frac = root.groupby(["m2_ke_nm", "sequence_scope", "policy_mode_set_name", "gain_threshold", "selected_mode_ids"], dropna=False, sort=True).agg(
            fraction_roots_with_at_least_one_natural_trigger=("natural_trigger_count", lambda s: float((pd.to_numeric(s, errors="coerce") >= 1).mean())),
            fraction_roots_with_at_least_two_natural_triggers=("at_least_two_natural_triggers", lambda s: float(pd.Series(s).astype(bool).mean())),
        ).reset_index()
        amp = amp.merge(frac, on=["m2_ke_nm", "sequence_scope", "policy_mode_set_name", "gain_threshold", "selected_mode_ids"], how="left")
    return root, amp, pd.DataFrame(window_rows)


def schedule_signature(group: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "historical_window_index",
        "update_index",
        "block_length",
        "closure_reason",
        "triggered_naturally",
        "maximum_latency_reached",
        "historical_window_boundary_flush",
    ]
    out: list[dict[str, Any]] = []
    for _, row in group.sort_values(["update_index"], kind="mergesort").iterrows():
        out.append({col: json_scalar(row.get(col, "")) for col in cols})
    return out


def build_schedule_equivalence(updates: pd.DataFrame) -> pd.DataFrame:
    if updates.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for key, group in updates.groupby(["run_name", "m2_ke_nm", "draw_index", "sequence_scope", "gain_threshold"], dropna=False, sort=True):
        astro = group[group["policy_mode_set_name"] == "astrometric_core"]
        high = group[group["policy_mode_set_name"] == "high_information_calibration"]
        if astro.empty or high.empty:
            continue
        astro_sig = schedule_signature(astro)
        high_sig = schedule_signature(high)
        first_diff = -1
        for idx, (a, h) in enumerate(zip(astro_sig, high_sig)):
            if a != h:
                first_diff = idx
                break
        if first_diff < 0 and len(astro_sig) != len(high_sig):
            first_diff = min(len(astro_sig), len(high_sig))
        rows.append(
            {
                **dict(zip(["run_name", "m2_ke_nm", "draw_index", "sequence_scope", "gain_threshold"], key)),
                "exact_schedule_match": astro_sig == high_sig,
                "update_count_match": len(astro_sig) == len(high_sig),
                "block_lengths_match": [r["block_length"] for r in astro_sig] == [r["block_length"] for r in high_sig],
                "closure_reasons_match": [r["closure_reason"] for r in astro_sig] == [r["closure_reason"] for r in high_sig],
                "first_difference_index": first_diff,
                "astrometric_schedule_signature": json.dumps(astro_sig, sort_keys=True),
                "high_information_schedule_signature": json.dumps(high_sig, sort_keys=True),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = df.groupby(["m2_ke_nm", "sequence_scope", "gain_threshold"], dropna=False, sort=True)["exact_schedule_match"].mean().reset_index(name="exact_schedule_match_fraction")
    return pd.concat([df, summary.assign(run_name="__amplitude_summary__", draw_index=np.nan, update_count_match=np.nan, block_lengths_match=np.nan, closure_reasons_match=np.nan, first_difference_index=np.nan, astrometric_schedule_signature="", high_information_schedule_signature="")], ignore_index=True, sort=False)


def assignment_lookup(assignments: pd.DataFrame) -> dict[tuple[str, int], str]:
    out: dict[tuple[str, int], str] = {}
    if assignments.empty:
        return out
    for _, row in assignments.iterrows():
        mode = parse_scalar_mode_id(row.get("canonical_mode_id"))
        if mode is not None and row.get("physical_concept") not in {"selected high-information WFE modes", "all-trackable threshold set"}:
            out[(row["run_name"], mode)] = str(row["physical_concept"])
    return out


def build_controlling_modes(gains: pd.DataFrame, updates: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if gains.empty:
        return pd.DataFrame(), pd.DataFrame()
    lookup = assignment_lookup(assignments)
    ctrl = gains[bool_series(gains, "controlling_mode", False)].copy()
    if ctrl.empty:
        return pd.DataFrame(), pd.DataFrame()
    ctrl["canonical_mode_id"] = numeric_series(ctrl, "canonical_mode_id").astype("Int64")
    ctrl["physical_interpretation"] = [
        lookup.get((row["run_name"], int(row["canonical_mode_id"])), row.get("mode_physical_interpretation", row.get("dominant_physical_group", "")))
        for _, row in ctrl.iterrows()
    ]
    upd_cols = ["run_name", "sequence_scope", "policy_mode_set_name", "gain_threshold", "update_index"]
    update_flags = updates[upd_cols + [c for c in ("triggered_naturally", "maximum_latency_reached") if c in updates]].copy() if not updates.empty else pd.DataFrame(columns=upd_cols)
    ctrl = ctrl.merge(update_flags, on=upd_cols, how="left")
    ctrl["gain_at_closure"] = numeric_series(ctrl, "current_relative_gain")
    group_cols = ["m2_ke_nm", "sequence_scope", "policy_mode_set_name", "gain_threshold", "physical_interpretation", "canonical_mode_id"]
    rows: list[dict[str, Any]] = []
    total_by_policy = ctrl.groupby(["m2_ke_nm", "sequence_scope", "policy_mode_set_name", "gain_threshold"], dropna=False).size()
    for key, group in ctrl.groupby(group_cols, dropna=False, sort=True):
        total = total_by_policy.loc[key[:4]]
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "controlling_update_count": int(len(group)),
                "controlling_update_fraction": float(len(group) / total) if total else np.nan,
                "median_gain_at_closure": float(pd.to_numeric(group["gain_at_closure"], errors="coerce").median()),
                "natural_trigger_fraction": float(bool_series(group, "triggered_naturally", False).mean()),
                "maximum_latency_fraction": float(bool_series(group, "maximum_latency_reached", False).mean()),
            }
        )
    return ctrl.sort_values(["m2_ke_nm", "draw_index", "sequence_scope", "policy_mode_set_name", "gain_threshold", "update_index"], kind="mergesort").reset_index(drop=True), pd.DataFrame(rows)


def build_fixed_prior(candidates: pd.DataFrame, prefix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame()
    root = candidates.copy().sort_values(["m2_ke_nm", "draw_index", "window_index", "gain_threshold", "required_top_mode_count"], kind="mergesort").reset_index(drop=True)
    amp = flatten_numeric_summary(
        root,
        ["m2_ke_nm", "gain_threshold", "required_top_mode_count"],
        ["natural_crossing_prefix", "resolved_candidate_block_length"],
    )
    if "maximum_latency_reached" in root:
        frac = root.groupby(["m2_ke_nm", "gain_threshold", "required_top_mode_count"], dropna=False, sort=True)["maximum_latency_reached"].apply(lambda s: float(pd.Series(s).astype(bool).mean())).reset_index(name="maximum_latency_fraction")
        amp = amp.merge(frac, on=["m2_ke_nm", "gain_threshold", "required_top_mode_count"], how="left")
    return root, amp


def parse_composition(value: Any) -> tuple[dict[str, float], str]:
    if isinstance(value, dict):
        comp = value
    else:
        try:
            comp = json.loads(str(value))
        except (TypeError, json.JSONDecodeError):
            comp = {}
    parsed = {str(k): safe_float(v, 0.0) for k, v in comp.items()}
    dominant = max(parsed, key=parsed.get) if parsed else ""
    return parsed, dominant


def build_quasi(quasi: pd.DataFrame, overlaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    if not quasi.empty:
        for _, r in quasi.iterrows():
            comp, dominant = parse_composition(r.get("group_physical_composition", "{}"))
            rows.append(
                {
                    "run_name": r["run_name"],
                    "m2_ke_nm": r["m2_ke_nm"],
                    "draw_index": r["draw_index"],
                    "quasi_degeneracy_group": r.get("quasi_degeneracy_group", r.get("degeneracy_group", "")),
                    "member_mode_ids": r.get("member_mode_ids", ""),
                    "group_dimension": safe_int(r.get("group_dimension")),
                    "eigenvalue_min": safe_float(r.get("eigenvalue_min")),
                    "eigenvalue_max": safe_float(r.get("eigenvalue_max")),
                    "adjacent_relative_gaps": r.get("adjacent_relative_gaps", r.get("relative_gaps", "")),
                    "parsed_physical_composition": json.dumps(comp, sort_keys=True),
                    "dominant_physical_group": dominant,
                    "minimum_subspace_singular_value": safe_float(r.get("minimum_subspace_singular_value")),
                    "median_subspace_singular_value": safe_float(r.get("median_subspace_singular_value")),
                    "maximum_principal_angle_deg": safe_float(r.get("maximum_principal_angle_deg")),
                    "interpretation_note": r.get("individual_mode_interpretation_note", "quasi-degenerate: prefer subspace stability over individual modes"),
                }
            )
    by_root = pd.DataFrame(rows)
    amp_rows: list[dict[str, Any]] = []
    if not by_root.empty:
        included_roots = by_root.groupby("m2_ke_nm")["run_name"].nunique().to_dict()
        for amp, group in by_root.groupby("m2_ke_nm", dropna=False, sort=True):
            amp_rows.append(
                {
                    "m2_ke_nm": amp,
                    "roots_with_at_least_one_quasi_degenerate_group": int(group["run_name"].nunique()),
                    "fraction_roots_with_a_group": np.nan,
                    "number_of_groups": int(len(group)),
                    "group_dimension_distribution": ";".join(f"{k}:{v}" for k, v in sorted(Counter(group["group_dimension"].astype(str)).items())),
                    "dominant_composition_distribution": ";".join(f"{k}:{v}" for k, v in sorted(Counter(group["dominant_physical_group"].astype(str)).items())),
                    "minimum_subspace_singular_value_min": float(pd.to_numeric(group["minimum_subspace_singular_value"], errors="coerce").min()),
                    "median_subspace_singular_value_median": float(pd.to_numeric(group["median_subspace_singular_value"], errors="coerce").median()),
                    "maximum_principal_angle_deg_max": float(pd.to_numeric(group["maximum_principal_angle_deg"], errors="coerce").max()),
                }
            )
    stability_rows: list[dict[str, Any]] = []
    if not overlaps.empty and "comparison_scope" in overlaps:
        qo = overlaps[overlaps["comparison_scope"] == "window_rate_quasi_subspace"].copy()
        for key, group in qo.groupby(["run_name", "m2_ke_nm", "draw_index", "degeneracy_group"], dropna=False, sort=True):
            group = group.sort_values("window_index", kind="mergesort")
            windows = sorted(pd.to_numeric(group["window_index"], errors="coerce").dropna().unique())
            split = len(windows) // 2
            early = group[group["window_index"].isin(windows[:split])]
            late = group[group["window_index"].isin(windows[split:])]
            sv = numeric_series(group, "minimum_subspace_singular_value")
            ang = numeric_series(group, "maximum_principal_angle_deg")
            stability_rows.append(
                {
                    **dict(zip(["run_name", "m2_ke_nm", "draw_index", "quasi_degeneracy_group"], key)),
                    "first_window_singular_value": safe_float(sv.iloc[0]) if len(sv) else np.nan,
                    "first_window_principal_angle_deg": safe_float(ang.iloc[0]) if len(ang) else np.nan,
                    "early_half_median_singular_value": float(numeric_series(early, "minimum_subspace_singular_value").median()) if not early.empty else np.nan,
                    "early_half_median_principal_angle_deg": float(numeric_series(early, "maximum_principal_angle_deg").median()) if not early.empty else np.nan,
                    "late_half_median_singular_value": float(numeric_series(late, "minimum_subspace_singular_value").median()) if not late.empty else np.nan,
                    "late_half_median_principal_angle_deg": float(numeric_series(late, "maximum_principal_angle_deg").median()) if not late.empty else np.nan,
                    "final_window_singular_value": safe_float(sv.iloc[-1]) if len(sv) else np.nan,
                    "final_window_principal_angle_deg": safe_float(ang.iloc[-1]) if len(ang) else np.nan,
                    "singular_value_improvement_late_minus_early": float(numeric_series(late, "minimum_subspace_singular_value").median() - numeric_series(early, "minimum_subspace_singular_value").median()) if not early.empty and not late.empty else np.nan,
                    "principal_angle_improvement_early_minus_late_deg": float(numeric_series(early, "maximum_principal_angle_deg").median() - numeric_series(late, "maximum_principal_angle_deg").median()) if not early.empty and not late.empty else np.nan,
                }
            )
    return by_root, pd.DataFrame(amp_rows), pd.DataFrame(stability_rows)


def check_policy_independent(group: pd.DataFrame, column: str) -> tuple[float, bool, float]:
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    if values.empty:
        return np.nan, True, np.nan
    spread = float(values.max() - values.min())
    scale = max(abs(float(values.median())), 1.0)
    ok = spread <= FLOAT_TOL_ATOL + FLOAT_TOL_RTOL * scale
    return float(values.median()), ok, spread


def physical_sigma(physical: pd.DataFrame, run_name: str, label: str, scope: str) -> float:
    group = physical[(physical["run_name"] == run_name) & (physical["theta_label"] == label) & (physical["analysis_scope"] == scope)].copy()
    if group.empty:
        return np.nan
    if scope == "frozen_factor_observation_prefix" and "elapsed_time_s" in group:
        group = group.sort_values("elapsed_time_s", kind="mergesort")
    return safe_float(group.iloc[-1].get("posterior_marginal_sigma"))


def build_formal_uncertainty(seq: pd.DataFrame, physical: pd.DataFrame, strict: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    if seq.empty and physical.empty:
        return pd.DataFrame(), pd.DataFrame()
    for run_name, group in seq.groupby("run_name", dropna=False, sort=True) if not seq.empty else []:
        meta = group.iloc[0]
        carry = group[group["sequence_scope"] == "observation_carry_window_bounded"]
        restart = group[group["sequence_scope"] == "window_restart"]
        carry_sep, carry_sep_ok, carry_sep_spread = check_policy_independent(carry, "final_separation_sigma")
        carry_plate, carry_plate_ok, carry_plate_spread = check_policy_independent(carry, "final_plate_scale_sigma")
        restart_sep, restart_sep_ok, restart_sep_spread = check_policy_independent(restart, "final_separation_sigma")
        restart_plate, restart_plate_ok, restart_plate_spread = check_policy_independent(restart, "final_plate_scale_sigma")
        if strict and not (carry_sep_ok and carry_plate_ok and restart_sep_ok and restart_plate_ok):
            raise RuntimeError(f"Policy-dependent final covariance for {run_name}")
        sep30 = physical_sigma(physical, str(run_name), "source.separation_as", "late_tail_projection_30s")
        sep300 = physical_sigma(physical, str(run_name), "source.separation_as", "frozen_factor_observation_prefix")
        sep1800 = physical_sigma(physical, str(run_name), "source.separation_as", "late_tail_projection_1800s")
        plate30 = physical_sigma(physical, str(run_name), "optics.plate_scale_as_per_pix", "late_tail_projection_30s")
        plate300 = physical_sigma(physical, str(run_name), "optics.plate_scale_as_per_pix", "frozen_factor_observation_prefix")
        plate1800 = physical_sigma(physical, str(run_name), "optics.plate_scale_as_per_pix", "late_tail_projection_1800s")
        rows.append(
            {
                "run_name": run_name,
                "m2_ke_nm": meta.get("m2_ke_nm", np.nan),
                "draw_index": meta.get("draw_index", np.nan),
                "window_local_separation_sigma_30s_uas": restart_sep * ARCSEC_TO_UAS if np.isfinite(restart_sep) else sep30 * ARCSEC_TO_UAS,
                "cumulative_separation_sigma_300s_uas": carry_sep * ARCSEC_TO_UAS if np.isfinite(carry_sep) else sep300 * ARCSEC_TO_UAS,
                "late_tail_projected_separation_sigma_1800s_uas": sep1800 * ARCSEC_TO_UAS,
                "window_local_plate_scale_sigma_30s_uas_per_pix": restart_plate * ARCSEC_TO_UAS if np.isfinite(restart_plate) else plate30 * ARCSEC_TO_UAS,
                "cumulative_plate_scale_sigma_300s_uas_per_pix": carry_plate * ARCSEC_TO_UAS if np.isfinite(carry_plate) else plate300 * ARCSEC_TO_UAS,
                "late_tail_projected_plate_scale_sigma_1800s_uas_per_pix": plate1800 * ARCSEC_TO_UAS,
                "separation_30s_over_300s_contraction_ratio": (restart_sep / carry_sep) if np.isfinite(restart_sep) and np.isfinite(carry_sep) and carry_sep else np.nan,
                "separation_300s_over_1800s_formal_contraction_ratio": (carry_sep / sep1800) if np.isfinite(carry_sep) and np.isfinite(sep1800) and sep1800 else np.nan,
                "separation_deviation_from_ideal_sqrt_time_scaling": (carry_sep / sep1800) / math.sqrt(1800.0 / 300.0) if np.isfinite(carry_sep) and np.isfinite(sep1800) and sep1800 else np.nan,
                "plate_scale_30s_over_300s_contraction_ratio": (restart_plate / carry_plate) if np.isfinite(restart_plate) and np.isfinite(carry_plate) and carry_plate else np.nan,
                "plate_scale_300s_over_1800s_formal_contraction_ratio": (carry_plate / plate1800) if np.isfinite(carry_plate) and np.isfinite(plate1800) and plate1800 else np.nan,
                "plate_scale_deviation_from_ideal_sqrt_time_scaling": (carry_plate / plate1800) / math.sqrt(1800.0 / 300.0) if np.isfinite(carry_plate) and np.isfinite(plate1800) and plate1800 else np.nan,
                "carry_policy_independent": bool(carry_sep_ok and carry_plate_ok),
                "restart_policy_independent": bool(restart_sep_ok and restart_plate_ok),
                "carry_separation_sigma_spread": carry_sep_spread,
                "carry_plate_scale_sigma_spread": carry_plate_spread,
                "restart_separation_sigma_spread": restart_sep_spread,
                "restart_plate_scale_sigma_spread": restart_plate_spread,
                "projection_status": "formal_stationary_late_tail_projection_not_achieved_accuracy",
            }
        )
    root = pd.DataFrame(rows).sort_values(["m2_ke_nm", "draw_index", "run_name"], kind="mergesort").reset_index(drop=True) if rows else pd.DataFrame()
    amp = flatten_numeric_summary(
        root,
        ["m2_ke_nm"],
        [
            "window_local_separation_sigma_30s_uas",
            "cumulative_separation_sigma_300s_uas",
            "late_tail_projected_separation_sigma_1800s_uas",
            "window_local_plate_scale_sigma_30s_uas_per_pix",
            "cumulative_plate_scale_sigma_300s_uas_per_pix",
            "late_tail_projected_plate_scale_sigma_1800s_uas_per_pix",
            "separation_30s_over_300s_contraction_ratio",
            "separation_300s_over_1800s_formal_contraction_ratio",
            "separation_deviation_from_ideal_sqrt_time_scaling",
        ],
    )
    return root, amp


def extract_actual_error(sep: pd.DataFrame) -> dict[str, Any]:
    if sep.empty:
        return {}
    row = sep.iloc[0]
    signed = safe_float(row.get("final_sep_err_uas", row.get("signed_final_separation_error_uas", row.get("final_error_uas"))))
    sigma = safe_float(row.get("final_posterior_sigma_sep_uas", row.get("iterative_final_posterior_sigma_uas", row.get("final_posterior_sigma"))))
    return {
        "signed_final_separation_error_uas": signed,
        "absolute_final_separation_error_uas": abs(signed) if np.isfinite(signed) else np.nan,
        "iterative_final_posterior_sigma_uas": sigma,
    }


def build_accuracy(inventory: pd.DataFrame, tables_by_run: Mapping[str, Mapping[str, pd.DataFrame]], formal: pd.DataFrame, physical_rates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    rate_sep = physical_rates[physical_rates.get("physical_concept", pd.Series(dtype=str)) == "source separation"] if not physical_rates.empty else pd.DataFrame()
    formal_lookup = {row["run_name"]: row for _, row in formal.iterrows()} if not formal.empty else {}
    rate_lookup = {row["run_name"]: row for _, row in rate_sep.iterrows()} if not rate_sep.empty else {}
    for meta in included_metadata(inventory):
        sep = tables_by_run[str(meta["run_name"])].get("separation_error_summary.csv", pd.DataFrame())
        actual = extract_actual_error(sep)
        f = formal_lookup.get(meta["run_name"], {})
        r = rate_lookup.get(meta["run_name"], {})
        sigma300 = safe_float(f.get("cumulative_separation_sigma_300s_uas")) if isinstance(f, pd.Series) else np.nan
        abs_err = safe_float(actual.get("absolute_final_separation_error_uas"))
        rows.append(
            {
                "run_name": meta["run_name"],
                "m2_ke_nm": meta["m2_ke_nm"],
                "draw_index": meta["draw_index"],
                **actual,
                "formal_information_sigma_30s_uas": safe_float(f.get("window_local_separation_sigma_30s_uas")) if isinstance(f, pd.Series) else np.nan,
                "formal_information_sigma_300s_uas": sigma300,
                "formal_late_tail_projection_1800s_uas": safe_float(f.get("late_tail_projected_separation_sigma_1800s_uas")) if isinstance(f, pd.Series) else np.nan,
                "absolute_error_over_300s_formal_sigma": abs_err / sigma300 if np.isfinite(abs_err) and np.isfinite(sigma300) and sigma300 else np.nan,
                "psd_projection_fraction": safe_float(meta.get("projected_matrix_fraction")),
                "separation_information_rate": safe_float(r.get("canonical_eigenvalue_rate")) if isinstance(r, pd.Series) else np.nan,
            }
        )
    root = pd.DataFrame(rows).sort_values(["m2_ke_nm", "draw_index", "run_name"], kind="mergesort").reset_index(drop=True)
    amp = flatten_numeric_summary(
        root,
        ["m2_ke_nm"],
        [
            "signed_final_separation_error_uas",
            "absolute_final_separation_error_uas",
            "iterative_final_posterior_sigma_uas",
            "formal_information_sigma_300s_uas",
            "absolute_error_over_300s_formal_sigma",
        ],
    )
    corr_rows: list[dict[str, Any]] = []
    pairs = [
        ("signed_final_separation_error_uas", "separation_information_rate"),
        ("absolute_final_separation_error_uas", "separation_information_rate"),
        ("signed_final_separation_error_uas", "formal_information_sigma_300s_uas"),
        ("absolute_final_separation_error_uas", "formal_information_sigma_300s_uas"),
        ("absolute_final_separation_error_uas", "psd_projection_fraction"),
    ]
    for label, group in [("all", root), *[(f"m2_ke_nm={amp:g}", g) for amp, g in root.groupby("m2_ke_nm", dropna=False, sort=True)]]:
        for x, y in pairs:
            data = group[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            corr_rows.append(
                {
                    "group": label,
                    "m2_ke_nm": safe_float(label.split("=")[-1]) if label.startswith("m2") else np.nan,
                    "x_metric": x,
                    "y_metric": y,
                    "N": int(len(data)),
                    "pearson_correlation": float(data[x].corr(data[y], method="pearson")) if len(data) >= 2 else np.nan,
                    "rank_correlation": float(data[x].rank().corr(data[y].rank(), method="pearson")) if len(data) >= 2 else np.nan,
                    "interpretation": "descriptive only; no causal claim; per-amplitude N is small",
                }
            )
    return root, amp, pd.DataFrame(corr_rows)


def build_psd(inventory: pd.DataFrame, tables_by_run: Mapping[str, Mapping[str, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root_rows: list[dict[str, Any]] = []
    for meta in included_metadata(inventory):
        inv = tables_by_run[str(meta["run_name"])].get("information_rate/information_rate_input_inventory.csv", pd.DataFrame())
        psd = psd_root_summary(inv)
        root_rows.append({"run_name": meta["run_name"], "m2_ke_nm": meta["m2_ke_nm"], "draw_index": meta["draw_index"], **psd})
    root = pd.DataFrame(root_rows).sort_values(["m2_ke_nm", "draw_index", "run_name"], kind="mergesort").reset_index(drop=True) if root_rows else pd.DataFrame()
    amp_rows: list[dict[str, Any]] = []
    for amp, group in root.groupby("m2_ke_nm", dropna=False, sort=True):
        amp_rows.append(
            {
                "m2_ke_nm": amp,
                "root_count": int(group["run_name"].nunique()),
                "projected_matrix_count": int(pd.to_numeric(group["projected_matrix_count"], errors="coerce").sum()),
                "accepted_matrix_count": int(pd.to_numeric(group["accepted_matrix_count"], errors="coerce").sum()),
                "mean_projected_matrix_fraction": float(pd.to_numeric(group["projected_matrix_fraction"], errors="coerce").mean()),
                "clipped_eigenvalue_count": int(pd.to_numeric(group["clipped_eigenvalue_count"], errors="coerce").sum()),
                "maximum_raw_negative_magnitude": float(pd.to_numeric(group["maximum_raw_negative_magnitude"], errors="coerce").max()),
                "maximum_relative_frobenius_correction": float(pd.to_numeric(group["maximum_relative_projection_correction"], errors="coerce").max()),
                "maximum_absolute_correction": float(pd.to_numeric(group["maximum_absolute_correction"], errors="coerce").max()),
                "materially_indefinite_count": int(pd.to_numeric(group["materially_indefinite_count"], errors="coerce").sum()),
                "projection_status_counts": semicolon_join(group["projection_status_counts"]),
            }
        )
    amp_df = pd.DataFrame(amp_rows)
    conclusions: list[str] = []
    if not amp_df.empty:
        inactive = amp_df[(pd.to_numeric(amp_df["m2_ke_nm"], errors="coerce") <= 0.1) & (pd.to_numeric(amp_df["projected_matrix_count"], errors="coerce") == 0)]
        if set(np.round(inactive["m2_ke_nm"].astype(float), 12)) == set(np.round([0.01, 0.05, 0.1], 12)):
            conclusions.append("Projection is inactive through 0.1 nm.")
        intermittent = amp_df[(np.isclose(pd.to_numeric(amp_df["m2_ke_nm"], errors="coerce"), 0.5)) & (pd.to_numeric(amp_df["mean_projected_matrix_fraction"], errors="coerce").between(0.0, 1.0, inclusive="neither"))]
        if not intermittent.empty:
            conclusions.append("Projection becomes intermittent at 0.5 nm.")
        widespread = amp_df[(np.isclose(pd.to_numeric(amp_df["m2_ke_nm"], errors="coerce"), 1.0)) & (pd.to_numeric(amp_df["mean_projected_matrix_fraction"], errors="coerce") > 0.5)]
        if not widespread.empty:
            conclusions.append("Projection is widespread at 1.0 nm.")
        if pd.to_numeric(amp_df["maximum_relative_frobenius_correction"], errors="coerce").max() < 1.0e-13:
            conclusions.append("Projection corrections remain below 1e-13 relative scale.")
    return root, amp_df, {"conclusions": conclusions}


def make_plots(outdir: Path, frames: Mapping[str, pd.DataFrame], warnings: list[str]) -> list[str]:
    paths: list[str] = []
    if plt is None:
        warnings.append("matplotlib_unavailable")
        return paths
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(plot_dir / name, dpi=150)
        plt.close(fig)
        paths.append(f"plots/{name}")

    def scatter_by_concept(ax: Any, df: pd.DataFrame, x: str, y: str, concepts: Sequence[str] | None = None) -> None:
        use = df if concepts is None else df[df["physical_concept"].isin(concepts)]
        for concept, group in use.groupby("physical_concept", dropna=False, sort=True):
            ax.scatter(group[x], group[y], label=str(concept), alpha=0.75)
            med = group.groupby(x)[y].median().reset_index()
            ax.plot(med[x], med[y], linewidth=1.0)

    try:
        df = frames["physical_rates_root"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not df.empty:
            scatter_by_concept(ax, df, "m2_ke_nm", "canonical_eigenvalue_rate", ["source separation", "plate scale", "total/log flux", "contrast"])
            ax.set_yscale("log")
            ax.legend(fontsize=7)
        ax.set_xlabel("M2 high-order WFE KE (nm)")
        ax.set_ylabel("prior-normalized gain rate (1/s)")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[0])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[0]}:{exc}")
    try:
        df = frames["assignments_root"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not df.empty:
            scatter_by_concept(ax, df, "m2_ke_nm", "squared_loading_used_for_assignment", ["source separation", "plate scale", "total/log flux", "contrast"])
            ax.legend(fontsize=7)
        ax.set_xlabel("M2 high-order WFE KE (nm)")
        ax.set_ylabel("assignment squared loading")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[1])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[1]}:{exc}")
    try:
        df = frames["gain3_root"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not df.empty:
            use = df[df["policy_mode_set_name"].isin(["astrometric_core", "high_information_calibration"])]
            xpos = np.arange(len(use))
            ax.scatter(xpos, use["first_natural_block_length"], label="first", alpha=0.7)
            ax.scatter(xpos, use["second_natural_block_length"], label="second", alpha=0.7)
            ax.set_xticks([])
            ax.legend(fontsize=7)
        ax.set_xlabel("gain-3 root/policy rows")
        ax.set_ylabel("natural block length (subblocks)")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[2])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[2]}:{exc}")
    try:
        df = frames["gain3_amp"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not df.empty:
            use = df[df["policy_mode_set_name"].isin(HEADLINE_POLICIES)]
            for policy, group in use.groupby("policy_mode_set_name", sort=True):
                if "fraction_roots_with_at_least_two_natural_triggers" in group:
                    ax.plot(group["m2_ke_nm"], group["fraction_roots_with_at_least_two_natural_triggers"], marker="o", label=f"{policy} two natural")
                if "maximum_latency_fraction_median" in group:
                    ax.plot(group["m2_ke_nm"], group["maximum_latency_fraction_median"], marker="x", linestyle="--", label=f"{policy} latency")
            ax.legend(fontsize=6)
        ax.set_xlabel("M2 high-order WFE KE (nm)")
        ax.set_ylabel("fraction")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[3])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[3]}:{exc}")
    try:
        df = frames["controlling_amp"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not df.empty:
            use = df[df["policy_mode_set_name"] == "source_core"]
            for concept, group in use.groupby("physical_interpretation", sort=True):
                ax.plot(group["m2_ke_nm"], group["controlling_update_fraction"], marker="o", label=str(concept))
            ax.legend(fontsize=7)
        ax.set_xlabel("M2 high-order WFE KE (nm)")
        ax.set_ylabel("source-core controlling fraction")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[4])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[4]}:{exc}")
    try:
        df = frames["accuracy_root"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not df.empty:
            ax.scatter(df["m2_ke_nm"], df["signed_final_separation_error_uas"], label="signed actual error", alpha=0.75)
            ax.scatter(df["m2_ke_nm"], df["absolute_final_separation_error_uas"], label="absolute actual error", alpha=0.75)
            ax.scatter(df["m2_ke_nm"], df["formal_information_sigma_300s_uas"], label="formal 300s sigma", alpha=0.75)
            ax.set_yscale("symlog", linthresh=1.0)
            ax.legend(fontsize=7)
        ax.set_xlabel("M2 high-order WFE KE (nm)")
        ax.set_ylabel("separation quantity (uas)")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[5])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[5]}:{exc}")
    try:
        df = frames["accuracy_root"]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        if not df.empty:
            for amp, group in df.groupby("m2_ke_nm", sort=True):
                ax.scatter(group["formal_information_sigma_300s_uas"], group["absolute_final_separation_error_uas"], label=f"{float(amp):g} nm", alpha=0.75)
            ax.legend(fontsize=7)
        ax.set_xlabel("formal 300s sigma (uas)")
        ax.set_ylabel("absolute final separation error (uas)")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[6])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[6]}:{exc}")
    try:
        df = frames["quasi_stability"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not df.empty:
            ax.scatter(df["early_half_median_singular_value"], df["late_half_median_singular_value"], alpha=0.75)
            ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, alpha=0.4)
        ax.set_xlabel("early median subspace singular value")
        ax.set_ylabel("late median subspace singular value")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[7])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[7]}:{exc}")
    try:
        df = frames["psd_root"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if not df.empty:
            ax.scatter(df["m2_ke_nm"], df["projected_matrix_fraction"], alpha=0.75)
            med = df.groupby("m2_ke_nm")["projected_matrix_fraction"].median().reset_index()
            ax.plot(med["m2_ke_nm"], med["projected_matrix_fraction"], color="black", marker="o", linewidth=1.0)
        ax.set_xlabel("M2 high-order WFE KE (nm)")
        ax.set_ylabel("projected summary fraction")
        ax.grid(True, alpha=0.25)
        save(fig, PLOT_NAMES[8])
    except Exception as exc:
        warnings.append(f"{PLOT_NAMES[8]}:{exc}")
    return paths


def markdown_table(df: pd.DataFrame, max_rows: int = 8) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    cols = list(view.columns)
    lines = [
        "| " + " | ".join(str(col) for col in cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in view.iterrows():
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append("" if not np.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_markdown(summary: Mapping[str, Any], frames: Mapping[str, pd.DataFrame]) -> str:
    validation = summary["input_validation"]
    headlines = [
        f"Roots discovered: {validation['roots_discovered']}; included: {validation['roots_included']}; excluded: {validation['roots_excluded']}.",
        f"Amplitudes represented: {', '.join(str(v) for v in summary['amplitude_list'])}.",
        "Canonical mode IDs are treated as root-local; family physical summaries are joined through `adaptive_mode_set_resolution.csv`.",
        "Sequential products are covariance-only frozen-factor diagnostics and do not simulate posterior means or innovation gates.",
        "The two sequential scopes are reported separately: cumulative observation carry and window-local restart.",
        "Formal covariance contraction is reported beside actual estimator error; formal sigma is not estimator accuracy.",
    ]
    psd_conclusions = summary.get("PSD-projection summary", {}).get("conclusions", [])
    headlines.extend(psd_conclusions[:2])
    lines = ["# M2-Center Information-Rate Family Summary", ""]
    for item in headlines[:8]:
        lines.append(f"- {item}")
    sections = [
        ("1. Dataset and audit validity", markdown_table(frames["input_inventory"][["run_name", "m2_ke_nm", "draw_index", "inclusion_status", "exclusion_reason"]], 6)),
        ("2. Physical mode assignment consistency", markdown_table(frames["assignments_amp"], 8)),
        ("3. Information-rate hierarchy and stability", markdown_table(frames["physical_rates_amp"], 8)),
        ("4. Fixed-prior acquisition timing", markdown_table(frames["fixed_candidates_amp"], 8)),
        ("5. Sequential acquisition behavior", markdown_table(frames["sequential_amp"], 8)),
        ("6. Gain-3 astrometric policy", markdown_table(frames["gain3_amp"][frames["gain3_amp"].get("policy_mode_set_name", pd.Series(dtype=str)).isin(HEADLINE_POLICIES)] if not frames["gain3_amp"].empty else pd.DataFrame(), 8)),
        ("7. Source-core contrast limitation", markdown_table(frames["controlling_amp"][frames["controlling_amp"].get("policy_mode_set_name", pd.Series(dtype=str)) == "source_core"] if not frames["controlling_amp"].empty else pd.DataFrame(), 8)),
        ("8. High-information WFE policy comparison", markdown_table(frames["schedule_equivalence"], 8)),
        ("9. Quasi-degenerate M2 subspaces", markdown_table(frames["quasi_amp"], 8)),
        ("10. Formal uncertainty versus actual estimator error", markdown_table(frames["accuracy_amp"], 8)),
        ("11. PSD-projection numerical diagnostics", markdown_table(frames["psd_amp"], 8)),
    ]
    for title, table in sections:
        lines.extend(["", f"## {title}", "", table])
    lines.extend(
        [
            "",
            "## 12. Guidance for future implementation",
            "",
            "The family products support evaluating an acquisition gate based on `astrometric_core`, with `high_information_calibration` included only when schedule-equivalence rows show it does not lengthen the information-only schedule. A gain threshold near 3 corresponds to a single-update sigma target of 0.5. This should be treated as acquisition support before a fixed-reference precision-accumulation phase, not as an indefinitely recurring operational cadence.",
            "",
            "Contrast and weak modes should continue to be estimated and monitored, but the source-core controlling-mode table should be used before allowing them to control the main astrometric cadence. Any future controller also needs innovation, score, or reference-stability checks; information support alone does not determine whether an estimator update is necessary or unbiased.",
            "",
            "## 13. Limitations",
            "",
            "The aggregation is read-only and does not recompute Fisher matrices, rerun inference, or implement a controller. The 1800-second values are formal stationary late-tail projections, not achieved 30-minute astrometric accuracy. Quasi-degenerate groups should be interpreted through subspace singular values and principal angles rather than individual mode identity.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.per_root_dir is None and not args.review_root and not args.review_glob:
        raise SystemExit("Provide --per-root-dir, --review-root, or --review-glob.")
    expected_amplitudes = [float(token) for token in args.expected_amplitudes.split(",") if token.strip()]
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    original_roots = read_root_list(args.root_list)
    review_roots = discover_review_roots(args)
    inventory, tables_by_run, validation_diag = build_input_inventory(review_roots, original_roots, expected_commit=args.expected_commit, strict=args.strict)
    expectation_issues = validate_family_expectations(
        inventory,
        expected_root_count=args.expected_root_count,
        expected_draws_per_amplitude=args.expected_draws_per_amplitude,
        expected_amplitudes=expected_amplitudes,
        strict=args.strict,
    )

    rates = collect_table(inventory, tables_by_run, "information_rate/information_rate_by_mode.csv")
    window_rates = collect_table(inventory, tables_by_run, "information_rate/information_rate_by_window_mode.csv")
    resolution = collect_table(inventory, tables_by_run, "information_rate/adaptive_mode_set_resolution.csv")
    seq_summary = collect_table(inventory, tables_by_run, "information_rate/adaptive_cadence_sequential_summary.csv")
    updates = collect_table(inventory, tables_by_run, "information_rate/adaptive_cadence_sequential_updates.csv")
    gains = collect_table(inventory, tables_by_run, "information_rate/adaptive_cadence_sequential_mode_gains.csv")
    candidates = collect_table(inventory, tables_by_run, "information_rate/adaptive_cadence_candidates.csv")
    prefix = collect_table(inventory, tables_by_run, "information_rate/adaptive_cadence_prefix_diagnostics.csv")
    physical = collect_table(inventory, tables_by_run, "information_rate/information_by_physical_label.csv")
    overlaps = collect_table(inventory, tables_by_run, "information_rate/mode_overlap.csv")
    quasi = collect_table(inventory, tables_by_run, "information_rate/quasi_degenerate_subspace_summary.csv")

    assignments_root, assignments_amp = build_physical_assignments(resolution, rates)
    physical_rates_root, physical_rates_amp, stability_root, stability_amp = build_physical_rates(assignments_root, rates, window_rates)
    sequential_root, sequential_amp = build_sequential(seq_summary)
    gain3_root, gain3_amp, gain3_window = build_gain3(updates, args.headline_gain_threshold)
    schedule_equivalence = build_schedule_equivalence(updates)
    controlling_root, controlling_amp = build_controlling_modes(gains, updates, assignments_root)
    fixed_root, fixed_amp = build_fixed_prior(candidates, prefix)
    quasi_root, quasi_amp, quasi_stability = build_quasi(quasi, overlaps)
    formal_root, formal_amp = build_formal_uncertainty(sequential_root, physical, args.strict)
    accuracy_root, accuracy_amp, correlations = build_accuracy(inventory, tables_by_run, formal_root, physical_rates_root)
    psd_root, psd_amp, psd_summary_extra = build_psd(inventory, tables_by_run)

    frames = {
        "input_inventory": inventory,
        "assignments_root": assignments_root,
        "assignments_amp": assignments_amp,
        "physical_rates_root": physical_rates_root,
        "physical_rates_amp": physical_rates_amp,
        "stability_root": stability_root,
        "stability_amp": stability_amp,
        "sequential_root": sequential_root,
        "sequential_amp": sequential_amp,
        "gain3_root": gain3_root,
        "gain3_amp": gain3_amp,
        "gain3_window": gain3_window,
        "schedule_equivalence": schedule_equivalence,
        "controlling_root": controlling_root,
        "controlling_amp": controlling_amp,
        "fixed_candidates_root": fixed_root,
        "fixed_candidates_amp": fixed_amp,
        "quasi_root": quasi_root,
        "quasi_amp": quasi_amp,
        "quasi_stability": quasi_stability,
        "formal_root": formal_root,
        "formal_amp": formal_amp,
        "accuracy_root": accuracy_root,
        "accuracy_amp": accuracy_amp,
        "correlations": correlations,
        "psd_root": psd_root,
        "psd_amp": psd_amp,
    }
    output_map = {
        "family_input_inventory.csv": inventory,
        "family_physical_mode_assignments_by_root.csv": assignments_root,
        "family_physical_mode_assignments_by_amplitude.csv": assignments_amp,
        "family_physical_information_rates_by_root.csv": physical_rates_root,
        "family_physical_information_rates_by_amplitude.csv": physical_rates_amp,
        "family_physical_rate_stability_by_root.csv": stability_root,
        "family_physical_rate_stability_by_amplitude.csv": stability_amp,
        "family_sequential_policy_by_root.csv": sequential_root,
        "family_sequential_policy_by_amplitude.csv": sequential_amp,
        "family_gain3_acquisition_by_root.csv": gain3_root,
        "family_gain3_acquisition_by_amplitude.csv": gain3_amp,
        "family_gain3_window_restart_events.csv": gain3_window,
        "family_policy_schedule_equivalence.csv": schedule_equivalence,
        "family_controlling_modes.csv": controlling_root,
        "family_controlling_modes_by_amplitude.csv": controlling_amp,
        "family_fixed_prior_candidates.csv": fixed_root,
        "family_fixed_prior_candidates_by_amplitude.csv": fixed_amp,
        "family_quasi_degenerate_subspaces_by_root.csv": quasi_root,
        "family_quasi_degenerate_subspaces_by_amplitude.csv": quasi_amp,
        "family_quasi_subspace_window_stability.csv": quasi_stability,
        "family_formal_uncertainty_by_root.csv": formal_root,
        "family_formal_uncertainty_by_amplitude.csv": formal_amp,
        "family_accuracy_and_information_by_root.csv": accuracy_root,
        "family_accuracy_and_information_by_amplitude.csv": accuracy_amp,
        "family_accuracy_information_correlations.csv": correlations,
        "family_psd_projection_by_root.csv": psd_root,
        "family_psd_projection_by_amplitude.csv": psd_amp,
    }
    for name, frame in output_map.items():
        write_csv(frame, outdir / name)

    plot_warnings: list[str] = []
    plot_paths = [] if args.no_plots else make_plots(outdir, frames, plot_warnings)
    amplitude_counts = inventory[inventory["inclusion_status"] == "included"].groupby("m2_ke_nm")["draw_index"].nunique().to_dict() if not inventory.empty else {}
    warning_inventory = Counter()
    for text in inventory.get("warning_statuses", pd.Series(dtype=str)).astype(str):
        for token in text.split(";"):
            if token:
                warning_inventory[token.split(":")[0]] += int(token.split(":")[1]) if ":" in token and token.split(":")[1].isdigit() else 1
    validation_summary = {
        "roots_discovered": int(len(inventory)),
        "roots_included": int((inventory["inclusion_status"] == "included").sum()) if not inventory.empty else 0,
        "roots_excluded": int((inventory["inclusion_status"] != "included").sum()) if not inventory.empty else 0,
        "commit_matches": int((inventory["commit_matches_expected"] == True).sum()) if args.expected_commit and not inventory.empty else None,
        "status_ok": int((inventory["status"] == "ok").sum()) if not inventory.empty else 0,
        "accepted_summaries_300": int((inventory["accepted_summary_count"] == 300).sum()) if not inventory.empty else 0,
        "sequential_rows_40": int((inventory["sequential_summary_row_count"] == 40).sum()) if not inventory.empty else 0,
        "invariance_pass": int(inventory["invariance_all_pass"].astype(bool).sum()) if not inventory.empty else 0,
        "materially_indefinite_inputs": int(pd.to_numeric(psd_root.get("materially_indefinite_count", pd.Series(dtype=float)), errors="coerce").sum()) if not psd_root.empty else 0,
        "expectation_issues": expectation_issues,
        "settings_consistent": validation_diag["settings_consistent"],
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_utc(),
        "script_path": str(SCRIPT_PATH),
        "script_git_commit": repository_commit(),
        "expected_input_commit": args.expected_commit,
        "per_root_directory": str(args.per_root_dir) if args.per_root_dir else "",
        "root_list_path": str(args.root_list) if args.root_list else "",
        "root_count": int(len(inventory)),
        "amplitude_list": sorted(float(v) for v in inventory["m2_ke_nm"].dropna().unique()) if not inventory.empty else [],
        "draws_per_amplitude": {str(k): int(v) for k, v in amplitude_counts.items()},
        "input_validation": validation_summary,
        "warning_inventory": dict(sorted(warning_inventory.items())),
        "information_rate_settings": validation_diag["information_rate_settings"],
        "physical_assignment_summary": assignments_amp.to_dict(orient="records"),
        "physical_rate_summary": physical_rates_amp.to_dict(orient="records"),
        "gain_3_acquisition_summary": gain3_amp.to_dict(orient="records"),
        "policy_equivalence_summary": schedule_equivalence[schedule_equivalence.get("run_name", pd.Series(dtype=str)) == "__amplitude_summary__"].to_dict(orient="records") if not schedule_equivalence.empty else [],
        "source_core_controlling_mode_summary": controlling_amp[controlling_amp.get("policy_mode_set_name", pd.Series(dtype=str)) == "source_core"].to_dict(orient="records") if not controlling_amp.empty else [],
        "quasi_degenerate_subspace_summary": quasi_amp.to_dict(orient="records"),
        "formal_uncertainty_summary": formal_amp.to_dict(orient="records"),
        "actual_estimator_error_summary": accuracy_amp.to_dict(orient="records"),
        "accuracy_information_correlations": correlations.to_dict(orient="records"),
        "PSD-projection summary": {**psd_summary_extra, "by_amplitude": psd_amp.to_dict(orient="records")},
        "output_file_inventory": sorted([*output_map.keys(), "family_information_rate_summary.json", "family_information_rate_summary.md", "family_aggregation_manifest.json", *plot_paths]),
        "caveats": [
            "Formal covariance accumulation is not actual estimator accuracy.",
            "Sequential cadence outputs are frozen-factor covariance-only diagnostics.",
            "Observation-carry and window-restart scopes are not pooled.",
            "Canonical mode IDs are root-local.",
            "All-trackable membership is threshold dependent.",
            "The 1800-second result is a formal stationary late-tail projection, not achieved 30-minute accuracy.",
        ],
        "future-controller guidance": [
            "Evaluate an astrometric_core acquisition gate near gain 3.",
            "Include high-information WFE modes only where schedule-equivalence diagnostics show no cadence penalty.",
            "Use a small number of information-supported acquisition updates before fixed-reference precision accumulation.",
            "Do not let contrast or weak modes control main astrometric cadence unless controlling-mode evidence supports it.",
            "Combine information support with innovation or reference-stability checks before implementation.",
        ],
    }
    manifest = {
        "invocation_arguments": vars(args),
        "generated_at": summary["generated_at"],
        "repository_commit": summary["script_git_commit"],
        "expected_input_commit": args.expected_commit,
        "observed_completion_sentinel_commits": sorted(set(inventory["recorded_commit"].astype(str))) if not inventory.empty else [],
        "input_per_root_directory": str(args.per_root_dir) if args.per_root_dir else "",
        "root_list_path": str(args.root_list) if args.root_list else "",
        "root_count": int(len(inventory)),
        "amplitude_and_draw_inventory": {str(k): sorted(int(v) for v in g["draw_index"].dropna().unique()) for k, g in inventory.groupby("m2_ke_nm", dropna=False)} if not inventory.empty else {},
        "required_input_files": list(TOP_LEVEL_REQUIRED + INFORMATION_REQUIRED),
        "optional_input_files": list(OPTIONAL_INPUTS),
        "missing_file_inventory": validation_diag["missing_files"],
        "missing_expected_run_names": validation_diag["missing_expected_run_names"],
        "unexpected_review_run_names": validation_diag["unexpected_review_run_names"],
        "output_file_inventory": summary["output_file_inventory"],
        "warnings": plot_warnings,
        "strict_mode_status": "pass" if args.strict and validation_summary["roots_excluded"] == 0 and not expectation_issues else ("non_strict" if not args.strict else "failed"),
    }
    write_json(outdir / "family_information_rate_summary.json", summary)
    write_json(outdir / "family_aggregation_manifest.json", manifest)
    (outdir / "family_information_rate_summary.md").write_text(build_markdown(summary, frames), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
