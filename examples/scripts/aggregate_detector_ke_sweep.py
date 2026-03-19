"""Aggregate detector knowledge-error sweeps across prescribed-MC experiments.

This script combines multiple *experiment directories* (for example `ke_1e-3`,
`ke_1e-2`, ...) into two outer-sweep CSVs:

- `sweep_runs.csv`: one row per run across all loaded experiments.
- `sweep_summary.csv`: one row per detector knowledge-error setting with grouped
  error statistics across all infer quantities present in the run tables.

Expected layout
---------------
Each experiment directory is expected to contain:

- `prescription.yaml` (or `prescription.yml` / `prescription.json`)
- `results.csv` written by `examples/recipes/prescribed_monte_carlo.py`
  in **row orientation** (`run_id` is a data column).

Typical sweep root layout:

```
Results/detector_ke_sweep/
  ke_0/
    prescription.yaml
    results.csv
    manifest.json
    runs/
  ke_1e-3/
    prescription.yaml
    results.csv
    ...
```

What is read from each `prescription.*`
---------------------------------------
Detector mismatch metadata is sourced from:

- `experiment.inference_system.detector.layers[*]`

for `pixel_offsets` and `pixel_response` (when present), including:

- `knowledge_error.model`
- `knowledge_error.scale`
- calibration paths (`dx_path`, `dy_path`, `prf_path`)

When `experiment.inference_system` is absent, this script treats the experiment
as having no inference-side detector mismatch and records:

- `pixel_offsets_knowledge_error_scale = 0.0`
- `pixel_response_knowledge_error_scale = 0.0`
- model/path columns as null

What is read from each `results.csv`
------------------------------------
Run-level columns are preserved as written by prescribed-MC (row orientation),
including flattened scalar/vector infer-key outputs like:

- `truth.<infer_key>`
- `init.<infer_key>`
- `final.<infer_key>`
- `init_delta.<infer_key>`
- `final_delta.<infer_key>`

For vector-valued infer keys (for example Zernikes), row-oriented outputs are
already flattened into component columns such as
`final_delta.optics.primary.zernike_coeffs_nm[0]`. This script summarizes each
component independently.

Derived outputs
---------------
`sweep_runs.csv` adds sweep metadata plus:

- `sep_error_as`
- `abs_sep_error_as`

where separation error uses:

1. `final_delta.source.separation_as` when available
2. fallback `final.source.separation_as - truth.source.separation_as`

`sweep_summary.csv` groups by detector KE setting and reports:

- `n_total`
- `n_success`
- per-quantity metrics:
  - `<quantity>_mean_bias`
  - `<quantity>_std_bias`
  - `<quantity>_mean_abs_error`
  - `<quantity>_median_abs_error`
  - `<quantity>_rmse`

Examples
--------
Aggregate a default sweep root:

```
python examples/scripts/aggregate_detector_ke_sweep.py \
  --root Results/detector_ke_sweep
```

Strict mode (fail on missing/malformed experiments):

```
python examples/scripts/aggregate_detector_ke_sweep.py \
  --root Results/detector_ke_sweep \
  --strict
```
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from dluxshera.config.io import load_config_file

ERROR_METRIC_SUFFIXES = (
    "mean_bias",
    "std_bias",
    "mean_abs_error",
    "median_abs_error",
    "rmse",
)


class AggregationStats:
    """Simple container for experiment discovery/load counts."""

    def __init__(self, discovered: int, loaded: int, skipped: int) -> None:
        self.discovered = int(discovered)
        self.loaded = int(loaded)
        self.skipped = int(skipped)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for detector KE sweep aggregation."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate many prescribed-MC detector knowledge-error experiment "
            "directories into sweep_runs.csv and sweep_summary.csv."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("Results/detector_ke_sweep"),
        help="Sweep root directory containing experiment subdirectories (default: Results/detector_ke_sweep).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="ke_*",
        help="Glob pattern for experiment directories under --root (default: ke_*).",
    )
    parser.add_argument(
        "--out-runs",
        type=Path,
        default=Path("sweep_runs.csv"),
        help="Output filename/path for run-level CSV (default: sweep_runs.csv).",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("sweep_summary.csv"),
        help="Output filename/path for grouped summary CSV (default: sweep_summary.csv).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on missing/malformed experiment inputs instead of warning+skip.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-directory processing logs.",
    )
    return parser.parse_args()


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message)


def _fail_or_warn(message: str, *, strict: bool) -> None:
    if strict:
        raise ValueError(message)
    _warn(message)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            out = float(stripped)
        except ValueError:
            return None
        if not math.isfinite(out):
            return None
        return out
    return None


def _first_layer_by_name(layers: Any, layer_name: str) -> dict[str, Any] | None:
    if not isinstance(layers, list):
        return None
    for layer in layers:
        if isinstance(layer, dict) and layer.get("name") == layer_name:
            return layer
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def extract_detector_knowledge_error_metadata(prescription: dict[str, Any]) -> dict[str, Any]:
    """Extract inference-detector knowledge-error metadata from a prescription."""
    out: dict[str, Any] = {
        "inference_system_present": False,
        "pixel_offsets_knowledge_error_scale": 0.0,
        "pixel_offsets_knowledge_error_model": None,
        "pixel_offsets_dx_path": None,
        "pixel_offsets_dy_path": None,
        "pixel_response_knowledge_error_scale": 0.0,
        "pixel_response_knowledge_error_model": None,
        "pixel_response_prf_path": None,
    }

    experiment_cfg = prescription.get("experiment")
    if not isinstance(experiment_cfg, dict):
        return out
    inference_system = experiment_cfg.get("inference_system")
    if not isinstance(inference_system, dict):
        return out

    out["inference_system_present"] = True
    detector_cfg = inference_system.get("detector")
    if not isinstance(detector_cfg, dict):
        return out
    layers = detector_cfg.get("layers")

    offsets_layer = _first_layer_by_name(layers, "pixel_offsets")
    if offsets_layer is not None:
        out["pixel_offsets_dx_path"] = _string_or_none(offsets_layer.get("dx_path"))
        out["pixel_offsets_dy_path"] = _string_or_none(offsets_layer.get("dy_path"))
        ke_cfg = offsets_layer.get("knowledge_error")
        if isinstance(ke_cfg, dict):
            out["pixel_offsets_knowledge_error_model"] = _string_or_none(ke_cfg.get("model"))
            scale = _coerce_float(ke_cfg.get("scale"))
            if scale is not None:
                out["pixel_offsets_knowledge_error_scale"] = scale

    response_layer = _first_layer_by_name(layers, "pixel_response")
    if response_layer is not None:
        out["pixel_response_prf_path"] = _string_or_none(response_layer.get("prf_path"))
        ke_cfg = response_layer.get("knowledge_error")
        if isinstance(ke_cfg, dict):
            out["pixel_response_knowledge_error_model"] = _string_or_none(ke_cfg.get("model"))
            scale = _coerce_float(ke_cfg.get("scale"))
            if scale is not None:
                out["pixel_response_knowledge_error_scale"] = scale

    return out


def _find_prescription_path(experiment_dir: Path) -> Path | None:
    preferred = [
        experiment_dir / "prescription.yaml",
        experiment_dir / "prescription.yml",
        experiment_dir / "prescription.json",
    ]
    for candidate in preferred:
        if candidate.exists() and candidate.is_file():
            return candidate

    matches = sorted(
        path
        for path in experiment_dir.glob("prescription.*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}
    )
    if not matches:
        return None
    return matches[0]


def _read_row_results_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError("results.csv has no header row.")
        if "run_id" not in fieldnames:
            if "key" in fieldnames:
                raise ValueError(
                    "results.csv appears to be column-oriented ('key' column present). "
                    "This sweep aggregator expects row-oriented results.csv."
                )
            raise ValueError("results.csv missing required 'run_id' column.")

        rows = [dict(row) for row in reader]

    if not rows:
        raise ValueError("results.csv has no data rows.")
    return rows, list(fieldnames)


def _merge_column_order(existing: list[str], new_columns: list[str]) -> list[str]:
    merged = list(existing)
    seen = set(existing)
    for column in new_columns:
        if column in seen:
            continue
        merged.append(column)
        seen.add(column)
    return merged


def _component_sort_key(name: str) -> tuple[str, int]:
    match = re.match(r"^(.*)\[(\d+)\]$", name)
    if match:
        return match.group(1), int(match.group(2))
    return name, -1


def discover_error_components(result_columns: list[str]) -> list[str]:
    """Discover infer-quantity components that can produce final errors."""
    final_delta_components = {
        column[len("final_delta."):]
        for column in result_columns
        if column.startswith("final_delta.")
    }
    final_components = {
        column[len("final."):]
        for column in result_columns
        if column.startswith("final.")
    }
    truth_components = {
        column[len("truth."):]
        for column in result_columns
        if column.startswith("truth.")
    }
    fallback_components = final_components & truth_components
    discovered = final_delta_components | fallback_components
    return sorted(discovered, key=_component_sort_key)


def compute_component_error(row: dict[str, Any], component: str) -> float | None:
    """Compute final error for one infer component.

    Preference order:
    1) `final_delta.<component>`
    2) `final.<component> - truth.<component>`
    """
    direct = _coerce_float(row.get(f"final_delta.{component}"))
    if direct is not None:
        return direct

    final_val = _coerce_float(row.get(f"final.{component}"))
    truth_val = _coerce_float(row.get(f"truth.{component}"))
    if final_val is None or truth_val is None:
        return None
    return final_val - truth_val


def _is_success_status(value: Any) -> bool:
    if value is None:
        return False
    status = str(value).strip().lower()
    return status in {"ok", "success", "succeeded", "complete", "completed"}


def _group_sort_key(group_row: dict[str, Any]) -> tuple[float, str]:
    scale = _coerce_float(group_row.get("pixel_offsets_knowledge_error_scale"))
    scale_sort = scale if scale is not None else float("inf")
    model = _string_or_none(group_row.get("pixel_offsets_knowledge_error_model")) or ""
    return scale_sort, model


def build_sweep_summary_rows(
    run_rows: list[dict[str, Any]],
    components: list[str],
) -> list[dict[str, Any]]:
    """Build grouped summary rows keyed by pixel-offsets KE setting."""
    grouped: dict[tuple[float | None, str | None], list[dict[str, Any]]] = {}
    for row in run_rows:
        scale = _coerce_float(row.get("pixel_offsets_knowledge_error_scale"))
        model = _string_or_none(row.get("pixel_offsets_knowledge_error_model"))
        grouped.setdefault((scale, model), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (scale, model), rows in grouped.items():
        sweep_labels = sorted(
            {
                str(value)
                for value in (row.get("sweep_label") for row in rows)
                if value not in (None, "")
            }
        )
        experiment_dirs = sorted(
            {
                str(value)
                for value in (row.get("experiment_dir") for row in rows)
                if value not in (None, "")
            }
        )
        base_row: dict[str, Any] = {
            "pixel_offsets_knowledge_error_scale": scale,
            "pixel_offsets_knowledge_error_model": model,
            "pixel_response_knowledge_error_scale": _coerce_float(
                rows[0].get("pixel_response_knowledge_error_scale")
            ),
            "pixel_response_knowledge_error_model": rows[0].get(
                "pixel_response_knowledge_error_model"
            ),
            "sweep_labels": ";".join(sweep_labels),
            "experiment_dirs": ";".join(experiment_dirs),
            "n_total": len(rows),
            "n_success": sum(1 for row in rows if _is_success_status(row.get("status"))),
        }

        for component in components:
            values = [
                value
                for value in (compute_component_error(row, component) for row in rows)
                if value is not None
            ]
            if not values:
                continue

            arr = np.asarray(values, dtype=float)
            abs_arr = np.abs(arr)
            base_row[f"{component}_mean_bias"] = float(np.mean(arr))
            base_row[f"{component}_std_bias"] = float(np.std(arr))
            base_row[f"{component}_mean_abs_error"] = float(np.mean(abs_arr))
            base_row[f"{component}_median_abs_error"] = float(np.median(abs_arr))
            base_row[f"{component}_rmse"] = float(np.sqrt(np.mean(arr**2)))

        summary_rows.append(base_row)

    return sorted(summary_rows, key=_group_sort_key)


def _resolve_output_path(root: Path, out_path: Path) -> Path:
    if out_path.is_absolute():
        return out_path
    return root / out_path


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate_detector_ke_sweep(
    *,
    root: Path,
    pattern: str = "ke_*",
    strict: bool = False,
    verbose: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[str],
    AggregationStats,
]:
    """Aggregate run rows and grouped summary rows for a detector KE sweep."""
    discovered_dirs = sorted(path for path in root.glob(pattern) if path.is_dir())
    discovered = len(discovered_dirs)
    loaded = 0
    skipped = 0

    merged_rows: list[dict[str, Any]] = []
    result_columns: list[str] = []

    for experiment_dir in discovered_dirs:
        _log(f"[scan] {experiment_dir}", verbose=verbose)
        prescription_path = _find_prescription_path(experiment_dir)
        if prescription_path is None:
            skipped += 1
            _fail_or_warn(
                f"{experiment_dir}: no prescription.yaml/.yml/.json found.",
                strict=strict,
            )
            continue

        results_path = experiment_dir / "results.csv"
        if not results_path.exists():
            skipped += 1
            _fail_or_warn(
                f"{experiment_dir}: missing results.csv; skipping experiment.",
                strict=strict,
            )
            continue

        try:
            prescription = load_config_file(prescription_path)
            metadata = extract_detector_knowledge_error_metadata(prescription)
            run_rows, fieldnames = _read_row_results_csv(results_path)
        except Exception as exc:
            skipped += 1
            _fail_or_warn(
                f"{experiment_dir}: failed to load inputs ({exc}); skipping experiment.",
                strict=strict,
            )
            continue

        result_columns = _merge_column_order(result_columns, fieldnames)
        rel_dir = experiment_dir.relative_to(root).as_posix()
        sweep_label = experiment_dir.name

        for row in run_rows:
            merged_row: dict[str, Any] = dict(row)
            merged_row.update(metadata)
            merged_row["experiment_dir"] = rel_dir
            merged_row["sweep_label"] = sweep_label
            merged_row["prescription_path"] = prescription_path.name

            sep_error = compute_component_error(merged_row, "source.separation_as")
            merged_row["sep_error_as"] = sep_error
            merged_row["abs_sep_error_as"] = abs(sep_error) if sep_error is not None else None
            merged_rows.append(merged_row)

        loaded += 1
        _log(
            f"[ok] {experiment_dir.name}: loaded {len(run_rows)} run rows from results.csv",
            verbose=verbose,
        )

    if not merged_rows:
        raise ValueError(
            "No valid experiment results were loaded. "
            "Check sweep directories, prescriptions, and row-oriented results.csv files."
        )

    components = discover_error_components(result_columns)
    summary_rows = build_sweep_summary_rows(merged_rows, components)
    stats = AggregationStats(discovered=discovered, loaded=loaded, skipped=skipped)
    return merged_rows, summary_rows, components, result_columns, stats


def _run_sort_key(row: dict[str, Any]) -> tuple[float, str, str]:
    scale = _coerce_float(row.get("pixel_offsets_knowledge_error_scale"))
    scale_sort = scale if scale is not None else float("inf")
    sweep_label = _string_or_none(row.get("sweep_label")) or ""
    run_id = _string_or_none(row.get("run_id")) or ""
    return scale_sort, sweep_label, run_id


def _run_fieldnames(result_columns: list[str]) -> list[str]:
    metadata_columns = [
        "experiment_dir",
        "sweep_label",
        "prescription_path",
        "inference_system_present",
        "pixel_offsets_knowledge_error_scale",
        "pixel_offsets_knowledge_error_model",
        "pixel_offsets_dx_path",
        "pixel_offsets_dy_path",
        "pixel_response_knowledge_error_scale",
        "pixel_response_knowledge_error_model",
        "pixel_response_prf_path",
    ]
    preferred_run_columns = [
        "run_id",
        "status",
        "seed",
        "loss_init",
        "loss_final",
        "chi2_final",
        "num_steps_completed",
        "run_note",
        "plan_label",
    ]
    derived_columns = ["sep_error_as", "abs_sep_error_as"]

    preferred_set = set(preferred_run_columns)
    remaining_result_columns = [
        column for column in result_columns if column not in preferred_set
    ]
    return metadata_columns + preferred_run_columns + remaining_result_columns + derived_columns


def _summary_fieldnames(components: list[str]) -> list[str]:
    base = [
        "pixel_offsets_knowledge_error_scale",
        "pixel_offsets_knowledge_error_model",
        "pixel_response_knowledge_error_scale",
        "pixel_response_knowledge_error_model",
        "sweep_labels",
        "experiment_dirs",
        "n_total",
        "n_success",
    ]
    metric_columns: list[str] = []
    for component in components:
        for suffix in ERROR_METRIC_SUFFIXES:
            metric_columns.append(f"{component}_{suffix}")
    return base + metric_columns


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--root does not exist or is not a directory: {root}")

    try:
        run_rows, summary_rows, components, result_columns, stats = aggregate_detector_ke_sweep(
            root=root,
            pattern=args.pattern,
            strict=args.strict,
            verbose=args.verbose,
        )
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    run_rows_sorted = sorted(run_rows, key=_run_sort_key)
    summary_rows_sorted = sorted(summary_rows, key=_group_sort_key)

    out_runs = _resolve_output_path(root, args.out_runs)
    out_summary = _resolve_output_path(root, args.out_summary)

    _write_csv(out_runs, run_rows_sorted, _run_fieldnames(result_columns))
    _write_csv(out_summary, summary_rows_sorted, _summary_fieldnames(components))

    print(
        "Sweep aggregation complete: "
        f"discovered={stats.discovered}, loaded={stats.loaded}, skipped={stats.skipped}"
    )
    print(f"Wrote run table: {out_runs}")
    print(f"Wrote grouped summary: {out_summary}")


if __name__ == "__main__":
    main()
