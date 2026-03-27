"""Aggregate outer prescribed-MC sweeps across many experiment directories.

This script combines multiple experiment directories under one sweep root into:

- `sweep_runs.csv`: one row per run across all loaded experiments
- `sweep_summary.csv`: one grouped summary row per sweep point

Inputs
------
Each experiment directory is expected to contain:

- `prescription.yaml` (or `.yml` / `.json`)
- row-oriented `results.csv` from `examples/recipes/prescribed_monte_carlo.py`

When a root-level `sweep_manifest.json` is present, it is used to guide
experiment discovery and to define the outer sweep axis. This supports both:

- detector KE sweeps generated with `--mode detector_ke`
- scalar-field sweeps generated with `--mode scalar_field`

When no root manifest is present, the script falls back to detector-KE metadata
found in experiment prescriptions and remains compatible with older detector KE
sweep layouts.

Outputs
-------
The run table preserves the row-oriented results columns and enriches them with:

- generic sweep columns (`sweep_mode`, `sweep_target`, `sweep_value`, ...)
- configured detector metadata from prescriptions when present
- realized detector metadata from `runs/*/meta.json` when present
- derived separation error columns (`sep_error_as`, `abs_sep_error_as`)

The summary table groups rows by the resolved outer sweep axis and reports:

- `n_total`
- `n_success`
- per-component bias/error metrics for each infer quantity discovered in
  `results.csv`
"""
from __future__ import annotations

import argparse
import csv
import json
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
MODE_DETECTOR_KE = "detector_ke"
MODE_SCALAR_FIELD = "scalar_field"


class AggregationStats:
    """Simple container for experiment discovery/load counts."""

    def __init__(self, discovered: int, loaded: int, skipped: int) -> None:
        self.discovered = int(discovered)
        self.loaded = int(loaded)
        self.skipped = int(skipped)


class SweepSpec:
    """Lightweight sweep-root metadata used to generalize grouping/discovery."""

    def __init__(
        self,
        *,
        manifest_path: Path,
        mode: str | None = None,
        sweep_name: str | None = None,
        layer: str | None = None,
        field_path: str | None = None,
        experiments_by_dir: dict[str, dict[str, Any]] | None = None,
        experiment_order: list[str] | None = None,
    ) -> None:
        self.manifest_path = manifest_path
        self.mode = mode
        self.sweep_name = sweep_name
        self.layer = layer
        self.field_path = field_path
        self.experiments_by_dir = (
            dict(experiments_by_dir) if experiments_by_dir is not None else {}
        )
        self.experiment_order = list(experiment_order) if experiment_order is not None else []

    @property
    def manifest_present(self) -> bool:
        return True


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for outer prescribed-MC sweep aggregation."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate many prescribed-MC experiment directories into "
            "sweep_runs.csv and sweep_summary.csv."
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
        default="*",
        help="Glob pattern for experiment directories under --root (default: *).",
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


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _get_nested(mapping: Any, dotted_key: str) -> Any:
    current = mapping
    for key in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_sweep_value_label(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return format(value, ".12g")
    text = str(value).strip()
    return text if text else None


def _default_sweep_metadata(*, sweep_name: str | None = None) -> dict[str, Any]:
    return {
        "sweep_manifest_present": False,
        "sweep_mode": None,
        "sweep_name": sweep_name,
        "sweep_target": None,
        "sweep_layer": None,
        "sweep_field_path": None,
        "sweep_value": None,
        "sweep_value_label": None,
        "sweep_value_numeric": None,
    }


def _layer_metadata_prefix(layer_name: str) -> str | None:
    mapping = {
        "pixel_offsets": "pixel_offsets",
        "pixel_response": "pixel_response",
    }
    return mapping.get(layer_name)


def _configured_detector_ke_value(
    metadata: dict[str, Any],
    *,
    layer_name: str,
) -> Any:
    prefix = _layer_metadata_prefix(layer_name)
    if prefix is None:
        return None
    return metadata.get(f"{prefix}_configured_scale")


def _detector_layer_has_configured_ke(metadata: dict[str, Any], *, layer_name: str) -> bool:
    prefix = _layer_metadata_prefix(layer_name)
    if prefix is None:
        return False

    scale = _coerce_float(metadata.get(f"{prefix}_configured_scale"))
    model = _string_or_none(metadata.get(f"{prefix}_configured_model"))
    policy = _string_or_none(metadata.get(f"{prefix}_configured_realization_policy"))
    if layer_name == "pixel_offsets":
        path_present = any(
            _string_or_none(metadata.get(key)) is not None
            for key in ("pixel_offsets_dx_path", "pixel_offsets_dy_path")
        )
    else:
        path_present = _string_or_none(metadata.get("pixel_response_prf_path")) is not None

    return (
        model is not None
        or policy is not None
        or path_present
        or (scale is not None and scale != 0.0)
    )


def _infer_detector_sweep_layer(metadata: dict[str, Any]) -> str | None:
    offsets_present = _detector_layer_has_configured_ke(metadata, layer_name="pixel_offsets")
    response_present = _detector_layer_has_configured_ke(metadata, layer_name="pixel_response")
    if offsets_present and not response_present:
        return "pixel_offsets"
    if response_present and not offsets_present:
        return "pixel_response"
    if offsets_present:
        return "pixel_offsets"
    if response_present:
        return "pixel_response"
    return None


def _detector_sweep_target(layer_name: str | None) -> str | None:
    if layer_name is None:
        return None
    return f"detector.layers.{layer_name}.knowledge_error.scale"


def load_sweep_spec(
    root: Path,
    *,
    strict: bool,
    verbose: bool,
) -> SweepSpec | None:
    """Load optional root-level sweep metadata used to guide aggregation."""
    manifest_path = root / "sweep_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        payload = _read_json_dict(manifest_path)
    except Exception as exc:
        _fail_or_warn(
            f"{manifest_path}: failed to load sweep manifest ({exc}); falling back to directory scan.",
            strict=strict,
        )
        return None

    mode = _string_or_none(payload.get("mode"))
    sweep_name = _string_or_none(payload.get("sweep_name")) or root.name
    layer = _string_or_none(payload.get("layer"))
    field_path = _string_or_none(payload.get("field_path"))

    experiments_by_dir: dict[str, dict[str, Any]] = {}
    experiment_order: list[str] = []
    experiments_payload = payload.get("experiments")
    if isinstance(experiments_payload, list):
        for entry in experiments_payload:
            if not isinstance(entry, dict):
                continue
            directory = _string_or_none(entry.get("directory"))
            if directory is None or directory in experiments_by_dir:
                continue
            experiments_by_dir[directory] = dict(entry)
            experiment_order.append(directory)

    spec = SweepSpec(
        manifest_path=manifest_path,
        mode=mode,
        sweep_name=sweep_name,
        layer=layer,
        field_path=field_path,
        experiments_by_dir=experiments_by_dir,
        experiment_order=experiment_order,
    )
    _log(
        f"[sweep] loaded manifest {manifest_path.name} mode={spec.mode or 'unknown'}",
        verbose=verbose,
    )
    return spec


def _discover_experiment_dirs(
    *,
    root: Path,
    pattern: str,
    sweep_spec: SweepSpec | None,
    strict: bool,
    verbose: bool,
) -> list[Path]:
    discovered_dirs: list[Path] = []
    seen: set[Path] = set()

    if sweep_spec is not None:
        for directory in sweep_spec.experiment_order:
            candidate = root / directory
            if candidate.exists() and candidate.is_dir():
                discovered_dirs.append(candidate)
                seen.add(candidate)
                continue
            _fail_or_warn(
                f"{root}: sweep manifest references missing experiment directory '{directory}'.",
                strict=strict,
            )

    for candidate in sorted(path for path in root.glob(pattern) if path.is_dir()):
        if candidate in seen:
            continue
        discovered_dirs.append(candidate)
        seen.add(candidate)

    if sweep_spec is not None and discovered_dirs:
        _log(
            f"[sweep] discovered {len(discovered_dirs)} experiment directories",
            verbose=verbose,
        )
    return discovered_dirs


def extract_sweep_axis_metadata(
    prescription: dict[str, Any],
    *,
    experiment_dir_name: str,
    sweep_spec: SweepSpec | None,
    detector_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Extract generic sweep-axis metadata for grouping and sorting."""
    out = _default_sweep_metadata(
        sweep_name=sweep_spec.sweep_name if sweep_spec is not None else None
    )
    record = None
    if sweep_spec is not None:
        out["sweep_manifest_present"] = True
        out["sweep_mode"] = sweep_spec.mode
        out["sweep_name"] = sweep_spec.sweep_name
        record = sweep_spec.experiments_by_dir.get(experiment_dir_name)

    raw_value: Any = None
    value_label: str | None = None

    if sweep_spec is not None and sweep_spec.mode == MODE_SCALAR_FIELD:
        out["sweep_field_path"] = sweep_spec.field_path
        out["sweep_target"] = sweep_spec.field_path
        if record is not None:
            raw_value = record.get("value")
            value_label = _string_or_none(record.get("value_label"))
        if raw_value is None and sweep_spec.field_path is not None:
            raw_value = _get_nested(prescription, sweep_spec.field_path)
        if value_label is None:
            value_label = _format_sweep_value_label(raw_value)
    else:
        layer_name = sweep_spec.layer if sweep_spec is not None else None
        if layer_name is None:
            layer_name = _infer_detector_sweep_layer(detector_metadata)
        if layer_name is not None:
            out["sweep_mode"] = out["sweep_mode"] or MODE_DETECTOR_KE
            out["sweep_layer"] = layer_name
            out["sweep_target"] = _detector_sweep_target(layer_name)
            if record is not None:
                raw_value = record.get("scale")
                value_label = _string_or_none(record.get("scale_label"))
            if raw_value is None:
                raw_value = _configured_detector_ke_value(
                    detector_metadata,
                    layer_name=layer_name,
                )
            if value_label is None:
                value_label = _format_sweep_value_label(raw_value)

    if raw_value is None and record is not None:
        raw_value = record.get("value", record.get("scale"))
    if value_label is None and record is not None:
        value_label = _string_or_none(record.get("value_label"))
        if value_label is None:
            value_label = _string_or_none(record.get("scale_label"))

    if value_label is None:
        value_label = experiment_dir_name

    out["sweep_value"] = raw_value
    out["sweep_value_label"] = value_label
    out["sweep_value_numeric"] = _coerce_float(raw_value)
    return out


def extract_detector_knowledge_error_metadata(prescription: dict[str, Any]) -> dict[str, Any]:
    """Extract inference-detector knowledge-error metadata from a prescription."""
    out: dict[str, Any] = {
        "inference_system_present": False,
        "pixel_offsets_configured_scale": 0.0,
        "pixel_offsets_configured_model": None,
        "pixel_offsets_configured_realization_policy": None,
        "pixel_offsets_dx_path": None,
        "pixel_offsets_dy_path": None,
        "pixel_response_configured_scale": 0.0,
        "pixel_response_configured_model": None,
        "pixel_response_configured_realization_policy": None,
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
            out["pixel_offsets_configured_model"] = _string_or_none(ke_cfg.get("model"))
            out["pixel_offsets_configured_realization_policy"] = _string_or_none(
                ke_cfg.get("realization_policy")
            )
            scale = _coerce_float(ke_cfg.get("scale"))
            if scale is not None:
                out["pixel_offsets_configured_scale"] = scale

    response_layer = _first_layer_by_name(layers, "pixel_response")
    if response_layer is not None:
        out["pixel_response_prf_path"] = _string_or_none(response_layer.get("prf_path"))
        ke_cfg = response_layer.get("knowledge_error")
        if isinstance(ke_cfg, dict):
            out["pixel_response_configured_model"] = _string_or_none(ke_cfg.get("model"))
            out["pixel_response_configured_realization_policy"] = _string_or_none(
                ke_cfg.get("realization_policy")
            )
            scale = _coerce_float(ke_cfg.get("scale"))
            if scale is not None:
                out["pixel_response_configured_scale"] = scale

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


def _discover_run_artifact_roots(experiment_dir: Path) -> list[Path]:
    roots: list[Path] = []
    preferred = experiment_dir / "runs"
    if preferred.exists() and preferred.is_dir():
        roots.append(preferred)
    for candidate in sorted(experiment_dir.glob("runs*")):
        if not candidate.is_dir():
            continue
        if candidate == preferred:
            continue
        roots.append(candidate)
    return roots


def _read_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected top-level object in {path}")
    return payload


def _load_run_meta_index(
    experiment_dir: Path,
    *,
    strict: bool,
    verbose: bool,
) -> dict[str, dict[str, Any]]:
    """Index run `meta.json` payloads by run_id for one experiment directory."""
    index: dict[str, dict[str, Any]] = {}
    run_roots = _discover_run_artifact_roots(experiment_dir)
    if not run_roots:
        _fail_or_warn(
            f"{experiment_dir}: no run artifact directory found (expected runs/ or runs*).",
            strict=strict,
        )
        return index

    for run_root in run_roots:
        run_dirs = sorted(path for path in run_root.iterdir() if path.is_dir())
        for run_dir in run_dirs:
            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                _fail_or_warn(
                    f"{run_dir}: missing meta.json.",
                    strict=strict,
                )
                continue
            try:
                meta = _read_json_dict(meta_path)
            except Exception as exc:
                _fail_or_warn(
                    f"{meta_path}: failed to read run metadata ({exc}).",
                    strict=strict,
                )
                continue

            run_id = _string_or_none(meta.get("run_id")) or run_dir.name
            if run_id in index:
                _warn(
                    f"{experiment_dir}: duplicate run_id '{run_id}' meta; "
                    f"keeping first from {index[run_id]['meta_path']}, ignoring {meta_path}."
                )
                continue
            index[run_id] = {
                "meta": meta,
                "meta_path": meta_path,
                "runs_root": run_root,
            }

    _log(
        f"[meta] {experiment_dir.name}: loaded {len(index)} run meta records",
        verbose=verbose,
    )
    return index


def _extract_layer_meta(layers_payload: Any, layer_name: str) -> dict[str, Any] | None:
    if isinstance(layers_payload, dict):
        direct = layers_payload.get(layer_name)
        if isinstance(direct, dict):
            return direct
        for layer in layers_payload.values():
            if isinstance(layer, dict) and _string_or_none(layer.get("name")) == layer_name:
                return layer
    if isinstance(layers_payload, list):
        for layer in layers_payload:
            if isinstance(layer, dict) and _string_or_none(layer.get("name")) == layer_name:
                return layer
    return None


def _realized_detector_ke_defaults(*, run_meta_present: bool) -> dict[str, Any]:
    return {
        "run_meta_present": run_meta_present,
        "detector_ke_realization_mode": None,
        "inference_cfg_hash": None,
        "inference_forward_spec_hash": None,
        "pixel_offsets_realized_model": None,
        "pixel_offsets_realized_scale": None,
        "pixel_offsets_realized_realization_policy": None,
        "pixel_offsets_realized_seed": None,
        "pixel_offsets_realized_seed_source": None,
        "pixel_response_realized_model": None,
        "pixel_response_realized_scale": None,
        "pixel_response_realized_realization_policy": None,
        "pixel_response_realized_seed": None,
        "pixel_response_realized_seed_source": None,
    }


def extract_realized_detector_knowledge_error_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """Extract inference-side realized detector KE metadata from run meta payload."""
    out = _realized_detector_ke_defaults(run_meta_present=True)
    out["detector_ke_realization_mode"] = _string_or_none(
        _get_nested(meta, "prescribed.detector_ke_realization_mode")
    )
    out["inference_cfg_hash"] = _string_or_none(
        _get_nested(meta, "prescribed.inference_cfg_hash")
    )
    out["inference_forward_spec_hash"] = _string_or_none(
        _get_nested(meta, "prescribed.inference_forward_spec_hash")
    )

    detector_ke = meta.get("detector_knowledge_error")
    if not isinstance(detector_ke, dict):
        return out
    inference_ke = detector_ke.get("inference")
    if not isinstance(inference_ke, dict):
        return out
    layers_payload = inference_ke.get("layers")

    offsets_layer = _extract_layer_meta(layers_payload, "pixel_offsets")
    if isinstance(offsets_layer, dict):
        out["pixel_offsets_realized_model"] = _string_or_none(offsets_layer.get("model"))
        out["pixel_offsets_realized_scale"] = _coerce_float(offsets_layer.get("scale"))
        out["pixel_offsets_realized_realization_policy"] = _string_or_none(
            offsets_layer.get("realization_policy")
        )
        out["pixel_offsets_realized_seed"] = _coerce_int(offsets_layer.get("seed"))
        out["pixel_offsets_realized_seed_source"] = _string_or_none(
            offsets_layer.get("seed_source")
        )

    response_layer = _extract_layer_meta(layers_payload, "pixel_response")
    if isinstance(response_layer, dict):
        out["pixel_response_realized_model"] = _string_or_none(response_layer.get("model"))
        out["pixel_response_realized_scale"] = _coerce_float(response_layer.get("scale"))
        out["pixel_response_realized_realization_policy"] = _string_or_none(
            response_layer.get("realization_policy")
        )
        out["pixel_response_realized_seed"] = _coerce_int(response_layer.get("seed"))
        out["pixel_response_realized_seed_source"] = _string_or_none(
            response_layer.get("seed_source")
        )

    return out


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


def _row_sweep_value_numeric(row: dict[str, Any]) -> float | None:
    value = _coerce_float(row.get("sweep_value_numeric"))
    if value is not None:
        return value
    return _coerce_float(row.get("pixel_offsets_configured_scale"))


def _row_sweep_value_label(row: dict[str, Any]) -> str:
    label = _string_or_none(row.get("sweep_value_label"))
    if label is not None:
        return label
    fallback = _format_sweep_value_label(row.get("sweep_value"))
    if fallback is not None:
        return fallback
    return _string_or_none(row.get("sweep_label")) or ""


def _row_sweep_target(row: dict[str, Any]) -> str:
    target = _string_or_none(row.get("sweep_target"))
    if target is not None:
        return target

    response_present = _detector_layer_has_configured_ke(row, layer_name="pixel_response")
    offsets_present = _detector_layer_has_configured_ke(row, layer_name="pixel_offsets")
    if response_present and not offsets_present:
        return _detector_sweep_target("pixel_response") or ""
    if offsets_present:
        return _detector_sweep_target("pixel_offsets") or ""
    return ""


def _sort_key_for_sweep_value(numeric: float | None, label: str) -> tuple[int, float, str]:
    if numeric is not None:
        return 0, numeric, label
    return 1, float("inf"), label


def _group_sort_key(group_row: dict[str, Any]) -> tuple[str, int, float, str]:
    target = _row_sweep_target(group_row)
    label = _row_sweep_value_label(group_row)
    numeric = _row_sweep_value_numeric(group_row)
    kind, numeric_sort, label_sort = _sort_key_for_sweep_value(numeric, label)
    return target, kind, numeric_sort, label_sort


def build_sweep_summary_rows(
    run_rows: list[dict[str, Any]],
    components: list[str],
) -> list[dict[str, Any]]:
    """Build grouped summary rows keyed by the inferred/declared sweep axis."""
    grouped: dict[tuple[str, float | None, str], list[dict[str, Any]]] = {}
    for row in run_rows:
        target = _row_sweep_target(row)
        numeric_value = _row_sweep_value_numeric(row)
        label = _row_sweep_value_label(row)
        grouped.setdefault((target, numeric_value, label), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (target, numeric_value, label), rows in grouped.items():
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
        realization_modes = sorted(
            {
                str(value)
                for value in (row.get("detector_ke_realization_mode") for row in rows)
                if value not in (None, "")
            }
        )
        first_row = rows[0]
        base_row: dict[str, Any] = {
            "sweep_manifest_present": first_row.get("sweep_manifest_present"),
            "sweep_mode": first_row.get("sweep_mode"),
            "sweep_name": first_row.get("sweep_name"),
            "sweep_target": target,
            "sweep_layer": first_row.get("sweep_layer"),
            "sweep_field_path": first_row.get("sweep_field_path"),
            "sweep_value": first_row.get("sweep_value"),
            "sweep_value_label": label,
            "sweep_value_numeric": numeric_value,
            "pixel_offsets_configured_scale": _coerce_float(
                first_row.get("pixel_offsets_configured_scale")
            ),
            "pixel_offsets_configured_model": first_row.get(
                "pixel_offsets_configured_model"
            ),
            "pixel_offsets_configured_realization_policy": first_row.get(
                "pixel_offsets_configured_realization_policy"
            ),
            "pixel_response_configured_scale": _coerce_float(
                first_row.get("pixel_response_configured_scale")
            ),
            "pixel_response_configured_model": first_row.get(
                "pixel_response_configured_model"
            ),
            "pixel_response_configured_realization_policy": first_row.get(
                "pixel_response_configured_realization_policy"
            ),
            "sweep_labels": ";".join(sweep_labels),
            "experiment_dirs": ";".join(experiment_dirs),
            "detector_ke_realization_modes": ";".join(realization_modes),
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
    pattern: str = "*",
    strict: bool = False,
    verbose: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    list[str],
    AggregationStats,
]:
    """Aggregate run rows and grouped summary rows for an outer sweep root."""
    sweep_spec = load_sweep_spec(root, strict=strict, verbose=verbose)
    discovered_dirs = _discover_experiment_dirs(
        root=root,
        pattern=pattern,
        sweep_spec=sweep_spec,
        strict=strict,
        verbose=verbose,
    )
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
            sweep_metadata = extract_sweep_axis_metadata(
                prescription,
                experiment_dir_name=experiment_dir.name,
                sweep_spec=sweep_spec,
                detector_metadata=metadata,
            )
            run_rows, fieldnames = _read_row_results_csv(results_path)
        except Exception as exc:
            skipped += 1
            _fail_or_warn(
                f"{experiment_dir}: failed to load inputs ({exc}); skipping experiment.",
                strict=strict,
            )
            continue

        run_meta_index = _load_run_meta_index(
            experiment_dir,
            strict=strict,
            verbose=verbose,
        )

        result_columns = _merge_column_order(result_columns, fieldnames)
        rel_dir = experiment_dir.relative_to(root).as_posix()
        sweep_label = experiment_dir.name
        missing_meta_count = 0

        for row in run_rows:
            merged_row: dict[str, Any] = dict(row)
            merged_row.update(metadata)
            merged_row.update(sweep_metadata)
            merged_row["experiment_dir"] = rel_dir
            merged_row["sweep_label"] = sweep_label
            merged_row["prescription_path"] = prescription_path.name

            run_id = _string_or_none(merged_row.get("run_id"))
            meta_record = run_meta_index.get(run_id) if run_id is not None else None
            if meta_record is not None:
                meta_payload = meta_record.get("meta")
                if isinstance(meta_payload, dict):
                    merged_row.update(
                        extract_realized_detector_knowledge_error_metadata(meta_payload)
                    )
                else:
                    merged_row.update(_realized_detector_ke_defaults(run_meta_present=False))
                try:
                    merged_row["run_meta_path"] = (
                        Path(meta_record["meta_path"]).relative_to(experiment_dir).as_posix()
                    )
                except Exception:
                    merged_row["run_meta_path"] = str(meta_record.get("meta_path"))
            else:
                merged_row.update(_realized_detector_ke_defaults(run_meta_present=False))
                merged_row["run_meta_path"] = None
                missing_meta_count += 1
                if strict and run_id is not None:
                    raise ValueError(
                        f"{experiment_dir}: missing run metadata for run_id '{run_id}'."
                    )

            sep_error = compute_component_error(merged_row, "source.separation_as")
            merged_row["sep_error_as"] = sep_error
            merged_row["abs_sep_error_as"] = abs(sep_error) if sep_error is not None else None
            merged_rows.append(merged_row)

        if missing_meta_count > 0:
            _fail_or_warn(
                f"{experiment_dir}: missing run meta.json for {missing_meta_count} "
                f"of {len(run_rows)} result rows.",
                strict=strict,
            )

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


def _run_sort_key(row: dict[str, Any]) -> tuple[str, int, float, str, str]:
    numeric = _row_sweep_value_numeric(row)
    label = _row_sweep_value_label(row)
    target = _row_sweep_target(row)
    kind, scale_sort, sweep_label = _sort_key_for_sweep_value(numeric, label)
    run_id = _string_or_none(row.get("run_id")) or ""
    return target, kind, scale_sort, sweep_label, run_id


def _run_fieldnames(result_columns: list[str]) -> list[str]:
    metadata_columns = [
        "sweep_manifest_present",
        "sweep_mode",
        "sweep_name",
        "sweep_target",
        "sweep_layer",
        "sweep_field_path",
        "sweep_value",
        "sweep_value_label",
        "sweep_value_numeric",
        "experiment_dir",
        "sweep_label",
        "prescription_path",
        "inference_system_present",
        "pixel_offsets_configured_scale",
        "pixel_offsets_configured_model",
        "pixel_offsets_configured_realization_policy",
        "pixel_offsets_dx_path",
        "pixel_offsets_dy_path",
        "pixel_response_configured_scale",
        "pixel_response_configured_model",
        "pixel_response_configured_realization_policy",
        "pixel_response_prf_path",
        "run_meta_present",
        "run_meta_path",
        "detector_ke_realization_mode",
        "inference_cfg_hash",
        "inference_forward_spec_hash",
        "pixel_offsets_realized_model",
        "pixel_offsets_realized_scale",
        "pixel_offsets_realized_realization_policy",
        "pixel_offsets_realized_seed",
        "pixel_offsets_realized_seed_source",
        "pixel_response_realized_model",
        "pixel_response_realized_scale",
        "pixel_response_realized_realization_policy",
        "pixel_response_realized_seed",
        "pixel_response_realized_seed_source",
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
        "sweep_manifest_present",
        "sweep_mode",
        "sweep_name",
        "sweep_target",
        "sweep_layer",
        "sweep_field_path",
        "sweep_value",
        "sweep_value_label",
        "sweep_value_numeric",
        "pixel_offsets_configured_scale",
        "pixel_offsets_configured_model",
        "pixel_offsets_configured_realization_policy",
        "pixel_response_configured_scale",
        "pixel_response_configured_model",
        "pixel_response_configured_realization_policy",
        "sweep_labels",
        "experiment_dirs",
        "detector_ke_realization_modes",
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
