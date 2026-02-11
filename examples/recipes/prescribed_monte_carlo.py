"""Prescribed Monte Carlo experiment.

Purpose: provide a robust way to define and run monte carlo experiments
Local helpers are defined here for now and document whether they are reusable.

Execution flow
--------------
- Load the prescription JSON and optional per-run overrides CSV.
- Resolve run specs and seeds, plus experiment-wide config/store overrides.
- Build truth/init stores for each run, then generate synthetic observations with
  optional noise.
- Run optimization in eigen space (FIM-based) or primitive parameter space.
- Write run artifacts under runs/<run_id>/..., including summaries and logs.
- Aggregate manifest.json and results.csv across runs at the experiment root.

Notes behavior
--------------
- Experiment-level note: set `experiment.notes` in the prescription (aliases:
  `experiment.note`, `experiment.comment`, `experiment.comments`). This value
  is written once to top-level `manifest.json["notes"]`.
- Per-run note: set `note`/`notes`/`comment`/`comments` in overrides rows.
  The resolved value is stored as `run_note` in run summaries and aggregate
  outputs.

CLI arguments (mirrors `main`)
------------------------------
- --prescription: JSON experiment recipe (defaults to template when omitted).
- --overrides: optional CSV of per-run overrides (defaults to template when omitted).
- --outdir: root output directory for the experiment (experiment root). If
  omitted, the output directory is derived from `--run-name` or a timestamp.
- --run-name: optional name segment used to build Results/<run-name> when
  `--outdir` is not provided. If both are omitted, a timestamp tag is used.
  This remains as a convenience for quick naming when you do not want to type
  a full `--outdir` path, rather than being deprecated.
- --dry-run: resolve and preview run specs without executing optimization.
- --aggregate-only: skip execution and only build manifest.json/results.csv from
  existing run artifacts inside `--outdir`.
- --num-preview: limit how many resolved run specs are printed during preview.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import dataclasses
import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dluxshera.inference.optimization import (
    EigenThetaMap,
    fim_theta,
    generate_fim_labels,
    make_binder_nll_fn,
    map_labels_to_keys,
    run_shera_gd,
)
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.run_artifacts import (
    _now_iso_local_ms,
    build_param_summary,
    patch_summary,
)
from dluxshera.params.packing import (
    build_eigen_index_map,
    build_index_map,
    pack_params,
    unpack_params as store_unpack_params,
)
from dluxshera.params.spec import build_inference_spec_basic, make_inference_subspec
from dluxshera.params.store import ParameterStore, strip_structural
from dluxshera.systems.three_plane import (
    SHERA_FLIGHT_CONFIG,
    SHERA_TESTBED_CONFIG,
    SheraThreePlaneConfig,
    SheraThreePlaneBinder,
    build_forward_spec_from_config,
)

DEFAULT_PRESCRIPTION_PATH = Path(
    "examples/recipes/prescription_templates/prescription.json"
)
DEFAULT_OVERRIDES_PATH = Path("examples/recipes/prescription_templates/overrides.csv")
PLAN_FREE_TEXT_COLUMNS = frozenset({"note", "notes", "comment", "comments"})
EXPERIMENT_NOTE_KEYS = ("notes", "note", "comment", "comments")

def _timestamp_tag() -> str:
    """Return a sortable timestamp string for labeling output directories.

    Used by `_resolve_outdir` when no explicit output directory/run name is given,
    providing consistent time-based naming in the main execution flow.
    This helper is broadly reusable as a generic timestamp label utility.
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_prescription(path: Path) -> dict[str, Any]:
    """Load a prescription JSON file from disk.

    Called early in `main` to materialize the experiment recipe that drives run
    spec resolution and defaults. Keys that start with `_` are treated as
    private annotations/disabled fields and are recursively stripped from the
    loaded object before any downstream parsing/validation runs. This is
    generally reusable for JSON config loading, but kept local because it
    assumes the specific prescription schema expected by this script.
    """
    with path.open("r", encoding="utf-8") as handle:
        return _strip_private_keys(json.load(handle))


def _strip_private_keys(obj: Any) -> Any:
    """Recursively remove keys prefixed with `_` from nested JSON-like objects.

    This keeps template comments/disabled example fields out of runtime parsing
    so strict validators only see active prescription content.
    """
    if isinstance(obj, dict):
        return {
            key: _strip_private_keys(value)
            for key, value in obj.items()
            if not key.startswith("_")
        }
    if isinstance(obj, list):
        return [_strip_private_keys(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_strip_private_keys(item) for item in obj)
    return obj


def _parse_cell(value: str | None) -> Any:
    """Parse a CSV cell value into a typed Python value.

    Invoked by `_load_plan_csv` when interpreting plan rows, converting strings
    to `None`, booleans, numbers, or JSON lists as needed. The parser is reusable
    for plan-like CSV parsing but is tailored to this script's CSV conventions.
    """
    if value is None:
        return None
    raw = value.strip()
    # CSV cells that include embedded JSON are sometimes exported as a quoted
    # JSON string literal (e.g. '"[1, 2, 3]"'). Peel wrapping quotes so vector
    # values continue through the normal JSON parsing path below.
    while len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"\"", "'"}:
        raw = raw[1:-1].strip()
    if raw == "" or raw.lower() in {"null", "none"}:
        return None
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(parsed, list):
            return [float(item) if isinstance(item, (int, float)) else item for item in parsed]
        return parsed
    if raw.lstrip("-").isdigit():
        try:
            return int(raw)
        except ValueError:
            pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _parse_plan_value(key: str, value: str | None) -> Any:
    """Parse a plan CSV value with key-aware handling for free-text fields."""
    normalized_key = key.strip().lower()
    if normalized_key in PLAN_FREE_TEXT_COLUMNS:
        if value is None:
            return None
        raw = value.strip()
        if raw == "" or raw.lower() in {"null", "none"}:
            return None
        return raw
    return _parse_cell(value)


def _load_plan_csv(path: Path) -> list[dict[str, Any]]:
    """Load plan rows from CSV in wide or transposed formats.

    Called by `main` to read the optional plan file that overrides per-run
    settings. It supports both runs-as-rows and keys-as-rows layouts and relies
    on `_parse_cell` to coerce values. This helper is reusable for experiment
    plan ingestion but is coupled to the specific plan format rules used here.
    """
    with path.open("r", encoding="utf-8") as handle:
        lines = []
        for line in handle:
            stripped = line.lstrip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"#')
                or stripped.startswith("'#")
            ):
                continue
            if not stripped.strip():
                continue
            lines.append(line)

    if not lines:
        return []

    header = next(csv.reader([lines[0]]))
    if header and header[0] == "key":
        reader = csv.reader(lines)
        header = next(reader)
        empty_header_indices = [
            idx
            for idx, cell in enumerate(header)
            if cell is None or not cell.strip()
        ]
        print(f"Plan CSV header length: {len(header)}")
        print(
            "Plan CSV empty header cells: "
            f"{len(empty_header_indices)} at indices {empty_header_indices}"
        )
        run_headers = header[1:]
        last_non_empty = None
        for idx, run_header in enumerate(run_headers):
            if run_header is not None and run_header.strip():
                last_non_empty = idx
        if last_non_empty is not None:
            run_headers = run_headers[: last_non_empty + 1]
        # Avoid Excel-generated trailing columns by skipping empty headers.
        run_columns = [
            run_header
            for run_header in run_headers
            if run_header is not None and run_header.strip()
        ]
        trimmed_run_columns = [
            run_header.strip() if run_header is not None else "" for run_header in run_columns
        ]
        print(f"Plan CSV run columns after trimming: {trimmed_run_columns}")
        rows: list[dict[str, Any]] = []
        for run_col in run_columns:
            label = run_col.strip() if run_col is not None else ""
            rows.append({"_plan_label": label or None})
        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                trailing_cells = row[len(header):] if len(row) > len(header) else []
                print(
                    "Warning: Plan CSV row "
                    f"{row_idx} length {len(row)} differs from header length "
                    f"{len(header)}; trailing cells: {trailing_cells}"
                )
            key = row[0].strip() if len(row) > 0 and row[0] is not None else ""
            if not key:
                continue
            for idx, _ in enumerate(run_columns):
                value = row[idx + 1] if len(row) > idx + 1 else ""
                if (
                    (value is None or value.strip() == "")
                    and key.strip().lower() not in PLAN_FREE_TEXT_COLUMNS
                ):
                    continue
                rows[idx][key] = _parse_plan_value(key, value)
        return rows

    reader = csv.DictReader(lines)
    rows: list[dict[str, Any]] = []
    for row in reader:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if (
                (value is None or value.strip() == "")
                and key.strip().lower() not in PLAN_FREE_TEXT_COLUMNS
            ):
                continue
            parsed[key] = _parse_plan_value(key, value)
        rows.append(parsed)
    return rows


def _apply_experiment_n_runs(
    plan_rows: list[dict[str, Any]],
    n_runs: int | None,
) -> tuple[list[dict[str, Any]], dict[str, int | None]]:
    """Apply experiment.n_runs precedence to the plan-defined run rows.

    When experiment.n_runs is set, it becomes authoritative: the plan is padded
    with default-only rows or truncated as needed so downstream previews and run
    execution operate on the resolved run count. When experiment.n_runs is not
    set, the plan length defines the run count and an empty plan is rejected.
    """
    plan_rows_copy = [row.copy() for row in plan_rows]
    plan_runs = len(plan_rows_copy)

    if n_runs is None:
        if plan_runs == 0:
            raise ValueError(
                "Unable to resolve run count: experiment.n_runs is not set and "
                "overrides.csv defines 0 runs."
            )
        return (
            plan_rows_copy,
            {
                "plan_runs": plan_runs,
                "resolved_runs": plan_runs,
                "padded_runs": 0,
                "truncated_runs": 0,
                "n_runs": None,
            },
        )

    try:
        resolved_n_runs = int(n_runs)
    except (TypeError, ValueError) as exc:
        raise ValueError("experiment.n_runs must be an integer.") from exc

    if resolved_n_runs <= 0:
        raise ValueError("experiment.n_runs must be a positive integer.")

    padded = 0
    truncated = 0
    if plan_runs > resolved_n_runs:
        truncated = plan_runs - resolved_n_runs
        plan_rows_copy = plan_rows_copy[:resolved_n_runs]
    elif plan_runs < resolved_n_runs:
        padded = resolved_n_runs - plan_runs
        plan_rows_copy.extend({"_plan_label": None} for _ in range(padded))

    return (
        plan_rows_copy,
        {
            "plan_runs": plan_runs,
            "resolved_runs": len(plan_rows_copy),
            "padded_runs": padded,
            "truncated_runs": truncated,
            "n_runs": resolved_n_runs,
        },
    )


def _detect_prescription_overrides_candidates(
    outdir: Path,
) -> tuple[Path | None, Path | None]:
    """Scan an output directory for candidate prescription/overrides files.

    This is a best-effort helper for `main` when `--prescription` or `--overrides` is
    omitted but an output directory is provided. Detection rules are intentionally
    conservative; update this helper when additional filename conventions are
    introduced in prescribed Monte Carlo workflows.
    """
    if not outdir.exists():
        return None, None

    prescription_candidates = sorted(
        candidate
        for candidate in outdir.rglob("*.json")
        if "prescription" in candidate.name.lower()
    )
    overrides_candidates = sorted(
        (candidate for candidate in outdir.rglob("*.csv") if "overrides" in candidate.name.lower()),
        key=lambda candidate: (
            candidate.name.lower() != "overrides.csv",
            candidate.name.lower() == "overrides_wide.csv",
            str(candidate),
        ),
    )

    if len(prescription_candidates) > 1:
        joined = "\n".join(f"- {candidate}" for candidate in prescription_candidates)
        raise ValueError(
            "Multiple prescription candidates found in "
            f"{outdir}. Provide --prescription to disambiguate:\n{joined}"
        )
    if len(overrides_candidates) > 1:
        joined = "\n".join(f"- {candidate}" for candidate in overrides_candidates)
        raise ValueError(
            "Multiple overrides candidates found in "
            f"{outdir}. Provide --overrides to disambiguate:\n{joined}"
        )

    prescription_path = prescription_candidates[0] if prescription_candidates else None
    overrides_path = overrides_candidates[0] if overrides_candidates else None

    return prescription_path, overrides_path


def _resolve_prescription_and_overrides(
    args: argparse.Namespace, outdir: Path | None
) -> tuple[Path, Path | None]:
    """Resolve prescription/overrides paths from CLI args and optional outdir scan."""
    prescription_path = args.prescription
    overrides_path = args.overrides
    explicit_prescription = prescription_path is not None

    if (prescription_path is None or overrides_path is None) and outdir is not None:
        detected_prescription, detected_overrides = _detect_prescription_overrides_candidates(
            outdir
        )
        if prescription_path is None and detected_prescription is not None:
            prescription_path = detected_prescription
            explicit_prescription = True
        if overrides_path is None and detected_overrides is not None:
            overrides_path = detected_overrides

    if overrides_path is not None and prescription_path is None:
        overrides_label = overrides_path if overrides_path is not None else "unknown"
        outdir_label = f"{outdir}" if outdir is not None else "no outdir provided"
        raise ValueError(
            "Overrides path was provided or detected "
            f"({overrides_label}, outdir scan: {outdir_label}) but no prescription was "
            "provided or detected. Pass --prescription explicitly."
        )

    if prescription_path is None:
        outdir_label = f"found in {outdir}" if outdir is not None else "outdir not provided"
        print(
            "WARNING: No prescription path provided/detected (no prescription JSON "
            f"{outdir_label}); falling back to template at {DEFAULT_PRESCRIPTION_PATH}"
        )
        prescription_path = DEFAULT_PRESCRIPTION_PATH
    if overrides_path is None and not explicit_prescription:
        outdir_label = f"found in {outdir}" if outdir is not None else "outdir not provided"
        print(
            "WARNING: No overrides path provided/detected (no overrides CSV "
            f"{outdir_label}); falling back to template at {DEFAULT_OVERRIDES_PATH}"
        )
        overrides_path = DEFAULT_OVERRIDES_PATH

    return Path(prescription_path), Path(overrides_path) if overrides_path is not None else None


def _set_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a nested key path within a dictionary.

    Used by `_unflatten_row` to expand dotted keys into nested structures.
    This is a generic dictionary utility and is a good candidate for shared
    helpers if nested config handling is needed elsewhere.
    """
    current = target
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `updates` into `target`, recursing over nested dicts.

    Called by `_resolve_run_spec_with_id` to apply per-run overrides on top of
    prescription defaults. This is a reusable merge utility, but it assumes dict
    values are the only mergeable structures (no list merging).
    """
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _unflatten_row(row: dict[str, Any]) -> dict[str, Any]:
    """Expand dotted keys in a plan row into nested dictionaries.

    Used in `_resolve_run_spec_with_id` to translate plan CSV columns into the
    nested configuration structure expected by the prescription. This is
    reusable for dot-notation configs but is coupled to the plan CSV layout.
    """
    structured: dict[str, Any] = {}
    for key, value in row.items():
        if "." in key:
            _set_nested(structured, key.split("."), value)
        else:
            structured[key] = value
    return structured


def _extract_prior_overrides(
    row: dict[str, Any],
    *,
    prefix: str = "prior",
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Extract per-run prior overrides from a flat plan row.

    This helper runs before `_unflatten_row` so plan keys like
    `prior.<infer_key>.<field>` are parsed without being incorrectly nested.
    """
    row_clean: dict[str, Any] = {}
    overrides: dict[str, dict[str, Any]] = {}
    prefix_token = f"{prefix}."
    for key, value in row.items():
        if not key.startswith(prefix_token):
            row_clean[key] = value
            continue
        tokens = key.split(".")
        if len(tokens) < 3:
            print(f"WARNING: Invalid prior override key '{key}'; skipping.")
            continue
        field = tokens[-1]
        infer_key = ".".join(tokens[1:-1])
        if not infer_key:
            print(f"WARNING: Invalid prior override key '{key}'; missing infer key.")
            continue
        if field == "std":
            field = "sigma"
        if field not in {"sigma", "dist"}:
            print(
                f"WARNING: Unsupported prior override field '{field}' in '{key}'; skipping."
            )
            continue
        if value is None:
            print(
                f"WARNING: prior override '{key}' set to null; ignoring this override."
            )
            continue
        overrides.setdefault(infer_key, {})[field] = value
    return row_clean, overrides


def _normalize_sigma_override(
    value: Any,
    *,
    base_sigma: Any,
    store_value: Any,
    infer_key: str,
) -> Any:
    """Normalize a sigma override, broadcasting scalars for vector parameters."""
    if isinstance(value, list):
        return value
    if isinstance(value, (int, float, np.floating)):
        scalar = float(value)
        length = None
        if isinstance(base_sigma, (list, tuple, np.ndarray)):
            length = len(base_sigma)
        elif store_value is not None:
            try:
                store_array = np.asarray(store_value)
                if store_array.size > 1:
                    length = store_array.size
            except Exception:
                length = None
        if length:
            return [scalar] * length
        return scalar
    print(
        f"WARNING: prior override sigma for '{infer_key}' should be numeric or list; "
        f"got {type(value).__name__}."
    )
    return value


def _apply_prior_overrides(
    base_prior_info: dict[str, Any],
    overrides: dict[str, dict[str, Any]],
    *,
    infer_keys: tuple[str, ...],
    base_store: ParameterStore | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Merge per-run overrides into a deep-copied prior info payload."""
    merged = copy.deepcopy(base_prior_info)
    applied: list[str] = []
    for infer_key, fields in overrides.items():
        if infer_key not in infer_keys:
            print(
                f"WARNING: prior override for unknown infer key '{infer_key}'; skipping."
            )
            continue
        if infer_key not in merged:
            print(
                f"WARNING: prior override for '{infer_key}' ignored because no base prior "
                "is defined."
            )
            continue
        entry = dict(merged.get(infer_key, {}))
        for field, value in fields.items():
            if field == "sigma":
                entry["sigma"] = _normalize_sigma_override(
                    value,
                    base_sigma=entry.get("sigma"),
                    store_value=base_store.get(infer_key) if base_store else None,
                    infer_key=infer_key,
                )
            elif field == "dist":
                entry["dist"] = value
        merged[infer_key] = entry
        applied.append(infer_key)
    return merged, applied


def _resolve_run_spec(presc: dict[str, Any], row: dict[str, Any], index: int) -> dict[str, Any]:
    """Resolve a run spec from a plan row using default run indexing.

    This is a compatibility wrapper used for older call sites; in this script
    `_resolve_run_spec_with_id` is used directly to accommodate disabled rows.
    It is reusable only for the prescription schema defined here.
    """
    return _resolve_run_spec_with_id(presc, row, index=index, run_id_index=index)


def _resolve_run_spec_with_id(
    presc: dict[str, Any],
    row: dict[str, Any],
    *,
    index: int,
    run_id_index: int | None,
) -> dict[str, Any]:
    """Build a fully resolved run specification from a plan row.

    Called in `main` to combine prescription defaults with plan overrides,
    generate run IDs, and ensure required sections/seed are present. This helper
    is tightly coupled to the prescription schema and run ID conventions in
    this script, so it is not intended as a general utility.
    """
    defaults = copy.deepcopy(presc.get("defaults", {}))
    resolved = copy.deepcopy(defaults)

    experiment = presc.get("experiment", {})
    run_id_prefix = experiment.get("run_id_prefix", "run")

    row = dict(row)
    row.pop("_plan_label", None)
    # prior.<infer_key>.<field> overrides must be extracted before unflattening
    # so dotted infer keys are preserved rather than nested.
    row_clean, prior_overrides = _extract_prior_overrides(row)
    structured_row = _unflatten_row(row_clean)
    run_id = structured_row.pop("run_id", None)
    if run_id:
        resolved["run_id"] = run_id
    elif run_id_index is not None:
        resolved["run_id"] = f"{run_id_prefix}_{run_id_index:04d}"

    seed_override = structured_row.pop("seed", None)

    _deep_update(resolved, structured_row)

    resolved["model"] = copy.deepcopy(presc.get("model", {}))

    resolved.setdefault("init", {})
    resolved.setdefault("noise", {})
    resolved.setdefault("eigen", {})
    resolved.setdefault("truth", {})
    resolved.setdefault("optimizer", {})
    resolved.setdefault("model", {})

    base_seed = presc.get("defaults", {}).get("seed")
    if base_seed is None:
        raise ValueError("Prescription defaults must include a base seed.")
    if seed_override is None:
        # Derive a per-run seed from the default base seed so runs remain
        # reproducible without explicit overrides.
        seed_index = run_id_index if run_id_index is not None else index
        base_key = jr.PRNGKey(int(base_seed))
        run_key = jr.fold_in(base_key, int(seed_index))
        resolved_seed = int(np.asarray(run_key)[0])
    else:
        resolved_seed = seed_override
    resolved["seed"] = int(resolved_seed)

    if prior_overrides:
        resolved["prior_overrides"] = prior_overrides

    return resolved


def _get_nested(payload: dict[str, Any], keys: list[str]) -> Any:
    """Safely fetch a nested value from a dictionary by key path.

    Used throughout `main` and helpers like `_print_preview` to access optional
    nested fields without raising `KeyError`. This is a generic utility that
    could be shared across scripts.
    """
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_present_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    """Return the first non-empty string value found for candidate keys."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _print_preview(run_specs: list[dict[str, Any]], limit: int | None = None) -> None:
    """Print a tabular preview of resolved run specs to stdout.

    Invoked in `main` after run resolution to show key fields before execution.
    The logic depends on the specific nested keys in this script, so it is
    tightly coupled and not ideal for a shared utility without customization.
    """
    headers = [
        "run_id",
        "enabled",
        "seed",
        "init.mode",
        "eigen.use_eigen",
        "eigen.whiten_basis",
        "eigen.truncate_k",
        "eigen.truncate_by_eigval",
        "truth.x",
        "truth.y",
        "init.x",
        "init.y",
    ]

    def cell(spec: dict[str, Any], key: str) -> str:
        if key == "run_id":
            value = spec.get("run_id")
        elif key == "enabled":
            value = _row_enabled(spec)
        elif key == "seed":
            value = spec.get("seed")
        elif key == "init.mode":
            value = _get_nested(spec, ["init", "mode"])
        elif key == "eigen.use_eigen":
            value = _get_nested(spec, ["eigen", "use_eigen"])
        elif key == "eigen.whiten_basis":
            value = _get_nested(spec, ["eigen", "whiten_basis"])
        elif key == "eigen.truncate_k":
            value = _get_nested(spec, ["eigen", "truncate_k"])
        elif key == "eigen.truncate_by_eigval":
            value = _get_nested(spec, ["eigen", "truncate_by_eigval"])
        elif key == "truth.x":
            value = _get_nested(spec, ["truth", "binary", "x_position_as"])
        elif key == "truth.y":
            value = _get_nested(spec, ["truth", "binary", "y_position_as"])
        elif key == "init.x":
            value = _get_nested(spec, ["init", "binary", "x_position_as"])
        elif key == "init.y":
            value = _get_nested(spec, ["init", "binary", "y_position_as"])
        else:
            value = None
        return "" if value is None else str(value)

    preview = run_specs if limit is None else run_specs[:limit]
    rows = [[cell(spec, key) for key in headers] for spec in preview]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    divider = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(divider)
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def _resolve_outdir(outdir: str | None, run_name: str | None) -> Path:
    """Resolve the experiment output directory based on CLI inputs.

    Called in `main` to determine where runs and aggregate outputs are written,
    falling back to a timestamped Results directory via `_timestamp_tag` when
    neither an explicit outdir nor run name is supplied. This is a reusable
    pattern for experiment output naming, but naming conventions are specific
    to this script.
    """
    if outdir:
        return Path(outdir)
    if run_name:
        return Path("Results") / run_name
    return Path("Results") / f"prescribed_mc_{_timestamp_tag()}"


def _row_enabled(row: dict[str, Any]) -> bool:
    """Determine whether a plan row is enabled for execution.

    Used in `main` to skip disabled plan rows and in `_collect_run_entries` to
    ignore runs when aggregating. This is reusable for boolean-like CSV fields,
    though the accepted string values are tailored to this script.
    """
    value = row.get("enabled")
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
    return bool(value)


def _flatten_store_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested store overrides into dotted key/value pairs.

    Used in `main` to apply store overrides to `ParameterStore` and to build
    truth/init overrides from run specs. This is reusable for nested dicts, but
    it assumes dotted-key semantics aligned with this script's store API.
    """
    flattened: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, entry in value.items():
                joined = f"{prefix}.{key}" if prefix else key
                _walk(joined, entry)
        else:
            flattened[prefix] = value

    _walk("", payload)
    return flattened


def _partition_overrides_by_kind(
    overrides_flat: dict[str, Any],
    forward_spec: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split flattened overrides into primitive, derived, and unknown keys."""
    primitive_overrides: dict[str, Any] = {}
    derived_overrides: dict[str, Any] = {}
    unknown_overrides: dict[str, Any] = {}

    for key, value in overrides_flat.items():
        if key not in forward_spec:
            unknown_overrides[key] = value
            continue
        kind = forward_spec.get(key).kind
        if kind == "primitive":
            primitive_overrides[key] = value
        elif kind == "derived":
            derived_overrides[key] = value
        else:
            unknown_overrides[key] = value

    return primitive_overrides, derived_overrides, unknown_overrides


def _resolve_config_id(config_id: str | None) -> SheraThreePlaneConfig:
    """Resolve a config ID string into a concrete SheraThreePlaneConfig.

    Called in `main` to translate the prescription's `model.config_id` into an
    actual config object. This is tightly coupled to Shera configs and should
    remain local to this script or a Shera-specific utility module.
    """
    if not config_id:
        raise ValueError("Prescription must include model.config_id.")
    mapping = {
        "SHERA_TESTBED_CONFIG": SHERA_TESTBED_CONFIG,
        "SHERA_FLIGHT_CONFIG": SHERA_FLIGHT_CONFIG,
        "shera_testbed": SHERA_TESTBED_CONFIG,
        "shera_flight": SHERA_FLIGHT_CONFIG,
    }
    if config_id in mapping:
        return mapping[config_id]
    raise ValueError(f"Unknown config_id '{config_id}'.")


def _apply_config_overrides(
    cfg: SheraThreePlaneConfig,
    overrides: dict[str, Any],
) -> SheraThreePlaneConfig:
    """Apply validated overrides to a SheraThreePlaneConfig.

    Used in `main` after resolving the base config to enforce override keys and
    normalize list fields. This is Shera-specific and not generally reusable
    outside this experiment workflow.
    """
    if not overrides:
        return cfg
    field_names = {field.name for field in dataclasses.fields(cfg)}
    unknown = [key for key in overrides if key not in field_names]
    if unknown:
        raise ValueError(
            "Unknown config override(s): " + ", ".join(sorted(unknown))
        )
    normalized = dict(overrides)
    for key in ("primary_noll_indices", "secondary_noll_indices"):
        if key in normalized and isinstance(normalized[key], list):
            normalized[key] = tuple(normalized[key])
    return cfg.replace(**normalized)


def _repo_relative_path(path: str | Path | None, *, repo_root: Path) -> str | None:
    """Return a repo-relative path string for reporting/manifest metadata.

    Used by `_config_payload` and `_write_experiment_outputs` to make paths
    reproducible in manifest outputs. This is reusable for any tooling that
    wants stable paths relative to a repo root.
    """
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, repo_root)).as_posix()


def _config_payload(cfg: SheraThreePlaneConfig, *, repo_root: Path) -> dict[str, Any]:
    """Serialize a SheraThreePlaneConfig and normalize path fields.

    Called in `main` when recording run metadata for the optimizer artifacts.
    It is tightly coupled to the Shera config schema and is not generic.
    """
    payload = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    if isinstance(payload, dict) and "diffractive_pupil_path" in payload:
        payload = {
            **payload,
            "diffractive_pupil_path": _repo_relative_path(
                payload.get("diffractive_pupil_path"), repo_root=repo_root
            ),
        }
    return payload


def _reduced_chi2_between_images(
    data_image: Any,
    model_image: Any,
    *,
    variance_image: Any,
) -> float:
    """Compute reduced chi-squared between data and model PSF images.

    Used in `main` to report image-space fit quality for the seeded initial
    model and optimized final model in each run summary. The computation mirrors
    `plot_psf_comparison` by building a z-score image from residuals and
    variance, then averaging squared z values by image degrees of freedom.

    This helper is a good candidate for migration into a shared utility module
    if additional experiment scripts need the same image-comparison metric.
    """
    data_arr = np.asarray(data_image, dtype=float)
    model_arr = np.asarray(model_image, dtype=float)
    var_arr = np.asarray(variance_image, dtype=float)

    safe_var = np.where(var_arr > 0.0, var_arr, np.nan)
    z_score = (data_arr - model_arr) / np.sqrt(safe_var)
    dof = data_arr.size
    if dof <= 0:
        return float("nan")
    return float(np.nansum(z_score**2) / dof)


def _maybe_warn_missing_artifacts(run_dir: Path) -> None:
    """Warn if required run artifacts are missing from a run directory.

    Used after patching run summaries in `main` to flag incomplete runs. This is
    reusable for artifact validation but depends on this script's artifact list.
    """
    required = ["meta.json", "summary.json", "trace.npz"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        print(
            f"WARNING: run artifacts missing in {run_dir}: {', '.join(missing)}"
        )


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    """Load a JSON file expected to contain an object, or return None if absent.

    Used by `_collect_run_entries` when aggregating run metadata. This helper is
    reusable for safe JSON object loading across scripts.
    """
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _get_run_created_at(summary: dict[str, Any] | None, meta: dict[str, Any] | None) -> str | None:
    """Find the run creation timestamp from summary/meta payloads.

    Used by `_build_results_rows` to populate results.csv creation metadata.
    This logic is tied to the fields emitted by this script's run artifacts.
    """
    if summary:
        return summary.get("created_at") or summary.get("run_created_at")
    if meta:
        return meta.get("created_at")
    return None


def _infer_vector_length(
    value: Any,
    *,
    key: str,
    field: str,
    current_len: int | None,
) -> int | None:
    """Infer or validate vector length for parameter summaries.

    Called by `_collect_param_layout` to ensure consistent vector lengths across
    runs when building results columns. This helper is reusable for validating
    vector/scalar consistency but depends on this script's summary schema.
    """
    if isinstance(value, list):
        length = len(value)
        if current_len is None:
            return length
        if current_len != length:
            raise ValueError(
                f"Vector length mismatch for '{field}.{key}': {current_len} vs {length}"
            )
        return current_len
    if current_len is not None:
        raise ValueError(
            f"Scalar/vector mismatch for '{field}.{key}' (expected length {current_len})"
        )
    return None


def _collect_param_layout(
    infer_keys: tuple[str, ...],
    summaries: list[dict[str, Any] | None],
) -> dict[str, dict[str, Any]]:
    """Analyze parameter summaries to determine column layout.

    Used in `_build_results_rows` to decide which parameters are vectors and
    whether truth values exist. This is coupled to the summary schema produced
    by this script's optimization artifacts.
    """
    layout: dict[str, dict[str, Any]] = {}
    for key in infer_keys:
        vector_len: int | None = None
        has_truth = False
        for summary in summaries:
            if not summary:
                continue
            param_summary = summary.get("param_summary") or {}
            entry = param_summary.get(key)
            if not isinstance(entry, dict):
                continue
            if "truth" in entry:
                has_truth = True
            for field in ("truth", "init", "final", "init_delta", "final_delta"):
                if field in entry:
                    vector_len = _infer_vector_length(
                        entry[field],
                        key=key,
                        field=field,
                        current_len=vector_len,
                    )
        layout[key] = {"vector_len": vector_len, "has_truth": has_truth}
    return layout


def _param_columns(infer_keys: tuple[str, ...], layout: dict[str, dict[str, Any]]) -> list[str]:
    """Build the ordered list of parameter column names for results.csv.

    Called by `_build_results_rows` to determine output columns based on the
    inferred parameter layout. This is tightly coupled to the results schema.
    """
    columns: list[str] = []
    for key in infer_keys:
        entry = layout.get(key, {})
        vector_len = entry.get("vector_len")
        has_truth = entry.get("has_truth", False)
        fields = []
        if has_truth:
            fields.append("truth")
        fields.extend(["init", "final"])
        if has_truth:
            fields.extend(["init_delta", "final_delta"])
        for field in fields:
            if vector_len is None:
                columns.append(f"{field}.{key}")
            else:
                for index in range(vector_len):
                    columns.append(f"{field}.{key}[{index}]")
    return columns


def _assign_param_values(
    row: dict[str, Any],
    *,
    key: str,
    field: str,
    value: Any,
    vector_len: int | None,
) -> None:
    """Write scalar or vector parameter values into a flat results row.

    Used by `_build_results_rows` to emit per-parameter values in the expected
    column shape. This helper is specialized to the results.csv schema here.
    """
    if vector_len is None:
        row[f"{field}.{key}"] = value
        return
    if value is None:
        for index in range(vector_len):
            row[f"{field}.{key}[{index}]"] = None
        return
    if not isinstance(value, list):
        raise ValueError(f"Expected list for '{field}.{key}'")
    if len(value) != vector_len:
        raise ValueError(
            f"Vector length mismatch for '{field}.{key}': {len(value)} vs {vector_len}"
        )
    for index, entry in enumerate(value):
        row[f"{field}.{key}[{index}]"] = entry


def _build_results_rows(
    run_entries: list[dict[str, Any]],
    infer_keys: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Construct rows and column metadata for the aggregate results.csv file.

    Called by `_write_results_csv` to transform run summaries into flat rows
    while adding derived metrics. This is tightly coupled to the run artifacts
    and results schema defined by this script.
    """
    summaries = [entry.get("summary") for entry in run_entries]
    layout = _collect_param_layout(infer_keys, summaries)
    columns = _param_columns(infer_keys, layout)

    rows: list[dict[str, Any]] = []
    for entry in run_entries:
        run_id = entry["run_id"]
        summary = entry.get("summary")
        meta = entry.get("meta")
        plan_label = entry.get("plan_label")

        status = summary.get("status") if summary else "error"
        loss_init = summary.get("loss_init") if summary else None
        loss_final = summary.get("loss_final") if summary else None
        chi2_init = summary.get("chi2_init") if summary else None
        chi2_final = summary.get("chi2_final") if summary else None
        loss_truth = None
        if summary:
            loss_truth = summary.get("loss_truth", summary.get("loss_true"))

        improvement_ratio = None
        if summary and "improvement_ratio" in summary:
            improvement_ratio = summary.get("improvement_ratio")
        elif loss_init is not None and loss_final not in (None, 0):
            improvement_ratio = loss_init / loss_final

        created_at = _get_run_created_at(summary, meta)
        num_steps = summary.get("num_steps_completed") if summary else None

        prescribed_meta = meta.get("prescribed") if meta else {}
        optimizer_meta = meta.get("optimizer") if meta else {}
        precond_meta = (
            optimizer_meta.get("preconditioning") if optimizer_meta else {}
        )
        precond_method = precond_meta.get("method")

        row: dict[str, Any] = {
            "run_id": run_id,
            "status": status,
            "created_at": created_at,
            "run_note": summary.get("run_note") if summary else None,
            "loss_init": loss_init,
            "loss_final": loss_final,
            "chi2_init": chi2_init,
            "chi2_final": chi2_final,
            "loss_truth": loss_truth,
            "num_steps_completed": num_steps,
            "improvement_ratio": improvement_ratio,
            "plan_label": plan_label,
            "seed": (
                summary.get("run_seed")
                if summary and summary.get("run_seed") is not None
                else prescribed_meta.get("seed")
            ),
            "init.mode": prescribed_meta.get("init_mode"),
            "optimizer.n_iter": optimizer_meta.get("num_steps"),
            "optimizer.base_lr": optimizer_meta.get("learning_rate"),
            "eigen.use_eigen": (
                prescribed_meta.get("use_eigen")
                if prescribed_meta.get("use_eigen") is not None
                else (precond_method == "eigen" if precond_method else None)
            ),
            "eigen.whiten_basis": precond_meta.get("whiten_basis"),
            "eigen.truncate_k": precond_meta.get("truncate_k"),
            "eigen.truncate_by_eigval": precond_meta.get("truncate_by_eigval"),
            "noise.add_noise": prescribed_meta.get("add_noise"),
        }

        param_summary = summary.get("param_summary") if summary else None
        for key in infer_keys:
            key_layout = layout.get(key, {})
            vector_len = key_layout.get("vector_len")
            has_truth = key_layout.get("has_truth", False)
            entry_values = param_summary.get(key) if isinstance(param_summary, dict) else None

            if has_truth:
                truth_value = entry_values.get("truth") if isinstance(entry_values, dict) else None
                _assign_param_values(
                    row, key=key, field="truth", value=truth_value, vector_len=vector_len
                )
            init_value = entry_values.get("init") if isinstance(entry_values, dict) else None
            final_value = entry_values.get("final") if isinstance(entry_values, dict) else None
            _assign_param_values(row, key=key, field="init", value=init_value, vector_len=vector_len)
            _assign_param_values(
                row, key=key, field="final", value=final_value, vector_len=vector_len
            )
            if has_truth:
                init_delta = entry_values.get("init_delta") if isinstance(entry_values, dict) else None
                final_delta = entry_values.get("final_delta") if isinstance(entry_values, dict) else None
                _assign_param_values(
                    row,
                    key=key,
                    field="init_delta",
                    value=init_delta,
                    vector_len=vector_len,
                )
                _assign_param_values(
                    row,
                    key=key,
                    field="final_delta",
                    value=final_delta,
                    vector_len=vector_len,
                )

        rows.append(row)

    return rows, columns


def _write_results_csv(
    out_path: Path,
    run_entries: list[dict[str, Any]],
    infer_keys: tuple[str, ...],
) -> list[str]:
    """Write the aggregate results.csv file for all runs.

    Invoked by `_write_experiment_outputs` after runs complete or during
    aggregation-only mode. This is a workflow-specific writer and not intended
    for reuse outside this experiment layout.
    """
    rows, param_columns = _build_results_rows(run_entries, infer_keys)
    base_columns = [
        "run_id",
        "status",
        "created_at",
        "run_note",
        "loss_init",
        "loss_final",
        "chi2_init",
        "chi2_final",
        "loss_truth",
        "num_steps_completed",
        "improvement_ratio",
        "plan_label",
        "seed",
        "init.mode",
        "optimizer.n_iter",
        "optimizer.base_lr",
        "eigen.use_eigen",
        "eigen.whiten_basis",
        "eigen.truncate_k",
        "eigen.truncate_by_eigval",
        "noise.add_noise",
    ]
    columns = base_columns + param_columns
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})
    return columns


def _write_manifest(
    out_path: Path,
    *,
    created_at: str,
    script: str,
    prescription_path: str | None,
    plan_path: str | None,
    config_id: str | None,
    notes: str | None,
    overrides_config_keys: list[str],
    overrides_store_keys: list[str],
    runs: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> None:
    """Write the experiment manifest.json describing runs and outputs.

    Called by `_write_experiment_outputs` to summarize run metadata and artifact
    locations. This is specific to the prescribed Monte Carlo experiment format.
    """
    manifest = {
        "created_at": created_at,
        "script": script,
        "prescription_path": prescription_path,
        "plan_path": plan_path,
        "config_id": config_id,
        "notes": notes,
        "overrides": {
            "config_keys": overrides_config_keys,
            "store_keys": overrides_store_keys,
        },
        "runs_dir": "runs",
        "runs": runs,
        "artifacts": artifacts,
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _collect_run_entries(
    runs_dir: Path,
    plan_rows: list[dict[str, Any]],
    run_specs: list[dict[str, Any]],
    plan_labels: list[str | None],
) -> list[dict[str, Any]]:
    """Collect run metadata from on-disk artifacts for aggregation.

    Used in `main` for aggregate-only mode and after execution to build entries
    for results/manifest output. This is coupled to the run directory layout
    and artifact filenames used by this script.
    """
    entries: list[dict[str, Any]] = []
    for row, spec, label in zip(plan_rows, run_specs, plan_labels):
        if not _row_enabled(row):
            continue
        run_id = spec.get("run_id")
        if not run_id:
            continue
        run_dir = runs_dir / run_id
        summary = _load_json_dict(run_dir / "summary.json")
        meta = _load_json_dict(run_dir / "meta.json")
        entries.append(
            {
                "run_id": run_id,
                "run_dir": run_dir,
                "summary": summary,
                "meta": meta,
                "plan_label": label,
            }
        )
    return entries


def _build_manifest_runs(run_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize run entries for inclusion in manifest.json.

    Called by `_write_experiment_outputs` to turn entries into compact manifest
    records. This is tied to the manifest schema used by this script.
    """
    manifest_runs: list[dict[str, Any]] = []
    for entry in run_entries:
        run_id = entry["run_id"]
        summary = entry.get("summary")
        run_dir = entry.get("run_dir")
        status = summary.get("status") if summary else "error"
        run_payload: dict[str, Any] = {
            "run_id": run_id,
            "path": f"runs/{run_id}",
            "status": status,
        }
        if summary:
            for key in ("loss_init", "loss_final", "num_steps_completed"):
                if key in summary:
                    run_payload[key] = summary.get(key)
            run_note = summary.get("run_note")
            if run_note is not None:
                run_payload["run_note"] = run_note
        else:
            run_payload["message"] = f"summary.json missing in runs/{run_id}"
        manifest_runs.append(run_payload)
    return manifest_runs


def _write_experiment_outputs(
    *,
    outdir: Path,
    prescription: dict[str, Any],
    prescription_path: Path,
    plan_path: Path | None,
    run_entries: list[dict[str, Any]],
    infer_keys: tuple[str, ...],
    repo_root: Path,
) -> None:
    """Write aggregated outputs (results.csv and manifest.json) for the run set.

    Called in `main` after execution or in aggregate-only mode to produce
    experiment-level artifacts. This is specific to the prescribed Monte Carlo
    workflow and not intended as a generic utility.
    """
    experiment = prescription.get("experiment", {})
    results_filename = (
        experiment.get("results_filename")
        or experiment.get("results_table_name")
        or "results.csv"
    )
    results_path = outdir / results_filename
    _write_results_csv(results_path, run_entries, infer_keys)

    manifest_path = outdir / "manifest.json"
    overrides = prescription.get("overrides", {})
    config_keys = sorted((overrides.get("config") or {}).keys())
    store_keys = sorted((overrides.get("store") or {}).keys())
    manifest_runs = _build_manifest_runs(run_entries)
    experiment_notes = _first_present_string(experiment, EXPERIMENT_NOTE_KEYS)

    _write_manifest(
        manifest_path,
        created_at=_now_iso_local_ms(),
        script=_repo_relative_path(Path(__file__), repo_root=repo_root)
        or "examples/recipes/prescribed_monte_carlo.py",
        prescription_path=_repo_relative_path(prescription_path, repo_root=repo_root),
        plan_path=_repo_relative_path(plan_path, repo_root=repo_root),
        config_id=_get_nested(prescription, ["model", "config_id"]),
        notes=experiment_notes,
        overrides_config_keys=config_keys,
        overrides_store_keys=store_keys,
        runs=manifest_runs,
        artifacts=[
            {"path": "manifest.json"},
            {"path": results_filename},
        ],
    )


def main() -> None:
    """Run the prescribed Monte Carlo experiment pipeline.

    Args:
        --prescription: Path to the JSON experiment recipe. This file sets the
            experiment defaults, model config, and per-run seed rules, and may
            include an experiment-level note (`experiment.notes`; aliases
            `experiment.note`/`experiment.comment`/`experiment.comments`) that
            is persisted once in top-level manifest.json `notes`.
        --overrides: Optional CSV of per-run overrides. Rows
            cannot override model or overrides.* settings; they only mutate
            run-level fields. Per-run notes come from
            note/notes/comment/comments fields and are persisted as `run_note`.
        --outdir: Root output directory for the experiment. When supplied, this
            is treated as the experiment root, and all run artifacts
            (runs/<run_id>/...), manifest.json, and results.csv are written
            underneath this directory.
        --run-name: Optional name segment used to construct Results/<run-name>
            when --outdir is omitted. If both are omitted, a timestamp-based
            directory name is used. Supplying both uses --outdir verbatim and
            ignores --run-name.
        --dry-run: Resolve run specs and print previews without executing the
            optimization runs or writing run artifacts.
        --aggregate-only: Skip execution and only aggregate manifest.json and
            results.csv from existing run artifacts inside the resolved outdir.
        --num-preview: Limit the number of resolved run specs shown in preview
            output (useful with large plans).
    """
    parser = argparse.ArgumentParser(description="Prescribed Monte Carlo scaffold")
    parser.add_argument(
        "--prescription",
        type=Path,
        default=None,
        help="Path to prescription JSON (defaults to template if omitted)",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="Path to per-run overrides CSV (defaults to template if omitted)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Experiment root directory (no auto-suffix when provided).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Convenience name for Results/<run-name> when --outdir is omitted.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        default=False,
        help="Generate manifest + results.csv from existing runs without executing.",
    )
    parser.add_argument("--num-preview", type=int, default=None)

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    # Local toggle for writing per-run metric artifacts (metric.npz).
    # Default off to avoid extra artifact I/O unless explicitly needed.
    output_metric = False

    outdir_hint = Path(args.outdir) if args.outdir else None
    prescription_path, overrides_path = _resolve_prescription_and_overrides(args, outdir_hint)
    resolved_prescription = _repo_relative_path(prescription_path, repo_root=repo_root)
    resolved_overrides = _repo_relative_path(overrides_path, repo_root=repo_root)
    print(f"Resolved prescription path: {resolved_prescription or prescription_path}")
    if overrides_path is None:
        print("Resolved overrides path: None (per-run overrides disabled)")
    else:
        print(f"Resolved overrides path: {resolved_overrides or overrides_path}")

    # Plan/prescription parsing and run spec resolution: load the JSON recipe
    # and the optional overrides CSV that mutates per-run settings.
    prescription = _load_prescription(prescription_path)
    if overrides_path is None:
        plan_rows = []
    else:
        plan_rows = _load_plan_csv(overrides_path)
    n_runs = _get_nested(prescription, ["experiment", "n_runs"])
    # experiment.n_runs is authoritative: it pads or truncates the plan so
    # previews and run execution operate on the resolved run count.
    plan_rows, run_policy = _apply_experiment_n_runs(plan_rows, n_runs)
    enabled_count = sum(1 for row in plan_rows if _row_enabled(row))
    disabled_count = len(plan_rows) - enabled_count
    if run_policy["n_runs"] is None:
        print(
            "Run count policy: experiment.n_runs not set; "
            f"plan defines {run_policy['plan_runs']} run(s)."
        )
    else:
        print(
            "Run count policy: experiment.n_runs="
            f"{run_policy['n_runs']} (plan defines {run_policy['plan_runs']}) "
            f"-> resolved {run_policy['resolved_runs']} run(s)."
        )
        if run_policy["padded_runs"]:
            print(
                "  Added "
                f"{run_policy['padded_runs']} default-only run(s) to match experiment.n_runs."
            )
        if run_policy["truncated_runs"]:
            print(
                "WARNING: experiment.n_runs is authoritative; truncated "
                f"{run_policy['truncated_runs']} plan-defined run(s)."
            )
    print(
        f"Run enablement: {enabled_count} enabled, {disabled_count} disabled."
    )

    seed_preview_rows = plan_rows[:5]
    seed_preview = [row.get("seed") for row in seed_preview_rows if "seed" in row]
    if seed_preview:
        print(
            f"Plan seed preview (first {len(seed_preview_rows)} rows): {seed_preview}"
        )

    for row in plan_rows:
        forbidden = [
            key
            for key in row
            if not key.startswith("_")
            if key == "model"
            or key.startswith("model.")
            or key == "overrides"
            or key.startswith("overrides.")
        ]
        if forbidden:
            raise ValueError(
                "Plan rows cannot override model settings; remove: "
                + ", ".join(sorted(forbidden))
            )

    model_config_id = _get_nested(prescription, ["model", "config_id"])
    config_overrides = _get_nested(prescription, ["overrides", "config"]) or {}
    store_overrides = _get_nested(prescription, ["overrides", "store"]) or {}
    config_override_keys = (
        ", ".join(sorted(config_overrides.keys())) if config_overrides else "none"
    )
    store_override_keys = (
        ", ".join(sorted(store_overrides.keys())) if store_overrides else "none"
    )

    plan_labels = [row.get("_plan_label") for row in plan_rows]
    run_specs: list[dict[str, Any]] = []
    run_id_index = 0
    for index, row in enumerate(plan_rows):
        enabled = _row_enabled(row)
        if enabled:
            run_id_index += 1
            # Resolve each run spec by merging defaults + row overrides and
            # assigning an indexed run_id for enabled rows.
            resolved = _resolve_run_spec_with_id(
                prescription,
                row,
                index=index + 1,
                run_id_index=run_id_index,
            )
        else:
            # Disabled rows keep the resolved fields but do not receive a run_id.
            resolved = _resolve_run_spec_with_id(
                prescription,
                row,
                index=index + 1,
                run_id_index=None,
            )
        run_specs.append(resolved)

    outdir = _resolve_outdir(args.outdir, args.run_name)
    print(f"Resolved outdir: {outdir}")
    print(f"Model config_id: {model_config_id}")
    print(f"Config overrides: {config_override_keys}")
    print(f"Store overrides: {store_override_keys}")
    if any(label is not None for label in plan_labels):
        print("Plan column labels -> run_id mapping:")
        for label, spec in zip(plan_labels, run_specs):
            label_display = label or "(auto)"
            print(f"  {label_display} -> {spec.get('run_id')}")
    print(f"Resolved {len(run_specs)} run(s). Preview:")
    _print_preview(run_specs, args.num_preview)

    runs_with_prior_overrides = [
        spec
        for spec in run_specs
        if spec.get("prior_overrides")
    ]
    if runs_with_prior_overrides:
        print(
            "Prior overrides: "
            f"{len(runs_with_prior_overrides)} run(s) include per-run prior overrides."
        )
        for spec in runs_with_prior_overrides[:5]:
            run_id = spec.get("run_id") or "(disabled)"
            overrides = spec.get("prior_overrides", {})
            flattened = [
                f"{infer_key}.{field}"
                for infer_key, fields in overrides.items()
                for field in fields
            ]
            print(f"  {run_id}: {', '.join(flattened)}")
    else:
        print("Prior overrides: 0 runs include per-run prior overrides.")

    if args.dry_run:
        print("Dry run enabled; exiting before optimization.")
        return

    jax.config.update("jax_enable_x64", True)

    cfg = _resolve_config_id(model_config_id)
    cfg = _apply_config_overrides(cfg, config_overrides)

    forward_spec = build_forward_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic(cfg)

    infer_keys = tuple(prescription.get("infer_keys", []))
    if not infer_keys:
        raise ValueError("Prescription must include non-empty infer_keys.")
    inference_subspec = make_inference_subspec(
        base_spec=inference_spec,
        infer_keys=infer_keys,
        cfg=cfg,
    )

    # Store overrides and derived refresh steps: apply nested overrides via
    # dotted keys, then recompute any derived parameters to keep the store
    # self-consistent for forward/inference use.
    base_store = ParameterStore.from_spec_defaults(forward_spec)
    if store_overrides:
        base_store = base_store.replace(_flatten_store_overrides(store_overrides))
    base_store = base_store.refresh_derived(forward_spec)

    prior_info = prescription.get("priors", {})
    prior_spec = PriorSpec.from_info(base_store, prior_info)
    prior_spec_cache: dict[str, PriorSpec] = {}

    fim_cache: dict[str, dict[str, Any]] = {}
    fim_cache_last: dict[str, Any] | None = None

    def _stable_hash(payload: Any) -> str:
        if dataclasses.is_dataclass(payload):
            payload = dataclasses.asdict(payload)
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _hash_array(value: Any) -> str:
        array = np.asarray(value)
        return hashlib.sha256(array.tobytes()).hexdigest()

    cfg_hash = _stable_hash(cfg)
    forward_spec_hash = _stable_hash(forward_spec)

    if args.aggregate_only:
        if not outdir.exists():
            raise FileNotFoundError(f"Output directory not found: {outdir}")
        runs_dir = outdir / "runs"
        if not runs_dir.exists():
            print(
                "WARNING: Runs directory not found for aggregate-only mode; "
                f"continuing with empty run artifacts: {runs_dir}"
            )
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        runs_dir = outdir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

    run_id_prefix = _get_nested(prescription, ["experiment", "run_id_prefix"]) or "run"
    run_counter = 0

    if args.aggregate_only:
        run_entries = _collect_run_entries(runs_dir, plan_rows, run_specs, plan_labels)
        _write_experiment_outputs(
            outdir=outdir,
            prescription=prescription,
            prescription_path=prescription_path,
            plan_path=overrides_path,
            run_entries=run_entries,
            infer_keys=infer_keys,
            repo_root=repo_root,
        )
        print(f"Wrote experiment manifest/results to: {outdir}")
        return

    t0_experiment = time.time()

    for index, (row, run_spec) in enumerate(zip(plan_rows, run_specs)):
        if not _row_enabled(row):
            continue

        run_counter += 1
        run_id = run_spec.get("run_id") or f"{run_id_prefix}_{run_counter:04d}"
        run_spec["run_id"] = run_id

        print(f"\n--- Run {run_counter} ({run_id}) ---")
        t0_run = time.time()

        seed_value = run_spec.get("seed")
        if seed_value is None:
            raise ValueError(f"Run {run_id} resolved to a null seed.")
        seed = int(seed_value)
        rng_key = jr.PRNGKey(seed)
        rng_key, init_key = jr.split(rng_key)
        rng_key, noise_key = jr.split(rng_key)

        print(f"Generating synthetic data...")
        # Synthetic data generation + noise handling: build the "truth" store,
        # refresh derived values, then forward-model data and inject noise.
        truth_overrides = _flatten_store_overrides(run_spec.get("truth", {}))
        truth_store = base_store.replace(truth_overrides)
        truth_store = truth_store.refresh_derived(forward_spec)

        binder = SheraThreePlaneBinder(cfg, forward_spec, truth_store)

        # Generate the synthetic data
        data_psf = binder.model()

        # Optionally add noise to the data
        add_noise_value = _get_nested(run_spec, ["noise", "add_noise"])
        if add_noise_value is None:
            add_noise_value = _get_nested(prescription, ["defaults", "noise", "add_noise"])
        add_noise = bool(add_noise_value)
        if add_noise:
            rng_key, split_key = jr.split(rng_key)
            if np.min(data_psf) > 100:
                # Use gaussian approximation to shot noise if image is bright enough
                data = np.sqrt(data_psf) * jr.normal(split_key, data_psf.shape) + data_psf
            else:
                # Otherwise use poisson statistics
                data = jr.poisson(split_key, data_psf).astype(data_psf.dtype)
                # Casting back to float is important for the optimization
        else:  # No noise
            data = data_psf

        # Define data variance = data_psf -> Shot noise dominated
        # Add a minimum variance floor to avoid any division by zero
        data_var = jnp.maximum(data_psf, 1.0)

        noise_model = "gaussian"
        reduce = "sum"
        nll_loss_fn, _ = make_binder_nll_fn(
            binder=binder,
            infer_keys=infer_keys,
            data=data,
            var=data_var,
            noise_model=noise_model,
            reduce=reduce,
            theta0_store=truth_store,
        )
        fim_labels = generate_fim_labels(infer_keys, cfg=cfg, store=truth_store)
        loss_fn = nll_loss_fn

        # FIM computation and eigen preconditioning flow: pack_params converts
        # a structured ParameterStore into a flat vector for optimization/FIM.
        theta_true = pack_params(inference_subspec, truth_store)
        loss_true = float(loss_fn(theta_true))

        fim_cfg = run_spec.get("fim", {})
        reuse_fim_value = fim_cfg.get("reuse_fim")
        if reuse_fim_value is None:
            reuse_fim_value = _get_nested(run_spec, ["eigen", "reuse_fim"])
        if reuse_fim_value is None:
            reuse_fim_value = _get_nested(run_spec, ["optimizer", "reuse_fim"])
        if reuse_fim_value is None:
            reuse_fim_value = run_spec.get("reuse_fim")
        if reuse_fim_value is None:
            reuse_fim_value = _get_nested(prescription, ["defaults", "fim", "reuse_fim"])
        reuse_fim = bool(reuse_fim_value) if reuse_fim_value is not None else False

        fim_point = theta_true
        # FIM cache key notes:
        # - Default behavior is strict: reuse uses only exact cache matches unless
        #   reuse_fim=True is set in the prescription or plan. When reuse_fim=True,
        #   a cache miss can still reuse the last cached FIM with a warning.
        # - Safe reuse inputs include full theta_true, infer_keys, cfg/forward_spec
        #   identifiers, data/data_var hashes, and noise_model (all must match).
        cache_key_payload = {
            "infer_keys": infer_keys,
            "model_config_id": model_config_id,
            "cfg_hash": cfg_hash,
            "forward_spec_hash": forward_spec_hash,
            "theta_true_hash": _hash_array(theta_true),
            "add_noise": add_noise,
            "data_hash": _hash_array(data),
            "data_var_hash": _hash_array(data_var),
            "noise_model": noise_model,
            "reduce": reduce,
        }
        fim_cache_key = _stable_hash(cache_key_payload)[:12]
        cache_key = json.dumps(cache_key_payload, sort_keys=True, default=str)
        cache_entry = fim_cache.get(cache_key)
        if cache_entry is not None:
            print("FIM cache hit; reusing cached FIM.")
            F = cache_entry["F"]
            fim_diag = cache_entry["fim_diag"]
            fim_cache_hit = True
            fim_cache_last = cache_entry
        elif reuse_fim and fim_cache_last is not None:
            print(
                "WARNING: FIM cache miss for strict key; reusing previous cached FIM "
                "because reuse_fim=True. Inputs may be misaligned."
            )
            F = fim_cache_last["F"]
            fim_diag = fim_cache_last["fim_diag"]
            fim_cache_hit = False
        else:
            print("FIM cache miss; computing new FIM...")
            F = fim_theta(nll_loss_fn, fim_point)
            fim_diag = jnp.diag(F)
            fim_cache_hit = False
            fim_cache[cache_key] = {"F": F, "fim_diag": fim_diag}
            fim_cache_last = fim_cache[cache_key]
            print("FIM computed and cached for later.")

        eigen_cfg = run_spec.get("eigen", {})
        use_eigen_value = eigen_cfg.get("use_eigen")
        if use_eigen_value is None:
            use_eigen_value = _get_nested(prescription, ["defaults", "eigen", "use_eigen"])
        use_eigen = bool(use_eigen_value)
        whiten_basis_value = eigen_cfg.get("whiten_basis")
        if whiten_basis_value is None:
            whiten_basis_value = _get_nested(
                prescription, ["defaults", "eigen", "whiten_basis"]
            )
        whiten_basis = bool(whiten_basis_value)
        truncate_k = eigen_cfg.get("truncate_k")
        truncate_by_eigval = eigen_cfg.get("truncate_by_eigval")

        if use_eigen:
            # Switch to eigen theta_space when preconditioning in FIM eigenbasis.
            # EigenThetaMap encapsulates z<->theta transforms and truncation.
            theta_space = "eigen"
            precond_meta_base = {
                "method": "eigen",
                "whiten_basis": whiten_basis,
                "truncate_k": truncate_k,
                "truncate_by_eigval": truncate_by_eigval,
            }
        else:
            eigen_map = None
            theta_space = "primitive"
            precond_meta_base = {"method": "fim_diag"}

        init_cfg = run_spec.get("init", {})
        init_mode = init_cfg.get("mode") or _get_nested(
            prescription, ["defaults", "init", "mode"]
        )
        init_overrides = {
            key: value for key, value in init_cfg.items() if key != "mode"
        }
        init_overrides_flat = _flatten_store_overrides(init_overrides)
        prior_overrides = run_spec.get("prior_overrides") or {}

        if init_mode == "prior":
            # Per-run prior overrides only affect prior sampling and do not
            # change init.mode semantics.
            run_prior_spec = prior_spec
            if prior_overrides:
                run_prior_info, applied_keys = _apply_prior_overrides(
                    prior_info,
                    prior_overrides,
                    infer_keys=infer_keys,
                    base_store=base_store,
                )
                if applied_keys:
                    cache_key = _stable_hash(run_prior_info)
                    run_prior_spec = prior_spec_cache.get(cache_key)
                    if run_prior_spec is None:
                        run_prior_spec = PriorSpec.from_info(base_store, run_prior_info)
                        prior_spec_cache[cache_key] = run_prior_spec
                    applied_entries = []
                    for infer_key in applied_keys:
                        entry = run_prior_info.get(infer_key, {})
                        override_fields = prior_overrides.get(infer_key, {})
                        for field in override_fields:
                            if field in {"sigma", "dist"} and field in entry:
                                applied_entries.append(
                                    f"{infer_key}.{field}={entry[field]}"
                                )
                    if applied_entries:
                        print(
                            "Applying prior overrides: "
                            + ", ".join(applied_entries)
                        )
                else:
                    print(
                        "WARNING: prior overrides were provided but none were applied; "
                        "using base priors."
                    )
            init_store = run_prior_spec.sample_near(
                center_store=truth_store,
                rng_key=init_key,
                keys=infer_keys,
            )
            if init_overrides_flat:
                init_store = init_store.replace(init_overrides_flat)
        elif init_mode == "explicit":
            # Explicit init precedence for coupled exposure/log-flux fields:
            # 1) `imaging.exposure_time_s` only -> `refresh_derived` recomputes
            #    `binary.log_flux_total` from primitives.
            # 2) `binary.log_flux_total` only -> explicit value is used.
            # 3) both provided -> explicit `binary.log_flux_total` wins because
            #    derived keys are re-applied after `refresh_derived`.
            init_store = truth_store
            if init_overrides_flat:
                (
                    init_primitive_overrides,
                    init_derived_overrides,
                    init_unknown_overrides,
                ) = _partition_overrides_by_kind(init_overrides_flat, forward_spec)
                if init_unknown_overrides:
                    print(
                        "WARNING: explicit init overrides include keys that are not "
                        "declared as primitive/derived in forward_spec; applying "
                        "them as direct overrides: "
                        + ", ".join(sorted(init_unknown_overrides))
                    )

                if init_primitive_overrides:
                    init_store = init_store.replace(init_primitive_overrides)
                init_store = init_store.refresh_derived(forward_spec)

                if init_derived_overrides:
                    infer_derived_keys = [
                        key
                        for key in infer_keys
                        if key in init_derived_overrides
                    ]
                    init_store = init_store.replace(init_derived_overrides)
                    print(
                        "Init explicit precedence: explicit derived overrides are "
                        "authoritative after refresh"
                        + (
                            f" (infer keys: {', '.join(infer_derived_keys)})."
                            if infer_derived_keys
                            else "."
                        )
                    )
                elif init_primitive_overrides:
                    print(
                        "Init explicit precedence: no explicit derived overrides; "
                        "derived values (e.g., binary.log_flux_total) come from "
                        "forward transforms after primitive overrides."
                    )

                if init_unknown_overrides:
                    init_store = init_store.replace(init_unknown_overrides)
            else:
                init_store = init_store.refresh_derived(forward_spec)
            if prior_overrides:
                print(
                    f"Note: prior overrides provided for run_id={run_id} but "
                    f"init.mode={init_mode}; ignoring prior overrides."
                )
        else:
            raise ValueError(f"Unknown init.mode '{init_mode}'")
        if init_mode == "prior":
            init_store = init_store.refresh_derived(forward_spec)

        _, theta0 = make_binder_nll_fn(
            binder=binder,
            infer_keys=infer_keys,
            data=data,
            var=data_var,
            noise_model="gaussian",
            reduce="sum",
            theta0_store=init_store,
        )

        if use_eigen:
            if truncate_k is not None and truncate_by_eigval is not None:
                print(
                    "truncate_k is set; ignoring truncate_by_eigval="
                    f"{truncate_by_eigval}."
                )

            theta_ref = theta0
            # Build full eigen map and optionally truncate modes (k or eigval).
            eigen_map_full = EigenThetaMap.from_fim(
                F,
                theta_ref,
                whiten=whiten_basis,
            )
            eigvals_full = (
                np.asarray(eigen_map_full.eigvals)
                if eigen_map_full.eigvals is not None
                else None
            )

            if truncate_k is not None:
                k = int(truncate_k)
            elif truncate_by_eigval is not None and eigvals_full is not None:
                k = int(np.sum(eigvals_full >= truncate_by_eigval))
            else:
                k = None

            if k is not None:
                if k <= 0:
                    print("truncate_by_eigval removed all modes; keeping top-1.")
                    k = 1
                eigen_map = EigenThetaMap.from_fim(
                    F,
                    theta_ref,
                    truncate=k,
                    whiten=whiten_basis,
                )
            else:
                eigen_map = eigen_map_full

            eigvals_kept = (
                np.asarray(eigen_map.eigvals)
                if eigen_map.eigvals is not None
                else np.array([])
            )

            # Transform theta into z coordinates for the optimizer when in
            # eigen space; theta_space controls how run_shera_gd logs artifacts.
            z0 = eigen_map.z_from_theta(theta0)
            if whiten_basis:
                lr_vec = np.ones_like(z0)
                curvature_vec = np.ones_like(z0)
            else:
                lr_vec = 1.0 / (eigvals_kept + 1e-12)
                curvature_vec = eigvals_kept

            index_map = build_eigen_index_map(eigen_map)
            # loss_opt maps optimizer z -> physical theta so the loss stays
            # defined in the original parameter space.
            loss_opt = lambda z: loss_fn(eigen_map.theta_from_z(z))
            theta0_opt = z0
            metric_payload = None
            if output_metric:
                metric_payload = {
                    "theta_ref": np.asarray(theta0_opt),
                    "metric_diag": np.asarray(curvature_vec),
                    "lr_scale": np.asarray(lr_vec),
                }
            precond_meta = {
                **precond_meta_base,
                "lr_vec": np.asarray(lr_vec),
            }
        else:
            # Primitive theta_space keeps parameters untransformed and applies
            # a diagonal preconditioner from the FIM.
            index_map = build_index_map(inference_subspec, init_store, theta=theta0)
            lr_vec = 1.0 / (np.asarray(fim_diag) + 1e-12)
            curvature_vec = fim_diag
            loss_opt = loss_fn
            theta0_opt = theta0
            metric_payload = None
            if output_metric:
                metric_payload = {
                    "theta_ref": np.asarray(theta0_opt),
                    "metric_diag": np.asarray(curvature_vec),
                    "lr_scale": np.asarray(lr_vec),
                }
            precond_meta = {
                **precond_meta_base,
                "lr_vec": np.asarray(lr_vec),
            }

        labels_by_key = map_labels_to_keys(
            infer_keys,
            fim_labels,
            store=init_store if use_eigen else None,
            index_map=None if use_eigen else index_map,
        )

        optimizer_cfg = run_spec.get("optimizer", {})
        n_iter_value = optimizer_cfg.get("n_iter")
        if n_iter_value is None:
            raise ValueError(f"Run {run_id} resolved to a null optimizer.n_iter.")
        base_lr_value = optimizer_cfg.get("base_lr")
        if base_lr_value is None:
            raise ValueError(f"Run {run_id} resolved to a null optimizer.base_lr.")
        n_iter = int(n_iter_value)
        base_lr = float(base_lr_value)

        # Loss function setup and optimization execution: run_shera_gd consumes
        # the chosen theta_space, preconditioner, and metadata for artifacts.
        config_payload = _config_payload(cfg, repo_root=repo_root)
        theta_final_opt, trace, artifacts = run_shera_gd(
            loss_fn=loss_opt,
            theta0=theta0_opt,
            index_map=index_map,
            learning_rate=base_lr,
            lr_vec=lr_vec,
            num_steps=n_iter,
            runs_dir=runs_dir,
            run_id=run_id,
            return_artifacts=True,
            theta_space=theta_space,
            metric=metric_payload,
            extra_meta={
                "optimizer": {"preconditioning": precond_meta},
                "theta": {"labels_by_key": labels_by_key},
                "model": {
                    "config_id": model_config_id,
                    "config": config_payload,
                },
                "prescribed": {
                    "index": run_counter - 1,
                    "seed": seed,
                    "run_id": run_id,
                    "init_mode": init_mode,
                    "add_noise": add_noise,
                    "use_eigen": use_eigen,
                    "reuse_fim": reuse_fim,
                    "fim_cache_key": fim_cache_key,
                    "fim_cache_hit": fim_cache_hit,
                },
            },
        )

        # Convert back to physical theta if we optimized in eigen z-space.
        if use_eigen and eigen_map is not None:
            theta_final = eigen_map.theta_from_z(theta_final_opt)
        else:
            theta_final = theta_final_opt

        # Rehydrate structured parameters after optimization: unpack_params
        # restores the ParameterStore layout for reporting and artifacts.
        final_store = store_unpack_params(inference_subspec, theta_final, init_store)
        init_psf = binder.model(
            strip_structural(init_store, structural_keys=binder.structural_store_keys())
        )
        final_psf = binder.model(
            strip_structural(final_store, structural_keys=binder.structural_store_keys())
        )

        loss_init = float(loss_fn(theta0))
        loss_final = float(loss_fn(theta_final))
        chi2_init = _reduced_chi2_between_images(
            data,
            init_psf,
            variance_image=data_var,
        )
        chi2_final = _reduced_chi2_between_images(
            data,
            final_psf,
            variance_image=data_var,
        )
        improvement_ratio = loss_init / loss_final if loss_final != 0 else float("nan")

        if artifacts is not None:
            run_dir = Path(artifacts["run_dir"]) if artifacts.get("run_dir") else None
            if run_dir is not None:
                truth_dict = {key: truth_store.get(key) for key in infer_keys}
                init_dict = {key: init_store.get(key) for key in infer_keys}
                final_dict = {key: final_store.get(key) for key in infer_keys}
                param_summary = build_param_summary(
                    init_dict, final_dict, truth=truth_dict
                )
                patch_summary(
                    run_dir,
                    {
                        "param_summary": param_summary,
                        "loss_true": loss_true,
                        "chi2_init": chi2_init,
                        "chi2_final": chi2_final,
                        "improvement_ratio": improvement_ratio,
                        "run_note": run_spec.get("note"),
                        "run_seed": seed,
                        "run_created_at": _now_iso_local_ms(),
                    },
                )
                _maybe_warn_missing_artifacts(run_dir)

        t1_run = time.time()
        print(
            "Run summary: loss(true)={:.6g}, loss(init)={:.6g}, loss(final)={:.6g}, "
            "chi2(init)={:.6g}, chi2(final)={:.6g}, time={:.3f} sec".format(
                loss_true,
                loss_init,
                loss_final,
                chi2_init,
                chi2_final,
                t1_run - t0_run,
            )
        )

    # Artifact/manifest/results aggregation: collect per-run metadata and write
    # experiment-level manifest + results tables.
    run_entries = _collect_run_entries(runs_dir, plan_rows, run_specs, plan_labels)
    _write_experiment_outputs(
        outdir=outdir,
        prescription=prescription,
        prescription_path=prescription_path,
        plan_path=overrides_path,
        run_entries=run_entries,
        infer_keys=infer_keys,
        repo_root=repo_root,
    )
    t1_experiment = time.time()
    print(
        "\nExecution complete in {:.3f} sec. Wrote runs to: {}".format(
            t1_experiment - t0_experiment,
            runs_dir,
        )
    )
    print(f"Wrote experiment manifest/results to: {outdir}")

if __name__ == "__main__":
    main()
