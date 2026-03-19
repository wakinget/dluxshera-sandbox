"""Prescribed Monte Carlo experiment.

Purpose: provide a robust way to define and run monte carlo experiments.
Local helpers are defined here for now and document whether they are reusable.

Execution flow
--------------
- Load the prescription YAML/JSON (native system/experiment config) and optional
  run plan CSV specified inside the experiment config. Plan paths are resolved
  relative to the config file when relative. If no plan is specified, defaults
  are derived solely from the experiment config.
- Resolve run specs from experiment-level controls (seed, optimizer, eigenmodes,
  infer_keys, noise, outputs, init, priors) and experiment.monte_carlo settings.
- Build data and inference systems (optionally distinct via experiment.inference_system),
  generate synthetic observations with optional noise, and run inference.
- Run optimization in eigen space (FIM-based) or primitive parameter space.
- Write run artifacts under runs/<run_id>/..., including summaries and logs.
- Aggregate manifest.json and results.csv across runs at the experiment root.
  `results.csv` defaults to column orientation (first column `key`, one column
  per `run_id`), with an optional compatibility mode for row orientation.

Notes behavior
--------------
- Experiment-level note: set `experiment.notes` in the prescription (aliases:
  `experiment.note`, `experiment.comment`, `experiment.comments`). This value
  is written once to top-level `manifest.json["notes"]`.
- Per-run note: set `note`/`notes`/`comment`/`comments` in run plan rows. The
  resolved value is stored as `run_note` in run summaries and aggregate outputs.
- Detector knowledge-error realization policy: detector-layer
  `knowledge_error.realization_policy` supports `fixed_per_experiment` (default)
  and `per_run`. Explicit `knowledge_error.seed` remains authoritative. The
  inference-side base config is kept unseeded outside the run loop so `per_run`
  layers are realized fresh from each resolved run seed.

CLI arguments (mirrors `main`)
------------------------------
- --prescription: YAML/JSON experiment config (defaults to template when omitted).
- --outdir: explicit experiment root directory. When provided, this overrides
  `experiment.outputs.outdir`.
- --run-name: optional name segment used to build Results/<run-name> when
  neither `--outdir` nor `experiment.outputs.outdir` is set.
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
import matplotlib.pyplot as plt

from dluxshera.config.io import load_user_config, load_config_file
from dluxshera.config.resolver import resolve_config, resolve_system_config
from dluxshera.inference.optimization import (
    EigenThetaMap,
    fim_theta,
    generate_fim_labels,
    make_binder_nll_fn,
    map_labels_to_keys,
    run_shera_gd,
    diagnose_first_step,
)
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.run_artifacts import (
    _now_iso_local_ms,
    build_param_summary,
    patch_summary,
)
from dluxshera.inference.signals import build_signals
from dluxshera.params.packing import (
    build_eigen_index_map,
    build_index_map,
    pack_params,
    unpack_params as store_unpack_params,
)
from dluxshera.params.store import ParameterStore
from dluxshera.plot.plotting import (
    apply_plot_defaults,
    get_default_cmaps,
    plot_eigenvalue_spectrum,
    plot_fim,
    plot_parameter_history,
    plot_psf_comparison,
    plot_signals_grid,
    plot_pixel_offset_maps,
)
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.noise import (
    apply_observation_noise,
    make_subseed,
)

DEFAULT_PRESCRIPTION_PATH = Path(
    "examples/recipes/prescribed_mc_template/prescription.yaml"
)
PLAN_FREE_TEXT_COLUMNS = frozenset({"note", "notes", "comment", "comments"})
EXPERIMENT_NOTE_KEYS = ("notes", "note", "comment", "comments")

# Plotting defaults
_ = get_default_cmaps()
apply_plot_defaults()
plt.rcParams["image.cmap"] = "inferno_nan"

LEGACY_KEY_MAP = {
    "binary.separation_as": "source.separation_as",
    "binary.position_angle_deg": "source.position_angle_deg",
    "binary.x_position_as": "source.x_position_as",
    "binary.y_position_as": "source.y_position_as",
    "binary.log_flux_total": "source.log_flux_total",
    "binary.contrast": "source.contrast",
    "system.plate_scale_as_per_pix": "optics.plate_scale_as_per_pix",
    "primary.zernike_coeffs_nm": "optics.primary.zernike_coeffs_nm",
    "secondary.zernike_coeffs_nm": "optics.secondary.zernike_coeffs_nm",
    "imaging.exposure_time_s": "source.exposure_time_s",
}


def _get_pixel_offset_maps(binder: SheraBinder):
    """Return (dx_map, dy_map) from the binder if pixel_offsets layer exists."""
    layer = binder.detector.layers.get("pixel_offsets") if hasattr(binder, "detector") else None
    if layer is None or not hasattr(layer, "dx_map") or not hasattr(layer, "dy_map"):
        return None
    return np.asarray(layer.dx_map), np.asarray(layer.dy_map)


def _get_pixel_response_map(binder: SheraBinder):
    """Return pixel_response map from the binder if the layer exists."""
    layer = binder.detector.layers.get("pixel_response") if hasattr(binder, "detector") else None
    if layer is None or not hasattr(layer, "pixel_response"):
        return None
    return np.asarray(layer.pixel_response)


def _refresh_preserving_derived_infer_keys(store, *, infer_keys: tuple[str, ...], spec):
    """Refresh derived values while preserving sampled values for derived infer keys."""
    sampled_derived: dict[str, Any] = {}
    for key in infer_keys:
        if key not in spec:
            continue
        if spec.get(key).kind != "derived":
            continue
        try:
            sampled_derived[key] = store.get(key)
        except KeyError:
            continue

    refreshed = store.refresh_derived(spec)
    if sampled_derived:
        refreshed = refreshed.replace(sampled_derived)
    return refreshed


def _trace_with_initial_point(
    trace: dict[str, Any],
    *,
    theta0: np.ndarray,
    loss0: float | None = None,
) -> dict[str, Any]:
    """Return a trace copy with iteration-0 (sampled init) prepended."""
    theta_hist = np.asarray(trace["theta"])
    theta0_arr = np.asarray(theta0)

    prepend_theta0 = True
    if theta_hist.shape[0] > 0 and np.allclose(theta_hist[0], theta0_arr, rtol=0.0, atol=0.0):
        prepend_theta0 = False

    trace_with_init = dict(trace)
    if prepend_theta0:
        trace_with_init["theta"] = np.concatenate((theta0_arr[None, ...], theta_hist), axis=0)

        if "loss" in trace:
            loss_hist = np.asarray(trace["loss"])
            loss0_value = np.nan if loss0 is None else float(loss0)
            trace_with_init["loss"] = np.concatenate(
                (np.asarray([loss0_value], dtype=loss_hist.dtype), loss_hist),
                axis=0,
            )
    else:
        trace_with_init["theta"] = theta_hist

    return trace_with_init


def _require_experiment_seed(experiment_cfg: dict[str, Any]) -> int:
    seed = experiment_cfg.get("seed")
    if seed is None:
        raise ValueError("experiment.seed is required for prescribed Monte Carlo.")
    return int(seed)


def _eigen_defaults(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    eigen_block = experiment_cfg.get("eigenmodes") or experiment_cfg.get("eigen") or {}
    return {
        "use_eigen": bool(eigen_block.get("enable", eigen_block.get("use_eigen", False))),
        "whiten_basis": bool(eigen_block.get("whiten", eigen_block.get("whiten_basis", False))),
        "truncate_k": eigen_block.get("truncate_k"),
        "truncate_by_eigval": eigen_block.get("truncate_by_eigval"),
        "reuse_fim": eigen_block.get("reuse_fim"),
    }


def _init_defaults(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    init_block = copy.deepcopy(experiment_cfg.get("init", {}) or {})
    sampling = init_block.pop("sampling", None)
    if sampling and "mode" not in init_block:
        init_block["mode"] = "prior" if sampling == "prior" else "explicit"
    return init_block


def _noise_defaults(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    noise_block = copy.deepcopy(experiment_cfg.get("noise", {}) or {})
    if "enabled" in noise_block and "add_noise" not in noise_block:
        noise_block["add_noise"] = noise_block["enabled"]
    return noise_block


def _outputs_defaults(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(experiment_cfg.get("outputs", {}) or {})


def _optimizer_defaults(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(experiment_cfg.get("optimizer", {}) or {})


def _mc_defaults_from_experiment(
    experiment_cfg: dict[str, Any],
    mc_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build effective MC defaults using new experiment-level schema."""
    defaults_block = mc_cfg.get("defaults", {}) if isinstance(mc_cfg.get("defaults", {}), dict) else {}
    defaults = copy.deepcopy(defaults_block)
    defaults["seed"] = _require_experiment_seed(experiment_cfg)

    defaults.setdefault("truth", {})
    defaults.setdefault("optimizer", {})
    defaults.setdefault("eigen", {})
    defaults.setdefault("fim", {})
    defaults.setdefault("noise", {})
    defaults.setdefault("outputs", {})
    defaults.setdefault("init", {})

    _deep_update(defaults["optimizer"], _optimizer_defaults(experiment_cfg))
    _deep_update(defaults["eigen"], _eigen_defaults(experiment_cfg))
    if mc_cfg.get("reuse_fim") is not None and defaults["fim"].get("reuse_fim") is None:
        defaults["fim"]["reuse_fim"] = mc_cfg.get("reuse_fim")
    eigen_defaults = _eigen_defaults(experiment_cfg)
    if eigen_defaults.get("reuse_fim") is not None and defaults["fim"].get("reuse_fim") is None:
        defaults["fim"]["reuse_fim"] = eigen_defaults["reuse_fim"]
    _deep_update(defaults["noise"], _noise_defaults(experiment_cfg))
    _deep_update(defaults["outputs"], _outputs_defaults(experiment_cfg))
    _deep_update(defaults["init"], _init_defaults(experiment_cfg))

    mc_effective = dict(mc_cfg)
    mc_effective["defaults"] = defaults
    return mc_effective, defaults

def _timestamp_tag() -> str:
    """Return a sortable timestamp string for labeling output directories.

    Used by `_resolve_outdir` when no explicit output directory/run name is given,
    providing consistent time-based naming in the main execution flow.
    This helper is broadly reusable as a generic timestamp label utility.
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_prescription(path: Path) -> dict[str, Any]:
    """Load a prescription config (YAML or JSON) from disk.

    Keys that start with `_` are treated as private/disabled template fields and
    are recursively stripped from the loaded mapping. The resulting mapping is
    used downstream to detect legacy vs. native schema and then fed into the
    canonical `resolve_config` path.
    """
    loaded = load_config_file(path)
    return _strip_private_keys(loaded)


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


def _is_legacy_prescription(presc: dict[str, Any]) -> bool:
    """Detect legacy prescription schema (no top-level system block)."""
    return "system" not in presc and "model" in presc


def _migrate_param_key(key: str) -> str:
    """Translate legacy parameter keys to migrated schema."""
    return LEGACY_KEY_MAP.get(key, key)


def _migrate_key_mapping(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a mapping with legacy param keys migrated."""
    return {_migrate_param_key(k): v for k, v in payload.items()}


def _upgrade_legacy_prescription(
    legacy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate legacy prescription schema into native config mapping.

    Returns a tuple of (user_cfg, compatibility_meta) where ``user_cfg`` is a
    canonical config mapping containing ``system`` and ``experiment`` blocks
    ready for ``resolve_config``, and ``compatibility_meta`` carries any legacy
    override payloads that still need special handling (e.g., structural
    config overrides applied post-resolution).
    """
    experiment_block = legacy.get("experiment", {}) or {}
    legacy_overrides = legacy.get("overrides", {}) or {}
    config_overrides = _migrate_key_mapping(legacy_overrides.get("config", {}) or {})
    store_overrides = _migrate_key_mapping(legacy_overrides.get("store", {}) or {})

    # Legacy infer_keys/priors live at the top level.
    infer_keys = tuple(_migrate_param_key(key) for key in legacy.get("infer_keys", []) or [])
    priors = _migrate_key_mapping(legacy.get("priors", {}) or {})

    mc_defaults = copy.deepcopy(legacy.get("defaults", {}) or {})
    if store_overrides:
        mc_defaults.setdefault("truth", {})
        _deep_update(mc_defaults["truth"], store_overrides)

    mc_block = {
        "n_runs": experiment_block.get("n_runs"),
        "run_id_prefix": experiment_block.get("run_id_prefix"),
        "results_filename": (
            experiment_block.get("results_filename")
            or experiment_block.get("results_table_name")
        ),
        "results_orientation": experiment_block.get("results_orientation"),
        "defaults": mc_defaults,
    }

    notes = _first_present_string(experiment_block, EXPERIMENT_NOTE_KEYS)
    experiment_cfg = {
        "notes": notes,
        "infer_keys": infer_keys,
        "priors": priors,
        "prescribed_mc": mc_block,
    }

    model_cfg = legacy.get("model", {}) or {}
    system_preset = _resolve_config_id(model_cfg.get("config_id"))
    user_cfg = {
        "system": {"preset": system_preset},
        "experiment": experiment_cfg,
    }

    return user_cfg, {"config_overrides": config_overrides, "legacy_schema": True}

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
    """Apply monte_carlo.n_runs precedence to the plan-defined run rows.

    When experiment.monte_carlo.n_runs is set, it becomes authoritative: the
    plan is padded with default-only rows or truncated as needed so downstream
    previews and run execution operate on the resolved run count. When
    experiment.monte_carlo.n_runs is not set, the plan length defines the run
    count and an empty plan is rejected.
    """
    plan_rows_copy = [row.copy() for row in plan_rows]
    plan_runs = len(plan_rows_copy)

    if n_runs is None:
        if plan_runs == 0:
            raise ValueError(
                "Unable to resolve run count: experiment.monte_carlo.n_runs is not set and "
                "run_plan.csv defines 0 runs."
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
        raise ValueError("experiment.monte_carlo.n_runs must be an integer.") from exc

    if resolved_n_runs <= 0:
        raise ValueError("experiment.monte_carlo.n_runs must be a positive integer.")

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


def _detect_prescription_candidate(outdir: Path) -> Path | None:
    """Scan an output directory for a candidate prescription file.

    This helper is used when --prescription is omitted but an outdir is
    provided. Detection rules are intentionally conservative; update this helper
    when additional filename conventions are introduced in prescribed Monte
    Carlo workflows.
    """
    if not outdir.exists():
        return None

    prescription_candidates = sorted(
        candidate
        for candidate in outdir.rglob("*")
        if candidate.suffix.lower() in {".json", ".yaml", ".yml"}
        if "prescription" in candidate.name.lower()
    )

    if len(prescription_candidates) > 1:
        joined = "\n".join(f"- {candidate}" for candidate in prescription_candidates)
        raise ValueError(
            "Multiple prescription candidates found in "
            f"{outdir}. Provide --prescription to disambiguate:\n{joined}"
        )

    return prescription_candidates[0] if prescription_candidates else None


def _resolve_prescription(args: argparse.Namespace, outdir: Path | None) -> Path:
    """Resolve prescription path from CLI args and optional outdir scan."""
    prescription_path = args.prescription
    if prescription_path is None and outdir is not None:
        prescription_path = _detect_prescription_candidate(outdir)

    if prescription_path is None:
        outdir_label = f"found in {outdir}" if outdir is not None else "outdir not provided"
        print(
            "WARNING: No prescription path provided/detected (no prescription config "
            f"{outdir_label}); falling back to template at {DEFAULT_PRESCRIPTION_PATH}"
        )
        prescription_path = DEFAULT_PRESCRIPTION_PATH

    return Path(prescription_path)


def _resolve_path_relative_to_prescription(
    value: str | Path | None,
    *,
    prescription_path: Path,
    field_name: str,
) -> Path | None:
    """Resolve optional paths relative to the prescription file when relative."""
    if value is None:
        return None

    if isinstance(value, Path):
        path_value = value
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        path_value = Path(stripped)
    else:
        raise ValueError(f"{field_name} must be a path string or null.")

    if path_value.is_absolute():
        return path_value
    return (prescription_path.parent / path_value).resolve()


def _resolve_plan_csv_path(plan_csv: str | Path | None, *, prescription_path: Path) -> Path | None:
    """Resolve plan CSV path relative to the prescription file when relative."""
    return _resolve_path_relative_to_prescription(
        plan_csv,
        prescription_path=prescription_path,
        field_name="experiment.monte_carlo.run_plan",
    )


def _stable_hash_payload(payload: Any) -> str:
    """Return a stable SHA256 hash for JSON-serializable payloads."""
    if dataclasses.is_dataclass(payload):
        payload = dataclasses.asdict(payload)
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _hash_array_bytes(value: Any) -> str:
    """Return a SHA256 hash of an array's raw bytes."""
    array = np.asarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _normalize_ke_realization_policy(value: Any) -> str:
    """Normalize detector knowledge-error realization policy strings."""
    if value is None:
        return "fixed_per_experiment"
    policy = str(value).strip().lower()
    if policy not in {"fixed_per_experiment", "per_run"}:
        raise ValueError(
            "detector.layers[*].knowledge_error.realization_policy must be "
            "'fixed_per_experiment' or 'per_run'."
        )
    return policy


def _layer_metadata_key(layer_name: str | None, idx: int, *, used: set[str]) -> str:
    """Build a stable layer metadata key, disambiguating duplicate layer names."""
    base = (layer_name or f"layer_{idx}").strip() if layer_name is not None else f"layer_{idx}"
    if not base:
        base = f"layer_{idx}"
    key = base
    if key in used:
        key = f"{base}_{idx}"
    used.add(key)
    return key


def _seed_detector_knowledge_errors_with_policy(
    system_cfg: dict[str, Any],
    *,
    experiment_seed: int,
    token_prefix: str,
    run_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach detector knowledge-error seeds, honoring realization policies.

    Rules per detector layer with a ``knowledge_error`` mapping:
    - Explicit ``knowledge_error.seed`` is preserved as-is.
    - ``realization_policy`` defaults to ``fixed_per_experiment`` when omitted.
    - ``fixed_per_experiment`` derives seed from ``experiment_seed``.
    - ``per_run`` derives seed from ``run_seed`` when provided, otherwise falls
      back to ``experiment_seed``.

    Callers that want fresh ``per_run`` realizations must pass an unseeded base
    config each time. Once a concrete seed is attached to a layer, it is treated
    as explicit and preserved on subsequent calls.
    """
    cfg = copy.deepcopy(system_cfg)
    detector_cfg = cfg.get("detector", {}) if isinstance(cfg, dict) else {}
    layers = detector_cfg.get("layers", []) if isinstance(detector_cfg, dict) else []

    seeded_layers = []
    metadata_layers: dict[str, dict[str, Any]] = {}
    used_meta_keys: set[str] = set()
    has_per_run = False

    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            seeded_layers.append(layer)
            continue

        layer_copy = dict(layer)
        layer_name = layer_copy.get("name")
        knowledge_error = layer_copy.get("knowledge_error")

        if isinstance(knowledge_error, dict):
            seeded_ke = dict(knowledge_error)
            policy = _normalize_ke_realization_policy(seeded_ke.get("realization_policy"))
            seeded_ke["realization_policy"] = policy
            has_per_run = has_per_run or policy == "per_run"

            explicit_seed = seeded_ke.get("seed") is not None
            if not explicit_seed:
                seed_base = (
                    run_seed
                    if policy == "per_run" and run_seed is not None
                    else experiment_seed
                )
                seed_token = f"{token_prefix}.{layer_name or 'layer'}.{idx}"
                seeded_ke["seed"] = make_subseed(seed_base, seed_token)

            layer_copy["knowledge_error"] = seeded_ke

            layer_name_text = None
            if layer_name is not None:
                raw_name = str(layer_name).strip()
                layer_name_text = raw_name if raw_name else None
            meta_key = _layer_metadata_key(
                layer_name_text,
                idx,
                used=used_meta_keys,
            )
            metadata_layers[meta_key] = {
                "name": layer_name,
                "index": idx,
                "model": seeded_ke.get("model"),
                "scale": seeded_ke.get("scale"),
                "realization_policy": policy,
                "seed": seeded_ke.get("seed"),
                "seed_source": (
                    "explicit"
                    if explicit_seed
                    else (
                        "run_seed"
                        if policy == "per_run" and run_seed is not None
                        else "experiment_seed"
                    )
                ),
            }

        seeded_layers.append(layer_copy)

    if isinstance(detector_cfg, dict):
        detector_cfg = dict(detector_cfg)
        detector_cfg["layers"] = seeded_layers
        cfg["detector"] = detector_cfg

    metadata = {
        "token_prefix": token_prefix,
        "experiment_seed": experiment_seed,
        "run_seed": run_seed,
        "has_per_run_realization": has_per_run,
        "layers": metadata_layers,
    }
    return cfg, metadata


def _detector_ke_has_per_run_realization(system_cfg: dict[str, Any]) -> bool:
    """Return whether any detector layer requests ``realization_policy=per_run``.

    This is intentionally read-only so callers can inspect the configured
    policy without materializing detector knowledge-error seeds.
    """
    detector_cfg = system_cfg.get("detector", {}) if isinstance(system_cfg, dict) else {}
    layers = detector_cfg.get("layers", []) if isinstance(detector_cfg, dict) else []

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        knowledge_error = layer.get("knowledge_error")
        if not isinstance(knowledge_error, dict):
            continue
        policy = _normalize_ke_realization_policy(knowledge_error.get("realization_policy"))
        if policy == "per_run":
            return True
    return False


def _seed_detector_knowledge_errors(
    system_cfg: dict[str, Any],
    *,
    base_seed: int,
    token_prefix: str,
    run_seed: int | None = None,
) -> dict[str, Any]:
    """Backward-compatible wrapper around policy-aware detector KE seeding."""
    cfg, _ = _seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=base_seed,
        token_prefix=token_prefix,
        run_seed=run_seed,
    )
    return cfg


def _build_fim_cache_key_payload(
    *,
    infer_keys: tuple[str, ...],
    system_label: str,
    cfg_hash: str,
    forward_spec_hash: str,
    theta_true_hash: str,
    loss_kind: str,
) -> dict[str, Any]:
    """Build a structured FIM-cache payload to hash into a cache key."""
    return {
        "infer_keys": infer_keys,
        "model_config_id": system_label,
        "cfg_hash": cfg_hash,
        "forward_spec_hash": forward_spec_hash,
        "theta_true_hash": theta_true_hash,
        "loss_kind": loss_kind,
    }


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


def _resolve_run_spec(mc_cfg: dict[str, Any], row: dict[str, Any], index: int) -> dict[str, Any]:
    """Resolve a run spec from a plan row using default run indexing.

    This is a compatibility wrapper used for older call sites; in this script
    `_resolve_run_spec_with_id` is used directly to accommodate disabled rows.
    It is reusable only for the prescription schema defined here.
    """
    return _resolve_run_spec_with_id(mc_cfg, row, index=index, run_id_index=index)


def _resolve_run_spec_with_id(
    mc_cfg: dict[str, Any],
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
    defaults = copy.deepcopy(mc_cfg.get("defaults", {}))
    resolved = copy.deepcopy(defaults)

    run_id_prefix = mc_cfg.get("run_id_prefix", "run")

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

    resolved.setdefault("init", {})
    resolved.setdefault("noise", {})
    resolved.setdefault("eigen", {})
    resolved.setdefault("truth", {})
    resolved.setdefault("optimizer", {})
    resolved.setdefault("outputs", {})

    base_seed = mc_cfg.get("defaults", {}).get("seed")
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


def _get_prescribed_mc_cfg(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    """Fetch and validate the prescribed_mc/monte_carlo block."""
    mc_cfg = experiment_cfg.get("prescribed_mc") or experiment_cfg.get("monte_carlo")
    if mc_cfg is None:
        raise ValueError(
            "Experiment config must include an 'experiment.monte_carlo' (or legacy experiment.prescribed_mc) block."
        )
    if not isinstance(mc_cfg, dict):
        raise ValueError("experiment.monte_carlo must be a mapping/dict.")
    return mc_cfg


def _resolve_loss_kind(run_spec: dict[str, Any], mc_defaults: dict[str, Any]) -> str:
    """Resolve optimizer.loss with run-spec override and experiment default."""
    loss_kind = (
        _get_nested(run_spec, ["optimizer", "loss"])
        or _get_nested(mc_defaults, ["optimizer", "loss"])
        or "nll"
    )
    if loss_kind not in {"nll", "map"}:
        raise ValueError(
            f"Unsupported optimizer.loss={loss_kind!r}; expected 'nll' or 'map'."
        )
    return loss_kind


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
        "optimizer.kind",
        "optimizer.loss",
        "optimizer.kwargs",
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
        elif key == "optimizer.kind":
            value = _get_nested(spec, ["optimizer", "kind"])
        elif key == "optimizer.loss":
            value = _get_nested(spec, ["optimizer", "loss"])
        elif key == "optimizer.kwargs":
            kwargs_val = _get_nested(spec, ["optimizer", "kwargs"])
            if kwargs_val:
                try:
                    value = json.dumps(kwargs_val, sort_keys=True)
                except Exception:
                    value = str(kwargs_val)
            else:
                value = ""
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


def _resolve_outdir(
    *,
    cli_outdir: str | None,
    run_name: str | None,
    experiment_cfg: dict[str, Any],
    prescription_path: Path,
) -> tuple[Path, str]:
    """Resolve the experiment output directory with CLI/config precedence.

    Precedence:
    1) --outdir
    2) experiment.outputs.outdir (relative to prescription file when relative)
    3) --run-name
    4) Results/prescribed_mc_<timestamp>
    """
    if cli_outdir and cli_outdir.strip():
        return Path(cli_outdir), "CLI --outdir"

    outputs_cfg = experiment_cfg.get("outputs", {})
    if outputs_cfg is None:
        outputs_cfg = {}
    if not isinstance(outputs_cfg, dict):
        raise ValueError("experiment.outputs must be a mapping/dict when provided.")

    cfg_outdir = _resolve_path_relative_to_prescription(
        outputs_cfg.get("outdir"),
        prescription_path=prescription_path,
        field_name="experiment.outputs.outdir",
    )
    if cfg_outdir is not None:
        return cfg_outdir, "experiment.outputs.outdir"

    if run_name:
        return Path("Results") / run_name, "CLI --run-name"
    return Path("Results") / f"prescribed_mc_{_timestamp_tag()}", "auto timestamp"


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
            flattened[_migrate_param_key(prefix)] = value

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


def _resolve_config_id(config_id: str | None) -> str:
    """Map a legacy config_id to a system preset name (legacy compatibility)."""
    if not config_id:
        raise ValueError("Prescription must include model.config_id.")
    mapping = {
        "SHERA_TESTBED_CONFIG": "SHERA_TESTBED_3P",
        "SHERA_FLIGHT_CONFIG": "SHERA_FLIGHT_3P",
        "shera_testbed": "SHERA_TESTBED_3P",
        "shera_flight": "SHERA_FLIGHT_3P",
    }
    if config_id in mapping:
        return mapping[config_id]
    raise ValueError(f"Unknown config_id '{config_id}'.")


def _apply_config_overrides(
    system_cfg: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Apply basic overrides onto a resolved system config mapping."""
    if not overrides:
        return system_cfg

    translated: dict[str, Any] = {}
    for key, value in overrides.items():
        migrated = _migrate_param_key(key)
        # Map common structural fields onto optics/source blocks.
        if migrated in {
            "optics.pupil_npix",
            "optics.psf_npix",
            "optics.oversample",
            "optics.primary_noll_indices",
            "optics.secondary_noll_indices",
            "optics.dp_path",
            "optics.dp_design_wavelength_m",
            "optics.pixel_pitch_m",
            "optics.m1_diameter_m",
            "optics.m2_diameter_m",
        }:
            translated.setdefault("optics", {})[migrated.split(".", 1)[1]] = value
        elif migrated in {
            "source.wavelength_m",
            "source.bandwidth_m",
            "source.n_lambda",
        }:
            translated.setdefault("source", {})[migrated.split(".", 1)[1]] = value
        else:
            translated.setdefault("system", {})[migrated] = value

    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        out = copy.deepcopy(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
        return out

    return _deep_merge(system_cfg, translated)


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


def _config_payload(cfg: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Serialize a resolved system config mapping and normalize path fields."""
    payload = copy.deepcopy(cfg)
    dp_path = None
    if isinstance(payload, dict):
        dp_path = (
            payload.get("optics", {}).get("dp_path")
            or payload.get("optics", {}).get("diffractive_pupil_path")
        )
    if dp_path is not None:
        payload.setdefault("optics", {})
        payload["optics"]["dp_path"] = _repo_relative_path(dp_path, repo_root=repo_root)
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
        optimizer_kind = optimizer_meta.get("kind")
        optimizer_kwargs = optimizer_meta.get("kwargs")
        optimizer_loss = optimizer_meta.get("loss")

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
            "optimizer.kind": optimizer_kind,
            "optimizer.kwargs": (
                json.dumps(optimizer_kwargs, sort_keys=True)
                if isinstance(optimizer_kwargs, dict) and optimizer_kwargs
                else None
            ),
            "optimizer.loss": optimizer_loss,
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


def _transpose_results_rows(
    rows: list[dict[str, Any]],
    metric_columns: list[str],
    run_ids: list[str],
) -> list[dict[str, Any]]:
    """Transpose run-major rows into metric-major rows for aggregate CSV output.

    Each output row is keyed by metric name and has one column per run ID.
    Missing values are left as empty strings, matching existing CSV behavior.
    """
    rows_by_run_id = {
        row.get("run_id"): row
        for row in rows
        if isinstance(row.get("run_id"), str)
    }
    transposed_rows: list[dict[str, Any]] = []
    for metric_key in metric_columns:
        metric_row: dict[str, Any] = {"key": metric_key}
        for run_id in run_ids:
            source_row = rows_by_run_id.get(run_id)
            metric_row[run_id] = (
                source_row.get(metric_key, "") if source_row is not None else ""
            )
        transposed_rows.append(metric_row)
    return transposed_rows


def _write_results_csv(
    out_path: Path,
    run_entries: list[dict[str, Any]],
    infer_keys: tuple[str, ...],
    *,
    results_orientation: str,
) -> list[str]:
    """Write the aggregate results.csv file for all runs.

    Invoked by `_write_experiment_outputs` after runs complete or during
    aggregation-only mode. Preferred schema is column-oriented with a leading
    `key` column and one column per `run_id`. Pass `results_orientation="row"`
    for compatibility with run-major rows where `run_id` is a data column.
    This is a workflow-specific writer and not intended for reuse outside this
    experiment layout.
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
        "optimizer.kind",
        "optimizer.kwargs",
        "optimizer.loss",
        "eigen.use_eigen",
        "eigen.whiten_basis",
        "eigen.truncate_k",
        "eigen.truncate_by_eigval",
        "noise.add_noise",
    ]
    metric_columns = base_columns + param_columns
    if results_orientation == "col":
        run_ids = [entry["run_id"] for entry in run_entries]
        rows_to_write = _transpose_results_rows(rows, metric_columns, run_ids)
        columns = ["key", *run_ids]
    else:
        rows_to_write = rows
        columns = metric_columns
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows_to_write:
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
    experiment_cfg: dict[str, Any],
    mc_cfg: dict[str, Any],
    prescription_path: Path,
    plan_path: Path | None,
    run_entries: list[dict[str, Any]],
    infer_keys: tuple[str, ...],
    repo_root: Path,
    results_orientation: str,
    system_label: str | None,
    config_override_keys: list[str],
    truth_default_keys: list[str],
) -> None:
    """Write aggregated outputs (results.csv and manifest.json) for the run set.

    Called in `main` after execution or in aggregate-only mode to produce
    experiment-level artifacts. This is specific to the prescribed Monte Carlo
    workflow and not intended as a generic utility. `results.csv` can be written
    in column orientation (`key` + `run_id` columns; preferred default) or row
    orientation (`run_id` as a field; compatibility mode).
    """
    results_filename = mc_cfg.get("results_filename") or "results.csv"
    results_path = outdir / results_filename
    _write_results_csv(
        results_path,
        run_entries,
        infer_keys,
        results_orientation=results_orientation,
    )

    manifest_path = outdir / "manifest.json"
    manifest_runs = _build_manifest_runs(run_entries)
    experiment_notes = _first_present_string(experiment_cfg, EXPERIMENT_NOTE_KEYS)

    _write_manifest(
        manifest_path,
        created_at=_now_iso_local_ms(),
        script=_repo_relative_path(Path(__file__), repo_root=repo_root)
        or "examples/recipes/prescribed_monte_carlo.py",
        prescription_path=_repo_relative_path(prescription_path, repo_root=repo_root),
        plan_path=_repo_relative_path(plan_path, repo_root=repo_root),
        config_id=system_label,
        notes=experiment_notes,
        overrides_config_keys=config_override_keys,
        overrides_store_keys=truth_default_keys,
        runs=manifest_runs,
        artifacts=[
            {"path": "manifest.json"},
            {"path": results_filename, "orientation": results_orientation},
        ],
    )


def main() -> None:
    """Run the prescribed Monte Carlo experiment pipeline.

    Args:
        --prescription: Path to the YAML/JSON experiment config. The file must
            contain native `system` and `experiment` blocks (with
            `experiment.monte_carlo` for Monte Carlo settings; legacy
            `experiment.prescribed_mc` is still accepted). Experiment notes live
            in `experiment.notes` (aliases note/comment/comments).
        --outdir: Root output directory for the experiment. When supplied, this
            overrides `experiment.outputs.outdir`.
        --run-name: Optional name segment used to construct Results/<run-name>
            when neither --outdir nor experiment.outputs.outdir is set. If all
            of those are omitted, a timestamp-based directory name is used.
        --dry-run: Resolve run specs and print previews without executing the
            optimization runs or writing run artifacts.
        --aggregate-only: Skip execution and only aggregate manifest.json and
            results.csv from existing run artifacts inside the resolved outdir.
        --results-orientation: Output schema for results.csv. `col` (default)
            writes a leading `key` column plus one column per run_id. `row`
            writes one row per run with `run_id` as a data column.
        --num-preview: Limit the number of resolved run specs shown in preview
            output (useful with large plans).
    """
    parser = argparse.ArgumentParser(description="Prescribed Monte Carlo scaffold")
    parser.add_argument(
        "--prescription",
        type=Path,
        default=None,
        help="Path to prescription YAML/JSON (defaults to template if omitted)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Experiment root directory; overrides experiment.outputs.outdir.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Convenience name for Results/<run-name> when outdir is not otherwise set.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        default=False,
        help="Generate manifest + results.csv from existing runs without executing.",
    )
    parser.add_argument(
        "--results-orientation",
        choices=("row", "col"),
        default="col",
        help=(
            "results.csv schema: 'col' writes key + run_id columns (default), "
            "'row' writes one row per run for compatibility. CLI overrides "
            "prescription experiment.monte_carlo.results_orientation when provided."
        ),
    )
    parser.add_argument("--num-preview", type=int, default=None)

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    # Local toggle for writing per-run metric artifacts (metric.npz).
    # Default off to avoid extra artifact I/O unless explicitly needed.
    output_metric = False

    outdir_hint = Path(args.outdir) if args.outdir else None
    prescription_path = _resolve_prescription(args, outdir_hint)
    resolved_prescription = _repo_relative_path(prescription_path, repo_root=repo_root)
    print(f"Resolved prescription path: {resolved_prescription or prescription_path}")

    raw_prescription = _load_prescription(prescription_path)
    if _is_legacy_prescription(raw_prescription):
        user_cfg, compat_meta = _upgrade_legacy_prescription(raw_prescription)
    else:
        user_cfg = _strip_private_keys(
            load_user_config(
                config_path=prescription_path,
                system_preset=None,
                experiment_preset=None,
            )
        )
        compat_meta = {"config_overrides": {}, "legacy_schema": False}

    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    experiment_cfg = resolved_cfg.get("experiment")

    if system_cfg is None:
        raise ValueError("Prescribed Monte Carlo requires a top-level 'system' block.")
    if experiment_cfg is None:
        raise ValueError("Prescribed Monte Carlo requires a top-level 'experiment' block.")

    mc_cfg_raw = _get_prescribed_mc_cfg(experiment_cfg)
    mc_cfg, mc_defaults = _mc_defaults_from_experiment(experiment_cfg, mc_cfg_raw)
    if not isinstance(mc_defaults, dict):
        raise ValueError("experiment.monte_carlo.defaults must be a mapping/dict.")

    def _normalize_orientation(value: str | None) -> str:
        if value is None:
            return "col"
        if value not in {"col", "row"}:
            raise ValueError("experiment.monte_carlo.results_orientation must be 'col' or 'row'")
        return value

    presc_orientation = _normalize_orientation(mc_cfg.get("results_orientation"))
    results_orientation = _normalize_orientation(
        args.results_orientation if args.results_orientation is not None else presc_orientation
    )

    # Resolve plan CSV: prescription config controls plan usage; explicit null/omission disables plans.
    plan_cfg_provided = "run_plan" in mc_cfg or "plan_csv" in mc_cfg
    plan_csv_cfg = mc_cfg.get("run_plan")
    if plan_csv_cfg is None:
        plan_csv_cfg = mc_cfg.get("plan_csv")
    plan_path = _resolve_plan_csv_path(plan_csv_cfg, prescription_path=prescription_path)

    if plan_csv_cfg is None:
        plan_rows = []
        reason = "explicitly disabled in prescription" if plan_cfg_provided else "no run plan specified"
        print(f"Resolved run plan path: None ({reason})")
    else:
        resolved_plan = _repo_relative_path(plan_path, repo_root=repo_root) if plan_path else None
        print(f"Resolved run plan path: {resolved_plan or plan_path}")
        plan_rows = _load_plan_csv(plan_path) if plan_path is not None else []

    n_runs = mc_cfg.get("n_runs")
    # experiment.n_runs is authoritative: it pads or truncates the plan so
    # previews and run execution operate on the resolved run count.
    plan_rows, run_policy = _apply_experiment_n_runs(plan_rows, n_runs)
    enabled_count = sum(1 for row in plan_rows if _row_enabled(row))
    disabled_count = len(plan_rows) - enabled_count
    if run_policy["n_runs"] is None:
        print(
            "Run count policy: experiment.monte_carlo.n_runs not set; "
            f"plan defines {run_policy['plan_runs']} run(s)."
        )
    else:
        print(
            "Run count policy: experiment.monte_carlo.n_runs="
            f"{run_policy['n_runs']} (plan defines {run_policy['plan_runs']}) "
            f"-> resolved {run_policy['resolved_runs']} run(s)."
        )
        if run_policy["padded_runs"]:
            print(
                "  Added "
                f"{run_policy['padded_runs']} default-only run(s) to match experiment.monte_carlo.n_runs."
            )
        if run_policy["truncated_runs"]:
            print(
                "WARNING: experiment.monte_carlo.n_runs is authoritative; truncated "
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
            if key == "system"
            or key.startswith("system.")
            or key == "experiment"
            or key.startswith("experiment.")
        ]
        if forbidden:
            raise ValueError(
                "Plan rows cannot override system/experiment settings; remove: "
                + ", ".join(sorted(forbidden))
            )

    config_overrides = compat_meta.get("config_overrides", {}) or {}
    config_override_keys = (
        ", ".join(sorted(config_overrides.keys())) if config_overrides else "none"
    )
    truth_defaults = mc_defaults.get("truth", {}) or {}
    truth_default_keys = sorted(_flatten_store_overrides(truth_defaults).keys())

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
                mc_cfg,
                row,
                index=index + 1,
                run_id_index=run_id_index,
            )
        else:
            # Disabled rows keep the resolved fields but do not receive a run_id.
            resolved = _resolve_run_spec_with_id(
                mc_cfg,
                row,
                index=index + 1,
                run_id_index=None,
            )
        run_specs.append(resolved)

    outdir, outdir_source = _resolve_outdir(
        cli_outdir=args.outdir,
        run_name=args.run_name,
        experiment_cfg=experiment_cfg,
        prescription_path=prescription_path,
    )
    print(f"Resolved outdir: {outdir} ({outdir_source})")
    system_label = _get_nested(user_cfg, ["system", "preset"]) or "custom"
    print(f"System preset: {system_label}")
    print(f"Structural config overrides: {config_override_keys}")
    if truth_default_keys:
        print("Truth defaults provided for keys: " + ", ".join(truth_default_keys))
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

    base_seed = _require_experiment_seed(experiment_cfg)

    system_cfg = _apply_config_overrides(system_cfg, config_overrides)

    inference_system_cfg_base = experiment_cfg.get("inference_system")
    if inference_system_cfg_base is not None:
        inference_system_cfg_base = resolve_system_config(inference_system_cfg_base)
    else:
        inference_system_cfg_base = copy.deepcopy(system_cfg)

    system_cfg, data_detector_ke_meta = _seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=base_seed,
        token_prefix="data.detector",
    )
    if _detector_ke_has_per_run_realization(inference_system_cfg_base):
        print(
            "Inference detector knowledge_error includes realization_policy=per_run; "
            "inference-side detector realization will be resampled per run seed."
        )

    forward_spec_data = compose_forward_spec(system_cfg)
    forward_spec_infer = compose_forward_spec(inference_system_cfg_base)

    infer_keys_raw = tuple(experiment_cfg.get("infer_keys", []))
    infer_keys = tuple(_migrate_param_key(key) for key in infer_keys_raw)
    if not infer_keys:
        raise ValueError("Prescription must include non-empty infer_keys.")
    missing_infer_keys = [key for key in infer_keys if key not in forward_spec_infer]
    if missing_infer_keys:
        print(
            "WARNING: dropping infer_keys not present in forward_spec: "
            + ", ".join(missing_infer_keys)
        )
    infer_keys = tuple(key for key in infer_keys if key in forward_spec_infer)
    if not infer_keys:
        raise ValueError("No valid infer_keys remain after migration/filtering.")
    # Store overrides and derived refresh steps: apply nested overrides via
    # dotted keys, then recompute any derived parameters to keep the store
    # self-consistent for forward/inference use.
    base_store_data = ParameterStore.from_spec_defaults(forward_spec_data)
    if truth_defaults:
        base_store_data = base_store_data.replace(_flatten_store_overrides(truth_defaults))
    base_store_data = base_store_data.refresh_derived(forward_spec_data)

    base_store_infer = ParameterStore.from_spec_defaults(forward_spec_infer)
    if truth_defaults:
        base_store_infer = base_store_infer.replace(_flatten_store_overrides(truth_defaults))
    base_store_infer = base_store_infer.refresh_derived(forward_spec_infer)

    prior_info_raw = experiment_cfg.get("priors", {})
    prior_info = _migrate_key_mapping(prior_info_raw)
    prior_spec = PriorSpec.from_info(base_store_infer, prior_info)
    prior_spec_cache: dict[str, PriorSpec] = {}

    fim_cache: dict[str, dict[str, Any]] = {}
    fim_cache_last: dict[str, Any] | None = None

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

    run_id_prefix = mc_cfg.get("run_id_prefix") or "run"
    run_counter = 0

    if args.aggregate_only:
        run_entries = _collect_run_entries(runs_dir, plan_rows, run_specs, plan_labels)
        _write_experiment_outputs(
            outdir=outdir,
            experiment_cfg=experiment_cfg,
            mc_cfg=mc_cfg,
            prescription_path=prescription_path,
            plan_path=plan_path,
            run_entries=run_entries,
            infer_keys=infer_keys,
            repo_root=repo_root,
            results_orientation=results_orientation,
            system_label=system_label,
            config_override_keys=sorted(config_overrides.keys()),
            truth_default_keys=truth_default_keys,
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

        inference_system_cfg_run, inference_detector_ke_meta = _seed_detector_knowledge_errors_with_policy(
            inference_system_cfg_base,
            experiment_seed=base_seed,
            token_prefix="inference.detector",
            run_seed=seed,
        )
        forward_spec_infer_run = compose_forward_spec(inference_system_cfg_run)
        if tuple(forward_spec_infer_run.keys()) != tuple(forward_spec_infer.keys()):
            raise ValueError(
                "Run-specific inference forward spec keys differ from experiment-level keys. "
                "Per-run detector knowledge-error realization currently requires invariant model structure."
            )
        inference_subspec = forward_spec_infer_run.subset(infer_keys)
        cfg_hash = _stable_hash_payload(inference_system_cfg_run)
        forward_spec_hash = _stable_hash_payload(forward_spec_infer_run)

        print(f"Generating synthetic data...")
        # Synthetic data generation + noise handling: build the "truth" store,
        # refresh derived values, then forward-model data and inject noise.
        truth_overrides = _flatten_store_overrides(run_spec.get("truth", {}))
        truth_store_data = base_store_data.replace(truth_overrides)
        truth_store_data = truth_store_data.refresh_derived(forward_spec_data)

        truth_store_infer = base_store_infer.replace(truth_overrides)
        aligned_truth = {}
        for key in forward_spec_infer_run.keys():
            try:
                aligned_truth[key] = truth_store_data.get(key)
            except KeyError:
                continue
        if aligned_truth:
            truth_store_infer = truth_store_infer.replace(aligned_truth)
        truth_store_infer = truth_store_infer.refresh_derived(forward_spec_infer_run)

        binder_data = SheraBinder(system_cfg, forward_spec_data, truth_store_data)
        binder_infer = SheraBinder(inference_system_cfg_run, forward_spec_infer_run, truth_store_infer)

        # Generate the synthetic data
        data_psf = binder_data.model()

        # Resolve noise configuration and apply through shared helper
        noise_cfg_run = copy.deepcopy(mc_defaults.get("noise", {}))
        noise_cfg_run = _deep_update(noise_cfg_run, run_spec.get("noise", {}) or {})
        add_noise_value = noise_cfg_run.get("enabled", noise_cfg_run.get("add_noise"))
        add_noise = bool(add_noise_value) if add_noise_value is not None else False
        outputs_cfg = run_spec.get("outputs", {}) or {}
        plots_flag = outputs_cfg.get("plots")
        if plots_flag is None:
            plots_flag = _get_nested(mc_defaults, ["outputs", "plots"])
        save_plots = bool(plots_flag) if plots_flag is not None else False

        data, data_var = apply_observation_noise(
            data_psf,
            noise_cfg=noise_cfg_run,
            rng_key=noise_key,
            detector_spec=getattr(binder_data.detector, "spec", None),
            exposure_time_s=truth_store_data.get("source.exposure_time_s", default=None),
        )

        noise_model = "gaussian"
        reduce = "sum"
        nll_loss_fn, _ = make_binder_nll_fn(
            binder=binder_infer,
            infer_keys=infer_keys,
            data=data,
            var=data_var,
            noise_model=noise_model,
            reduce=reduce,
            theta0_store=truth_store_infer,
        )
        fim_labels = generate_fim_labels(
            infer_keys,
            cfg=inference_system_cfg_run,
            store=truth_store_infer,
        )
        loss_fn = nll_loss_fn

        # FIM computation and eigen preconditioning flow: pack_params converts
        # a structured ParameterStore into a flat vector for optimization/FIM.
        theta_true = pack_params(inference_subspec, truth_store_infer)
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
            reuse_fim_value = _get_nested(mc_defaults, ["fim", "reuse_fim"])
        reuse_fim = bool(reuse_fim_value) if reuse_fim_value is not None else False

        eigen_cfg = run_spec.get("eigen", {})
        use_eigen_value = eigen_cfg.get("use_eigen")
        if use_eigen_value is None:
            use_eigen_value = _get_nested(mc_defaults, ["eigen", "use_eigen"])
        use_eigen = bool(use_eigen_value)
        whiten_basis_value = eigen_cfg.get("whiten_basis")
        if whiten_basis_value is None:
            whiten_basis_value = _get_nested(
                mc_defaults, ["eigen", "whiten_basis"]
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
            mc_defaults, ["init", "mode"]
        )
        init_overrides = {
            key: value for key, value in init_cfg.items() if key != "mode"
        }
        init_overrides_flat = _flatten_store_overrides(init_overrides)
        prior_overrides = _migrate_key_mapping(run_spec.get("prior_overrides") or {})

        if init_mode == "prior":
            # Per-run prior overrides only affect prior sampling and do not
            # change init.mode semantics.
            run_prior_spec = prior_spec
            if prior_overrides:
                run_prior_info, applied_keys = _apply_prior_overrides(
                    prior_info,
                    prior_overrides,
                    infer_keys=infer_keys,
                    base_store=base_store_infer,
                )
                if applied_keys:
                    cache_key = _stable_hash_payload(run_prior_info)
                    run_prior_spec = prior_spec_cache.get(cache_key)
                    if run_prior_spec is None:
                        run_prior_spec = PriorSpec.from_info(base_store_infer, run_prior_info)
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
                center_store=truth_store_infer,
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
            init_store = truth_store_infer
            if init_overrides_flat:
                (
                    init_primitive_overrides,
                    init_derived_overrides,
                    init_unknown_overrides,
                ) = _partition_overrides_by_kind(init_overrides_flat, forward_spec_infer_run)
                if init_unknown_overrides:
                    print(
                        "WARNING: explicit init overrides include keys that are not "
                        "declared as primitive/derived in forward_spec; applying "
                        "them as direct overrides: "
                        + ", ".join(sorted(init_unknown_overrides))
                    )

                if init_primitive_overrides:
                    init_store = init_store.replace(init_primitive_overrides)
                    init_store = init_store.refresh_derived(forward_spec_infer_run)

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
                init_store = init_store.refresh_derived(forward_spec_infer_run)
            if prior_overrides:
                print(
                    f"Note: prior overrides provided for run_id={run_id} but "
                    f"init.mode={init_mode}; ignoring prior overrides."
                )
        else:
            raise ValueError(f"Unknown init.mode '{init_mode}'")
        if init_mode == "prior":
            init_store = _refresh_preserving_derived_infer_keys(
                init_store,
                infer_keys=infer_keys,
                spec=forward_spec_infer_run,
            )

        _, theta0 = make_binder_nll_fn(
            binder=binder_infer,
            infer_keys=infer_keys,
            data=data,
            var=data_var,
            noise_model="gaussian",
            reduce="sum",
            theta0_store=init_store,
        )

        def map_loss_fn(theta: np.ndarray) -> np.ndarray:
            store_theta = store_unpack_params(inference_subspec, theta, init_store)
            nll_loss = nll_loss_fn(theta)
            prior_gaussian_loss = run_prior_spec.quadratic_penalty(
                store_theta,
                center_store=init_store, # TODO: truth_store or init_store?
                keys=infer_keys,
            )
            return nll_loss + prior_gaussian_loss

        loss_kind = _resolve_loss_kind(run_spec, mc_defaults)
        if loss_kind == "map":
            loss_fn = map_loss_fn
        elif loss_kind == "nll":
            loss_fn = nll_loss_fn

        # Pack truth store for FIM/diagnostics and evaluate baseline loss.
        theta_true = pack_params(inference_subspec, truth_store_infer)
        loss_true = float(loss_fn(theta_true))

        fim_point = theta_true
        # FIM cache key includes loss_kind so MAP and NLL cache separately.
        cache_key_payload = _build_fim_cache_key_payload(
            infer_keys=infer_keys,
            system_label=system_label,
            cfg_hash=cfg_hash,
            forward_spec_hash=forward_spec_hash,
            theta_true_hash=_hash_array_bytes(theta_true),
            loss_kind=loss_kind,
        )
        fim_cache_key = _stable_hash_payload(cache_key_payload)[:12]
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
            F = fim_theta(loss_fn, fim_point)
            fim_diag = jnp.diag(F)
            fim_cache_hit = False
            fim_cache[cache_key] = {"F": F, "fim_diag": fim_diag}
            fim_cache_last = fim_cache[cache_key]
            print("FIM computed and cached for later.")

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
                curvature_vec = np.maximum(eigvals_kept, 1e-8)
                lr_vec = 1.0 / (curvature_vec + 1e-12)

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
            curvature_vec = np.maximum(np.asarray(fim_diag), 1e-8)
            lr_vec = 1.0 / (curvature_vec + 1e-12)
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
        optimizer_cfg = run_spec.get("optimizer", {})
        n_iter_value = optimizer_cfg.get("n_iter")
        if n_iter_value is None:
            raise ValueError(f"Run {run_id} resolved to a null optimizer.n_iter.")
        base_lr_value = optimizer_cfg.get("base_lr")
        if base_lr_value is None:
            raise ValueError(f"Run {run_id} resolved to a null optimizer.base_lr.")
        opt_kind = optimizer_cfg.get("kind", "sgd")
        optimizer_kwargs = optimizer_cfg.get("kwargs", {})
        n_iter = int(n_iter_value)
        base_lr = float(base_lr_value)

        run_diagnosis = False
        if run_diagnosis:
            diag = diagnose_first_step(
                loss_fn=loss_opt,
                theta0=theta0_opt,
                learning_rate=base_lr,
                lr_vec=lr_vec if use_eigen else None,
                optimizer_kind=opt_kind,
                index_map=index_map if not use_eigen else None,
                verbose=True,
            )
            print("First-step diagnostic:")
            print(
                f"  loss0={diag['loss0']:.6g} finite={diag['loss0_finite']} | "
                f"grad_finite={diag['grad0_finite']} | theta1_finite={diag['theta1_finite']} | "
                f"loss1={diag['loss1']:.6g} finite={diag['loss1_finite']}"
            )
            print(
                f"  grad0 min/max={diag['grad0_min']:.3e}/{diag['grad0_max']:.3e} | "
                f"delta min/max={diag['delta_min']:.3e}/{diag['delta_max']:.3e}"
            )
            if lr_vec is not None:
                print(
                    f"  lr_vec min/max={diag['lr_vec_min']:.3e}/{diag['lr_vec_max']:.3e}"
                )
            if diag.get("top_grad"):
                topg = ", ".join(f"{i}:{v:.2e}" for i, v in diag["top_grad"])
                print(f"  top |grad|: {topg}")
            if diag.get("top_delta"):
                topl = ", ".join(f"{i}:{v:.2e}" for i, v in diag["top_delta"])
                print(f"  top |delta|: {topl}")

        labels_by_key = map_labels_to_keys(
            infer_keys,
            generate_fim_labels(
                infer_keys,
                cfg=inference_system_cfg_run,
                store=init_store if use_eigen else truth_store_infer,
            ),
            store=init_store if use_eigen else None,
            index_map=None if use_eigen else index_map,
        )

        loss_kind = _resolve_loss_kind(run_spec, mc_defaults)

        # Loss function setup and optimization execution: run_shera_gd consumes
        # the chosen theta_space, preconditioner, and metadata for artifacts.
        config_payload = _config_payload(system_cfg, repo_root=repo_root)
        theta_final_opt, trace, artifacts = run_shera_gd(
            loss_fn=loss_opt,
            theta0=theta0_opt,
            index_map=index_map,
            learning_rate=base_lr,
            lr_vec=lr_vec,
            num_steps=n_iter,
            optimizer_kind=opt_kind,
            optimizer_kwargs=optimizer_kwargs,
            runs_dir=runs_dir,
            run_id=run_id,
            return_artifacts=True,
            theta_space=theta_space,
            metric=metric_payload,
            extra_meta={
                "optimizer": {
                    "preconditioning": precond_meta,
                    "kind": opt_kind,
                    "kwargs": optimizer_kwargs,
                    "loss": loss_kind,
                },
                "theta": {"labels_by_key": labels_by_key},
                "model": {
                    "config_id": system_label,
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
                    "inference_cfg_hash": cfg_hash,
                    "inference_forward_spec_hash": forward_spec_hash,
                    "detector_ke_realization_mode": (
                        "per_run"
                        if inference_detector_ke_meta.get("has_per_run_realization")
                        else "fixed_per_experiment"
                    ),
                },
                "detector_knowledge_error": {
                    "data": data_detector_ke_meta,
                    "inference": inference_detector_ke_meta,
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
        init_psf = binder_infer.model(
            binder_infer.strip_structural(init_store)
        )
        final_psf = binder_infer.model(
            binder_infer.strip_structural(final_store)
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
                truth_dict = {key: truth_store_data.get(key) for key in infer_keys}
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

                if save_plots:
                    plots_dir = run_dir / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    fim_labels = generate_fim_labels(
                        infer_keys,
                        cfg=inference_system_cfg_run,
                        store=init_store if use_eigen else truth_store_infer,
                    )
                    plot_fim(
                        F,
                        fim_labels,
                        save_path=plots_dir / "fim.png",
                        vmin=4,
                        vmax=14,
                        show=False,
                    )

                    if use_eigen and eigen_map is not None:
                        eigvals, eigvecs = np.linalg.eigh(np.asarray(F))
                        sort_idx = np.argsort(eigvals)[::-1]
                        eigvals = eigvals[sort_idx]
                        eigvecs = eigvecs[:, sort_idx]
                        spectrum_truncate_k = None
                        if truncate_k is not None:
                            spectrum_truncate_k = int(truncate_k)
                        elif truncate_by_eigval is not None:
                            spectrum_truncate_k = int(np.sum(eigvals >= truncate_by_eigval))
                            if spectrum_truncate_k <= 0:
                                spectrum_truncate_k = 1
                        plot_eigenvalue_spectrum(
                            eigvals,
                            eigvecs,
                            labels=fim_labels,
                            truncate_k=spectrum_truncate_k,
                            label_boxes=False,
                            save_path=plots_dir / "eigenvalue_spectrum.png",
                            show=False,
                        )

                    psf_extent_as = (
                        binder_data.optics.psf_npixels * binder_data.optics.psf_pixel_scale / 2
                        * np.array([-1, 1, -1, 1])
                    )

                    plot_psf_comparison(
                        data=data,
                        model=init_psf,
                        var=data_var,
                        extent=psf_extent_as,
                        model_label="Initial Model",
                        save_path=plots_dir / "initial_psf_comparison.png",
                        show=False,
                    )

                    plot_psf_comparison(
                        data=data,
                        model=final_psf,
                        var=data_var,
                        extent=psf_extent_as,
                        model_label="Final Model",
                        save_path=plots_dir / "final_psf_comparison.png",
                        show=False,
                    )

                    pixel_offsets_data = _get_pixel_offset_maps(binder_data)
                    pixel_offsets_infer = _get_pixel_offset_maps(binder_infer)
                    if pixel_offsets_data is not None and pixel_offsets_infer is not None:
                        data_dx, data_dy = pixel_offsets_data
                        infer_dx, infer_dy = pixel_offsets_infer
                        if data_dx.shape == infer_dx.shape and data_dy.shape == infer_dy.shape:
                            plot_pixel_offset_maps(
                                data_dx,
                                data_dy,
                                infer_dx,
                                infer_dy,
                                cmap="viridis_nan",
                                save_path=plots_dir / "pixel_offset_maps.png",
                                show=False,
                            )
                        else:
                            print("Skipping pixel_offset_maps.png: data/inference map shapes differ.")
                    else:
                        print("Skipping pixel_offset_maps.png: pixel_offsets layer missing.")

                    prf_data = _get_pixel_response_map(binder_data)
                    prf_infer = _get_pixel_response_map(binder_infer)
                    if prf_data is not None and prf_infer is not None:
                        if prf_data.shape == prf_infer.shape:
                            plot_pixel_response_maps(
                                prf_data,
                                prf_infer,
                                cmap="viridis_nan",
                                save_path=plots_dir / "pixel_response_maps.png",
                                show=False,
                            )
                        else:
                            print("Skipping pixel_response_maps.png: data/inference prf shapes differ.")
                    else:
                        print("Skipping pixel_response_maps.png: pixel_response layer missing.")

                    trace_for_signals = _trace_with_initial_point(
                        trace,
                        theta0=theta0_opt,
                        loss0=loss_init,
                    )
                    losses = np.asarray(trace_for_signals["loss"])
                    iterations = np.arange(losses.shape[0])
                    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
                    axes = axes.flatten()
                    plot_parameter_history(
                        names=("Loss",),
                        histories=(losses,),
                        true_vals=(float(loss_true),),
                        ax=axes[0],
                        title="Optimization Loss History",
                        show=False,
                        close=False,
                    )
                    window = min(10, losses.shape[0])
                    start = losses.shape[0] - window
                    axes[1].plot(iterations[start:], losses[start:])
                    axes[1].set_title(f"Last {window} Iterations, Final= {losses[-1]:.3f}")
                    axes[1].set_xlabel("Iteration")
                    axes[1].set_ylabel("Loss")
                    axes[1].axhline(loss_true, linestyle="--", color="k", alpha=0.6)
                    final_delta = np.abs(losses[-1] - loss_true)
                    if final_delta != 0:
                        axes[1].set_ylim(loss_true - 3 * final_delta, loss_true + 3 * final_delta)
                    fig.tight_layout()
                    fig.savefig(plots_dir / "loss_history.png", dpi=300)
                    plt.close(fig)

                    if use_eigen and eigen_map is not None:
                        decoder = lambda z: _refresh_preserving_derived_infer_keys(
                            store_unpack_params(
                                inference_subspec,
                                eigen_map.theta_from_z(z),
                                init_store,
                            ),
                            infer_keys=infer_keys,
                            spec=forward_spec_infer_run,
                        )
                    else:
                        decoder = lambda theta: _refresh_preserving_derived_infer_keys(
                            store_unpack_params(
                                inference_subspec,
                                theta,
                                init_store,
                            ),
                            infer_keys=infer_keys,
                            spec=forward_spec_infer_run,
                        )

                    signals = build_signals(
                        trace_for_signals,
                        meta={},
                        decoder=decoder,
                        truth=truth_store_data,
                        signal_set="intro",
                    )
                    plot_signals_grid(
                        signals,
                        plots_dir,
                        include_zernike_rms=False,
                        show_final_values=True,
                        figsize=(15, 10),
                        show=False,
                    )

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
        experiment_cfg=experiment_cfg,
        mc_cfg=mc_cfg,
        prescription_path=prescription_path,
        plan_path=plan_path,
        run_entries=run_entries,
        infer_keys=infer_keys,
        repo_root=repo_root,
        results_orientation=results_orientation,
        system_label=system_label,
        config_override_keys=sorted(config_overrides.keys()),
        truth_default_keys=truth_default_keys,
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
