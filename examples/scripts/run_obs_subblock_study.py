"""Run minimal screening studies for observation sub-block cases.

This harness stays intentionally lightweight. It reuses the existing staged
case runner for trace/render preparation and then runs one explicit screening
mode under ``<case_root>/study/<mode>/``:

- ``full_case``: trace -> render -> optional quick-look -> inference
- ``fisher_only``: prepare a case, build the inference objective, compute dense
  Fisher/Schur summaries, and skip optimization
- ``profile_objective``: scan one fixed shared-parameter assumption over a
  scalar grid and optimize nuisance registration at each grid point
- ``nuisance_absorption``: render truth under one shared value, solve nuisance
  registration under a different fixed shared assumption, and summarize the
  induced fast-state bias
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from astropy.io import fits

from dluxshera.config.io import load_config_file, load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_io import now_iso_local_ms
from dluxshera.utils.obs_subblock_keys import (
    OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS,
    parse_obs_subblock_key_address,
)
from dluxshera.utils.obs_subblock_trace import load_obs_subblock_trace_csv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results"
DEFAULT_TRACE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_trace_template"
    / "subblock_trace_prescription.yaml"
)
DEFAULT_RENDER_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_template"
    / "subblock_generation_prescription.yaml"
)
DEFAULT_INFERENCE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_inference_template"
    / "subblock_inference_prescription.yaml"
)

MODE_FULL_CASE = "full_case"
MODE_FISHER_ONLY = "fisher_only"
MODE_PROFILE_OBJECTIVE = "profile_objective"
MODE_NUISANCE_ABSORPTION = "nuisance_absorption"
SUPPORTED_MODES = (
    MODE_FULL_CASE,
    MODE_FISHER_ONLY,
    MODE_PROFILE_OBJECTIVE,
    MODE_NUISANCE_ABSORPTION,
)

SUMMARY_SCHEMA_VERSION = "obs_subblock_study_summary.v1"
TRACE_STAGE = "trace"
RENDER_STAGE = "render"


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_case_runner_module():
    return _load_module(
        REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_case.py",
        "obs_subblock_case_runner_for_study",
    )


def _load_inference_recipe_module():
    return _load_module(
        REPO_ROOT / "examples" / "recipes" / "observation_subblock_inference.py",
        "obs_subblock_inference_recipe_for_study",
    )


def parse_study_mode(raw_mode: str) -> str:
    """Validate and normalize the requested study mode."""

    mode = str(raw_mode).strip()
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported study mode {raw_mode!r}. Expected one of: "
            + ", ".join(SUPPORTED_MODES)
            + "."
        )
    return mode


def parse_scalar_candidate_parameter(raw_key: str | None) -> str | None:
    """Validate the narrow first-pass candidate-parameter contract."""

    if raw_key is None:
        return None
    address = parse_obs_subblock_key_address(str(raw_key))
    if address.index is not None:
        raise ValueError(
            "The study harness currently supports one scalar candidate "
            "parameter at a time; indexed vector components are not supported."
        )
    if address.base_key not in OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS:
        raise ValueError(
            "Unsupported scalar candidate parameter "
            f"{address.canonical!r}. Supported scalar keys are: "
            + ", ".join(OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS)
        )
    return address.canonical


def parse_scalar_grid(raw: str | Sequence[float] | None) -> tuple[float, ...]:
    """Parse a comma-separated or sequence-valued scalar scan grid."""

    if raw is None:
        return ()
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    values: list[float] = []
    for token in tokens:
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(
                "--scan-values must be a comma-separated list of floats."
            ) from exc
    if not values:
        raise ValueError("--scan-values must contain at least one value.")
    return tuple(values)


def _study_value_token(value: float) -> str:
    """Return a compact filesystem-safe token for one scalar study value."""

    text = f"{float(value):.6g}"
    text = text.replace("+", "")
    text = text.replace("-0", "-")
    return text.replace(".", "p")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def _ensure_mapping(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        parent[key] = {}
        return parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict.")
    return value


def _set_nested_scalar(mapping: dict[str, Any], dotted_key: str, value: float) -> None:
    """Set a scalar dotted-key value inside a nested mapping."""

    current = mapping
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            current[part] = {}
            child = current[part]
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot set nested scalar {dotted_key!r}; path component {part!r} "
                "is not a mapping."
            )
        current = child
    current[parts[-1]] = float(value)


def _get_nested_scalar(mapping: dict[str, Any] | None, dotted_key: str) -> float | None:
    """Read a scalar dotted-key value from a nested mapping when present."""

    if not isinstance(mapping, dict):
        return None
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        return None
    return float(current)


def _case_layout(case_root: Path):
    case_module = _load_case_runner_module()
    return case_module.build_case_layout(case_root.resolve())


def _study_root(case_root: Path, mode: str) -> Path:
    return case_root.resolve() / "study" / mode


def _study_templates_dir(case_root: Path, mode: str) -> Path:
    return _study_root(case_root, mode) / "templates"


def _truth_value_from_render_manifest(manifest_path: Path | None, candidate_key: str) -> float | None:
    """Resolve the rendered truth-side candidate value when available."""

    if not candidate_key:
        return None
    if manifest_path is None or not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    shared_truth = manifest.get("shared_truth")
    truth_value = _get_nested_scalar(
        shared_truth if isinstance(shared_truth, dict) else None,
        candidate_key,
    )
    if truth_value is not None:
        return truth_value
    system_payload = manifest.get("system")
    if not isinstance(system_payload, dict):
        return None
    resolved_cfg = system_payload.get("resolved_config")
    if not isinstance(resolved_cfg, dict):
        return None
    return _get_nested_scalar(resolved_cfg, candidate_key)


def _build_study_templates(
    *,
    mode: str,
    case_root: Path,
    trace_template: Path,
    render_template: Path,
    inference_template: Path,
    candidate_key: str | None,
    truth_value: float | None,
    assumed_value: float | None,
) -> dict[str, Any]:
    """Write study-local template copies with narrow mode-specific patching."""

    templates_dir = _study_templates_dir(case_root, mode)
    templates_dir.mkdir(parents=True, exist_ok=True)

    trace_cfg = load_config_file(trace_template)
    render_cfg = load_config_file(render_template)
    inference_cfg = load_config_file(inference_template)

    if candidate_key is not None and truth_value is not None:
        trace_system_cfg = _ensure_mapping(trace_cfg, "system", path="root")
        _set_nested_scalar(trace_system_cfg, candidate_key, truth_value)

        render_experiment_cfg = _ensure_mapping(render_cfg, "experiment", path="root")
        render_truth_cfg = _ensure_mapping(
            render_experiment_cfg,
            "truth",
            path="experiment",
        )
        _set_nested_scalar(render_truth_cfg, candidate_key, truth_value)

    if candidate_key is not None and assumed_value is not None:
        inference_system_cfg = _ensure_mapping(inference_cfg, "system", path="root")
        _set_nested_scalar(inference_system_cfg, candidate_key, assumed_value)

    if mode in {MODE_FISHER_ONLY, MODE_PROFILE_OBJECTIVE, MODE_NUISANCE_ABSORPTION}:
        inference_experiment_cfg = _ensure_mapping(inference_cfg, "experiment", path="root")
        inference_block_cfg = _ensure_mapping(
            inference_experiment_cfg,
            "inference",
            path="experiment",
        )
        diagnostics_cfg = _ensure_mapping(
            inference_block_cfg,
            "diagnostics",
            path="experiment.inference",
        )
        diagnostics_cfg["plots"] = False

        if mode == MODE_NUISANCE_ABSORPTION:
            diagnostics_cfg["compare_to_truth_when_available"] = True

    if mode == MODE_FISHER_ONLY:
        if candidate_key is None:
            raise ValueError("fisher_only mode requires --candidate.")
        inference_experiment_cfg = _ensure_mapping(inference_cfg, "experiment", path="root")
        inference_block_cfg = _ensure_mapping(
            inference_experiment_cfg,
            "inference",
            path="experiment",
        )
        active_cfg = _ensure_mapping(
            inference_block_cfg,
            "active",
            path="experiment.inference",
        )
        shared_keys = list(active_cfg.get("shared_keys", []))
        if shared_keys and shared_keys != [candidate_key]:
            raise ValueError(
                "fisher_only currently supports one scalar shared candidate and "
                "expects the base inference template to have shared_keys: []."
            )
        if candidate_key not in shared_keys:
            shared_keys.append(candidate_key)
        active_cfg["shared_keys"] = shared_keys

        init_cfg = _ensure_mapping(
            inference_block_cfg,
            "init",
            path="experiment.inference",
        )
        shared_init_cfg = _ensure_mapping(
            init_cfg,
            "shared",
            path="experiment.inference.init",
        )
        reference_value = truth_value
        if reference_value is None:
            reference_value = assumed_value
        if reference_value is None:
            reference_value = _get_nested_scalar(inference_cfg.get("system"), candidate_key)
        if reference_value is None:
            raise ValueError(
                "Unable to resolve a reference value for fisher_only mode. "
                "Provide --truth-value, --assumed-value, or set the candidate in "
                "the inference template system block."
            )
        shared_init_cfg[candidate_key] = float(reference_value)

    trace_path = templates_dir / "trace_template.json"
    render_path = templates_dir / "render_template.json"
    inference_path = templates_dir / "inference_template.json"
    _write_json(trace_path, trace_cfg)
    _write_json(render_path, render_cfg)
    _write_json(inference_path, inference_cfg)

    resolved_assumed = (
        None
        if candidate_key is None
        else _get_nested_scalar(inference_cfg.get("system"), candidate_key)
    )
    resolved_truth = truth_value
    if candidate_key is not None and resolved_truth is None:
        render_truth_cfg = render_cfg.get("experiment", {}).get("truth")
        if isinstance(render_truth_cfg, dict):
            resolved_truth = _get_nested_scalar(render_truth_cfg, candidate_key)
        if resolved_truth is None:
            resolved_truth = _get_nested_scalar(render_cfg.get("system"), candidate_key)

    return {
        "paths": {
            "trace": trace_path,
            "render": render_path,
            "inference": inference_path,
        },
        "resolved_truth_value": resolved_truth,
        "resolved_assumed_value": resolved_assumed,
    }


def _prepare_case_render_artifacts(
    *,
    case_root: Path,
    template_paths: dict[str, Path],
    candidate_key: str | None,
    truth_value: float | None,
    n_frames: int | None,
    dt_s: float | None,
    exposure_time_s: float | None,
    noise_mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Ensure the case has render-ready artifacts for a screening study."""

    case_module = _load_case_runner_module()
    layout = case_module.build_case_layout(case_root.resolve())
    trace_input = case_module._discover_case_trace_input(layout)
    render_inputs = case_module._discover_case_render_inputs(layout)

    stages_to_run: list[str] = []
    render_truth_mismatch = False
    if truth_value is not None and candidate_key is not None:
        current_truth = _truth_value_from_render_manifest(
            render_inputs.manifest.path,
            candidate_key=candidate_key,
        )
        if current_truth is None or not np.isclose(current_truth, truth_value):
            render_truth_mismatch = True

    need_render = render_inputs.cube.path is None or render_truth_mismatch
    if need_render:
        if trace_input.path is None:
            stages_to_run.append(TRACE_STAGE)
        stages_to_run.append(RENDER_STAGE)

    case_prep_summary: dict[str, Any] | None = None
    if stages_to_run:
        case_prep_summary = case_module.run_case_workflow(
            case_root=case_root,
            stages=tuple(stages_to_run),
            trace_template=template_paths["trace"],
            render_template=template_paths["render"],
            inference_template=template_paths["inference"],
            n_frames=n_frames,
            dt_s=dt_s,
            exposure_time_s=exposure_time_s,
            noise_mode=noise_mode,
            dry_run=dry_run,
        )
        render_inputs = case_module._discover_case_render_inputs(layout)

    return {
        "layout": layout,
        "render_inputs": render_inputs,
        "case_prep_summary": case_prep_summary,
        "stages_executed": stages_to_run,
    }


def _prepare_inference_context(
    *,
    config_path: Path,
    reference_mode: str = "truth_when_available",
) -> dict[str, Any]:
    """Reuse the inference recipe helpers to build objective/Fisher context."""

    recipe = _load_inference_recipe_module()
    cfg_path = config_path.resolve()

    user_cfg = load_user_config(
        config_path=cfg_path,
        system_preset=None,
        experiment_preset=None,
    )
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    experiment_cfg = resolved_cfg.get("experiment")
    if system_cfg is None or experiment_cfg is None:
        raise ValueError("Inference study context requires top-level system and experiment blocks.")

    experiment = recipe._validate_experiment_cfg(experiment_cfg)
    inference_cfg = experiment["inference"]
    data_cfg = inference_cfg["data"]

    cube_path = recipe._resolve_relative_path(
        data_cfg["cube"],
        config_path=cfg_path,
        field_name="experiment.inference.data.cube",
    )
    if not cube_path.exists():
        raise FileNotFoundError(f"Observation cube FITS not found: {cube_path}")

    manifest_path_value = data_cfg.get("manifest")
    manifest_path = (
        recipe._resolve_relative_path(
            manifest_path_value,
            config_path=cfg_path,
            field_name="experiment.inference.data.manifest",
        )
        if manifest_path_value is not None
        else recipe.find_obs_subblock_sidecar_manifest(cube_path)
    )
    manifest_input = recipe._load_manifest(manifest_path)

    explicit_trace_path = data_cfg.get("truth_trace")
    trace_path = (
        recipe._resolve_relative_path(
            explicit_trace_path,
            config_path=cfg_path,
            field_name="experiment.inference.data.truth_trace",
        )
        if explicit_trace_path is not None
        else None
    )
    trace_path = recipe._infer_trace_path(
        trace_path=trace_path,
        manifest=manifest_input,
        manifest_path=manifest_path,
    )

    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data, dtype=float)
    if cube.ndim != 3:
        raise ValueError(
            "Observation sub-block cube must have shape (n_frame, ny, nx), "
            f"got {cube.shape}."
        )

    n_frame = int(cube.shape[0])
    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    active_layout = recipe._build_active_state_layout(
        active_cfg=inference_cfg["active"],
        forward_spec=forward_spec,
        reference_store=base_store,
        n_frame=n_frame,
    )
    binder = SheraBinder(system_cfg, forward_spec, base_store)

    reference_image = np.asarray(binder.model(binder.strip_structural(base_store)))
    frame_shape = tuple(int(value) for value in cube.shape[1:])
    if tuple(reference_image.shape) != frame_shape:
        raise ValueError(
            "Observation sub-block cube frame shape is incompatible with the "
            f"configured fixed shared model. cube_frame_shape={frame_shape}, "
            f"model_frame_shape={tuple(reference_image.shape)}."
        )

    trace = None
    if trace_path is not None:
        validate_cfg = inference_cfg["validate"]
        trace = load_obs_subblock_trace_csv(
            trace_path,
            required_varying_keys=(),
            require_contiguous_frame_index=validate_cfg["require_contiguous_frame_index"],
            require_monotonic_time=validate_cfg["require_monotonic_time"],
        )
        if trace.frame_count != n_frame:
            trace = None

    truth_frame_matrix = recipe._build_truth_frame_matrix(
        trace,
        layout=active_layout,
        base_store=base_store,
        n_frame=n_frame,
    )
    initial_state = recipe._resolve_initial_active_state(
        layout=active_layout,
        base_store=base_store,
        init_cfg=inference_cfg["init"],
    )
    theta0 = recipe._pack_active_state(active_layout, initial_state)
    variance_cube = recipe._build_variance_cube(
        data_cube=cube,
        noise_model_cfg=inference_cfg["objective"]["noise_model"],
    )
    objective_bundle = recipe._build_objective_bundle(
        layout=active_layout,
        binder=binder,
        forward_spec=forward_spec,
        base_store=base_store,
        cube_data=cube,
        variance_cube=variance_cube,
        objective_cfg=inference_cfg["objective"],
        priors_cfg=inference_cfg["priors"],
        temporal_cfg=inference_cfg["temporal"],
    )
    theta_reference, theta_reference_source = recipe._resolve_theta_preconditioning_reference(
        layout=active_layout,
        theta0=np.asarray(theta0),
        initial_state=initial_state,
        truth=truth_frame_matrix,
        reference_mode=reference_mode,
    )

    return {
        "recipe": recipe,
        "config_path": cfg_path,
        "cube_path": cube_path,
        "trace_path": trace_path,
        "manifest_path": manifest_path,
        "system_cfg": system_cfg,
        "experiment": experiment,
        "inference_cfg": inference_cfg,
        "layout": active_layout,
        "base_store": base_store,
        "binder": binder,
        "cube": cube,
        "variance_cube": variance_cube,
        "truth": truth_frame_matrix,
        "initial_state": initial_state,
        "theta0": np.asarray(theta0, dtype=float),
        "theta_reference": np.asarray(theta_reference, dtype=float),
        "theta_reference_source": theta_reference_source,
        "objective_bundle": objective_bundle,
    }


def _evaluate_fisher_only(
    *,
    config_path: Path,
    output_dir: Path,
    candidate_key: str,
) -> dict[str, Any]:
    """Compute a dense Fisher/Schur screening summary without optimization."""

    context = _prepare_inference_context(config_path=config_path)
    recipe = context["recipe"]
    layout = context["layout"]

    if list(layout.shared_keys) != [candidate_key]:
        raise ValueError(
            "fisher_only currently supports exactly one shared active key, "
            f"the requested candidate {candidate_key!r}."
        )

    theta_ref = context["theta_reference"]
    fim = np.asarray(
        recipe.fim_theta(
            context["objective_bundle"].total_loss_fn,
            theta_ref,
        ),
        dtype=float,
    )
    if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
        raise ValueError("Dense Fisher matrix must be square.")

    nuisance_dim = int(layout.n_frame * layout.frame_width)
    candidate_dim = int(layout.shared_width)
    if candidate_dim != 1:
        raise ValueError("fisher_only currently supports exactly one scalar shared candidate.")

    nuisance_block = fim[:nuisance_dim, :nuisance_dim]
    candidate_cross = fim[:nuisance_dim, nuisance_dim:]
    candidate_block = fim[nuisance_dim:, nuisance_dim:]
    if nuisance_dim == 0:
        schur = candidate_block.copy()
    else:
        schur = candidate_block - candidate_cross.T @ np.linalg.pinv(nuisance_block) @ candidate_cross

    nuisance_eigs = (
        np.linalg.eigvalsh(0.5 * (nuisance_block + nuisance_block.T))
        if nuisance_dim > 0
        else np.asarray([], dtype=float)
    )
    direct_candidate_info = float(np.asarray(candidate_block, dtype=float).squeeze())
    schur_scalar = float(np.asarray(schur, dtype=float).squeeze())

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "fisher_blocks.npz",
        fim=fim,
        nuisance_block=nuisance_block,
        candidate_cross=candidate_cross,
        candidate_block=candidate_block,
        schur=schur,
        theta_reference=np.asarray(theta_ref, dtype=float),
    )

    summary = {
        "mode": MODE_FISHER_ONLY,
        "candidate_parameter": candidate_key,
        "theta_reference_source": context["theta_reference_source"],
        "theta_dim": int(layout.theta_size),
        "nuisance_dim": nuisance_dim,
        "candidate_dim": candidate_dim,
        "frame_keys": list(layout.frame_keys),
        "shared_keys": list(layout.shared_keys),
        "direct_candidate_information": direct_candidate_info,
        "schur_complement_information": schur_scalar,
        "candidate_information_retained_fraction": (
            None
            if direct_candidate_info == 0.0
            else float(schur_scalar / direct_candidate_info)
        ),
        "nuisance_block_min_eig": (
            None if nuisance_eigs.size == 0 else float(np.min(nuisance_eigs))
        ),
        "nuisance_block_max_eig": (
            None if nuisance_eigs.size == 0 else float(np.max(nuisance_eigs))
        ),
        "dense_fim_trace": float(np.trace(fim)),
        "dense_fim_fro_norm": float(np.linalg.norm(fim)),
        "artifacts": {
            "fisher_summary_json": str((output_dir / "fisher_summary.json").resolve()),
            "fisher_blocks_npz": str((output_dir / "fisher_blocks.npz").resolve()),
        },
    }
    _write_json(output_dir / "fisher_summary.json", summary)
    return summary


def _default_inference_runner(config_path: Path, run_root: Path, dry_run: bool) -> dict[str, Any]:
    case_module = _load_case_runner_module()
    return case_module._default_inference_runner(config_path, run_root, dry_run)


def _build_study_inference_config(
    *,
    template_path: Path,
    run_root: Path,
    render_inputs,
    exposure_time_s: float | None,
    candidate_key: str | None,
    assumed_value: float | None,
    force_truth_comparison: bool,
    disable_plots: bool,
) -> dict[str, Any]:
    """Build one run-specific inference config for study-mode execution."""

    case_module = _load_case_runner_module()
    cfg = case_module.build_inference_case_config(
        template_path=template_path,
        config_dir=run_root,
        case_root=run_root,
        render_inputs=render_inputs,
        exposure_time_s=exposure_time_s,
    )

    if candidate_key is not None and assumed_value is not None:
        system_cfg = case_module._ensure_mapping(cfg, "system", path="root")
        _set_nested_scalar(system_cfg, candidate_key, assumed_value)

    inference_cfg = case_module._ensure_mapping(cfg["experiment"], "inference", path="experiment")
    diagnostics_cfg = case_module._ensure_mapping(
        inference_cfg,
        "diagnostics",
        path="experiment.inference",
    )
    if disable_plots:
        diagnostics_cfg["plots"] = False
    if force_truth_comparison:
        diagnostics_cfg["compare_to_truth_when_available"] = True
    return cfg


def _read_truth_comparison_metrics(
    *,
    csv_path: Path,
    frame_keys: Sequence[str],
) -> dict[str, Any]:
    """Summarize per-key recovered-minus-truth residuals from one CSV artifact."""

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Truth comparison CSV is empty: {csv_path}")

    by_key: dict[str, list[float]] = {key: [] for key in frame_keys}
    for row in rows:
        for key in frame_keys:
            residual_key = f"{key}_residual"
            if residual_key not in row:
                raise ValueError(
                    f"Truth comparison CSV is missing required column {residual_key!r}."
                )
            by_key[key].append(float(row[residual_key]))

    per_key: dict[str, Any] = {}
    stacked: list[float] = []
    for key, values in by_key.items():
        arr = np.asarray(values, dtype=float)
        stacked.extend(arr.tolist())
        per_key[key] = {
            "mean_bias": float(np.mean(arr)),
            "rms_residual": float(np.sqrt(np.mean(np.square(arr)))),
            "max_abs_residual": float(np.max(np.abs(arr))),
        }

    all_arr = np.asarray(stacked, dtype=float)
    return {
        "frame_count": len(rows),
        "per_key": per_key,
        "overall": {
            "mean_bias": float(np.mean(all_arr)),
            "rms_residual": float(np.sqrt(np.mean(np.square(all_arr)))),
            "max_abs_residual": float(np.max(np.abs(all_arr))),
        },
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of flat dict rows to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run_profile_objective(
    *,
    case_root: Path,
    study_root: Path,
    candidate_key: str,
    scan_values: Sequence[float],
    template_path: Path,
    render_inputs,
    exposure_time_s: float | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Run repeated nuisance-only optimizations over a scalar assumed-value grid."""

    runs_dir = study_root / "runs"
    rows: list[dict[str, Any]] = []
    for value in scan_values:
        run_label = f"{candidate_key.split('.')[-1]}_{_study_value_token(value)}"
        run_root = runs_dir / run_label
        cfg = _build_study_inference_config(
            template_path=template_path,
            run_root=run_root,
            render_inputs=render_inputs,
            exposure_time_s=exposure_time_s,
            candidate_key=candidate_key,
            assumed_value=float(value),
            force_truth_comparison=False,
            disable_plots=True,
        )
        config_path = run_root / "inference_config.json"
        _write_json(config_path, cfg)

        row: dict[str, Any] = {
            "run_label": run_label,
            "assumed_value": float(value),
            "config_path": str(config_path.resolve()),
            "planned_output_dir": str((run_root / "inference").resolve()),
        }
        if dry_run:
            row["completed"] = False
            row["status"] = "planned"
        else:
            result = _default_inference_runner(config_path, run_root, False)
            row.update(
                {
                    "completed": True,
                    "status": "ok",
                    "initial_loss": float(result["initial_loss"]),
                    "final_loss": float(result["final_loss"]),
                    "loss_delta": float(result["final_loss"] - result["initial_loss"]),
                    "output_dir": str(Path(result["output_dir"]).resolve()),
                    "manifest_json": result["artifacts"].get("manifest_json"),
                }
            )
        rows.append(row)

    curve_csv_path = study_root / "profile_curve.csv"
    _write_rows_csv(curve_csv_path, rows)

    best_run = None
    completed_rows = [row for row in rows if row.get("completed")]
    if completed_rows:
        best_run = min(completed_rows, key=lambda row: float(row["final_loss"]))

    summary_path = study_root / "summary.json"
    summary = {
        "mode": MODE_PROFILE_OBJECTIVE,
        "case_root": str(case_root.resolve()),
        "candidate_parameter": candidate_key,
        "scan_values": [float(value) for value in scan_values],
        "dry_run": bool(dry_run),
        "summary_path": str(summary_path.resolve()),
        "curve_csv": str(curve_csv_path.resolve()),
        "run_count": len(rows),
        "best_run": best_run,
    }
    _write_json(summary_path, summary)
    return summary


def _run_nuisance_absorption(
    *,
    case_root: Path,
    study_root: Path,
    candidate_key: str,
    assumed_value: float,
    template_path: Path,
    render_inputs,
    exposure_time_s: float | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Run one wrong-model nuisance solve and summarize the induced fast-state bias."""

    run_root = study_root / "run"
    cfg = _build_study_inference_config(
        template_path=template_path,
        run_root=run_root,
        render_inputs=render_inputs,
        exposure_time_s=exposure_time_s,
        candidate_key=candidate_key,
        assumed_value=assumed_value,
        force_truth_comparison=True,
        disable_plots=True,
    )
    config_path = run_root / "inference_config.json"
    _write_json(config_path, cfg)

    summary = {
        "mode": MODE_NUISANCE_ABSORPTION,
        "case_root": str(case_root.resolve()),
        "candidate_parameter": candidate_key,
        "assumed_value": float(assumed_value),
        "dry_run": bool(dry_run),
        "summary_path": str((study_root / "summary.json").resolve()),
        "config_path": str(config_path.resolve()),
        "planned_output_dir": str((run_root / "inference").resolve()),
    }
    if dry_run:
        _write_json(study_root / "summary.json", summary)
        return summary

    result = _default_inference_runner(config_path, run_root, False)
    truth_comparison_path_value = result["artifacts"].get("truth_comparison_csv")
    if not isinstance(truth_comparison_path_value, str) or not truth_comparison_path_value.strip():
        raise ValueError(
            "nuisance_absorption requires truth-comparison outputs. Ensure the "
            "render manifest or explicit truth trace is available."
        )
    truth_comparison_path = Path(truth_comparison_path_value).resolve()
    bias_summary = _read_truth_comparison_metrics(
        csv_path=truth_comparison_path,
        frame_keys=result["frame_keys"],
    )

    summary.update(
        {
            "output_dir": str(Path(result["output_dir"]).resolve()),
            "initial_loss": float(result["initial_loss"]),
            "final_loss": float(result["final_loss"]),
            "loss_delta": float(result["final_loss"] - result["initial_loss"]),
            "truth_comparison_csv": str(truth_comparison_path),
            "bias_summary": bias_summary,
        }
    )
    _write_json(study_root / "summary.json", summary)
    return summary


def run_obs_subblock_study(
    *,
    mode: str,
    case_root: Path,
    case_stages: str | Sequence[str] = "trace,render,quicklook,inference",
    trace_template: Path = DEFAULT_TRACE_TEMPLATE,
    render_template: Path = DEFAULT_RENDER_TEMPLATE,
    inference_template: Path = DEFAULT_INFERENCE_TEMPLATE,
    candidate_key: str | None = None,
    truth_value: float | None = None,
    assumed_value: float | None = None,
    scan_values: Sequence[float] = (),
    n_frames: int | None = None,
    dt_s: float | None = None,
    exposure_time_s: float | None = None,
    noise_mode: str = "inherit",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one observation sub-block screening study."""

    study_mode = parse_study_mode(mode)
    candidate = parse_scalar_candidate_parameter(candidate_key)
    if study_mode != MODE_FULL_CASE and candidate is None:
        raise ValueError(f"{study_mode} mode requires --candidate.")
    if study_mode == MODE_PROFILE_OBJECTIVE and not scan_values:
        raise ValueError("profile_objective mode requires --scan-values.")
    if study_mode == MODE_NUISANCE_ABSORPTION and assumed_value is None:
        raise ValueError("nuisance_absorption mode requires --assumed-value.")

    case_root = case_root.resolve()
    study_root = _study_root(case_root, study_mode)
    study_root.mkdir(parents=True, exist_ok=True)
    summary_path = study_root / "summary.json"

    template_info = _build_study_templates(
        mode=study_mode,
        case_root=case_root,
        trace_template=trace_template.resolve(),
        render_template=render_template.resolve(),
        inference_template=inference_template.resolve(),
        candidate_key=candidate,
        truth_value=truth_value,
        assumed_value=assumed_value,
    )
    template_paths = dict(template_info["paths"])

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "mode": study_mode,
        "case_root": str(case_root),
        "study_root": str(study_root),
        "summary_path": str(summary_path.resolve()),
        "dry_run": bool(dry_run),
        "candidate_parameter": candidate,
        "truth_value_requested": None if truth_value is None else float(truth_value),
        "assumed_value_requested": None if assumed_value is None else float(assumed_value),
        "scan_values_requested": [float(value) for value in scan_values],
        "templates": {
            "trace": str(template_paths["trace"]),
            "render": str(template_paths["render"]),
            "inference": str(template_paths["inference"]),
        },
        "resolved_template_values": {
            "truth_value": template_info["resolved_truth_value"],
            "assumed_value": template_info["resolved_assumed_value"],
        },
    }

    if study_mode == MODE_FULL_CASE:
        case_module = _load_case_runner_module()
        case_summary = case_module.run_case_workflow(
            case_root=case_root,
            stages=case_stages,
            trace_template=template_paths["trace"],
            render_template=template_paths["render"],
            inference_template=template_paths["inference"],
            n_frames=n_frames,
            dt_s=dt_s,
            exposure_time_s=exposure_time_s,
            noise_mode=noise_mode,
            dry_run=dry_run,
        )
        summary["case_summary_path"] = case_summary["summary_path"]
        summary["case_stages_requested"] = list(case_summary["stages_requested"])
        _write_json(summary_path, summary)
        return summary

    prep = _prepare_case_render_artifacts(
        case_root=case_root,
        template_paths=template_paths,
        candidate_key=candidate,
        truth_value=truth_value,
        n_frames=n_frames,
        dt_s=dt_s,
        exposure_time_s=exposure_time_s,
        noise_mode=noise_mode,
        dry_run=dry_run,
    )
    render_inputs = prep["render_inputs"]
    summary["case_prep_stages_executed"] = list(prep["stages_executed"])
    if prep["case_prep_summary"] is not None:
        summary["case_prep_summary_path"] = prep["case_prep_summary"]["summary_path"]
    summary["render_inputs"] = {
        "cube": None if render_inputs.cube.path is None else str(render_inputs.cube.path),
        "truth_trace": (
            None if render_inputs.truth_trace.path is None else str(render_inputs.truth_trace.path)
        ),
        "manifest": (
            None if render_inputs.manifest.path is None else str(render_inputs.manifest.path)
        ),
    }

    if render_inputs.cube.path is None and not dry_run:
        raise ValueError(
            f"{study_mode} mode requires a rendered cube. Run trace/render first or "
            "let the harness prepare them."
        )

    if candidate is not None:
        rendered_truth = _truth_value_from_render_manifest(render_inputs.manifest.path, candidate)
        if rendered_truth is not None:
            summary["rendered_truth_value"] = rendered_truth

    if study_mode == MODE_FISHER_ONLY:
        fisher_config_root = study_root / "fisher"
        fisher_config = _build_study_inference_config(
            template_path=template_paths["inference"],
            run_root=fisher_config_root,
            render_inputs=render_inputs,
            exposure_time_s=exposure_time_s,
            candidate_key=None,
            assumed_value=None,
            force_truth_comparison=False,
            disable_plots=True,
        )
        fisher_config_path = fisher_config_root / "inference_config.json"
        _write_json(fisher_config_path, fisher_config)
        summary["fisher_config_path"] = str(fisher_config_path.resolve())
        if dry_run:
            _write_json(summary_path, summary)
            return summary
        fisher_summary = _evaluate_fisher_only(
            config_path=fisher_config_path,
            output_dir=study_root,
            candidate_key=candidate,
        )
        summary["fisher_summary"] = fisher_summary
        _write_json(summary_path, summary)
        return summary

    if study_mode == MODE_PROFILE_OBJECTIVE:
        profile_summary = _run_profile_objective(
            case_root=case_root,
            study_root=study_root,
            candidate_key=candidate,
            scan_values=scan_values,
            template_path=template_paths["inference"],
            render_inputs=render_inputs,
            exposure_time_s=exposure_time_s,
            dry_run=dry_run,
        )
        summary["profile_summary"] = profile_summary
        _write_json(summary_path, summary)
        return summary

    nuisance_summary = _run_nuisance_absorption(
        case_root=case_root,
        study_root=study_root,
        candidate_key=candidate,
        assumed_value=float(assumed_value),
        template_path=template_paths["inference"],
        render_inputs=render_inputs,
        exposure_time_s=exposure_time_s,
        dry_run=dry_run,
    )
    summary["nuisance_absorption_summary"] = nuisance_summary
    _write_json(summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a lightweight observation sub-block screening study."
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_MODES,
        required=True,
        help="Explicit study mode to run.",
    )
    case_group = parser.add_mutually_exclusive_group(required=True)
    case_group.add_argument(
        "--case-root",
        type=Path,
        default=None,
        help="Explicit case root directory.",
    )
    case_group.add_argument(
        "--case-name",
        type=Path,
        default=None,
        help="Relative case path under --results-root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Parent directory used with --case-name.",
    )
    parser.add_argument(
        "--stages",
        default="trace,render,quicklook,inference",
        help="Stage list used only by full_case mode.",
    )
    parser.add_argument(
        "--trace-template",
        type=Path,
        default=DEFAULT_TRACE_TEMPLATE,
        help="Trace template YAML/JSON path.",
    )
    parser.add_argument(
        "--render-template",
        type=Path,
        default=DEFAULT_RENDER_TEMPLATE,
        help="Render template YAML/JSON path.",
    )
    parser.add_argument(
        "--inference-template",
        type=Path,
        default=DEFAULT_INFERENCE_TEMPLATE,
        help="Inference template YAML/JSON path.",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Canonical scalar candidate parameter key, for example optics.plate_scale_as_per_pix.",
    )
    parser.add_argument(
        "--truth-value",
        type=float,
        default=None,
        help="Optional truth-side scalar value used for render-side patching.",
    )
    parser.add_argument(
        "--assumed-value",
        type=float,
        default=None,
        help="Optional assumed fixed scalar value used for inference-side patching.",
    )
    parser.add_argument(
        "--scan-values",
        default=None,
        help="Comma-separated scalar grid used by profile_objective mode.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Optional trace n_frames override passed through to case prep.",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=None,
        help="Optional trace dt_s override passed through to case prep.",
    )
    parser.add_argument(
        "--exposure-time-s",
        type=float,
        default=None,
        help="Optional system.source.exposure_time_s override passed through to case prep.",
    )
    parser.add_argument(
        "--noise",
        choices=("inherit", "enabled", "disabled"),
        default="inherit",
        help="Optional render noise override passed through to case prep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write study templates and summary without executing renders or inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    case_module = _load_case_runner_module()
    case_root = case_module.resolve_case_root(
        case_root=args.case_root,
        case_name=args.case_name,
        results_root=args.results_root,
    )
    scan_values = parse_scalar_grid(args.scan_values) if args.scan_values is not None else ()
    summary = run_obs_subblock_study(
        mode=args.mode,
        case_root=case_root,
        case_stages=args.stages,
        trace_template=args.trace_template,
        render_template=args.render_template,
        inference_template=args.inference_template,
        candidate_key=args.candidate,
        truth_value=args.truth_value,
        assumed_value=args.assumed_value,
        scan_values=scan_values,
        n_frames=args.n_frames,
        dt_s=args.dt_s,
        exposure_time_s=args.exposure_time_s,
        noise_mode=args.noise,
        dry_run=bool(args.dry_run),
    )
    print(f"Study mode: {summary['mode']}")
    print(f"Case root: {summary['case_root']}")
    print(f"Study root: {summary['study_root']}")
    return summary


if __name__ == "__main__":
    main()
