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
FISHER_DENSE_TO_STRUCTURED_THRESHOLD_DIM = 30


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


def _study_log(message: str, **fields: Any) -> None:
    """Print one flushed diagnostic line for study execution."""

    parts = [f"[obs_subblock_study] {message}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    print(" ".join(parts), flush=True)


def _select_fisher_curvature_method(
    *,
    theta_size: int,
    threshold_dim: int = FISHER_DENSE_TO_STRUCTURED_THRESHOLD_DIM,
) -> str:
    """Select the current Fisher curvature method for the narrow study path."""

    if int(theta_size) > int(threshold_dim):
        return "structured_arrowhead"
    return "dense_full_theta_hessian"


def _path_value_or_missing(root: Any, dotted_path: str) -> tuple[bool, Any]:
    """Resolve a dotted attribute/index path without raising on misses."""

    current = root
    for segment in dotted_path.split("."):
        if current is None:
            return False, None
        try:
            current = getattr(current, segment)
            continue
        except AttributeError:
            pass
        if isinstance(current, dict) and segment in current:
            current = current[segment]
            continue
        if isinstance(current, (list, tuple)) and segment.isdigit():
            index = int(segment)
            if 0 <= index < len(current):
                current = current[index]
                continue
        return False, None
    return True, current


def _scalar_or_none(value: Any) -> float | None:
    """Return a Python float for scalar numeric values when possible."""

    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim != 0:
        return None
    try:
        return float(arr)
    except Exception:
        return None


def _model_delta_summary(
    model_ref: np.ndarray,
    model_perturbed: np.ndarray,
) -> dict[str, float]:
    """Summarize cube-space differences between reference and perturbed models."""

    delta = np.asarray(model_perturbed, dtype=float) - np.asarray(model_ref, dtype=float)
    rms_model_delta = float(np.sqrt(np.mean(np.square(delta))))
    rms_model_ref = float(np.sqrt(np.mean(np.square(np.asarray(model_ref, dtype=float)))))
    denom = max(rms_model_ref, 1.0e-30)
    return {
        "max_abs_model_delta": float(np.max(np.abs(delta))),
        "rms_model_delta": rms_model_delta,
        "relative_rms_model_delta": float(rms_model_delta / denom),
    }


def _array_stats(values: np.ndarray) -> dict[str, Any]:
    """Return compact machine-readable summary stats for one numeric array."""

    arr = np.asarray(values, dtype=float)
    flat = arr.reshape((-1,))
    finite_mask = np.isfinite(flat)
    finite = flat[finite_mask]
    total_count = int(flat.size)
    finite_count = int(finite.size)
    stats: dict[str, Any] = {
        "shape": [int(value) for value in arr.shape],
        "total_count": total_count,
        "finite_count": finite_count,
        "nonfinite_count": int(total_count - finite_count),
        "zero_count": int(np.count_nonzero(flat == 0.0)),
        "nonpositive_count": int(np.count_nonzero(flat <= 0.0)),
    }
    if finite.size == 0:
        stats.update(
            {
                "sum": None,
                "mean": None,
                "min": None,
                "max": None,
                "p01": None,
                "p50": None,
                "p99": None,
            }
        )
        return stats

    stats.update(
        {
            "sum": float(np.sum(finite)),
            "mean": float(np.mean(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "p01": float(np.percentile(finite, 1.0)),
            "p50": float(np.percentile(finite, 50.0)),
            "p99": float(np.percentile(finite, 99.0)),
        }
    )
    return stats


def _resolve_manifest_artifact_path(
    manifest: dict[str, Any] | None,
    *,
    manifest_path: Path | None,
    artifact_name: str,
) -> Path | None:
    """Resolve one manifest artifact path relative to the manifest location."""

    if manifest is None or manifest_path is None:
        return None
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    candidate = artifacts.get(artifact_name)
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    return (manifest_path.parent / candidate).resolve()


def _finite_difference_information_from_cube_derivative(
    *,
    dmodel_dp: np.ndarray,
    variance_cube: np.ndarray,
    frame_reduce: str,
    subblock_reduce: str,
) -> float:
    """Approximate conditional Fisher information from image-space sensitivity."""

    weighted = np.square(np.asarray(dmodel_dp, dtype=float)) / np.asarray(variance_cube, dtype=float)
    flat = weighted.reshape((weighted.shape[0], -1))
    if frame_reduce == "sum":
        per_frame = np.nansum(flat, axis=1)
    elif frame_reduce == "mean":
        per_frame = np.nanmean(flat, axis=1)
    else:
        raise ValueError(f"Unsupported frame_reduce {frame_reduce!r} in finite-difference diagnostic.")

    if subblock_reduce == "sum":
        return float(np.nansum(per_frame))
    if subblock_reduce == "mean":
        return float(np.nanmean(per_frame))
    raise ValueError(
        f"Unsupported subblock_reduce {subblock_reduce!r} in finite-difference diagnostic."
    )


def _build_fisher_noise_audit(context: dict[str, Any]) -> dict[str, Any]:
    """Summarize cube/variance provenance for one fisher_only case."""

    cube = np.asarray(context["cube"], dtype=float)
    variance_cube = np.asarray(context["variance_cube"], dtype=float)
    inference_cfg = context["inference_cfg"]
    noise_model_cfg = inference_cfg["objective"]["noise_model"]
    variance_model = str(noise_model_cfg["variance_model"])
    manifest = context.get("manifest")
    manifest_path = context.get("manifest_path")

    raw_data_stats = _array_stats(cube)
    effective_variance_stats = _array_stats(variance_cube)
    data_floor_value = 1.0e-9
    data_based_variance_cube = np.maximum(cube, data_floor_value)
    data_based_variance_stats = _array_stats(data_based_variance_cube)

    render_variance_path = _resolve_manifest_artifact_path(
        manifest,
        manifest_path=manifest_path,
        artifact_name="variance_fits",
    )
    render_variance_stats = None
    if render_variance_path is not None and render_variance_path.exists():
        with fits.open(render_variance_path) as hdul:
            render_variance_cube = np.asarray(hdul[0].data, dtype=float)
        render_variance_stats = _array_stats(render_variance_cube)

    render_noise = {}
    if isinstance(manifest, dict):
        render_noise_value = manifest.get("noise")
        if isinstance(render_noise_value, dict):
            render_noise = dict(render_noise_value)

    cube_mean = raw_data_stats["mean"]
    variance_mean = effective_variance_stats["mean"]
    data_variance_mean = data_based_variance_stats["mean"]
    render_variance_mean = (
        None if render_variance_stats is None else render_variance_stats.get("mean")
    )

    return {
        "variance_model": variance_model,
        "variance_source": (
            "provided_cube"
            if variance_model == "provided_cube"
            else ("data_cube" if variance_model == "data" else variance_model)
        ),
        "noise_model_kind": str(noise_model_cfg["kind"]),
        "provided_variance_path": (
            None
            if variance_model != "provided_cube"
            else noise_model_cfg.get("path")
        ),
        "render_manifest_path": None if manifest_path is None else str(manifest_path.resolve()),
        "render_variance_artifact_available": bool(
            render_variance_path is not None and render_variance_path.exists()
        ),
        "render_variance_artifact_path": (
            None if render_variance_path is None else str(render_variance_path.resolve())
        ),
        "render_variance_artifact_used": bool(variance_model == "provided_cube"),
        "render_noise": render_noise,
        "cube_stats": raw_data_stats,
        "variance_stats": effective_variance_stats,
        "data_as_variance_stats": data_based_variance_stats,
        "render_variance_stats": render_variance_stats,
        "data_variance_floor_value": data_floor_value,
        "data_variance_floor_clipped_count": int(np.count_nonzero(cube <= data_floor_value)),
        "variance_mean_over_cube_mean": (
            None
            if cube_mean in (None, 0.0) or variance_mean is None
            else float(variance_mean / cube_mean)
        ),
        "data_variance_mean_over_cube_mean": (
            None
            if cube_mean in (None, 0.0) or data_variance_mean is None
            else float(data_variance_mean / cube_mean)
        ),
        "render_variance_mean_over_cube_mean": (
            None
            if cube_mean in (None, 0.0) or render_variance_mean in (None, 0.0)
            else float(render_variance_mean / cube_mean)
        ),
    }


def _classify_candidate_runtime_status(
    *,
    candidate_found_in_layout: bool,
    field_found: bool,
    binding_present: bool,
    store_changed: bool,
    model_changes: bool,
    finite_difference_f_pp: float | None,
    fisher_f_pp: float | None,
) -> str:
    """Classify whether the candidate is live in the current objective path."""

    tol = 1.0e-12
    if not candidate_found_in_layout or not field_found or not binding_present:
        return "candidate_not_found_or_not_runtime_bindable"
    if not store_changed:
        return "candidate_does_not_change_model"
    if not model_changes:
        return "candidate_changes_store_but_not_model"
    if (
        finite_difference_f_pp is not None
        and finite_difference_f_pp > tol
        and (fisher_f_pp is None or abs(fisher_f_pp) <= tol)
    ):
        return "fisher_assembly_suspect"
    return "candidate_changes_model"


def _jsonify_value(value: Any) -> Any:
    """Convert arrays/scalars into compact JSON-friendly payloads."""

    if value is None:
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return value
    if arr.ndim == 0:
        try:
            return float(arr)
        except Exception:
            return value
    return arr.tolist()


def _stores_scalar_changed(reference_value: Any, perturbed_value: Any, *, tol: float = 1.0e-12) -> bool:
    """Return whether two scalar-like values differ beyond tolerance."""

    ref_scalar = _scalar_or_none(reference_value)
    pert_scalar = _scalar_or_none(perturbed_value)
    if ref_scalar is None or pert_scalar is None:
        return not np.array_equal(np.asarray(reference_value), np.asarray(perturbed_value))
    return abs(ref_scalar - pert_scalar) > tol


def _candidate_perturbed_value(reference_value: float, relative_offset: float) -> float:
    """Apply a relative scalar perturbation with a small absolute fallback."""

    if reference_value == 0.0:
        return float(relative_offset)
    return float(reference_value * (1.0 + relative_offset))


def _theta_with_updated_candidate(
    theta_reference: np.ndarray,
    *,
    candidate_index: int,
    candidate_value: float,
) -> np.ndarray:
    """Return a theta copy with one scalar candidate entry replaced."""

    theta = np.asarray(theta_reference, dtype=float).copy()
    theta[int(candidate_index)] = float(candidate_value)
    return theta


def _evaluate_candidate_sensitivity(
    *,
    context: dict[str, Any],
    candidate_key: str,
    fisher_f_pp: float | None,
    truth_value: float | None,
) -> dict[str, Any]:
    """Diagnose whether one shared scalar candidate is live in the objective."""

    recipe = context["recipe"]
    layout = context["layout"]
    theta_reference = np.asarray(context["theta_reference"], dtype=float)
    objective_bundle = context["objective_bundle"]
    inference_cfg = context["inference_cfg"]
    binder = context["binder"]
    base_store = context["base_store"]
    forward_spec = context["forward_spec"]

    theta_candidate_index: int | None = None
    shared_candidate_index: int | None = None
    candidate_found_in_layout = candidate_key in layout.shared_keys
    if candidate_found_in_layout:
        shared_candidate_index = list(layout.shared_keys).index(candidate_key)
        theta_candidate_index = int(layout.n_frame * layout.frame_width + shared_candidate_index)

    field = forward_spec.get(candidate_key) if candidate_key in forward_spec else None
    field_found = field is not None
    binding_present = bool(getattr(field, "binding", None)) if field_found else False
    candidate_field = None if field is None else _candidate_field_payload(field)
    component_name = None
    runtime_path = None
    if field is not None:
        try:
            component_name = str(binder._component_for_field(field))
            runtime_path = str(binder._binding_path_for_field(field))
        except Exception:
            component_name = None
            runtime_path = None

    objective_cfg = inference_cfg["objective"]
    frame_reduce = str(objective_cfg["frame_reduce"])
    subblock_reduce = str(objective_cfg["subblock_reduce"])

    theta_ref_loss = float(np.asarray(objective_bundle.total_loss_fn(theta_reference), dtype=float))
    model_ref = np.asarray(objective_bundle.predict_cube_fn(theta_reference), dtype=float)

    candidate_reference_value = (
        None
        if theta_candidate_index is None
        else float(theta_reference[int(theta_candidate_index)])
    )
    base_store_value = _scalar_or_none(base_store.get(candidate_key, default=None))

    theta_state_ref = recipe._unpack_active_state(layout, recipe.jnp.asarray(theta_reference))
    reference_shared = np.asarray(theta_state_ref.shared, dtype=float)
    reference_primitive_overrides, reference_derived_overrides = recipe._build_runtime_overrides(
        reference_store=base_store,
        key_specs=layout.shared_specs,
        values=recipe.jnp.asarray(reference_shared),
    )
    reference_store = recipe._apply_runtime_active_values(
        reference_store=base_store,
        forward_spec=forward_spec,
        key_specs=layout.shared_specs,
        values=recipe.jnp.asarray(reference_shared),
    )
    reference_store_value = _scalar_or_none(reference_store.get(candidate_key, default=None))
    reference_frame_store = reference_store
    reference_frame_store_value = None
    if int(layout.n_frame) > 0 and int(layout.frame_width) > 0:
        reference_frame_store = recipe._apply_runtime_active_values(
            reference_store=reference_store,
            forward_spec=forward_spec,
            key_specs=layout.frame_specs,
            values=recipe.jnp.asarray(np.asarray(theta_state_ref.frame[0], dtype=float)),
        )
        if hasattr(recipe, "_preserve_shared_derived_active_values"):
            reference_frame_store = recipe._preserve_shared_derived_active_values(
                frame_store=reference_frame_store,
                shared_store=reference_store,
                shared_specs=layout.shared_specs,
            )
        reference_frame_store_value = _scalar_or_none(
            reference_frame_store.get(candidate_key, default=None)
        )

    perturbation_rows: list[dict[str, Any]] = []
    store_changed = False
    runtime_value_changed = False
    model_changes = False
    first_positive_payload: dict[str, Any] | None = None

    for label, relative_offset in (
        ("plus_1pct", 0.01),
        ("minus_1pct", -0.01),
        ("plus_10pct", 0.10),
    ):
        if theta_candidate_index is None or candidate_reference_value is None:
            break

        perturbed_value = _candidate_perturbed_value(candidate_reference_value, relative_offset)
        theta_perturbed = _theta_with_updated_candidate(
            theta_reference,
            candidate_index=theta_candidate_index,
            candidate_value=perturbed_value,
        )
        perturbed_loss = float(
            np.asarray(objective_bundle.total_loss_fn(theta_perturbed), dtype=float)
        )
        model_perturbed = np.asarray(objective_bundle.predict_cube_fn(theta_perturbed), dtype=float)
        model_delta_summary = _model_delta_summary(model_ref, model_perturbed)

        perturbed_state = recipe._unpack_active_state(layout, recipe.jnp.asarray(theta_perturbed))
        perturbed_shared = np.asarray(perturbed_state.shared, dtype=float)
        primitive_overrides, derived_overrides = recipe._build_runtime_overrides(
            reference_store=base_store,
            key_specs=layout.shared_specs,
            values=recipe.jnp.asarray(perturbed_shared),
        )
        perturbed_store = recipe._apply_runtime_active_values(
            reference_store=base_store,
            forward_spec=forward_spec,
            key_specs=layout.shared_specs,
            values=recipe.jnp.asarray(perturbed_shared),
        )
        perturbed_store_value = _scalar_or_none(perturbed_store.get(candidate_key, default=None))
        perturbed_frame_store = perturbed_store
        perturbed_frame_store_value = None
        if int(layout.n_frame) > 0 and int(layout.frame_width) > 0:
            perturbed_frame_store = recipe._apply_runtime_active_values(
                reference_store=perturbed_store,
                forward_spec=forward_spec,
                key_specs=layout.frame_specs,
                values=recipe.jnp.asarray(np.asarray(perturbed_state.frame[0], dtype=float)),
            )
            if hasattr(recipe, "_preserve_shared_derived_active_values"):
                perturbed_frame_store = recipe._preserve_shared_derived_active_values(
                    frame_store=perturbed_frame_store,
                    shared_store=perturbed_store,
                    shared_specs=layout.shared_specs,
                )
            perturbed_frame_store_value = _scalar_or_none(
                perturbed_frame_store.get(candidate_key, default=None)
            )

        telescope_ref = binder._apply_runtime_updates(reference_frame_store)
        telescope_perturbed = binder._apply_runtime_updates(perturbed_frame_store)
        runtime_reference_value = None
        runtime_perturbed_value = None
        runtime_reference_found = False
        runtime_perturbed_found = False
        if component_name is not None and runtime_path is not None:
            runtime_reference_found, runtime_reference_raw = _path_value_or_missing(
                getattr(telescope_ref, component_name),
                runtime_path,
            )
            runtime_perturbed_found, runtime_perturbed_raw = _path_value_or_missing(
                getattr(telescope_perturbed, component_name),
                runtime_path,
            )
            runtime_reference_value = _scalar_or_none(runtime_reference_raw)
            runtime_perturbed_value = _scalar_or_none(runtime_perturbed_raw)

        current_store_changed = _stores_scalar_changed(reference_store_value, perturbed_store_value)
        current_runtime_changed = _stores_scalar_changed(
            runtime_reference_value,
            runtime_perturbed_value,
        )
        current_model_changes = (
            model_delta_summary["max_abs_model_delta"] > 1.0e-12
            or model_delta_summary["rms_model_delta"] > 1.0e-12
        )

        store_changed = store_changed or current_store_changed
        runtime_value_changed = runtime_value_changed or current_runtime_changed
        model_changes = model_changes or current_model_changes

        payload = {
            "label": label,
            "relative_offset": float(relative_offset),
            "candidate_perturbed_value": float(perturbed_value),
            "loss_ref": theta_ref_loss,
            "loss_perturbed": perturbed_loss,
            "loss_delta": float(perturbed_loss - theta_ref_loss),
            "store_value_ref": reference_store_value,
            "store_value_perturbed": perturbed_store_value,
            "store_changed": bool(current_store_changed),
            "frame_store_value_ref": reference_frame_store_value,
            "frame_store_value_perturbed": perturbed_frame_store_value,
            "frame_store_preserves_candidate": bool(
                _stores_scalar_changed(perturbed_store_value, perturbed_frame_store_value)
                is False
            ),
            "runtime_reference_found": bool(runtime_reference_found),
            "runtime_perturbed_found": bool(runtime_perturbed_found),
            "runtime_value_ref": runtime_reference_value,
            "runtime_value_perturbed": runtime_perturbed_value,
            "runtime_value_changed": bool(current_runtime_changed),
            "primitive_overrides": {
                key: _jsonify_value(value) for key, value in primitive_overrides.items()
            },
            "derived_overrides": {
                key: _jsonify_value(value) for key, value in derived_overrides.items()
            },
            **model_delta_summary,
        }
        perturbation_rows.append(payload)
        if relative_offset > 0.0 and first_positive_payload is None:
            first_positive_payload = payload

    finite_difference_payload: dict[str, Any]
    if theta_candidate_index is None or candidate_reference_value is None:
        finite_difference_payload = {
            "eps_rel": 0.01,
            "eps_abs": None,
            "max_abs_dmodel_dp": None,
            "rms_dmodel_dp": None,
            "finite_difference_f_pp": None,
        }
    else:
        eps_rel = 0.01
        eps_abs = max(abs(candidate_reference_value) * eps_rel, 1.0e-8)
        theta_plus = _theta_with_updated_candidate(
            theta_reference,
            candidate_index=theta_candidate_index,
            candidate_value=float(candidate_reference_value + eps_abs),
        )
        theta_minus = _theta_with_updated_candidate(
            theta_reference,
            candidate_index=theta_candidate_index,
            candidate_value=float(candidate_reference_value - eps_abs),
        )
        model_plus = np.asarray(objective_bundle.predict_cube_fn(theta_plus), dtype=float)
        model_minus = np.asarray(objective_bundle.predict_cube_fn(theta_minus), dtype=float)
        dmodel_dp = (model_plus - model_minus) / (2.0 * eps_abs)
        finite_difference_f_pp = _finite_difference_information_from_cube_derivative(
            dmodel_dp=dmodel_dp,
            variance_cube=context["variance_cube"],
            frame_reduce=frame_reduce,
            subblock_reduce=subblock_reduce,
        )
        finite_difference_payload = {
            "eps_rel": float(eps_rel),
            "eps_abs": float(eps_abs),
            "max_abs_dmodel_dp": float(np.max(np.abs(dmodel_dp))),
            "rms_dmodel_dp": float(np.sqrt(np.mean(np.square(dmodel_dp)))),
            "finite_difference_f_pp": float(finite_difference_f_pp),
        }

    conclusion = _classify_candidate_runtime_status(
        candidate_found_in_layout=candidate_found_in_layout,
        field_found=field_found,
        binding_present=binding_present,
        store_changed=store_changed,
        model_changes=model_changes,
        finite_difference_f_pp=finite_difference_payload["finite_difference_f_pp"],
        fisher_f_pp=fisher_f_pp,
    )

    return {
        "candidate_parameter": candidate_key,
        "candidate_reference_value": candidate_reference_value,
        "truth_value": None if truth_value is None else float(truth_value),
        "theta_candidate_index": theta_candidate_index,
        "active_layout": _active_layout_payload(layout),
        "resolved_spec": candidate_field,
        "binding": {
            "field_found": bool(field_found),
            "binding_present": bool(binding_present),
            "component": component_name,
            "runtime_path": runtime_path,
        },
        "store_diagnostics": {
            "base_store_value": base_store_value,
            "reference_store_value": reference_store_value,
            "reference_frame_store_value": reference_frame_store_value,
            "store_changed_under_1pct": bool(store_changed),
            "runtime_value_changed_under_1pct": bool(runtime_value_changed),
            "reference_primitive_overrides": {
                key: _jsonify_value(value) for key, value in reference_primitive_overrides.items()
            },
            "reference_derived_overrides": {
                key: _jsonify_value(value) for key, value in reference_derived_overrides.items()
            },
        },
        "objective": {
            "frame_reduce": frame_reduce,
            "subblock_reduce": subblock_reduce,
            "loss_ref": theta_ref_loss,
        },
        "model_perturbations": perturbation_rows,
        "finite_difference": finite_difference_payload,
        "compact": {
            "candidate_model_rms_delta_1pct": (
                None if first_positive_payload is None else first_positive_payload["rms_model_delta"]
            ),
            "candidate_loss_delta_1pct": (
                None if first_positive_payload is None else first_positive_payload["loss_delta"]
            ),
            "frame_store_preserves_candidate": (
                None
                if first_positive_payload is None
                else first_positive_payload["frame_store_preserves_candidate"]
            ),
            "finite_difference_f_pp": finite_difference_payload["finite_difference_f_pp"],
            "candidate_runtime_status": conclusion,
        },
        "conclusion": conclusion,
    }


def derive_scalar_information_metrics(
    *,
    f_pp: float | None,
    i_marg: float | None,
) -> dict[str, Any]:
    """Derive scalar Fisher screening summaries with safe invalid handling."""

    f_pp_is_finite = f_pp is not None and np.isfinite(f_pp)
    i_marg_is_finite = i_marg is not None and np.isfinite(i_marg)

    sigma_cond: float | None
    if f_pp_is_finite and float(f_pp) > 0.0:
        sigma_cond = float(1.0 / np.sqrt(float(f_pp)))
    elif f_pp_is_finite and float(f_pp) == 0.0:
        sigma_cond = float("inf")
    else:
        sigma_cond = None

    sigma_marg: float | None
    if i_marg_is_finite and float(i_marg) > 0.0:
        sigma_marg = float(1.0 / np.sqrt(float(i_marg)))
    elif i_marg_is_finite and float(i_marg) == 0.0:
        sigma_marg = float("inf")
    else:
        sigma_marg = None

    absorption_fraction: float | None = None
    if f_pp_is_finite and i_marg_is_finite and float(f_pp) > 0.0:
        absorption_fraction = float(1.0 - (float(i_marg) / float(f_pp)))

    if not f_pp_is_finite:
        marginalization_status = "nonfinite_conditional_information"
    elif float(f_pp) < 0.0:
        marginalization_status = "negative_conditional_information"
    elif not i_marg_is_finite:
        marginalization_status = "nonfinite_marginal_information"
    elif float(i_marg) < 0.0:
        marginalization_status = "negative_marginal_information"
    elif float(i_marg) == 0.0:
        marginalization_status = "zero_marginal_information"
    else:
        marginalization_status = "ok"

    return {
        "f_pp": None if f_pp is None else float(f_pp),
        "i_marg": None if i_marg is None else float(i_marg),
        "sigma_cond": sigma_cond,
        "sigma_marg": sigma_marg,
        "absorption_fraction": absorption_fraction,
        "f_pp_is_finite": bool(f_pp_is_finite),
        "i_marg_is_finite": bool(i_marg_is_finite),
        "valid_conditional_sigma": bool(
            sigma_cond is not None and np.isfinite(sigma_cond) and float(f_pp) > 0.0
        ),
        "valid_marginal_sigma": bool(
            sigma_marg is not None and np.isfinite(sigma_marg) and float(i_marg) > 0.0
        ),
        "marginalization_status": marginalization_status,
    }


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


def _resolve_target_name(cfg: dict[str, Any] | None) -> str | None:
    """Return `system.source.target` when available."""

    if not isinstance(cfg, dict):
        return None
    source_cfg = cfg.get("source")
    if not isinstance(source_cfg, dict):
        return None
    target = source_cfg.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def _candidate_field_payload(field: Any) -> dict[str, Any]:
    """Return a JSON-friendly summary of one resolved forward-spec field."""

    return {
        "canonical_key": str(getattr(field, "key", "")),
        "kind": getattr(field, "kind", None),
        "units": getattr(field, "units", None),
        "shape": None if getattr(field, "shape", None) is None else list(field.shape),
        "structural": bool(getattr(field, "structural", False)),
        "binding": getattr(field, "binding", None),
        "transform": getattr(field, "transform", None),
        "depends_on": [str(key) for key in getattr(field, "depends_on", ())],
        "default": _scalar_or_none(getattr(field, "default", None)),
        "scalar": getattr(field, "shape", None) in (None, ()),
    }


def _active_layout_payload(layout: Any) -> dict[str, Any]:
    """Return a compact JSON-friendly active-layout summary."""

    return {
        "n_frame": int(layout.n_frame),
        "frame_width": int(layout.frame_width),
        "shared_width": int(layout.shared_width),
        "theta_size": int(layout.theta_size),
        "frame_keys": list(layout.frame_keys),
        "shared_keys": list(layout.shared_keys),
    }


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
        "resolved_target_name": (
            _resolve_target_name(inference_cfg.get("system"))
            or _resolve_target_name(render_cfg.get("system"))
            or _resolve_target_name(trace_cfg.get("system"))
        ),
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
        if dry_run:
            _study_log(
                "prepare_case_render_artifacts.plan_only",
                case_root=case_root,
                planned_stages=stages_to_run,
            )
        else:
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
    _study_log(
        "prepare_inference_context.start",
        config_path=cfg_path,
        reference_mode=reference_mode,
    )

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
    _study_log(
        "prepare_inference_context.cube_loaded",
        cube_path=cube_path,
        n_frame=n_frame,
        frame_shape=tuple(int(value) for value in cube.shape[1:]),
    )
    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    active_layout = recipe._build_active_state_layout(
        active_cfg=inference_cfg["active"],
        forward_spec=forward_spec,
        reference_store=base_store,
        n_frame=n_frame,
    )
    _study_log(
        "prepare_inference_context.layout_ready",
        frame_keys=list(active_layout.frame_keys),
        shared_keys=list(active_layout.shared_keys),
        frame_width=active_layout.frame_width,
        shared_width=active_layout.shared_width,
        theta_size=active_layout.theta_size,
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
        config_path=cfg_path,
    )
    _study_log(
        "prepare_inference_context.objective_bundle.start",
        theta_size=active_layout.theta_size,
        n_frame=active_layout.n_frame,
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
    _study_log(
        "prepare_inference_context.objective_bundle.done",
        theta_size=active_layout.theta_size,
        n_frame=active_layout.n_frame,
        frame_width=active_layout.frame_width,
        shared_width=active_layout.shared_width,
    )
    theta_reference, theta_reference_source = recipe._resolve_theta_preconditioning_reference(
        layout=active_layout,
        theta0=np.asarray(theta0),
        initial_state=initial_state,
        truth=truth_frame_matrix,
        reference_mode=reference_mode,
    )
    _study_log(
        "prepare_inference_context.done",
        theta_reference_source=theta_reference_source,
        theta_size=active_layout.theta_size,
    )

    return {
        "recipe": recipe,
        "config_path": cfg_path,
        "cube_path": cube_path,
        "trace_path": trace_path,
        "manifest_path": manifest_path,
        "manifest": manifest_input,
        "system_cfg": system_cfg,
        "experiment": experiment,
        "inference_cfg": inference_cfg,
        "forward_spec": forward_spec,
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
    truth_value: float | None = None,
    noise_mode: str | None = None,
    target_name: str | None = None,
) -> dict[str, Any]:
    """Compute a dense Fisher/Schur screening summary without optimization."""

    context = _prepare_inference_context(config_path=config_path)
    recipe = context["recipe"]
    layout = context["layout"]
    dense_fim_shape = (int(layout.theta_size), int(layout.theta_size))
    dense_fim_bytes = int(np.dtype(np.float64).itemsize * dense_fim_shape[0] * dense_fim_shape[1])
    fisher_method = _select_fisher_curvature_method(theta_size=int(layout.theta_size))
    _study_log(
        "fisher_only.start",
        config_path=config_path.resolve(),
        candidate=candidate_key,
        target=target_name,
        noise_mode=noise_mode,
        n_frame=layout.n_frame,
        frame_width=layout.frame_width,
        shared_width=layout.shared_width,
        theta_size=layout.theta_size,
        fisher_method=fisher_method,
        fisher_method_threshold_dim=FISHER_DENSE_TO_STRUCTURED_THRESHOLD_DIM,
        dense_fim_shape=dense_fim_shape,
        dense_fim_bytes=dense_fim_bytes,
    )

    if list(layout.shared_keys) != [candidate_key]:
        raise ValueError(
            "fisher_only currently supports exactly one shared active key, "
            f"the requested candidate {candidate_key!r}."
        )

    theta_ref = context["theta_reference"]
    nuisance_dim = int(layout.n_frame * layout.frame_width)
    candidate_dim = int(layout.shared_width)
    if candidate_dim != 1:
        raise ValueError("fisher_only currently supports exactly one scalar shared candidate.")
    structured_blocks = None
    frame_blocks_np = None
    coupling_blocks_np = None
    shared_blocks_np = None
    fim = None
    dense_global_fim_materialized = fisher_method == "dense_full_theta_hessian"

    if fisher_method == "dense_full_theta_hessian":
        _study_log(
            "fisher_only.fim_theta.start",
            theta_size=layout.theta_size,
            fisher_method=fisher_method,
        )
        fim = np.asarray(
            recipe.fim_theta(
                context["objective_bundle"].total_loss_fn,
                theta_ref,
            ),
            dtype=float,
        )
        _study_log(
            "fisher_only.fim_theta.done",
            theta_size=layout.theta_size,
            dense_fim_shape=tuple(int(v) for v in fim.shape),
        )
        if fim.ndim != 2 or fim.shape[0] != fim.shape[1]:
            raise ValueError("Dense Fisher matrix must be square.")

        nuisance_block = fim[:nuisance_dim, :nuisance_dim]
        candidate_cross = fim[:nuisance_dim, nuisance_dim:]
        candidate_block = fim[nuisance_dim:, nuisance_dim:]
        _study_log(
            "fisher_only.partition.done",
            nuisance_dim=nuisance_dim,
            candidate_dim=candidate_dim,
        )
        if nuisance_dim == 0:
            schur = candidate_block.copy()
        else:
            schur = candidate_block - candidate_cross.T @ np.linalg.pinv(nuisance_block) @ candidate_cross

        nuisance_block_sym = 0.5 * (nuisance_block + nuisance_block.T)
        nuisance_eigs = (
            np.linalg.eigvalsh(nuisance_block_sym)
            if nuisance_dim > 0
            else np.asarray([], dtype=float)
        )
        nuisance_rank = int(np.linalg.matrix_rank(nuisance_block)) if nuisance_dim > 0 else 0
        nuisance_cond = (
            float(np.linalg.cond(nuisance_block))
            if nuisance_dim > 0
            else None
        )
        direct_candidate_info = float(np.asarray(candidate_block, dtype=float).squeeze())
        schur_scalar = float(np.asarray(schur, dtype=float).squeeze())
    else:
        _study_log(
            "fisher_only.structured_arrowhead.start",
            theta_size=layout.theta_size,
            n_frame=layout.n_frame,
            frame_width=layout.frame_width,
            shared_width=layout.shared_width,
        )
        theta_state_ref = recipe._unpack_active_state(layout, recipe.jnp.asarray(theta_ref))
        structured_blocks = recipe.build_independent_frame_curvature_blocks(
            frame_loss_fn=context["objective_bundle"].frame_data_term_fn,
            frame_theta_ref=theta_state_ref.frame,
            shared_theta_ref=theta_state_ref.shared,
            subblock_reduce=str(context["inference_cfg"]["objective"]["subblock_reduce"]),
            kind="structured_arrowhead",
        )
        frame_blocks_np = np.asarray(
            [np.asarray(block.frame_block, dtype=float) for block in structured_blocks.blocks],
            dtype=float,
        )
        coupling_blocks_np = np.asarray(
            [np.asarray(block.coupling_block, dtype=float) for block in structured_blocks.blocks],
            dtype=float,
        )
        shared_blocks_np = np.asarray(
            [np.asarray(block.shared_block, dtype=float) for block in structured_blocks.blocks],
            dtype=float,
        )
        candidate_block = np.sum(shared_blocks_np, axis=0)
        nuisance_rank = 0
        nuisance_cond = None
        nuisance_eigs_list: list[float] = []
        schur = np.array(candidate_block, copy=True, dtype=float)
        for frame_block, coupling_block in zip(frame_blocks_np, coupling_blocks_np):
            nuisance_rank += int(np.linalg.matrix_rank(frame_block))
            frame_cond = float(np.linalg.cond(frame_block))
            if nuisance_cond is None or frame_cond > nuisance_cond:
                nuisance_cond = frame_cond
            nuisance_eigs_list.extend(np.linalg.eigvalsh(0.5 * (frame_block + frame_block.T)).tolist())
            schur = schur - coupling_block.T @ np.linalg.pinv(frame_block) @ coupling_block
        nuisance_block = None
        candidate_cross = None
        nuisance_eigs = np.asarray(nuisance_eigs_list, dtype=float)
        direct_candidate_info = float(np.asarray(candidate_block, dtype=float).squeeze())
        schur_scalar = float(np.asarray(schur, dtype=float).squeeze())
        _study_log(
            "fisher_only.structured_arrowhead.done",
            n_blocks=len(structured_blocks.blocks),
            shared_dim=structured_blocks.shared_dim,
        )

    nuisance_status = "none"
    if nuisance_dim > 0:
        if nuisance_rank < nuisance_dim:
            nuisance_status = "rank_deficient"
        elif nuisance_cond is None or not np.isfinite(nuisance_cond):
            nuisance_status = "singular_or_nonfinite_condition"
        elif nuisance_cond > 1.0e12:
            nuisance_status = "ill_conditioned"
        else:
            nuisance_status = "ok"

    scalar_metrics = derive_scalar_information_metrics(
        f_pp=direct_candidate_info,
        i_marg=schur_scalar,
    )
    candidate_reference_value = float(np.asarray(theta_ref[nuisance_dim:], dtype=float).squeeze())
    resolved_target_name = target_name or _resolve_target_name(context["system_cfg"])
    _study_log(
        "fisher_only.candidate_sensitivity.start",
        candidate=candidate_key,
        theta_candidate_index=nuisance_dim,
    )
    candidate_sensitivity = _evaluate_candidate_sensitivity(
        context=context,
        candidate_key=candidate_key,
        fisher_f_pp=direct_candidate_info,
        truth_value=truth_value,
    )
    noise_audit = _build_fisher_noise_audit(context)
    _study_log(
        "fisher_only.candidate_sensitivity.done",
        conclusion=candidate_sensitivity["conclusion"],
        finite_difference_f_pp=candidate_sensitivity["compact"]["finite_difference_f_pp"],
        frame_store_preserves_candidate=candidate_sensitivity["compact"][
            "frame_store_preserves_candidate"
        ],
        candidate_model_rms_delta_1pct=candidate_sensitivity["compact"][
            "candidate_model_rms_delta_1pct"
        ],
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _study_log(
        "fisher_only.save_npz.start",
        output_dir=output_dir.resolve(),
    )
    npz_payload: dict[str, Any] = {
        "candidate_block": candidate_block,
        "schur": schur,
        "theta_reference": np.asarray(theta_ref, dtype=float),
    }
    if fim is not None:
        npz_payload["fim"] = fim
    if nuisance_block is not None:
        npz_payload["nuisance_block"] = nuisance_block
    if candidate_cross is not None:
        npz_payload["candidate_cross"] = candidate_cross
    if frame_blocks_np is not None:
        npz_payload["frame_blocks"] = frame_blocks_np
    if coupling_blocks_np is not None:
        npz_payload["coupling_blocks"] = coupling_blocks_np
    if shared_blocks_np is not None:
        npz_payload["shared_blocks"] = shared_blocks_np
    np.savez(output_dir / "fisher_blocks.npz", **npz_payload)
    _study_log(
        "fisher_only.save_npz.done",
        output_path=(output_dir / "fisher_blocks.npz").resolve(),
    )
    _write_json(output_dir / "candidate_sensitivity.json", candidate_sensitivity)
    _write_json(output_dir / "noise_audit.json", noise_audit)

    summary = {
        "mode": MODE_FISHER_ONLY,
        "candidate_parameter": candidate_key,
        "fisher_method": fisher_method,
        "target_name": resolved_target_name,
        "truth_value": None if truth_value is None else float(truth_value),
        "candidate_reference_value": candidate_reference_value,
        "noise_mode": noise_mode,
        "frame_count": int(layout.n_frame),
        "active_layout": {
            "n_frame": int(layout.n_frame),
            "frame_width": int(layout.frame_width),
            "shared_width": int(layout.shared_width),
            "theta_size": int(layout.theta_size),
        },
        "dense_fim_shape": [int(v) for v in dense_fim_shape],
        "dense_fim_bytes": dense_fim_bytes,
        "dense_global_fim_materialized": dense_global_fim_materialized,
        "fisher_method_threshold_dim": int(FISHER_DENSE_TO_STRUCTURED_THRESHOLD_DIM),
        "theta_reference_source": context["theta_reference_source"],
        "theta_dim": int(layout.theta_size),
        "nuisance_dim": nuisance_dim,
        "candidate_dim": candidate_dim,
        "frame_keys": list(layout.frame_keys),
        "shared_keys": list(layout.shared_keys),
        "nuisance_block_rank": nuisance_rank,
        "nuisance_block_condition_number": nuisance_cond,
        "nuisance_block_status": nuisance_status,
        "used_pseudoinverse": True,
        **scalar_metrics,
        "candidate_model_rms_delta_1pct": candidate_sensitivity["compact"][
            "candidate_model_rms_delta_1pct"
        ],
        "candidate_loss_delta_1pct": candidate_sensitivity["compact"][
            "candidate_loss_delta_1pct"
        ],
        "frame_store_preserves_candidate": candidate_sensitivity["compact"][
            "frame_store_preserves_candidate"
        ],
        "finite_difference_f_pp": candidate_sensitivity["compact"]["finite_difference_f_pp"],
        "candidate_runtime_status": candidate_sensitivity["compact"][
            "candidate_runtime_status"
        ],
        "direct_candidate_information": direct_candidate_info,
        "schur_complement_information": schur_scalar,
        "candidate_information_retained_fraction": (
            None
            if scalar_metrics["f_pp"] is None
            or scalar_metrics["i_marg"] is None
            or direct_candidate_info == 0.0
            else float(schur_scalar / direct_candidate_info)
        ),
        "nuisance_block_min_eig": (
            None if nuisance_eigs.size == 0 else float(np.min(nuisance_eigs))
        ),
        "nuisance_block_max_eig": (
            None if nuisance_eigs.size == 0 else float(np.max(nuisance_eigs))
        ),
        "dense_fim_trace": None if fim is None else float(np.trace(fim)),
        "dense_fim_fro_norm": None if fim is None else float(np.linalg.norm(fim)),
        "structured_block_count": (
            None if structured_blocks is None else int(len(structured_blocks.blocks))
        ),
        "noise_audit": noise_audit,
        "artifacts": {
            "fisher_summary_json": str((output_dir / "fisher_summary.json").resolve()),
            "fisher_blocks_npz": str((output_dir / "fisher_blocks.npz").resolve()),
            "candidate_sensitivity_json": str(
                (output_dir / "candidate_sensitivity.json").resolve()
            ),
            "noise_audit_json": str((output_dir / "noise_audit.json").resolve()),
        },
    }
    _write_json(output_dir / "fisher_summary.json", summary)
    _study_log(
        "fisher_only.done",
        summary_path=(output_dir / "fisher_summary.json").resolve(),
        marginalization_status=scalar_metrics["marginalization_status"],
    )
    return summary


def _default_inference_runner(config_path: Path, run_root: Path, dry_run: bool) -> dict[str, Any]:
    case_module = _load_case_runner_module()
    return case_module._default_inference_runner(config_path, run_root, dry_run)


def _resolve_render_variance_artifact(manifest_path: Path | None) -> Path | None:
    """Return the render variance artifact path when advertised in the manifest."""

    if manifest_path is None or not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    return _resolve_manifest_artifact_path(
        manifest,
        manifest_path=manifest_path,
        artifact_name="variance_fits",
    )


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
    use_render_variance: bool = False,
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

    if use_render_variance:
        variance_path = _resolve_render_variance_artifact(render_inputs.manifest.path)
        if variance_path is None:
            raise ValueError(
                "use_render_variance=True requires a render manifest with a "
                "variance_fits artifact."
            )
        objective_cfg = case_module._ensure_mapping(
            inference_cfg,
            "objective",
            path="experiment.inference",
        )
        noise_model_cfg = case_module._ensure_mapping(
            objective_cfg,
            "noise_model",
            path="experiment.inference.objective",
        )
        noise_model_cfg["variance_model"] = "provided_cube"
        noise_model_cfg["path"] = case_module._path_for_config(
            variance_path,
            config_dir=run_root,
        )
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
            use_render_variance=False,
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
        use_render_variance=False,
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
    use_render_variance: bool = False,
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
        "target_name": template_info["resolved_target_name"],
        "n_frames_requested": n_frames,
        "dt_s_requested": dt_s,
        "exposure_time_s_requested": exposure_time_s,
        "noise_mode_requested": noise_mode,
        "use_render_variance_requested": bool(use_render_variance),
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
        _study_log(
            "run_obs_subblock_study.fisher_only.case",
            case_root=case_root,
            candidate=candidate,
            target=summary["target_name"],
            noise_mode=noise_mode,
            requested_n_frames=n_frames,
        )
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
            use_render_variance=bool(use_render_variance),
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
            truth_value=summary.get("rendered_truth_value", template_info["resolved_truth_value"]),
            noise_mode=noise_mode,
            target_name=summary["target_name"],
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
