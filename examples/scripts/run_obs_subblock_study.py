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
import math
import os
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from astropy.io import fits

from dluxshera.config.io import load_config_file, load_user_config
from dluxshera.config.numeric import coerce_numeric_value, normalize_optimizer_kwargs
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.observation_belief import ObservationThetaLayout, SubblockSummary
from dluxshera.inference.observation_summary import (
    ImageBackedSubblockSummaryArtifact,
    build_combined_local_parameter_layout,
    load_subblock_summary,
    partition_local_curvature,
    schur_reduce_local_quadratic,
)
from dluxshera.inference.structured_curvature import (
    build_independent_frame_theta_phi_quadratic_blocks,
    compare_structured_and_dense_schur_outputs,
    materialize_structured_schur_sidecar_blocks,
    schur_reduce_independent_frame_blocks,
)
from dluxshera.inference.schedules import validate_optimizer_schedule_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_io import now_iso_local_ms
from dluxshera.utils.obs_subblock_keys import (
    OBS_SUBBLOCK_SUPPORTED_INDEXED_KEYS,
    OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS,
    ObsSubblockKeyAddress,
    apply_obs_subblock_runtime_overrides_without_refresh,
    get_obs_subblock_mapping_value,
    get_obs_subblock_store_value,
    parse_obs_subblock_key_address,
    parse_obs_subblock_varying_keys,
    set_obs_subblock_mapping_value,
    validate_supported_obs_subblock_key_addresses,
)
from dluxshera.utils.obs_subblock_trace import load_obs_subblock_trace_csv

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results"

# ---------------------------------------------------------------------------
# General study workflow defaults
# ---------------------------------------------------------------------------
#
# Source precedence for generated configs:
# 1. start from the base template selected for the study mode;
# 2. apply workflow defaults only for fields explicitly owned by this script;
# 3. apply generated path/data patches such as cube, manifest, truth trace,
#    output directories, and truth-comparison requests;
# 4. apply explicit CLI overrides, where this script exposes them.
#
# Template-owned defaults are intentionally not duplicated as script defaults.
# For example, optimizer kind/base_lr/n_iter and most inference diagnostics are
# read from the inference template unless a targeted CLI override is provided.

DEFAULT_TRACE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_trace_template"
    / "subblock_trace_prescription.yaml"
)
# Registration-iid template used by the image-backed Schur summary validation
# workflow unless the user passes --trace-template.
DEFAULT_SCHUR_TRACE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_trace_template"
    / "subblock_trace_registration_iid_prescription.yaml"
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
MODE_SCHUR_SUMMARY = "schur_summary"
SUPPORTED_MODES = (
    MODE_FULL_CASE,
    MODE_FISHER_ONLY,
    MODE_PROFILE_OBJECTIVE,
    MODE_NUISANCE_ABSORPTION,
    MODE_SCHUR_SUMMARY,
)

SUMMARY_SCHEMA_VERSION = "obs_subblock_study_summary.v1"
TRACE_STAGE = "trace"
RENDER_STAGE = "render"
FISHER_DENSE_TO_STRUCTURED_THRESHOLD_DIM = 30


# ---------------------------------------------------------------------------
# Schur summary workflow defaults
# ---------------------------------------------------------------------------

SOURCE_INFERENCE_TEMPLATE = "inference_template"
SOURCE_SCHUR_WORKFLOW_DEFAULT = "schur_workflow_default"
SOURCE_CLI_OVERRIDE = "cli_override"
SOURCE_GENERATED_CONFIG_PATCH = "generated_config_patch"
SOURCE_INFERENCE_RECIPE_DEFAULT = "inference_recipe_default"
SOURCE_NOT_APPLICABLE = "not_applicable"
TEMPLATE_OWNED_DEFAULT = "template_owned"

DEFAULT_SCHUR_THETA_KEYS = (
    "source.separation_as",
    "source.log_flux_total",
    "source.contrast",
    "optics.plate_scale_as_per_pix",
)
# Optional indexed M1/M2 Zernike components used only when
# ``--enable-zernikes`` is requested. They are irrelevant to the first scalar
# smoke test but provide a stable default family for later validation runs.
DEFAULT_SCHUR_ZERNIKE_INDICES = (0, 1, 2, 3, 4, 5)
DEFAULT_SCHUR_DAMPING = 1.0e-8
# Safety guard for the v0 dense Hessian path over the packed ``[Theta, phi]``
# local vector. This should prevent accidental large-frame or full-Zernike runs
# from materializing an oversized dense Hessian before a structured path exists.
DEFAULT_SCHUR_MAX_DENSE_DIM = 40
DEFAULT_VALIDATE_STRUCTURED_AGAINST_DENSE = False
DEFAULT_SCHUR_PHI_REF = "truth_when_available"
SCHUR_CURVATURE_METHOD_AUTO = "auto"
SCHUR_CURVATURE_METHOD_DENSE = "dense"
SCHUR_CURVATURE_METHOD_STRUCTURED = "structured_independent_frames"
SUPPORTED_SCHUR_CURVATURE_METHODS = (
    SCHUR_CURVATURE_METHOD_AUTO,
    SCHUR_CURVATURE_METHOD_DENSE,
    SCHUR_CURVATURE_METHOD_STRUCTURED,
)
SUPPORTED_SMOKE_THETA_KEYS = frozenset(
    {
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
        "optics.plate_scale_as_per_pix",
    }
)
SCHUR_SUMMARY_PLAN_FILENAME = "schur_summary_plan.json"
SCHUR_SUMMARY_AUDIT_FILENAME = "schur_summary_audit.json"
FRAME_TRUTH_PREVIEW_FILENAME = "frame_truth_preview.json"
SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES = ("warn", "mask", "reject")
SUPPORTED_SCHUR_FRAME_QUALITY_MISSING_POLICIES = ("allow_all", "error")
SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS = ("original", "kept")
DEFAULT_SCHUR_FRAME_QUALITY_POLICY = "warn"
DEFAULT_SCHUR_FRAME_CHI2_THRESHOLD = 5.0
DEFAULT_SCHUR_FRAME_QUALITY_MISSING = "allow_all"
DEFAULT_SCHUR_FRAME_MASK_DENOMINATOR = "original"
DEFAULT_SCHUR_FRAME_MASK_MIN_GOOD_FRAMES = 1


@dataclass(frozen=True)
class SchurFrameQualityReport:
    threshold: float
    total_frame_count: int
    good_frame_indices: tuple[int, ...]
    bad_frame_indices: tuple[int, ...]
    good_frame_count: int
    bad_frame_count: int
    per_frame_reduced_chi2: tuple[float, ...]
    max_frame_reduced_chi2: float | None
    median_frame_reduced_chi2: float | None
    block_reduced_chi2: float | None
    source_manifest_json: str | None
    source_status: str
    warning: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "total_frame_count": int(self.total_frame_count),
            "good_frame_indices": list(self.good_frame_indices),
            "bad_frame_indices": list(self.bad_frame_indices),
            "good_frame_count": int(self.good_frame_count),
            "bad_frame_count": int(self.bad_frame_count),
            "per_frame_reduced_chi2": [float(v) for v in self.per_frame_reduced_chi2],
            "max_frame_reduced_chi2": self.max_frame_reduced_chi2,
            "median_frame_reduced_chi2": self.median_frame_reduced_chi2,
            "block_reduced_chi2": self.block_reduced_chi2,
            "source_manifest_json": self.source_manifest_json,
            "source_status": self.source_status,
            "warning": self.warning,
            "error": self.error,
        }


@dataclass(frozen=True)
class SchurSummaryWorkflowDefaults:
    """Collect top-level policy values owned by the Schur smoke workflow."""

    trace_template: Path
    render_template: Path
    inference_template: Path
    theta_keys: tuple[str, ...]
    zernike_indices: tuple[int, ...]
    schur_damping: float
    max_dense_dim: int
    validate_structured_against_dense: bool
    phi_ref: str
    schur_curvature_method: str
    plan_filename: str
    audit_filename: str
    frame_truth_preview_filename: str


@dataclass(frozen=True)
class SchurReferenceInferencePolicy:
    """Document which recovered-reference inference defaults remain template-owned."""

    optimizer_kind: str = TEMPLATE_OWNED_DEFAULT
    base_lr: str = TEMPLATE_OWNED_DEFAULT
    n_iter: str = TEMPLATE_OWNED_DEFAULT
    schedule: str = TEMPLATE_OWNED_DEFAULT
    preconditioning_enabled: str = TEMPLATE_OWNED_DEFAULT
    preconditioning_method: str = TEMPLATE_OWNED_DEFAULT
    preconditioning_reference: str = TEMPLATE_OWNED_DEFAULT
    preconditioning_damping: str = TEMPLATE_OWNED_DEFAULT
    preconditioning_eig_floor_rel: str = TEMPLATE_OWNED_DEFAULT
    preconditioning_eig_floor_abs: str = TEMPLATE_OWNED_DEFAULT
    diagnostics: str = TEMPLATE_OWNED_DEFAULT


SCHUR_WORKFLOW_DEFAULTS = SchurSummaryWorkflowDefaults(
    trace_template=DEFAULT_SCHUR_TRACE_TEMPLATE,
    render_template=DEFAULT_RENDER_TEMPLATE,
    inference_template=DEFAULT_INFERENCE_TEMPLATE,
    theta_keys=DEFAULT_SCHUR_THETA_KEYS,
    zernike_indices=DEFAULT_SCHUR_ZERNIKE_INDICES,
    schur_damping=DEFAULT_SCHUR_DAMPING,
    max_dense_dim=DEFAULT_SCHUR_MAX_DENSE_DIM,
    validate_structured_against_dense=DEFAULT_VALIDATE_STRUCTURED_AGAINST_DENSE,
    phi_ref=DEFAULT_SCHUR_PHI_REF,
    schur_curvature_method=SCHUR_CURVATURE_METHOD_AUTO,
    plan_filename=SCHUR_SUMMARY_PLAN_FILENAME,
    audit_filename=SCHUR_SUMMARY_AUDIT_FILENAME,
    frame_truth_preview_filename=FRAME_TRUTH_PREVIEW_FILENAME,
)
SCHUR_REFERENCE_INFERENCE_POLICY = SchurReferenceInferencePolicy()

# Diagnostics profiles are optional CLI convenience patches for recovered
# reference review. When omitted, diagnostics remain inference-template owned.
SCHUR_REFERENCE_DIAGNOSTICS_PROFILES: dict[str, dict[str, bool]] = {
    "none": {
        "plots": False,
        "compare_to_truth_when_available": False,
        "first_step_report": False,
        "save_first_step_json": False,
        "save_fim_debug": False,
        "finite_difference_check": False,
        "plot_parameter_history_heatmap": False,
        "plot_parameter_residual_history_heatmap": False,
        "plot_parameter_history_lines": False,
        "plot_parameter_residual_history_lines": False,
    },
    "basic": {
        "plots": True,
        "compare_to_truth_when_available": True,
        "first_step_report": False,
        "save_first_step_json": False,
        "save_fim_debug": False,
        "finite_difference_check": False,
        "plot_parameter_history_heatmap": False,
        "plot_parameter_residual_history_heatmap": False,
        "plot_parameter_history_lines": False,
        "plot_parameter_residual_history_lines": False,
    },
    "review": {
        "plots": True,
        "compare_to_truth_when_available": True,
        "first_step_report": True,
        "save_first_step_json": True,
        "save_fim_debug": False,
        "finite_difference_check": False,
        "plot_parameter_history_heatmap": False,
        "plot_parameter_residual_history_heatmap": False,
        "plot_parameter_history_lines": True,
        "plot_parameter_residual_history_lines": True,
    },
    "full": {
        "plots": True,
        "compare_to_truth_when_available": True,
        "first_step_report": True,
        "save_first_step_json": True,
        "save_fim_debug": True,
        "finite_difference_check": True,
        "plot_parameter_history_heatmap": True,
        "plot_parameter_residual_history_heatmap": True,
        "plot_parameter_history_lines": True,
        "plot_parameter_residual_history_lines": True,
    },
}

TRACE_TRUTH_OVERRIDE_KEYS = {
    "trace_x0_as": "source.x_position_as",
    "trace_y0_as": "source.y_position_as",
    "trace_pa0_deg": "source.position_angle_deg",
}
TRACE_JITTER_OVERRIDE_KEYS = {
    "trace_jitter_x_sigma_as": "source.x_position_as",
    "trace_jitter_y_sigma_as": "source.y_position_as",
    "trace_jitter_pa_sigma_deg": "source.position_angle_deg",
}
INFERENCE_INIT_OVERRIDE_KEYS = {
    "init_x_as": "source.x_position_as",
    "init_y_as": "source.y_position_as",
    "init_pa_deg": "source.position_angle_deg",
}


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


def parse_candidate_parameter_address(
    raw_key: str | None,
    *,
    forward_spec: Any | None = None,
    reference_store: Any | None = None,
) -> ObsSubblockKeyAddress | None:
    """Parse and validate one scalar-or-indexed candidate key address."""

    if raw_key is None:
        return None
    address = parse_obs_subblock_key_address(str(raw_key))
    validate_supported_obs_subblock_key_addresses(
        (address,),
        forward_spec=forward_spec,
        reference_store=reference_store,
    )
    return address


def parse_scalar_candidate_parameter(raw_key: str | None) -> str | None:
    """Backward-compatible wrapper returning the canonical candidate string."""

    address = parse_candidate_parameter_address(raw_key)
    return None if address is None else address.canonical


def _candidate_metadata(candidate_key: str | None) -> dict[str, Any]:
    """Return stable candidate metadata fields for summaries and tables."""

    if candidate_key is None:
        return {
            "candidate_parameter": None,
            "candidate_base_key": None,
            "candidate_index": None,
        }
    address = parse_obs_subblock_key_address(candidate_key)
    return {
        "candidate_parameter": address.canonical,
        "candidate_base_key": address.base_key,
        "candidate_index": address.index,
    }


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


def parse_theta_keys(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse one comma-separated observation-level Theta key list."""

    if raw is None:
        return DEFAULT_SCHUR_THETA_KEYS
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    values: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token:
            continue
        address = parse_obs_subblock_key_address(token)
        canonical = address.canonical
        if canonical in seen:
            raise ValueError(f"Duplicate theta key: {canonical}.")
        seen.add(canonical)
        values.append(canonical)
    if not values:
        raise ValueError("--theta-keys must contain at least one key.")
    return tuple(values)


def parse_key_value_float_overrides(
    raw_items: Sequence[str] | None,
    *,
    option_name: str,
) -> dict[str, float]:
    """Parse repeatable ``KEY=VALUE`` numeric observation-key overrides."""

    if raw_items is None:
        return {}

    parsed: dict[str, float] = {}
    for raw_item in raw_items:
        text = str(raw_item).strip()
        if not text or "=" not in text:
            raise ValueError(f"{option_name} entries must use KEY=VALUE syntax.")
        raw_key, raw_value = text.split("=", 1)
        raw_key = raw_key.strip()
        if not raw_key:
            raise ValueError(f"{option_name} entries must include a non-empty KEY.")
        address = parse_obs_subblock_key_address(raw_key)
        validate_supported_obs_subblock_key_addresses((address,))
        canonical = address.canonical
        if canonical in parsed:
            raise ValueError(f"Duplicate {option_name} override for {canonical}.")
        parsed[canonical] = float(
            coerce_numeric_value(
                raw_value,
                path=f"{option_name}.{canonical}",
                allow_str=True,
            )
        )
    return parsed


def normalize_schur_phi_ref_mode(raw_mode: str) -> str:
    """Normalize legacy and preferred Schur-summary fast-state reference modes."""

    mode = str(raw_mode).strip()
    if mode == "truth":
        return "truth_when_available"
    if mode not in {"recovered", "truth_when_available", "init"}:
        raise ValueError(
            "phi_ref must be one of: recovered, truth_when_available, init."
        )
    return mode


def normalize_schur_curvature_method(raw_method: str | None) -> str:
    """Normalize the requested Schur-summary curvature export method."""

    method = SCHUR_CURVATURE_METHOD_AUTO if raw_method is None else str(raw_method).strip()
    if method not in SUPPORTED_SCHUR_CURVATURE_METHODS:
        allowed = ", ".join(SUPPORTED_SCHUR_CURVATURE_METHODS)
        raise ValueError(
            f"schur_curvature_method must be one of: {allowed}."
        )
    return method


def _structured_schur_support_from_inference_cfg(
    *,
    inference_cfg: Mapping[str, Any],
    n_frames: int | None,
) -> dict[str, Any]:
    """Return config-level support metadata for structured Schur export."""

    active_cfg = inference_cfg.get("active", {})
    temporal_cfg = inference_cfg.get("temporal", {})
    frame_model_cfg = (
        temporal_cfg.get("frame_model", {})
        if isinstance(temporal_cfg, Mapping)
        else {}
    )
    frame_keys = tuple(str(key) for key in active_cfg.get("frame_keys", ()))
    shared_keys = tuple(str(key) for key in active_cfg.get("shared_keys", ()))
    frame_model_kind = (
        str(frame_model_cfg.get("kind"))
        if isinstance(frame_model_cfg, Mapping)
        else None
    )
    reasons: list[str] = []
    if n_frames is None or int(n_frames) <= 0:
        reasons.append("n_frames is unknown or non-positive")
    if frame_model_kind != "independent":
        reasons.append("temporal.frame_model.kind is not independent")
    if not frame_keys:
        reasons.append("no frame-local active keys are configured")
    if shared_keys:
        reasons.append("shared active subblock state is configured")
    supported = not reasons
    return {
        "supported": bool(supported),
        "unsupported_reasons": reasons,
        "frame_model_kind": frame_model_kind,
        "n_frames": None if n_frames is None else int(n_frames),
        "frame_phi_dim": int(len(frame_keys)),
        "shared_phi_dim": int(len(shared_keys)),
        "required_assumptions": (
            "structured_independent_frames currently requires independent "
            "frame model, frame-local active state, no active shared subblock "
            "state, and per-frame objective access through "
            "ObjectiveBundle.frame_data_term_fn."
        ),
    }


def _structured_schur_support_from_context(context: Mapping[str, Any]) -> dict[str, Any]:
    """Return runtime support metadata for structured Schur export."""

    layout = context["layout"]
    inference_cfg = context["inference_cfg"]
    support = _structured_schur_support_from_inference_cfg(
        inference_cfg=inference_cfg,
        n_frames=int(layout.n_frame),
    )
    objective_bundle = context.get("objective_bundle")
    if not hasattr(objective_bundle, "frame_data_term_fn"):
        support["supported"] = False
        support.setdefault("unsupported_reasons", []).append(
            "ObjectiveBundle.frame_data_term_fn is unavailable"
        )
    if int(layout.shared_width) != int(support["shared_phi_dim"]):
        support["supported"] = False
        support.setdefault("unsupported_reasons", []).append(
            "runtime shared active width does not match config metadata"
        )
    support["frame_phi_dim"] = int(layout.frame_width)
    support["shared_phi_dim"] = int(layout.shared_width)
    support["n_frames"] = int(layout.n_frame)
    return support


def _select_schur_curvature_method(
    *,
    requested_method: str,
    combined_dim: int,
    max_dense_dim: int,
    structured_support: Mapping[str, Any],
) -> str:
    """Select the effective Schur curvature method or raise a clear error."""

    requested = normalize_schur_curvature_method(requested_method)
    dense_allowed = int(combined_dim) <= int(max_dense_dim)
    structured_supported = bool(structured_support.get("supported"))
    unsupported_reasons = ", ".join(
        str(reason) for reason in structured_support.get("unsupported_reasons", ())
    )
    structured_message = (
        "structured_independent_frames currently requires independent frame "
        "model, frame-local active state, no active shared subblock state, and "
        "per-frame objective access through ObjectiveBundle.frame_data_term_fn."
    )

    if requested == SCHUR_CURVATURE_METHOD_DENSE:
        _validate_schur_dense_dimension(
            combined_dim=int(combined_dim),
            max_dense_dim=int(max_dense_dim),
        )
        return SCHUR_CURVATURE_METHOD_DENSE
    if requested == SCHUR_CURVATURE_METHOD_STRUCTURED:
        if not structured_supported:
            detail = f" Failed assumption(s): {unsupported_reasons}." if unsupported_reasons else ""
            raise ValueError(structured_message + detail)
        return SCHUR_CURVATURE_METHOD_STRUCTURED

    if dense_allowed:
        return SCHUR_CURVATURE_METHOD_DENSE
    if structured_supported:
        return SCHUR_CURVATURE_METHOD_STRUCTURED
    detail = f" Failed assumption(s): {unsupported_reasons}." if unsupported_reasons else ""
    raise ValueError(
        f"Combined dense dimension {combined_dim} exceeds max_dense_dim={max_dense_dim}, "
        "and structured Schur export is not available. "
        + structured_message
        + detail
    )


def _dense_vs_structured_comparison_state(
    *,
    requested: bool,
    curvature_method_used: str | None,
    combined_dim: int,
    max_dense_dim: int,
) -> dict[str, Any]:
    """Return validation-only dense comparison state for audit payloads."""

    method = (
        None
        if curvature_method_used is None
        else normalize_schur_curvature_method(curvature_method_used)
    )
    if method != SCHUR_CURVATURE_METHOD_STRUCTURED:
        return {
            "dense_vs_structured_comparison_requested": bool(requested),
            "dense_vs_structured_comparison_run": False,
            "dense_vs_structured_comparison_skipped_reason": "curvature_method_not_structured",
            "max_dense_dim": int(max_dense_dim),
            "combined_dim": int(combined_dim),
        }
    if not requested:
        return {
            "dense_vs_structured_comparison_requested": False,
            "dense_vs_structured_comparison_run": False,
            "dense_vs_structured_comparison_skipped_reason": "not_requested",
            "max_dense_dim": int(max_dense_dim),
            "combined_dim": int(combined_dim),
        }
    if int(combined_dim) > int(max_dense_dim):
        return {
            "dense_vs_structured_comparison_requested": True,
            "dense_vs_structured_comparison_run": False,
            "dense_vs_structured_comparison_skipped_reason": "combined_dim_exceeds_max_dense_dim",
            "max_dense_dim": int(max_dense_dim),
            "combined_dim": int(combined_dim),
        }
    return {
        "dense_vs_structured_comparison_requested": True,
        "dense_vs_structured_comparison_run": True,
        "dense_vs_structured_comparison_skipped_reason": None,
        "max_dense_dim": int(max_dense_dim),
        "combined_dim": int(combined_dim),
    }


def classify_schur_summary_theta_keys(
    theta_keys: Sequence[str] | str | None,
) -> dict[str, list[str]]:
    """Classify Theta keys by current dense-autodiff Schur-summary support status."""

    canonical_keys = parse_theta_keys(theta_keys)
    supported: list[str] = []
    experimental: list[str] = []
    for label in canonical_keys:
        if label in SUPPORTED_SMOKE_THETA_KEYS:
            supported.append(label)
        else:
            experimental.append(label)
    return {
        "supported": supported,
        "experimental": experimental,
        "blocked": [],
    }


def validate_schur_summary_theta_keys(
    theta_keys: Sequence[str] | str | None,
) -> dict[str, list[str]]:
    """Validate requested Schur-summary Theta keys before JAX tracing begins.

    The current dense image-backed exporter applies active Theta values as a
    direct overlay on a resolved base store. Source photometry dependents are
    patched explicitly with JAX-safe array operations instead of differentiating
    through a full ``refresh_derived(...)`` call.
    """

    classification = classify_schur_summary_theta_keys(theta_keys)
    return classification


def parse_zernike_indices(raw: str | Sequence[int] | None) -> tuple[int, ...]:
    """Parse one comma-separated Zernike index list."""

    if raw is None:
        return DEFAULT_SCHUR_ZERNIKE_INDICES
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    values: list[int] = []
    seen: set[int] = set()
    for token in tokens:
        if not token:
            continue
        value = int(token)
        if value in seen:
            raise ValueError(f"Duplicate Zernike index: {value}.")
        seen.add(value)
        values.append(value)
    if not values:
        raise ValueError("--zernike-indices must contain at least one index.")
    return tuple(values)


def _build_observation_theta_layout(
    *,
    theta_keys: Sequence[str],
    enable_zernikes: bool,
    zernike_indices: Sequence[int],
) -> ObservationThetaLayout:
    """Build the observation-level Theta layout for Schur-summary export."""

    validate_schur_summary_theta_keys(theta_keys)
    theta_keys_resolved = parse_theta_keys(theta_keys)
    requested = set(theta_keys_resolved)
    addresses = parse_obs_subblock_varying_keys(theta_keys_resolved)
    validate_supported_obs_subblock_key_addresses(addresses)
    requested_primary = sorted(
        address.index
        for address in addresses
        if address.base_key == "optics.primary.zernike_coeffs_nm"
        and address.index is not None
    )
    requested_secondary = sorted(
        address.index
        for address in addresses
        if address.base_key == "optics.secondary.zernike_coeffs_nm"
        and address.index is not None
    )
    configured_indices = list(parse_zernike_indices(zernike_indices))
    primary_indices = requested_primary or (configured_indices if enable_zernikes else [])
    secondary_indices = requested_secondary or (configured_indices if enable_zernikes else [])

    source_cfg = {
        "separation_as": "source.separation_as" in requested,
        "log_flux_total": "source.log_flux_total" in requested,
        "contrast": "source.contrast" in requested,
    }
    optics_cfg: dict[str, Any] = {
        "plate_scale_as_per_pix": "optics.plate_scale_as_per_pix" in requested,
        "primary_zernikes": {
            "enabled": bool(primary_indices),
            "indices": list(primary_indices),
        },
        "secondary_zernikes": {
            "enabled": bool(secondary_indices),
            "indices": list(secondary_indices),
        },
    }
    return ObservationThetaLayout.from_config(
        {
            "theta_layout": {
                "source": source_cfg,
                "optics": optics_cfg,
            }
        }
    )


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
    recipe = context["recipe"]
    inference_cfg = context["inference_cfg"]
    noise_model_cfg = inference_cfg["objective"]["noise_model"]
    variance_model = str(noise_model_cfg["variance_model"])
    manifest = context.get("manifest")
    manifest_path = context.get("manifest_path")

    raw_data_stats = _array_stats(cube)
    effective_variance_stats = _array_stats(variance_cube)
    data_floor_value, data_floor_source = recipe._resolve_data_variance_floor(
        noise_model_cfg,
        path="experiment.inference.objective.noise_model.variance_floor",
    )
    data_based_variance_cube = np.maximum(cube, data_floor_value)
    data_based_variance_stats = _array_stats(data_based_variance_cube)
    data_floor_clipped_count = int(np.count_nonzero(cube <= data_floor_value))
    data_floor_total = int(cube.size)

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
        "data_variance_floor_source": data_floor_source,
        "data_variance_floor_clipped_count": data_floor_clipped_count,
        "data_variance_floor_clipped_fraction": (
            None
            if data_floor_total == 0
            else float(data_floor_clipped_count / data_floor_total)
        ),
        "data_variance_min_before_floor": (
            None if data_floor_total == 0 else float(np.min(cube))
        ),
        "data_variance_min_after_floor": (
            None
            if data_based_variance_cube.size == 0
            else float(np.min(data_based_variance_cube))
        ),
        "data_variance_median_after_floor": (
            None
            if data_based_variance_cube.size == 0
            else float(np.median(data_based_variance_cube))
        ),
        "data_variance_max_after_floor": (
            None
            if data_based_variance_cube.size == 0
            else float(np.max(data_based_variance_cube))
        ),
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
    candidate_address = parse_obs_subblock_key_address(candidate_key)

    theta_candidate_index: int | None = None
    shared_candidate_index: int | None = None
    candidate_found_in_layout = candidate_key in layout.shared_keys
    if candidate_found_in_layout:
        shared_candidate_index = list(layout.shared_keys).index(candidate_key)
        theta_candidate_index = int(layout.n_frame * layout.frame_width + shared_candidate_index)

    field = (
        forward_spec.get(candidate_address.base_key)
        if candidate_address.base_key in forward_spec
        else None
    )
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
    base_store_value = get_obs_subblock_store_value(base_store, address=candidate_address)

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
    reference_store_value = get_obs_subblock_store_value(
        reference_store,
        address=candidate_address,
    )
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
        reference_frame_store_value = get_obs_subblock_store_value(
            reference_frame_store,
            address=candidate_address,
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
        perturbed_store_value = get_obs_subblock_store_value(
            perturbed_store,
            address=candidate_address,
        )
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
            perturbed_frame_store_value = get_obs_subblock_store_value(
                perturbed_frame_store,
                address=candidate_address,
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
        **_candidate_metadata(candidate_key),
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


def _candidate_token(candidate_key: str) -> str:
    """Return a compact filesystem-safe token for one canonical candidate key."""

    address = parse_obs_subblock_key_address(candidate_key)
    token = address.base_key.replace(".", "_")
    if address.index is not None:
        token = f"{token}_i{address.index}"
    return token


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


def _maxrss_to_mb(value: float) -> float:
    """Normalize ``resource.ru_maxrss`` to MB on macOS and Linux."""

    if sys.platform == "darwin":
        return float(value) / (1024.0 * 1024.0)
    return float(value) / 1024.0


def _current_rss_mb() -> tuple[float | None, str | None]:
    """Return current process RSS in MB using isolated best-effort probes."""

    if sys.platform == "darwin":
        try:
            completed = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if completed.returncode == 0 and completed.stdout.strip():
                return float(completed.stdout.strip().splitlines()[-1]) / 1024.0, None
            return None, (completed.stderr or "ps returned no rss").strip()
        except Exception as exc:
            return None, str(exc)
    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        try:
            pages = int(statm_path.read_text(encoding="utf-8").split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size) / (1024.0 * 1024.0), None
        except Exception as exc:
            return None, str(exc)
    return None, "current RSS unavailable on this platform"


def _json_safe(value: Any) -> Any:
    """Convert diagnostic metadata to JSON-safe values without raising."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def array_memory_metadata(value: Any) -> dict[str, Any]:
    """Return shape/dtype/byte metadata for an array-like diagnostic value."""

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    nbytes = getattr(value, "nbytes", None)
    return {
        "shape": None if shape is None else [int(dim) for dim in tuple(shape)],
        "dtype": None if dtype is None else str(dtype),
        "nbytes": None if nbytes is None else int(nbytes),
        "mb": None if nbytes is None else float(nbytes) / (1024.0 * 1024.0),
    }


def named_array_memory_metadata(**arrays: Any) -> dict[str, Any]:
    """Build JSON-safe array metadata for stage diagnostics."""

    return {
        name: array_memory_metadata(array)
        for name, array in arrays.items()
        if array is not None
    }


def capture_memory_snapshot(stage: str, **metadata: Any) -> dict[str, Any]:
    """Capture one best-effort process memory snapshot for diagnostics.

    This records Python heap statistics only when ``tracemalloc`` is enabled.
    Those numbers do not include JAX/XLA native allocations; RSS/max RSS are the
    broad process-level evidence for those allocations.
    """

    measurement_errors: dict[str, str] = {}
    rss_mb, rss_error = _current_rss_mb()
    if rss_error:
        measurement_errors["rss_mb"] = rss_error
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_rss_mb = _maxrss_to_mb(float(usage.ru_maxrss))
    except Exception as exc:
        peak_rss_mb = None
        measurement_errors["peak_rss_mb"] = str(exc)

    tracemalloc_current_mb = None
    tracemalloc_peak_mb = None
    if tracemalloc.is_tracing():
        try:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_current_mb = float(current) / (1024.0 * 1024.0)
            tracemalloc_peak_mb = float(peak) / (1024.0 * 1024.0)
        except Exception as exc:
            measurement_errors["tracemalloc"] = str(exc)

    payload = {
        "timestamp": now_iso_local_ms(),
        "monotonic_seconds": time.monotonic(),
        "stage": str(stage),
        "pid": os.getpid(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "rss_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "tracemalloc_current_mb": tracemalloc_current_mb,
        "tracemalloc_peak_mb": tracemalloc_peak_mb,
        "metadata": _json_safe(metadata),
    }
    if measurement_errors:
        payload["measurement_errors"] = measurement_errors
    return payload


class MemoryDiagnosticsRecorder:
    """Append memory snapshots to JSONL and retain a compact in-process audit."""

    def __init__(self, path: Path, *, enable_tracemalloc: bool = True):
        self.path = path.resolve()
        self.stages: list[str] = []
        self.peak_rss_mb_observed: float | None = None
        self.last_record: dict[str, Any] | None = None
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()

    def record(self, stage: str, **metadata: Any) -> dict[str, Any]:
        snapshot = capture_memory_snapshot(stage, **metadata)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, default=str) + "\n")
        self.stages.append(str(stage))
        peak = snapshot.get("peak_rss_mb")
        if isinstance(peak, (int, float)):
            self.peak_rss_mb_observed = (
                float(peak)
                if self.peak_rss_mb_observed is None
                else max(self.peak_rss_mb_observed, float(peak))
            )
        self.last_record = snapshot
        return snapshot

    def audit_payload(self, **metadata: Any) -> dict[str, Any]:
        return {
            "diagnostics_enabled": True,
            "timeline_jsonl": str(self.path),
            "stages_recorded": list(self.stages),
            "peak_rss_mb_observed": self.peak_rss_mb_observed,
            "last_stage": None if self.last_record is None else self.last_record.get("stage"),
            "last_successful_stage": None
            if self.last_record is None
            else self.last_record.get("stage"),
            **_json_safe(metadata),
        }


def _ensure_mapping(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        parent[key] = {}
        return parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict.")
    return value


def _non_null_float_overrides(values: Mapping[str, float | None]) -> dict[str, float]:
    """Return only CLI override values that were explicitly provided."""

    return {key: float(value) for key, value in values.items() if value is not None}


def _trace_plan_entry(trace_cfg: dict[str, Any], key: str) -> dict[str, Any]:
    experiment_cfg = _ensure_mapping(trace_cfg, "experiment", path="root")
    trace_block = _ensure_mapping(experiment_cfg, "trace", path="experiment")
    plan_cfg = _ensure_mapping(trace_block, "plan", path="experiment.trace")
    entry = plan_cfg.get(key)
    if not isinstance(entry, dict):
        raise ValueError(
            f"Trace override requested for {key!r}, but experiment.trace.plan.{key} "
            "is not present in the trace template."
        )
    return entry


def _apply_trace_truth_overrides(
    trace_cfg: dict[str, Any],
    *,
    truth_overrides: Mapping[str, float],
    jitter_overrides: Mapping[str, float],
    seed: int | None,
) -> dict[str, Any]:
    """Patch narrow smoke-test trace controls into a trace template copy.

    Policy notes:
    - Starts from the selected trace template.
    - Applies only explicit ``--trace-*`` CLI overrides.
    - Does not invent stochastic effects; jitter overrides require a compatible
      ``iid_jitter`` or ``random_walk`` effect already present in the template.
    """

    applied: dict[str, Any] = {"truth": {}, "jitter": {}, "seed": None}
    for cli_name, value in truth_overrides.items():
        trace_key = TRACE_TRUTH_OVERRIDE_KEYS[cli_name]
        entry = _trace_plan_entry(trace_cfg, trace_key)
        entry["base"] = float(value)
        applied["truth"][trace_key] = {
            "field": "base",
            "value": float(value),
            "source": f"--{cli_name.replace('_', '-')}",
        }

    for cli_name, value in jitter_overrides.items():
        trace_key = TRACE_JITTER_OVERRIDE_KEYS[cli_name]
        entry = _trace_plan_entry(trace_cfg, trace_key)
        effects = entry.get("effects")
        if not isinstance(effects, list):
            raise ValueError(
                f"Trace jitter override requested for {trace_key!r}, but "
                f"experiment.trace.plan.{trace_key}.effects is not a list."
            )
        jitter_effect = None
        jitter_field = None
        for effect in effects:
            if not isinstance(effect, dict):
                continue
            if effect.get("kind") == "iid_jitter":
                jitter_effect = effect
                jitter_field = "sigma"
                break
            if effect.get("kind") == "random_walk":
                jitter_effect = effect
                jitter_field = "sigma_step"
                break
        if jitter_effect is None or jitter_field is None:
            raise ValueError(
                f"Trace jitter override requested for {trace_key!r}, but no "
                "iid_jitter or random_walk effect exists in the trace template."
            )
        jitter_effect[jitter_field] = float(value)
        applied["jitter"][trace_key] = {
            "effect_kind": str(jitter_effect.get("kind")),
            "field": jitter_field,
            "value": float(value),
            "source": f"--{cli_name.replace('_', '-')}",
        }

    if seed is not None:
        experiment_cfg = _ensure_mapping(trace_cfg, "experiment", path="root")
        experiment_cfg["seed"] = int(seed)
        applied["seed"] = {"value": int(seed), "source": "--trace-seed"}
    return applied


def _apply_inference_init_overrides(
    inference_cfg: dict[str, Any],
    *,
    init_overrides: Mapping[str, float],
) -> dict[str, Any]:
    """Patch narrow registration-init controls into an inference template copy.

    Policy notes:
    - Starts from the selected inference template.
    - Applies only explicit ``--init-*`` CLI overrides.
    - Does not change active keys or frame-init mode; those remain
      template-owned.
    """

    applied: dict[str, Any] = {}
    if not init_overrides:
        return applied
    experiment_cfg = _ensure_mapping(inference_cfg, "experiment", path="root")
    inference_block = _ensure_mapping(experiment_cfg, "inference", path="experiment")
    init_cfg = _ensure_mapping(inference_block, "init", path="experiment.inference")
    frame_cfg = _ensure_mapping(init_cfg, "frame", path="experiment.inference.init")
    values_cfg = _ensure_mapping(
        frame_cfg,
        "values",
        path="experiment.inference.init.frame",
    )
    active_cfg = _ensure_mapping(inference_block, "active", path="experiment.inference")
    frame_keys = set(str(key) for key in active_cfg.get("frame_keys", ()))
    for cli_name, value in init_overrides.items():
        init_key = INFERENCE_INIT_OVERRIDE_KEYS[cli_name]
        if init_key not in frame_keys:
            raise ValueError(
                f"Inference init override requested for {init_key!r}, but that key "
                "is not listed in experiment.inference.active.frame_keys."
            )
        values_cfg[init_key] = float(value)
        applied[init_key] = {
            "field": "experiment.inference.init.frame.values",
            "value": float(value),
            "source": f"--{cli_name.replace('_', '-')}",
        }
    return applied


def _has_nested_key(root: Mapping[str, Any], dotted_path: str) -> bool:
    """Return whether one dotted mapping path exists."""

    current: Any = root
    for segment in dotted_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return False
        current = current[segment]
    return True


def _field_source(
    cfg: Mapping[str, Any],
    dotted_path: str,
    *,
    cli_override: bool = False,
    generated_patch: bool = False,
) -> str:
    """Classify the source of one generated-config value."""

    if cli_override:
        return SOURCE_CLI_OVERRIDE
    if generated_patch:
        return SOURCE_GENERATED_CONFIG_PATCH
    if _has_nested_key(cfg, dotted_path):
        return SOURCE_INFERENCE_TEMPLATE
    return SOURCE_INFERENCE_RECIPE_DEFAULT


def _apply_reference_diagnostics_profile(
    diagnostics_cfg: dict[str, Any],
    *,
    profile: str | None,
) -> dict[str, str]:
    """Patch diagnostics from a named CLI profile.

    Policy notes:
    - When ``profile`` is ``None``, diagnostics remain template-owned.
    - A named profile intentionally overrides matching template values because
      it is an explicit CLI review request.
    - The returned mapping records each patched field as ``cli_override``.
    """

    if profile is None:
        return {}
    if profile not in SCHUR_REFERENCE_DIAGNOSTICS_PROFILES:
        raise ValueError(
            "reference_diagnostics_profile must be one of: "
            + ", ".join(sorted(SCHUR_REFERENCE_DIAGNOSTICS_PROFILES))
        )
    sources: dict[str, str] = {}
    for key, value in SCHUR_REFERENCE_DIAGNOSTICS_PROFILES[profile].items():
        diagnostics_cfg[key] = bool(value)
        sources[key] = SOURCE_CLI_OVERRIDE
    return sources


def parse_reference_optimizer_kwargs(raw_values: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeatable ``KEY=VALUE`` optimizer kwargs from the CLI."""

    if not raw_values:
        return {}
    parsed: dict[str, Any] = {}
    for raw in raw_values:
        text = str(raw).strip()
        if "=" not in text:
            raise ValueError(
                "reference optimizer kwargs must use KEY=VALUE syntax; "
                f"received {raw!r}."
            )
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("reference optimizer kwargs must use non-empty keys.")
        parsed[key] = value.strip()
    return parsed


def parse_csv_ints(raw: str | None, *, field_name: str) -> tuple[int, ...] | None:
    """Parse a comma-separated integer list used by schedule CLI flags."""

    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"{field_name} must be a comma-separated list of integers.")
    values: list[int] = []
    for index, part in enumerate(parts):
        numeric = coerce_numeric_value(
            part,
            path=f"{field_name}[{index}]",
            finite_only=True,
        )
        integer = int(numeric)
        if float(integer) != float(numeric):
            raise ValueError(f"{field_name}[{index}] must be an integer.")
        values.append(integer)
    return tuple(values)


def parse_csv_floats(raw: str | None, *, field_name: str) -> tuple[float, ...] | None:
    """Parse a comma-separated float list used by schedule CLI flags."""

    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if not parts or any(part == "" for part in parts):
        raise ValueError(f"{field_name} must be a comma-separated list of floats.")
    return tuple(
        float(
            coerce_numeric_value(
                part,
                path=f"{field_name}[{index}]",
                finite_only=True,
            )
        )
        for index, part in enumerate(parts)
    )


def parse_reference_schedule_config(
    *,
    kind: str | None,
    warmup_steps: int | None,
    start_factor: float | None,
    min_factor: float | None,
    boundaries: str | None,
    factors: str | None,
    decay_rate: float | None,
    transition_steps: int | None,
    staircase: bool,
) -> dict[str, Any] | None:
    """Build an optional recovered-reference optimizer schedule override."""

    provided_without_kind = {
        "warmup_steps": warmup_steps,
        "start_factor": start_factor,
        "min_factor": min_factor,
        "boundaries": boundaries,
        "factors": factors,
        "decay_rate": decay_rate,
        "transition_steps": transition_steps,
        "staircase": staircase,
    }
    if kind is None:
        if any(value not in {None, False, ""} for value in provided_without_kind.values()):
            raise ValueError(
                "reference schedule fields require --reference-schedule-kind."
            )
        return None

    schedule_kind = str(kind).strip().lower()
    schedule: dict[str, Any] = {"kind": schedule_kind}

    if schedule_kind == "constant":
        return schedule
    if schedule_kind == "linear_warmup":
        if warmup_steps is None:
            raise ValueError(
                "linear_warmup requires --reference-schedule-warmup-steps."
            )
        if start_factor is None:
            raise ValueError(
                "linear_warmup requires --reference-schedule-start-factor."
            )
        schedule["warmup_steps"] = int(warmup_steps)
        schedule["start_factor"] = float(start_factor)
        return schedule
    if schedule_kind == "piecewise_constant":
        parsed_boundaries = parse_csv_ints(
            boundaries,
            field_name="--reference-schedule-boundaries",
        )
        parsed_factors = parse_csv_floats(
            factors,
            field_name="--reference-schedule-factors",
        )
        if parsed_boundaries is None or parsed_factors is None:
            raise ValueError(
                "piecewise_constant requires both --reference-schedule-boundaries "
                "and --reference-schedule-factors."
            )
        schedule["boundaries"] = list(parsed_boundaries)
        schedule["factors"] = list(parsed_factors)
        return schedule
    if schedule_kind == "exponential_decay":
        if decay_rate is None or transition_steps is None:
            raise ValueError(
                "exponential_decay requires --reference-schedule-decay-rate and "
                "--reference-schedule-transition-steps."
            )
        schedule["decay_rate"] = float(decay_rate)
        schedule["transition_steps"] = int(transition_steps)
        if staircase:
            schedule["staircase"] = True
        return schedule
    if schedule_kind == "cosine_decay":
        if min_factor is None:
            raise ValueError("cosine_decay requires --reference-schedule-min-factor.")
        schedule["min_factor"] = float(min_factor)
        return schedule
    if schedule_kind == "linear_warmup_cosine_decay":
        if warmup_steps is None:
            raise ValueError(
                "linear_warmup_cosine_decay requires --reference-schedule-warmup-steps."
            )
        if start_factor is None:
            raise ValueError(
                "linear_warmup_cosine_decay requires --reference-schedule-start-factor."
            )
        if min_factor is None:
            raise ValueError(
                "linear_warmup_cosine_decay requires --reference-schedule-min-factor."
            )
        schedule["warmup_steps"] = int(warmup_steps)
        schedule["start_factor"] = float(start_factor)
        schedule["min_factor"] = float(min_factor)
        return schedule

    raise ValueError(
        "reference_schedule_kind must be one of: constant, linear_warmup, "
        "piecewise_constant, exponential_decay, cosine_decay, "
        "linear_warmup_cosine_decay."
    )


def parse_reference_preconditioning_lr_clip(raw: str | None) -> tuple[float, float] | None:
    """Parse ``MIN,MAX`` learning-rate clip bounds for reference preconditioning."""

    if raw is None:
        return None
    parts = [part.strip() for part in str(raw).split(",")]
    if len(parts) != 2 or not all(parts):
        raise ValueError("--reference-preconditioning-lr-clip must be MIN,MAX.")
    lr_min = float(
        coerce_numeric_value(
            parts[0],
            path="--reference-preconditioning-lr-clip[0]",
            must_be_positive=True,
        )
    )
    lr_max = float(
        coerce_numeric_value(
            parts[1],
            path="--reference-preconditioning-lr-clip[1]",
            must_be_positive=True,
        )
    )
    if lr_max < lr_min:
        raise ValueError("--reference-preconditioning-lr-clip max must be >= min.")
    return (lr_min, lr_max)


def apply_reference_optimizer_overrides(
    inference_cfg: dict[str, Any],
    *,
    optimizer_kind: str | None = None,
    base_lr: float | None = None,
    n_iter: int | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
    preconditioning_enabled: bool | None = None,
    preconditioning_method: str | None = None,
    preconditioning_reference: str | None = None,
    preconditioning_damping: float | None = None,
    preconditioning_eig_floor_rel: float | None = None,
    preconditioning_eig_floor_abs: float | None = None,
    preconditioning_lr_clip: tuple[float, float] | None = None,
) -> dict[str, str]:
    """Patch explicit recovered-reference optimizer overrides into config.

    Called only by the Schur-summary study config builder.  Omitted values are
    left untouched so template-owned optimizer settings remain authoritative.
    Returned source labels use ``SOURCE_CLI_OVERRIDE`` for every patched field.
    """

    optimizer_cfg = _ensure_mapping(
        inference_cfg,
        "optimizer",
        path="experiment.inference",
    )
    current_kind = str(optimizer_cfg.get("kind", "adam")).strip().lower()
    sources: dict[str, str] = {}

    if optimizer_kind is not None:
        normalized_kind = str(optimizer_kind).strip().lower()
        if normalized_kind not in {"sgd", "adam"}:
            raise ValueError("reference_optimizer_kind must be 'sgd' or 'adam'.")
        optimizer_cfg["kind"] = normalized_kind
        current_kind = normalized_kind
        sources["optimizer_kind"] = SOURCE_CLI_OVERRIDE

    if base_lr is not None:
        optimizer_cfg["base_lr"] = float(
            coerce_numeric_value(
                base_lr,
                path="reference_base_lr",
                must_be_positive=True,
            )
        )
        sources["base_lr"] = SOURCE_CLI_OVERRIDE

    if n_iter is not None:
        n_iter_int = int(n_iter)
        if n_iter_int <= 0:
            raise ValueError("reference_n_iter must be > 0.")
        optimizer_cfg["n_iter"] = n_iter_int
        sources["n_iter"] = SOURCE_CLI_OVERRIDE

    if optimizer_kwargs:
        optimizer_cfg["kwargs"] = normalize_optimizer_kwargs(
            current_kind,
            optimizer_kwargs,
            path="reference_optimizer_kwargs",
        )
        sources["optimizer_kwargs"] = SOURCE_CLI_OVERRIDE

    if schedule is not None:
        schedule_n_iter = int(optimizer_cfg.get("n_iter", 100))
        optimizer_cfg["schedule"] = validate_optimizer_schedule_config(
            dict(schedule),
            n_iter=schedule_n_iter,
            path="reference_schedule",
        )
        sources["schedule"] = SOURCE_CLI_OVERRIDE

    if (
        preconditioning_enabled is not None
        or preconditioning_method is not None
        or preconditioning_reference is not None
        or preconditioning_damping is not None
        or preconditioning_eig_floor_rel is not None
        or preconditioning_eig_floor_abs is not None
        or preconditioning_lr_clip is not None
    ):
        preconditioning_cfg = _ensure_mapping(
            optimizer_cfg,
            "preconditioning",
            path="experiment.inference.optimizer",
        )
        if preconditioning_enabled is not None:
            preconditioning_cfg["enabled"] = bool(preconditioning_enabled)
            sources["preconditioning_enabled"] = SOURCE_CLI_OVERRIDE
        if preconditioning_method is not None:
            recipe = _load_inference_recipe_module()
            preconditioning_cfg["method"] = recipe._normalize_preconditioning_method(
                preconditioning_method
            )
            sources["preconditioning_method"] = SOURCE_CLI_OVERRIDE
        if preconditioning_reference is not None:
            if preconditioning_reference not in {"truth_when_available", "initial"}:
                raise ValueError(
                    "reference_preconditioning_reference must be "
                    "'truth_when_available' or 'initial'."
                )
            preconditioning_cfg["reference"] = str(preconditioning_reference)
            sources["preconditioning_reference"] = SOURCE_CLI_OVERRIDE
        if preconditioning_damping is not None:
            preconditioning_cfg["damping"] = float(
                coerce_numeric_value(
                    preconditioning_damping,
                    path="reference_preconditioning_damping",
                    must_be_nonnegative=True,
                )
            )
            sources["preconditioning_damping"] = SOURCE_CLI_OVERRIDE
        if preconditioning_eig_floor_rel is not None:
            preconditioning_cfg["eig_floor_rel"] = float(
                coerce_numeric_value(
                    preconditioning_eig_floor_rel,
                    path="reference_preconditioning_eig_floor_rel",
                    must_be_nonnegative=True,
                )
            )
            sources["preconditioning_eig_floor_rel"] = SOURCE_CLI_OVERRIDE
        if preconditioning_eig_floor_abs is not None:
            preconditioning_cfg["eig_floor_abs"] = float(
                coerce_numeric_value(
                    preconditioning_eig_floor_abs,
                    path="reference_preconditioning_eig_floor_abs",
                    must_be_nonnegative=True,
                )
            )
            sources["preconditioning_eig_floor_abs"] = SOURCE_CLI_OVERRIDE
        if preconditioning_lr_clip is not None:
            preconditioning_cfg["lr_clip"] = [
                float(preconditioning_lr_clip[0]),
                float(preconditioning_lr_clip[1]),
            ]
            sources["preconditioning_lr_clip"] = SOURCE_CLI_OVERRIDE

    return sources


def _reference_optimizer_override_sources(
    *,
    optimizer_kind: str | None = None,
    base_lr: float | None = None,
    n_iter: int | None = None,
    optimizer_kwargs: Mapping[str, Any] | None = None,
    schedule: Mapping[str, Any] | None = None,
    preconditioning_enabled: bool | None = None,
    preconditioning_method: str | None = None,
    preconditioning_reference: str | None = None,
    preconditioning_damping: float | None = None,
    preconditioning_eig_floor_rel: float | None = None,
    preconditioning_eig_floor_abs: float | None = None,
    preconditioning_lr_clip: tuple[float, float] | None = None,
) -> dict[str, str]:
    """Return provenance labels for explicitly requested optimizer overrides."""

    raw = {
        "optimizer_kind": optimizer_kind,
        "base_lr": base_lr,
        "n_iter": n_iter,
        "optimizer_kwargs": optimizer_kwargs if optimizer_kwargs else None,
        "schedule": dict(schedule) if schedule is not None else None,
        "preconditioning_enabled": preconditioning_enabled,
        "preconditioning_method": preconditioning_method,
        "preconditioning_reference": preconditioning_reference,
        "preconditioning_damping": preconditioning_damping,
        "preconditioning_eig_floor_rel": preconditioning_eig_floor_rel,
        "preconditioning_eig_floor_abs": preconditioning_eig_floor_abs,
        "preconditioning_lr_clip": preconditioning_lr_clip,
    }
    return {key: SOURCE_CLI_OVERRIDE for key, value in raw.items() if value is not None}


def _build_schur_config_provenance(
    *,
    schur_config: Mapping[str, Any],
    reference_preconditioning_enabled: bool | None,
    reference_preconditioning_reference: str | None,
    reference_diagnostics_profile: str | None,
    force_truth_comparison: bool,
    reference_optimizer_sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build compact source labels for policy-sensitive inference config fields."""

    base = "experiment.inference"
    diagnostics_profile_fields = (
        set(SCHUR_REFERENCE_DIAGNOSTICS_PROFILES[reference_diagnostics_profile])
        if reference_diagnostics_profile is not None
        else set()
    )
    optimizer_override_sources = dict(reference_optimizer_sources or {})
    diagnostics_fields = (
        "plots",
        "compare_to_truth_when_available",
        "first_step_report",
        "save_first_step_json",
        "save_fim_debug",
        "finite_difference_check",
        "plot_parameter_history_heatmap",
        "plot_parameter_residual_history_heatmap",
        "plot_parameter_history_lines",
        "plot_parameter_residual_history_lines",
        "top_k",
    )
    diagnostics_sources = {
        key: _field_source(
            schur_config,
            f"{base}.diagnostics.{key}",
            cli_override=key in diagnostics_profile_fields,
            generated_patch=(
                key == "compare_to_truth_when_available"
                and bool(force_truth_comparison)
                and key not in diagnostics_profile_fields
            ),
        )
        for key in diagnostics_fields
    }
    return {
        "optimizer": {
            "optimizer_kind": _field_source(
                schur_config,
                f"{base}.optimizer.kind",
                cli_override=optimizer_override_sources.get("optimizer_kind")
                == SOURCE_CLI_OVERRIDE,
            ),
            "base_lr": _field_source(
                schur_config,
                f"{base}.optimizer.base_lr",
                cli_override=optimizer_override_sources.get("base_lr")
                == SOURCE_CLI_OVERRIDE,
            ),
            "n_iter": _field_source(
                schur_config,
                f"{base}.optimizer.n_iter",
                cli_override=optimizer_override_sources.get("n_iter")
                == SOURCE_CLI_OVERRIDE,
            ),
            "optimizer_kwargs": _field_source(
                schur_config,
                f"{base}.optimizer.kwargs",
                cli_override=optimizer_override_sources.get("optimizer_kwargs")
                == SOURCE_CLI_OVERRIDE,
            ),
            "schedule": (
                SOURCE_CLI_OVERRIDE
                if optimizer_override_sources.get("schedule") == SOURCE_CLI_OVERRIDE
                else (
                    SOURCE_INFERENCE_TEMPLATE
                    if _has_nested_key(schur_config, f"{base}.optimizer.schedule")
                    else TEMPLATE_OWNED_DEFAULT
                )
            ),
        },
        "preconditioning": {
            "preconditioning_enabled": _field_source(
                schur_config,
                f"{base}.optimizer.preconditioning.enabled",
                cli_override=(
                    reference_preconditioning_enabled is not None
                    or optimizer_override_sources.get("preconditioning_enabled")
                    == SOURCE_CLI_OVERRIDE
                ),
            ),
            "preconditioning_method": _field_source(
                schur_config,
                f"{base}.optimizer.preconditioning.method",
                cli_override=optimizer_override_sources.get("preconditioning_method")
                == SOURCE_CLI_OVERRIDE,
            ),
            "preconditioning_reference": _field_source(
                schur_config,
                f"{base}.optimizer.preconditioning.reference",
                cli_override=(
                    reference_preconditioning_reference is not None
                    or optimizer_override_sources.get("preconditioning_reference")
                    == SOURCE_CLI_OVERRIDE
                ),
            ),
            "preconditioning_damping": _field_source(
                schur_config,
                f"{base}.optimizer.preconditioning.damping",
                cli_override=optimizer_override_sources.get("preconditioning_damping")
                == SOURCE_CLI_OVERRIDE,
            ),
            "preconditioning_eig_floor_rel": _field_source(
                schur_config,
                f"{base}.optimizer.preconditioning.eig_floor_rel",
                cli_override=optimizer_override_sources.get("preconditioning_eig_floor_rel")
                == SOURCE_CLI_OVERRIDE,
            ),
            "preconditioning_eig_floor_abs": _field_source(
                schur_config,
                f"{base}.optimizer.preconditioning.eig_floor_abs",
                cli_override=optimizer_override_sources.get("preconditioning_eig_floor_abs")
                == SOURCE_CLI_OVERRIDE,
            ),
            "preconditioning_lr_clip": _field_source(
                schur_config,
                f"{base}.optimizer.preconditioning.lr_clip",
                cli_override=optimizer_override_sources.get("preconditioning_lr_clip")
                == SOURCE_CLI_OVERRIDE,
            ),
        },
        "diagnostics": diagnostics_sources,
        "diagnostics_profile": (
            SOURCE_CLI_OVERRIDE
            if reference_diagnostics_profile is not None
            else SOURCE_NOT_APPLICABLE
        ),
    }


def _resolve_template_system_context(template_path: Path) -> dict[str, Any]:
    """Resolve one template's system block and default store for candidate work."""

    user_cfg = load_user_config(
        config_path=template_path.resolve(),
        system_preset=None,
        experiment_preset=None,
    )
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    if not isinstance(system_cfg, dict):
        raise ValueError(f"Template {template_path} must resolve a system mapping.")
    forward_spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    store, system_store_overlay = _apply_resolved_system_observation_values_to_store(
        store=store,
        forward_spec=forward_spec,
        system_cfg=system_cfg,
    )
    return {
        "resolved_cfg": resolved_cfg,
        "system_cfg": system_cfg,
        "forward_spec": forward_spec,
        "store": store,
        "system_store_overlay": system_store_overlay,
    }


def _resolve_candidate_mapping_value(
    mapping: dict[str, Any] | None,
    *,
    candidate_key: str,
) -> float | None:
    """Read one scalar-or-indexed candidate value from a nested mapping."""

    return get_obs_subblock_mapping_value(
        mapping,
        address=parse_obs_subblock_key_address(candidate_key),
    )


def _system_config_reference_addresses(
    *,
    system_cfg: Mapping[str, Any],
    reference_store: ParameterStore,
) -> tuple[ObsSubblockKeyAddress, ...]:
    """Return observation-level resolved-system keys explicitly present in config."""

    addresses: list[ObsSubblockKeyAddress] = []
    for key in OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS:
        address = parse_obs_subblock_key_address(key)
        if get_obs_subblock_mapping_value(system_cfg, address=address) is not None:
            addresses.append(address)
    for base_key in OBS_SUBBLOCK_SUPPORTED_INDEXED_KEYS:
        vector_value = reference_store.get(base_key, None)
        if vector_value is None:
            continue
        array_value = np.asarray(vector_value)
        if array_value.ndim != 1:
            continue
        for index in range(int(array_value.shape[0])):
            address = ObsSubblockKeyAddress(base_key=base_key, index=index)
            if get_obs_subblock_mapping_value(system_cfg, address=address) is not None:
                addresses.append(address)
    return tuple(addresses)


def _apply_resolved_system_observation_values_to_store(
    *,
    store: ParameterStore,
    forward_spec: Any,
    system_cfg: Mapping[str, Any],
) -> tuple[ParameterStore, dict[str, Any]]:
    """Overlay explicit resolved-config observation values after derived refresh.

    Three-plane plate scale is represented as a derived store key, so a plain
    ``refresh_derived(...)`` recomputes it from geometry and can erase a
    deliberate inference/reference override even though the resolved config
    contains the biased value. Observation-subblock workflows treat explicit
    resolved config values for supported observation-level keys as the fixed
    runtime reference state; this overlay is therefore applied after refresh.
    """

    addresses = _system_config_reference_addresses(
        system_cfg=system_cfg,
        reference_store=store,
    )
    if not addresses:
        return store, {"applied": False, "items": []}

    validate_supported_obs_subblock_key_addresses(
        addresses,
        forward_spec=forward_spec,
        reference_store=store,
    )
    overrides: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    for address in addresses:
        config_value = get_obs_subblock_mapping_value(system_cfg, address=address)
        if config_value is None:
            continue
        store_value_before = get_obs_subblock_store_value(store, address=address)
        if address.index is None:
            overrides[address.base_key] = float(config_value)
        else:
            vector_value = jnp.asarray(
                overrides.get(address.base_key, store.get(address.base_key))
            )
            overrides[address.base_key] = vector_value.at[address.index].set(
                float(config_value)
            )
        items.append(
            {
                "key": address.canonical,
                "config_value": float(config_value),
                "store_value_before_overlay": float(store_value_before),
                "absolute_delta": float(config_value - store_value_before),
            }
        )

    if not overrides:
        return store, {"applied": False, "items": []}
    updated = apply_obs_subblock_runtime_overrides_without_refresh(
        store,
        overrides_flat=overrides,
        forward_spec=forward_spec,
    )
    for item in items:
        address = parse_obs_subblock_key_address(str(item["key"]))
        item["store_value_after_overlay"] = get_obs_subblock_store_value(
            updated,
            address=address,
        )
    return updated, {"applied": True, "items": items}


def _set_candidate_mapping_value(
    mapping: dict[str, Any],
    *,
    candidate_key: str,
    value: float,
    reference_store: ParameterStore | None = None,
) -> None:
    """Patch one scalar-or-indexed candidate value into a nested mapping."""

    address = parse_obs_subblock_key_address(candidate_key)
    reference_vector = None
    if address.index is not None and reference_store is not None:
        reference_vector = np.asarray(reference_store.get(address.base_key), dtype=float)
    set_obs_subblock_mapping_value(
        mapping,
        address=address,
        value=value,
        reference_vector=reference_vector,
    )


def _disabled_theta_reference_overrides_payload() -> dict[str, Any]:
    return {"enabled": False, "items": []}


def resolve_theta_reference_overrides(
    *,
    inference_config: dict[str, Any],
    theta_reference_offsets: Mapping[str, float] | None = None,
    theta_reference_values: Mapping[str, float] | None = None,
    render_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Patch biased Theta values into the inference/reference config only."""

    offsets = dict(theta_reference_offsets or {})
    values = dict(theta_reference_values or {})
    if not offsets and not values:
        return _disabled_theta_reference_overrides_payload()

    offset_addresses = parse_obs_subblock_varying_keys(tuple(offsets))
    value_addresses = parse_obs_subblock_varying_keys(tuple(values))
    by_key_offset = {address.canonical: address for address in offset_addresses}
    by_key_value = {address.canonical: address for address in value_addresses}
    conflicts = sorted(set(by_key_offset) & set(by_key_value))
    if conflicts:
        raise ValueError(
            "Cannot specify both --theta-reference-offset and "
            "--theta-reference-value for: "
            + ", ".join(conflicts)
            + "."
        )

    resolved_cfg = resolve_config(inference_config)
    system_cfg = resolved_cfg.get("system")
    if not isinstance(system_cfg, dict):
        raise ValueError("Theta reference overrides require a resolved system block.")
    forward_spec = compose_forward_spec(system_cfg)
    reference_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(
        forward_spec
    )
    reference_store, _ = _apply_resolved_system_observation_values_to_store(
        store=reference_store,
        forward_spec=forward_spec,
        system_cfg=system_cfg,
    )
    all_addresses = tuple(by_key_offset.values()) + tuple(by_key_value.values())
    validate_supported_obs_subblock_key_addresses(
        all_addresses,
        forward_spec=forward_spec,
        reference_store=reference_store,
    )

    config_system_cfg = _ensure_mapping(inference_config, "system", path="root")
    items: list[dict[str, Any]] = []
    for canonical in [*by_key_offset, *by_key_value]:
        address = by_key_offset.get(canonical) or by_key_value[canonical]
        base_value = get_obs_subblock_store_value(reference_store, address=address)
        truth_value = _truth_value_from_render_manifest(render_manifest_path, canonical)
        if canonical in by_key_offset:
            offset = float(offsets[canonical])
            reference_value = float(base_value + offset)
            mode = "offset"
        else:
            offset = None
            reference_value = float(values[canonical])
            mode = "value"

        _set_candidate_mapping_value(
            config_system_cfg,
            candidate_key=canonical,
            value=reference_value,
            reference_store=reference_store,
        )
        items.append(
            {
                "key": canonical,
                "mode": mode,
                "truth_value": None if truth_value is None else float(truth_value),
                "reference_base_value": float(base_value),
                "offset": None if offset is None else float(offset),
                "reference_value": float(reference_value),
                "applied_to": "inference_reference_only",
            }
        )

    return {"enabled": True, "items": items}


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


def resolve_study_trace_template(
    *,
    mode: str,
    trace_template: Path | None,
) -> tuple[Path, str]:
    """Resolve the source trace template and provenance for one study mode.

    Policy notes:
    - Explicit ``--trace-template`` wins for every mode.
    - ``schur_summary`` otherwise uses the registration-iid Schur template.
    - Older study modes otherwise use the general trace template.
    """

    if trace_template is not None:
        return trace_template.expanduser().resolve(), "cli_override"
    if parse_study_mode(mode) == MODE_SCHUR_SUMMARY:
        return DEFAULT_SCHUR_TRACE_TEMPLATE.resolve(), "schur_summary_default"
    return DEFAULT_TRACE_TEMPLATE.resolve(), "general_default"


def _truth_value_from_render_manifest(manifest_path: Path | None, candidate_key: str) -> float | None:
    """Resolve the rendered truth-side candidate value when available."""

    if not candidate_key:
        return None
    if manifest_path is None or not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    shared_truth = manifest.get("shared_truth")
    truth_value = _resolve_candidate_mapping_value(
        shared_truth if isinstance(shared_truth, dict) else None,
        candidate_key=candidate_key,
    )
    if truth_value is not None:
        return truth_value
    system_payload = manifest.get("system")
    if not isinstance(system_payload, dict):
        return None
    resolved_cfg = system_payload.get("resolved_config")
    if not isinstance(resolved_cfg, dict):
        return None
    return _resolve_candidate_mapping_value(resolved_cfg, candidate_key=candidate_key)


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
    trace_truth_overrides: Mapping[str, float] | None = None,
    trace_jitter_overrides: Mapping[str, float] | None = None,
    trace_seed: int | None = None,
    inference_init_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Write study-local template copies with narrow mode-specific patching.

    Policy notes:
    - Template copies are the first durable run record.
    - Candidate truth/assumed patches are study-mode specific.
    - Schur trace/init CLI patches are explicit and recorded.
    - Schur diagnostics are not silently rewritten here; recovered-reference
      diagnostics are handled by the final generated inference config helper.
    """

    templates_dir = _study_templates_dir(case_root, mode)
    templates_dir.mkdir(parents=True, exist_ok=True)

    trace_cfg = load_config_file(trace_template)
    render_cfg = load_config_file(render_template)
    inference_cfg = load_config_file(inference_template)
    applied_trace_overrides = _apply_trace_truth_overrides(
        trace_cfg,
        truth_overrides=trace_truth_overrides or {},
        jitter_overrides=trace_jitter_overrides or {},
        seed=trace_seed,
    )
    applied_inference_init_overrides = _apply_inference_init_overrides(
        inference_cfg,
        init_overrides=inference_init_overrides or {},
    )
    trace_context = None
    render_context = None
    inference_context = None
    candidate_address = None
    if candidate_key is not None:
        trace_context = _resolve_template_system_context(trace_template)
        render_context = _resolve_template_system_context(render_template)
        inference_context = _resolve_template_system_context(inference_template)
        candidate_address = parse_candidate_parameter_address(
            candidate_key,
            forward_spec=inference_context["forward_spec"],
            reference_store=inference_context["store"],
        )
        validate_supported_obs_subblock_key_addresses(
            (candidate_address,),
            forward_spec=trace_context["forward_spec"],
            reference_store=trace_context["store"],
        )
        validate_supported_obs_subblock_key_addresses(
            (candidate_address,),
            forward_spec=render_context["forward_spec"],
            reference_store=render_context["store"],
        )
        validate_supported_obs_subblock_key_addresses(
            (candidate_address,),
            forward_spec=inference_context["forward_spec"],
            reference_store=inference_context["store"],
        )
        candidate_key = candidate_address.canonical

    if candidate_key is not None and truth_value is not None:
        trace_system_cfg = _ensure_mapping(trace_cfg, "system", path="root")
        _set_candidate_mapping_value(
            trace_system_cfg,
            candidate_key=candidate_key,
            value=truth_value,
            reference_store=None if trace_context is None else trace_context["store"],
        )

        render_experiment_cfg = _ensure_mapping(render_cfg, "experiment", path="root")
        render_truth_cfg = _ensure_mapping(
            render_experiment_cfg,
            "truth",
            path="experiment",
        )
        _set_candidate_mapping_value(
            render_truth_cfg,
            candidate_key=candidate_key,
            value=truth_value,
            reference_store=None if render_context is None else render_context["store"],
        )

    if candidate_key is not None and assumed_value is not None:
        inference_system_cfg = _ensure_mapping(inference_cfg, "system", path="root")
        _set_candidate_mapping_value(
            inference_system_cfg,
            candidate_key=candidate_key,
            value=assumed_value,
            reference_store=None if inference_context is None else inference_context["store"],
        )

    if mode in {
        MODE_FISHER_ONLY,
        MODE_PROFILE_OBJECTIVE,
        MODE_NUISANCE_ABSORPTION,
        MODE_SCHUR_SUMMARY,
    }:
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
        if mode != MODE_SCHUR_SUMMARY:
            diagnostics_cfg["plots"] = True

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
            reference_value = _resolve_candidate_mapping_value(
                inference_cfg.get("system"),
                candidate_key=candidate_key,
            )
        if reference_value is None and candidate_address is not None and inference_context is not None:
            reference_value = get_obs_subblock_store_value(
                inference_context["store"],
                address=candidate_address,
            )
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
        else _resolve_candidate_mapping_value(
            inference_cfg.get("system"),
            candidate_key=candidate_key,
        )
    )
    if (
        resolved_assumed is None
        and candidate_address is not None
        and inference_context is not None
    ):
        resolved_assumed = get_obs_subblock_store_value(
            inference_context["store"],
            address=candidate_address,
        )
    resolved_truth = truth_value
    if candidate_key is not None and resolved_truth is None:
        render_truth_cfg = render_cfg.get("experiment", {}).get("truth")
        if isinstance(render_truth_cfg, dict):
            resolved_truth = _resolve_candidate_mapping_value(
                render_truth_cfg,
                candidate_key=candidate_key,
            )
        if resolved_truth is None:
            resolved_truth = _resolve_candidate_mapping_value(
                render_cfg.get("system"),
                candidate_key=candidate_key,
            )
        if (
            resolved_truth is None
            and candidate_address is not None
            and render_context is not None
        ):
            resolved_truth = get_obs_subblock_store_value(
                render_context["store"],
                address=candidate_address,
            )

    return {
        "paths": {
            "trace": trace_path,
            "render": render_path,
            "inference": inference_path,
        },
        "source_template_paths": {
            "trace": trace_template.resolve(),
            "render": render_template.resolve(),
            "inference": inference_template.resolve(),
        },
        "resolved_truth_value": resolved_truth,
        "resolved_assumed_value": resolved_assumed,
        "resolved_target_name": (
            _resolve_target_name(inference_cfg.get("system"))
            or _resolve_target_name(render_cfg.get("system"))
            or _resolve_target_name(trace_cfg.get("system"))
        ),
        "applied_overrides": {
            "trace": applied_trace_overrides,
            "inference_init": applied_inference_init_overrides,
        },
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
    render_seed: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ensure the case has render-ready artifacts for a screening study.

    Policy notes:
    - Uses the already selected case-local trace/render templates.
    - Dry runs plan missing trace/render stages but do not create image data.
    - Actual runs delegate mechanics to ``run_obs_subblock_case.py``.
    """

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
                render_seed=render_seed,
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
    base_store, system_store_overlay = _apply_resolved_system_observation_values_to_store(
        store=base_store,
        forward_spec=forward_spec,
        system_cfg=system_cfg,
    )
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
        "system_store_overlay": system_store_overlay,
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
        **_candidate_metadata(candidate_key),
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
    variance_floor: float | None = None,
    reference_optimizer_kind: str | None = None,
    reference_base_lr: float | None = None,
    reference_n_iter: int | None = None,
    reference_optimizer_kwargs: Mapping[str, Any] | None = None,
    reference_schedule: Mapping[str, Any] | None = None,
    reference_preconditioning_enabled: bool | None = None,
    reference_preconditioning_method: str | None = None,
    reference_preconditioning_reference: str | None = None,
    reference_preconditioning_damping: float | None = None,
    reference_preconditioning_eig_floor_rel: float | None = None,
    reference_preconditioning_eig_floor_abs: float | None = None,
    reference_preconditioning_lr_clip: tuple[float, float] | None = None,
    reference_diagnostics_profile: str | None = None,
) -> dict[str, Any]:
    """Build one run-specific inference config for study-mode execution.

    Policy notes:
    - Starts from the selected inference template via the case runner.
    - Applies generated path patches for cube, truth trace, manifest, and
      output directories.
    - Does not silently override optimizer or preconditioning template values.
    - Explicit reference preconditioning CLI overrides intentionally patch the
      generated config and win over template values.
    - A reference diagnostics profile intentionally patches matching
      diagnostics values; otherwise diagnostics remain template-owned except
      for generated truth-comparison requests.
    """

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
        template_context = _resolve_template_system_context(template_path)
        _set_candidate_mapping_value(
            system_cfg,
            candidate_key=candidate_key,
            value=assumed_value,
            reference_store=template_context["store"],
        )

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
    _apply_reference_diagnostics_profile(
        diagnostics_cfg,
        profile=reference_diagnostics_profile,
    )
    if variance_floor is not None:
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
        noise_model_cfg["variance_floor"] = float(variance_floor)

    apply_reference_optimizer_overrides(
        inference_cfg,
        optimizer_kind=reference_optimizer_kind,
        base_lr=reference_base_lr,
        n_iter=reference_n_iter,
        optimizer_kwargs=reference_optimizer_kwargs,
        schedule=reference_schedule,
        preconditioning_enabled=reference_preconditioning_enabled,
        preconditioning_method=reference_preconditioning_method,
        preconditioning_reference=reference_preconditioning_reference,
        preconditioning_damping=reference_preconditioning_damping,
        preconditioning_eig_floor_rel=reference_preconditioning_eig_floor_rel,
        preconditioning_eig_floor_abs=reference_preconditioning_eig_floor_abs,
        preconditioning_lr_clip=reference_preconditioning_lr_clip,
    )

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
        noise_model_cfg.pop("variance_floor", None)
        noise_model_cfg["path"] = case_module._path_for_config(
            variance_path,
            config_dir=run_root,
        )
    return cfg


def _phi_labels_for_active_layout(recipe: Any, active_layout: Any) -> tuple[str, ...]:
    """Return explicit phi labels in the packed active-state order."""

    return tuple(f"phi.{label}" for label in recipe._theta_labels_for_layout(active_layout))


def _theta_labels_for_observation_layout(theta_layout: ObservationThetaLayout) -> tuple[str, ...]:
    """Return explicit observation-level labels in packed Theta order."""

    return tuple(f"theta.{label}" for label in theta_layout.labels)


def _theta_addresses_for_layout(
    *,
    theta_layout: ObservationThetaLayout,
    forward_spec: Any,
    base_store: ParameterStore,
) -> tuple[ObsSubblockKeyAddress, ...]:
    """Resolve packed observation-level labels into validated store addresses."""

    addresses = parse_obs_subblock_varying_keys(theta_layout.labels)
    validate_supported_obs_subblock_key_addresses(
        addresses,
        forward_spec=forward_spec,
        reference_store=base_store,
    )
    return addresses


def _observation_theta_ref_from_store(
    *,
    theta_layout: ObservationThetaLayout,
    base_store: ParameterStore,
) -> np.ndarray:
    """Read the physical-basis Theta reference vector from the resolved store."""

    return np.asarray(
        [
            get_obs_subblock_store_value(base_store, address=parse_obs_subblock_key_address(label))
            for label in theta_layout.labels
        ],
        dtype=float,
    )


def validate_theta_reference_override_consistency(
    *,
    theta_reference_overrides: Mapping[str, Any] | None,
    theta_labels: Sequence[str],
    theta_ref: Sequence[float] | np.ndarray,
    resolved_config: Mapping[str, Any] | None = None,
    store: ParameterStore | None = None,
    prior_context: Mapping[str, Any] | None = None,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-12,
    raise_on_mismatch: bool = True,
) -> dict[str, Any]:
    """Validate that biased reference overrides reached all summary handoffs."""

    payload = dict(theta_reference_overrides or _disabled_theta_reference_overrides_payload())
    items = payload.get("items")
    if not payload.get("enabled") or not isinstance(items, Sequence):
        return {"passed": True, "items": []}

    labels = tuple(theta_labels)
    theta_array = np.asarray(theta_ref, dtype=float)
    if theta_array.shape != (len(labels),):
        raise ValueError(
            "theta_ref shape does not match theta_labels for override consistency "
            f"validation: theta_ref_shape={theta_array.shape}, n_labels={len(labels)}."
        )

    system_mapping: Mapping[str, Any] | None = resolved_config
    if isinstance(resolved_config, Mapping) and isinstance(resolved_config.get("system"), Mapping):
        system_mapping = resolved_config["system"]
    prior_by_label = {}
    if isinstance(prior_context, Mapping):
        raw_prior = prior_context.get("theta_ref_by_label")
        if isinstance(raw_prior, Mapping):
            prior_by_label = dict(raw_prior)

    diagnostics: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for raw_item in items:
        if not isinstance(raw_item, Mapping):
            continue
        key = str(raw_item.get("key", "")).strip()
        if not key:
            continue
        expected = float(raw_item["reference_value"])
        address = parse_obs_subblock_key_address(key)
        resolved_value = (
            None
            if system_mapping is None
            else get_obs_subblock_mapping_value(system_mapping, address=address)
        )
        store_value = (
            None
            if store is None
            else get_obs_subblock_store_value(store, address=address)
        )
        theta_ref_value = None
        if key in labels:
            theta_ref_value = float(theta_array[labels.index(key)])
        prior_value = None
        if key in prior_by_label:
            prior_value = float(prior_by_label[key])

        compared_values = [
            value
            for value in (resolved_value, store_value, theta_ref_value, prior_value)
            if value is not None
        ]
        absolute_error = (
            None
            if not compared_values
            else float(max(abs(float(value) - expected) for value in compared_values))
        )
        relative_error = (
            None
            if absolute_error is None
            else float(absolute_error / max(abs(expected), np.finfo(float).tiny))
        )
        item_mismatches: list[str] = []
        for label, value in (
            ("resolved_config_value", resolved_value),
            ("store_value", store_value),
            ("theta_ref_value", theta_ref_value),
            ("prior_context_value", prior_value),
        ):
            if value is None:
                continue
            if not math.isclose(float(value), expected, rel_tol=rtol, abs_tol=atol):
                item_mismatches.append(
                    f"{key} {label}={float(value)!r} expected={expected!r}"
                )
        if key in labels and theta_ref_value is None:
            item_mismatches.append(f"{key} missing from theta_ref despite label match")
        if item_mismatches:
            mismatches.extend(item_mismatches)

        diagnostics.append(
            {
                "key": key,
                "expected_reference_value": expected,
                "resolved_config_value": None
                if resolved_value is None
                else float(resolved_value),
                "store_value": None if store_value is None else float(store_value),
                "theta_ref_value": None
                if theta_ref_value is None
                else float(theta_ref_value),
                "prior_context_value": None
                if prior_value is None
                else float(prior_value),
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "passed": not item_mismatches,
                "mismatches": item_mismatches,
            }
        )

    consistency = {
        "passed": not mismatches,
        "items": diagnostics,
        "rtol": float(rtol),
        "atol": float(atol),
    }
    if mismatches and raise_on_mismatch:
        raise ValueError(
            "Theta reference override consistency failed: "
            + "; ".join(mismatches)
        )
    return consistency


def _apply_theta_overrides(
    *,
    reference_store: ParameterStore,
    forward_spec: Any,
    theta_addresses: Sequence[ObsSubblockKeyAddress],
    theta_values: jnp.ndarray,
) -> ParameterStore:
    """Apply observation-level Theta values to one resolved system store.

    Notes
    -----
    The traced Schur-summary objective intentionally does *not* call full
    ``ParameterStore.refresh_derived(...)`` here. That broader refresh path is
    fine outside autodiff, but it reaches transform functions that still use
    Python ``float(...)`` coercion for some source-photometry terms. The local
    image-backed inference semantics instead mirror the canonical pack/unpack
    path: active Theta values are authoritative overlays on a resolved base
    store, and only minimal dependent runtime quantities are updated explicitly.
    """

    overrides: dict[str, Any] = {}
    for index, address in enumerate(theta_addresses):
        value = theta_values[index]
        if address.index is None:
            overrides[address.base_key] = value
            continue
        if address.base_key in overrides:
            vector_value = jnp.asarray(overrides[address.base_key])
        else:
            vector_value = jnp.asarray(reference_store.get(address.base_key))
        overrides[address.base_key] = vector_value.at[address.index].set(value)

    return apply_obs_subblock_runtime_overrides_without_refresh(
        reference_store,
        overrides_flat=overrides,
        forward_spec=forward_spec,
    )


def _build_combined_local_objective(
    *,
    context: dict[str, Any],
    theta_layout: ObservationThetaLayout,
    objective_kind: str,
) -> tuple[Any, dict[str, Any]]:
    """Build one local objective over the packed vector ``[Theta, phi]``.

    The slow/shared observation-level ``Theta`` entries are applied to the
    resolved system store first. The packed fast ``phi`` vector is then unpacked
    into the frame-wise registration state used by the existing subblock
    inference recipe. The resulting objective is the image-backed local
    quadratic source for the dense Schur reduction.
    """

    recipe = context["recipe"]
    active_layout = context["layout"]
    binder = context["binder"]
    forward_spec = context["forward_spec"]
    base_store = context["base_store"]
    inference_cfg = context["inference_cfg"]
    theta_addresses = _theta_addresses_for_layout(
        theta_layout=theta_layout,
        forward_spec=forward_spec,
        base_store=base_store,
    )

    data_cube = recipe.jnp.asarray(context["cube"])
    variance_cube = recipe.jnp.asarray(context["variance_cube"])
    frame_reduce = str(inference_cfg["objective"]["frame_reduce"])
    subblock_reduce = str(inference_cfg["objective"]["subblock_reduce"])
    prior_term_fn = recipe._build_prior_term_fn(inference_cfg["priors"])
    temporal_term_fn = recipe._build_temporal_term_fn(inference_cfg["temporal"])

    priors_nonempty = bool(inference_cfg["priors"]["frame"] or inference_cfg["priors"]["shared"])
    temporal_kind = str(inference_cfg["temporal"]["frame_model"]["kind"])
    if objective_kind not in {"data_only", "full_objective"}:
        raise ValueError("summary objective must be 'data_only' or 'full_objective'.")
    if objective_kind == "full_objective" and (
        priors_nonempty or temporal_kind != "independent"
    ):
        raise ValueError(
            "v0 schur_summary only supports full_objective when priors are empty "
            "and temporal.frame_model.kind='independent'."
        )

    def _shared_store_from_local(
        observation_theta_values: jnp.ndarray,
        fast_phi_state: Any,
    ) -> ParameterStore:
        theta_store = _apply_theta_overrides(
            reference_store=base_store,
            forward_spec=forward_spec,
            theta_addresses=theta_addresses,
            theta_values=observation_theta_values,
        )
        return recipe._apply_runtime_active_values(
            reference_store=theta_store,
            forward_spec=forward_spec,
            key_specs=active_layout.shared_specs,
            values=fast_phi_state.shared,
        )

    def _frame_model(shared_store: ParameterStore, frame_values: jnp.ndarray) -> jnp.ndarray:
        frame_store = recipe._apply_runtime_active_values(
            reference_store=shared_store,
            forward_spec=forward_spec,
            key_specs=active_layout.frame_specs,
            values=frame_values,
        )
        frame_store = recipe._preserve_shared_derived_active_values(
            frame_store=frame_store,
            shared_store=shared_store,
            shared_specs=active_layout.shared_specs,
        )
        frame_delta = binder.strip_structural(frame_store)
        return binder.model(frame_delta)

    def _data_term(observation_theta_values: jnp.ndarray, fast_phi_state: Any) -> jnp.ndarray:
        shared_store = _shared_store_from_local(observation_theta_values, fast_phi_state)

        def _frame_loss(
            frame_values: jnp.ndarray,
            data_frame: jnp.ndarray,
            variance_frame: jnp.ndarray,
        ) -> jnp.ndarray:
            model_frame = _frame_model(shared_store, frame_values)
            return recipe.gaussian_image_nll(
                model_frame,
                data_frame,
                variance_frame,
                reduce=frame_reduce,
            )

        per_frame = jax.vmap(_frame_loss)(fast_phi_state.frame, data_cube, variance_cube)
        return recipe._reduce_subblock_terms(per_frame, reduce=subblock_reduce)

    def _combined_loss(local_vector: jnp.ndarray) -> jnp.ndarray:
        observation_theta_values = local_vector[: theta_layout.size]
        fast_phi_values = local_vector[theta_layout.size :]
        fast_phi_state = recipe._unpack_active_state(active_layout, fast_phi_values)
        data_term = _data_term(observation_theta_values, fast_phi_state)
        if objective_kind == "data_only":
            return data_term
        return data_term + prior_term_fn(fast_phi_state) + temporal_term_fn(fast_phi_state)

    metadata = {
        "objective_kind_requested": objective_kind,
        "objective_kind_used": (
            "data_only"
            if objective_kind == "data_only"
            else "full_objective_equivalent_to_data_only_current_template"
        ),
        "priors_nonempty": bool(priors_nonempty),
        "temporal_kind": temporal_kind,
        "inference_objective": inference_cfg["objective"],
    }
    return _combined_loss, metadata


def _build_structured_schur_frame_objective(
    *,
    context: dict[str, Any],
    theta_layout: ObservationThetaLayout,
    objective_kind: str,
) -> tuple[Any, dict[str, Any]]:
    """Build one frame-local objective over ``[Theta, phi_i]``.

    This is the script orchestration layer for structured Schur export. The
    reusable curvature assembly and framewise Schur reduction live in
    ``dluxshera.inference.structured_curvature``; this helper only adapts the
    image-backed inference context into the required frame-term callable.
    """

    recipe = context["recipe"]
    active_layout = context["layout"]
    binder = context["binder"]
    forward_spec = context["forward_spec"]
    base_store = context["base_store"]
    inference_cfg = context["inference_cfg"]
    theta_addresses = _theta_addresses_for_layout(
        theta_layout=theta_layout,
        forward_spec=forward_spec,
        base_store=base_store,
    )

    if int(active_layout.shared_width) != 0:
        raise ValueError(
            "structured_independent_frames currently requires no active shared "
            "subblock state."
        )

    data_cube = recipe.jnp.asarray(context["cube"])
    variance_cube = recipe.jnp.asarray(context["variance_cube"])
    frame_reduce = str(inference_cfg["objective"]["frame_reduce"])
    subblock_reduce = str(inference_cfg["objective"]["subblock_reduce"])
    priors_nonempty = bool(
        inference_cfg["priors"]["frame"] or inference_cfg["priors"]["shared"]
    )
    temporal_kind = str(inference_cfg["temporal"]["frame_model"]["kind"])
    if objective_kind not in {"data_only", "full_objective"}:
        raise ValueError("summary objective must be 'data_only' or 'full_objective'.")
    if objective_kind == "full_objective" and (
        priors_nonempty or temporal_kind != "independent"
    ):
        raise ValueError(
            "structured schur_summary only supports full_objective when priors "
            "are empty and temporal.frame_model.kind='independent'."
        )

    def _theta_store(observation_theta_values: jnp.ndarray) -> ParameterStore:
        return _apply_theta_overrides(
            reference_store=base_store,
            forward_spec=forward_spec,
            theta_addresses=theta_addresses,
            theta_values=observation_theta_values,
        )

    def _frame_model(theta_store: ParameterStore, frame_values: jnp.ndarray) -> jnp.ndarray:
        frame_store = recipe._apply_runtime_active_values(
            reference_store=theta_store,
            forward_spec=forward_spec,
            key_specs=active_layout.frame_specs,
            values=frame_values,
        )
        frame_delta = binder.strip_structural(frame_store)
        return binder.model(frame_delta)

    def _frame_loss(
        observation_theta_values: jnp.ndarray,
        frame_phi_values: jnp.ndarray,
        frame_index: int,
    ) -> jnp.ndarray:
        theta_store = _theta_store(observation_theta_values)
        model_frame = _frame_model(theta_store, frame_phi_values)
        return recipe.gaussian_image_nll(
            model_frame,
            data_cube[int(frame_index)],
            variance_cube[int(frame_index)],
            reduce=frame_reduce,
        )

    metadata = {
        "objective_kind_requested": objective_kind,
        "objective_kind_used": (
            "data_only"
            if objective_kind == "data_only"
            else "full_objective_equivalent_to_data_only_current_template"
        ),
        "priors_nonempty": bool(priors_nonempty),
        "temporal_kind": temporal_kind,
        "inference_objective": inference_cfg["objective"],
        "structured_frame_objective": True,
        "subblock_reduce": subblock_reduce,
    }
    return _frame_loss, metadata


def _resolve_phi_reference_for_summary(
    *,
    context: dict[str, Any],
    phi_ref_mode: str,
    recovered_theta: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Resolve the local fast-state reference vector for one summary export.

    Policy notes:
    - ``init`` uses optimizer initialization from the generated config.
    - ``truth_when_available`` uses the truth trace when complete.
    - ``recovered`` requires a completed reference inference solve.
    """

    recipe = context["recipe"]
    active_layout = context["layout"]
    normalized_mode = normalize_schur_phi_ref_mode(phi_ref_mode)
    if normalized_mode == "init":
        return np.asarray(context["theta0"], dtype=float), "init_state"
    if normalized_mode == "recovered":
        if recovered_theta is None:
            raise ValueError("phi_ref='recovered' requires recovered_theta.")
        return np.asarray(recovered_theta, dtype=float), "recovered_inference_solution"
    if normalized_mode == "truth_when_available":
        theta_truth, source = recipe._resolve_theta_preconditioning_reference(
            layout=active_layout,
            theta0=recipe.jnp.asarray(context["theta0"]),
            initial_state=context["initial_state"],
            truth=context["truth"],
            reference_mode="truth_when_available",
        )
        return np.asarray(theta_truth, dtype=float), str(source)
    raise ValueError("phi_ref must be one of: recovered, truth_when_available, init.")


def _validate_schur_dense_dimension(*, combined_dim: int, max_dense_dim: int) -> None:
    """Fail early when the v0 dense Hessian path would exceed its size guard."""

    if int(combined_dim) > int(max_dense_dim):
        raise ValueError(
            f"Combined dense dimension {combined_dim} exceeds max_dense_dim={max_dense_dim}. "
            "Reduce n_frames or Theta size, or wait for a structured-curvature path."
        )


def _estimate_phi_labels_from_inference_cfg(
    *,
    inference_cfg: Mapping[str, Any],
    n_frames: int | None,
) -> tuple[str, ...]:
    """Estimate packed phi labels from config-level active frame/shared keys."""

    active_cfg = inference_cfg.get("active", {})
    frame_keys = tuple(str(key) for key in active_cfg.get("frame_keys", ()))
    shared_keys = tuple(str(key) for key in active_cfg.get("shared_keys", ()))
    labels: list[str] = []
    if n_frames is not None:
        for frame_index in range(int(n_frames)):
            labels.extend(f"phi.frame[{frame_index}].{key}" for key in frame_keys)
    labels.extend(f"phi.shared.{key}" for key in shared_keys)
    return tuple(labels)


def _coerce_preview_cell(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except ValueError:
        return text


def _read_csv_preview_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        rows = [
            {str(key): _coerce_preview_cell(value) for key, value in row.items()}
            for row in reader
        ]
    return columns, rows


def _column_stats(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for column in columns:
        values: list[float] = []
        for row in rows:
            value = row.get(column)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                values.append(float(value))
        if values:
            arr = np.asarray(values, dtype=float)
            stats[column] = {
                "min": float(np.min(arr)),
                "median": float(np.median(arr)),
                "max": float(np.max(arr)),
            }
    return stats


def _write_frame_truth_preview(
    *,
    trace_csv_path: Path | None,
    preview_path: Path,
    head_rows: int = 5,
    tail_rows: int = 5,
) -> dict[str, Any] | None:
    """Write a compact frame-truth preview derived from a generated trace CSV."""

    if trace_csv_path is None or not trace_csv_path.exists():
        return None
    columns, rows = _read_csv_preview_rows(trace_csv_path)
    selected_columns = [
        column
        for column in (
            "frame_index",
            "time_s",
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
            "optics.plate_scale_as_per_pix",
        )
        if column in columns
    ]
    if not selected_columns:
        selected_columns = columns[: min(len(columns), 8)]

    def _select(row: Mapping[str, Any]) -> dict[str, Any]:
        return {column: row.get(column) for column in selected_columns}

    payload = {
        "schema_version": "frame_truth_preview.v1",
        "source_csv_path": str(trace_csv_path.resolve()),
        "selected_columns": selected_columns,
        "row_count": int(len(rows)),
        "first_rows": [_select(row) for row in rows[:head_rows]],
        "last_rows": [_select(row) for row in rows[-tail_rows:]],
        "column_stats": _column_stats(rows, selected_columns),
    }
    _write_json(preview_path, payload)
    return payload


def _effect_summary(effects: Any) -> list[dict[str, Any]]:
    if not isinstance(effects, list):
        return []
    return [dict(effect) for effect in effects if isinstance(effect, dict)]


def _trace_key_summary(trace_plan: Mapping[str, Any], key: str) -> dict[str, Any] | None:
    entry = trace_plan.get(key)
    if not isinstance(entry, Mapping):
        return None
    effects = _effect_summary(entry.get("effects"))
    summary = {
        "base": _scalar_or_none(entry.get("base")),
        "effect_kinds": [str(effect.get("kind")) for effect in effects],
        "effects": effects,
        "iid_jitter_sigma": None,
        "random_walk_sigma_step": None,
    }
    for effect in effects:
        if effect.get("kind") == "iid_jitter":
            summary["iid_jitter_sigma"] = _scalar_or_none(effect.get("sigma"))
        if effect.get("kind") == "random_walk":
            summary["random_walk_sigma_step"] = _scalar_or_none(effect.get("sigma_step"))
    return summary


def _build_trace_truth_summary(
    *,
    trace_template_path: Path,
    trace_template_source: str,
    trace_config_path: Path,
    trace_cfg: Mapping[str, Any],
    generated_trace_csv_path: Path | None,
    n_frames_requested: int | None,
    dt_s_requested: float | None,
    exposure_time_s_requested: float | None,
    preview: Mapping[str, Any] | None,
    applied_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize trace truth policy and generated-frame previews for the plan."""

    experiment_cfg = trace_cfg.get("experiment", {})
    trace_block = experiment_cfg.get("trace", {}) if isinstance(experiment_cfg, Mapping) else {}
    trace_plan = trace_block.get("plan", {}) if isinstance(trace_block, Mapping) else {}
    system_source = trace_cfg.get("system", {}).get("source", {}) if isinstance(trace_cfg.get("system"), Mapping) else {}
    n_frames = n_frames_requested
    if n_frames is None and isinstance(trace_block, Mapping):
        value = trace_block.get("n_frames")
        n_frames = None if value is None else int(value)
    dt_s = dt_s_requested
    if dt_s is None and isinstance(trace_block, Mapping):
        value = trace_block.get("dt_s")
        dt_s = None if value is None else float(value)
    exposure_time_s = exposure_time_s_requested
    if exposure_time_s is None and isinstance(system_source, Mapping):
        exposure_time_s = _scalar_or_none(system_source.get("exposure_time_s"))
    key_summaries = {
        "x": _trace_key_summary(trace_plan, "source.x_position_as")
        if isinstance(trace_plan, Mapping)
        else None,
        "y": _trace_key_summary(trace_plan, "source.y_position_as")
        if isinstance(trace_plan, Mapping)
        else None,
        "position_angle": _trace_key_summary(trace_plan, "source.position_angle_deg")
        if isinstance(trace_plan, Mapping)
        else None,
    }
    column_stats = {} if preview is None else dict(preview.get("column_stats", {}))
    first_rows = [] if preview is None else list(preview.get("first_rows", []))
    last_rows = [] if preview is None else list(preview.get("last_rows", []))
    return {
        "trace_template_path": str(trace_template_path.resolve()),
        "trace_template_source": str(trace_template_source),
        "template_description": (
            str(experiment_cfg.get("notes"))
            if isinstance(experiment_cfg, Mapping)
            and experiment_cfg.get("notes") is not None
            else None
        ),
        "registration_iid_template_used": bool(
            trace_template_path.resolve() == DEFAULT_SCHUR_TRACE_TEMPLATE.resolve()
        ),
        "trace_config_path": str(trace_config_path.resolve()),
        "generated_trace_csv_path": None
        if generated_trace_csv_path is None
        else str(generated_trace_csv_path.resolve()),
        "n_frames": n_frames,
        "dt_s": dt_s,
        "exposure_time_s": exposure_time_s,
        "seed": experiment_cfg.get("seed") if isinstance(experiment_cfg, Mapping) else None,
        "truth_model_by_key": key_summaries,
        "nominal_or_base_values": {
            "source.x_position_as": None
            if key_summaries["x"] is None
            else key_summaries["x"]["base"],
            "source.y_position_as": None
            if key_summaries["y"] is None
            else key_summaries["y"]["base"],
            "source.position_angle_deg": None
            if key_summaries["position_angle"] is None
            else key_summaries["position_angle"]["base"],
        },
        "jitter_amplitudes": {
            "source.x_position_as": None
            if key_summaries["x"] is None
            else (
                key_summaries["x"]["iid_jitter_sigma"]
                if key_summaries["x"]["iid_jitter_sigma"] is not None
                else key_summaries["x"]["random_walk_sigma_step"]
            ),
            "source.y_position_as": None
            if key_summaries["y"] is None
            else (
                key_summaries["y"]["iid_jitter_sigma"]
                if key_summaries["y"]["iid_jitter_sigma"] is not None
                else key_summaries["y"]["random_walk_sigma_step"]
            ),
            "source.position_angle_deg": None
            if key_summaries["position_angle"] is None
            else (
                key_summaries["position_angle"]["iid_jitter_sigma"]
                if key_summaries["position_angle"]["iid_jitter_sigma"] is not None
                else key_summaries["position_angle"]["random_walk_sigma_step"]
            ),
        },
        "first_generated_frame_values": first_rows[0] if first_rows else None,
        "last_generated_frame_values": last_rows[-1] if last_rows else None,
        "generated_column_stats": column_stats,
        "cli_overrides_applied": dict(applied_overrides),
    }


def _build_inference_init_summary(
    *,
    inference_cfg: Mapping[str, Any],
    n_frames: int | None,
    applied_overrides: Mapping[str, Any],
) -> dict[str, Any]:
    inference_block = inference_cfg.get("inference", inference_cfg)
    active_cfg = inference_block.get("active", {}) if isinstance(inference_block, Mapping) else {}
    init_cfg = inference_block.get("init", {}) if isinstance(inference_block, Mapping) else {}
    frame_cfg = init_cfg.get("frame", {}) if isinstance(init_cfg, Mapping) else {}
    shared_cfg = init_cfg.get("shared", {}) if isinstance(init_cfg, Mapping) else {}
    frame_keys = [str(key) for key in active_cfg.get("frame_keys", [])]
    shared_keys = [str(key) for key in active_cfg.get("shared_keys", [])]
    frame_values = dict(frame_cfg.get("values", {})) if isinstance(frame_cfg, Mapping) else {}
    shared_values = dict(shared_cfg) if isinstance(shared_cfg, Mapping) else {}
    first_frame_values = {
        key: _scalar_or_none(frame_values.get(key))
        for key in (
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        )
        if key in frame_keys
    }
    return {
        "active_frame_keys": frame_keys,
        "active_shared_keys": shared_keys,
        "frame_init_mode": frame_cfg.get("mode") if isinstance(frame_cfg, Mapping) else None,
        "shared_init_mode": "explicit_values" if shared_values else "empty",
        "configured_frame_init_values": frame_values,
        "configured_shared_init_values": shared_values,
        "init_value_sources": {
            key: (
                "cli_override"
                if key in applied_overrides
                else str(frame_cfg.get("mode", "template"))
            )
            for key in frame_keys
        },
        "packed_theta_size": (
            None
            if n_frames is None
            else int(n_frames) * len(frame_keys) + len(shared_keys)
        ),
        "first_frame_initial_values": first_frame_values,
        "cli_overrides_applied": dict(applied_overrides),
    }


def _extract_reference_diagnostics_plan(
    inference_cfg: Mapping[str, Any],
    *,
    sources: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Extract reference-inference diagnostic settings and their sources."""

    diagnostics_cfg = inference_cfg.get("diagnostics", {})
    keys = (
        "plots",
        "compare_to_truth_when_available",
        "first_step_report",
        "save_first_step_json",
        "save_fim_debug",
        "finite_difference_check",
        "plot_parameter_history_heatmap",
        "plot_parameter_residual_history_heatmap",
        "plot_parameter_history_lines",
        "plot_parameter_residual_history_lines",
        "top_k",
    )
    defaults = {
        "plots": True,
        "compare_to_truth_when_available": True,
        "first_step_report": False,
        "save_first_step_json": False,
        "save_fim_debug": False,
        "finite_difference_check": False,
        "plot_parameter_history_heatmap": False,
        "plot_parameter_residual_history_heatmap": False,
        "plot_parameter_history_lines": False,
        "plot_parameter_residual_history_lines": False,
        "top_k": 10,
    }
    return {
        "settings": {key: diagnostics_cfg.get(key, defaults[key]) for key in keys},
        "sources": {
            key: (sources or {}).get(key, SOURCE_INFERENCE_TEMPLATE)
            for key in keys
        },
    }


def _extract_reference_inference_plan(
    inference_cfg: Mapping[str, Any],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract recovered-reference optimizer, diagnostics, and provenance.

    Policy notes:
    - Values are read from the final generated inference config.
    - Source labels describe whether each value came from the inference
      template, an inference-recipe default, a generated config patch, or an
      explicit CLI override.
    - Missing preconditioning fields use the inference recipe defaults; this
      helper must not imply that the Schur workflow silently enabled them.
    """

    optimizer_cfg = inference_cfg.get("optimizer", {})
    preconditioning_cfg = optimizer_cfg.get("preconditioning", {})
    optimizer_sources = dict((provenance or {}).get("optimizer", {}))
    preconditioning_sources = dict((provenance or {}).get("preconditioning", {}))
    diagnostics_sources = dict((provenance or {}).get("diagnostics", {}))
    return {
        "optimizer_kind": str(optimizer_cfg.get("kind", "adam")),
        "base_lr": optimizer_cfg.get("base_lr", 1e-2),
        "n_iter": optimizer_cfg.get("n_iter", 100),
        "optimizer_kwargs": dict(optimizer_cfg.get("kwargs", {})),
        "schedule": optimizer_cfg.get("schedule"),
        "preconditioning_enabled": bool(preconditioning_cfg.get("enabled", False)),
        "preconditioning_method": str(preconditioning_cfg.get("method", "auto")),
        "preconditioning_reference": str(
            preconditioning_cfg.get("reference", "truth_when_available")
        ),
        "preconditioning_damping": preconditioning_cfg.get("damping", 1.0e-6),
        "preconditioning_eig_floor_rel": preconditioning_cfg.get(
            "eig_floor_rel",
            1.0e-6,
        ),
        "preconditioning_eig_floor_abs": preconditioning_cfg.get(
            "eig_floor_abs",
            1.0e-8,
        ),
        "preconditioning_lr_clip": preconditioning_cfg.get("lr_clip"),
        "sources": {
            **optimizer_sources,
            **preconditioning_sources,
        },
        "diagnostics": _extract_reference_diagnostics_plan(
            inference_cfg,
            sources=diagnostics_sources,
        ),
        "diagnostics_profile_source": (provenance or {}).get(
            "diagnostics_profile",
            SOURCE_NOT_APPLICABLE,
        ),
    }


def _summarize_local_surrogate_validation(csv_path: Path) -> dict[str, Any]:
    """Summarize the fixed-phi local surrogate validation CSV for audit review."""

    if not csv_path.exists():
        return {
            "path": str(csv_path.resolve()),
            "exists": False,
            "labels_validated": [],
            "perturbation_count": 0,
            "warnings": ["local_surrogate_validation.csv was not found"],
            "interpretation_note": (
                "Schur-reduced predictions are nuisance-adjusted; fixed-phi actual "
                "deltas are not."
            ),
        }
    _columns, rows = _read_csv_preview_rows(csv_path)
    labels = sorted({str(row.get("label")) for row in rows if row.get("label") is not None})
    ratios: list[float] = []
    sign_matches: list[bool] = []
    warnings: list[str] = []
    for row in rows:
        predicted = row.get("predicted_delta")
        actual = row.get("actual_delta_fixed_phi")
        if not isinstance(predicted, (int, float)) or not isinstance(actual, (int, float)):
            continue
        if float(predicted) == 0.0 or float(actual) == 0.0:
            continue
        sign_matches.append(bool(np.sign(float(predicted)) == np.sign(float(actual))))
        ratios.append(float(actual) / float(predicted))
    if sign_matches and not all(sign_matches):
        warnings.append("At least one fixed-phi actual delta has a different sign.")
    if not rows:
        warnings.append("Validation CSV is empty.")
    ratio_stats = None
    if ratios:
        arr = np.asarray(ratios, dtype=float)
        ratio_stats = {
            "min": float(np.min(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }
    return {
        "path": str(csv_path.resolve()),
        "exists": True,
        "labels_validated": labels,
        "perturbation_count": int(len(rows)),
        "matching_sign_count": int(sum(1 for value in sign_matches if value)),
        "sign_comparison_count": int(len(sign_matches)),
        "all_nonzero_deltas_match_sign": bool(sign_matches and all(sign_matches)),
        "actual_fixed_phi_to_schur_predicted_ratio_stats": ratio_stats,
        "warnings": warnings,
        "interpretation_note": (
            "Schur-reduced predictions are nuisance-adjusted; fixed-phi actual "
            "deltas are fixed-phi objective slices, so imperfect agreement is expected."
        ),
    }


def _build_schur_summary_audit(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    summary_payload: Mapping[str, Any] | None,
    recovered_reference_metadata: Mapping[str, Any],
    frame_truth_preview: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the post-run Schur audit from the plan and observed artifacts.

    Policy notes:
    - The plan remains the before-run source of expected behavior.
    - The audit records whether recovered-reference inference actually ran and
      points back to the final generated inference config consumed by the
      inference recipe.
    """

    planned_artifacts = dict(plan.get("planned_artifacts", {}))
    summary_artifacts = dict(summary_payload.get("artifacts", {})) if summary_payload else {}
    surrogate_path = Path(
        summary_artifacts.get("local_surrogate_validation_csv")
        or planned_artifacts.get("local_surrogate_validation_csv", "")
    )
    local_validation = (
        _summarize_local_surrogate_validation(surrogate_path)
        if str(surrogate_path)
        else {"exists": False, "warnings": ["local surrogate path was not planned"]}
    )
    reference_ran = bool(plan.get("reference_inference_will_run")) and bool(
        recovered_reference_metadata
    )
    return {
        "schema_version": "schur_summary_audit.v1",
        "created_at": now_iso_local_ms(),
        "case_name": plan.get("case_name"),
        "mode": MODE_SCHUR_SUMMARY,
        "selected_stages": list(plan.get("selected_stages", [])),
        "plan_json": str(plan_path.resolve()),
        "actual_artifacts": {
            "plan_json": str(plan_path.resolve()),
            "audit_json": planned_artifacts.get("schur_summary_audit_json"),
            "frame_truth_preview_json": planned_artifacts.get("frame_truth_preview_json"),
            **summary_artifacts,
        },
        "trace_truth": plan.get("trace_truth"),
        "trace_template": {
            "trace_template_path": plan.get("trace_template_path"),
            "trace_template_source": plan.get("trace_template_source"),
            "registration_iid_trace_template_used": plan.get(
                "registration_iid_trace_template_used"
            ),
            "case_local_trace_template_copy": plan.get("trace_config_path"),
            "generated_case_trace_config_path": plan.get(
                "generated_case_trace_config_path"
            ),
        },
        "frame_truth_preview": {
            "path": planned_artifacts.get("frame_truth_preview_json"),
            "written": frame_truth_preview is not None,
            "row_count": None if frame_truth_preview is None else frame_truth_preview.get("row_count"),
        },
        "render_summary": {
            "cube_path": plan.get("cube_path"),
            "render_config_path": plan.get("render_config_path"),
            "generated_case_render_config_path": plan.get(
                "generated_case_render_config_path"
            ),
            "render_noise_mode": plan.get("render_noise_mode"),
        },
        "inference_init": plan.get("inference_init"),
        "summary_export_inference_config_path": plan.get(
            "summary_export_inference_config_path"
        ),
        "reference_inference": {
            "status": "ran" if reference_ran else "not_run",
            "will_run": bool(plan.get("reference_inference_will_run")),
            "not_run_reason": None
            if reference_ran
            else plan.get("reference_inference_not_run_reason"),
            "config_if_run": plan.get("reference_inference_config_if_run"),
            "output_path": plan.get("reference_inference_output_path"),
            "final_generated_config_path": plan.get(
                "final_reference_inference_config_path"
            ),
            "recovered_reference_source": (
                None
                if not recovered_reference_metadata
                else recovered_reference_metadata.get("recovered_trace_csv")
                or recovered_reference_metadata.get("manifest_json")
                or recovered_reference_metadata.get("output_dir")
            ),
        },
        "preconditioning": plan.get("preconditioning"),
        "reference_diagnostics": (
            dict(plan.get("reference_inference_config_if_run", {})).get(
                "diagnostics"
            )
        ),
        "phi_reference": {
            "phi_ref_mode": plan.get("phi_ref_mode"),
            "phi_ref_source": None
            if summary_payload is None
            else summary_payload.get("phi_ref_source"),
            "n_phi": plan.get("n_phi"),
            "phi_labels": plan.get("phi_labels"),
        },
        "theta_reference": {
            "theta_ref": None if summary_payload is None else summary_payload.get("theta_ref"),
            "theta_labels": plan.get("theta_labels"),
            "theta_reference_consistency": None
            if summary_payload is None
            else summary_payload.get("theta_reference_consistency"),
        },
        "theta_reference_overrides": plan.get(
            "theta_reference_overrides",
            _disabled_theta_reference_overrides_payload(),
        ),
        "schur_summary_diagnostics_path": summary_artifacts.get("schur_diagnostics_json")
        or planned_artifacts.get("schur_diagnostics_json"),
        "curvature": {
            "schur_curvature_method_requested": plan.get(
                "schur_curvature_method_requested"
            ),
            "schur_curvature_method_planned": plan.get(
                "schur_curvature_method_planned"
            ),
            "schur_curvature_method_used": None
            if summary_payload is None
            else summary_payload.get("schur_curvature_method_used"),
            "dense_global_hessian_materialized": None
            if summary_payload is None
            else summary_payload.get("dense_global_hessian_materialized"),
            "structured_curvature_used": None
            if summary_payload is None
            else summary_payload.get("structured_curvature_used"),
            "structured_supported_layout": plan.get("structured_supported_layout"),
            "structured_reduce_weight": None
            if summary_payload is None
            else summary_payload.get("structured_reduce_weight"),
            "dense_vs_structured_comparison": None
            if summary_payload is None
            else summary_payload.get("dense_vs_structured_comparison"),
            "dense_vs_structured_comparison_requested": plan.get(
                "dense_vs_structured_comparison_requested"
            ),
            "dense_vs_structured_comparison_run": None
            if summary_payload is None
            else summary_payload.get("dense_vs_structured_comparison_run"),
            "dense_vs_structured_comparison_skipped_reason": None
            if summary_payload is None
            else summary_payload.get(
                "dense_vs_structured_comparison_skipped_reason"
            ),
            "max_dense_dim": plan.get("max_dense_dim"),
            "combined_dim": plan.get("combined_dim"),
        },
        "frame_quality": None
        if summary_payload is None
        else summary_payload.get("frame_quality"),
        "local_surrogate_validation": local_validation,
        "observation_prior_recommendation": {
            "prior_mean_source": "summary_theta_ref",
            "reason": (
                "Real-summary observation updates default the prior mean to the "
                "summary theta_ref context."
            ),
        },
    }


def _resolve_schur_frame_quality_manifest(
    *,
    recovered_reference_metadata: Mapping[str, Any],
    summary_json_dir: Path,
) -> Path | None:
    manifest_value = recovered_reference_metadata.get("manifest_json")
    candidates: list[Path] = []
    if isinstance(manifest_value, str) and manifest_value.strip():
        raw = Path(manifest_value).expanduser()
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(summary_json_dir / raw)
            output_dir = recovered_reference_metadata.get("output_dir")
            if isinstance(output_dir, str) and output_dir.strip():
                candidates.append(Path(output_dir).expanduser() / raw)
            run_root = recovered_reference_metadata.get("run_root")
            if isinstance(run_root, str) and run_root.strip():
                candidates.append(Path(run_root).expanduser() / raw)
    output_dir_value = recovered_reference_metadata.get("output_dir")
    if isinstance(output_dir_value, str) and output_dir_value.strip():
        candidates.append(Path(output_dir_value).expanduser() / "manifest.json")
    run_root_value = recovered_reference_metadata.get("run_root")
    if isinstance(run_root_value, str) and run_root_value.strip():
        candidates.append(Path(run_root_value).expanduser() / "inference" / "manifest.json")

    for candidate in candidates:
        path = candidate if candidate.is_absolute() else candidate.resolve()
        if path.exists():
            return path.resolve()
    fallback_root = summary_json_dir / "reference_inference"
    if fallback_root.exists():
        matches = sorted(fallback_root.glob("**/manifest.json"))
        if matches:
            return matches[0].resolve()
    return None


def build_schur_frame_quality_report(
    *,
    recovered_reference_metadata: Mapping[str, Any],
    summary_json_dir: Path,
    n_frames: int,
    chi2_threshold: float,
) -> SchurFrameQualityReport:
    if chi2_threshold <= 0.0 or not math.isfinite(float(chi2_threshold)):
        raise ValueError("chi2_threshold must be a positive finite float.")
    if int(n_frames) <= 0:
        raise ValueError("n_frames must be positive.")
    all_indices = tuple(range(int(n_frames)))
    if not recovered_reference_metadata:
        return SchurFrameQualityReport(
            threshold=float(chi2_threshold),
            total_frame_count=int(n_frames),
            good_frame_indices=all_indices,
            bad_frame_indices=(),
            good_frame_count=int(n_frames),
            bad_frame_count=0,
            per_frame_reduced_chi2=(),
            max_frame_reduced_chi2=None,
            median_frame_reduced_chi2=None,
            block_reduced_chi2=None,
            source_manifest_json=None,
            source_status="unavailable",
            warning="Recovered-reference metadata is unavailable.",
        )
    manifest_path = _resolve_schur_frame_quality_manifest(
        recovered_reference_metadata=recovered_reference_metadata,
        summary_json_dir=summary_json_dir,
    )
    if manifest_path is None:
        return SchurFrameQualityReport(
            threshold=float(chi2_threshold),
            total_frame_count=int(n_frames),
            good_frame_indices=all_indices,
            bad_frame_indices=(),
            good_frame_count=int(n_frames),
            bad_frame_count=0,
            per_frame_reduced_chi2=(),
            max_frame_reduced_chi2=None,
            median_frame_reduced_chi2=None,
            block_reduced_chi2=None,
            source_manifest_json=None,
            source_status="missing",
            warning="Recovered-reference manifest was not found.",
        )
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics = (
            manifest_payload.get("metrics", {})
            if isinstance(manifest_payload.get("metrics"), Mapping)
            else {}
        )
        chi2 = manifest_payload.get("chi2") or metrics.get("chi2") or {}
        final_model = chi2.get("final_model", {}) if isinstance(chi2, Mapping) else {}
        raw_values = final_model.get("per_frame_reduced_chi2", [])
        values = tuple(float(value) for value in raw_values)
        if len(values) != int(n_frames):
            raise ValueError(
                f"Expected {n_frames} per-frame chi2 values, found {len(values)}."
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("per_frame_reduced_chi2 contains non-finite values.")
        bad = tuple(
            index for index, value in enumerate(values) if value > float(chi2_threshold)
        )
        good = tuple(index for index in all_indices if index not in set(bad))
        block = final_model.get("block_reduced_chi2")
        block_value = None if block is None else float(block)
        arr = np.asarray(values, dtype=float)
        return SchurFrameQualityReport(
            threshold=float(chi2_threshold),
            total_frame_count=int(n_frames),
            good_frame_indices=good,
            bad_frame_indices=bad,
            good_frame_count=len(good),
            bad_frame_count=len(bad),
            per_frame_reduced_chi2=values,
            max_frame_reduced_chi2=float(np.max(arr)),
            median_frame_reduced_chi2=float(np.median(arr)),
            block_reduced_chi2=block_value,
            source_manifest_json=str(manifest_path),
            source_status="found",
            warning=None if not bad else "One or more frames exceed the chi2 threshold.",
        )
    except Exception as exc:
        return SchurFrameQualityReport(
            threshold=float(chi2_threshold),
            total_frame_count=int(n_frames),
            good_frame_indices=all_indices,
            bad_frame_indices=(),
            good_frame_count=int(n_frames),
            bad_frame_count=0,
            per_frame_reduced_chi2=(),
            max_frame_reduced_chi2=None,
            median_frame_reduced_chi2=None,
            block_reduced_chi2=None,
            source_manifest_json=str(manifest_path),
            source_status="parse_error",
            warning="Recovered-reference frame-quality manifest could not be parsed.",
            error=str(exc),
        )


def _metadata_for_reused_reference_inference(
    *,
    value: str | Path,
    study_root: Path,
) -> dict[str, Any]:
    raw_text = str(value)
    root = study_root / "reference_inference" if raw_text == "auto" else Path(raw_text)
    root = root.expanduser()
    manifest_candidates: list[Path] = []
    if root.is_file():
        manifest_candidates.append(root)
    else:
        manifest_candidates.extend(
            [
                root / "manifest.json",
                root / "inference" / "manifest.json",
            ]
        )
        if root.exists():
            manifest_candidates.extend(sorted(root.glob("**/manifest.json")))
    manifest_path = next(
        (candidate.resolve() for candidate in manifest_candidates if candidate.exists()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError(f"Could not find reused reference manifest under {root}")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = (
        manifest_payload.get("artifacts", {})
        if isinstance(manifest_payload.get("artifacts"), Mapping)
        else {}
    )
    trace_value = artifacts.get("recovered_trace_csv")
    trace_path: Path | None = None
    if isinstance(trace_value, str) and trace_value.strip():
        raw_trace = Path(trace_value).expanduser()
        trace_path = raw_trace if raw_trace.is_absolute() else manifest_path.parent / raw_trace
    if trace_path is None or not trace_path.exists():
        matches = sorted(manifest_path.parent.glob("*recovered_trace.csv"))
        if matches:
            trace_path = matches[0]
    if trace_path is None or not trace_path.exists():
        raise FileNotFoundError(
            f"Could not find recovered trace CSV next to reused manifest {manifest_path}"
        )
    return {
        "run_root": str(manifest_path.parent.parent.resolve()),
        "output_dir": str(manifest_path.parent.resolve()),
        "manifest_json": str(manifest_path.resolve()),
        "recovered_trace_csv": str(trace_path.resolve()),
        "reuse_reference_inference": True,
    }


def _recovered_theta_from_trace_csv(
    *,
    context: Mapping[str, Any],
    recovered_trace_csv: Path,
) -> np.ndarray:
    recipe = context["recipe"]
    layout = context["layout"]
    trace = load_obs_subblock_trace_csv(
        recovered_trace_csv,
        required_varying_keys=layout.frame_keys,
        require_contiguous_frame_index=True,
        require_monotonic_time=True,
    )
    if trace.frame_count != int(layout.n_frame):
        raise ValueError(
            f"Recovered trace frame count {trace.frame_count} does not match layout "
            f"frame count {int(layout.n_frame)}."
        )
    frame = np.empty((int(layout.n_frame), int(layout.frame_width)), dtype=float)
    for frame_index, row in enumerate(trace.rows):
        for key_index, key in enumerate(layout.frame_keys):
            frame[frame_index, key_index] = float(row[key])
    state = recipe.ActiveState(
        frame=recipe.jnp.asarray(frame),
        shared=context["initial_state"].shared,
    )
    return np.asarray(recipe._pack_active_state(layout, state), dtype=float)


def _schur_frame_quality_mask_state(
    *,
    report: SchurFrameQualityReport,
    policy: str,
    missing_policy: str,
    mask_denominator: str,
    min_good_frames: int,
    subblock_reduce: str,
) -> dict[str, Any]:
    policy = str(policy)
    missing_policy = str(missing_policy)
    mask_denominator = str(mask_denominator)
    if policy not in SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES:
        raise ValueError(f"Unsupported schur frame-quality policy: {policy}")
    if missing_policy not in SUPPORTED_SCHUR_FRAME_QUALITY_MISSING_POLICIES:
        raise ValueError(f"Unsupported schur frame-quality missing policy: {missing_policy}")
    if mask_denominator not in SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS:
        raise ValueError(f"Unsupported schur frame mask denominator: {mask_denominator}")

    quality_available = report.source_status == "found"
    if policy == "reject":
        pass
    if policy == "mask" and not quality_available and missing_policy == "error":
        raise RuntimeError("frame_quality_unavailable")

    included = tuple(range(report.total_frame_count))
    frame_scale = 1.0
    included_weight_policy = "all_frames"
    if policy == "mask" and quality_available and report.bad_frame_count:
        if report.good_frame_count < int(min_good_frames):
            raise RuntimeError("frame_quality_too_few_good_frames")
        included = report.good_frame_indices
        included_weight_policy = f"mask_denominator_{mask_denominator}"
        if mask_denominator == "kept" and str(subblock_reduce) == "mean":
            frame_scale = float(report.total_frame_count) / float(report.good_frame_count)
    return {
        "included_frame_indices": list(included),
        "included_frame_count": len(included),
        "frame_scale": float(frame_scale),
        "included_frame_weight_policy": included_weight_policy,
        "effective_frame_fraction": float(len(included)) / float(report.total_frame_count),
        "quality_available": bool(quality_available),
    }


def _build_schur_summary_plan(
    *,
    case_root: Path,
    study_root: Path,
    template_paths: Mapping[str, Path],
    source_template_paths: Mapping[str, Path],
    trace_template_source: str,
    schur_config_path: Path,
    schur_config: Mapping[str, Any],
    schur_config_provenance: Mapping[str, Any],
    render_inputs: Any,
    case_prep_stages: Sequence[str],
    n_frames_requested: int | None,
    dt_s_requested: float | None,
    exposure_time_s_requested: float | None,
    noise_mode: str,
    theta_keys: Sequence[str],
    enable_zernikes: bool,
    zernike_indices: Sequence[int],
    schur_damping: float,
    max_dense_dim: int,
    schur_curvature_method: str,
    phi_ref_mode: str,
    summary_objective: str,
    validate_surrogate: bool,
    validate_structured_against_dense: bool = DEFAULT_VALIDATE_STRUCTURED_AGAINST_DENSE,
    validation_steps: int = 5,
    schur_frame_quality_policy: str = DEFAULT_SCHUR_FRAME_QUALITY_POLICY,
    schur_frame_chi2_threshold: float = DEFAULT_SCHUR_FRAME_CHI2_THRESHOLD,
    schur_frame_quality_missing: str = DEFAULT_SCHUR_FRAME_QUALITY_MISSING,
    schur_frame_mask_denominator: str = DEFAULT_SCHUR_FRAME_MASK_DENOMINATOR,
    schur_frame_mask_min_good_frames: int = DEFAULT_SCHUR_FRAME_MASK_MIN_GOOD_FRAMES,
    frame_truth_preview: Mapping[str, Any] | None,
    applied_trace_overrides: Mapping[str, Any],
    applied_inference_init_overrides: Mapping[str, Any],
    theta_reference_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a review-friendly plan for the Schur-summary smoke path.

    The plan intentionally uses config-level and case-layout information so
    `--dry-run` can explain the run before any dense JAX differentiation begins.

    Policy notes:
    - Starts from copied templates and the final generated summary-export
      inference config.
    - Reports script-owned Schur defaults separately from template-owned
      recovered-reference defaults.
    - Records source labels for optimizer, preconditioning, and diagnostics so
      explicit template values are not mistaken for hidden script defaults.
    """

    theta_classification = validate_schur_summary_theta_keys(theta_keys)
    theta_layout = _build_observation_theta_layout(
        theta_keys=theta_keys,
        enable_zernikes=enable_zernikes,
        zernike_indices=zernike_indices,
    )
    experiment_cfg = schur_config.get("experiment", {})
    inference_cfg = experiment_cfg.get("inference", {})
    objective_cfg = inference_cfg.get("objective", {})
    n_frames_effective = (
        None if n_frames_requested is None else int(n_frames_requested)
    )
    trace_cfg = load_config_file(template_paths["trace"])
    if n_frames_effective is None:
        n_frames_value = _path_value_or_missing(trace_cfg, "experiment.trace.n_frames")
        if n_frames_value[0]:
            n_frames_effective = int(n_frames_value[1])
    phi_labels = _estimate_phi_labels_from_inference_cfg(
        inference_cfg=inference_cfg,
        n_frames=n_frames_effective,
    )
    n_phi = len(phi_labels)
    combined_dim = int(theta_layout.size + n_phi)
    dense_hessian_allowed = combined_dim <= int(max_dense_dim)
    structured_support = _structured_schur_support_from_inference_cfg(
        inference_cfg=inference_cfg,
        n_frames=n_frames_effective,
    )
    curvature_method_requested = normalize_schur_curvature_method(
        schur_curvature_method
    )
    try:
        curvature_method_planned = _select_schur_curvature_method(
            requested_method=curvature_method_requested,
            combined_dim=combined_dim,
            max_dense_dim=int(max_dense_dim),
            structured_support=structured_support,
        )
    except ValueError:
        curvature_method_planned = None
    dense_comparison_state = _dense_vs_structured_comparison_state(
        requested=bool(validate_structured_against_dense),
        curvature_method_used=curvature_method_planned,
        combined_dim=combined_dim,
        max_dense_dim=int(max_dense_dim),
    )
    reference_inference = _extract_reference_inference_plan(
        inference_cfg,
        provenance=schur_config_provenance,
    )
    preconditioning_actually_used = bool(
        phi_ref_mode == "recovered"
        and reference_inference["preconditioning_enabled"]
    )
    preconditioning_not_used_reason = None
    if not preconditioning_actually_used:
        if phi_ref_mode != "recovered":
            preconditioning_not_used_reason = "reference inference did not run"
        elif not reference_inference["preconditioning_enabled"]:
            preconditioning_not_used_reason = "preconditioning disabled in inference config"
    planned_artifacts = {
        "schur_summary_plan_json": str((study_root / SCHUR_SUMMARY_PLAN_FILENAME).resolve()),
        "schur_summary_audit_json": str(
            (study_root / SCHUR_SUMMARY_AUDIT_FILENAME).resolve()
        ),
        "frame_truth_preview_json": str(
            (study_root / FRAME_TRUTH_PREVIEW_FILENAME).resolve()
        ),
        "subblock_summary_json": str((study_root / "subblock_summary.json").resolve()),
        "subblock_summary_matrices_npz": str(
            (study_root / "subblock_summary_matrices.npz").resolve()
        ),
        "schur_diagnostics_json": str((study_root / "schur_diagnostics.json").resolve()),
        "combined_curvature_diagnostics_json": str(
            (study_root / "combined_curvature_diagnostics.json").resolve()
        ),
        "local_surrogate_validation_csv": str(
            (study_root / "local_surrogate_validation.csv").resolve()
        ),
        "local_surrogate_validation_png": str(
            (study_root / "local_surrogate_validation.png").resolve()
        ),
    }
    warnings: list[str] = []
    if theta_classification["experimental"]:
        warnings.append(
            "Experimental Theta keys requested: "
            + ", ".join(theta_classification["experimental"])
            + ". The first smoke path is only documented for "
            + ", ".join(DEFAULT_SCHUR_THETA_KEYS)
            + "."
        )
    if not dense_hessian_allowed:
        if curvature_method_planned == SCHUR_CURVATURE_METHOD_STRUCTURED:
            warnings.append(
                f"Combined dense dimension {combined_dim} exceeds max_dense_dim={max_dense_dim}; "
                "auto will use structured_independent_frames."
            )
        else:
            warnings.append(
                f"Combined dense dimension {combined_dim} exceeds max_dense_dim={max_dense_dim}."
            )
    if (
        phi_ref_mode == "recovered"
        and reference_inference["optimizer_kind"] == "sgd"
        and not reference_inference["preconditioning_enabled"]
    ):
        warnings.append(
            "phi_ref=recovered will use unpreconditioned SGD for the reference "
            "registration solve. The recovered linearization point may be weak; "
            "prefer phi_ref=truth_when_available for the first smoke test."
        )
    warnings.append(
        "Dense image-backed Schur export remains the small-case validation path; "
        "structured_independent_frames is the first operational independent-frame "
        "registration-only exporter."
    )
    planned_stages = list(case_prep_stages)
    if phi_ref_mode == "recovered":
        planned_stages.append("reference_inference")
    planned_stages.append("schur_summary_export")
    return {
        "case_name": case_root.name,
        "case_root": str(case_root.resolve()),
        "study_root": str(study_root.resolve()),
        "mode": MODE_SCHUR_SUMMARY,
        "selected_stages": planned_stages,
        "n_frames": n_frames_effective,
        "render_noise_mode": str(noise_mode),
        "cube_path": None if render_inputs.cube.path is None else str(render_inputs.cube.path),
        "trace_config_path": str(template_paths["trace"].resolve()),
        "trace_template_path": str(source_template_paths["trace"].resolve()),
        "trace_template_source": str(trace_template_source),
        "registration_iid_trace_template_used": bool(
            source_template_paths["trace"].resolve()
            == DEFAULT_SCHUR_TRACE_TEMPLATE.resolve()
        ),
        "generated_case_trace_config_path": str((case_root / "trace_config.json").resolve()),
        "render_config_path": str(template_paths["render"].resolve()),
        "generated_case_render_config_path": str((case_root / "render_config.json").resolve()),
        "inference_config_path": str(template_paths["inference"].resolve()),
        "summary_export_inference_config_path": str(schur_config_path.resolve()),
        "final_reference_inference_config_path": str(schur_config_path.resolve()),
        "generated_config_source_precedence": [
            "inference_template",
            "schur_workflow_default_for_script_owned_fields",
            "generated_config_patch",
            "cli_override",
        ],
        "schur_workflow_defaults": {
            "trace_template": str(SCHUR_WORKFLOW_DEFAULTS.trace_template.resolve()),
            "render_template": str(SCHUR_WORKFLOW_DEFAULTS.render_template.resolve()),
            "inference_template": str(
                SCHUR_WORKFLOW_DEFAULTS.inference_template.resolve()
            ),
            "theta_keys": list(SCHUR_WORKFLOW_DEFAULTS.theta_keys),
            "zernike_indices": list(SCHUR_WORKFLOW_DEFAULTS.zernike_indices),
            "schur_damping": float(SCHUR_WORKFLOW_DEFAULTS.schur_damping),
            "max_dense_dim": int(SCHUR_WORKFLOW_DEFAULTS.max_dense_dim),
            "validate_structured_against_dense": bool(
                SCHUR_WORKFLOW_DEFAULTS.validate_structured_against_dense
            ),
            "phi_ref": SCHUR_WORKFLOW_DEFAULTS.phi_ref,
            "schur_curvature_method": (
                SCHUR_WORKFLOW_DEFAULTS.schur_curvature_method
            ),
            "schur_frame_quality_policy": DEFAULT_SCHUR_FRAME_QUALITY_POLICY,
            "schur_frame_chi2_threshold": DEFAULT_SCHUR_FRAME_CHI2_THRESHOLD,
            "schur_frame_quality_missing": DEFAULT_SCHUR_FRAME_QUALITY_MISSING,
            "schur_frame_mask_denominator": DEFAULT_SCHUR_FRAME_MASK_DENOMINATOR,
        },
        "reference_inference_policy": {
            "optimizer_kind": SCHUR_REFERENCE_INFERENCE_POLICY.optimizer_kind,
            "base_lr": SCHUR_REFERENCE_INFERENCE_POLICY.base_lr,
            "n_iter": SCHUR_REFERENCE_INFERENCE_POLICY.n_iter,
            "preconditioning_enabled": (
                SCHUR_REFERENCE_INFERENCE_POLICY.preconditioning_enabled
            ),
            "preconditioning_method": (
                SCHUR_REFERENCE_INFERENCE_POLICY.preconditioning_method
            ),
            "preconditioning_reference": (
                SCHUR_REFERENCE_INFERENCE_POLICY.preconditioning_reference
            ),
            "diagnostics": SCHUR_REFERENCE_INFERENCE_POLICY.diagnostics,
        },
        "reference_inference_output_path": (
            None
            if phi_ref_mode != "recovered"
            else str((study_root / "reference_inference" / "inference").resolve())
        ),
        "theta_labels": list(theta_layout.labels),
        "theta_reference_overrides": dict(
            theta_reference_overrides or _disabled_theta_reference_overrides_payload()
        ),
        "theta_key_support": theta_classification,
        "phi_labels": list(phi_labels),
        "n_theta": int(theta_layout.size),
        "n_phi": int(n_phi),
        "combined_dim": int(combined_dim),
        "max_dense_dim": int(max_dense_dim),
        "dense_hessian_allowed": bool(dense_hessian_allowed),
        "schur_curvature_method_requested": curvature_method_requested,
        "schur_curvature_method_planned": curvature_method_planned,
        "schur_frame_quality": {
            "policy": str(schur_frame_quality_policy),
            "chi2_threshold": float(schur_frame_chi2_threshold),
            "missing_policy": str(schur_frame_quality_missing),
            "mask_denominator": str(schur_frame_mask_denominator),
            "mask_min_good_frames": int(schur_frame_mask_min_good_frames),
        },
        **dense_comparison_state,
        "structured_curvature_used": bool(
            curvature_method_planned == SCHUR_CURVATURE_METHOD_STRUCTURED
        ),
        "dense_global_hessian_materialized": bool(
            curvature_method_planned == SCHUR_CURVATURE_METHOD_DENSE
        ),
        "structured_supported_layout": bool(structured_support["supported"]),
        "structured_support": structured_support,
        "structured_reduce_weight": (
            None
            if n_frames_effective is None
            else (
                1.0
                if str(objective_cfg.get("subblock_reduce")) == "sum"
                else 1.0 / float(n_frames_effective)
            )
        ),
        "phi_ref_mode": str(phi_ref_mode),
        "reference_inference_will_run": bool(phi_ref_mode == "recovered"),
        "reference_inference_status": (
            "configured_to_run" if phi_ref_mode == "recovered" else "not_run"
        ),
        "reference_inference_not_run_reason": (
            None
            if phi_ref_mode == "recovered"
            else f"phi_ref_mode={phi_ref_mode}"
        ),
        "reference_inference_config_if_run": reference_inference,
        "reference_inference": {
            **reference_inference,
            "status": "configured_to_run" if phi_ref_mode == "recovered" else "not_run",
            "reason": None
            if phi_ref_mode == "recovered"
            else f"phi_ref_mode={phi_ref_mode}",
        },
        "preconditioning": {
            "preconditioning_configured_enabled": bool(
                reference_inference["preconditioning_enabled"]
            ),
            "preconditioning_method": reference_inference["preconditioning_method"],
            "preconditioning_reference": reference_inference["preconditioning_reference"],
            "preconditioning_damping": reference_inference["preconditioning_damping"],
            "preconditioning_eig_floor_rel": reference_inference[
                "preconditioning_eig_floor_rel"
            ],
            "preconditioning_eig_floor_abs": reference_inference[
                "preconditioning_eig_floor_abs"
            ],
            "preconditioning_actually_used": bool(preconditioning_actually_used),
            "preconditioning_not_used_reason": preconditioning_not_used_reason,
            "sources": {
                key: reference_inference["sources"].get(key)
                for key in (
                    "preconditioning_enabled",
                    "preconditioning_method",
                    "preconditioning_reference",
                    "preconditioning_damping",
                    "preconditioning_eig_floor_rel",
                    "preconditioning_eig_floor_abs",
                    "preconditioning_lr_clip",
                )
            },
        },
        "preconditioning_configured_enabled": bool(
            reference_inference["preconditioning_enabled"]
        ),
        "preconditioning_configured_enabled_source": reference_inference["sources"].get(
            "preconditioning_enabled"
        ),
        "preconditioning_method": reference_inference["preconditioning_method"],
        "preconditioning_method_source": reference_inference["sources"].get(
            "preconditioning_method"
        ),
        "preconditioning_reference": reference_inference["preconditioning_reference"],
        "preconditioning_reference_source": reference_inference["sources"].get(
            "preconditioning_reference"
        ),
        "preconditioning_actually_used": bool(preconditioning_actually_used),
        "preconditioning_not_used_reason": preconditioning_not_used_reason,
        "trace_truth": _build_trace_truth_summary(
            trace_template_path=source_template_paths["trace"],
            trace_template_source=trace_template_source,
            trace_config_path=template_paths["trace"],
            trace_cfg=trace_cfg,
            generated_trace_csv_path=render_inputs.truth_trace.path,
            n_frames_requested=n_frames_requested,
            dt_s_requested=dt_s_requested,
            exposure_time_s_requested=exposure_time_s_requested,
            preview=frame_truth_preview,
            applied_overrides=applied_trace_overrides,
        ),
        "frame_truth_preview_path": planned_artifacts["frame_truth_preview_json"],
        "inference_init": _build_inference_init_summary(
            inference_cfg=inference_cfg,
            n_frames=n_frames_effective,
            applied_overrides=applied_inference_init_overrides,
        ),
        "observation_prior_recommendation": {
            "prior_mean_source": "summary_theta_ref",
            "theta_ref_source": "summary_export_inference_config",
        },
        "schur_damping": float(schur_damping),
        "summary_objective_kind": str(summary_objective),
        "variance_model": str(objective_cfg.get("noise_model", {}).get("variance_model")),
        "validate_surrogate": bool(validate_surrogate),
        "validate_structured_against_dense": bool(validate_structured_against_dense),
        "validation_steps": int(validation_steps),
        "planned_artifacts": planned_artifacts,
        "known_limitations_or_warnings": warnings,
    }


def _combined_curvature_diagnostics(
    *,
    blocks: Any,
) -> dict[str, Any]:
    return {
        "dimensions": {
            "combined_dim": int(blocks.layout.size),
            "n_theta": int(blocks.layout.n_theta),
            "n_phi": int(blocks.layout.n_phi),
        },
        "combined_gradient_norm": float(np.linalg.norm(blocks.combined_gradient)),
        "combined_curvature_trace": float(np.trace(blocks.combined_curvature)),
        "combined_curvature_frobenius_norm": float(np.linalg.norm(blocks.combined_curvature)),
        "partition_shapes": {
            "h_tt": list(blocks.h_tt.shape),
            "h_tp": list(blocks.h_tp.shape),
            "h_pp": list(blocks.h_pp.shape),
            "g_theta": list(blocks.g_theta.shape),
            "g_phi": list(blocks.g_phi.shape),
        },
    }


def _local_surrogate_validation_rows(
    *,
    combined_loss_fn: Any,
    theta_layout: ObservationThetaLayout,
    theta_ref: np.ndarray,
    phi_ref: np.ndarray,
    reduced_information: np.ndarray,
    reduced_score: np.ndarray,
    max_labels: int = 2,
    validation_steps: int = 5,
) -> list[dict[str, Any]]:
    """Compare reduced quadratic predictions against fixed-phi objective slices."""

    if validation_steps <= 1:
        raise ValueError("validation_steps must be > 1.")
    step_map = {
        "source.separation_as": 5.0e-3,
        "source.log_flux_total": 5.0e-2,
        "source.contrast": 5.0e-2,
        "optics.plate_scale_as_per_pix": 1.0e-4,
    }
    selected_labels = [
        label
        for label in (
            "source.separation_as",
            "optics.plate_scale_as_per_pix",
            "source.log_flux_total",
            "source.contrast",
        )
        if label in theta_layout.labels
    ][:max_labels]
    rows: list[dict[str, Any]] = []
    if not selected_labels:
        return rows

    combined_ref = np.concatenate((theta_ref, phi_ref), axis=0)
    ref_loss = float(np.asarray(combined_loss_fn(jnp.asarray(combined_ref)), dtype=float))
    grid = np.linspace(-1.0, 1.0, int(validation_steps), dtype=float)
    for label in selected_labels:
        theta_index = theta_layout.labels.index(label)
        base_step = step_map.get(label, 1.0)
        for step_scale in grid:
            delta_theta = np.zeros((theta_layout.size,), dtype=float)
            delta_theta[theta_index] = float(step_scale * base_step)
            combined_eval = np.concatenate((theta_ref + delta_theta, phi_ref), axis=0)
            actual_loss = float(
                np.asarray(combined_loss_fn(jnp.asarray(combined_eval)), dtype=float)
            )
            predicted_delta = float(
                reduced_score @ delta_theta
                + 0.5 * delta_theta @ reduced_information @ delta_theta
            )
            rows.append(
                {
                    "label": label,
                    "theta_index": int(theta_index),
                    "step_scale": float(step_scale),
                    "step_size": float(delta_theta[theta_index]),
                    "predicted_delta": predicted_delta,
                    "actual_delta_fixed_phi": float(actual_loss - ref_loss),
                }
            )
    return rows


def _write_structured_schur_summary_artifact(
    *,
    summary: SubblockSummary,
    combined_layout: Any,
    theta_ref: np.ndarray,
    phi_ref: np.ndarray,
    sidecar_blocks: Mapping[str, np.ndarray],
    structured_reduction: Any,
    summary_json_path: Path,
    matrix_npz_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    """Write a loader-compatible structured Schur summary artifact.

    The existing loader contract is preserved by writing the same reduced
    arrays plus dense ``H_tt/H_tp/H_pp`` sidecar blocks. These sidecar blocks
    are materialized from structured per-frame Hessians; the full packed dense
    Hessian is not obtained through global autodiff in this path.
    """

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_npz_path,
        theta_ref=np.asarray(theta_ref, dtype=float),
        phi_ref=np.asarray(phi_ref, dtype=float),
        reduced_information=summary.reduced_information,
        reduced_score=summary.reduced_score,
        h_tt=np.asarray(sidecar_blocks["h_tt"], dtype=float),
        h_tp=np.asarray(sidecar_blocks["h_tp"], dtype=float),
        h_pp=np.asarray(sidecar_blocks["h_pp"], dtype=float),
        g_theta=np.asarray(sidecar_blocks["g_theta"], dtype=float),
        g_phi=np.asarray(sidecar_blocks["g_phi"], dtype=float),
    )
    created_at = now_iso_local_ms()
    payload = {
        "schema_version": "image_backed_subblock_summary.v1",
        "created_at": created_at,
        "generator": metadata.get("generator"),
        "subblock_id": summary.subblock_id,
        "summary_kind": summary.summary_kind,
        "case_root": metadata.get("case_root"),
        "cube_path": metadata.get("cube_path"),
        "manifest_path": metadata.get("manifest_path"),
        "truth_trace_path": metadata.get("truth_trace_path"),
        "config_path": metadata.get("config_path"),
        "objective": metadata.get("objective"),
        "system": metadata.get("system"),
        "prior_context": metadata.get("prior_context"),
        "recovered_reference": metadata.get("recovered_reference"),
        "theta_labels": list(summary.theta_labels),
        "phi_labels": list(combined_layout.phi_labels),
        "combined_labels": list(combined_layout.combined_labels),
        "theta_ref": np.asarray(theta_ref, dtype=float).tolist(),
        "phi_ref": np.asarray(phi_ref, dtype=float).tolist(),
        "dimensions": {
            "n_theta": int(combined_layout.n_theta),
            "n_phi": int(combined_layout.n_phi),
            "combined_dim": int(combined_layout.size),
        },
        "diagnostics": structured_reduction.to_diagnostics_dict(),
        "summary_diagnostics": dict(summary.diagnostics),
        "matrix_artifact_path": matrix_npz_path.name,
        "metadata": dict(metadata),
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _plot_local_surrogate_validation(
    *,
    rows: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot predicted versus fixed-phi actual local objective deltas."""

    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = tuple(dict.fromkeys(str(row["label"]) for row in rows))
    for label in labels:
        label_rows = [row for row in rows if row["label"] == label]
        x = np.asarray([float(row["step_size"]) for row in label_rows], dtype=float)
        predicted = np.asarray(
            [float(row["predicted_delta"]) for row in label_rows],
            dtype=float,
        )
        actual = np.asarray(
            [float(row["actual_delta_fixed_phi"]) for row in label_rows],
            dtype=float,
        )
        ax.plot(x, predicted, marker="o", label=f"{label} predicted")
        ax.plot(x, actual, marker="s", linestyle="--", label=f"{label} actual")
    ax.set_xlabel("Theta Perturbation")
    ax.set_ylabel("Objective Delta")
    ax.set_title("Local Reduced-Quadratic Validation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _evaluate_schur_summary(
    *,
    config_path: Path,
    output_dir: Path,
    case_root: Path,
    theta_keys: Sequence[str],
    enable_zernikes: bool,
    zernike_indices: Sequence[int],
    schur_damping: float,
    max_dense_dim: int,
    schur_curvature_method: str,
    phi_ref: str,
    summary_objective: str,
    validate_surrogate: bool,
    validate_structured_against_dense: bool = DEFAULT_VALIDATE_STRUCTURED_AGAINST_DENSE,
    validation_steps: int = 5,
    recovered_theta: np.ndarray | None = None,
    recovered_reference_metadata: Mapping[str, Any] | None = None,
    schur_frame_quality_policy: str = DEFAULT_SCHUR_FRAME_QUALITY_POLICY,
    schur_frame_chi2_threshold: float = DEFAULT_SCHUR_FRAME_CHI2_THRESHOLD,
    schur_frame_quality_missing: str = DEFAULT_SCHUR_FRAME_QUALITY_MISSING,
    schur_frame_mask_denominator: str = DEFAULT_SCHUR_FRAME_MASK_DENOMINATOR,
    schur_frame_mask_min_good_frames: int = DEFAULT_SCHUR_FRAME_MASK_MIN_GOOD_FRAMES,
    theta_reference_overrides: Mapping[str, Any] | None = None,
    memory_recorder: MemoryDiagnosticsRecorder | None = None,
) -> dict[str, Any]:
    """Export one image-backed Schur-reduced summary from a prepared subblock.

    The workflow here is deliberately linear:

    1. resolve the image-backed inference context for one prepared cube,
    2. build the observation-level Theta layout and packed fast ``phi`` labels,
    3. choose a local reference point ``[Theta_ref, phi_ref]``,
    4. select dense or structured independent-frame curvature,
    5. compute the reduced ``Theta`` quadratic with the selected path, and
    6. persist a loader-compatible ``SubblockSummary`` plus diagnostics.
    """

    def record(stage: str, **metadata: Any) -> None:
        if memory_recorder is not None:
            memory_recorder.record(stage, **metadata)

    record("theta_phi_layout.start", config_path=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    theta_reference_override_payload = dict(
        theta_reference_overrides or _disabled_theta_reference_overrides_payload()
    )
    context = _prepare_inference_context(config_path=config_path)
    theta_layout = _build_observation_theta_layout(
        theta_keys=theta_keys,
        enable_zernikes=enable_zernikes,
        zernike_indices=zernike_indices,
    )
    observation_theta_ref_values = _observation_theta_ref_from_store(
        theta_layout=theta_layout,
        base_store=context["base_store"],
    )
    prior_theta_ref_by_label = {
        label: float(observation_theta_ref_values[index])
        for index, label in enumerate(theta_layout.labels)
    }
    theta_reference_consistency = validate_theta_reference_override_consistency(
        theta_reference_overrides=theta_reference_override_payload,
        theta_labels=theta_layout.labels,
        theta_ref=observation_theta_ref_values,
        resolved_config=context["system_cfg"],
        store=context["base_store"],
        prior_context={"theta_ref_by_label": prior_theta_ref_by_label},
    )
    if recovered_theta is None and normalize_schur_phi_ref_mode(phi_ref) == "recovered":
        recovered_trace_value = (recovered_reference_metadata or {}).get(
            "recovered_trace_csv"
        )
        if isinstance(recovered_trace_value, str) and recovered_trace_value.strip():
            recovered_theta = _recovered_theta_from_trace_csv(
                context=context,
                recovered_trace_csv=Path(recovered_trace_value),
            )
    record("phi_ref.resolve.start", phi_ref_mode=phi_ref)
    fast_phi_ref_values, phi_ref_source = _resolve_phi_reference_for_summary(
        context=context,
        phi_ref_mode=phi_ref,
        recovered_theta=recovered_theta,
    )
    record(
        "phi_ref.resolve.done",
        phi_ref_mode=phi_ref,
        phi_ref_source=phi_ref_source,
        arrays=named_array_memory_metadata(phi_ref=fast_phi_ref_values),
    )
    recipe = context["recipe"]
    phi_labels = _phi_labels_for_active_layout(recipe, context["layout"])
    combined_layout = build_combined_local_parameter_layout(
        _theta_labels_for_observation_layout(theta_layout),
        phi_labels,
    )
    combined_dim = int(theta_layout.size + fast_phi_ref_values.size)
    structured_support = _structured_schur_support_from_context(context)
    record(
        "theta_phi_layout.done",
        n_frames=int(context["layout"].n_frame),
        n_theta=int(theta_layout.size),
        n_phi=int(fast_phi_ref_values.size),
        combined_dim=combined_dim,
        frame_phi_dim=int(context["layout"].frame_width),
        shared_phi_dim=int(context["layout"].shared_width),
        arrays=named_array_memory_metadata(
            theta_ref=observation_theta_ref_values,
            phi_ref=fast_phi_ref_values,
        ),
    )
    frame_quality_report = build_schur_frame_quality_report(
        recovered_reference_metadata=recovered_reference_metadata or {},
        summary_json_dir=output_dir,
        n_frames=int(context["layout"].n_frame),
        chi2_threshold=float(schur_frame_chi2_threshold),
    )
    frame_quality_state = _schur_frame_quality_mask_state(
        report=frame_quality_report,
        policy=schur_frame_quality_policy,
        missing_policy=schur_frame_quality_missing,
        mask_denominator=schur_frame_mask_denominator,
        min_good_frames=int(schur_frame_mask_min_good_frames),
        subblock_reduce=str(context["inference_cfg"]["objective"].get("subblock_reduce")),
    )
    frame_quality_metadata = {
        "policy": str(schur_frame_quality_policy),
        "chi2_threshold": float(schur_frame_chi2_threshold),
        "missing_policy": str(schur_frame_quality_missing),
        "mask_denominator": str(schur_frame_mask_denominator),
        **frame_quality_report.to_dict(),
        **frame_quality_state,
    }
    if str(schur_frame_quality_policy) == "reject" and (
        frame_quality_report.bad_frame_count
        or frame_quality_report.source_status != "found"
    ):
        rejection = {
            "reason": (
                "frame_quality_failed"
                if frame_quality_report.bad_frame_count
                else "frame_quality_unavailable"
            ),
            "frame_quality": frame_quality_metadata,
            "bad_frame_indices": list(frame_quality_report.bad_frame_indices),
            "threshold": float(schur_frame_chi2_threshold),
            "per_frame_reduced_chi2": list(frame_quality_report.per_frame_reduced_chi2),
        }
        _write_json(output_dir / "schur_summary_rejection.json", rejection)
        raise RuntimeError(str(rejection["reason"]))
    curvature_method_requested = normalize_schur_curvature_method(
        schur_curvature_method
    )
    record(
        "curvature_method.select.start",
        schur_curvature_method_requested=curvature_method_requested,
        combined_dim=combined_dim,
        max_dense_dim=int(max_dense_dim),
        structured_supported_layout=bool(structured_support["supported"]),
    )
    curvature_method_used = _select_schur_curvature_method(
        requested_method=curvature_method_requested,
        combined_dim=combined_dim,
        max_dense_dim=int(max_dense_dim),
        structured_support=structured_support,
    )
    if (
        curvature_method_used == SCHUR_CURVATURE_METHOD_DENSE
        and str(schur_frame_quality_policy) == "mask"
        and frame_quality_report.bad_frame_count
    ):
        frame_quality_state = {
            **frame_quality_state,
            "included_frame_indices": list(range(frame_quality_report.total_frame_count)),
            "included_frame_count": int(frame_quality_report.total_frame_count),
            "frame_scale": 1.0,
            "included_frame_weight_policy": "all_frames_dense_path_mask_not_applied",
            "effective_frame_fraction": 1.0,
        }
        frame_quality_metadata = {
            **frame_quality_metadata,
            **frame_quality_state,
            "warning": (
                "Frame-quality mask was requested, but the dense Schur path "
                "does not support frame masking; all frames were included."
            ),
        }
    record(
        "curvature_method.select.done",
        schur_curvature_method_requested=curvature_method_requested,
        schur_curvature_method_used=curvature_method_used,
        dense_global_hessian_materialized=(
            curvature_method_used == SCHUR_CURVATURE_METHOD_DENSE
        ),
        structured_curvature_used=(
            curvature_method_used == SCHUR_CURVATURE_METHOD_STRUCTURED
        ),
    )

    record("objective_context.start", summary_objective=summary_objective)
    combined_loss_fn, objective_metadata = _build_combined_local_objective(
        context=context,
        theta_layout=theta_layout,
        objective_kind=summary_objective,
    )
    record(
        "objective_context.done",
        objective_kind_used=objective_metadata["objective_kind_used"],
        subblock_reduce=context["inference_cfg"]["objective"].get("subblock_reduce"),
    )
    combined_reference_vector = np.concatenate(
        (observation_theta_ref_values, fast_phi_ref_values),
        axis=0,
    )
    dense_global_hessian_materialized = False
    structured_curvature_used = (
        curvature_method_used == SCHUR_CURVATURE_METHOD_STRUCTURED
    )
    dense_vs_structured_comparison: dict[str, Any] | None = None
    dense_comparison_state = _dense_vs_structured_comparison_state(
        requested=bool(validate_structured_against_dense),
        curvature_method_used=curvature_method_used,
        combined_dim=combined_dim,
        max_dense_dim=int(max_dense_dim),
    )
    combined_gradient: np.ndarray | None = None
    combined_curvature: np.ndarray | None = None
    blocks: Any | None = None
    reduced: Any | None = None
    structured_blocks: Any | None = None
    structured_reduction: Any | None = None
    structured_sidecar_blocks: dict[str, np.ndarray] | None = None

    if curvature_method_used == SCHUR_CURVATURE_METHOD_DENSE:
        record(
            "dense_curvature.start",
            combined_dim=combined_dim,
            arrays=named_array_memory_metadata(
                combined_reference_vector=combined_reference_vector
            ),
        )
        dense_global_hessian_materialized = True
        combined_gradient = np.asarray(
            jax.grad(combined_loss_fn)(
                jnp.asarray(combined_reference_vector, dtype=float)
            ),
            dtype=float,
        )
        combined_curvature = np.asarray(
            jax.hessian(combined_loss_fn)(
                jnp.asarray(combined_reference_vector, dtype=float)
            ),
            dtype=float,
        )
        blocks = partition_local_curvature(
            layout=combined_layout,
            combined_gradient=combined_gradient,
            combined_curvature=combined_curvature,
        )
        record(
            "dense_curvature.done",
            arrays=named_array_memory_metadata(
                combined_gradient=combined_gradient,
                combined_curvature=combined_curvature,
                H_tt=blocks.h_tt,
                H_tp=blocks.h_tp,
                H_pp=blocks.h_pp,
            ),
        )
        record("dense_schur_reduce.start")
        reduced = schur_reduce_local_quadratic(
            blocks=blocks,
            damping=float(schur_damping),
        )
        reduced_information = reduced.reduced_information
        reduced_score = reduced.reduced_score
        record(
            "dense_schur_reduce.done",
            arrays=named_array_memory_metadata(
                reduced_information=reduced_information,
                reduced_score=reduced_score,
            ),
        )
    else:
        fast_phi_state = recipe._unpack_active_state(
            context["layout"],
            jnp.asarray(fast_phi_ref_values, dtype=float),
        )
        frame_phi_ref = np.asarray(fast_phi_state.frame, dtype=float)
        frame_loss_fn, structured_objective_metadata = (
            _build_structured_schur_frame_objective(
                context=context,
                theta_layout=theta_layout,
                objective_kind=summary_objective,
            )
        )
        objective_metadata.update(structured_objective_metadata)
        record(
            "structured_curvature.blocks.start",
            n_frames=int(context["layout"].n_frame),
            n_theta=int(theta_layout.size),
            frame_phi_dim=int(context["layout"].frame_width),
            shared_phi_dim=int(context["layout"].shared_width),
            subblock_reduce=str(
                context["inference_cfg"]["objective"]["subblock_reduce"]
            ),
            arrays=named_array_memory_metadata(frame_phi_ref=frame_phi_ref),
        )
        structured_blocks = build_independent_frame_theta_phi_quadratic_blocks(
            frame_loss_fn=frame_loss_fn,
            theta_ref=observation_theta_ref_values,
            frame_phi_ref=frame_phi_ref,
            subblock_reduce=str(
                context["inference_cfg"]["objective"]["subblock_reduce"]
            ),
            kind=SCHUR_CURVATURE_METHOD_STRUCTURED,
        )
        record(
            "structured_curvature.blocks.done",
            n_frames=int(context["layout"].n_frame),
            reduce_weight=float(structured_blocks.reduce_weight),
            per_frame_arrays=[
                {
                    "frame_index": int(block.frame_index),
                    "H_tt": array_memory_metadata(block.h_tt),
                    "H_tp": array_memory_metadata(block.h_tphi),
                    "H_pp": array_memory_metadata(block.h_phiphi),
                    "g_theta": array_memory_metadata(block.g_theta),
                    "g_phi": array_memory_metadata(block.g_phi),
                }
                for block in structured_blocks.blocks
            ],
        )
        record(
            "structured_schur_reduce.start",
            subblock_reduce=str(
                context["inference_cfg"]["objective"]["subblock_reduce"]
            ),
            reduce_weight=float(structured_blocks.reduce_weight),
        )
        structured_reduction = schur_reduce_independent_frame_blocks(
            structured_blocks,
            damping=float(schur_damping),
            frame_indices=frame_quality_state["included_frame_indices"],
            frame_scale=float(frame_quality_state["frame_scale"]),
        )
        record(
            "structured_schur_reduce.done",
            arrays=named_array_memory_metadata(
                reduced_information=structured_reduction.reduced_information,
                reduced_score=structured_reduction.reduced_score,
            ),
        )
        record("materialize_sidecar_blocks.start")
        structured_sidecar_blocks = materialize_structured_schur_sidecar_blocks(
            structured_blocks,
            frame_indices=frame_quality_state["included_frame_indices"],
            frame_scale=float(frame_quality_state["frame_scale"]),
        )
        record(
            "materialize_sidecar_blocks.done",
            arrays={
                key: array_memory_metadata(value)
                for key, value in structured_sidecar_blocks.items()
            },
        )
        reduced_information = structured_reduction.reduced_information
        reduced_score = structured_reduction.reduced_score

        if dense_comparison_state["dense_vs_structured_comparison_run"]:
            record(
                "dense_vs_structured_comparison.start",
                combined_dim=combined_dim,
                max_dense_dim=int(max_dense_dim),
            )
            dense_gradient_for_comparison = np.asarray(
                jax.grad(combined_loss_fn)(
                    jnp.asarray(combined_reference_vector, dtype=float)
                ),
                dtype=float,
            )
            dense_curvature_for_comparison = np.asarray(
                jax.hessian(combined_loss_fn)(
                    jnp.asarray(combined_reference_vector, dtype=float)
                ),
                dtype=float,
            )
            dense_blocks_for_comparison = partition_local_curvature(
                layout=combined_layout,
                combined_gradient=dense_gradient_for_comparison,
                combined_curvature=dense_curvature_for_comparison,
            )
            dense_reduced_for_comparison = schur_reduce_local_quadratic(
                blocks=dense_blocks_for_comparison,
                damping=float(schur_damping),
            )
            dense_vs_structured_comparison = (
                compare_structured_and_dense_schur_outputs(
                    structured_information=reduced_information,
                    structured_score=reduced_score,
                    dense_information=dense_reduced_for_comparison.reduced_information,
                    dense_score=dense_reduced_for_comparison.reduced_score,
                )
            )
            record(
                "dense_vs_structured_comparison.done",
                comparison=dense_vs_structured_comparison,
            )

    record(
        "subblock_summary.build.start",
        arrays=named_array_memory_metadata(
            reduced_information=reduced_information,
            reduced_score=reduced_score,
        ),
    )
    reduced_summary = SubblockSummary.from_reduced_form(
        subblock_id=f"{case_root.name}_subblock_summary",
        theta_labels=theta_layout.labels,
        theta_ref=observation_theta_ref_values,
        reduced_information=reduced_information,
        reduced_score=reduced_score,
        summary_kind="image_backed_schur",
        damping_used=float(schur_damping),
        diagnostics={
            "phi_ref_source": phi_ref_source,
            "objective_kind": objective_metadata["objective_kind_used"],
            "n_phi": int(combined_layout.n_phi),
            "local_fit_reference_kind": phi_ref,
            "schur_curvature_method_requested": curvature_method_requested,
            "schur_curvature_method_used": curvature_method_used,
            "dense_global_hessian_materialized": bool(
                dense_global_hessian_materialized
            ),
            "structured_curvature_used": bool(structured_curvature_used),
            "structured_supported_layout": bool(structured_support["supported"]),
            "structured_reduce_weight": None
            if structured_blocks is None
            else float(structured_blocks.reduce_weight),
            "n_frames": int(context["layout"].n_frame),
            "theta_dim": int(theta_layout.size),
            "frame_phi_dim": int(context["layout"].frame_width),
            "shared_phi_dim": int(context["layout"].shared_width),
            "dense_vs_structured_comparison": dense_vs_structured_comparison,
            "frame_quality_policy": str(schur_frame_quality_policy),
            "frame_quality_total_frame_count": int(
                frame_quality_report.total_frame_count
            ),
            "frame_quality_good_frame_count": int(frame_quality_report.good_frame_count),
            "frame_quality_bad_frame_count": int(frame_quality_report.bad_frame_count),
            "frame_quality_bad_frame_indices": list(
                frame_quality_report.bad_frame_indices
            ),
            "frame_quality_chi2_threshold": float(schur_frame_chi2_threshold),
            "frame_quality_effective_frame_fraction": float(
                frame_quality_metadata["effective_frame_fraction"]
            ),
            "theta_reference_overrides": theta_reference_override_payload,
            "theta_reference_consistency": theta_reference_consistency,
            **dense_comparison_state,
        },
    )
    record(
        "subblock_summary.build.done",
        n_frames=int(context["layout"].n_frame),
        n_theta=int(theta_layout.size),
        n_phi=int(fast_phi_ref_values.size),
        combined_dim=combined_dim,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_dir / "subblock_summary.json"
    summary_npz_path = output_dir / "subblock_summary_matrices.npz"
    schur_diag_path = output_dir / "schur_diagnostics.json"
    curvature_diag_path = output_dir / "combined_curvature_diagnostics.json"
    surrogate_csv_path = output_dir / "local_surrogate_validation.csv"
    surrogate_png_path = output_dir / "local_surrogate_validation.png"

    artifact_metadata = {
        "generator": "examples/scripts/run_obs_subblock_study.py",
        "case_root": str(case_root.resolve()),
        "config_path": str(config_path.resolve()),
        "cube_path": str(context["cube_path"]),
        "manifest_path": None
        if context["manifest_path"] is None
        else str(Path(context["manifest_path"]).resolve()),
        "truth_trace_path": None
        if context["trace_path"] is None
        else str(Path(context["trace_path"]).resolve()),
        "theta_layout": theta_layout.to_dict(),
        "theta_reference_overrides": theta_reference_override_payload,
        "theta_reference_consistency": theta_reference_consistency,
        "phi_ref_source": phi_ref_source,
        "phi_ref_mode": normalize_schur_phi_ref_mode(phi_ref),
        "prior_context": {
            "recommended_prior_mean_source": "summary_theta_ref",
            "theta_ref_by_label": prior_theta_ref_by_label,
            "effective_store_values": {
                "source.exposure_time_s": (
                    float(np.asarray(context["base_store"].get("source.exposure_time_s")))
                    if "source.exposure_time_s" in getattr(
                        context["base_store"],
                        "_values",
                        {},
                    )
                    else None
                ),
            },
            "provenance": {
                "system_preset": context["system_cfg"].get("preset"),
                "render_config_path": str((case_root / "render_config.json").resolve())
                if (case_root / "render_config.json").exists()
                else None,
                "summary_export_config_path": str(config_path.resolve()),
                "cube_path": str(context["cube_path"]),
            },
        },
        "objective": objective_metadata,
        "recovered_reference": dict(recovered_reference_metadata or {}),
        "system": {
            "preset": context["system_cfg"].get("preset"),
            "resolved_config": context["system_cfg"],
            "store_overlay": context.get("system_store_overlay"),
        },
        "curvature": {
            "schur_curvature_method_requested": curvature_method_requested,
            "schur_curvature_method_used": curvature_method_used,
            "dense_global_hessian_materialized": bool(
                dense_global_hessian_materialized
            ),
            "structured_curvature_used": bool(structured_curvature_used),
            "structured_supported_layout": bool(structured_support["supported"]),
            "structured_support": dict(structured_support),
            "structured_reduce_weight": None
            if structured_blocks is None
            else float(structured_blocks.reduce_weight),
            "dense_vs_structured_comparison": dense_vs_structured_comparison,
            **dense_comparison_state,
        },
        "frame_quality": frame_quality_metadata,
    }
    if curvature_method_used == SCHUR_CURVATURE_METHOD_DENSE:
        if reduced is None or blocks is None:
            raise RuntimeError("Dense Schur path did not produce dense blocks.")
        record(
            "materialize_sidecar_blocks.start",
            path="dense",
            arrays=named_array_memory_metadata(
                H_tt=blocks.h_tt,
                H_tp=blocks.h_tp,
                H_pp=blocks.h_pp,
            ),
        )
        record(
            "materialize_sidecar_blocks.done",
            path="dense",
            arrays=named_array_memory_metadata(
                H_tt=blocks.h_tt,
                H_tp=blocks.h_tp,
                H_pp=blocks.h_pp,
                g_theta=blocks.g_theta,
                g_phi=blocks.g_phi,
            ),
        )
        artifact = ImageBackedSubblockSummaryArtifact(
            summary=reduced_summary,
            layout=combined_layout,
            theta_ref=observation_theta_ref_values,
            phi_ref=fast_phi_ref_values,
            reduced=reduced,
            metadata=artifact_metadata,
            combined_gradient=combined_gradient,
            combined_curvature=combined_curvature,
        )
        record(
            "subblock_summary.write.start",
            summary_json_path=summary_json_path,
            matrix_npz_path=summary_npz_path,
            dense_global_hessian_materialized=True,
        )
        artifact.write(
            summary_json_path=summary_json_path,
            matrix_npz_path=summary_npz_path,
        )
        record(
            "subblock_summary.write.done",
            summary_json_path=summary_json_path,
            matrix_npz_path=summary_npz_path,
            summary_json_written=summary_json_path.exists(),
            matrix_npz_written=summary_npz_path.exists(),
        )
        schur_diagnostics = reduced.to_diagnostics_dict()
        curvature_diagnostics = _combined_curvature_diagnostics(blocks=blocks)
    else:
        if structured_reduction is None or structured_sidecar_blocks is None:
            raise RuntimeError("Structured Schur path did not produce sidecar blocks.")
        record(
            "subblock_summary.write.start",
            summary_json_path=summary_json_path,
            matrix_npz_path=summary_npz_path,
            dense_global_hessian_materialized=False,
        )
        _write_structured_schur_summary_artifact(
            summary=reduced_summary,
            combined_layout=combined_layout,
            theta_ref=observation_theta_ref_values,
            phi_ref=fast_phi_ref_values,
            sidecar_blocks=structured_sidecar_blocks,
            structured_reduction=structured_reduction,
            summary_json_path=summary_json_path,
            matrix_npz_path=summary_npz_path,
            metadata=artifact_metadata,
        )
        record(
            "subblock_summary.write.done",
            summary_json_path=summary_json_path,
            matrix_npz_path=summary_npz_path,
            summary_json_written=summary_json_path.exists(),
            matrix_npz_written=summary_npz_path.exists(),
        )
        schur_diagnostics = structured_reduction.to_diagnostics_dict()
        curvature_diagnostics = {
            "dimensions": {
                "combined_dim": int(combined_layout.size),
                "n_theta": int(combined_layout.n_theta),
                "n_phi": int(combined_layout.n_phi),
                "n_frames": int(context["layout"].n_frame),
                "frame_phi_dim": int(context["layout"].frame_width),
                "shared_phi_dim": int(context["layout"].shared_width),
            },
            "dense_global_hessian_materialized": False,
            "structured_curvature_used": True,
            "structured_blocks": structured_blocks.to_debug_payload(
                include_blocks=False
            ),
            "sidecar_policy": (
                "Dense H_tt/H_tp/H_pp sidecar arrays are materialized from "
                "structured per-frame curvature blocks for compatibility; the "
                "full packed Hessian is not formed by global autodiff."
            ),
            "partition_shapes": {
                "h_tt": list(structured_sidecar_blocks["h_tt"].shape),
                "h_tp": list(structured_sidecar_blocks["h_tp"].shape),
                "h_pp": list(structured_sidecar_blocks["h_pp"].shape),
                "g_theta": list(structured_sidecar_blocks["g_theta"].shape),
                "g_phi": list(structured_sidecar_blocks["g_phi"].shape),
            },
        }
    schur_diagnostics.update(
        {
            "schur_curvature_method_requested": curvature_method_requested,
            "schur_curvature_method_used": curvature_method_used,
            "dense_global_hessian_materialized": bool(
                dense_global_hessian_materialized
            ),
            "structured_curvature_used": bool(structured_curvature_used),
            "structured_supported_layout": bool(structured_support["supported"]),
            "structured_reduce_weight": None
            if structured_blocks is None
            else float(structured_blocks.reduce_weight),
            "dense_vs_structured_comparison": dense_vs_structured_comparison,
            "frame_quality": frame_quality_metadata,
            **dense_comparison_state,
        }
    )
    _write_json(schur_diag_path, schur_diagnostics)
    _write_json(curvature_diag_path, curvature_diagnostics)

    validation_rows: list[dict[str, Any]] = []
    if validate_surrogate:
        validation_rows = _local_surrogate_validation_rows(
            combined_loss_fn=combined_loss_fn,
            theta_layout=theta_layout,
            theta_ref=observation_theta_ref_values,
            phi_ref=fast_phi_ref_values,
            reduced_information=reduced_information,
            reduced_score=reduced_score,
            validation_steps=validation_steps,
        )
        _write_rows_csv(surrogate_csv_path, validation_rows)
        _plot_local_surrogate_validation(
            rows=validation_rows,
            output_path=surrogate_png_path,
        )
    else:
        _write_rows_csv(surrogate_csv_path, validation_rows)

    loaded_summary = load_subblock_summary(summary_json_path)
    record(
        "subblock_summary.validate.done",
        loaded_theta_labels=list(loaded_summary.theta_labels),
        summary_json_written=summary_json_path.exists(),
        matrix_npz_written=summary_npz_path.exists(),
    )
    summary_payload = {
        "mode": MODE_SCHUR_SUMMARY,
        "schema_version": "image_backed_subblock_summary.v1",
        "case_root": str(case_root.resolve()),
        "config_path": str(config_path.resolve()),
        "cube_path": str(context["cube_path"]),
        "summary_json_path": str(summary_json_path.resolve()),
        "summary_npz_path": str(summary_npz_path.resolve()),
        "schur_diagnostics_json": str(schur_diag_path.resolve()),
        "combined_curvature_diagnostics_json": str(curvature_diag_path.resolve()),
        "loaded_summary_theta_labels": list(loaded_summary.theta_labels),
        "theta_labels": list(theta_layout.labels),
        "phi_labels": list(phi_labels),
        "theta_ref": observation_theta_ref_values.tolist(),
        "theta_reference_overrides": theta_reference_override_payload,
        "theta_reference_consistency": theta_reference_consistency,
        "phi_ref_source": phi_ref_source,
        "phi_ref_mode": normalize_schur_phi_ref_mode(phi_ref),
        "combined_dim": combined_dim,
        "n_theta": int(theta_layout.size),
        "n_phi": int(fast_phi_ref_values.size),
        "schur_curvature_method_requested": curvature_method_requested,
        "schur_curvature_method_used": curvature_method_used,
        "dense_global_hessian_materialized": bool(dense_global_hessian_materialized),
        "structured_curvature_used": bool(structured_curvature_used),
        "structured_supported_layout": bool(structured_support["supported"]),
        "structured_reduce_weight": None
        if structured_blocks is None
        else float(structured_blocks.reduce_weight),
        "frame_quality": frame_quality_metadata,
        "frame_quality_bad_frame_count": int(frame_quality_report.bad_frame_count),
        "frame_quality_effective_frame_fraction": float(
            frame_quality_metadata["effective_frame_fraction"]
        ),
        "frame_phi_dim": int(context["layout"].frame_width),
        "shared_phi_dim": int(context["layout"].shared_width),
        "dense_vs_structured_comparison": dense_vs_structured_comparison,
        **dense_comparison_state,
        "validate_surrogate": bool(validate_surrogate),
        "validate_structured_against_dense": bool(validate_structured_against_dense),
        "validation_row_count": int(len(validation_rows)),
        "artifacts": {
            "subblock_summary_json": str(summary_json_path.resolve()),
            "subblock_summary_matrices_npz": str(summary_npz_path.resolve()),
            "schur_diagnostics_json": str(schur_diag_path.resolve()),
            "combined_curvature_diagnostics_json": str(curvature_diag_path.resolve()),
            "local_surrogate_validation_csv": str(surrogate_csv_path.resolve()),
            "local_surrogate_validation_png": (
                None
                if not validate_surrogate
                else str(surrogate_png_path.resolve())
            ),
        },
    }
    return summary_payload


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
        run_label = f"{_candidate_token(candidate_key)}_{_study_value_token(value)}"
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
        **_candidate_metadata(candidate_key),
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
        **_candidate_metadata(candidate_key),
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
    trace_template: Path | None = None,
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
    theta_keys: Sequence[str] = DEFAULT_SCHUR_THETA_KEYS,
    enable_zernikes: bool = False,
    zernike_indices: Sequence[int] = DEFAULT_SCHUR_ZERNIKE_INDICES,
    schur_damping: float = DEFAULT_SCHUR_DAMPING,
    max_dense_dim: int = DEFAULT_SCHUR_MAX_DENSE_DIM,
    schur_curvature_method: str = SCHUR_CURVATURE_METHOD_AUTO,
    phi_ref: str = DEFAULT_SCHUR_PHI_REF,
    variance_floor: float | None = None,
    summary_objective: str = "full_objective",
    validate_surrogate: bool = True,
    validate_structured_against_dense: bool = DEFAULT_VALIDATE_STRUCTURED_AGAINST_DENSE,
    validation_steps: int = 5,
    schur_frame_quality_policy: str = DEFAULT_SCHUR_FRAME_QUALITY_POLICY,
    schur_frame_chi2_threshold: float = DEFAULT_SCHUR_FRAME_CHI2_THRESHOLD,
    schur_frame_quality_missing: str = DEFAULT_SCHUR_FRAME_QUALITY_MISSING,
    schur_frame_mask_denominator: str = DEFAULT_SCHUR_FRAME_MASK_DENOMINATOR,
    schur_frame_mask_min_good_frames: int = DEFAULT_SCHUR_FRAME_MASK_MIN_GOOD_FRAMES,
    trace_x0_as: float | None = None,
    trace_y0_as: float | None = None,
    trace_pa0_deg: float | None = None,
    trace_jitter_x_sigma_as: float | None = None,
    trace_jitter_y_sigma_as: float | None = None,
    trace_jitter_pa_sigma_deg: float | None = None,
    trace_seed: int | None = None,
    render_seed: int | None = None,
    init_x_as: float | None = None,
    init_y_as: float | None = None,
    init_pa_deg: float | None = None,
    reference_preconditioning_enabled: bool | None = None,
    reference_optimizer_kind: str | None = None,
    reference_base_lr: float | None = None,
    reference_n_iter: int | None = None,
    reference_optimizer_kwargs: Mapping[str, Any] | None = None,
    reference_schedule: Mapping[str, Any] | None = None,
    reference_preconditioning_method: str | None = None,
    reference_preconditioning_reference: str | None = None,
    reference_preconditioning_damping: float | None = None,
    reference_preconditioning_eig_floor_rel: float | None = None,
    reference_preconditioning_eig_floor_abs: float | None = None,
    reference_preconditioning_lr_clip: tuple[float, float] | None = None,
    reference_diagnostics_profile: str | None = None,
    reuse_reference_inference: str | Path | None = None,
    theta_reference_offsets: Mapping[str, float] | None = None,
    theta_reference_values: Mapping[str, float] | None = None,
    memory_diagnostics: bool = False,
    memory_diagnostics_file: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one observation sub-block screening study."""

    study_mode = parse_study_mode(mode)
    candidate_address = parse_candidate_parameter_address(candidate_key)
    candidate = None if candidate_address is None else candidate_address.canonical
    normalized_phi_ref = normalize_schur_phi_ref_mode(phi_ref)
    normalized_schur_curvature_method = normalize_schur_curvature_method(
        schur_curvature_method
    )
    theta_reference_offset_overrides = dict(theta_reference_offsets or {})
    theta_reference_value_overrides = dict(theta_reference_values or {})
    theta_reference_conflicts = sorted(
        set(theta_reference_offset_overrides) & set(theta_reference_value_overrides)
    )
    if theta_reference_conflicts:
        raise ValueError(
            "Cannot specify both theta reference value and offset for: "
            + ", ".join(theta_reference_conflicts)
            + "."
        )
    if schur_frame_quality_policy not in SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES:
        raise ValueError("Unsupported --schur-frame-quality-policy.")
    if schur_frame_quality_missing not in SUPPORTED_SCHUR_FRAME_QUALITY_MISSING_POLICIES:
        raise ValueError("Unsupported --schur-frame-quality-missing.")
    if schur_frame_mask_denominator not in SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS:
        raise ValueError("Unsupported --schur-frame-mask-denominator.")
    if float(schur_frame_chi2_threshold) <= 0.0 or not math.isfinite(float(schur_frame_chi2_threshold)):
        raise ValueError("--schur-frame-chi2-threshold must be positive.")
    if int(schur_frame_mask_min_good_frames) < 1:
        raise ValueError("--schur-frame-mask-min-good-frames must be >= 1.")
    if study_mode in {
        MODE_FISHER_ONLY,
        MODE_PROFILE_OBJECTIVE,
        MODE_NUISANCE_ABSORPTION,
    } and candidate is None:
        raise ValueError(f"{study_mode} mode requires --candidate.")
    if study_mode == MODE_PROFILE_OBJECTIVE and not scan_values:
        raise ValueError("profile_objective mode requires --scan-values.")
    if study_mode == MODE_NUISANCE_ABSORPTION and assumed_value is None:
        raise ValueError("nuisance_absorption mode requires --assumed-value.")

    resolved_trace_template, trace_template_source = resolve_study_trace_template(
        mode=study_mode,
        trace_template=trace_template,
    )
    case_root = case_root.resolve()
    study_root = _study_root(case_root, study_mode)
    study_root.mkdir(parents=True, exist_ok=True)
    summary_path = study_root / "summary.json"
    memory_recorder: MemoryDiagnosticsRecorder | None = None
    memory_audit_path: Path | None = None
    if memory_diagnostics and study_mode == MODE_SCHUR_SUMMARY:
        memory_timeline_path = (
            memory_diagnostics_file.resolve()
            if memory_diagnostics_file is not None
            else study_root / "schur_summary_memory_timeline.jsonl"
        )
        memory_recorder = MemoryDiagnosticsRecorder(memory_timeline_path)
        memory_audit_path = study_root / "schur_summary_memory_audit.json"
        memory_recorder.record(
            "schur_summary.start",
            case_root=case_root,
            study_root=study_root,
            n_frames=n_frames,
            phi_ref=normalized_phi_ref,
            schur_curvature_method_requested=normalized_schur_curvature_method,
        )

    template_info = _build_study_templates(
        mode=study_mode,
        case_root=case_root,
        trace_template=resolved_trace_template,
        render_template=render_template.resolve(),
        inference_template=inference_template.resolve(),
        candidate_key=candidate,
        truth_value=truth_value,
        assumed_value=assumed_value,
        trace_truth_overrides=_non_null_float_overrides(
            {
                "trace_x0_as": trace_x0_as,
                "trace_y0_as": trace_y0_as,
                "trace_pa0_deg": trace_pa0_deg,
            }
        ),
        trace_jitter_overrides=_non_null_float_overrides(
            {
                "trace_jitter_x_sigma_as": trace_jitter_x_sigma_as,
                "trace_jitter_y_sigma_as": trace_jitter_y_sigma_as,
                "trace_jitter_pa_sigma_deg": trace_jitter_pa_sigma_deg,
            }
        ),
        trace_seed=trace_seed,
        inference_init_overrides=_non_null_float_overrides(
            {
                "init_x_as": init_x_as,
                "init_y_as": init_y_as,
                "init_pa_deg": init_pa_deg,
            }
        ),
    )
    template_paths = dict(template_info["paths"])
    source_template_paths = dict(template_info["source_template_paths"])

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "mode": study_mode,
        "case_root": str(case_root),
        "study_root": str(study_root),
        "summary_path": str(summary_path.resolve()),
        "dry_run": bool(dry_run),
        **_candidate_metadata(candidate),
        "target_name": template_info["resolved_target_name"],
        "n_frames_requested": n_frames,
        "dt_s_requested": dt_s,
        "exposure_time_s_requested": exposure_time_s,
        "noise_mode_requested": noise_mode,
        "use_render_variance_requested": bool(use_render_variance),
        "truth_value_requested": None if truth_value is None else float(truth_value),
        "assumed_value_requested": None if assumed_value is None else float(assumed_value),
        "scan_values_requested": [float(value) for value in scan_values],
        "schur_summary_requested": {
            "theta_keys": list(parse_theta_keys(theta_keys)),
            "enable_zernikes": bool(enable_zernikes),
            "zernike_indices": list(parse_zernike_indices(zernike_indices)),
            "schur_damping": float(schur_damping),
            "max_dense_dim": int(max_dense_dim),
            "schur_curvature_method": normalized_schur_curvature_method,
            "phi_ref": normalized_phi_ref,
            "variance_floor": variance_floor,
            "summary_objective": str(summary_objective),
            "validate_surrogate": bool(validate_surrogate),
            "validate_structured_against_dense": bool(validate_structured_against_dense),
            "validation_steps": int(validation_steps),
            "schur_frame_quality_policy": str(schur_frame_quality_policy),
            "schur_frame_chi2_threshold": float(schur_frame_chi2_threshold),
            "schur_frame_quality_missing": str(schur_frame_quality_missing),
            "schur_frame_mask_denominator": str(schur_frame_mask_denominator),
            "schur_frame_mask_min_good_frames": int(
                schur_frame_mask_min_good_frames
            ),
            "trace_truth_cli_overrides": {
                "trace_x0_as": trace_x0_as,
                "trace_y0_as": trace_y0_as,
                "trace_pa0_deg": trace_pa0_deg,
                "trace_jitter_x_sigma_as": trace_jitter_x_sigma_as,
                "trace_jitter_y_sigma_as": trace_jitter_y_sigma_as,
                "trace_jitter_pa_sigma_deg": trace_jitter_pa_sigma_deg,
                "trace_seed": trace_seed,
                "render_seed": render_seed,
            },
            "inference_init_cli_overrides": {
                "init_x_as": init_x_as,
                "init_y_as": init_y_as,
                "init_pa_deg": init_pa_deg,
            },
            "reference_inference_cli_overrides": {
                "reference_optimizer_kind": reference_optimizer_kind,
                "reference_base_lr": reference_base_lr,
                "reference_n_iter": reference_n_iter,
                "reference_optimizer_kwargs": dict(reference_optimizer_kwargs or {}),
                "reference_schedule": (
                    None if reference_schedule is None else dict(reference_schedule)
                ),
                "reference_preconditioning_enabled": reference_preconditioning_enabled,
                "reference_preconditioning_method": reference_preconditioning_method,
                "reference_preconditioning_reference": reference_preconditioning_reference,
                "reference_preconditioning_damping": reference_preconditioning_damping,
                "reference_preconditioning_eig_floor_rel": reference_preconditioning_eig_floor_rel,
                "reference_preconditioning_eig_floor_abs": reference_preconditioning_eig_floor_abs,
                "reference_preconditioning_lr_clip": (
                    None
                    if reference_preconditioning_lr_clip is None
                    else list(reference_preconditioning_lr_clip)
                ),
            },
            "reference_diagnostics_profile": reference_diagnostics_profile,
            "reuse_reference_inference": None
            if reuse_reference_inference is None
            else str(reuse_reference_inference),
            "memory_diagnostics": bool(memory_diagnostics),
            "memory_diagnostics_file": None
            if memory_diagnostics_file is None
            else str(memory_diagnostics_file),
            "theta_reference_cli_overrides": {
                "offsets": dict(theta_reference_offset_overrides),
                "values": dict(theta_reference_value_overrides),
            },
        },
        "templates": {
            "trace": str(template_paths["trace"]),
            "render": str(template_paths["render"]),
            "inference": str(template_paths["inference"]),
        },
        "source_templates": {
            "trace": str(source_template_paths["trace"]),
            "trace_source": trace_template_source,
            "render": str(source_template_paths["render"]),
            "inference": str(source_template_paths["inference"]),
        },
        "resolved_template_values": {
            "truth_value": template_info["resolved_truth_value"],
            "assumed_value": template_info["resolved_assumed_value"],
        },
        "cli_overrides_applied": template_info["applied_overrides"],
    }
    if memory_recorder is not None:
        summary["memory_diagnostics"] = {
            "enabled": True,
            "timeline_jsonl": str(memory_recorder.path),
            "audit_json": None if memory_audit_path is None else str(memory_audit_path),
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

    if memory_recorder is not None:
        memory_recorder.record(
            "case_prepare.start",
            n_frames=n_frames,
            noise_mode=noise_mode,
            dry_run=dry_run,
        )
    prep = _prepare_case_render_artifacts(
        case_root=case_root,
        template_paths=template_paths,
        candidate_key=candidate,
        truth_value=truth_value,
        n_frames=n_frames,
        dt_s=dt_s,
        exposure_time_s=exposure_time_s,
        noise_mode=noise_mode,
        render_seed=render_seed,
        dry_run=dry_run,
    )
    if memory_recorder is not None:
        memory_recorder.record(
            "case_prepare.done",
            stages_executed=list(prep["stages_executed"]),
            cube_path=None
            if prep["render_inputs"].cube.path is None
            else prep["render_inputs"].cube.path,
            manifest_path=None
            if prep["render_inputs"].manifest.path is None
            else prep["render_inputs"].manifest.path,
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
            variance_floor=variance_floor,
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

    if study_mode == MODE_SCHUR_SUMMARY:
        validate_schur_summary_theta_keys(theta_keys)
        summary_run_root = study_root / "summary_export"
        schur_config = _build_study_inference_config(
            template_path=template_paths["inference"],
            run_root=summary_run_root,
            render_inputs=render_inputs,
            exposure_time_s=exposure_time_s,
            candidate_key=None,
            assumed_value=None,
            force_truth_comparison=(normalized_phi_ref == "truth_when_available"),
            disable_plots=False,
            use_render_variance=bool(use_render_variance),
            variance_floor=variance_floor,
            reference_optimizer_kind=reference_optimizer_kind,
            reference_base_lr=reference_base_lr,
            reference_n_iter=reference_n_iter,
            reference_optimizer_kwargs=reference_optimizer_kwargs,
            reference_schedule=reference_schedule,
            reference_preconditioning_enabled=reference_preconditioning_enabled,
            reference_preconditioning_method=reference_preconditioning_method,
            reference_preconditioning_reference=reference_preconditioning_reference,
            reference_preconditioning_damping=reference_preconditioning_damping,
            reference_preconditioning_eig_floor_rel=reference_preconditioning_eig_floor_rel,
            reference_preconditioning_eig_floor_abs=reference_preconditioning_eig_floor_abs,
            reference_preconditioning_lr_clip=reference_preconditioning_lr_clip,
            reference_diagnostics_profile=reference_diagnostics_profile,
        )
        theta_reference_override_metadata = resolve_theta_reference_overrides(
            inference_config=schur_config,
            theta_reference_offsets=theta_reference_offset_overrides,
            theta_reference_values=theta_reference_value_overrides,
            render_manifest_path=render_inputs.manifest.path,
        )
        schur_config_path = summary_run_root / "inference_config.json"
        _write_json(schur_config_path, schur_config)
        summary["schur_config_path"] = str(schur_config_path.resolve())
        summary["theta_reference_overrides"] = theta_reference_override_metadata
        schur_config_provenance = _build_schur_config_provenance(
            schur_config=schur_config,
            reference_optimizer_sources=_reference_optimizer_override_sources(
                optimizer_kind=reference_optimizer_kind,
                base_lr=reference_base_lr,
                n_iter=reference_n_iter,
                optimizer_kwargs=reference_optimizer_kwargs,
                schedule=reference_schedule,
                preconditioning_enabled=reference_preconditioning_enabled,
                preconditioning_method=reference_preconditioning_method,
                preconditioning_reference=reference_preconditioning_reference,
                preconditioning_damping=reference_preconditioning_damping,
                preconditioning_eig_floor_rel=reference_preconditioning_eig_floor_rel,
                preconditioning_eig_floor_abs=reference_preconditioning_eig_floor_abs,
                preconditioning_lr_clip=reference_preconditioning_lr_clip,
            ),
            reference_preconditioning_enabled=reference_preconditioning_enabled,
            reference_preconditioning_reference=reference_preconditioning_reference,
            reference_diagnostics_profile=reference_diagnostics_profile,
            force_truth_comparison=(normalized_phi_ref == "truth_when_available"),
        )

        frame_truth_preview = _write_frame_truth_preview(
            trace_csv_path=render_inputs.truth_trace.path,
            preview_path=study_root / FRAME_TRUTH_PREVIEW_FILENAME,
        )
        schur_plan = _build_schur_summary_plan(
            case_root=case_root,
            study_root=study_root,
            template_paths=template_paths,
            source_template_paths=source_template_paths,
            trace_template_source=trace_template_source,
            schur_config_path=schur_config_path,
            schur_config=schur_config,
            schur_config_provenance=schur_config_provenance,
            render_inputs=render_inputs,
            case_prep_stages=summary["case_prep_stages_executed"],
            n_frames_requested=n_frames,
            dt_s_requested=dt_s,
            exposure_time_s_requested=exposure_time_s,
            noise_mode=noise_mode,
            theta_keys=parse_theta_keys(theta_keys),
            enable_zernikes=bool(enable_zernikes),
            zernike_indices=parse_zernike_indices(zernike_indices),
            schur_damping=float(schur_damping),
            max_dense_dim=int(max_dense_dim),
            schur_curvature_method=normalized_schur_curvature_method,
            phi_ref_mode=normalized_phi_ref,
            summary_objective=str(summary_objective),
            validate_surrogate=bool(validate_surrogate),
            validate_structured_against_dense=bool(validate_structured_against_dense),
            validation_steps=int(validation_steps),
            schur_frame_quality_policy=str(schur_frame_quality_policy),
            schur_frame_chi2_threshold=float(schur_frame_chi2_threshold),
            schur_frame_quality_missing=str(schur_frame_quality_missing),
            schur_frame_mask_denominator=str(schur_frame_mask_denominator),
            schur_frame_mask_min_good_frames=int(schur_frame_mask_min_good_frames),
            frame_truth_preview=frame_truth_preview,
            applied_trace_overrides=template_info["applied_overrides"]["trace"],
            applied_inference_init_overrides=template_info["applied_overrides"][
                "inference_init"
            ],
            theta_reference_overrides=theta_reference_override_metadata,
        )
        schur_plan_path = study_root / SCHUR_SUMMARY_PLAN_FILENAME
        _write_json(schur_plan_path, schur_plan)
        summary["schur_summary_plan_path"] = str(schur_plan_path.resolve())
        summary["schur_summary_plan"] = schur_plan
        summary["planned_artifacts"] = dict(schur_plan["planned_artifacts"])
        _study_log(
            "schur_summary.plan",
            case_name=schur_plan["case_name"],
            phi_ref=schur_plan["phi_ref_mode"],
            n_theta=schur_plan["n_theta"],
            n_phi=schur_plan["n_phi"],
            combined_dim=schur_plan["combined_dim"],
            max_dense_dim=schur_plan["max_dense_dim"],
            dense_hessian_allowed=schur_plan["dense_hessian_allowed"],
            effective_method=schur_plan["schur_curvature_method_planned"],
            dense_comparison_requested=schur_plan[
                "dense_vs_structured_comparison_requested"
            ],
            dense_comparison_run=schur_plan["dense_vs_structured_comparison_run"],
            reference_optimizer_kind=schur_plan["reference_inference_config_if_run"][
                "optimizer_kind"
            ],
            reference_base_lr=schur_plan["reference_inference_config_if_run"][
                "base_lr"
            ],
            reference_n_iter=schur_plan["reference_inference_config_if_run"][
                "n_iter"
            ],
            plan_path=schur_plan_path.resolve(),
        )
        for warning in schur_plan["known_limitations_or_warnings"]:
            _study_log("schur_summary.warning", detail=warning)
        if dry_run:
            audit = _build_schur_summary_audit(
                plan=schur_plan,
                plan_path=schur_plan_path,
                summary_payload=None,
                recovered_reference_metadata={},
                frame_truth_preview=frame_truth_preview,
            )
            audit_path = study_root / SCHUR_SUMMARY_AUDIT_FILENAME
            _write_json(audit_path, audit)
            summary["schur_summary_audit_path"] = str(audit_path.resolve())
            summary["schur_summary_audit"] = audit
            if memory_recorder is not None:
                memory_recorder.record(
                    "schur_summary.done",
                    dry_run=True,
                    summary_json_written=False,
                    matrix_npz_written=False,
                )
                if memory_audit_path is not None:
                    _write_json(
                        memory_audit_path,
                        memory_recorder.audit_payload(
                            n_frames=schur_plan.get("n_frames"),
                            schur_curvature_method_used=None,
                            summary_json_written=False,
                            matrix_npz_written=False,
                        ),
                    )
            _write_json(summary_path, summary)
            return summary

        recovered_theta = None
        recovered_reference_metadata: dict[str, Any] = {}
        if normalized_phi_ref == "recovered":
            recovered_run_root = study_root / "reference_inference"
            if reuse_reference_inference is not None:
                recovered_reference_metadata = _metadata_for_reused_reference_inference(
                    value=reuse_reference_inference,
                    study_root=study_root,
                )
                summary["recovered_inference"] = recovered_reference_metadata
                summary["reused_reference_inference"] = True
            elif memory_recorder is not None:
                memory_recorder.record(
                    "reference_inference.start",
                    run_root=recovered_run_root,
                    config_path=schur_config_path,
                )
            if reuse_reference_inference is None:
                recovered_result = _default_inference_runner(
                    schur_config_path,
                    recovered_run_root,
                    False,
                )
                recovered_theta = np.asarray(recovered_result["theta_final"], dtype=float)
                if memory_recorder is not None:
                    memory_recorder.record(
                        "reference_inference.done",
                        run_root=recovered_run_root,
                        output_dir=Path(recovered_result["output_dir"]),
                        arrays=named_array_memory_metadata(theta_final=recovered_theta),
                    )
                recovered_reference_metadata = {
                    "run_root": str(recovered_run_root.resolve()),
                    "output_dir": str(Path(recovered_result["output_dir"]).resolve()),
                    "manifest_json": recovered_result["artifacts"].get("manifest_json"),
                    "recovered_trace_csv": recovered_result["artifacts"].get("recovered_trace_csv"),
                }
                if memory_recorder is not None:
                    memory_recorder.record(
                        "reference_inference.manifest_loaded",
                        manifest_json=recovered_reference_metadata.get("manifest_json"),
                        recovered_trace_csv=recovered_reference_metadata.get(
                            "recovered_trace_csv"
                        ),
                    )
                summary["recovered_inference"] = recovered_reference_metadata

        schur_summary = _evaluate_schur_summary(
            config_path=schur_config_path,
            output_dir=study_root,
            case_root=case_root,
            theta_keys=parse_theta_keys(theta_keys),
            enable_zernikes=bool(enable_zernikes),
            zernike_indices=parse_zernike_indices(zernike_indices),
            schur_damping=float(schur_damping),
            max_dense_dim=int(max_dense_dim),
            schur_curvature_method=normalized_schur_curvature_method,
            phi_ref=normalized_phi_ref,
            summary_objective=str(summary_objective),
            validate_surrogate=bool(validate_surrogate),
            validate_structured_against_dense=bool(validate_structured_against_dense),
            validation_steps=int(validation_steps),
            recovered_theta=recovered_theta,
            recovered_reference_metadata=recovered_reference_metadata,
            schur_frame_quality_policy=str(schur_frame_quality_policy),
            schur_frame_chi2_threshold=float(schur_frame_chi2_threshold),
            schur_frame_quality_missing=str(schur_frame_quality_missing),
            schur_frame_mask_denominator=str(schur_frame_mask_denominator),
            schur_frame_mask_min_good_frames=int(schur_frame_mask_min_good_frames),
            theta_reference_overrides=theta_reference_override_metadata,
            memory_recorder=memory_recorder,
        )
        summary["schur_summary"] = schur_summary
        frame_truth_preview = _write_frame_truth_preview(
            trace_csv_path=render_inputs.truth_trace.path,
            preview_path=study_root / FRAME_TRUTH_PREVIEW_FILENAME,
        )
        audit = _build_schur_summary_audit(
            plan=schur_plan,
            plan_path=schur_plan_path,
            summary_payload=schur_summary,
            recovered_reference_metadata=recovered_reference_metadata,
            frame_truth_preview=frame_truth_preview,
        )
        audit_path = study_root / SCHUR_SUMMARY_AUDIT_FILENAME
        _write_json(audit_path, audit)
        summary["schur_summary_audit_path"] = str(audit_path.resolve())
        summary["schur_summary_audit"] = audit
        if memory_recorder is not None:
            memory_recorder.record(
                "schur_summary.done",
                n_frames=schur_summary.get("n_frames"),
                schur_curvature_method_used=schur_summary.get(
                    "schur_curvature_method_used"
                ),
                summary_json_written=Path(
                    schur_summary["artifacts"]["subblock_summary_json"]
                ).exists(),
                matrix_npz_written=Path(
                    schur_summary["artifacts"]["subblock_summary_matrices_npz"]
                ).exists(),
            )
            if memory_audit_path is not None:
                _write_json(
                    memory_audit_path,
                    memory_recorder.audit_payload(
                        n_frames=schur_summary.get("n_frames"),
                        schur_curvature_method_used=schur_summary.get(
                            "schur_curvature_method_used"
                        ),
                        summary_json_written=Path(
                            schur_summary["artifacts"]["subblock_summary_json"]
                        ).exists(),
                        matrix_npz_written=Path(
                            schur_summary["artifacts"]["subblock_summary_matrices_npz"]
                        ).exists(),
                    ),
                )
                summary["memory_diagnostics"]["audit_json"] = str(
                    memory_audit_path.resolve()
                )
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
        default=None,
        help=(
            "Trace template YAML/JSON path. Defaults to the registration-iid "
            "template for schur_summary mode and the general trace template for "
            "other modes."
        ),
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
        help=(
            "Canonical scalar or indexed candidate key, for example "
            "optics.plate_scale_as_per_pix or optics.primary.zernike_coeffs_nm[3]."
        ),
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
        "--theta-keys",
        default=",".join(DEFAULT_SCHUR_THETA_KEYS),
        help=(
            "Comma-separated observation-level Theta keys for schur_summary mode. "
            "Defaults to the four-scalar smoke-test set: "
            "source.separation_as,source.log_flux_total,"
            "source.contrast,optics.plate_scale_as_per_pix."
        ),
    )
    parser.add_argument(
        "--theta-reference-offset",
        action="append",
        default=None,
        metavar="KEY=DELTA",
        help=(
            "Repeatable schur_summary inference/reference-only Theta offset. "
            "The render truth context is unchanged."
        ),
    )
    parser.add_argument(
        "--theta-reference-value",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Repeatable schur_summary inference/reference-only Theta value. "
            "Cannot be combined with an offset for the same key."
        ),
    )
    parser.add_argument(
        "--enable-zernikes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable primary/secondary Zernike groups in schur_summary mode.",
    )
    parser.add_argument(
        "--zernike-indices",
        default=",".join(str(index) for index in DEFAULT_SCHUR_ZERNIKE_INDICES),
        help="Comma-separated Zernike indices used when Zernikes are enabled.",
    )
    parser.add_argument(
        "--schur-damping",
        type=float,
        default=DEFAULT_SCHUR_DAMPING,
        help="Non-negative nuisance damping used by the Schur reduction.",
    )
    parser.add_argument(
        "--max-dense-dim",
        type=int,
        default=DEFAULT_SCHUR_MAX_DENSE_DIM,
        help="Maximum dense combined dimension allowed for schur_summary mode.",
    )
    parser.add_argument(
        "--schur-curvature-method",
        choices=SUPPORTED_SCHUR_CURVATURE_METHODS,
        default=SCHUR_CURVATURE_METHOD_AUTO,
        help=(
            "Curvature path for schur_summary export. auto uses dense below "
            "the max-dense-dim guard and structured_independent_frames for "
            "supported independent-frame layouts above the guard."
        ),
    )
    parser.add_argument(
        "--phi-ref",
        choices=("recovered", "truth_when_available", "truth", "init"),
        default=DEFAULT_SCHUR_PHI_REF,
        help=(
            "Reference fast-state source for schur_summary mode. "
            "Use truth_when_available for the first smoke test."
        ),
    )
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=None,
        help="Optional inference noise-model variance floor override.",
    )
    parser.add_argument(
        "--summary-objective",
        choices=("data_only", "full_objective"),
        default="full_objective",
        help="Local objective variant used for schur_summary mode.",
    )
    parser.add_argument(
        "--validate-surrogate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run local fixed-phi surrogate validation for schur_summary mode.",
    )
    parser.add_argument(
        "--validate-structured-against-dense",
        action="store_true",
        default=False,
        help=(
            "Validation-only: compare structured Schur output to dense autodiff "
            "when structured mode is used and combined_dim <= max_dense_dim."
        ),
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=5,
        help="Number of perturbation points per validated Theta label.",
    )
    parser.add_argument(
        "--schur-frame-quality-policy",
        choices=SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES,
        default=DEFAULT_SCHUR_FRAME_QUALITY_POLICY,
        help="Frame-quality policy for Schur summary export.",
    )
    parser.add_argument(
        "--schur-frame-chi2-threshold",
        type=float,
        default=DEFAULT_SCHUR_FRAME_CHI2_THRESHOLD,
        help="Reduced chi-squared threshold used to flag bad frames.",
    )
    parser.add_argument(
        "--schur-frame-quality-missing",
        choices=SUPPORTED_SCHUR_FRAME_QUALITY_MISSING_POLICIES,
        default=DEFAULT_SCHUR_FRAME_QUALITY_MISSING,
        help="Behavior when recovered-reference frame-quality diagnostics are unavailable.",
    )
    parser.add_argument(
        "--schur-frame-mask-denominator",
        choices=SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS,
        default=DEFAULT_SCHUR_FRAME_MASK_DENOMINATOR,
        help="Denominator convention for mask policy with mean-reduced subblocks.",
    )
    parser.add_argument(
        "--schur-frame-mask-min-good-frames",
        type=int,
        default=DEFAULT_SCHUR_FRAME_MASK_MIN_GOOD_FRAMES,
        help="Minimum good-frame count required when applying mask policy.",
    )
    parser.add_argument(
        "--trace-x0-as",
        type=float,
        default=None,
        help="Override experiment.trace.plan.source.x_position_as.base.",
    )
    parser.add_argument(
        "--trace-y0-as",
        type=float,
        default=None,
        help="Override experiment.trace.plan.source.y_position_as.base.",
    )
    parser.add_argument(
        "--trace-pa0-deg",
        type=float,
        default=None,
        help="Override experiment.trace.plan.source.position_angle_deg.base.",
    )
    parser.add_argument(
        "--trace-jitter-x-sigma-as",
        type=float,
        default=None,
        help="Override X iid_jitter.sigma or random_walk.sigma_step in the trace plan.",
    )
    parser.add_argument(
        "--trace-jitter-y-sigma-as",
        type=float,
        default=None,
        help="Override Y iid_jitter.sigma or random_walk.sigma_step in the trace plan.",
    )
    parser.add_argument(
        "--trace-jitter-pa-sigma-deg",
        type=float,
        default=None,
        help="Override PA iid_jitter.sigma or random_walk.sigma_step in the trace plan.",
    )
    parser.add_argument(
        "--trace-seed",
        type=int,
        default=None,
        help="Override experiment.seed in the copied trace template.",
    )
    parser.add_argument(
        "--render-seed",
        type=int,
        default=None,
        help="Override experiment.seed in the copied render template.",
    )
    parser.add_argument(
        "--init-x-as",
        type=float,
        default=None,
        help="Override experiment.inference.init.frame.values.source.x_position_as.",
    )
    parser.add_argument(
        "--init-y-as",
        type=float,
        default=None,
        help="Override experiment.inference.init.frame.values.source.y_position_as.",
    )
    parser.add_argument(
        "--init-pa-deg",
        type=float,
        default=None,
        help="Override experiment.inference.init.frame.values.source.position_angle_deg.",
    )
    parser.add_argument(
        "--reference-optimizer-kind",
        choices=("sgd", "adam"),
        default=None,
        help="Override experiment.inference.optimizer.kind for recovered-reference inference.",
    )
    parser.add_argument(
        "--reference-base-lr",
        type=float,
        default=None,
        help="Override experiment.inference.optimizer.base_lr for recovered-reference inference.",
    )
    parser.add_argument(
        "--reference-n-iter",
        type=int,
        default=None,
        help="Override experiment.inference.optimizer.n_iter for recovered-reference inference.",
    )
    parser.add_argument(
        "--reference-optimizer-kwarg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Repeatable optimizer kwarg override, for example b1=0.8.",
    )
    parser.add_argument(
        "--reference-schedule-kind",
        choices=(
            "constant",
            "linear_warmup",
            "piecewise_constant",
            "exponential_decay",
            "cosine_decay",
            "linear_warmup_cosine_decay",
        ),
        default=None,
        help="Optional recovered-reference scalar LR schedule override.",
    )
    parser.add_argument("--reference-schedule-warmup-steps", type=int, default=None)
    parser.add_argument("--reference-schedule-start-factor", type=float, default=None)
    parser.add_argument("--reference-schedule-min-factor", type=float, default=None)
    parser.add_argument("--reference-schedule-boundaries", default=None)
    parser.add_argument("--reference-schedule-factors", default=None)
    parser.add_argument("--reference-schedule-decay-rate", type=float, default=None)
    parser.add_argument("--reference-schedule-transition-steps", type=int, default=None)
    parser.add_argument(
        "--reference-schedule-staircase",
        action="store_true",
        default=False,
    )
    preconditioning_group = parser.add_mutually_exclusive_group()
    preconditioning_group.add_argument(
        "--reference-preconditioning-enabled",
        dest="reference_preconditioning_enabled",
        action="store_const",
        const=True,
        default=None,
        help="Enable optimizer preconditioning in the generated reference-inference config.",
    )
    preconditioning_group.add_argument(
        "--reference-preconditioning-disabled",
        dest="reference_preconditioning_enabled",
        action="store_const",
        const=False,
        help="Disable optimizer preconditioning in the generated reference-inference config.",
    )
    parser.add_argument(
        "--reference-preconditioning-method",
        default=None,
        help="Override experiment.inference.optimizer.preconditioning.method.",
    )
    parser.add_argument(
        "--reference-preconditioning-reference",
        choices=("initial", "truth_when_available"),
        default=None,
        help=(
            "Override optimizer preconditioning reference for recovered-reference "
            "inference. Template value is used when omitted."
        ),
    )
    parser.add_argument(
        "--reference-preconditioning-damping",
        type=float,
        default=None,
        help="Override experiment.inference.optimizer.preconditioning.damping.",
    )
    parser.add_argument(
        "--reference-preconditioning-eig-floor-rel",
        type=float,
        default=None,
        help="Override experiment.inference.optimizer.preconditioning.eig_floor_rel.",
    )
    parser.add_argument(
        "--reference-preconditioning-eig-floor-abs",
        type=float,
        default=None,
        help="Override experiment.inference.optimizer.preconditioning.eig_floor_abs.",
    )
    parser.add_argument(
        "--reference-preconditioning-lr-clip",
        default=None,
        help="Override experiment.inference.optimizer.preconditioning.lr_clip as MIN,MAX.",
    )
    parser.add_argument(
        "--reference-diagnostics-profile",
        choices=tuple(sorted(SCHUR_REFERENCE_DIAGNOSTICS_PROFILES)),
        default=None,
        help=(
            "Optional diagnostics patch for recovered-reference review. "
            "Template diagnostics are used when omitted."
        ),
    )
    parser.add_argument(
        "--reuse-reference-inference",
        default=None,
        help=(
            "Reuse an existing recovered-reference inference manifest/dir instead "
            "of rerunning optimization. Use 'auto' for the case-local "
            "study/schur_summary/reference_inference directory."
        ),
    )
    parser.add_argument(
        "--memory-diagnostics",
        action="store_true",
        default=False,
        help="Enable stage-level memory diagnostics for Schur-summary runs.",
    )
    parser.add_argument(
        "--memory-diagnostics-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL output path for memory diagnostics. Defaults under "
            "the schur_summary study directory."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write study templates and summary without executing renders or inference.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    jax.config.update("jax_enable_x64", True)
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
        theta_keys=parse_theta_keys(args.theta_keys),
        enable_zernikes=bool(args.enable_zernikes),
        zernike_indices=parse_zernike_indices(args.zernike_indices),
        schur_damping=float(args.schur_damping),
        max_dense_dim=int(args.max_dense_dim),
        schur_curvature_method=str(args.schur_curvature_method),
        phi_ref=str(args.phi_ref),
        variance_floor=args.variance_floor,
        summary_objective=str(args.summary_objective),
        validate_surrogate=bool(args.validate_surrogate),
        validate_structured_against_dense=bool(
            args.validate_structured_against_dense
        ),
        validation_steps=int(args.validation_steps),
        schur_frame_quality_policy=str(args.schur_frame_quality_policy),
        schur_frame_chi2_threshold=float(args.schur_frame_chi2_threshold),
        schur_frame_quality_missing=str(args.schur_frame_quality_missing),
        schur_frame_mask_denominator=str(args.schur_frame_mask_denominator),
        schur_frame_mask_min_good_frames=int(args.schur_frame_mask_min_good_frames),
        trace_x0_as=args.trace_x0_as,
        trace_y0_as=args.trace_y0_as,
        trace_pa0_deg=args.trace_pa0_deg,
        trace_jitter_x_sigma_as=args.trace_jitter_x_sigma_as,
        trace_jitter_y_sigma_as=args.trace_jitter_y_sigma_as,
        trace_jitter_pa_sigma_deg=args.trace_jitter_pa_sigma_deg,
        trace_seed=args.trace_seed,
        render_seed=args.render_seed,
        init_x_as=args.init_x_as,
        init_y_as=args.init_y_as,
        init_pa_deg=args.init_pa_deg,
        reference_optimizer_kind=args.reference_optimizer_kind,
        reference_base_lr=args.reference_base_lr,
        reference_n_iter=args.reference_n_iter,
        reference_optimizer_kwargs=parse_reference_optimizer_kwargs(
            args.reference_optimizer_kwarg
        ),
        reference_schedule=parse_reference_schedule_config(
            kind=args.reference_schedule_kind,
            warmup_steps=args.reference_schedule_warmup_steps,
            start_factor=args.reference_schedule_start_factor,
            min_factor=args.reference_schedule_min_factor,
            boundaries=args.reference_schedule_boundaries,
            factors=args.reference_schedule_factors,
            decay_rate=args.reference_schedule_decay_rate,
            transition_steps=args.reference_schedule_transition_steps,
            staircase=bool(args.reference_schedule_staircase),
        ),
        reference_preconditioning_enabled=args.reference_preconditioning_enabled,
        reference_preconditioning_method=args.reference_preconditioning_method,
        reference_preconditioning_reference=args.reference_preconditioning_reference,
        reference_preconditioning_damping=args.reference_preconditioning_damping,
        reference_preconditioning_eig_floor_rel=args.reference_preconditioning_eig_floor_rel,
        reference_preconditioning_eig_floor_abs=args.reference_preconditioning_eig_floor_abs,
        reference_preconditioning_lr_clip=parse_reference_preconditioning_lr_clip(
            args.reference_preconditioning_lr_clip
        ),
        reference_diagnostics_profile=args.reference_diagnostics_profile,
        reuse_reference_inference=args.reuse_reference_inference,
        theta_reference_offsets=parse_key_value_float_overrides(
            args.theta_reference_offset,
            option_name="--theta-reference-offset",
        ),
        theta_reference_values=parse_key_value_float_overrides(
            args.theta_reference_value,
            option_name="--theta-reference-value",
        ),
        memory_diagnostics=bool(args.memory_diagnostics),
        memory_diagnostics_file=args.memory_diagnostics_file,
        dry_run=bool(args.dry_run),
    )
    print(f"Study mode: {summary['mode']}")
    print(f"Case root: {summary['case_root']}")
    print(f"Study root: {summary['study_root']}")
    return summary


if __name__ == "__main__":
    main()
