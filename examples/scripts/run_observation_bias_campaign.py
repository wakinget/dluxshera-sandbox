"""Run a small observation-level Zernike bias campaign.

This script orchestrates image-backed Schur-summary sub-block exports with the
existing ``run_obs_subblock_study.py --mode schur_summary`` entrypoint, then
accumulates the summaries into one observation-level belief update per bias
case. The native state stays in the physical parameter basis; eigenmodes are
diagnostic transforms of the accumulated or posterior precision matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shlex
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dluxshera-matplotlib"),
)

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _HAVE_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    _HAVE_MATPLOTLIB = False
import numpy as np

from dluxshera.config.io import load_config_file
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.observation_belief import (
    MatrixDiagnostics,
    ObservationBeliefState,
    ObservationThetaLayout,
    ObservationUpdatePolicy,
    SubblockSummary,
    accumulate_summary_information,
    build_observation_eigenbasis,
    build_prior_whitened_information_gain_matrix,
    build_system_observation_theta_layout,
    update_observation_belief,
    update_observation_belief_with_policy,
)
from dluxshera.inference.observation_forecast import (
    DEFAULT_SYSTEM_PRESET,
    build_default_prior_sigma,
    build_prior_mean_from_store,
    resolve_prior_context_for_summaries,
)
from dluxshera.inference.observation_summary import (
    SUMMARY_SCALE_POLICY_ALLOW_OPTIMIZER,
    SUMMARY_SCALE_POLICY_REQUIRE_SUMMED,
    load_subblock_summary,
    load_subblock_summary_artifact_payload,
    validate_summary_information_scale,
)
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_io import now_iso_local_ms, timestamp_tag
from dluxshera.utils.campaigns import (
    format_shell_command,
    json_ready,
    load_existing_campaign_plan,
    write_shell_command,
)
from dluxshera.utils.campaign_model_split import (
    CampaignModelSplit,
    build_campaign_model_split,
    hash_campaign_model_config,
    template_hash_row,
    validate_campaign_model_split_artifacts,
    write_campaign_model_split_templates,
)
from dluxshera.utils.detector_layer_overrides import (
    apply_detector_layer_overrides,
    detector_blur_warnings,
    detector_layer_stack,
    patch_smear_layer_for_policy,
)
from dluxshera.utils.campaign_trace_sources import (
    PreparedTraceSourcePlan,
    PreparedTraceSubblock,
    prepare_campaign_trace_source,
    trace_subblock_command_flags,
    validate_stored_trace_source_artifacts,
)
from dluxshera.utils.iterative_campaigns import (
    apply_physical_reference_update,
    posterior_float,
    posterior_offsets_from_rows,
    posterior_rows_by_label,
    separation_update_diagnostics,
    vector_update_diagnostics,
)
from dluxshera.utils.campaign_truth import realize_campaign_truth as _realize_campaign_truth
from dluxshera.utils.obs_subblock_cli import (
    REFERENCE_EARLY_STOPPING_FLAG_MAP,
    REFERENCE_OPTIMIZER_FLAG_MAP,
    REFERENCE_PRECONDITIONING_FLAG_MAP,
    REFERENCE_SCHEDULE_FLAG_MAP,
    SCHUR_FRAME_QUALITY_FLAG_MAP,
    append_reference_optimizer_flags,
    append_schur_frame_quality_flags,
)
from dluxshera.utils.obs_subblock_keys import (
    parse_obs_subblock_key_address,
)
from dluxshera.utils.seeding import derive_campaign_subblock_seeds
from dluxshera.utils.noise import (
    NormalizedSubblockNoiseConfig,
    make_subseed,
    normalize_subblock_noise_config,
)
from dluxshera.utils.smear_audit import (
    build_smear_summary_rows,
    kernel_matches,
    load_named_smear_kernel,
    plan_smear_rows,
)
from dluxshera.utils.subprocess_diagnostics import (
    require_resource_time_available,
    run_subprocess_with_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "observation_bias_campaign"
SUBBLOCK_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_study.py"
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
CAMPAIGN_SCHEMA_VERSION = "observation_bias_campaign.v1"
DEFAULT_LOCAL_ELIMINATED_KEYS = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)
SUPPORTED_SEED_POLICIES = (
    "different_jitter_different_noise",
    "same_jitter_different_noise",
    "different_jitter_same_noise",
)
SUPPORTED_EIGEN_SOURCES = (
    "accumulated_information",
    "posterior_precision",
)
SUPPORTED_FORECAST_MODES = (
    "replicate",
    "fixed_information_score_noise",
)
SUPPORTED_ITERATIVE_FORECAST_MODES = (
    "replicate_information",
    "stochastic_score_noise",
)
SUPPORTED_REFERENCE_DIAGNOSTICS_PROFILES = ("none", "basic", "review", "full")
SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES = ("warn", "mask", "reject")
SUPPORTED_SCHUR_FRAME_QUALITY_MISSING = ("allow_all", "error")
SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS = ("original", "kept")
SUPPORTED_ITERATIVE_UPDATE_MODES = (
    "physical_full",
    "eigen_full",
    "eigen_damped",
    "eigen_truncated",
)
SUPPORTED_RENDER_RETENTION_POLICIES = (
    "keep",
    "delete_after_window",
)
RENDER_RETENTION_DEFAULT = "keep"
RENDER_RETENTION_ALLOWED_SUFFIXES = ("_cube.fits", "_variance.fits")
SUBBLOCK_OPTION_FLAG_MAP = {
    "summary_information_scale": "--summary-information-scale",
    "exposure_time_s": "--exposure-time-s",
    "variance_floor": "--variance-floor",
    **REFERENCE_OPTIMIZER_FLAG_MAP,
    **REFERENCE_SCHEDULE_FLAG_MAP,
    **REFERENCE_PRECONDITIONING_FLAG_MAP,
    **REFERENCE_EARLY_STOPPING_FLAG_MAP,
    **SCHUR_FRAME_QUALITY_FLAG_MAP,
}


@dataclass(frozen=True)
class BiasCase:
    case_name: str
    theta_reference_offsets: dict[str, float]
    case_origin: str = "explicit"
    prior_sigma_by_label: dict[str, float] | None = None
    prior_draw_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CampaignPlan:
    run_root: Path
    layout: ObservationThetaLayout
    layout_metadata: dict[str, Any]
    prior_truth: np.ndarray
    cases: tuple[BiasCase, ...]
    subblock_commands: dict[str, list[list[str]]]
    summary_paths: dict[str, list[Path]]
    subblock_plans: dict[str, list[dict[str, Any]]]
    prior_draw_rows_by_case: dict[str, list[dict[str, Any]]]
    config: dict[str, Any]
    partition: dict[str, Any]
    case_generation: dict[str, Any]
    truth_realization: dict[str, Any]
    truth_realization_rows: list[dict[str, Any]]
    trace_source_plan: PreparedTraceSourcePlan
    iterative: dict[str, Any]
    iterative_plan_rows: list[dict[str, Any]]
    expected_output_rows: list[dict[str, Any]]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_ready(dict(payload)), sort_keys=True) + "\n")


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure_dir(path.parent)
    rows = list(rows)
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
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_observation_bias_templates(
    run_root: Path,
    *,
    model_split: Any,
    normalized_noise: NormalizedSubblockNoiseConfig | None = None,
) -> dict[str, Path]:
    template_root = run_root / "templates"
    payloads = {
        "trace": load_config_file(DEFAULT_SCHUR_TRACE_TEMPLATE),
        "render": load_config_file(DEFAULT_RENDER_TEMPLATE),
        "inference": load_config_file(DEFAULT_INFERENCE_TEMPLATE),
    }
    if normalized_noise is not None:
        _patch_render_payload_noise(payloads["render"], normalized_noise)
    return write_campaign_model_split_templates(
        template_root=template_root,
        trace_payload=payloads["trace"],
        render_payload=payloads["render"],
        inference_payload=payloads["inference"],
        split=model_split,
    )


def _patch_render_payload_noise(
    render_payload: dict[str, Any],
    normalized_noise: NormalizedSubblockNoiseConfig,
) -> dict[str, Any]:
    block = normalized_noise.render_template_noise_block()
    if block is None:
        return render_payload
    experiment = render_payload.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("render template experiment block must be a mapping.")
    noise = experiment.setdefault("noise", {})
    if not isinstance(noise, dict):
        raise ValueError("render template experiment.noise block must be a mapping.")
    noise.update(block)
    return render_payload


def _write_noise_provenance(
    *,
    run_root: Path,
    normalized_noise: NormalizedSubblockNoiseConfig,
    render_template_noise_block: Mapping[str, Any] | None,
) -> dict[str, Path]:
    noise_root = run_root / "noise"
    noise_root.mkdir(parents=True, exist_ok=True)
    normalized_path = noise_root / "noise_request_normalized.json"
    render_path = noise_root / "noise_render_provenance.json"
    inference_path = noise_root / "noise_inference_provenance.json"
    normalized_payload = normalized_noise.to_dict()
    normalized_path.write_text(json.dumps(json_ready(normalized_payload), indent=2), encoding="utf-8")
    render_path.write_text(
        json.dumps(
            json_ready(
                {
                    "structured_noise_supported": True,
                    "render_terms_forwarded": render_template_noise_block is not None,
                    "legacy_noise_mode": normalized_noise.legacy_noise_mode,
                    "render_template_noise_block": dict(render_template_noise_block or {}),
                    "warnings": list(normalized_noise.warnings),
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    inference_path.write_text(
        json.dumps(
            json_ready(
                {
                    "inference_variance_floor_forwarded": normalized_noise.variance_floor is not None,
                    "inference_variance_floor": normalized_noise.variance_floor,
                    "variance_floor_source": normalized_noise.variance_floor_source,
                    "use_render_variance": normalized_noise.use_render_variance,
                    "use_render_variance_resolved": normalized_noise.use_render_variance_resolved,
                    "use_render_variance_source": normalized_noise.use_render_variance_source,
                    "warnings": list(normalized_noise.warnings),
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "noise_request_normalized_json": normalized_path,
        "noise_render_provenance_json": render_path,
        "noise_inference_provenance_json": inference_path,
    }


def _smear_render_mode(subblock_cfg: Mapping[str, Any]) -> str:
    processing = subblock_cfg.get("trajectory_processing")
    smear = processing.get("smear", {}) if isinstance(processing, Mapping) else {}
    render = smear.get("render", {}) if isinstance(smear, Mapping) else {}
    enabled = bool(smear.get("enabled", False)) if isinstance(smear, Mapping) else False
    return str(render.get("mode", "metadata_only" if enabled else "disabled"))


def _smear_cfg(subblock_cfg: Mapping[str, Any]) -> dict[str, Any]:
    processing = subblock_cfg.get("trajectory_processing")
    smear = processing.get("smear", {}) if isinstance(processing, Mapping) else {}
    return dict(smear) if isinstance(smear, Mapping) else {}


def _update_subblock_smear_provenance(
    *,
    trace_row: Mapping[str, Any],
    subblock_index: int,
    window_index: int | str,
    subblock_cfg: Mapping[str, Any],
    template_paths: Mapping[str, Path],
    representative_kernel: Mapping[str, Any],
) -> None:
    provenance_raw = trace_row.get("smear_provenance_json")
    if not provenance_raw:
        return
    provenance_path = Path(str(provenance_raw))
    if not provenance_path.exists():
        return
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    smear_cfg = _smear_cfg(subblock_cfg)
    render_cfg = smear_cfg.get("render", {}) if isinstance(smear_cfg.get("render"), Mapping) else {}
    inference_cfg = smear_cfg.get("inference", {}) if isinstance(smear_cfg.get("inference"), Mapping) else {}
    target_layer = str(render_cfg.get("target_layer", render_cfg.get("layer_name", "smear")))
    render_path = Path(template_paths["render"]).resolve()
    inference_path = Path(template_paths["inference"]).resolve()
    render_kernel = load_named_smear_kernel(render_path, layer_name=target_layer)
    inference_kernel = load_named_smear_kernel(inference_path, layer_name=target_layer)
    render_match = kernel_matches(representative_kernel, render_kernel)
    inference_match = kernel_matches(representative_kernel, inference_kernel)
    warnings_out = list(payload.get("warnings", [])) if isinstance(payload.get("warnings"), list) else []
    if not render_match:
        warnings_out.append("render_template_smear_kernel_mismatch")
    if not inference_match:
        warnings_out.append("inference_template_smear_kernel_mismatch")
    payload.update(
        {
            "subblock_index": int(subblock_index),
            "window_index": window_index,
            "render_mode": str(render_cfg.get("mode", payload.get("render_mode", ""))),
            "inference_mode": str(inference_cfg.get("mode", payload.get("inference_mode", "matched_subblock_constant"))),
            "target_layer": target_layer,
            "source": representative_kernel.get("source", payload.get("source", "")),
            "exposure_time_s": representative_kernel.get("exposure_time_s", payload.get("exposure_time_s")),
            "plate_scale_as_per_pix": representative_kernel.get(
                "plate_scale_as_per_pix",
                payload.get("plate_scale_as_per_pix"),
            ),
            "representative_kernel": dict(representative_kernel),
            "truth_kernel": dict(representative_kernel),
            "model_kernel": dict(representative_kernel)
            if str(inference_cfg.get("mode", "matched_subblock_constant")) in {"matched", "matched_subblock_constant"}
            else dict(payload.get("model_kernel", representative_kernel)),
            "matched_model": bool(inference_match),
            "pa_smear_mode": "ignored_for_line_kernel",
            "input_frame_truth_csv": str(trace_row.get("frame_truth_path", payload.get("input_frame_truth_csv", ""))),
            "frame_smear_truth_csv": str(trace_row.get("smear_truth_csv", payload.get("frame_smear_truth_csv", ""))),
            "frame_smear_model_csv": str(trace_row.get("smear_model_csv", payload.get("frame_smear_model_csv", ""))),
            "render_template_path": str(render_path),
            "inference_template_path": str(inference_path),
            "render_template_kernel": render_kernel,
            "inference_template_kernel": inference_kernel,
            "render_template_match": bool(render_match),
            "inference_template_match": bool(inference_match),
            "warnings": warnings_out,
        }
    )
    provenance_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_subblock_smear_templates(
    *,
    run_root: Path,
    subblock_index: int,
    window_index: int | str,
    model_split: CampaignModelSplit,
    subblock_cfg: Mapping[str, Any],
    trace_row: Mapping[str, Any],
    normalized_noise: NormalizedSubblockNoiseConfig | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    kernel_raw = trace_row.get("smear_representative_kernel_json")
    if not kernel_raw:
        raise ValueError(
            "subblock_constant_layer requested, but no trajectory-derived smear kernel "
            f"was prepared for subblock {subblock_index}."
        )
    representative_kernel = json.loads(str(kernel_raw))
    smear_cfg = _smear_cfg(subblock_cfg)
    truth_system, truth_policy = patch_smear_layer_for_policy(
        model_split.truth_system_cfg,
        smear_cfg,
        representative_kernel=representative_kernel,
        context=f"subblock_{subblock_index:06d}.truth",
        strict=True,
    )
    inference_cfg = dict(smear_cfg)
    inference_mode = str(
        (inference_cfg.get("inference", {}) or {}).get("mode", "matched_subblock_constant")
        if isinstance(inference_cfg.get("inference", {}), Mapping)
        else "matched_subblock_constant"
    )
    if inference_mode == "disabled":
        render = inference_cfg.get("render", {}) if isinstance(inference_cfg.get("render"), Mapping) else {}
        inference_cfg = {
            "enabled": False,
            "render": {"mode": "disabled", "target_layer": render.get("target_layer", "smear")},
        }
    elif inference_mode in {"matched", "matched_subblock_constant"}:
        pass
    elif inference_mode in {"solve_subblock_smear", "per_frame", "mismatch_scaled", "mismatch_angle_offset"}:
        raise ValueError(f"smear.inference.mode={inference_mode!r} is future/deferred and not implemented.")
    else:
        raise ValueError(f"Unsupported smear.inference.mode {inference_mode!r}.")
    inference_system, inference_policy = patch_smear_layer_for_policy(
        model_split.inference_system_cfg,
        inference_cfg,
        representative_kernel=representative_kernel,
        context=f"subblock_{subblock_index:06d}.inference",
        strict=True,
    )
    sub_split = replace(
        model_split,
        truth_system_cfg=truth_system,
        inference_system_cfg=inference_system,
        truth_config_hash=hash_campaign_model_config(truth_system),
        inference_config_hash=hash_campaign_model_config(inference_system),
        provenance={
            **model_split.provenance,
            "subblock_smear_template_patch": {
                "subblock_index": int(subblock_index),
                "representative_kernel": representative_kernel,
                "truth_policy": truth_policy,
                "inference_policy": inference_policy,
            },
        },
    )
    template_root = run_root / "trajectory" / f"subblock_{subblock_index:06d}" / "templates"
    payloads = {
        "trace": load_config_file(DEFAULT_SCHUR_TRACE_TEMPLATE),
        "render": load_config_file(DEFAULT_RENDER_TEMPLATE),
        "inference": load_config_file(DEFAULT_INFERENCE_TEMPLATE),
    }
    if normalized_noise is not None:
        _patch_render_payload_noise(payloads["render"], normalized_noise)
    paths = write_campaign_model_split_templates(
        template_root=template_root,
        trace_payload=payloads["trace"],
        render_payload=payloads["render"],
        inference_payload=payloads["inference"],
        split=sub_split,
    )
    _update_subblock_smear_provenance(
        trace_row=trace_row,
        subblock_index=subblock_index,
        window_index=window_index,
        subblock_cfg=subblock_cfg,
        template_paths=paths,
        representative_kernel=representative_kernel,
    )
    return paths, template_hash_row(paths, sub_split)


def _resolve_total_subblocks_for_campaign(
    experiment_cfg: Mapping[str, Any],
    subblock_cfg: Mapping[str, Any],
    iterative_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    iterative_enabled = bool(iterative_cfg.get("enabled", False))
    raw_n = subblock_cfg.get("n_subblocks")
    trace = subblock_cfg.get("trace_source", {}) if isinstance(subblock_cfg.get("trace_source"), Mapping) else {}
    window = trace.get("window", {}) if isinstance(trace.get("window"), Mapping) else {}
    trace_window_n = window.get("n_subblocks")
    if iterative_enabled:
        if iterative_cfg.get("subblocks_per_window") is None:
            if raw_n is None:
                raise ValueError(
                    "experiment.iterative.enabled=true requires iterative.subblocks_per_window "
                    "or an explicit subblocks.n_subblocks fallback."
                )
            subblocks_per_window = int(raw_n)
        else:
            subblocks_per_window = int(iterative_cfg["subblocks_per_window"])
        windows_per_draw = int(iterative_cfg.get("windows_per_draw", 1))
        total = windows_per_draw * subblocks_per_window
        source = "experiment.iterative.windows_per_draw*subblocks_per_window"
        if raw_n is not None and int(raw_n) != total:
            raise ValueError(
                "experiment.subblocks.n_subblocks conflicts with iterative grouping: "
                f"{int(raw_n)} != {windows_per_draw}*{subblocks_per_window}={total}."
            )
    else:
        windows_per_draw = int(iterative_cfg.get("windows_per_draw", 1))
        subblocks_per_window = int(iterative_cfg.get("subblocks_per_window", raw_n or 1))
        total = int(raw_n if raw_n is not None else 1)
        source = "experiment.subblocks.n_subblocks" if raw_n is not None else "default:1"
    if total <= 0:
        raise ValueError("resolved total subblock count must be positive.")
    if trace_window_n is not None and int(trace_window_n) != total:
        raise ValueError(
            "experiment.subblocks.trace_source.window.n_subblocks conflicts with resolved "
            f"subblock count: {int(trace_window_n)} != {total}."
        )
    return {
        "resolved_total_subblocks": int(total),
        "resolved_windows_per_draw": int(windows_per_draw),
        "resolved_subblocks_per_window": int(subblocks_per_window),
        "subblock_count_source": source,
        "trace_source_window_n_subblocks": None if trace_window_n is None else int(trace_window_n),
        "consistency_status": "consistent",
    }


def _matrix_diagnostics(matrix: np.ndarray) -> MatrixDiagnostics:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    if matrix.size == 0:
        return MatrixDiagnostics(0, 0.0, 0.0, 1.0, 0.0, 0.0)
    eigenvalues = np.linalg.eigvalsh(matrix)
    tol = np.finfo(float).eps * max(matrix.shape) * max(float(np.max(np.abs(eigenvalues))), 1.0)
    positive = eigenvalues[eigenvalues > tol]
    condition = float("inf") if positive.size == 0 else float(np.max(positive) / np.min(positive))
    return MatrixDiagnostics(
        rank_estimate=int(np.count_nonzero(np.abs(eigenvalues) > tol)),
        min_eigenvalue=float(np.min(eigenvalues)),
        max_eigenvalue=float(np.max(eigenvalues)),
        condition_number=condition,
        trace=float(np.trace(matrix)),
        frobenius_norm=float(np.linalg.norm(matrix)),
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "unnamed"


def _label_group(label: str) -> str:
    if label.startswith("source."):
        return "source"
    if label == "optics.plate_scale_as_per_pix":
        return "optics.plate_scale"
    if label.startswith("optics.primary.zernike_coeffs_nm"):
        return "optics.primary_zernikes"
    if label.startswith("optics.secondary.zernike_coeffs_nm"):
        return "optics.secondary_zernikes"
    return "other"


def _parameter_unit(label: str) -> str:
    if label == "source.separation_as":
        return "arcsec"
    if label == "source.log_flux_total":
        return "log flux"
    if label == "source.contrast":
        return "dimensionless"
    if label == "optics.plate_scale_as_per_pix":
        return "arcsec / pixel"
    if "zernike_coeffs_nm" in label:
        return "nm"
    return "arb"


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-30:
        return float("nan")
    return float(numerator / denominator)


def _load_campaign_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"experiment": _default_experiment_config()}
    payload = load_config_file(path.resolve())
    experiment = payload.get("experiment", payload)
    if not isinstance(experiment, Mapping):
        raise ValueError("Campaign config must contain a mapping-valued experiment block.")
    if "experiment" not in payload:
        payload = {"experiment": dict(experiment)}
    return dict(payload)


def _default_experiment_config() -> dict[str, Any]:
    return {
        "kind": "observation_bias_campaign",
        "seed": 42,
        "run_name": "full_zernike_bias_smoke",
        "subblocks": {
            "n_subblocks": 3,
            "n_frames": 3,
            "noise": "disabled",
            "phi_ref": "truth_when_available",
            "schur_curvature_method": "auto",
            "max_dense_dim": 40,
            "schur_damping": 1.0e-8,
            "summary_information_scale": "summed_likelihood",
        },
        "seeding": {
            "seed_policy": "different_jitter_different_noise",
            "base_seed": 42,
        },
        "observation_theta": {
            "source": {
                "separation_as": True,
                "log_flux_total": True,
                "contrast": True,
            },
            "optics": {
                "plate_scale_as_per_pix": True,
                "primary_zernikes": {
                    "enabled": True,
                    "indices": "from_system",
                    "include": None,
                    "exclude": [],
                },
                "secondary_zernikes": {
                    "enabled": True,
                    "indices": "from_system",
                    "include": None,
                    "exclude": [],
                },
            },
        },
        "eigenbasis": {
            "enabled": True,
            "sources": ["accumulated_information", "posterior_precision"],
            "whiten": True,
            "eig_floor_abs": 0.0,
            "eig_floor_rel": 1.0e-12,
            "top_k_contributors": 8,
        },
        "bias_cases": [
            {"case_name": "zero_bias_full_zernike", "theta_reference_offsets": {}},
        ],
        "prior_draws": {
            "enabled": False,
            "n_cases": 3,
            "center": "truth",
            "distribution": "normal",
            "draw_seed": 12345,
            "case_name_template": "prior_draw_{draw_index:03d}",
            "sigmas": {
                "source.separation_as": {"kind": "absolute", "sigma": 1.0e-5, "unit": "arcsec"},
                "source.log_flux_total": {"kind": "absolute", "sigma": 1.0e-5, "unit": "log_flux"},
                "source.contrast": {"kind": "fractional", "sigma": 1.0e-5},
                "optics.plate_scale_as_per_pix": {"kind": "fractional", "sigma": 1.0e-5},
                "optics.primary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 1.0e-1, "unit": "nm"},
                "optics.secondary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 1.0e-1, "unit": "nm"},
            },
        },
        "truth_realization": {
            "enabled": False,
            "seed": 20260521,
            "mode": "zernike_per_coefficient_sigma",
            "zernikes": {
                "primary": {
                    "enabled": True,
                    "indices": "from_observation_theta",
                    "mean_nm": 0.0,
                    "sigma_nm": 5.0,
                },
                "secondary": {
                    "enabled": True,
                    "indices": "from_observation_theta",
                    "mean_nm": 0.0,
                    "sigma_nm": 2.0,
                },
            },
        },
        "forecast": {
            "enabled": True,
            "modes": ["replicate", "fixed_information_score_noise"],
            "n_subblocks_grid": [1, 3, 5, 10, 30, 100, 300, 1000, 1800],
            "subblock_duration_s": 1.0,
            "single_observation_n_subblocks": 1800,
            "replicate": {"enabled": True},
            "fixed_information_score_noise": {
                "enabled": True,
                "n_trials": 100,
                "seed": 2026,
                "score_noise_alpha": 1.0,
                "score_noise_eig_floor_abs": 0.0,
                "score_noise_eig_floor_rel": 1.0e-12,
                "truth_mode": "campaign_truth",
            },
            "plots": True,
        },
    }


def _experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment", config)
    if not isinstance(experiment, Mapping):
        raise ValueError("Campaign experiment config must be a mapping.")
    return dict(experiment)


def resolve_campaign_system_preset(
    *,
    config: Mapping[str, Any],
    experiment_cfg: Mapping[str, Any],
    cli_system_preset: str | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Resolve campaign preset with explicit observation-bias precedence.

    Precedence is top-level ``system.preset``, ``experiment.system_preset``,
    ``experiment.system.preset``, an explicit CLI fallback, then the legacy
    observation-bias default.
    """

    top_system = config.get("system") if isinstance(config.get("system"), Mapping) else {}
    exp_system = experiment_cfg.get("system") if isinstance(experiment_cfg.get("system"), Mapping) else {}
    candidates = {
        "cli.system_preset": cli_system_preset,
        "top_level_system.preset": top_system.get("preset") if isinstance(top_system, Mapping) else None,
        "experiment.system_preset": experiment_cfg.get("system_preset"),
        "experiment.system.preset": exp_system.get("preset") if isinstance(exp_system, Mapping) else None,
        "legacy_default": DEFAULT_SYSTEM_PRESET,
    }
    source = "legacy_default"
    resolved = DEFAULT_SYSTEM_PRESET
    for key in (
        "top_level_system.preset",
        "experiment.system_preset",
        "experiment.system.preset",
        "cli.system_preset",
    ):
        value = candidates.get(key)
        if value is not None:
            resolved = str(value)
            source = key
            break
    warnings_out: list[str] = []
    source_kind = str(experiment_cfg.get("source_campaign_kind", experiment_cfg.get("kind", "")))
    is_full_fidelity = source_kind == "full_fidelity_binary_iterative"
    values = {
        key: str(value)
        for key, value in candidates.items()
        if key != "legacy_default" and value is not None
    }
    if is_full_fidelity and len(set(values.values())) > 1:
        message = (
            "Conflicting system preset candidates in full-fidelity translated config: "
            + json.dumps(values, sort_keys=True)
        )
        if strict:
            raise ValueError(message)
        warnings_out.append(message)
    return {
        "system_preset": resolved,
        "source": source,
        "candidates": candidates,
        "warnings": warnings_out,
        "full_fidelity_source": is_full_fidelity,
    }


def _resolve_system_store(
    *,
    config: Mapping[str, Any],
    system_preset: str | None,
    preset_resolution: Mapping[str, Any] | None = None,
    exposure_time_s: float | None = None,
) -> tuple[ParameterStore, dict[str, Any], dict[str, Any]]:
    user_cfg = dict(config)
    preset = str((preset_resolution or {}).get("system_preset") or (DEFAULT_SYSTEM_PRESET if system_preset is None else system_preset))
    if "system" not in user_cfg:
        user_cfg = {**user_cfg, "system": {"preset": preset}}
    elif preset is not None:
        system_cfg = dict(user_cfg["system"])
        system_cfg["preset"] = preset
        user_cfg["system"] = system_cfg
    if exposure_time_s is not None:
        exposure = float(exposure_time_s)
        if exposure <= 0.0 or not math.isfinite(exposure):
            raise ValueError("experiment.subblocks.exposure_time_s must be positive.")
        system_cfg = dict(user_cfg.get("system", {}) or {})
        source_cfg = dict(system_cfg.get("source", {}) or {})
        source_cfg["exposure_time_s"] = exposure
        system_cfg["source"] = source_cfg
        user_cfg["system"] = system_cfg
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    if not isinstance(system_cfg, Mapping):
        raise ValueError("Campaign system resolution requires a resolved system block.")
    experiment_cfg = _experiment_config(config)
    detector_overrides = experiment_cfg.get("detector_overrides")
    detector_override_provenance: dict[str, Any] | None = None
    detector_stack_from_preset = detector_layer_stack(system_cfg)
    if isinstance(detector_overrides, Mapping):
        system_cfg, detector_override_provenance = apply_detector_layer_overrides(
            system_cfg,
            detector_overrides,
            context="observation_bias_campaign.global",
        )
    forward_spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    subblock_cfg = experiment_cfg.get("subblocks", {}) if isinstance(experiment_cfg.get("subblocks"), Mapping) else {}
    trajectory_processing = (
        subblock_cfg.get("trajectory_processing", {})
        if isinstance(subblock_cfg.get("trajectory_processing"), Mapping)
        else {}
    )
    smear_cfg = trajectory_processing.get("smear", {}) if isinstance(trajectory_processing.get("smear"), Mapping) else {}
    provenance = {
        "system_preset": system_cfg.get("preset", preset),
        "source_kind": system_cfg.get("source", {}).get("kind"),
        "source_target": system_cfg.get("source", {}).get("target"),
        "optics_kind": system_cfg.get("optics", {}).get("kind"),
        "detector_model": system_cfg.get("detector", {}).get("model"),
        "detector_layer_stack_from_preset": detector_stack_from_preset,
        "detector_layer_stack_after_global_overrides": detector_layer_stack(system_cfg),
        "detector_layer_overrides": detector_override_provenance,
        "detector_blur_warnings": detector_blur_warnings(system_cfg, smear_cfg=smear_cfg),
        "system_preset_resolution": dict(preset_resolution or {}),
    }
    return store, dict(system_cfg), provenance


def _subblock_exposure_time_s(experiment_cfg: Mapping[str, Any]) -> float | None:
    subblock_cfg = experiment_cfg.get("subblocks", {}) or {}
    if not isinstance(subblock_cfg, Mapping) or subblock_cfg.get("exposure_time_s") is None:
        return None
    exposure = float(subblock_cfg["exposure_time_s"])
    if exposure <= 0.0 or not math.isfinite(exposure):
        raise ValueError("experiment.subblocks.exposure_time_s must be positive.")
    return exposure


def _store_scalar_value(store: ParameterStore, key: str) -> float | None:
    try:
        return float(np.asarray(store.get(key)))
    except Exception:
        return None


def _validate_partition_config(partition_cfg: Mapping[str, Any] | None, layout: ObservationThetaLayout) -> dict[str, Any]:
    cfg = dict(partition_cfg or {})
    local_eliminated = tuple(cfg.get("local_eliminated_keys", DEFAULT_LOCAL_ELIMINATED_KEYS))
    if local_eliminated != DEFAULT_LOCAL_ELIMINATED_KEYS:
        raise ValueError(
            "For this campaign, state_partition.local_eliminated_keys must match "
            + ", ".join(DEFAULT_LOCAL_ELIMINATED_KEYS)
            + "."
        )
    shared_active = tuple(cfg.get("subblock_shared_active_keys", ()) or ())
    if shared_active:
        raise NotImplementedError(
            "state_partition.subblock_shared_active_keys is accepted for future "
            "work but is not implemented by this registration-only campaign."
        )
    return {
        "observation_theta_keys": list(layout.labels),
        "local_eliminated_keys": list(local_eliminated),
        "subblock_shared_active_keys": list(shared_active),
        "report_only_keys": list(cfg.get("report_only_keys", ()) or ()),
    }


def _parse_bias_cases(
    experiment_cfg: Mapping[str, Any],
    *,
    layout: ObservationThetaLayout,
    layout_metadata: Mapping[str, Any],
) -> tuple[BiasCase, ...]:
    label_set = set(layout.labels)
    raw_cases = experiment_cfg.get("bias_cases", [])
    cases: list[BiasCase] = []
    if raw_cases is not None:
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise ValueError("experiment.bias_cases must be a list.")
        for raw_case in raw_cases:
            if not isinstance(raw_case, Mapping):
                raise ValueError("Each bias case must be a mapping.")
            name = str(raw_case.get("case_name", "")).strip()
            if not name:
                raise ValueError("Each bias case requires a non-empty case_name.")
            offsets_raw = raw_case.get("theta_reference_offsets", {}) or {}
            if not isinstance(offsets_raw, Mapping):
                raise ValueError(f"Bias case {name!r} offsets must be a mapping.")
            offsets: dict[str, float] = {}
            for raw_key, raw_value in offsets_raw.items():
                address = parse_obs_subblock_key_address(str(raw_key))
                key = address.canonical
                if key not in label_set:
                    raise ValueError(
                        f"Bias case {name!r} references {key!r}, which is not in "
                        "the resolved observation theta layout."
                    )
                offsets[key] = float(raw_value)
            prior_sigmas_raw = raw_case.get("prior_sigmas")
            prior_sigma_by_label: dict[str, float] | None = None
            if prior_sigmas_raw is not None:
                if not isinstance(prior_sigmas_raw, Mapping):
                    raise ValueError(f"Bias case {name!r} prior_sigmas must be a mapping.")
                prior_sigma_by_label = {}
                for label, value in prior_sigmas_raw.items():
                    canonical = parse_obs_subblock_key_address(str(label)).canonical
                    if canonical not in label_set:
                        raise ValueError(
                            f"Bias case {name!r} prior_sigmas references {canonical!r}, "
                            "which is not in the resolved observation theta layout."
                        )
                    sigma = float(value)
                    if sigma <= 0.0 or not math.isfinite(sigma):
                        raise ValueError(f"Bias case {name!r} prior sigma must be positive.")
                    prior_sigma_by_label[canonical] = sigma
            cases.append(
                BiasCase(
                    case_name=name,
                    theta_reference_offsets=offsets,
                    case_origin="explicit",
                    prior_sigma_by_label=prior_sigma_by_label,
                )
            )

    auto_cfg = experiment_cfg.get("auto_cases", {}) or {}
    if isinstance(auto_cfg, Mapping) and bool(auto_cfg.get("enabled", False)):
        cases.extend(
            _generate_auto_cases(
                layout=layout,
                layout_metadata=layout_metadata,
                amplitude_nm=float(auto_cfg.get("zernike_pair_amplitude_nm", 5.0)),
                include_zero_bias=bool(auto_cfg.get("include_zero_bias", True)),
                include_matched=bool(auto_cfg.get("include_matched_pairs", True)),
                include_differential=bool(auto_cfg.get("include_differential_pairs", True)),
                max_cases=int(auto_cfg.get("max_cases", 6)),
            )
        )

    seen: set[str] = set()
    unique_cases: list[BiasCase] = []
    for case in cases:
        if case.case_name in seen:
            raise ValueError(f"Duplicate bias case name: {case.case_name}.")
        seen.add(case.case_name)
        unique_cases.append(case)
    return tuple(unique_cases)


def _resolve_case_generation(
    experiment_cfg: Mapping[str, Any],
    *,
    parsed_cases: Sequence[BiasCase],
    prior_draw_cases: Sequence[BiasCase],
) -> tuple[tuple[BiasCase, ...], dict[str, Any]]:
    raw_cfg = experiment_cfg.get("case_generation", {}) or {}
    if not isinstance(raw_cfg, Mapping):
        raise ValueError("experiment.case_generation must be a mapping when provided.")
    include_key_present = "include_implicit_zero_bias" in raw_cfg
    include_implicit = bool(raw_cfg.get("include_implicit_zero_bias", True))
    explicit_present = any(case.case_origin == "explicit" for case in parsed_cases)
    auto_present = any(case.case_origin.startswith("auto") for case in parsed_cases)
    has_configured_cases = bool(parsed_cases)
    has_prior_draw_cases = bool(prior_draw_cases)

    add_implicit = False
    zero_bias_status = "not_added"
    if explicit_present or auto_present:
        if include_key_present and include_implicit:
            add_implicit = True
            zero_bias_status = "implicit_requested_with_configured_cases"
        else:
            zero_bias_status = "not_added_configured_cases_present"
    elif has_prior_draw_cases:
        if include_implicit:
            add_implicit = True
            zero_bias_status = "implicit_default_with_prior_draws"
        else:
            zero_bias_status = "disabled"
    else:
        if include_implicit:
            add_implicit = True
            zero_bias_status = "implicit_default_no_cases"
        else:
            raise ValueError(
                "No bias cases or prior draws were configured, and "
                "case_generation.include_implicit_zero_bias is false."
            )

    cases: list[BiasCase] = []
    if add_implicit:
        cases.append(
            BiasCase("zero_bias_full_zernike", {}, case_origin="implicit_zero_bias")
        )
    cases.extend(parsed_cases)
    cases.extend(prior_draw_cases)
    names = [case.case_name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate case names across configured, implicit, and prior-draw cases.")
    metadata = {
        "include_implicit_zero_bias": include_implicit,
        "include_implicit_zero_bias_configured": include_key_present,
        "zero_bias_case_status": zero_bias_status,
        "n_configured_cases": len(parsed_cases),
        "n_prior_draw_cases": len(prior_draw_cases),
        "n_total_cases": len(cases),
    }
    return tuple(cases), metadata


def _generate_auto_cases(
    *,
    layout: ObservationThetaLayout,
    layout_metadata: Mapping[str, Any],
    amplitude_nm: float,
    include_zero_bias: bool,
    include_matched: bool,
    include_differential: bool,
    max_cases: int,
) -> list[BiasCase]:
    cases: list[BiasCase] = []
    if include_zero_bias:
        cases.append(BiasCase("zero_bias_full_zernike", {}))
    primary = set(int(index) for index in layout_metadata.get("primary_zernike_indices", []))
    secondary = set(int(index) for index in layout_metadata.get("secondary_zernike_indices", []))
    for index in sorted(primary & secondary):
        m1 = f"optics.primary.zernike_coeffs_nm[{index}]"
        m2 = f"optics.secondary.zernike_coeffs_nm[{index}]"
        if m1 not in layout.labels or m2 not in layout.labels:
            continue
        if include_matched:
            cases.append(
                BiasCase(
                    f"matched_zernike_pair_{index}_plus",
                    {m1: float(amplitude_nm), m2: float(amplitude_nm)},
                    case_origin="auto_matched_pair",
                )
            )
        if include_differential:
            cases.append(
                BiasCase(
                    f"differential_zernike_pair_{index}_plus",
                    {m1: float(amplitude_nm), m2: -float(amplitude_nm)},
                    case_origin="auto_differential_pair",
                )
            )
        if len(cases) >= max_cases:
            break
    return cases[:max_cases]


def _apply_cli_overrides(experiment_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(json.dumps(experiment_cfg))
    subblocks = dict(cfg.get("subblocks", {}) or {})
    seeding = dict(cfg.get("seeding", {}) or {})
    for arg_name, key in (
        ("n_subblocks", "n_subblocks"),
        ("n_frames", "n_frames"),
        ("noise", "noise"),
        ("phi_ref", "phi_ref"),
        ("max_dense_dim", "max_dense_dim"),
        ("schur_curvature_method", "schur_curvature_method"),
        ("summary_information_scale", "summary_information_scale"),
        ("render_retention", "render_retention"),
        ("profile_runtime_detail", "profile_runtime_detail"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            subblocks[key] = value
    if bool(getattr(args, "profile_runtime", False)):
        subblocks["profile_runtime"] = True
    if bool(getattr(args, "memory_diagnostics", False)):
        subblocks["memory_diagnostics"] = True
    trace_source = dict(subblocks.get("trace_source", {}) or {})
    if getattr(args, "trace_source_mode", None) is not None:
        trace_source["mode"] = args.trace_source_mode
    if getattr(args, "trajectory_csv", None) is not None:
        source_cfg = dict(trace_source.get("source", {}) or {})
        source_cfg["kind"] = "airbus_csv"
        source_cfg["path"] = str(args.trajectory_csv)
        trace_source["source"] = source_cfg
    window_cfg = dict(trace_source.get("window", {}) or {})
    if getattr(args, "trajectory_start_s", None) is not None:
        window_cfg["start_s"] = float(args.trajectory_start_s)
    if getattr(args, "trajectory_duration_s", None) is not None:
        window_cfg["duration_s"] = float(args.trajectory_duration_s)
        window_cfg.pop("n_subblocks", None)
    if getattr(args, "trajectory_n_subblocks", None) is not None:
        window_cfg["n_subblocks"] = int(args.trajectory_n_subblocks)
    if window_cfg:
        trace_source["window"] = window_cfg
    sampling_cfg = dict(trace_source.get("sampling", {}) or {})
    if getattr(args, "trajectory_frame_dt_s", None) is not None:
        sampling_cfg["frame_dt_s"] = float(args.trajectory_frame_dt_s)
    if sampling_cfg:
        trace_source["sampling"] = sampling_cfg
    if getattr(args, "trajectory_output_keys", None) is not None:
        trace_source["output_keys"] = [
            part.strip()
            for part in str(args.trajectory_output_keys).split(",")
            if part.strip()
        ]
    if getattr(args, "trajectory_plan", None) is not None:
        trace_source["mode"] = "external_plan"
        trace_source["campaign_plan"] = str(args.trajectory_plan)
    if trace_source:
        subblocks["trace_source"] = trace_source
    cfg["subblocks"] = subblocks
    if getattr(args, "seed_policy", None) is not None:
        seeding["seed_policy"] = args.seed_policy
    if getattr(args, "base_seed", None) is not None:
        seeding["base_seed"] = int(args.base_seed)
    if seeding:
        cfg["seeding"] = seeding
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    return cfg


def _resolve_seeding_config(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    seeding_cfg = dict(experiment_cfg.get("seeding", {}) or {})
    policy = str(
        seeding_cfg.get(
            "seed_policy",
            experiment_cfg.get("subblocks", {}).get(
                "seed_policy", "different_jitter_different_noise"
            ),
        )
    )
    if policy not in SUPPORTED_SEED_POLICIES:
        raise ValueError(f"Unsupported seed_policy: {policy}")
    base_seed = int(
        seeding_cfg.get(
            "base_seed",
            experiment_cfg.get("subblocks", {}).get(
                "base_seed",
                experiment_cfg.get("seed", 42),
            ),
        )
    )
    return {"seed_policy": policy, "base_seed": base_seed}


def _derive_subblock_seeds(
    *,
    run_name: str,
    case_name: str,
    subblock_index: int,
    seed_policy: str,
    base_seed: int,
) -> dict[str, int]:
    derived = derive_campaign_subblock_seeds(
        base_seed=int(base_seed),
        seed_policy=str(seed_policy),
        campaign_token=str(run_name),
        case_token=str(case_name),
        subblock_index=int(subblock_index),
    )
    return {
        "seed_policy": derived.policy,
        "base_seed": int(derived.base_seed),
        "subblock_seed": int(derived.subblock_seed),
        "trace_seed": int(derived.trace_seed),
        "noise_seed": int(derived.noise_seed),
    }


def _resolve_iterative_config(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw_cfg = experiment_cfg.get("iterative", {}) or {}
    if not isinstance(raw_cfg, Mapping):
        raise ValueError("experiment.iterative must be a mapping when provided.")
    enabled = bool(raw_cfg.get("enabled", False))
    windows_per_draw = int(raw_cfg.get("windows_per_draw", 1))
    subblocks_per_window = int(
        raw_cfg.get(
            "subblocks_per_window",
            experiment_cfg.get("subblocks", {}).get("n_subblocks", 1),
        )
    )
    update_gain = float(raw_cfg.get("update_gain", 1.0))
    safety_cfg = dict(raw_cfg.get("update_safety", raw_cfg.get("safety", {})) or {})
    update_mode = str(raw_cfg.get("update_mode", "physical_full"))
    eigen_cfg = dict(raw_cfg.get("eigenbasis", {}) or {})
    min_sigma_by_label = safety_cfg.get("min_sigma_by_label")
    max_abs_update_by_label = safety_cfg.get("max_abs_update_by_label")
    posterior_sigma_policy = str(
        safety_cfg.get(
            "posterior_sigma_policy",
            "inflate_by_factor"
            if safety_cfg.get("posterior_sigma_inflation", safety_cfg.get("inflation_factor")) is not None
            else "reported_only",
        )
    )
    if posterior_sigma_policy not in {
        "reported_only",
        "inflate_by_factor",
        "floor_only",
        "process_noise_floor",
    }:
        raise ValueError(
            "experiment.iterative.update_safety.posterior_sigma_policy must be one of "
            "reported_only, inflate_by_factor, floor_only, process_noise_floor."
        )
    if posterior_sigma_policy in {"floor_only", "process_noise_floor"}:
        raise NotImplementedError(
            f"posterior_sigma_policy={posterior_sigma_policy!r} is documented as future/deferred."
        )
    posterior_sigma_inflation = (
        float(safety_cfg.get("posterior_sigma_inflation", safety_cfg.get("inflation_factor", 1.0)))
        if posterior_sigma_policy == "inflate_by_factor"
        else 1.0
    )
    if update_mode not in SUPPORTED_ITERATIVE_UPDATE_MODES:
        raise ValueError(
            "experiment.iterative.update_mode must be one of "
            + ", ".join(SUPPORTED_ITERATIVE_UPDATE_MODES)
            + "."
        )
    if windows_per_draw <= 0:
        raise ValueError("experiment.iterative.windows_per_draw must be positive.")
    if subblocks_per_window <= 0:
        raise ValueError("experiment.iterative.subblocks_per_window must be positive.")
    if not math.isfinite(update_gain):
        raise ValueError("experiment.iterative.update_gain must be finite.")
    policy = ObservationUpdatePolicy(
        update_mode=update_mode,
        update_gain=update_gain,
        basis_source=str(eigen_cfg.get("basis_source", "posterior_precision")),
        gate_source=eigen_cfg.get("gate_source", "accumulated_information"),
        whiten=bool(eigen_cfg.get("whiten", True)),
        eig_floor_abs=float(eigen_cfg.get("eig_floor_abs", 0.0)),
        eig_floor_rel=float(eigen_cfg.get("eig_floor_rel", 0.0)),
        damping_mode=str(eigen_cfg.get("damping_mode", "none")),
        damping_value=float(eigen_cfg.get("damping_value", 1.0)),
        damping_n_modes=eigen_cfg.get("damping_n_modes", 0),
        min_kept_modes=(
            None
            if eigen_cfg.get("min_kept_modes") is None
            else int(eigen_cfg["min_kept_modes"])
        ),
        max_kept_modes=(
            None
            if eigen_cfg.get("max_kept_modes") is None
            else int(eigen_cfg["max_kept_modes"])
        ),
        top_k_contributors=int(eigen_cfg.get("top_k_contributors", 8)),
    )
    return {
        "enabled": enabled,
        "windows_per_draw": windows_per_draw,
        "subblocks_per_window": subblocks_per_window,
        "update_gain": update_gain,
        "update_mode": update_mode,
        "eigenbasis": {
            "basis_source": policy.basis_source,
            "gate_source": policy.gate_source,
            "whiten": policy.whiten,
            "eig_floor_abs": policy.eig_floor_abs,
            "eig_floor_rel": policy.eig_floor_rel,
            "damping_mode": policy.damping_mode,
            "damping_value": policy.damping_value,
            "damping_n_modes": policy.damping_n_modes,
            "min_kept_modes": policy.min_kept_modes,
            "max_kept_modes": policy.max_kept_modes,
            "top_k_contributors": policy.top_k_contributors,
        },
        "update_policy": {
            "update_mode": policy.update_mode,
            "update_gain": policy.update_gain,
            "basis_source": policy.basis_source,
            "gate_source": policy.gate_source,
            "whiten": policy.whiten,
            "eig_floor_abs": policy.eig_floor_abs,
            "eig_floor_rel": policy.eig_floor_rel,
            "damping_mode": policy.damping_mode,
            "damping_value": policy.damping_value,
            "damping_n_modes": policy.damping_n_modes,
            "min_kept_modes": policy.min_kept_modes,
            "max_kept_modes": policy.max_kept_modes,
            "top_k_contributors": policy.top_k_contributors,
        },
        "carry_prior_mean_with_reference": bool(
            raw_cfg.get("carry_prior_mean_with_reference", True)
        ),
        "update_safety": {
            "enabled": bool(safety_cfg.get("enabled", False)),
            "posterior_sigma_policy": posterior_sigma_policy,
            "posterior_sigma_inflation": posterior_sigma_inflation,
            "min_sigma_by_label": dict(min_sigma_by_label or {}),
            "process_noise_by_label": dict(safety_cfg.get("process_noise_by_label") or {}),
            "max_abs_update_by_label": dict(max_abs_update_by_label or {}),
            "reject_on_bad_frame_quality": bool(safety_cfg.get("reject_on_bad_frame_quality", True)),
            "reject_on_nonfinite_posterior": bool(safety_cfg.get("reject_on_nonfinite_posterior", True)),
            "policy_note": "Physical-label safety limits are applied after the policy-controlled physical update.",
        },
        "future_update_modes": [],
    }


def _resolve_render_retention_policy(experiment_cfg: Mapping[str, Any]) -> str:
    """Return the explicit rendered-FITS retention policy for subblocks."""

    subblocks = experiment_cfg.get("subblocks", {}) or {}
    if not isinstance(subblocks, Mapping):
        raise ValueError("experiment.subblocks must be a mapping when provided.")
    policy = str(subblocks.get("render_retention", RENDER_RETENTION_DEFAULT))
    if policy not in SUPPORTED_RENDER_RETENTION_POLICIES:
        raise ValueError(
            "experiment.subblocks.render_retention must be one of "
            + ", ".join(SUPPORTED_RENDER_RETENTION_POLICIES)
            + "."
        )
    return policy


def _resolve_iterative_forecast_config(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw_cfg = experiment_cfg.get("iterative_forecast", {}) or {}
    if not isinstance(raw_cfg, Mapping):
        raise ValueError("experiment.iterative_forecast must be a mapping when provided.")
    enabled = bool(raw_cfg.get("enabled", False))
    iterative_cfg = _resolve_iterative_config(experiment_cfg)
    actual_windows = int(raw_cfg.get("actual_windows", iterative_cfg.get("windows_per_draw", 1)))
    projected_windows = int(raw_cfg.get("projected_windows", actual_windows))
    subblocks_per_window = int(
        raw_cfg.get(
            "subblocks_per_window",
            iterative_cfg.get("subblocks_per_window", 1),
        )
    )
    modes = tuple(str(mode) for mode in raw_cfg.get("modes", ["replicate_information"]))
    invalid = sorted(set(modes) - set(SUPPORTED_ITERATIVE_FORECAST_MODES))
    if invalid:
        raise ValueError(
            "Unsupported iterative_forecast mode(s): " + ", ".join(invalid)
        )
    stochastic_trials = int(raw_cfg.get("stochastic_trials", raw_cfg.get("n_trials", 128)))
    if actual_windows <= 0:
        raise ValueError("experiment.iterative_forecast.actual_windows must be positive.")
    if projected_windows < actual_windows:
        raise ValueError(
            "experiment.iterative_forecast.projected_windows must be >= actual_windows."
        )
    if subblocks_per_window <= 0:
        raise ValueError(
            "experiment.iterative_forecast.subblocks_per_window must be positive."
        )
    if "stochastic_score_noise" in modes and stochastic_trials <= 0:
        raise ValueError(
            "experiment.iterative_forecast.stochastic_trials must be positive "
            "when stochastic_score_noise is enabled."
        )
    prefixes = tuple(
        int(value)
        for value in raw_cfg.get("output_prefixes", [actual_windows, projected_windows])
    )
    if any(value <= 0 for value in prefixes):
        raise ValueError("experiment.iterative_forecast.output_prefixes must be positive.")
    observation_duration_s = float(
        raw_cfg.get(
            "observation_duration_s",
            float(projected_windows * subblocks_per_window)
            * float(raw_cfg.get("subblock_duration_s", 1.0)),
        )
    )
    seed = int(raw_cfg.get("seed", experiment_cfg.get("seed", 42)))
    return {
        "enabled": enabled,
        "actual_windows": actual_windows,
        "projected_windows": projected_windows,
        "subblocks_per_window": subblocks_per_window,
        "projected_subblocks_total": int(projected_windows * subblocks_per_window),
        "actual_subblocks_total_per_case": int(actual_windows * subblocks_per_window),
        "observation_duration_s": observation_duration_s,
        "modes": list(modes),
        "stochastic_trials": stochastic_trials,
        "seed": seed,
        "forecast_from": str(raw_cfg.get("forecast_from", "realized_windows")),
        "output_prefixes": sorted(set(prefixes)),
        "plots": bool(raw_cfg.get("plots", True)),
    }


def _case_draw_index(case: BiasCase) -> int | str:
    if case.prior_draw_metadata and case.prior_draw_metadata.get("draw_index") is not None:
        return int(case.prior_draw_metadata["draw_index"])
    return ""


def _iterative_subblock_name(
    *,
    case_name: str,
    window_index: int,
    subblock_index: int,
) -> str:
    return (
        f"{case_name}/window_{int(window_index):03d}/"
        f"subblock_{int(subblock_index):03d}"
    )


def _iterative_window_case_name(case_name: str, window_index: int) -> str:
    return f"{case_name}/windows/window_{int(window_index):03d}"


def _resolve_eigen_sources(eigen_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    if "sources" in eigen_cfg:
        raw_sources = eigen_cfg.get("sources")
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise ValueError("eigenbasis.sources must be a list of source names.")
        sources = tuple(str(item) for item in raw_sources)
    elif "source_matrix" in eigen_cfg:
        sources = (str(eigen_cfg.get("source_matrix")),)
    else:
        sources = ("accumulated_information",)
    if not sources:
        raise ValueError("At least one eigenbasis source is required.")
    invalid = [name for name in sources if name not in SUPPORTED_EIGEN_SOURCES]
    if invalid:
        raise ValueError(
            "Unsupported eigenbasis source(s): " + ", ".join(sorted(set(invalid)))
        )
    if len(set(sources)) != len(sources):
        raise ValueError("eigenbasis sources must not contain duplicates.")
    return sources


def _match_wildcard_pattern(pattern: str, label: str) -> bool:
    if "[*]" not in pattern:
        return pattern == label
    escaped = re.escape(pattern).replace(r"\[\*\]", r"\[\d+\]")
    return re.fullmatch(escaped, label) is not None


def _resolve_prior_draw_sigmas(
    *,
    labels: Sequence[str],
    truth_by_label: Mapping[str, float],
    sigmas_cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    if not isinstance(sigmas_cfg, Mapping):
        raise ValueError("prior_draws.sigmas must be a mapping.")
    resolved: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}
    for raw_rule_label, raw_rule in sigmas_cfg.items():
        rule_label = str(raw_rule_label).strip()
        if not rule_label:
            raise ValueError("prior_draw sigma rule labels must be non-empty.")
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"Sigma rule {rule_label!r} must be a mapping.")
        kind = str(raw_rule.get("kind", "absolute"))
        if kind not in {"absolute", "fractional"}:
            raise ValueError(f"Unsupported sigma kind {kind!r} for rule {rule_label!r}.")
        sigma_value = float(raw_rule.get("sigma", 0.0))
        if sigma_value <= 0.0 or not math.isfinite(sigma_value):
            raise ValueError(f"Sigma for rule {rule_label!r} must be positive and finite.")
        unit = None if raw_rule.get("unit") is None else str(raw_rule.get("unit"))
        matching_labels = [label for label in labels if _match_wildcard_pattern(rule_label, label)]
        if not matching_labels:
            raise ValueError(f"Sigma rule {rule_label!r} matched no theta labels.")
        for label in matching_labels:
            if kind == "absolute":
                sigma = sigma_value
            else:
                truth = float(truth_by_label[label])
                sigma = abs(truth) * sigma_value
            if sigma <= 0.0 or not math.isfinite(sigma):
                raise ValueError(
                    f"Resolved sigma for label {label!r} from rule {rule_label!r} "
                    "must be positive and finite."
                )
            resolved[label] = float(sigma)
            meta[label] = {
                "sigma_kind": kind,
                "sigma_source_rule": rule_label,
                "unit": _parameter_unit(label) if unit is None else unit,
                "sigma_config_value": sigma_value,
            }
    missing = [label for label in labels if label not in resolved]
    if missing:
        raise ValueError(
            "prior_draws.sigmas did not resolve all observation labels; missing: "
            + ", ".join(missing)
        )
    return resolved, meta


def _generate_prior_draw_cases(
    *,
    experiment_cfg: Mapping[str, Any],
    labels: Sequence[str],
    truth_by_label: Mapping[str, float],
) -> tuple[list[BiasCase], dict[str, list[dict[str, Any]]]]:
    raw_cfg = experiment_cfg.get("prior_draws", {}) or {}
    if not isinstance(raw_cfg, Mapping) or not bool(raw_cfg.get("enabled", False)):
        return [], {}
    center = str(raw_cfg.get("center", "truth"))
    if center != "truth":
        raise ValueError("prior_draws.center currently supports only 'truth'.")
    distribution = str(raw_cfg.get("distribution", "normal"))
    if distribution != "normal":
        raise ValueError("prior_draws.distribution currently supports only 'normal'.")
    conditions_raw = raw_cfg.get("conditions")
    has_conditions = conditions_raw is not None
    n_cases = int(
        raw_cfg.get(
            "n_cases",
            raw_cfg.get("draws_per_condition", raw_cfg.get("prior_draws_per_condition", 0)),
        )
    )
    if n_cases <= 0:
        raise ValueError(
            "prior_draws.n_cases must be positive when enabled. For conditional draws, "
            "n_cases is interpreted as draws per condition."
        )
    draw_seed = int(raw_cfg.get("draw_seed", 12345))
    draw_index_start = int(raw_cfg.get("draw_index_start", 0))
    condition_index_start = int(raw_cfg.get("condition_index_start", 0))
    global_draw_index_start_raw = raw_cfg.get("global_draw_index_start")
    global_draw_index_start = (
        int(global_draw_index_start_raw)
        if global_draw_index_start_raw is not None
        else None
    )
    rng_skip_draws = int(raw_cfg.get("rng_skip_draws", 0))
    if draw_index_start < 0:
        raise ValueError("prior_draws.draw_index_start must be non-negative.")
    if condition_index_start < 0:
        raise ValueError("prior_draws.condition_index_start must be non-negative.")
    if global_draw_index_start is not None and global_draw_index_start < 0:
        raise ValueError("prior_draws.global_draw_index_start must be non-negative.")
    if rng_skip_draws < 0:
        raise ValueError("prior_draws.rng_skip_draws must be non-negative.")
    case_name_template = str(raw_cfg.get("case_name_template", "prior_draw_{draw_index:03d}"))
    label_order = tuple(labels)
    rng = np.random.default_rng(draw_seed)
    if rng_skip_draws:
        # Sharded campaigns use this opt-in offset to reproduce the exact RNG
        # stream position of draws selected from a larger parent campaign.
        rng.normal(loc=0.0, scale=1.0, size=rng_skip_draws * len(label_order))
    cases: list[BiasCase] = []
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    condition_entries: list[dict[str, Any]]
    if has_conditions:
        if not isinstance(conditions_raw, Sequence) or isinstance(conditions_raw, (str, bytes)):
            raise ValueError("prior_draws.conditions must be a list when provided.")
        condition_entries = []
        seen_conditions: set[str] = set()
        for local_condition_index, raw_condition in enumerate(conditions_raw):
            if not isinstance(raw_condition, Mapping):
                raise ValueError("Each prior_draws.conditions entry must be a mapping.")
            condition_name = str(raw_condition.get("condition_name", "")).strip()
            if not condition_name:
                raise ValueError("Each prior_draws.conditions entry requires condition_name.")
            if condition_name in seen_conditions:
                raise ValueError(f"Duplicate prior draw condition_name: {condition_name}.")
            seen_conditions.add(condition_name)
            condition_sigmas = dict(raw_cfg.get("sigmas", {}) or {})
            condition_sigmas.update(dict(raw_condition.get("sigmas", {}) or {}))
            condition_entries.append(
                {
                    "condition_index": int(condition_index_start + local_condition_index),
                    "condition_name": condition_name,
                    "sigmas": condition_sigmas,
                }
            )
    else:
        condition_entries = [
            {
                "condition_index": int(condition_index_start),
                "condition_name": "",
                "sigmas": raw_cfg.get("sigmas", {}),
            }
        ]
    generated_draw_ordinal = 0
    for condition in condition_entries:
        sigma_by_label, sigma_meta = _resolve_prior_draw_sigmas(
            labels=labels,
            truth_by_label=truth_by_label,
            sigmas_cfg=condition["sigmas"],
        )
        sigma_vector = np.asarray([sigma_by_label[label] for label in label_order], dtype=float)
        condition_name = str(condition["condition_name"])
        condition_index = int(condition["condition_index"])
        for local_draw_index in range(n_cases):
            draw_index = draw_index_start + local_draw_index
            if global_draw_index_start is not None:
                global_draw_index = global_draw_index_start + generated_draw_ordinal
            else:
                global_draw_index = (
                    condition_index * n_cases + draw_index
                    if has_conditions
                    else draw_index
                )
            format_kwargs = {
                "draw_index": int(draw_index),
                "global_draw_index": int(global_draw_index),
                "condition_index": int(condition_index),
                "condition_name": condition_name,
            }
            try:
                case_name = case_name_template.format(**format_kwargs)
            except KeyError as exc:
                raise ValueError(
                    "prior_draws.case_name_template may use draw_index, global_draw_index, "
                    "condition_index, and condition_name."
                ) from exc
            if has_conditions and "{condition_name" not in case_name_template:
                case_name = f"{condition_name}_{case_name}"
            z_vector = rng.normal(loc=0.0, scale=1.0, size=len(label_order))
            delta = z_vector * sigma_vector
            offsets = {label: float(delta[i]) for i, label in enumerate(label_order)}
            case = BiasCase(
                case_name=case_name,
                theta_reference_offsets=offsets,
                case_origin="prior_draw",
                prior_sigma_by_label={label: float(sigma_by_label[label]) for label in label_order},
                prior_draw_metadata={
                    "draw_seed": int(draw_seed),
                    "draw_index": int(draw_index),
                    "global_draw_index": int(global_draw_index),
                    "condition_index": int(condition_index),
                    "condition_name": condition_name,
                    "distribution": distribution,
                    "center": center,
                },
            )
            cases.append(case)
            rows: list[dict[str, Any]] = []
            for i, label in enumerate(label_order):
                truth_value = float(truth_by_label[label])
                offset = float(delta[i])
                sigma = float(sigma_vector[i])
                z_value = float(z_vector[i])
                rows.append(
                    {
                        "case_name": case_name,
                        "theta_label": label,
                        "truth_value": truth_value,
                        "prior_mean": truth_value + offset,
                        "reference_value": truth_value + offset,
                        "prior_sigma": sigma,
                        "draw_z": z_value,
                        "theta_reference_offset": offset,
                        "sigma_kind": sigma_meta[label]["sigma_kind"],
                        "sigma_source_rule": sigma_meta[label]["sigma_source_rule"],
                        "unit": sigma_meta[label]["unit"],
                        "sigma_config_value": sigma_meta[label]["sigma_config_value"],
                        "draw_seed": int(draw_seed),
                        "draw_index": int(draw_index),
                        "global_draw_index": int(global_draw_index),
                        "condition_index": int(condition_index),
                        "condition_name": condition_name,
                    }
                )
            rows_by_case[case_name] = rows
            generated_draw_ordinal += 1
    return cases, rows_by_case


def _validate_choice(value: Any, *, key: str, choices: Sequence[str]) -> str:
    text = str(value)
    if text not in choices:
        raise ValueError(
            f"experiment.subblocks.{key} must be one of "
            + ", ".join(choices)
            + "."
        )
    return text


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def resolve_subblock_command_options(
    subblock_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize optional flags forwarded to the subblock runner."""

    subblock_cfg = dict(subblock_cfg)
    if subblock_cfg.get("variance_floor") is None:
        noise_cfg = subblock_cfg.get("noise")
        if isinstance(noise_cfg, Mapping) and noise_cfg.get("variance_floor") is not None:
            subblock_cfg["variance_floor"] = noise_cfg["variance_floor"]
        else:
            noise_model = subblock_cfg.get("noise_model")
            if isinstance(noise_model, Mapping):
                terms = noise_model.get("render_template_terms")
                if isinstance(terms, Mapping) and terms.get("variance_floor") is not None:
                    subblock_cfg["variance_floor"] = terms["variance_floor"]
    options: dict[str, Any] = {}
    if subblock_cfg.get("exposure_time_s") is not None:
        exposure_time_s = float(subblock_cfg["exposure_time_s"])
        if exposure_time_s <= 0.0 or not math.isfinite(exposure_time_s):
            raise ValueError("experiment.subblocks.exposure_time_s must be positive.")
        options["exposure_time_s"] = exposure_time_s
    if subblock_cfg.get("reference_diagnostics_profile") is not None:
        options["reference_diagnostics_profile"] = _validate_choice(
            subblock_cfg["reference_diagnostics_profile"],
            key="reference_diagnostics_profile",
            choices=SUPPORTED_REFERENCE_DIAGNOSTICS_PROFILES,
        )
    if subblock_cfg.get("summary_information_scale") is not None:
        options["summary_information_scale"] = _validate_choice(
            subblock_cfg["summary_information_scale"],
            key="summary_information_scale",
            choices=("summed_likelihood", "optimizer"),
        )
    if subblock_cfg.get("schur_frame_quality_policy") is not None:
        options["schur_frame_quality_policy"] = _validate_choice(
            subblock_cfg["schur_frame_quality_policy"],
            key="schur_frame_quality_policy",
            choices=SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES,
        )
    if subblock_cfg.get("schur_frame_chi2_threshold") is not None:
        threshold = float(subblock_cfg["schur_frame_chi2_threshold"])
        if threshold <= 0.0 or not math.isfinite(threshold):
            raise ValueError("experiment.subblocks.schur_frame_chi2_threshold must be positive.")
        options["schur_frame_chi2_threshold"] = threshold
    if subblock_cfg.get("schur_frame_quality_missing") is not None:
        options["schur_frame_quality_missing"] = _validate_choice(
            subblock_cfg["schur_frame_quality_missing"],
            key="schur_frame_quality_missing",
            choices=SUPPORTED_SCHUR_FRAME_QUALITY_MISSING,
        )
    if subblock_cfg.get("schur_frame_mask_denominator") is not None:
        options["schur_frame_mask_denominator"] = _validate_choice(
            subblock_cfg["schur_frame_mask_denominator"],
            key="schur_frame_mask_denominator",
            choices=SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS,
        )
    if subblock_cfg.get("schur_frame_mask_min_good_frames") is not None:
        min_good = int(subblock_cfg["schur_frame_mask_min_good_frames"])
        if min_good < 1:
            raise ValueError(
                "experiment.subblocks.schur_frame_mask_min_good_frames must be >= 1."
            )
        options["schur_frame_mask_min_good_frames"] = min_good

    optional_keys = (
        "reference_optimizer_kind",
        "reference_base_lr",
        "reference_n_iter",
        "reference_optimizer_kwargs",
        "reference_schedule_kind",
        "reference_schedule_warmup_steps",
        "reference_schedule_start_factor",
        "reference_schedule_min_factor",
        "reference_schedule_boundaries",
        "reference_schedule_factors",
        "reference_schedule_decay_rate",
        "reference_schedule_transition_steps",
        "reference_schedule_staircase",
        "reference_preconditioning_method",
        "reference_preconditioning_reference",
        "reference_preconditioning_damping",
        "reference_preconditioning_eig_floor_rel",
        "reference_preconditioning_eig_floor_abs",
        "reference_preconditioning_lr_clip",
        "reference_early_stopping_min_iter",
        "reference_early_stopping_patience",
        "reference_early_stopping_loss_rtol",
        "reference_early_stopping_loss_atol",
        "reference_early_stopping_step_atol",
        "reference_early_stopping_grad_norm_atol",
        "variance_floor",
    )
    for key in optional_keys:
        if subblock_cfg.get(key) is not None:
            value = subblock_cfg[key]
            if key in {
                "reference_n_iter",
                "reference_schedule_warmup_steps",
                "reference_schedule_transition_steps",
                "reference_early_stopping_min_iter",
                "reference_early_stopping_patience",
            }:
                value = int(value)
            elif key in {
                "reference_base_lr",
                "reference_schedule_start_factor",
                "reference_schedule_min_factor",
                "reference_schedule_decay_rate",
                "reference_preconditioning_damping",
                "reference_preconditioning_eig_floor_rel",
                "reference_preconditioning_eig_floor_abs",
                "reference_early_stopping_loss_rtol",
                "reference_early_stopping_loss_atol",
                "reference_early_stopping_step_atol",
                "reference_early_stopping_grad_norm_atol",
                "variance_floor",
            }:
                value = float(value)
            options[key] = value
    if subblock_cfg.get("reference_preconditioning_enabled") is not None:
        options["reference_preconditioning_enabled"] = bool(
            subblock_cfg["reference_preconditioning_enabled"]
        )
    if subblock_cfg.get("reference_early_stopping_enabled") is not None:
        options["reference_early_stopping_enabled"] = bool(
            subblock_cfg["reference_early_stopping_enabled"]
        )
    if subblock_cfg.get("use_render_variance") is not None:
        options["use_render_variance"] = bool(subblock_cfg["use_render_variance"])
    bool_switch_keys = {
        "reference_preconditioning_enabled",
        "reference_early_stopping_enabled",
        "use_render_variance",
    }
    forwarded_flags = [
        SUBBLOCK_OPTION_FLAG_MAP[key]
        for key in SUBBLOCK_OPTION_FLAG_MAP
        if key in options and key not in bool_switch_keys
    ]
    if options.get("use_render_variance") is True:
        forwarded_flags.append("--use-render-variance")
    if "reference_preconditioning_enabled" in options:
        forwarded_flags.append(
            "--reference-preconditioning-enabled"
            if options["reference_preconditioning_enabled"]
            else "--reference-preconditioning-disabled"
        )
    if options.get("reference_early_stopping_enabled") is True:
        forwarded_flags.append("--reference-early-stopping")
    if bool(subblock_cfg.get("profile_runtime", False)):
        options["profile_runtime"] = True
        forwarded_flags.append("--profile-runtime")
        detail = str(subblock_cfg.get("profile_runtime_detail", "basic"))
        if detail not in {"basic", "full"}:
            raise ValueError("experiment.subblocks.profile_runtime_detail must be 'basic' or 'full'.")
        options["profile_runtime_detail"] = detail
        forwarded_flags.extend(["--profile-runtime-detail", detail])
    if bool(subblock_cfg.get("memory_diagnostics", False)):
        options["memory_diagnostics"] = True
        forwarded_flags.append("--memory-diagnostics")
    options["forwarded_flags"] = forwarded_flags
    return options


def build_subblock_command(
    *,
    case_root_parent: Path,
    case_subblock_name: str,
    theta_labels: Sequence[str],
    offsets: Mapping[str, float],
    subblock_cfg: Mapping[str, Any],
    trace_seed: int | None = None,
    noise_seed: int | None = None,
    trace_subblock: PreparedTraceSubblock | None = None,
    template_paths: Mapping[str, Path] | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SUBBLOCK_SCRIPT),
        "--results-root",
        str(case_root_parent),
        "--case-name",
        case_subblock_name,
        "--mode",
        "schur_summary",
        "--trace-template",
        str((template_paths or {}).get("trace", DEFAULT_SCHUR_TRACE_TEMPLATE)),
        "--render-template",
        str((template_paths or {}).get("render", DEFAULT_RENDER_TEMPLATE)),
        "--inference-template",
        str((template_paths or {}).get("inference", DEFAULT_INFERENCE_TEMPLATE)),
        "--n-frames",
        str(int(subblock_cfg.get("n_frames", 3))),
        "--noise",
        "inherit" if isinstance(subblock_cfg.get("noise"), Mapping) else str(subblock_cfg.get("noise", "disabled")),
        "--theta-keys",
        ",".join(str(label) for label in theta_labels),
        "--phi-ref",
        str(subblock_cfg.get("phi_ref", "truth_when_available")),
        "--schur-curvature-method",
        str(subblock_cfg.get("schur_curvature_method", "auto")),
        "--max-dense-dim",
        str(int(subblock_cfg.get("max_dense_dim", 40))),
        "--schur-damping",
        str(float(subblock_cfg.get("schur_damping", 1.0e-8))),
    ]
    for key in sorted(offsets):
        command.extend(["--theta-reference-offset", f"{key}={float(offsets[key])}"])
    if trace_seed is not None:
        command.extend(["--trace-seed", str(int(trace_seed))])
    if noise_seed is not None:
        command.extend(["--render-seed", str(int(noise_seed))])
    trace_jitter_cfg = dict(subblock_cfg.get("trace_jitter", {}) or {})
    if trace_jitter_cfg.get("x_sigma_as") is not None:
        command.extend(
            ["--trace-jitter-x-sigma-as", str(float(trace_jitter_cfg["x_sigma_as"]))]
        )
    if trace_jitter_cfg.get("y_sigma_as") is not None:
        command.extend(
            ["--trace-jitter-y-sigma-as", str(float(trace_jitter_cfg["y_sigma_as"]))]
        )
    if trace_jitter_cfg.get("pa_sigma_deg") is not None:
        command.extend(
            [
                "--trace-jitter-pa-sigma-deg",
                str(float(trace_jitter_cfg["pa_sigma_deg"])),
            ]
        )
    command_options = resolve_subblock_command_options(subblock_cfg)
    for key, flag in SUBBLOCK_OPTION_FLAG_MAP.items():
        if key in {
            *REFERENCE_OPTIMIZER_FLAG_MAP,
            *REFERENCE_SCHEDULE_FLAG_MAP,
            *REFERENCE_PRECONDITIONING_FLAG_MAP,
            *REFERENCE_EARLY_STOPPING_FLAG_MAP,
            *SCHUR_FRAME_QUALITY_FLAG_MAP,
        }:
            continue
        if key in command_options:
            command.extend([flag, _format_cli_value(command_options[key])])
    if command_options.get("use_render_variance") is True:
        command.append("--use-render-variance")
    append_reference_optimizer_flags(command, command_options)
    append_schur_frame_quality_flags(command, command_options)
    if command_options.get("profile_runtime") is True:
        command.append("--profile-runtime")
        command.extend(
            [
                "--profile-runtime-detail",
                str(command_options.get("profile_runtime_detail", "basic")),
            ]
        )
    if command_options.get("memory_diagnostics") is True:
        command.append("--memory-diagnostics")
    if trace_subblock is not None:
        command.extend(trace_subblock_command_flags(trace_subblock))
    return command


def build_campaign_plan(
    *,
    config_path: Path | None,
    results_root: Path,
    run_name: str | None,
    system_preset: str | None,
    args: argparse.Namespace | None = None,
) -> CampaignPlan:
    config = _load_campaign_config(config_path)
    experiment_cfg = _experiment_config(config)
    if args is not None:
        experiment_cfg = _apply_cli_overrides(experiment_cfg, args)
    if run_name is not None:
        experiment_cfg["run_name"] = run_name
    resolved_run_name = str(experiment_cfg.get("run_name") or f"observation_bias_campaign_{timestamp_tag()}")
    run_root = Path(results_root).resolve() / resolved_run_name
    preset_resolution = resolve_campaign_system_preset(
        config=config,
        experiment_cfg=experiment_cfg,
        cli_system_preset=system_preset,
        strict=True,
    )
    subblock_exposure_time_s = _subblock_exposure_time_s(experiment_cfg)
    system_store, system_cfg, system_provenance = _resolve_system_store(
        config=config,
        system_preset=str(preset_resolution["system_preset"]),
        preset_resolution=preset_resolution,
        exposure_time_s=subblock_exposure_time_s,
    )
    observation_theta_cfg = experiment_cfg.get("observation_theta", {}) or {}
    layout, layout_metadata = build_system_observation_theta_layout(
        system_store,
        config=observation_theta_cfg,
    )
    layout_metadata["system"] = system_provenance
    layout_metadata["system_preset_resolution"] = preset_resolution
    layout_metadata["resolved_system"] = system_cfg
    partition = _validate_partition_config(experiment_cfg.get("state_partition"), layout)
    prior_truth = build_prior_mean_from_store(layout.labels, store=system_store)
    base_truth_by_label = {
        label: float(prior_truth[index]) for index, label in enumerate(layout.labels)
    }
    truth_realization = _realize_campaign_truth(
        experiment_cfg=experiment_cfg,
        labels=layout.labels,
        base_truth_by_label=base_truth_by_label,
    )
    truth_by_label = dict(base_truth_by_label)
    truth_by_label.update(truth_realization.truth_overrides_by_label)
    prior_truth = np.asarray([truth_by_label[label] for label in layout.labels], dtype=float)
    subblock_cfg = dict(experiment_cfg.get("subblocks", {}) or {})
    reuse_existing_artifacts = bool(
        args is not None
        and (
            getattr(args, "aggregate_only", False)
            or (
                getattr(args, "resume", False)
                and (run_root / "campaign_plan.json").exists()
            )
        )
    )
    source_cfg = system_cfg.get("source", {}) if isinstance(system_cfg.get("source"), Mapping) else {}
    smear_cfg = (
        subblock_cfg.get("trajectory_processing", {}).get("smear", {})
        if isinstance(subblock_cfg.get("trajectory_processing"), Mapping)
        else {}
    )
    normalized_noise = normalize_subblock_noise_config(
        subblock_cfg,
        detector_cfg=system_cfg,
        exposure_time_s=subblock_exposure_time_s,
        strict=False,
    )
    model_split = build_campaign_model_split(
        base_system_cfg=system_cfg,
        spectral_model_cfg=experiment_cfg.get("spectral_model"),
        high_order_wfe_cfg=experiment_cfg.get("high_order_wfe"),
        detector_calibration_knowledge_error_cfg=experiment_cfg.get(
            "detector_calibration_knowledge_error"
        ),
        scalar_reference_offsets=truth_realization.truth_overrides_by_label,
        detector_noise_metadata={
            **normalized_noise.to_dict(),
            "enabled": bool(normalized_noise.enabled),
            "noise_mode": normalized_noise.legacy_noise_mode,
        },
        run_root=run_root,
        artifact_root=run_root / "model_split",
        seed_context={
            "wrapper": "observation_bias_campaign",
            "run_name": resolved_run_name,
            "base_seed": int(experiment_cfg.get("seed", 42)),
        },
        source_kind=str(source_cfg.get("kind", "binary")),
        target=source_cfg.get("target"),
        write_artifacts=not reuse_existing_artifacts,
        trajectory_smear_metadata=smear_cfg if isinstance(smear_cfg, Mapping) else None,
    )
    system_cfg = model_split.truth_system_cfg
    reference_system_cfg = model_split.inference_system_cfg
    layout_metadata["resolved_system"] = system_cfg
    layout_metadata["reference_system"] = reference_system_cfg
    layout_metadata["model_split"] = model_split.to_dict()
    layout_metadata["high_order_wfe"] = model_split.provenance.get("high_order_wfe", {})
    if reuse_existing_artifacts:
        template_paths = {
            "trace": run_root / "templates" / "trace_template.json",
            "render": run_root / "templates" / "render_template.json",
            "inference": run_root / "templates" / "inference_template.json",
        }
        missing_templates = [
            str(path) for path in template_paths.values() if not Path(path).exists()
        ]
        if missing_templates:
            if bool(getattr(args, "aggregate_only", False)):
                template_paths = {
                    "trace": DEFAULT_SCHUR_TRACE_TEMPLATE,
                    "render": DEFAULT_RENDER_TEMPLATE,
                    "inference": DEFAULT_INFERENCE_TEMPLATE,
                }
            else:
                raise FileNotFoundError(
                    "Stored campaign template artifacts are required for resume; missing: "
                    + ", ".join(missing_templates)
                )
    else:
        template_paths = _write_observation_bias_templates(
            run_root,
            model_split=model_split,
            normalized_noise=normalized_noise,
        )
    template_hashes = template_hash_row(template_paths, model_split)
    noise_provenance_paths = _write_noise_provenance(
        run_root=run_root,
        normalized_noise=normalized_noise,
        render_template_noise_block=normalized_noise.render_template_noise_block(),
    )
    layout_metadata["noise"] = normalized_noise.to_dict()
    layout_metadata["noise_provenance_paths"] = {key: str(path) for key, path in noise_provenance_paths.items()}
    configured_cases = _parse_bias_cases(
        experiment_cfg,
        layout=layout,
        layout_metadata=layout_metadata,
    )
    prior_draw_cases, prior_draw_rows_by_case = _generate_prior_draw_cases(
        experiment_cfg=experiment_cfg,
        labels=layout.labels,
        truth_by_label=truth_by_label,
    )
    cases, case_generation = _resolve_case_generation(
        experiment_cfg,
        parsed_cases=configured_cases,
        prior_draw_cases=prior_draw_cases,
    )
    effective_subblock_cfg = dict(subblock_cfg)
    effective_subblock_cfg["noise"] = normalized_noise.legacy_noise_mode
    if normalized_noise.variance_floor is not None and normalized_noise.variance_floor != "auto":
        effective_subblock_cfg["variance_floor"] = normalized_noise.variance_floor
    effective_subblock_cfg["use_render_variance"] = normalized_noise.use_render_variance_resolved
    effective_subblock_cfg["render_retention"] = _resolve_render_retention_policy(
        experiment_cfg
    )
    effective_subblock_cfg["noise_model"] = {
        **dict(effective_subblock_cfg.get("noise_model", {}) or {}),
        "normalized_subblock_noise": normalized_noise.to_dict(),
        "render_template_terms": normalized_noise.render_template_noise_block() or {},
    }
    subblock_command_options = resolve_subblock_command_options(effective_subblock_cfg)
    iterative_cfg = _resolve_iterative_config(experiment_cfg)
    iterative_cfg["render_retention"] = str(effective_subblock_cfg["render_retention"])
    iterative_forecast_cfg = _resolve_iterative_forecast_config(experiment_cfg)
    if bool(iterative_forecast_cfg.get("enabled", False)):
        if not bool(iterative_cfg.get("enabled", False)):
            raise ValueError("experiment.iterative_forecast.enabled=true requires iterative.enabled=true.")
        if int(iterative_forecast_cfg["subblocks_per_window"]) != int(iterative_cfg["subblocks_per_window"]):
            raise ValueError(
                "experiment.iterative_forecast.subblocks_per_window must match "
                "experiment.iterative.subblocks_per_window."
            )
        iterative_cfg = {
            **iterative_cfg,
            "windows_per_draw": int(iterative_forecast_cfg["actual_windows"]),
        }
        experiment_cfg["iterative"] = iterative_cfg
        experiment_cfg["iterative_forecast"] = iterative_forecast_cfg
    seeding_cfg = _resolve_seeding_config(experiment_cfg)
    subblock_resolution = _resolve_total_subblocks_for_campaign(
        experiment_cfg,
        effective_subblock_cfg,
        iterative_cfg,
    )
    n_subblocks = int(subblock_resolution["resolved_total_subblocks"])
    subblock_root = run_root / "subblock_runs"
    trace_source_plan = prepare_campaign_trace_source(
        trace_source_cfg=subblock_cfg.get("trace_source"),
        run_root=run_root,
        artifact_root=run_root / "trajectory",
        source_kind="binary",
        active_frame_keys=DEFAULT_LOCAL_ELIMINATED_KEYS,
        n_subblocks=n_subblocks,
        n_frames_per_subblock=int(effective_subblock_cfg.get("n_frames", 3)),
        frame_dt_s=float(effective_subblock_cfg.get("exposure_time_s", 0.05)),
        subblock_duration_s=float(
            experiment_cfg.get("forecast", {}).get("subblock_duration_s", 1.0)
        ),
        default_output_keys=DEFAULT_LOCAL_ELIMINATED_KEYS,
        reuse_existing=bool(
            args is not None
            and (
                getattr(args, "aggregate_only", False)
                or (
                    getattr(args, "resume", False)
                    and (run_root / "campaign_plan.json").exists()
                )
            )
        ),
        trajectory_processing_cfg=effective_subblock_cfg.get("trajectory_processing"),
        plate_scale_as_per_pix=_store_scalar_value(
            system_store,
            "optics.plate_scale_as_per_pix",
        ),
    )
    commands: dict[str, list[list[str]]] = {}
    summary_paths: dict[str, list[Path]] = {}
    subblock_plans: dict[str, list[dict[str, Any]]] = {}
    iterative_plan_rows: list[dict[str, Any]] = []
    expected_output_rows: list[dict[str, Any]] = []
    template_hash_rows: list[dict[str, Any]] = []
    seen_template_hash_keys: set[tuple[str, str, str]] = set()
    for case in cases:
        commands[case.case_name] = []
        summary_paths[case.case_name] = []
        subblock_plans[case.case_name] = []
        for subblock_index in range(n_subblocks):
            window_index = (
                int(subblock_index) // int(iterative_cfg["subblocks_per_window"])
                if bool(iterative_cfg["enabled"])
                else 0
            )
            window_subblock_index = (
                int(subblock_index) % int(iterative_cfg["subblocks_per_window"])
                if bool(iterative_cfg["enabled"])
                else int(subblock_index)
            )
            subblock_name = (
                _iterative_subblock_name(
                    case_name=case.case_name,
                    window_index=window_index,
                    subblock_index=window_subblock_index,
                )
                if bool(iterative_cfg["enabled"])
                else f"{case.case_name}/subblock_{subblock_index:03d}"
            )
            seeds = _derive_subblock_seeds(
                run_name=resolved_run_name,
                case_name=case.case_name,
                subblock_index=subblock_index,
                seed_policy=str(seeding_cfg["seed_policy"]),
                base_seed=int(seeding_cfg["base_seed"]),
            )
            summary_path = (
                subblock_root
                / subblock_name
                / "study"
                / "schur_summary"
                / "subblock_summary.json"
            )
            trace_row = dict(trace_source_plan.rows[subblock_index])
            if _smear_render_mode(effective_subblock_cfg) == "subblock_constant_layer":
                active_template_paths, active_template_hashes = _write_subblock_smear_templates(
                    run_root=run_root,
                    subblock_index=subblock_index,
                    window_index=int(window_index) if bool(iterative_cfg["enabled"]) else "",
                    model_split=model_split,
                    subblock_cfg=effective_subblock_cfg,
                    trace_row=trace_row,
                    normalized_noise=normalized_noise,
                )
            else:
                active_template_paths = template_paths
                active_template_hashes = template_hashes
            template_key = (
                str(active_template_hashes.get("trace_template_hash", "")),
                str(active_template_hashes.get("render_template_hash", "")),
                str(active_template_hashes.get("inference_template_hash", "")),
            )
            if template_key not in seen_template_hash_keys:
                seen_template_hash_keys.add(template_key)
                template_hash_rows.append(active_template_hashes)
            commands[case.case_name].append(
                build_subblock_command(
                    case_root_parent=subblock_root,
                    case_subblock_name=subblock_name,
                    theta_labels=layout.labels,
                    offsets=case.theta_reference_offsets,
                    subblock_cfg=effective_subblock_cfg,
                    trace_seed=int(seeds["trace_seed"]),
                    noise_seed=int(seeds["noise_seed"]),
                    trace_subblock=trace_source_plan.subblocks[subblock_index],
                    template_paths=active_template_paths,
                )
            )
            summary_paths[case.case_name].append(summary_path)
            plan_options = {
                key: value
                for key, value in subblock_command_options.items()
                if key != "forwarded_flags"
            }
            subblock_plan_row = {
                    "case_name": case.case_name,
                    "subblock_index": int(subblock_index),
                    "window_index": int(window_index) if bool(iterative_cfg["enabled"]) else "",
                    "window_subblock_index": int(window_subblock_index)
                    if bool(iterative_cfg["enabled"])
                    else "",
                    "subblock_name": subblock_name,
                    "summary_path": str(summary_path),
                    "case_origin": case.case_origin,
                    "n_frames": int(effective_subblock_cfg.get("n_frames", 3)),
                    "noise": str(effective_subblock_cfg.get("noise", "disabled")),
                    "phi_ref": str(effective_subblock_cfg.get("phi_ref", "truth_when_available")),
                    "schur_curvature_method": str(
                        effective_subblock_cfg.get("schur_curvature_method", "auto")
                    ),
                    "max_dense_dim": int(effective_subblock_cfg.get("max_dense_dim", 40)),
                    "schur_damping": float(effective_subblock_cfg.get("schur_damping", 1.0e-8)),
                    "noise_enabled": bool(normalized_noise.enabled),
                    "shot_noise": bool(normalized_noise.shot_noise),
                    "read_noise": bool(normalized_noise.read_noise),
                    "dark_current": bool(normalized_noise.dark_current),
                    "read_noise_electrons": normalized_noise.read_noise_electrons,
                    "dark_current_e_per_s": normalized_noise.dark_current_e_per_s,
                    "variance_floor": normalized_noise.variance_floor,
                    "variance_floor_source": normalized_noise.variance_floor_source,
                    "use_render_variance": normalized_noise.use_render_variance_resolved,
                    "noise_request_normalized_json": str(noise_provenance_paths["noise_request_normalized_json"]),
                    "forwarded_flags": ",".join(
                        str(flag) for flag in subblock_command_options["forwarded_flags"]
                    ),
                    "high_order_wfe_enabled": bool(model_split.enabled_components.get("high_order_wfe", {}).get("enabled", False)),
                    "high_order_wfe_summary_json": model_split.artifact_paths.get("high_order_wfe_high_order_wfe_summary_json", ""),
                    **active_template_hashes,
                    **plan_options,
                    **seeds,
                    **trace_row,
                }
            subblock_plans[case.case_name].append(subblock_plan_row)
            if bool(iterative_cfg["enabled"]):
                window_case_name = _iterative_window_case_name(
                    case.case_name,
                    int(window_index),
                )
                window_case_root = run_root / "cases" / window_case_name
                posterior_path = window_case_root / "posterior_by_label.csv"
                window_summary_path = window_case_root / "science_summary.csv"
                reference_update_path = window_case_root / "iterative_reference_update.json"
                window_diagnostic_path = window_case_root / "iterative_window_diagnostics.csv"
                eigen_update_diagnostics_path = (
                    window_case_root / "eigen_update_diagnostics.json"
                )
                eigen_update_modes_path = window_case_root / "eigen_update_modes.csv"
                realized_command_path = (
                    window_case_root
                    / "commands"
                    / f"subblock_{int(window_subblock_index):03d}.sh"
                )
                row = {
                    **subblock_plan_row,
                    "draw_index": _case_draw_index(case),
                    "global_subblock_index": int(subblock_index),
                    "case_posterior_path": str(posterior_path),
                    "expected_case_posterior_path": str(posterior_path),
                    "window_summary_path": str(window_summary_path),
                    "iterative_reference_update_path": str(reference_update_path),
                    "window_diagnostic_path": str(window_diagnostic_path),
                    "eigen_update_diagnostics_path": str(
                        eigen_update_diagnostics_path
                    ),
                    "eigen_update_modes_path": str(eigen_update_modes_path),
                    "window_case_name": window_case_name,
                    "planned_command_template": " ".join(commands[case.case_name][-1]),
                    "realized_command_path": str(realized_command_path),
                    "realized_after_reference_update": True,
                    "update_gain": float(iterative_cfg["update_gain"]),
                    "update_mode": str(iterative_cfg["update_mode"]),
                    "update_safety_json": json.dumps(
                        iterative_cfg.get("update_safety", {}), sort_keys=True
                    ),
                    "carry_prior_mean_with_reference": bool(
                        iterative_cfg["carry_prior_mean_with_reference"]
                    ),
                    "summary_information_scale": str(
                        effective_subblock_cfg.get("summary_information_scale", "")
                    ),
                    "trace_source_mode": str(trace_source_plan.mode),
                    "high_order_wfe_enabled": bool(model_split.enabled_components.get("high_order_wfe", {}).get("enabled", False)),
                    "high_order_wfe_summary_json": model_split.artifact_paths.get("high_order_wfe_high_order_wfe_summary_json", ""),
                    **active_template_hashes,
                    "theta_reference_offsets_window0_json": json.dumps(
                        dict(case.theta_reference_offsets), sort_keys=True
                    )
                    if int(window_index) == 0
                    else "",
                    "initial_truth_reference_provenance": "campaign_truth_plus_case_offsets",
                }
                iterative_plan_rows.append(row)
                expected_output_rows.append(
                    {
                        "case_name": case.case_name,
                        "case_origin": case.case_origin,
                        "draw_index": _case_draw_index(case),
                        "window_index": int(window_index),
                        "subblock_index": int(window_subblock_index),
                        "global_subblock_index": int(subblock_index),
                        "summary_path": str(summary_path),
                        "case_posterior_path": str(posterior_path),
                        "window_summary_path": str(window_summary_path),
                        "iterative_reference_update_path": str(reference_update_path),
                        "window_diagnostic_path": str(window_diagnostic_path),
                        "eigen_update_diagnostics_path": str(
                            eigen_update_diagnostics_path
                        ),
                        "eigen_update_modes_path": str(eigen_update_modes_path),
                        "window_case_name": window_case_name,
                        "realized_command_path": str(realized_command_path),
                        "realized_after_reference_update": True,
                        "update_gain": float(iterative_cfg["update_gain"]),
                        "update_mode": str(iterative_cfg["update_mode"]),
                        "update_safety_json": json.dumps(
                            iterative_cfg.get("update_safety", {}), sort_keys=True
                        ),
                        "summary_information_scale": str(
                            effective_subblock_cfg.get("summary_information_scale", "")
                        ),
                        "trace_source_mode": str(trace_source_plan.mode),
                        "high_order_wfe_enabled": bool(model_split.enabled_components.get("high_order_wfe", {}).get("enabled", False)),
                        "high_order_wfe_summary_json": model_split.artifact_paths.get("high_order_wfe_high_order_wfe_summary_json", ""),
                        **active_template_hashes,
                        "trace_seed": int(seeds["trace_seed"]),
                        "noise_seed": int(seeds["noise_seed"]),
                    }
                )
    resolved_config = {
        **config,
        "experiment": experiment_cfg,
        "system": system_cfg,
        "reference_system": reference_system_cfg,
    }
    resolved_config["experiment"]["subblocks"] = effective_subblock_cfg
    resolved_config["experiment"]["high_order_wfe_summary"] = model_split.provenance.get("high_order_wfe", {})
    resolved_config["experiment"]["model_split"] = model_split.to_dict()
    resolved_config["experiment"]["template_hashes"] = template_hash_rows or [template_hashes]
    resolved_config["experiment"]["subblock_resolution"] = subblock_resolution
    resolved_config["experiment"]["noise"] = {
        "structured_noise_supported": True,
        "normalized": normalized_noise.to_dict(),
        "artifact_paths": {key: str(path) for key, path in noise_provenance_paths.items()},
    }
    return CampaignPlan(
        run_root=run_root,
        layout=layout,
        layout_metadata=layout_metadata,
        prior_truth=prior_truth,
        cases=cases,
        subblock_commands=commands,
        summary_paths=summary_paths,
        subblock_plans=subblock_plans,
        prior_draw_rows_by_case=prior_draw_rows_by_case,
        config=resolved_config,
        partition=partition,
        case_generation=case_generation,
        truth_realization=dict(truth_realization.summary),
        truth_realization_rows=list(truth_realization.rows),
        trace_source_plan=trace_source_plan,
        iterative=iterative_cfg,
        iterative_plan_rows=iterative_plan_rows,
        expected_output_rows=expected_output_rows,
    )


def _plan_payload(plan: CampaignPlan) -> dict[str, Any]:
    subblock_cfg = plan.config["experiment"].get("subblocks", {})
    subblock_command_options = resolve_subblock_command_options(subblock_cfg)
    n_phi = 3 * int(subblock_cfg.get("n_frames", 3))
    combined_dim = int(plan.layout.size + n_phi)
    max_dense_dim = int(subblock_cfg.get("max_dense_dim", 40))
    eigen_cfg = dict(plan.config["experiment"].get("eigenbasis", {}) or {})
    forecast_cfg = _forecast_config(plan.config["experiment"])
    iterative_cfg = _resolve_iterative_config(plan.config["experiment"])
    iterative_forecast_cfg = _resolve_iterative_forecast_config(plan.config["experiment"])
    actual_n_summaries = int(plan.config["experiment"].get("subblocks", {}).get("n_subblocks", 3))
    smear_summary_rows = _smear_summary_rows_for_plan(plan, strict=True)
    forecast_grid = parse_forecast_grid(
        forecast_cfg.get("n_subblocks_grid"),
        actual_n_summaries=actual_n_summaries,
    )
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "run_root": str(plan.run_root),
        "theta_layout": plan.layout.to_dict(),
        "layout_metadata": plan.layout_metadata,
        "state_partition": plan.partition,
        "case_generation": dict(plan.case_generation),
        "truth_realization": dict(plan.truth_realization),
        "prior_truth_by_label": {
            label: float(plan.prior_truth[index])
            for index, label in enumerate(plan.layout.labels)
        },
        "trace_source": dict(plan.trace_source_plan.summary),
        "model_split": dict(
            plan.config["experiment"].get("model_split", {})
        ),
        "template_hashes": list(
            plan.config["experiment"].get("template_hashes", [])
        ),
        "smear_audit": {
            "summary_csv": str((plan.run_root / "trajectory" / "smear_summary.csv").resolve()),
            "n_subblocks": len(smear_summary_rows),
            "rows": smear_summary_rows,
            "notes": [
                "Actual subblock_constant_layer kernels are audited from per-subblock render/inference templates.",
            ]
            if smear_summary_rows
            else [],
        },
        "high_order_wfe": dict(
            plan.config["experiment"].get("high_order_wfe_summary", {})
        ),
        "subblock_command_options": dict(subblock_command_options),
        "seeding": _resolve_seeding_config(plan.config["experiment"]),
        "iterative": dict(plan.iterative),
        "iterative_forecast": {
            **dict(iterative_forecast_cfg),
            "enabled": bool(iterative_forecast_cfg.get("enabled", False)),
            "rendered_subblocks_total": int(
                len(plan.cases)
                * int(iterative_forecast_cfg.get("actual_windows", iterative_cfg.get("windows_per_draw", 1)))
                * int(iterative_forecast_cfg.get("subblocks_per_window", iterative_cfg.get("subblocks_per_window", 1)))
            )
            if bool(iterative_forecast_cfg.get("enabled", False))
            else 0,
            "projected_endpoint_subblocks_per_case": int(
                iterative_forecast_cfg.get("projected_subblocks_total", 0)
            ),
            "case_output_dir": str(plan.run_root / "analysis"),
            "artifacts": [
                "analysis/final_observation_summary.csv",
                "analysis/final_observation_summary.json",
                "analysis/projected_observation_forecast.csv",
                "analysis/window_evolution_actual_and_projected.csv",
            ],
        },
        "eigenbasis": {
            **eigen_cfg,
            "resolved_sources": list(_resolve_eigen_sources(eigen_cfg)),
        },
        "forecast": {
            "enabled": bool(forecast_cfg.get("enabled", False)),
            "iterative_warning": (
                "forecast.enabled=true with iterative.enabled=true can be much heavier; "
                "forecast is not needed for first iterative validation."
                if bool(forecast_cfg.get("enabled", False))
                and bool(iterative_cfg.get("enabled", False))
                else ""
            ),
            "modes": list(forecast_cfg.get("modes", [])),
            "n_subblocks_grid": list(forecast_grid),
            "single_observation_n_subblocks": int(
                forecast_cfg.get("single_observation_n_subblocks", 1800)
            ),
            "subblock_duration_s": float(forecast_cfg.get("subblock_duration_s", 1.0)),
            "case_output_dirs": {
                case.case_name: str(plan.run_root / "cases" / case.case_name / "forecast")
                for case in plan.cases
            },
            "limitations": {
                "replicate": (
                    "Repeats actual reduced information and score summaries; "
                    "not an independent shot-noise realization model."
                ),
                "fixed_information_score_noise": (
                    "Keeps template information fixed and samples score noise "
                    "with covariance alpha * S."
                ),
            },
        },
        "bias_cases": [
            {
                "case_name": case.case_name,
                "theta_reference_offsets": dict(case.theta_reference_offsets),
                "case_origin": case.case_origin,
                "has_case_prior_sigma": bool(case.prior_sigma_by_label),
                "is_prior_draw": bool(case.case_origin == "prior_draw"),
            }
            for case in plan.cases
        ],
        "prior_draw_rows_by_case": {
            case_name: list(rows)
            for case_name, rows in plan.prior_draw_rows_by_case.items()
        },
        "dimension_estimate": {
            "n_theta": int(plan.layout.size),
            "n_phi": int(n_phi),
            "combined_dim": int(combined_dim),
            "max_dense_dim": int(max_dense_dim),
            "dense_schur_allowed": bool(combined_dim <= max_dense_dim),
            "structured_schur_recommended": bool(combined_dim > max_dense_dim),
        },
        "subblock_commands": {
            name: [" ".join(command) for command in commands]
            for name, commands in plan.subblock_commands.items()
        },
        "summary_paths": {
            name: [str(path) for path in paths]
            for name, paths in plan.summary_paths.items()
        },
        "subblock_plan": {
            name: list(items) for name, items in plan.subblock_plans.items()
        },
        "iterative_plan": list(plan.iterative_plan_rows),
        "expected_outputs": list(plan.expected_output_rows),
        "notes": [
            "Trace-source mode iid_jitter preserves legacy trace-template behavior.",
            "Trajectory mode writes frame_truth.csv for render truth and starting_guess_prediction.csv for optimizer initialization only.",
            "Trajectory mode uses one pointing history shared across bias cases.",
            "Iterative physical_full mode stores later-window command templates at plan time and exact realized commands after each reference update.",
        ],
    }


def execute_subblocks(
    plan: CampaignPlan,
    *,
    resume: bool,
    max_workers: int,
    fail_fast: bool,
    quiet: bool,
    resource_time: bool | str | None,
) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    jobs: list[tuple[str, Path, list[str]]] = []
    for case_name, commands in plan.subblock_commands.items():
        for command, summary_path in zip(commands, plan.summary_paths[case_name], strict=True):
            if resume and summary_path.exists():
                continue
            jobs.append((case_name, summary_path, command))
    if not jobs:
        return
    require_resource_time_available(resource_time)

    status_rows: list[dict[str, Any]] = []

    def _job(item: tuple[str, Path, list[str]]) -> tuple[str, Path, dict[str, Any]]:
        case_name, summary_path, command = item
        case_root = summary_path.parent.parent
        diagnostics_path = case_root / "subprocess_diagnostics.json"
        diag = run_subprocess_with_diagnostics(
            command=command,
            cwd=REPO_ROOT,
            env=dict(env),
            stdout_log=case_root / "subprocess.stdout.log",
            stderr_log=case_root / "subprocess.stderr.log",
            diagnostics_json=diagnostics_path,
            resource_time=resource_time,
        )
        return case_name, summary_path, {
            "case_name": case_name,
            "summary_path": str(summary_path),
            "status": "ok" if int(diag.return_code) == 0 else "failed",
            "return_code": int(diag.return_code),
            "failure_class": diag.failure_class,
            "failure_hint": diag.failure_hint,
            "subprocess_diagnostics_path": str(diagnostics_path),
            "stdout_log": str(diag.stdout_log),
            "stderr_log": str(diag.stderr_log),
            "last_stderr_line": diag.last_stderr_line,
            "resource_time_maximum_resident_set_mb": diag.resource_time.get(
                "maximum_resident_set_mb"
            ),
            "resource_time_mode_effective": diag.resource_time.get(
                "resource_time_mode_effective",
                diag.resource_time.get("mode_effective"),
            ),
            "stderr_tail": "\n".join(diag.stderr_tail),
        }

    def _record(row: Mapping[str, Any]) -> None:
        status_rows.append(dict(row))
        _write_csv_rows(plan.run_root / "subblock_status.csv", status_rows)
        _write_csv_rows(plan.run_root / "memory_failure_summary.csv", status_rows)
        failed = sum(item.get("status") == "failed" for item in status_rows)
        _write_json(
            plan.run_root / "progress.json",
            {
                "schema_version": "observation_bias_campaign_progress.v1",
                "updated_at": now_iso_local_ms(),
                "run_root": str(plan.run_root),
                "total_subblocks_planned": len(jobs),
                "completed_count": len(status_rows),
                "failed_count": int(failed),
                "pending_count": len(jobs) - len(status_rows),
                "last_completed": dict(row),
            },
        )

    def _raise_for_failure(row: Mapping[str, Any]) -> None:
        if int(row.get("return_code", 0)) == 0:
            return
        raise RuntimeError(
            f"Subprocess failed ({row['return_code']}) for {row['summary_path']}: "
            f"{row['subprocess_diagnostics_path']}\nchild stderr tail:\n"
            f"{row.get('stderr_tail') or '<empty>'}"
        )

    if max_workers <= 1:
        for job in jobs:
            try:
                case_name, summary_path, row = _job(job)
                _record(row)
                _raise_for_failure(row)
                if not quiet:
                    print(f"[observation_bias_campaign] completed {case_name}: {summary_path}", flush=True)
            except RuntimeError as exc:
                if fail_fast:
                    raise RuntimeError(str(exc)) from exc
                print(str(exc), file=sys.stderr, flush=True)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            try:
                case_name, summary_path, row = future.result()
                _record(row)
                _raise_for_failure(row)
                if not quiet:
                    print(f"[observation_bias_campaign] completed {case_name}: {summary_path}", flush=True)
            except RuntimeError as exc:
                if fail_fast:
                    raise RuntimeError(str(exc)) from exc
                print(str(exc), file=sys.stderr, flush=True)


def _execution_context_payload(
    *,
    run_root: Path,
    max_workers: int,
    resource_time: bool | str | None,
) -> dict[str, Any]:
    env = os.environ
    return {
        "schema_version": "observation_bias_execution_context.v1",
        "created_at": now_iso_local_ms(),
        "run_root": str(run_root),
        "max_workers": int(max_workers),
        "resource_time": resource_time,
        "slurm": {
            "job_id": env.get("SLURM_JOB_ID"),
            "job_name": env.get("SLURM_JOB_NAME"),
            "cpus_per_task": env.get("SLURM_CPUS_PER_TASK"),
            "mem_per_node": env.get("SLURM_MEM_PER_NODE"),
            "mem_per_cpu": env.get("SLURM_MEM_PER_CPU"),
            "job_nodelist": env.get("SLURM_JOB_NODELIST"),
        },
        "threading": {
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": env.get("NUMEXPR_NUM_THREADS"),
        },
        "jax": {
            "XLA_FLAGS": env.get("XLA_FLAGS"),
            "JAX_COMPILATION_CACHE_DIR": env.get("JAX_COMPILATION_CACHE_DIR"),
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": env.get(
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"
            ),
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": env.get(
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"
            ),
        },
    }


def _flatten_subblock_plan_rows(plan: CampaignPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name, items in plan.subblock_plans.items():
        for item, command in zip(items, plan.subblock_commands[case_name], strict=True):
            rows.append(
                {
                    **item,
                    "command": " ".join(command),
                }
            )
    return rows


def _smear_summary_rows_for_plan(plan: CampaignPlan, *, strict: bool = False) -> list[dict[str, Any]]:
    rows = _flatten_subblock_plan_rows(plan)
    if not any(row.get("smear_representative_kernel_json") for row in rows):
        return []
    return build_smear_summary_rows(rows, run_root=plan.run_root, strict=strict)


def _flatten_prior_draw_rows(plan: CampaignPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name, case_rows in plan.prior_draw_rows_by_case.items():
        for row in case_rows:
            rows.append(dict(row))
    return rows


def _bias_case_rows(plan: CampaignPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in plan.cases:
        rows.append(
            {
                "case_name": case.case_name,
                "case_origin": case.case_origin,
                "has_case_prior_sigma": bool(case.prior_sigma_by_label),
                "has_prior_draw_metadata": bool(case.prior_draw_metadata),
                "n_theta_offsets": len(case.theta_reference_offsets),
            }
        )
    return rows


def _load_case_summaries(
    paths: Sequence[Path],
    *,
    summary_scale_policy: str = SUMMARY_SCALE_POLICY_REQUIRE_SUMMED,
) -> tuple[list[SubblockSummary], dict[str, Any]]:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing subblock summaries: " + ", ".join(str(path) for path in missing)
        )
    summaries = [load_subblock_summary(path) for path in paths]
    labels = tuple(summaries[0].theta_labels)
    for index, summary in enumerate(summaries[1:], start=1):
        if tuple(summary.theta_labels) != labels:
            raise ValueError(
                f"Summary {index} labels differ from the first summary for this case."
            )
    summary_scale_validation = validate_summary_information_scale(
        [load_subblock_summary_artifact_payload(path) for path in paths],
        policy=summary_scale_policy,
        summary_paths=paths,
    )
    return summaries, summary_scale_validation


def _truth_reference_maps(
    *,
    plan: CampaignPlan,
    case: BiasCase,
    summaries: Sequence[SubblockSummary],
    summary_paths: Sequence[Path],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    labels = tuple(plan.layout.labels)
    truth = {label: float(plan.prior_truth[index]) for index, label in enumerate(labels)}
    reference = {
        label: float(summaries[0].theta_ref[index]) for index, label in enumerate(labels)
    }
    offsets = {label: 0.0 for label in labels}
    offsets.update({key: float(value) for key, value in case.theta_reference_offsets.items()})
    for path in summary_paths:
        try:
            payload = load_subblock_summary_artifact_payload(path)
        except Exception:
            continue
        override_payload = payload.get("metadata", {}).get("theta_reference_overrides")
        if not isinstance(override_payload, Mapping):
            override_payload = payload.get("theta_reference_overrides")
        items = override_payload.get("items", []) if isinstance(override_payload, Mapping) else []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            key = str(item.get("key", ""))
            if key not in labels:
                continue
            if item.get("truth_value") is not None:
                truth[key] = float(item["truth_value"])
            elif item.get("reference_base_value") is not None:
                truth[key] = float(item["reference_base_value"])
            if item.get("reference_value") is not None:
                reference[key] = float(item["reference_value"])
            if item.get("offset") is not None:
                offsets[key] = float(item["offset"])
    for key, offset in offsets.items():
        if abs(offset) > 0.0 and key in reference:
            truth.setdefault(key, float(reference[key] - offset))
    return truth, reference, offsets


def _posterior_rows(
    *,
    case_name: str,
    labels: Sequence[str],
    truth: Mapping[str, float],
    reference: Mapping[str, float],
    offsets: Mapping[str, float],
    posterior_mean: np.ndarray,
    posterior_sigma: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        truth_value = float(truth[label])
        reference_value = float(reference[label])
        posterior_value = float(posterior_mean[index])
        posterior_error = posterior_value - truth_value
        reference_error = reference_value - truth_value
        posterior_shift = posterior_value - reference_value
        rows.append(
            {
                "case_name": case_name,
                "theta_label": label,
                "truth_value": truth_value,
                "reference_value": reference_value,
                "theta_reference_offset": float(offsets.get(label, 0.0)),
                "posterior_mean": posterior_value,
                "posterior_shift": posterior_shift,
                "posterior_error": posterior_error,
                "posterior_sigma": float(posterior_sigma[index]),
                "posterior_error_over_sigma": _safe_fraction(posterior_error, float(posterior_sigma[index])),
                "correction_fraction": _safe_fraction(posterior_shift, truth_value - reference_value),
                "residual_fraction": _safe_fraction(posterior_error, reference_error),
                "label_group": _label_group(label),
                "unit": _parameter_unit(label),
            }
        )
    return rows


def _science_row(
    *,
    case_name: str,
    labels: Sequence[str],
    truth: Mapping[str, float],
    reference: Mapping[str, float],
    posterior_mean: np.ndarray,
    posterior_sigma: np.ndarray,
) -> dict[str, Any]:
    label = "source.separation_as"
    index = labels.index(label)
    truth_value = float(truth[label])
    reference_value = float(reference[label])
    posterior_value = float(posterior_mean[index])
    error = posterior_value - truth_value
    sigma = float(posterior_sigma[index])
    prior_error = reference_value - truth_value
    return {
        "case_name": case_name,
        "truth_separation_as": truth_value,
        "reference_separation_as": reference_value,
        "posterior_separation_as": posterior_value,
        "posterior_separation_shift_as": posterior_value - reference_value,
        "posterior_separation_error_as": error,
        "posterior_separation_error_microas": error * 1.0e6,
        "posterior_separation_sigma_as": sigma,
        "posterior_separation_sigma_microas": sigma * 1.0e6,
        "posterior_separation_error_over_sigma": _safe_fraction(error, sigma),
        "separation_correction_fraction": _safe_fraction(posterior_value - reference_value, truth_value - reference_value),
        "moves_separation_toward_truth": bool(abs(error) < abs(prior_error)),
    }


def _top_contributors(
    *,
    basis: Any,
    mode_index: int,
    top_k: int,
) -> list[tuple[str, float]]:
    return basis.mode_contributors(mode_index, top_k=max(1, top_k))


def _eigen_rows(
    *,
    case_name: str,
    source_name: str,
    labels: Sequence[str],
    matrix: np.ndarray,
    prior_sigma: np.ndarray,
    reference_error: np.ndarray,
    posterior_error: np.ndarray,
    posterior_shift: np.ndarray,
    eigen_cfg: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    Any,
]:
    whiten = bool(eigen_cfg.get("whiten", True))
    source_matrix = (
        build_prior_whitened_information_gain_matrix(matrix, prior_sigma)
        if whiten
        else np.asarray(matrix, dtype=float)
    )
    basis = build_observation_eigenbasis(
        source_matrix,
        labels,
        eig_floor_abs=float(eigen_cfg.get("eig_floor_abs", 0.0)),
        eig_floor_rel=float(eigen_cfg.get("eig_floor_rel", 1.0e-12)),
    )
    top_k = int(eigen_cfg.get("top_k_contributors", 8))
    raw_sigmas = basis.raw_sigma_along_modes()
    eigenvalue_rows: list[dict[str, Any]] = []
    contributor_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    projected_reference_error = reference_error / prior_sigma if whiten else reference_error
    projected_posterior_error = posterior_error / prior_sigma if whiten else posterior_error
    projected_posterior_shift = posterior_shift / prior_sigma if whiten else posterior_shift
    prior_proj = basis.physical_delta_to_eigen(projected_reference_error)
    posterior_proj = basis.physical_delta_to_eigen(projected_posterior_error)
    shift_proj = basis.physical_delta_to_eigen(projected_posterior_shift)
    for mode_index, eigenvalue in enumerate(basis.eigenvalues):
        eigenvalue_rows.append(
            {
                "case_name": case_name,
                "mode_index": int(mode_index),
                "eigenvalue": float(eigenvalue),
                "sqrt_eigenvalue": float(math.sqrt(max(float(eigenvalue), 0.0))),
                "sigma_equivalent": float(raw_sigmas[mode_index]),
                "is_weak": bool(basis.weak_mode_mask[mode_index]),
                "is_retained": bool(not basis.was_floored()[mode_index]),
            }
        )
        contributors = _top_contributors(basis=basis, mode_index=mode_index, top_k=top_k)
        for rank, (label, coeff) in enumerate(contributors, start=1):
            contributor_rows.append(
                {
                    "case_name": case_name,
                    "mode_index": int(mode_index),
                    "rank": int(rank),
                    "theta_label": label,
                    "coefficient": float(coeff),
                    "abs_coefficient": float(abs(coeff)),
                    "label_group": _label_group(label),
                    "unit": _parameter_unit(label),
                }
            )
        top_summary = "; ".join(f"{label}:{coeff:+.4f}" for label, coeff in contributors[:3])
        projection_rows.append(
            {
                "case_name": case_name,
                "mode_index": int(mode_index),
                "eigenvalue": float(eigenvalue),
                "prior_error_projection": float(prior_proj[mode_index]),
                "posterior_error_projection": float(posterior_proj[mode_index]),
                "posterior_shift_projection": float(shift_proj[mode_index]),
                "correction_fraction_projection": _safe_fraction(
                    float(shift_proj[mode_index]),
                    -float(prior_proj[mode_index]),
                ),
                "top_contributor_summary": top_summary,
            }
        )
    weak_mode_rows: list[dict[str, Any]] = []
    for mode_index, eigenvalue in enumerate(basis.eigenvalues):
        if not bool(basis.weak_mode_mask[mode_index]):
            continue
        vector = basis.eigenvectors[:, int(mode_index)]
        abs_vector = np.abs(vector)
        primary_weight = float(
            np.sum(
                abs_vector[
                    [
                        i
                        for i, label in enumerate(labels)
                        if label.startswith("optics.primary.zernike_coeffs_nm")
                    ]
                ]
            )
        )
        secondary_weight = float(
            np.sum(
                abs_vector[
                    [
                        i
                        for i, label in enumerate(labels)
                        if label.startswith("optics.secondary.zernike_coeffs_nm")
                    ]
                ]
            )
        )
        source_scalar_weight = float(
            np.sum(
                abs_vector[
                    [i for i, label in enumerate(labels) if label.startswith("source.")]
                ]
            )
        )
        plate_scale_weight = float(
            np.sum(
                abs_vector[
                    [
                        i
                        for i, label in enumerate(labels)
                        if label == "optics.plate_scale_as_per_pix"
                    ]
                ]
            )
        )
        separation_weight = float(
            np.sum(
                abs_vector[
                    [i for i, label in enumerate(labels) if label == "source.separation_as"]
                ]
            )
        )
        group_weights = {
            "primary_zernikes": primary_weight,
            "secondary_zernikes": secondary_weight,
            "source_scalars": source_scalar_weight,
            "plate_scale": plate_scale_weight,
        }
        dominant_group = max(group_weights, key=lambda item: group_weights[item])
        notes = ""
        if dominant_group in {"primary_zernikes", "secondary_zernikes"}:
            notes = "M1/M2-dominated weak mode candidate"
        weak_mode_rows.append(
            {
                "case_name": case_name,
                "source_matrix": source_name,
                "mode_index": int(mode_index),
                "eigenvalue": float(eigenvalue),
                "sigma_equivalent": float(raw_sigmas[mode_index]),
                "top_contributor_summary": projection_rows[mode_index]["top_contributor_summary"],
                "primary_zernike_weight": primary_weight,
                "secondary_zernike_weight": secondary_weight,
                "source_scalar_weight": source_scalar_weight,
                "plate_scale_weight": plate_scale_weight,
                "separation_weight": separation_weight,
                "dominant_group": dominant_group,
                "notes": notes,
            }
        )
    summary = {
        "case_name": case_name,
        "source_matrix": source_name,
        "source_matrix_whitened": bool(whiten),
        "condition_number": float(basis.condition_number),
        "eig_floor_abs": float(basis.eig_floor_abs),
        "eig_floor_rel": float(basis.eig_floor_rel),
        "n_modes": int(len(labels)),
        "n_weak_modes": int(np.count_nonzero(basis.weak_mode_mask)),
    }
    return (
        eigenvalue_rows,
        contributor_rows,
        projection_rows,
        weak_mode_rows,
        summary,
        basis,
    )


def _plot_eigenvalue_spectrum(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAVE_MATPLOTLIB or plt is None:
        return
    if not rows:
        return
    x = np.asarray([int(row["mode_index"]) + 1 for row in rows], dtype=int)
    y = np.asarray([float(row["eigenvalue"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x, np.clip(y, 1.0e-30, None), marker="o")
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_group_error(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAVE_MATPLOTLIB or plt is None:
        return
    if not rows:
        return
    groups = sorted(set(str(row["label_group"]) for row in rows))
    values = [
        np.nanmax([
            abs(float(row["posterior_error_over_sigma"]))
            for row in rows
            if row["label_group"] == group and math.isfinite(float(row["posterior_error_over_sigma"]))
        ] or [0.0])
        for group in groups
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(groups, values)
    ax.set_ylabel("max |posterior error / sigma|")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _forecast_config(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(experiment_cfg.get("forecast", {}) or {})
    if "enabled" not in cfg:
        cfg["enabled"] = False
    if "modes" not in cfg:
        cfg["modes"] = ["replicate", "fixed_information_score_noise"]
    cfg.setdefault("n_subblocks_grid", [1, 3, 5, 10, 30, 100, 300, 1000, 1800])
    cfg.setdefault("subblock_duration_s", 1.0)
    cfg.setdefault("single_observation_n_subblocks", 1800)
    cfg.setdefault("plots", True)
    cfg.setdefault("replicate", {"enabled": True})
    cfg.setdefault(
        "fixed_information_score_noise",
        {
            "enabled": True,
            "n_trials": 100,
            "seed": 2026,
            "score_noise_alpha": 1.0,
            "score_noise_eig_floor_abs": 0.0,
            "score_noise_eig_floor_rel": 1.0e-12,
            "truth_mode": "campaign_truth",
        },
    )
    modes = tuple(str(mode) for mode in cfg.get("modes", ()))
    invalid = [mode for mode in modes if mode not in SUPPORTED_FORECAST_MODES]
    if invalid:
        raise ValueError("Unsupported forecast mode(s): " + ", ".join(sorted(invalid)))
    cfg["modes"] = list(modes)
    return cfg


def parse_forecast_grid(
    raw: Sequence[int] | None,
    *,
    actual_n_summaries: int,
) -> tuple[int, ...]:
    values = list(raw or [1, 3, 5, 10, 30, 100, 300, 1000, 1800])
    values.append(int(actual_n_summaries))
    parsed: list[int] = []
    for value in values:
        count = int(value)
        if count < 1:
            raise ValueError("forecast n_subblocks values must be >= 1.")
        parsed.append(count)
    return tuple(sorted(set(parsed)))


def replicate_summaries_for_count(
    summaries: Sequence[SubblockSummary],
    n_subblocks: int,
) -> tuple[SubblockSummary, ...]:
    if not summaries:
        raise ValueError("At least one summary is required for forecast replication.")
    if int(n_subblocks) < 1:
        raise ValueError("n_subblocks must be >= 1.")
    return tuple(summaries[index % len(summaries)] for index in range(int(n_subblocks)))


def _sample_score_noise(
    information: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float,
    eig_floor_abs: float,
    eig_floor_rel: float,
) -> np.ndarray:
    if alpha < 0.0:
        raise ValueError("score_noise_alpha must be non-negative.")
    matrix = 0.5 * (np.asarray(information, dtype=float) + np.asarray(information, dtype=float).T)
    if alpha == 0.0:
        return np.zeros((matrix.shape[0],), dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    max_eig = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    floor = max(float(eig_floor_abs), float(eig_floor_rel) * max(max_eig, 0.0))
    effective = np.clip(eigenvalues, floor, None)
    standard = rng.normal(loc=0.0, scale=1.0, size=effective.shape)
    return eigenvectors @ (np.sqrt(alpha * effective) * standard)


def synthesize_score_noise_summaries(
    summaries: Sequence[SubblockSummary],
    *,
    n_subblocks: int,
    theta_true: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    eig_floor_abs: float,
    eig_floor_rel: float,
) -> tuple[SubblockSummary, ...]:
    templates = replicate_summaries_for_count(summaries, n_subblocks)
    synthetic: list[SubblockSummary] = []
    truth = np.asarray(theta_true, dtype=float)
    for index, template in enumerate(templates):
        expected = template.reduced_information @ (template.theta_ref - truth)
        score = expected + _sample_score_noise(
            template.reduced_information,
            rng=rng,
            alpha=float(alpha),
            eig_floor_abs=float(eig_floor_abs),
            eig_floor_rel=float(eig_floor_rel),
        )
        synthetic.append(
            SubblockSummary.from_reduced_form(
                subblock_id=f"forecast_{index:06d}_{template.subblock_id}",
                theta_labels=template.theta_labels,
                theta_ref=template.theta_ref,
                reduced_information=template.reduced_information,
                reduced_score=score,
                summary_kind=f"{template.summary_kind}_score_noise_forecast",
                diagnostics={
                    "template_subblock_id": template.subblock_id,
                    "forecast_sequence_index": int(index),
                    "score_noise_alpha": float(alpha),
                },
            )
        )
    return tuple(synthetic)


def _forecast_duration_fields(
    *,
    n_subblocks: int,
    actual_n_summaries: int,
    subblock_duration_s: float,
    single_observation_n_subblocks: int,
) -> dict[str, Any]:
    duration_s = float(n_subblocks) * float(subblock_duration_s)
    return {
        "n_actual_summaries": int(actual_n_summaries),
        "n_subblocks": int(n_subblocks),
        "equivalent_duration_s": duration_s,
        "equivalent_duration_min": duration_s / 60.0,
        "is_actual_summary_count": bool(int(n_subblocks) == int(actual_n_summaries)),
        "is_single_observation_target": bool(
            int(n_subblocks) == int(single_observation_n_subblocks)
        ),
    }


def build_forecast_information_row(
    *,
    case_name: str,
    forecast_mode: str,
    labels: Sequence[str],
    summaries: Sequence[SubblockSummary],
    n_subblocks: int,
    actual_n_summaries: int,
    subblock_duration_s: float,
    single_observation_n_subblocks: int,
) -> dict[str, Any]:
    info = accumulate_summary_information(labels, summaries)
    diagnostics = _matrix_diagnostics(info)
    return {
        "case_name": case_name,
        "forecast_mode": forecast_mode,
        **_forecast_duration_fields(
            n_subblocks=n_subblocks,
            actual_n_summaries=actual_n_summaries,
            subblock_duration_s=subblock_duration_s,
            single_observation_n_subblocks=single_observation_n_subblocks,
        ),
        "accumulated_information_rank_estimate": int(diagnostics.rank_estimate),
        "accumulated_information_min_eigenvalue": float(diagnostics.min_eigenvalue),
        "accumulated_information_condition_number": float(diagnostics.condition_number),
        "accumulated_information_trace": float(diagnostics.trace),
    }


def build_forecast_science_row(
    *,
    case_name: str,
    forecast_mode: str,
    labels: Sequence[str],
    update_result: Any,
    accumulated_information: np.ndarray,
    prior_sigma: np.ndarray,
    theta_true: np.ndarray,
    n_subblocks: int,
    actual_n_summaries: int,
    subblock_duration_s: float,
    single_observation_n_subblocks: int,
    prior_sigma_source: str,
    n_trials: int | None = None,
) -> dict[str, Any]:
    label = "source.separation_as"
    index = labels.index(label) if label in labels else None
    posterior_sigma = update_result.posterior.sigma()
    posterior_diag = _matrix_diagnostics(update_result.posterior.precision)
    info_diag = _matrix_diagnostics(accumulated_information)
    row: dict[str, Any] = {
        "case_name": case_name,
        "forecast_mode": forecast_mode,
        **_forecast_duration_fields(
            n_subblocks=n_subblocks,
            actual_n_summaries=actual_n_summaries,
            subblock_duration_s=subblock_duration_s,
            single_observation_n_subblocks=single_observation_n_subblocks,
        ),
        "theta_dim": int(len(labels)),
        "separation_label_found": bool(index is not None),
        "posterior_precision_rank_estimate": int(posterior_diag.rank_estimate),
        "posterior_precision_min_eigenvalue": float(posterior_diag.min_eigenvalue),
        "posterior_precision_condition_number": float(posterior_diag.condition_number),
        "accumulated_information_rank_estimate": int(info_diag.rank_estimate),
        "accumulated_information_condition_number": float(info_diag.condition_number),
        "prior_sigma_source": str(prior_sigma_source),
    }
    if n_trials is not None:
        row["n_trials"] = int(n_trials)
    if index is None:
        row.update(
            {
                "separation_posterior_sigma_as": float("nan"),
                "separation_posterior_sigma_microas": float("nan"),
                "separation_posterior_error_as": float("nan"),
                "separation_posterior_error_microas": float("nan"),
                "separation_posterior_error_over_sigma": float("nan"),
                "separation_posterior_sigma_over_prior_sigma": float("nan"),
            }
        )
        return row
    sigma = float(posterior_sigma[index])
    error = float(update_result.posterior.mean[index] - theta_true[index])
    row.update(
        {
            "separation_posterior_sigma_as": sigma,
            "separation_posterior_sigma_microas": sigma * 1.0e6,
            "separation_posterior_error_as": error,
            "separation_posterior_error_microas": error * 1.0e6,
            "separation_posterior_error_over_sigma": _safe_fraction(error, sigma),
            "separation_posterior_sigma_over_prior_sigma": _safe_fraction(
                sigma,
                float(prior_sigma[index]),
            ),
        }
    )
    return row


def build_forecast_posterior_rows(
    *,
    case_name: str,
    forecast_mode: str,
    labels: Sequence[str],
    update_result: Any,
    theta_true: np.ndarray,
    prior_sigma: np.ndarray,
    n_subblocks: int,
    trial_index: int | None = None,
) -> list[dict[str, Any]]:
    sigma = update_result.posterior.sigma()
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        error = float(update_result.posterior.mean[index] - theta_true[index])
        row: dict[str, Any] = {
            "case_name": case_name,
            "forecast_mode": forecast_mode,
            "n_subblocks": int(n_subblocks),
            "theta_label": label,
            "truth_value": float(theta_true[index]),
            "posterior_mean": float(update_result.posterior.mean[index]),
            "posterior_error": error,
            "posterior_sigma": float(sigma[index]),
            "posterior_error_over_sigma": _safe_fraction(error, float(sigma[index])),
            "posterior_sigma_over_prior_sigma": _safe_fraction(
                float(sigma[index]),
                float(prior_sigma[index]),
            ),
            "label_group": _label_group(label),
            "unit": _parameter_unit(label),
        }
        if trial_index is not None:
            row["trial_index"] = int(trial_index)
        rows.append(row)
    return rows


def _plot_forecast_line(
    *,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    y_key: str,
    ylabel: str,
    actual_n_summaries: int,
    single_observation_n_subblocks: int,
) -> None:
    if not _HAVE_MATPLOTLIB or plt is None or not rows:
        return
    x = np.asarray([int(row["n_subblocks"]) for row in rows], dtype=int)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=float)
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x[order], y[order], marker="o")
    ax.axvline(int(actual_n_summaries), color="0.4", linestyle="--", linewidth=1)
    ax.axvline(int(single_observation_n_subblocks), color="0.2", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Subblocks")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_forecast_manifest(
    *,
    path: Path,
    case_name: str,
    mode: str,
    summaries: Sequence[SubblockSummary],
    summary_paths: Sequence[Path],
    prior: ObservationBeliefState,
    prior_sigma: np.ndarray,
    theta_true: np.ndarray,
    forecast_cfg: Mapping[str, Any],
    grid: Sequence[int],
    extra: Mapping[str, Any] | None = None,
) -> None:
    limitations = {
        "replicate": (
            "This mode repeats the actual summary matrices and score vectors. "
            "It is an information accumulation sanity check and not an independent "
            "shot-noise realization model."
        ),
        "fixed_information_score_noise": (
            "This mode keeps the observed/template information matrices fixed and "
            "draws independent score noise with covariance alpha * S. Alpha is "
            "Fisher-consistent when alpha=1 but is not yet empirically calibrated "
            "unless a calibration artifact is supplied."
        ),
    }
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "case_name": case_name,
        "forecast_mode": mode,
        "limitations": limitations.get(mode, ""),
        "input_summary_paths": [str(path) for path in summary_paths],
        "n_actual_summaries": int(len(summaries)),
        "input_summary_theta_refs": [summary.theta_ref.tolist() for summary in summaries],
        "input_summary_matrix_diagnostics": [
            _matrix_diagnostics(summary.reduced_information).to_dict()
            for summary in summaries
        ],
        "prior": prior.to_dict(),
        "prior_sigma": np.asarray(prior_sigma, dtype=float).tolist(),
        "truth_vector": np.asarray(theta_true, dtype=float).tolist(),
        "forecast_grid": [int(value) for value in grid],
        "subblock_duration_s": float(forecast_cfg.get("subblock_duration_s", 1.0)),
        "single_observation_n_subblocks": int(
            forecast_cfg.get("single_observation_n_subblocks", 1800)
        ),
    }
    if extra:
        payload.update(dict(extra))
    _write_json(path, payload)


def run_case_forecast(
    *,
    case_name: str,
    case_root: Path,
    summaries: Sequence[SubblockSummary],
    summary_paths: Sequence[Path],
    prior: ObservationBeliefState,
    prior_sigma: np.ndarray,
    theta_true: np.ndarray,
    theta_reference: np.ndarray,
    labels: Sequence[str],
    forecast_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = _forecast_config({"forecast": forecast_cfg})
    if not bool(cfg.get("enabled", False)):
        return {
            "forecast_results": [],
            "forecast_trial_results": [],
            "forecast_information_diagnostics": [],
            "forecast_summary": {"enabled": False},
        }
    grid = parse_forecast_grid(
        cfg.get("n_subblocks_grid"),
        actual_n_summaries=len(summaries),
    )
    subblock_duration_s = float(cfg.get("subblock_duration_s", 1.0))
    single_observation_n_subblocks = int(cfg.get("single_observation_n_subblocks", 1800))
    prior_sigma_source = str(prior.metadata.get("prior_sigma_source", "unknown"))
    forecast_root = case_root / "forecast"
    all_results: list[dict[str, Any]] = []
    all_trial_results: list[dict[str, Any]] = []
    all_trial_posterior_rows: list[dict[str, Any]] = []
    all_information_rows: list[dict[str, Any]] = []
    modes = tuple(str(mode) for mode in cfg.get("modes", ()))

    if "replicate" in modes and bool(dict(cfg.get("replicate", {}) or {}).get("enabled", True)):
        mode = "replicate"
        mode_root = forecast_root / mode
        rows: list[dict[str, Any]] = []
        posterior_rows: list[dict[str, Any]] = []
        info_rows: list[dict[str, Any]] = []
        for count in grid:
            sequence = replicate_summaries_for_count(summaries, count)
            update = update_observation_belief(prior, sequence)
            accumulated = accumulate_summary_information(labels, sequence)
            row = build_forecast_science_row(
                case_name=case_name,
                forecast_mode=mode,
                labels=labels,
                update_result=update,
                accumulated_information=accumulated,
                prior_sigma=prior_sigma,
                theta_true=theta_true,
                n_subblocks=count,
                actual_n_summaries=len(summaries),
                subblock_duration_s=subblock_duration_s,
                single_observation_n_subblocks=single_observation_n_subblocks,
                prior_sigma_source=prior_sigma_source,
            )
            rows.append(row)
            posterior_rows.extend(
                build_forecast_posterior_rows(
                    case_name=case_name,
                    forecast_mode=mode,
                    labels=labels,
                    update_result=update,
                    theta_true=theta_true,
                    prior_sigma=prior_sigma,
                    n_subblocks=count,
                )
            )
            info_rows.append(
                build_forecast_information_row(
                    case_name=case_name,
                    forecast_mode=mode,
                    labels=labels,
                    summaries=sequence,
                    n_subblocks=count,
                    actual_n_summaries=len(summaries),
                    subblock_duration_s=subblock_duration_s,
                    single_observation_n_subblocks=single_observation_n_subblocks,
                )
            )
        _write_csv_rows(mode_root / "forecast_results.csv", rows)
        _write_csv_rows(mode_root / "posterior_by_label_by_n_subblocks.csv", posterior_rows)
        _write_csv_rows(mode_root / "information_diagnostics.csv", info_rows)
        _write_forecast_manifest(
            path=mode_root / "manifest.json",
            case_name=case_name,
            mode=mode,
            summaries=summaries,
            summary_paths=summary_paths,
            prior=prior,
            prior_sigma=prior_sigma,
            theta_true=theta_true,
            forecast_cfg=cfg,
            grid=grid,
            extra={"theta_reference": np.asarray(theta_reference, dtype=float).tolist()},
        )
        if bool(cfg.get("plots", True)):
            _plot_forecast_line(
                path=mode_root / "separation_sigma_vs_n_subblocks.png",
                rows=rows,
                y_key="separation_posterior_sigma_microas",
                ylabel="Separation sigma (microas)",
                actual_n_summaries=len(summaries),
                single_observation_n_subblocks=single_observation_n_subblocks,
            )
            _plot_forecast_line(
                path=mode_root / "prior_normalized_sigma_vs_n_subblocks.png",
                rows=rows,
                y_key="separation_posterior_sigma_over_prior_sigma",
                ylabel="Separation sigma / prior sigma",
                actual_n_summaries=len(summaries),
                single_observation_n_subblocks=single_observation_n_subblocks,
            )
        all_results.extend(rows)
        all_information_rows.extend(info_rows)

    noise_cfg = dict(cfg.get("fixed_information_score_noise", {}) or {})
    if "fixed_information_score_noise" in modes and bool(noise_cfg.get("enabled", True)):
        mode = "fixed_information_score_noise"
        mode_root = forecast_root / mode
        n_trials = int(noise_cfg.get("n_trials", 100))
        if n_trials <= 0:
            raise ValueError("fixed_information_score_noise.n_trials must be positive.")
        seed = int(noise_cfg.get("seed", 2026))
        alpha = float(noise_cfg.get("score_noise_alpha", 1.0))
        if alpha < 0.0:
            raise ValueError("fixed_information_score_noise.score_noise_alpha must be non-negative.")
        eig_floor_abs = float(noise_cfg.get("score_noise_eig_floor_abs", 0.0))
        eig_floor_rel = float(noise_cfg.get("score_noise_eig_floor_rel", 1.0e-12))
        max_count = max(grid)
        trial_rows: list[dict[str, Any]] = []
        trial_posterior_rows: list[dict[str, Any]] = []
        synthesis_rows: list[dict[str, Any]] = []
        information_rows: list[dict[str, Any]] = []
        for trial_index in range(n_trials):
            rng = np.random.default_rng(make_subseed(seed, f"{case_name}.{mode}.{trial_index}"))
            sequence = synthesize_score_noise_summaries(
                summaries,
                n_subblocks=max_count,
                theta_true=theta_true,
                rng=rng,
                alpha=alpha,
                eig_floor_abs=eig_floor_abs,
                eig_floor_rel=eig_floor_rel,
            )
            synthesis_rows.append(
                {
                    "case_name": case_name,
                    "forecast_mode": mode,
                    "trial_index": int(trial_index),
                    "seed": int(seed),
                    "n_generated_summaries": int(len(sequence)),
                    "score_noise_alpha": float(alpha),
                    "score_noise_eig_floor_abs": float(eig_floor_abs),
                    "score_noise_eig_floor_rel": float(eig_floor_rel),
                }
            )
            for count in grid:
                prefix = sequence[:count]
                update = update_observation_belief(prior, prefix)
                accumulated = accumulate_summary_information(labels, prefix)
                row = build_forecast_science_row(
                    case_name=case_name,
                    forecast_mode=mode,
                    labels=labels,
                    update_result=update,
                    accumulated_information=accumulated,
                    prior_sigma=prior_sigma,
                    theta_true=theta_true,
                    n_subblocks=count,
                    actual_n_summaries=len(summaries),
                    subblock_duration_s=subblock_duration_s,
                    single_observation_n_subblocks=single_observation_n_subblocks,
                    prior_sigma_source=prior_sigma_source,
                    n_trials=n_trials,
                )
                row["trial_index"] = int(trial_index)
                trial_rows.append(row)
                trial_posterior_rows.extend(
                    build_forecast_posterior_rows(
                        case_name=case_name,
                        forecast_mode=mode,
                        labels=labels,
                        update_result=update,
                        theta_true=theta_true,
                        prior_sigma=prior_sigma,
                        n_subblocks=count,
                        trial_index=trial_index,
                    )
                )
                if trial_index == 0:
                    information_rows.append(
                        build_forecast_information_row(
                            case_name=case_name,
                            forecast_mode=mode,
                            labels=labels,
                            summaries=prefix,
                            n_subblocks=count,
                            actual_n_summaries=len(summaries),
                            subblock_duration_s=subblock_duration_s,
                            single_observation_n_subblocks=single_observation_n_subblocks,
                        )
                    )
        aggregate_rows: list[dict[str, Any]] = []
        for count in grid:
            subset = [row for row in trial_rows if int(row["n_subblocks"]) == int(count)]
            errors = np.asarray(
                [float(row["separation_posterior_error_microas"]) for row in subset],
                dtype=float,
            )
            sigmas = np.asarray(
                [float(row["separation_posterior_sigma_microas"]) for row in subset],
                dtype=float,
            )
            base = dict(subset[0])
            base.pop("trial_index", None)
            base.update(
                {
                    "n_trials": int(n_trials),
                    "separation_error_mean_microas": float(np.mean(errors)),
                    "separation_error_std_microas": float(np.std(errors, ddof=0)),
                    "separation_error_rms_microas": float(np.sqrt(np.mean(np.square(errors)))),
                    "separation_error_p16_microas": float(np.percentile(errors, 16)),
                    "separation_error_p50_microas": float(np.percentile(errors, 50)),
                    "separation_error_p84_microas": float(np.percentile(errors, 84)),
                    "separation_sigma_mean_microas": float(np.mean(sigmas)),
                    "separation_sigma_p16_microas": float(np.percentile(sigmas, 16)),
                    "separation_sigma_p50_microas": float(np.percentile(sigmas, 50)),
                    "separation_sigma_p84_microas": float(np.percentile(sigmas, 84)),
                }
            )
            aggregate_rows.append(base)
        _write_csv_rows(mode_root / "forecast_results.csv", aggregate_rows)
        _write_csv_rows(mode_root / "trial_forecast_results.csv", trial_rows)
        _write_csv_rows(mode_root / "trial_posterior_by_label.csv", trial_posterior_rows)
        _write_csv_rows(mode_root / "information_diagnostics.csv", information_rows)
        _write_csv_rows(mode_root / "stochastic_synthesis_diagnostics.csv", synthesis_rows)
        _write_forecast_manifest(
            path=mode_root / "manifest.json",
            case_name=case_name,
            mode=mode,
            summaries=summaries,
            summary_paths=summary_paths,
            prior=prior,
            prior_sigma=prior_sigma,
            theta_true=theta_true,
            forecast_cfg=cfg,
            grid=grid,
            extra={
                "score_noise_settings": dict(noise_cfg),
                "theta_reference": np.asarray(theta_reference, dtype=float).tolist(),
            },
        )
        if bool(cfg.get("plots", True)):
            _plot_forecast_line(
                path=mode_root / "separation_sigma_vs_n_subblocks.png",
                rows=aggregate_rows,
                y_key="separation_sigma_p50_microas",
                ylabel="Median separation sigma (microas)",
                actual_n_summaries=len(summaries),
                single_observation_n_subblocks=single_observation_n_subblocks,
            )
            _plot_forecast_line(
                path=mode_root / "separation_error_vs_n_subblocks.png",
                rows=aggregate_rows,
                y_key="separation_error_rms_microas",
                ylabel="Separation error RMS (microas)",
                actual_n_summaries=len(summaries),
                single_observation_n_subblocks=single_observation_n_subblocks,
            )
        all_results.extend(aggregate_rows)
        all_trial_results.extend(trial_rows)
        all_trial_posterior_rows.extend(trial_posterior_rows)
        all_information_rows.extend(information_rows)

    summary = {
        "enabled": True,
        "case_name": case_name,
        "forecast_grid": list(grid),
        "modes": list(modes),
        "forecast_root": str(forecast_root),
        "n_actual_summaries": int(len(summaries)),
        "single_observation_n_subblocks": int(single_observation_n_subblocks),
        "subblock_duration_s": float(subblock_duration_s),
    }
    _write_json(forecast_root / "forecast_summary.json", summary)
    return {
        "forecast_results": all_results,
        "forecast_trial_results": all_trial_results,
        "forecast_trial_posterior_rows": all_trial_posterior_rows,
        "forecast_information_diagnostics": all_information_rows,
        "forecast_summary": summary,
    }


def _observation_update_policy(plan: CampaignPlan) -> ObservationUpdatePolicy:
    return ObservationUpdatePolicy(**dict(plan.iterative.get("update_policy", {})))


def _write_eigen_update_artifacts(
    *,
    case_root: Path,
    labels: Sequence[str],
    result: Any,
) -> None:
    diagnostics = dict(result.diagnostics)
    diagnostics["physical_update_full_by_label"] = {
        label: float(result.physical_update_full[index])
        for index, label in enumerate(labels)
    }
    diagnostics["physical_update_applied_by_label"] = {
        label: float(result.physical_update_applied[index])
        for index, label in enumerate(labels)
    }
    _write_json(case_root / "eigen_update_diagnostics.json", diagnostics)
    _write_csv_rows(case_root / "eigen_update_modes.csv", result.mode_rows)


def aggregate_case(
    *,
    plan: CampaignPlan,
    case: BiasCase,
    prior_source: str,
    summary_scale_policy: str = SUMMARY_SCALE_POLICY_REQUIRE_SUMMED,
) -> dict[str, Any]:
    case_root = plan.run_root / "cases" / case.case_name
    _ensure_dir(case_root)
    summary_paths = plan.summary_paths[case.case_name]
    summaries, summary_scale_validation = _load_case_summaries(
        summary_paths,
        summary_scale_policy=summary_scale_policy,
    )
    summary_reference_mean = np.asarray(summaries[0].theta_ref, dtype=float)
    use_case_prior = bool(case.prior_sigma_by_label)
    if use_case_prior:
        prior_mean = summary_reference_mean.copy()
        prior_mean_source = (
            "prior_draw_theta_ref"
            if case.case_origin == "prior_draw"
            else "case_specific_theta_ref"
        )
        prior_mean_provenance = {
            "policy": "case_specific_prior",
            "case_origin": case.case_origin,
            "summary_paths": [str(path) for path in summary_paths],
        }
        if case.prior_draw_metadata is not None:
            prior_mean_provenance["prior_draw_metadata"] = dict(case.prior_draw_metadata)
        prior_warnings: list[str] = []
    else:
        prior_context = resolve_prior_context_for_summaries(
            summaries,
            summary_paths=summary_paths,
            prior_source=prior_source,
            allow_summary_theta_ref_default=True,
        )
        prior_mean = np.asarray(prior_context.prior_mean, dtype=float)
        prior_mean_source = prior_context.prior_mean_source
        prior_mean_provenance = dict(prior_context.provenance)
        prior_warnings = list(prior_context.warnings)
    if case.prior_sigma_by_label:
        sigma_defaults = build_default_prior_sigma(plan.layout.labels)
        prior_sigma = np.asarray(sigma_defaults, dtype=float)
        for i, label in enumerate(plan.layout.labels):
            if label in case.prior_sigma_by_label:
                prior_sigma[i] = float(case.prior_sigma_by_label[label])
        if np.any(prior_sigma <= 0.0) or not np.all(np.isfinite(prior_sigma)):
            raise ValueError(
                f"Case {case.case_name!r} resolved non-positive or non-finite prior sigma."
            )
        prior_sigma_source = (
            "prior_draw_config" if case.case_origin == "prior_draw" else "case_prior_sigmas"
        )
    else:
        prior_sigma = build_default_prior_sigma(plan.layout.labels)
        prior_sigma_source = "build_default_prior_sigma"
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=plan.layout.labels,
        mean=prior_mean,
        sigma=prior_sigma,
        metadata={
            "prior_mean_source": prior_mean_source,
            "prior_mean_provenance": prior_mean_provenance,
            "prior_sigma_source": prior_sigma_source,
            "case_origin": case.case_origin,
        },
    )
    update_policy = _observation_update_policy(plan)
    update = update_observation_belief_with_policy(
        prior,
        summaries,
        policy=update_policy,
    )
    _write_eigen_update_artifacts(
        case_root=case_root,
        labels=plan.layout.labels,
        result=update,
    )
    accumulated = accumulate_summary_information(plan.layout.labels, summaries)
    posterior_sigma = update.posterior.sigma()
    truth, reference, offsets = _truth_reference_maps(
        plan=plan,
        case=case,
        summaries=summaries,
        summary_paths=summary_paths,
    )
    posterior_rows = _posterior_rows(
        case_name=case.case_name,
        labels=plan.layout.labels,
        truth=truth,
        reference=reference,
        offsets=offsets,
        posterior_mean=update.posterior.mean,
        posterior_sigma=posterior_sigma,
    )
    science = _science_row(
        case_name=case.case_name,
        labels=plan.layout.labels,
        truth=truth,
        reference=reference,
        posterior_mean=update.posterior.mean,
        posterior_sigma=posterior_sigma,
    )
    _write_csv_rows(
        case_root / "summary_paths.csv",
        [{"case_name": case.case_name, "summary_path": str(path)} for path in summary_paths],
    )
    _write_csv_rows(case_root / "posterior_by_label.csv", posterior_rows)
    _write_csv_rows(case_root / "science_summary.csv", [science])
    eigen_cfg = dict(plan.config["experiment"].get("eigenbasis", {}) or {})
    eigen_sources = _resolve_eigen_sources(eigen_cfg)
    eigenvalue_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    contributor_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    projection_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    weak_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    eigen_summary_by_source: dict[str, dict[str, Any]] = {}
    if bool(eigen_cfg.get("enabled", True)):
        reference_error = np.asarray([reference[label] - truth[label] for label in plan.layout.labels], dtype=float)
        posterior_error = np.asarray([update.posterior.mean[index] - truth[label] for index, label in enumerate(plan.layout.labels)], dtype=float)
        posterior_shift = np.asarray([update.posterior.mean[index] - reference[label] for index, label in enumerate(plan.layout.labels)], dtype=float)
        for source_name in eigen_sources:
            matrix = (
                update.posterior.precision
                if source_name == "posterior_precision"
                else accumulated
            )
            (
                eigen_rows,
                contributor_rows,
                projection_rows,
                weak_rows,
                eigen_summary,
                _,
            ) = _eigen_rows(
                case_name=case.case_name,
                source_name=source_name,
                labels=plan.layout.labels,
                matrix=matrix,
                prior_sigma=prior_sigma,
                reference_error=reference_error,
                posterior_error=posterior_error,
                posterior_shift=posterior_shift,
                eigen_cfg=eigen_cfg,
            )
            eigenvalue_rows_by_source[source_name] = eigen_rows
            contributor_rows_by_source[source_name] = contributor_rows
            projection_rows_by_source[source_name] = projection_rows
            weak_rows_by_source[source_name] = weak_rows
            eigen_summary_by_source[source_name] = eigen_summary
            if len(eigen_sources) == 1:
                _write_csv_rows(case_root / "eigenvalues.csv", eigen_rows)
                _write_csv_rows(case_root / "eigenmode_contributors.csv", contributor_rows)
                _write_csv_rows(case_root / "eigenmode_projection.csv", projection_rows)
                _write_csv_rows(case_root / "weak_mode_summary.csv", weak_rows)
                _write_json(case_root / "eigenbasis_summary.json", eigen_summary)
            _write_csv_rows(
                case_root / f"eigenvalues_{source_name}.csv",
                eigen_rows,
            )
            _write_csv_rows(
                case_root / f"eigenmode_contributors_{source_name}.csv",
                contributor_rows,
            )
            _write_csv_rows(
                case_root / f"eigenmode_projection_{source_name}.csv",
                projection_rows,
            )
            _write_csv_rows(
                case_root / f"weak_mode_summary_{source_name}.csv",
                weak_rows,
            )
            _write_json(
                case_root / f"eigenbasis_summary_{source_name}.json",
                eigen_summary,
            )
            _plot_eigenvalue_spectrum(
                case_root / f"eigenvalue_spectrum_{source_name}.png",
                eigen_rows,
            )
        _plot_group_error(case_root / f"posterior_error_over_sigma_by_group_{_slugify(case.case_name)}.png", posterior_rows)
    matrix_diagnostics = {
        "accumulated_information": _matrix_diagnostics(accumulated).to_dict(),
        "posterior_precision": _matrix_diagnostics(update.posterior.precision).to_dict(),
        "prior_precision": _matrix_diagnostics(prior.precision).to_dict(),
    }
    _write_json(case_root / "matrix_diagnostics.json", matrix_diagnostics)
    update_summary = {
        "case_name": case.case_name,
        "n_summaries": len(summaries),
        "prior": prior.to_dict(),
        "posterior": update.posterior.to_dict(),
        "update_policy": dict(update.diagnostics),
        "posterior_full": update.posterior_full.to_dict(),
        "prior_context": {
            "prior_mean_source": prior_mean_source,
            "provenance": prior_mean_provenance,
            "warnings": prior_warnings,
            "prior_sigma_source": prior_sigma_source,
        },
        "summary_scale_validation": summary_scale_validation,
    }
    _write_json(case_root / "observation_update_summary.json", update_summary)
    case_manifest = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "case_name": case.case_name,
        "case_root": str(case_root),
        "case_origin": case.case_origin,
        "seeding": _resolve_seeding_config(plan.config["experiment"]),
        "case_generation": dict(plan.case_generation),
        "subblock_command_options": resolve_subblock_command_options(
            plan.config["experiment"].get("subblocks", {}) or {}
        ),
        "theta_reference_offsets": dict(case.theta_reference_offsets),
        "case_prior_sigma_by_label": (
            None if case.prior_sigma_by_label is None else dict(case.prior_sigma_by_label)
        ),
        "prior_draw_metadata": (
            None if case.prior_draw_metadata is None else dict(case.prior_draw_metadata)
        ),
        "prior_draw_rows": list(plan.prior_draw_rows_by_case.get(case.case_name, [])),
        "theta_layout": plan.layout.to_dict(),
        "layout_metadata": plan.layout_metadata,
        "state_partition": plan.partition,
        "summary_paths": [str(path) for path in summary_paths],
        "summary_scale_validation": summary_scale_validation,
        "subblock_plan": list(plan.subblock_plans.get(case.case_name, [])),
        "planned_commands": [" ".join(command) for command in plan.subblock_commands[case.case_name]],
        "outputs": {
            "posterior_by_label_csv": str((case_root / "posterior_by_label.csv").resolve()),
            "science_summary_csv": str((case_root / "science_summary.csv").resolve()),
            "eigen_sources": list(eigen_sources),
            "subblock_plan_csv": str((case_root / "subblock_plan.csv").resolve()),
        },
    }
    _write_csv_rows(
        case_root / "subblock_plan.csv",
        list(plan.subblock_plans.get(case.case_name, [])),
    )
    _write_csv_rows(
        case_root / "prior_draws.csv",
        list(plan.prior_draw_rows_by_case.get(case.case_name, [])),
    )
    _write_json(case_root / "case_manifest.json", case_manifest)
    theta_true_vector = np.asarray([truth[label] for label in plan.layout.labels], dtype=float)
    theta_reference_vector = np.asarray(
        [reference[label] for label in plan.layout.labels],
        dtype=float,
    )
    forecast_payload = run_case_forecast(
        case_name=case.case_name,
        case_root=case_root,
        summaries=summaries,
        summary_paths=summary_paths,
        prior=prior,
        prior_sigma=prior_sigma,
        theta_true=theta_true_vector,
        theta_reference=theta_reference_vector,
        labels=plan.layout.labels,
        forecast_cfg=plan.config["experiment"].get("forecast", {}) or {},
    )
    return {
        "case_name": case.case_name,
        "case_root": str(case_root),
        "posterior_rows": posterior_rows,
        "science": science,
        "eigen_sources": list(eigen_sources),
        "eigenvalues_by_source": eigenvalue_rows_by_source,
        "eigenmode_contributors_by_source": contributor_rows_by_source,
        "eigenmode_projection_by_source": projection_rows_by_source,
        "weak_mode_summary_by_source": weak_rows_by_source,
        "matrix_diagnostics": matrix_diagnostics,
        "eigenbasis_summary_by_source": eigen_summary_by_source,
        "forecast_results": forecast_payload["forecast_results"],
        "forecast_trial_results": forecast_payload["forecast_trial_results"],
        "forecast_information_diagnostics": forecast_payload[
            "forecast_information_diagnostics"
        ],
        "forecast_summary": forecast_payload["forecast_summary"],
        "summary_scale_validation": summary_scale_validation,
    }


def _plot_campaign_bar(path: Path, rows: Sequence[Mapping[str, Any]], *, key: str, ylabel: str) -> None:
    if not _HAVE_MATPLOTLIB or plt is None:
        return
    if not rows:
        return
    names = [str(row["case_name"]) for row in rows]
    values = [float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 5))
    ax.bar(names, values)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _campaign_forecast_summary(
    *,
    forecast_rows: Sequence[Mapping[str, Any]],
    forecast_trial_rows: Sequence[Mapping[str, Any]],
    forecast_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = _forecast_config({"forecast": forecast_cfg})
    target = int(cfg.get("single_observation_n_subblocks", 1800))
    target_rows = [row for row in forecast_rows if int(row.get("n_subblocks", -1)) == target]
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "single_observation_n_subblocks": int(target),
        "subblock_duration_s": float(cfg.get("subblock_duration_s", 1.0)),
        "single_observation_target_rows": [
            {
                "case_name": row.get("case_name"),
                "forecast_mode": row.get("forecast_mode"),
                "n_subblocks": int(row.get("n_subblocks", target)),
                "equivalent_duration_min": float(row.get("equivalent_duration_min", 0.0)),
                "n_actual_summaries": int(row.get("n_actual_summaries", 0)),
                "separation_posterior_sigma_microas": row.get(
                    "separation_posterior_sigma_microas"
                ),
                "separation_posterior_error_microas": row.get(
                    "separation_posterior_error_microas"
                ),
                "separation_error_rms_microas": row.get(
                    "separation_error_rms_microas"
                ),
                "prior_sigma_source": row.get("prior_sigma_source"),
            }
            for row in target_rows
        ],
        "n_forecast_rows": int(len(forecast_rows)),
        "n_trial_rows": int(len(forecast_trial_rows)),
    }


def _plot_forecast_campaign_bars(
    *,
    run_root: Path,
    forecast_rows: Sequence[Mapping[str, Any]],
    forecast_cfg: Mapping[str, Any],
) -> None:
    if not _HAVE_MATPLOTLIB or plt is None or not forecast_rows:
        return
    cfg = _forecast_config({"forecast": forecast_cfg})
    target = int(cfg.get("single_observation_n_subblocks", 1800))
    rows = [row for row in forecast_rows if int(row.get("n_subblocks", -1)) == target]
    if not rows:
        return

    def _bar(path: Path, key_candidates: Sequence[str], ylabel: str) -> None:
        labels: list[str] = []
        values: list[float] = []
        for row in rows:
            key = next((candidate for candidate in key_candidates if candidate in row), None)
            if key is None:
                continue
            value = row.get(key)
            if value is None:
                continue
            labels.append(f"{row['case_name']}\n{row['forecast_mode']}")
            values.append(float(value))
        if not values:
            return
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(values)), 5))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    _bar(
        run_root / "separation_sigma_forecast_by_case.png",
        ("separation_sigma_p50_microas", "separation_posterior_sigma_microas"),
        "Separation sigma at target (microas)",
    )
    _bar(
        run_root / "separation_error_forecast_by_case.png",
        ("separation_error_rms_microas", "separation_posterior_error_microas"),
        "Separation error at target (microas)",
    )


def aggregate_campaign(
    plan: CampaignPlan,
    *,
    prior_source: str,
    allow_optimizer_scale_summaries: bool = False,
) -> dict[str, Any]:
    summary_scale_policy = (
        SUMMARY_SCALE_POLICY_ALLOW_OPTIMIZER
        if allow_optimizer_scale_summaries
        else SUMMARY_SCALE_POLICY_REQUIRE_SUMMED
    )
    all_posterior_rows: list[dict[str, Any]] = []
    all_science_rows: list[dict[str, Any]] = []
    all_eigenvalues_by_source: dict[str, list[dict[str, Any]]] = {}
    all_contributors_by_source: dict[str, list[dict[str, Any]]] = {}
    all_projection_by_source: dict[str, list[dict[str, Any]]] = {}
    all_weak_by_source: dict[str, list[dict[str, Any]]] = {}
    all_forecast_rows: list[dict[str, Any]] = []
    all_forecast_trial_rows: list[dict[str, Any]] = []
    all_forecast_information_rows: list[dict[str, Any]] = []
    all_subblock_plan_rows = _flatten_subblock_plan_rows(plan)
    case_payloads: list[dict[str, Any]] = []
    for case in plan.cases:
        payload = aggregate_case(
            plan=plan,
            case=case,
            prior_source=prior_source,
            summary_scale_policy=summary_scale_policy,
        )
        case_payloads.append(payload)
        all_posterior_rows.extend(payload["posterior_rows"])
        all_science_rows.append(payload["science"])
        for source in payload["eigen_sources"]:
            all_eigenvalues_by_source.setdefault(source, []).extend(
                payload["eigenvalues_by_source"].get(source, [])
            )
            all_contributors_by_source.setdefault(source, []).extend(
                payload["eigenmode_contributors_by_source"].get(source, [])
            )
            all_projection_by_source.setdefault(source, []).extend(
                payload["eigenmode_projection_by_source"].get(source, [])
            )
            all_weak_by_source.setdefault(source, []).extend(
                payload["weak_mode_summary_by_source"].get(source, [])
            )
        all_forecast_rows.extend(payload.get("forecast_results", []))
        all_forecast_trial_rows.extend(payload.get("forecast_trial_results", []))
        all_forecast_information_rows.extend(
            payload.get("forecast_information_diagnostics", [])
        )
    _write_csv_rows(plan.run_root / "posterior_by_label.csv", all_posterior_rows)
    _write_csv_rows(plan.run_root / "science_summary.csv", all_science_rows)
    _write_csv_rows(plan.run_root / "subblock_plan.csv", all_subblock_plan_rows)
    _write_csv_rows(plan.run_root / "trajectory" / "smear_summary.csv", _smear_summary_rows_for_plan(plan, strict=True))
    _write_csv_rows(plan.run_root / "forecast_results.csv", all_forecast_rows)
    _write_csv_rows(plan.run_root / "forecast_trial_results.csv", all_forecast_trial_rows)
    _write_csv_rows(
        plan.run_root / "forecast_information_diagnostics.csv",
        all_forecast_information_rows,
    )
    sources = _resolve_eigen_sources(plan.config["experiment"].get("eigenbasis", {}) or {})
    for source in sources:
        eigen_rows = all_eigenvalues_by_source.get(source, [])
        contributor_rows = all_contributors_by_source.get(source, [])
        projection_rows = all_projection_by_source.get(source, [])
        weak_rows = all_weak_by_source.get(source, [])
        _write_csv_rows(plan.run_root / f"eigenvalues_{source}.csv", eigen_rows)
        _write_csv_rows(
            plan.run_root / f"eigenmode_contributors_{source}.csv",
            contributor_rows,
        )
        _write_csv_rows(
            plan.run_root / f"eigenmode_projection_{source}.csv",
            projection_rows,
        )
        _write_csv_rows(plan.run_root / f"weak_mode_summary_{source}.csv", weak_rows)
        if len(sources) == 1:
            _write_csv_rows(plan.run_root / "eigenvalues.csv", eigen_rows)
            _write_csv_rows(
                plan.run_root / "eigenmode_contributors.csv",
                contributor_rows,
            )
            _write_csv_rows(
                plan.run_root / "eigenmode_projection.csv",
                projection_rows,
            )
            _write_csv_rows(plan.run_root / "weak_mode_summary.csv", weak_rows)
    _plot_campaign_bar(
        plan.run_root / "separation_error_by_case.png",
        all_science_rows,
        key="posterior_separation_error_microas",
        ylabel="Posterior separation error (microas)",
    )
    _plot_campaign_bar(
        plan.run_root / "separation_sigma_by_case.png",
        all_science_rows,
        key="posterior_separation_sigma_microas",
        ylabel="Posterior separation sigma (microas)",
    )
    _plot_forecast_campaign_bars(
        run_root=plan.run_root,
        forecast_rows=all_forecast_rows,
        forecast_cfg=plan.config["experiment"].get("forecast", {}) or {},
    )
    forecast_summary = _campaign_forecast_summary(
        forecast_rows=all_forecast_rows,
        forecast_trial_rows=all_forecast_trial_rows,
        forecast_cfg=plan.config["experiment"].get("forecast", {}) or {},
    )
    _write_json(plan.run_root / "forecast_summary.json", forecast_summary)
    summary = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "run_root": str(plan.run_root),
        "n_cases": len(plan.cases),
        "n_theta": plan.layout.size,
        "eigen_sources": list(sources),
        "truth_realization": dict(plan.truth_realization),
        "forecast": forecast_summary,
        "summary_scale_policy": summary_scale_policy,
        "case_outputs": [
            {
                "case_name": payload["case_name"],
                "case_root": payload["case_root"],
                "science": payload["science"],
                "summary_scale_validation": payload["summary_scale_validation"],
            }
            for payload in case_payloads
        ],
    }
    _write_json(plan.run_root / "campaign_summary.json", summary)
    return summary


def _truth_by_label(plan: CampaignPlan) -> dict[str, float]:
    return {
        label: float(plan.prior_truth[index])
        for index, label in enumerate(plan.layout.labels)
    }


def _rows_for_iterative_window(
    plan: CampaignPlan,
    *,
    case_name: str,
    window_index: int,
) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in plan.iterative_plan_rows
        if str(row.get("case_name")) == str(case_name)
        and int(row.get("window_index", -1)) == int(window_index)
    ]


def _window_plan(
    plan: CampaignPlan,
    *,
    case: BiasCase,
    window_index: int,
    current_offsets: Mapping[str, float],
) -> tuple[CampaignPlan, BiasCase]:
    rows = sorted(
        _rows_for_iterative_window(plan, case_name=case.case_name, window_index=window_index),
        key=lambda row: int(row["window_subblock_index"]),
    )
    if not rows:
        raise RuntimeError(
            f"Stored iterative plan has no rows for case={case.case_name!r} "
            f"window={window_index}."
        )
    subblock_cfg = dict(plan.config["experiment"].get("subblocks", {}) or {})
    commands: list[list[str]] = []
    summary_paths: list[Path] = []
    subblock_rows: list[dict[str, Any]] = []
    subblock_root = plan.run_root / "subblock_runs"
    for row in rows:
        global_index = int(row["global_subblock_index"])
        summary_path = Path(str(row["summary_path"]))
        summary_paths.append(summary_path)
        command = build_subblock_command(
            case_root_parent=subblock_root,
            case_subblock_name=str(row["subblock_name"]),
            theta_labels=plan.layout.labels,
            offsets=current_offsets,
            subblock_cfg=subblock_cfg,
            trace_seed=int(row["trace_seed"]),
            noise_seed=int(row["noise_seed"]),
            trace_subblock=plan.trace_source_plan.subblocks[global_index],
            template_paths={
                "trace": Path(str(row["trace_template_path"])),
                "render": Path(str(row["render_template_path"])),
                "inference": Path(str(row["inference_template_path"])),
            },
        )
        commands.append(command)
        subblock_rows.append(
            {
                **row,
                "theta_reference_offsets_realized_json": json.dumps(
                    {str(k): float(v) for k, v in current_offsets.items()},
                    sort_keys=True,
                ),
            }
        )
    window_case = BiasCase(
        case_name=_iterative_window_case_name(case.case_name, window_index),
        theta_reference_offsets={str(k): float(v) for k, v in current_offsets.items()},
        case_origin=case.case_origin,
        prior_sigma_by_label=case.prior_sigma_by_label,
        prior_draw_metadata=case.prior_draw_metadata,
    )
    window_plan_obj = replace(
        plan,
        cases=(window_case,),
        subblock_commands={window_case.case_name: commands},
        summary_paths={window_case.case_name: summary_paths},
        subblock_plans={window_case.case_name: subblock_rows},
    )
    return window_plan_obj, window_case


def _write_realized_window_commands(
    *,
    window_plan_obj: CampaignPlan,
    window_case: BiasCase,
) -> list[dict[str, Any]]:
    command_rows: list[dict[str, Any]] = []
    rows = list(window_plan_obj.subblock_plans[window_case.case_name])
    commands = list(window_plan_obj.subblock_commands[window_case.case_name])
    for row, command in zip(rows, commands, strict=True):
        command_path = Path(str(row.get("realized_command_path", "")))
        if not str(command_path):
            command_path = (
                window_plan_obj.run_root
                / "cases"
                / window_case.case_name
                / "commands"
                / f"subblock_{int(row.get('window_subblock_index', row.get('subblock_index', 0))):03d}.sh"
            )
        write_shell_command(command_path, command)
        command_rows.append(
            {
                "case_name": row.get("case_name", ""),
                "window_case_name": window_case.case_name,
                "window_index": row.get("window_index", ""),
                "window_subblock_index": row.get("window_subblock_index", ""),
                "global_subblock_index": row.get("global_subblock_index", row.get("subblock_index", "")),
                "summary_path": row.get("summary_path", ""),
                "command_path": str(command_path),
                "command": format_shell_command(command),
            }
        )
    _write_csv_rows(
        window_plan_obj.run_root / "cases" / window_case.case_name / "commands" / "commands.csv",
        command_rows,
    )
    return command_rows


def _augment_iterative_status_rows(
    *,
    plan: CampaignPlan,
    window_case: BiasCase,
    command_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    command_by_summary = {str(row.get("summary_path", "")): row for row in command_rows}
    status_rows = _read_csv_rows(plan.run_root / "subblock_status.csv")
    out: list[dict[str, Any]] = []
    for status in status_rows:
        summary_path = str(status.get("summary_path", ""))
        command_row = command_by_summary.get(summary_path, {})
        if not command_row:
            continue
        row = {
            **status,
            "case_name": command_row.get("case_name", ""),
            "window_case_name": window_case.case_name,
            "window_index": command_row.get("window_index", ""),
            "window_subblock_index": command_row.get("window_subblock_index", ""),
            "global_subblock_index": command_row.get("global_subblock_index", ""),
            "command_path": command_row.get("command_path", ""),
            "failure_reason": status.get("failure_hint", ""),
            "elapsed_seconds": "",
        }
        diag_path = Path(str(status.get("subprocess_diagnostics_path", "")))
        if diag_path.exists():
            try:
                diag_payload = json.loads(diag_path.read_text(encoding="utf-8"))
                row["elapsed_seconds"] = diag_payload.get("elapsed_seconds", "")
            except Exception:
                pass
        out.append(row)
    if out:
        _write_csv_rows(
            plan.run_root / "cases" / window_case.case_name / "subblock_status.csv",
            out,
        )
    return out


def _norm_for_prefix(labels: Sequence[str], offsets: Mapping[str, float], prefix: str) -> float:
    values = [float(offsets.get(label, 0.0)) for label in labels if label.startswith(prefix)]
    return float(np.linalg.norm(np.asarray(values, dtype=float))) if values else float("nan")


def _binary_iterative_diagnostic_row(
    *,
    plan: CampaignPlan,
    case: BiasCase,
    window_index: int,
    current_offsets: Mapping[str, float],
    posterior_offsets: Mapping[str, float],
    next_offsets: Mapping[str, float],
    posterior_rows: Mapping[str, Mapping[str, Any]],
    posterior_offset_status: Mapping[str, str],
    previous_residual_norm: float | None,
    previous_next_reference_norm: float | None,
    n_subblocks: int,
    eigen_update_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    labels = tuple(plan.layout.labels)
    base = vector_update_diagnostics(
        labels=labels,
        current_offsets=current_offsets,
        posterior_offsets=posterior_offsets,
        next_offsets=next_offsets,
        previous_residual_norm=previous_residual_norm,
        previous_next_reference_norm=previous_next_reference_norm,
    )
    sep_label = "source.separation_as"
    sep_sigma = float("nan")
    if sep_label in posterior_rows:
        sep_sigma = posterior_float(
            posterior_rows[sep_label],
            ("posterior_sigma", "sigma", "std"),
        )
    missing_labels = [
        label for label in labels if posterior_offset_status.get(label) != "ok"
    ]
    eigen_diag = dict(eigen_update_diagnostics or {})
    relative = np.asarray(eigen_diag.get("eigenvalue_relative", []), dtype=float)
    kept = np.asarray(eigen_diag.get("kept_mode_mask", []), dtype=bool)
    min_kept = (
        float(np.min(relative[kept]))
        if relative.size and kept.size == relative.size and np.any(kept)
        else float("nan")
    )
    max_rejected = (
        float(np.max(relative[~kept]))
        if relative.size and kept.size == relative.size and np.any(~kept)
        else float("nan")
    )
    physical_full = np.asarray(eigen_diag.get("physical_update_full", []), dtype=float)
    physical_applied = np.asarray(
        eigen_diag.get("physical_update_applied", []),
        dtype=float,
    )
    physical_after_safety = np.asarray(
        [
            float(next_offsets.get(label, 0.0))
            - float(current_offsets.get(label, 0.0))
            for label in labels
        ],
        dtype=float,
    )
    eigen_full = np.asarray(eigen_diag.get("eigen_update_full", []), dtype=float)
    eigen_applied = np.asarray(eigen_diag.get("eigen_update_applied", []), dtype=float)
    return {
        "case_name": case.case_name,
        "case_origin": case.case_origin,
        "draw_index": _case_draw_index(case),
        "window_index": int(window_index),
        "update_gain": float(plan.iterative["update_gain"]),
        "update_mode": str(plan.iterative["update_mode"]),
        "eigen_basis_source": eigen_diag.get(
            "basis_source",
            plan.iterative.get("eigenbasis", {}).get("basis_source", ""),
        ),
        "eigen_gate_source": eigen_diag.get(
            "gate_source",
            plan.iterative.get("eigenbasis", {}).get("gate_source", ""),
        ),
        "eigen_whiten": eigen_diag.get(
            "whiten",
            plan.iterative.get("eigenbasis", {}).get("whiten", False),
        ),
        "eigen_n_modes_total": int(eigen_diag.get("n_modes_total", 0)),
        "eigen_n_modes_kept": int(eigen_diag.get("n_modes_kept", 0)),
        "eigen_min_kept_eigenvalue_rel": min_kept,
        "eigen_max_rejected_eigenvalue_rel": max_rejected,
        "physical_update_norm_full": float(np.linalg.norm(physical_full)),
        "physical_update_norm_applied": float(
            np.linalg.norm(physical_after_safety)
            if physical_after_safety.size
            else np.linalg.norm(physical_applied)
        ),
        "eigen_update_norm_full": (
            float(np.linalg.norm(eigen_full)) if eigen_full.size else float("nan")
        ),
        "eigen_update_norm_applied": (
            float(np.linalg.norm(eigen_applied))
            if eigen_applied.size
            else float("nan")
        ),
        "n_subblocks": int(n_subblocks),
        **base,
        **separation_update_diagnostics(
            current_offsets=current_offsets,
            posterior_offsets=posterior_offsets,
            next_offsets=next_offsets,
        ),
        "posterior_sigma_separation_microas": sep_sigma * 1.0e6
        if math.isfinite(sep_sigma)
        else float("nan"),
        "posterior_missing_label_count": int(len(missing_labels)),
        "posterior_missing_labels": ",".join(missing_labels),
        "posterior_offset_status_json": json.dumps(dict(posterior_offset_status), sort_keys=True),
        "source_scalar_reference_error_norm_before": _norm_for_prefix(labels, current_offsets, "source."),
        "source_scalar_posterior_error_norm_after": _norm_for_prefix(labels, posterior_offsets, "source."),
        "plate_scale_reference_error_norm_before": abs(
            float(current_offsets.get("optics.plate_scale_as_per_pix", 0.0))
        )
        if "optics.plate_scale_as_per_pix" in labels
        else float("nan"),
        "plate_scale_posterior_error_norm_after": abs(
            float(posterior_offsets.get("optics.plate_scale_as_per_pix", 0.0))
        )
        if "optics.plate_scale_as_per_pix" in labels
        else float("nan"),
        "m1_zernike_reference_error_norm_before": _norm_for_prefix(
            labels, current_offsets, "optics.primary.zernike_coeffs_nm"
        ),
        "m1_zernike_posterior_error_norm_after": _norm_for_prefix(
            labels, posterior_offsets, "optics.primary.zernike_coeffs_nm"
        ),
        "m2_zernike_reference_error_norm_before": _norm_for_prefix(
            labels, current_offsets, "optics.secondary.zernike_coeffs_nm"
        ),
        "m2_zernike_posterior_error_norm_after": _norm_for_prefix(
            labels, posterior_offsets, "optics.secondary.zernike_coeffs_nm"
        ),
    }


def _apply_iterative_update_safety(
    *,
    labels: Sequence[str],
    current_offsets: Mapping[str, float],
    proposed_offsets: Mapping[str, float],
    safety_cfg: Mapping[str, Any],
) -> dict[str, float]:
    next_offsets = {
        str(label): float(proposed_offsets.get(label, current_offsets.get(label, 0.0)))
        for label in labels
    }
    if not bool(safety_cfg.get("enabled", False)):
        return next_offsets
    limits = dict(safety_cfg.get("max_abs_update_by_label", {}) or {})
    for label in labels:
        raw_limit = limits.get(label)
        if raw_limit is None:
            continue
        limit = float(raw_limit)
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError(
                "experiment.iterative.update_safety.max_abs_update_by_label "
                f"contains invalid limit for {label!r}."
            )
        current = float(current_offsets.get(label, 0.0))
        delta = float(next_offsets[label] - current)
        next_offsets[label] = current + float(np.clip(delta, -limit, limit))
    return next_offsets


def _resolved_next_offsets(
    *,
    plan: CampaignPlan,
    current_offsets: Mapping[str, float],
    posterior_offsets: Mapping[str, float],
    posterior_rows: Mapping[str, Mapping[str, Any]],
    posterior_truth: Mapping[str, float],
    eigen_update_diagnostics: Mapping[str, Any],
) -> dict[str, float]:
    if eigen_update_diagnostics:
        proposed = dict(posterior_offsets)
    elif str(plan.iterative.get("update_mode", "physical_full")) == "physical_full":
        proposed = apply_physical_reference_update(
            current_offsets=current_offsets,
            posterior_rows_by_label=posterior_rows,
            truth_by_label=posterior_truth,
            update_gain=float(plan.iterative["update_gain"]),
        )
    else:
        raise RuntimeError(
            "Eigen update mode requires eigen_update_diagnostics.json for "
            "resume or aggregate-only processing."
        )
    return _apply_iterative_update_safety(
        labels=plan.layout.labels,
        current_offsets=current_offsets,
        proposed_offsets=proposed,
        safety_cfg=plan.iterative.get("update_safety", {}),
    )


def _scalar_nested_value(payload: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if math.isfinite(scalar) else None


def _summary_exposure_context_values(summary_paths: Sequence[Path]) -> list[float]:
    values: list[float] = []
    candidate_paths = (
        ("metadata", "prior_context", "effective_store_values", "source.exposure_time_s"),
        ("prior_context", "effective_store_values", "source.exposure_time_s"),
        ("metadata", "system", "resolved_config", "source", "exposure_time_s"),
        ("system", "resolved_config", "source", "exposure_time_s"),
    )
    for path in summary_paths:
        try:
            payload = load_subblock_summary_artifact_payload(path)
        except Exception:
            continue
        for candidate in candidate_paths:
            value = _scalar_nested_value(payload, candidate)
            if value is not None:
                values.append(value)
                break
    return values


def _truth_or_prior_store_exposure_time_s(plan: CampaignPlan) -> float | None:
    for payload in (plan.layout_metadata, plan.config):
        value = _scalar_nested_value(payload, ("resolved_system", "source", "exposure_time_s"))
        if value is not None:
            return value
        value = _scalar_nested_value(payload, ("system", "source", "exposure_time_s"))
        if value is not None:
            return value
    return None


def _campaign_exposure_time_s(plan: CampaignPlan) -> float | None:
    value = _subblock_exposure_time_s(plan.config.get("experiment", {}) or {})
    if value is not None:
        return value
    return _truth_or_prior_store_exposure_time_s(plan)


def _exposure_values_consistent(
    *,
    campaign: float | None,
    summary_values: Sequence[float],
    truth_or_prior_store: float | None,
    expected_summary_count: int,
) -> bool:
    if campaign is None or truth_or_prior_store is None:
        return False
    if len(summary_values) != int(expected_summary_count):
        return False
    values = [value for value in [campaign, truth_or_prior_store] if value is not None]
    values.extend(float(value) for value in summary_values)
    reference = values[0]
    return all(math.isclose(reference, value, rel_tol=1.0e-12, abs_tol=1.0e-15) for value in values)


def iterative_context_diagnostics(
    *,
    plan: CampaignPlan,
    summary_paths: Sequence[Path],
) -> dict[str, Any]:
    campaign_exposure = _campaign_exposure_time_s(plan)
    summary_values = _summary_exposure_context_values(summary_paths)
    truth_or_prior_store = _truth_or_prior_store_exposure_time_s(plan)
    source_diag: dict[str, Any] = {
        "campaign": campaign_exposure,
        "summary_values": summary_values,
        "summary_count": int(len(summary_paths)),
        "summary_values_count": int(len(summary_values)),
        "has_all_summary_values": bool(len(summary_values) == len(summary_paths)),
        "truth_or_prior_store": truth_or_prior_store,
        "consistent": _exposure_values_consistent(
            campaign=campaign_exposure,
            summary_values=summary_values,
            truth_or_prior_store=truth_or_prior_store,
            expected_summary_count=len(summary_paths),
        ),
    }
    log_flux_diag: dict[str, Any] = {"base": "log10"}
    if campaign_exposure is not None and truth_or_prior_store is not None:
        log_flux_diag[
            "expected_log10_offset_if_truth_or_prior_vs_campaign"
        ] = float(math.log10(truth_or_prior_store / campaign_exposure))
    log_flux_diag[
        "expected_log10_offset_if_1800s_vs_0p05s"
    ] = float(math.log10(1800.0 / 0.05))
    return {
        "source.exposure_time_s": source_diag,
        "source.log_flux_total": log_flux_diag,
    }


def validate_iterative_log_flux_exposure_context(
    *,
    plan: CampaignPlan,
    context_diagnostics: Mapping[str, Any],
) -> None:
    if "source.log_flux_total" not in set(plan.layout.labels):
        return
    exposure_diag = context_diagnostics.get("source.exposure_time_s", {})
    if not isinstance(exposure_diag, Mapping) or not bool(exposure_diag.get("consistent", False)):
        raise RuntimeError(
            "source.log_flux_total iterative update requires consistent exposure context "
            "across campaign, subblock summaries, and truth/prior store. "
            f"Diagnostics: {json.dumps(exposure_diag, sort_keys=True)}"
        )


def _posterior_truth_by_label(
    *,
    labels: Sequence[str],
    posterior_rows: Mapping[str, Mapping[str, Any]],
    fallback_truth: Mapping[str, float],
) -> dict[str, float]:
    truth = {str(label): float(fallback_truth[label]) for label in labels if label in fallback_truth}
    for label in labels:
        row = posterior_rows.get(label)
        if row is None:
            continue
        value = posterior_float(row, ("truth_value",))
        if math.isfinite(value):
            truth[str(label)] = float(value)
    return truth


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True


def _render_retention_provenance_paths(window_root: Path) -> dict[str, Path]:
    root = window_root / "render_retention"
    return {
        "latest_json": root / "cleanup_latest.json",
        "history_jsonl": root / "cleanup_history.jsonl",
    }


def _subblock_root_from_summary_path(summary_path: Path) -> Path:
    return summary_path.parent.parent.parent


def _iterative_window_render_candidates(
    *,
    summary_paths: Sequence[Path],
) -> tuple[list[Path], list[dict[str, Any]]]:
    candidates: dict[Path, Path] = {}
    skipped: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        subblock_root = _subblock_root_from_summary_path(Path(summary_path))
        subblock_window_root = subblock_root.parent
        render_root = subblock_root / "render"
        if subblock_window_root.is_symlink() or subblock_root.is_symlink():
            skipped.append(
                {
                    "path": str(subblock_root),
                    "reason": "subblock_root_or_window_root_symlink",
                }
            )
            continue
        if not _path_is_within(render_root, subblock_root):
            skipped.append(
                {
                    "path": str(render_root),
                    "reason": "render_root_outside_subblock_root",
                }
            )
            continue
        if render_root.is_symlink():
            skipped.append({"path": str(render_root), "reason": "render_root_symlink"})
            continue
        if not render_root.is_dir():
            continue
        for suffix in RENDER_RETENTION_ALLOWED_SUFFIXES:
            for candidate in render_root.glob(f"*{suffix}"):
                if not candidate.is_file() or candidate.is_symlink():
                    skipped.append(
                        {
                            "path": str(candidate),
                            "reason": "not_regular_file_or_symlink",
                        }
                    )
                    continue
                if not _path_is_within(candidate, subblock_window_root):
                    skipped.append(
                        {
                            "path": str(candidate),
                            "reason": "candidate_outside_window_root",
                        }
                    )
                    continue
                if not candidate.name.endswith(RENDER_RETENTION_ALLOWED_SUFFIXES):
                    skipped.append(
                        {
                            "path": str(candidate),
                            "reason": "basename_not_allowed",
                        }
                    )
                    continue
                candidates[candidate.resolve(strict=False)] = candidate
    return sorted(candidates.values(), key=lambda path: str(path)), skipped


def _iterative_window_completion_guard(
    *,
    plan: CampaignPlan,
    window_case: BiasCase,
    window_index: int,
    posterior_path: Path,
    summary_paths: Sequence[Path],
) -> tuple[bool, dict[str, Any]]:
    window_root = plan.run_root / "cases" / window_case.case_name
    reference_update_path = window_root / "iterative_reference_update.json"
    window_summary_path = window_root / "science_summary.csv"
    diagnostics_path = window_root / "iterative_window_diagnostics.csv"
    summary_checks: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        check = {"path": str(summary_path), "exists": summary_path.exists(), "readable": False}
        if summary_path.exists():
            try:
                load_subblock_summary_artifact_payload(summary_path)
                check["readable"] = True
            except Exception as exc:
                check["error"] = str(exc)
        summary_checks.append(check)
    posterior_rows: Mapping[str, Mapping[str, Any]] = {}
    posterior_error = ""
    if posterior_path.exists():
        try:
            posterior_rows = posterior_rows_by_label(posterior_path)
        except Exception as exc:
            posterior_error = str(exc)
    reference_payload: dict[str, Any] = {}
    reference_error = ""
    if reference_update_path.exists():
        try:
            raw_reference = json.loads(reference_update_path.read_text(encoding="utf-8"))
            reference_payload = dict(raw_reference) if isinstance(raw_reference, Mapping) else {}
        except Exception as exc:
            reference_error = str(exc)
    guard = {
        "canonical_completed_window_artifacts": {
            "posterior_by_label_csv": str(posterior_path),
            "iterative_reference_update_json": str(reference_update_path),
            "science_summary_csv": str(window_summary_path),
            "iterative_window_diagnostics_csv": str(diagnostics_path),
        },
        "window_index": int(window_index),
        "window_case_name": window_case.case_name,
        "posterior_exists": posterior_path.exists(),
        "posterior_readable": bool(posterior_rows),
        "posterior_row_count": int(len(posterior_rows)),
        "posterior_error": posterior_error,
        "reference_update_exists": reference_update_path.exists(),
        "reference_update_status": str(reference_payload.get("status", "")),
        "reference_update_readable": bool(reference_payload),
        "reference_update_error": reference_error,
        "reference_update_posterior_table_path": str(
            reference_payload.get("posterior_table_path", "")
        ),
        "reference_update_posterior_table_matches": bool(
            str(reference_payload.get("posterior_table_path", ""))
            == str(posterior_path)
        ),
        "window_summary_exists": window_summary_path.exists(),
        "window_diagnostics_exists": diagnostics_path.exists(),
        "summary_checks": summary_checks,
        "summary_count": int(len(summary_checks)),
        "summary_readable_count": int(
            sum(bool(row.get("readable")) for row in summary_checks)
        ),
    }
    ok = (
        bool(posterior_rows)
        and bool(reference_payload)
        and str(reference_payload.get("status", "")) == "ok"
        and str(reference_payload.get("posterior_table_path", "")) == str(posterior_path)
        and bool(window_summary_path.exists())
        and bool(diagnostics_path.exists())
        and all(bool(row.get("readable")) for row in summary_checks)
    )
    guard["status"] = "ok" if ok else "failed"
    return ok, guard


def _cleanup_completed_iterative_window_renders(
    *,
    plan: CampaignPlan,
    window_case: BiasCase,
    window_index: int,
    posterior_path: Path,
    summary_paths: Sequence[Path],
) -> dict[str, Any]:
    policy = str(plan.iterative.get("render_retention", RENDER_RETENTION_DEFAULT))
    window_root = plan.run_root / "cases" / window_case.case_name
    provenance_paths = _render_retention_provenance_paths(window_root)
    payload: dict[str, Any] = {
        "schema_version": "observation_bias_render_retention_cleanup.v1",
        "created_at": now_iso_local_ms(),
        "retention_policy": policy,
        "run_root": str(plan.run_root),
        "case_name": window_case.case_name,
        "window_index": int(window_index),
        "window_root": str(window_root),
        "allowed_filename_suffixes": list(RENDER_RETENTION_ALLOWED_SUFFIXES),
        "files_deleted": [],
        "files_deleted_count": 0,
        "logical_bytes_deleted": 0,
        "candidate_count": 0,
        "skipped": [],
        "errors": [],
    }
    if policy == "keep":
        payload["cleanup_status"] = "skipped_policy_keep"
        return payload
    if policy != "delete_after_window":
        payload["cleanup_status"] = "failed"
        payload["errors"].append({"reason": f"unsupported_policy:{policy}"})
        _write_json(provenance_paths["latest_json"], json_ready(payload))
        _append_jsonl(provenance_paths["history_jsonl"], payload)
        return payload
    summary_paths = [Path(path) for path in summary_paths]
    guard_ok, guard = _iterative_window_completion_guard(
        plan=plan,
        window_case=window_case,
        window_index=window_index,
        posterior_path=posterior_path,
        summary_paths=summary_paths,
    )
    payload["completion_guard"] = guard
    if not guard_ok:
        payload["cleanup_status"] = "guard_failed"
        _write_json(provenance_paths["latest_json"], json_ready(payload))
        _append_jsonl(provenance_paths["history_jsonl"], payload)
        return payload
    candidates, skipped = _iterative_window_render_candidates(
        summary_paths=summary_paths,
    )
    payload["candidate_count"] = int(len(candidates))
    payload["skipped"] = skipped
    for candidate in candidates:
        try:
            size_bytes = int(candidate.stat().st_size)
        except Exception as exc:
            payload["errors"].append(
                {
                    "path": str(candidate),
                    "size_bytes": 0,
                    "error": str(exc),
                }
            )
            continue
        try:
            candidate.unlink()
        except Exception as exc:
            payload["errors"].append(
                {
                    "path": str(candidate),
                    "size_bytes": size_bytes,
                    "error": str(exc),
                }
            )
            continue
        payload["files_deleted"].append({"path": str(candidate), "size_bytes": size_bytes})
        payload["logical_bytes_deleted"] = int(payload["logical_bytes_deleted"]) + size_bytes
    payload["cleanup_status"] = "failed" if payload["errors"] else "ok"
    payload["files_deleted_count"] = int(len(payload["files_deleted"]))
    _write_json(provenance_paths["latest_json"], json_ready(payload))
    _append_jsonl(provenance_paths["history_jsonl"], payload)
    return payload


def execute_iterative_campaign(
    plan: CampaignPlan,
    *,
    resume: bool,
    max_workers: int,
    fail_fast: bool,
    quiet: bool,
    prior_source: str,
    allow_optimizer_scale_summaries: bool,
    resource_time: bool | str | None,
) -> dict[str, Any]:
    summary_scale_policy = (
        SUMMARY_SCALE_POLICY_ALLOW_OPTIMIZER
        if allow_optimizer_scale_summaries
        else SUMMARY_SCALE_POLICY_REQUIRE_SUMMED
    )
    truth = _truth_by_label(plan)
    diagnostics: list[dict[str, Any]] = []
    current_offsets_by_case = {
        case.case_name: dict(case.theta_reference_offsets) for case in plan.cases
    }
    previous_residual_by_case: dict[str, float] = {}
    previous_next_reference_by_case: dict[str, float] = {}
    previous_next_reference_by_case: dict[str, float] = {}
    all_status_rows: list[dict[str, Any]] = _read_csv_rows(
        plan.run_root / "subblock_status_iterative.csv"
    )
    for case in plan.cases:
        for window_index in range(int(plan.iterative["windows_per_draw"])):
            current_offsets = current_offsets_by_case[case.case_name]
            window_plan_obj, window_case = _window_plan(
                plan,
                case=case,
                window_index=window_index,
                current_offsets=current_offsets,
            )
            command_rows = _write_realized_window_commands(
                window_plan_obj=window_plan_obj,
                window_case=window_case,
            )
            expected_posterior = (
                window_plan_obj.run_root
                / "cases"
                / window_case.case_name
                / "posterior_by_label.csv"
            )
            if not (resume and expected_posterior.exists()):
                try:
                    execute_subblocks(
                        window_plan_obj,
                        resume=resume,
                        max_workers=max(1, int(max_workers)),
                        fail_fast=fail_fast,
                        quiet=quiet,
                        resource_time=resource_time,
                    )
                finally:
                    window_status = _augment_iterative_status_rows(
                        plan=plan,
                        window_case=window_case,
                        command_rows=command_rows,
                    )
                    all_status_rows.extend(window_status)
                    _write_csv_rows(
                        plan.run_root / "subblock_status_iterative.csv",
                        all_status_rows,
                    )
                aggregate_case(
                    plan=window_plan_obj,
                    case=window_case,
                    prior_source=prior_source,
                    summary_scale_policy=summary_scale_policy,
                )
            posterior_rows = posterior_rows_by_label(expected_posterior)
            if not posterior_rows:
                raise RuntimeError(
                    f"Missing iterative window posterior: {expected_posterior}"
                )
            context_diagnostics = iterative_context_diagnostics(
                plan=plan,
                summary_paths=window_plan_obj.summary_paths[window_case.case_name],
            )
            validate_iterative_log_flux_exposure_context(
                plan=plan,
                context_diagnostics=context_diagnostics,
            )
            posterior_truth = _posterior_truth_by_label(
                labels=plan.layout.labels,
                posterior_rows=posterior_rows,
                fallback_truth=truth,
            )
            posterior_offsets, posterior_offset_status = posterior_offsets_from_rows(
                labels=plan.layout.labels,
                posterior_rows_by_label=posterior_rows,
                truth_by_label=posterior_truth,
                fallback_offsets=current_offsets,
            )
            realized_root = plan.run_root / "cases" / window_case.case_name
            eigen_update_path = realized_root / "eigen_update_diagnostics.json"
            eigen_update_diagnostics = (
                json.loads(eigen_update_path.read_text(encoding="utf-8"))
                if eigen_update_path.exists()
                else {}
            )
            next_offsets = _resolved_next_offsets(
                plan=plan,
                current_offsets=current_offsets,
                posterior_offsets=posterior_offsets,
                posterior_rows=posterior_rows,
                posterior_truth=posterior_truth,
                eigen_update_diagnostics=eigen_update_diagnostics,
            )
            if eigen_update_diagnostics:
                safety_applied = {
                    label: float(next_offsets.get(label, 0.0))
                    - float(current_offsets.get(label, 0.0))
                    for label in plan.layout.labels
                }
                eigen_update_diagnostics["physical_update_after_safety_by_label"] = (
                    safety_applied
                )
                eigen_update_diagnostics["physical_update_norm_after_safety"] = float(
                    np.linalg.norm(list(safety_applied.values()))
                )
                _write_json(eigen_update_path, eigen_update_diagnostics)
            row = _binary_iterative_diagnostic_row(
                plan=plan,
                case=case,
                window_index=window_index,
                current_offsets=current_offsets,
                posterior_offsets=posterior_offsets,
                next_offsets=next_offsets,
                posterior_rows=posterior_rows,
                posterior_offset_status=posterior_offset_status,
                previous_residual_norm=previous_residual_by_case.get(case.case_name),
                previous_next_reference_norm=previous_next_reference_by_case.get(case.case_name),
                n_subblocks=int(plan.iterative["subblocks_per_window"]),
                eigen_update_diagnostics=eigen_update_diagnostics,
            )
            diagnostics.append(row)
            previous_residual_by_case[case.case_name] = float(
                row["posterior_error_norm_after"]
            )
            previous_next_reference_by_case[case.case_name] = float(
                row["next_reference_error_norm"]
            )
            _write_csv_rows(realized_root / "iterative_window_diagnostics.csv", [row])
            _write_json(
                realized_root / "iterative_reference_update.json",
                {
                    "schema_version": "observation_bias_iterative_reference_update.v1",
                    "case_name": case.case_name,
                    "window_case_name": window_case.case_name,
                    "window_index": int(window_index),
                    "update_gain": float(plan.iterative["update_gain"]),
                    "update_mode": str(plan.iterative["update_mode"]),
                    "current_offsets": dict(current_offsets),
                    "posterior_offsets": dict(posterior_offsets),
                    "next_offsets": dict(next_offsets),
                    "eigen_update_diagnostics_path": str(eigen_update_path),
                    "eigen_update_modes_path": str(
                        realized_root / "eigen_update_modes.csv"
                    ),
                    "truth_by_label": dict(posterior_truth),
                    "posterior_table_path": str(expected_posterior),
                    "diagnostics": row,
                    "context_diagnostics": context_diagnostics,
                    "posterior_offset_status": dict(posterior_offset_status),
                    "created_at": now_iso_local_ms(),
                    "status": "ok",
                    "not_scientific_if_synthetic_inputs": False,
                },
            )
            cleanup_result = _cleanup_completed_iterative_window_renders(
                plan=plan,
                window_case=window_case,
                window_index=window_index,
                posterior_path=expected_posterior,
                summary_paths=window_plan_obj.summary_paths[window_case.case_name],
            )
            if str(cleanup_result.get("cleanup_status", "")) == "failed":
                print(
                    "[observation_bias_campaign] render-retention cleanup failed "
                    f"for {window_case.case_name}: "
                    f"{cleanup_result.get('errors')}",
                    file=sys.stderr,
                    flush=True,
                )
            current_offsets_by_case[case.case_name] = next_offsets
    analysis_root = plan.run_root / "analysis"
    _write_csv_rows(analysis_root / "iterative_window_diagnostics.csv", diagnostics)
    return aggregate_iterative_outputs(plan)


def _iterative_inventory_rows(plan: CampaignPlan) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    path_fields = (
        ("summary", "summary_path"),
        ("case_posterior", "case_posterior_path"),
        ("window_summary", "window_summary_path"),
        ("iterative_reference_update", "iterative_reference_update_path"),
        ("window_diagnostic", "window_diagnostic_path"),
        ("eigen_update_diagnostics", "eigen_update_diagnostics_path"),
        ("eigen_update_modes", "eigen_update_modes_path"),
        ("realized_command", "realized_command_path"),
    )
    for row in plan.expected_output_rows:
        for kind, field in path_fields:
            value = str(row.get(field, ""))
            if not value:
                continue
            path = Path(value)
            exists = path.exists()
            status = {
                "kind": kind,
                "path_field": field,
                "path": str(path),
                "exists": bool(exists),
                "size_bytes": int(path.stat().st_size) if exists and path.is_file() else 0,
                "case_name": row.get("case_name", ""),
                "window_case_name": row.get("window_case_name", ""),
                "window_index": row.get("window_index", ""),
                "subblock_index": row.get("subblock_index", ""),
                "global_subblock_index": row.get("global_subblock_index", ""),
                "summary_path": row.get("summary_path", ""),
                "case_posterior_path": row.get("case_posterior_path", ""),
                "window_summary_path": row.get("window_summary_path", ""),
            }
            inventory.append(status)
            if not exists:
                missing.append(status)
    return inventory, missing


def _counts_by_kind(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind", "unknown"))
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def _iterative_status_summary(plan: CampaignPlan) -> dict[str, Any]:
    rows = _read_csv_rows(plan.run_root / "subblock_status_iterative.csv")
    failed = [row for row in rows if str(row.get("status", "")) == "failed"]
    completed = [row for row in rows if str(row.get("status", "")) == "ok"]
    expected = len(plan.expected_output_rows)
    windows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("case_name", "")), str(row.get("window_index", "")))
        bucket = windows.setdefault(key, {"rows": 0, "failed": 0})
        bucket["rows"] += 1
        if str(row.get("status", "")) == "failed":
            bucket["failed"] += 1
    incomplete_windows = [
        {"case_name": case_name, "window_index": window_index, **bucket}
        for (case_name, window_index), bucket in windows.items()
        if int(bucket["rows"]) < int(plan.iterative.get("subblocks_per_window", 0))
        or int(bucket["failed"]) > 0
    ]
    return {
        "status_rows": int(len(rows)),
        "total_expected_subblocks": int(expected),
        "completed_subblocks": int(len(completed)),
        "failed_subblocks": int(len(failed)),
        "incomplete_windows": incomplete_windows,
        "incomplete_window_count": int(len(incomplete_windows)),
        "first_failure": failed[0] if failed else None,
    }


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _iterative_forecast_detector_ke_fields(plan: CampaignPlan) -> dict[str, Any]:
    cfg = plan.config.get("experiment", {}).get("detector_calibration_knowledge_error", {})
    if not isinstance(cfg, Mapping):
        cfg = {}
    offsets = cfg.get("pixel_offsets", {}) if isinstance(cfg.get("pixel_offsets"), Mapping) else {}
    response = cfg.get("pixel_response", {}) if isinstance(cfg.get("pixel_response"), Mapping) else {}
    return {
        "detector_ke_pixel_offsets_sigma_pix": offsets.get("sigma_pix", ""),
        "detector_ke_pixel_response_sigma_fractional": response.get("sigma_fractional", ""),
    }


def _iterative_forecast_prior_sigma_microas(plan: CampaignPlan, case: BiasCase) -> float:
    if case.prior_sigma_by_label and "source.separation_as" in case.prior_sigma_by_label:
        return float(case.prior_sigma_by_label["source.separation_as"]) * 1.0e6
    prior_draws = plan.config.get("experiment", {}).get("prior_draws", {})
    if isinstance(prior_draws, Mapping):
        sigmas = prior_draws.get("sigmas", {})
        if isinstance(sigmas, Mapping):
            sep = sigmas.get("source.separation_as", {})
            if isinstance(sep, Mapping) and sep.get("sigma") is not None:
                return float(sep["sigma"]) * 1.0e6
    return float("nan")


def _project_iterative_case_rows(
    *,
    plan: CampaignPlan,
    case: BiasCase,
    actual_rows: Sequence[Mapping[str, Any]],
    forecast_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = sorted(actual_rows, key=lambda row: int(row.get("window_index", 0)))
    if not rows:
        return [], {}
    actual_windows = int(forecast_cfg["actual_windows"])
    projected_windows = int(forecast_cfg["projected_windows"])
    subblocks_per_window = int(forecast_cfg["subblocks_per_window"])
    final_actual = rows[-1]
    first_actual = rows[0]
    actual_final_error = _finite_float(
        final_actual.get("separation_next_reference_error_microas"),
        _finite_float(final_actual.get("separation_posterior_error_after_microas")),
    )
    actual_sigma = _finite_float(final_actual.get("posterior_sigma_separation_microas"))
    initial_error = _finite_float(first_actual.get("separation_reference_error_before_microas"))
    final_vector_norm_actual = _finite_float(final_actual.get("next_reference_error_norm"))
    ratios: list[float] = []
    for row in rows:
        before = _finite_float(row.get("separation_reference_error_before_microas"))
        after = _finite_float(row.get("separation_next_reference_error_microas"))
        if math.isfinite(before) and abs(before) > 1.0e-12 and math.isfinite(after):
            ratios.append(float(after / before))
    if not ratios:
        ratios = [1.0]
    vector_ratios: list[float] = []
    for row in rows:
        before = _finite_float(row.get("reference_error_norm_before"))
        after = _finite_float(row.get("next_reference_error_norm"))
        if math.isfinite(before) and before > 1.0e-30 and math.isfinite(after):
            vector_ratios.append(float(after / before))
    if not vector_ratios:
        vector_ratios = [1.0]

    evolution: list[dict[str, Any]] = []
    for row in rows:
        idx = int(row.get("window_index", 0))
        evolution.append(
            {
                "case_name": case.case_name,
                "condition_name": (case.prior_draw_metadata or {}).get("condition_name", ""),
                "draw_index": _case_draw_index(case),
                "window_index": idx,
                "window_kind": "actual",
                "forecast_mode": "realized",
                "n_subblocks_cumulative": int((idx + 1) * subblocks_per_window),
                "separation_error_microas": _finite_float(row.get("separation_next_reference_error_microas")),
                "posterior_sigma_separation_microas": _finite_float(row.get("posterior_sigma_separation_microas")),
                "vector_norm": _finite_float(row.get("next_reference_error_norm")),
                "update_cosine": _finite_float(row.get("update_cosine_with_ideal")),
            }
        )

    projected_error = actual_final_error
    projected_vector_norm = final_vector_norm_actual
    for idx in range(actual_windows, projected_windows):
        ratio = ratios[(idx - actual_windows) % len(ratios)]
        vector_ratio = vector_ratios[(idx - actual_windows) % len(vector_ratios)]
        projected_error = float(projected_error * ratio) if math.isfinite(projected_error) else projected_error
        projected_vector_norm = (
            float(projected_vector_norm * vector_ratio)
            if math.isfinite(projected_vector_norm)
            else projected_vector_norm
        )
        sigma = (
            float(actual_sigma * math.sqrt(actual_windows / float(idx + 1)))
            if math.isfinite(actual_sigma) and actual_windows > 0
            else float("nan")
        )
        evolution.append(
            {
                "case_name": case.case_name,
                "condition_name": (case.prior_draw_metadata or {}).get("condition_name", ""),
                "draw_index": _case_draw_index(case),
                "window_index": idx,
                "window_kind": "projected",
                "forecast_mode": "replicate_information",
                "n_subblocks_cumulative": int((idx + 1) * subblocks_per_window),
                "separation_error_microas": projected_error,
                "posterior_sigma_separation_microas": sigma,
                "vector_norm": projected_vector_norm,
                "update_cosine": "",
            }
        )
    projected_sigma = (
        float(actual_sigma * math.sqrt(actual_windows / float(projected_windows)))
        if math.isfinite(actual_sigma) and actual_windows > 0
        else float("nan")
    )
    actual_abs = [
        abs(_finite_float(row.get("separation_next_reference_error_microas")))
        for row in rows
        if math.isfinite(_finite_float(row.get("separation_next_reference_error_microas")))
    ]
    projected_abs = [
        abs(_finite_float(row.get("separation_error_microas")))
        for row in evolution
        if str(row.get("window_kind")) == "projected"
        and math.isfinite(_finite_float(row.get("separation_error_microas")))
    ]
    all_projected_abs = actual_abs + projected_abs
    final = {
        "case_name": case.case_name,
        "condition_name": (case.prior_draw_metadata or {}).get("condition_name", ""),
        "draw_index": _case_draw_index(case),
        "actual_windows": actual_windows,
        "projected_windows": projected_windows,
        "subblocks_per_window": subblocks_per_window,
        "actual_subblocks_total": int(actual_windows * subblocks_per_window),
        "projected_subblocks_total": int(projected_windows * subblocks_per_window),
        "observation_duration_s": float(forecast_cfg["observation_duration_s"]),
        "phi_ref": str(plan.config.get("experiment", {}).get("subblocks", {}).get("phi_ref", "")),
        "update_mode": str(plan.iterative.get("update_mode", "")),
        "eigen_damping_mode": str(plan.iterative.get("eigenbasis", {}).get("damping_mode", "")),
        "separation_prior_sigma_microas": _iterative_forecast_prior_sigma_microas(plan, case),
        "initial_separation_error_microas": initial_error,
        "actual_final_window_index": int(rows[-1].get("window_index", actual_windows - 1)),
        "actual_final_separation_error_microas": actual_final_error,
        "actual_final_abs_separation_error_microas": abs(actual_final_error) if math.isfinite(actual_final_error) else float("nan"),
        "actual_final_posterior_sigma_separation_microas": actual_sigma,
        "projected_final_separation_error_microas": projected_error,
        "projected_final_abs_separation_error_microas": abs(projected_error) if math.isfinite(projected_error) else float("nan"),
        "projected_final_posterior_sigma_separation_microas": projected_sigma,
        "projected_error_over_sigma": (
            float(projected_error / projected_sigma)
            if math.isfinite(projected_error) and math.isfinite(projected_sigma) and projected_sigma > 0.0
            else float("nan")
        ),
        "diverged_flag": bool(
            not math.isfinite(projected_error)
            or (
                math.isfinite(initial_error)
                and abs(projected_error) > max(abs(initial_error) * 10.0, 1.0e6)
            )
        ),
        "max_abs_separation_error_actual_microas": max(actual_abs) if actual_abs else float("nan"),
        "max_abs_separation_error_projected_microas": max(all_projected_abs) if all_projected_abs else float("nan"),
        "n_actual_windows_separation_worsened": int(
            sum(
                1
                for prev, cur in zip(actual_abs, actual_abs[1:])
                if math.isfinite(prev) and math.isfinite(cur) and cur > prev
            )
        ),
        "n_projected_windows_separation_worsened": int(
            sum(
                1
                for prev, cur in zip(all_projected_abs, all_projected_abs[1:])
                if math.isfinite(prev) and math.isfinite(cur) and cur > prev
            )
        ),
        "median_actual_update_cosine": float(
            np.nanmedian([_finite_float(row.get("update_cosine_with_ideal")) for row in rows])
        ),
        "median_projected_update_cosine": "",
        "final_vector_norm_actual": final_vector_norm_actual,
        "final_vector_norm_projected": projected_vector_norm,
        **_iterative_forecast_detector_ke_fields(plan),
        "forecast_mode": "replicate_information",
        "forecast_notes": (
            "Projected from realized iterative windows by cycling observed "
            "separation/reference transition ratios and scaling posterior sigma "
            "with accumulated projected window count."
        ),
    }
    return evolution, final


def _stochastic_iterative_forecast_trials(
    *,
    final_rows: Sequence[Mapping[str, Any]],
    forecast_cfg: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    if "stochastic_score_noise" not in set(forecast_cfg.get("modes", [])):
        return [], {}
    rng = np.random.default_rng(int(forecast_cfg.get("seed", 42)))
    n_trials = int(forecast_cfg.get("stochastic_trials", 128))
    trial_rows: list[dict[str, Any]] = []
    quantiles: dict[str, dict[str, float]] = {}
    for final in final_rows:
        case_name = str(final.get("case_name", ""))
        mean = _finite_float(final.get("projected_final_separation_error_microas"), 0.0)
        sigma = _finite_float(final.get("projected_final_posterior_sigma_separation_microas"), 0.0)
        samples = rng.normal(loc=mean, scale=max(sigma, 0.0), size=n_trials)
        for trial_index, sample in enumerate(samples):
            trial_rows.append(
                {
                    "case_name": case_name,
                    "condition_name": final.get("condition_name", ""),
                    "draw_index": final.get("draw_index", ""),
                    "forecast_mode": "stochastic_score_noise",
                    "trial_index": int(trial_index),
                    "projected_final_separation_error_microas": float(sample),
                    "projected_final_abs_separation_error_microas": float(abs(sample)),
                    "projected_final_posterior_sigma_separation_microas": sigma,
                    "projected_error_over_sigma": float(sample / sigma) if sigma > 0.0 else float("nan"),
                    "seed": int(forecast_cfg.get("seed", 42)),
                }
            )
        quantiles[case_name] = {
            "projected_p16_separation_error_microas": float(np.percentile(samples, 16)),
            "projected_p50_separation_error_microas": float(np.percentile(samples, 50)),
            "projected_p84_separation_error_microas": float(np.percentile(samples, 84)),
            "projected_p95_abs_separation_error_microas": float(np.percentile(np.abs(samples), 95)),
        }
    return trial_rows, quantiles


def _plot_iterative_forecast_evolution(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAVE_MATPLOTLIB or plt is None or not rows:
        return
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row.get("case_name", "")), []).append(row)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for case_name, case_rows in sorted(by_case.items()):
        ordered = sorted(case_rows, key=lambda row: int(row.get("window_index", 0)))
        x = [int(row.get("window_index", 0)) + 1 for row in ordered]
        y = [_finite_float(row.get("separation_error_microas")) for row in ordered]
        ax.plot(x, y, marker="o", linewidth=1.2, label=case_name)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Window")
    ax.set_ylabel("Separation error (microas)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_iterative_observation_forecast_artifacts(
    plan: CampaignPlan,
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    forecast_cfg = _resolve_iterative_forecast_config(plan.config["experiment"])
    analysis_root = plan.run_root / "analysis"
    if not bool(forecast_cfg.get("enabled", False)):
        return {"enabled": False}
    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in diagnostics:
        by_case.setdefault(str(row.get("case_name", "")), []).append(row)
    evolution_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for case in plan.cases:
        case_evolution, final = _project_iterative_case_rows(
            plan=plan,
            case=case,
            actual_rows=by_case.get(case.case_name, []),
            forecast_cfg=forecast_cfg,
        )
        evolution_rows.extend(case_evolution)
        if final:
            final_rows.append(final)
    trial_rows, quantiles = _stochastic_iterative_forecast_trials(
        final_rows=final_rows,
        forecast_cfg=forecast_cfg,
    )
    for row in final_rows:
        row.update(quantiles.get(str(row.get("case_name", "")), {}))
        if quantiles.get(str(row.get("case_name", ""))):
            row["forecast_notes"] = (
                str(row["forecast_notes"])
                + " Stochastic score-noise trials sample the projected endpoint "
                "residual with the projected separation sigma."
            )
    _write_csv_rows(analysis_root / "window_evolution_actual_and_projected.csv", evolution_rows)
    _write_csv_rows(analysis_root / "projected_observation_forecast.csv", final_rows)
    _write_csv_rows(analysis_root / "final_observation_summary.csv", final_rows)
    if trial_rows:
        _write_csv_rows(analysis_root / "projected_observation_forecast_trials.csv", trial_rows)
    _write_json(
        analysis_root / "final_observation_summary.json",
        {
            "schema_version": "iterative_observation_forecast_summary.v1",
            "created_at": now_iso_local_ms(),
            "forecast_config": dict(forecast_cfg),
            "rows": final_rows,
            "trial_rows": int(len(trial_rows)),
            "notes": [
                "Projected rows are forecasts from actual realized iterative windows, not fully rendered later windows.",
            ],
        },
    )
    if bool(forecast_cfg.get("plots", True)):
        _plot_iterative_forecast_evolution(
            analysis_root / "actual_vs_projected_separation_evolution.png",
            evolution_rows,
        )
    return {
        "enabled": True,
        "forecast_config": dict(forecast_cfg),
        "n_final_rows": int(len(final_rows)),
        "n_evolution_rows": int(len(evolution_rows)),
        "n_trial_rows": int(len(trial_rows)),
        "artifacts": {
            "final_observation_summary_csv": str(analysis_root / "final_observation_summary.csv"),
            "final_observation_summary_json": str(analysis_root / "final_observation_summary.json"),
            "projected_observation_forecast_csv": str(analysis_root / "projected_observation_forecast.csv"),
            "projected_observation_forecast_trials_csv": str(analysis_root / "projected_observation_forecast_trials.csv") if trial_rows else "",
            "window_evolution_actual_and_projected_csv": str(analysis_root / "window_evolution_actual_and_projected.csv"),
            "actual_vs_projected_plot": str(analysis_root / "actual_vs_projected_separation_evolution.png"),
        },
    }


def aggregate_iterative_outputs(plan: CampaignPlan) -> dict[str, Any]:
    analysis_root = plan.run_root / "analysis"
    inventory, missing = _iterative_inventory_rows(plan)
    _write_csv_rows(analysis_root / "output_inventory.csv", inventory)
    _write_csv_rows(analysis_root / "missing_outputs.csv", missing)
    missing_by_kind = _counts_by_kind(missing)
    existing_by_kind = _counts_by_kind([row for row in inventory if bool(row.get("exists"))])
    status_summary = _iterative_status_summary(plan)
    diagnostics: list[dict[str, Any]] = []
    truth = _truth_by_label(plan)
    current_offsets_by_case = {
        case.case_name: dict(case.theta_reference_offsets) for case in plan.cases
    }
    previous_residual_by_case: dict[str, float] = {}
    previous_next_reference_by_case: dict[str, float] = {}
    cases_by_name = {case.case_name: case for case in plan.cases}
    windows = sorted(
        {
            (str(row.get("case_name")), int(row.get("window_index", -1)))
            for row in plan.expected_output_rows
            if row.get("case_name") in cases_by_name and int(row.get("window_index", -1)) >= 0
        },
        key=lambda item: (item[0], item[1]),
    )
    for case_name, window_index in windows:
        case = cases_by_name[case_name]
        window_rows = _rows_for_iterative_window(
            plan,
            case_name=case_name,
            window_index=window_index,
        )
        if not window_rows:
            continue
        posterior_path = Path(str(window_rows[0]["case_posterior_path"]))
        if not posterior_path.exists():
            continue
        current_offsets = current_offsets_by_case[case_name]
        posterior_rows = posterior_rows_by_label(posterior_path)
        context_diagnostics = iterative_context_diagnostics(
            plan=plan,
            summary_paths=[Path(str(row["summary_path"])) for row in window_rows],
        )
        validate_iterative_log_flux_exposure_context(
            plan=plan,
            context_diagnostics=context_diagnostics,
        )
        posterior_truth = _posterior_truth_by_label(
            labels=plan.layout.labels,
            posterior_rows=posterior_rows,
            fallback_truth=truth,
        )
        posterior_offsets, posterior_offset_status = posterior_offsets_from_rows(
            labels=plan.layout.labels,
            posterior_rows_by_label=posterior_rows,
            truth_by_label=posterior_truth,
            fallback_offsets=current_offsets,
        )
        window_root = posterior_path.parent
        eigen_update_path = window_root / "eigen_update_diagnostics.json"
        eigen_update_diagnostics = (
            json.loads(eigen_update_path.read_text(encoding="utf-8"))
            if eigen_update_path.exists()
            else {}
        )
        next_offsets = _resolved_next_offsets(
            plan=plan,
            current_offsets=current_offsets,
            posterior_offsets=posterior_offsets,
            posterior_rows=posterior_rows,
            posterior_truth=posterior_truth,
            eigen_update_diagnostics=eigen_update_diagnostics,
        )
        row = _binary_iterative_diagnostic_row(
            plan=plan,
            case=case,
            window_index=window_index,
            current_offsets=current_offsets,
            posterior_offsets=posterior_offsets,
            next_offsets=next_offsets,
            posterior_rows=posterior_rows,
            posterior_offset_status=posterior_offset_status,
            previous_residual_norm=previous_residual_by_case.get(case_name),
            previous_next_reference_norm=previous_next_reference_by_case.get(case_name),
            n_subblocks=int(plan.iterative["subblocks_per_window"]),
            eigen_update_diagnostics=eigen_update_diagnostics,
        )
        diagnostics.append(row)
        previous_residual_by_case[case_name] = float(row["posterior_error_norm_after"])
        previous_next_reference_by_case[case_name] = float(row["next_reference_error_norm"])
        current_offsets_by_case[case_name] = next_offsets
    _write_csv_rows(analysis_root / "iterative_window_diagnostics.csv", diagnostics)
    iterative_forecast_summary = write_iterative_observation_forecast_artifacts(
        plan,
        diagnostics,
    )
    status = {
        "schema_version": "observation_bias_iterative_aggregate_status.v1",
        "created_at": now_iso_local_ms(),
        "run_root": str(plan.run_root),
        "used_stored_plan": True,
        "iterative_enabled": bool(plan.iterative.get("enabled", False)),
        "expected_output_rows": int(len(plan.expected_output_rows)),
        "missing_output_rows": int(len(missing)),
        "inventory_rows": int(len(inventory)),
        "existing_outputs_by_kind": existing_by_kind,
        "missing_outputs_by_kind": missing_by_kind,
        "missing_summaries": int(missing_by_kind.get("summary", 0)),
        "missing_posterior_tables": int(missing_by_kind.get("case_posterior", 0)),
        "completed_subblocks": int(status_summary["completed_subblocks"]),
        "failed_subblocks": int(status_summary["failed_subblocks"]),
        "incomplete_windows": int(status_summary["incomplete_window_count"]),
        "first_failure": status_summary["first_failure"],
        "iterative_window_diagnostic_rows": int(len(diagnostics)),
        "windows_per_draw": int(plan.iterative.get("windows_per_draw", 0)),
        "subblocks_per_window": int(plan.iterative.get("subblocks_per_window", 0)),
        "update_gain": float(plan.iterative.get("update_gain", 1.0)),
        "update_mode": str(plan.iterative.get("update_mode", "")),
        "iterative_forecast": iterative_forecast_summary,
    }
    _write_json(analysis_root / "aggregate_status.json", status)
    _write_json(plan.run_root / "campaign_summary.json", status)
    return status


def _stored_layout_labels(stored_plan: Mapping[str, Any]) -> list[str]:
    layout = stored_plan.get("theta_layout", {})
    if isinstance(layout, Mapping):
        labels = layout.get("labels", [])
        if isinstance(labels, list):
            return [str(label) for label in labels]
    return []


def _validate_aggregate_only_stored_plan(
    *,
    current_plan: CampaignPlan,
    stored_plan: Mapping[str, Any],
    stored_plan_path: Path,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    validated: list[str] = []

    def check(field: str, current: Any, stored: Any) -> None:
        validated.append(field)
        if current != stored:
            mismatches.append({"field": field, "current": current, "stored": stored})

    check("schema_version", CAMPAIGN_SCHEMA_VERSION, stored_plan.get("schema_version"))
    check("run_root", str(current_plan.run_root), str(stored_plan.get("run_root", "")))
    check(
        "case_set",
        sorted(current_plan.summary_paths.keys()),
        sorted(str(key) for key in (stored_plan.get("summary_paths", {}) or {}).keys()),
    )
    stored_labels = _stored_layout_labels(stored_plan)
    if stored_labels:
        check("layout_labels", list(current_plan.layout.labels), stored_labels)
    stored_iterative = stored_plan.get("iterative", {})
    if isinstance(stored_iterative, Mapping):
        check("iterative.enabled", bool(current_plan.iterative.get("enabled", False)), bool(stored_iterative.get("enabled", False)))
        check("iterative.update_mode", str(current_plan.iterative.get("update_mode", "")), str(stored_iterative.get("update_mode", "")))
        check("iterative.windows_per_draw", int(current_plan.iterative.get("windows_per_draw", 0)), int(stored_iterative.get("windows_per_draw", 0)))
        check("iterative.subblocks_per_window", int(current_plan.iterative.get("subblocks_per_window", 0)), int(stored_iterative.get("subblocks_per_window", 0)))
    stored_trace = stored_plan.get("trace_source", {})
    if isinstance(stored_trace, Mapping):
        check("trace_source.mode", str(current_plan.trace_source_plan.mode), str(stored_trace.get("mode", "")))
    stored_split = stored_plan.get("model_split", {})
    current_split = current_plan.config["experiment"].get("model_split", {})
    if isinstance(stored_split, Mapping) and isinstance(current_split, Mapping):
        check("model_split.truth_config_hash", current_split.get("truth_config_hash"), stored_split.get("truth_config_hash"))
        check("model_split.inference_config_hash", current_split.get("inference_config_hash"), stored_split.get("inference_config_hash"))
        check("model_split.components", current_split.get("components", {}), stored_split.get("components", {}))
    stored_wfe = stored_plan.get("high_order_wfe", {})
    current_wfe = current_plan.config["experiment"].get("high_order_wfe_summary", {})
    if isinstance(stored_wfe, Mapping) and isinstance(current_wfe, Mapping):
        check("high_order_wfe.enabled", bool(current_wfe.get("provenance", {}).get("enabled", False)), bool(stored_wfe.get("provenance", {}).get("enabled", False)))
        check("high_order_wfe.truth_seed", current_wfe.get("provenance", {}).get("truth_seed"), stored_wfe.get("provenance", {}).get("truth_seed"))
        check("high_order_wfe.mirrors", current_wfe.get("provenance", {}).get("mirrors", []), stored_wfe.get("provenance", {}).get("mirrors", []))
    stored_options = stored_plan.get("subblock_command_options", {})
    current_options = resolve_subblock_command_options(
        current_plan.config["experiment"].get("subblocks", {}) or {}
    )
    if isinstance(stored_options, Mapping):
        check(
            "summary_information_scale",
            str(current_options.get("summary_information_scale", "")),
            str(stored_options.get("summary_information_scale", "")),
        )
    stored_expected = stored_plan.get("expected_outputs", [])
    if isinstance(stored_expected, list):
        check("expected_outputs_count", len(current_plan.expected_output_rows), len(stored_expected))
    stored_iterative_plan = stored_plan.get("iterative_plan", [])
    if isinstance(stored_iterative_plan, list):
        check("iterative_plan_count", len(current_plan.iterative_plan_rows), len(stored_iterative_plan))
    payload = {
        "schema_version": "observation_bias_aggregate_only_plan_validation.v1",
        "created_at": now_iso_local_ms(),
        "stored_plan_used": True,
        "stored_plan_path": str(stored_plan_path),
        "validated_fields": validated,
        "mismatches": mismatches,
        "status": "ok" if not mismatches else "mismatch",
    }
    _write_json(current_plan.run_root / "analysis" / "aggregate_only_plan_validation.json", payload)
    if mismatches:
        raise ValueError(
            "Aggregate-only stored plan validation failed. Run aggregate-only against "
            "the original run root/config, or explicitly inspect the stored plan before "
            "continuing. Mismatches: "
            + "; ".join(
                f"{m['field']}: current={m['current']!r} stored={m['stored']!r}"
                for m in mismatches
            )
        )
    return payload


def _resolve_aggregate_only_run_root(
    *,
    config_path: Path | None,
    results_root: Path,
    run_name: str | None,
) -> Path | None:
    if run_name:
        return Path(results_root).resolve() / str(run_name)
    if config_path is None:
        return None
    try:
        cfg = load_config_file(config_path)
    except Exception:
        return None
    experiment = cfg.get("experiment", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(experiment, Mapping) and experiment.get("run_name"):
        return Path(results_root).resolve() / str(experiment["run_name"])
    return None


def _stored_plan_has_subblock_constant_smear(stored_plan: Mapping[str, Any]) -> bool:
    for row in plan_smear_rows(stored_plan):
        if row.get("smear_representative_kernel_json"):
            return True
    return False


def _layout_from_stored_plan(stored_plan: Mapping[str, Any]) -> ObservationThetaLayout:
    theta = stored_plan.get("theta_layout", {})
    if not isinstance(theta, Mapping):
        raise ValueError("Stored campaign_plan.json does not contain theta_layout.")
    labels = tuple(str(label) for label in theta.get("labels", ()))
    groups = tuple(str(group) for group in theta.get("label_groups", ()))
    return ObservationThetaLayout(labels=labels, label_groups=groups)


def _rehydrate_trace_source_plan(
    *,
    stored_plan: Mapping[str, Any],
    run_root: Path,
) -> PreparedTraceSourcePlan:
    rows = plan_smear_rows(stored_plan)
    trace_summary = stored_plan.get("trace_source", {})
    if not isinstance(trace_summary, Mapping):
        trace_summary = {}
    subblocks: list[PreparedTraceSubblock] = []
    for row in rows:
        subblocks.append(
            PreparedTraceSubblock(
                subblock_index=int(row.get("subblock_index", len(subblocks))),
                frame_truth_path=Path(str(row["frame_truth_path"])) if row.get("frame_truth_path") else None,
                starting_guess_prediction_path=Path(str(row["starting_guess_prediction_path"])) if row.get("starting_guess_prediction_path") else None,
                smear_truth_path=Path(str(row["smear_truth_csv"])) if row.get("smear_truth_csv") else None,
                smear_model_path=Path(str(row["smear_model_csv"])) if row.get("smear_model_csv") else None,
                smear_provenance_path=Path(str(row["smear_provenance_json"])) if row.get("smear_provenance_json") else None,
                trace_source_mode=str(row.get("trace_source_mode", trace_summary.get("mode", ""))),
                time_start_s=float(row["trajectory_time_start_s"]) if row.get("trajectory_time_start_s") not in (None, "") else None,
                time_end_s=float(row["trajectory_time_end_s"]) if row.get("trajectory_time_end_s") not in (None, "") else None,
                n_frames=int(row.get("n_frames", 0) or 0),
                output_keys=tuple(str(row.get("trajectory_output_keys", "")).split(",")) if row.get("trajectory_output_keys") else (),
                active_frame_keys=tuple(str(row.get("trajectory_active_frame_keys", "")).split(",")) if row.get("trajectory_active_frame_keys") else (),
                diagnostics={},
                provenance=dict(row),
            )
        )
    return PreparedTraceSourcePlan(
        mode=str(trace_summary.get("mode", "")),
        run_root=run_root,
        source_kind=str(trace_summary.get("source_kind", "")),
        output_keys=tuple(),
        active_frame_keys=tuple(),
        subblocks=tuple(subblocks),
        summary=dict(trace_summary),
        rows=tuple(dict(row) for row in rows),
    )


def _rehydrate_campaign_plan_from_stored(
    *,
    stored_plan: Mapping[str, Any],
    run_root: Path,
) -> CampaignPlan:
    resolved_config_path = run_root / "resolved_config.json"
    if resolved_config_path.exists():
        config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    else:
        config = {
            "experiment": {
                "subblocks": {},
                "model_split": stored_plan.get("model_split", {}),
                "template_hashes": stored_plan.get("template_hashes", []),
                "high_order_wfe_summary": stored_plan.get("high_order_wfe", {}),
            }
        }
    layout = _layout_from_stored_plan(stored_plan)
    truth_by_label = stored_plan.get("prior_truth_by_label", {})
    prior_truth = np.asarray(
        [float(truth_by_label.get(label, 0.0)) for label in layout.labels],
        dtype=float,
    )
    bias_cases_payload = stored_plan.get("bias_cases", [])
    cases = tuple(
        BiasCase(
            case_name=str(row.get("case_name", "")),
            theta_reference_offsets=dict(row.get("theta_reference_offsets", {}) or {}),
            case_origin=str(row.get("case_origin", "explicit")),
        )
        for row in bias_cases_payload
        if isinstance(row, Mapping)
    )
    summary_paths_payload = stored_plan.get("summary_paths", {})
    summary_paths = {
        str(case): [Path(str(path)) for path in paths]
        for case, paths in summary_paths_payload.items()
        if isinstance(paths, list)
    } if isinstance(summary_paths_payload, Mapping) else {}
    subblock_commands_payload = stored_plan.get("subblock_commands", {})
    subblock_commands = {
        str(case): [shlex.split(str(command)) for command in commands]
        for case, commands in subblock_commands_payload.items()
        if isinstance(commands, list)
    } if isinstance(subblock_commands_payload, Mapping) else {}
    subblock_plan_payload = stored_plan.get("subblock_plan", {})
    subblock_plans = {
        str(case): list(rows)
        for case, rows in subblock_plan_payload.items()
        if isinstance(rows, list)
    } if isinstance(subblock_plan_payload, Mapping) else {}
    return CampaignPlan(
        run_root=run_root,
        layout=layout,
        layout_metadata=dict(stored_plan.get("layout_metadata", {}) or {}),
        prior_truth=prior_truth,
        cases=cases,
        subblock_commands=subblock_commands,
        summary_paths=summary_paths,
        subblock_plans=subblock_plans,
        prior_draw_rows_by_case=dict(stored_plan.get("prior_draw_rows_by_case", {}) or {}),
        config=config,
        partition=dict(stored_plan.get("state_partition", {}) or {}),
        case_generation=dict(stored_plan.get("case_generation", {}) or {}),
        truth_realization=dict(stored_plan.get("truth_realization", {}) or {}),
        truth_realization_rows=_read_csv_rows(run_root / "truth_realization_by_label.csv"),
        trace_source_plan=_rehydrate_trace_source_plan(stored_plan=stored_plan, run_root=run_root),
        iterative=dict(stored_plan.get("iterative", {}) or {}),
        iterative_plan_rows=list(stored_plan.get("iterative_plan", []) or []),
        expected_output_rows=list(stored_plan.get("expected_outputs", []) or []),
    )


def _validate_stored_replay_plan(
    *,
    plan: CampaignPlan,
    stored_plan: Mapping[str, Any],
    stored_plan_path: Path,
    config_path: Path | None,
) -> dict[str, Any]:
    validate_stored_trace_source_artifacts(stored_plan)
    validate_campaign_model_split_artifacts(stored_plan)
    current_config = _load_campaign_config(config_path)
    current_experiment = current_config.get("experiment", {})
    current_iterative = _resolve_iterative_config(current_experiment)
    current_prior_draws = dict(current_experiment.get("prior_draws", {}) or {})
    current_case_count = (
        int(current_prior_draws.get("n_cases", 0))
        if bool(current_prior_draws.get("enabled", False))
        else len(current_experiment.get("bias_cases", []) or [])
    )
    current_expected_outputs = (
        current_case_count
        * int(current_iterative["windows_per_draw"])
        * int(current_iterative["subblocks_per_window"])
    )
    validations = [
        {
            "field": "case_count",
            "stored": len(plan.cases),
            "current": current_case_count,
            "match": len(plan.cases) == current_case_count,
        },
        {
            "field": "expected_outputs_count",
            "stored": len(plan.expected_output_rows),
            "current": current_expected_outputs,
            "match": len(plan.expected_output_rows) == current_expected_outputs,
        },
        {
            "field": "iterative_plan_count",
            "stored": len(plan.iterative_plan_rows),
            "current": current_expected_outputs,
            "match": len(plan.iterative_plan_rows) == current_expected_outputs,
        },
    ]
    mismatches = [row for row in validations if not bool(row["match"])]
    payload = {
        "schema_version": "observation_bias_aggregate_only_plan_validation.v1",
        "created_at": now_iso_local_ms(),
        "stored_plan_used": True,
        "stored_plan_path": str(stored_plan_path),
        "replay_mode": "stored_plan_subblock_constant_smear",
        "validated_fields": validations
        + [
            {"field": "theta_layout.labels", "stored": list(plan.layout.labels), "current": list(plan.layout.labels), "match": True},
        ],
        "mismatches": mismatches,
        "status": "ok" if not mismatches else "mismatch",
    }
    _write_json(plan.run_root / "analysis" / "aggregate_only_plan_validation.json", payload)
    if mismatches:
        raise ValueError(
            "Aggregate-only config does not match the stored campaign plan: "
            + ", ".join(str(row["field"]) for row in mismatches)
        )
    return payload


def run_observation_bias_campaign(
    *,
    config_path: Path | None,
    results_root: Path,
    run_name: str | None = None,
    dry_run: bool = False,
    aggregate_only: bool = False,
    resume: bool = False,
    max_workers: int = 1,
    fail_fast: bool = True,
    quiet: bool = False,
    system_preset: str | None = None,
    prior_source: str = "summary_theta_ref",
    allow_optimizer_scale_summaries: bool = False,
    resource_time: bool | str | None = None,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    plan_args = args
    if plan_args is None:
        plan_args = argparse.Namespace(
            aggregate_only=aggregate_only,
            resume=resume,
            run_name=None,
            n_subblocks=None,
            n_frames=None,
            trace_source_mode=None,
            trajectory_csv=None,
            trajectory_start_s=None,
            trajectory_duration_s=None,
            trajectory_n_subblocks=None,
            trajectory_frame_dt_s=None,
            trajectory_output_keys=None,
            trajectory_plan=None,
            noise=None,
            phi_ref=None,
            max_dense_dim=None,
            schur_curvature_method=None,
            summary_information_scale=None,
            render_retention=None,
            seed_policy=None,
            base_seed=None,
        )
    if aggregate_only:
        replay_run_root = _resolve_aggregate_only_run_root(
            config_path=config_path,
            results_root=results_root,
            run_name=run_name,
        )
        if replay_run_root is not None:
            stored_plan_path = replay_run_root / "campaign_plan.json"
            stored_plan = load_existing_campaign_plan(stored_plan_path)
            if stored_plan is not None and _stored_plan_has_subblock_constant_smear(stored_plan):
                plan = _rehydrate_campaign_plan_from_stored(
                    stored_plan=stored_plan,
                    run_root=replay_run_root,
                )
                _validate_stored_replay_plan(
                    plan=plan,
                    stored_plan=stored_plan,
                    stored_plan_path=stored_plan_path,
                    config_path=config_path,
                )
                if bool(plan.iterative.get("enabled", False)):
                    return aggregate_iterative_outputs(plan)
                return aggregate_campaign(
                    plan,
                    prior_source=prior_source,
                    allow_optimizer_scale_summaries=allow_optimizer_scale_summaries,
                )
    plan = build_campaign_plan(
        config_path=config_path,
        results_root=results_root,
        run_name=run_name,
        system_preset=system_preset,
        args=plan_args,
    )
    skip_plan_rewrite = False
    if aggregate_only:
        stored_plan_path = plan.run_root / "campaign_plan.json"
        stored_plan = load_existing_campaign_plan(stored_plan_path)
        if stored_plan is not None:
            validate_stored_trace_source_artifacts(stored_plan)
            validate_campaign_model_split_artifacts(stored_plan)
            _validate_aggregate_only_stored_plan(
                current_plan=plan,
                stored_plan=stored_plan,
                stored_plan_path=stored_plan_path,
            )
            skip_plan_rewrite = True
            stored_paths = stored_plan.get("summary_paths")
            if isinstance(stored_paths, Mapping):
                summary_paths = {
                    str(case): [Path(str(path)) for path in paths]
                    for case, paths in stored_paths.items()
                    if isinstance(paths, list)
                }
                subblock_commands_payload = stored_plan.get("subblock_commands")
                if isinstance(subblock_commands_payload, Mapping):
                    subblock_commands = {
                        str(case): [str(command).split(" ") for command in commands]
                        for case, commands in subblock_commands_payload.items()
                        if isinstance(commands, list)
                    }
                else:
                    subblock_commands = plan.subblock_commands
                subblock_plan_payload = stored_plan.get("subblock_plan")
                if isinstance(subblock_plan_payload, Mapping):
                    subblock_plans = {
                        str(case): list(rows)
                        for case, rows in subblock_plan_payload.items()
                        if isinstance(rows, list)
                    }
                else:
                    subblock_plans = plan.subblock_plans
                iterative_payload = stored_plan.get("iterative")
                iterative = (
                    dict(iterative_payload)
                    if isinstance(iterative_payload, Mapping)
                    else plan.iterative
                )
                iterative_plan_payload = stored_plan.get("iterative_plan")
                iterative_plan_rows = (
                    list(iterative_plan_payload)
                    if isinstance(iterative_plan_payload, list)
                    else plan.iterative_plan_rows
                )
                expected_payload = stored_plan.get("expected_outputs")
                expected_output_rows = (
                    list(expected_payload)
                    if isinstance(expected_payload, list)
                    else plan.expected_output_rows
                )
                stored_truth_by_label = stored_plan.get("prior_truth_by_label")
                prior_truth = plan.prior_truth
                if isinstance(stored_truth_by_label, Mapping):
                    prior_truth = np.asarray(
                        [
                            float(
                                stored_truth_by_label.get(
                                    str(label),
                                    plan.prior_truth[index],
                                )
                            )
                            for index, label in enumerate(plan.layout.labels)
                        ],
                        dtype=float,
                    )
                plan = replace(
                    plan,
                    prior_truth=prior_truth,
                    summary_paths=summary_paths,
                    subblock_commands=subblock_commands,
                    subblock_plans=subblock_plans,
                    iterative=iterative,
                    iterative_plan_rows=iterative_plan_rows,
                    expected_output_rows=expected_output_rows,
                    case_generation=dict(stored_plan.get("case_generation", plan.case_generation)),
                    truth_realization=dict(stored_plan.get("truth_realization", plan.truth_realization)),
                    partition=dict(stored_plan.get("state_partition", plan.partition)),
                )
    _ensure_dir(plan.run_root)
    if not skip_plan_rewrite:
        _write_json(plan.run_root / "campaign_plan.json", _plan_payload(plan))
        _write_json(plan.run_root / "model_split.json", plan.config["experiment"].get("model_split", {}))
        _write_json(
            plan.run_root / "model_split_summary.json",
            {
                "schema_version": "campaign_model_split.v1.run_root_summary",
                "truth_config_hash": plan.config["experiment"].get("model_split", {}).get("truth_config_hash"),
                "inference_config_hash": plan.config["experiment"].get("model_split", {}).get("inference_config_hash"),
                "components": plan.config["experiment"].get("model_split", {}).get("components", {}),
                "artifact_paths": plan.config["experiment"].get("model_split", {}).get("artifact_paths", {}),
            },
        )
        _write_json(plan.run_root / "truth_realization.json", plan.truth_realization)
        _write_csv_rows(plan.run_root / "truth_realization_by_label.csv", plan.truth_realization_rows)
        _write_json(plan.run_root / "resolved_config.json", plan.config)
        _write_csv_rows(plan.run_root / "subblock_plan.csv", _flatten_subblock_plan_rows(plan))
        _write_csv_rows(plan.run_root / "trajectory" / "smear_summary.csv", _smear_summary_rows_for_plan(plan, strict=True))
        _write_csv_rows(plan.run_root / "iterative_plan.csv", plan.iterative_plan_rows)
        _write_csv_rows(plan.run_root / "template_hashes.csv", plan.config["experiment"].get("template_hashes", []))
        _write_csv_rows(plan.run_root / "expected_outputs.csv", plan.expected_output_rows)
        _write_csv_rows(plan.run_root / "bias_cases.csv", _bias_case_rows(plan))
        _write_csv_rows(plan.run_root / "prior_draws.csv", _flatten_prior_draw_rows(plan))
        _write_json(
            plan.run_root / "execution_context.json",
            _execution_context_payload(
                run_root=plan.run_root,
                max_workers=max(1, int(max_workers)),
                resource_time=resource_time,
            ),
        )
    if dry_run:
        payload = _plan_payload(plan)
        if not quiet:
            print(json.dumps(payload, indent=2), flush=True)
        return payload
    if not aggregate_only:
        if bool(plan.iterative.get("enabled", False)):
            return execute_iterative_campaign(
                plan,
                resume=resume,
                max_workers=max(1, int(max_workers)),
                fail_fast=fail_fast,
                quiet=quiet,
                prior_source=prior_source,
                allow_optimizer_scale_summaries=allow_optimizer_scale_summaries,
                resource_time=resource_time,
            )
        execute_subblocks(
            plan,
            resume=resume,
            max_workers=max(1, int(max_workers)),
            fail_fast=fail_fast,
            quiet=quiet,
            resource_time=resource_time,
        )
    if bool(plan.iterative.get("enabled", False)):
        return aggregate_iterative_outputs(plan)
    return aggregate_campaign(
        plan,
        prior_source=prior_source,
        allow_optimizer_scale_summaries=allow_optimizer_scale_summaries,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an observation-level Zernike bias campaign.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n-subblocks", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--trace-source-mode", choices=("iid_jitter", "trajectory", "external_plan"), default=None)
    parser.add_argument("--trajectory-csv", type=Path, default=None)
    parser.add_argument("--trajectory-start-s", type=float, default=None)
    parser.add_argument("--trajectory-duration-s", type=float, default=None)
    parser.add_argument("--trajectory-n-subblocks", type=int, default=None)
    parser.add_argument("--trajectory-frame-dt-s", type=float, default=None)
    parser.add_argument("--trajectory-output-keys", default=None)
    parser.add_argument("--trajectory-plan", type=Path, default=None)
    parser.add_argument("--noise", choices=("inherit", "enabled", "disabled"), default=None)
    parser.add_argument("--phi-ref", choices=("truth_when_available", "recovered"), default=None)
    parser.add_argument("--max-dense-dim", type=int, default=None)
    parser.add_argument("--schur-curvature-method", choices=("auto", "dense", "structured_independent_frames"), default=None)
    parser.add_argument(
        "--summary-information-scale",
        choices=("summed_likelihood", "optimizer"),
        default=None,
    )
    parser.add_argument(
        "--render-retention",
        choices=SUPPORTED_RENDER_RETENTION_POLICIES,
        default=None,
        help="Rendered FITS retention policy for iterative window campaigns.",
    )
    parser.add_argument(
        "--allow-optimizer-scale-summaries",
        action="store_true",
        help="Allow legacy/debug optimizer-scale or unclassified real summaries.",
    )
    parser.add_argument("--seed-policy", choices=SUPPORTED_SEED_POLICIES, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument(
        "--resource-time",
        dest="resource_time",
        nargs="?",
        const="enabled",
        choices=("auto", "enabled", "gnu", "disabled"),
        default=None,
    )
    parser.add_argument("--no-resource-time", dest="resource_time", action="store_const", const="disabled")
    parser.add_argument("--profile-runtime", action="store_true", default=False)
    parser.add_argument("--profile-runtime-detail", choices=("basic", "full"), default=None)
    parser.add_argument("--memory-diagnostics", action="store_true", default=False)
    parser.add_argument("--system-preset", default=None)
    parser.add_argument(
        "--prior-source",
        choices=("auto", "summary_theta_ref", "resolved_system", "default_system"),
        default="summary_theta_ref",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_observation_bias_campaign(
        config_path=args.config,
        results_root=args.results_root,
        run_name=args.run_name,
        dry_run=bool(args.dry_run),
        aggregate_only=bool(args.aggregate_only),
        resume=bool(args.resume),
        max_workers=int(args.max_workers),
        fail_fast=bool(args.fail_fast),
        quiet=bool(args.quiet),
        system_preset=args.system_preset,
        prior_source=str(args.prior_source),
        allow_optimizer_scale_summaries=bool(args.allow_optimizer_scale_summaries),
        resource_time="auto" if args.resource_time is None else str(args.resource_time),
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
