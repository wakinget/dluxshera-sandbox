"""Run a worked Fisher screening study for one scalar or indexed candidate.

This entrypoint stays intentionally narrow:

- one shared candidate parameter at a time
- one target at a time
- an explicit frame-count x noise-mode matrix
- the existing ``fisher_only`` harness path underneath

The candidate may be an ordinary scalar key such as
``optics.plate_scale_as_per_pix`` or one indexed component such as
``optics.primary.zernike_coeffs_nm[3]``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from dluxshera.config.io import load_config_file, load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_keys import (
    get_obs_subblock_mapping_value,
    get_obs_subblock_store_value,
    parse_obs_subblock_key_address,
    validate_supported_obs_subblock_key_addresses,
)


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
STUDY_SCHEMA_VERSION = "candidate_fisher_screen.v1"
STUDY_MODE = "fisher_only"
# CANDIDATE_LIST = ("optics.plate_scale_as_per_pix",
#                   "source.separation_as",
#                   "source.log_flux_total",
#                   "source.contrast",
#                   "optics.primary.zernike_coeffs_nm[0]",
#                   "optics.secondary.zernike_coeffs_nm[0]")
DEFAULT_CANDIDATE_KEY = "optics.plate_scale_as_per_pix"
DEFAULT_TARGET_NAME = "ALPHA_CEN"
DEFAULT_FRAME_COUNTS = (1, 2, 5, 10, 20, 50)
SUPPORTED_NOISE_MODES = ("noiseless", "shot_noise_only")


def _screen_log(study_root: Path, message: str, **fields: Any) -> None:
    """Print and append one flushed progress line for the worked Fisher study."""

    parts = [f"[candidate_fisher_screen] {message}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    line = " ".join(parts)
    print(line, flush=True)
    study_root.mkdir(parents=True, exist_ok=True)
    with (study_root / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


@dataclass(frozen=True)
class WorkedFisherCase:
    """One explicit case in the worked single-candidate Fisher matrix."""

    candidate_key: str
    candidate_base_key: str
    candidate_index: int | None
    target_name: str
    frame_count: int
    noise_mode: str
    case_name: str
    case_root: Path


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_study_module():
    return _load_module(
        REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_study.py",
        "obs_subblock_candidate_fisher_study_module",
    )


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


def _write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        for row in rows:
            writer.writerow(row)


def _ensure_mapping(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        parent[key] = {}
        return parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict.")
    return value


def _candidate_metadata(candidate_key: str) -> dict[str, Any]:
    address = parse_obs_subblock_key_address(candidate_key)
    return {
        "candidate": address.canonical,
        "candidate_base_key": address.base_key,
        "candidate_index": address.index,
    }


def _slugify_text(text: str) -> str:
    """Return a compact filesystem-safe slug for free-text labels."""

    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower())
    slug = slug.strip("_")
    return slug or "unnamed"


def _candidate_slug(candidate_key: str) -> str:
    address = parse_obs_subblock_key_address(candidate_key)
    slug = address.base_key.replace(".", "_")
    if address.index is not None:
        slug = f"{slug}_i{address.index}"
    return slug


def _candidate_plot_label(candidate_key: str) -> str:
    return parse_obs_subblock_key_address(candidate_key).canonical


def _plot_title(candidate_key: str, plot_title: str) -> str:
    """Return the standardized multi-line title for Fisher screen plots."""

    return f"Fisher Screening\n{_candidate_plot_label(candidate_key)}\n{plot_title}"


def _artifact_prefix(candidate_key: str) -> str:
    return f"{_candidate_slug(candidate_key)}_fisher"


def _derive_study_root(
    *,
    candidate_key: str,
    target_name: str,
    results_root: Path = DEFAULT_RESULTS_ROOT,
) -> Path:
    return results_root.resolve() / (
        f"{_candidate_slug(candidate_key)}_fisher_{_slugify_text(target_name)}"
    )


def _resolve_template_context(template_path: Path) -> dict[str, Any]:
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
    return {
        "resolved_cfg": resolved_cfg,
        "system_cfg": system_cfg,
        "forward_spec": forward_spec,
        "store": store,
    }


def _resolve_candidate_mapping_value(
    mapping: dict[str, Any] | None,
    *,
    candidate_key: str,
) -> float | None:
    return get_obs_subblock_mapping_value(
        mapping,
        address=parse_obs_subblock_key_address(candidate_key),
    )


def _resolve_target_name(cfg: dict[str, Any] | None) -> str | None:
    if not isinstance(cfg, dict):
        return None
    source_cfg = cfg.get("source")
    if not isinstance(source_cfg, dict):
        return None
    target = source_cfg.get("target")
    if not isinstance(target, str) or not target.strip():
        return None
    return target.strip()


def parse_frame_counts(raw: str | Sequence[int] | None) -> tuple[int, ...]:
    """Parse a comma-separated or sequence-valued frame-count list."""

    if raw is None:
        return DEFAULT_FRAME_COUNTS
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    values: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError("--frame-counts must be a comma-separated list of integers.") from exc
        if value <= 0:
            raise ValueError("--frame-counts must contain only positive integers.")
        values.append(value)
    if not values:
        raise ValueError("--frame-counts must contain at least one value.")
    return tuple(values)


def parse_noise_modes(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse and validate the narrow first-pass noise-mode list."""

    if raw is None:
        return SUPPORTED_NOISE_MODES
    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    values: list[str] = []
    for token in tokens:
        if not token:
            continue
        if token not in SUPPORTED_NOISE_MODES:
            raise ValueError(
                f"Unsupported noise mode {token!r}. Expected one of: "
                + ", ".join(SUPPORTED_NOISE_MODES)
                + "."
            )
        values.append(token)
    if not values:
        raise ValueError("--noise-modes must contain at least one value.")
    return tuple(values)


def build_candidate_fisher_case_specs(
    *,
    study_root: Path,
    candidate_key: str = DEFAULT_CANDIDATE_KEY,
    frame_counts: Sequence[int] = DEFAULT_FRAME_COUNTS,
    noise_modes: Sequence[str] = SUPPORTED_NOISE_MODES,
    target_name: str = DEFAULT_TARGET_NAME,
) -> tuple[WorkedFisherCase, ...]:
    """Expand the explicit worked-study matrix into stable case roots."""

    metadata = _candidate_metadata(candidate_key)
    specs: list[WorkedFisherCase] = []
    target_slug = _slugify_text(target_name)
    candidate_slug = _candidate_slug(candidate_key)
    for noise_mode in noise_modes:
        if noise_mode not in SUPPORTED_NOISE_MODES:
            raise ValueError(
                f"Unsupported noise mode {noise_mode!r}. Expected one of: "
                + ", ".join(SUPPORTED_NOISE_MODES)
                + "."
            )
        for frame_count in frame_counts:
            if int(frame_count) <= 0:
                raise ValueError("Frame counts must be positive integers.")
            case_name = (
                f"{target_slug}_{candidate_slug}_n{int(frame_count):03d}_{noise_mode}"
            )
            specs.append(
                WorkedFisherCase(
                    candidate_key=metadata["candidate"],
                    candidate_base_key=metadata["candidate_base_key"],
                    candidate_index=metadata["candidate_index"],
                    target_name=target_name,
                    frame_count=int(frame_count),
                    noise_mode=noise_mode,
                    case_name=case_name,
                    case_root=(study_root / "cases" / case_name).resolve(),
                )
            )
    return tuple(specs)


def resolve_candidate_truth_and_target(
    *,
    candidate_key: str,
    render_template: Path,
    inference_template: Path,
) -> tuple[float, str]:
    """Resolve one candidate truth/reference value and target name from templates."""

    address = parse_obs_subblock_key_address(candidate_key)
    candidate_value: float | None = None
    resolved_target: str | None = None
    render_context = _resolve_template_context(render_template)
    inference_context = _resolve_template_context(inference_template)
    validate_supported_obs_subblock_key_addresses(
        (address,),
        forward_spec=render_context["forward_spec"],
        reference_store=render_context["store"],
    )
    validate_supported_obs_subblock_key_addresses(
        (address,),
        forward_spec=inference_context["forward_spec"],
        reference_store=inference_context["store"],
    )
    for template_context in (render_context, inference_context):
        system_cfg = template_context["system_cfg"]
        resolved_cfg = template_context["resolved_cfg"]
        if candidate_value is None:
            experiment_cfg = resolved_cfg.get("experiment")
            truth_cfg = None
            if isinstance(experiment_cfg, dict):
                truth_cfg = experiment_cfg.get("truth")
            if isinstance(truth_cfg, dict):
                candidate_value = _resolve_candidate_mapping_value(
                    truth_cfg,
                    candidate_key=candidate_key,
                )
            if candidate_value is None:
                candidate_value = _resolve_candidate_mapping_value(
                    system_cfg,
                    candidate_key=candidate_key,
                )
            if candidate_value is None:
                candidate_value = get_obs_subblock_store_value(
                    template_context["store"],
                    address=address,
                )
        if resolved_target is None:
            resolved_target = _resolve_target_name(system_cfg)
    if candidate_value is None:
        raise ValueError(
            "Unable to resolve the baseline candidate value from the configured templates."
        )
    if resolved_target is None:
        resolved_target = DEFAULT_TARGET_NAME
    return float(candidate_value), resolved_target


def write_noise_mode_render_template(
    *,
    base_render_template: Path,
    output_path: Path,
    noise_mode: str,
) -> Path:
    """Write one render-template copy with explicit noiseless or shot-noise settings."""

    if noise_mode not in SUPPORTED_NOISE_MODES:
        raise ValueError(
            f"Unsupported noise mode {noise_mode!r}. Expected one of: "
            + ", ".join(SUPPORTED_NOISE_MODES)
            + "."
        )

    cfg = load_config_file(base_render_template.resolve())
    experiment_cfg = _ensure_mapping(cfg, "experiment", path="root")
    noise_cfg = _ensure_mapping(experiment_cfg, "noise", path="experiment")
    noise_cfg["enabled"] = noise_mode == "shot_noise_only"
    noise_cfg["photon_noise"] = True
    noise_cfg["read_noise"] = False
    noise_cfg["dark_current"] = False
    _write_json(output_path, cfg)
    return output_path


def build_case_row(
    *,
    case: WorkedFisherCase,
    truth_value: float,
    case_summary: dict[str, Any] | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Flatten one case summary into the aggregate row contract."""

    base_row: dict[str, Any] = {
        "target": case.target_name,
        "candidate": case.candidate_key,
        "candidate_base_key": case.candidate_base_key,
        "candidate_index": case.candidate_index,
        "study_mode": STUDY_MODE,
        "frame_count": int(case.frame_count),
        "noise_mode": case.noise_mode,
        "truth_value": float(truth_value),
        "case_name": case.case_name,
        "case_root": str(case.case_root),
        "case_status": "error" if error_message is not None else "planned",
        "error_message": error_message,
        "case_summary_path": None,
        "fisher_summary_json": None,
        "fisher_blocks_npz": None,
        "reference_value": None,
        "nuisance_keys": None,
        "fisher_method": None,
        "f_pp": None,
        "i_marg": None,
        "sigma_cond": None,
        "sigma_marg": None,
        "absorption_fraction": None,
        "candidate_runtime_status": None,
        "finite_difference_f_pp": None,
        "candidate_model_rms_delta_1pct": None,
        "candidate_loss_delta_1pct": None,
        "frame_store_preserves_candidate": None,
        "f_pp_is_finite": None,
        "i_marg_is_finite": None,
        "valid_conditional_sigma": None,
        "valid_marginal_sigma": None,
        "marginalization_status": None,
        "nuisance_block_status": None,
    }
    if case_summary is None:
        return base_row

    base_row["case_summary_path"] = case_summary.get("summary_path")
    fisher_summary = case_summary.get("fisher_summary")
    if not isinstance(fisher_summary, dict):
        if error_message is None:
            base_row["case_status"] = "planned" if case_summary.get("dry_run") else "missing_summary"
        return base_row

    artifacts = fisher_summary.get("artifacts")
    base_row.update(
        {
            "case_status": "ok",
            "fisher_summary_json": (
                None if not isinstance(artifacts, dict) else artifacts.get("fisher_summary_json")
            ),
            "fisher_blocks_npz": (
                None if not isinstance(artifacts, dict) else artifacts.get("fisher_blocks_npz")
            ),
            "reference_value": fisher_summary.get("candidate_reference_value"),
            "nuisance_keys": "|".join(fisher_summary.get("frame_keys", [])),
            "fisher_method": fisher_summary.get("fisher_method"),
            "f_pp": fisher_summary.get("f_pp"),
            "i_marg": fisher_summary.get("i_marg"),
            "sigma_cond": fisher_summary.get("sigma_cond"),
            "sigma_marg": fisher_summary.get("sigma_marg"),
            "absorption_fraction": fisher_summary.get("absorption_fraction"),
            "candidate_runtime_status": fisher_summary.get("candidate_runtime_status"),
            "finite_difference_f_pp": fisher_summary.get("finite_difference_f_pp"),
            "candidate_model_rms_delta_1pct": fisher_summary.get(
                "candidate_model_rms_delta_1pct"
            ),
            "candidate_loss_delta_1pct": fisher_summary.get("candidate_loss_delta_1pct"),
            "frame_store_preserves_candidate": fisher_summary.get(
                "frame_store_preserves_candidate"
            ),
            "f_pp_is_finite": fisher_summary.get("f_pp_is_finite"),
            "i_marg_is_finite": fisher_summary.get("i_marg_is_finite"),
            "valid_conditional_sigma": fisher_summary.get("valid_conditional_sigma"),
            "valid_marginal_sigma": fisher_summary.get("valid_marginal_sigma"),
            "marginalization_status": fisher_summary.get("marginalization_status"),
            "nuisance_block_status": fisher_summary.get("nuisance_block_status"),
        }
    )
    return base_row


def _stat_value(mapping: dict[str, Any] | None, key: str) -> Any:
    """Read one scalar stat from a mapping when available."""

    if not isinstance(mapping, dict):
        return None
    return mapping.get(key)


def build_noise_audit_row(
    *,
    case: WorkedFisherCase,
    truth_value: float,
    case_summary: dict[str, Any] | None,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Flatten one case's cube/variance audit into the aggregate audit contract."""

    row: dict[str, Any] = {
        "target": case.target_name,
        "candidate": case.candidate_key,
        "candidate_base_key": case.candidate_base_key,
        "candidate_index": case.candidate_index,
        "frame_count": int(case.frame_count),
        "noise_mode": case.noise_mode,
        "truth_value": float(truth_value),
        "case_name": case.case_name,
        "case_root": str(case.case_root),
        "case_status": "error" if error_message is not None else "planned",
        "error_message": error_message,
        "variance_model": None,
        "variance_source": None,
        "render_variance_artifact_available": None,
        "render_variance_artifact_used": None,
        "cube_sum": None,
        "cube_mean": None,
        "cube_min": None,
        "cube_max": None,
        "cube_zero_count": None,
        "variance_sum": None,
        "variance_mean": None,
        "variance_min": None,
        "variance_max": None,
        "variance_zero_count": None,
        "data_as_variance_mean": None,
        "data_as_variance_min": None,
        "data_variance_floor_value": None,
        "data_variance_floor_source": None,
        "data_variance_floor_clipped_count": None,
        "data_variance_floor_clipped_fraction": None,
        "render_variance_mean": None,
        "render_variance_min": None,
        "variance_mean_over_cube_mean": None,
        "data_variance_mean_over_cube_mean": None,
        "render_variance_mean_over_cube_mean": None,
        "f_pp": None,
        "i_marg": None,
        "sigma_cond": None,
        "sigma_marg": None,
        "absorption_fraction": None,
    }
    if case_summary is None:
        return row

    fisher_summary = case_summary.get("fisher_summary")
    if not isinstance(fisher_summary, dict):
        if error_message is None:
            row["case_status"] = "planned" if case_summary.get("dry_run") else "missing_summary"
        return row

    noise_audit = fisher_summary.get("noise_audit")
    cube_stats = noise_audit.get("cube_stats") if isinstance(noise_audit, dict) else None
    variance_stats = noise_audit.get("variance_stats") if isinstance(noise_audit, dict) else None
    data_variance_stats = (
        noise_audit.get("data_as_variance_stats") if isinstance(noise_audit, dict) else None
    )
    render_variance_stats = (
        noise_audit.get("render_variance_stats") if isinstance(noise_audit, dict) else None
    )

    row.update(
        {
            "case_status": "ok",
            "variance_model": None if not isinstance(noise_audit, dict) else noise_audit.get("variance_model"),
            "variance_source": None if not isinstance(noise_audit, dict) else noise_audit.get("variance_source"),
            "render_variance_artifact_available": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("render_variance_artifact_available")
            ),
            "render_variance_artifact_used": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("render_variance_artifact_used")
            ),
            "cube_sum": _stat_value(cube_stats, "sum"),
            "cube_mean": _stat_value(cube_stats, "mean"),
            "cube_min": _stat_value(cube_stats, "min"),
            "cube_max": _stat_value(cube_stats, "max"),
            "cube_zero_count": _stat_value(cube_stats, "zero_count"),
            "variance_sum": _stat_value(variance_stats, "sum"),
            "variance_mean": _stat_value(variance_stats, "mean"),
            "variance_min": _stat_value(variance_stats, "min"),
            "variance_max": _stat_value(variance_stats, "max"),
            "variance_zero_count": _stat_value(variance_stats, "zero_count"),
            "data_as_variance_mean": _stat_value(data_variance_stats, "mean"),
            "data_as_variance_min": _stat_value(data_variance_stats, "min"),
            "data_variance_floor_value": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("data_variance_floor_value")
            ),
            "data_variance_floor_source": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("data_variance_floor_source")
            ),
            "data_variance_floor_clipped_count": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("data_variance_floor_clipped_count")
            ),
            "data_variance_floor_clipped_fraction": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("data_variance_floor_clipped_fraction")
            ),
            "render_variance_mean": _stat_value(render_variance_stats, "mean"),
            "render_variance_min": _stat_value(render_variance_stats, "min"),
            "variance_mean_over_cube_mean": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("variance_mean_over_cube_mean")
            ),
            "data_variance_mean_over_cube_mean": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("data_variance_mean_over_cube_mean")
            ),
            "render_variance_mean_over_cube_mean": (
                None
                if not isinstance(noise_audit, dict)
                else noise_audit.get("render_variance_mean_over_cube_mean")
            ),
            "f_pp": fisher_summary.get("f_pp"),
            "i_marg": fisher_summary.get("i_marg"),
            "sigma_cond": fisher_summary.get("sigma_cond"),
            "sigma_marg": fisher_summary.get("sigma_marg"),
            "absorption_fraction": fisher_summary.get("absorption_fraction"),
        }
    )
    return row


def build_noise_audit_comparisons(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build per-frame-count noiseless vs shot-noise comparison rows."""

    candidate = None
    candidate_base_key = None
    candidate_index = None
    target = None
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("case_status") != "ok":
            continue
        frame_count = row.get("frame_count")
        noise_mode = row.get("noise_mode")
        if frame_count is None or noise_mode is None:
            continue
        candidate = row.get("candidate")
        candidate_base_key = row.get("candidate_base_key")
        candidate_index = row.get("candidate_index")
        target = row.get("target")
        by_key[(int(frame_count), str(noise_mode))] = row

    comparisons: list[dict[str, Any]] = []
    for frame_count in sorted({key[0] for key in by_key}):
        noiseless = by_key.get((frame_count, "noiseless"))
        shot = by_key.get((frame_count, "shot_noise_only"))
        if noiseless is None or shot is None:
            continue

        def _ratio(numerator_key: str, denominator_key: str | None = None) -> float | None:
            denom_key = numerator_key if denominator_key is None else denominator_key
            numerator = shot.get(numerator_key)
            denominator = noiseless.get(denom_key)
            try:
                numerator_f = float(numerator)
                denominator_f = float(denominator)
            except (TypeError, ValueError):
                return None
            if denominator_f == 0.0:
                return None
            return float(numerator_f / denominator_f)

        comparisons.append(
            {
                "candidate": candidate,
                "candidate_base_key": candidate_base_key,
                "candidate_index": candidate_index,
                "target": target,
                "frame_count": int(frame_count),
                "shot_to_noiseless_f_pp_ratio": _ratio("f_pp"),
                "shot_to_noiseless_i_marg_ratio": _ratio("i_marg"),
                "shot_to_noiseless_sigma_marg_ratio": _ratio("sigma_marg"),
                "shot_to_noiseless_cube_mean_ratio": _ratio("cube_mean"),
                "shot_to_noiseless_variance_mean_ratio": _ratio("variance_mean"),
                "shot_to_noiseless_data_as_variance_mean_ratio": _ratio(
                    "data_as_variance_mean"
                ),
                "noiseless_variance_model": noiseless.get("variance_model"),
                "shot_noise_variance_model": shot.get("variance_model"),
                "noiseless_data_variance_floor_value": noiseless.get(
                    "data_variance_floor_value"
                ),
                "shot_noise_data_variance_floor_value": shot.get(
                    "data_variance_floor_value"
                ),
                "noiseless_data_variance_floor_clipped_count": noiseless.get(
                    "data_variance_floor_clipped_count"
                ),
                "shot_noise_data_variance_floor_clipped_count": shot.get(
                    "data_variance_floor_clipped_count"
                ),
            }
        )

    return comparisons


def _augment_case_outputs(
    *,
    case: WorkedFisherCase,
    truth_value: float,
    case_summary: dict[str, Any],
) -> dict[str, Any]:
    """Add stable study-matrix metadata to case-local Fisher outputs."""

    summary_path = (case.case_root / "study" / STUDY_MODE / "summary.json").resolve()
    fallback_summary_path_value = case_summary.get("summary_path")
    if not summary_path.exists() and isinstance(fallback_summary_path_value, str):
        fallback_summary_path = Path(fallback_summary_path_value).resolve()
        if fallback_summary_path.exists():
            summary_path = fallback_summary_path

    if summary_path.exists():
        summary_payload = _read_json(summary_path)
    else:
        summary_payload = dict(case_summary)

    summary_payload["case_root"] = str(case.case_root.resolve())
    summary_payload["study_root"] = str(summary_path.parent.resolve())
    summary_payload["summary_path"] = str(summary_path.resolve())
    summary_payload["candidate_fisher_case"] = {
        "target": case.target_name,
        "candidate": case.candidate_key,
        "candidate_base_key": case.candidate_base_key,
        "candidate_index": case.candidate_index,
        "frame_count": int(case.frame_count),
        "noise_mode": case.noise_mode,
        "truth_value": float(truth_value),
        "study_mode": STUDY_MODE,
    }

    fisher_summary = summary_payload.get("fisher_summary")
    if not isinstance(fisher_summary, dict):
        if summary_path.parent.exists():
            _write_json(summary_path, summary_payload)
        return summary_payload

    fisher_summary.update(
        {
            "target_name": case.target_name,
            "truth_value": float(truth_value),
            "noise_mode": case.noise_mode,
            "frame_count": int(case.frame_count),
            "study_mode": STUDY_MODE,
            "candidate_parameter": case.candidate_key,
            "candidate_base_key": case.candidate_base_key,
            "candidate_index": case.candidate_index,
        }
    )
    artifacts = fisher_summary.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    local_artifacts = {
        "fisher_summary_json": summary_path.parent / "fisher_summary.json",
        "fisher_blocks_npz": summary_path.parent / "fisher_blocks.npz",
        "candidate_sensitivity_json": summary_path.parent / "candidate_sensitivity.json",
        "noise_audit_json": summary_path.parent / "noise_audit.json",
    }
    for key, artifact_path in local_artifacts.items():
        if artifact_path.exists() or key not in artifacts:
            artifacts[key] = str(artifact_path.resolve())
    fisher_summary["artifacts"] = artifacts
    fisher_summary_path = Path(artifacts["fisher_summary_json"]).resolve()
    _write_json(fisher_summary_path, fisher_summary)
    summary_payload["fisher_summary"] = fisher_summary
    _write_json(summary_path, summary_payload)
    return summary_payload


def _load_existing_case_summary(case: WorkedFisherCase) -> dict[str, Any] | None:
    """Load one reusable case-local Fisher summary when it already exists."""

    summary_path = (case.case_root / "study" / STUDY_MODE / "summary.json").resolve()
    fisher_summary_path = (case.case_root / "study" / STUDY_MODE / "fisher_summary.json").resolve()
    if not summary_path.exists() and not fisher_summary_path.exists():
        return None

    summary_payload: dict[str, Any] = {}
    if summary_path.exists():
        loaded = _read_json(summary_path)
        if isinstance(loaded, dict):
            summary_payload = loaded

    fisher_summary = summary_payload.get("fisher_summary")
    if not isinstance(fisher_summary, dict):
        if not fisher_summary_path.exists():
            return None
        fisher_summary = _read_json(fisher_summary_path)
        if not isinstance(fisher_summary, dict):
            return None
        summary_payload["fisher_summary"] = fisher_summary

    summary_payload.setdefault("dry_run", False)
    summary_payload["case_root"] = str(case.case_root.resolve())
    summary_payload["study_root"] = str(summary_path.parent.resolve())
    summary_payload["summary_path"] = str(summary_path.resolve())
    return summary_payload


def plot_metric_vs_frame_count(
    *,
    rows: Sequence[dict[str, Any]],
    metric_key: str,
    output_path: Path,
    y_label: str,
    title: str,
    log_y: bool,
    frame_counts: Sequence[int],
) -> Path:
    """Plot one scalar summary against frame count for the two study noise modes."""

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for noise_mode in SUPPORTED_NOISE_MODES:
        points: list[tuple[int, float]] = []
        for row in rows:
            if row.get("noise_mode") != noise_mode:
                continue
            value = row.get(metric_key)
            frame_count = row.get("frame_count")
            if value is None or frame_count is None:
                continue
            if isinstance(value, bool):
                continue
            try:
                y = float(value)
                x = int(frame_count)
            except (TypeError, ValueError):
                continue
            if not y_label.startswith("Absorption") and not (y > 0.0):
                continue
            if y != y or y == float("inf") or y == float("-inf"):
                continue
            points.append((x, y))
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        ax.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            linewidth=1.8,
            label=noise_mode,
        )

    ax.set_xlabel("Frame Count")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks([int(value) for value in frame_counts])
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def write_candidate_fisher_artifacts(
    *,
    study_root: Path,
    candidate_key: str,
    rows: Sequence[dict[str, Any]],
    noise_audit_rows: Sequence[dict[str, Any]],
    case_summaries: Sequence[dict[str, Any]],
    truth_value: float,
    target_name: str,
    frame_counts: Sequence[int],
    noise_modes: Sequence[str],
    dry_run: bool,
) -> dict[str, Any]:
    """Write the aggregate CSV/JSON/plot outputs for the worked study."""

    candidate_meta = _candidate_metadata(candidate_key)
    artifact_prefix = _artifact_prefix(candidate_key)
    csv_path = study_root / f"{artifact_prefix}_summary.csv"
    json_path = study_root / f"{artifact_prefix}_summary.json"
    noise_audit_csv = study_root / f"{artifact_prefix}_noise_audit.csv"
    noise_audit_json = study_root / f"{artifact_prefix}_noise_audit.json"
    sigma_marg_plot = study_root / f"{artifact_prefix}_sigma_marg_vs_frame_count.png"
    absorption_plot = study_root / f"{artifact_prefix}_absorption_fraction_vs_frame_count.png"
    sigma_cond_plot = study_root / f"{artifact_prefix}_sigma_cond_vs_frame_count.png"
    variance_mean_plot = study_root / f"{artifact_prefix}_variance_mean_vs_frame_count.png"

    _write_rows_csv(csv_path, rows)
    _write_rows_csv(noise_audit_csv, noise_audit_rows)
    plot_metric_vs_frame_count(
        rows=rows,
        metric_key="sigma_marg",
        output_path=sigma_marg_plot,
        y_label="Marginalized Sigma",
        title=_plot_title(candidate_key, "Marginalized Sigma"),
        log_y=True,
        frame_counts=frame_counts,
    )
    plot_metric_vs_frame_count(
        rows=rows,
        metric_key="absorption_fraction",
        output_path=absorption_plot,
        y_label="Absorption Fraction",
        title=_plot_title(candidate_key, "Absorption Fraction"),
        log_y=False,
        frame_counts=frame_counts,
    )
    plot_metric_vs_frame_count(
        rows=rows,
        metric_key="sigma_cond",
        output_path=sigma_cond_plot,
        y_label="Conditional Sigma",
        title=_plot_title(candidate_key, "Conditional Sigma"),
        log_y=True,
        frame_counts=frame_counts,
    )
    plot_metric_vs_frame_count(
        rows=noise_audit_rows,
        metric_key="variance_mean",
        output_path=variance_mean_plot,
        y_label="Variance Mean",
        title=_plot_title(candidate_key, "Variance Mean"),
        log_y=True,
        frame_counts=frame_counts,
    )

    noise_audit_comparisons = build_noise_audit_comparisons(noise_audit_rows)
    _write_json(
        noise_audit_json,
        {
            **candidate_meta,
            "target": target_name,
            "frame_counts": [int(value) for value in frame_counts],
            "noise_modes": list(noise_modes),
            "rows": list(noise_audit_rows),
            "comparisons": noise_audit_comparisons,
        },
    )

    summary = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_mode": STUDY_MODE,
        **candidate_meta,
        "study_root": str(study_root.resolve()),
        "target": target_name,
        "truth_value": float(truth_value),
        "frame_counts": [int(value) for value in frame_counts],
        "noise_modes": list(noise_modes),
        "dry_run": bool(dry_run),
        "case_count": len(rows),
        "successful_case_count": sum(1 for row in rows if row.get("case_status") == "ok"),
        "failed_case_count": sum(1 for row in rows if row.get("case_status") == "error"),
        "artifacts": {
            "aggregate_csv": str(csv_path.resolve()),
            "aggregate_json": str(json_path.resolve()),
            "noise_audit_csv": str(noise_audit_csv.resolve()),
            "noise_audit_json": str(noise_audit_json.resolve()),
            "sigma_marg_plot": str(sigma_marg_plot.resolve()),
            "absorption_fraction_plot": str(absorption_plot.resolve()),
            "sigma_cond_plot": str(sigma_cond_plot.resolve()),
            "variance_mean_plot": str(variance_mean_plot.resolve()),
            "progress_log": str((study_root / "progress.log").resolve()),
        },
        "cases": list(rows),
        "noise_audit_rows": list(noise_audit_rows),
        "noise_audit_comparisons": noise_audit_comparisons,
        "case_summaries": list(case_summaries),
    }
    _write_json(json_path, summary)
    return summary


def run_candidate_fisher_screen(
    *,
    study_root: Path | None = None,
    candidate_key: str = DEFAULT_CANDIDATE_KEY,
    trace_template: Path = DEFAULT_TRACE_TEMPLATE,
    render_template: Path = DEFAULT_RENDER_TEMPLATE,
    inference_template: Path = DEFAULT_INFERENCE_TEMPLATE,
    frame_counts: Sequence[int] = DEFAULT_FRAME_COUNTS,
    noise_modes: Sequence[str] = SUPPORTED_NOISE_MODES,
    reuse_existing_cases: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the worked Fisher matrix for one scalar or indexed candidate."""

    candidate_key = parse_obs_subblock_key_address(candidate_key).canonical
    study_module = _load_study_module()
    truth_value, resolved_target = resolve_candidate_truth_and_target(
        candidate_key=candidate_key,
        render_template=render_template,
        inference_template=inference_template,
    )
    target_name = resolved_target or DEFAULT_TARGET_NAME
    study_root = (
        _derive_study_root(
            candidate_key=candidate_key,
            target_name=target_name,
            results_root=DEFAULT_RESULTS_ROOT,
        )
        if study_root is None
        else study_root
    )
    study_root = study_root.resolve()
    study_root.mkdir(parents=True, exist_ok=True)
    _screen_log(
        study_root,
        "study.start",
        study_root_path=study_root,
        candidate=candidate_key,
        frame_counts=[int(value) for value in frame_counts],
        noise_modes=list(noise_modes),
        dry_run=bool(dry_run),
    )
    _screen_log(
        study_root,
        "study.config",
        candidate=candidate_key,
        target=target_name,
        truth_value=truth_value,
    )
    cases = build_candidate_fisher_case_specs(
        study_root=study_root,
        candidate_key=candidate_key,
        frame_counts=frame_counts,
        noise_modes=noise_modes,
        target_name=target_name,
    )

    template_dir = study_root / "templates"
    noise_templates = {
        noise_mode: write_noise_mode_render_template(
            base_render_template=render_template.resolve(),
            output_path=template_dir / f"render_{noise_mode}.json",
            noise_mode=noise_mode,
        )
        for noise_mode in noise_modes
    }

    rows: list[dict[str, Any]] = []
    noise_audit_rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    for case in cases:
        _screen_log(
            study_root,
            "case.start",
            case_name=case.case_name,
            frame_count=case.frame_count,
            noise_mode=case.noise_mode,
            case_root=case.case_root,
        )
        existing_case_summary = (
            None
            if dry_run or not reuse_existing_cases
            else _load_existing_case_summary(case)
        )
        if existing_case_summary is not None:
            case_summary = _augment_case_outputs(
                case=case,
                truth_value=truth_value,
                case_summary=existing_case_summary,
            )
            _screen_log(
                study_root,
                "case.reuse",
                case_name=case.case_name,
                case_status="ok",
                case_summary_path=case_summary.get("summary_path"),
            )
            case_summaries.append(case_summary)
            rows.append(
                build_case_row(
                    case=case,
                    truth_value=truth_value,
                    case_summary=case_summary,
                )
            )
            noise_audit_rows.append(
                build_noise_audit_row(
                    case=case,
                    truth_value=truth_value,
                    case_summary=case_summary,
                )
            )
            continue
        try:
            case_summary = study_module.run_obs_subblock_study(
                mode=STUDY_MODE,
                case_root=case.case_root,
                trace_template=trace_template.resolve(),
                render_template=noise_templates[case.noise_mode],
                inference_template=inference_template.resolve(),
                candidate_key=case.candidate_key,
                truth_value=truth_value,
                n_frames=int(case.frame_count),
                noise_mode=(
                    "disabled" if case.noise_mode == "noiseless" else "enabled"
                ),
                use_render_variance=True,
                dry_run=bool(dry_run),
            )
            case_summary = _augment_case_outputs(
                case=case,
                truth_value=truth_value,
                case_summary=case_summary,
            )
            fisher_summary = case_summary.get("fisher_summary")
            _screen_log(
                study_root,
                "case.done",
                case_name=case.case_name,
                case_status="ok",
                fisher_method=(
                    None
                    if not isinstance(fisher_summary, dict)
                    else fisher_summary.get("fisher_method")
                ),
                marginalization_status=(
                    None
                    if not isinstance(fisher_summary, dict)
                    else fisher_summary.get("marginalization_status")
                ),
            )
            case_summaries.append(case_summary)
            rows.append(
                build_case_row(
                    case=case,
                    truth_value=truth_value,
                    case_summary=case_summary,
                )
            )
            noise_audit_rows.append(
                build_noise_audit_row(
                    case=case,
                    truth_value=truth_value,
                    case_summary=case_summary,
                )
            )
        except Exception as exc:
            existing_case_summary = (
                None if dry_run else _load_existing_case_summary(case)
            )
            if existing_case_summary is not None:
                case_summary = _augment_case_outputs(
                    case=case,
                    truth_value=truth_value,
                    case_summary=existing_case_summary,
                )
                case_summaries.append(case_summary)
                rows.append(
                    build_case_row(
                        case=case,
                        truth_value=truth_value,
                        case_summary=case_summary,
                    )
                )
                noise_audit_rows.append(
                    build_noise_audit_row(
                        case=case,
                        truth_value=truth_value,
                        case_summary=case_summary,
                    )
                )
                _screen_log(
                    study_root,
                    "case.reuse_after_error",
                    case_name=case.case_name,
                    case_status="ok",
                    error_message=str(exc),
                    case_summary_path=case_summary.get("summary_path"),
                )
                continue
            rows.append(
                build_case_row(
                    case=case,
                    truth_value=truth_value,
                    case_summary=None,
                    error_message=str(exc),
                )
            )
            noise_audit_rows.append(
                build_noise_audit_row(
                    case=case,
                    truth_value=truth_value,
                    case_summary=None,
                    error_message=str(exc),
                )
            )
            _screen_log(
                study_root,
                "case.done",
                case_name=case.case_name,
                case_status="error",
                error_message=str(exc),
            )

    summary = write_candidate_fisher_artifacts(
        study_root=study_root,
        candidate_key=candidate_key,
        rows=rows,
        noise_audit_rows=noise_audit_rows,
        case_summaries=case_summaries,
        truth_value=truth_value,
        target_name=target_name,
        frame_counts=frame_counts,
        noise_modes=noise_modes,
        dry_run=dry_run,
    )
    if summary.get("noise_audit_comparisons"):
        largest = max(
            (
                row
                for row in summary["noise_audit_comparisons"]
                if row.get("shot_to_noiseless_f_pp_ratio") is not None
            ),
            key=lambda row: float(row["shot_to_noiseless_f_pp_ratio"]),
            default=None,
        )
        if largest is not None:
            _screen_log(
                study_root,
                "noise.audit.summary",
                frame_count=largest["frame_count"],
                shot_to_noiseless_f_pp_ratio=largest["shot_to_noiseless_f_pp_ratio"],
                shot_to_noiseless_variance_mean_ratio=largest[
                    "shot_to_noiseless_variance_mean_ratio"
                ],
            )
    _screen_log(
        study_root,
        "study.done",
        case_count=summary["case_count"],
        successful_case_count=summary["successful_case_count"],
        failed_case_count=summary["failed_case_count"],
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the worked fisher_only screening study for one scalar or indexed candidate."
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit study output root. When omitted, the script writes to "
            "Results/<candidate_slug>_fisher_<target_slug>/."
        ),
    )
    parser.add_argument(
        "--candidate",
        default=DEFAULT_CANDIDATE_KEY,
        help=(
            "Canonical candidate key, for example optics.plate_scale_as_per_pix or "
            "optics.primary.zernike_coeffs_nm[3]."
        ),
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
        "--frame-counts",
        default=None,
        help="Optional comma-separated frame counts. Default: 1,2,5,10,20,50.",
    )
    parser.add_argument(
        "--noise-modes",
        default=None,
        help="Optional comma-separated noise modes. Default: noiseless,shot_noise_only.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help=(
            "Recompute Fisher cases even when a case-local study summary already exists. "
            "By default, reruns reuse existing successful case outputs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the case matrix and write aggregate placeholders without running Fisher cases.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    frame_counts = parse_frame_counts(args.frame_counts)
    noise_modes = parse_noise_modes(args.noise_modes)
    summary = run_candidate_fisher_screen(
        study_root=args.study_root,
        candidate_key=args.candidate,
        trace_template=args.trace_template,
        render_template=args.render_template,
        inference_template=args.inference_template,
        frame_counts=frame_counts,
        noise_modes=noise_modes,
        reuse_existing_cases=not bool(args.force_rerun),
        dry_run=bool(args.dry_run),
    )
    print(f"Study root: {summary['study_root']}")
    print(f"Cases: {summary['case_count']}")
    print(f"Aggregate JSON: {summary['artifacts']['aggregate_json']}")
    return summary


if __name__ == "__main__":
    main()
