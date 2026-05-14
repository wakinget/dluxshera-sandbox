"""Run scalar-bias capture sweeps for image-backed Schur summaries."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dluxshera-matplotlib"),
)

from dluxshera.inference.observation_belief import (
    ObservationBeliefState,
    update_observation_belief,
)
from dluxshera.inference.observation_forecast import build_default_prior_sigma
from dluxshera.inference.observation_summary import load_subblock_summary
from dluxshera.utils.obs_subblock_io import now_iso_local_ms


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_SCHEMA_VERSION = "scalar_bias_capture_study.v1"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "scalar_bias_capture"
DEFAULT_STUDY_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_study.py"
DEFAULT_PARAMETERS = (
    "source.separation_as",
    "source.log_flux_total",
    "source.contrast",
    "optics.plate_scale_as_per_pix",
)
DEFAULT_BIAS_PPM_GRID = (-100.0, -30.0, -10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_TRACE_SEED = 0
DEFAULT_RENDER_SEED = 0
SUPPORTED_TRACE_SEED_POLICIES = ("same_trace_all_cases", "parameter_specific_trace", "case_specific_trace")

NUISANCE_COLUMNS = (
    "case_id",
    "case_name",
    "biased_parameter",
    "bias_ppm",
    "fractional_bias",
    "truth_value",
    "reference_value",
    "theta_reference_offset",
    "planned_truth_value",
    "planned_reference_value",
    "planned_theta_reference_offset",
    "actual_reference_value",
    "effective_truth_value",
    "reference_value_source",
    "truth_value_source",
    "planned_minus_actual_reference",
    "planned_actual_reference_abs_diff",
    "planned_actual_reference_rel_diff",
    "planned_actual_reference_mismatch_warning",
    "summary_json",
    "recovered_trace_csv",
    "truth_comparison_csv",
    "mean_dx_as",
    "std_dx_as",
    "rms_dx_as",
    "max_abs_dx_as",
    "mean_dy_as",
    "std_dy_as",
    "rms_dy_as",
    "max_abs_dy_as",
    "mean_dpa_deg",
    "std_dpa_deg",
    "rms_dpa_deg",
    "max_abs_dpa_deg",
    "initial_block_reduced_chi2",
    "final_block_reduced_chi2",
    "max_final_frame_reduced_chi2",
    "median_final_frame_reduced_chi2",
    "frame_quality_bad_frame_count",
    "frame_quality_good_frame_count",
    "status",
)

CORRECTION_COLUMNS = (
    "case_id",
    "case_name",
    "biased_parameter",
    "bias_ppm",
    "truth_value",
    "reference_value",
    "theta_reference_offset",
    "planned_truth_value",
    "planned_reference_value",
    "planned_theta_reference_offset",
    "actual_reference_value",
    "effective_truth_value",
    "reference_value_source",
    "truth_value_source",
    "planned_minus_actual_reference",
    "planned_actual_reference_abs_diff",
    "planned_actual_reference_rel_diff",
    "planned_actual_reference_mismatch_warning",
    "posterior_value_biased_parameter",
    "posterior_shift_biased_parameter",
    "expected_shift_to_truth",
    "posterior_error_biased_parameter",
    "correction_fraction_biased_parameter",
    "residual_fraction_biased_parameter",
    "posterior_sigma_biased_parameter",
    "posterior_error_over_sigma_biased_parameter",
    "moves_biased_parameter_toward_truth",
    "posterior_error_sign_flip",
    "score_norm",
    "reduced_information_rank",
    "reduced_information_min_eigenvalue",
    "reduced_information_condition_number",
    "update_solve_method",
    "surrogate_validation_available",
    "surrogate_validation_max_abs_error",
    "surrogate_validation_status",
    "theta_reference_consistency_passed",
    "status",
)

SCIENCE_COLUMNS = (
    "case_id",
    "case_name",
    "biased_parameter",
    "bias_ppm",
    "truth_separation_as",
    "reference_separation_as",
    "planned_truth_separation_as",
    "actual_reference_separation_as",
    "truth_value_source",
    "reference_value_source",
    "planned_minus_actual_reference_separation_as",
    "posterior_separation_as",
    "posterior_separation_shift_as",
    "posterior_separation_error_as",
    "posterior_separation_error_microas",
    "posterior_separation_sigma_as",
    "posterior_separation_sigma_microas",
    "posterior_separation_error_over_sigma",
    "separation_correction_fraction_if_biased",
    "separation_residual_fraction_if_biased",
    "moves_separation_toward_truth",
    "status",
)

POSTERIOR_COLUMNS = (
    "case_id",
    "case_name",
    "biased_parameter",
    "bias_ppm",
    "theta_label",
    "truth_value",
    "reference_value",
    "planned_truth_value",
    "planned_reference_value",
    "actual_reference_value",
    "effective_truth_value",
    "reference_value_source",
    "truth_value_source",
    "planned_minus_actual_reference",
    "posterior_mean",
    "posterior_shift",
    "posterior_error",
    "posterior_sigma",
    "posterior_error_over_sigma",
    "injected_bias",
    "correction_fraction",
    "residual_fraction",
)


@dataclass(frozen=True)
class ScalarBiasStudyConfig:
    results_root: Path
    run_name: str
    parameters: tuple[str, ...]
    theta_keys: tuple[str, ...]
    bias_ppm_grid: tuple[float, ...]
    n_frames: int
    noise: str
    trace_template: Path | None
    seed: int
    trace_seed_policy: str
    max_workers: int
    dry_run: bool = False
    resume: bool = False
    aggregate_only: bool = False
    fail_fast: bool = False
    quiet: bool = False
    plots: bool = False
    phi_ref: str = "recovered"
    max_dense_dim: int = 40
    schur_curvature_method: str = "auto"
    schur_damping: float = 1.0e-8
    schur_frame_quality_policy: str = "warn"
    reference_schedule_kind: str = "linear_warmup"
    reference_schedule_warmup_steps: int = 10
    reference_schedule_start_factor: float = 0.125
    reference_diagnostics_profile: str = "basic"
    reference_n_iter: int | None = None
    zero_bias_reference_n_iter: int | None = None

    @property
    def run_root(self) -> Path:
        return self.results_root / self.run_name


@dataclass(frozen=True)
class ScalarBiasCaseSpec:
    case_id: int
    case_name: str
    biased_parameter: str
    bias_ppm: float
    fractional_bias: float
    truth_value: float
    reference_value: float
    theta_reference_offset: float
    theta_reference_offset_expression: str
    n_frames: int
    noise: str
    trace_seed: int
    render_seed: int
    theta_keys: tuple[str, ...]
    phi_ref: str
    reference_n_iter: int | None
    results_root: Path
    case_root: Path
    summary_json_expected: Path
    command: tuple[str, ...]
    command_path: Path
    stdout_log: Path
    stderr_log: Path
    status: str = "planned"
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float | None = None
    return_code: int | None = None
    failure_reason: str = ""
    last_stdout_line: str = ""
    last_stderr_line: str = ""

    def to_plan_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_name": self.case_name,
            "biased_parameter": self.biased_parameter,
            "bias_ppm": self.bias_ppm,
            "fractional_bias": self.fractional_bias,
            "truth_value": self.truth_value,
            "reference_value": self.reference_value,
            "theta_reference_offset": self.theta_reference_offset,
            "theta_reference_offset_expression": self.theta_reference_offset_expression,
            "n_frames": self.n_frames,
            "noise": self.noise,
            "trace_seed": self.trace_seed,
            "render_seed": self.render_seed,
            "theta_keys": ",".join(self.theta_keys),
            "phi_ref": self.phi_ref,
            "reference_n_iter": self.reference_n_iter,
            "results_root": str(self.results_root),
            "case_root": str(self.case_root),
            "summary_json_expected": str(self.summary_json_expected),
            "command": " ".join(self.command),
            "command_path": str(self.command_path),
            "stdout_log": str(self.stdout_log),
            "stderr_log": str(self.stderr_log),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed_seconds,
            "return_code": self.return_code,
            "failure_reason": self.failure_reason,
            "last_stdout_line": self.last_stdout_line,
            "last_stderr_line": self.last_stderr_line,
        }


def _load_obs_subblock_study_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_obs_subblock_study_for_scalar_bias",
        DEFAULT_STUDY_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {DEFAULT_STUDY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_ppm_grid(raw: str | Sequence[float] | None) -> tuple[float, ...]:
    if raw is None:
        return DEFAULT_BIAS_PPM_GRID
    tokens = [part.strip() for part in str(raw).split(",")] if isinstance(raw, str) else [str(value).strip() for value in raw]
    values: list[float] = []
    seen: set[float] = set()
    for token in tokens:
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"Malformed PPM value {token!r}.") from exc
        key = float(value)
        if key in seen:
            raise ValueError(f"Duplicate PPM value: {token}.")
        seen.add(key)
        values.append(value)
    if not values:
        raise ValueError("bias PPM grid must contain at least one value.")
    return tuple(values)


def parse_parameter_list(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    values = DEFAULT_PARAMETERS if raw is None else tuple(part.strip() for part in str(raw).split(",")) if isinstance(raw, str) else tuple(str(part).strip() for part in raw)
    parameters = tuple(value for value in values if value)
    if not parameters:
        raise ValueError("parameter list must contain at least one parameter.")
    if len(set(parameters)) != len(parameters):
        raise ValueError("parameter list must not contain duplicates.")
    unsupported = [value for value in parameters if value not in DEFAULT_PARAMETERS]
    if unsupported:
        raise ValueError("Unsupported scalar bias parameter(s): " + ", ".join(unsupported))
    module = _load_obs_subblock_study_module()
    module.validate_schur_summary_theta_keys(parameters)
    return parameters


def slugify_parameter(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(label).strip())
    slug = slug.strip("_").lower()
    return slug or "unnamed"


def format_signed_ppm(ppm: float) -> str:
    if float(ppm) == 0.0:
        return "z0"
    prefix = "p" if ppm > 0.0 else "m"
    value = abs(float(ppm))
    if value.is_integer():
        body = str(int(value))
    else:
        body = ("%g" % value).replace(".", "p")
    return prefix + body


def noise_slug(noise: str) -> str:
    return "noiseless" if str(noise) == "disabled" else slugify_parameter(str(noise))


def compute_reference_offset(label: str, truth_value: float, ppm: float) -> tuple[float, float, str]:
    fractional = float(ppm) * 1.0e-6
    if label == "source.log_flux_total":
        if 1.0 + fractional <= 0.0:
            raise ValueError("source.log_flux_total PPM bias must be greater than -1e6.")
        offset = math.log10(1.0 + fractional)
        return truth_value + offset, offset, "log10(1 + ppm * 1e-6)"
    offset = float(truth_value) * fractional
    return float(truth_value) + offset, offset, "truth_value * ppm * 1e-6"


def resolve_truth_values_for_parameters(
    parameters: Sequence[str],
    *,
    trace_template: Path | None = None,
) -> dict[str, float]:
    from dluxshera.inference.observation_forecast import build_prior_mean_from_store
    from dluxshera.config.io import load_user_config
    from dluxshera.config.resolver import resolve_config
    from dluxshera.params.store import ParameterStore
    from dluxshera.systems.base import compose_forward_spec

    config_path = trace_template if trace_template is not None else DEFAULT_STUDY_SCRIPT.parent.parent / "recipes" / "observation_subblock_trace_template" / "subblock_trace_registration_iid_prescription.yaml"
    user_cfg = load_user_config(config_path=config_path.resolve(), system_preset=None, experiment_preset=None)
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    if not isinstance(system_cfg, Mapping):
        raise ValueError(f"Unable to resolve system block from {config_path}.")
    forward_spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    values = build_prior_mean_from_store(parameters, store=store)
    return {label: float(values[index]) for index, label in enumerate(parameters)}


def _case_seed(base_seed: int, policy: str, parameter_index: int, case_index: int) -> int:
    if policy == "same_trace_all_cases":
        return int(base_seed)
    if policy == "parameter_specific_trace":
        return int(base_seed) + 1009 * int(parameter_index)
    if policy == "case_specific_trace":
        return int(base_seed) + int(case_index)
    raise ValueError(f"Unsupported trace seed policy: {policy}")


def _effective_reference_n_iter(config: ScalarBiasStudyConfig, *, bias_ppm: float) -> int | None:
    value = config.reference_n_iter
    if float(bias_ppm) == 0.0 and config.zero_bias_reference_n_iter is not None:
        value = config.zero_bias_reference_n_iter
    return value


def build_schur_summary_command(config: ScalarBiasStudyConfig, spec_seed_args: Mapping[str, Any]) -> tuple[str, ...]:
    case_root = Path(spec_seed_args["case_root"])
    offset = float(spec_seed_args["theta_reference_offset"])
    parameter = str(spec_seed_args["biased_parameter"])
    command = [
        sys.executable,
        str(DEFAULT_STUDY_SCRIPT),
        "--mode",
        "schur_summary",
        "--case-root",
        str(case_root),
        "--n-frames",
        str(config.n_frames),
        "--noise",
        config.noise,
        "--theta-keys",
        ",".join(config.theta_keys),
        "--phi-ref",
        config.phi_ref,
        "--reference-schedule-kind",
        config.reference_schedule_kind,
        "--reference-schedule-warmup-steps",
        str(config.reference_schedule_warmup_steps),
        "--reference-schedule-start-factor",
        str(config.reference_schedule_start_factor),
        "--reference-diagnostics-profile",
        config.reference_diagnostics_profile,
        "--max-dense-dim",
        str(config.max_dense_dim),
        "--schur-curvature-method",
        config.schur_curvature_method,
        "--schur-damping",
        str(config.schur_damping),
        "--schur-frame-quality-policy",
        config.schur_frame_quality_policy,
        "--trace-seed",
        str(spec_seed_args["trace_seed"]),
        "--render-seed",
        str(spec_seed_args["render_seed"]),
    ]
    if config.trace_template is not None:
        command.extend(["--trace-template", str(config.trace_template)])
    effective_reference_n_iter = _effective_reference_n_iter(
        config,
        bias_ppm=float(spec_seed_args.get("bias_ppm", 0.0)),
    )
    if effective_reference_n_iter is not None:
        command.extend(["--reference-n-iter", str(int(effective_reference_n_iter))])
    if offset != 0.0:
        command.extend(["--theta-reference-offset", f"{parameter}={offset:.17g}"])
    return tuple(command)


def build_case_plan(
    config: ScalarBiasStudyConfig,
    truth_values: Mapping[str, float],
) -> list[ScalarBiasCaseSpec]:
    cases: list[ScalarBiasCaseSpec] = []
    commands_root = config.run_root / "commands"
    logs_root = config.run_root / "logs"
    case_id = 0
    for parameter_index, parameter in enumerate(config.parameters):
        truth = float(truth_values[parameter])
        for ppm in config.bias_ppm_grid:
            fractional = float(ppm) * 1.0e-6
            reference, offset, expression = compute_reference_offset(parameter, truth, ppm)
            name = f"{slugify_parameter(parameter)}_ppm_{format_signed_ppm(ppm)}_{config.n_frames}f_{noise_slug(config.noise)}"
            case_root = config.run_root / "cases" / name
            summary_json = case_root / "study" / "schur_summary" / "subblock_summary.json"
            trace_seed = _case_seed(config.seed, config.trace_seed_policy, parameter_index, case_id)
            seed_args = {
                "case_root": case_root,
                "biased_parameter": parameter,
                "theta_reference_offset": offset,
                "trace_seed": trace_seed,
                "render_seed": config.seed,
                "bias_ppm": float(ppm),
            }
            command = build_schur_summary_command(config, seed_args)
            reference_n_iter = _effective_reference_n_iter(config, bias_ppm=float(ppm))
            command_path = commands_root / f"{name}.sh"
            stdout_log = logs_root / f"{name}.stdout.log"
            stderr_log = logs_root / f"{name}.stderr.log"
            cases.append(
                ScalarBiasCaseSpec(
                    case_id=case_id,
                    case_name=name,
                    biased_parameter=parameter,
                    bias_ppm=float(ppm),
                    fractional_bias=fractional,
                    truth_value=truth,
                    reference_value=reference,
                    theta_reference_offset=offset,
                    theta_reference_offset_expression=expression,
                    n_frames=config.n_frames,
                    noise=config.noise,
                    trace_seed=trace_seed,
                    render_seed=config.seed,
                    theta_keys=config.theta_keys,
                    phi_ref=config.phi_ref,
                    reference_n_iter=reference_n_iter,
                    results_root=config.results_root,
                    case_root=case_root,
                    summary_json_expected=summary_json,
                    command=command,
                    command_path=command_path,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                )
            )
            case_id += 1
    return cases


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = [dict(row) for row in rows]
    if columns is None:
        fieldnames: list[str] = []
        for row in row_list:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    else:
        fieldnames = list(columns)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def _nan() -> float:
    return float("nan")


def _render_command_script(command: Sequence[str]) -> str:
    quoted = " ".join(shlex.quote(token) for token in command)
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        f"PYTHONPATH=src {quoted}\n"
    )


def write_command_file(spec: ScalarBiasCaseSpec) -> None:
    spec.command_path.parent.mkdir(parents=True, exist_ok=True)
    spec.command_path.write_text(_render_command_script(spec.command), encoding="utf-8")


def _last_nonempty_line(path: Path) -> str:
    if not path.exists():
        return ""
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            candidate = line.strip()
            if candidate:
                last = candidate
    return last


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or not math.isfinite(float(denominator)):
        return _nan()
    return float(numerator) / float(denominator)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _residual_stats(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": _nan(), "std": _nan(), "rms": _nan(), "max_abs": _nan()}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "rms": float(np.sqrt(np.mean(np.square(arr)))),
        "max_abs": float(np.max(np.abs(arr))),
    }


def _path_from_summary(summary: Mapping[str, Any], *keys: str) -> Path | None:
    current: Any = summary
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    if not isinstance(current, str) or not current:
        return None
    return Path(current).expanduser().resolve()


def _find_artifacts(spec: ScalarBiasCaseSpec) -> dict[str, Path | None]:
    study_root = spec.case_root / "study" / "schur_summary"
    summary_path = study_root / "summary.json"
    payload = _read_json(summary_path) if summary_path.exists() else {}
    schur_summary = payload.get("schur_summary", {}) if isinstance(payload, Mapping) else {}
    recovered = payload.get("recovered_inference", {}) if isinstance(payload, Mapping) else {}
    artifacts = schur_summary.get("artifacts", {}) if isinstance(schur_summary, Mapping) else {}
    render_inputs = payload.get("render_inputs", {}) if isinstance(payload, Mapping) else {}
    return {
        "study_summary_json": summary_path if summary_path.exists() else None,
        "subblock_summary_json": Path(artifacts.get("subblock_summary_json", spec.summary_json_expected)).resolve() if isinstance(artifacts, Mapping) else spec.summary_json_expected,
        "schur_diagnostics_json": study_root / "schur_diagnostics.json",
        "combined_curvature_diagnostics_json": study_root / "combined_curvature_diagnostics.json",
        "surrogate_validation_csv": study_root / "local_surrogate_validation.csv",
        "manifest_json": Path(recovered["manifest_json"]).resolve() if isinstance(recovered, Mapping) and recovered.get("manifest_json") else None,
        "recovered_trace_csv": Path(recovered["recovered_trace_csv"]).resolve() if isinstance(recovered, Mapping) and recovered.get("recovered_trace_csv") else None,
        "truth_comparison_csv": None,
        "truth_trace_csv": Path(render_inputs["truth_trace"]).resolve() if isinstance(render_inputs, Mapping) and render_inputs.get("truth_trace") else None,
    }


def _truth_comparison_from_manifest(manifest_path: Path | None) -> Path | None:
    if manifest_path is None or not manifest_path.exists():
        return None
    payload = _read_json(manifest_path)
    artifacts = payload.get("artifacts", {})
    for key in ("truth_comparison_csv", "comparison_csv"):
        value = artifacts.get(key) if isinstance(artifacts, Mapping) else None
        if isinstance(value, str) and value:
            path = Path(value)
            return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()
    matches = sorted(manifest_path.parent.glob("*truth_comparison*.csv"))
    return matches[0].resolve() if matches else None


def _lookup_nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _extract_chi2_from_manifest(manifest_path: Path | None) -> dict[str, float]:
    values = {
        "initial_block_reduced_chi2": _nan(),
        "final_block_reduced_chi2": _nan(),
    }
    if manifest_path is None or not manifest_path.exists():
        return values
    payload = _read_json(manifest_path)
    initial = _lookup_nested(payload, ("metrics", "chi2", "initial_model", "block_reduced_chi2"))
    if initial is None:
        initial = _lookup_nested(payload, ("diagnostics", "chi2", "initial_model", "block_reduced_chi2"))
    final = _lookup_nested(payload, ("metrics", "chi2", "final_model", "block_reduced_chi2"))
    if final is None:
        final = _lookup_nested(payload, ("diagnostics", "chi2", "final_model", "block_reduced_chi2"))
    if isinstance(initial, (int, float)) and math.isfinite(float(initial)):
        values["initial_block_reduced_chi2"] = float(initial)
    if isinstance(final, (int, float)) and math.isfinite(float(final)):
        values["final_block_reduced_chi2"] = float(final)
    return values


def _summary_theta_ref_map(summary_payload: Mapping[str, Any]) -> dict[str, float]:
    labels = tuple(str(label) for label in summary_payload.get("theta_labels", ()))
    theta_ref = np.asarray(summary_payload.get("theta_ref", ()), dtype=float)
    if len(labels) != int(theta_ref.size):
        raise ValueError("Summary theta_labels/theta_ref shape mismatch.")
    return {label: float(theta_ref[index]) for index, label in enumerate(labels)}


def _effective_theta_context(
    spec: ScalarBiasCaseSpec,
    summary_payload: Mapping[str, Any],
) -> dict[str, Any]:
    actual_reference = _summary_theta_ref_map(summary_payload)
    effective_truth = dict(actual_reference)
    actual_biased_reference = float(actual_reference[spec.biased_parameter])
    effective_truth[spec.biased_parameter] = actual_biased_reference - float(spec.theta_reference_offset)
    planned_reference = float(spec.reference_value)
    planned_truth = float(spec.truth_value)
    planned_minus_actual = planned_reference - actual_biased_reference
    abs_diff = abs(planned_minus_actual)
    rel_diff = _safe_ratio(abs_diff, abs(actual_biased_reference))
    return {
        "actual_reference_by_label": actual_reference,
        "effective_truth_by_label": effective_truth,
        "planned_truth_value": planned_truth,
        "planned_reference_value": planned_reference,
        "planned_theta_reference_offset": float(spec.theta_reference_offset),
        "actual_reference_value": actual_biased_reference,
        "effective_truth_value": float(effective_truth[spec.biased_parameter]),
        "reference_value_source": "summary_theta_ref",
        "truth_value_source": (
            "actual_reference_minus_injected_offset"
            if spec.theta_reference_offset != 0.0
            else "actual_reference"
        ),
        "planned_minus_actual_reference": planned_minus_actual,
        "planned_actual_reference_abs_diff": abs_diff,
        "planned_actual_reference_rel_diff": rel_diff,
        "planned_actual_reference_mismatch_warning": bool(abs_diff > 1.0e-12),
    }


def compute_registration_absorption_metrics(
    spec: ScalarBiasCaseSpec,
    artifacts: Mapping[str, Path | None],
    study_payload: Mapping[str, Any],
    theta_context: Mapping[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": spec.case_id,
        "case_name": spec.case_name,
        "biased_parameter": spec.biased_parameter,
        "bias_ppm": spec.bias_ppm,
        "fractional_bias": spec.fractional_bias,
        "truth_value": theta_context["effective_truth_value"],
        "reference_value": theta_context["actual_reference_value"],
        "theta_reference_offset": spec.theta_reference_offset,
        "planned_truth_value": theta_context["planned_truth_value"],
        "planned_reference_value": theta_context["planned_reference_value"],
        "planned_theta_reference_offset": theta_context["planned_theta_reference_offset"],
        "actual_reference_value": theta_context["actual_reference_value"],
        "effective_truth_value": theta_context["effective_truth_value"],
        "reference_value_source": theta_context["reference_value_source"],
        "truth_value_source": theta_context["truth_value_source"],
        "planned_minus_actual_reference": theta_context["planned_minus_actual_reference"],
        "planned_actual_reference_abs_diff": theta_context["planned_actual_reference_abs_diff"],
        "planned_actual_reference_rel_diff": theta_context["planned_actual_reference_rel_diff"],
        "planned_actual_reference_mismatch_warning": theta_context["planned_actual_reference_mismatch_warning"],
        "summary_json": str(artifacts.get("subblock_summary_json") or ""),
        "recovered_trace_csv": str(artifacts.get("recovered_trace_csv") or ""),
        "truth_comparison_csv": "",
        "status": "ok",
    }
    truth_comparison = artifacts.get("truth_comparison_csv") or _truth_comparison_from_manifest(artifacts.get("manifest_json"))
    if truth_comparison is not None:
        row["truth_comparison_csv"] = str(truth_comparison)

    residuals = {
        "dx": [],
        "dy": [],
        "dpa": [],
    }
    if truth_comparison is not None and truth_comparison.exists():
        for csv_row in _load_csv_rows(truth_comparison):
            candidates = {
                "dx": ("source.x_position_as_residual", "x_position_as_residual", "dx_as"),
                "dy": ("source.y_position_as_residual", "y_position_as_residual", "dy_as"),
                "dpa": ("source.position_angle_deg_residual", "position_angle_deg_residual", "dpa_deg"),
            }
            for key, names in candidates.items():
                for name in names:
                    if name in csv_row and csv_row[name] != "":
                        residuals[key].append(float(csv_row[name]))
                        break
    elif artifacts.get("recovered_trace_csv") is not None and artifacts.get("truth_trace_csv") is not None:
        recovered_rows = _load_csv_rows(Path(artifacts["recovered_trace_csv"]))
        truth_rows = {row.get("frame_index", str(index)): row for index, row in enumerate(_load_csv_rows(Path(artifacts["truth_trace_csv"])))}
        key_pairs = {
            "dx": ("source.x_position_as", "source.x_position_as"),
            "dy": ("source.y_position_as", "source.y_position_as"),
            "dpa": ("source.position_angle_deg", "source.position_angle_deg"),
        }
        for index, recovered_row in enumerate(recovered_rows):
            truth_row = truth_rows.get(recovered_row.get("frame_index", str(index)))
            if truth_row is None:
                continue
            for out_key, (rec_key, true_key) in key_pairs.items():
                if rec_key in recovered_row and true_key in truth_row:
                    residuals[out_key].append(float(recovered_row[rec_key]) - float(truth_row[true_key]))

    for prefix, suffix in (("dx", "as"), ("dy", "as"), ("dpa", "deg")):
        stats = _residual_stats(residuals[prefix])
        row[f"mean_d{prefix[1:]}_{suffix}" if prefix != "dpa" else "mean_dpa_deg"] = stats["mean"]
        row[f"std_d{prefix[1:]}_{suffix}" if prefix != "dpa" else "std_dpa_deg"] = stats["std"]
        row[f"rms_d{prefix[1:]}_{suffix}" if prefix != "dpa" else "rms_dpa_deg"] = stats["rms"]
        row[f"max_abs_d{prefix[1:]}_{suffix}" if prefix != "dpa" else "max_abs_dpa_deg"] = stats["max_abs"]
    schur_summary = study_payload.get("schur_summary", {}) if isinstance(study_payload, Mapping) else {}
    frame_quality = schur_summary.get("frame_quality", {}) if isinstance(schur_summary, Mapping) else {}
    chi2_values = _extract_chi2_from_manifest(artifacts.get("manifest_json"))
    row["initial_block_reduced_chi2"] = chi2_values["initial_block_reduced_chi2"]
    if not math.isfinite(float(row["initial_block_reduced_chi2"])):
        row["initial_block_reduced_chi2"] = schur_summary.get("initial_block_reduced_chi2", _nan()) if isinstance(schur_summary, Mapping) else _nan()
    row["final_block_reduced_chi2"] = chi2_values["final_block_reduced_chi2"]
    if not math.isfinite(float(row["final_block_reduced_chi2"])):
        row["final_block_reduced_chi2"] = schur_summary.get("final_block_reduced_chi2", frame_quality.get("block_reduced_chi2", _nan()) if isinstance(frame_quality, Mapping) else _nan())
    row["max_final_frame_reduced_chi2"] = frame_quality.get("max_frame_reduced_chi2", _nan()) if isinstance(frame_quality, Mapping) else _nan()
    row["median_final_frame_reduced_chi2"] = frame_quality.get("median_frame_reduced_chi2", _nan()) if isinstance(frame_quality, Mapping) else _nan()
    row["frame_quality_bad_frame_count"] = frame_quality.get("bad_frame_count", _nan()) if isinstance(frame_quality, Mapping) else _nan()
    row["frame_quality_good_frame_count"] = frame_quality.get("good_frame_count", _nan()) if isinstance(frame_quality, Mapping) else _nan()
    return row


def run_single_summary_observation_update(summary_json: Path) -> tuple[Any, np.ndarray, np.ndarray]:
    summary = load_subblock_summary(summary_json)
    prior_sigma = build_default_prior_sigma(summary.theta_labels)
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=summary.theta_labels,
        mean=summary.theta_ref,
        sigma=prior_sigma,
        metadata={"prior_mean_source": "summary_theta_ref"},
    )
    update = update_observation_belief(prior, [summary])
    return update, np.asarray(update.posterior.mean, dtype=float), update.posterior.sigma()


def _diagnostic_value(summary_payload: Mapping[str, Any], key: str) -> Any:
    for parent_key in ("summary_diagnostics", "diagnostics"):
        parent = summary_payload.get(parent_key)
        if isinstance(parent, Mapping) and key in parent:
            return parent[key]
    return None


def _surrogate_validation_summary(path: Path | None) -> tuple[bool, float, str]:
    if path is None or not path.exists():
        return False, _nan(), ""
    rows = _load_csv_rows(path)
    errors: list[float] = []
    statuses: list[str] = []
    for row in rows:
        for key in ("abs_error", "absolute_error", "error_abs"):
            if key in row and row[key] != "":
                errors.append(float(row[key]))
                break
        if "status" in row and row["status"]:
            statuses.append(row["status"])
    return True, float(np.max(errors)) if errors else _nan(), ";".join(sorted(set(statuses)))


def _theta_reference_consistency_passed(summary_payload: Mapping[str, Any]) -> Any:
    metadata = summary_payload.get("metadata")
    if isinstance(metadata, Mapping):
        direct = metadata.get("theta_reference_consistency_passed")
        if isinstance(direct, bool):
            return direct
        nested = metadata.get("theta_reference_consistency")
        if isinstance(nested, Mapping):
            nested_value = nested.get("passed")
            if isinstance(nested_value, bool):
                return nested_value
    top = summary_payload.get("theta_reference_consistency")
    if isinstance(top, Mapping):
        top_value = top.get("passed")
        if isinstance(top_value, bool):
            return top_value
    return ""


def compute_correction_response_metrics(
    spec: ScalarBiasCaseSpec,
    summary_payload: Mapping[str, Any],
    posterior_mean: np.ndarray,
    posterior_sigma: np.ndarray,
    update: Any,
    artifacts: Mapping[str, Path | None],
    theta_context: Mapping[str, Any],
) -> dict[str, Any]:
    labels = tuple(str(label) for label in summary_payload["theta_labels"])
    idx = labels.index(spec.biased_parameter)
    actual_reference = float(theta_context["actual_reference_value"])
    effective_truth = float(theta_context["effective_truth_value"])
    posterior_value = float(posterior_mean[idx])
    posterior_shift = posterior_value - actual_reference
    expected_shift = effective_truth - actual_reference
    posterior_error = posterior_value - effective_truth
    injected_bias = actual_reference - effective_truth
    surrogate_available, surrogate_max_error, surrogate_status = _surrogate_validation_summary(artifacts.get("surrogate_validation_csv"))
    correction_fraction = _safe_ratio(posterior_shift, expected_shift)
    residual_fraction = _safe_ratio(abs(posterior_error), abs(injected_bias))
    return {
        "case_id": spec.case_id,
        "case_name": spec.case_name,
        "biased_parameter": spec.biased_parameter,
        "bias_ppm": spec.bias_ppm,
        "truth_value": effective_truth,
        "reference_value": actual_reference,
        "theta_reference_offset": spec.theta_reference_offset,
        "planned_truth_value": theta_context["planned_truth_value"],
        "planned_reference_value": theta_context["planned_reference_value"],
        "planned_theta_reference_offset": theta_context["planned_theta_reference_offset"],
        "actual_reference_value": theta_context["actual_reference_value"],
        "effective_truth_value": theta_context["effective_truth_value"],
        "reference_value_source": theta_context["reference_value_source"],
        "truth_value_source": theta_context["truth_value_source"],
        "planned_minus_actual_reference": theta_context["planned_minus_actual_reference"],
        "planned_actual_reference_abs_diff": theta_context["planned_actual_reference_abs_diff"],
        "planned_actual_reference_rel_diff": theta_context["planned_actual_reference_rel_diff"],
        "planned_actual_reference_mismatch_warning": theta_context["planned_actual_reference_mismatch_warning"],
        "posterior_value_biased_parameter": posterior_value,
        "posterior_shift_biased_parameter": posterior_shift,
        "expected_shift_to_truth": expected_shift,
        "posterior_error_biased_parameter": posterior_error,
        "correction_fraction_biased_parameter": correction_fraction,
        "residual_fraction_biased_parameter": residual_fraction,
        "posterior_sigma_biased_parameter": float(posterior_sigma[idx]),
        "posterior_error_over_sigma_biased_parameter": _safe_ratio(posterior_error, float(posterior_sigma[idx])),
        "moves_biased_parameter_toward_truth": bool(abs(posterior_error) < abs(injected_bias)) if injected_bias != 0.0 else "",
        "posterior_error_sign_flip": bool(injected_bias != 0.0 and posterior_error != 0.0 and math.copysign(1.0, injected_bias) != math.copysign(1.0, posterior_error)),
        "score_norm": _diagnostic_value(summary_payload, "score_norm"),
        "reduced_information_rank": _diagnostic_value(summary_payload, "rank_estimate"),
        "reduced_information_min_eigenvalue": _diagnostic_value(summary_payload, "min_eigenvalue"),
        "reduced_information_condition_number": _diagnostic_value(summary_payload, "condition_number"),
        "update_solve_method": update.posterior.metadata.get("solve_method", ""),
        "surrogate_validation_available": surrogate_available,
        "surrogate_validation_max_abs_error": surrogate_max_error,
        "surrogate_validation_status": surrogate_status,
        "theta_reference_consistency_passed": _theta_reference_consistency_passed(summary_payload),
        "status": "ok",
    }


def compute_science_leakage_metrics(
    spec: ScalarBiasCaseSpec,
    summary_payload: Mapping[str, Any],
    posterior_mean: np.ndarray,
    posterior_sigma: np.ndarray,
    theta_context: Mapping[str, Any],
) -> dict[str, Any]:
    labels = tuple(str(label) for label in summary_payload["theta_labels"])
    idx = labels.index("source.separation_as")
    theta_ref = np.asarray(summary_payload["theta_ref"], dtype=float)
    actual_reference_by_label = theta_context["actual_reference_by_label"]
    effective_truth_by_label = theta_context["effective_truth_by_label"]
    truth = float(effective_truth_by_label["source.separation_as"])
    planned_truth = (
        float(theta_context["planned_truth_value"])
        if spec.biased_parameter == "source.separation_as"
        else float(actual_reference_by_label["source.separation_as"])
    )
    reference = float(theta_ref[idx])
    posterior = float(posterior_mean[idx])
    sigma = float(posterior_sigma[idx])
    shift = posterior - reference
    error = posterior - truth
    injected = reference - truth
    expected = truth - reference
    return {
        "case_id": spec.case_id,
        "case_name": spec.case_name,
        "biased_parameter": spec.biased_parameter,
        "bias_ppm": spec.bias_ppm,
        "truth_separation_as": truth,
        "reference_separation_as": reference,
        "planned_truth_separation_as": planned_truth,
        "actual_reference_separation_as": float(actual_reference_by_label["source.separation_as"]),
        "truth_value_source": "effective_truth_from_summary_theta_ref",
        "reference_value_source": "summary_theta_ref",
        "planned_minus_actual_reference_separation_as": planned_truth - float(actual_reference_by_label["source.separation_as"]),
        "posterior_separation_as": posterior,
        "posterior_separation_shift_as": shift,
        "posterior_separation_error_as": error,
        "posterior_separation_error_microas": error * 1.0e6,
        "posterior_separation_sigma_as": sigma,
        "posterior_separation_sigma_microas": sigma * 1.0e6,
        "posterior_separation_error_over_sigma": _safe_ratio(error, sigma),
        "separation_correction_fraction_if_biased": _safe_ratio(shift, expected) if spec.biased_parameter == "source.separation_as" else _nan(),
        "separation_residual_fraction_if_biased": _safe_ratio(abs(error), abs(injected)) if spec.biased_parameter == "source.separation_as" else _nan(),
        "moves_separation_toward_truth": bool(abs(error) < abs(injected)) if spec.biased_parameter == "source.separation_as" and injected != 0.0 else "",
        "status": "ok",
    }


def compute_posterior_by_parameter_rows(
    spec: ScalarBiasCaseSpec,
    summary_payload: Mapping[str, Any],
    posterior_mean: np.ndarray,
    posterior_sigma: np.ndarray,
    theta_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    labels = tuple(str(label) for label in summary_payload["theta_labels"])
    theta_ref = np.asarray(summary_payload["theta_ref"], dtype=float)
    effective_truth_by_label = theta_context["effective_truth_by_label"]
    actual_reference_by_label = theta_context["actual_reference_by_label"]
    planned_truth = float(theta_context["planned_truth_value"])
    planned_reference = float(theta_context["planned_reference_value"])
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        truth = float(effective_truth_by_label[label])
        reference = float(theta_ref[index])
        posterior = float(posterior_mean[index])
        sigma = float(posterior_sigma[index])
        injected = reference - truth
        expected = truth - reference
        error = posterior - truth
        shift = posterior - reference
        rows.append(
            {
                "case_id": spec.case_id,
                "case_name": spec.case_name,
                "biased_parameter": spec.biased_parameter,
                "bias_ppm": spec.bias_ppm,
                "theta_label": label,
                "truth_value": truth,
                "reference_value": reference,
                "planned_truth_value": planned_truth if label == spec.biased_parameter else float(actual_reference_by_label[label]),
                "planned_reference_value": planned_reference if label == spec.biased_parameter else float(actual_reference_by_label[label]),
                "actual_reference_value": float(actual_reference_by_label[label]),
                "effective_truth_value": truth,
                "reference_value_source": "summary_theta_ref",
                "truth_value_source": (
                    "actual_reference_minus_injected_offset"
                    if label == spec.biased_parameter and spec.theta_reference_offset != 0.0
                    else "actual_reference"
                ),
                "planned_minus_actual_reference": (
                    planned_reference - float(actual_reference_by_label[label])
                    if label == spec.biased_parameter
                    else 0.0
                ),
                "posterior_mean": posterior,
                "posterior_shift": shift,
                "posterior_error": error,
                "posterior_sigma": sigma,
                "posterior_error_over_sigma": _safe_ratio(error, sigma),
                "injected_bias": injected,
                "correction_fraction": _safe_ratio(shift, expected),
                "residual_fraction": _safe_ratio(abs(error), abs(injected)),
            }
        )
    return rows


def run_case(spec: ScalarBiasCaseSpec, *, quiet: bool = False, resume: bool = False) -> ScalarBiasCaseSpec:
    if resume and spec.summary_json_expected.exists():
        return spec.__class__(**{**spec.__dict__, "status": "skipped_existing", "return_code": 0})
    write_command_file(spec)
    spec.stdout_log.parent.mkdir(parents=True, exist_ok=True)
    spec.stderr_log.parent.mkdir(parents=True, exist_ok=True)
    started = now_iso_local_ms()
    t0 = time.time()
    if not quiet:
        print(
            f"[scalar-bias] running {spec.case_name} "
            f"stdout={spec.stdout_log} stderr={spec.stderr_log}",
            flush=True,
        )
    with spec.stdout_log.open("w", encoding="utf-8") as stdout_handle, spec.stderr_log.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            spec.command,
            cwd=REPO_ROOT,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    finished = now_iso_local_ms()
    elapsed_seconds = float(time.time() - t0)
    last_stdout_line = _last_nonempty_line(spec.stdout_log)
    last_stderr_line = _last_nonempty_line(spec.stderr_log)
    status = "ok" if completed.returncode == 0 and spec.summary_json_expected.exists() else "failed"
    reason = ""
    if status != "ok":
        reason = (
            f"return_code={completed.returncode}; summary_exists={spec.summary_json_expected.exists()}; "
            f"stdout_log={spec.stdout_log}; stderr_log={spec.stderr_log}; "
            f"stderr_tail={last_stderr_line or '<empty>'}"
        )
        if not quiet:
            print(
                f"[scalar-bias] failed {spec.case_name} return_code={completed.returncode} "
                f"stderr_tail={last_stderr_line or '<empty>'}",
                flush=True,
            )
    return spec.__class__(
        **{
            **spec.__dict__,
            "status": status,
            "started_at": started,
            "finished_at": finished,
            "elapsed_seconds": elapsed_seconds,
            "return_code": int(completed.returncode),
            "failure_reason": reason,
            "last_stdout_line": last_stdout_line,
            "last_stderr_line": last_stderr_line,
        }
    )


def aggregate_cases(
    config: ScalarBiasStudyConfig,
    cases: Sequence[ScalarBiasCaseSpec],
    truth_values: Mapping[str, float],
) -> dict[str, Any]:
    nuisance_rows: list[dict[str, Any]] = []
    correction_rows: list[dict[str, Any]] = []
    science_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    completed = 0
    failed = 0
    for spec in cases:
        row = spec.to_plan_row()
        try:
            artifacts = _find_artifacts(spec)
            summary_json = artifacts.get("subblock_summary_json")
            if summary_json is None or not Path(summary_json).exists():
                raise FileNotFoundError(f"Missing subblock summary: {summary_json}")
            study_payload = _read_json(Path(artifacts["study_summary_json"])) if artifacts.get("study_summary_json") else {}
            summary_payload = _read_json(Path(summary_json))
            theta_context = _effective_theta_context(spec, summary_payload)
            update, posterior_mean, posterior_sigma = run_single_summary_observation_update(Path(summary_json))
            nuisance_rows.append(compute_registration_absorption_metrics(spec, artifacts, study_payload, theta_context))
            correction_rows.append(compute_correction_response_metrics(spec, summary_payload, posterior_mean, posterior_sigma, update, artifacts, theta_context))
            science_rows.append(compute_science_leakage_metrics(spec, summary_payload, posterior_mean, posterior_sigma, theta_context))
            posterior_rows.extend(compute_posterior_by_parameter_rows(spec, summary_payload, posterior_mean, posterior_sigma, theta_context))
            row["aggregate_status"] = "ok"
            completed += 1
        except Exception as exc:
            failed += 1
            row["aggregate_status"] = "failed"
            row["failure_reason"] = str(exc)
            base = {
                "case_id": spec.case_id,
                "case_name": spec.case_name,
                "biased_parameter": spec.biased_parameter,
                "bias_ppm": spec.bias_ppm,
                "status": "failed",
            }
            nuisance_rows.append(base)
            correction_rows.append(base)
            science_rows.append(base)
        status_rows.append(row)

    run_root = config.run_root
    paths = {
        "case_status_csv": run_root / "case_status.csv",
        "nuisance_absorption_sensitivity_csv": run_root / "nuisance_absorption_sensitivity.csv",
        "summary_correction_response_csv": run_root / "summary_correction_response.csv",
        "science_leakage_matrix_csv": run_root / "science_leakage_matrix.csv",
        "posterior_by_parameter_csv": run_root / "posterior_by_parameter.csv",
        "aggregate_summary_json": run_root / "aggregate_summary.json",
        "commands_root": run_root / "commands",
        "logs_root": run_root / "logs",
    }
    write_csv_rows(paths["case_status_csv"], status_rows)
    write_csv_rows(paths["nuisance_absorption_sensitivity_csv"], nuisance_rows, NUISANCE_COLUMNS)
    write_csv_rows(paths["summary_correction_response_csv"], correction_rows, CORRECTION_COLUMNS)
    write_csv_rows(paths["science_leakage_matrix_csv"], science_rows, SCIENCE_COLUMNS)
    write_csv_rows(paths["posterior_by_parameter_csv"], posterior_rows, POSTERIOR_COLUMNS)
    summary = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "run_name": config.run_name,
        "run_root": str(run_root.resolve()),
        "counts": {
            "planned": len(cases),
            "aggregated_ok": completed,
            "aggregate_failed": failed,
        },
        "config": config_to_dict(config),
        "truth_values": dict(truth_values),
        "artifacts": {key: str(path.resolve()) for key, path in paths.items()},
    }
    write_json(paths["aggregate_summary_json"], summary)
    return summary


def config_to_dict(config: ScalarBiasStudyConfig) -> dict[str, Any]:
    payload = dict(config.__dict__)
    for key in ("results_root", "trace_template"):
        value = payload.get(key)
        payload[key] = None if value is None else str(value)
    payload["parameters"] = list(config.parameters)
    payload["theta_keys"] = list(config.theta_keys)
    payload["bias_ppm_grid"] = list(config.bias_ppm_grid)
    return payload


def write_plan(config: ScalarBiasStudyConfig, cases: Sequence[ScalarBiasCaseSpec], truth_values: Mapping[str, float]) -> None:
    config.run_root.mkdir(parents=True, exist_ok=True)
    (config.run_root / "commands").mkdir(parents=True, exist_ok=True)
    (config.run_root / "logs").mkdir(parents=True, exist_ok=True)
    for case in cases:
        write_command_file(case)
    rows = [case.to_plan_row() for case in cases]
    write_csv_rows(config.run_root / "run_plan.csv", rows)
    write_json(
        config.run_root / "run_plan.json",
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "created_at": now_iso_local_ms(),
            "config": config_to_dict(config),
            "truth_values": dict(truth_values),
            "seed_policy": {
                "trace_seed_policy": config.trace_seed_policy,
                "seed": config.seed,
                "implemented_policies": list(SUPPORTED_TRACE_SEED_POLICIES),
            },
            "schur_defaults": {
                "phi_ref": config.phi_ref,
                "max_dense_dim": config.max_dense_dim,
                "schur_curvature_method": config.schur_curvature_method,
                "schur_damping": config.schur_damping,
            "schur_frame_quality_policy": config.schur_frame_quality_policy,
            "reference_schedule_kind": config.reference_schedule_kind,
            "reference_schedule_warmup_steps": config.reference_schedule_warmup_steps,
            "reference_schedule_start_factor": config.reference_schedule_start_factor,
            "reference_diagnostics_profile": config.reference_diagnostics_profile,
            "reference_n_iter": config.reference_n_iter,
            "zero_bias_reference_n_iter": config.zero_bias_reference_n_iter,
        },
            "cases": rows,
            "paths": {
                "commands_root": str((config.run_root / "commands").resolve()),
                "logs_root": str((config.run_root / "logs").resolve()),
            },
        },
    )


def write_plots(config: ScalarBiasStudyConfig) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    run_root = config.run_root
    plot_root = run_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)

    nuisance = _load_csv_rows(run_root / "nuisance_absorption_sensitivity.csv")
    correction = _load_csv_rows(run_root / "summary_correction_response.csv")
    science = _load_csv_rows(run_root / "science_leakage_matrix.csv")
    for parameter in config.parameters:
        slug = slugify_parameter(parameter)
        nrows = [row for row in nuisance if row.get("biased_parameter") == parameter and row.get("status") == "ok"]
        if nrows:
            xs = [float(row["bias_ppm"]) for row in nrows]
            plt.figure(figsize=(7, 4))
            for key, label in (("mean_dx_as", "mean dx"), ("mean_dy_as", "mean dy"), ("mean_dpa_deg", "mean dpa")):
                plt.plot(xs, [float(row[key]) for row in nrows], marker="o", label=label)
            plt.axhline(0.0, color="black", linewidth=0.8)
            plt.xlabel("Bias PPM")
            plt.ylabel("Recovered minus truth")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_root / f"registration_bias_vs_ppm_{slug}.png", dpi=150)
            plt.close()
        crows = [row for row in correction if row.get("biased_parameter") == parameter and row.get("status") == "ok"]
        if crows:
            xs = [float(row["bias_ppm"]) for row in crows]
            plt.figure(figsize=(7, 4))
            for key, label in (("correction_fraction_biased_parameter", "correction fraction"), ("residual_fraction_biased_parameter", "residual fraction"), ("posterior_error_over_sigma_biased_parameter", "error / sigma")):
                plt.plot(xs, [float(row[key]) for row in crows], marker="o", label=label)
            plt.axhline(0.0, color="black", linewidth=0.8)
            plt.xlabel("Bias PPM")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_root / f"correction_fraction_vs_ppm_{slug}.png", dpi=150)
            plt.close()
        srows = [row for row in science if row.get("biased_parameter") == parameter and row.get("status") == "ok"]
        if srows:
            xs = [float(row["bias_ppm"]) for row in srows]
            plt.figure(figsize=(7, 4))
            plt.plot(xs, [float(row["posterior_separation_error_microas"]) for row in srows], marker="o", label="error microas")
            plt.plot(xs, [float(row["posterior_separation_error_over_sigma"]) for row in srows], marker="o", label="error / sigma")
            plt.axhline(0.0, color="black", linewidth=0.8)
            plt.xlabel("Bias PPM")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_root / f"separation_leakage_vs_ppm_{slug}.png", dpi=150)
            plt.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--parameters", default=",".join(DEFAULT_PARAMETERS))
    parser.add_argument("--theta-keys", default=",".join(DEFAULT_PARAMETERS))
    parser.add_argument("--bias-ppm-grid", default=",".join("%g" % value for value in DEFAULT_BIAS_PPM_GRID))
    parser.add_argument("--n-frames", type=int, default=20)
    parser.add_argument("--noise", default="disabled")
    parser.add_argument("--trace-template", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRACE_SEED)
    parser.add_argument("--trace-seed-policy", choices=SUPPORTED_TRACE_SEED_POLICIES, default="same_trace_all_cases")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--plots", dest="plots", action="store_true", default=False)
    parser.add_argument("--no-plots", dest="plots", action="store_false")
    parser.add_argument("--reference-n-iter", type=int, default=None)
    parser.add_argument("--zero-bias-reference-n-iter", type=int, default=None)
    parser.add_argument("--belief-update-script", type=Path, default=None, help="Reserved for subprocess-based update experiments; direct helpers are used by default.")
    return parser


def _normalize_argv(argv: list[str] | None) -> list[str] | None:
    """Allow comma-separated negative PPM grids after ``--bias-ppm-grid``."""

    if argv is None:
        argv = sys.argv[1:]
    normalized: list[str] = []
    skip_next = False
    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if token == "--bias-ppm-grid" and index + 1 < len(argv):
            normalized.append(f"--bias-ppm-grid={argv[index + 1]}")
            skip_next = True
        else:
            normalized.append(token)
    return normalized


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(_normalize_argv(argv))
    parameters = parse_parameter_list(args.parameters)
    theta_keys = parse_parameter_list(args.theta_keys)
    missing = [parameter for parameter in parameters if parameter not in theta_keys]
    if missing:
        raise ValueError("--theta-keys must include every biased --parameters entry: " + ", ".join(missing))
    if "source.separation_as" not in theta_keys:
        raise ValueError("--theta-keys must include source.separation_as for science leakage metrics.")
    if args.max_workers != 1:
        raise ValueError("Only --max-workers 1 is implemented in the first scalar-bias runner.")
    config = ScalarBiasStudyConfig(
        results_root=args.results_root,
        run_name=args.run_name,
        parameters=parameters,
        theta_keys=theta_keys,
        bias_ppm_grid=parse_ppm_grid(args.bias_ppm_grid),
        n_frames=int(args.n_frames),
        noise=str(args.noise),
        trace_template=args.trace_template,
        seed=int(args.seed),
        trace_seed_policy=str(args.trace_seed_policy),
        max_workers=int(args.max_workers),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        aggregate_only=bool(args.aggregate_only),
        fail_fast=bool(args.fail_fast),
        quiet=bool(args.quiet),
        plots=bool(args.plots),
        reference_n_iter=args.reference_n_iter,
        zero_bias_reference_n_iter=args.zero_bias_reference_n_iter,
    )
    truth_values = resolve_truth_values_for_parameters(config.theta_keys, trace_template=config.trace_template)
    cases = build_case_plan(config, truth_values)
    write_plan(config, cases, truth_values)
    write_json(
        config.run_root / "manifest.json",
        {
            "schema_version": STUDY_SCHEMA_VERSION,
            "created_at": now_iso_local_ms(),
            "run_name": config.run_name,
            "run_root": str(config.run_root.resolve()),
            "run_plan_csv": str((config.run_root / "run_plan.csv").resolve()),
            "run_plan_json": str((config.run_root / "run_plan.json").resolve()),
            "commands_root": str((config.run_root / "commands").resolve()),
            "logs_root": str((config.run_root / "logs").resolve()),
            "config": config_to_dict(config),
        },
    )
    executed_cases = list(cases)
    if not config.dry_run and not config.aggregate_only:
        executed_cases = []
        for spec in cases:
            result = run_case(spec, quiet=config.quiet, resume=config.resume)
            executed_cases.append(result)
            if result.status == "failed" and config.fail_fast:
                break
    summary = aggregate_cases(config, executed_cases, truth_values) if not config.dry_run or config.aggregate_only else {
        "run_name": config.run_name,
        "run_root": str(config.run_root.resolve()),
        "counts": {"planned": len(cases), "aggregated_ok": 0, "aggregate_failed": 0},
    }
    if config.plots and not config.dry_run:
        write_plots(config)
    if not config.quiet:
        print(f"[scalar-bias] plan: {config.run_root / 'run_plan.csv'}")
        print(f"[scalar-bias] aggregate: {config.run_root / 'aggregate_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
