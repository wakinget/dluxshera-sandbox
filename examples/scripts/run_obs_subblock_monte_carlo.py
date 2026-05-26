"""Run reproducible Monte Carlo campaigns over observation sub-block summaries.

This script is a reusable, plan-first harness around
``examples/scripts/run_obs_subblock_study.py``.  It repeatedly builds independent
observation sub-block trials, executes each trial in a subprocess, validates the
image-backed ``SubblockSummary`` artifact, and writes aggregate diagnostics for
the empirical distribution of reduced information matrices and reduced scores.

The first supported aggregation path is Schur-summary-aware and intentionally
local to this script.  It is called by the CLI entrypoint below and is a
candidate for migration only after additional campaigns need the same plan,
execution, and aggregation semantics.  Shapes follow the observation-level
summary contract: ``S_b`` is ``(n_theta, n_theta)``, ``g_b`` and ``theta_ref``
are ``(n_theta,)``, and labels are required to match across accepted summaries.

Seed semantics are stable across reruns.  Seeds are derived with
``dluxshera.utils.noise.make_subseed`` from the base seed, run name, trial id,
and policy-specific token strings.  The resulting trace and render/noise seeds
are recorded in the manifest and run plan; no arithmetic on trial ids is used.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import shlex
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dluxshera.config.io import deep_merge, load_config_file
from dluxshera.config.numeric import normalize_optimizer_kwargs
from dluxshera.inference.observation_summary import (
    load_subblock_summary,
    load_subblock_summary_artifact_payload,
)
from dluxshera.inference.schedules import validate_optimizer_schedule_config
from dluxshera.utils.noise import make_subseed
from dluxshera.utils.obs_subblock_cli import append_reference_optimizer_flags
from dluxshera.utils.subprocess_diagnostics import (
    require_resource_time_available,
    run_subprocess_with_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "obs_subblock_monte_carlo"
STUDY_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_study.py"
DEFAULT_THETA_KEYS = (
    "source.separation_as",
    "source.log_flux_total",
    "source.contrast",
    "optics.plate_scale_as_per_pix",
)
DEFAULT_MC_MAX_DENSE_DIM = 40
DEFAULT_MC_REFERENCE_OPTIMIZER_KIND = "sgd"
DEFAULT_MC_REFERENCE_BASE_LR = 0.7
DEFAULT_MC_REFERENCE_N_ITER = 80
SUPPORTED_SEED_POLICIES = (
    "different_jitter_different_noise",
    "same_jitter_different_noise",
    "different_jitter_same_noise",
    "plan_csv",
)
SUPPORTED_REFERENCE_DIAGNOSTICS_PROFILES = ("none", "basic", "review", "full")
PLAN_COLUMNS = (
    "trial_id",
    "trial_name",
    "case_name",
    "seed_policy",
    "base_seed",
    "trial_seed",
    "trace_seed",
    "noise_seed",
    "n_frames",
    "noise_mode",
    "theta_keys",
    "phi_ref",
    "schur_curvature_method",
    "max_dense_dim",
    "variance_floor",
    "reference_diagnostics_profile",
    "reference_optimizer_kind",
    "reference_base_lr",
    "reference_n_iter",
    "reference_schedule_kind",
    "reference_schedule_json",
    "reference_preconditioning_enabled",
    "reference_preconditioning_method",
    "reference_preconditioning_reference",
    "reference_preconditioning_lr_clip",
    "schur_frame_quality_policy",
    "schur_frame_chi2_threshold",
    "schur_frame_quality_missing",
    "schur_frame_mask_denominator",
    "schur_frame_mask_min_good_frames",
    "summary_information_scale",
    "results_root",
    "case_root",
    "command_path",
    "stdout_log",
    "stderr_log",
    "expected_summary_json",
    "status",
    "return_code",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "summary_json_path",
    "matrix_npz_path",
    "failure_reason",
    "failure_class",
    "failure_hint",
    "last_stdout_line",
    "last_stderr_line",
    "memory_diagnostics_path",
    "last_memory_stage",
    "last_memory_rss_mb",
    "last_memory_peak_rss_mb",
)
STATUS_COLUMNS = PLAN_COLUMNS
TINY = 1.0e-12
FIT_WARNING_MAX_FRAME_REDUCED_CHI2 = 5.0
SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES = ("warn", "mask", "reject")
SUPPORTED_SCHUR_FRAME_QUALITY_MISSING_POLICIES = ("allow_all", "error")
SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS = ("original", "kept")


@dataclass(frozen=True)
class MonteCarloRunConfig:
    """Collect script-local Monte Carlo controls.

    The CLI and optional config file both normalize into this dataclass before
    plan generation.  It is called by ``main`` and all public helpers in this
    file; it is intentionally script-local until multiple workflows need the
    same campaign abstraction.

    Seed reproducibility is controlled by ``seed`` and ``seed_policy``.  Derived
    trace and render/noise seeds are produced only by ``derive_trial_seeds`` and
    then persisted in the plan, making ``run_plan.csv`` the source of truth for
    resumed and aggregate-only runs.
    """

    run_name: str
    results_root: Path = DEFAULT_RESULTS_ROOT
    n_trials: int = 1
    max_workers: int = 1
    seed: int = 42
    seed_policy: str = "different_jitter_different_noise"
    plan_csv: Path | None = None
    study_mode: str = "schur_summary"
    n_frames: int = 10
    noise: str = "enabled"
    theta_keys: tuple[str, ...] = DEFAULT_THETA_KEYS
    phi_ref: str = "recovered"
    schur_curvature_method: str = "auto"
    variance_floor: float = 1.0
    reference_diagnostics_profile: str = "none"
    reference_optimizer_kind: str | None = DEFAULT_MC_REFERENCE_OPTIMIZER_KIND
    reference_base_lr: float | None = DEFAULT_MC_REFERENCE_BASE_LR
    reference_n_iter: int | None = DEFAULT_MC_REFERENCE_N_ITER
    reference_optimizer_kwargs: dict[str, Any] | None = None
    reference_schedule: dict[str, Any] | None = None
    reference_preconditioning_enabled: bool | None = None
    reference_preconditioning_method: str | None = None
    reference_preconditioning_reference: str | None = None
    reference_preconditioning_damping: float | None = None
    reference_preconditioning_eig_floor_rel: float | None = None
    reference_preconditioning_eig_floor_abs: float | None = None
    reference_preconditioning_lr_clip: tuple[float, float] | None = None
    reference_early_stopping_enabled: bool | None = None
    reference_early_stopping_min_iter: int | None = None
    reference_early_stopping_patience: int | None = None
    reference_early_stopping_loss_rtol: float | None = None
    reference_early_stopping_loss_atol: float | None = None
    reference_early_stopping_step_atol: float | None = None
    reference_early_stopping_grad_norm_atol: float | None = None
    reference_init_mode: str | None = None
    reuse_reference_inference: str | None = None
    schur_damping: float | None = None
    schur_frame_quality_policy: str = "warn"
    schur_frame_chi2_threshold: float = FIT_WARNING_MAX_FRAME_REDUCED_CHI2
    schur_frame_quality_missing: str = "allow_all"
    schur_frame_mask_denominator: str = "original"
    schur_frame_mask_min_good_frames: int = 1
    max_dense_dim: int | None = DEFAULT_MC_MAX_DENSE_DIM
    summary_objective: str | None = None
    summary_information_scale: str = "summed_likelihood"
    validate_surrogate: bool | None = None
    aggregation_enabled: bool = True
    truth_mode: str = "summary_theta_ref"
    truth_json: Path | None = None
    plots: bool = True
    resume: bool = False
    aggregate_only: bool = False
    dry_run: bool = False
    fail_fast: bool = False
    quiet: bool = False
    progress_interval_s: float = 30.0
    tail_lines: int = 1
    memory_diagnostics: bool = False
    resource_time: bool | str | None = None
    profile_runtime: bool = False
    profile_runtime_detail: str = "basic"
    memory_progress_tail_lines: int = 3
    verbose: bool = False

    @property
    def run_root(self) -> Path:
        return (self.results_root / self.run_name).resolve()


@dataclass(frozen=True)
class MonteCarloTrialSpec:
    """Describe one planned subprocess-backed observation sub-block trial.

    ``trace_seed`` controls stochastic trace generation, while ``noise_seed`` is
    passed as the render seed used by the image renderer and observation-noise
    path.  The command produced by ``build_trial_command`` writes case-local
    artifacts under ``case_root`` and should be reproducible from this spec
    alone.
    """

    trial_id: int
    trial_name: str
    case_name: str
    seed_policy: str
    base_seed: int
    trial_seed: int
    trace_seed: int
    noise_seed: int
    n_frames: int
    noise_mode: str
    theta_keys: tuple[str, ...]
    phi_ref: str
    schur_curvature_method: str
    max_dense_dim: int | None
    variance_floor: float
    reference_diagnostics_profile: str
    reference_optimizer_kind: str | None
    reference_base_lr: float | None
    reference_n_iter: int | None
    reference_schedule: dict[str, Any] | None
    reference_preconditioning_enabled: bool | None
    reference_preconditioning_method: str | None
    reference_preconditioning_reference: str | None
    reference_preconditioning_lr_clip: tuple[float, float] | None
    schur_frame_quality_policy: str
    schur_frame_chi2_threshold: float
    schur_frame_quality_missing: str
    schur_frame_mask_denominator: str
    schur_frame_mask_min_good_frames: int
    summary_information_scale: str
    results_root: Path
    case_root: Path
    command_path: Path
    stdout_log: Path
    stderr_log: Path
    expected_summary_json: Path

    @property
    def expected_matrix_npz(self) -> Path:
        return self.expected_summary_json.with_name("subblock_summary_matrices.npz")

    @property
    def memory_diagnostics_path(self) -> Path:
        return self.expected_summary_json.with_name("schur_summary_memory_timeline.jsonl")

    @property
    def runtime_profile_summary_path(self) -> Path:
        return self.expected_summary_json.with_name("runtime_profile_summary.json")

    @property
    def runtime_profile_timeline_path(self) -> Path:
        return self.expected_summary_json.with_name("runtime_profile_timeline.jsonl")


@dataclass(frozen=True)
class MonteCarloTrialResult:
    """Record execution status for one trial subprocess."""

    trial_id: int
    status: str
    return_code: int | None
    started_at: str | None
    finished_at: str | None
    elapsed_seconds: float | None
    summary_json_path: Path | None
    matrix_npz_path: Path | None
    failure_reason: str | None = None
    failure_class: str | None = None
    failure_hint: str | None = None
    last_stdout_line: str | None = None
    last_stderr_line: str | None = None
    memory_diagnostics_path: Path | None = None
    last_memory_stage: str | None = None
    last_memory_rss_mb: float | None = None
    last_memory_peak_rss_mb: float | None = None


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def mc_log(message: str, *, quiet: bool = False, force: bool = False, **fields: Any) -> None:
    """Print one flushed parent-level Monte Carlo progress message.

    This script-local helper is used by ``main``, ``run_trial_pool``, and
    aggregation boundaries.  It intentionally reports structured parent events
    instead of streaming child subprocess logs, keeping parallel runs readable
    while detailed stdout/stderr remain in per-trial log files.
    """

    if quiet and not force:
        return
    parts = [f"[obs_subblock_mc] {message}"]
    for key, value in fields.items():
        if value is None:
            continue
        text = str(value)
        if any(ch.isspace() for ch in text) or text == "":
            text = json.dumps(text)
        parts.append(f"{key}={text}")
    print(" ".join(parts), flush=True)


def tail_text_file(path: Path, *, n_lines: int, max_chars: int = 240) -> tuple[str, ...]:
    """Return the last ``n_lines`` from a possibly active text log file."""

    if n_lines <= 0 or not path.exists():
        return ()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    out: list[str] = []
    for line in lines[-int(n_lines):]:
        stripped = line.rstrip("\n\r")
        if len(stripped) > max_chars:
            stripped = stripped[: max_chars - 3] + "..."
        out.append(stripped)
    return tuple(out)


def classify_subprocess_failure(return_code: int | None) -> tuple[str | None, str | None]:
    """Return derived failure class/hint without overwriting the raw reason."""

    if return_code == -9:
        return "probable_sigkill", "possible_memory_pressure_or_external_kill"
    if return_code not in {None, 0}:
        return "subprocess_nonzero_exit", "inspect_trial_stdout_stderr"
    return None, None


def read_last_memory_diagnostic(path: Path) -> dict[str, Any] | None:
    """Read the last valid memory JSONL record, returning None if unavailable."""

    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return None


def tail_memory_diagnostics(path: Path, *, n_lines: int) -> tuple[dict[str, Any], ...]:
    if n_lines <= 0 or not path.exists():
        return ()
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ()
    records: list[dict[str, Any]] = []
    for line in reversed(lines):
        if len(records) >= int(n_lines):
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            records.append(dict(payload))
    return tuple(reversed(records))


def memory_failure_fields(
    *,
    spec: MonteCarloTrialSpec,
    result: MonteCarloTrialResult | None = None,
) -> dict[str, Any]:
    """Build parent-side failure diagnostics from logs and optional JSONL."""

    memory_path = (
        result.memory_diagnostics_path
        if result is not None and result.memory_diagnostics_path is not None
        else spec.memory_diagnostics_path
    )
    last_memory = read_last_memory_diagnostic(memory_path)
    return {
        "last_stdout_line": (tail_text_file(spec.stdout_log, n_lines=1) or ("",))[-1],
        "last_stderr_line": (tail_text_file(spec.stderr_log, n_lines=1) or ("",))[-1],
        "memory_diagnostics_path": str(memory_path) if memory_path.exists() else "",
        "runtime_profile_summary_path": str(spec.runtime_profile_summary_path),
        "runtime_profile_timeline_path": str(spec.runtime_profile_timeline_path),
        "last_memory_stage": "" if last_memory is None else last_memory.get("stage", ""),
        "last_memory_rss_mb": "" if last_memory is None else last_memory.get("rss_mb", ""),
        "last_memory_peak_rss_mb": ""
        if last_memory is None
        else last_memory.get("peak_rss_mb", ""),
    }


def parse_theta_keys(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = raw
    keys = tuple(str(part).strip() for part in parts if str(part).strip())
    if not keys:
        raise ValueError("theta_keys must contain at least one key.")
    if len(set(keys)) != len(keys):
        raise ValueError("theta_keys must not contain duplicates.")
    return keys


def parse_reference_optimizer_kwargs(raw_values: Sequence[str] | Mapping[str, Any] | None) -> dict[str, Any]:
    """Parse CLI/config optimizer kwargs for recovered-reference trials."""

    if raw_values is None:
        return {}
    if isinstance(raw_values, Mapping):
        return dict(raw_values)
    parsed: dict[str, Any] = {}
    for raw in raw_values:
        text = str(raw).strip()
        if "=" not in text:
            raise ValueError(
                "--reference-optimizer-kwarg values must use KEY=VALUE syntax; "
                f"received {raw!r}."
            )
        key, value = text.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("--reference-optimizer-kwarg keys must be non-empty.")
        parsed[key] = value.strip()
    return parsed


def parse_csv_ints(raw: str | Sequence[int] | None, *, field_name: str) -> tuple[int, ...] | None:
    if raw is None or raw == "":
        return None
    parts = [part.strip() for part in raw.split(",")] if isinstance(raw, str) else raw
    if not parts:
        raise ValueError(f"{field_name} must contain at least one integer.")
    values: list[int] = []
    for index, part in enumerate(parts):
        integer = int(part)
        if float(integer) != float(part):
            raise ValueError(f"{field_name}[{index}] must be an integer.")
        values.append(integer)
    return tuple(values)


def parse_csv_floats(
    raw: str | Sequence[float] | None,
    *,
    field_name: str,
) -> tuple[float, ...] | None:
    if raw is None or raw == "":
        return None
    parts = [part.strip() for part in raw.split(",")] if isinstance(raw, str) else raw
    if not parts:
        raise ValueError(f"{field_name} must contain at least one float.")
    return tuple(float(part) for part in parts)


def parse_reference_schedule_config(
    *,
    kind: str | None,
    warmup_steps: int | None,
    start_factor: float | None,
    min_factor: float | None,
    boundaries: str | Sequence[int] | None,
    factors: str | Sequence[float] | None,
    decay_rate: float | None,
    transition_steps: int | None,
    staircase: bool,
    n_iter: int | None,
) -> dict[str, Any] | None:
    """Normalize one optional recovered-reference scalar LR schedule config."""

    if kind is None:
        if any(
            value not in {None, False, ""}
            for value in (
                warmup_steps,
                start_factor,
                min_factor,
                boundaries,
                factors,
                decay_rate,
                transition_steps,
                staircase,
            )
        ):
            raise ValueError(
                "reference schedule fields require --reference-schedule-kind "
                "or a config schedule.kind."
            )
        return None

    schedule_kind = str(kind).strip().lower()
    schedule: dict[str, Any] = {"kind": schedule_kind}
    if schedule_kind == "linear_warmup":
        if warmup_steps is None or start_factor is None:
            raise ValueError(
                "linear_warmup requires warmup_steps and start_factor."
            )
        schedule["warmup_steps"] = int(warmup_steps)
        schedule["start_factor"] = float(start_factor)
    elif schedule_kind == "piecewise_constant":
        parsed_boundaries = parse_csv_ints(
            boundaries,
            field_name="reference_schedule_boundaries",
        )
        parsed_factors = parse_csv_floats(
            factors,
            field_name="reference_schedule_factors",
        )
        if parsed_boundaries is None or parsed_factors is None:
            raise ValueError(
                "piecewise_constant requires boundaries and factors."
            )
        schedule["boundaries"] = list(parsed_boundaries)
        schedule["factors"] = list(parsed_factors)
    elif schedule_kind == "exponential_decay":
        if decay_rate is None or transition_steps is None:
            raise ValueError(
                "exponential_decay requires decay_rate and transition_steps."
            )
        schedule["decay_rate"] = float(decay_rate)
        schedule["transition_steps"] = int(transition_steps)
        if staircase:
            schedule["staircase"] = True
    elif schedule_kind == "cosine_decay":
        if min_factor is None:
            raise ValueError("cosine_decay requires min_factor.")
        schedule["min_factor"] = float(min_factor)
    elif schedule_kind == "linear_warmup_cosine_decay":
        if warmup_steps is None or start_factor is None or min_factor is None:
            raise ValueError(
                "linear_warmup_cosine_decay requires warmup_steps, start_factor, and min_factor."
            )
        schedule["warmup_steps"] = int(warmup_steps)
        schedule["start_factor"] = float(start_factor)
        schedule["min_factor"] = float(min_factor)
    elif schedule_kind == "constant":
        pass
    else:
        raise ValueError(
            "reference_schedule_kind must be one of: constant, linear_warmup, "
            "piecewise_constant, exponential_decay, cosine_decay, "
            "linear_warmup_cosine_decay."
        )

    return validate_optimizer_schedule_config(
        schedule,
        n_iter=int(n_iter if n_iter is not None else DEFAULT_MC_REFERENCE_N_ITER),
        path="reference_schedule",
    )


def _format_schedule_json(value: Mapping[str, Any] | None) -> str:
    return "" if value is None else json.dumps(value, sort_keys=True, separators=(",", ":"))


def parse_reference_lr_clip(raw: str | Sequence[float] | None) -> tuple[float, float] | None:
    """Parse recovered-reference preconditioning learning-rate clip bounds."""

    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        parts: Sequence[Any] = [part.strip() for part in raw.split(",")]
    else:
        parts = raw
    if len(parts) != 2:
        raise ValueError("reference_preconditioning_lr_clip must be MIN,MAX.")
    low = float(parts[0])
    high = float(parts[1])
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high <= 0.0:
        raise ValueError("reference_preconditioning_lr_clip values must be positive finite floats.")
    if high < low:
        raise ValueError("reference_preconditioning_lr_clip max must be >= min.")
    return (low, high)


def _format_lr_clip(value: tuple[float, float] | None) -> str:
    return "" if value is None else f"{float(value[0]):.12g},{float(value[1]):.12g}"


def _parse_optional_bool(value: Any) -> bool | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"Expected optional bool, got {value!r}.")


def filesystem_safe_token(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    token = "".join(ch if ch in allowed else "_" for ch in str(value))
    return token.strip("._-") or "run"


def derive_trial_seeds(
    *,
    base_seed: int,
    run_name: str,
    trial_id: int,
    seed_policy: str,
) -> dict[str, int]:
    """Derive deterministic trial, trace, and render/noise seeds.

    Called by ``build_trial_plan``.  The helper deliberately reuses
    ``make_subseed`` so seed derivation stays compatible with other
    JAX-oriented Monte Carlo utilities in the repository.  Token strings are
    stable and are persisted indirectly through the recorded seeds and manifest.
    """

    if seed_policy not in SUPPORTED_SEED_POLICIES:
        raise ValueError(f"Unsupported seed policy: {seed_policy}")
    token = f"{run_name}.trial.{int(trial_id):06d}"
    trial_seed = make_subseed(int(base_seed), token)
    if seed_policy == "same_jitter_different_noise":
        trace_seed = make_subseed(int(base_seed), f"{run_name}.shared_trace")
        noise_seed = make_subseed(int(base_seed), f"{token}.noise")
    elif seed_policy == "different_jitter_same_noise":
        trace_seed = make_subseed(int(base_seed), f"{token}.trace")
        noise_seed = make_subseed(int(base_seed), f"{run_name}.shared_noise")
    else:
        trace_seed = make_subseed(int(base_seed), f"{token}.trace")
        noise_seed = make_subseed(int(base_seed), f"{token}.noise")
    return {
        "trial_seed": int(trial_seed),
        "trace_seed": int(trace_seed),
        "noise_seed": int(noise_seed),
    }


def _trial_from_row(row: Mapping[str, Any], *, run_root: Path) -> MonteCarloTrialSpec:
    trial_id = int(row["trial_id"])
    case_root = Path(str(row["case_root"])).expanduser()
    if not case_root.is_absolute():
        case_root = (run_root / case_root).resolve()
    command_path = Path(str(row["command_path"]))
    stdout_log = Path(str(row["stdout_log"]))
    stderr_log = Path(str(row["stderr_log"]))
    expected_summary_json = Path(str(row["expected_summary_json"]))
    return MonteCarloTrialSpec(
        trial_id=trial_id,
        trial_name=str(row.get("trial_name") or f"trial_{trial_id:06d}"),
        case_name=str(row.get("case_name") or f"trials/trial_{trial_id:06d}/case"),
        seed_policy=str(row["seed_policy"]),
        base_seed=int(row["base_seed"]),
        trial_seed=int(row["trial_seed"]),
        trace_seed=int(row["trace_seed"]),
        noise_seed=int(row["noise_seed"]),
        n_frames=int(row["n_frames"]),
        noise_mode=str(row["noise_mode"]),
        theta_keys=parse_theta_keys(str(row["theta_keys"])),
        phi_ref=str(row["phi_ref"]),
        schur_curvature_method=str(row["schur_curvature_method"]),
        max_dense_dim=(
            int(row["max_dense_dim"])
            if row.get("max_dense_dim")
            else DEFAULT_MC_MAX_DENSE_DIM
        ),
        variance_floor=float(row["variance_floor"]),
        reference_diagnostics_profile=str(row.get("reference_diagnostics_profile") or "none"),
        reference_optimizer_kind=str(row["reference_optimizer_kind"]) if row.get("reference_optimizer_kind") else None,
        reference_base_lr=float(row["reference_base_lr"]) if row.get("reference_base_lr") else None,
        reference_n_iter=int(row["reference_n_iter"]) if row.get("reference_n_iter") else None,
        reference_schedule=(
            json.loads(str(row["reference_schedule_json"]))
            if row.get("reference_schedule_json")
            else None
        ),
        reference_preconditioning_enabled=_parse_optional_bool(
            row.get("reference_preconditioning_enabled")
        ),
        reference_preconditioning_method=(
            str(row["reference_preconditioning_method"])
            if row.get("reference_preconditioning_method")
            else None
        ),
        reference_preconditioning_reference=(
            str(row["reference_preconditioning_reference"])
            if row.get("reference_preconditioning_reference")
            else None
        ),
        reference_preconditioning_lr_clip=parse_reference_lr_clip(
            row.get("reference_preconditioning_lr_clip")
        ),
        schur_frame_quality_policy=str(row.get("schur_frame_quality_policy") or "warn"),
        schur_frame_chi2_threshold=float(
            row.get("schur_frame_chi2_threshold") or FIT_WARNING_MAX_FRAME_REDUCED_CHI2
        ),
        schur_frame_quality_missing=str(row.get("schur_frame_quality_missing") or "allow_all"),
        schur_frame_mask_denominator=str(row.get("schur_frame_mask_denominator") or "original"),
        schur_frame_mask_min_good_frames=int(row.get("schur_frame_mask_min_good_frames") or 1),
        summary_information_scale=str(
            row.get("summary_information_scale") or "summed_likelihood"
        ),
        results_root=Path(str(row["results_root"])).resolve(),
        case_root=case_root.resolve(),
        command_path=command_path if command_path.is_absolute() else (run_root / command_path).resolve(),
        stdout_log=stdout_log if stdout_log.is_absolute() else (run_root / stdout_log).resolve(),
        stderr_log=stderr_log if stderr_log.is_absolute() else (run_root / stderr_log).resolve(),
        expected_summary_json=(
            expected_summary_json
            if expected_summary_json.is_absolute()
            else (run_root / expected_summary_json).resolve()
        ),
    )


def build_trial_plan(config: MonteCarloRunConfig) -> list[MonteCarloTrialSpec]:
    """Build the durable run plan for a Monte Carlo campaign."""

    run_root = config.run_root
    if config.seed_policy == "plan_csv":
        if config.plan_csv is None:
            raise ValueError("seed_policy='plan_csv' requires --plan-csv.")
        with config.plan_csv.open("r", encoding="utf-8", newline="") as handle:
            return [_trial_from_row(row, run_root=run_root) for row in csv.DictReader(handle)]

    plan: list[MonteCarloTrialSpec] = []
    for trial_id in range(int(config.n_trials)):
        seeds = derive_trial_seeds(
            base_seed=config.seed,
            run_name=config.run_name,
            trial_id=trial_id,
            seed_policy=config.seed_policy,
        )
        trial_name = f"trial_{trial_id:06d}"
        case_name = f"trials/{trial_name}/case"
        case_root = run_root / case_name
        study_root = case_root / "study" / "schur_summary"
        plan.append(
            MonteCarloTrialSpec(
                trial_id=trial_id,
                trial_name=trial_name,
                case_name=case_name,
                seed_policy=config.seed_policy,
                base_seed=int(config.seed),
                trial_seed=seeds["trial_seed"],
                trace_seed=seeds["trace_seed"],
                noise_seed=seeds["noise_seed"],
                n_frames=int(config.n_frames),
                noise_mode=str(config.noise),
                theta_keys=tuple(config.theta_keys),
                phi_ref=str(config.phi_ref),
                schur_curvature_method=str(config.schur_curvature_method),
                max_dense_dim=config.max_dense_dim,
                variance_floor=float(config.variance_floor),
                reference_diagnostics_profile=str(config.reference_diagnostics_profile),
                reference_optimizer_kind=config.reference_optimizer_kind,
                reference_base_lr=config.reference_base_lr,
                reference_n_iter=config.reference_n_iter,
                reference_schedule=(
                    None
                    if config.reference_schedule is None
                    else dict(config.reference_schedule)
                ),
                reference_preconditioning_enabled=config.reference_preconditioning_enabled,
                reference_preconditioning_method=config.reference_preconditioning_method,
                reference_preconditioning_reference=config.reference_preconditioning_reference,
                reference_preconditioning_lr_clip=config.reference_preconditioning_lr_clip,
                schur_frame_quality_policy=config.schur_frame_quality_policy,
                schur_frame_chi2_threshold=float(config.schur_frame_chi2_threshold),
                schur_frame_quality_missing=config.schur_frame_quality_missing,
                schur_frame_mask_denominator=config.schur_frame_mask_denominator,
                schur_frame_mask_min_good_frames=int(config.schur_frame_mask_min_good_frames),
                summary_information_scale=str(config.summary_information_scale),
                results_root=run_root,
                case_root=case_root.resolve(),
                command_path=(run_root / "commands" / f"{trial_name}.sh").resolve(),
                stdout_log=(run_root / "logs" / f"{trial_name}_stdout.log").resolve(),
                stderr_log=(run_root / "logs" / f"{trial_name}_stderr.log").resolve(),
                expected_summary_json=(study_root / "subblock_summary.json").resolve(),
            )
        )
    return plan


def load_run_plan_csv(path: Path, *, run_root: Path) -> list[MonteCarloTrialSpec]:
    """Load an existing ``run_plan.csv`` as the source of truth."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_trial_from_row(row, run_root=run_root) for row in csv.DictReader(handle)]


def build_trial_command(spec: MonteCarloTrialSpec, config: MonteCarloRunConfig) -> list[str]:
    """Return the subprocess command for one Schur-summary trial."""

    command = [
        sys.executable,
        str(STUDY_SCRIPT),
        "--mode",
        str(config.study_mode),
        "--case-root",
        str(spec.case_root),
        "--n-frames",
        str(spec.n_frames),
        "--noise",
        spec.noise_mode,
        "--theta-keys",
        ",".join(spec.theta_keys),
        "--phi-ref",
        spec.phi_ref,
        "--schur-curvature-method",
        spec.schur_curvature_method,
        "--trace-seed",
        str(spec.trace_seed),
        "--render-seed",
        str(spec.noise_seed),
        "--reference-diagnostics-profile",
        str(config.reference_diagnostics_profile),
        "--schur-frame-quality-policy",
        str(spec.schur_frame_quality_policy),
        "--schur-frame-chi2-threshold",
        str(spec.schur_frame_chi2_threshold),
        "--schur-frame-quality-missing",
        str(spec.schur_frame_quality_missing),
        "--schur-frame-mask-denominator",
        str(spec.schur_frame_mask_denominator),
        "--schur-frame-mask-min-good-frames",
        str(spec.schur_frame_mask_min_good_frames),
        "--summary-information-scale",
        str(spec.summary_information_scale),
    ]
    if spec.variance_floor is not None:
        command.extend(["--variance-floor", str(spec.variance_floor)])
    if config.schur_damping is not None:
        command.extend(["--schur-damping", str(config.schur_damping)])
    if spec.max_dense_dim is not None:
        command.extend(["--max-dense-dim", str(spec.max_dense_dim)])
    if config.summary_objective is not None:
        command.extend(["--summary-objective", str(config.summary_objective)])
    if config.validate_surrogate is not None:
        command.append("--validate-surrogate" if config.validate_surrogate else "--no-validate-surrogate")
    reference_cfg = {**dataclasses.asdict(config), "reference_schedule": spec.reference_schedule}
    append_reference_optimizer_flags(command, reference_cfg)
    if config.memory_diagnostics:
        command.append("--memory-diagnostics")
    if config.profile_runtime:
        command.append("--profile-runtime")
        command.extend(["--profile-runtime-detail", str(config.profile_runtime_detail)])
    return command


def write_command_file(spec: MonteCarloTrialSpec, command: Sequence[str]) -> None:
    spec.command_path.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join(shlex.quote(str(part)) for part in command)
    spec.command_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"export MPLCONFIGDIR={shlex.quote(str(spec.command_path.parent.parent / 'matplotlib'))}\n"
        f"PYTHONPATH={shlex.quote(str(REPO_ROOT / 'src'))}${{PYTHONPATH:+:${{PYTHONPATH}}}} {text}\n",
        encoding="utf-8",
    )
    spec.command_path.chmod(0o755)


def _plan_row(spec: MonteCarloTrialSpec, result: MonteCarloTrialResult | None = None) -> dict[str, Any]:
    row = {
        "trial_id": spec.trial_id,
        "trial_name": spec.trial_name,
        "case_name": spec.case_name,
        "seed_policy": spec.seed_policy,
        "base_seed": spec.base_seed,
        "trial_seed": spec.trial_seed,
        "trace_seed": spec.trace_seed,
        "noise_seed": spec.noise_seed,
        "n_frames": spec.n_frames,
        "noise_mode": spec.noise_mode,
        "theta_keys": ",".join(spec.theta_keys),
        "phi_ref": spec.phi_ref,
        "schur_curvature_method": spec.schur_curvature_method,
        "max_dense_dim": "" if spec.max_dense_dim is None else spec.max_dense_dim,
        "variance_floor": spec.variance_floor,
        "reference_diagnostics_profile": spec.reference_diagnostics_profile,
        "reference_optimizer_kind": spec.reference_optimizer_kind or "",
        "reference_base_lr": "" if spec.reference_base_lr is None else spec.reference_base_lr,
        "reference_n_iter": "" if spec.reference_n_iter is None else spec.reference_n_iter,
        "reference_schedule_kind": (
            "" if spec.reference_schedule is None else str(spec.reference_schedule.get("kind", ""))
        ),
        "reference_schedule_json": _format_schedule_json(spec.reference_schedule),
        "reference_preconditioning_enabled": (
            "" if spec.reference_preconditioning_enabled is None else spec.reference_preconditioning_enabled
        ),
        "reference_preconditioning_method": spec.reference_preconditioning_method or "",
        "reference_preconditioning_reference": spec.reference_preconditioning_reference or "",
        "reference_preconditioning_lr_clip": _format_lr_clip(
            spec.reference_preconditioning_lr_clip
        ),
        "schur_frame_quality_policy": spec.schur_frame_quality_policy,
        "schur_frame_chi2_threshold": spec.schur_frame_chi2_threshold,
        "schur_frame_quality_missing": spec.schur_frame_quality_missing,
        "schur_frame_mask_denominator": spec.schur_frame_mask_denominator,
        "schur_frame_mask_min_good_frames": spec.schur_frame_mask_min_good_frames,
        "summary_information_scale": spec.summary_information_scale,
        "results_root": str(spec.results_root),
        "case_root": str(spec.case_root),
        "command_path": str(spec.command_path),
        "stdout_log": str(spec.stdout_log),
        "stderr_log": str(spec.stderr_log),
        "expected_summary_json": str(spec.expected_summary_json),
        "status": "planned",
        "return_code": "",
        "started_at": "",
        "finished_at": "",
        "elapsed_seconds": "",
        "summary_json_path": "",
        "matrix_npz_path": "",
        "failure_reason": "",
        "failure_class": "",
        "failure_hint": "",
        "last_stdout_line": "",
        "last_stderr_line": "",
        "memory_diagnostics_path": "",
        "last_memory_stage": "",
        "last_memory_rss_mb": "",
        "last_memory_peak_rss_mb": "",
    }
    if result is not None:
        row.update(
            {
                "status": result.status,
                "return_code": "" if result.return_code is None else result.return_code,
                "started_at": result.started_at or "",
                "finished_at": result.finished_at or "",
                "elapsed_seconds": "" if result.elapsed_seconds is None else f"{result.elapsed_seconds:.6f}",
                "summary_json_path": "" if result.summary_json_path is None else str(result.summary_json_path),
                "matrix_npz_path": "" if result.matrix_npz_path is None else str(result.matrix_npz_path),
                "failure_reason": result.failure_reason or "",
                "failure_class": result.failure_class or "",
                "failure_hint": result.failure_hint or "",
                "last_stdout_line": result.last_stdout_line or "",
                "last_stderr_line": result.last_stderr_line or "",
                "memory_diagnostics_path": ""
                if result.memory_diagnostics_path is None
                else str(result.memory_diagnostics_path),
                "last_memory_stage": result.last_memory_stage or "",
                "last_memory_rss_mb": ""
                if result.last_memory_rss_mb is None
                else result.last_memory_rss_mb,
                "last_memory_peak_rss_mb": ""
                if result.last_memory_peak_rss_mb is None
                else result.last_memory_peak_rss_mb,
            }
        )
    return row


def write_run_plan_csv(path: Path, plan: Sequence[MonteCarloTrialSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAN_COLUMNS)
        writer.writeheader()
        for spec in plan:
            writer.writerow(_plan_row(spec))


def write_run_status_csv(
    path: Path,
    plan: Sequence[MonteCarloTrialSpec],
    results: Mapping[int, MonteCarloTrialResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        for spec in plan:
            writer.writerow(_plan_row(spec, results.get(spec.trial_id)))


def write_progress_json(
    path: Path,
    *,
    total_trials: int,
    results: Mapping[int, MonteCarloTrialResult],
    active: Mapping[int, tuple[MonteCarloTrialSpec, float]],
    started_at_monotonic: float,
) -> None:
    """Write a lightweight progress snapshot for external monitoring."""

    counts = _status_counts(results)
    now = time.monotonic()
    payload = {
        "last_heartbeat_at": now_iso_utc(),
        "total_trials": int(total_trials),
        **counts,
        "active_trial_ids": sorted(active),
        "active_elapsed_seconds": {
            str(trial_id): round(now - started, 3)
            for trial_id, (_spec, started) in active.items()
        },
        "elapsed_seconds": round(now - started_at_monotonic, 3),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_trial_status(path: Path) -> dict[int, MonteCarloTrialResult]:
    if not path.exists():
        return {}
    out: dict[int, MonteCarloTrialResult] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trial_id = int(row["trial_id"])
            summary = Path(row["summary_json_path"]) if row.get("summary_json_path") else None
            matrix = Path(row["matrix_npz_path"]) if row.get("matrix_npz_path") else None
            elapsed = float(row["elapsed_seconds"]) if row.get("elapsed_seconds") else None
            return_code = int(row["return_code"]) if row.get("return_code") else None
            memory_path = (
                Path(row["memory_diagnostics_path"])
                if row.get("memory_diagnostics_path")
                else None
            )
            rss = float(row["last_memory_rss_mb"]) if row.get("last_memory_rss_mb") else None
            peak = (
                float(row["last_memory_peak_rss_mb"])
                if row.get("last_memory_peak_rss_mb")
                else None
            )
            failure_class = row.get("failure_class") or None
            failure_hint = row.get("failure_hint") or None
            if failure_class is None:
                failure_class, failure_hint = classify_subprocess_failure(return_code)
            out[trial_id] = MonteCarloTrialResult(
                trial_id=trial_id,
                status=str(row.get("status") or "unknown"),
                return_code=return_code,
                started_at=row.get("started_at") or None,
                finished_at=row.get("finished_at") or None,
                elapsed_seconds=elapsed,
                summary_json_path=summary,
                matrix_npz_path=matrix,
                failure_reason=row.get("failure_reason") or None,
                failure_class=failure_class,
                failure_hint=failure_hint,
                last_stdout_line=row.get("last_stdout_line") or None,
                last_stderr_line=row.get("last_stderr_line") or None,
                memory_diagnostics_path=memory_path,
                last_memory_stage=row.get("last_memory_stage") or None,
                last_memory_rss_mb=rss,
                last_memory_peak_rss_mb=peak,
            )
    return out


def write_manifest(config: MonteCarloRunConfig, plan: Sequence[MonteCarloTrialSpec]) -> None:
    payload = {
        "schema_version": "obs_subblock_monte_carlo_manifest.v1",
        "created_at": now_iso_utc(),
        "run_name": config.run_name,
        "run_root": str(config.run_root),
        "n_trials_planned": len(plan),
        "max_workers": int(config.max_workers),
        "base_seed": int(config.seed),
        "seed_policy": config.seed_policy,
        "seed_helper": "dluxshera.utils.noise.make_subseed",
        "study_defaults": {
            "study_mode": config.study_mode,
            "n_frames": config.n_frames,
            "noise": config.noise,
            "theta_keys": list(config.theta_keys),
            "phi_ref": config.phi_ref,
            "schur_curvature_method": config.schur_curvature_method,
            "schur_frame_quality_policy": config.schur_frame_quality_policy,
            "schur_frame_chi2_threshold": config.schur_frame_chi2_threshold,
            "schur_frame_quality_missing": config.schur_frame_quality_missing,
            "schur_frame_mask_denominator": config.schur_frame_mask_denominator,
            "schur_frame_mask_min_good_frames": config.schur_frame_mask_min_good_frames,
            "max_dense_dim": config.max_dense_dim,
            "variance_floor": config.variance_floor,
            "reference_diagnostics_profile": config.reference_diagnostics_profile,
        },
        "reference_optimizer_overrides": _reference_optimizer_override_payload(config),
        "mc_workflow_defaults": {
            "max_dense_dim": DEFAULT_MC_MAX_DENSE_DIM,
            "reference_optimizer_kind": DEFAULT_MC_REFERENCE_OPTIMIZER_KIND,
            "reference_base_lr": DEFAULT_MC_REFERENCE_BASE_LR,
            "reference_n_iter": DEFAULT_MC_REFERENCE_N_ITER,
        },
        "aggregation": {
            "enabled": bool(config.aggregation_enabled),
            "truth_mode": config.truth_mode,
            "plots": bool(config.plots),
        },
        "memory_diagnostics": {
            "enabled": bool(config.memory_diagnostics),
            "memory_progress_tail_lines": int(config.memory_progress_tail_lines),
        },
    }
    path = config.run_root / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _reference_optimizer_override_payload(config: MonteCarloRunConfig) -> dict[str, Any]:
    return {
        "kind": config.reference_optimizer_kind,
        "base_lr": config.reference_base_lr,
        "n_iter": config.reference_n_iter,
        "kwargs": dict(config.reference_optimizer_kwargs or {}),
        "schedule": None if config.reference_schedule is None else dict(config.reference_schedule),
        "preconditioning": {
            "enabled": config.reference_preconditioning_enabled,
            "method": config.reference_preconditioning_method,
            "reference": config.reference_preconditioning_reference,
            "damping": config.reference_preconditioning_damping,
            "eig_floor_rel": config.reference_preconditioning_eig_floor_rel,
            "eig_floor_abs": config.reference_preconditioning_eig_floor_abs,
            "lr_clip": (
                None
                if config.reference_preconditioning_lr_clip is None
                else list(config.reference_preconditioning_lr_clip)
            ),
        },
    }


def valid_existing_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        load_subblock_summary(path)
    except Exception:
        return False
    return True


def run_trial_subprocess(
    spec: MonteCarloTrialSpec,
    command: Sequence[str],
    *,
    resume: bool = False,
    resource_time: bool | str | None = None,
) -> MonteCarloTrialResult:
    """Execute one trial command and capture per-trial logs."""

    if resume and valid_existing_summary(spec.expected_summary_json):
        return MonteCarloTrialResult(
            trial_id=spec.trial_id,
            status="skipped_completed",
            return_code=0,
            started_at=None,
            finished_at=None,
            elapsed_seconds=0.0,
            summary_json_path=spec.expected_summary_json,
            matrix_npz_path=spec.expected_matrix_npz if spec.expected_matrix_npz.exists() else None,
        )

    env = dict(os.environ)
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    env.setdefault("MPLCONFIGDIR", str(spec.command_path.parent.parent / "matplotlib"))
    diagnostics_json = spec.case_root / "subprocess_diagnostics.json"
    diag = run_subprocess_with_diagnostics(
        command=[str(part) for part in command],
        cwd=REPO_ROOT,
        env=env,
        stdout_log=spec.stdout_log,
        stderr_log=spec.stderr_log,
        diagnostics_json=diagnostics_json,
        resource_time=resource_time,
    )
    return_code = int(diag.return_code)
    rejection_path = spec.expected_summary_json.with_name("schur_summary_rejection.json")
    if return_code != 0 and rejection_path.exists():
        status = "rejected"
        reason = "schur_summary_rejection"
        try:
            rejection_payload = json.loads(rejection_path.read_text(encoding="utf-8"))
            if isinstance(rejection_payload, Mapping) and rejection_payload.get("reason"):
                reason = str(rejection_payload["reason"])
        except Exception:
            reason = "schur_summary_rejection"
    elif return_code != 0:
        status = "failed"
        reason = f"subprocess_return_code_{return_code}"
    elif not spec.expected_summary_json.exists():
        status = "rejected"
        reason = "missing_expected_summary_json"
    else:
        status = "completed"
        reason = None
    failure_class, failure_hint = classify_subprocess_failure(return_code)
    failure_fields = memory_failure_fields(spec=spec)
    return MonteCarloTrialResult(
        trial_id=spec.trial_id,
        status=status,
        return_code=return_code,
        started_at=diag.started_at,
        finished_at=diag.finished_at,
        elapsed_seconds=float(diag.elapsed_seconds),
        summary_json_path=spec.expected_summary_json if spec.expected_summary_json.exists() else None,
        matrix_npz_path=spec.expected_matrix_npz if spec.expected_matrix_npz.exists() else None,
        failure_reason=reason,
        failure_class=failure_class,
        failure_hint=failure_hint,
        last_stdout_line=failure_fields["last_stdout_line"] or None,
        last_stderr_line=failure_fields["last_stderr_line"] or None,
        memory_diagnostics_path=spec.memory_diagnostics_path
        if spec.memory_diagnostics_path.exists()
        else None,
        last_memory_stage=failure_fields["last_memory_stage"] or None,
        last_memory_rss_mb=(
            None
            if failure_fields["last_memory_rss_mb"] in {"", None}
            else float(failure_fields["last_memory_rss_mb"])
        ),
        last_memory_peak_rss_mb=(
            None
            if failure_fields["last_memory_peak_rss_mb"] in {"", None}
            else float(failure_fields["last_memory_peak_rss_mb"])
        ),
    )


def run_trial_pool(
    plan: Sequence[MonteCarloTrialSpec],
    config: MonteCarloRunConfig,
    *,
    commands: Mapping[int, Sequence[str]],
) -> dict[int, MonteCarloTrialResult]:
    """Run trial subprocesses with a small thread-backed process pool."""

    execution_started = time.monotonic()
    results = load_trial_status(config.run_root / "run_status.csv")
    pending = [
        spec
        for spec in plan
        if not (
            config.resume
            and valid_existing_summary(spec.expected_summary_json)
        )
    ]
    if pending:
        require_resource_time_available(config.resource_time)
    for spec in plan:
        if config.resume and valid_existing_summary(spec.expected_summary_json):
            results[spec.trial_id] = MonteCarloTrialResult(
                trial_id=spec.trial_id,
                status="skipped_completed",
                return_code=0,
                started_at=None,
                finished_at=None,
                elapsed_seconds=0.0,
                summary_json_path=spec.expected_summary_json,
                matrix_npz_path=spec.expected_matrix_npz if spec.expected_matrix_npz.exists() else None,
            )

    skipped = sum(1 for result in results.values() if result.status == "skipped_completed")
    mc_log(
        "execution.start",
        quiet=config.quiet,
        pending=len(pending),
        skipped=skipped,
        max_workers=max(1, int(config.max_workers)),
    )
    if skipped:
        mc_log("execution.resume", quiet=config.quiet, skipped=skipped)

    pending_queue = list(pending)
    active: dict[Future[MonteCarloTrialResult], tuple[MonteCarloTrialSpec, float]] = {}
    next_heartbeat = time.monotonic() + max(float(config.progress_interval_s), 0.0)
    progress_path = config.run_root / "progress.json"

    def submit_ready(pool: ThreadPoolExecutor) -> None:
        while pending_queue and len(active) < max(1, int(config.max_workers)):
            spec = pending_queue.pop(0)
            kwargs: dict[str, Any] = {"resume": False}
            if config.resource_time is not None:
                kwargs["resource_time"] = config.resource_time
            future = pool.submit(
                run_trial_subprocess,
                spec,
                commands[spec.trial_id],
                **kwargs,
            )
            active[future] = (spec, time.monotonic())
            fields: dict[str, Any] = {
                "trial_id": spec.trial_id,
                "active": f"{len(active)}/{max(1, int(config.max_workers))}",
                "trace_seed": spec.trace_seed,
                "noise_seed": spec.noise_seed,
                "stdout_log": spec.stdout_log,
            }
            if config.verbose:
                fields.update(
                    command_path=spec.command_path,
                    expected_summary_json=spec.expected_summary_json,
                )
            mc_log("trial.start", quiet=config.quiet, **fields)

    def emit_heartbeat() -> None:
        active_by_id = {
            spec.trial_id: (spec, started)
            for spec, started in active.values()
        }
        counts = _status_counts(results)
        elapsed = int(time.monotonic() - execution_started)
        mc_log(
            "heartbeat",
            quiet=config.quiet,
            completed=f"{counts['completed']}/{len(plan)}",
            active=len(active),
            failed=counts["failed"],
            rejected=counts["rejected"],
            skipped=counts["skipped"],
            elapsed_s=elapsed,
        )
        if int(config.tail_lines) > 0:
            for spec, started in sorted(active.values(), key=lambda item: item[0].trial_id):
                trial_elapsed = int(time.monotonic() - started)
                stdout_lines = tail_text_file(spec.stdout_log, n_lines=int(config.tail_lines))
                stderr_lines = tail_text_file(spec.stderr_log, n_lines=int(config.tail_lines))
                for line in stdout_lines:
                    mc_log(
                        "active",
                        quiet=config.quiet,
                        trial_id=spec.trial_id,
                        elapsed_s=trial_elapsed,
                        stream="stdout",
                        last_line=line,
                    )
                for line in stderr_lines:
                    mc_log(
                        "active",
                        quiet=config.quiet,
                        trial_id=spec.trial_id,
                        elapsed_s=trial_elapsed,
                        stream="stderr",
                        last_line=line,
                    )
        write_progress_json(
            progress_path,
            total_trials=len(plan),
            results=results,
            active=active_by_id,
            started_at_monotonic=execution_started,
        )

    with ThreadPoolExecutor(max_workers=max(1, int(config.max_workers))) as pool:
        submit_ready(pool)
        while active:
            timeout = max(float(config.progress_interval_s), 0.0)
            done, _not_done = wait(tuple(active), timeout=timeout, return_when=FIRST_COMPLETED)
            now = time.monotonic()
            if not done and now >= next_heartbeat:
                emit_heartbeat()
                next_heartbeat = now + max(float(config.progress_interval_s), 0.0)
                continue
            for future in done:
                spec, started = active.pop(future)
                result = future.result()
                results[spec.trial_id] = result
                write_run_status_csv(config.run_root / "run_status.csv", plan, results)
                event = "trial.done" if result.status in {"completed", "skipped_completed"} else "trial.failed"
                mc_log(
                    event,
                    quiet=config.quiet,
                    trial_id=spec.trial_id,
                    status=result.status,
                    elapsed_s=(
                        int(result.elapsed_seconds)
                        if result.elapsed_seconds is not None
                        else int(time.monotonic() - started)
                    ),
                    summary_json=result.summary_json_path,
                    failure_reason=result.failure_reason,
                    failure_class=result.failure_class,
                    failure_hint=result.failure_hint,
                    last_memory_stage=result.last_memory_stage,
                    last_memory_peak_rss_mb=result.last_memory_peak_rss_mb,
                )
                if config.memory_diagnostics and result.status == "failed":
                    for record in tail_memory_diagnostics(
                        spec.memory_diagnostics_path,
                        n_lines=int(config.memory_progress_tail_lines),
                    ):
                        mc_log(
                            "trial.memory_tail",
                            quiet=config.quiet,
                            trial_id=spec.trial_id,
                            stage=record.get("stage"),
                            rss_mb=record.get("rss_mb"),
                            peak_rss_mb=record.get("peak_rss_mb"),
                        )
                if config.fail_fast and result.status not in {"completed", "skipped_completed"}:
                    emit_heartbeat()
                    raise RuntimeError(f"Trial {spec.trial_name} failed: {result.failure_reason}")
            submit_ready(pool)
            if now >= next_heartbeat and active:
                emit_heartbeat()
                next_heartbeat = now + max(float(config.progress_interval_s), 0.0)
    counts = _status_counts(results)
    mc_log(
        "execution.done",
        quiet=config.quiet,
        completed=counts["completed"],
        failed=counts["failed"],
        rejected=counts["rejected"],
        skipped=counts["skipped"],
        elapsed_s=int(time.monotonic() - execution_started),
    )
    return results


def _status_counts(results: Mapping[int, MonteCarloTrialResult]) -> dict[str, int]:
    completed = sum(1 for result in results.values() if result.status == "completed")
    skipped = sum(1 for result in results.values() if result.status == "skipped_completed")
    failed = sum(1 for result in results.values() if result.status == "failed")
    rejected = sum(1 for result in results.values() if result.status == "rejected")
    return {
        "completed": completed,
        "failed": failed,
        "rejected": rejected,
        "skipped": skipped,
    }


def resolve_matrix_path(summary_json_path: Path, payload: Mapping[str, Any]) -> Path:
    raw = payload.get("matrix_artifact_path")
    if not isinstance(raw, str) or not raw:
        return summary_json_path.with_name("subblock_summary_matrices.npz")
    path = Path(raw)
    if not path.is_absolute():
        path = (summary_json_path.parent / path).resolve()
    return path


def safe_correlation(info: np.ndarray) -> np.ndarray:
    diag = np.diag(info).astype(float)
    denom = np.sqrt(np.maximum(diag, 0.0))
    outer = np.outer(denom, denom)
    corr = np.full_like(info, np.nan, dtype=float)
    mask = outer > 0.0
    corr[mask] = info[mask] / outer[mask]
    return corr


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {key: None for key in ("mean", "std", "min", "max", "median", "p16", "p84")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p16": float(np.percentile(arr, 16)),
        "p84": float(np.percentile(arr, 84)),
    }


def compute_score_residual_metrics(
    *,
    info: np.ndarray,
    score: np.ndarray,
    theta_ref: np.ndarray,
    theta_true: np.ndarray | None,
    eig_floor_abs: float = 1.0e-10,
    eig_floor_rel: float = 1.0e-10,
) -> dict[str, Any] | None:
    """Compute residual and PSD-safe whitened residual diagnostics.

    Called by ``aggregate_schur_summary_trials`` when a truth vector is
    available.  Residuals use ``r_b = g_b - S_b @ (theta_ref_b - theta_true)``.
    Whitening uses an eigendecomposition of ``S_b`` with absolute and relative
    eigenvalue flooring; this is diagnostic only and should not be interpreted
    as a final score-noise model.
    """

    if theta_true is None:
        return None
    residual = score - info @ (theta_ref - theta_true)
    sym = 0.5 * (info + info.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    raw_max = float(np.max(eigvals)) if eigvals.size else 0.0
    floor = max(float(eig_floor_abs), float(eig_floor_rel) * max(raw_max, 0.0))
    floored = np.maximum(eigvals, floor)
    whitened = eigvecs @ ((eigvecs.T @ residual) / np.sqrt(floored))
    return {
        "residual": residual,
        "whitened": whitened,
        "whitened_norm": float(np.linalg.norm(whitened)),
        "whitening_eig_floor_abs": float(eig_floor_abs),
        "whitening_eig_floor_rel": float(eig_floor_rel),
        "raw_min_eigenvalue": float(np.min(eigvals)) if eigvals.size else math.nan,
        "raw_max_eigenvalue": raw_max,
        "n_eigenvalues_floored": int(np.sum(eigvals < floor)),
    }


def _load_explicit_truth(path: Path, labels: Sequence[str]) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Explicit truth JSON must be a mapping from theta label to value.")
    missing = [label for label in labels if label not in payload]
    if missing:
        raise ValueError(f"Explicit truth JSON is missing labels: {missing}")
    return np.asarray([float(payload[label]) for label in labels], dtype=float)


def aggregate_schur_summary_trials(
    *,
    config: MonteCarloRunConfig,
    plan: Sequence[MonteCarloTrialSpec],
    results: Mapping[int, MonteCarloTrialResult],
) -> dict[str, Any]:
    """Aggregate accepted image-backed Schur summaries into tables and plots.

    The aggregation is intentionally conservative: labels must match across
    accepted summaries, matrices must be finite and square, and rejected trials
    are written to ``failed_trials.csv`` with reasons.  Future non-Schur
    aggregators can reuse the plan/status inputs while replacing this function.
    """

    aggregate_root = config.run_root / "aggregate"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    accepted_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    eigen_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    whitened_rows: list[dict[str, Any]] = []
    all_infos: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_eigs: list[np.ndarray] = []
    theta_labels_ref: tuple[str, ...] | None = None
    theta_true: np.ndarray | None = None
    manifest_warnings: list[str] = []

    for spec in plan:
        result = results.get(spec.trial_id)
        if result is None:
            failed_rows.append(
                _failed_row(spec, "not_executed", status="planned_not_run")
            )
            continue
        if result.status in {"planned", "not_run"}:
            failed_rows.append(
                _failed_row(spec, "not_executed", status="planned_not_run", result=result)
            )
            continue
        if result.return_code not in {0, None} or result.status in {"failed", "rejected"}:
            failed_rows.append(
                _failed_row(
                    spec,
                    result.failure_reason or result.status,
                    status=(
                        "failed_subprocess"
                        if result.status == "failed" or result.return_code not in {0, None}
                        else "completed_rejected"
                    ),
                    result=result,
                )
            )
            continue
        summary_path = result.summary_json_path or spec.expected_summary_json
        if not summary_path.exists():
            failed_rows.append(
                _failed_row(spec, "missing_summary_json", status="completed_missing_summary", result=result)
            )
            continue
        try:
            payload = load_subblock_summary_artifact_payload(summary_path)
            summary = load_subblock_summary(summary_path)
        except Exception as exc:
            failed_rows.append(
                _failed_row(spec, f"load_failed: {exc}", status="completed_load_failed", result=result)
            )
            continue

        labels = tuple(summary.theta_labels)
        if theta_labels_ref is None:
            theta_labels_ref = labels
            if config.truth_mode == "summary_theta_ref":
                theta_true = np.asarray(summary.theta_ref, dtype=float)
            elif config.truth_mode == "explicit":
                if config.truth_json is None:
                    raise ValueError("truth_mode='explicit' requires --truth-json.")
                theta_true = _load_explicit_truth(config.truth_json, labels)
        elif labels != theta_labels_ref:
            failed_rows.append(
                _failed_row(spec, "theta_labels_mismatch", status="completed_rejected", result=result)
            )
            continue

        info = np.asarray(summary.reduced_information, dtype=float)
        score = np.asarray(summary.reduced_score, dtype=float)
        if info.shape != (len(labels), len(labels)):
            failed_rows.append(
                _failed_row(spec, "reduced_information_shape_mismatch", status="completed_rejected", result=result)
            )
            continue
        if score.shape != (len(labels),):
            failed_rows.append(
                _failed_row(spec, "reduced_score_shape_mismatch", status="completed_rejected", result=result)
            )
            continue
        if not np.all(np.isfinite(info)) or not np.all(np.isfinite(score)):
            failed_rows.append(
                _failed_row(spec, "nonfinite_summary_arrays", status="completed_rejected", result=result)
            )
            continue
        eigvals = np.linalg.eigvalsh(0.5 * (info + info.T))
        psd_tolerance = 1.0e-8 * max(1.0, float(np.max(np.abs(eigvals))))
        psd_issue = bool(np.min(eigvals) < -psd_tolerance)

        matrix_path = resolve_matrix_path(summary_path, payload)
        recovered_required_missing = (
            spec.phi_ref == "recovered"
            and not (
                isinstance(payload.get("recovered_reference"), Mapping)
                or (spec.case_root / "study" / "schur_summary" / "reference_inference").exists()
            )
        )
        if recovered_required_missing:
            failed_rows.append(
                _failed_row(spec, "missing_recovered_reference_artifacts", status="completed_rejected", result=result)
            )
            continue

        corr = safe_correlation(info)
        residual = compute_score_residual_metrics(
            info=info,
            score=score,
            theta_ref=np.asarray(summary.theta_ref, dtype=float),
            theta_true=theta_true,
        )
        diag = np.diag(info)
        normalized_score = score / np.sqrt(np.maximum(diag, TINY))
        diagnostics = dict(summary.diagnostics)
        dims = payload.get("dimensions") if isinstance(payload.get("dimensions"), Mapping) else {}
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        fit_warning = extract_reference_frame_quality_from_summary_payload(
            payload,
            summary_json_path=summary_path,
            threshold=float(config.schur_frame_chi2_threshold),
        )

        accepted_rows.append(
            {
                "trial_id": spec.trial_id,
                "summary_json_path": str(summary_path),
                "theta_labels_hash": hashlib.sha1("|".join(labels).encode("utf-8")).hexdigest(),
                "theta_labels": "|".join(labels),
                "status": "accepted",
            }
        )
        summary_rows.append(
            {
                "trial_id": spec.trial_id,
                "trial_name": spec.trial_name,
                "summary_json_path": str(summary_path),
                "matrix_npz_path": str(matrix_path),
                "subblock_id": summary.subblock_id,
                "theta_labels": "|".join(labels),
                "phi_ref_source": _nested_get(payload, ("metadata", "phi_ref_source")),
                "phi_ref_mode": spec.phi_ref,
                "n_theta": len(labels),
                "n_phi": dims.get("n_phi", ""),
                "n_frames": spec.n_frames,
                "max_dense_dim": "" if spec.max_dense_dim is None else spec.max_dense_dim,
                "noise_mode": spec.noise_mode,
                "trace_seed": spec.trace_seed,
                "noise_seed": spec.noise_seed,
                "schur_curvature_method_used": _curvature_method_used(payload),
                "structured_curvature_used": bool(metadata.get("structured_curvature_used", False)),
                "frame_quality_policy": diagnostics.get("frame_quality_policy", ""),
                "frame_quality_good_frame_count": diagnostics.get(
                    "frame_quality_good_frame_count",
                    "",
                ),
                "frame_quality_bad_frame_count": diagnostics.get(
                    "frame_quality_bad_frame_count",
                    "",
                ),
                "frame_quality_bad_frame_indices": "|".join(
                    str(index)
                    for index in diagnostics.get("frame_quality_bad_frame_indices", [])
                )
                if isinstance(
                    diagnostics.get("frame_quality_bad_frame_indices"), list
                )
                else diagnostics.get("frame_quality_bad_frame_indices", ""),
                "frame_quality_effective_frame_fraction": diagnostics.get(
                    "frame_quality_effective_frame_fraction",
                    "",
                ),
                "variance_floor": spec.variance_floor,
                "reference_diagnostics_profile": spec.reference_diagnostics_profile,
                "reference_optimizer_kind": spec.reference_optimizer_kind or "",
                "reference_base_lr": "" if spec.reference_base_lr is None else spec.reference_base_lr,
                "reference_n_iter": "" if spec.reference_n_iter is None else spec.reference_n_iter,
                "reference_preconditioning_enabled": (
                    ""
                    if spec.reference_preconditioning_enabled is None
                    else spec.reference_preconditioning_enabled
                ),
                "reference_preconditioning_method": spec.reference_preconditioning_method or "",
                "reference_preconditioning_reference": spec.reference_preconditioning_reference or "",
                "reference_preconditioning_lr_clip": _format_lr_clip(
                    spec.reference_preconditioning_lr_clip
                ),
                "status": "accepted_with_psd_warning" if psd_issue else "accepted",
                "trace": float(np.trace(info)),
                "frobenius_norm": float(np.linalg.norm(info, ord="fro")),
                "rank_estimate": int(np.linalg.matrix_rank(info)),
                "min_eigenvalue": float(np.min(eigvals)),
                "max_eigenvalue": float(np.max(eigvals)),
                "condition_number": _condition_number(eigvals),
                "score_norm": float(np.linalg.norm(score)),
                "diagnostics_present": bool(diagnostics),
                **fit_warning,
            }
        )
        for index, label in enumerate(labels):
            value = float(diag[index])
            diag_rows.append(
                {
                    "trial_id": spec.trial_id,
                    "label": label,
                    "index": index,
                    "value": value,
                    "log10_value": math.log10(value) if value > 0.0 and math.isfinite(value) else "",
                }
            )
            score_rows.append(
                {
                    "trial_id": spec.trial_id,
                    "label": label,
                    "index": index,
                    "value": float(score[index]),
                    "normalized_value": float(normalized_score[index]),
                }
            )
            if residual is not None:
                whitened_rows.append(
                    {
                        "trial_id": spec.trial_id,
                        "label": label,
                        "index": index,
                        "score_residual": float(residual["residual"][index]),
                        "whitened_residual": float(residual["whitened"][index]),
                        "whitened_residual_norm": float(residual["whitened_norm"]),
                        "whitening_eig_floor_abs": residual["whitening_eig_floor_abs"],
                        "whitening_eig_floor_rel": residual["whitening_eig_floor_rel"],
                        "raw_min_eigenvalue": residual["raw_min_eigenvalue"],
                        "raw_max_eigenvalue": residual["raw_max_eigenvalue"],
                        "n_eigenvalues_floored": residual["n_eigenvalues_floored"],
                    }
                )
        for index, value in enumerate(eigvals):
            eigen_rows.append({"trial_id": spec.trial_id, "eigen_index": index, "eigenvalue": float(value)})
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                corr_rows.append(
                    {
                        "trial_id": spec.trial_id,
                        "i": i,
                        "j": j,
                        "label_i": label_i,
                        "label_j": label_j,
                        "correlation": float(corr[i, j]) if np.isfinite(corr[i, j]) else "",
                    }
                )
        all_infos.append(info)
        all_scores.append(score)
        all_eigs.append(eigvals)

    if theta_labels_ref is None:
        manifest_warnings.append("No summaries accepted; aggregate tables contain failures only.")
    elif theta_true is None:
        manifest_warnings.append("Truth unavailable; residual and whitened residual outputs skipped.")

    _write_csv(aggregate_root / "accepted_summary_paths.csv", accepted_rows)
    _write_csv(aggregate_root / "failed_trials.csv", failed_rows)
    memory_failure_rows = [
        _memory_failure_summary_row(spec, result)
        for spec in plan
        for result in [results.get(spec.trial_id)]
        if result is not None
        and (result.status == "failed" or result.return_code not in {0, None})
    ]
    if memory_failure_rows:
        _write_csv(aggregate_root / "memory_failure_summary.csv", memory_failure_rows)
    _write_csv(aggregate_root / "summary_metrics.csv", summary_rows)
    _write_csv(aggregate_root / "matrix_diagonal_entries.csv", diag_rows)
    _write_csv(aggregate_root / "matrix_correlation_entries.csv", corr_rows)
    _write_csv(aggregate_root / "score_entries.csv", score_rows)
    _write_csv(aggregate_root / "eigenvalue_metrics.csv", eigen_rows)
    _write_csv(aggregate_root / "whitened_score_residuals.csv", whitened_rows)

    aggregate_summary = _build_aggregate_summary(
        config=config,
        plan=plan,
        results=results,
        failed_rows=failed_rows,
        summary_rows=summary_rows,
        diag_rows=diag_rows,
        score_rows=score_rows,
        eigen_rows=eigen_rows,
        whitened_rows=whitened_rows,
        theta_labels=theta_labels_ref or (),
        warnings=manifest_warnings,
    )
    (aggregate_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate_summary, indent=2),
        encoding="utf-8",
    )
    if config.plots and accepted_rows:
        write_aggregate_plots(
            aggregate_root=aggregate_root,
            theta_labels=theta_labels_ref or (),
            infos=all_infos,
            scores=all_scores,
            eigs=all_eigs,
            whitened_rows=whitened_rows,
        )
    return aggregate_summary


def _condition_number(eigvals: np.ndarray) -> float:
    positive = eigvals[eigvals > TINY]
    if positive.size == 0:
        return math.inf
    return float(np.max(np.abs(eigvals)) / np.min(positive))


def _nested_get(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return "" if current is None else current


def _curvature_method_used(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    diagnostics = payload.get("diagnostics") if isinstance(payload.get("diagnostics"), Mapping) else {}
    for source in (metadata, diagnostics, payload):
        value = source.get("schur_curvature_method_used") or source.get("curvature_method_used")
        if value is not None:
            return str(value)
    return ""


def _candidate_recovered_manifest_paths(
    payload: Mapping[str, Any],
    *,
    summary_json_path: Path,
) -> list[Path]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    recovered = (
        metadata.get("recovered_reference")
        if isinstance(metadata.get("recovered_reference"), Mapping)
        else {}
    )
    candidates: list[Path] = []
    manifest_value = recovered.get("manifest_json") if isinstance(recovered, Mapping) else None
    summary_dir = summary_json_path.parent
    case_value = metadata.get("case_root") or payload.get("case_root")
    case_root = Path(case_value).expanduser() if isinstance(case_value, str) and case_value.strip() else None
    output_value = recovered.get("output_dir") if isinstance(recovered, Mapping) else None
    output_dir = Path(output_value).expanduser() if isinstance(output_value, str) and output_value.strip() else None
    if isinstance(manifest_value, str) and manifest_value.strip():
        raw = Path(manifest_value).expanduser()
        candidates.append(raw)
        if not raw.is_absolute():
            candidates.append(summary_dir / raw)
            if case_root is not None:
                candidates.append(case_root / raw)
            if output_dir is not None:
                candidates.append(output_dir / raw)
    if output_dir is not None:
        candidates.append(output_dir / "manifest.json")
    if case_root is not None:
        candidates.extend(
            sorted(
                (case_root / "study" / "schur_summary" / "reference_inference").glob(
                    "**/manifest.json"
                )
            )
        )
    return candidates


def _resolve_recovered_manifest_path(
    payload: Mapping[str, Any],
    *,
    summary_json_path: Path,
) -> Path | None:
    for candidate in _candidate_recovered_manifest_paths(
        payload,
        summary_json_path=summary_json_path,
    ):
        path = candidate if candidate.is_absolute() else candidate.resolve()
        if path.exists():
            return path.resolve()
    return None


def extract_reference_frame_quality_from_summary_payload(
    payload: Mapping[str, Any],
    *,
    summary_json_path: Path,
    threshold: float,
) -> dict[str, Any]:
    base = {
        "reference_final_max_frame_reduced_chi2": "",
        "reference_final_median_frame_reduced_chi2": "",
        "reference_final_block_reduced_chi2": "",
        "reference_failed_frame_count_chi2_gt_threshold": "",
        "reference_failed_frame_indices_chi2_gt_threshold": "",
        "reference_frame_chi2_threshold": float(threshold),
        "reference_frame_quality_source": "",
        "reference_frame_quality_error": "",
        "fit_warning": "",
    }
    manifest_path = _resolve_recovered_manifest_path(
        payload,
        summary_json_path=summary_json_path,
    )
    if manifest_path is None:
        return {**base, "reference_frame_quality_source": "missing_manifest"}
    try:
        manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        metrics = (
            manifest_payload.get("metrics", {})
            if isinstance(manifest_payload.get("metrics"), Mapping)
            else {}
        )
        chi2 = manifest_payload.get("chi2") or metrics.get("chi2") or {}
        final_model = chi2.get("final_model", {}) if isinstance(chi2, Mapping) else {}
        per_frame = final_model.get("per_frame_reduced_chi2", [])
        values = []
        for value in per_frame:
            numeric = float(value)
            if math.isfinite(numeric):
                values.append(numeric)
        max_frame = max(values) if values else None
        median_frame = float(np.median(np.asarray(values, dtype=float))) if values else None
        failed = [index for index, value in enumerate(values) if value > float(threshold)]
        block = final_model.get("block_reduced_chi2")
        block_value = None if block is None else float(block)
    except Exception as exc:
        return {
            **base,
            "reference_frame_quality_source": "manifest_parse_error",
            "reference_frame_quality_error": str(exc),
        }
    warning = ""
    if max_frame is not None and max_frame > float(threshold):
        warning = "reference_final_max_frame_reduced_chi2_high"
    return {
        **base,
        "reference_final_max_frame_reduced_chi2": ""
        if max_frame is None
        else max_frame,
        "reference_final_median_frame_reduced_chi2": ""
        if median_frame is None
        else median_frame,
        "reference_final_block_reduced_chi2": ""
        if block_value is None
        else block_value,
        "reference_failed_frame_count_chi2_gt_threshold": len(failed),
        "reference_failed_frame_indices_chi2_gt_threshold": "|".join(
            str(index) for index in failed
        ),
        "reference_frame_quality_source": "found",
        "fit_warning": warning,
    }


def _fit_warning_from_recovered_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return extract_reference_frame_quality_from_summary_payload(
        payload,
        summary_json_path=Path("."),
        threshold=FIT_WARNING_MAX_FRAME_REDUCED_CHI2,
    )


def _failed_row(
    spec: MonteCarloTrialSpec,
    reason: str,
    *,
    status: str = "failed_or_rejected",
    result: MonteCarloTrialResult | None = None,
) -> dict[str, Any]:
    row = {
        "trial_id": spec.trial_id,
        "trial_name": spec.trial_name,
        "summary_json_path": str(spec.expected_summary_json),
        "status": status,
        "failure_reason": str(reason),
        "return_code": "" if result is None or result.return_code is None else result.return_code,
        "failure_class": "" if result is None else result.failure_class or "",
        "failure_hint": "" if result is None else result.failure_hint or "",
    }
    if result is not None:
        row.update(memory_failure_fields(spec=spec, result=result))
    return row


def _reference_inference_manifest_exists(spec: MonteCarloTrialSpec) -> bool:
    root = spec.case_root / "study" / "schur_summary" / "reference_inference"
    if not root.exists():
        return False
    try:
        return any(path.name == "manifest.json" for path in root.rglob("manifest.json"))
    except OSError:
        return False


def _memory_failure_summary_row(
    spec: MonteCarloTrialSpec,
    result: MonteCarloTrialResult,
) -> dict[str, Any]:
    fields = memory_failure_fields(spec=spec, result=result)
    return {
        "trial_id": spec.trial_id,
        "status": result.status,
        "return_code": "" if result.return_code is None else result.return_code,
        "failure_reason": result.failure_reason or "",
        "failure_class": result.failure_class or "",
        "failure_hint": result.failure_hint or "",
        "elapsed_seconds": ""
        if result.elapsed_seconds is None
        else f"{result.elapsed_seconds:.6f}",
        "last_stdout_line": fields["last_stdout_line"],
        "last_stderr_line": fields["last_stderr_line"],
        "memory_diagnostics_path": fields["memory_diagnostics_path"],
        "last_memory_stage": fields["last_memory_stage"],
        "last_memory_rss_mb": fields["last_memory_rss_mb"],
        "last_memory_peak_rss_mb": fields["last_memory_peak_rss_mb"],
        "expected_summary_json_exists": spec.expected_summary_json.exists(),
        "reference_inference_manifest_exists": _reference_inference_manifest_exists(spec),
        "subblock_summary_json_exists": spec.expected_summary_json.exists(),
        "matrix_npz_exists": spec.expected_matrix_npz.exists(),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _build_aggregate_summary(
    *,
    config: MonteCarloRunConfig,
    plan: Sequence[MonteCarloTrialSpec],
    results: Mapping[int, MonteCarloTrialResult],
    failed_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    diag_rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    eigen_rows: Sequence[Mapping[str, Any]],
    whitened_rows: Sequence[Mapping[str, Any]],
    theta_labels: Sequence[str],
    warnings: Sequence[str],
) -> dict[str, Any]:
    diag_by_label = {
        label: quantiles([float(row["value"]) for row in diag_rows if row.get("label") == label])
        for label in theta_labels
    }
    score_by_label = {
        label: quantiles([float(row["value"]) for row in score_rows if row.get("label") == label])
        for label in theta_labels
    }
    eig_by_index: dict[str, dict[str, float | None]] = {}
    for row in eigen_rows:
        key = str(row["eigen_index"])
        eig_by_index.setdefault(key, {"_values": []})["_values"].append(float(row["eigenvalue"]))  # type: ignore[index]
    eig_quantiles = {key: quantiles(value["_values"]) for key, value in eig_by_index.items()}  # type: ignore[index]
    condition_values = [float(row["condition_number"]) for row in summary_rows if np.isfinite(float(row["condition_number"]))]
    score_norm_values = [float(row["score_norm"]) for row in summary_rows]
    whitened_values = [
        float(row["whitened_residual"])
        for row in whitened_rows
        if row.get("whitened_residual") not in {"", None}
    ]
    whitened_arr = np.asarray(whitened_values, dtype=float)
    alpha = None
    if whitened_arr.size:
        by_label_variances = []
        for label in theta_labels:
            vals = np.asarray(
                [float(row["whitened_residual"]) for row in whitened_rows if row.get("label") == label],
                dtype=float,
            )
            if vals.size:
                by_label_variances.append(float(np.var(vals)))
        alpha = float(np.median(by_label_variances)) if by_label_variances else None
    failed_subprocess_count = sum(
        1 for row in failed_rows if row.get("status") == "failed_subprocess"
    )
    planned_not_run_count = sum(
        1 for row in failed_rows if row.get("status") == "planned_not_run"
    )
    completed_missing_summary_count = sum(
        1 for row in failed_rows if row.get("status") == "completed_missing_summary"
    )
    failed_probable_sigkill_count = sum(
        1
        for result in results.values()
        if result.status == "failed" and result.failure_class == "probable_sigkill"
    )
    failed_other_count = sum(
        1
        for result in results.values()
        if result.status == "failed" and result.failure_class != "probable_sigkill"
    )
    memory_diagnostics_available_count = sum(
        1
        for spec in plan
        if (
            (
                results.get(spec.trial_id) is not None
                and results[spec.trial_id].memory_diagnostics_path is not None
                and results[spec.trial_id].memory_diagnostics_path.exists()
            )
            or spec.memory_diagnostics_path.exists()
        )
    )
    return {
        "schema_version": "obs_subblock_monte_carlo_aggregate.v1",
        "created_at": now_iso_utc(),
        "n_trials_planned": len(plan),
        "n_trials_completed": len(summary_rows),
        "n_trials_failed": failed_subprocess_count,
        "n_trials_rejected_or_invalid": len(failed_rows)
        - failed_subprocess_count
        - planned_not_run_count,
        "n_planned_not_run": planned_not_run_count,
        "n_completed_missing_summary": completed_missing_summary_count,
        "n_failed_probable_sigkill": failed_probable_sigkill_count,
        "n_failed_other": failed_other_count,
        "n_memory_diagnostics_available": memory_diagnostics_available_count,
        "n_summaries_accepted": len(summary_rows),
        "theta_labels": list(theta_labels),
        "seed_policy": config.seed_policy,
        "base_seed": int(config.seed),
        "study_defaults": {
            "study_mode": config.study_mode,
            "n_frames": config.n_frames,
            "noise": config.noise,
            "phi_ref": config.phi_ref,
            "schur_curvature_method": config.schur_curvature_method,
            "schur_frame_quality_policy": config.schur_frame_quality_policy,
            "schur_frame_chi2_threshold": config.schur_frame_chi2_threshold,
            "schur_frame_quality_missing": config.schur_frame_quality_missing,
            "schur_frame_mask_denominator": config.schur_frame_mask_denominator,
            "schur_frame_mask_min_good_frames": config.schur_frame_mask_min_good_frames,
            "max_dense_dim": config.max_dense_dim,
            "variance_floor": config.variance_floor,
            "reference_diagnostics_profile": config.reference_diagnostics_profile,
        },
        "s_diagonal_by_label": diag_by_label,
        "score_by_label": score_by_label,
        "eigenvalue_quantiles": eig_quantiles,
        "condition_number_quantiles": quantiles(condition_values),
        "score_norm_quantiles": quantiles(score_norm_values),
        "whitened_residual_mean": None if not whitened_arr.size else float(np.mean(whitened_arr)),
        "whitened_residual_std": None if not whitened_arr.size else float(np.std(whitened_arr)),
        "alpha_calibration_suggestion": {
            "whitened_residual_component_variance_mean": None
            if not whitened_arr.size
            else float(np.var(whitened_arr)),
            "whitened_residual_component_variance_median": alpha,
            "suggested_score_noise_alpha": alpha,
            "interpretation": "Empirical diagnostic only; not a final score-noise model.",
        },
        "warnings": list(warnings),
    }


def write_aggregate_plots(
    *,
    aggregate_root: Path,
    theta_labels: Sequence[str],
    infos: Sequence[np.ndarray],
    scores: Sequence[np.ndarray],
    eigs: Sequence[np.ndarray],
    whitened_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write headless Matplotlib diagnostic plots for accepted summaries."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plots_dir = aggregate_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    info_arr = np.asarray(infos, dtype=float)
    score_arr = np.asarray(scores, dtype=float)
    eig_arr = np.asarray(eigs, dtype=float)

    n = max(1, len(theta_labels))
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for index, label in enumerate(theta_labels):
        ax = axes[index // cols][index % cols]
        ax.hist(info_arr[:, index, index], bins=min(20, max(3, len(info_arr))))
        ax.set_title(label)
    _hide_unused_axes(axes, n)
    fig.tight_layout()
    fig.savefig(plots_dir / "s_diagonal_histograms.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)
    for index, label in enumerate(theta_labels):
        ax = axes[index // cols][index % cols]
        ax.hist(score_arr[:, index], bins=min(20, max(3, len(score_arr))))
        ax.set_title(label)
    _hide_unused_axes(axes, n)
    fig.tight_layout()
    fig.savefig(plots_dir / "score_entry_histograms.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(np.linalg.norm(score_arr, axis=1), bins=min(20, max(3, len(score_arr))))
    ax.set_xlabel("score norm")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(plots_dir / "score_norm_histogram.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = np.arange(eig_arr.shape[1])
    ax.plot(x, np.median(eig_arr, axis=0), marker="o", label="median")
    ax.fill_between(x, np.percentile(eig_arr, 16, axis=0), np.percentile(eig_arr, 84, axis=0), alpha=0.25)
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("eigenvalue")
    fig.tight_layout()
    fig.savefig(plots_dir / "eigenvalue_spectrum_quantiles.png", dpi=140)
    plt.close(fig)

    corr_arr = np.asarray([safe_correlation(info) for info in infos], dtype=float)
    for name, matrix in (
        ("correlation_mean_heatmap.png", np.nanmean(corr_arr, axis=0)),
        ("correlation_std_heatmap.png", np.nanstd(corr_arr, axis=0)),
    ):
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(matrix, vmin=-1 if "mean" in name else None, vmax=1 if "mean" in name else None)
        ax.set_xticks(np.arange(len(theta_labels)), labels=theta_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(theta_labels)), labels=theta_labels)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(plots_dir / name, dpi=140)
        plt.close(fig)

    values = [
        float(row["whitened_residual"])
        for row in whitened_rows
        if row.get("whitened_residual") not in {"", None}
    ]
    if values:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.hist(values, bins=min(30, max(4, len(values))))
        ax.set_xlabel("whitened residual component")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(plots_dir / "whitened_score_residual_histogram.png", dpi=140)
        plt.close(fig)


def _hide_unused_axes(axes: np.ndarray, used: int) -> None:
    for index, ax in enumerate(axes.ravel()):
        if index >= used:
            ax.set_visible(False)


def _config_from_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    raw = load_config_file(path)
    trial = raw.get("trial", {})
    if not isinstance(trial, Mapping):
        trial = {}
    reference_optimizer = trial.get("reference_optimizer", {})
    if not isinstance(reference_optimizer, Mapping):
        reference_optimizer = {}
    reference_optimizer_schedule = raw.get("trial", {}).get("reference_optimizer_schedule")
    if (
        not isinstance(reference_optimizer_schedule, Mapping)
        and reference_optimizer_schedule is not None
    ):
        reference_optimizer_schedule = {}
    preconditioning = reference_optimizer.get("preconditioning", {})
    if not isinstance(preconditioning, Mapping):
        preconditioning = {}
    schedule = reference_optimizer.get("schedule")
    if not isinstance(schedule, Mapping):
        schedule = reference_optimizer_schedule if isinstance(reference_optimizer_schedule, Mapping) else {}
    return {
        "run_name": raw.get("run", {}).get("run_name"),
        "results_root": raw.get("run", {}).get("results_root"),
        "n_trials": raw.get("run", {}).get("n_trials"),
        "max_workers": raw.get("run", {}).get("max_workers"),
        "seed": raw.get("run", {}).get("seed"),
        "seed_policy": raw.get("run", {}).get("seed_policy"),
        "resume": raw.get("run", {}).get("resume"),
        "fail_fast": raw.get("run", {}).get("fail_fast"),
        "study_mode": trial.get("study_mode"),
        "n_frames": trial.get("n_frames"),
        "noise": trial.get("noise"),
        "theta_keys": trial.get("theta_keys"),
        "phi_ref": trial.get("phi_ref"),
        "schur_curvature_method": trial.get("schur_curvature_method"),
        "schur_frame_quality_policy": trial.get("schur_frame_quality_policy"),
        "schur_frame_chi2_threshold": trial.get("schur_frame_chi2_threshold"),
        "schur_frame_quality_missing": trial.get("schur_frame_quality_missing"),
        "schur_frame_mask_denominator": trial.get("schur_frame_mask_denominator"),
        "schur_frame_mask_min_good_frames": trial.get("schur_frame_mask_min_good_frames"),
        "max_dense_dim": trial.get("max_dense_dim"),
        "variance_floor": trial.get("variance_floor"),
        "summary_information_scale": trial.get("summary_information_scale"),
        "reference_diagnostics_profile": trial.get("reference_diagnostics_profile"),
        "reference_optimizer_kind": reference_optimizer.get("kind"),
        "reference_base_lr": reference_optimizer.get("base_lr"),
        "reference_n_iter": reference_optimizer.get("n_iter"),
        "reference_optimizer_kwargs": reference_optimizer.get("kwargs"),
        "reference_schedule": schedule,
        "reference_preconditioning_enabled": preconditioning.get("enabled"),
        "reference_preconditioning_method": preconditioning.get("method"),
        "reference_preconditioning_reference": preconditioning.get("reference"),
        "reference_preconditioning_damping": preconditioning.get("damping"),
        "reference_preconditioning_eig_floor_rel": preconditioning.get("eig_floor_rel"),
        "reference_preconditioning_eig_floor_abs": preconditioning.get("eig_floor_abs"),
        "reference_preconditioning_lr_clip": preconditioning.get("lr_clip"),
        "aggregation_enabled": raw.get("aggregation", {}).get("enabled"),
        "truth_mode": raw.get("aggregation", {}).get("truth_mode"),
        "plots": raw.get("aggregation", {}).get("plots"),
        "quiet": raw.get("run", {}).get("quiet"),
        "progress_interval_s": raw.get("run", {}).get("progress_interval_s"),
        "tail_lines": raw.get("run", {}).get("tail_lines"),
        "memory_diagnostics": raw.get("run", {}).get("memory_diagnostics"),
        "memory_progress_tail_lines": raw.get("run", {}).get(
            "memory_progress_tail_lines"
        ),
        "verbose": raw.get("run", {}).get("verbose"),
    }


def _drop_none(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def build_config_from_args(args: argparse.Namespace) -> MonteCarloRunConfig:
    defaults = dataclasses.asdict(MonteCarloRunConfig(run_name="obs_subblock_mc"))
    file_cfg = _drop_none(_config_from_file(args.config))
    cli_cfg = _drop_none(
        {
            "run_name": args.run_name,
            "results_root": args.results_root,
            "n_trials": args.n_trials,
            "max_workers": args.max_workers,
            "seed": args.seed,
            "seed_policy": args.seed_policy,
            "plan_csv": args.plan_csv,
            "study_mode": args.study_mode,
            "n_frames": args.n_frames,
            "noise": args.noise,
            "theta_keys": args.theta_keys,
            "phi_ref": args.phi_ref,
            "schur_curvature_method": args.schur_curvature_method,
            "schur_frame_quality_policy": args.schur_frame_quality_policy,
            "schur_frame_chi2_threshold": args.schur_frame_chi2_threshold,
            "schur_frame_quality_missing": args.schur_frame_quality_missing,
            "schur_frame_mask_denominator": args.schur_frame_mask_denominator,
            "schur_frame_mask_min_good_frames": args.schur_frame_mask_min_good_frames,
            "variance_floor": args.variance_floor,
            "reference_diagnostics_profile": args.reference_diagnostics_profile,
            "reference_optimizer_kind": args.reference_optimizer_kind,
            "reference_base_lr": args.reference_base_lr,
            "reference_n_iter": args.reference_n_iter,
            "reference_optimizer_kwargs": (
                parse_reference_optimizer_kwargs(args.reference_optimizer_kwarg)
                if args.reference_optimizer_kwarg
                else None
            ),
            "reference_schedule": parse_reference_schedule_config(
                kind=args.reference_schedule_kind,
                warmup_steps=args.reference_schedule_warmup_steps,
                start_factor=args.reference_schedule_start_factor,
                min_factor=args.reference_schedule_min_factor,
                boundaries=args.reference_schedule_boundaries,
                factors=args.reference_schedule_factors,
                decay_rate=args.reference_schedule_decay_rate,
                transition_steps=args.reference_schedule_transition_steps,
                staircase=bool(args.reference_schedule_staircase),
                n_iter=args.reference_n_iter,
            ),
            "reference_preconditioning_enabled": args.reference_preconditioning_enabled,
            "reference_preconditioning_method": args.reference_preconditioning_method,
            "reference_preconditioning_reference": args.reference_preconditioning_reference,
            "reference_preconditioning_damping": args.reference_preconditioning_damping,
            "reference_preconditioning_eig_floor_rel": args.reference_preconditioning_eig_floor_rel,
            "reference_preconditioning_eig_floor_abs": args.reference_preconditioning_eig_floor_abs,
            "reference_preconditioning_lr_clip": (
                parse_reference_lr_clip(args.reference_preconditioning_lr_clip)
                if args.reference_preconditioning_lr_clip is not None
                else None
            ),
            "reference_early_stopping_enabled": args.reference_early_stopping_enabled,
            "reference_early_stopping_min_iter": args.reference_early_stopping_min_iter,
            "reference_early_stopping_patience": args.reference_early_stopping_patience,
            "reference_early_stopping_loss_rtol": args.reference_early_stopping_loss_rtol,
            "reference_early_stopping_loss_atol": args.reference_early_stopping_loss_atol,
            "reference_early_stopping_step_atol": args.reference_early_stopping_step_atol,
            "reference_early_stopping_grad_norm_atol": args.reference_early_stopping_grad_norm_atol,
            "reference_init_mode": args.reference_init_mode,
            "reuse_reference_inference": args.reuse_reference_inference,
            "schur_damping": args.schur_damping,
            "max_dense_dim": args.max_dense_dim,
            "summary_objective": args.summary_objective,
            "summary_information_scale": args.summary_information_scale,
            "validate_surrogate": args.validate_surrogate,
            "truth_mode": args.truth_mode,
            "truth_json": args.truth_json,
            "progress_interval_s": args.progress_interval,
            "tail_lines": args.tail_lines,
            "memory_progress_tail_lines": args.memory_progress_tail_lines,
            "resource_time": (
                None
                if args.resource_time is None
                else ("enabled" if args.resource_time else "disabled")
            ),
        }
    )
    merged = deep_merge(defaults, file_cfg)
    merged = deep_merge(merged, cli_cfg)
    for key in ("resume", "aggregate_only", "dry_run", "fail_fast", "quiet", "verbose"):
        value = getattr(args, key)
        if value is not None:
            merged[key] = bool(value)
    if args.no_plots:
        merged["plots"] = False
    if args.memory_diagnostics:
        merged["memory_diagnostics"] = True
    if args.profile_runtime:
        merged["profile_runtime"] = True
    if args.profile_runtime_detail is not None:
        merged["profile_runtime_detail"] = args.profile_runtime_detail
    merged["run_name"] = filesystem_safe_token(str(merged["run_name"]))
    merged["results_root"] = Path(merged["results_root"]).expanduser()
    merged["theta_keys"] = parse_theta_keys(merged["theta_keys"])
    merged["reference_optimizer_kwargs"] = parse_reference_optimizer_kwargs(
        merged.get("reference_optimizer_kwargs")
    )
    merged["reference_schedule"] = parse_reference_schedule_config(
        kind=(
            None
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("kind")
        ),
        warmup_steps=(
            None
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("warmup_steps")
        ),
        start_factor=(
            None
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("start_factor")
        ),
        min_factor=(
            None
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("min_factor")
        ),
        boundaries=(
            None
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("boundaries")
        ),
        factors=(
            None
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("factors")
        ),
        decay_rate=(
            None
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("decay_rate")
        ),
        transition_steps=(
            None
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("transition_steps")
        ),
        staircase=bool(
            False
            if merged.get("reference_schedule") is None
            else dict(merged["reference_schedule"]).get("staircase", False)
        ),
        n_iter=merged.get("reference_n_iter"),
    )
    merged["reference_preconditioning_lr_clip"] = parse_reference_lr_clip(
        merged.get("reference_preconditioning_lr_clip")
    )
    if merged["seed_policy"] not in SUPPORTED_SEED_POLICIES:
        raise ValueError(f"Unsupported --seed-policy: {merged['seed_policy']}")
    if merged.get("reference_optimizer_kind") not in {None, "sgd", "adam"}:
        raise ValueError("reference_optimizer_kind must be 'sgd' or 'adam'.")
    if merged["reference_optimizer_kwargs"]:
        merged["reference_optimizer_kwargs"] = normalize_optimizer_kwargs(
            str(merged.get("reference_optimizer_kind") or "adam"),
            merged["reference_optimizer_kwargs"],
            path="reference_optimizer_kwargs",
        )
    if merged.get("reference_base_lr") is not None:
        merged["reference_base_lr"] = float(merged["reference_base_lr"])
        if merged["reference_base_lr"] <= 0.0 or not math.isfinite(merged["reference_base_lr"]):
            raise ValueError("reference_base_lr must be a positive finite float.")
    if merged.get("reference_n_iter") is not None:
        merged["reference_n_iter"] = int(merged["reference_n_iter"])
        if merged["reference_n_iter"] <= 0:
            raise ValueError("reference_n_iter must be > 0.")
    if merged.get("reference_preconditioning_enabled") is not None:
        merged["reference_preconditioning_enabled"] = _parse_optional_bool(
            merged["reference_preconditioning_enabled"]
        )
    if merged.get("reference_preconditioning_reference") not in {
        None,
        "initial",
        "truth_when_available",
    }:
        raise ValueError(
            "reference_preconditioning_reference must be 'initial' or "
            "'truth_when_available'."
        )
    for key in (
        "reference_preconditioning_damping",
        "reference_preconditioning_eig_floor_rel",
        "reference_preconditioning_eig_floor_abs",
    ):
        if merged.get(key) is not None:
            merged[key] = float(merged[key])
            if merged[key] < 0.0 or not math.isfinite(merged[key]):
                raise ValueError(f"{key} must be a nonnegative finite float.")
    if merged["reference_diagnostics_profile"] not in SUPPORTED_REFERENCE_DIAGNOSTICS_PROFILES:
        raise ValueError(
            "Unsupported --reference-diagnostics-profile: "
            f"{merged['reference_diagnostics_profile']}. Expected one of: "
            f"{', '.join(SUPPORTED_REFERENCE_DIAGNOSTICS_PROFILES)}."
        )
    if merged["schur_frame_quality_policy"] not in SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES:
        raise ValueError("Unsupported --schur-frame-quality-policy.")
    if merged["schur_frame_quality_missing"] not in SUPPORTED_SCHUR_FRAME_QUALITY_MISSING_POLICIES:
        raise ValueError("Unsupported --schur-frame-quality-missing.")
    if merged["schur_frame_mask_denominator"] not in SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS:
        raise ValueError("Unsupported --schur-frame-mask-denominator.")
    if merged["summary_information_scale"] not in {
        "summed_likelihood",
        "optimizer",
    }:
        raise ValueError("Unsupported --summary-information-scale.")
    merged["schur_frame_chi2_threshold"] = float(merged["schur_frame_chi2_threshold"])
    if (
        merged["schur_frame_chi2_threshold"] <= 0.0
        or not math.isfinite(merged["schur_frame_chi2_threshold"])
    ):
        raise ValueError("--schur-frame-chi2-threshold must be positive.")
    merged["schur_frame_mask_min_good_frames"] = int(
        merged["schur_frame_mask_min_good_frames"]
    )
    if merged["schur_frame_mask_min_good_frames"] < 1:
        raise ValueError("--schur-frame-mask-min-good-frames must be >= 1.")
    if int(merged["max_workers"]) < 1:
        raise ValueError("--max-workers must be >= 1.")
    if merged.get("max_dense_dim") is not None:
        merged["max_dense_dim"] = int(merged["max_dense_dim"])
        if merged["max_dense_dim"] < 1:
            raise ValueError("--max-dense-dim must be >= 1.")
    if float(merged["progress_interval_s"]) < 0.0:
        raise ValueError("--progress-interval must be >= 0.")
    if int(merged["tail_lines"]) < 0:
        raise ValueError("--tail-lines must be >= 0.")
    if int(merged["memory_progress_tail_lines"]) < 0:
        raise ValueError("--memory-progress-tail-lines must be >= 0.")
    return MonteCarloRunConfig(**merged)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run observation sub-block Monte Carlo trials.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-policy", choices=SUPPORTED_SEED_POLICIES, default=None)
    parser.add_argument("--plan-csv", type=Path, default=None)
    parser.add_argument("--study-mode", default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--noise", choices=("inherit", "enabled", "disabled"), default=None)
    parser.add_argument("--theta-keys", default=None)
    parser.add_argument("--phi-ref", choices=("recovered", "truth_when_available", "truth", "init"), default=None)
    parser.add_argument("--schur-curvature-method", choices=("auto", "dense", "structured_independent_frames"), default=None)
    parser.add_argument("--schur-frame-quality-policy", choices=SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES, default=None)
    parser.add_argument("--schur-frame-chi2-threshold", type=float, default=None)
    parser.add_argument("--schur-frame-quality-missing", choices=SUPPORTED_SCHUR_FRAME_QUALITY_MISSING_POLICIES, default=None)
    parser.add_argument("--schur-frame-mask-denominator", choices=SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS, default=None)
    parser.add_argument("--schur-frame-mask-min-good-frames", type=int, default=None)
    parser.add_argument("--variance-floor", type=float, default=None)
    parser.add_argument(
        "--reference-diagnostics-profile",
        choices=SUPPORTED_REFERENCE_DIAGNOSTICS_PROFILES,
        default=None,
        help=(
            "Reference-inference diagnostics profile forwarded to "
            "run_obs_subblock_study.py. Use none for benchmarks, basic for "
            "small review runs, review/full for heavier diagnostics."
        ),
    )
    parser.add_argument("--reference-optimizer-kind", choices=("sgd", "adam"), default=None)
    parser.add_argument("--reference-base-lr", type=float, default=None)
    parser.add_argument("--reference-n-iter", type=int, default=None)
    parser.add_argument(
        "--reference-optimizer-kwarg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
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
    )
    preconditioning_group.add_argument(
        "--reference-preconditioning-disabled",
        dest="reference_preconditioning_enabled",
        action="store_const",
        const=False,
    )
    parser.add_argument("--reference-preconditioning-method", default=None)
    parser.add_argument(
        "--reference-preconditioning-reference",
        choices=("initial", "truth_when_available"),
        default=None,
    )
    parser.add_argument("--reference-preconditioning-damping", type=float, default=None)
    parser.add_argument("--reference-preconditioning-eig-floor-rel", type=float, default=None)
    parser.add_argument("--reference-preconditioning-eig-floor-abs", type=float, default=None)
    parser.add_argument("--reference-preconditioning-lr-clip", default=None)
    parser.add_argument("--reference-early-stopping", dest="reference_early_stopping_enabled", action="store_true", default=None)
    parser.add_argument("--reference-early-stopping-min-iter", type=int, default=None)
    parser.add_argument("--reference-early-stopping-patience", type=int, default=None)
    parser.add_argument("--reference-early-stopping-loss-rtol", type=float, default=None)
    parser.add_argument("--reference-early-stopping-loss-atol", type=float, default=None)
    parser.add_argument("--reference-early-stopping-step-atol", type=float, default=None)
    parser.add_argument("--reference-early-stopping-grad-norm-atol", type=float, default=None)
    parser.add_argument("--reference-init-mode", choices=("initial", "truth_when_available"), default=None)
    parser.add_argument("--reuse-reference-inference", default=None)
    parser.add_argument("--schur-damping", type=float, default=None)
    parser.add_argument("--max-dense-dim", type=int, default=None)
    parser.add_argument("--summary-objective", choices=("data_only", "full_objective"), default=None)
    parser.add_argument(
        "--summary-information-scale",
        choices=("summed_likelihood", "optimizer"),
        default=None,
    )
    parser.add_argument("--validate-surrogate", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--truth-mode", choices=("summary_theta_ref", "explicit"), default=None)
    parser.add_argument("--truth-json", type=Path, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--aggregate-only", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--fail-fast", action="store_true", default=None)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--progress-interval", type=float, default=None)
    parser.add_argument("--tail-lines", type=int, default=None)
    parser.add_argument("--profile-runtime", action="store_true", default=False, help="Forward runtime profiling to child study runs.")
    parser.add_argument("--profile-runtime-detail", choices=("basic","full"), default="basic", help="Forwarded runtime profiling detail level.")
    parser.add_argument(
        "--memory-diagnostics",
        action="store_true",
        default=False,
        help=(
            "Forward --memory-diagnostics to each study subprocess and include "
            "parent-side memory failure classification."
        ),
    )
    parser.add_argument("--resource-time", dest="resource_time", action="store_true", default=None)
    parser.add_argument("--no-resource-time", dest="resource_time", action="store_false")
    parser.add_argument("--memory-progress-tail-lines", type=int, default=None)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-plots", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    config = build_config_from_args(args)
    existing_plan_path = config.run_root / "run_plan.csv"
    if config.aggregate_only and existing_plan_path.exists() and config.plan_csv is None:
        plan = load_run_plan_csv(existing_plan_path, run_root=config.run_root)
    else:
        plan = build_trial_plan(config)
    commands = {spec.trial_id: build_trial_command(spec, config) for spec in plan}

    config.run_root.mkdir(parents=True, exist_ok=True)
    for spec in plan:
        write_command_file(spec, commands[spec.trial_id])
    write_manifest(config, plan)
    if not (config.aggregate_only and existing_plan_path.exists() and config.plan_csv is None):
        write_run_plan_csv(config.run_root / "run_plan.csv", plan)
    mc_log(
        "plan.ready",
        quiet=config.quiet,
        run_name=config.run_name,
        trials=len(plan),
        max_workers=config.max_workers,
        run_root=config.run_root,
    )

    if config.dry_run:
        write_run_status_csv(config.run_root / "run_status.csv", plan, {})
        mc_log(
            "dry_run.done",
            quiet=config.quiet,
            run_plan=config.run_root / "run_plan.csv",
            commands_dir=config.run_root / "commands",
        )
        return {"run_root": str(config.run_root), "n_trials_planned": len(plan), "dry_run": True}

    if config.aggregate_only:
        mc_log("aggregate_only.start", quiet=config.quiet, status_csv=config.run_root / "run_status.csv")
        results = load_trial_status(config.run_root / "run_status.csv")
    else:
        results = run_trial_pool(plan, config, commands=commands)
        write_run_status_csv(config.run_root / "run_status.csv", plan, results)

    aggregate_summary: dict[str, Any] | None = None
    if config.aggregation_enabled:
        accepted_candidates = sum(
            1
            for spec in plan
            if results.get(spec.trial_id) is not None
            and (results[spec.trial_id].summary_json_path or spec.expected_summary_json).exists()
        )
        mc_log(
            "aggregate.start",
            quiet=config.quiet,
            accepted_candidates=accepted_candidates,
            aggregate_root=config.run_root / "aggregate",
        )
        aggregate_summary = aggregate_schur_summary_trials(
            config=config,
            plan=plan,
            results=results,
        )
        mc_log(
            "aggregate.done",
            quiet=config.quiet,
            accepted=aggregate_summary["n_summaries_accepted"],
            failed=aggregate_summary["n_trials_failed"],
            rejected=aggregate_summary["n_trials_rejected_or_invalid"],
            planned=aggregate_summary["n_planned_not_run"],
            plots_dir=config.run_root / "aggregate" / "plots",
        )
    mc_log("run.done", quiet=config.quiet, run_root=config.run_root, trials=len(plan))
    return {
        "run_root": str(config.run_root),
        "n_trials_planned": len(plan),
        "aggregate_summary": aggregate_summary,
    }


if __name__ == "__main__":
    main()
