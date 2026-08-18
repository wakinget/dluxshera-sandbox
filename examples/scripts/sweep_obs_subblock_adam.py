"""Run a focused Adam sweep for observation sub-block inference.

This script is intentionally narrow: it tunes Adam hyperparameters for the
current three-frame, registration-only observation sub-block toy problem. It
starts from one base inference prescription, writes one patched prescription
per sweep point, runs ``examples/recipes/observation_subblock_inference.py`` for
each point, computes truth-based metrics, and writes experiment-level aggregate
outputs.

Use this when you have already rendered the validated three-frame toy cube and
want a small, repeatable table for choosing Adam settings. The expected active
frame keys are:

- ``source.x_position_as``
- ``source.y_position_as``
- ``source.position_angle_deg``

Shared active keys must be empty. The workflow does not try to be a generic
optimizer benchmark framework, does not add scheduler support, and does not
compare broad optimizer families. It keeps preconditioning out of the sweep
question by changing only the optimizer/objective fields needed for the Adam
grid and the per-run plotting switch used to keep repeated runs lightweight.
"""

import argparse
import csv
import importlib.util
import json
import math
import traceback
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dluxshera.config.io import load_config_file
from dluxshera.utils.obs_subblock_io import now_iso_local_ms, timestamp_tag


GENERATOR_ID = "examples/scripts/sweep_obs_subblock_adam.py"
MANIFEST_SCHEMA_VERSION = "obs_subblock_adam_sweep_manifest.v1"

CURRENT_TOY_FRAME_KEYS = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)

DEFAULT_TRUTH_SCALES = {
    "source.x_position_as": 1.0e-3,
    "source.y_position_as": 1.0e-3,
    "source.position_angle_deg": 1.0e-2,
}

DEFAULT_BASE_LRS = (0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
DEFAULT_B1S = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
DEFAULT_B2S = (0.999,)
DEFAULT_EPS_VALUES = (1.0e-8,)
DEFAULT_TAIL_K = 10
DEFAULT_SETTLING_TOLERANCE_BAND = 0.10
DEFAULT_RINGING_DEADBAND = 0.05

RESULTS_CSV = "results.csv"
RANKED_SUMMARY_CSV = "ranked_summary.csv"
MANIFEST_JSON = "manifest.json"
RECOMMENDATION_JSON = "recommendation.json"
RECOMMENDATION_MD = "recommendation.md"

RESULT_FIELD_ORDER = (
    "rank",
    "run_id",
    "status",
    "completed",
    "truth_metrics_available",
    "error",
    "optimizer.kind",
    "optimizer.base_lr",
    "optimizer.kwargs.b1",
    "optimizer.kwargs.b2",
    "optimizer.kwargs.eps",
    "objective.frame_reduce",
    "objective.subblock_reduce",
    "optimizer.n_iter",
    "frame_count",
    "output_dir",
    "config_path",
    "truth_score_curve_csv",
    "normalized_residual_history_csv",
    "initial_truth_score",
    "final_truth_score",
    "iter_to_90pct_improvement",
    "settling_iter_tol",
    "ringing_index",
    "tail_std_last_k",
    "max_overshoot_ratio",
    "initial_loss",
    "final_loss",
)


@dataclass(frozen=True)
class AdamSweepPoint:
    """One explicit Adam hyperparameter point in the staged sweep grid."""

    base_lr: float
    b1: float
    b2: float
    eps: float

    @property
    def optimizer_kwargs(self) -> dict[str, float]:
        return {"b1": self.b1, "b2": self.b2, "eps": self.eps}

    @property
    def run_id(self) -> str:
        return (
            f"adam_lr{_float_token(self.base_lr)}"
            f"_b1{_float_token(self.b1)}"
            f"_b2{_float_token(self.b2)}"
            f"_eps{_float_token(self.eps)}"
        )


def _float_token(value: float) -> str:
    """Format a float for compact, filesystem-safe run labels."""

    text = f"{float(value):.6g}"
    text = text.replace("+", "")
    text = text.replace("-0", "-")
    return text.replace(".", "p")


def _parse_float_list(raw: str, *, label: str) -> tuple[float, ...]:
    values: list[float] = []
    for item in raw.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError as exc:
            raise ValueError(f"{label} must be a comma-separated list of floats.") from exc
    if not values:
        raise ValueError(f"{label} must contain at least one value.")
    return tuple(values)


def _validate_grid_values(
    *,
    base_lrs: Sequence[float],
    b1s: Sequence[float],
    b2s: Sequence[float],
    eps_values: Sequence[float],
) -> None:
    if any(value <= 0.0 or not math.isfinite(value) for value in base_lrs):
        raise ValueError("All base learning rates must be positive finite values.")
    if any(value < 0.0 or value >= 1.0 or not math.isfinite(value) for value in b1s):
        raise ValueError("All Adam b1 values must satisfy 0 <= b1 < 1.")
    if any(value < 0.0 or value >= 1.0 or not math.isfinite(value) for value in b2s):
        raise ValueError("All Adam b2 values must satisfy 0 <= b2 < 1.")
    if any(value <= 0.0 or not math.isfinite(value) for value in eps_values):
        raise ValueError("All Adam eps values must be positive finite values.")


def build_adam_grid(
    *,
    base_lrs: Sequence[float] = DEFAULT_BASE_LRS,
    b1s: Sequence[float] = DEFAULT_B1S,
    b2s: Sequence[float] = DEFAULT_B2S,
    eps_values: Sequence[float] = DEFAULT_EPS_VALUES,
) -> list[AdamSweepPoint]:
    """Build the small staged Adam grid requested for the current toy problem."""

    _validate_grid_values(
        base_lrs=base_lrs,
        b1s=b1s,
        b2s=b2s,
        eps_values=eps_values,
    )
    points: list[AdamSweepPoint] = []
    for base_lr in base_lrs:
        for b1 in b1s:
            for b2 in b2s:
                for eps in eps_values:
                    points.append(
                        AdamSweepPoint(
                            base_lr=float(base_lr),
                            b1=float(b1),
                            b2=float(b2),
                            eps=float(eps),
                        )
                    )
    return points


def _resolve_relative_config_path(value: str, *, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _resolve_data_paths_for_run_config(
    cfg: dict[str, Any],
    *,
    base_config_path: Path,
) -> dict[str, Any]:
    """Make data paths absolute before writing generated per-run configs.

    The base prescription may use paths relative to itself. Generated configs
    live under the sweep output directory, so relative data paths would otherwise
    be reinterpreted from the wrong directory by the inference recipe.
    """

    patched = deepcopy(cfg)
    data = patched.get("experiment", {}).get("inference", {}).get("data", {})
    if not isinstance(data, dict):
        return patched
    for key in ("cube", "truth_trace", "manifest"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            data[key] = str(
                _resolve_relative_config_path(value, config_path=base_config_path)
            )
    return patched


def validate_current_toy_config(base_cfg: dict[str, Any]) -> tuple[str, ...]:
    """Validate that the base prescription targets the supported toy problem."""

    experiment = base_cfg.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("Base config must contain an experiment mapping.")
    if experiment.get("kind") != "subblock_inference":
        raise ValueError("Base config experiment.kind must be 'subblock_inference'.")

    inference = experiment.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("Base config must contain experiment.inference.")
    active = inference.get("active")
    if not isinstance(active, dict):
        raise ValueError("Base config must contain experiment.inference.active.")

    frame_keys = tuple(active.get("frame_keys") or ())
    shared_keys = tuple(active.get("shared_keys") or ())
    if frame_keys != CURRENT_TOY_FRAME_KEYS:
        raise ValueError(
            "This sweep only supports the current registration-only toy frame keys: "
            f"{list(CURRENT_TOY_FRAME_KEYS)}."
        )
    if shared_keys:
        raise ValueError("This sweep only supports shared_keys: [] for now.")
    return frame_keys


def patch_config_for_adam_point(
    base_cfg: dict[str, Any],
    point: AdamSweepPoint,
    *,
    runs_root: Path,
    per_run_plots: bool,
) -> dict[str, Any]:
    """Return a per-run config with only the targeted sweep overrides applied."""

    cfg = deepcopy(base_cfg)
    experiment = cfg.setdefault("experiment", {})
    inference = experiment.setdefault("inference", {})
    objective = inference.setdefault("objective", {})
    objective["frame_reduce"] = "mean"
    objective["subblock_reduce"] = "sum"
    objective.pop("reduce", None)

    optimizer = inference.setdefault("optimizer", {})
    optimizer["kind"] = "adam"
    optimizer["base_lr"] = float(point.base_lr)
    optimizer["kwargs"] = point.optimizer_kwargs

    diagnostics = inference.setdefault("diagnostics", {})
    diagnostics["plots"] = bool(per_run_plots)

    outputs = experiment.setdefault("outputs", {})
    outputs["outdir"] = str(runs_root)
    return cfg


def final_truth_score(
    residual_matrix: np.ndarray,
    active_keys: Sequence[str],
    *,
    scales: dict[str, float] | None = None,
) -> float:
    """Compute the combined normalized RMS truth residual score.

    Rows are frames and columns are active keys. Each column is divided by a
    fixed key scale before computing one RMS over all entries. The default
    scales encode the first-pass registration tolerances used by this sweep.
    """

    keys = tuple(active_keys)
    residuals = np.asarray(residual_matrix, dtype=float)
    if residuals.ndim != 2:
        raise ValueError("residual_matrix must be 2D with shape (n_frame, n_key).")
    if residuals.shape[1] != len(keys):
        raise ValueError("residual_matrix width must match active_keys length.")
    if residuals.size == 0:
        raise ValueError("residual_matrix must not be empty.")

    scale_map = DEFAULT_TRUTH_SCALES if scales is None else scales
    normalized = np.empty_like(residuals, dtype=float)
    for key_index, key in enumerate(keys):
        scale = float(scale_map[key])
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError(f"Scale for {key!r} must be positive and finite.")
        normalized[:, key_index] = residuals[:, key_index] / scale
    return float(np.sqrt(np.mean(normalized**2)))


def normalize_residual_history(
    residual_history: np.ndarray,
    active_keys: Sequence[str],
    *,
    scales: dict[str, float] | None = None,
) -> np.ndarray:
    """Normalize per-iteration residual history with the truth-score scales.

    The expected shape is ``(iteration, frame, active_key)``. The returned array
    uses the same shape and expresses every component in normalized residual
    units, so a value of ``1.0`` means one configured scale unit for that key.
    """

    keys = tuple(active_keys)
    residuals = np.asarray(residual_history, dtype=float)
    if residuals.ndim != 3:
        raise ValueError(
            "residual_history must have shape (n_iteration, n_frame, n_key)."
        )
    if residuals.shape[2] != len(keys):
        raise ValueError("residual_history key axis must match active_keys length.")
    if residuals.size == 0:
        raise ValueError("residual_history must not be empty.")

    scale_map = DEFAULT_TRUTH_SCALES if scales is None else scales
    normalized = np.empty_like(residuals, dtype=float)
    for key_index, key in enumerate(keys):
        scale = float(scale_map[key])
        if scale <= 0.0 or not math.isfinite(scale):
            raise ValueError(f"Scale for {key!r} must be positive and finite.")
        normalized[:, :, key_index] = residuals[:, :, key_index] / scale
    return normalized


def settling_iter_tol(
    normalized_residual_history: np.ndarray,
    *,
    tolerance_band: float = DEFAULT_SETTLING_TOLERANCE_BAND,
) -> int | None:
    """Return the first iteration that stays inside the tolerance band.

    A run is considered settled at iteration ``t`` when every normalized
    residual component in every later sample ``u >= t`` has absolute value less
    than or equal to ``tolerance_band``. If that never happens, the final
    iteration index is returned as a clear "not settled before the end" sentinel
    that sorts later than earlier settled runs.
    """

    if tolerance_band < 0.0 or not math.isfinite(tolerance_band):
        raise ValueError("tolerance_band must be finite and >= 0.")
    history = np.asarray(normalized_residual_history, dtype=float)
    if history.ndim == 1:
        history = history.reshape((history.shape[0], 1))
    if history.ndim < 2 or history.shape[0] == 0 or not np.all(np.isfinite(history)):
        return None

    flattened = history.reshape((history.shape[0], -1))
    within_band_by_iter = np.all(np.abs(flattened) <= tolerance_band, axis=1)
    # Work backward so suffix_settled[t] means every sample from t through the
    # final iteration is inside the band.
    suffix_settled = np.logical_and.accumulate(within_band_by_iter[::-1])[::-1]
    matches = np.nonzero(suffix_settled)[0]
    if matches.size == 0:
        return int(history.shape[0] - 1)
    return int(matches[0])


def ringing_index(
    normalized_residual_history: np.ndarray,
    *,
    deadband: float = DEFAULT_RINGING_DEADBAND,
) -> float | None:
    """Aggregate weighted sign changes outside a normalized residual deadband.

    For each frame/key component, samples with ``abs(value) <= deadband`` are
    ignored. A sign change between adjacent remaining samples contributes the
    smaller adjacent absolute amplitude. Summing those weights across all
    frame-varying active components penalizes meaningful ring-down while
    avoiding tiny late-stage jitter around zero.
    """

    if deadband < 0.0 or not math.isfinite(deadband):
        raise ValueError("deadband must be finite and >= 0.")
    history = np.asarray(normalized_residual_history, dtype=float)
    if history.ndim == 1:
        history = history.reshape((history.shape[0], 1))
    if history.ndim < 2 or history.shape[0] == 0 or not np.all(np.isfinite(history)):
        return None

    flattened = history.reshape((history.shape[0], -1))
    total = 0.0
    for component in range(flattened.shape[1]):
        values = flattened[:, component]
        significant = values[np.abs(values) > deadband]
        if significant.size < 2:
            continue
        signs = np.sign(significant)
        sign_changes = signs[1:] * signs[:-1] < 0.0
        if not np.any(sign_changes):
            continue
        adjacent_weights = np.minimum(np.abs(significant[1:]), np.abs(significant[:-1]))
        total += float(np.sum(adjacent_weights[sign_changes]))
    return total


def key_residual_metrics(
    residual_matrix: np.ndarray,
    active_keys: Sequence[str],
) -> dict[str, float]:
    """Compute per-key RMS and max absolute residuals across frames."""

    keys = tuple(active_keys)
    residuals = np.asarray(residual_matrix, dtype=float)
    if residuals.ndim != 2 or residuals.shape[1] != len(keys):
        raise ValueError("residual_matrix must have shape (n_frame, len(active_keys)).")

    metrics: dict[str, float] = {}
    for key_index, key in enumerate(keys):
        column = residuals[:, key_index]
        metrics[f"rms_residual.{key}"] = float(np.sqrt(np.mean(column**2)))
        metrics[f"max_abs_residual.{key}"] = float(np.max(np.abs(column)))
    return metrics


def iter_to_90pct_improvement(score_curve: Sequence[float]) -> int | None:
    """Return the first iteration reaching 90 percent of final improvement."""

    scores = np.asarray(score_curve, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        return None
    initial = float(scores[0])
    final = float(scores[-1])
    threshold = initial - 0.9 * (initial - final)
    matches = np.nonzero(scores <= threshold)[0]
    if matches.size == 0:
        return None
    return int(matches[0])


def tail_std_last_k(score_curve: Sequence[float], *, k: int = DEFAULT_TAIL_K) -> float | None:
    """Return score standard deviation over the final ``k`` recorded iterations."""

    if k <= 0:
        raise ValueError("k must be > 0.")
    scores = np.asarray(score_curve, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        return None
    tail = scores[-min(k, scores.size) :]
    return float(np.std(tail))


def max_overshoot_ratio(score_curve: Sequence[float]) -> float | None:
    """Return max score during the run divided by the initial score."""

    scores = np.asarray(score_curve, dtype=float)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        return None
    initial = float(scores[0])
    max_score = float(np.max(scores))
    if initial == 0.0:
        return 1.0 if max_score == 0.0 else math.inf
    return float(max_score / initial)


def load_truth_comparison_matrices(
    path: Path,
    *,
    active_keys: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load truth, recovered, and residual matrices from a run comparison CSV."""

    truth_rows: list[list[float]] = []
    recovered_rows: list[list[float]] = []
    residual_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            truth_row: list[float] = []
            recovered_row: list[float] = []
            residual_row: list[float] = []
            for key in active_keys:
                truth = float(row[f"{key}_truth"])
                recovered = float(row[f"{key}_recovered"])
                residual = row.get(f"{key}_residual")
                truth_row.append(truth)
                recovered_row.append(recovered)
                residual_row.append(
                    recovered - truth if residual in (None, "") else float(residual)
                )
            truth_rows.append(truth_row)
            recovered_rows.append(recovered_row)
            residual_rows.append(residual_row)

    if not truth_rows:
        raise ValueError(f"Truth comparison CSV contains no rows: {path}")
    return (
        np.asarray(truth_rows, dtype=float),
        np.asarray(recovered_rows, dtype=float),
        np.asarray(residual_rows, dtype=float),
    )


def _theta_history_from_trace(
    *,
    theta0: Sequence[float],
    theta_trace: np.ndarray,
) -> np.ndarray:
    """Return theta history including the initial state when needed."""

    theta0_np = np.asarray(theta0, dtype=float).ravel()
    trace_np = np.asarray(theta_trace, dtype=float)
    if trace_np.ndim == 1:
        trace_np = trace_np.reshape((1, trace_np.size))
    if trace_np.ndim != 2:
        raise ValueError("theta_trace must be 2D.")
    if trace_np.shape[1] != theta0_np.size:
        raise ValueError("theta_trace width must match theta0 size.")

    if trace_np.shape[0] == 0:
        theta_history = theta0_np[None, :]
    elif np.allclose(trace_np[0], theta0_np, rtol=0.0, atol=1e-12):
        theta_history = trace_np
    else:
        theta_history = np.vstack((theta0_np[None, :], trace_np))
    return theta_history


def residual_history_from_theta_history(
    *,
    theta0: Sequence[float],
    theta_trace: np.ndarray,
    truth_matrix: np.ndarray,
    active_keys: Sequence[str],
) -> np.ndarray:
    """Decode frame-active theta history into recovered-minus-truth residuals."""

    theta_history = _theta_history_from_trace(theta0=theta0, theta_trace=theta_trace)

    truth = np.asarray(truth_matrix, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != len(active_keys):
        raise ValueError("truth_matrix must have shape (n_frame, len(active_keys)).")
    frame_count = int(truth.shape[0])
    frame_width = len(active_keys)
    frame_theta_width = frame_count * frame_width
    if theta_history.shape[1] < frame_theta_width:
        raise ValueError("theta history is too narrow for frame active keys.")

    # The observation sub-block recipe packs frame state first, frame-major, then
    # shared state. This sweep validates shared_keys == [], but slicing only the
    # frame block keeps the scoring helper aligned with the recipe layout.
    frame_history = theta_history[:, :frame_theta_width].reshape(
        (theta_history.shape[0], frame_count, frame_width)
    )
    return frame_history - truth[None, :, :]


def score_curve_from_theta_history(
    *,
    theta0: Sequence[float],
    theta_trace: np.ndarray,
    truth_matrix: np.ndarray,
    active_keys: Sequence[str],
) -> np.ndarray:
    """Decode frame-active theta history and compute one truth score per step."""

    residual_history = residual_history_from_theta_history(
        theta0=theta0,
        theta_trace=theta_trace,
        truth_matrix=truth_matrix,
        active_keys=active_keys,
    )
    scores = [
        final_truth_score(residual_values, active_keys)
        for residual_values in residual_history
    ]
    return np.asarray(scores, dtype=float)


def write_score_curve_csv(path: Path, score_curve: Sequence[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["iteration", "truth_score"])
        writer.writeheader()
        for iteration, score in enumerate(score_curve):
            writer.writerow({"iteration": int(iteration), "truth_score": float(score)})


def write_normalized_residual_history_csv(
    path: Path,
    normalized_residual_history: np.ndarray,
    *,
    active_keys: Sequence[str],
) -> None:
    """Write a compact per-run artifact used by oscillation-aware metrics."""

    history = np.asarray(normalized_residual_history, dtype=float)
    if history.ndim != 3 or history.shape[2] != len(active_keys):
        raise ValueError(
            "normalized_residual_history must have shape "
            "(n_iteration, n_frame, len(active_keys))."
        )

    fieldnames = ["iteration"]
    for frame_index in range(history.shape[1]):
        for key in active_keys:
            fieldnames.append(f"frame[{frame_index}].{key}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for iteration, frame_values in enumerate(history):
            row: dict[str, float | int] = {"iteration": int(iteration)}
            for frame_index in range(history.shape[1]):
                for key_index, key in enumerate(active_keys):
                    row[f"frame[{frame_index}].{key}"] = float(
                        frame_values[frame_index, key_index]
                    )
            writer.writerow(row)


def _numeric_rank_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return math.inf
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.inf
    return numeric if math.isfinite(numeric) else math.inf


def rank_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank rows using completion as a gate and truth metrics as sort keys."""

    ranked = [dict(row) for row in rows]
    ranked.sort(
        key=lambda row: (
            0 if bool(row.get("completed")) else 1,
            _numeric_rank_value(row, "final_truth_score"),
            _numeric_rank_value(row, "iter_to_90pct_improvement"),
            _numeric_rank_value(row, "settling_iter_tol"),
            _numeric_rank_value(row, "ringing_index"),
            _numeric_rank_value(row, "tail_std_last_k"),
            _numeric_rank_value(row, "max_overshoot_ratio"),
            str(row.get("run_id", "")),
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def _fieldnames_for_rows(rows: Sequence[dict[str, Any]]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for key in RESULT_FIELD_ORDER:
        if any(key in row for row in rows):
            ordered.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    return ordered


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames_for_rows(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _best_completed_row(ranked_rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for row in ranked_rows:
        if bool(row.get("completed")) and math.isfinite(
            _numeric_rank_value(row, "final_truth_score")
        ):
            return row
    return None


def recommendation_from_ranked_rows(
    ranked_rows: Sequence[dict[str, Any]]
) -> dict[str, Any] | None:
    """Build a compact recommendation payload from the top ranked row."""

    best = _best_completed_row(ranked_rows)
    if best is None:
        return None
    optimizer = {
        "kind": "adam",
        "base_lr": float(best["optimizer.base_lr"]),
        "kwargs": {
            "b1": float(best["optimizer.kwargs.b1"]),
            "b2": float(best["optimizer.kwargs.b2"]),
            "eps": float(best["optimizer.kwargs.eps"]),
        },
    }
    return {
        "run_id": best["run_id"],
        "rank": int(best["rank"]),
        "optimizer": optimizer,
        "objective": {
            "frame_reduce": best.get("objective.frame_reduce", "mean"),
            "subblock_reduce": best.get("objective.subblock_reduce", "sum"),
        },
        "metrics": {
            "final_truth_score": _numeric_rank_value(best, "final_truth_score"),
            "iter_to_90pct_improvement": _numeric_rank_value(
                best, "iter_to_90pct_improvement"
            ),
            "settling_iter_tol": _numeric_rank_value(best, "settling_iter_tol"),
            "ringing_index": _numeric_rank_value(best, "ringing_index"),
            "tail_std_last_k": _numeric_rank_value(best, "tail_std_last_k"),
            "max_overshoot_ratio": _numeric_rank_value(best, "max_overshoot_ratio"),
        },
        "summary": (
            "Best tested Adam configuration for the current three-frame "
            "registration-only toy problem."
        ),
    }


def _write_recommendation_files(
    *,
    output_dir: Path,
    recommendation: dict[str, Any] | None,
) -> dict[str, str]:
    json_path = output_dir / RECOMMENDATION_JSON
    md_path = output_dir / RECOMMENDATION_MD
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(recommendation, handle, indent=2)

    if recommendation is None:
        text = "# Adam Sweep Recommendation\n\nNo completed, rankable Adam run was found.\n"
    else:
        opt = recommendation["optimizer"]
        kwargs = opt["kwargs"]
        metrics = recommendation["metrics"]
        text = (
            "# Adam Sweep Recommendation\n\n"
            f"Recommended run: `{recommendation['run_id']}`\n\n"
            "```yaml\n"
            "optimizer:\n"
            "  kind: adam\n"
            f"  base_lr: {opt['base_lr']}\n"
            "  kwargs:\n"
            f"    b1: {kwargs['b1']}\n"
            f"    b2: {kwargs['b2']}\n"
            f"    eps: {kwargs['eps']}\n"
            "objective:\n"
            f"  frame_reduce: {recommendation['objective']['frame_reduce']}\n"
            f"  subblock_reduce: {recommendation['objective']['subblock_reduce']}\n"
            "```\n\n"
            f"Final truth score: {metrics['final_truth_score']:.6g}\n\n"
            f"Iteration to 90 percent improvement: "
            f"{metrics['iter_to_90pct_improvement']:.6g}\n\n"
            f"Settling iteration at tolerance: {metrics['settling_iter_tol']:.6g}\n\n"
            f"Ringing index: {metrics['ringing_index']:.6g}\n\n"
            f"Tail standard deviation: {metrics['tail_std_last_k']:.6g}\n\n"
            f"Max overshoot ratio: {metrics['max_overshoot_ratio']:.6g}\n"
        )
    md_path.write_text(text, encoding="utf-8")
    return {
        "recommendation_json": RECOMMENDATION_JSON,
        "recommendation_md": RECOMMENDATION_MD,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _plot_metric_vs_lr(
    *,
    rows: Sequence[dict[str, Any]],
    metric_name: str,
    ylabel: str,
    output_path: Path,
    log_y: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        value = _numeric_rank_value(row, metric_name)
        lr = _numeric_rank_value(row, "optimizer.base_lr")
        if not (math.isfinite(value) and math.isfinite(lr)):
            continue
        key = (
            float(row["optimizer.kwargs.b1"]),
            float(row["optimizer.kwargs.b2"]),
        )
        groups.setdefault(key, []).append(row)

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (b1, b2), group_rows in sorted(groups.items()):
        group_rows = sorted(group_rows, key=lambda row: float(row["optimizer.base_lr"]))
        x = [float(row["optimizer.base_lr"]) for row in group_rows]
        y = [_numeric_rank_value(row, metric_name) for row in group_rows]
        ax.plot(x, y, marker="o", linewidth=1.2, label=f"b1={b1:g}, b2={b2:g}")
    ax.set_xscale("log")
    if log_y and all(_numeric_rank_value(row, metric_name) > 0 for row in rows):
        ax.set_yscale("log")
    ax.set_xlabel("optimizer.base_lr")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_plots(
    *,
    output_dir: Path,
    ranked_rows: Sequence[dict[str, Any]],
) -> dict[str, str]:
    completed_rows = [row for row in ranked_rows if bool(row.get("completed"))]
    plots: dict[str, str] = {}
    final_path = output_dir / "final_truth_score_vs_base_lr.png"
    quickness_path = output_dir / "iter_to_90pct_improvement_vs_base_lr.png"
    _plot_metric_vs_lr(
        rows=completed_rows,
        metric_name="final_truth_score",
        ylabel="final_truth_score",
        output_path=final_path,
        log_y=True,
    )
    if final_path.exists():
        plots["final_truth_score_vs_base_lr_png"] = final_path.name
    _plot_metric_vs_lr(
        rows=completed_rows,
        metric_name="iter_to_90pct_improvement",
        ylabel="iter_to_90pct_improvement",
        output_path=quickness_path,
        log_y=False,
    )
    if quickness_path.exists():
        plots["iter_to_90pct_improvement_vs_base_lr_png"] = quickness_path.name
    return plots


def write_aggregate_outputs(
    *,
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    base_config_path: Path,
    grid: Sequence[AdamSweepPoint],
    started_at: str,
    completed_at: str,
    write_plots: bool = True,
) -> dict[str, Path]:
    """Write results.csv, ranked_summary.csv, manifest.json, and recommendation."""

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_CSV
    ranked_path = output_dir / RANKED_SUMMARY_CSV
    manifest_path = output_dir / MANIFEST_JSON

    run_order_rows = [dict(row) for row in rows]
    ranked_rows = rank_rows(rows)
    write_rows_csv(results_path, run_order_rows)
    write_rows_csv(ranked_path, ranked_rows)

    recommendation = recommendation_from_ranked_rows(ranked_rows)
    recommendation_outputs = _write_recommendation_files(
        output_dir=output_dir,
        recommendation=recommendation,
    )
    plot_outputs = write_summary_plots(output_dir=output_dir, ranked_rows=ranked_rows) if write_plots else {}

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": completed_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "generator": GENERATOR_ID,
        "base_config_path": str(base_config_path),
        "target_problem": {
            "description": "Current 3-frame registration-only observation sub-block toy problem.",
            "frame_keys": list(CURRENT_TOY_FRAME_KEYS),
            "shared_keys": [],
        },
        "grid": {
            "optimizer.kind": "adam",
            "objective.frame_reduce": "mean",
            "objective.subblock_reduce": "sum",
            "base_lrs": sorted({point.base_lr for point in grid}),
            "b1s": sorted({point.b1 for point in grid}),
            "b2s": sorted({point.b2 for point in grid}),
            "eps_values": sorted({point.eps for point in grid}),
            "run_count": len(grid),
        },
        "metric_definitions": {
            "final_truth_score": (
                "RMS over recovered-minus-truth residuals after normalizing each "
                "active key by fixed scale."
            ),
            "truth_scales": DEFAULT_TRUTH_SCALES,
            "iter_to_90pct_improvement": (
                "First iteration t where S_t <= S_0 - 0.9 * (S_0 - S_T)."
            ),
            "settling_iter_tol": (
                "First iteration index where every normalized residual component "
                f"stays within +/-{DEFAULT_SETTLING_TOLERANCE_BAND:g} through "
                "the final recorded iteration; final iteration index means the "
                "run did not settle earlier."
            ),
            "ringing_index": (
                "Sum over frame/key components of weighted sign changes outside "
                f"a +/-{DEFAULT_RINGING_DEADBAND:g} normalized deadband; each "
                "sign change is weighted by the smaller adjacent amplitude."
            ),
            "tail_std_last_k": f"Standard deviation over the final {DEFAULT_TAIL_K} score samples.",
            "max_overshoot_ratio": "Max score during the run divided by initial score.",
        },
        "ranking_policy": [
            "successful completion first",
            "lowest final_truth_score",
            "lowest iter_to_90pct_improvement",
            "lowest settling_iter_tol",
            "lowest ringing_index",
            "lowest tail_std_last_k",
            "lowest max_overshoot_ratio",
        ],
        "outputs": {
            "results_csv": RESULTS_CSV,
            "ranked_summary_csv": RANKED_SUMMARY_CSV,
            **recommendation_outputs,
            **plot_outputs,
        },
        "recommendation": recommendation,
        "runs": ranked_rows,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(manifest), handle, indent=2)

    return {
        "results_csv": results_path,
        "ranked_summary_csv": ranked_path,
        "manifest_json": manifest_path,
        "recommendation_json": output_dir / RECOMMENDATION_JSON,
        "recommendation_md": output_dir / RECOMMENDATION_MD,
        **{key: output_dir / value for key, value in plot_outputs.items()},
    }


def _load_inference_recipe():
    recipe_path = Path(__file__).resolve().parents[1] / "recipes" / "observation_subblock_inference.py"
    spec = importlib.util.spec_from_file_location("observation_subblock_inference_for_adam_sweep", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load inference recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_run_config(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(cfg), handle, indent=2)


def _base_result_row(
    *,
    point: AdamSweepPoint,
    run_id: str,
    run_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "planned",
        "completed": False,
        "truth_metrics_available": False,
        "error": None,
        "optimizer.kind": "adam",
        "optimizer.base_lr": point.base_lr,
        "optimizer.kwargs.b1": point.b1,
        "optimizer.kwargs.b2": point.b2,
        "optimizer.kwargs.eps": point.eps,
        "objective.frame_reduce": "mean",
        "objective.subblock_reduce": "sum",
        "output_dir": str(run_dir),
        "config_path": str(config_path),
    }


def _update_row_with_truth_metrics(
    *,
    row: dict[str, Any],
    result: dict[str, Any],
    active_keys: Sequence[str],
    run_dir: Path,
) -> None:
    artifacts = {name: Path(path) for name, path in result.get("artifacts", {}).items()}
    comparison_path = artifacts.get("truth_comparison_csv")
    if comparison_path is None or not comparison_path.exists():
        row["truth_metrics_available"] = False
        row["status"] = "completed_without_truth_metrics"
        return

    truth_matrix, _, residual_matrix = load_truth_comparison_matrices(
        comparison_path,
        active_keys=active_keys,
    )
    residual_history = residual_history_from_theta_history(
        theta0=result["theta0"],
        theta_trace=np.asarray(result["trace_history"]["theta"], dtype=float),
        truth_matrix=truth_matrix,
        active_keys=active_keys,
    )
    normalized_history = normalize_residual_history(
        residual_history,
        active_keys,
    )
    score_curve = score_curve_from_theta_history(
        theta0=result["theta0"],
        theta_trace=np.asarray(result["trace_history"]["theta"], dtype=float),
        truth_matrix=truth_matrix,
        active_keys=active_keys,
    )
    curve_path = run_dir / "truth_score_curve.csv"
    residual_history_path = run_dir / "normalized_residual_history.csv"
    write_score_curve_csv(curve_path, score_curve)
    write_normalized_residual_history_csv(
        residual_history_path,
        normalized_history,
        active_keys=active_keys,
    )

    row.update(key_residual_metrics(residual_matrix, active_keys))
    row["truth_metrics_available"] = True
    row["truth_score_curve_csv"] = str(curve_path)
    row["normalized_residual_history_csv"] = str(residual_history_path)
    row["initial_truth_score"] = float(score_curve[0])
    row["final_truth_score"] = final_truth_score(residual_matrix, active_keys)
    row["iter_to_90pct_improvement"] = iter_to_90pct_improvement(score_curve)
    row["settling_iter_tol"] = settling_iter_tol(
        normalized_history,
        tolerance_band=DEFAULT_SETTLING_TOLERANCE_BAND,
    )
    row["ringing_index"] = ringing_index(
        normalized_history,
        deadband=DEFAULT_RINGING_DEADBAND,
    )
    row["tail_std_last_k"] = tail_std_last_k(score_curve, k=DEFAULT_TAIL_K)
    row["max_overshoot_ratio"] = max_overshoot_ratio(score_curve)
    row["status"] = "ok"


def run_sweep(
    *,
    base_config_path: Path,
    results_dir: Path,
    experiment_name: str | None,
    grid: Sequence[AdamSweepPoint],
    no_progress: bool,
    per_run_plots: bool,
    summary_plots: bool,
    dry_run: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    """Run the staged Adam sweep and return aggregate artifact paths."""

    base_config_path = base_config_path.resolve()
    base_cfg = load_config_file(base_config_path)
    active_keys = validate_current_toy_config(base_cfg)

    run_grid = list(grid)
    if not run_grid:
        raise ValueError("Sweep grid is empty.")

    experiment_dir = results_dir / (experiment_name or f"adam_sweep_{timestamp_tag()}")
    runs_root = experiment_dir / "runs"

    if dry_run:
        return {
            "dry_run": True,
            "base_config_path": str(base_config_path),
            "output_dir": str(experiment_dir),
            "run_count": len(run_grid),
            "run_ids": [point.run_id for point in run_grid],
        }

    started_at = now_iso_local_ms()
    inference_recipe = _load_inference_recipe()
    rows: list[dict[str, Any]] = []

    for index, point in enumerate(run_grid, start=1):
        run_id = point.run_id
        run_dir = runs_root / run_id
        run_config_path = run_dir / "sweep_run_config.json"
        patched_cfg = patch_config_for_adam_point(
            base_cfg,
            point,
            runs_root=runs_root,
            per_run_plots=per_run_plots,
        )
        patched_cfg = _resolve_data_paths_for_run_config(
            patched_cfg,
            base_config_path=base_config_path,
        )
        _write_run_config(run_config_path, patched_cfg)

        row = _base_result_row(
            point=point,
            run_id=run_id,
            run_dir=run_dir,
            config_path=run_config_path,
        )
        print(f"[{index}/{len(run_grid)}] Running {run_id}")
        try:
            recipe_args = [
                "--config",
                str(run_config_path),
                "--results-dir",
                str(runs_root),
                "--run-name",
                run_id,
            ]
            if no_progress:
                recipe_args.append("--no-progress")
            result = inference_recipe.main(recipe_args)
            row["completed"] = True
            row["frame_count"] = int(result.get("frame_count", 0))
            row["initial_loss"] = float(result.get("initial_loss", math.nan))
            row["final_loss"] = float(result.get("final_loss", math.nan))
            optimizer_cfg = (
                patched_cfg.get("experiment", {})
                .get("inference", {})
                .get("optimizer", {})
            )
            row["optimizer.n_iter"] = optimizer_cfg.get("n_iter")
            _update_row_with_truth_metrics(
                row=row,
                result=result,
                active_keys=active_keys,
                run_dir=run_dir,
            )
        except Exception as exc:
            row["status"] = "failed"
            row["completed"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            if fail_fast:
                raise
        rows.append(row)

    completed_at = now_iso_local_ms()
    outputs = write_aggregate_outputs(
        output_dir=experiment_dir,
        rows=rows,
        base_config_path=base_config_path,
        grid=run_grid,
        started_at=started_at,
        completed_at=completed_at,
        write_plots=summary_plots,
    )
    return {
        "dry_run": False,
        "output_dir": str(experiment_dir),
        "run_count": len(run_grid),
        "outputs": {name: str(path) for name, path in outputs.items()},
        "recommendation": recommendation_from_ranked_rows(rank_rows(rows)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused Adam hyperparameter sweep for the current "
            "observation sub-block inference toy problem."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "examples/recipes/observation_subblock_inference_template/subblock_inference_prescription.yaml"
        ),
        help="Base observation sub-block inference prescription.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("Results/obs_subblock_adam_sweeps"),
        help="Root directory for sweep experiment outputs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment directory name under --results-dir.",
    )
    parser.add_argument(
        "--base-lrs",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_BASE_LRS),
        help="Comma-separated Adam base_lr values.",
    )
    parser.add_argument(
        "--b1s",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_B1S),
        help="Comma-separated Adam b1 values.",
    )
    parser.add_argument(
        "--b2s",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_B2S),
        help="Comma-separated Adam b2 values.",
    )
    parser.add_argument(
        "--eps-values",
        type=str,
        default=",".join(f"{value:g}" for value in DEFAULT_EPS_VALUES),
        help="Comma-separated Adam eps values.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on planned grid points for quick smoke runs.",
    )
    parser.add_argument(
        "--per-run-plots",
        action="store_true",
        help="Keep inference recipe plots enabled for each sweep run.",
    )
    parser.add_argument(
        "--no-summary-plots",
        action="store_true",
        help="Skip experiment-level summary PNGs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the base config/grid and print planned run IDs without running inference.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable per-run optimizer progress bars.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed sweep point instead of recording the failure row.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    grid = build_adam_grid(
        base_lrs=_parse_float_list(args.base_lrs, label="--base-lrs"),
        b1s=_parse_float_list(args.b1s, label="--b1s"),
        b2s=_parse_float_list(args.b2s, label="--b2s"),
        eps_values=_parse_float_list(args.eps_values, label="--eps-values"),
    )
    if args.max_runs is not None:
        if args.max_runs <= 0:
            raise ValueError("--max-runs must be > 0 when provided.")
        grid = grid[: args.max_runs]

    result = run_sweep(
        base_config_path=args.config,
        results_dir=args.results_dir,
        experiment_name=args.experiment_name,
        grid=grid,
        no_progress=bool(args.no_progress),
        per_run_plots=bool(args.per_run_plots),
        summary_plots=not bool(args.no_summary_plots),
        dry_run=bool(args.dry_run),
        fail_fast=bool(args.fail_fast),
    )
    if result["dry_run"]:
        print(f"Dry run: planned {result['run_count']} Adam runs.")
        print(f"Output directory: {result['output_dir']}")
        for run_id in result["run_ids"]:
            print(f"  {run_id}")
    else:
        print(f"Wrote Adam sweep outputs under: {result['output_dir']}")
        recommendation = result.get("recommendation")
        if recommendation is not None:
            print("Recommended Adam configuration:")
            print(json.dumps(recommendation["optimizer"], indent=2))
    return result


if __name__ == "__main__":
    main()
