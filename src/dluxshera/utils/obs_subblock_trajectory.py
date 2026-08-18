"""Trajectory preparation helpers for observation sub-block campaigns.

This module converts raw pointing samples into the canonical per-frame trace
CSV consumed by the observation sub-block renderer, plus a separate per-frame
starting-guess prediction table for recovered-reference inference.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .trajectory_filters import (
    TrajectoryFilterSpec,
    apply_trajectory_filter,
    parse_trajectory_filter_config,
)


AIRBUS_DEFAULT_MAPPING: dict[str, dict[str, Any]] = {
    "source.x_position_as": {"source_column": "x_as", "scale": 1.0, "unit": "arcsec"},
    "source.y_position_as": {"source_column": "y_as", "scale": 1.0, "unit": "arcsec"},
    "source.position_angle_deg": {
        "source_column": "z_as",
        "scale": 1.0 / 3600.0,
        "unit": "deg",
    },
}
DEFAULT_OUTPUT_KEYS: tuple[str, ...] = tuple(AIRBUS_DEFAULT_MAPPING)
TRAJECTORY_NOTES: tuple[str, ...] = (
    "Dynamic cropping is not implemented.",
    "psf_npixels / ROI-origin realism is not tested by this path.",
    "High-order WFE map insertion is not implemented.",
    "Trajectory affects frame-level source registration truth and starting guesses only.",
)


@dataclass(frozen=True)
class RawTrajectory:
    """Canonical raw trajectory samples before parameter mapping."""

    time_s: np.ndarray
    columns: Mapping[str, np.ndarray]
    source_path: Path
    source_kind: str

    @property
    def sample_count(self) -> int:
        return int(self.time_s.size)

    @property
    def span(self) -> tuple[float, float]:
        return (float(self.time_s[0]), float(self.time_s[-1]))


@dataclass(frozen=True)
class CanonicalTrajectory:
    """Canonical parameter-valued trajectory samples."""

    time_s: np.ndarray
    values: Mapping[str, np.ndarray]
    raw: RawTrajectory
    mapping: Mapping[str, Mapping[str, Any]]
    filter_provenance: Mapping[str, Any] | None = None
    offset_provenance: Mapping[str, Any] | None = None
    unfiltered_values: Mapping[str, np.ndarray] | None = None


@dataclass(frozen=True)
class SubblockTrajectory:
    """Prepared frame truth and fit diagnostics for one subblock."""

    subblock_index: int
    frame_times_s: np.ndarray
    time_relative_s: np.ndarray
    truth: Mapping[str, np.ndarray]
    prediction: Mapping[str, np.ndarray]
    residual: Mapping[str, np.ndarray]
    fit_coefficients: Mapping[str, tuple[float, float]]
    diagnostics: Mapping[str, Mapping[str, float]]

    @property
    def time_start_s(self) -> float:
        return float(self.frame_times_s[0])

    @property
    def time_end_s(self) -> float:
        return float(self.frame_times_s[-1])

    @property
    def n_frames(self) -> int:
        return int(self.frame_times_s.size)


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _read_airbus_rows(path: Path) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Read Airbus CSV rows with or without a header.

    The repository example is a headerless four-column file: time, X, Y, Z.
    Tests and future exports may use named columns.
    """

    if not path.exists():
        raise FileNotFoundError(f"Airbus trajectory CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.readline()
        if not sample:
            raise ValueError(f"Airbus trajectory CSV is empty: {path}")
        handle.seek(0)
        first_cells = [cell.strip() for cell in sample.strip().split(",")]
        has_header = any(cell.lower() in {"time", "time_s", "x", "y", "z"} for cell in first_cells)

        times: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        if has_header:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"Airbus trajectory CSV has no header: {path}")
            names = {name.strip().lower(): name for name in reader.fieldnames if name is not None}
            x_name = names.get("x")
            y_name = names.get("y")
            z_name = names.get("z")
            time_name = names.get("time_s") or names.get("time")
            if x_name is None or y_name is None or z_name is None:
                raise ValueError("Airbus CSV header must include X, Y, and Z columns.")
            for row_index, row in enumerate(reader, start=2):
                if time_name is not None and row.get(time_name, "").strip() != "":
                    times.append(_finite_float(row[time_name], name=f"row {row_index} time"))
                xs.append(_finite_float(row[x_name], name=f"row {row_index} X"))
                ys.append(_finite_float(row[y_name], name=f"row {row_index} Y"))
                zs.append(_finite_float(row[z_name], name=f"row {row_index} Z"))
        else:
            reader = csv.reader(handle)
            for row_index, row in enumerate(reader, start=1):
                if not row:
                    continue
                if len(row) < 4:
                    raise ValueError(
                        "Headerless Airbus CSV rows must contain time, X, Y, Z "
                        f"columns; row {row_index} has {len(row)}."
                    )
                times.append(_finite_float(row[0], name=f"row {row_index} time"))
                xs.append(_finite_float(row[1], name=f"row {row_index} X"))
                ys.append(_finite_float(row[2], name=f"row {row_index} Y"))
                zs.append(_finite_float(row[3], name=f"row {row_index} Z"))

    if not xs:
        raise ValueError(f"Airbus trajectory CSV has no data rows: {path}")
    time_array = np.asarray(times, dtype=float) if times else None
    return (
        time_array,
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(zs, dtype=float),
    )


def load_airbus_csv(
    path: Path,
    *,
    sample_dt_s: float = 0.1,
    time_mode: str = "inferred_uniform",
    start_s: float = 0.0,
) -> RawTrajectory:
    """Load the Airbus pointing CSV into canonical raw X/Y/Z arcsec columns."""

    sample_dt = _finite_float(sample_dt_s, name="sample_dt_s")
    if sample_dt <= 0.0:
        raise ValueError("sample_dt_s must be > 0.")
    raw_time, x_as, y_as, z_as = _read_airbus_rows(path)
    if time_mode == "inferred_uniform":
        time_s = float(start_s) + np.arange(x_as.size, dtype=float) * sample_dt
    elif time_mode == "csv":
        if raw_time is None or raw_time.size != x_as.size:
            raise ValueError("time.mode='csv' requires one time value per data row.")
        time_s = raw_time.astype(float)
    else:
        raise ValueError("Airbus time_mode must be 'inferred_uniform' or 'csv'.")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("Airbus trajectory time samples must be strictly increasing.")
    return RawTrajectory(
        time_s=time_s,
        columns={"x_as": x_as, "y_as": y_as, "z_as": z_as},
        source_path=path.resolve(),
        source_kind="airbus_csv",
    )


def map_trajectory(
    raw: RawTrajectory,
    *,
    mapping: Mapping[str, Mapping[str, Any]] | None = None,
    output_keys: Sequence[str] | None = None,
) -> CanonicalTrajectory:
    """Map raw source columns into canonical dLuxShera parameter keys."""

    mapping = dict(AIRBUS_DEFAULT_MAPPING if mapping is None else mapping)
    keys = tuple(DEFAULT_OUTPUT_KEYS if output_keys is None else output_keys)
    values: dict[str, np.ndarray] = {}
    for key in keys:
        if key not in mapping:
            raise ValueError(f"No trajectory mapping configured for output key {key!r}.")
        item = mapping[key]
        source_column = str(item.get("source_column", "")).strip()
        if source_column not in raw.columns:
            raise ValueError(
                f"Trajectory source column {source_column!r} for {key!r} is unavailable."
            )
        scale = _finite_float(item.get("scale", 1.0), name=f"mapping.{key}.scale")
        series = np.asarray(raw.columns[source_column], dtype=float) * scale
        if not np.all(np.isfinite(series)):
            raise ValueError(f"Mapped trajectory values for {key!r} are not finite.")
        values[key] = series
    return CanonicalTrajectory(time_s=raw.time_s, values=values, raw=raw, mapping=mapping)


def apply_filter_to_trajectory(
    trajectory: CanonicalTrajectory,
    *,
    config: Mapping[str, Any] | None,
) -> CanonicalTrajectory:
    """Return a trajectory with configured columns filtered at source cadence."""

    spec = parse_trajectory_filter_config(config)
    columns = tuple(key for key in spec.columns if key in trajectory.values)
    if not columns:
        if spec.enabled and spec.kind != "none":
            raise ValueError(
                "trajectory filter columns do not match mapped trajectory keys: "
                + ", ".join(spec.columns)
            )
        columns = tuple(trajectory.values)
    values = np.column_stack([np.asarray(trajectory.values[key], dtype=float) for key in columns])
    filtered, provenance = apply_trajectory_filter(
        np.asarray(trajectory.time_s, dtype=float),
        values,
        TrajectoryFilterSpec(**{**spec.__dict__, "columns": columns}),
        axis=0,
    )
    output = {key: np.asarray(value, dtype=float).copy() for key, value in trajectory.values.items()}
    if spec.enabled and spec.kind != "none":
        for index, key in enumerate(columns):
            output[key] = np.asarray(filtered[:, index], dtype=float)
    return CanonicalTrajectory(
        time_s=trajectory.time_s,
        values=output,
        raw=trajectory.raw,
        mapping=trajectory.mapping,
        filter_provenance=provenance,
        offset_provenance=trajectory.offset_provenance,
        unfiltered_values=trajectory.values,
    )


def _series_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "peak_to_peak": float(np.max(arr) - np.min(arr)),
    }


def parse_trajectory_offsets_config(
    config: Mapping[str, Any] | None,
    *,
    available_keys: Sequence[str] | None = None,
) -> dict[str, float]:
    """Parse constant trajectory offsets keyed by canonical frame parameter."""

    cfg = dict(config or {})
    if not cfg:
        return {}
    keys = set(DEFAULT_OUTPUT_KEYS if available_keys is None else available_keys)
    bad = sorted(set(str(key) for key in cfg) - keys)
    if bad:
        raise ValueError(
            "trajectory offsets contain unsupported keys: " + ", ".join(bad)
        )
    out: dict[str, float] = {}
    for key, value in cfg.items():
        out[str(key)] = _finite_float(value, name=f"trajectory offsets {key}")
    return out


def apply_offsets_to_trajectory(
    trajectory: CanonicalTrajectory,
    *,
    offsets: Mapping[str, Any] | None,
    stage: str,
) -> CanonicalTrajectory:
    """Return a trajectory with constant offsets added to mapped parameters."""

    parsed = parse_trajectory_offsets_config(offsets, available_keys=trajectory.values)
    values = {
        key: np.asarray(series, dtype=float).copy()
        for key, series in trajectory.values.items()
    }
    shifted, provenance = apply_offsets_to_values(values, offsets=parsed, stage=stage)
    return CanonicalTrajectory(
        time_s=trajectory.time_s,
        values=shifted,
        raw=trajectory.raw,
        mapping=trajectory.mapping,
        filter_provenance=trajectory.filter_provenance,
        offset_provenance=provenance,
        unfiltered_values=trajectory.unfiltered_values,
    )


def apply_offsets_to_values(
    values: Mapping[str, Sequence[float]],
    *,
    offsets: Mapping[str, Any] | None,
    stage: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return value columns shifted by constant canonical-parameter offsets."""

    parsed = parse_trajectory_offsets_config(offsets, available_keys=values)
    shifted = {
        key: np.asarray(series, dtype=float).copy()
        for key, series in values.items()
    }
    provenance: dict[str, Any] = {
        "schema_version": "trajectory_offsets_provenance.v1",
        "enabled": bool(parsed),
        "stage": str(stage),
        "requested_offsets": dict(parsed),
        "columns": list(parsed),
        "statistics": {},
        "std_unchanged_tolerance": 1.0e-12,
    }
    for key, offset in parsed.items():
        before = _series_stats(shifted[key])
        series = shifted[key] + float(offset)
        after = _series_stats(series)
        shifted[key] = series
        provenance["statistics"][key] = {
            "requested_offset": float(offset),
            "pre_offset": before,
            "post_offset": after,
            "mean_delta": float(after["mean"] - before["mean"]),
            "std_delta": float(after["std"] - before["std"]),
            "peak_to_peak_delta": float(after["peak_to_peak"] - before["peak_to_peak"]),
            "residual_std_unchanged": bool(
                np.isclose(after["std"], before["std"], atol=1.0e-12, rtol=1.0e-12)
            ),
        }
    return shifted, provenance


def derive_window_duration(
    *,
    duration_s: float | None,
    n_subblocks: int | None,
    subblock_duration_s: float,
) -> float:
    """Resolve and validate window duration from duration or subblock count."""

    if duration_s is None and n_subblocks is None:
        raise ValueError("Either duration_s or n_subblocks must be provided.")
    if n_subblocks is not None:
        if int(n_subblocks) < 1:
            raise ValueError("n_subblocks must be >= 1.")
        implied = int(n_subblocks) * float(subblock_duration_s)
        if duration_s is not None and not np.isclose(float(duration_s), implied):
            raise ValueError(
                "duration_s and n_subblocks are inconsistent with subblock_duration_s."
            )
        return float(implied)
    assert duration_s is not None
    duration = _finite_float(duration_s, name="duration_s")
    if duration <= 0.0:
        raise ValueError("duration_s must be > 0.")
    return duration


def build_frame_times(
    *,
    start_s: float,
    duration_s: float,
    frame_dt_s: float = 0.05,
    n_frames_per_subblock: int = 20,
    subblock_duration_s: float = 1.0,
) -> np.ndarray:
    """Return non-overlapping subblock frame times for a selected window."""

    frame_dt = _finite_float(frame_dt_s, name="frame_dt_s")
    if frame_dt <= 0.0:
        raise ValueError("frame_dt_s must be > 0.")
    n_frames = int(n_frames_per_subblock)
    if n_frames < 1:
        raise ValueError("n_frames_per_subblock must be >= 1.")
    subblock_duration = _finite_float(subblock_duration_s, name="subblock_duration_s")
    if subblock_duration <= 0.0:
        raise ValueError("subblock_duration_s must be > 0.")
    duration = _finite_float(duration_s, name="duration_s")
    n_subblocks_float = duration / subblock_duration
    n_subblocks = int(round(n_subblocks_float))
    if not np.isclose(n_subblocks_float, n_subblocks):
        raise ValueError("duration_s must be an integer multiple of subblock_duration_s.")
    frame_offsets = np.arange(n_frames, dtype=float) * frame_dt
    if frame_offsets[-1] >= subblock_duration + 1.0e-12:
        raise ValueError("n_frames_per_subblock * frame_dt_s must fit within subblock_duration_s.")
    times = [
        float(start_s) + block * subblock_duration + offset
        for block in range(n_subblocks)
        for offset in frame_offsets
    ]
    return np.asarray(times, dtype=float)


def interpolate_trajectory(
    trajectory: CanonicalTrajectory,
    *,
    frame_times_s: Sequence[float],
    method: str = "linear",
) -> dict[str, np.ndarray]:
    """Interpolate canonical trajectory values onto frame times."""

    if method != "linear":
        raise ValueError("Only linear trajectory interpolation is currently supported.")
    frame_times = np.asarray(frame_times_s, dtype=float)
    if frame_times.size == 0:
        raise ValueError("At least one frame time is required.")
    raw_start = float(trajectory.time_s[0])
    raw_stop = float(trajectory.time_s[-1])
    if float(frame_times[0]) < raw_start - 1.0e-12 or float(frame_times[-1]) > raw_stop + 1.0e-12:
        raise ValueError(
            "Selected frame times exceed raw trajectory domain: "
            f"frames=[{frame_times[0]}, {frame_times[-1]}], raw=[{raw_start}, {raw_stop}]."
        )
    return {
        key: np.interp(frame_times, trajectory.time_s, np.asarray(values, dtype=float))
        for key, values in trajectory.values.items()
    }


def split_subblocks(
    *,
    frame_times_s: Sequence[float],
    truth_values: Mapping[str, Sequence[float]],
    n_frames_per_subblock: int = 20,
    fit_keys: Sequence[str] | None = None,
) -> list[SubblockTrajectory]:
    """Split interpolated frame truth into subblocks and linear predictions."""

    frame_times = np.asarray(frame_times_s, dtype=float)
    n_frames = int(n_frames_per_subblock)
    if frame_times.size % n_frames != 0:
        raise ValueError("Total frame count must be a multiple of n_frames_per_subblock.")
    keys = tuple(truth_values)
    fit_key_set = set(keys if fit_keys is None else fit_keys)
    blocks: list[SubblockTrajectory] = []
    for block_index, start in enumerate(range(0, frame_times.size, n_frames)):
        stop = start + n_frames
        times = frame_times[start:stop]
        relative = times - times[0]
        truth = {key: np.asarray(truth_values[key], dtype=float)[start:stop] for key in keys}
        prediction: dict[str, np.ndarray] = {}
        residual: dict[str, np.ndarray] = {}
        coefficients: dict[str, tuple[float, float]] = {}
        diagnostics: dict[str, dict[str, float]] = {}
        for key in keys:
            series = truth[key]
            if key in fit_key_set and series.size >= 2:
                slope, intercept = np.polyfit(relative, series, deg=1)
                pred = intercept + slope * relative
            else:
                intercept = float(series[0])
                slope = 0.0
                pred = np.full_like(series, intercept, dtype=float)
            res = series - pred
            prediction[key] = pred
            residual[key] = res
            coefficients[key] = (float(intercept), float(slope))
            diagnostics[key] = {
                "rms_residual": float(np.sqrt(np.mean(np.square(res)))),
                "max_abs_residual": float(np.max(np.abs(res))),
            }
        blocks.append(
            SubblockTrajectory(
                subblock_index=block_index,
                frame_times_s=times,
                time_relative_s=relative,
                truth=truth,
                prediction=prediction,
                residual=residual,
                fit_coefficients=coefficients,
                diagnostics=diagnostics,
            )
        )
    return blocks


def frame_truth_rows(block: SubblockTrajectory, output_keys: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index, time_s in enumerate(block.frame_times_s):
        row: dict[str, Any] = {"frame_index": frame_index, "time_s": float(time_s)}
        for key in output_keys:
            row[key] = float(block.truth[key][frame_index])
        rows.append(row)
    return rows


def starting_guess_rows(block: SubblockTrajectory, output_keys: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame_index, time_s in enumerate(block.frame_times_s):
        row: dict[str, Any] = {
            "subblock_index": int(block.subblock_index),
            "frame_index": int(frame_index),
            "time_s": float(time_s),
            "time_relative_s": float(block.time_relative_s[frame_index]),
        }
        for key in output_keys:
            intercept, slope = block.fit_coefficients[key]
            diag = block.diagnostics[key]
            row[f"{key}_truth"] = float(block.truth[key][frame_index])
            row[f"{key}_linear_fit"] = float(block.prediction[key][frame_index])
            row[f"{key}_residual"] = float(block.residual[key][frame_index])
            row[f"{key}_fit_intercept"] = intercept
            row[f"{key}_fit_slope_per_s"] = slope
            row[f"rms_{key}_residual"] = diag["rms_residual"]
            row[f"max_abs_{key}_residual"] = diag["max_abs_residual"]
        rows.append(row)
    return rows


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def trajectory_rows(
    trajectory: CanonicalTrajectory,
    *,
    values: Mapping[str, Sequence[float]] | None = None,
) -> list[dict[str, Any]]:
    selected = trajectory.values if values is None else values
    rows: list[dict[str, Any]] = []
    for index, time_s in enumerate(trajectory.time_s):
        row: dict[str, Any] = {"sample_index": int(index), "time_s": float(time_s)}
        for key, series in selected.items():
            row[key] = float(np.asarray(series, dtype=float)[index])
        rows.append(row)
    return rows


def write_trajectory_filter_artifacts(
    *,
    outdir: Path,
    trajectory: CanonicalTrajectory,
) -> dict[str, Path]:
    """Write raw/filtered trajectory CSVs and filter provenance sidecars."""

    outdir.mkdir(parents=True, exist_ok=True)
    raw_values = trajectory.unfiltered_values or trajectory.values
    keys = tuple(trajectory.values)
    fieldnames = ("sample_index", "time_s", *keys)
    raw_path = outdir / "trajectory_raw.csv"
    filtered_path = outdir / "trajectory_filtered.csv"
    provenance_path = outdir / "trajectory_filter_provenance.json"
    summary_path = outdir / "trajectory_filter_summary.csv"
    diagnostic_path = outdir / "trajectory_filter_diagnostic.png"
    offset_provenance_path = outdir / "trajectory_offset_provenance.json"
    offset_summary_path = outdir / "trajectory_offset_summary.csv"
    write_rows_csv(raw_path, trajectory_rows(trajectory, values=raw_values), fieldnames)
    write_rows_csv(filtered_path, trajectory_rows(trajectory), fieldnames)
    provenance = dict(trajectory.filter_provenance or {})
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary_rows = []
    input_rms = dict(provenance.get("input_rms_by_column", {}) or {})
    output_rms = dict(provenance.get("output_rms_by_column", {}) or {})
    removed_rms = dict(provenance.get("removed_rms_by_column", {}) or {})
    for key in keys:
        summary_rows.append(
            {
                "column": key,
                "input_rms": input_rms.get(key, ""),
                "output_rms": output_rms.get(key, ""),
                "removed_rms": removed_rms.get(key, ""),
            }
        )
    write_rows_csv(summary_path, summary_rows, ("column", "input_rms", "output_rms", "removed_rms"))
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, max(2.5, 2.0 * len(keys))))
        FigureCanvasAgg(fig)
        axes = fig.subplots(len(keys), 1, sharex=True)
        if len(keys) == 1:
            axes = [axes]
        for axis, key in zip(axes, keys):
            axis.plot(trajectory.time_s, np.asarray(raw_values[key], dtype=float), label="raw", linewidth=1.0)
            axis.plot(trajectory.time_s, np.asarray(trajectory.values[key], dtype=float), label="filtered", linewidth=1.0)
            axis.set_ylabel(key)
            axis.grid(True, alpha=0.25)
        axes[-1].set_xlabel("time_s")
        axes[0].legend(loc="best")
        fig.tight_layout()
        fig.savefig(diagnostic_path, dpi=140)
    except Exception as exc:
        with diagnostic_path.with_suffix(".txt").open("w", encoding="utf-8") as handle:
            handle.write(f"trajectory filter diagnostic plot unavailable: {exc}\n")
    written = {
        "trajectory_raw_csv": raw_path.resolve(),
        "trajectory_filtered_csv": filtered_path.resolve(),
        "trajectory_filter_provenance_json": provenance_path.resolve(),
        "trajectory_filter_summary_csv": summary_path.resolve(),
        "trajectory_filter_diagnostic_png": diagnostic_path.resolve(),
    }
    offset_provenance = dict(trajectory.offset_provenance or {})
    if offset_provenance:
        with offset_provenance_path.open("w", encoding="utf-8") as handle:
            json.dump(offset_provenance, handle, indent=2, sort_keys=True)
            handle.write("\n")
        offset_rows = []
        for key, item in dict(offset_provenance.get("statistics", {}) or {}).items():
            pre = dict(item.get("pre_offset", {}) or {})
            post = dict(item.get("post_offset", {}) or {})
            offset_rows.append(
                {
                    "column": key,
                    "requested_offset": item.get("requested_offset", ""),
                    "pre_mean": pre.get("mean", ""),
                    "post_mean": post.get("mean", ""),
                    "mean_delta": item.get("mean_delta", ""),
                    "pre_std": pre.get("std", ""),
                    "post_std": post.get("std", ""),
                    "std_delta": item.get("std_delta", ""),
                    "pre_min": pre.get("min", ""),
                    "post_min": post.get("min", ""),
                    "pre_max": pre.get("max", ""),
                    "post_max": post.get("max", ""),
                    "pre_peak_to_peak": pre.get("peak_to_peak", ""),
                    "post_peak_to_peak": post.get("peak_to_peak", ""),
                    "peak_to_peak_delta": item.get("peak_to_peak_delta", ""),
                    "residual_std_unchanged": item.get("residual_std_unchanged", ""),
                }
            )
        write_rows_csv(
            offset_summary_path,
            offset_rows,
            (
                "column",
                "requested_offset",
                "pre_mean",
                "post_mean",
                "mean_delta",
                "pre_std",
                "post_std",
                "std_delta",
                "pre_min",
                "post_min",
                "pre_max",
                "post_max",
                "pre_peak_to_peak",
                "post_peak_to_peak",
                "peak_to_peak_delta",
                "residual_std_unchanged",
            ),
        )
        written["trajectory_offset_provenance_json"] = offset_provenance_path.resolve()
        written["trajectory_offset_summary_csv"] = offset_summary_path.resolve()
    return written


def write_subblock_artifacts(
    block: SubblockTrajectory,
    *,
    outdir: Path,
    output_keys: Sequence[str],
) -> dict[str, Path]:
    """Write frame_truth.csv and starting_guess_prediction.csv for one block."""

    truth_path = outdir / "frame_truth.csv"
    guess_path = outdir / "starting_guess_prediction.csv"
    truth_fieldnames = ("frame_index", "time_s", *output_keys)
    guess_fieldnames: list[str] = [
        "subblock_index",
        "frame_index",
        "time_s",
        "time_relative_s",
    ]
    for key in output_keys:
        guess_fieldnames.extend(
            [
                f"{key}_truth",
                f"{key}_linear_fit",
                f"{key}_residual",
                f"{key}_fit_intercept",
                f"{key}_fit_slope_per_s",
                f"rms_{key}_residual",
                f"max_abs_{key}_residual",
            ]
        )
    write_rows_csv(truth_path, frame_truth_rows(block, output_keys), truth_fieldnames)
    write_rows_csv(guess_path, starting_guess_rows(block, output_keys), guess_fieldnames)
    return {"frame_truth_csv": truth_path.resolve(), "starting_guess_prediction_csv": guess_path.resolve()}


def prepare_airbus_subblocks(
    *,
    path: Path,
    start_s: float,
    duration_s: float | None = None,
    n_subblocks: int | None = None,
    sample_dt_s: float = 0.1,
    frame_dt_s: float = 0.05,
    subblock_duration_s: float = 1.0,
    n_frames_per_subblock: int = 20,
    output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
    fit_keys: Sequence[str] | None = None,
    time_mode: str = "inferred_uniform",
    time_start_s: float = 0.0,
    interpolation: str = "linear",
    filter_config: Mapping[str, Any] | None = None,
    offsets_config: Mapping[str, Any] | None = None,
) -> tuple[CanonicalTrajectory, np.ndarray, list[SubblockTrajectory]]:
    """Load Airbus trajectory and return prepared subblocks."""

    raw = load_airbus_csv(
        path,
        sample_dt_s=sample_dt_s,
        time_mode=time_mode,
        start_s=time_start_s,
    )
    trajectory = map_trajectory(raw, output_keys=output_keys)
    filter_spec = parse_trajectory_filter_config(filter_config)
    if filter_spec.apply_stage == "before_window":
        trajectory = apply_filter_to_trajectory(trajectory, config=filter_config)
        trajectory = apply_offsets_to_trajectory(
            trajectory,
            offsets=offsets_config,
            stage="after_filter_before_window_interpolation",
        )
    resolved_duration = derive_window_duration(
        duration_s=duration_s,
        n_subblocks=n_subblocks,
        subblock_duration_s=subblock_duration_s,
    )
    frame_times = build_frame_times(
        start_s=start_s,
        duration_s=resolved_duration,
        frame_dt_s=frame_dt_s,
        n_frames_per_subblock=n_frames_per_subblock,
        subblock_duration_s=subblock_duration_s,
    )
    truth = interpolate_trajectory(
        trajectory,
        frame_times_s=frame_times,
        method=interpolation,
    )
    if filter_spec.apply_stage == "after_window":
        columns = tuple(key for key in filter_spec.columns if key in truth)
        if not columns and filter_spec.enabled and filter_spec.kind != "none":
            raise ValueError(
                "trajectory filter columns do not match interpolated trajectory keys: "
                + ", ".join(filter_spec.columns)
            )
        if columns:
            values = np.column_stack([np.asarray(truth[key], dtype=float) for key in columns])
            filtered, provenance = apply_trajectory_filter(
                frame_times,
                values,
                TrajectoryFilterSpec(**{**filter_spec.__dict__, "columns": columns}),
                axis=0,
            )
            if filter_spec.enabled and filter_spec.kind != "none":
                provenance.setdefault("warnings", []).append(
                    "filter apply_stage=after_window can introduce edge artifacts in the selected segment"
                )
                for index, key in enumerate(columns):
                    truth[key] = np.asarray(filtered[:, index], dtype=float)
            trajectory = CanonicalTrajectory(
                time_s=trajectory.time_s,
                values=trajectory.values,
                raw=trajectory.raw,
                mapping=trajectory.mapping,
                filter_provenance=provenance,
                offset_provenance=trajectory.offset_provenance,
                unfiltered_values=trajectory.values,
            )
        truth, offset_provenance = apply_offsets_to_values(
            truth,
            offsets=offsets_config,
            stage="after_window_filter_before_subblock_split",
        )
        trajectory = CanonicalTrajectory(
            time_s=trajectory.time_s,
            values=trajectory.values,
            raw=trajectory.raw,
            mapping=trajectory.mapping,
            filter_provenance=trajectory.filter_provenance,
            offset_provenance=offset_provenance,
            unfiltered_values=trajectory.unfiltered_values,
        )
    blocks = split_subblocks(
        frame_times_s=frame_times,
        truth_values=truth,
        n_frames_per_subblock=n_frames_per_subblock,
        fit_keys=fit_keys,
    )
    return trajectory, frame_times, blocks


__all__ = [
    "AIRBUS_DEFAULT_MAPPING",
    "DEFAULT_OUTPUT_KEYS",
    "TRAJECTORY_NOTES",
    "CanonicalTrajectory",
    "RawTrajectory",
    "SubblockTrajectory",
    "build_frame_times",
    "derive_window_duration",
    "frame_truth_rows",
    "interpolate_trajectory",
    "apply_offsets_to_trajectory",
    "apply_offsets_to_values",
    "apply_filter_to_trajectory",
    "load_airbus_csv",
    "map_trajectory",
    "parse_trajectory_offsets_config",
    "prepare_airbus_subblocks",
    "split_subblocks",
    "starting_guess_rows",
    "write_rows_csv",
    "write_subblock_artifacts",
    "write_trajectory_filter_artifacts",
]
