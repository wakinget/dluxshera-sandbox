"""Reusable filtering helpers for canonical trajectory traces.

The filter cutoff period fields are convenience aliases for frequency:
``cutoff_hz = 1 / cutoff_period_s``. They are not RC time constants.
Zero-phase filtering uses forward/backward filtering and is appropriate for
offline trajectory truth preprocessing; causal filtering keeps phase lag and is
more representative of realtime estimator studies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SUPPORTED_FILTER_KINDS = ("none", "low_pass", "high_pass", "band_pass")
SUPPORTED_FILTER_METHODS = ("bessel",)
DEFAULT_FILTER_COLUMNS = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)


@dataclass(frozen=True)
class TrajectoryFilterSpec:
    enabled: bool = False
    kind: str = "none"
    method: str = "bessel"
    order: int = 4
    cutoff_hz: float | None = None
    cutoff_period_s: float | None = None
    low_cutoff_hz: float | None = None
    high_cutoff_hz: float | None = None
    low_cutoff_period_s: float | None = None
    high_cutoff_period_s: float | None = None
    zero_phase: bool = True
    pad_policy: str = "default"
    columns: tuple[str, ...] = DEFAULT_FILTER_COLUMNS
    units: str | None = None
    apply_stage: str = "before_window"
    write_unfiltered_comparison: bool = False


def _optional_float(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trajectory filter {name} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"trajectory filter {name} must be finite.")
    return parsed


def _period_to_hz(period_s: float | None, *, name: str) -> float | None:
    if period_s is None:
        return None
    if period_s <= 0.0:
        raise ValueError(f"trajectory filter {name} must be > 0.")
    return 1.0 / period_s


def _coalesce_cutoff(
    explicit_hz: float | None,
    period_s: float | None,
    *,
    hz_name: str,
    period_name: str,
) -> float | None:
    from_period = _period_to_hz(period_s, name=period_name)
    if explicit_hz is not None:
        if explicit_hz <= 0.0:
            raise ValueError(f"trajectory filter {hz_name} must be > 0.")
        if from_period is not None and not np.isclose(explicit_hz, from_period):
            raise ValueError(
                f"trajectory filter {hz_name} and {period_name} disagree; "
                f"period implies {from_period:g} Hz."
            )
        return explicit_hz
    return from_period


def _as_columns(value: Any) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_FILTER_COLUMNS
    if isinstance(value, str):
        columns = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Sequence):
        columns = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        raise ValueError("trajectory filter columns must be list[str] or comma-separated string.")
    if not columns:
        raise ValueError("trajectory filter columns must contain at least one column.")
    return columns


def parse_trajectory_filter_config(config: Mapping[str, Any] | None) -> TrajectoryFilterSpec:
    """Parse and validate a trajectory filter configuration block."""

    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    kind = str(cfg.get("kind", "none" if not enabled else "high_pass")).strip().lower()
    method = str(cfg.get("method", "bessel")).strip().lower()
    order = int(cfg.get("order", 4))
    if kind not in SUPPORTED_FILTER_KINDS:
        raise ValueError("trajectory filter kind must be one of: " + ", ".join(SUPPORTED_FILTER_KINDS))
    if method not in SUPPORTED_FILTER_METHODS:
        raise ValueError("trajectory filter method must be one of: " + ", ".join(SUPPORTED_FILTER_METHODS))
    if order < 1:
        raise ValueError("trajectory filter order must be >= 1.")
    pad_policy = str(cfg.get("pad_policy", "default")).strip().lower()
    if pad_policy not in {"default"}:
        raise ValueError("trajectory filter pad_policy currently supports only 'default'.")
    apply_stage = str(cfg.get("apply_stage", "before_window")).strip().lower()
    if apply_stage not in {"before_window", "after_window"}:
        raise ValueError("trajectory filter apply_stage must be before_window or after_window.")

    cutoff_period_s = _optional_float(cfg.get("cutoff_period_s"), name="cutoff_period_s")
    low_period_s = _optional_float(cfg.get("low_cutoff_period_s"), name="low_cutoff_period_s")
    high_period_s = _optional_float(cfg.get("high_cutoff_period_s"), name="high_cutoff_period_s")
    cutoff_hz = _coalesce_cutoff(
        _optional_float(cfg.get("cutoff_hz"), name="cutoff_hz"),
        cutoff_period_s,
        hz_name="cutoff_hz",
        period_name="cutoff_period_s",
    )
    low_hz = _coalesce_cutoff(
        _optional_float(cfg.get("low_cutoff_hz"), name="low_cutoff_hz"),
        low_period_s,
        hz_name="low_cutoff_hz",
        period_name="low_cutoff_period_s",
    )
    high_hz = _coalesce_cutoff(
        _optional_float(cfg.get("high_cutoff_hz"), name="high_cutoff_hz"),
        high_period_s,
        hz_name="high_cutoff_hz",
        period_name="high_cutoff_period_s",
    )
    if kind in {"low_pass", "high_pass"} and enabled and cutoff_hz is None:
        raise ValueError(f"trajectory filter kind={kind!r} requires cutoff_hz or cutoff_period_s.")
    if kind == "band_pass" and enabled:
        if low_hz is None or high_hz is None:
            raise ValueError("trajectory band_pass filter requires low and high cutoff fields.")
        if low_hz >= high_hz:
            raise ValueError("trajectory band_pass requires low_cutoff_hz < high_cutoff_hz.")

    return TrajectoryFilterSpec(
        enabled=enabled,
        kind=kind,
        method=method,
        order=order,
        cutoff_hz=cutoff_hz,
        cutoff_period_s=cutoff_period_s,
        low_cutoff_hz=low_hz,
        high_cutoff_hz=high_hz,
        low_cutoff_period_s=low_period_s,
        high_cutoff_period_s=high_period_s,
        zero_phase=bool(cfg.get("zero_phase", True)),
        pad_policy=pad_policy,
        columns=_as_columns(cfg.get("columns")),
        units=None if cfg.get("units") is None else str(cfg.get("units")),
        apply_stage=apply_stage,
        write_unfiltered_comparison=bool(cfg.get("write_unfiltered_comparison", False)),
    )


def _sample_summary(time_s: np.ndarray) -> dict[str, float]:
    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1:
        raise ValueError("trajectory filter time_s must be one-dimensional.")
    if times.size < 2:
        raise ValueError("trajectory filter requires at least two time samples.")
    if not np.all(np.isfinite(times)):
        raise ValueError("trajectory filter time_s must contain only finite values.")
    diffs = np.diff(times)
    if np.any(diffs <= 0.0):
        raise ValueError("trajectory filter time_s must be strictly increasing.")
    dt = float(np.median(diffs))
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("trajectory filter inferred sample dt must be positive.")
    if not np.allclose(diffs, dt, rtol=1.0e-5, atol=max(1.0e-12, abs(dt) * 1.0e-7)):
        raise ValueError(
            "trajectory filter requires approximately uniform sampling; resample before filtering."
        )
    sample_rate = 1.0 / dt
    return {
        "sample_dt_s": dt,
        "sample_rate_hz": sample_rate,
        "nyquist_hz": 0.5 * sample_rate,
    }


def _filter_cutoffs(spec: TrajectoryFilterSpec) -> float | tuple[float, float] | None:
    if spec.kind in {"low_pass", "high_pass"}:
        return spec.cutoff_hz
    if spec.kind == "band_pass":
        assert spec.low_cutoff_hz is not None and spec.high_cutoff_hz is not None
        return (spec.low_cutoff_hz, spec.high_cutoff_hz)
    return None


def _rms_by_column(values: np.ndarray, columns: Sequence[str]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return {str(columns[0] if columns else "value"): float(np.sqrt(np.mean(np.square(arr))))}
    if arr.shape[1] != len(columns):
        return {f"column_{i}": float(np.sqrt(np.mean(np.square(arr[:, i])))) for i in range(arr.shape[1])}
    return {
        str(column): float(np.sqrt(np.mean(np.square(arr[:, index]))))
        for index, column in enumerate(columns)
    }


def _normalise_values(values: np.ndarray, *, axis: int) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim != 2:
        raise ValueError("trajectory filter values must be one-dimensional or two-dimensional.")
    if axis == 0:
        return arr, False
    if axis == 1:
        return np.swapaxes(arr, 0, 1), False
    raise ValueError("trajectory filter axis must be 0 or 1.")


def _restore_values(values_2d: np.ndarray, *, was_1d: bool, axis: int) -> np.ndarray:
    if was_1d:
        return values_2d[:, 0]
    if axis == 0:
        return values_2d
    return np.swapaxes(values_2d, 0, 1)


def _bessel_provenance_response(sos: np.ndarray, sample_rate_hz: float) -> dict[str, Any]:
    try:
        from scipy import signal

        freq, response = signal.sosfreqz(sos, worN=256, fs=sample_rate_hz)
        gain = np.abs(response)
        return {
            "frequency_hz": [float(item) for item in freq],
            "gain": [float(item) for item in gain],
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"warning": f"frequency response unavailable: {exc}"}


def apply_trajectory_filter(
    time_s: np.ndarray,
    values: np.ndarray,
    spec: TrajectoryFilterSpec,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a trajectory filter to one or more value columns."""

    sample = _sample_summary(time_s)
    values_2d, was_1d = _normalise_values(values, axis=axis)
    if values_2d.shape[0] != np.asarray(time_s).size:
        raise ValueError("trajectory filter values length must match time_s along the sample axis.")
    if not np.all(np.isfinite(values_2d)):
        raise ValueError("trajectory filter values must contain only finite values.")

    if was_1d and spec.columns == DEFAULT_FILTER_COLUMNS:
        columns = ("value",)
    else:
        columns = tuple(spec.columns[: values_2d.shape[1]])
    if len(columns) != values_2d.shape[1]:
        columns = tuple(f"column_{index}" for index in range(values_2d.shape[1]))
    provenance: dict[str, Any] = {
        "schema_version": "trajectory_filter_provenance.v1",
        "enabled": bool(spec.enabled),
        "kind": spec.kind,
        "method": spec.method,
        "order": int(spec.order),
        "zero_phase": bool(spec.zero_phase),
        "pad_policy": spec.pad_policy,
        "cutoff_hz": spec.cutoff_hz,
        "cutoff_period_s": spec.cutoff_period_s,
        "low_cutoff_hz": spec.low_cutoff_hz,
        "low_cutoff_period_s": spec.low_cutoff_period_s,
        "high_cutoff_hz": spec.high_cutoff_hz,
        "high_cutoff_period_s": spec.high_cutoff_period_s,
        "sample_dt_s": sample["sample_dt_s"],
        "sample_rate_hz": sample["sample_rate_hz"],
        "nyquist_hz": sample["nyquist_hz"],
        "columns_filtered": list(columns) if spec.enabled and spec.kind != "none" else [],
        "input_rms_by_column": _rms_by_column(values_2d, columns),
        "output_rms_by_column": {},
        "removed_rms_by_column": {},
        "warnings": [],
    }

    if not spec.enabled or spec.kind == "none":
        provenance["output_rms_by_column"] = _rms_by_column(values_2d, columns)
        provenance["removed_rms_by_column"] = _rms_by_column(np.zeros_like(values_2d), columns)
        return _restore_values(values_2d.copy(), was_1d=was_1d, axis=axis), provenance

    cutoff = _filter_cutoffs(spec)
    nyquist = sample["nyquist_hz"]
    cutoff_values = cutoff if isinstance(cutoff, tuple) else (cutoff,)
    for value in cutoff_values:
        if value is None or value <= 0.0:
            raise ValueError("trajectory filter cutoff frequencies must be positive.")
        if value >= nyquist:
            raise ValueError(
                f"trajectory filter cutoff {value:g} Hz must be below Nyquist {nyquist:g} Hz."
            )

    if spec.method != "bessel":
        raise ValueError(f"trajectory filter method {spec.method!r} is not implemented.")
    try:
        from scipy import signal
    except ImportError as exc:
        raise ImportError(
            "trajectory filter method='bessel' requires SciPy. Install scipy or disable filtering."
        ) from exc

    btype = {"low_pass": "lowpass", "high_pass": "highpass", "band_pass": "bandpass"}[spec.kind]
    sos = signal.bessel(
        int(spec.order),
        cutoff,
        btype=btype,
        analog=False,
        output="sos",
        fs=sample["sample_rate_hz"],
        norm="phase",
    )
    if spec.zero_phase:
        # Match scipy.signal.sosfiltfilt's default padlen expression closely so
        # short trajectory windows fail before SciPy raises an opaque error.
        n_sections = sos.shape[0]
        zeros_at_origin = (sos[:, 2] == 0).sum()
        poles_at_origin = (sos[:, 5] == 0).sum()
        padlen = 3 * (2 * n_sections + 1 - min(zeros_at_origin, poles_at_origin))
        if values_2d.shape[0] <= padlen:
            raise ValueError(
                "trajectory filter zero-phase padding requires more samples than "
                f"padlen={padlen}; got {values_2d.shape[0]}. Filter a longer segment "
                "or use zero_phase=false."
            )
        filtered = signal.sosfiltfilt(sos, values_2d, axis=0)
    else:
        provenance["warnings"].append("causal sosfilt was used; phase/group delay is expected")
        filtered = signal.sosfilt(sos, values_2d, axis=0)

    removed = values_2d - filtered
    provenance["output_rms_by_column"] = _rms_by_column(filtered, columns)
    provenance["removed_rms_by_column"] = _rms_by_column(removed, columns)
    provenance["frequency_response"] = _bessel_provenance_response(sos, sample["sample_rate_hz"])
    return _restore_values(filtered, was_1d=was_1d, axis=axis), provenance


def apply_trajectory_filters_to_table(
    rows_or_table: Any,
    *,
    time_key: str,
    value_keys: Sequence[str],
    config: Mapping[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    """Apply configured filtering to a sequence of mapping rows or dict-of-arrays table."""

    spec = parse_trajectory_filter_config(config)
    keys = tuple(str(key) for key in value_keys)
    if isinstance(rows_or_table, Mapping):
        time_s = np.asarray(rows_or_table[time_key], dtype=float)
        values = np.column_stack([np.asarray(rows_or_table[key], dtype=float) for key in keys])
        filtered, provenance = apply_trajectory_filter(
            time_s,
            values,
            TrajectoryFilterSpec(**{**spec.__dict__, "columns": keys}),
            axis=0,
        )
        table = dict(rows_or_table)
        for index, key in enumerate(keys):
            table[key] = filtered[:, index]
        return table, provenance

    rows = [dict(row) for row in rows_or_table]
    time_s = np.asarray([row[time_key] for row in rows], dtype=float)
    values = np.asarray([[row[key] for key in keys] for row in rows], dtype=float)
    filtered, provenance = apply_trajectory_filter(
        time_s,
        values,
        TrajectoryFilterSpec(**{**spec.__dict__, "columns": keys}),
        axis=0,
    )
    for row_index, row in enumerate(rows):
        for column_index, key in enumerate(keys):
            row[key] = float(filtered[row_index, column_index])
    return rows, provenance


__all__ = [
    "DEFAULT_FILTER_COLUMNS",
    "SUPPORTED_FILTER_KINDS",
    "SUPPORTED_FILTER_METHODS",
    "TrajectoryFilterSpec",
    "apply_trajectory_filter",
    "apply_trajectory_filters_to_table",
    "parse_trajectory_filter_config",
]
