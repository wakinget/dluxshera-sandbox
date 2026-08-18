"""Trace loading and validation helpers for observation sub-block rendering.

CSV-backed explicit traces are the canonical interface for observation-subblock
rendering. The required per-frame value columns are caller-configurable via
``required_varying_keys`` so this loader works for both v1 ``x/y/PA`` traces
and generalized Phase 4 varying-key sets.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .obs_subblock_keys import OBS_SUBBLOCK_V1_DEFAULT_VARYING_KEYS

DEFAULT_TRACE_VARYING_KEYS: tuple[str, ...] = (
    *OBS_SUBBLOCK_V1_DEFAULT_VARYING_KEYS,
)

# Backward-compatible alias retained for older internal imports.
APPLIED_V1_VARYING_KEYS: tuple[str, ...] = DEFAULT_TRACE_VARYING_KEYS

TRACE_BASE_COLUMNS: tuple[str, ...] = (
    "frame_index",
    "time_s",
)

REQUIRED_TRACE_COLUMNS: tuple[str, ...] = (
    *TRACE_BASE_COLUMNS,
    *DEFAULT_TRACE_VARYING_KEYS,
)


@dataclass(frozen=True)
class ObsSubblockTrace:
    """Represent a validated observation sub-block trace table.

    Parameters
    ----------
    rows : tuple[dict[str, Any], ...]
        Per-frame rows sorted by ``frame_index``. Required v1 fields are typed
        as ``int``/``float``; extra columns are preserved as strings or
        ``None``.
    required_columns : tuple[str, ...]
        Required trace columns for this load operation (base columns plus
        requested varying-key columns).
    extra_columns : tuple[str, ...]
        Input trace columns that are not required by v1 rendering.
    source_path : Path
        Resolved CSV path used to load this trace.
    """

    rows: tuple[dict[str, Any], ...]
    required_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    source_path: Path

    @property
    def frame_count(self) -> int:
        """Return the number of frames in the trace."""

        return len(self.rows)

    @property
    def time_start_s(self) -> float | None:
        """Return first frame time in seconds, if any rows are present."""

        if not self.rows:
            return None
        return float(self.rows[0]["time_s"])

    @property
    def time_stop_s(self) -> float | None:
        """Return last frame time in seconds, if any rows are present."""

        if not self.rows:
            return None
        return float(self.rows[-1]["time_s"])


def _require_cell(row: dict[str, str | None], key: str, *, row_number: int) -> str:
    """Return a required CSV cell value or raise a clear ValueError."""

    value = row.get(key, "")
    text = "" if value is None else value.strip()
    if text == "":
        raise ValueError(f"Trace row {row_number}: missing required value for {key!r}.")
    return text


def _parse_frame_index(text: str, *, row_number: int) -> int:
    """Parse and validate a ``frame_index`` CSV cell."""

    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(
            f"Trace row {row_number}: frame_index must be an integer, got {text!r}."
        ) from exc
    if value < 0:
        raise ValueError(f"Trace row {row_number}: frame_index must be >= 0.")
    return value


def _parse_float(text: str, *, key: str, row_number: int) -> float:
    """Parse and validate a float CSV cell."""

    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Trace row {row_number}: {key} must be numeric, got {text!r}."
        ) from exc
    if not math.isfinite(value):
        raise ValueError(
            f"Trace row {row_number}: {key} must be finite, got {text!r}."
        )
    return value


def _normalize_extra_cell(text: str | None) -> str | None:
    """Normalize optional extra-column text."""

    stripped = "" if text is None else text.strip()
    if stripped == "":
        return None
    return stripped


def _normalize_required_varying_keys(
    required_varying_keys: Sequence[str] | None,
) -> tuple[str, ...]:
    if required_varying_keys is None:
        return DEFAULT_TRACE_VARYING_KEYS

    normalized: list[str] = []
    seen: set[str] = set()
    for idx, key in enumerate(required_varying_keys):
        if not isinstance(key, str):
            raise ValueError(
                "required_varying_keys must contain only strings; "
                f"index {idx} has {type(key).__name__}."
            )
        text = key.strip()
        if not text:
            raise ValueError(
                f"required_varying_keys[{idx}] is blank after stripping."
            )
        if text in seen:
            raise ValueError(f"Duplicate entry in required_varying_keys: {text!r}.")
        seen.add(text)
        normalized.append(text)
    return tuple(normalized)


def load_obs_subblock_trace_csv(
    path: Path,
    *,
    required_varying_keys: Sequence[str] | None = None,
    require_contiguous_frame_index: bool = True,
    require_monotonic_time: bool = True,
) -> ObsSubblockTrace:
    """Load and validate an observation sub-block trace CSV.

    Parameters
    ----------
    path : Path
        CSV path containing required trace columns.
    required_varying_keys : Sequence[str] | None, optional
        Varying-key columns required in addition to ``frame_index`` and
        ``time_s``. When omitted, defaults to ``x/y/PA`` for backward
        compatibility.

    Returns
    -------
    ObsSubblockTrace
        Validated trace rows sorted by ``frame_index``.

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    ValueError
        If required columns are missing, rows are malformed, duplicate
        ``frame_index`` values are present, or enabled validation checks fail.
    """

    if not path.exists():
        raise FileNotFoundError(f"Trace CSV not found: {path}")

    varying_keys = _normalize_required_varying_keys(required_varying_keys)
    required_columns = (*TRACE_BASE_COLUMNS, *varying_keys)

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Trace CSV has no header row: {path}")
        fieldnames = [name.strip() for name in reader.fieldnames if name is not None]
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise ValueError(
                "Trace CSV is missing required columns: "
                + ", ".join(missing)
            )

        extra_columns = tuple(
            column for column in fieldnames if column not in required_columns
        )
        parsed_rows: list[dict[str, Any]] = []

        for row_number, row in enumerate(reader, start=2):
            frame_index_text = _require_cell(row, "frame_index", row_number=row_number)
            time_text = _require_cell(row, "time_s", row_number=row_number)

            parsed_row: dict[str, Any] = {
                "frame_index": _parse_frame_index(frame_index_text, row_number=row_number),
                "time_s": _parse_float(time_text, key="time_s", row_number=row_number),
            }
            for key in varying_keys:
                parsed_row[key] = _parse_float(
                    _require_cell(row, key, row_number=row_number),
                    key=key,
                    row_number=row_number,
                )
            for column in extra_columns:
                parsed_row[column] = _normalize_extra_cell(row.get(column, ""))

            parsed_rows.append(parsed_row)

    if not parsed_rows:
        raise ValueError("Trace CSV must contain at least one frame row.")

    sorted_rows = sorted(parsed_rows, key=lambda item: int(item["frame_index"]))
    frame_indices = [int(row["frame_index"]) for row in sorted_rows]
    counts = Counter(frame_indices)
    duplicate_values = sorted(value for value, count in counts.items() if count > 1)
    if duplicate_values:
        raise ValueError(
            "Trace contains duplicate frame_index values: "
            + ", ".join(str(value) for value in duplicate_values)
        )

    if require_contiguous_frame_index:
        expected = list(range(len(sorted_rows)))
        if frame_indices != expected:
            raise ValueError(
                "Trace frame_index must be contiguous 0..N-1 after sorting, "
                f"got {frame_indices}."
            )

    if require_monotonic_time:
        times = [float(row["time_s"]) for row in sorted_rows]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("Trace time_s must be monotonic non-decreasing.")

    return ObsSubblockTrace(
        rows=tuple(sorted_rows),
        required_columns=required_columns,
        extra_columns=extra_columns,
        source_path=path.resolve(),
    )


__all__ = [
    "APPLIED_V1_VARYING_KEYS",
    "DEFAULT_TRACE_VARYING_KEYS",
    "ObsSubblockTrace",
    "REQUIRED_TRACE_COLUMNS",
    "TRACE_BASE_COLUMNS",
    "load_obs_subblock_trace_csv",
]
