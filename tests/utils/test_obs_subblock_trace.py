from __future__ import annotations

from pathlib import Path

import pytest

from dluxshera.utils.obs_subblock_trace import load_obs_subblock_trace_csv


def _write_trace(path: Path, text: str) -> Path:
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def test_trace_missing_required_column_raises(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_missing.csv",
        """
        frame_index,time_s,source.x_position_as,source.y_position_as
        0,0.0,0.0,0.0
        """,
    )

    with pytest.raises(ValueError, match="missing required columns"):
        load_obs_subblock_trace_csv(trace_path)


def test_trace_non_contiguous_frame_index_raises(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_non_contiguous.csv",
        """
        frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg
        0,0.0,0.0,0.0,90.0
        2,0.1,0.1,-0.1,90.2
        """,
    )

    with pytest.raises(ValueError, match="frame_index must be contiguous"):
        load_obs_subblock_trace_csv(trace_path)


def test_trace_non_monotonic_time_raises(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_non_monotonic.csv",
        """
        frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg
        0,0.1,0.0,0.0,90.0
        1,0.0,0.1,-0.1,90.2
        """,
    )

    with pytest.raises(ValueError, match="time_s must be monotonic"):
        load_obs_subblock_trace_csv(trace_path)


def test_trace_extra_columns_preserved(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_extra_columns.csv",
        """
        frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg,tag,mode
        1,0.05,0.1,-0.1,90.1,b,science
        0,0.00,0.0,0.0,90.0,a,guide
        """,
    )

    trace = load_obs_subblock_trace_csv(trace_path)

    assert trace.extra_columns == ("tag", "mode")
    assert trace.frame_count == 2
    assert trace.rows[0]["frame_index"] == 0
    assert trace.rows[0]["tag"] == "a"
    assert trace.rows[1]["frame_index"] == 1
    assert trace.rows[1]["mode"] == "science"
