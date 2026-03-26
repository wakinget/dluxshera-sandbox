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


def test_trace_non_contiguous_allowed_when_flag_disabled(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_non_contiguous_allowed.csv",
        """
        frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg
        0,0.0,0.0,0.0,90.0
        2,0.1,0.1,-0.1,90.2
        """,
    )

    trace = load_obs_subblock_trace_csv(
        trace_path, require_contiguous_frame_index=False
    )
    assert trace.frame_count == 2
    assert [row["frame_index"] for row in trace.rows] == [0, 2]


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


def test_trace_non_monotonic_allowed_when_flag_disabled(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_non_monotonic_allowed.csv",
        """
        frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg
        0,0.1,0.0,0.0,90.0
        1,0.0,0.1,-0.1,90.2
        """,
    )

    trace = load_obs_subblock_trace_csv(trace_path, require_monotonic_time=False)
    assert trace.frame_count == 2
    assert [row["time_s"] for row in trace.rows] == [0.1, 0.0]


def test_trace_duplicate_frame_index_is_always_hard_error(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_duplicate_frame_index.csv",
        """
        frame_index,time_s,source.x_position_as,source.y_position_as,source.position_angle_deg
        0,0.0,0.0,0.0,90.0
        0,0.1,0.1,-0.1,90.2
        """,
    )

    with pytest.raises(ValueError, match="duplicate frame_index"):
        load_obs_subblock_trace_csv(
            trace_path,
            require_contiguous_frame_index=False,
            require_monotonic_time=False,
        )


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


def test_trace_generalized_required_varying_columns_are_parsed(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_generalized.csv",
        """
        frame_index,time_s,source.x_position_as,optics.plate_scale_as_per_pix,optics.primary.zernike_coeffs_nm[3],tag
        0,0.00,0.0,0.11,1.0,a
        1,0.05,0.1,0.12,1.5,b
        """,
    )

    trace = load_obs_subblock_trace_csv(
        trace_path,
        required_varying_keys=(
            "source.x_position_as",
            "optics.plate_scale_as_per_pix",
            "optics.primary.zernike_coeffs_nm[3]",
        ),
    )

    assert trace.required_columns == (
        "frame_index",
        "time_s",
        "source.x_position_as",
        "optics.plate_scale_as_per_pix",
        "optics.primary.zernike_coeffs_nm[3]",
    )
    assert trace.rows[0]["optics.plate_scale_as_per_pix"] == 0.11
    assert trace.rows[1]["optics.primary.zernike_coeffs_nm[3]"] == 1.5
    assert trace.extra_columns == ("tag",)


def test_trace_generalized_required_column_must_be_finite(tmp_path):
    trace_path = _write_trace(
        tmp_path / "trace_non_finite_generalized.csv",
        """
        frame_index,time_s,source.x_position_as,optics.plate_scale_as_per_pix
        0,0.00,0.0,0.11
        1,0.05,0.1,nan
        """,
    )

    with pytest.raises(ValueError, match="must be finite"):
        load_obs_subblock_trace_csv(
            trace_path,
            required_varying_keys=(
                "source.x_position_as",
                "optics.plate_scale_as_per_pix",
            ),
        )
