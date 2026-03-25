from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dluxshera.plot.obs_subblock import (
    apply_intensity_stretch,
    compute_cube_display_limits,
    make_obs_subblock_summary_figure,
    make_obs_subblock_trace_summary_figure,
    write_obs_subblock_preview_gif,
)
from dluxshera.utils.obs_subblock_trace import ObsSubblockTrace


def _toy_cube(n_frame: int = 8, ny: int = 16, nx: int = 16) -> np.ndarray:
    cube = np.zeros((n_frame, ny, nx), dtype=float)
    for i in range(n_frame):
        cube[i, ny // 2, (i + 3) % nx] = 2.0
        cube[i] += 0.05 * i
    return cube


def _toy_trace(n_frame: int = 8) -> ObsSubblockTrace:
    rows = []
    for i in range(n_frame):
        rows.append(
            {
                "frame_index": i,
                "time_s": 0.1 * i,
                "source.x_position_as": 0.001 * i,
                "source.y_position_as": -0.001 * i,
                "source.position_angle_deg": 10.0 + i,
            }
        )

    return ObsSubblockTrace(
        rows=tuple(rows),
        required_columns=(
            "frame_index",
            "time_s",
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
        extra_columns=(),
        source_path=Path("dummy.csv"),
    )


def test_compute_cube_display_limits_percentile_bounds():
    cube = _toy_cube()
    vmin, vmax = compute_cube_display_limits(cube, pmin=5.0, pmax=95.0)
    assert np.isfinite(vmin)
    assert np.isfinite(vmax)
    assert vmax > vmin


def test_apply_intensity_stretch_log_rejects_negative_range():
    image = np.array([[-1.0, 0.0], [1.0, 2.0]])
    try:
        apply_intensity_stretch(image, vmin=-1.0, vmax=2.0, stretch="log")
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for log stretch with vmin < 0")


def test_summary_and_trace_summary_smoke(tmp_path: Path):
    cube = _toy_cube()
    trace = _toy_trace(n_frame=cube.shape[0])

    fig_summary, axes_summary = make_obs_subblock_summary_figure(cube, stretch="sqrt")
    assert axes_summary.shape == (2, 3)
    summary_path = tmp_path / "summary.png"
    fig_summary.savefig(summary_path)
    plt.close(fig_summary)

    fig_trace, axes_trace = make_obs_subblock_trace_summary_figure(trace)
    assert axes_trace.shape == (2, 2)
    trace_path = tmp_path / "trace_summary.png"
    fig_trace.savefig(trace_path)
    plt.close(fig_trace)

    assert summary_path.exists()
    assert trace_path.exists()


def test_preview_gif_writes_file(tmp_path: Path):
    cube = _toy_cube()
    trace = _toy_trace(n_frame=cube.shape[0])
    gif_path = tmp_path / "preview.gif"
    write_obs_subblock_preview_gif(
        cube,
        output_path=gif_path,
        trace=trace,
        stride=2,
        pmin=1.0,
        pmax=99.0,
        stretch="linear",
    )
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0
