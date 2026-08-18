from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from dluxshera.utils.full_fidelity_review import (  # noqa: E402
    plot_trajectory_review_components,
    trajectory_filter_provenance_table,
    trajectory_timing_summary_table,
)
from dluxshera.utils.obs_subblock_trajectory import (  # noqa: E402
    CanonicalTrajectory,
    RawTrajectory,
    SubblockTrajectory,
)


def _synthetic_review(kind: str = "high_pass") -> dict:
    time_s = np.linspace(0.0, 2.0, 21)
    raw_values = {
        "source.x_position_as": 10.0 + 0.2 * time_s + 0.01 * np.sin(20.0 * time_s),
        "source.y_position_as": -5.0 + 0.1 * time_s + 0.02 * np.cos(18.0 * time_s),
        "source.position_angle_deg": 0.5 + 0.001 * np.sin(15.0 * time_s),
    }
    filtered_values = {key: value - np.mean(value) for key, value in raw_values.items()}
    raw = RawTrajectory(
        time_s=time_s,
        columns={},
        source_path=Path(__file__),
        source_kind="synthetic",
    )
    trajectory = CanonicalTrajectory(
        time_s=time_s,
        values=filtered_values,
        raw=raw,
        mapping={},
        unfiltered_values=raw_values,
        filter_provenance={
            "enabled": True,
            "kind": kind,
            "method": "bessel",
            "order": 4,
            "cutoff_period_s": 15.0 if kind != "band_pass" else None,
            "cutoff_hz": 1.0 / 15.0 if kind != "band_pass" else None,
            "low_cutoff_period_s": 30.0 if kind == "band_pass" else None,
            "low_cutoff_hz": 1.0 / 30.0 if kind == "band_pass" else None,
            "high_cutoff_period_s": 5.0 if kind == "band_pass" else None,
            "high_cutoff_hz": 1.0 / 5.0 if kind == "band_pass" else None,
        },
    )

    blocks = []
    for index, start in enumerate((0.0, 1.0)):
        frame_times = np.asarray([start, start + 0.1, start + 0.2])
        truth = {
            key: np.interp(frame_times, time_s, values)
            for key, values in filtered_values.items()
        }
        prediction = {
            key: np.linspace(values[0], values[-1], values.size)
            for key, values in truth.items()
        }
        residual = {key: truth[key] - prediction[key] for key in truth}
        diagnostics = {
            key: {"rms_residual": float(np.sqrt(np.mean(np.square(value))))}
            for key, value in residual.items()
        }
        blocks.append(
            SubblockTrajectory(
                subblock_index=index,
                frame_times_s=frame_times,
                time_relative_s=frame_times - frame_times[0],
                truth=truth,
                prediction=prediction,
                residual=residual,
                fit_coefficients={key: (float(values[0]), 0.0) for key, values in truth.items()},
                diagnostics=diagnostics,
            )
        )

    return {
        "available": True,
        "trajectory": trajectory,
        "frame_times_s": np.concatenate([block.frame_times_s for block in blocks]),
        "blocks": blocks,
    }


def test_trajectory_plot_uses_separate_raw_filtered_removed_axes() -> None:
    figures = plot_trajectory_review_components(
        _synthetic_review("high_pass"),
        keys=["source.x_position_as"],
    )

    assert len(figures) == 1
    axes = figures[0].axes
    assert len(axes) == 4
    titles = [axis.get_title() for axis in axes]
    assert titles[0].startswith("raw trajectory:")
    assert titles[1].startswith("high-pass filtered residual:")
    assert "low-frequency component removed" in titles[2]
    assert "selected subblock frame samples" in titles[3]


def test_selected_segment_plot_does_not_connect_across_subblock_gaps() -> None:
    fig = plot_trajectory_review_components(
        _synthetic_review("high_pass"),
        keys=["source.x_position_as"],
    )[0]
    selected_axis = fig.axes[3]
    line_spans = [
        float(np.max(line.get_xdata()) - np.min(line.get_xdata()))
        for line in selected_axis.lines
        if len(line.get_xdata()) > 1
    ]

    assert line_spans
    assert max(line_spans) <= 0.2 + 1.0e-12


def test_timing_summary_reports_first_and_last_frame_times() -> None:
    rows = trajectory_timing_summary_table(_synthetic_review())

    assert rows[0]["subblock_index"] == 0
    assert rows[0]["first_frame_time_s"] == 0.0
    assert rows[0]["last_frame_time_s"] == 0.2
    assert rows[1]["first_frame_time_s"] == 1.0
    assert rows[1]["last_frame_time_s"] == 1.2
    assert rows[0]["fit_model"] == "linear per subblock"


def test_filter_provenance_labels_removed_component_by_filter_kind() -> None:
    expected = {
        "high_pass": "low-frequency component removed",
        "low_pass": "high-frequency residual removed",
        "band_pass": "out-of-band component removed",
    }
    for kind, label in expected.items():
        rows = trajectory_filter_provenance_table(_synthetic_review(kind))

        assert rows
        assert rows[0]["filter_kind"] == kind
        assert rows[0]["removed_component_label"] == label
        assert np.isfinite(rows[0]["raw_rms"])
        assert np.isfinite(rows[0]["filtered_rms"])
        assert np.isfinite(rows[0]["removed_rms"])
