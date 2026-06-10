from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dluxshera.utils.obs_subblock_trajectory import (
    CanonicalTrajectory,
    RawTrajectory,
    build_frame_times,
    interpolate_trajectory,
    load_airbus_csv,
    map_trajectory,
    prepare_airbus_subblocks,
    split_subblocks,
    write_subblock_artifacts,
)
from dluxshera.utils.trajectory_smear import (
    derive_model_smear_rows,
    derive_truth_smear_rows,
    parse_smear_config,
    representative_line_kernel,
    write_smear_sidecars,
)


def _write_airbus_fixture(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "100.0,1.0,2.0,3600.0",
                "100.1,1.1,2.2,7200.0",
                "100.2,1.2,2.4,10800.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_airbus_csv_ingest_maps_xy_arcsec_and_z_to_degrees(tmp_path):
    raw = load_airbus_csv(_write_airbus_fixture(tmp_path / "airbus.csv"), sample_dt_s=0.1)
    trajectory = map_trajectory(raw)

    assert raw.sample_count == 3
    assert np.allclose(raw.time_s, [0.0, 0.1, 0.2])
    assert np.allclose(trajectory.values["source.x_position_as"], [1.0, 1.1, 1.2])
    assert np.allclose(trajectory.values["source.y_position_as"], [2.0, 2.2, 2.4])
    assert np.allclose(trajectory.values["source.position_angle_deg"], [1.0, 2.0, 3.0])


def test_interpolation_and_middle_window_values_are_exact_for_linear_series():
    raw = RawTrajectory(
        time_s=np.asarray([0.0, 0.1, 0.2], dtype=float),
        columns={
            "x_as": np.asarray([0.0, 10.0, 20.0], dtype=float),
            "y_as": np.asarray([1.0, 2.0, 3.0], dtype=float),
            "z_as": np.asarray([0.0, 360.0, 720.0], dtype=float),
        },
        source_path=Path("fixture.csv"),
        source_kind="airbus_csv",
    )
    trajectory = map_trajectory(raw)
    values = interpolate_trajectory(trajectory, frame_times_s=[0.05, 0.10, 0.15])

    assert np.allclose(values["source.x_position_as"], [5.0, 10.0, 15.0])
    assert np.allclose(values["source.y_position_as"], [1.5, 2.0, 2.5])
    assert np.allclose(values["source.position_angle_deg"], [0.05, 0.10, 0.15])


def test_subblock_chunking_has_no_duplicate_boundary_frames():
    frame_times = build_frame_times(
        start_s=0.0,
        duration_s=2.0,
        frame_dt_s=0.05,
        n_frames_per_subblock=20,
        subblock_duration_s=1.0,
    )

    assert np.isclose(frame_times[0], 0.0)
    assert np.isclose(frame_times[19], 0.95)
    assert np.isclose(frame_times[20], 1.0)
    assert len(np.unique(frame_times)) == len(frame_times)


def test_linear_starting_guess_residuals_zero_for_linear_and_nonzero_for_curved():
    times = build_frame_times(
        start_s=0.0,
        duration_s=1.0,
        frame_dt_s=0.05,
        n_frames_per_subblock=20,
        subblock_duration_s=1.0,
    )
    linear_blocks = split_subblocks(
        frame_times_s=times,
        truth_values={"source.x_position_as": 2.0 + 3.0 * times},
        n_frames_per_subblock=20,
    )
    curved_blocks = split_subblocks(
        frame_times_s=times,
        truth_values={"source.x_position_as": times * times},
        n_frames_per_subblock=20,
    )

    assert linear_blocks[0].diagnostics["source.x_position_as"]["rms_residual"] < 1.0e-12
    assert curved_blocks[0].diagnostics["source.x_position_as"]["rms_residual"] > 0.0


def test_csv_artifacts_and_xy_only_output_omit_position_angle(tmp_path):
    _, _, blocks = prepare_airbus_subblocks(
        path=_write_airbus_fixture(tmp_path / "airbus.csv"),
        start_s=0.0,
        duration_s=0.1,
        sample_dt_s=0.1,
        frame_dt_s=0.05,
        subblock_duration_s=0.1,
        n_frames_per_subblock=2,
        output_keys=("source.x_position_as", "source.y_position_as"),
    )

    artifacts = write_subblock_artifacts(
        blocks[0],
        outdir=tmp_path / "subblock_000000",
        output_keys=("source.x_position_as", "source.y_position_as"),
    )
    truth_text = artifacts["frame_truth_csv"].read_text(encoding="utf-8")
    guess_text = artifacts["starting_guess_prediction_csv"].read_text(encoding="utf-8")

    assert "source.x_position_as" in truth_text
    assert "source.y_position_as" in guess_text
    assert "source.position_angle_deg" not in truth_text
    assert "source.position_angle_deg" not in guess_text


def _linear_xy_trajectory() -> CanonicalTrajectory:
    raw = RawTrajectory(
        time_s=np.asarray([0.0, 1.0, 2.0], dtype=float),
        columns={
            "x_as": np.asarray([0.0, 2.0, 4.0], dtype=float),
            "y_as": np.asarray([0.0, 1.0, 2.0], dtype=float),
            "z_as": np.asarray([0.0, 0.0, 0.0], dtype=float),
        },
        source_path=Path("linear.csv"),
        source_kind="airbus_csv",
    )
    return map_trajectory(raw)


def test_smear_derivation_from_linear_trajectory_matches_expected_dx_dy_angle():
    trajectory = _linear_xy_trajectory()
    block = split_subblocks(
        frame_times_s=[0.5, 1.0],
        truth_values={
            "source.x_position_as": [1.0, 2.0],
            "source.y_position_as": [0.5, 1.0],
        },
        n_frames_per_subblock=2,
    )[0]
    cfg = parse_smear_config(
        {
            "smear": {
                "enabled": True,
                "exposure": {"time_s": 0.2},
                "truth": {"kernel_size": 9, "sigma_perp_detector_pix": 0.25},
            }
        },
        exposure_time_s=0.2,
        plate_scale_as_per_pix=0.1,
    )

    rows = derive_truth_smear_rows(trajectory, block, cfg)

    assert np.isclose(rows[0]["smear_dx_as"], 0.4)
    assert np.isclose(rows[0]["smear_dy_as"], 0.2)
    assert np.isclose(rows[0]["smear_length_as"], np.hypot(0.4, 0.2))
    assert np.isclose(rows[0]["smear_length_detector_pix"], np.hypot(0.4, 0.2) / 0.1)
    assert np.isclose(rows[0]["smear_theta_deg"], np.degrees(np.arctan2(0.2, 0.4)))


def test_zero_motion_smear_sidecars_are_valid(tmp_path):
    raw = RawTrajectory(
        time_s=np.asarray([0.0, 1.0], dtype=float),
        columns={
            "x_as": np.asarray([1.0, 1.0], dtype=float),
            "y_as": np.asarray([2.0, 2.0], dtype=float),
            "z_as": np.asarray([0.0, 0.0], dtype=float),
        },
        source_path=tmp_path / "zero.csv",
        source_kind="airbus_csv",
    )
    raw.source_path.write_text("0,1,2,0\n1,1,2,0\n", encoding="utf-8")
    trajectory = map_trajectory(raw)
    block = split_subblocks(
        frame_times_s=[0.5],
        truth_values={"source.x_position_as": [1.0], "source.y_position_as": [2.0]},
        n_frames_per_subblock=1,
    )[0]
    cfg = parse_smear_config({"smear": {"enabled": True}}, exposure_time_s=0.2, plate_scale_as_per_pix=0.1)

    artifacts = write_smear_sidecars(outdir=tmp_path / "subblock", trajectory=trajectory, block=block, cfg=cfg)

    assert artifacts["smear_truth_csv"].exists()
    assert artifacts["smear_model_csv"].exists()
    assert artifacts["smear_provenance_json"].exists()
    assert artifacts["smear_truth_length_pix_max"] == 0.0


def test_smear_edge_policy_error_and_kernel_validation():
    cfg = parse_smear_config({"smear": {"enabled": True}}, exposure_time_s=0.2, plate_scale_as_per_pix=0.1)
    trajectory = _linear_xy_trajectory()
    block = split_subblocks(
        frame_times_s=[0.0],
        truth_values={"source.x_position_as": [0.0], "source.y_position_as": [0.0]},
        n_frames_per_subblock=1,
    )[0]

    with pytest.raises(ValueError, match="outside trajectory domain"):
        derive_truth_smear_rows(trajectory, block, cfg)
    with pytest.raises(ValueError, match="positive odd integer"):
        parse_smear_config(
            {"smear": {"enabled": True, "truth": {"kernel_size": 8}}},
            exposure_time_s=0.2,
            plate_scale_as_per_pix=0.1,
        )


def test_model_smear_mismatch_modes_and_representative_are_deterministic():
    trajectory = _linear_xy_trajectory()
    block = split_subblocks(
        frame_times_s=[0.5, 1.0],
        truth_values={
            "source.x_position_as": [1.0, 2.0],
            "source.y_position_as": [0.5, 1.0],
        },
        n_frames_per_subblock=2,
    )[0]
    base_cfg = parse_smear_config({"smear": {"enabled": True}}, exposure_time_s=0.2, plate_scale_as_per_pix=0.1)
    truth = derive_truth_smear_rows(trajectory, block, base_cfg)
    matched = derive_model_smear_rows(truth, base_cfg)
    scaled_cfg = parse_smear_config(
        {"smear": {"enabled": True, "inference": {"mode": "scaled", "length_scale": 0.8}}},
        exposure_time_s=0.2,
        plate_scale_as_per_pix=0.1,
    )
    disabled_cfg = parse_smear_config(
        {"smear": {"enabled": True, "inference": {"mode": "disabled"}}},
        exposure_time_s=0.2,
        plate_scale_as_per_pix=0.1,
    )
    angle_cfg = parse_smear_config(
        {"smear": {"enabled": True, "inference": {"mode": "angle_offset", "theta_offset_deg": 5.0}}},
        exposure_time_s=0.2,
        plate_scale_as_per_pix=0.1,
    )

    assert np.isclose(
        matched[0]["smear_length_detector_pix"],
        truth[0]["smear_length_detector_pix"],
    )
    assert np.isclose(matched[0]["smear_theta_deg"], truth[0]["smear_theta_deg"])
    scaled = derive_model_smear_rows(truth, scaled_cfg)
    assert np.isclose(
        scaled[0]["smear_length_detector_pix"],
        0.8 * truth[0]["smear_length_detector_pix"],
    )
    disabled = derive_model_smear_rows(truth, disabled_cfg)
    assert disabled[0]["smear_enabled"] is False
    angled = derive_model_smear_rows(truth, angle_cfg)
    assert np.isclose(angled[0]["smear_theta_deg"], truth[0]["smear_theta_deg"] + 5.0)
    kernel = representative_line_kernel(truth, representative="median")
    assert kernel["kind"] == "line"
    assert kernel["kernel_size"] == 9
