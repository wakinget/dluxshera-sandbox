from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from dluxshera.utils.campaign_trace_sources import prepare_campaign_trace_source


def _write_airbus_fixture(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "0.0,1.0,2.0,3600.0",
                "0.1,1.1,2.2,7200.0",
                "0.2,1.2,2.4,10800.0",
                "0.3,1.3,2.6,14400.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_iid_trace_source_plan_has_no_external_artifacts(tmp_path):
    plan = prepare_campaign_trace_source(
        trace_source_cfg={"mode": "iid_jitter"},
        run_root=tmp_path,
        source_kind="binary",
        active_frame_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
        n_subblocks=2,
        n_frames_per_subblock=3,
        frame_dt_s=0.05,
        subblock_duration_s=1.0,
        default_output_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
    )

    assert plan.mode == "iid_jitter"
    assert len(plan.rows) == 2
    assert plan.rows[0]["frame_truth_path"] == ""
    assert not (tmp_path / "trajectory").exists()


def test_trajectory_trace_source_binary_writes_xy_pa_artifacts(tmp_path):
    plan = prepare_campaign_trace_source(
        trace_source_cfg={
            "mode": "trajectory",
            "source": {"kind": "airbus_csv", "path": str(_write_airbus_fixture(tmp_path / "airbus.csv"))},
            "window": {"start_s": 0.0, "n_subblocks": 1},
            "sampling": {
                "frame_dt_s": 0.05,
                "subblock_duration_s": 0.1,
                "n_frames_per_subblock": 2,
            },
            "output_keys": [
                "source.x_position_as",
                "source.y_position_as",
                "source.position_angle_deg",
            ],
        },
        run_root=tmp_path,
        source_kind="binary",
        active_frame_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
        n_subblocks=1,
        n_frames_per_subblock=2,
        frame_dt_s=0.05,
        subblock_duration_s=0.1,
        default_output_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
    )

    row = plan.rows[0]
    truth = Path(row["frame_truth_path"])
    guess = Path(row["starting_guess_prediction_path"])
    assert truth.exists()
    assert guess.exists()
    assert "rms_source.position_angle_deg_residual" in row
    assert "source.position_angle_deg" in truth.read_text(encoding="utf-8")


def test_trajectory_trace_source_offsets_write_shifted_artifacts_and_provenance(tmp_path):
    plan = prepare_campaign_trace_source(
        trace_source_cfg={
            "mode": "trajectory",
            "source": {"kind": "airbus_csv", "path": str(_write_airbus_fixture(tmp_path / "airbus.csv"))},
            "processing": {
                "offsets": {
                    "source.x_position_as": 1.0,
                    "source.y_position_as": -1.0,
                    "source.position_angle_deg": 0.5,
                }
            },
            "window": {"start_s": 0.0, "n_subblocks": 1},
            "sampling": {
                "frame_dt_s": 0.05,
                "subblock_duration_s": 0.1,
                "n_frames_per_subblock": 2,
            },
            "output_keys": [
                "source.x_position_as",
                "source.y_position_as",
                "source.position_angle_deg",
            ],
        },
        run_root=tmp_path,
        source_kind="binary",
        active_frame_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
        n_subblocks=1,
        n_frames_per_subblock=2,
        frame_dt_s=0.05,
        subblock_duration_s=0.1,
        default_output_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
    )

    row = plan.rows[0]
    assert row["trajectory_offsets_enabled"] is True
    assert Path(row["trajectory_offset_provenance_json"]).exists()
    truth_rows = list(csv.DictReader(Path(row["frame_truth_path"]).open("r", encoding="utf-8")))
    guess_rows = list(csv.DictReader(Path(row["starting_guess_prediction_path"]).open("r", encoding="utf-8")))
    assert float(truth_rows[0]["source.x_position_as"]) == pytest.approx(2.0)
    assert float(guess_rows[0]["source.x_position_as_linear_fit"]) == pytest.approx(2.0)
    provenance = json.loads(Path(row["trajectory_offset_provenance_json"]).read_text(encoding="utf-8"))
    stats = provenance["statistics"]["source.x_position_as"]
    assert stats["mean_delta"] == pytest.approx(1.0)
    assert stats["residual_std_unchanged"] is True


def test_trajectory_trace_source_single_star_xy_only_does_not_require_pa(tmp_path):
    plan = prepare_campaign_trace_source(
        trace_source_cfg={
            "mode": "trajectory",
            "source": {"kind": "airbus_csv", "path": str(_write_airbus_fixture(tmp_path / "airbus.csv"))},
            "window": {"start_s": 0.0, "n_subblocks": 1},
            "sampling": {
                "frame_dt_s": 0.05,
                "subblock_duration_s": 0.1,
                "n_frames_per_subblock": 2,
            },
            "output_keys": ["source.x_position_as", "source.y_position_as"],
        },
        run_root=tmp_path,
        source_kind="single_star",
        active_frame_keys=("source.x_position_as", "source.y_position_as"),
        n_subblocks=1,
        n_frames_per_subblock=2,
        frame_dt_s=0.05,
        subblock_duration_s=0.1,
        default_output_keys=("source.x_position_as", "source.y_position_as"),
    )

    text = Path(plan.rows[0]["starting_guess_prediction_path"]).read_text(encoding="utf-8")
    assert "source.x_position_as_linear_fit" in text
    assert "source.position_angle_deg" not in text


def test_trajectory_trace_source_smear_start_zero_uses_endpoint_extrapolation(tmp_path):
    plan = prepare_campaign_trace_source(
        trace_source_cfg={
            "mode": "trajectory",
            "source": {"kind": "airbus_csv", "path": str(_write_airbus_fixture(tmp_path / "airbus.csv"))},
            "window": {"start_s": 0.0, "n_subblocks": 1},
            "sampling": {
                "frame_dt_s": 0.05,
                "subblock_duration_s": 0.05,
                "n_frames_per_subblock": 1,
            },
            "output_keys": ["source.x_position_as", "source.y_position_as"],
        },
        run_root=tmp_path,
        source_kind="single_star",
        active_frame_keys=("source.x_position_as", "source.y_position_as"),
        n_subblocks=1,
        n_frames_per_subblock=1,
        frame_dt_s=0.05,
        subblock_duration_s=0.05,
        default_output_keys=("source.x_position_as", "source.y_position_as"),
        trajectory_processing_cfg={
            "smear": {
                "enabled": True,
                "edge_policy": "symmetric_linear_extrapolate",
                "max_extrapolation_s": "auto",
                "exposure": {"time_s": 0.05},
            }
        },
        plate_scale_as_per_pix=0.1,
    )

    row = plan.rows[0]
    assert Path(row["smear_truth_csv"]).exists()
    assert Path(row["smear_model_csv"]).exists()
    provenance = json.loads(Path(row["smear_provenance_json"]).read_text(encoding="utf-8"))
    endpoint = provenance["endpoint_extrapolation"]
    assert endpoint["used"] is True
    assert endpoint["leading_used"] is True
    assert endpoint["exposures"][0]["exposure_start_s"] == pytest.approx(-0.025)
    assert endpoint["exposures"][0]["exposure_mid_s"] == pytest.approx(0.0)


def test_external_plan_reuses_existing_paths_and_fails_when_missing(tmp_path):
    source = prepare_campaign_trace_source(
        trace_source_cfg={
            "mode": "trajectory",
            "source": {"kind": "airbus_csv", "path": str(_write_airbus_fixture(tmp_path / "airbus.csv"))},
            "window": {"start_s": 0.0, "n_subblocks": 1},
            "sampling": {
                "frame_dt_s": 0.05,
                "subblock_duration_s": 0.1,
                "n_frames_per_subblock": 2,
            },
            "output_keys": ["source.x_position_as", "source.y_position_as"],
        },
        run_root=tmp_path / "source",
        source_kind="single_star",
        active_frame_keys=("source.x_position_as", "source.y_position_as"),
        n_subblocks=1,
        n_frames_per_subblock=2,
        frame_dt_s=0.05,
        subblock_duration_s=0.1,
        default_output_keys=("source.x_position_as", "source.y_position_as"),
    )
    campaign_plan = tmp_path / "campaign_plan.json"
    subblock_plan = tmp_path / "subblock_plan.csv"
    campaign_plan.write_text(json.dumps({"trace_source": source.summary}), encoding="utf-8")
    with subblock_plan.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source.rows[0].keys()))
        writer.writeheader()
        writer.writerow(source.rows[0])

    reused = prepare_campaign_trace_source(
        trace_source_cfg={
            "mode": "external_plan",
            "campaign_plan": str(campaign_plan),
            "subblock_plan": str(subblock_plan),
        },
        run_root=tmp_path / "reuse",
        source_kind="single_star",
        active_frame_keys=("source.x_position_as", "source.y_position_as"),
        n_subblocks=1,
        n_frames_per_subblock=2,
        frame_dt_s=0.05,
        subblock_duration_s=0.1,
        default_output_keys=("source.x_position_as", "source.y_position_as"),
    )
    assert reused.mode == "external_plan"
    assert reused.subblocks[0].frame_truth_path.exists()

    Path(source.rows[0]["frame_truth_path"]).unlink()
    with pytest.raises(FileNotFoundError, match="External trace-source artifact missing"):
        prepare_campaign_trace_source(
            trace_source_cfg={
                "mode": "external_plan",
                "campaign_plan": str(campaign_plan),
                "subblock_plan": str(subblock_plan),
            },
            run_root=tmp_path / "reuse2",
            source_kind="single_star",
            active_frame_keys=("source.x_position_as", "source.y_position_as"),
            n_subblocks=1,
            n_frames_per_subblock=2,
            frame_dt_s=0.05,
            subblock_duration_s=0.1,
            default_output_keys=("source.x_position_as", "source.y_position_as"),
        )


def test_trajectory_smear_resume_rehydrates_representative_kernel(tmp_path):
    airbus_path = _write_airbus_fixture(tmp_path / "airbus.csv")
    run_root = tmp_path / "campaign"

    trace_source_cfg = {
        "mode": "trajectory",
        "source": {
            "kind": "airbus_csv",
            "path": str(airbus_path),
        },
        "window": {
            "start_s": 0.0,
            "n_subblocks": 1,
        },
        "sampling": {
            "frame_dt_s": 0.05,
            "subblock_duration_s": 0.1,
            "n_frames_per_subblock": 2,
        },
        "output_keys": [
            "source.x_position_as",
            "source.y_position_as",
        ],
    }

    processing_cfg = {
        "smear": {
            "enabled": True,
            "edge_policy": "symmetric_linear_extrapolate",
            "max_extrapolation_s": "auto",
            "exposure": {
                "time_s": 0.05,
            },
            "inference": {
                "mode": "matched_subblock_constant",
            },
            "render": {
                "mode": "subblock_constant_layer",
            },
        }
    }

    initial = prepare_campaign_trace_source(
        trace_source_cfg=trace_source_cfg,
        run_root=run_root,
        source_kind="single_star",
        active_frame_keys=(
            "source.x_position_as",
            "source.y_position_as",
        ),
        n_subblocks=1,
        n_frames_per_subblock=2,
        frame_dt_s=0.05,
        subblock_duration_s=0.1,
        default_output_keys=(
            "source.x_position_as",
            "source.y_position_as",
        ),
        trajectory_processing_cfg=processing_cfg,
        plate_scale_as_per_pix=0.1,
    )

    initial_row = initial.rows[0]
    initial_kernel = json.loads(
        initial_row["smear_representative_kernel_json"]
    )
    assert initial_kernel["kind"] == "line"

    resumed = prepare_campaign_trace_source(
        trace_source_cfg=trace_source_cfg,
        run_root=run_root,
        source_kind="single_star",
        active_frame_keys=(
            "source.x_position_as",
            "source.y_position_as",
        ),
        n_subblocks=1,
        n_frames_per_subblock=2,
        frame_dt_s=0.05,
        subblock_duration_s=0.1,
        default_output_keys=(
            "source.x_position_as",
            "source.y_position_as",
        ),
        reuse_existing=True,
        trajectory_processing_cfg=processing_cfg,
        plate_scale_as_per_pix=0.1,
    )

    resumed_row = resumed.rows[0]
    resumed_kernel = json.loads(
        resumed_row["smear_representative_kernel_json"]
    )

    assert resumed_kernel == initial_kernel
    assert (
        resumed_row["smear_truth_length_pix_median"]
        == initial_row["smear_truth_length_pix_median"]
    )
    assert (
        resumed_row["smear_model_length_pix_median"]
        == initial_row["smear_model_length_pix_median"]
    )
