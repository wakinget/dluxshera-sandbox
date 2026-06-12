from __future__ import annotations

import json
from pathlib import Path

from dluxshera.utils.campaign_trace_sources import prepare_campaign_trace_source


def _write_airbus_fixture(path: Path) -> Path:
    rows = []
    for index in range(400):
        time_s = 0.1 * index
        slow = 10.0 + 0.05 * time_s
        fast = 0.2 * ((-1.0) ** index)
        rows.append(f"{time_s:.3f},{slow + fast:.8f},{2.0 * slow - fast:.8f},{3600.0 + slow:.8f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_trajectory_campaign_filter_writes_provenance_and_csv_source_schema_works(tmp_path):
    source_path = _write_airbus_fixture(tmp_path / "airbus.csv")

    plan = prepare_campaign_trace_source(
        trace_source_cfg={
            "mode": "trajectory",
            "source": {
                "kind": "csv",
                "format": "airbus_xyz_arcsec",
                "path": str(source_path),
            },
            "window": {"start_s": 1.0, "n_subblocks": 2},
            "sampling": {
                "frame_dt_s": 0.05,
                "subblock_duration_s": 1.0,
                "n_frames_per_subblock": 20,
            },
            "processing": {
                "filter": {
                    "enabled": True,
                    "kind": "high_pass",
                    "method": "bessel",
                    "order": 4,
                    "cutoff_period_s": 15.0,
                    "zero_phase": True,
                    "columns": ["source.x_position_as", "source.y_position_as"],
                }
            },
            "output_keys": ["source.x_position_as", "source.y_position_as"],
        },
        run_root=tmp_path,
        source_kind="single_star",
        active_frame_keys=("source.x_position_as", "source.y_position_as"),
        n_subblocks=2,
        n_frames_per_subblock=20,
        frame_dt_s=0.05,
        subblock_duration_s=1.0,
        default_output_keys=("source.x_position_as", "source.y_position_as"),
    )

    assert plan.summary["trajectory_source_kind"] == "csv"
    assert plan.summary["trajectory_source_format"] == "airbus_xyz_arcsec"
    assert plan.summary["filter"]["enabled"] is True
    provenance_path = Path(plan.summary["filter"]["provenance_json"])
    filtered_path = tmp_path / "trajectory" / "trajectory_filtered.csv"
    raw_path = tmp_path / "trajectory" / "trajectory_raw.csv"
    assert provenance_path.exists()
    assert filtered_path.exists()
    assert raw_path.exists()
    assert Path(plan.rows[0]["frame_truth_path"]).exists()
    assert plan.rows[0]["trajectory_filter_enabled"] is True
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["kind"] == "high_pass"
    assert provenance["cutoff_period_s"] == 15.0
    assert "source.x_position_as" in provenance["removed_rms_by_column"]


def test_legacy_airbus_csv_source_kind_still_works(tmp_path):
    plan = prepare_campaign_trace_source(
        trace_source_cfg={
            "mode": "trajectory",
            "source": {"kind": "airbus_csv", "path": str(_write_airbus_fixture(tmp_path / "airbus.csv"))},
            "window": {"start_s": 0.0, "n_subblocks": 1},
            "sampling": {
                "frame_dt_s": 0.05,
                "subblock_duration_s": 1.0,
                "n_frames_per_subblock": 20,
            },
            "output_keys": ["source.x_position_as", "source.y_position_as"],
        },
        run_root=tmp_path / "legacy",
        source_kind="single_star",
        active_frame_keys=("source.x_position_as", "source.y_position_as"),
        n_subblocks=1,
        n_frames_per_subblock=20,
        frame_dt_s=0.05,
        subblock_duration_s=1.0,
        default_output_keys=("source.x_position_as", "source.y_position_as"),
    )

    assert plan.summary["trajectory_source_kind"] == "airbus_csv"
    assert Path(plan.rows[0]["frame_truth_path"]).exists()
