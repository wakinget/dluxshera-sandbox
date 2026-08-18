from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_single_star_calibration_demo.py"
BIAS_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_observation_bias_campaign.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_airbus_fixture(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "0.0,1.0,2.0,3600.0",
                "0.1,1.1,2.2,7200.0",
                "0.2,1.2,2.4,10800.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_single_star_dry_run_trajectory_trace_source_xy_only(tmp_path):
    module = _load_module(SINGLE_SCRIPT, "single_star_trajectory_wrapper_test")
    airbus = _write_airbus_fixture(tmp_path / "airbus.csv")

    module.main(
        [
            "--results-root",
            str(tmp_path / "results"),
            "--run-name",
            "single",
            "--trace-source-mode",
            "trajectory",
            "--trajectory-csv",
            str(airbus),
            "--trajectory-start-s",
            "0.0",
            "--trajectory-n-subblocks",
            "1",
            "--trajectory-frame-dt-s",
            "0.05",
            "--n-subblocks",
            "1",
            "--n-frames",
            "2",
            "--dry-run",
            "--quiet",
        ]
    )

    run_root = tmp_path / "results" / "single"
    plan = json.loads((run_root / "campaign_plan.json").read_text(encoding="utf-8"))
    rows = _read_csv(run_root / "subblock_plan.csv")
    assert plan["trace_source"]["mode"] == "trajectory"
    assert plan["active_frame_keys"] == ["source.x_position_as", "source.y_position_as"]
    assert rows
    assert rows[0]["trace_source_mode"] == "trajectory"
    assert Path(rows[0]["frame_truth_path"]).exists()
    assert Path(rows[0]["starting_guess_prediction_path"]).exists()
    assert "--external-frame-truth-csv" in rows[0]["command"]
    assert "--starting-guess-mode starting_guess_csv" in rows[0]["command"]
    assert "source.position_angle_deg" not in Path(rows[0]["frame_truth_path"]).read_text(
        encoding="utf-8"
    )


def test_observation_bias_dry_run_trajectory_trace_source_xy_pa(tmp_path):
    module = _load_module(BIAS_SCRIPT, "bias_trajectory_wrapper_test")
    airbus = _write_airbus_fixture(tmp_path / "airbus.csv")

    module.main(
        [
            "--results-root",
            str(tmp_path / "results"),
            "--run-name",
            "bias",
            "--trace-source-mode",
            "trajectory",
            "--trajectory-csv",
            str(airbus),
            "--trajectory-start-s",
            "0.0",
            "--trajectory-n-subblocks",
            "1",
            "--trajectory-frame-dt-s",
            "0.05",
            "--n-subblocks",
            "1",
            "--n-frames",
            "2",
            "--dry-run",
            "--quiet",
        ]
    )

    run_root = tmp_path / "results" / "bias"
    plan = json.loads((run_root / "campaign_plan.json").read_text(encoding="utf-8"))
    rows = _read_csv(run_root / "subblock_plan.csv")
    assert plan["trace_source"]["trajectory_window_policy"] == "shared_across_cases"
    assert plan["trace_source"]["active_frame_keys"] == [
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    ]
    assert rows[0]["trace_source_mode"] == "trajectory"
    assert "rms_source.position_angle_deg_residual" in rows[0]
    assert "--external-frame-truth-csv" in rows[0]["command"]
    assert "--starting-guess-csv" in rows[0]["command"]
    assert rows[0]["summary_information_scale"] == "summed_likelihood"
