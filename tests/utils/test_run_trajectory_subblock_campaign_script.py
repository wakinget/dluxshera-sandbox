from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_trajectory_subblock_campaign.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_trajectory_subblock_campaign_script_tests",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_script_dry_run_writes_plan_and_subblock_artifacts(tmp_path):
    module = _load_script_module()
    airbus = tmp_path / "airbus.csv"
    airbus.write_text(
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

    plan = module.main(
        [
            "--run-name",
            "tiny",
            "--results-root",
            str(tmp_path / "results"),
            "--trajectory-csv",
            str(airbus),
            "--duration-s",
            "0.1",
            "--subblock-duration-s",
            "0.1",
            "--frame-dt-s",
            "0.05",
            "--n-frames-per-subblock",
            "2",
            "--dry-run",
        ]
    )

    run_root = Path(plan["run_root"])
    assert (run_root / "campaign_plan.json").exists()
    assert (run_root / "trajectory_ingest_summary.json").exists()
    assert (run_root / "subblock_plan.csv").exists()
    subblock = run_root / "subblocks" / "subblock_000000"
    assert (subblock / "frame_truth.csv").exists()
    assert (subblock / "starting_guess_prediction.csv").exists()
    assert (subblock / "render_config.json").exists()
    assert (subblock / "inference_config.json").exists()
    assert (subblock / "command.sh").exists()

    campaign_plan = json.loads((run_root / "campaign_plan.json").read_text(encoding="utf-8"))
    assert campaign_plan["n_subblocks"] == 1
    assert campaign_plan["child_results"] == []
    assert "source.position_angle_deg" in campaign_plan["active_frame_keys"]
