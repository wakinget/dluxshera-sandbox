from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_observation_belief_update_demo.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_observation_belief_update_demo_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_observation_belief_demo_dry_run_plans_without_writing(tmp_path: Path):
    module = _load_script_module()

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "dry_run_case",
            "--n-subblocks",
            "3",
            "--seed",
            "7",
            "--zernike-indices",
            "0,1",
            "--dry-run",
        ]
    )

    assert result["dry_run"] is True
    assert result["artifacts"] == {}
    assert not (tmp_path / "dry_run_case").exists()
    assert result["summary"]["n_subblocks"] == 3


def test_observation_belief_demo_writes_required_artifacts(tmp_path: Path):
    module = _load_script_module()

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "artifact_case",
            "--n-subblocks",
            "4",
            "--seed",
            "11",
            "--zernike-indices",
            "0,1",
        ]
    )

    assert result["dry_run"] is False
    run_dir = tmp_path / "artifact_case"
    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    for key in (
        "observation_update_summary_json",
        "posterior_table_csv",
        "eigenmode_table_csv",
        "cumulative_update_table_csv",
    ):
        assert key in artifacts
        assert artifacts[key].exists()
        assert artifacts[key].stat().st_size > 0

    summary = json.loads(
        artifacts["observation_update_summary_json"].read_text(encoding="utf-8")
    )
    posterior_rows = _read_csv_rows(artifacts["posterior_table_csv"])
    eigen_rows = _read_csv_rows(artifacts["eigenmode_table_csv"])
    cumulative_rows = _read_csv_rows(artifacts["cumulative_update_table_csv"])

    assert summary["update"]["n_summaries"] == 4
    assert len(posterior_rows) == len(summary["theta_layout"]["labels"])
    assert len(eigen_rows) == len(summary["theta_layout"]["labels"])
    assert len(cumulative_rows) == 4
    assert summary["eigenbasis"]["weak_mode_count"] >= 1
    assert (run_dir / "synthetic_subblock_summaries").is_dir()
