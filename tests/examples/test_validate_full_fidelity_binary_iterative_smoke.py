from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "scripts" / "validate_full_fidelity_binary_iterative_smoke.py"
CONFIG_PATH = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


def load_module() -> Any:
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "validate_full_fidelity_binary_iterative_smoke",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_validation_can_run_in_temp_results_root(tmp_path: Path) -> None:
    module = load_module()
    report = module.run_validation(
        config_path=CONFIG_PATH,
        results_root=tmp_path,
        run_name="validator_unit_dryrun",
        stage="dry-run",
        max_workers=1,
    )

    run_root = tmp_path / "validator_unit_dryrun"
    assert report["stages"]["dry_run"]["status"] == "passed"
    assert (run_root / "campaign_plan.json").is_file()
    assert (run_root / "validation_report.json").is_file()
    assert (run_root / "validation_report.md").is_file()


def test_validation_reports_missing_artifact_failures_clearly(tmp_path: Path) -> None:
    module = load_module()
    module.run_validation(
        config_path=CONFIG_PATH,
        results_root=tmp_path,
        run_name="validator_unit_missing",
        stage="dry-run",
        max_workers=1,
    )
    run_root = tmp_path / "validator_unit_missing"
    (run_root / "model_split_summary.json").unlink()

    stage = module.validate_dry_run_artifacts(
        run_root=run_root,
        config_path=CONFIG_PATH,
        expected_run_name="validator_unit_missing",
    )

    assert stage["status"] == "failed"
    assert any(
        failure["name"] == "artifact_exists:model_split_summary.json"
        for failure in stage["failures"]
    )


def test_aggregate_only_validation_detects_stored_plan_mismatch(tmp_path: Path) -> None:
    module = load_module()
    module.run_validation(
        config_path=CONFIG_PATH,
        results_root=tmp_path,
        run_name="validator_unit_mismatch",
        stage="dry-run",
        max_workers=1,
    )
    raw = module._load_wrapper_module().load_config_file(CONFIG_PATH)
    raw["experiment"]["n_cases"] = 2
    raw["experiment"]["prior_draws"]["n_cases"] = 2
    mismatched_config = tmp_path / "mismatched.yaml"
    mismatched_config.write_text(json.dumps(raw), encoding="utf-8")

    stage = module.run_aggregate_only_validation(
        config_path=mismatched_config,
        results_root=tmp_path,
        run_name="validator_unit_mismatch",
    )

    assert stage["status"] == "failed"
    assert stage["command_returncode"] != 0
    assert any(
        mismatch.get("field") in {"case_set", "expected_outputs_count", "iterative_plan_count"}
        for mismatch in stage.get("mismatches", [])
    )
