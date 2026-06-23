from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_full_fidelity_binary_iterative_campaign.py"
)
CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)
PROJECTED_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_info_damped_detector_ke_projected_30min_v1.yaml"
)


def load_module() -> Any:
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_full_fidelity_binary_iterative_campaign", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_full_fidelity_binary_iterative_smoke_dry_run_writes_split_plans(tmp_path: Path) -> None:
    module = load_module()
    status = module.run_full_fidelity_binary_iterative_campaign(
        config_path=CONFIG_PATH,
        results_root=tmp_path,
        run_name="full_fidelity_unit_dryrun",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )
    run_root = Path(status["run_root"])
    assert (run_root / "campaign_plan.json").is_file()
    assert (run_root / "resolved_config.json").is_file()
    assert (run_root / "model_split.json").is_file()
    assert (run_root / "model_split_summary.json").is_file()
    assert (run_root / "model_split" / "model_split.json").is_file()
    assert (run_root / "model_split" / "model_split_summary.json").is_file()
    assert (run_root / "subblock_plan.csv").is_file()
    assert (run_root / "expected_outputs.csv").is_file()
    assert (run_root / "iterative_plan.csv").is_file()
    assert (run_root / "template_hashes.csv").is_file()

    plan = json.loads((run_root / "campaign_plan.json").read_text(encoding="utf-8"))
    split = plan["model_split"]
    assert split["schema_version"] == "campaign_model_split.v1"
    assert split["truth_config_hash"] != split["inference_config_hash"]
    assert split["components"]["spectral_model"]["enabled"] is True
    assert split["components"]["high_order_wfe"]["enabled"] is True
    assert split["components"]["trajectory_smear"]["mode"] == "subblock_constant_layer"
    assert (run_root / "trajectory" / "subblock_000000" / "templates" / "render_template.json").is_file()
    assert (run_root / "trajectory" / "subblock_000000" / "templates" / "inference_template.json").is_file()
    render_template = json.loads(
        (run_root / "trajectory" / "subblock_000000" / "templates" / "render_template.json").read_text(
            encoding="utf-8"
        )
    )
    layers = render_template["system"]["detector"]["layers"]
    smear = next(layer for layer in layers if layer.get("name") == "smear")
    assert smear["kernel"]["kind"] == "line"
    assert smear["kernel"]["units"] == "detector_pix"

    rows = _read_csv(run_root / "iterative_plan.csv")
    resolved_cfg = json.loads((run_root / "resolved_config.json").read_text(encoding="utf-8"))
    resolved = resolved_cfg["experiment"]["subblock_resolution"]
    assert len(rows) == int(resolved["resolved_total_subblocks"])
    assert all(row["trace_template_hash"] for row in rows)
    assert all(row["model_split_json"] for row in rows)
    assert all("posterior_sigma_inflation" in row["update_safety_json"] for row in rows)


def test_projected_30min_config_dry_run_plans_actual_windows_only(tmp_path: Path) -> None:
    module = load_module()
    status = module.run_full_fidelity_binary_iterative_campaign(
        config_path=PROJECTED_CONFIG_PATH,
        results_root=tmp_path,
        run_name="projected_30min_unit_dryrun",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )
    run_root = Path(status["run_root"])
    plan = json.loads((run_root / "campaign_plan.json").read_text(encoding="utf-8"))
    resolved = json.loads((run_root / "resolved_config.json").read_text(encoding="utf-8"))
    rows = _read_csv(run_root / "iterative_plan.csv")

    assert len(plan["bias_cases"]) == 6
    assert len(rows) == 6 * 10 * 30
    assert plan["iterative"]["windows_per_draw"] == 10
    assert plan["iterative"]["subblocks_per_window"] == 30
    forecast = plan["iterative_forecast"]
    assert forecast["enabled"] is True
    assert forecast["rendered_subblocks_total"] == 1800
    assert forecast["projected_endpoint_subblocks_per_case"] == 1800
    assert forecast["actual_windows"] == 10
    assert forecast["projected_windows"] == 60
    assert resolved["experiment"]["subblocks"]["phi_ref"] == "truth_when_available"
    assert resolved["experiment"]["detector_calibration_knowledge_error"]["apply_to"] == "inference"
    assert resolved["experiment"]["detector_calibration_knowledge_error"]["pixel_offsets"]["sigma_pix"] == 0.001
    assert resolved["experiment"]["detector_calibration_knowledge_error"]["pixel_response"]["sigma_fractional"] == 0.001
    assert resolved["experiment"]["prior_draws"]["sigmas"]["source.separation_as"]["sigma"] == 1.0e-04
