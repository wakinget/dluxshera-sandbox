from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_registration_solve_smoke_hpc_v1.yaml"
)
SMOKE = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


def _load_wrapper():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_full_fidelity_registration_solve_smoke_config_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_registration_solve_config_translates_and_preserves_fidelity_decks() -> None:
    module = _load_wrapper()
    cfg = _yaml(CONFIG)
    smoke = _yaml(SMOKE)
    translated = module._full_fidelity_to_observation_bias(cfg, run_name="unit_registration")
    experiment = translated["experiment"]

    assert cfg["experiment"]["subblocks"]["phi_ref"] == "recovered"
    assert experiment["subblocks"]["phi_ref"] == "recovered"
    assert cfg["experiment"]["system_preset"] == "SHERA_FLIGHT_3P_CONV"
    for key in (
        "spectral_model",
        "high_order_wfe",
        "detector_overrides",
    ):
        assert cfg["experiment"][key] == smoke["experiment"][key]
    for key in ("noise", "trace_source", "trajectory_processing", "trace_jitter"):
        assert cfg["experiment"]["subblocks"][key] == smoke["experiment"]["subblocks"][key]


def test_registration_solve_dryrun_has_12_subblocks_and_recovered_phi(tmp_path: Path) -> None:
    module = _load_wrapper()
    status = module.run_full_fidelity_binary_iterative_campaign(
        config_path=CONFIG,
        results_root=tmp_path,
        run_name="registration_solve_dryrun",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )
    plan = json.loads((Path(status["run_root"]) / "campaign_plan.json").read_text(encoding="utf-8"))
    commands = [command for command_list in plan["subblock_commands"].values() for command in command_list]
    subblock_rows = [row for rows in plan["subblock_plan"].values() for row in rows]

    assert plan["smear_audit"]["n_subblocks"] == 6
    assert len(commands) == 12
    assert all("--phi-ref recovered" in command for command in commands)
    assert all("--phi-ref truth_when_available" not in command for command in commands)
    assert plan["subblock_command_options"]["forwarded_flags"]
    assert {row["phi_ref"] for row in subblock_rows} == {"recovered"}
