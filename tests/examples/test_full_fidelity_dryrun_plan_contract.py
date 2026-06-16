from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from dluxshera.utils.full_fidelity_defaults import DEFAULT_FULL_FIDELITY_SYSTEM_PRESET


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
CONFIG = (
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
        "run_full_fidelity_binary_iterative_dryrun_contract_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_fidelity_smoke_dryrun_uses_conv_preset_and_smear_layer(tmp_path: Path) -> None:
    module = _load_wrapper()
    status = module.run_full_fidelity_binary_iterative_campaign(
        config_path=CONFIG,
        results_root=tmp_path,
        run_name="preset_contract_dryrun",
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
    system_meta = plan["layout_metadata"]["system"]
    resolved_system = plan["layout_metadata"]["resolved_system"]
    layers = resolved_system["detector"]["layers"]
    names = [layer.get("name") for layer in layers]
    flags = plan["subblock_command_options"]["forwarded_flags"]

    assert system_meta["system_preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert resolved_system["preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert "smear" in names
    assert flags.count("--reference-early-stopping") == 1

    render_templates = sorted((run_root / "trajectory").glob("subblock_*/templates/render_template.json"))
    inference_templates = sorted((run_root / "trajectory").glob("subblock_*/templates/inference_template.json"))
    assert render_templates
    assert len(render_templates) == len(inference_templates)

    render = json.loads(render_templates[0].read_text(encoding="utf-8"))
    render_system = render.get("system", render.get("instrument", {}))
    render_layers = render_system["detector"]["layers"]
    render_names = [layer.get("name") for layer in render_layers]
    assert "smear" in render_names
