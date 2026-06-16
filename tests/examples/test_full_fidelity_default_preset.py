from __future__ import annotations

from pathlib import Path

import yaml

from dluxshera.utils.full_fidelity_defaults import DEFAULT_FULL_FIDELITY_SYSTEM_PRESET


ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = ROOT / "examples" / "recipes" / "full_fidelity_algorithm_campaign_template"


def _experiment(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["experiment"]


def test_full_fidelity_review_config_uses_conv_preset() -> None:
    exp = _experiment(RECIPE_ROOT / "full_fidelity_binary_iterative_review.yaml")
    assert exp["system_preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET


def test_full_fidelity_smoke_config_uses_conv_preset() -> None:
    exp = _experiment(RECIPE_ROOT / "full_fidelity_binary_iterative_smoke.yaml")
    assert exp["system_preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    render_smear = exp["subblocks"]["trajectory_processing"]["smear"]["render"]
    assert render_smear["mode"] == "subblock_constant_layer"
    assert render_smear["target_layer"] == "smear"
    assert render_smear["require_existing_layer"] is True
    assert render_smear["allow_layer_injection"] is False
