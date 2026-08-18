from __future__ import annotations

import importlib.util
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
        "run_full_fidelity_binary_iterative_preset_translation_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_fidelity_translation_writes_observation_bias_preset_locations() -> None:
    module = _load_wrapper()
    raw = module.load_config_file(CONFIG)

    assert raw["experiment"]["system_preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET

    translated = module._full_fidelity_to_observation_bias(raw, run_name="preset_translation")
    experiment = translated["experiment"]

    assert translated["system"]["preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert experiment["system_preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert experiment["system"]["preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert experiment["source_campaign_kind"] == "full_fidelity_binary_iterative"


def test_full_fidelity_wrapper_does_not_literal_default_to_legacy_preset() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'DEFAULT_FULL_FIDELITY_SYSTEM_PRESET' in text
    assert '"SHERA_FLIGHT_3P"' not in text
