from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

from dluxshera.utils.full_fidelity_defaults import DEFAULT_FULL_FIDELITY_SYSTEM_PRESET


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"


def _load_wrapper():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_full_fidelity_binary_iterative_campaign_migration_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_fidelity_wrapper_uses_shared_default_when_config_omits_preset() -> None:
    module = _load_wrapper()
    cfg = {
        "experiment": {
            "kind": "full_fidelity_binary_iterative_smoke",
            "source_kind": "binary_target",
            "target": "ALPHA_CEN",
            "subblocks": {},
        }
    }
    translated = module._full_fidelity_to_observation_bias(cfg, run_name=None)
    assert translated["experiment"]["system"]["preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET


def test_full_fidelity_wrapper_forwards_detector_overrides() -> None:
    module = _load_wrapper()
    recipe = yaml.safe_load(
        (ROOT / "examples" / "recipes" / "full_fidelity_algorithm_campaign_template" / "full_fidelity_binary_iterative_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    translated = module._full_fidelity_to_observation_bias(recipe, run_name="migration_test")
    assert translated["experiment"]["system"]["preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert translated["experiment"]["detector_overrides"]["layers"]["jitter"]["action"] == "update"
