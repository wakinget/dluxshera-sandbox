from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_observation_bias_campaign.py"


def _module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_observation_bias_campaign_seed_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_experiment_seed_is_default_campaign_base_seed() -> None:
    module = _module()

    resolved = module._resolve_seeding_config({"seed": 123})

    assert resolved["base_seed"] == 123
    assert resolved["seed_policy"] == "different_jitter_different_noise"


def test_explicit_base_seed_overrides_experiment_seed() -> None:
    module = _module()

    resolved = module._resolve_seeding_config(
        {"seed": 123, "seeding": {"base_seed": 999, "seed_policy": "same_jitter_different_noise"}}
    )

    assert resolved["base_seed"] == 999
    assert resolved["seed_policy"] == "same_jitter_different_noise"
