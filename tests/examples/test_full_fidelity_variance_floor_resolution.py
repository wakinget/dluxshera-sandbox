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
    spec = importlib.util.spec_from_file_location("run_observation_bias_campaign_variance_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_subblocks_noise_variance_floor_is_canonical_forwarded_option() -> None:
    module = _module()

    options = module.resolve_subblock_command_options(
        {"noise": {"enabled": True, "variance_floor": 0.5}}
    )

    assert options["variance_floor"] == 0.5


def test_legacy_variance_floor_still_falls_back() -> None:
    module = _module()

    options = module.resolve_subblock_command_options({"variance_floor": 1.25})

    assert options["variance_floor"] == 1.25
