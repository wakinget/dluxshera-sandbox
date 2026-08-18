from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_observation_bias_campaign.py"


def _module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_observation_bias_campaign_subblocks_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iterative_enabled_derives_total_subblocks_from_window_grouping() -> None:
    module = _module()

    resolved = module._resolve_total_subblocks_for_campaign(
        {},
        {},
        {"enabled": True, "windows_per_draw": 3, "subblocks_per_window": 5},
    )

    assert resolved["resolved_total_subblocks"] == 15
    assert resolved["subblock_count_source"] == "experiment.iterative.windows_per_draw*subblocks_per_window"


def test_inconsistent_explicit_n_subblocks_fails() -> None:
    module = _module()

    with pytest.raises(ValueError, match="conflicts with iterative grouping"):
        module._resolve_total_subblocks_for_campaign(
            {},
            {"n_subblocks": 4},
            {"enabled": True, "windows_per_draw": 3, "subblocks_per_window": 5},
        )
