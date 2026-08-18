from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dluxshera.utils.full_fidelity_defaults import DEFAULT_FULL_FIDELITY_SYSTEM_PRESET


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_observation_bias_campaign.py"


def _load_observation_bias():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_observation_bias_preset_resolution_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_plain_observation_bias_without_preset_uses_legacy_default() -> None:
    module = _load_observation_bias()
    config = {"experiment": {"kind": "observation_bias_campaign"}}

    resolved = module.resolve_campaign_system_preset(
        config=config,
        experiment_cfg=config["experiment"],
        strict=True,
    )

    assert resolved["system_preset"] == module.DEFAULT_SYSTEM_PRESET
    assert resolved["system_preset"] != DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert resolved["source"] == "legacy_default"


def test_plain_observation_bias_cli_preset_is_fallback_when_config_omits_preset() -> None:
    module = _load_observation_bias()
    config = {"experiment": {"kind": "observation_bias_campaign"}}

    resolved = module.resolve_campaign_system_preset(
        config=config,
        experiment_cfg=config["experiment"],
        cli_system_preset=DEFAULT_FULL_FIDELITY_SYSTEM_PRESET,
        strict=True,
    )

    assert resolved["system_preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert resolved["source"] == "cli.system_preset"


def test_full_fidelity_translated_config_resolves_conv_preset() -> None:
    module = _load_observation_bias()
    config = {
        "system": {"preset": DEFAULT_FULL_FIDELITY_SYSTEM_PRESET},
        "experiment": {
            "kind": "observation_bias_campaign",
            "source_campaign_kind": "full_fidelity_binary_iterative",
            "system_preset": DEFAULT_FULL_FIDELITY_SYSTEM_PRESET,
            "system": {"preset": DEFAULT_FULL_FIDELITY_SYSTEM_PRESET},
        },
    }

    resolved = module.resolve_campaign_system_preset(
        config=config,
        experiment_cfg=config["experiment"],
        strict=True,
    )

    assert resolved["system_preset"] == DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
    assert resolved["source"] == "top_level_system.preset"


def test_full_fidelity_translated_config_conflict_fails_strict() -> None:
    module = _load_observation_bias()
    config = {
        "system": {"preset": DEFAULT_FULL_FIDELITY_SYSTEM_PRESET},
        "experiment": {
            "kind": "observation_bias_campaign",
            "source_campaign_kind": "full_fidelity_binary_iterative",
            "system_preset": module.DEFAULT_SYSTEM_PRESET,
        },
    }

    with pytest.raises(ValueError, match="Conflicting system preset"):
        module.resolve_campaign_system_preset(
            config=config,
            experiment_cfg=config["experiment"],
            strict=True,
        )
