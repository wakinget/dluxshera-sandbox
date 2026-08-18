from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"


def _module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_full_fidelity_binary_iterative_kind_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_canonical_kind_translates_with_canonical_source_kind() -> None:
    module = _module()
    translated = module._full_fidelity_to_observation_bias(
        {"experiment": {"kind": "full_fidelity_binary_iterative"}},
        run_name=None,
    )

    assert translated["experiment"]["source_campaign_kind"] == "full_fidelity_binary_iterative"
    assert "source_campaign_alias" not in translated["experiment"]


def test_deprecated_alias_records_alias_but_normalizes_source_kind() -> None:
    module = _module()
    translated = module._full_fidelity_to_observation_bias(
        {"experiment": {"kind": "full_fidelity_binary_iterative_smoke"}},
        run_name=None,
    )

    assert translated["experiment"]["source_campaign_kind"] == "full_fidelity_binary_iterative"
    assert translated["experiment"]["source_campaign_alias"] == "full_fidelity_binary_iterative_smoke"
