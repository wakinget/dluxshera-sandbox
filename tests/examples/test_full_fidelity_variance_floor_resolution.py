from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from dluxshera.config.io import load_config_file


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_observation_bias_campaign.py"
AUDIT = ROOT / "examples" / "scripts" / "audit_full_fidelity_config.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


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


def test_conflicting_variance_floor_aliases_fail_strict_audit(tmp_path: Path) -> None:
    audit = _module_from(AUDIT, "audit_full_fidelity_config_variance_test")
    cfg = load_config_file(CONFIG)
    cfg["experiment"]["subblocks"]["variance_floor"] = 99.0
    path = tmp_path / "conflict.yaml"
    import yaml

    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="deprecated and disagrees"):
        audit.build_audit(path, tmp_path / "audit", strict=True)


def _module_from(path: Path, name: str):
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
