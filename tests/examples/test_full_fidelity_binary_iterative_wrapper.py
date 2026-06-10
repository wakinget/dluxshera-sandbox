from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
CONFIG_PATH = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


def load_module() -> Any:
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_full_fidelity_binary_iterative_campaign",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_config_translation_accepts_full_fidelity_smoke() -> None:
    module = load_module()
    raw = module.load_config_file(CONFIG_PATH)
    translated = module._full_fidelity_to_observation_bias(raw, run_name="unit")
    experiment = translated["experiment"]

    assert experiment["kind"] == "observation_bias_campaign"
    assert experiment["source_campaign_kind"] == "full_fidelity_binary_iterative_smoke"
    assert experiment["system"]["source"]["kind"]
    assert experiment["system"]["source"]["target"]
    for key in (
        "spectral_model",
        "high_order_wfe",
        "subblocks",
        "iterative",
        "observation_theta",
        "full_fidelity_smoke_contract",
    ):
        assert key in experiment


def test_config_translation_rejects_unsupported_kind() -> None:
    module = load_module()
    try:
        module._full_fidelity_to_observation_bias({"experiment": {"kind": "bad"}}, run_name=None)
    except ValueError as exc:
        assert "full_fidelity_binary_iterative_smoke" in str(exc)
    else:
        raise AssertionError("unsupported kind was accepted")


def test_wrapper_help_works() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--aggregate-only" in completed.stdout
