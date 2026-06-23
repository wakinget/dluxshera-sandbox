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
SKELETON_PATH = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_algorithm_campaign_v1.yaml"
)
DAMPED_CONFIG_PATH = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_zernike_2x2_self_correction_hpc_v1_eigen_bottom_damped.yaml"
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
    assert experiment["source_campaign_kind"] == "full_fidelity_binary_iterative"
    assert "source_campaign_alias" not in experiment
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


def test_config_translation_preserves_iterative_eigenbasis_policy() -> None:
    module = load_module()
    raw = module.load_config_file(CONFIG_PATH)
    raw["experiment"]["iterative"]["update_mode"] = "eigen_truncated"
    raw["experiment"]["iterative"]["update_gain"] = 0.25
    raw["experiment"]["iterative"]["eigenbasis"] = {
        "basis_source": "posterior_precision",
        "gate_source": "accumulated_information",
        "whiten": True,
        "eig_floor_rel": 1.0e-8,
        "min_kept_modes": 1,
        "damping_mode": "bottom_n",
        "damping_n_modes": 8,
        "damping_value": 0.1,
    }

    translated = module._full_fidelity_to_observation_bias(raw, run_name="unit")
    iterative = translated["experiment"]["iterative"]

    assert iterative["update_mode"] == "eigen_truncated"
    assert iterative["update_gain"] == 0.25
    assert iterative["eigenbasis"]["min_kept_modes"] == 1
    assert iterative["eigenbasis"]["damping_mode"] == "bottom_n"
    assert iterative["eigenbasis"]["damping_n_modes"] == 8
    assert iterative["eigenbasis"]["damping_value"] == 0.1


def test_config_translation_forwards_detector_calibration_knowledge_error() -> None:
    module = load_module()
    raw = module.load_config_file(CONFIG_PATH)
    raw["experiment"]["detector_calibration_knowledge_error"] = {
        "enabled": True,
        "apply_to": "inference",
        "realization_policy": "fixed_per_experiment",
        "pixel_offsets": {"enabled": True, "sigma_pix": 0.001},
        "pixel_response": {"enabled": True, "sigma_fractional": 0.001},
    }

    translated = module._full_fidelity_to_observation_bias(raw, run_name="unit")

    detector_ke = translated["experiment"]["detector_calibration_knowledge_error"]
    assert detector_ke["enabled"] is True
    assert detector_ke["apply_to"] == "inference"
    assert detector_ke["pixel_offsets"]["sigma_pix"] == 0.001
    assert detector_ke["pixel_response"]["sigma_fractional"] == 0.001


def test_damped_hpc_config_translates_bottom_n_policy() -> None:
    module = load_module()
    raw = module.load_config_file(DAMPED_CONFIG_PATH)

    translated = module._full_fidelity_to_observation_bias(raw, run_name="unit")
    iterative = translated["experiment"]["iterative"]

    assert iterative["update_mode"] == "eigen_damped"
    assert iterative["eigenbasis"]["damping_mode"] == "bottom_n"
    assert iterative["eigenbasis"]["damping_n_modes"] == 8
    assert iterative["eigenbasis"]["damping_value"] == 0.1
    assert iterative["eigenbasis"]["eig_floor_abs"] == 0.0
    assert iterative["eigenbasis"]["eig_floor_rel"] == 0.0
    assert iterative["eigenbasis"]["min_kept_modes"] is None
    assert iterative["eigenbasis"]["max_kept_modes"] is None


def test_config_translation_rejects_unsupported_kind() -> None:
    module = load_module()
    try:
        module._full_fidelity_to_observation_bias({"experiment": {"kind": "bad"}}, run_name=None)
    except ValueError as exc:
        assert "full_fidelity_binary_iterative" in str(exc)
    else:
        raise AssertionError("unsupported kind was accepted")


def test_config_translation_rejects_future_skeleton_helpfully() -> None:
    module = load_module()
    raw = module.load_config_file(SKELETON_PATH)

    try:
        module._full_fidelity_to_observation_bias(raw, run_name=None)
    except ValueError as exc:
        text = str(exc)
        assert "full_fidelity_algorithm_campaign" in text
        assert "non-executable" in text
        assert "full_fidelity_binary_iterative" in text
    else:
        raise AssertionError("future skeleton was accepted")


def test_wrapper_validation_smoke_has_no_hidden_spectral_fast() -> None:
    module = load_module()
    raw = module.load_config_file(CONFIG_PATH)

    warnings = module.validate_full_fidelity_smoke_config(raw)

    assert "fast" not in raw["experiment"]["spectral_model"]
    assert not any("spectral_model.fast" in warning for warning in warnings)


def test_wrapper_validation_warns_about_synthetic_spectral_fast() -> None:
    module = load_module()
    raw = module.load_config_file(CONFIG_PATH)
    raw["experiment"]["spectral_model"]["fast"] = True

    warnings = module.validate_full_fidelity_smoke_config(raw)

    assert any("spectral_model.fast" in warning for warning in warnings)
    assert any("truth<=7" in warning and "inference<=5" in warning for warning in warnings)


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
    assert "full_fidelity_algorithm_campaign" in completed.stdout
