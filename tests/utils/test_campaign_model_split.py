from __future__ import annotations

import copy
from pathlib import Path

from dluxshera.utils.campaign_model_split import (
    build_campaign_model_split,
    hash_campaign_model_config,
)


BASE_SYSTEM = {
    "source": {
        "kind": "alpha_cen",
        "target": "ALPHA_CEN",
        "wavelength_m": 5.5e-7,
        "bandwidth_m": 4.0e-8,
        "n_lambda": 3,
        "log_flux_total": 0.0,
        "contrast": 0.5,
    },
    "optics": {"kind": "three_plane"},
}


def test_hash_campaign_model_config_is_stable_and_sensitive() -> None:
    cfg = copy.deepcopy(BASE_SYSTEM)
    assert hash_campaign_model_config(cfg) == hash_campaign_model_config(copy.deepcopy(cfg))
    changed = copy.deepcopy(cfg)
    changed["source"]["n_lambda"] = 5
    assert hash_campaign_model_config(cfg) != hash_campaign_model_config(changed)


def test_disabled_components_preserve_existing_configs(tmp_path: Path) -> None:
    split = build_campaign_model_split(
        base_system_cfg=BASE_SYSTEM,
        run_root=tmp_path,
        artifact_root=tmp_path / "model_split",
        seed_context={"test": "disabled"},
        write_artifacts=False,
    )
    assert split.truth_system_cfg == BASE_SYSTEM
    assert split.inference_system_cfg == BASE_SYSTEM
    assert split.truth_config_hash == split.inference_config_hash
    assert split.enabled_components["spectral_model"]["enabled"] is False
    assert split.enabled_components["high_order_wfe"]["enabled"] is False


def test_spectral_mismatch_returns_distinct_configs(tmp_path: Path) -> None:
    split = build_campaign_model_split(
        base_system_cfg=BASE_SYSTEM,
        spectral_model_cfg={
            "enabled": True,
            "source_seds": {"mode": "target"},
            "truth": {"n_lambda": 5, "wavelength_min_nm": 500.0, "wavelength_max_nm": 700.0},
            "inference": {"n_lambda": 3, "wavelength_min_nm": 525.0, "wavelength_max_nm": 675.0},
        },
        run_root=tmp_path,
        artifact_root=tmp_path / "model_split",
        seed_context={"test": "spectral"},
        source_kind="alpha_cen",
        target="ALPHA_CEN",
        write_artifacts=False,
    )
    assert split.truth_config_hash != split.inference_config_hash
    assert split.truth_system_cfg["source"]["n_lambda"] == 5
    assert split.inference_system_cfg["source"]["n_lambda"] == 3
    assert split.enabled_components["spectral_model"]["enabled"] is True


def test_high_order_wfe_mismatch_composes_after_spectral(tmp_path: Path) -> None:
    split = build_campaign_model_split(
        base_system_cfg=BASE_SYSTEM,
        spectral_model_cfg={
            "enabled": True,
            "truth": {"n_lambda": 5},
            "inference": {"n_lambda": 3},
            "source_seds": {"mode": "target"},
        },
        high_order_wfe_cfg={
            "enabled": True,
            "truth": {"npix": 16, "amplitude_nm_rms": 0.3, "mirrors": ["primary"], "seed": 123},
            "inference": {"knowledge_error": {"enabled": True, "amplitude_nm_rms": 0.1}},
            "artifacts": {"write_maps": False, "write_summary_json": False},
        },
        run_root=tmp_path,
        artifact_root=tmp_path / "model_split",
        seed_context={"test": "order"},
        source_kind="alpha_cen",
        target="ALPHA_CEN",
        write_artifacts=False,
    )
    assert split.truth_system_cfg["source"]["n_lambda"] == 5
    assert split.inference_system_cfg["source"]["n_lambda"] == 3
    assert "high_order_wfe" in split.truth_system_cfg["optics"]
    assert "high_order_wfe" in split.inference_system_cfg["optics"]
    assert split.truth_config_hash != split.inference_config_hash
