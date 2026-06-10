from __future__ import annotations

from pathlib import Path

import numpy as np

from dluxshera.utils.campaign_high_order_wfe import apply_high_order_wfe_campaign_config


def base_system() -> dict:
    return {
        "optics": {"pupil_npix": 16},
        "source": {"kind": "single_star"},
    }


def enabled_cfg(error_nm: float = 0.3) -> dict:
    return {
        "enabled": True,
        "truth": {
            "enabled": True,
            "mirrors": ["primary", "secondary"],
            "mode": "synthetic",
            "npix": 16,
            "amplitude_nm_rms": 1.0,
            "pairing": "independent",
            "remove_low_order_zernikes": True,
        },
        "inference": {
            "enabled": True,
            "mode": "knowledge_error",
            "knowledge_error": {
                "enabled": True,
                "amplitude_nm_rms": error_nm,
            },
            "use_truth_common_map": True,
        },
        "artifacts": {"write_maps": False, "write_summary_json": True},
    }


def test_disabled_high_order_wfe_preserves_system_config(tmp_path: Path) -> None:
    system = base_system()
    result = apply_high_order_wfe_campaign_config(
        system_cfg=system,
        high_order_wfe_cfg={"enabled": False},
        seed_context={"run": "unit"},
        artifact_root=tmp_path,
    )

    assert result.truth_system_cfg == system
    assert result.inference_system_cfg == system
    assert result.provenance["enabled"] is False


def test_enabled_high_order_wfe_is_deterministic_and_writes_summary(tmp_path: Path) -> None:
    a = apply_high_order_wfe_campaign_config(
        system_cfg=base_system(),
        high_order_wfe_cfg=enabled_cfg(),
        seed_context={"run": "unit", "case": 1},
        artifact_root=tmp_path / "a",
    )
    b = apply_high_order_wfe_campaign_config(
        system_cfg=base_system(),
        high_order_wfe_cfg=enabled_cfg(),
        seed_context={"run": "unit", "case": 1},
        artifact_root=tmp_path / "b",
    )

    assert a.provenance["truth_seed"] == b.provenance["truth_seed"]
    assert a.provenance["primary"]["knowledge_error_rms_nm"] == 0.3
    assert "high_order_wfe_summary_json" in a.artifact_paths
    assert Path(a.artifact_paths["high_order_wfe_summary_json"]).exists()

    truth = np.load(
        a.truth_system_cfg["optics"]["high_order_wfe"]["primary"]["map"]["array_path"]
    )
    inf_common = np.load(
        a.inference_system_cfg["optics"]["high_order_wfe"]["primary"]["map"]["array_path"]
    )
    inf_error = np.load(
        a.inference_system_cfg["optics"]["high_order_wfe"]["primary"]["knowledge_error"]["array_path"]
    )
    assert np.allclose(truth, inf_common)
    assert not np.allclose(inf_error, 0.0)


def test_zero_knowledge_error_control_matches_truth_reference() -> None:
    result = apply_high_order_wfe_campaign_config(
        system_cfg=base_system(),
        high_order_wfe_cfg=enabled_cfg(error_nm=0.0),
        seed_context={"run": "unit"},
        write_artifacts=False,
    )
    assert result.inference_system_cfg["optics"]["high_order_wfe"]["primary"]["knowledge_error"]["enabled"] is True
    err = np.asarray(
        result.inference_system_cfg["optics"]["high_order_wfe"]["primary"]["knowledge_error"]["array_nm"]
    )
    assert np.allclose(err, 0.0)
