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


def mirror_cfg(active_mirror: str, error_nm: float = 0.1) -> dict:
    cfg = enabled_cfg(error_nm)
    cfg["truth"]["seed"] = 101
    cfg["inference"]["knowledge_error"]["seed"] = 202
    cfg["inference"]["knowledge_error"]["mirrors"] = {
        "primary": {
            "enabled": active_mirror == "primary",
            "amplitude_nm_rms": error_nm if active_mirror == "primary" else 0.0,
        },
        "secondary": {
            "enabled": active_mirror == "secondary",
            "amplitude_nm_rms": error_nm if active_mirror == "secondary" else 0.0,
        },
    }
    return cfg


def _knowledge_error_array(result, mirror: str) -> np.ndarray | None:
    block = result.inference_system_cfg["optics"]["high_order_wfe"][mirror]["knowledge_error"]
    if not block.get("enabled", False):
        return None
    if "array_nm" in block:
        return np.asarray(block["array_nm"], dtype=float)
    return np.load(block["array_path"])


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


def test_legacy_scalar_knowledge_error_applies_to_both_mirrors() -> None:
    result = apply_high_order_wfe_campaign_config(
        system_cfg=base_system(),
        high_order_wfe_cfg=enabled_cfg(error_nm=0.1),
        seed_context={"run": "legacy"},
        write_artifacts=False,
    )

    primary = result.provenance["primary"]
    secondary = result.provenance["secondary"]
    assert primary["knowledge_error_enabled"] is True
    assert secondary["knowledge_error_enabled"] is True
    assert primary["requested_knowledge_error_rms_nm"] == 0.1
    assert secondary["requested_knowledge_error_rms_nm"] == 0.1
    assert _knowledge_error_array(result, "primary") is not None
    assert _knowledge_error_array(result, "secondary") is not None


def test_mirror_specific_primary_ke_preserves_secondary_truth_match() -> None:
    result = apply_high_order_wfe_campaign_config(
        system_cfg=base_system(),
        high_order_wfe_cfg=mirror_cfg("primary", error_nm=0.1),
        seed_context={"run": "m1"},
        write_artifacts=False,
    )

    truth_block = result.truth_system_cfg["optics"]["high_order_wfe"]
    inf_block = result.inference_system_cfg["optics"]["high_order_wfe"]
    assert "primary" in truth_block
    assert "secondary" in truth_block
    assert inf_block["primary"]["knowledge_error"]["enabled"] is True
    assert inf_block["secondary"]["knowledge_error"]["enabled"] is False
    assert result.provenance["primary"]["truth_inference_difference_rms_nm"] == 0.1
    assert result.provenance["secondary"]["truth_inference_difference_rms_nm"] == 0.0


def test_mirror_specific_secondary_ke_preserves_primary_truth_match() -> None:
    result = apply_high_order_wfe_campaign_config(
        system_cfg=base_system(),
        high_order_wfe_cfg=mirror_cfg("secondary", error_nm=0.1),
        seed_context={"run": "m2"},
        write_artifacts=False,
    )

    inf_block = result.inference_system_cfg["optics"]["high_order_wfe"]
    assert inf_block["primary"]["knowledge_error"]["enabled"] is False
    assert inf_block["secondary"]["knowledge_error"]["enabled"] is True
    assert result.provenance["primary"]["truth_inference_difference_rms_nm"] == 0.0
    assert result.provenance["secondary"]["truth_inference_difference_rms_nm"] == 0.1


def test_fixed_ke_seed_hash_survives_run_draw_field_and_amplitude_changes() -> None:
    base = mirror_cfg("primary", error_nm=0.1)
    varied = mirror_cfg("primary", error_nm=1.0)

    a = apply_high_order_wfe_campaign_config(
        system_cfg=base_system(),
        high_order_wfe_cfg=base,
        seed_context={"run": "a", "draw": 0, "field": "center"},
        write_artifacts=False,
    )
    b = apply_high_order_wfe_campaign_config(
        system_cfg=base_system(),
        high_order_wfe_cfg=varied,
        seed_context={"run": "b", "draw": 9, "field": "xp1"},
        write_artifacts=False,
    )

    assert a.provenance["primary"]["normalised_knowledge_error_map_hash"]
    assert (
        a.provenance["primary"]["normalised_knowledge_error_map_hash"]
        == b.provenance["primary"]["normalised_knowledge_error_map_hash"]
    )
    arr_a = _knowledge_error_array(a, "primary")
    arr_b = _knowledge_error_array(b, "primary")
    assert arr_a is not None
    assert arr_b is not None
    assert np.allclose(arr_b, arr_a * 10.0)


def test_high_order_wfe_invalid_mirror_and_negative_amplitude_fail() -> None:
    bad_mirror = mirror_cfg("primary", error_nm=0.1)
    bad_mirror["inference"]["knowledge_error"]["mirrors"]["tertiary"] = {
        "enabled": True,
        "amplitude_nm_rms": 0.1,
    }
    with np.testing.assert_raises_regex(ValueError, "Unsupported"):
        apply_high_order_wfe_campaign_config(
            system_cfg=base_system(),
            high_order_wfe_cfg=bad_mirror,
            seed_context={},
            write_artifacts=False,
        )

    bad_amp = mirror_cfg("primary", error_nm=0.1)
    bad_amp["inference"]["knowledge_error"]["mirrors"]["primary"]["amplitude_nm_rms"] = -1.0
    with np.testing.assert_raises_regex(ValueError, "must be >= 0"):
        apply_high_order_wfe_campaign_config(
            system_cfg=base_system(),
            high_order_wfe_cfg=bad_amp,
            seed_context={},
            write_artifacts=False,
        )
