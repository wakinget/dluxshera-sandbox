from __future__ import annotations

import pytest

from dluxshera.utils.noise import normalize_subblock_noise_config


def test_scalar_noise_enabled_normalizes_legacy_behavior() -> None:
    normalized = normalize_subblock_noise_config({"noise": "enabled"})

    assert normalized.enabled is True
    assert normalized.legacy_noise_mode == "enabled"
    assert normalized.shot_noise is True
    assert normalized.photon_noise is True


def test_scalar_noise_disabled_disables_render_terms() -> None:
    normalized = normalize_subblock_noise_config({"noise": "disabled"})

    assert normalized.enabled is False
    assert normalized.legacy_noise_mode == "disabled"
    assert normalized.render_template_noise_block()["enabled"] is False
    assert normalized.render_template_noise_block()["photon_noise"] is False


def test_scalar_noise_inherit_leaves_template_settings_intact() -> None:
    normalized = normalize_subblock_noise_config({"noise": "inherit"})

    assert normalized.legacy_noise_mode == "inherit"
    assert normalized.render_template_noise_block() is None


def test_structured_noise_maps_shot_to_photon_and_preserves_terms() -> None:
    normalized = normalize_subblock_noise_config(
        {
            "noise": {
                "enabled": True,
                "shot_noise": True,
                "read_noise": True,
                "read_noise_electrons": 2.5,
                "dark_current": False,
                "variance_floor": 0.5,
            },
            "use_render_variance": "auto",
        },
        exposure_time_s=0.05,
    )

    block = normalized.render_template_noise_block()
    assert block is not None
    assert block["shot_noise"] is True
    assert block["photon_noise"] is True
    assert block["read_noise"] is True
    assert block["dark_current"] is False
    assert block["read_noise_electrons"] == 2.5
    assert normalized.variance_floor == 0.5
    assert normalized.variance_floor_source == "experiment.subblocks.noise.variance_floor"
    assert normalized.use_render_variance_resolved is True


def test_explicit_read_noise_override_wins() -> None:
    normalized = normalize_subblock_noise_config(
        {"noise": {"enabled": True, "read_noise": True, "read_noise_electrons": 7.0}},
        detector_cfg={"detector": {"model": "GSENSE2020BSI", "read_noise_electrons": 3.0}},
    )

    assert normalized.read_noise_electrons == 7.0
    assert normalized.read_noise_source == "config_override"


def test_missing_read_noise_fails_strict_when_enabled() -> None:
    with pytest.raises(ValueError, match="read_noise=true"):
        normalize_subblock_noise_config(
            {
                "noise": {
                    "enabled": True,
                    "read_noise": True,
                    "use_detector_read_noise": False,
                }
            },
            strict=True,
        )


def test_dark_current_disabled_does_not_require_amplitude() -> None:
    normalized = normalize_subblock_noise_config(
        {"noise": {"enabled": True, "dark_current": False}},
        strict=True,
    )

    assert normalized.dark_current is False
    assert normalized.dark_current_source == "disabled"


def test_variance_floor_auto_resolves_from_enabled_terms() -> None:
    normalized = normalize_subblock_noise_config(
        {
            "noise": {
                "enabled": True,
                "read_noise": True,
                "read_noise_electrons": 2.0,
                "dark_current": True,
                "dark_current_e_per_s": 10.0,
                "variance_floor": "auto",
            }
        },
        exposure_time_s=0.5,
        strict=True,
    )

    assert normalized.variance_floor == pytest.approx(9.0)
    assert normalized.variance_floor_source == "auto"
