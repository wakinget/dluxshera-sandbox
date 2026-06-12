from __future__ import annotations

import jax.numpy as jnp
import pytest

from dluxshera.utils.noise import (
    expected_noise_variance,
    normalize_noise_request,
    resolve_detector_noise_spec,
)


def _system(read_noise: float | None = 1.25, dark_current: float | None = 0.2) -> dict:
    detector: dict = {"model": "HWK4123"}
    if read_noise is not None:
        detector["read_noise"] = read_noise
    if dark_current is not None:
        detector["dark_current"] = dark_current
    return {
        "source": {"exposure_time_s": 10.0},
        "detector": detector,
    }


def test_scalar_noise_enabled_normalizes_to_structured_enabled() -> None:
    cfg = normalize_noise_request("enabled")
    assert cfg["enabled"] is True
    assert cfg["shot_noise"] is True
    assert cfg["read_noise"] is False
    assert cfg["dark_current"] is False


def test_scalar_noise_disabled_normalizes_to_structured_disabled() -> None:
    cfg = normalize_noise_request("disabled")
    assert cfg["enabled"] is False
    assert cfg["shot_noise"] is False
    assert cfg["read_noise"] is False
    assert cfg["dark_current"] is False


def test_structured_noise_request_is_parsed() -> None:
    cfg = normalize_noise_request(
        {
            "enabled": True,
            "shot_noise": True,
            "read_noise": True,
            "dark_current": True,
            "variance_floor": 3.0,
        }
    )
    assert cfg["shot_noise"] is True
    assert cfg["read_noise"] is True
    assert cfg["dark_current"] is True
    assert cfg["variance_floor"] == 3.0


def test_detector_read_noise_resolves_from_detector_spec() -> None:
    info = resolve_detector_noise_spec(_system(read_noise=1.25), {"enabled": True, "read_noise": True})
    assert info["read_noise_electrons"] == 1.25
    assert info["read_noise_source"] == "detector_spec:read_noise"


def test_explicit_read_noise_override_wins_over_detector_spec() -> None:
    info = resolve_detector_noise_spec(
        _system(read_noise=1.25),
        {"enabled": True, "read_noise": True, "read_noise_electrons": 4.5},
    )
    assert info["read_noise_electrons"] == 4.5
    assert info["read_noise_source"] == "config_override"


def test_missing_detector_read_noise_warns_or_fails_strict() -> None:
    cfg = {"enabled": True, "read_noise": True, "use_detector_read_noise": False}
    info = resolve_detector_noise_spec(_system(read_noise=None), cfg)
    assert info["read_noise_source"] == "missing"
    assert info["warnings"]
    with pytest.raises(ValueError):
        resolve_detector_noise_spec(_system(read_noise=None), cfg, strict=True)


def test_dark_current_resolves_and_scales_with_exposure_time() -> None:
    info = resolve_detector_noise_spec(_system(dark_current=0.2), {"enabled": True, "dark_current": True})
    assert info["dark_current_e_per_s"] == 0.2
    assert info["dark_current_source"] == "detector_spec:dark_current"
    variance = expected_noise_variance(
        jnp.ones((2, 2)),
        noise_cfg={"enabled": True, "shot_noise": False, "dark_current": True},
        detector_noise=info,
    )
    assert jnp.allclose(variance, jnp.ones((2, 2)) * 2.0)


def test_expected_variance_terms_and_floor_are_separate() -> None:
    image = jnp.array([[2.0, 8.0]])
    info = {
        "read_noise_electrons": 3.0,
        "dark_current_e_per_s": 0.5,
        "exposure_time_s": 4.0,
    }
    variance = expected_noise_variance(
        image,
        noise_cfg={"enabled": True, "shot_noise": True, "read_noise": True, "dark_current": True},
        detector_noise=info,
    )
    assert jnp.allclose(variance, image + 9.0 + 2.0)
    floored = expected_noise_variance(
        image * 0.0,
        noise_cfg={"enabled": True, "shot_noise": True},
        detector_noise=info,
        variance_floor=1.0,
    )
    assert jnp.allclose(floored, jnp.ones_like(image))


def test_shot_variance_signal_dependent_and_read_variance_constant() -> None:
    image = jnp.array([[1.0, 5.0]])
    shot = expected_noise_variance(image, noise_cfg={"enabled": True, "shot_noise": True})
    read = expected_noise_variance(
        image,
        noise_cfg={"enabled": True, "shot_noise": False, "read_noise": True},
        detector_noise={"read_noise_electrons": 2.0},
    )
    assert not jnp.allclose(shot[0, 0], shot[0, 1])
    assert jnp.allclose(read, jnp.ones_like(image) * 4.0)
