from __future__ import annotations

import pytest
import jax.numpy as jnp
import jax.random as jr

from dluxshera.components.detectors import DetectorSpec, GSENSE2020BSI_SPEC
from dluxshera.utils.noise import apply_observation_noise


def test_photon_noise_poisson_and_variance():
    image = jnp.array([[5.0, 10.0]], dtype=float)
    rng = jr.PRNGKey(0)
    noise_cfg = {"enabled": True, "photon_noise": True}

    noisy, var = apply_observation_noise(image, noise_cfg=noise_cfg, rng_key=rng)

    rng_base, photon_key = jr.split(rng)
    expected = jr.poisson(photon_key, image).astype(image.dtype)
    expected_var = jnp.maximum(image, 0.0)

    assert jnp.array_equal(noisy, expected)
    assert jnp.array_equal(var, expected_var)

    # Reproducibility
    noisy_2, var_2 = apply_observation_noise(image, noise_cfg=noise_cfg, rng_key=rng)
    assert jnp.array_equal(noisy, noisy_2)
    assert jnp.array_equal(var, var_2)


def test_read_noise_adds_gaussian_and_variance():
    image = jnp.ones((2, 2), dtype=float) * 3.0
    rng = jr.PRNGKey(1)
    noise_cfg = {"enabled": True, "photon_noise": False, "read_noise": True}
    spec = DetectorSpec(read_noise=2.0)

    noisy, var = apply_observation_noise(
        image,
        noise_cfg=noise_cfg,
        rng_key=rng,
        detector_spec=spec,
    )

    rng_after, read_key = jr.split(rng)
    expected = image + 2.0 * jr.normal(read_key, image.shape, dtype=image.dtype)
    expected_var = jnp.zeros_like(image) + 4.0

    assert jnp.allclose(noisy, expected)
    assert jnp.allclose(var, expected_var)


def test_dark_current_adds_gaussian_and_variance():
    image = jnp.ones((1, 1), dtype=float) * 2.0
    rng = jr.PRNGKey(2)
    noise_cfg = {"enabled": True, "photon_noise": False, "dark_current": True}
    spec = DetectorSpec(dark_current=0.5)
    exposure = 10.0

    noisy, var = apply_observation_noise(
        image,
        noise_cfg=noise_cfg,
        rng_key=rng,
        detector_spec=spec,
        exposure_time_s=exposure,
    )

    _, dark_key = jr.split(rng)
    dc_var = spec.dark_current * exposure
    expected = image + jnp.sqrt(dc_var) * jr.normal(dark_key, image.shape, dtype=image.dtype)
    expected_var = jnp.zeros_like(image) + dc_var

    assert jnp.allclose(noisy, expected)
    assert jnp.allclose(var, expected_var)


def test_flags_disable_additional_noise():
    image = jnp.array([[3.0]], dtype=float)
    rng = jr.PRNGKey(3)
    noise_cfg = {"enabled": True, "photon_noise": False, "read_noise": False, "dark_current": False}

    noisy, var = apply_observation_noise(image, noise_cfg=noise_cfg, rng_key=rng)

    assert jnp.array_equal(noisy, image)
    assert jnp.array_equal(var, jnp.zeros_like(image))


def test_missing_detector_metadata_raises():
    image = jnp.array([[1.0]])
    rng = jr.PRNGKey(4)
    noise_cfg_read = {"enabled": True, "photon_noise": False, "read_noise": True}
    with pytest.raises(ValueError):
        apply_observation_noise(image, noise_cfg=noise_cfg_read, rng_key=rng)

    noise_cfg_dark = {"enabled": True, "photon_noise": False, "dark_current": True}
    with pytest.raises(ValueError):
        apply_observation_noise(image, noise_cfg=noise_cfg_dark, rng_key=rng, detector_spec=GSENSE2020BSI_SPEC)

    with pytest.raises(ValueError):
        apply_observation_noise(
            image,
            noise_cfg=noise_cfg_dark,
            rng_key=rng,
            detector_spec=DetectorSpec(dark_current=0.1),
            exposure_time_s=None,
        )
