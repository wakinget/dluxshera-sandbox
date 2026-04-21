"""Smoke test for the canonical astrometry recipe."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr


def load_recipe_module():
    repo_root = Path(__file__).resolve().parents[2]
    recipe_path = repo_root / "examples" / "recipes" / "canonical_astrometry.py"
    spec = importlib.util.spec_from_file_location("canonical_astrometry_recipe", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_astrometry_recipe_runs(tmp_path):
    recipe = load_recipe_module()
    recipe.main(fast=True, results_dir=tmp_path)
    assert (tmp_path / "initial_psf_comparison.png").exists()


def test_canonical_validation_honors_infer_keys_and_noise_block():
    recipe = load_recipe_module()

    validated = recipe._validate_experiment(
        {
            "seed": 7,
            "infer_keys": [
                "source.y_position_as",
                "source.x_position_as",
            ],
            "noise": {
                "enabled": False,
                "photon_noise": False,
                "read_noise": True,
                "dark_current": True,
            },
        }
    )

    assert validated["infer_keys"] == (
        "source.y_position_as",
        "source.x_position_as",
    )
    assert validated["noise"] == {
        "enabled": False,
        "photon_noise": False,
        "read_noise": True,
        "dark_current": True,
    }


def test_canonical_cli_defaults_to_bundled_prescription():
    recipe = load_recipe_module()

    args = recipe._build_parser().parse_args([])

    assert args.prescription == recipe.PRESCRIPTION


def test_canonical_default_prescription_resolves_noiseless_noise_config():
    recipe = load_recipe_module()

    user_cfg = recipe.load_user_config(
        config_path=recipe.PRESCRIPTION,
        system_preset=recipe.DEFAULT_SYSTEM_PRESET,
        experiment_preset=recipe.DEFAULT_EXPERIMENT_PRESET,
    )
    resolved_cfg = recipe.resolve_config(user_cfg)
    experiment = recipe._validate_experiment(resolved_cfg["experiment"])

    assert experiment["noise"] == {
        "enabled": False,
        "photon_noise": False,
        "read_noise": False,
        "dark_current": False,
    }


def test_render_synthetic_observation_skips_noise_helper_when_disabled(monkeypatch):
    recipe = load_recipe_module()
    image = jnp.array([[2.0, 5.0]], dtype=float)

    class FakeBinder:
        class detector:
            spec = object()

        def model(self):
            return image

    class FakeStore:
        def get(self, key, default=None):
            return 10.0 if key == "source.exposure_time_s" else default

    def fail_apply_observation_noise(*args, **kwargs):
        raise AssertionError("apply_observation_noise should not run when noise is disabled")

    monkeypatch.setattr(recipe, "apply_observation_noise", fail_apply_observation_noise)

    rng_key = jr.PRNGKey(11)
    data_psf, data, data_var, next_rng = recipe._render_synthetic_observation(
        binder=FakeBinder(),
        truth_store=FakeStore(),
        noise_cfg={
            "enabled": False,
            "photon_noise": True,
            "read_noise": True,
            "dark_current": True,
        },
        rng_key=rng_key,
    )

    assert jnp.array_equal(data_psf, image)
    assert jnp.array_equal(data, image)
    assert jnp.array_equal(data_var, jnp.maximum(image, 1.0))
    assert jnp.array_equal(next_rng, rng_key)


def test_render_synthetic_observation_uses_noise_helper_when_enabled(monkeypatch):
    recipe = load_recipe_module()
    image = jnp.array([[3.0, 4.0]], dtype=float)
    expected_var = jnp.array([[9.0, 9.0]], dtype=float)
    calls = {}

    class FakeDetector:
        spec = object()

    class FakeBinder:
        detector = FakeDetector()

        def model(self):
            return image

    class FakeStore:
        def get(self, key, default=None):
            return 123.0 if key == "source.exposure_time_s" else default

    def fake_apply_observation_noise(
        image_arg,
        *,
        noise_cfg,
        rng_key,
        detector_spec,
        exposure_time_s,
    ):
        calls["image"] = image_arg
        calls["noise_cfg"] = noise_cfg
        calls["rng_key"] = rng_key
        calls["detector_spec"] = detector_spec
        calls["exposure_time_s"] = exposure_time_s
        return image_arg + 1.0, expected_var

    monkeypatch.setattr(recipe, "apply_observation_noise", fake_apply_observation_noise)

    rng_key = jr.PRNGKey(23)
    expected_next_rng, expected_noise_key = jr.split(rng_key)
    data_psf, data, data_var, next_rng = recipe._render_synthetic_observation(
        binder=FakeBinder(),
        truth_store=FakeStore(),
        noise_cfg={
            "enabled": True,
            "photon_noise": True,
            "read_noise": False,
            "dark_current": False,
        },
        rng_key=rng_key,
    )

    assert jnp.array_equal(data_psf, image)
    assert jnp.array_equal(data, image + 1.0)
    assert jnp.array_equal(data_var, expected_var)
    assert jnp.array_equal(next_rng, expected_next_rng)
    assert jnp.array_equal(calls["image"], image)
    assert calls["noise_cfg"]["enabled"] is True
    assert jnp.array_equal(calls["rng_key"], expected_noise_key)
    assert calls["exposure_time_s"] == 123.0
