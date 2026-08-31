from __future__ import annotations

import numpy as np
import pytest

from dluxshera.ml.metrics import compute_regression_metrics
from dluxshera.ml.noise import NoiseConfig, apply_pair_noise


def test_noise_disabled_is_exact_identity_copy() -> None:
    image_a = np.asarray([[1.0, 2.0]], dtype=np.float32)
    image_b = np.asarray([[3.0, 4.0]], dtype=np.float32)
    out_a, out_b = apply_pair_noise(image_a, image_b, NoiseConfig(enabled=False))
    np.testing.assert_array_equal(out_a, image_a)
    np.testing.assert_array_equal(out_b, image_b)
    assert out_a is not image_a
    assert out_b is not image_b


def test_seeded_observation_noise_is_reproducible_and_asymmetric() -> None:
    image_a = np.full((4, 4), 10.0, dtype=np.float32)
    image_b = np.full((4, 4), 20.0, dtype=np.float32)
    cfg = NoiseConfig(enabled=True, apply_to="observation", seed=99, photon_noise=True)
    first_a, first_b = apply_pair_noise(image_a, image_b, cfg, pair_record_id="pair")
    second_a, second_b = apply_pair_noise(image_a, image_b, cfg, pair_record_id="pair")
    np.testing.assert_array_equal(first_a, image_a)
    np.testing.assert_array_equal(first_b, second_b)
    assert not np.array_equal(first_b, image_b)

    both_cfg = NoiseConfig(enabled=True, apply_to="both", seed=99, photon_noise=True)
    both_a, both_b = apply_pair_noise(image_a, image_b, both_cfg, pair_record_id="pair")
    assert not np.array_equal(both_a, image_a)
    assert not np.array_equal(both_b, image_b)


def test_photon_noise_negative_inputs_are_explicit() -> None:
    cfg = NoiseConfig(enabled=True, apply_to="observation", photon_noise=True)
    with pytest.raises(ValueError, match="non-negative"):
        apply_pair_noise(np.ones((2, 2)), -np.ones((2, 2)), cfg)
    clipped = NoiseConfig(
        enabled=True,
        apply_to="observation",
        photon_noise=True,
        negative_policy="clip",
    )
    _, out_b = apply_pair_noise(np.ones((2, 2)), -np.ones((2, 2)), clipped)
    assert np.all(out_b == 0.0)


def test_regression_metrics_report_fisher_and_physical_errors() -> None:
    truth = np.asarray([[1.0, 0.0], [0.0, -2.0]], dtype=np.float32)
    pred = np.asarray([[1.5, 0.0], [0.0, -1.0]], dtype=np.float32)
    metrics = compute_regression_metrics(
        pred,
        truth,
        fisher_sigmas=[0.5, 2.0],
        parameter_labels=["contrast", "z0"],
    )
    assert metrics["sample_count"] == 2
    assert metrics["fisher_overall_rmse"] > 0.0
    assert metrics["physical_overall_rmse"] > 0.0
    assert set(metrics["fisher_per_parameter_rmse"]) == {"contrast", "z0"}
    assert metrics["fisher_alignment_cosine_mean"] is not None
