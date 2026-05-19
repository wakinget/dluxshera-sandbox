from __future__ import annotations

import numpy as np

from dluxshera.inference.structured_curvature import (
    build_residual_prior_temporal_curvature,
)


def _linear_series(times: np.ndarray, a: float, b: float) -> np.ndarray:
    centered = times - np.mean(times)
    return a + b * centered


def test_residual_prior_temporal_curvature_annihilates_linear_subspace():
    times = np.asarray([0.05, 0.15, 0.25, 0.35], dtype=float)
    keys = (
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    )
    h = build_residual_prior_temporal_curvature(
        frame_times_s=times,
        frame_keys=keys,
        residual_sigmas_by_key={
            "source.x_position_as": 0.01,
            "source.y_position_as": 0.01,
            "source.position_angle_deg": 1.0e-4,
        },
        reduce="sum",
        subblock_reduce="sum",
    )
    np.testing.assert_allclose(h, h.T, atol=1e-10)
    assert float(np.min(np.linalg.eigvalsh(h))) >= -5e-8
    n = len(times)
    phi = np.zeros((n, len(keys)), dtype=float)
    phi[:, 0] = _linear_series(times, 1.0, 2.0)
    phi[:, 1] = _linear_series(times, -0.5, 0.1)
    phi[:, 2] = _linear_series(times, 90.0, -0.02)
    packed = phi.reshape(-1)
    np.testing.assert_allclose(h @ packed, 0.0, atol=3e-6)


def test_residual_prior_temporal_curvature_penalizes_nonlinear_modes_and_scales():
    times = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    keys = (
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    )
    h_small = build_residual_prior_temporal_curvature(
        frame_times_s=times,
        frame_keys=keys,
        residual_sigmas_by_key={
            "source.x_position_as": 0.01,
            "source.y_position_as": 0.01,
            "source.position_angle_deg": 1.0e-4,
        },
        reduce="sum",
        subblock_reduce="sum",
    )
    h_large = build_residual_prior_temporal_curvature(
        frame_times_s=times,
        frame_keys=keys,
        residual_sigmas_by_key={
            "source.x_position_as": 0.02,
            "source.y_position_as": 0.02,
            "source.position_angle_deg": 2.0e-4,
        },
        reduce="sum",
        subblock_reduce="sum",
    )
    nonlinear = np.zeros((len(times), len(keys)), dtype=float)
    nonlinear[:, 0] = np.asarray([0.0, 0.1, -0.2, 0.1, 0.0], dtype=float)
    v = nonlinear.reshape(-1)
    q_small = float(v @ h_small @ v)
    q_large = float(v @ h_large @ v)
    assert q_small > 0.0
    np.testing.assert_allclose(q_small, 4.0 * q_large, rtol=1e-6)


def test_residual_prior_temporal_curvature_mean_reduce_scales_by_frame_count():
    times = np.asarray([0.0, 1.0, 2.0], dtype=float)
    keys = (
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    )
    h_sum = build_residual_prior_temporal_curvature(
        frame_times_s=times,
        frame_keys=keys,
        residual_sigmas_by_key={
            "source.x_position_as": 0.01,
            "source.y_position_as": 0.01,
            "source.position_angle_deg": 1.0e-4,
        },
        reduce="sum",
        subblock_reduce="sum",
    )
    h_mean = build_residual_prior_temporal_curvature(
        frame_times_s=times,
        frame_keys=keys,
        residual_sigmas_by_key={
            "source.x_position_as": 0.01,
            "source.y_position_as": 0.01,
            "source.position_angle_deg": 1.0e-4,
        },
        reduce="mean",
        subblock_reduce="sum",
    )
    np.testing.assert_allclose(h_mean, h_sum / float(len(times)), rtol=1e-7, atol=1e-12)
