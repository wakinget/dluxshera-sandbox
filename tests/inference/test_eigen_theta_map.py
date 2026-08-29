import jax.numpy as jnp
import numpy as np
import pytest

from dluxshera.inference.optimization import EigenThetaMap


def test_eigen_theta_map_roundtrip_unwhitened():
    F = jnp.array([[2.0, 0.1], [0.1, 1.0]])
    theta_ref = jnp.array([0.5, -1.0])
    eigen_map = EigenThetaMap.from_fim(F, theta_ref, whiten=False)

    theta = jnp.array([0.7, -0.8])
    z = eigen_map.z_from_theta(theta)
    theta_roundtrip = eigen_map.theta_from_z(z)

    assert jnp.allclose(theta, theta_roundtrip, atol=1e-6)


def test_eigen_theta_map_whitened_scales_quadratic():
    F = jnp.diag(jnp.array([4.0, 1.0, 0.25]))
    theta_ref = jnp.zeros(3)
    eigen_map = EigenThetaMap.from_fim(F, theta_ref, whiten=True)

    z = jnp.array([1.0, -2.0, 0.5])
    theta = eigen_map.theta_from_z(z)
    delta = theta - theta_ref

    quad_theta = 0.5 * delta @ (F @ delta)
    quad_z = 0.5 * jnp.sum(z ** 2)

    assert jnp.allclose(quad_theta, quad_z, atol=1e-5)


def test_eigen_theta_map_accepts_tiny_roundoff_negative_eigenvalue():
    F = jnp.diag(jnp.array([1.0, -1.0e-14]))
    theta_ref = jnp.array([0.5, -1.0])

    eigen_map = EigenThetaMap.from_fim(F, theta_ref, whiten=True)
    assert float(jnp.min(eigen_map.eigvals)) == 0.0

    z0 = eigen_map.z_from_theta(theta_ref)
    theta_roundtrip = eigen_map.theta_from_z(z0)

    assert jnp.all(jnp.isfinite(z0))
    assert jnp.all(jnp.isfinite(theta_roundtrip))
    assert jnp.allclose(theta_ref, theta_roundtrip, atol=1e-6)


def test_eigen_theta_map_rejects_material_negative_eigenvalue():
    F = jnp.diag(jnp.array([1.0, -1.0e-2]))
    theta_ref = jnp.zeros(2)

    with pytest.raises(ValueError, match="minimum eigenvalue"):
        EigenThetaMap.from_fim(F, theta_ref, whiten=True)


def test_eigen_theta_map_rejects_nonfinite_and_nonsquare_matrix():
    theta_ref = jnp.zeros(2)

    with pytest.raises(ValueError, match="non-finite"):
        EigenThetaMap.from_fim(jnp.array([[1.0, jnp.nan], [0.0, 1.0]]), theta_ref)

    with pytest.raises(ValueError, match="square"):
        EigenThetaMap.from_fim(jnp.ones((2, 3)), theta_ref)


def test_eigen_theta_map_accepts_high_dynamic_range_psd_without_ridge():
    eigvals = np.array([1.0e11, 1.0e3, 1.0e-3], dtype=float)
    F = np.diag(eigvals)
    theta_ref = np.array([1.0, -2.0, 0.5], dtype=float)

    eigen_map = EigenThetaMap.from_fim(F, theta_ref, whiten=True)

    kept = np.asarray(eigen_map.eigvals)
    assert kept[-1] == pytest.approx(1.0e-3)
    assert np.all(kept > 0.0)

    z0 = eigen_map.z_from_theta(theta_ref)
    theta_roundtrip = eigen_map.theta_from_z(z0)

    assert np.all(np.isfinite(np.asarray(z0)))
    assert np.all(np.isfinite(np.asarray(theta_roundtrip)))
    np.testing.assert_allclose(np.asarray(theta_roundtrip), theta_ref, rtol=0.0, atol=1e-12)


def test_eigen_theta_map_rejects_material_negative_mode_at_shera_scale():
    F = np.diag(np.array([2.12e11, 1.0e3, -4.38e-2], dtype=float))
    theta_ref = np.zeros(3, dtype=float)

    with pytest.raises(ValueError, match="material_negative_eigenvalues"):
        EigenThetaMap.from_fim(F, theta_ref, whiten=True)
