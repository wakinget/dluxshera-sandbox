# tests/test_eigen_theta_map.py
import jax.numpy as np
import numpy as onp

from dluxshera.inference.optimization import EigenThetaMap


def test_eigen_theta_map_roundtrip_unwhitened():
    # Simple 2D curvature; eigenvectors are just the identity
    F = np.diag(np.array([1.0, 4.0]))
    theta_ref = np.array([1.0, -2.0])

    etm = EigenThetaMap.from_fim(F, theta_ref, truncate=None, whiten=False)

    # Shapes are as expected
    assert etm.dim_theta == 2
    assert etm.dim_eigen == 2
    assert etm.eigvecs.shape == (2, 2)

    # Round-trip a point near theta_ref
    theta = np.array([1.1, -1.9])
    z = etm.to_eigen(theta)
    theta_rt = etm.from_eigen(z)

    onp.testing.assert_allclose(onp.array(theta_rt), onp.array(theta), rtol=1e-6, atol=1e-6)


def test_eigen_theta_map_truncate_and_whiten():
    # 3D curvature with distinct eigenvalues
    F = np.diag(np.array([1.0, 2.0, 5.0]))
    theta_ref = np.zeros(3)

    # Keep top-2 modes and whiten
    etm = EigenThetaMap.from_fim(F, theta_ref, truncate=2, whiten=True)

    assert etm.dim_theta == 3
    assert etm.dim_eigen == 2
    assert etm.eigvecs.shape == (3, 2)
    assert etm.eigvals.shape == (2,)

    # Check that round-trip still works in the retained subspace:
    #   take a perturbation only along the first two coordinates
    delta = np.array([0.0, -0.2, 0.1])
    theta = theta_ref + delta

    z = etm.to_eigen(theta)
    theta_rt = etm.from_eigen(z)

    # Because we've truncated, we only represent the projection of delta
    # onto the top-2 eigenmodes; in this diagonal case that's just the
    # first two coords, so we should match exactly.
    onp.testing.assert_allclose(onp.array(theta_rt), onp.array(theta), rtol=1e-6, atol=1e-6)
