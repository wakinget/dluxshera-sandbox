import jax.numpy as jnp

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
