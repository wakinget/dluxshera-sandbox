import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from dluxshera.inference.losses import gaussian_image_nll
from dluxshera.inference.optimization import (
    EigenThetaMap,
    fim_theta,
    fim_theta_shera,
    hessian_theta,
    make_binder_image_nll_fn,
)

jax.config.update("jax_enable_x64", True)


def _toy_predict(theta):
    return jnp.array(
        [
            [theta[0] + 2.0 * theta[1], theta[0] * theta[1]],
            [jnp.sin(theta[0]), theta[1] ** 2],
        ]
    )


def test_fim_theta_matches_jacobian_formula_for_image_variance():
    theta = jnp.array([0.3, -0.4], dtype=jnp.float64)
    var = jnp.array([[1.0, 2.0], [4.0, 8.0]], dtype=jnp.float64)

    F = fim_theta(_toy_predict, theta, var)

    J = jax.jacfwd(_toy_predict)(theta)
    J_white = J / jnp.sqrt(var)[..., None]
    expected = J_white.reshape((-1, theta.size)).T @ J_white.reshape((-1, theta.size))

    onp.testing.assert_allclose(onp.asarray(F), onp.asarray(expected), rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(onp.asarray(F), onp.asarray(F.T), rtol=0.0, atol=1e-14)
    assert onp.all(onp.isfinite(onp.asarray(F)))


def test_fim_theta_supports_scalar_variance_and_mean_reduction():
    theta = jnp.array([0.3, -0.4], dtype=jnp.float64)

    F_sum = fim_theta(_toy_predict, theta, 2.0)
    F_mean = fim_theta(_toy_predict, theta, 2.0, reduce="mean")

    J = jax.jacfwd(_toy_predict)(theta)
    J2 = (J / jnp.sqrt(2.0)).reshape((-1, theta.size))
    expected = J2.T @ J2

    onp.testing.assert_allclose(onp.asarray(F_sum), onp.asarray(expected), rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(
        onp.asarray(F_mean),
        onp.asarray(expected / _toy_predict(theta).size),
        rtol=1e-12,
        atol=1e-12,
    )


def test_fim_theta_rejects_invalid_variance_and_nonfinite_jacobian():
    theta = jnp.array([1.0], dtype=jnp.float64)

    with pytest.raises(ValueError, match="strictly positive"):
        fim_theta(lambda t: jnp.array([t[0]]), theta, jnp.array([0.0]))

    with pytest.raises(ValueError, match="non-finite"):
        fim_theta(lambda t: jnp.array([jnp.sqrt(t[0] - 2.0)]), theta, 1.0)


def test_matched_gaussian_hessian_equals_fisher_at_zero_residual():
    theta_truth = jnp.array([0.2, -0.7], dtype=jnp.float64)
    data = _toy_predict(theta_truth)
    var = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)

    def loss_fn(theta):
        return gaussian_image_nll(_toy_predict(theta), data, var)

    grad = jax.grad(loss_fn)(theta_truth)
    H = hessian_theta(loss_fn, theta_truth)
    F = fim_theta(_toy_predict, theta_truth, var)

    onp.testing.assert_allclose(onp.asarray(grad), onp.zeros(theta_truth.shape), atol=1e-12)
    onp.testing.assert_allclose(onp.asarray(H), onp.asarray(F), rtol=1e-12, atol=1e-12)


def test_mismatched_gaussian_hessian_differs_from_fisher_but_fisher_is_psd():
    theta_ref = jnp.array([1.0, 0.5], dtype=jnp.float64)
    data = jnp.array([3.0, -1.0], dtype=jnp.float64)
    var = jnp.array([2.0, 0.5], dtype=jnp.float64)

    def predict(theta):
        return jnp.array([theta[0] ** 2 + theta[1], theta[0] * theta[1]])

    def loss_fn(theta):
        return gaussian_image_nll(predict(theta), data, var)

    H = hessian_theta(loss_fn, theta_ref)
    F = fim_theta(predict, theta_ref, var)

    assert not onp.allclose(onp.asarray(H), onp.asarray(F), rtol=1e-8, atol=1e-10)
    evals = onp.linalg.eigvalsh(onp.asarray(F))
    assert onp.all(onp.isfinite(evals))
    assert evals.min() >= -1e-12


def test_map_metric_is_likelihood_fisher_plus_prior_penalty_hessian():
    theta_ref = jnp.array([0.5, -1.0], dtype=jnp.float64)
    var = jnp.array([2.0, 4.0], dtype=jnp.float64)
    prior_sigma = jnp.array([0.25, 0.5], dtype=jnp.float64)
    prior_mean = jnp.array([0.0, 0.0], dtype=jnp.float64)

    def predict(theta):
        return jnp.array([theta[0] + theta[1], theta[0] * theta[1]])

    def prior_penalty(theta):
        return 0.5 * jnp.sum(((theta - prior_mean) / prior_sigma) ** 2)

    likelihood_fisher = fim_theta(predict, theta_ref, var)
    prior_hessian = hessian_theta(prior_penalty, theta_ref)
    posterior_metric = likelihood_fisher + prior_hessian

    expected_prior = jnp.diag(1.0 / prior_sigma ** 2)
    onp.testing.assert_allclose(
        onp.asarray(prior_hessian),
        onp.asarray(expected_prior),
        rtol=1e-12,
        atol=1e-12,
    )
    onp.testing.assert_allclose(
        onp.asarray(posterior_metric),
        onp.asarray(likelihood_fisher + expected_prior),
        rtol=1e-12,
        atol=1e-12,
    )


def test_indefinite_observed_hessian_is_rejected_but_fisher_whitening_succeeds():
    theta_ref = jnp.array([1.0], dtype=jnp.float64)
    data = jnp.array([10.0], dtype=jnp.float64)
    var = jnp.array([1.0], dtype=jnp.float64)

    def predict(theta):
        return jnp.array([theta[0] ** 2])

    def loss_fn(theta):
        return gaussian_image_nll(predict(theta), data, var)

    H = hessian_theta(loss_fn, theta_ref)
    F = fim_theta(predict, theta_ref, var)

    assert float(onp.linalg.eigvalsh(onp.asarray(H)).min()) < -1.0
    assert float(onp.linalg.eigvalsh(onp.asarray(F)).min()) > 0.0

    with pytest.raises(ValueError, match="not PSD"):
        EigenThetaMap.from_fim(H, theta_ref, whiten=True)

    eigen_map = EigenThetaMap.from_fim(F, theta_ref, whiten=True)
    z0 = eigen_map.z_from_theta(theta_ref)
    theta_roundtrip = eigen_map.theta_from_z(z0)
    grad_z = jax.grad(lambda z: loss_fn(eigen_map.theta_from_z(z)))(z0)

    assert onp.all(onp.isfinite(onp.asarray(z0)))
    assert onp.all(onp.isfinite(onp.asarray(theta_roundtrip)))
    assert onp.all(onp.isfinite(onp.asarray(grad_z)))


@pytest.mark.slow
def test_fim_theta_shera_wrapper_consistency(
    shera_smoke_cfg,
    shera_smoke_forward,
    shera_smoke_binder_data,
):
    cfg = shera_smoke_cfg
    spec, store = shera_smoke_forward

    infer_keys = ["source.separation_as", "source.x_position_as"]

    _, data, var = shera_smoke_binder_data

    F_wrapped, theta0_wrapped = fim_theta_shera(
        cfg,
        spec,
        store,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
    )

    _loss_fn_manual, theta0_manual, predict_fn = make_binder_image_nll_fn(
        cfg,
        spec,
        store,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
        return_predict_fn=True,
    )
    F_manual = fim_theta(predict_fn, theta0_manual, var)

    assert jnp.allclose(theta0_wrapped, theta0_manual)
    assert jnp.allclose(F_wrapped, F_manual, atol=1e-6)
