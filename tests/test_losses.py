import jax.numpy as jnp
import numpy as np

from dluxshera.inference.losses import gaussian_image_nll


def test_gaussian_image_nll_matches_closed_form_sum():
    pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    data = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    var = 2.0
    expected_per_pixel = 0.5 * jnp.log(2.0 * jnp.pi * var)
    expected = expected_per_pixel * pred.size

    loss = gaussian_image_nll(pred, data, var, reduce="sum")
    np.testing.assert_allclose(np.array(loss), np.array(expected))


def test_gaussian_image_nll_supports_mean_and_none():
    pred = jnp.array([0.0, 1.0])
    data = jnp.array([1.0, 1.0])
    var = jnp.array([1.0, 4.0])

    per_pixel = gaussian_image_nll(pred, data, var, reduce=None)
    expected_per_pixel = 0.5 * ((pred - data) ** 2 / var + jnp.log(2.0 * jnp.pi * var))
    np.testing.assert_allclose(np.array(per_pixel), np.array(expected_per_pixel))

    mean_loss = gaussian_image_nll(pred, data, var, reduce="mean")
    np.testing.assert_allclose(np.array(mean_loss), np.array(expected_per_pixel.mean()))
