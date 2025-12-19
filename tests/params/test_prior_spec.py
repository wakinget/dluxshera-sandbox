import jax
import jax.numpy as jnp
import pytest

from dluxshera.inference.prior import PriorField, PriorSpec
from dluxshera.params.store import ParameterStore


def test_quadratic_penalty_matches_manual_computation():
    center = ParameterStore.from_dict({
        "alpha": jnp.array([1.0, 2.0]),
        "beta": 3.0,
    })
    sigmas = {"alpha": 0.5, "beta": 2.0}
    spec = PriorSpec.from_sigmas(center, sigmas)

    perturbed = center.replace({
        "alpha": jnp.array([1.5, 1.0]),
        "beta": 4.0,
    })

    penalty = spec.quadratic_penalty(perturbed, center)
    expected = jnp.sum((perturbed.get("alpha") - center.get("alpha")) ** 2 / (2 * sigmas["alpha"] ** 2))
    expected += (perturbed.get("beta") - center.get("beta")) ** 2 / (2 * sigmas["beta"] ** 2)

    assert float(penalty) == pytest.approx(float(expected))


def test_quadratic_penalty_uses_center_store_when_provided():
    stored_means = ParameterStore.from_dict({"gamma": 0.0})
    spec = PriorSpec(fields={"gamma": PriorField(mean=1.0, sigma=2.0)})

    store = stored_means.replace({"gamma": 2.5})
    override_center = stored_means.replace({"gamma": -1.0})

    penalty = spec.quadratic_penalty(store, center_store=override_center)
    expected = (2.5 - (-1.0)) ** 2 / (2 * 2.0**2)

    assert float(penalty) == pytest.approx(expected)


def test_sample_near_matches_manual_sampling():
    center = ParameterStore.from_dict({
        "delta": jnp.array([0.0, 1.0]),
    })
    spec = PriorSpec.from_sigmas(center, {"delta": jnp.array([0.1, 0.2])})

    rng_key = jax.random.PRNGKey(0)
    sampled = spec.sample_near(center, rng_key)

    # Recreate the sampling manually using the same key split logic
    (subkey,) = jax.random.split(rng_key, 1)
    noise = jax.random.normal(subkey, shape=center.get("delta").shape)
    expected = center.get("delta") + jnp.array([0.1, 0.2]) * noise

    assert jnp.allclose(sampled.get("delta"), expected)


def test_invalid_distribution_raises_error():
    spec = PriorSpec(fields={"zeta": PriorField(dist="Laplace", mean=0.0, sigma=1.0)})

    with pytest.raises(ValueError):
        spec.quadratic_penalty(ParameterStore.from_dict({"zeta": 0.0}))
