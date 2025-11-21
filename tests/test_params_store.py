import jax
import jax.numpy as jnp

from dluxshera.params.store import ParameterStore


def test_parameter_store_get_and_replace():
    store = ParameterStore.from_dict(
        {
            "binary.separation_mas": 100.0,
            "noise.jitter_rms_mas": 0.5,
        }
    )

    assert store.get("binary.separation_mas") == 100.0
    assert store.get("noise.jitter_rms_mas") == 0.5

    # default behavior for missing key
    assert store.get("does.not.exist", default=None) is None

    # replace returns a new store and does not mutate the original
    new_store = store.replace({"binary.separation_mas": 120.0})

    assert new_store.get("binary.separation_mas") == 120.0
    assert store.get("binary.separation_mas") == 100.0


def test_parameter_store_is_pytree():
    store = ParameterStore.from_dict(
        {
            "a": jnp.array([1.0, 2.0]),
            "b": 3.0,
        }
    )

    # Simple function that operates on the store as a whole
    def sum_all(s: ParameterStore) -> jnp.ndarray:
        total = 0.0
        for _, v in s.items():
            total = total + jnp.sum(v)
        return total

    # Should work with jit and grad without errors
    sum_jit = jax.jit(sum_all)
    result = sum_jit(store)

    assert jnp.allclose(result, 1.0 + 2.0 + 3.0)
