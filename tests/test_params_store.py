import jax
import jax.numpy as jnp
import pytest

from dluxshera.params.store import ParameterStore
from dluxshera.params.spec import build_inference_spec_basic


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

def test_parameter_store_validate_against_inference_spec_basic():
    """
    Integration test: ensure a store with the right keys validates cleanly
    against the basic inference spec, and that missing/extra keys are caught.
    """
    spec = build_inference_spec_basic()

    # Build a store with matching keys. We don't care about actual values here,
    # since validate_against() currently checks only the key set.
    values = {}

    for key in spec.keys():
        if key.endswith("zernike_coeffs"):
            # Zernikes are arbitrary-length arrays; use a small example.
            values[key] = jnp.zeros(3)
        else:
            values[key] = 0.0

    store = ParameterStore.from_dict(values)

    # Should validate without error when keys match exactly.
    store.validate_against(spec)

    # Now drop one key to force a missing-key error.
    some_key = next(iter(spec.keys()))
    values_missing = dict(values)
    values_missing.pop(some_key)
    store_missing = ParameterStore.from_dict(values_missing)

    with pytest.raises(ValueError):
        store_missing.validate_against(spec)

    # Now add an extra bogus key to force an extra-key error.
    values_extra = dict(values)
    values_extra["not.a.real.param"] = 1.0
    store_extra = ParameterStore.from_dict(values_extra)

    with pytest.raises(ValueError):
        store_extra.validate_against(spec)


def test_parameter_store_validate_against_with_flags():
    """
    Check that allow_missing and allow_extra flags relax validation as intended.
    """
    spec = build_inference_spec_basic()

    base_values = {}
    for key in spec.keys():
        base_values[key] = 0.0
    store = ParameterStore.from_dict(base_values)

    # Missing allowed
    values_missing = dict(base_values)
    values_missing.pop("binary.separation_as")
    store_missing = ParameterStore.from_dict(values_missing)
    store_missing.validate_against(spec, allow_missing=True)

    # Extra allowed
    values_extra = dict(base_values)
    values_extra["debug.flag"] = True
    store_extra = ParameterStore.from_dict(values_extra)
    store_extra.validate_against(spec, allow_extra=True)
