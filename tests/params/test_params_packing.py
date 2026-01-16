import math

import pytest
import jax.numpy as jnp

from dluxshera.params.spec import ParamField, ParamSpec
from dluxshera.params.store import ParameterStore
from dluxshera.params.packing import pack_params, unpack_params


def _make_simple_spec() -> ParamSpec:
    """
    Helper: build a small ParamSpec with three primitive keys:
    - 'a' : scalar
    - 'b' : 1D vector
    - 'c' : 2x2 matrix
    """
    fields = [
        ParamField(
            key="a",
            group="test",
            kind="primitive",
            doc="Scalar parameter.",
        ),
        ParamField(
            key="b",
            group="test",
            kind="primitive",
            doc="1D vector parameter.",
        ),
        ParamField(
            key="c",
            group="test",
            kind="primitive",
            doc="2x2 matrix parameter.",
        ),
    ]
    return ParamSpec(fields)


def test_pack_and_unpack_round_trip_scalar_vector_matrix():
    """
    Round-trip test:
      spec + store -> theta (pack) -> new_store (unpack)

    Checks:
      - theta has expected size and dtype.
      - all inference keys are restored with original values.
      - non-inference keys in the base_store are preserved.
    """
    spec = _make_simple_spec()

    # Values with different shapes + an extra key that should be left untouched
    base_store = ParameterStore.from_dict(
        {
            "a": 1.0,
            "b": jnp.array([2.0, 3.0]),
            "c": jnp.array([[4.0, 5.0], [6.0, 7.0]]),
            "extra": 42.0,
        }
    )

    subset_keys = ["a", "b", "c"]
    spec_subset = spec.subset(subset_keys)

    theta = pack_params(spec_subset, base_store, dtype=jnp.float64)

    # We only expect float64 if the JAX config is enabled,
    # otherwise JAX returns a float32.
    expected_dtype = jnp.asarray(0.0, dtype=jnp.float64).dtype

    # Expected sizes: a (1) + b (2) + c (4) = 7
    assert theta.shape == (7,)
    assert theta.dtype == expected_dtype

    new_store = unpack_params(spec_subset, theta, base_store)

    # Check that the packed/unpacked values match
    assert math.isclose(float(new_store.get("a")), 1.0)
    assert jnp.allclose(new_store.get("b"), jnp.array([2.0, 3.0]))
    assert jnp.allclose(
        new_store.get("c"),
        jnp.array([[4.0, 5.0], [6.0, 7.0]]),
    )

    # Non-inference key should be preserved
    assert new_store.get("extra") == 42.0


def test_pack_params_missing_key_raises():
    """
    If a key in the inference spec subset is missing from the store,
    pack_params should raise a KeyError with a helpful message.
    """
    fields = [
        ParamField(
            key="missing_key",
            group="test",
            kind="primitive",
            doc="This key will not be in the store.",
        )
    ]
    spec = ParamSpec(fields)
    store = ParameterStore.from_dict({})

    with pytest.raises(KeyError) as excinfo:
        _ = pack_params(spec, store)

    msg = str(excinfo.value)
    assert "missing_key" in msg
    assert "pack_params" in msg


def test_unpack_params_size_mismatch_raises():
    """
    If theta.size does not match the total size implied by the template
    values in base_store, unpack_params should raise a ValueError.
    """
    spec = _make_simple_spec()
    base_store = ParameterStore.from_dict(
        {
            "a": 1.0,
            "b": jnp.array([2.0, 3.0]),
            "c": jnp.array([[4.0, 5.0], [6.0, 7.0]]),
        }
    )
    spec_subset = spec.subset(["a", "b", "c"])

    # Correct size would be 7; we deliberately choose 6
    theta_bad = jnp.zeros((6,), dtype=jnp.float32)

    with pytest.raises(ValueError) as excinfo:
        _ = unpack_params(spec_subset, theta_bad, base_store)

    msg = str(excinfo.value)
    assert "size mismatch" in msg
    assert "theta.size" in msg


def test_pack_and_unpack_empty_subset_is_noop():
    """
    For an empty spec subset:
      - pack_params should return a length-0 vector.
      - unpack_params should return the base_store unchanged.
    """
    empty_spec = ParamSpec([])
    base_store = ParameterStore.from_dict({"foo": 1.23})

    theta = pack_params(empty_spec, base_store)
    assert theta.shape == (0,)

    new_store = unpack_params(empty_spec, theta, base_store)

    # Behavior is defined as "return base_store" for empty subsets.
    # We don't rely on object identity, but the contents should match.
    assert new_store.as_dict() == base_store.as_dict()
