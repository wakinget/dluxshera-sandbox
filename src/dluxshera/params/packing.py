# src/dluxshera/params/packing.py

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from .spec import ParamSpec, ParamKey
from .store import ParameterStore


def pack_params(
    spec_subset: ParamSpec,
    store: ParameterStore,
    *,
    dtype: Optional[jnp.dtype] = jnp.float32,
) -> jnp.ndarray:
    """
    Pack a subset of parameters from a ParameterStore into a flat 1D vector.

    Parameters
    ----------
    spec_subset:
        A ParamSpec containing *exactly* the keys you intend to infer,
        in the order you want them to appear in the packed vector.
        Typically constructed via `base_spec.subset(infer_keys)`.

    store:
        ParameterStore holding numeric values for each key in `spec_subset`.

    dtype:
        JAX dtype for the packed vector (default: float32). This allows you
        to choose float32 vs float64 depending on your JAX configuration.
        All values are cast to this dtype during packing.

    Returns
    -------
    theta : jnp.ndarray
        A 1D JAX array containing the concatenated parameter values.

    Raises
    ------
    KeyError
        If any key in `spec_subset` is missing from `store`.

    ValueError
        If any value for a key in `spec_subset` is `None` (we require
        concrete numeric values for all inferred parameters).
    """
    keys = list(spec_subset.keys())

    if not keys:
        # Empty subset → empty vector (useful for edge cases / tests).
        return jnp.zeros((0,), dtype=dtype or jnp.float32)

    pieces = []

    for key in keys:
        try:
            value = store.get(key)
        except KeyError as exc:
            raise KeyError(
                f"pack_params: store is missing value for key {key!r} "
                f"required by the inference subset."
            ) from exc

        if value is None:
            raise ValueError(
                f"pack_params: value for key {key!r} is None. "
                "All parameters in the inference subset must be concrete "
                "numeric values."
            )

        arr = jnp.asarray(value, dtype=dtype)
        pieces.append(arr.ravel())

    theta = jnp.concatenate(pieces) if pieces else jnp.zeros((0,), dtype=dtype)
    return theta


def unpack_params(
    spec_subset: ParamSpec,
    theta: jnp.ndarray,
    base_store: ParameterStore,
) -> ParameterStore:
    """
    Unpack a flat parameter vector into a new ParameterStore.

    This is the inverse of `pack_params` for a given `spec_subset` and
    `base_store`. The `base_store` provides the template shapes for each
    parameter key; those shapes are used to slice and reshape the flat
    vector back into structured values.

    Parameters
    ----------
    spec_subset:
        ParamSpec describing exactly which keys are encoded in `theta`,
        and in what order. Must match the spec used in `pack_params`.

    theta:
        Flat 1D JAX array containing the packed parameter values.

    base_store:
        ParameterStore that provides:
          - existing values for *all* parameters (inferred + fixed), and
          - the template shapes for each key in `spec_subset`.
        The returned ParameterStore is a copy of `base_store` with only
        the keys in `spec_subset` replaced by values unpacked from `theta`.

    Returns
    -------
    new_store : ParameterStore
        A new store where the parameters in `spec_subset` have been updated
        from `theta`, and all other parameters are unchanged.

    Raises
    ------
    KeyError
        If any key in `spec_subset` is missing from `base_store`.

    ValueError
        If `theta` does not have the expected total size implied by the
        shapes of the corresponding values in `base_store`, or if any
        template value is `None`.
    """
    keys = list(spec_subset.keys())
    n_theta = int(theta.size)

    # Early exit for empty subsets.
    if not keys:
        if n_theta != 0:
            raise ValueError(
                "unpack_params: non-empty theta provided for an empty "
                "spec_subset (no keys to unpack)."
            )
        return base_store

    # First pass: determine total expected size and remember shapes.
    shapes = {}
    sizes = {}
    total_expected = 0

    for key in keys:
        try:
            tmpl = base_store.get(key)
        except KeyError as exc:
            raise KeyError(
                f"unpack_params: base_store is missing template value for "
                f"key {key!r}."
            ) from exc

        if tmpl is None:
            raise ValueError(
                f"unpack_params: template value for key {key!r} is None. "
                "We need a concrete value in the base_store to infer the "
                "expected shape for unpacking."
            )

        tmpl_arr = jnp.asarray(tmpl)
        shape = tmpl_arr.shape
        size = int(tmpl_arr.size)

        shapes[key] = shape
        sizes[key] = size
        total_expected += size

    if n_theta != total_expected:
        raise ValueError(
            "unpack_params: size mismatch between theta and spec_subset. "
            f"Expected total size {total_expected} from base_store templates "
            f"but got theta.size={n_theta}."
        )

    # Second pass: slice and reshape theta into per-key arrays.
    offset = 0
    updates = {}

    for key in keys:
        size = sizes[key]
        shape = shapes[key]

        # Slice the relevant chunk
        chunk = theta[offset : offset + size]
        offset += size

        if size == 0:
            # Zero-sized arrays are allowed but rare; keep behavior simple.
            new_value = jnp.asarray(chunk).reshape(shape)
        else:
            new_value = chunk.reshape(shape)

        updates[key] = new_value

    # Apply updates on top of base_store, preserving all other keys.
    new_store = base_store.replace(updates)
    return new_store
