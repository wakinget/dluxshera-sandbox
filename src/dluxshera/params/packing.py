# src/dluxshera/params/packing.py

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Optional

import jax.numpy as jnp
import numpy as np

from .spec import ParamSpec, ParamKey
from .store import ParameterStore


def pack_params(
    spec_subset: ParamSpec,
    store: ParameterStore,
    *,
    dtype: Optional[jnp.dtype] = None,
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
        Optional JAX dtype for the packed vector. When ``None`` (default),
        values retain their existing dtype; otherwise they are cast to the
        requested dtype. This lets callers preserve high-precision truth
        values (e.g., float64) when constructing θ vectors for gradient
        checks.

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

    See Also
    --------
    unpack_params : Inverse operation that restores structured values.
    build_index_map : Metadata-only description of the packed layout.
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

        arr = jnp.asarray(value, dtype=dtype) if dtype is not None else jnp.asarray(value)
        pieces.append(arr.ravel())

    theta = jnp.concatenate(pieces) if pieces else jnp.zeros((0,), dtype=dtype)
    return theta


def _compute_layout_hash(entries: list[dict[str, object]]) -> str:
    """
    Compute a stable SHA-256 hash for an IndexMap layout.

    The hash is derived from the ordered (name, shape) pairs only. This keeps
    the hash invariant to offsets and other metadata while still detecting any
    change to the packed layout (e.g., reordering parameters or altering
    shapes).
    """
    payload = [(entry["name"], entry["shape"]) for entry in entries]
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_index_map(
    spec_subset: ParamSpec,
    store: ParameterStore,
    *,
    theta=None,
    block_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """
    Build a serializable IndexMap aligned with parameter packing order.

    ``build_index_map`` reports where each parameter lives inside the packed
    θ-vector produced by :func:`pack_params`. It is intended for artifact
    metadata (e.g., ``meta.json``) so downstream tooling can interpret θ without
    re-running the pack/unpack logic.

    Parameters
    ----------
    spec_subset:
        ParamSpec describing the keys and their order, matching the spec used
        for :func:`pack_params`.
    store:
        ParameterStore providing concrete numeric values for every key in
        ``spec_subset``. The values are only used to infer shapes and sizes.
    theta:
        Optional packed θ-vector. When provided, its length is validated
        against the total packed size implied by ``spec_subset`` and ``store``.
    block_fn:
        Optional callable mapping a parameter key to a block label. This allows
        grouping related parameters in visualization or reporting tools.

    Returns
    -------
    dict
        A JSON-serializable IndexMap with ``entries`` (one per parameter) and
        a ``layout_hash``. Each entry includes:

        - ``name``: parameter key.
        - ``start`` / ``stop``: half-open indices into θ.
        - ``shape``: original array shape.
        - ``block``: block label (defaults to the key).

    Notes
    -----
    Unlike :func:`pack_params`, this function does **not** return θ or perform
    any numerical packing. Its purpose is *metadata only*—to describe how an
    already-packed θ vector maps back to named parameters.
    """
    entries: list[dict[str, object]] = []
    offset = 0

    for key in spec_subset.keys():
        value = store.get(key)
        if value is None:
            raise ValueError(
                f"IndexMap requires concrete values; got None for key {key!r}."
            )
        arr = np.asarray(value)
        size = int(arr.size)
        shape = list(arr.shape)
        start = offset
        stop = offset + size
        block = block_fn(key) if block_fn is not None else key

        entries.append(
            {
                "name": key,
                "start": start,
                "stop": stop,
                "shape": shape,
                "block": block,
            }
        )

        offset = stop

    if theta is not None:
        theta_size = int(np.asarray(theta).size)
        if theta_size != offset:
            raise ValueError(
                "IndexMap size mismatch: packed size from spec/store does not "
                f"match theta.size ({offset} vs {theta_size})."
            )

    index_map = {
        "entries": entries,
        "layout_hash": _compute_layout_hash(entries),
    }

    return index_map


def build_eigen_index_map(eigen_map: Any) -> dict:
    """Build a serializable IndexMap aligned with an EigenThetaMap layout."""

    dim_eigen = getattr(eigen_map, "dim_eigen", None)
    if dim_eigen is None:
        raise ValueError("eigen_map must provide dim_eigen.")

    entries = []
    for i in range(int(dim_eigen)):
        entries.append(
            {
                "name": f"eigen.mode[{i:02d}]",
                "start": i,
                "stop": i + 1,
                "shape": [],
                "block": "eigen",
            }
        )

    index_map = {
        "entries": entries,
        "layout_hash": _compute_layout_hash(entries),
    }

    meta: dict[str, object] = {}
    for name in ("dim_theta", "dim_eigen", "whiten"):
        if hasattr(eigen_map, name):
            meta[name] = getattr(eigen_map, name)

    eigvals = getattr(eigen_map, "eigvals", None)
    if eigvals is not None:
        eigvals_arr = np.asarray(eigvals)
        if eigvals_arr.size:
            meta["eigval_summary"] = {
                "min": float(np.min(eigvals_arr)),
                "max": float(np.max(eigvals_arr)),
            }

    if meta:
        index_map["eigen"] = meta

    return index_map


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

    See Also
    --------
    pack_params : Packs structured values into a flat θ vector.
    build_index_map : Emits layout metadata without unpacking θ.
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
