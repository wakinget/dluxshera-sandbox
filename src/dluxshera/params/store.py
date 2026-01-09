"""
ParameterStore: immutable mapping from parameter keys to values.

This is the runtime "state" companion to ParamSpec: for a given model run,
a ParameterStore holds the concrete numeric values for some subset of the
parameters defined in a spec.

Design goals
------------
- Mostly-immutable API (replace() returns a new instance).
- JAX pytree so we can pass it through jit/grad/vmap.
- Lightweight, with no dependency on model code.

Key design points
-----------------

- Parameter keys are *string identifiers* (ParamKey = str), and in practice
  we use dotted, hierarchical names such as:

      "binary.separation_as"
      "imaging.plate_scale_as_per_pix"
      "noise.jitter_rms_as"

  These keys are part of the public "parameter API" of the model: they are
  used consistently in specs, priors, configs, logging, etc.

- Because keys may contain dots and other characters that are not valid
  Python identifiers, we cannot safely rely on keyword arguments for
  updating them. For example:

      store.replace(binary_separation_as=12.0)

  would create/modify a key called "binary_separation_as", which is *not*
  the same as the canonical key "binary.separation_as". This kind of
  mismatch is very easy to introduce and hard to debug.

- To avoid this, ParameterStore.replace() accepts a mapping of literal
  keys to values (e.g. replace({"binary.separation_as": 12.0})), and only
  supports **kwargs as a convenience for simple, identifier-like keys.

The overall goal is to treat parameter keys as opaque strings with stable
semantics, independent of Python's identifier syntax, while keeping the
store JAX-friendly and mostly immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .spec import ParamSpec, ParamKey


# Sentinel for "no default provided"
_MISSING = object()


@dataclass(frozen=True)
class ParameterStore:
    """
    Immutable-ish container mapping parameter keys to values.

    Values are typically Python floats/ints or jax.numpy arrays. Keys are
    ParamKey strings and may contain dots (e.g. "binary.separation_as").

    Because keys are arbitrary strings, updates should generally be passed
    as an explicit mapping to `replace()`, e.g.:

        store = store.replace({"binary.separation_as": 12.0})

    rather than using keyword arguments, which only work reliably for simple
    identifier-like keys that don't contain dots.
    """
    _values: Mapping[ParamKey, Any]

    # --- basic container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[ParamKey]:
        return iter(self._values)

    def __contains__(self, key: object) -> bool:
        return key in self._values

    def keys(self) -> Iterable[ParamKey]:
        return self._values.keys()

    def values(self) -> Iterable[Any]:
        return self._values.values()

    def items(self) -> Iterable[Tuple[ParamKey, Any]]:
        return self._values.items()

    def as_dict(self) -> Dict[ParamKey, Any]:
        """Return a shallow copy of the internal mapping as a plain dict."""
        return dict(self._values)

    # --- lookup -------------------------------------------------------------------

    def get(self, key: ParamKey, default: Any = _MISSING) -> Any:
        """
        Get the value for a key.

        If default is not provided and the key is missing, KeyError is raised.
        If default is provided and the key is missing, default is returned.
        """
        if key in self._values:
            return self._values[key]
        if default is _MISSING:
            raise KeyError(f"Unknown parameter key: {key!r}")
        return default

    # --- construction helpers -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[ParamKey, Any]) -> "ParameterStore":
        """
        Construct a ParameterStore from a mapping.

        This makes a shallow copy of the mapping to avoid surprising aliasing
        if a mutable dict is passed in.
        """
        return cls(dict(data))

    @classmethod
    def from_spec_defaults(cls, spec: ParamSpec) -> "ParameterStore":
        """
        Construct a primitives-only ParameterStore from ParamSpec defaults.

        Each ParamField's ``default`` is used as the initial value for that
        key. Derived fields are intentionally omitted so the resulting store
        contains only primitive parameters; derived values can be populated
        explicitly via :func:`refresh_derived` after primitives are set.

        Defaults are canonicalized when metadata is available:

        - If a field declares a ``dtype``, defaults are coerced via
          ``jnp.asarray(..., dtype=field.dtype)``.
        - If a field declares a ``shape``, defaults are normalized to that
          shape, including broadcasting scalar defaults to vector shapes.
        - ``None`` defaults are preserved as ``None``.

        No bounds or unit validation is performed here. Use
        ``store.validate_against(spec)`` to check key consistency, and any
        higher-level validation utilities for bounds or units.

        Notes:
            This method prioritizes predictable dtypes/shapes for downstream
            JAX workflows, avoiding subtle dtype drift from Python literals.
        """
        def _canonicalize_default(key: ParamKey, field) -> Any:
            value = field.default
            # Respect None values
            if value is None:
                return None

            # return value as-is if no dtype
            if field.dtype is None:
                return value

            # coerce to correct dtype
            arr = jnp.asarray(value, dtype=field.dtype)

            if field.shape is None:
                return arr.reshape(())

            expected_shape = tuple(field.shape)
            if arr.shape == expected_shape:
                return arr
            if arr.shape == ():
                return jnp.broadcast_to(arr, expected_shape)

            expected_size = math.prod(expected_shape)
            if arr.size == expected_size:
                return arr.reshape(expected_shape)

            raise ValueError(
                f"from_spec_defaults: default for '{key}' has shape {arr.shape} "
                f"(size {arr.size}), expected shape {expected_shape}."
            )

        values: Dict[ParamKey, Any] = {}
        for key, field in spec.items():
            if field.kind == "derived":
                continue
            values[key] = _canonicalize_default(key, field)
        return cls(values)

    def refresh_derived(
        self,
        spec: ParamSpec,
        *,
        resolver=None,
        system_id: Optional[str] = None,
        include_derived: bool = True,
    ) -> "ParameterStore":
        """Return a new store with derived keys recomputed for ``spec``.

        The effective system identifier is resolved in priority order:

        1. Explicit ``system_id`` argument
        2. ``spec.system_id`` (if present)
        3. The global default system configured for the transform resolver

        Derived transform modules are lazily imported via
        :func:`dluxshera.params.transforms.ensure_registered`.
        """

        from .transforms import DEFAULT_SYSTEM_ID, ensure_registered, get_resolver

        sid = system_id or getattr(spec, "system_id", None) or DEFAULT_SYSTEM_ID

        if resolver is None:
            ensure_registered(sid)
            effective_resolver = get_resolver(sid)
        else:
            effective_resolver = resolver
            try:
                # Attempt to register built-in transforms when the system is known;
                # ignore unknown systems for custom resolvers.
                ensure_registered(sid)
            except (ValueError, ModuleNotFoundError):
                pass

        return _refresh_derived_internal(
            store=self,
            spec=spec,
            resolver=effective_resolver,
            system_id=sid,
            include_derived=include_derived,
        )


    def replace(
        self,
        updates: Optional[Mapping[ParamKey, Any]] = None,
        **extra_updates: Any,
    ) -> "ParameterStore":
        """
        Return a new ParameterStore with updated values for the given keys.

        Parameters
        ----------
        updates:
            Optional mapping from ParamKey strings to values. This is the
            recommended way to update parameters, especially for hierarchical
            keys that contain dots, e.g.:

                store = store.replace({"binary.separation_as": 12.0})

        **extra_updates:
            Additional updates passed as keyword arguments. This is only
            safe for keys that are valid Python identifiers (no dots). For
            example:

                store = store.replace(n_wavelengths=3)

            Internally this will update the key "n_wavelengths".

        Notes
        -----
        For hierarchical parameter keys such as "binary.separation_as",
        always use the mapping form (the `updates` argument). Relying on
        keyword arguments for these keys would silently replace dots with
        underscores at the call site, creating a *different* key (e.g.
        "binary_separation_mas") and leading to hard-to-debug discrepancies
        between the spec and the store.
        """
        new_values: Dict[ParamKey, Any] = dict(self._values)

        if updates is not None:
            for key, value in updates.items():
                new_values[key] = value

        for key, value in extra_updates.items():
            new_values[key] = value

        return ParameterStore(new_values)

    # --- validation against a ParamSpec ------------------------------------------

    def validate_against(
            self,
            spec: ParamSpec,
            *,
            allow_missing: bool = False,
            allow_extra: bool = False,
            allow_derived: bool = False,
            require_derived: bool = False,
            check_dtype: bool = False,
            check_shape: bool = False,
    ) -> "ParameterStore":
        """
        Validate that this store is consistent with a given ParamSpec.

        By default this checks that the store and spec have consistent keys,
        requiring all non-derived keys from the spec and rejecting derived
        keys unless explicitly allowed. Derived keys are not required by
        default. Optional dtype/shape validation can be enabled to catch
        mismatches early.

        Parameters
        ----------
        spec:
            The ParamSpec to validate against.
        allow_missing:
            If False (default), require that every key in the spec appears
            in the store. If True, missing keys are allowed.
        allow_extra:
            If False (default), require that every key in the store appears
            in the spec. If True, extra keys are allowed.
        allow_derived:
            If False (default / strict mode), derived keys declared in the
            spec are considered invalid when present in the store. Set to
            True to enable override/debug flows where derived values are
            intentionally injected and should be accepted.
        require_derived:
            If True, require that all derived keys declared in the spec are
            present in the store. This is enforced regardless of
            ``allow_missing``. Requires ``allow_derived=True``.
        check_dtype:
            If True, validate dtype *kind* (float/int/bool/complex) for keys
            present in both the store and spec when the spec declares a
            dtype. Float32/float64 differences are tolerated.
        check_shape:
            If True, validate exact shape agreement for keys present in both
            the store and spec. ``field.shape is None`` requires a scalar
            shape ``()``; otherwise the stored value must match the specified
            shape exactly. No broadcasting is applied.

        Notes
        -----
        This method checks key consistency and optionally dtype/shape
        agreement. It does not validate bounds or units.

        Raises
        ------
        ValueError
            If the store contains unknown keys (when allow_extra is False),
            contains derived keys while allow_derived is False, or is missing
            required keys (when allow_missing is False). If require_derived is
            True, missing derived keys are always considered an error.

        Returns
        -------
        ParameterStore
            Returns self to allow simple chaining.
        """
        spec_keys = set(spec.keys())
        derived_keys = {k for k, f in spec.items() if f.kind == "derived"}
        required_keys = spec_keys - derived_keys
        store_keys = set(self.keys())

        extra_keys = store_keys - spec_keys
        missing_required = required_keys - store_keys
        present_derived = store_keys & derived_keys

        if require_derived and not allow_derived:
            raise ValueError(
                "ParameterStore.validate_against: require_derived=True is "
                "incompatible with allow_derived=False."
            )

        if require_derived:
            missing_derived = derived_keys - store_keys
            if missing_derived:
                raise ValueError(
                    "ParameterStore is missing derived keys required by spec: "
                    f"{sorted(missing_derived)}"
                )

        if present_derived and not allow_derived:
            raise ValueError(
                "ParameterStore contains derived keys while allow_derived=False: "
                f"{sorted(present_derived)}"
            )

        if not allow_extra and extra_keys:
            raise ValueError(
                f"ParameterStore contains keys not present in spec: "
                f"{sorted(extra_keys)}"
            )

        if not allow_missing and missing_required:
            raise ValueError(
                f"ParameterStore is missing keys required by spec: "
                f"{sorted(missing_required)}"
            )

        if check_dtype or check_shape:
            for key, field in spec.items():
                if key not in store_keys:
                    continue
                value = self._values[key]
                if value is None:
                    continue
                arr = jnp.asarray(value)

                if check_dtype and field.dtype is not None:
                    expected_kind: Optional[str]
                    if field.dtype is float:
                        expected_kind = "float"
                    elif field.dtype is int:
                        expected_kind = "int"
                    elif field.dtype is bool:
                        expected_kind = "bool"
                    elif field.dtype is complex:
                        expected_kind = "complex"
                    else:
                        try:
                            dtype = np.dtype(field.dtype)
                        except TypeError:
                            expected_kind = None
                        else:
                            if np.issubdtype(dtype, np.floating):
                                expected_kind = "float"
                            elif np.issubdtype(dtype, np.integer):
                                expected_kind = "int"
                            elif np.issubdtype(dtype, np.bool_):
                                expected_kind = "bool"
                            elif np.issubdtype(dtype, np.complexfloating):
                                expected_kind = "complex"
                            else:
                                expected_kind = None

                    if expected_kind is not None:
                        dtype_ok = False
                        if expected_kind == "float":
                            dtype_ok = np.issubdtype(arr.dtype, np.floating)
                        elif expected_kind == "int":
                            dtype_ok = np.issubdtype(arr.dtype, np.integer)
                        elif expected_kind == "bool":
                            dtype_ok = np.issubdtype(arr.dtype, np.bool_)
                        elif expected_kind == "complex":
                            dtype_ok = np.issubdtype(arr.dtype, np.complexfloating)

                        if not dtype_ok:
                            raise ValueError(
                                "ParameterStore dtype mismatch for "
                                f"'{key}': expected {expected_kind}-like, "
                                f"got {arr.dtype}"
                            )

                if check_shape:
                    expected_shape = () if field.shape is None else tuple(field.shape)
                    if arr.shape != expected_shape:
                        raise ValueError(
                            "ParameterStore shape mismatch for "
                            f"'{key}': expected {expected_shape}, got {arr.shape}"
                        )

        return self


def validate_inference_base_store(
    base_store: ParameterStore,
    subspec: ParamSpec,
    *,
    check_shapes: bool = True,
    check_dtypes: bool = False,
) -> None:
    """Validate that a base store is compatible with an inference subspec.

    This performs presence/shape/dtype checks (as configured) for every field
    in the subspec and raises a consolidated ValueError on any mismatch.
    """

    missing_keys = []
    shape_mismatches = []
    dtype_mismatches = []

    def _shape_of(value: Any) -> Optional[Tuple[int, ...]]:
        if hasattr(value, "shape"):
            shape_attr = getattr(value, "shape")
            if shape_attr is not None:
                try:
                    return tuple(shape_attr)
                except TypeError:
                    return None
        if isinstance(value, (tuple, list)):
            return (len(value),)
        return None

    def _dtype_matches(value: Any, expected) -> bool:
        try:
            actual_dtype = getattr(value, "dtype", None)
            if actual_dtype is not None:
                try:
                    return jnp.issubdtype(actual_dtype, jnp.dtype(expected))
                except TypeError:
                    pass
            return isinstance(value, expected)
        except Exception:
            return False

    for key, field in subspec.items():
        try:
            value = base_store.get(key)
        except KeyError:
            missing_keys.append(key)
            continue

        if check_shapes and field.shape is not None:
            actual_shape = _shape_of(value)
            if actual_shape != field.shape:
                shape_mismatches.append((key, field.shape, actual_shape))

        if check_dtypes and field.dtype is not None:
            if not _dtype_matches(value, field.dtype):
                dtype_mismatches.append((key, field.dtype, getattr(value, "dtype", type(value))))

    if missing_keys or shape_mismatches or dtype_mismatches:
        parts = []
        if missing_keys:
            parts.append(f"missing keys: {sorted(missing_keys)}")
        if shape_mismatches:
            formatted = [f"{k} (expected {exp}, found {found})" for k, exp, found in shape_mismatches]
            parts.append("shape mismatches: " + "; ".join(formatted))
        if dtype_mismatches:
            formatted = [f"{k} (expected {exp}, found {found})" for k, exp, found in dtype_mismatches]
            parts.append("dtype mismatches: " + "; ".join(formatted))
        raise ValueError("; ".join(parts))


def _derived_keys(spec: ParamSpec) -> set[ParamKey]:
    return {key for key, field in spec.items() if field.kind == "derived"}


def _refresh_derived_internal(
    *,
    store: ParameterStore,
    spec: ParamSpec,
    resolver,
    system_id: str,
    include_derived: bool,
) -> ParameterStore:
    primitive_store = strip_derived(store, spec, keep_extra=True)
    values = primitive_store.as_dict()
    if include_derived:
        for key in sorted(_derived_keys(spec)):
            value = resolver.compute(key, primitive_store, system_id=system_id)
            field = spec.get(key)
            if value is not None and field.dtype is not None:
                arr = jnp.asarray(value, dtype=field.dtype)
                if field.shape is None:
                    value = arr.reshape(())
                else:
                    expected_shape = tuple(field.shape)
                    if arr.shape == expected_shape:
                        value = arr
                    elif arr.shape == ():
                        value = jnp.broadcast_to(arr, expected_shape)
                    else:
                        expected_size = math.prod(expected_shape)
                        if arr.size == expected_size:
                            value = arr.reshape(expected_shape)
                        else:
                            raise ValueError(
                                f"_refresh_derived_internal: derived value for '{key}' "
                                f"has shape {arr.shape} (size {arr.size}), expected "
                                f"shape {expected_shape}."
                            )
            values[key] = value
    return ParameterStore.from_dict(values)


def strip_derived(
    store: ParameterStore,
    spec: ParamSpec,
    *,
    keep_extra: bool = True,
) -> ParameterStore:
    """
    Return a new store with all derived keys (per `spec`) removed.

    Parameters
    ----------
    store:
        ParameterStore to strip derived keys from.
    spec:
        ParamSpec whose `kind == "derived"` fields identify which keys to drop.
    keep_extra:
        If True (default), keys not present in the spec are preserved. If False,
        only primitive keys defined in the spec are kept.
    """

    derived_keys = _derived_keys(spec)
    spec_keys = set(spec.keys())

    filtered = {}
    for key, value in store.items():
        if key in derived_keys:
            continue
        if not keep_extra and key not in spec_keys:
            continue
        filtered[key] = value

    return ParameterStore.from_dict(filtered)


def strip_structural(
    store: ParameterStore,
    *,
    structural_keys: Optional[Iterable[ParamKey]] = None,
    structural_prefixes: Tuple[str, ...] = ("system.", "band."),
) -> ParameterStore:
    """
    Return a new store with structural keys removed.

    Parameters
    ----------
    store:
        ParameterStore to strip structural keys from.
    structural_keys:
        Optional explicit set/iterable of structural keys to remove. When
        provided, these keys are removed verbatim. When omitted, keys matching
        any of ``structural_prefixes`` are treated as structural.
    structural_prefixes:
        Prefixes used to detect structural keys when ``structural_keys`` is
        not provided.
    """

    if structural_keys is None:
        prefixes = tuple(structural_prefixes)
        structural_set = {
            key
            for key in store.keys()
            if any(key.startswith(prefix) for prefix in prefixes)
        }
    else:
        structural_set = set(structural_keys)

    filtered = {key: value for key, value in store.items() if key not in structural_set}
    return ParameterStore.from_dict(filtered)


def subset_store(
    store: ParameterStore,
    keys: Iterable[ParamKey],
) -> ParameterStore:
    """
    Return a new store containing only the requested keys.

    Raises ``KeyError`` if any requested key is missing from ``store``.
    """

    selected = {key: store.get(key) for key in keys}
    return ParameterStore.from_dict(selected)


def refresh_derived(
    store: ParameterStore,
    spec: ParamSpec,
    resolver=None,
    system_id: Optional[str] = None,
    *,
    include_derived: bool = True,
) -> ParameterStore:
    """
    Recompute derived parameters for a (spec, store, system) tuple.

    This helper removes any derived keys from the input store, resolves them
    through a resolver (defaulting to the system-aware registry), and returns
    a new store that preserves primitives/extras and optionally appends
    recomputed derived values.

    A canonical forward-modelling flow is::

        spec = build_forward_model_spec_from_config(cfg)
        store = ParameterStore.from_spec_defaults(spec)   # primitives only
        store = store.replace({...truth-level primitives...})
        store = store.refresh_derived(spec)

    Parameters
    ----------
    store:
        ParameterStore containing primitives (and possibly stale deriveds).
    spec:
        ParamSpec used to identify derived keys and their transforms.
    resolver:
        Optional object providing a `compute(key, store, system_id=...)` method
        (e.g., TransformRegistry or DerivedResolver). If omitted, the resolver
        for the inferred system is used.
    system_id:
        Optional system identifier passed through to the resolver. If omitted,
        ``spec.system_id`` is used when present; otherwise the resolver's
        default system is used.
    include_derived:
        If True (default), include recomputed derived keys in the returned
        store. If False, only primitives/extras are returned.
    """

    from .transforms import DEFAULT_SYSTEM_ID, ensure_registered, get_resolver

    sid = system_id or getattr(spec, "system_id", None) or DEFAULT_SYSTEM_ID

    if resolver is None:
        ensure_registered(sid)
        effective_resolver = get_resolver(sid)
    else:
        effective_resolver = resolver
        try:
            ensure_registered(sid)
        except (ValueError, ModuleNotFoundError):
            # Allow custom system IDs when the caller supplies a resolver.
            pass

    return _refresh_derived_internal(
        store=store,
        spec=spec,
        resolver=effective_resolver,
        system_id=sid,
        include_derived=include_derived,
    )


def check_consistency(
    store: ParameterStore,
    spec: ParamSpec,
    resolver,
    system_id: str,
    *,
    keys: Optional[Iterable[ParamKey]] = None,
    atol: float = 0.0,
    rtol: float = 0.0,
    raise_on_mismatch: bool = True,
) -> Dict[ParamKey, Optional[float]]:
    """
    Compare stored derived values against recomputed ones.

    This is primarily intended for tests/debugging of override flows where
    derived values may have been manually injected. Keys missing from the
    store are skipped (recorded as None).
    """

    derived_keys = _derived_keys(spec)
    if keys is not None:
        requested = set(keys)
        derived_keys = derived_keys & requested

    primitive_store = strip_derived(store, spec, keep_extra=True)
    diffs: Dict[ParamKey, Optional[float]] = {}

    for key in sorted(derived_keys):
        if key not in store:
            diffs[key] = None
            continue

        stored_value = store.get(key)
        recomputed = resolver.compute(key, primitive_store, system_id=system_id)

        stored_arr = jnp.asarray(stored_value)
        recomputed_arr = jnp.asarray(recomputed)
        abs_diff = jnp.max(jnp.abs(stored_arr - recomputed_arr))
        scale = atol + rtol * jnp.max(jnp.abs(recomputed_arr))

        diffs[key] = float(abs_diff)

        if raise_on_mismatch and bool(abs_diff > scale):
            raise AssertionError(
                f"Derived value for {key!r} differs from recomputed value: "
                f"abs_diff={float(abs_diff)} exceeds atol={atol}, rtol={rtol}"
            )

    return diffs


# --- JAX pytree registration ------------------------------------------------------


def _store_flatten(store: ParameterStore):
    """
    JAX pytree flatten function.

    We sort keys to have a deterministic leaf ordering. The aux data carries
    the key order so we can reconstruct on unflatten.
    """
    # Deterministic key order
    keys = tuple(sorted(store._values.keys()))
    children = [store._values[k] for k in keys]
    aux_data = keys
    return children, aux_data


def _store_unflatten(aux_data, children):
    """
    JAX pytree unflatten function.
    """
    keys = aux_data
    values = dict(zip(keys, children))
    return ParameterStore(values)


jax.tree_util.register_pytree_node(
    ParameterStore,
    _store_flatten,
    _store_unflatten,
)
