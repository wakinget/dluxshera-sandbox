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
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Optional, Tuple

import jax
import jax.numpy as jnp

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
        Construct a ParameterStore from the default values in a ParamSpec.

        Each ParamField's `default` attribute is used as the initial value
        for that key. Fields whose default is None are included with value
        None; it is the responsibility of the caller or model builder to
        replace these with concrete values before using the store in a
        forward model.

        Derived parameters declared in the spec are intentionally *omitted*;
        they can be populated explicitly via :func:`refresh_derived` once
        truth-level primitives have been set. This keeps the default store
        "primitives-only" and avoids accidental reliance on stale derived
        values.

        This does not perform any validation beyond copying defaults. Use
        `store.validate_against(spec)` if you want to check key consistency.
        """
        values: Dict[ParamKey, Any] = {}
        for key, field in spec.items():
            # Skip derived parameters
            if field.kind == "derived":
                continue
            values[key] = field.default
        return cls(values)


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
    ) -> "ParameterStore":
        """
        Validate that this store is consistent with a given ParamSpec.

        By default this checks only that the set of keys in the store and the
        spec match exactly and that no derived keys from the spec are present
        in the store (strict mode). Type/shape/bounds validation can be layered
        on later once we have concrete use cases.

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

        Raises
        ------
        ValueError
            If the store contains unknown keys (when allow_extra is False),
            contains derived keys while allow_derived is False, or is missing
            required keys (when allow_missing is False).

        Returns
        -------
        ParameterStore
            Returns self to allow simple chaining.
        """
        spec_keys = set(spec.keys())
        derived_keys = {k for k, f in spec.items() if f.kind == "derived"}
        store_keys = set(self.keys())

        extra_keys = store_keys - spec_keys
        missing_keys = spec_keys - store_keys
        present_derived = store_keys & derived_keys

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

        if not allow_missing and missing_keys:
            raise ValueError(
                f"ParameterStore is missing keys required by spec: "
                f"{sorted(missing_keys)}"
            )

        return self


def _derived_keys(spec: ParamSpec) -> set[ParamKey]:
    return {key for key, field in spec.items() if field.kind == "derived"}


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


def refresh_derived(
    store: ParameterStore,
    spec: ParamSpec,
    resolver,
    system_id: str,
    *,
    include_derived: bool = True,
) -> ParameterStore:
    """
    Recompute derived parameters for a (spec, store, system) tuple.

    This helper removes any derived keys from the input store, resolves them
    through the provided `resolver`, and returns a new store that preserves
    primitives/extras and optionally appends recomputed derived values.

    A canonical forward-modelling flow is::

        spec = build_forward_model_spec_from_config(cfg)
        store = ParameterStore.from_spec_defaults(spec)   # primitives only
        store = store.replace({...truth-level primitives...})
        store = refresh_derived(store, spec, TRANSFORMS, system_id)

    Parameters
    ----------
    store:
        ParameterStore containing primitives (and possibly stale deriveds).
    spec:
        ParamSpec used to identify derived keys and their transforms.
    resolver:
        Object providing a `compute(key, store, system_id=...)` method (e.g.,
        TransformRegistry or DerivedResolver).
    system_id:
        System identifier passed through to the resolver.
    include_derived:
        If True (default), include recomputed derived keys in the returned
        store. If False, only primitives/extras are returned.
    """

    primitive_store = strip_derived(store, spec, keep_extra=True)
    values = primitive_store.as_dict()
    if include_derived:
        for key in _derived_keys(spec):
            values[key] = resolver.compute(key, primitive_store, system_id=system_id)
    return ParameterStore.from_dict(values)


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
