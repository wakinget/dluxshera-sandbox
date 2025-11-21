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

      "binary.separation_mas"
      "imaging.psf_pixel_scale"
      "noise.jitter_rms_mas"

  These keys are part of the public "parameter API" of the model: they are
  used consistently in specs, priors, configs, logging, etc.

- Because keys may contain dots and other characters that are not valid
  Python identifiers, we cannot safely rely on keyword arguments for
  updating them. For example:

      store.replace(binary_separation_mas=120.0)

  would create/modify a key called "binary_separation_mas", which is *not*
  the same as the canonical key "binary.separation_mas". This kind of
  mismatch is very easy to introduce and hard to debug.

- To avoid this, ParameterStore.replace() accepts a mapping of literal
  keys to values (e.g. replace({"binary.separation_mas": 120.0})), and only
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

from .spec import ParamKey


# Sentinel for "no default provided"
_MISSING = object()


@dataclass(frozen=True)
class ParameterStore:
    """
    Immutable-ish container mapping parameter keys to values.

    Values are typically Python floats/ints or jax.numpy arrays. Keys are
    ParamKey strings and may contain dots (e.g. "binary.separation_mas").

    Because keys are arbitrary strings, updates should generally be passed
    as an explicit mapping to `replace()`, e.g.:

        store = store.replace({"binary.separation_mas": 120.0})

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

                store = store.replace({"binary.separation_mas": 120.0})

        **extra_updates:
            Additional updates passed as keyword arguments. This is only
            safe for keys that are valid Python identifiers (no dots). For
            example:

                store = store.replace(n_wavelengths=3)

            Internally this will update the key "n_wavelengths".

        Notes
        -----
        For hierarchical parameter keys such as "binary.separation_mas",
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


    # --- (future) validation hook -------------------------------------------------

    def validate(self, spec: "ParamSpec") -> "ParameterStore":
        """
        Placeholder for validation against a ParamSpec.

        For now, this is a no-op that simply returns self. Later we can:
        - check all keys exist in spec
        - check dtype / shape / bounds, etc.
        """
        # from .spec import ParamSpec  # delayed import to avoid cycles
        # TODO: implement real validation.
        return self


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
