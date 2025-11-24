from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .spec import ParamKey
from .store import ParameterStore


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class TransformError(Exception):
    """Base class for all transform-related errors."""


class TransformMissingDependencyError(TransformError):
    """
    Raised when a parameter cannot be resolved from either the store or
    the registered transforms.

    This can occur for top-level requests (no way to produce the key) or
    for dependencies of other transforms.
    """

    def __init__(self, key: ParamKey, requested_by: Optional[ParamKey] = None):
        self.key = key
        self.requested_by = requested_by

        if requested_by is None:
            msg = (
                f"Cannot resolve parameter {key!r}: no value in store and no "
                f"registered transform."
            )
        else:
            msg = (
                f"Cannot resolve dependency {key!r} required by transform for "
                f"{requested_by!r}: no value in store and no registered transform."
            )
        super().__init__(msg)


class TransformCycleError(TransformError):
    """
    Raised when a cycle is detected in the transform dependency graph.
    """

    def __init__(self, key: ParamKey, stack: Iterable[ParamKey]):
        self.key = key
        self.stack = tuple(stack)
        cycle_path = " -> ".join(list(self.stack) + [key])
        msg = (
            f"Detected cyclic dependency while resolving {key!r}. "
            f"Path: {cycle_path}"
        )
        super().__init__(msg)


class TransformDepthError(TransformError):
    """
    Raised when the recursion depth exceeds the maximum allowed depth.
    """

    def __init__(self, key: ParamKey, max_depth: int, stack: Iterable[ParamKey]):
        self.key = key
        self.max_depth = max_depth
        self.stack = tuple(stack)
        msg = (
            f"Maximum transform recursion depth ({max_depth}) exceeded while "
            f"resolving {key!r}. Current stack: {self.stack}"
        )
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Transform + Registry
# ---------------------------------------------------------------------------

TransformFn = Callable[[Mapping[ParamKey, Any]], Any]


@dataclass(frozen=True)
class Transform:
    """
    A single parameter transform.

    Parameters
    ----------
    key:
        The parameter key computed by this transform.
    depends_on:
        A sequence of parameter keys that this transform depends on. These
        keys may correspond to primitive values in the ParameterStore or to
        other derived parameters with registered transforms.
    fn:
        A pure function that takes a mapping from dependency keys to their
        resolved values and returns the value for `key`.
    doc:
        Optional human-readable description of the transform.
    """

    key: ParamKey
    depends_on: Tuple[ParamKey, ...]
    fn: TransformFn
    doc: Optional[str] = None


class TransformRegistry:
    """
    Registry mapping parameter keys to their corresponding transforms.

    The registry can resolve derived parameters recursively, using values from
    a ParameterStore as primitives. Resolution obeys the following rules:

      * If a key is present in the store, that value is returned directly and
        any transform registered for that key is ignored (stored values take
        precedent).
      * Otherwise, if a transform is registered for the key, its dependencies
        are resolved (recursively) and the transform function is evaluated.
      * If neither the store nor the registry can provide a value for a key,
        TransformMissingDependencyError is raised.
      * Cycles in the dependency graph are detected and raise
        TransformCycleError.
      * Recursion depth is limited by `max_depth` to avoid pathological cases.
    """

    def __init__(self) -> None:
        self._transforms: Dict[ParamKey, Transform] = {}

    # --- registration / lookup ---------------------------------------------------

    def register(self, transform: Transform) -> None:
        """
        Register a new transform.

        Raises
        ------
        ValueError
            If a transform is already registered for the same key.
        """
        key = transform.key
        if key in self._transforms:
            raise ValueError(
                f"A transform is already registered for key {key!r}."
            )
        self._transforms[key] = transform

    def has(self, key: ParamKey) -> bool:
        """Return True if a transform is registered for `key`."""
        return key in self._transforms

    def get(self, key: ParamKey) -> Transform:
        """Return the transform registered for `key`."""
        return self._transforms[key]

    # --- resolution --------------------------------------------------------------

    def compute(
        self,
        key: ParamKey,
        store: ParameterStore,
        *,
        max_depth: int = 32,
    ) -> Any:
        """
        Resolve the value for `key` given a ParameterStore.

        Parameters
        ----------
        key:
            The parameter key to resolve.
        store:
            ParameterStore providing primitive values and any manual overrides
            for derived parameters.
        max_depth:
            Maximum allowed recursion depth. If the dependency chain exceeds
            this depth, TransformDepthError is raised.

        Returns
        -------
        Any
            The resolved value for `key`.

        Raises
        ------
        TransformMissingDependencyError
            If `key` (or one of its dependencies) cannot be resolved from
            either the store or the registry.
        TransformCycleError
            If a cycle is detected in the transform dependency graph.
        TransformDepthError
            If recursion exceeds `max_depth`.
        """
        cache: Dict[ParamKey, Any] = {}
        stack: list[ParamKey] = []

        def _resolve(k: ParamKey, requested_by: Optional[ParamKey] = None) -> Any:
            # If we have already computed this derived value, reuse it.
            if k in cache:
                return cache[k]

            # Direct value from the store always wins.
            try:
                value = store.get(k)
            except KeyError:
                value_in_store = False
            else:
                value_in_store = True

            if value_in_store:
                cache[k] = value
                return value

            # No direct value; try to resolve via transform.
            if k in stack:
                # Cycle detected: current path plus this key.
                raise TransformCycleError(k, stack)

            if len(stack) >= max_depth:
                raise TransformDepthError(k, max_depth, stack)

            transform = self._transforms.get(k)
            if transform is None:
                # Nothing in store, nothing in registry: missing dependency.
                raise TransformMissingDependencyError(k, requested_by=requested_by)

            stack.append(k)
            try:
                # Resolve dependencies first.
                ctx: Dict[ParamKey, Any] = {}
                for dep in transform.depends_on:
                    ctx[dep] = _resolve(dep, requested_by=k)

                # Evaluate transform function.
                result = transform.fn(ctx)
                cache[k] = result
            finally:
                stack.pop()

            return result

        return _resolve(key, requested_by=None)
