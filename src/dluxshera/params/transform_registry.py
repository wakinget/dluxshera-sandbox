from __future__ import annotations

import importlib
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

    def __init__(
        self,
        key: ParamKey,
        requested_by: Optional[ParamKey] = None,
        *,
        system_id: Optional[str] = None,
    ):
        self.key = key
        self.requested_by = requested_by
        self.system_id = system_id

        system_fragment = "" if system_id is None else f" in system {system_id!r}"

        if requested_by is None:
            msg = (
                f"Cannot resolve parameter {key!r}{system_fragment}: no value in "
                "store and no registered transform."
            )
        else:
            msg = (
                f"Cannot resolve dependency {key!r} required by transform for {requested_by!r}"
                f"{system_fragment}: no value in store and no registered transform."
            )
        super().__init__(msg)


class TransformCycleError(TransformError):
    """
    Raised when a cycle is detected in the transform dependency graph.
    """

    def __init__(
        self,
        key: ParamKey,
        stack: Iterable[ParamKey],
        *,
        system_id: Optional[str] = None,
    ):
        self.key = key
        self.stack = tuple(stack)
        self.system_id = system_id
        system_fragment = "" if system_id is None else f" in system {system_id!r}"
        cycle_path = " -> ".join(list(self.stack) + [key])
        msg = (
            f"Detected cyclic dependency while resolving {key!r}{system_fragment}. "
            f"Path: {cycle_path}"
        )
        super().__init__(msg)


class TransformDepthError(TransformError):
    """
    Raised when the recursion depth exceeds the maximum allowed depth.
    """

    def __init__(
        self,
        key: ParamKey,
        max_depth: int,
        stack: Iterable[ParamKey],
        *,
        system_id: Optional[str] = None,
    ):
        self.key = key
        self.max_depth = max_depth
        self.stack = tuple(stack)
        self.system_id = system_id
        system_fragment = "" if system_id is None else f" in system {system_id!r}"
        msg = (
            f"Maximum transform recursion depth ({max_depth}) exceeded while resolving {key!r}{system_fragment}. "
            f"Current stack: {self.stack}"
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
        system_id: Optional[str] = None,
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
                raise TransformCycleError(k, stack, system_id=system_id)

            if len(stack) >= max_depth:
                raise TransformDepthError(k, max_depth, stack, system_id=system_id)

            transform = self._transforms.get(k)
            if transform is None:
                # Nothing in store, nothing in registry: missing dependency.
                raise TransformMissingDependencyError(
                    k, requested_by=requested_by, system_id=system_id
                )

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


# ---------------------------------------------------------------------------
# Scoped resolver
# ---------------------------------------------------------------------------


class DerivedResolver:
    """Manage per-system transform registries and resolve derived parameters."""

    def __init__(self, *, default_system_id: str = "default") -> None:
        self._registries: Dict[str, TransformRegistry] = {}
        self.default_system_id = default_system_id

    # --- registry access ---------------------------------------------------------

    def get_registry(self, system_id: Optional[str] = None) -> TransformRegistry:
        sid = system_id or self.default_system_id
        if sid not in self._registries:
            self._registries[sid] = TransformRegistry()
        return self._registries[sid]

    # --- registration helpers ----------------------------------------------------

    def register_transform(
        self,
        key: ParamKey,
        *,
        depends_on: Iterable[ParamKey] = (),
        doc: Optional[str] = None,
        system_id: Optional[str] = None,
    ):
        """Decorator for registering a transform under a specific system."""

        deps = tuple(depends_on)
        sid = system_id or self.default_system_id

        def decorator(fn: TransformFn) -> TransformFn:
            transform = Transform(
                key=key,
                depends_on=deps,
                fn=fn,
                doc=doc or fn.__doc__,
            )
            registry = self.get_registry(sid)
            registry.register(transform)
            return fn

        return decorator

    def register(
        self,
        transform: Transform,
        *,
        system_id: Optional[str] = None,
    ) -> None:
        registry = self.get_registry(system_id)
        registry.register(transform)

    # --- resolution --------------------------------------------------------------

    def has(self, key: ParamKey, *, system_id: Optional[str] = None) -> bool:
        registry = self.get_registry(system_id)
        return registry.has(key)

    def compute(
        self,
        key: ParamKey,
        store: ParameterStore,
        *,
        max_depth: int = 32,
        system_id: Optional[str] = None,
    ) -> Any:
        registry = self.get_registry(system_id)
        return registry.compute(
            key,
            store,
            max_depth=max_depth,
            system_id=system_id or self.default_system_id,
        )


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


# Default system ID preserves backward-compatibility with the Shera three-plane
# transforms previously registered in the global registry.
DEFAULT_SYSTEM_ID = "shera_threeplane"

# Scoped resolver instance that all modules should use.
DERIVED_RESOLVER = DerivedResolver(default_system_id=DEFAULT_SYSTEM_ID)

# Track which system-specific transform modules have been imported.
_REGISTERED_SYSTEMS: set[str] = set()


def ensure_registered(system_id: Optional[str]) -> None:
    """Lazily import system-specific transforms once per system ID."""

    sid = system_id or DEFAULT_SYSTEM_ID
    if sid in _REGISTERED_SYSTEMS:
        return

    if sid == "shera_threeplane":
        module = "dluxshera.params.transforms"
    elif sid == "shera_twoplane":
        module = "dluxshera.params.shera_twoplane_transforms"
    else:
        raise ValueError(f"Unknown system_id {sid!r} for transform registration")

    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing transform module {module!r} for system_id {sid!r}"
        ) from exc

    _REGISTERED_SYSTEMS.add(sid)


def get_resolver(system_id: Optional[str] = None) -> TransformRegistry:
    """Return the TransformRegistry for the given system_id (lazy-created)."""

    ensure_registered(system_id)
    return DERIVED_RESOLVER.get_registry(system_id)


def resolve_derived(
    key: ParamKey,
    store: ParameterStore,
    *,
    system_id: Optional[str] = None,
    max_depth: int = 32,
):
    """Resolve a derived parameter for the requested system."""

    ensure_registered(system_id)
    return DERIVED_RESOLVER.compute(
        key,
        store,
        max_depth=max_depth,
        system_id=system_id,
    )


def register_transform(
    key: ParamKey,
    *,
    depends_on: Iterable[ParamKey] = (),
    doc: Optional[str] = None,
    system_id: Optional[str] = None,
):
    """
    Convenience decorator for registering a transform function.

    Parameters
    ----------
    system_id:
        Identifier for the system this transform belongs to. Defaults to the
        resolver's default system, which currently corresponds to the Shera
        three-plane configuration.
    """

    return DERIVED_RESOLVER.register_transform(
        key,
        depends_on=depends_on,
        doc=doc,
        system_id=system_id,
    )


# Backward-compatible global registry alias for the default system.
TRANSFORMS = get_resolver()

__all__ = [
    "Transform",
    "TransformRegistry",
    "TransformError",
    "TransformMissingDependencyError",
    "TransformCycleError",
    "TransformDepthError",
    "DerivedResolver",
    "DERIVED_RESOLVER",
    "DEFAULT_SYSTEM_ID",
    "ensure_registered",
    "register_transform",
    "get_resolver",
    "resolve_derived",
    "TRANSFORMS",
]
