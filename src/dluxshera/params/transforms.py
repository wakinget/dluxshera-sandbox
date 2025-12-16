from __future__ import annotations

import importlib
from typing import Iterable, Optional

from .registry import (
    DerivedResolver,
    Transform,
    TransformRegistry,
    TransformCycleError,
    TransformDepthError,
    TransformError,
    TransformMissingDependencyError,
)
from .spec import ParamKey
from .store import ParameterStore

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
        module = "dluxshera.params.shera_threeplane_transforms"
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
