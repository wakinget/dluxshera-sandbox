"""
Parameter specification types for dLuxShera.

This module defines lightweight, mostly-immutable descriptions of model
parameters (ParamField) and collections of them (ParamSpec). At this stage,
these are *schema only* and do not depend on JAX or any runtime model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple


ParamKey = str  # simple alias for clarity


@dataclass(frozen=True)
class ParamField:
    """
    Declarative description of a single model parameter.

    This is intentionally lightweight and opinionated, but we keep the
    semantics fairly loose at this stage. We can tighten/extend as the
    refactor progresses.
    """

    # Unique identifier for this parameter, e.g. "optics.psf_pixel_scale"
    key: ParamKey

    # Logical group / namespace, e.g. "imaging", "primary", "secondary", "binary"
    group: str

    # High-level kind, e.g. "primitive", "derived", "hyper", "nuisance"
    kind: str

    # Optional units string for humans (e.g. "arcsec", "nm", "pixels")
    units: Optional[str] = None

    # Expected dtype for the parameter value, e.g. float, int, jnp.ndarray
    dtype: Any = float

    # Optional expected shape; None means "scalar or unconstrained"
    shape: Optional[Tuple[int, ...]] = None

    # Optional default value (used to seed a ParameterStore, not required)
    default: Any = None

    # Optional (lower, upper) bounds in physical units; None means "unbounded"
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None

    # Placeholder for prior information (to be formalized later)
    prior: Any = None

    # Optional name of a transform in the transform registry, if applicable
    transform: Optional[str] = None

    # Other parameter keys this one depends on (for derived quantities)
    depends_on: Tuple[ParamKey, ...] = field(default_factory=tuple)

    # Human-readable description
    doc: str = ""


class ParamSpec:
    """
    Collection of ParamField objects, indexed by key.

    This is designed to be a mostly-immutable container:
      - `.add()` returns a *new* ParamSpec
      - Internals are stored in a private dict for fast lookup

    At this stage it does not know about ParameterStore; we will add
    validation helpers once the store type is defined.
    """

    def __init__(self, fields: Iterable[ParamField] = ()) -> None:
        field_dict: Dict[ParamKey, ParamField] = {}
        for f in fields:
            if f.key in field_dict:
                raise ValueError(f"Duplicate parameter key in ParamSpec: {f.key!r}")
            field_dict[f.key] = f
        self._fields: Dict[ParamKey, ParamField] = field_dict

    # --- basic container protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._fields)

    def __iter__(self) -> Iterator[ParamKey]:
        return iter(self._fields)

    def __contains__(self, key: object) -> bool:
        return key in self._fields

    def keys(self) -> Iterable[ParamKey]:
        return self._fields.keys()

    def values(self) -> Iterable[ParamField]:
        return self._fields.values()

    def items(self) -> Iterable[Tuple[ParamKey, ParamField]]:
        return self._fields.items()

    def as_dict(self) -> Mapping[ParamKey, ParamField]:
        """Return a read-only mapping view of the fields."""
        return dict(self._fields)

    # --- lookup -------------------------------------------------------------------

    def get(self, key: ParamKey) -> ParamField:
        try:
            return self._fields[key]
        except KeyError as exc:
            raise KeyError(f"Unknown parameter key: {key!r}") from exc

    # --- construction / extension -------------------------------------------------

    def add(self, field: ParamField) -> "ParamSpec":
        """
        Return a new ParamSpec with an additional field.

        If the key already exists, this raises ValueError to avoid silent
        overwrites.
        """
        if field.key in self._fields:
            raise ValueError(f"ParamSpec already contains key {field.key!r}")
        new_fields = dict(self._fields)
        new_fields[field.key] = field
        return ParamSpec(new_fields.values())

    def merge(self, other: "ParamSpec") -> "ParamSpec":
        """
        Return a new ParamSpec combining this spec with another.

        If any keys overlap, this raises ValueError. Later we could allow
        controlled overrides, but for now we keep it strict.
        """
        new_fields = dict(self._fields)
        for key, field in other.items():
            if key in new_fields:
                raise ValueError(f"ParamSpec merge conflict on key {key!r}")
            new_fields[key] = field
        return ParamSpec(new_fields.values())
