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

    # --- subset / view helpers ---------------------------------------------------

    def subset(self, keys: Iterable[ParamKey]) -> "ParamSpec":
        """
        Return a new ParamSpec containing only the fields whose keys are in `keys`.

        This is useful for defining different inference configurations (e.g.
        astrometry-only vs joint astrometry+flux) from a larger base spec.
        """
        selected_fields = []
        for key in keys:
            try:
                selected_fields.append(self._fields[key])
            except KeyError as exc:
                raise KeyError(f"ParamSpec.subset: unknown key {key!r}") from exc
        return ParamSpec(selected_fields)


# ---------------------------------------------------------------------------
# Shera inference spec builders
# ---------------------------------------------------------------------------

def build_inference_spec_basic() -> ParamSpec:
    """
    Construct the baseline inference parameter specification for Shera runs.

    This specification contains the core effective parameters that the current
    Shera optimization and inference pipelines typically solve for:
      • binary separation and position angle
      • binary centroid offsets (X/Y)
      • binary flux ratio (A:B)
      • total effective log-flux detected from the binary
      • detector plate scale
      • primary and secondary mirror Zernike WFE coefficients

    All parameters here are treated as *primitive* in the inference model,
    even if they may have been derived from more physical quantities when
    generating synthetic data. This allows the inference to operate directly
    in the effective parameter space used by the forward model.
    """
    fields = [

        # ----------------------
        # Binary astrometry
        # ----------------------
        ParamField(
            key="binary.separation_as",
            group="binary",
            kind="primitive",
            units="as",
            dtype=float,
            shape=None,
            default=10.0,
            bounds=(0.0, None), # must be positive
            doc="Angular separation of the binary on the sky, in arcseconds.",
        ),
        ParamField(
            key="binary.position_angle_deg",
            group="binary",
            kind="primitive",
            units="deg",
            dtype=float,
            shape=None,
            default=90.0,
            bounds=(0.0, 360.0),
            doc=(
                "Position angle of the binary on the sky, in degrees East of North"
            ),
        ),
        ParamField(
            key="binary.x_position",
            group="binary",
            kind="primitive",
            units="as",
            dtype=float,
            shape=None,
            default=0.0,
            bounds=(None, None),
            doc="Binary centroid offset in the detector X direction (arcseconds).",
        ),
        ParamField(
            key="binary.y_position",
            group="binary",
            kind="primitive",
            units="as",
            dtype=float,
            shape=None,
            default=0.0,
            bounds=(None, None),
            doc="Binary centroid offset in the detector Y direction (arcseconds).",
        ),
        ParamField(
            key="binary.contrast",
            group="binary",
            kind="primitive",
            units=None,
            dtype=float,
            shape=None,
            default=3, # Ratio of A:B
            bounds=(0.0, None),
            doc=(
                "Flux ratio of the binary system, defined as Primary:Secondary "
                "(A:B). A ratio > 1 indicates the primary is brighter."
            ),
        ),

        # ----------------------
        # Source photometry
        # ----------------------
        ParamField(
            key="source.log_flux_total",
            group="source",
            kind="primitive",
            units="log10(photons)",
            dtype=float,
            shape=None,
            default=8.0,  # rough ballpark; tune later
            bounds=(None, None),
            doc=(
                "Effective total log10 photon count from both stars reaching "
                "the detector over the exposure. Treated as a primitive "
                "brightness/throughput parameter in inference."
            ),
        ),

        # ----------------------
        # Detector geometry
        # ----------------------
        ParamField(
            key="imaging.plate_scale_as_per_pix",
            group="imaging",
            kind="primitive",
            units="as / pixel",
            dtype=float,
            shape=None,
            default=0.355,
            bounds=(0.0, None),
            doc=(
                "Detector plate scale, in arcseconds per pixel. Although a "
                "three-plane optical system may determine plate scale from "
                "geometry, in inference we treat it as an effective primitive "
                "parameter."
            ),
        ),

        # ----------------------
        # Optical wavefront errors
        # ----------------------
        ParamField(
            key="primary.zernike_coeffs",
            group="primary",
            kind="primitive",
            units="nm",
            dtype=float,
            shape=None, # ← arbitrary length allowed
            default=None,
            bounds=(None, None),
            doc=(
                "Primary mirror Zernike wavefront error coefficients (nm). "
                "Length is variable and validated by the model builder."
            ),
        ),
        ParamField(
            key="secondary.zernike_coeffs",
            group="secondary",
            kind="primitive",
            units="nm",
            dtype=float,
            shape=None, # ← arbitrary length allowed
            default=None,
            bounds=(None, None),
            doc=(
                "Secondary mirror Zernike wavefront error coefficients (nm). "
                "Length is variable and validated by the model builder."
            ),
        ),
    ]

    return ParamSpec(fields)
