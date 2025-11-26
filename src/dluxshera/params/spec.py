"""
Parameter specification types for dLuxShera.

This module defines lightweight, mostly-immutable descriptions of model
parameters (ParamField) and collections of them (ParamSpec). At this stage,
these are *schema only* and do not depend on JAX or any runtime model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple
from ..optics.config import SheraThreePlaneConfig

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
            key="binary.log_flux_total",
            group="binary",
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
        # System geometry
        # ----------------------
        ParamField(
            key="system.plate_scale_as_per_pix",
            group="system",
            kind="primitive",
            units="as / pixel",
            dtype=float,
            shape=None,
            default=0.355,
            bounds=(0.0, None),
            doc=(
                "System plate scale, in arcseconds per pixel. Although a "
                "three-plane optical system may determine plate scale from "
                "geometry, in inference we treat it as an effective primitive "
                "parameter. In the forward model, the corresponding knob is "
                "`system.plate_scale_as_per_pix`."
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


# ---------------------------------------------------------------------------
# Forward modelling spec builders
# ---------------------------------------------------------------------------

def build_forward_model_spec_from_config(
    cfg: SheraThreePlaneConfig,
) -> ParamSpec:
    """
    Construct a ParamSpec describing the *truth-level* forward model
    configuration for a single Shera three-plane scenario.

    This spec is separate from the inference spec:

    - ForwardModelSpec:
        Holds the physical / configuration quantities used to compute
        truth-level derived parameters like:
          * the geometric PSF plate scale, and
          * the effective total log-flux at the detector.
        Many of these fields are mirrored from SheraThreePlaneConfig.

    - Inference spec:
        Holds the effective knobs actually exposed to the optimiser
        (binary separation, effective plate scale, log-flux, Zernike coeffs).

    Typical usage
    -------------
    For truth / synthetic-data generation you might:

      1) Build this spec from a SheraThreePlaneConfig.
      2) Construct a ParameterStore from the spec defaults.
      3) Override a few imaging/binary primitives
         (exposure time, throughput, flux density).
      4) Run transforms to compute:
           - `system.plate_scale_as_per_pix`
           - `binary.log_flux_total`
      5) Copy those derived values into your inference store.

    Notes
    -----
    All fields here live under semantic groups:

      - 'system.*' : geometry and detector sampling mirrored from config
      - 'band.*'   : bandpass properties
      - 'imaging.*': observation / exposure configuration
      - 'binary.*' : binary-level flux normalisation and derived flux

    Only `system.plate_scale_as_per_pix` and `binary.log_flux_total` are
    declared as kind='derived'; all others are primitives.
    """
    fields = [
        # --- System geometry: mirrored from SheraThreePlaneConfig ----------
        ParamField(
            key="system.m1_diameter_m",
            group="system",
            kind="primitive",
            units="m",
            dtype=float,
            shape=None,
            default=cfg.m1_diameter_m,
            bounds=(0.0, None),
            doc=(
                "Primary mirror clear diameter [meters]. Mirrored directly "
                "from SheraThreePlaneConfig; used e.g. for collecting area "
                "in flux calculations."
            ),
        ),
        ParamField(
            key="system.m2_diameter_m",
            group="system",
            kind="primitive",
            units="m",
            dtype=float,
            shape=None,
            default=cfg.m2_diameter_m,
            bounds=(0.0, None),
            doc=(
                "Secondary mirror diameter [meters]. Mirrored from config. "
                "Not used in P0 flux calculations by default, but available "
                "for future collecting-area refinements (e.g. central "
                "obscuration)."
            ),
        ),
        ParamField(
            key="system.m1_focal_length_m",
            group="system",
            kind="primitive",
            units="m",
            dtype=float,
            shape=None,
            default=cfg.m1_focal_length_m,
            bounds=(0.0, None),
            doc=(
                "Primary mirror effective focal length [meters]. Mirrored "
                "from SheraThreePlaneConfig; used to compute the effective "
                "telescope focal length in the three-plane layout."
            ),
        ),
        ParamField(
            key="system.m2_focal_length_m",
            group="system",
            kind="primitive",
            units="m",
            dtype=float,
            shape=None,
            default=cfg.m2_focal_length_m,
            bounds=(None, 0.0),
            doc=(
                "Secondary mirror effective focal length [meters]. Mirrored "
                "from SheraThreePlaneConfig; usually negative for a "
                "Cassegrain-like layout."
            ),
        ),
        ParamField(
            key="system.m1_m2_separation_m",
            group="system",
            kind="primitive",
            units="m",
            dtype=float,
            shape=None,
            default=cfg.m1_m2_separation_m,
            bounds=(0.0, None),
            doc=(
                "Axial separation between M1 and M2 [meters]. Mirrored from "
                "SheraThreePlaneConfig; together with the focal lengths, this "
                "defines the effective telescope focal length."
            ),
        ),
        ParamField(
            key="system.pixel_pitch_m",
            group="system",
            kind="primitive",
            units="m",
            dtype=float,
            shape=None,
            default=cfg.pixel_pitch_m,
            bounds=(0.0, None),
            doc=(
                "Physical detector pixel pitch [meters]. Mirrored from "
                "SheraThreePlaneConfig; used with the effective focal length "
                "to derive the geometric PSF plate scale."
            ),
        ),

        # --- Bandpass ----------------------------------------------------
        ParamField(
            key="band.wavelength_m",
            group="band",
            kind="primitive",
            units="m",
            dtype=float,
            shape=None,
            default=cfg.wavelength_m,
            bounds=(0.0, None),
            doc="Central wavelength of the bandpass [meters].",
        ),
        ParamField(
            key="band.bandwidth_m",
            group="band",
            kind="primitive",
            units="m",
            dtype=float,
            shape=None,
            default=cfg.bandwidth_m,
            bounds=(0.0, None),
            doc=(
                "Approximate bandpass width [meters]. Used for a simple "
                "flux ~ flux_density * bandwidth estimate in P0. "
                "More detailed bandpass modelling can be added later."
            ),
        ),

        # --- Imaging configuration ----------------------------------
        ParamField(
            key="imaging.exposure_time_s",
            group="imaging",
            kind="primitive",
            units="s",
            dtype=float,
            shape=None,
            default=1800.0,  # 30 min nominal; override as needed
            bounds=(0.0, None),
            doc=(
                "Single-exposure integration time [seconds]. Used in the "
                "flux transform to map a flux (photons/s) to a total "
                "photon count at the detector."
            ),
        ),
        ParamField(
            key="imaging.throughput",
            group="imaging",
            kind="primitive",
            units=None,
            dtype=float,
            shape=None,
            default=1.0,
            bounds=(0.0, 1.0),
            doc=(
                "Effective end-to-end throughput efficiency (0–1), capturing "
                "optical transmission, detector QE, and any other losses "
                "not explicitly modelled. P0 default is 1.0 (no loss)."
            ),
        ),

        # --- Source flux normalisation ----------------------------------
        ParamField(
            key="binary.spectral_flux_density",
            group="binary",
            kind="primitive",
            units="ph / s / m^2 / m",
            dtype=float,
            shape=None,
            default=1.7227e17,
            bounds=(0.0, None),
            doc=(
                "Mean photon flux density from the binary at the telescope "
                "entrance pupil, in units of photons/s/m^2 per *meter* of "
                "bandwidth, averaged over the band of interest.\n\n"
                "In practice you may have tabulated values in "
                "ph/s/m^2 per micron; in that case you should convert before "
                "setting this field, e.g. flux_per_m = flux_per_um / 1e-6.\n\n"
                "The default value (1.7227e17) is taken from the Toliman "
                "master spreadsheet for Alpha Cen A+B and is suitable as a "
                "reference point, but for general targets you should override "
                "this field based on the appropriate flux calibration."
            ),
        ),

        # --- Derived forward-model quantities ---------------------------
        ParamField(
            key="system.plate_scale_as_per_pix",
            group="system",
            kind="derived",
            units="as / pixel",
            dtype=float,
            shape=None,
            default=None,
            bounds=(0.0, None),
            doc=(
                "Geometric PSF plate scale at the detector, in arcseconds "
                "per pixel, derived from the three-plane telescope layout "
                "and detector pixel pitch. This is a truth-level quantity "
                "for the forward model; in inference, the corresponding "
                "knob is `system.plate_scale_as_per_pix`."
            ),
        ),
        ParamField(
            key="binary.log_flux_total",
            group="binary",
            kind="derived",
            units="log10(photons)",
            dtype=float,
            shape=None,
            default=None,
            bounds=(None, None),
            doc=(
                "Truth-level total log10 photon count from the source over "
                "the exposure at the detector plane.\n\n"
                "Derived from the mean flux density at the pupil, the "
                "telescope collecting area, the bandpass width, the "
                "exposure time, and the throughput efficiency. This value "
                "is typically copied into the inference ParameterStore under "
                "the same key, where it is then treated as a primitive knob."
            ),
        ),
    ]

    return ParamSpec(fields)
