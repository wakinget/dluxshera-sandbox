"""Three-plane Shera system definitions (binder, config, and presets)."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.resources as resources
from pathlib import Path
from typing import Optional, Tuple

import dLux as dl
import jax.numpy as jnp

from .base import BaseConfig, BaseSheraBinder
from ..builders.source import build_alpha_cen_source
from ..params.spec import ParamField, ParamSpec
from ..params.store import ParameterStore

SHERA_THREEPLANE_SYSTEM_ID = "shera_threeplane"


@dataclass(frozen=True)
class SheraThreePlaneConfig(BaseConfig):
    """
    Structural configuration for the Shera three-plane optical system.

    This dataclass collects *non-inferred* parameters that define the shape
    and layout of the optical model. These values:
      - control grid sizes, wavelength sampling, and geometry,
      - determine which Zernike bases exist on each surface,
      - are intended to remain fixed for a given run / instrument setup.

    They are separate from the ParameterStore/ParamSpec, which handle the
    *numeric state* of an inference run (binary separation, flux, Zernike
    coefficients, etc.).

    Note: Zernike bases are specified using Noll indices stored as immutable
    Python tuples of integers (see `primary_noll_indices` and
    `secondary_noll_indices`). Using tuples keeps this configuration layer
    backend-agnostic (no NumPy/JAX arrays), hashable, and safe as a frozen
    dataclass, while still being easy for the builder to consume when
    constructing Zernike `BasisOptic` layers.

    Structural hashing and caching
    ------------------------------
    The optics builder treats the following fields as *structural* when
    computing a hash for caching purposes:

      - Grid and sampling: `pupil_npix`, `psf_npix`, `oversample`.
      - Bandpass sampling (affects wavelength grid shapes elsewhere):
        `wavelength_m`, `bandwidth_m`, `n_lambda`.
      - Three-plane geometry: `m1_diameter_m`, `m2_diameter_m`,
        `m1_focal_length_m`, `m2_focal_length_m`, `m1_m2_separation_m`,
        `pixel_pitch_m`.
      - Aperture features: `n_struts`, `strut_width_m`, `strut_rotation_deg`.
      - Zernike basis selection: `primary_noll_indices`,
        `secondary_noll_indices` (coefficients live in the ParameterStore and
        are *not* part of the structural hash).
      - Diffractive pupil: `diffractive_pupil_path`,
        `dp_design_wavelength_m`.

    Metadata such as `design_name` is intentionally *not* included in the
    structural hash so that different labels can reuse the same cached optics
    structure.
    """

    design_name: Optional[str] = None
    """
    A human-readable identifier for this optical design.

    This has no effect on the optical model itself — it is purely metadata
    used for bookkeeping, logging, versioning, reproducibility, and selecting
    between different point designs (e.g., 'shera_testbed', 'shera_flight').
    """

    # ------------------------------------------------------------------
    # Pupil & PSF grids
    # ------------------------------------------------------------------
    pupil_npix: int = 256
    """Number of pixels across the pupil grid."""

    psf_npix: int = 256
    """Number of pixels across the detector/PSF cutout."""

    oversample: int = 3
    """PSF oversampling factor relative to the on-sky plate scale."""

    # ------------------------------------------------------------------
    # Wavelength sampling
    # ------------------------------------------------------------------
    wavelength_m: float = 550e-9
    """Central wavelength of the bandpass [meters]."""

    bandwidth_m: float = 110e-9
    """Width of the bandpass [meters]."""

    n_lambda: int = 3
    """Number of discrete wavelengths to sample across the bandpass."""

    # ------------------------------------------------------------------
    # System geometry (three-plane layout)
    # ------------------------------------------------------------------
    m1_diameter_m: float = 0.09
    """Primary mirror clear diameter [meters]."""

    m2_diameter_m: float = 0.025
    """Secondary mirror clear diameter [meters]."""

    m1_focal_length_m: float = 0.35796
    """Effective focal length of the primary mirror [meters]."""

    m2_focal_length_m: float = -0.041935
    """Effective focal length of the secondary mirror [meters]."""

    m1_m2_separation_m: float = 0.320
    """Axial separation between primary and secondary [meters]."""

    pixel_pitch_m: float = 6.5e-6
    """
    Physical pixel pitch of the detector [meters].

    In three-plane geometric transforms this, together with the effective
    focal length, determines the PSF plate scale.
    """

    # ------------------------------------------------------------------
    # Aperture features (spiders, etc.)
    # ------------------------------------------------------------------
    n_struts: int = 4
    """Number of support struts in the primary aperture."""

    strut_width_m: float = 0.002
    """
    Width of the support struts [meters].
    """

    strut_rotation_deg: float = 0.0
    """Rotation angle of the spider pattern [degrees]."""

    # ------------------------------------------------------------------
    # Zernike basis selection (structure, not coefficients)
    # ------------------------------------------------------------------
    primary_noll_indices: Tuple[int, ...] = ()
    """
    Noll indices defining the Zernike basis on the primary mirror.

    If this tuple is empty, the builder should *not* construct a Zernike
    BasisOptic for the primary. If non-empty, the builder will construct a
    basis using these indices and will expect `primary.zernike_coeffs_nm` in
    the ParameterStore to have matching length. Forward-model specs constructed
    from this config default the coefficients to a zero vector (no-aberration
    case) whenever a basis is present.
    """

    secondary_noll_indices: Tuple[int, ...] = ()
    """
    Noll indices defining the Zernike basis on the secondary mirror.

    Same convention as `primary_noll_indices`: an empty tuple means no
    Zernike basis is constructed for the secondary; a non-empty tuple means
    the builder constructs a BasisOptic and expects compatible coefficients
    under `secondary.zernike_coeffs_nm`. Forward-model specs built from this
    config seed those coefficients with a zero vector when the basis exists.
    """

    # ------------------------------------------------------------------
    # Diffractive pupil / fixed masks
    # ------------------------------------------------------------------
    diffractive_pupil_path: Optional[str] = None
    """
    Optional filesystem path to a diffractive pupil mask (e.g. a .npy file).

    If `diffractive_pupil_path` is None, no diffractive pupil is applied.

    Otherwise, interpretation depends on `dp_design_wavelength_m`:

    - If `dp_design_wavelength_m` is not None:
        The file is assumed to contain a dimensionless design pattern
        `P(x, y)` in [0, 1] representing a *normalized phase* over the
        interval [0, π] radians at the design wavelength. Specifically,

            P = 0 → 0 radians of phase
            P = 1 → π radians of phase

        at `dp_design_wavelength_m`. The builder will convert this to an
        OPD map in meters by computing the corresponding phase and mapping
        onto OPD via dlu.phase2opd. The resulting operation is:

            OPD_m = P * π * dp_design_wavelength_m / (2π).

        Equivalent to:

            OPD_m = P / 2 * dp_design_wavelength_m

        so P = 1 corresponds to half a wave of OPD at the design
        wavelength.

    - If `dp_design_wavelength_m` is None:
        The file is assumed to already encode an OPD map in meters on the
        same pupil grid as the primary and is applied directly without
        further scaling.

    """

    dp_design_wavelength_m: Optional[float] = None
    """
    Design wavelength for the provided diffractive pupil mask [meters].

    If this is not None, the diffractive pupil file is interpreted as a
    normalized phase pattern `P(x, y)` in [0, 1] spanning [0, π] radians
    at this wavelength:

        P = 0 → 0 radians of phase
        P = 1 → π radians of phase

    The builder converts this to an OPD map in meters by computing

        OPD_m = P * π * dp_design_wavelength_m / (2π)
              = 0.5 * P * dp_design_wavelength_m,

    so P = 1 corresponds to half a wave of OPD at `dp_design_wavelength_m`.

    If `dp_design_wavelength_m` is None, the diffractive pupil file is
    assumed to already encode an OPD map in meters on the primary pupil
    grid and is applied directly without further scaling.
    """


def build_forward_spec_from_config(cfg: SheraThreePlaneConfig) -> ParamSpec:
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
        It also carries the full unit-aware binary astrometry vocabulary
        (centroid, separation, position angle, contrast) and optionally
        Zernike coefficient arrays whose lengths follow the configured
        Noll index tuples (defaulting to zero vectors when present).

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

        # --- Binary astrometry and photometry ---------------------------------
        ParamField(
            key="binary.x_position_as",
            group="binary",
            kind="primitive",
            units="as",
            dtype=float,
            shape=None,
            default=0.0,
            bounds=(None, None),
            doc=(
                "On-sky X position of the binary system centroid in arcseconds. "
                "Positive values follow the detector X axis (typically increasing "
                "to the right)."
            ),
        ),
        ParamField(
            key="binary.y_position_as",
            group="binary",
            kind="primitive",
            units="as",
            dtype=float,
            shape=None,
            default=0.0,
            bounds=(None, None),
            doc=(
                "On-sky Y position of the binary system centroid in arcseconds. "
                "Positive values follow the detector Y axis (typically increasing "
                "upward)."
            ),
        ),
        ParamField(
            key="binary.separation_as",
            group="binary",
            kind="primitive",
            units="as",
            dtype=float,
            shape=None,
            default=10.0,
            bounds=(0.0, None),
            doc=(
                "Angular separation between primary and secondary components in "
                "arcseconds."
            ),
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
                "Position angle of the secondary relative to the primary, in "
                "degrees East of North."
            ),
        ),
        ParamField(
            key="binary.contrast",
            group="binary",
            kind="primitive",
            units=None,
            dtype=float,
            shape=None,
            default=3.0,
            bounds=(0.0, None),
            doc=(
                "Flux ratio of the binary system, defined as Primary:Secondary "
                "(A:B). A ratio > 1 indicates the primary is brighter."
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
            key="system.focal_length_m",
            group="system",
            kind="derived",
            units="m",
            dtype=float,
            shape=None,
            default=None,
            bounds=(0.0, None),
            transform="system_focal_length_m",
            depends_on=(
                "system.m1_focal_length_m",
                "system.m2_focal_length_m",
                "system.m1_m2_separation_m",
            ),
            doc=(
                "Effective telescope focal length [meters] for the Shera three-plane "
                "two-mirror relay.\n\n"
                "Computed from the primary and secondary focal lengths and their "
                "axial separation via\n\n"
                "    1 / f_eff = 1 / f1 + 1 / f2 - sep / (f1 * f2)\n\n"
                "This matches the relation used in SheraThreePlaneOptics and is used "
                "to derive the geometric plate scale at the detector."
            ),
        ),
        ParamField(
            key="system.plate_scale_as_per_pix",
            group="system",
            kind="derived",
            units="as / pixel",
            dtype=float,
            shape=None,
            default=None,
            bounds=(0.0, None),
            transform="system_plate_scale_as_per_pix",
            depends_on=(
                "system.focal_length_m",
                "system.pixel_pitch_m",
            ),
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
            transform="binary_log_flux_total",
            depends_on=(
                "system.m1_diameter_m",
                "band.bandwidth_m",
                "imaging.exposure_time_s",
                "imaging.throughput",
                "binary.spectral_flux_density",
            ),
            doc=(
                "Truth-level total log10 photon count from the binary over the "
                "exposure at the detector plane.\n\n"
                "Derived from the mean photon flux density at the pupil, the "
                "telescope collecting area (currently assuming a clear circular "
                "aperture), the bandpass width, the exposure time, and the "
                "throughput efficiency. This value is typically copied into the "
                "inference ParameterStore under the same key, where it is then "
                "treated as a primitive knob."
            ),
        ),
        ParamField(
            key="binary.raw_fluxes",
            group="binary",
            kind="derived",
            units="photons",
            dtype=float,
            shape=(2,),
            default=None,
            bounds=(0.0, None),
            transform="binary_raw_fluxes",
            depends_on=(
                "binary.log_flux_total",
                "binary.contrast",
            ),
            doc=(
                "Raw integrated fluxes for the binary components (photons for "
                "stars A and B). Derived from total log flux and contrast using "
                "the AlphaCen source convention."
            ),
        ),
    ]

    if cfg.primary_noll_indices:
        fields.append(
            ParamField(
                key="primary.zernike_coeffs_nm",
                group="primary",
                kind="primitive",
                units="nm",
                dtype=float,
                shape=(len(cfg.primary_noll_indices),),
                default=tuple(0.0 for _ in cfg.primary_noll_indices),
                bounds=(None, None),
                doc=(
                    "Primary mirror Zernike WFE coefficients (nm). Length matches "
                    "the configured primary_noll_indices tuple; defaults to a zero "
                    "vector for the no-aberration case."
                ),
            )
        )

    if cfg.secondary_noll_indices:
        fields.append(
            ParamField(
                key="secondary.zernike_coeffs_nm",
                group="secondary",
                kind="primitive",
                units="nm",
                dtype=float,
                shape=(len(cfg.secondary_noll_indices),),
                default=tuple(0.0 for _ in cfg.secondary_noll_indices),
                bounds=(None, None),
                doc=(
                    "Secondary mirror Zernike WFE coefficients (nm). Length matches "
                    "the configured secondary_noll_indices tuple; defaults to a zero "
                    "vector for the no-aberration case."
                ),
            )
        )

    return ParamSpec(fields, system_id=SHERA_THREEPLANE_SYSTEM_ID)


def default_diffractive_pupil_path() -> str:
    """Resolve the default diffractive pupil path as a string."""
    try:
        return str(resources.files("dluxshera.data") / "diffractive_pupil.npy")
    except Exception:
        return str(Path(__file__).resolve().parents[1] / "data" / "diffractive_pupil.npy")


# ---------------------------------------------------------------------
# Named point designs
# ---------------------------------------------------------------------

# Define the path to the default diffractive pupil file
DEFAULT_DP_PATH = default_diffractive_pupil_path()

SHERA_TESTBED_CONFIG = SheraThreePlaneConfig(
    design_name="shera_testbed",

    # --- system geometry ---
    m1_diameter_m=0.09,
    m2_diameter_m=0.025,
    m1_focal_length_m=0.35796,
    m2_focal_length_m=-0.041935,
    m1_m2_separation_m=0.320,
    pixel_pitch_m=6.5e-6,

    # --- grids & sampling ---
    pupil_npix=256,
    psf_npix=256,
    oversample=1,
    wavelength_m=550e-9,
    bandwidth_m=110e-9,
    n_lambda=3,

    # --- spiders / obscurations ---
    n_struts=4,
    strut_width_m=0.002,
    strut_rotation_deg=45.0,

    # --- Zernike basis structure ---
    # define Noll indices as an immutable Python tuple.
    primary_noll_indices=tuple(range(4, 12)),
    secondary_noll_indices=tuple(range(4, 12)),

    # --- diffractive pupil ---
    diffractive_pupil_path=DEFAULT_DP_PATH,
    dp_design_wavelength_m=550e-9,
)


SHERA_FLIGHT_CONFIG = SheraThreePlaneConfig(
    design_name="shera_flight",

    # --- system geometry ---
    m1_diameter_m=0.22,
    m2_diameter_m=0.025,
    m1_focal_length_m=0.604353,
    m2_focal_length_m=-0.0545,
    m1_m2_separation_m=0.55413,
    pixel_pitch_m=4.6e-6,

    # --- grids & sampling ---
    pupil_npix=256,
    psf_npix=256,
    oversample=1,
    wavelength_m=550e-9,
    bandwidth_m=41e-9,
    n_lambda=3,

    # --- spiders / obscurations ---
    n_struts=3,
    strut_width_m=0.002,
    strut_rotation_deg=-90.0,

    # --- Zernike basis structure ---
    # define Noll indices as an immutable Python tuple.
    primary_noll_indices=tuple(range(4, 12)),
    secondary_noll_indices=tuple(range(4, 12)),

    diffractive_pupil_path=DEFAULT_DP_PATH,
    dp_design_wavelength_m=550e-9,
)


@dataclass
class SheraThreePlaneBinder(BaseSheraBinder):
    """
    Canonical generative model for the Shera three-plane system.

    Binder is the successor to the legacy ``SheraThreePlane_Model`` facade and
    is intentionally treated as **mostly immutable**: instantiate it once for a
    given configuration + base forward store (with deriveds populated), then use
    ``.model(store_delta)`` to evaluate PSFs without mutating internal state.

    Key properties
    --------------
    - Holds the Shera config, forward ParamSpec, and a *forward-style* base
      ParameterStore (derived values already populated).
    - ``.model()`` is the primary API and is intentionally lightweight: with
      ``store_delta=None`` it fast-paths through the cached telescope. For
      non-structural overlays it merges ``store_delta`` onto the base store,
      then evaluates the direct builder path. Structural overrides require
      ``allow_rebuild=True`` and delegate to ``update_store()``.
    - ``.update_store()`` returns a new binder instance with the refreshed base
      store; the original binder remains unchanged.

    The ``with_store`` attribute is an alias of
    :meth:`BaseSheraBinder.with_store`, provided for parity with legacy APIs.
    It preserves the binder's immutable-style semantics by always returning a
    fresh binder instance rather than mutating in-place.
    """

    cfg: SheraThreePlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore

    def __init__(
        self,
        cfg: SheraThreePlaneConfig,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
    ) -> None:
        """Construct a binder for the three-plane Shera configuration.

        Parameters
        ----------
        cfg : SheraThreePlaneConfig
            Fully prepared Shera three-plane configuration. Any derived
            configuration values needed by the optics/source builders should
            already be present on this object.
        forward_spec : ParamSpec
            Parameter specification describing the full forward store,
            including structural keys and derived entries.
        base_forward_store : ParameterStore
            Forward-style base store with derived values populated. The store
            is validated against ``forward_spec`` and treated as immutable
            baseline state for subsequent evaluations.
        """
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        """Evaluate the Shera three-plane model directly.

        Uses the three-plane optics and alpha Cen source builders to assemble
        a fresh telescope from ``eff_store`` and returns the PSF model output.
        This path is used when a non-structural store overlay is supplied.
        """
        return self._build_telescope(eff_store).model()

    def _build_optics(self, store: ParameterStore):
        """Build the Shera three-plane optics stack.

        Delegates to ``build_shera_threeplane_optics`` with the configured
        three-plane configuration, validated store, and forward specification.
        """
        from ..builders.optics import build_shera_threeplane_optics

        return build_shera_threeplane_optics(
            self.cfg, store=store, spec=self.forward_spec
        )

    def _build_source(self, store: ParameterStore):
        """Build the Shera alpha Cen source for the three-plane system."""
        return build_alpha_cen_source(store, cfg=self.cfg)

    def _optics_runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return the three-plane runtime bindings for non-structural keys."""
        from ..builders.optics import THREEPLANE_RUNTIME_BINDINGS

        return THREEPLANE_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        """Return the structural hash derived from the three-plane config."""
        from ..builders.optics import structural_hash_from_config

        return structural_hash_from_config(self.cfg)

    with_store = BaseSheraBinder.with_store


__all__ = [
    "SHERA_THREEPLANE_SYSTEM_ID",
    "SheraThreePlaneConfig",
    "SheraThreePlaneBinder",
    "SHERA_TESTBED_CONFIG",
    "SHERA_FLIGHT_CONFIG",
    "DEFAULT_DP_PATH",
    "default_diffractive_pupil_path",
    "build_forward_spec_from_config",
]
