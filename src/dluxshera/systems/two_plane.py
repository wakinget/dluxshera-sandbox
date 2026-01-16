"""Two-plane Shera system definitions (binder, config, and presets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import dLux as dl
import jax.numpy as jnp

from .base import BaseConfig, BaseSheraBinder
from ..builders.source import build_alpha_cen_source
from ..params.spec import ParamField, ParamSpec
from ..params.store import ParameterStore
from ..utils.utils import DEFAULT_DP_PATH

SHERA_TWOPLANE_SYSTEM_ID = "shera_twoplane"


@dataclass(frozen=True)
class SheraTwoPlaneConfig(BaseConfig):
    """
    Structural configuration for the Shera two-plane optical system.

    This captures fixed geometry and sampling choices for the Toliman-like
    two-plane pupil→focal relay used by SheraTwoPlaneOptics. These values are
    separate from inference parameters (which live in ParameterStore/ParamSpec)
    and are intended to remain constant for a given instrument setup.
    """

    design_name: Optional[str] = None
    """Human-readable identifier for this optical design."""

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
    # System geometry (two-plane layout)
    # ------------------------------------------------------------------
    m1_diameter_m: float = 0.09
    """Primary mirror clear diameter [meters]."""

    m2_diameter_m: float = 0.025
    """Secondary mirror clear diameter [meters]."""

    n_struts: int = 4
    """Number of support struts in the primary aperture."""

    strut_width_m: float = 0.002
    """Width of the support struts [meters]."""

    strut_rotation_deg: float = -45.0
    """Rotation angle of the spider pattern [degrees]."""

    # ------------------------------------------------------------------
    # Fixed plate scale (primitive for the two-plane model)
    # ------------------------------------------------------------------
    plate_scale_as_per_pix: float = 0.355
    """
    Plate scale in arcseconds per pixel.

    For the two-plane Shera system we treat plate scale as a primitive rather
    than deriving it from telescope geometry.
    """

    # ------------------------------------------------------------------
    # Zernike basis selection (structure, not coefficients)
    # ------------------------------------------------------------------
    primary_noll_indices: Tuple[int, ...] = ()
    """
    Noll indices defining the Zernike basis on the primary mirror.

    If this tuple is empty, the builder does not construct a Zernike BasisOptic
    for the primary. If non-empty, the forward spec will expect
    `primary.zernike_coeffs_nm` of matching length.
    """

    # ------------------------------------------------------------------
    # Diffractive pupil / fixed masks
    # ------------------------------------------------------------------
    diffractive_pupil_path: Optional[str] = None
    """Optional filesystem path to a diffractive pupil mask (e.g. .npy)."""

    dp_design_wavelength_m: Optional[float] = None
    """Design wavelength for the diffractive pupil mask [meters]."""


def build_forward_spec_from_config(cfg: SheraTwoPlaneConfig) -> ParamSpec:
    """
    Construct a ParamSpec for the Shera two-plane forward model.

    This mirrors the three-plane forward spec semantics but treats the plate
    scale as a primitive (geometry is not modelled explicitly) and omits any
    secondary mirror Zernike basis. Binary astrometry fields are exposed as
    primitives, while the total log flux remains a derived quantity handled via
    the shared log-flux transform.
    """

    fields = [
        # --- System geometry (primitive plate scale) ---------------------
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
                "Primary mirror clear diameter [meters]. Used for collecting "
                "area in flux calculations."
            ),
        ),
        ParamField(
            key="system.plate_scale_as_per_pix",
            group="system",
            kind="primitive",
            units="as / pixel",
            dtype=float,
            shape=None,
            default=cfg.plate_scale_as_per_pix,
            bounds=(0.0, None),
            doc=(
                "Plate scale in arcseconds per pixel, treated as a primitive "
                "knob for the two-plane Shera system."
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
            doc="Approximate bandpass width [meters].",
        ),

        # --- Imaging configuration ----------------------------------
        ParamField(
            key="imaging.exposure_time_s",
            group="imaging",
            kind="primitive",
            units="s",
            dtype=float,
            shape=None,
            default=1800.0,
            bounds=(0.0, None),
            doc="Single-exposure integration time [seconds].",
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
                "optical transmission, detector QE, and other losses."
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
                "Positive values follow the detector X axis."
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
                "Positive values follow the detector Y axis."
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
            default=3,
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
                "entrance pupil, in units of photons/s/m^2 per meter of "
                "bandwidth."
            ),
        ),

        # --- Derived forward-model quantities ---------------------------
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
                "exposure at the detector plane."
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

    return ParamSpec(fields, system_id=SHERA_TWOPLANE_SYSTEM_ID)


# ---------------------------------------------------------------------
# Named point designs
# ---------------------------------------------------------------------

SHERA_TESTBED_CONFIG = SheraTwoPlaneConfig(
    design_name="shera_testbed",

    # --- system geometry ---
    m1_diameter_m=0.09,
    m2_diameter_m=0.025,
    plate_scale_as_per_pix=0.355,

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

    # --- diffractive pupil ---
    diffractive_pupil_path=DEFAULT_DP_PATH,
    dp_design_wavelength_m=550e-9,
)


SHERA_FLIGHT_CONFIG = SheraTwoPlaneConfig(
    design_name="shera_flight",

    # --- system geometry ---
    m1_diameter_m=0.22,
    m2_diameter_m=0.025,
    plate_scale_as_per_pix=0.123,

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

    diffractive_pupil_path=DEFAULT_DP_PATH,
    dp_design_wavelength_m=550e-9,
)

@dataclass
class SheraTwoPlaneBinder(BaseSheraBinder):
    """Generative model for the Shera two-plane system.

    Mirrors :class:`SheraThreePlaneBinder` semantics: mostly immutable, owns a
    forward-spec-validated base store, and exposes ``.model(store_delta)`` as the
    canonical evaluation path. ``.model`` fast-paths through the cached
    telescope when ``store_delta`` is omitted, and accepts non-structural
    overlays by default when an explicit delta is provided. Structural updates
    require ``allow_rebuild=True`` to rebuild the binder state. When
    ``.update_store()`` returns a new binder instance with the refreshed base
    store so the original binder remains unchanged.

    The ``with_store`` attribute is an alias of
    :meth:`BaseSheraBinder.with_store` to keep a stable public API. It preserves
    immutable-style semantics by always returning a new binder instance.
    """

    cfg: SheraTwoPlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore

    def __init__(
        self,
        cfg: SheraTwoPlaneConfig,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
    ) -> None:
        """Construct a binder for the two-plane Shera configuration.

        Parameters
        ----------
        cfg : SheraTwoPlaneConfig
            Fully prepared Shera two-plane configuration. Derived config
            values expected by the optics/source builders should already be
            present.
        forward_spec : ParamSpec
            Parameter specification describing the forward store, including
            structural keys and derived entries.
        base_forward_store : ParameterStore
            Forward-style base store with derived values populated. The store
            is validated against ``forward_spec`` and used as the immutable
            baseline for evaluations.
        """
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
        )

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        """Evaluate the Shera two-plane model directly.

        Builds a fresh telescope using the two-plane optics and alpha Cen
        source with ``eff_store`` and returns the modeled PSF output.
        """
        return self._build_telescope(eff_store).model()

    def _build_optics(self, store: ParameterStore):
        """Build the Shera two-plane optics stack."""
        from ..builders.optics import build_shera_twoplane_optics

        return build_shera_twoplane_optics(self.cfg, store=store, spec=self.forward_spec)

    def _build_source(self, store: ParameterStore):
        """Build the Shera alpha Cen source for the two-plane system."""
        return build_alpha_cen_source(store, cfg=self.cfg)

    def _optics_runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return the two-plane runtime bindings for non-structural keys."""
        from ..builders.optics import TWOPLANE_RUNTIME_BINDINGS

        return TWOPLANE_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        """Return the structural hash derived from the two-plane config."""
        from ..builders.optics import structural_hash_for_twoplane

        return structural_hash_for_twoplane(self.cfg)

    with_store = BaseSheraBinder.with_store


__all__ = [
    "SHERA_TWOPLANE_SYSTEM_ID",
    "SheraTwoPlaneConfig",
    "SheraTwoPlaneBinder",
    "SHERA_TESTBED_CONFIG",
    "SHERA_FLIGHT_CONFIG",
    "build_forward_spec_from_config",
]
