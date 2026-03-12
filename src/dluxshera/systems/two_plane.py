"""Two-plane Shera system definitions (binder, config, and presets)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

from .base import BaseConfig, compose_forward_spec
from ..params.spec import ParamSpec
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

    detector_model: Optional[str] = None
    """Detector model selector used by the detector builder metadata lookup."""

    detector_layers: Optional[list[dict[str, object]]] = None
    """Declarative detector layer pipeline for the detector builder."""

    system: Optional[dict[str, Any]] = None
    """Optional nested system config (e.g. ``system.optics.kind`` dispatch hints)."""

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

    throughput: float = 1.0
    """System throughput applied in forward modelling (unitless)."""

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
    `optics.primary.zernike_coeffs_nm` of matching length.
    """

    # ------------------------------------------------------------------
    # Diffractive pupil / fixed masks
    # ------------------------------------------------------------------
    diffractive_pupil_path: Optional[str] = None
    """Optional filesystem path to a diffractive pupil mask (e.g. .npy)."""

    dp_design_wavelength_m: Optional[float] = None
    """Design wavelength for the diffractive pupil mask [meters]."""

    def __post_init__(self):
        if self.detector_layers is None:
            default_layers = [
                {"name": "downsample", "kernel_size": int(self.oversample)},
                {"name": "pixel_offsets"},
                {"name": "pixel_response"},
                {"name": "jitter", "sigma": 1e-12, "kernel_size": 3},
            ]
            object.__setattr__(self, "detector_layers", default_layers)
        elif not isinstance(self.detector_layers, list):
            object.__setattr__(self, "detector_layers", list(self.detector_layers))


def build_forward_spec_from_config(cfg: SheraTwoPlaneConfig) -> ParamSpec:
    """Legacy wrapper for composed forward-spec construction.

    Forward specs are now authored compositionally from component contracts
    (source + optics + detector). This wrapper remains as a compatibility
    entry point for existing call sites and delegates to
    :func:`dluxshera.systems.base.compose_forward_spec`.
    """

    cfg_dict = asdict(cfg)
    system_cfg = {
        "source": {"kind": "alpha_cen", **cfg_dict},
        "optics": {"kind": "two_plane", **cfg_dict},
        "detector": {
            "model": cfg_dict.get("detector_model"),
            "layers": cfg_dict.get("detector_layers", None),
            **{k: v for k, v in cfg_dict.items() if k.startswith("detector_")},
        },
    }
    return compose_forward_spec({"system": system_cfg})


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

__all__ = [
    "SHERA_TWOPLANE_SYSTEM_ID",
    "SheraTwoPlaneConfig",
    "SHERA_TESTBED_CONFIG",
    "SHERA_FLIGHT_CONFIG",
    "build_forward_spec_from_config",
]
