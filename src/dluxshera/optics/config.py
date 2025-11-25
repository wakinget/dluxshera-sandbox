# src/dluxshera/optics/config.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class SheraThreePlaneConfig:
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
    # Telescope geometry (three-plane layout)
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

    # ------------------------------------------------------------------
    # Detector sampling
    # ------------------------------------------------------------------
    detector_pixel_pitch_m: float = 6.5e-6
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
    basis using these indices and will expect `primary.zernike_coeffs` in
    the ParameterStore to have matching length (or be None, in which case a
    zero vector is assumed).
    """

    secondary_noll_indices: Tuple[int, ...] = ()
    """
    Noll indices defining the Zernike basis on the secondary mirror.

    Same convention as `primary_noll_indices`: an empty tuple means no
    Zernike basis is constructed for the secondary; a non-empty tuple means
    the builder constructs a BasisOptic and expects compatible coefficients
    under `secondary.zernike_coeffs`.
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



# ---------------------------------------------------------------------
# Named point designs
# ---------------------------------------------------------------------

SHERA_TESTBED_CONFIG = SheraThreePlaneConfig(
    design_name="shera_testbed",

    # --- core telescope geometry ---
    m1_diameter_m=0.09,
    m2_diameter_m=0.025,
    m1_focal_length_m=0.35796,
    m2_focal_length_m=-0.041935,
    m1_m2_separation_m=0.320,

    # --- detector sampling ---
    detector_pixel_pitch_m=6.5e-6,

    # --- grids & sampling ---
    pupil_npix=256,
    psf_npix=256,
    oversample=3,
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
    diffractive_pupil_path="data/diffractive_pupil.npy",
    dp_design_wavelength_m=550e-9,
)


SHERA_FLIGHT_CONFIG = SheraThreePlaneConfig(
    design_name="shera_flight",

    # --- core telescope geometry ---
    m1_diameter_m=0.22,
    m2_diameter_m=0.025,
    m1_focal_length_m=0.604353,
    m2_focal_length_m=-0.0545,
    m1_m2_separation_m=0.55413,

    # --- detector sampling ---
    detector_pixel_pitch_m=4.6e-6,

    # --- grids & sampling ---
    pupil_npix=256,
    psf_npix=256,
    oversample=3,
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

    diffractive_pupil_path="data/diffractive_pupil.npy",
    dp_design_wavelength_m=550e-9,
)
