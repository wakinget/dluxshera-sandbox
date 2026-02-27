from __future__ import annotations

import jax.numpy as np
from jax import Array, vmap
from typing import TYPE_CHECKING

import dLux
import dLux.layers as dll
import dLux.utils as dlu
import dLuxToliman

from ..utils.utils import scale_array
from ..params.spec import ParamField, ParamSpec

if TYPE_CHECKING:
    from ..systems.three_plane import SheraThreePlaneConfig
    from ..systems.two_plane import SheraTwoPlaneConfig

MixedAlphaCen = lambda: dLuxToliman.sources.MixedAlphaCen

__all__ = [
    "SheraTwoPlaneOptics",
    "SheraThreePlaneOptics",
    "build_threeplane_optics_contract",
    "build_twoplane_optics_contract",
]

OpticalLayer = lambda: dLux.optical_layers.OpticalLayer
AngularOpticalSystem = lambda: dLux.optical_systems.AngularOpticalSystem
ThreePlaneOpticalSystem = lambda: dLux.optical_systems.ThreePlaneOpticalSystem


def build_threeplane_optics_contract(cfg: "SheraThreePlaneConfig") -> ParamSpec:
    """Return the three-plane optics parameter contract."""
    fields = [
        ParamField(
            "optics.pupil_npix",
            group="optics",
            kind="primitive",
            dtype=int,
            default=cfg.pupil_npix,
            structural=True,
        ),
        ParamField(
            "optics.psf_npix",
            group="optics",
            kind="primitive",
            dtype=int,
            default=cfg.psf_npix,
            structural=True,
        ),
        ParamField(
            "optics.oversample",
            group="optics",
            kind="primitive",
            dtype=int,
            default=cfg.oversample,
            structural=True,
        ),
        ParamField(
            "optics.m1_diameter_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.m1_diameter_m,
            structural=True,
        ),
        ParamField(
            "optics.m2_diameter_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.m2_diameter_m,
            structural=True,
        ),
        ParamField(
            "optics.m1_focal_length_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.m1_focal_length_m,
            structural=True,
        ),
        ParamField(
            "optics.m2_focal_length_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.m2_focal_length_m,
            structural=True,
        ),
        ParamField(
            "optics.m1_m2_separation_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.m1_m2_separation_m,
            structural=True,
        ),
        ParamField(
            "optics.n_struts",
            group="optics",
            kind="primitive",
            dtype=int,
            default=cfg.n_struts,
            structural=True,
        ),
        ParamField(
            "optics.strut_width_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.strut_width_m,
            structural=True,
        ),
        ParamField(
            "optics.strut_rotation_deg",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.strut_rotation_deg,
            structural=True,
        ),
        ParamField(
            "optics.primary_noll_indices",
            group="optics",
            kind="primitive",
            dtype=int,
            shape=(len(cfg.primary_noll_indices),),
            default=tuple(int(i) for i in cfg.primary_noll_indices),
            structural=True,
        ),
        ParamField(
            "optics.secondary_noll_indices",
            group="optics",
            kind="primitive",
            dtype=int,
            shape=(len(cfg.secondary_noll_indices),),
            default=tuple(int(i) for i in cfg.secondary_noll_indices),
            structural=True,
        ),
        ParamField(
            "optics.dp_path",
            group="optics",
            kind="primitive",
            dtype=str,
            default=cfg.dp_path,
            structural=True,
        ),
        ParamField(
            "optics.dp_design_wavelength_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.dp_design_wavelength_m,
            structural=True,
        ),
        ParamField(
            "optics.throughput",
            group="optics",
            kind="primitive",
            dtype=float,
            default=1.0,
            structural=False,
        ),
        ParamField(
            "optics.plate_scale_as_per_pix",
            group="optics",
            kind="derived",
            dtype=float,
            transform="optics.plate_scale_as_per_pix",
            depends_on=(
                "optics.focal_length_m",
                "detector.pixel_pitch_m",
            ),
            structural=False,
            binding="psf_pixel_scale",
        ),
    ]

    if cfg.primary_noll_indices:
        fields.append(
            ParamField(
                "optics.primary.zernike_coeffs_nm",
                group="optics",
                kind="primitive",
                dtype=float,
                shape=(len(cfg.primary_noll_indices),),
                default=tuple(0.0 for _ in cfg.primary_noll_indices),
                structural=False,
                binding="p1_layers.m1_aperture.coefficients",
            )
        )

    if cfg.secondary_noll_indices:
        fields.append(
            ParamField(
                "optics.secondary.zernike_coeffs_nm",
                group="optics",
                kind="primitive",
                dtype=float,
                shape=(len(cfg.secondary_noll_indices),),
                default=tuple(0.0 for _ in cfg.secondary_noll_indices),
                structural=False,
                binding="p2_layers.m2_aperture.coefficients",
            )
        )

    return ParamSpec(fields)


def build_twoplane_optics_contract(cfg: "SheraTwoPlaneConfig") -> ParamSpec:
    """Return the two-plane optics parameter contract."""
    fields = [
        ParamField(
            "optics.pupil_npix",
            group="optics",
            kind="primitive",
            dtype=int,
            default=cfg.pupil_npix,
            structural=True,
        ),
        ParamField(
            "optics.psf_npix",
            group="optics",
            kind="primitive",
            dtype=int,
            default=cfg.psf_npix,
            structural=True,
        ),
        ParamField(
            "optics.oversample",
            group="optics",
            kind="primitive",
            dtype=int,
            default=cfg.oversample,
            structural=True,
        ),
        ParamField(
            "optics.m1_diameter_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.m1_diameter_m,
            structural=True,
        ),
        ParamField(
            "optics.m2_diameter_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.m2_diameter_m,
            structural=True,
        ),
        ParamField(
            "optics.n_struts",
            group="optics",
            kind="primitive",
            dtype=int,
            default=cfg.n_struts,
            structural=True,
        ),
        ParamField(
            "optics.strut_width_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.strut_width_m,
            structural=True,
        ),
        ParamField(
            "optics.strut_rotation_deg",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.strut_rotation_deg,
            structural=True,
        ),
        ParamField(
            "optics.primary_noll_indices",
            group="optics",
            kind="primitive",
            dtype=int,
            shape=(len(cfg.primary_noll_indices),),
            default=tuple(int(i) for i in cfg.primary_noll_indices),
            structural=True,
        ),
        ParamField(
            "optics.dp_path",
            group="optics",
            kind="primitive",
            dtype=str,
            default=cfg.diffractive_pupil_path,
            structural=True,
        ),
        ParamField(
            "optics.dp_design_wavelength_m",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.dp_design_wavelength_m,
            structural=True,
        ),
        ParamField(
            "optics.throughput",
            group="optics",
            kind="primitive",
            dtype=float,
            default=1.0,
            structural=False,
        ),
        ParamField(
            "optics.plate_scale_as_per_pix",
            group="optics",
            kind="primitive",
            dtype=float,
            default=cfg.plate_scale_as_per_pix,
            structural=False,
            binding="psf_pixel_scale",
        ),
    ]

    if cfg.primary_noll_indices:
        fields.append(
            ParamField(
                "optics.primary.zernike_coeffs_nm",
                group="optics",
                kind="primitive",
                dtype=float,
                shape=(len(cfg.primary_noll_indices),),
                default=tuple(0.0 for _ in cfg.primary_noll_indices),
                structural=False,
                binding="layers.aperture.coefficients",
            )
        )

    return ParamSpec(fields)



class SheraTwoPlaneOptics(AngularOpticalSystem()):
    """Build a two-plane Shera optical system with optional aberrations.

    This class wires together a Shera-style entrance pupil, optional Zernike
    aberrations, and an optional diffractive pupil map for use with
    :class:`dLux.optical_systems.AngularOpticalSystem`.

    Key properties
    --------------
    - Supports Zernike aberrations via ``radial_orders`` or ``noll_indices``.
    - Accepts a diffractive pupil mask as either a path or array.
    - Interprets diffractive pupil data as normalized phase or OPD depending on
      ``dp_design_wavel``.

    Notes
    -----
    This object is intended to be treated as immutable. Update patterns should
    follow functional replacements on layers rather than in-place mutation.
    """

    def __init__(
        self,
        wf_npixels: int = 256,
        psf_npixels: int = 128,
        oversample: int = 2,
        psf_pixel_scale: float = 0.3547,  # as/pix
        mask: Array | str = None,
        radial_orders: Array = None,
        noll_indices: Array = None,
        coefficients: Array = None,
        m1_diameter: float = 0.09,
        m2_diameter: float = 0.025,
        n_struts: int = 4,
        strut_width: float = 0.002,
        strut_rotation_deg: float = -90.0,
        dp_design_wavel: float = 550e-9,
    ):
        """Construct a Shera two-plane optical system.

        Parameters
        ----------
        wf_npixels : int
            Pixel width of the wavefront layer.
        psf_npixels : int
            Pixel width of the PSF.
        oversample : int
            Nyquist oversampling factor of the PSF.
        psf_pixel_scale : float
            Pixel scale of the PSF in arcseconds per pixel.
        mask : Array
            Diffractive pupil mask to apply to the wavefront layer. Accepts a
            path to a .npy file or a 2D array. When provided alongside
            ``dp_design_wavel``, the input is interpreted as a normalized phase
            pattern ``P(x, y)`` ∈ [0, 1] spanning [0, π] radians at
            ``dp_design_wavel``. When ``dp_design_wavel`` is None, the input
            is interpreted directly as an OPD map in meters.
        radial_orders : Array = None
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indices argument should be used instead.
        noll_indices : Array
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus.
        coefficients : Array
            The coefficients of the Zernike polynomials.
        m1_diameter : float
            The outer diameter of the primary mirror in metres.
        m2_diameter : float
            The diameter of the secondary mirror in metres.
        n_struts : int
            The number of uniformly spaced struts holding the secondary mirror.
        strut_width : float
            The width of the struts in metres.
        strut_rotation_deg : float
            The angular rotation of the struts in degrees.
        dp_design_wavel : float
            Design wavelength in meters for interpreting normalized phase
            diffractive pupil masks.
        """

        # Diameter
        diameter = m1_diameter

        # Generate Aperture
        pupil_oversample = 2
        coords = dlu.pixel_coords(pupil_oversample * wf_npixels, m1_diameter)
        outer = dlu.circle(coords, m1_diameter / 2)
        inner = dlu.circle(coords, m2_diameter / 2, invert=True)
        strut_angles = np.linspace(0, 360, n_struts + 1)[:-1] + strut_rotation_deg
        spiders = dlu.spider(coords, strut_width, strut_angles)
        transmission = dlu.combine([outer, inner, spiders], pupil_oversample)

        # Hack this in for now, will be in dLux eventually
        if radial_orders is not None:
            radial_orders = np.array(radial_orders)

            if (radial_orders < 0).any():
                raise ValueError("Radial orders must be >= 0")

            noll_indices = []
            for order in radial_orders:
                start = dlu.triangular_number(order)
                stop = dlu.triangular_number(order + 1)
                noll_indices.append(np.arange(start, stop) + 1)
            noll_indices = np.concatenate(noll_indices).astype(int)

        if noll_indices is None:
            aperture = dll.TransmissiveLayer(transmission, normalise=True)
        else:
            # Generate Basis
            coords = dlu.pixel_coords(wf_npixels, diameter)
            basis = np.array(
                [dlu.zernike(i, coords, m1_diameter) for i in noll_indices]
            )

            if coefficients is None:
                coefficients = np.zeros(len(noll_indices))

            # Combine into BasisOptic class
            aperture = dll.BasisOptic(basis, transmission, coefficients, normalise=True)

        # Generate Mask
        if mask is None:
            dp_opd = np.zeros((wf_npixels, wf_npixels))
        else:
            if isinstance(mask, str):
                dp_array = np.load(mask)
            else:
                dp_array = mask

            if dp_array.shape[-2:] != (wf_npixels, wf_npixels):
                dp_array = scale_array(dp_array, wf_npixels, order=1)

            dp_array = np.array(dp_array)

            if dp_design_wavel is None:
                dp_opd = dp_array
            else:
                phase_rad = dp_array * np.pi
                dp_opd = dlu.phase2opd(phase_rad, dp_design_wavel)

        mask = dll.AberratedLayer(dp_opd)

        layers = [("aperture", aperture), ("pupil", mask)]

        # Propagator Properties
        psf_npixels = int(psf_npixels)
        oversample = float(oversample)
        psf_pixel_scale = float(psf_pixel_scale)

        super().__init__(
            wf_npixels=wf_npixels,
            diameter=diameter,
            layers=layers,
            psf_npixels=psf_npixels,
            oversample=int(oversample),
            psf_pixel_scale=psf_pixel_scale,
        )

    def _apply_aperture(self, wavelength, offset):
        """Apply aperture transmission and diffractive pupil terms."""
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.aperture
        wf = wf.normalise()
        wf += self.pupil
        return wf


class SheraThreePlaneOptics(ThreePlaneOpticalSystem()):
    """Build a three-plane Shera optical system with optional aberrations.

    This class defines a two-mirror optical train with a diffractive pupil on
    the primary plane. It uses :class:`dLux.optical_systems.ThreePlaneOpticalSystem`
    to handle multi-plane propagation and image formation.

    Key properties
    --------------
    - Independent Zernike aberrations can be applied to the M1 and M2 pupils.
    - Diffractive pupil inputs can be normalized phase or OPD maps.
    - Plate scale is derived from the focal length and detector pitch.

    Notes
    -----
    This object is intended to be treated as immutable. Update patterns should
    follow functional replacements on layers rather than in-place mutation.
    """

    m1_noll_ind: Array = None
    m2_noll_ind: Array = None

    def __init__(
        self,
        wf_npixels: int = 256,
        psf_npixels: int = 128,
        oversample: int = 2,
        detector_pixel_pitch: float = 4.6e-6,  # pixel pitch in meters/pixel
        mask=None,
        m1_noll_ind: Array = None,
        m1_coefficients: Array = None,
        m2_noll_ind: Array = None,
        m2_coefficients: Array = None,
        p1_diameter: float = 0.220,
        p2_diameter: float = 0.025,
        m1_focal_length: float = 0.604353,
        m2_focal_length: float = -0.0545,
        plane_separation: float = 0.554130,
        n_struts: int = 4,
        strut_width: float = 0.002,
        strut_rotation_deg: float = 45.0,
        dp_design_wavel: float | None = 550e-9,
    ):
        """Construct a Shera three-plane optical system.

        Parameters
        ----------
        wf_npixels : int
            Pixel width of the wavefront layer.
        psf_npixels : int
            Pixel width of the PSF.
        oversample : int
            Nyquist oversampling factor of the PSF.
        detector_pixel_pitch : float
            Detector pixel pitch in meters per pixel.
        mask : Array or str, optional
            Diffractive pupil mask as a 2D array or path to a ``.npy`` file.
        m1_noll_ind : Array, optional
            Zernike Noll indices for M1 aberrations.
        m1_coefficients : Array, optional
            Zernike coefficients for the M1 basis (meters of OPD).
        m2_noll_ind : Array, optional
            Zernike Noll indices for M2 aberrations.
        m2_coefficients : Array, optional
            Zernike coefficients for the M2 basis (meters of OPD).
        p1_diameter : float
            Primary pupil diameter in meters.
        p2_diameter : float
            Secondary pupil diameter in meters.
        m1_focal_length : float
            Primary mirror focal length in meters.
        m2_focal_length : float
            Secondary mirror focal length in meters.
        plane_separation : float
            Separation between the primary and secondary in meters.
        n_struts : int
            Number of uniformly spaced struts holding the secondary mirror.
        strut_width : float
            Strut width in meters.
        strut_rotation_deg : float
            Strut rotation in degrees.
        dp_design_wavel : float or None
            Design wavelength in meters for normalized phase masks. When None,
            the mask is treated as OPD in meters.
        """

        # Set attributes
        self.m1_noll_ind = m1_noll_ind
        self.m2_noll_ind = m2_noll_ind

        # Calculate optical system parameters
        phi_m1 = 1 / m1_focal_length
        phi_m2 = 1 / m2_focal_length
        phi_telescope = phi_m1 + phi_m2 - plane_separation * phi_m1 * phi_m2
        f_telescope = 1 / phi_telescope
        m1_magnification = 1 / (1 - plane_separation * phi_m1)
        psf_pixel_scale = dlu.rad2arcsec(detector_pixel_pitch / f_telescope)

        # Create the aperture masks (including spider geometry)
        pupil_oversample = 2
        coords = dlu.pixel_coords(pupil_oversample * wf_npixels, p1_diameter)
        outer = dlu.circle(coords, p1_diameter / 2)
        inner = dlu.circle(coords, p2_diameter / 2, invert=True)
        strut_angles = np.linspace(0, 360, n_struts + 1)[:-1] + strut_rotation_deg
        spiders = dlu.spider(coords, strut_width, strut_angles)
        m1_transmission = dlu.combine([outer, inner, spiders], pupil_oversample)
        m2_transmission = dlu.downsample(outer, pupil_oversample)

        # Generate Zernike basis for M1
        if self.m1_noll_ind is None:
            m1_aperture = dll.TransmissiveLayer(m1_transmission, normalise=True)
        else:
            coords = dlu.pixel_coords(wf_npixels, p1_diameter)
            basis = np.array(
                [dlu.zernike(i, coords, p1_diameter) for i in self.m1_noll_ind]
            )
            coefficients = (
                np.zeros(len(self.m1_noll_ind))
                if m1_coefficients is None
                else m1_coefficients
            )
            m1_aperture = dll.BasisOptic(basis, m1_transmission, coefficients, normalise=True)
            # Normalize basis so Zernike coefficients represent nanometers of OPD
            m1_aperture = m1_aperture.multiply("basis", 1e-9)

        # Generate Zernike basis for M2
        if self.m2_noll_ind is None:
            m2_aperture = dll.TransmissiveLayer(m2_transmission, normalise=True)
        else:
            coords = dlu.pixel_coords(wf_npixels, p2_diameter)
            basis = np.array(
                [dlu.zernike(i, coords, p2_diameter) for i in self.m2_noll_ind]
            )
            coefficients = (
                np.zeros(len(self.m2_noll_ind))
                if m2_coefficients is None
                else m2_coefficients
            )
            m2_aperture = dll.BasisOptic(basis, m2_transmission, coefficients, normalise=True)
            # Normalize basis so Zernike coefficients represent nanometers of OPD
            m2_aperture = m2_aperture.multiply("basis", 1e-9)

        # ------------------------------------------------------------------
        # Generate Diffractive Pupil Mask layer
        # ------------------------------------------------------------------
        # We always create a "dp" layer for p1_layers, but its effect depends
        # on whether a mask and design wavelength are provided:
        #
        # - mask is None:
        #       create a neutral aberration layer with zero OPD everywhere
        #       (no diffractive pupil applied).
        #
        # - mask is provided and dp_design_wavel is not None:
        #       interpret the data as a normalized phase pattern P(x, y) in
        #       [0, 1] spanning [0, π] radians at dp_design_wavel, and map to
        #       an OPD map via:
        #
        #           phase_rad = P * π
        #           opd_m = dlu.phase2opd(phase_rad, dp_design_wavel)
        #
        # - mask is provided and dp_design_wavel is None:
        #       interpret the data directly as an OPD map in meters.
        #
        # The mask input may be either:
        #   - a filesystem path (str) pointing to a .npy file, or
        #   - a 2D array already loaded in memory.
        #
        if mask is None:
            # Neutral DP: zero OPD across the pupil grid.
            dp_opd = np.zeros((wf_npixels, wf_npixels))
        else:
            # Load the raw map: either from disk (if mask is a path) or as-is
            # (if mask is already an array).
            if isinstance(mask, str):
                dp_array = np.load(mask)
            else:
                dp_array = mask

            # Ensure the map is sampled on the wf_npixels grid. If the helper
            # already returns a jax.numpy array, jnp.array() is a no-op.
            if dp_array.shape[-2:] != (wf_npixels, wf_npixels):
                dp_array = scale_array(dp_array, wf_npixels, order=1)

            dp_array = np.array(dp_array)

            if dp_design_wavel is None:
                # Map is already OPD in meters.
                dp_opd = dp_array
            else:
                # Normalized phase P ∈ [0, 1] → phase ∈ [0, π] radians.
                phase_rad = dp_array * np.pi
                dp_opd = dlu.phase2opd(phase_rad, dp_design_wavel)

        dp_layer = dll.AberratedLayer(dp_opd)

        p1_layers = [("m1_aperture", m1_aperture), ("dp", dp_layer)]
        p2_layers = [("m2_aperture", m2_aperture)]
        super().__init__(
            wf_npixels=wf_npixels,
            p1_diameter=p1_diameter,
            p2_diameter=p2_diameter,
            p1_layers=p1_layers,
            p2_layers=p2_layers,
            plane_separation=plane_separation,
            magnification=m1_magnification,
            psf_npixels=psf_npixels,
            psf_pixel_scale=psf_pixel_scale,
            oversample=oversample,
        )

    def _apply_aperture(self, wavelength, offset):
        """Apply aperture transmission and diffractive pupil terms."""
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.m1_aperture
        wf = wf.normalise()
        wf += self.dp
        return wf

# class SheraMultiPlaneSystem(MultiPlaneOpticalSystem()):
#     def __init__(
#         self,
#         wf_npixels: int = 256,
#         psf_npixels: int = 128,
#         oversample: int = 4,
#         detector_pixel_pitch: float = 4.6,  # um
#         mask: Array = None,
#         radial_orders: Array = None,
#         noll_indices: Array = None,
#         coefficients: Array = None,
#         m1_diameter: float = 0.220,
#         m2_diameter: float = 0.060,
#         m1_focal_length: float = 0.604353,
#         m2_focal_length: float = -0.0545,
#         plane_separation: float = 0.554130,
#         n_struts: int = 3,
#         strut_width: float = 0.002,
#         strut_rotation: float = -np.pi / 2,
#     ):
#         """
#         A pre-built dLux optics layer of the Shera optical system, including a secondary mirror.
#
#         Parameters
#         ----------
#         wf_npixels : int
#             The pixel width the wavefront layer.
#         psf_npixels : int
#             The pixel width of the PSF.
#         oversample : int
#             The Nyquist oversampling factor of the PSF.
#         psf_pixel_scale : float
#             The pixel scale of the PSF in arcseconds per pixel.
#         mask : Array
#             The diffractive mask array to apply to the wavefront layer.
#         radial_orders : Array = None
#             The radial orders of the zernike polynomials to be used for the
#             aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
#             [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
#             The order must be increasing but does not have to be consecutive.
#             If you want to specify specific zernikes across radial orders the
#             noll_indices argument should be used instead.
#         noll_indices : Array
#             The zernike noll indices to be used for the aberrations. [1, 2, 3]
#             would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
#             Tilt Y, Defocus.
#         coefficients : Array
#             The coefficients of the Zernike polynomials.
#         m1_diameter : float
#             The diameter of the primary mirror in metres.
#         m2_diameter : float
#             The diameter of the secondary mirror in metres.
#         m1_focal_length : float
#             The focal length of the primary mirror in metres.
#         m2_focal_length : float
#             The focal length of the secondary mirror in metres.
#         plane_separation : float
#             The separation between the primary and secondary in metres.
#         n_struts : int
#             The number of uniformly spaced struts holding the secondary mirror.
#         strut_width : float
#             The width of the struts in metres.
#         strut_rotation : float
#             The angular rotation of the struts in radians.
#         """
#
#         # Diameter
#         diameter = [m1_diameter, m2_diameter]
#
#
#         # Reduce the telescope system
#         phi_m1 = 1 / m1_focal_length  # diopters, M1 power
#         phi_m2 = 1 / m2_focal_length  # diopters, M2 power
#         phi_telescope = phi_m1 + phi_m2 - plane_separation * phi_m1 * phi_m2  # Overall Telescope power
#         f_telescope = 1 / phi_telescope  # Overall Telescope focal length
#         # Calculate M1 Magnification
#         m1_magnification = 1 / (1 - plane_separation * phi_m1)
#         psf_pixel_scale = dlu.rad2arcsec(detector_pixel_pitch*1e-6 / f_telescope)
#
#         # Generate Pupil Aperture
#         pupil_oversample = 5
#         coords = dlu.pixel_coords(pupil_oversample * wf_npixels, diameter[0])
#         outer = dlu.circle(coords, m1_diameter / 2)
#         inner = dlu.circle(coords, m2_diameter / 2, invert=True)
#         strut_angles = np.linspace(0,360, n_struts+1)
#         strut_angles = strut_angles[:-1] + np.rad2deg(strut_rotation)
#         spiders = dlu.spider(coords, strut_width, strut_angles)
#         m1_transmission = dlu.combine([outer, inner, spiders], pupil_oversample)
#         m2_transmission = dlu.downsample(outer, pupil_oversample)
#
#         # Generate a zernike basis
#         if radial_orders is not None:
#             radial_orders = np.array(radial_orders)
#
#             if (radial_orders < 0).any():
#                 raise ValueError("Radial orders must be >= 0")
#
#             noll_indices = []
#             for order in radial_orders:
#                 start = dlu.triangular_number(order)
#                 stop = dlu.triangular_number(order + 1)
#                 noll_indices.append(np.arange(start, stop) + 1)
#             noll_indices = np.concatenate(noll_indices).astype(int)
#
#         if noll_indices is None:
#             m1_aperture = dll.TransmissiveLayer(m1_transmission, normalise=True)
#             m2_aperture = dll.TransmissiveLayer(m2_transmission, normalise=True)
#         else:
#             # Generate Basis
#             coords = dlu.pixel_coords(wf_npixels, diameter[0])
#             basis = np.array(
#                 [dlu.zernike(i, coords, m1_diameter) for i in noll_indices]
#             )
#
#             if coefficients is None:
#                 coefficients = np.zeros(len(noll_indices))
#
#             # Combine into BasisOptic class
#             m1_aperture = dll.BasisOptic(basis, m1_transmission, coefficients, normalise=True)
#             m2_aperture = dll.BasisOptic(basis, m2_transmission, coefficients, normalise=True)
#
#         # # Generate Aperture
#         # aperture = dLux.apertures.ApertureFactory(
#         #     npixels=wf_npixels,
#         #     radial_orders=radial_orders,
#         #     noll_indices=noll_indices,
#         #     coefficients=coefficients,
#         #     secondary_ratio=m2_diameter / m1_diameter,
#         #     nstruts=n_struts,
#         #     strut_ratio=strut_width / m1_diameter,
#         #     strut_rotation=strut_rotation,
#         # )
#
#         # Generate DP Mask
#         if mask is None:
#             path = os.path.join(os.path.dirname(__file__), "diffractive_pupil.npy")
#             # print("DP Path: %s" % path)
#             # arr_in = np.load(path)
#             # ratio = wf_npixels / arr_in.shape[-1]
#             mask = scale_array(np.load(path), wf_npixels, order=1)
#
#             # Enforce full binary
#             mask = mask.at[np.where(mask <= 0.5)].set(0.0)
#             mask = mask.at[np.where(mask > 0.5)].set(1.0)
#
#             # Enforce full binary
#             mask = dlu.phase2opd(mask * np.pi, 550e-9)
#
#             # Turn into optic
#             mask = dll.AberratedLayer(mask)
#
#
#         # Define Layers for the Primary and for the Secondary
#         m1_layers = [("m1_aperture", m1_aperture), ("dp_mask", mask)]
#         m2_layers = [("m2_aperture", m2_aperture)]
#
#         # Propagator Properties
#         psf_npixels = int(psf_npixels)
#         oversample = float(oversample)
#         psf_pixel_scale = float(psf_pixel_scale)
#
#         super().__init__(
#             wf_npixels=wf_npixels,
#             diameter=diameter,
#             plane_layers=[m1_layers, m2_layers],
#             plane_separations=plane_separation,
#             plane_magnifications=m1_magnification,
#             # aperture=aperture,
#             # mask=mask,
#             psf_npixels=psf_npixels,
#             oversample=int(oversample),
#             psf_pixel_scale=psf_pixel_scale,
#         )
#
#     def _apply_aperture(self, wavelength, offset):
#         """
#         Overwrite so mask can be stored as array
#         """
#         wf = self._construct_wavefront(wavelength, offset)
#         wf *= self.m1_aperture
#         wf = wf.normalise()
#         wf += self.mask
#         return wf
