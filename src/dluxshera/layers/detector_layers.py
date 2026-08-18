"""Custom detector layers for dLuxShera."""

from __future__ import annotations

import interpax
import jax.numpy as np
from jax import Array
from jax.scipy.special import erf

from dLux.layers.detector_layers import DetectorLayer
from dLux.psfs import PSF


class ApplyConvolution(DetectorLayer):
    """Convolve a PSF with a normalized detector kernel.

    This layer intentionally reuses dLux's existing detector convolution path by
    generating an image-space kernel and delegating to ``PSF.convolve(...)``.
    That preserves dLux-style semantics:
      - linear convolution
      - ``mode="same"``
      - no auto-padding

    Supported kernel families:
      - ``gaussian``: anisotropic Gaussian with optional in-plane rotation
      - ``box``: axis-aligned rectangular pixel-aperture response
      - ``line``: finite linear smear with Gaussian cross-track softness

    Units
    -----
    Gaussian ``sigma_x``/``sigma_y``, box ``width_x``/``width_y``, and line
    ``length``/``sigma_perp`` are declared either in:
      - ``psf_pix``: current PSF-array pixel units
      - ``detector_pix``: detector-pixel units

    dLux detector layers only carry the current PSF sampling during ``apply``.
    Within the Shera detector stack, the sampling changes only through
    ``Downsample`` layers, so the builder precomputes ``detector_to_psf_scale``
    as the number of current-PSF pixels per detector pixel at this layer's
    position in the stack. Direct callers may provide that scale manually.
    """

    _SUPPORTED_KERNEL_KINDS = {"gaussian", "box", "line"}
    _SUPPORTED_UNITS = {"detector_pix", "psf_pix"}

    kernel_kind: str
    sigma_x: float
    sigma_y: float
    width_x: float
    width_y: float
    theta_deg: float
    length: float
    sigma_perp: float
    kernel_size: int
    units: str
    detector_to_psf_scale: float

    def __init__(
        self: DetectorLayer,
        *,
        kernel_kind: str,
        sigma_x: float = 1.0,
        sigma_y: float = 1.0,
        width_x: float = 1.0,
        width_y: float = 1.0,
        theta_deg: float = 0.0,
        length: float = 1.0,
        sigma_perp: float = 0.25,
        kernel_size: int,
        units: str,
        detector_to_psf_scale: float = 1.0,
    ):
        super().__init__()

        self.kernel_kind = str(kernel_kind)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.width_x = float(width_x)
        self.width_y = float(width_y)
        self.theta_deg = float(theta_deg)
        self.length = float(length)
        self.sigma_perp = float(sigma_perp)
        self.kernel_size = int(kernel_size)
        self.units = str(units)
        self.detector_to_psf_scale = float(detector_to_psf_scale)

        if self.kernel_kind not in self._SUPPORTED_KERNEL_KINDS:
            raise ValueError(
                "ApplyConvolution supports kernel_kind values: "
                f"{sorted(self._SUPPORTED_KERNEL_KINDS)}."
            )
        if self.kernel_kind == "gaussian":
            if self.sigma_x <= 0.0:
                raise ValueError("sigma_x must be positive.")
            if self.sigma_y <= 0.0:
                raise ValueError("sigma_y must be positive.")
        if self.kernel_kind == "box":
            if self.width_x <= 0.0:
                raise ValueError("width_x must be positive.")
            if self.width_y <= 0.0:
                raise ValueError("width_y must be positive.")
        if self.kernel_kind == "line":
            if self.length <= 0.0:
                raise ValueError("length must be positive.")
            if self.sigma_perp <= 0.0:
                raise ValueError("sigma_perp must be positive.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if self.units not in self._SUPPORTED_UNITS:
            raise ValueError(
                "units must be one of "
                f"{sorted(self._SUPPORTED_UNITS)}."
            )
        if self.detector_to_psf_scale <= 0.0:
            raise ValueError("detector_to_psf_scale must be positive.")

    def _sigma_in_psf_pixels(self: DetectorLayer) -> tuple[float, float]:
        """Return Gaussian sigmas in the current PSF array's pixel units."""
        if self.units == "psf_pix":
            return self.sigma_x, self.sigma_y

        return (
            self.sigma_x * self.detector_to_psf_scale,
            self.sigma_y * self.detector_to_psf_scale,
        )

    def _width_in_psf_pixels(self: DetectorLayer) -> tuple[float, float]:
        """Return box widths in the current PSF array's pixel units."""
        if self.units == "psf_pix":
            return self.width_x, self.width_y
        return (
            self.width_x * self.detector_to_psf_scale,
            self.width_y * self.detector_to_psf_scale,
        )

    def _line_params_in_psf_pixels(self: DetectorLayer) -> tuple[float, float]:
        """Return line-kernel length and cross-track sigma in PSF pixel units."""
        if self.units == "psf_pix":
            return self.length, self.sigma_perp
        return (
            self.length * self.detector_to_psf_scale,
            self.sigma_perp * self.detector_to_psf_scale,
        )

    def generate_kernel(self: DetectorLayer) -> Array:
        """Generate the normalized image-space convolution kernel."""
        coords = np.arange(self.kernel_size, dtype=float) - (self.kernel_size - 1) / 2.0
        y, x = np.meshgrid(coords, coords, indexing="ij")

        if self.kernel_kind == "gaussian":
            sigma_x, sigma_y = self._sigma_in_psf_pixels()
            theta = np.deg2rad(self.theta_deg)
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            x_rot = cos_theta * x + sin_theta * y
            y_rot = -sin_theta * x + cos_theta * y

            exponent = -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
            kernel = np.exp(exponent)
            return kernel / np.sum(kernel)

        if self.kernel_kind == "box":
            width_x, width_y = self._width_in_psf_pixels()
            inside_x = np.abs(x) <= (width_x / 2.0)
            inside_y = np.abs(y) <= (width_y / 2.0)
            kernel = np.where(inside_x & inside_y, 1.0, 0.0)
            return kernel / np.sum(kernel)

        length, sigma_perp = self._line_params_in_psf_pixels()
        theta = np.deg2rad(self.theta_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Rotate to along-track (u) / cross-track (v) coordinates.
        u = cos_theta * x + sin_theta * y
        v = -sin_theta * x + cos_theta * y

        # Anti-aliased finite-support line segment:
        # along-track weight is a softly windowed top-hat (fixed half-pixel edge),
        # cross-track profile is Gaussian with configurable thickness.
        edge_sigma = 0.5
        sqrt2 = np.sqrt(2.0)
        along = 0.5 * (
            erf((u + 0.5 * length) / (sqrt2 * edge_sigma))
            - erf((u - 0.5 * length) / (sqrt2 * edge_sigma))
        )
        cross = np.exp(-0.5 * (v / sigma_perp) ** 2)
        kernel = along * cross
        return kernel / np.sum(kernel)

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """Apply image-space convolution with dLux ``PSF.convolve`` semantics."""
        kernel = self.generate_kernel()
        return psf.convolve(kernel)


class ApplyPixelOffsets(DetectorLayer):
    """Warp a detector-resolution PSF using per-pixel center offsets.

    Attributes
    ----------
    dx_map : Array
        Horizontal detector-pixel offsets, sampled with x + dx.
    dy_map : Array
        Vertical detector-pixel offsets, sampled with y + dy.
    interp_method : str
        Interpolation method used for warping. Supported values are
        'nearest', 'linear', 'cubic' (default), 'catmull-rom',
        'cardinal', 'monotonic', 'monotonic-0', 'akima'.
    clip_nonnegative : bool
        If True, clip warped output values to be non-negative.
    detector_to_psf_scale : float
        Number of current-PSF pixels per detector pixel at this layer.
        When greater than 1, the layer samples the oversampled PSF back onto
        the detector grid using detector-pixel center coordinates.
    """

    _ALLOWED_METHODS = {
        "nearest",
        "linear",
        "cubic",
        "catmull-rom",
        "cardinal",
        "monotonic",
        "monotonic-0",
        "akima",
    }

    dx_map: Array
    dy_map: Array
    interp_method: str
    clip_nonnegative: bool
    detector_to_psf_scale: float

    def __init__(
        self: DetectorLayer,
        dx_map: Array,
        dy_map: Array,
        interp_method: str = "cubic",
        clip_nonnegative: bool = False,
        detector_to_psf_scale: float = 1.0,
    ):
        super().__init__()
        self.dx_map = np.asarray(dx_map, dtype=float)
        self.dy_map = np.asarray(dy_map, dtype=float)
        self.interp_method = str(interp_method)
        self.clip_nonnegative = bool(clip_nonnegative)
        self.detector_to_psf_scale = float(detector_to_psf_scale)

        if self.dx_map.ndim != 2:
            raise ValueError("dx_map must be a 2D array.")
        if self.dy_map.ndim != 2:
            raise ValueError("dy_map must be a 2D array.")
        if self.dx_map.shape != self.dy_map.shape:
            raise ValueError("dx_map and dy_map must have the same shape.")

        if self.interp_method not in self._ALLOWED_METHODS:
            raise ValueError(
                "interp_method must be one of "
                f"{sorted(self._ALLOWED_METHODS)}."
            )
        if self.detector_to_psf_scale <= 0.0:
            raise ValueError("detector_to_psf_scale must be positive.")

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """Apply detector-pixel offset warping to the PSF via ``interpax.interp2d``."""
        detector_height, detector_width = self.dx_map.shape
        psf_shape = tuple(int(v) for v in psf.data.shape)
        scale = float(self.detector_to_psf_scale)
        oversampled_shape = (
            int(round(detector_height * scale)),
            int(round(detector_width * scale)),
        )

        if psf_shape == (detector_height, detector_width):
            y0 = np.arange(detector_height, dtype=float)
            x0 = np.arange(detector_width, dtype=float)
            output_pixel_scale = psf.pixel_scale
            offset_scale = 1.0
            flux_scale = 1.0
        elif psf_shape == oversampled_shape:
            y0 = (np.arange(detector_height, dtype=float) + 0.5) * scale - 0.5
            x0 = (np.arange(detector_width, dtype=float) + 0.5) * scale - 0.5
            output_pixel_scale = psf.pixel_scale * scale
            offset_scale = scale
            flux_scale = scale**2
        else:
            raise ValueError(
                "dx_map/dy_map shape must match psf.data.shape, or "
                "psf.data.shape must equal detector_to_psf_scale * map shape; "
                f"got {self.dx_map.shape} vs {psf.data.shape} with "
                f"detector_to_psf_scale={scale}."
            )

        y, x = np.meshgrid(
            y0,
            x0,
            indexing="ij",
        )

        # Offsets are defined in detector-pixel units; scale them into the
        # current PSF grid when sampling an oversampled image.
        xq = x + self.dx_map * offset_scale
        yq = y + self.dy_map * offset_scale

        # Clamp-to-edge to emulate ndimage mode="nearest" boundary behavior.
        input_height, input_width = psf_shape
        xq = np.clip(xq, 0.0, input_width - 1.0)
        yq = np.clip(yq, 0.0, input_height - 1.0)

        # Regular grid in x and y
        xk = np.arange(input_width, dtype=float)
        yk = np.arange(input_height, dtype=float)

        # interpax.interp2d expects f to be shaped (Nx, Ny, ...), so transpose
        # the PSF data from (Ny, Nx) to (Nx, Ny).
        f = psf.data.T

        # Query points are expected as 1D arrays (Nq,)
        xq_flat = xq.reshape(-1)
        yq_flat = yq.reshape(-1)

        warped_flat = interpax.interp2d(
            xq_flat,
            yq_flat,
            xk,
            yk,
            f,
            method=self.interp_method,
            extrap=False,  # safe since we clamp
        )

        warped = warped_flat.reshape(detector_height, detector_width)
        warped = warped * flux_scale

        if self.clip_nonnegative:
            warped = np.clip(warped, 0.0)

        return PSF(data=warped, pixel_scale=output_pixel_scale)
