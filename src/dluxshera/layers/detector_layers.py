"""Custom detector layers for dLuxShera."""

from __future__ import annotations

import interpax
import jax.numpy as np
from jax import Array

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

    Only anisotropic Gaussian kernels are supported in this first version.

    Units
    -----
    ``sigma_x`` and ``sigma_y`` are declared either in:
      - ``psf_pix``: current PSF-array pixel units
      - ``detector_pix``: detector-pixel units

    dLux detector layers only carry the current PSF sampling during ``apply``.
    Within the Shera detector stack, the sampling changes only through
    ``Downsample`` layers, so the builder precomputes ``detector_to_psf_scale``
    as the number of current-PSF pixels per detector pixel at this layer's
    position in the stack. Direct callers may provide that scale manually.
    """

    _SUPPORTED_KERNEL_KINDS = {"gaussian"}
    _SUPPORTED_UNITS = {"detector_pix", "psf_pix"}

    kernel_kind: str
    sigma_x: float
    sigma_y: float
    theta_deg: float
    kernel_size: int
    units: str
    detector_to_psf_scale: float

    def __init__(
        self: DetectorLayer,
        *,
        kernel_kind: str,
        sigma_x: float,
        sigma_y: float,
        theta_deg: float,
        kernel_size: int,
        units: str,
        detector_to_psf_scale: float = 1.0,
    ):
        super().__init__()

        self.kernel_kind = str(kernel_kind)
        self.sigma_x = float(sigma_x)
        self.sigma_y = float(sigma_y)
        self.theta_deg = float(theta_deg)
        self.kernel_size = int(kernel_size)
        self.units = str(units)
        self.detector_to_psf_scale = float(detector_to_psf_scale)

        if self.kernel_kind not in self._SUPPORTED_KERNEL_KINDS:
            raise ValueError(
                "ApplyConvolution currently supports only "
                "kernel_kind='gaussian'."
            )
        if self.sigma_x <= 0.0:
            raise ValueError("sigma_x must be positive.")
        if self.sigma_y <= 0.0:
            raise ValueError("sigma_y must be positive.")
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

    def generate_kernel(self: DetectorLayer) -> Array:
        """Generate the normalized image-space convolution kernel."""
        sigma_x, sigma_y = self._sigma_in_psf_pixels()

        coords = np.arange(self.kernel_size, dtype=float) - (self.kernel_size - 1) / 2.0
        y, x = np.meshgrid(coords, coords, indexing="ij")

        theta = np.deg2rad(self.theta_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        x_rot = cos_theta * x + sin_theta * y
        y_rot = -sin_theta * x + cos_theta * y

        exponent = -0.5 * ((x_rot / sigma_x) ** 2 + (y_rot / sigma_y) ** 2)
        kernel = np.exp(exponent)
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

    def __init__(
        self: DetectorLayer,
        dx_map: Array,
        dy_map: Array,
        interp_method: str = "cubic",
        clip_nonnegative: bool = False,
    ):
        super().__init__()
        self.dx_map = np.asarray(dx_map, dtype=float)
        self.dy_map = np.asarray(dy_map, dtype=float)
        self.interp_method = str(interp_method)
        self.clip_nonnegative = bool(clip_nonnegative)

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

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """Apply detector-pixel offset warping to the PSF via ``interpax.interp2d``."""
        if self.dx_map.shape != psf.data.shape:
            raise ValueError(
                "dx_map/dy_map shape must match psf.data.shape; "
                f"got {self.dx_map.shape} vs {psf.data.shape}."
            )

        height, width = psf.data.shape

        # Build detector pixel coordinate grid (y, x)
        y, x = np.meshgrid(
            np.arange(height, dtype=float),
            np.arange(width, dtype=float),
            indexing="ij",
        )

        # Query points in (x, y) coordinates
        xq = x + self.dx_map
        yq = y + self.dy_map

        # Clamp-to-edge to emulate ndimage mode="nearest" boundary behavior.
        xq = np.clip(xq, 0.0, width - 1.0)
        yq = np.clip(yq, 0.0, height - 1.0)

        # Regular grid in x and y
        xk = np.arange(width, dtype=float)
        yk = np.arange(height, dtype=float)

        # interpax.interp2d expects f to be shaped (Nx, Ny, ...)
        # Our PSF is (Ny, Nx) = (height, width), so transpose it.
        f = psf.data.T  # shape (width, height)

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

        warped = warped_flat.reshape(height, width)

        if self.clip_nonnegative:
            warped = np.clip(warped, 0.0)

        return PSF(data=warped, pixel_scale=psf.pixel_scale)
