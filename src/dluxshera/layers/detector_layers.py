"""Custom detector layers for dLuxShera."""

from __future__ import annotations

import interpax
import jax.numpy as np
from jax import Array

from dLux.layers.detector_layers import DetectorLayer
from dLux.psfs import PSF


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
