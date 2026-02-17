"""Custom detector layers for dLuxShera."""

from __future__ import annotations

import jax.numpy as np
from jax import Array
from jax.scipy.ndimage import map_coordinates

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
    interp_order : int
        Interpolation order for warping (1 = bilinear, 3 = cubic).
    """

    dx_map: Array
    dy_map: Array
    interp_order: int

    def __init__(self: DetectorLayer, dx_map: Array, dy_map: Array, interp_order: int = 1):
        super().__init__()
        self.dx_map = np.asarray(dx_map, dtype=float)
        self.dy_map = np.asarray(dy_map, dtype=float)
        self.interp_order = int(interp_order)

        if self.dx_map.ndim != 2:
            raise ValueError("dx_map must be a 2D array.")
        if self.dy_map.ndim != 2:
            raise ValueError("dy_map must be a 2D array.")
        if self.dx_map.shape != self.dy_map.shape:
            raise ValueError("dx_map and dy_map must have the same shape.")
        if self.interp_order not in (1, 3):
            raise ValueError("interp_order must be 1 (bilinear) or 3 (cubic).")

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """Apply detector-pixel offset warping to the PSF."""
        if self.dx_map.shape != psf.data.shape:
            raise ValueError(
                "dx_map/dy_map shape must match psf.data.shape; "
                f"got {self.dx_map.shape} vs {psf.data.shape}."
            )

        y, x = np.meshgrid(
            np.arange(psf.data.shape[0], dtype=float),
            np.arange(psf.data.shape[1], dtype=float),
            indexing="ij",
        )
        coords = np.stack([y + self.dy_map, x + self.dx_map], axis=0)
        warped = map_coordinates(
            psf.data,
            coords,
            order=self.interp_order,
            mode="nearest",
        )
        return PSF(data=warped, pixel_scale=psf.pixel_scale)
