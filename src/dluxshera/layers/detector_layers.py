"""Custom detector layers for dLuxShera."""

from __future__ import annotations

import inspect

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
        ``'nearest'``, ``'linear'``, ``'cubic'``, ``'cubic2'``,
        ``'catmull-rom'``, ``'cardinal'``, ``'monotonic'``, ``'monotonic-0'``,
        ``'akima'``, and ``'fft'``.
    clip_nonnegative : bool
        If True, clip warped output values to be non-negative.
    """

    _INTERP2D_METHODS = {
        "nearest",
        "linear",
        "cubic",
        "cubic2",
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
        interp_method: str = "cubic2",
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
        if self.interp_method not in self._INTERP2D_METHODS.union({"fft"}):
            raise ValueError(
                "interp_method must be one of "
                f"{sorted(self._INTERP2D_METHODS.union({'fft'}))}."
            )

    def _call_interp2d(self, xk: Array, yk: Array, values: Array, xq: Array, yq: Array) -> Array:
        sig = inspect.signature(interpax.interp2d)
        params = sig.parameters
        kwargs = {}

        name_map = {
            "x": xk,
            "xk": xk,
            "xp": xk,
            "x_knots": xk,
            "y": yk,
            "yk": yk,
            "yp": yk,
            "y_knots": yk,
            "f": values,
            "z": values,
            "values": values,
            "v": values,
            "xq": xq,
            "x_new": xq,
            "xnew": xq,
            "xi": xq,
            "sx": xq,
            "yq": yq,
            "y_new": yq,
            "ynew": yq,
            "yi": yq,
            "sy": yq,
            "method": self.interp_method,
            "interp_method": self.interp_method,
        }

        for name in params:
            if name in name_map:
                kwargs[name] = name_map[name]

        return interpax.interp2d(**kwargs)

    def _call_fft_interp2d(self, xk: Array, yk: Array, values: Array, xq: Array, yq: Array) -> Array:
        """Call fft_interp2d if its API supports arbitrary query points.

        FFT interpolation has periodic / wrap-around semantics.
        """

        sig = inspect.signature(interpax.fft_interp2d)
        params = sig.parameters
        kwargs = {}

        name_map = {
            "x": xk,
            "xk": xk,
            "xp": xk,
            "y": yk,
            "yk": yk,
            "yp": yk,
            "f": values,
            "z": values,
            "values": values,
            "xq": xq,
            "x_new": xq,
            "xnew": xq,
            "xi": xq,
            "sx": xq,
            "yq": yq,
            "y_new": yq,
            "ynew": yq,
            "yi": yq,
            "sy": yq,
        }

        for name in params:
            if name in name_map:
                kwargs[name] = name_map[name]

        query_supported = any(name in params for name in ("xq", "yq", "xi", "yi", "sx", "sy"))
        if not query_supported:
            raise ValueError(
                "interp_method='fft' is not supported for per-pixel offsets because "
                "interpax.fft_interp2d in this environment does not accept arbitrary "
                "query coordinates. Use an interp2d method (e.g. 'linear' or 'cubic') instead."
            )

        return interpax.fft_interp2d(**kwargs)

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """Apply detector-pixel offset warping to the PSF."""
        if self.dx_map.shape != psf.data.shape:
            raise ValueError(
                "dx_map/dy_map shape must match psf.data.shape; "
                f"got {self.dx_map.shape} vs {psf.data.shape}."
            )

        height, width = psf.data.shape
        y, x = np.meshgrid(
            np.arange(height, dtype=float),
            np.arange(width, dtype=float),
            indexing="ij",
        )
        xq = x + self.dx_map
        yq = y + self.dy_map

        xk = np.arange(width, dtype=float)
        yk = np.arange(height, dtype=float)

        if self.interp_method in self._INTERP2D_METHODS:
            xq = np.clip(xq, 0.0, width - 1.0)
            yq = np.clip(yq, 0.0, height - 1.0)
            warped = self._call_interp2d(xk, yk, psf.data, xq, yq)
        else:
            warped = self._call_fft_interp2d(xk, yk, psf.data, xq, yq)

        if self.clip_nonnegative:
            warped = np.clip(warped, 0.0)

        return PSF(data=warped, pixel_scale=psf.pixel_scale)
