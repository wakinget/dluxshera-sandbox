"""Glass-box loss primitives for Shera inference.

This module intentionally keeps the math small and transparent so it can be
used by both legacy and Binder-based loss constructors.
"""
from __future__ import annotations

from typing import Literal, Optional

import jax.numpy as jnp

Reduction = Optional[Literal["sum", "mean"]]


def gaussian_image_nll(
    pred: jnp.ndarray,
    data: jnp.ndarray,
    var: jnp.ndarray,
    *,
    reduce: Reduction = "sum",
) -> jnp.ndarray:
    r"""Compute a Gaussian negative log-likelihood for image data.

    Parameters
    ----------
    pred
        Model prediction (e.g., PSF image). Broadcasts against ``data`` and
        ``var``.
    data
        Observed image data.
    var
        Per-pixel variance. Can be a scalar or array that broadcasts across
        ``pred``/``data``. No positivity checks are enforced here.
    reduce
        "sum" (default) sums over all pixels, "mean" averages, and ``None``
        returns the per-pixel NLL array without reduction.

    Returns
    -------
    jnp.ndarray
        Scalar NLL if reduced, otherwise an array matching the broadcasted
        inputs.

    Notes
    -----
    This implements the standard Gaussian NLL:

    .. math::

        \mathcal{L} = \tfrac{1}{2} \left[\frac{(m - d)^2}{\sigma^2} + \log(2\pi
        \sigma^2)\right]

    where ``m`` = ``pred``, ``d`` = ``data``, and ``var`` = ``σ²``. NaNs in the
    inputs are handled by the ``nan`` reduction variants.
    """
    pred = jnp.asarray(pred)
    data = jnp.asarray(data)
    var = jnp.asarray(var)

    nll = 0.5 * ((pred - data) ** 2 / var + jnp.log(2.0 * jnp.pi * var))

    if reduce == "sum":
        return jnp.nansum(nll)
    if reduce == "mean":
        return jnp.nanmean(nll)
    if reduce is None:
        return nll

    raise ValueError(f"Unknown reduction {reduce!r}; expected 'sum', 'mean', or None.")
