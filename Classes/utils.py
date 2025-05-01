# utils.py
import jax.numpy as np
import jax
from jax import grad, linearize, jit, lax
from jax import config as jax_config


__all__ = ["merge_cbar", "nanrms", "scale_array"]






def merge_cbar(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    return make_axes_locatable(ax).append_axes("right", size="5%", pad=0.0)

def nanrms(arr, axis=None):
    return np.nanmean(arr**2, axis=axis)**0.5

def scale_array(array: jax.Array, size_out: int, order: int) -> jax.Array:
    xs = np.linspace(0, array.shape[0], size_out)
    xs, ys = np.meshgrid(xs, xs)
    return jax.scipy.ndimage.map_coordinates(array, np.array([ys, xs]), order=order)
