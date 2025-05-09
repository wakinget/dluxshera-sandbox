# utils.py
import jax.numpy as np
import jax
# from jax import grad, linearize, jit, lax
from jax import config, tree, Array
import equinox as eqx
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = ["merge_cbar", "nanrms","set_array", "scale_array",
           "sinusoidal_grating_2D", "calculate_log_flux"]


def merge_cbar(ax):
    return make_axes_locatable(ax).append_axes("right", size="5%", pad=0.0)


def nanrms(arr, axis=None):
    return np.nanmean(arr**2, axis=axis)**0.5


def set_array(pytree, parameters):
    dtype = np.float64 if config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = tree.map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)


def scale_array(array: Array, size_out: int, order: int) -> Array:
    xs = np.linspace(0, array.shape[0], size_out)
    xs, ys = np.meshgrid(xs, xs)
    return jax.scipy.ndimage.map_coordinates(array, np.array([ys, xs]), order=order)


def sinusoidal_grating_2D(N, amplitude, frequency, angle=0, phase=None):
    """
    Generates a 2D sinusoidal grating pattern. Used to generate the DP grating

    Parameters:
    -----------
    N : int
        The size of the output 2D array (N x N).
    amplitude : float
        The amplitude of the sine wave.
    frequency : float
        The frequency of the sine wave (cycles per unit length).
    angle : float (Optional)
        The angle (in degrees) to rotate the sine wave pattern counter-clockwise.
    phase : float array (Optional)
        The phase of the sine wave specified as a full (N x N) array.
        Inputting the binary phase mask allows the user to 'flip' the phase
            of the sinusoid at the boundaries of the DP pattern


    Returns:
    --------
    sine_wave : np.array
        A 2D array containing the sine wave pattern.
    """

    if phase is None:
        phase = np.zeros((N, N))

    # Normalized coordinate grid
    x = np.linspace(-0.5, 0.5, N)
    y = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(x, y)

    # Convert angle to radians
    theta = np.deg2rad(angle)

    # Rotate coordinate system
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    # Compute the sine wave along rotated axes
    sine_wave = amplitude * (np.sin(2 * np.pi * frequency * X_rot - phase) +
                             np.sin(2 * np.pi * frequency * Y_rot - phase))

    return sine_wave

def calculate_log_flux(diameter, bandwidth, exposure_time, default_flux=1.7227e11):
    """
    Calculate the log_flux given the primary mirror diameter, bandwidth, and exposure time.

    Parameters
    ----------
    diameter : float
        Diameter of the primary mirror (meters).
    bandwidth : float
        Bandwidth of the observation (microns).
    exposure_time : float
        Exposure time (seconds).
    default_flux : float, optional
        Default total star flux in photons per second per meter^2 per micron of band
        Default of 1.7227e11 is taken from the Toliman Master Spreadsheet for aCenA+B

    Returns
    -------
    float
        The calculated log_flux.
    """
    # Calculate the aperture area (m^2)
    aperture_area = np.pi * (diameter / 2) ** 2

    # Calculate the total flux
    total_flux = default_flux * exposure_time * aperture_area * bandwidth

    # Return the log_flux
    return np.log10(total_flux)
