# utils.py
import jax.numpy as np
import numpy as onp
import jax
# from jax import grad, linearize, jit, lax
from jax import config, tree, Array
import equinox as eqx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, datetime, os


__all__ = ["merge_cbar", "nanrms","set_array", "scale_array",
           "sinusoidal_grating_2D", "calculate_log_flux",
           "plot_parameter_history"]

inferno = mpl.colormaps["inferno"]
seismic = mpl.colormaps["seismic"]
inferno.set_bad("k", 0.5)
seismic.set_bad("k", 0.5)
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120



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

    # Check if the bandwidth is specified in nanometers or microns
    if bandwidth > 1:
        warnings.warn(
            f"Bandwidth appears to be specified in nanometers ({bandwidth} nm). "
            "Expected units are microns. Please verify you have the correct units for bandwidth."
        )

    # Calculate the aperture area (m^2)
    aperture_area = np.pi * (diameter / 2) ** 2

    # Calculate the total flux
    total_flux = default_flux * exposure_time * aperture_area * bandwidth

    # Return the log_flux
    return np.log10(total_flux)


def get_sweep_values(center, span, steps):
    """
    Generate a symmetric array of sweep values around a center value.

    Parameters
    ----------
    center : float
        The central value of the sweep (e.g., the current parameter value).
    span : float
        The total range of the sweep (from min to max).
    steps : int
        The number of points in the sweep. If odd, the center is included.
        If even, the sweep avoids the center.

    Returns
    -------
    np.ndarray
        1D array of sweep values.
    """
    half = span / 2
    if steps % 2 == 0:
        step = span / steps
        return np.linspace(center - half + step/2, center + half - step/2, steps)
    else:
        return np.linspace(center - half, center + half, steps)


def print_param_summary(data_params, model_params, rtol=1e-10, atol=0):
    """
    Compare and summarize the starting parameters for data and model.

    Parameters
    ----------
    data_params : ModelParams
        Initial parameters used to generate synthetic data.
    model_params : ModelParams
        Initial parameters used to initialize the model.
    rtol : float
        Relative tolerance used in comparison.
    atol : float
        Absolute tolerance used in comparison.

    Notes
    -----
    Assumes both objects have `.keys` and `.get(key)` methods.
    """
    print("\n📌 Parameter Summary Before Sweep")
    print("-" * 100)

    all_keys = sorted(set(data_params.keys) | set(model_params.keys))

    for key in all_keys:
        val_data = data_params.get(key)
        val_model = model_params.get(key)

        try:
            match = np.allclose(val_data, val_model, rtol=rtol, atol=atol)
        except Exception:
            match = False

        val_data_arr = np.array(val_data)
        val_model_arr = np.array(val_model)

        val_data_str = onp.array2string(val_data_arr, precision=3, separator=", ")
        val_model_str = onp.array2string(val_model_arr, precision=3, separator=", ")

        if match:
            status = "✅ Match"
            diff_str = ""
        else:
            abs_diff = onp.abs(val_data_arr - val_model_arr)
            # Handle relative difference safely
            with onp.errstate(divide='ignore', invalid='ignore'):
                rel_diff = onp.abs(abs_diff / val_data_arr)
                rel_diff = onp.where(onp.isfinite(rel_diff), rel_diff,
                                     onp.nan)  # Replace inf/nan with NaN for later formatting
            # Format strings
            abs_diff_str = onp.array2string(abs_diff, precision=6, separator=", ")
            # Convert array to string manually, replacing NaN with "N/A"
            rel_diff_cleaned = ["N/A" if onp.isnan(x) else f"{x:.6e}" for x in onp.ravel(rel_diff)]
            rel_diff_str = "[" + ", ".join(rel_diff_cleaned) + "]"
            diff_str = f" | ΔABS: {abs_diff_str:18} | ΔREL: {rel_diff_str:18}"
            status = "❌ Mismatch"

        print(f"{key:20} | Data: {val_data_str:30} | Model: {val_model_str:30} | {status}{diff_str}")

    print("-" * 100 + "\n")


def get_param_scale_and_unit(param):
    """
    Returns a scaling factor and unit string for plotting a given parameter.

    Parameters
    ----------
    param : str
        Name of the parameter (e.g., 'x_position', 'coefficients').

    Returns
    -------
    scale : float
        Value by which to multiply the parameter for display (e.g., 1e6 for μas).
    unit : str
        Unit string for labeling axes (e.g., 'μas', 'nm').
    """
    mapping = {
        "x_position": (1e6, "μas"),
        "y_position": (1e6, "μas"),
        "separation": (1e6, "μas"),
        "coefficients": (1.0, "nm"),
        "m1_aperture.coefficients": (1.0, "nm"),
        "m2_aperture.coefficients": (1.0, "nm"),
        "zernike_amp": (1.0, "nm"),
        "contrast": (1.0, "Contrast B:A"),
        "position_angle": (1.0, "deg"),
        "log_flux": (1.0, "log photons"),
        "psf_pixel_scale": (1.0, "arcsec/px"),
        "wavelength": (1.0, "nm"),
    }

    return mapping.get(param, (1.0, "units"))