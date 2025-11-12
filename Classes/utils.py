# utils.py
import jax.numpy as np # Jax numpy
import numpy as onp # Original numpy
import jax
# from jax import grad, linearize, jit, lax
from jax import config, tree, Array
import equinox as eqx
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import time, datetime, os
import json
from typing import Any, Dict, Optional, Union
import pandas as pd


__all__ = ["nanrms","set_array", "scale_array",
           "sinusoidal_grating_2D", "calculate_log_flux",
           ]

inferno = mpl.colormaps["inferno"]
seismic = mpl.colormaps["seismic"]
inferno.set_bad("k", 0.5)
seismic.set_bad("k", 0.5)
plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120



def debug_param_type(model, key):
    val = model.get(key)
    print(f"{key:20s} | value={val} | type={type(val)}", end="")

    # Check if JAX array
    if isinstance(val, Array):
        print("  <-- JAX Array ✅")
    elif isinstance(val, onp.ndarray):
        print("  <-- NumPy array (not JAX) ⚠️")
    elif isinstance(val, float) or isinstance(val, int):
        print("  <-- Python scalar ⚠️ (not differentiable)")
    elif val is None:
        print("  <-- None ⚠️ (not in graph)")
    else:
        print("  <-- other type")


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


def save_prior_info(prior_info: dict, filepath: str):
    serializable = {k: list(v) for k, v in prior_info.items()}
    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)

def load_prior_info(filepath: str) -> dict:
    with open(filepath, "r") as f:
        raw = json.load(f)
    return {k: tuple(v) for k, v in raw.items()}

def construct_perturbation_vector(fim_labels, target_perturbations):
    """
    Constructs a parameter vector with perturbations aligned to FIM ordering.

    Parameters:
    ----------
    fim_labels : list of str
        Labels of the FIM/eigenvector parameters (e.g. ["x_position", ..., "m1 Z13", ...])
    target_perturbations : dict
        Dictionary mapping labels (e.g. "m1 Z13", "separation") to perturbation amplitudes

    Returns:
    -------
    a_true : np.ndarray
        Perturbation vector of shape (n_params,) to be projected onto eigenvectors
    """
    a_true = np.zeros(len(fim_labels))
    for i, label in enumerate(fim_labels):
        if label in target_perturbations:
            a_true = a_true.at[i].set(target_perturbations[label])
    return a_true


"""
Excel results saving utilities

These helpers assemble a single "results row" from your simulation dictionaries
(true values, initial values, history, residuals, etc.) and write it to an
Excel workbook consistently. Numeric scalars are coerced to plain Python types
so Excel won’t treat them as text, while 1D arrays/lists are joined into
readable strings.

Typical usage:

    row_written = save_results(
        save_path, save_name,
        true_vals=true_vals,
        initial_vals=initial_vals,
        history=history,
        residuals=residuals,
        data_params=data_params,
        model_initial_params=model_initial_params,
        data=data,
        misc={
            "rng_seed": default_params.rng_seed,
            "N_sweeps": N_sweeps,
            "N_sweepvals": N_sweepvals,
            "N_datasets": N_datasets,
            "N_observations": N_observations,
            "n_iter": n_iter,
            "sweep_i": sweep_i,
            "sweepval_i": sweepval_i,
            "dataset_i": dataset_i,
            "obs_i": obs_i,
            "m2_data_nolls": m2_data_nolls,
            "sweep_values": sweep_values,
            "xy_pointing_error": xy_pointing_error,
            "add_shot_noise": add_shot_noise,
            "sigma_read": sigma_read,
            "exposure_per_frame": exposure_per_frame,
            "N_frames": N_frames,
            "optimiser_label": optimiser_label,
            "nanrms_fn": nanrms,    # optional: function to compute RMS from arrays
        },
        sheet_name="Results",
        overwrite=False,            # append if file/sheet exists; overwrite if True
    )
"""

# ----------------------------- helpers ---------------------------------------
def to_py_number(x: Any) -> Any:
    """Convert 0-D JAX/NumPy scalars into plain Python scalars.

    Why:
        Pandas (and Excel) can treat NumPy/JAX scalars as `object` dtype,
        causing Excel to store numbers as text. Converting to Python
        `int/float/bool` fixes that.

    Args:
        x: Any value. If it's a 0-D numpy scalar (e.g., `onp.float64(3.14)`) or a
           0-D ndarray-like (JAX or NumPy), it will be converted to a Python scalar.

    Returns:
        Either a Python scalar (int/float/bool) for 0-D arrays/scalars, or the
        original value unchanged for everything else.
    """
    # NumPy scalar types (e.g., onp.float64)
    if isinstance(x, onp.generic):
        return x.item()

    # 0-D JAX/NumPy arrays (DeviceArray/jax.Array or np.ndarray with ndim == 0)
    # We avoid importing jax.Array for typing compatibility; use shape/ndim instead.
    try:
        if hasattr(x, "shape") and onp.ndim(x) == 0:
            return onp.asarray(x).item()
    except Exception:
        pass

    # Native Python scalars are fine
    if isinstance(x, (int, float, bool)):
        return x

    return x


def excel_friendly(val: Any) -> Any:
    """Make values safe for Excel columns.

    Behavior:
        - 0-D arrays/scalars -> Python numbers via `to_py_number`.
        - 1-D arrays/lists/tuples -> comma-separated string (e.g., "1, 2, 3").
        - Everything else -> returned unchanged.

    Rationale:
        Mixing arrays/strings/numbers in the same column forces an `object`
        dtype and often becomes "text" in Excel. For 1-D arrays where you want
        a single cell summary, a readable string is convenient.

    Args:
        val: Value to coerce.

    Returns:
        Coerced value suitable for placing in a single Excel cell.
    """
    v = to_py_number(val)
    if v is not val:
        return v

    # Arrays (JAX or NumPy) – detect via `.shape` and coerce with onp.asarray
    try:
        if hasattr(val, "shape"):
            arr = onp.asarray(val)
            if arr.ndim == 1:
                return ", ".join(map(str, arr.tolist()))
            return val  # leave multi-D arrays untouched
    except Exception:
        pass

    # Python lists/tuples
    if isinstance(val, (list, tuple)):
        return ", ".join(map(str, val))

    return val


# -------------------------- row construction ---------------------------------
def build_results_row(
    *,
    true_vals: Dict[str, Any],
    initial_vals: Dict[str, Any],
    history: Dict[str, Any],
    residuals: Dict[str, Any],
    data_params: Any,
    model_initial_params: Any,
    data: Union[onp.ndarray, Any],
    misc: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble one results row (a flat dict of Excel-friendly values).

    Centralizes logic for:
      - pulling scalar values from your run dictionaries (`true_vals`, etc.)
      - formatting numbers/arrays consistently (via `excel_friendly`)
      - computing derived fields (e.g., pointing magnitude via hypot)
      - selecting the final ("found") values from `history`

    Args:
        true_vals: Dictionary of ground-truth input parameters.
        initial_vals: Dictionary of initial parameter guesses.
        history: Dictionary of arrays recording parameter evolution per iteration.
        residuals: Dictionary of arrays with final residual metrics.
        data_params: Object with data-side Zernike and 1/f parameters (attributes).
        model_initial_params: Object with model-side Zernike and 1/f attributes.
        data: 2-D image array (NumPy or JAX) used to compute total flux.
              (Typed as Union[onp.ndarray, Any] for Python 3.9 compatibility and JAX support.)
        misc: Extra run metadata (rng_seed, sweep indices, noise settings, etc.).
              May include an optional callable under key `"nanrms_fn"` to compute
              RMS entries when arrays are present.

    Returns:
        A flat dict suitable for `pd.DataFrame([row])`.
    """
    hyp = onp.hypot
    nanrms_fn = misc.get("nanrms_fn", None)

    def last(key: str) -> Optional[Any]:
        return history[key][-1] if key in history else None

    def safe_rms(arr: Any) -> Any:
        if nanrms_fn is None or arr is None:
            return None
        return nanrms_fn(arr)

    # Safe attribute access with defaults for array-ish fields
    m1_noll = onp.asarray(getattr(data_params, "m1_zernike_noll", onp.array([])))
    m1_amp  = onp.asarray(getattr(data_params, "m1_zernike_amp", onp.array([])))
    m2_noll = onp.asarray(getattr(data_params, "m2_zernike_noll", onp.array([])))
    m2_amp  = onp.asarray(getattr(data_params, "m2_zernike_amp", onp.array([])))

    # Optional sweep arrays in misc
    m2_data_nolls = misc.get("m2_data_nolls")
    sweep_values  = misc.get("sweep_values")
    sweep_i       = misc.get("sweep_i", 0)
    sweepval_i    = misc.get("sweepval_i", 0)

    row: Dict[str, Any] = {
        # --- Run / sweep metadata ---
        "Starting RNG Seed":             excel_friendly(misc.get("rng_seed")),
        "N Sweeps":                      excel_friendly(misc.get("N_sweeps")),
        "N Sweep Values":                excel_friendly(misc.get("N_sweepvals")),
        "N Datasets":                    excel_friendly(misc.get("N_datasets")),
        "N Observations":                excel_friendly(misc.get("N_observations")),
        "Optimizer Iterations":          excel_friendly(misc.get("n_iter")),
        "Sweep #":                       excel_friendly(sweep_i + 1),
        "Sweep M2 Noll":                 excel_friendly(m2_data_nolls[sweep_i]) if m2_data_nolls is not None else None,
        "M2 Noll Amplitude":             excel_friendly(sweep_values[sweepval_i]) if sweep_values is not None else None,
        "Dataset":                       excel_friendly(misc.get("dataset_i", 0) + 1),
        "Observation":                   excel_friendly(misc.get("obs_i", 0) + 1),

        # --- Inputs / data summary ---
        "Pointing Error Amplitude (as)": excel_friendly(misc.get("xy_pointing_error")),
        "Input Source Position X (as)":  excel_friendly(true_vals.get("x_position")),
        "Input Source Position Y (as)":  excel_friendly(true_vals.get("y_position")),
        "Input Pointing Error Magnitude (as)": excel_friendly(
            hyp(true_vals.get("x_position", 0.0), true_vals.get("y_position", 0.0))
        ),
        "Input Pointing Error Angle (deg)": excel_friendly(
            hyp(true_vals.get("x_position", 0.0), true_vals.get("y_position", 0.0))
        ),
        "Input Source Separation (as)":  excel_friendly(true_vals.get("separation")),
        "Input Source Angle (deg)":      excel_friendly(true_vals.get("position_angle")),
        "Input Source Log Flux":         excel_friendly(true_vals.get("log_flux")),
        "Input Source Contrast (A:B)":   excel_friendly(true_vals.get("contrast")),
        "Input Source A Raw Flux":       excel_friendly(true_vals.get("raw_fluxes", [None, None])[0]),
        "Input Source B Raw Flux":       excel_friendly(true_vals.get("raw_fluxes", [None, None])[1]),
        "Input Platescale (as/pixel)":   excel_friendly(true_vals.get("psf_pixel_scale")),

        "Input M1 Zernikes (Noll Index)":                   ", ".join(map(str, m1_noll.tolist())),
        "Input M1 Zernike Coefficient RMS Amplitude (nm)":  excel_friendly(safe_rms(m1_amp)),
        "Input M1 Zernike Coefficient Amplitudes (nm)":     ", ".join(map(str, m1_amp.tolist())),
        "Input M1 Zernike OPD RMS Error (nm)":              excel_friendly(true_vals.get("m1_zernike_opd_rms_nm")),
        "Input M1 Calibrated 1/f Amplitude (nm rms)":       excel_friendly(getattr(model_initial_params, "m1_calibrated_amplitude", None) * 1e9 if hasattr(model_initial_params, "m1_calibrated_amplitude") else None),
        "Input M1 Calibrated 1/f Power Law":                excel_friendly(getattr(model_initial_params, "m1_calibrated_power_law", None)),
        "Input M1 Uncalibrated 1/f Amplitude (nm rms)":     excel_friendly(getattr(data_params, "m1_uncalibrated_amplitude", None) * 1e9 if hasattr(data_params, "m1_uncalibrated_amplitude") else None),
        "Input M1 Uncalibrated 1/f Power Law":              excel_friendly(getattr(data_params, "m1_uncalibrated_power_law", None)),

        "Input M2 Zernikes (Noll Index)":                   ", ".join(map(str, m2_noll.tolist())),
        "Input M2 Zernike Coefficient RMS Amplitude (nm)":  excel_friendly(safe_rms(m2_amp)),
        "Input M2 Zernike Coefficient Amplitudes (nm)":     ", ".join(map(str, m2_amp.tolist())),
        "Input M2 Zernike OPD RMS Error (nm)":              excel_friendly(true_vals.get("m2_zernike_opd_rms_nm")),
        "Input M2 Calibrated 1/f Amplitude (nm rms)":       excel_friendly(getattr(model_initial_params, "m2_calibrated_amplitude", None) * 1e9 if hasattr(model_initial_params, "m2_calibrated_amplitude") else None),
        "Input M2 Calibrated 1/f Power Law":                excel_friendly(getattr(model_initial_params, "m2_calibrated_power_law", None)),
        "Input M2 Uncalibrated 1/f Amplitude (nm rms)":     excel_friendly(getattr(data_params, "m2_uncalibrated_amplitude", None) * 1e9 if hasattr(data_params, "m2_uncalibrated_amplitude") else None),
        "Input M2 Uncalibrated 1/f Power Law":              excel_friendly(getattr(data_params, "m2_uncalibrated_power_law", None)),

        "Photon Noise":                     bool(misc.get("add_shot_noise", False)),
        "Read Noise (e- / frame)":          excel_friendly(misc.get("sigma_read")),
        "Exposure (s / frame)":             excel_friendly(misc.get("exposure_per_frame")),
        "Coadded Frames":                   excel_friendly(misc.get("N_frames")),
        "Optimiser":                        str(misc.get("optimiser_label", "")),
        "Total Flux in Data":               excel_friendly(onp.sum(onp.asarray(data))),

        # --- Initial guesses ---
        "Initial Source Position X (as)":   excel_friendly(initial_vals.get("x_position")),
        "Initial Source Position Y (as)":   excel_friendly(initial_vals.get("y_position")),
        "Initial Source Separation (as)":   excel_friendly(initial_vals.get("separation")),
        "Initial Source Angle (deg)":       excel_friendly(initial_vals.get("position_angle")),
        "Initial Source Log Flux":          excel_friendly(initial_vals.get("log_flux")),
        "Initial Source Contrast (A:B)":    excel_friendly(initial_vals.get("contrast")),
        "Initial Source A Raw Flux":        excel_friendly(initial_vals.get("raw_fluxes", [None, None])[0]),
        "Initial Source B Raw Flux":        excel_friendly(initial_vals.get("raw_fluxes", [None, None])[1]),
        "Initial Platescale (as/pixel)":    excel_friendly(initial_vals.get("psf_pixel_scale")),
        "Initial M1 Zernike Coefficient Amplitudes (nm)": ", ".join(
            map(str, onp.asarray(initial_vals.get("m1_aperture.coefficients", onp.array([]))).tolist())
        ),
        "Initial M2 Zernike Coefficient Amplitudes (nm)": ", ".join(
            map(str, onp.asarray(initial_vals.get("m2_aperture.coefficients", onp.array([]))).tolist())
        ) if "m2_aperture.coefficients" in initial_vals else "",

        # --- Found (last values from history) ---
        "Found Source Position X (as)":     excel_friendly(last("x_position")),
        "Found Source Position Y (as)":     excel_friendly(last("y_position")),
        "Found Source Separation (as)":     excel_friendly(last("separation")),
        "Found Source Angle (deg)":         excel_friendly(last("position_angle")),
        "Found Source Log Flux":            excel_friendly(last("log_flux")),
        "Found Source Contrast (A:B)":      excel_friendly(last("contrast")),
        "Found Source A Raw Flux":          excel_friendly(last("raw_fluxes")[0] if last("raw_fluxes") is not None else None),
        "Found Source B Raw Flux":          excel_friendly(last("raw_fluxes")[1] if last("raw_fluxes") is not None else None),
        "Found Platescale (as/pixel)":      excel_friendly(last("psf_pixel_scale")),
        "Found M1 Zernikes (Noll Index)":   ", ".join(map(str, onp.asarray(getattr(model_initial_params, "m1_zernike_noll", onp.array([]))).tolist())),
        "Found M1 Zernike Coefficient Amplitudes (nm)": ", ".join(
            map(str, onp.asarray(last("m1_aperture.coefficients")).tolist()) if last("m1_aperture.coefficients") is not None else []),
        "Found M2 Zernike Coefficient Amplitudes (nm)": ", ".join(
            map(str, onp.asarray(last("m2_aperture.coefficients")).tolist()) if last("m2_aperture.coefficients") is not None else []),

        # --- Residuals (final) ---
        "Residual Source Position X (uas)": excel_friendly(residuals.get("x_position", [None])[-1] * 1e6 if "x_position" in residuals else None),
        "Residual Source Position Y (uas)": excel_friendly(residuals.get("y_position", [None])[-1] * 1e6 if "y_position" in residuals else None),
        "Residual Source Separation (uas)": excel_friendly(residuals.get("separation", [None])[-1] * 1e6 if "separation" in residuals else None),
        "Residual Source Angle (as)":       excel_friendly(residuals.get("position_angle", [None])[-1] * 60**2 if "position_angle" in residuals else None),
        "Residual Source A Flux (ppm)":     excel_friendly(residuals.get("raw_flux_error_ppm", [[None, None]])[-1][0] if "raw_flux_error_ppm" in residuals else None),
        "Residual Source B Flux (ppm)":     excel_friendly(residuals.get("raw_flux_error_ppm", [[None, None]])[-1][1] if "raw_flux_error_ppm" in residuals else None),
        "Residual Platescale (ppm)":        excel_friendly(residuals.get("platescale_error_ppm", [None])[-1] if "platescale_error_ppm" in residuals else None),
        "Residual M1 Zernike OPD RMS Error (nm)": excel_friendly(residuals.get("m1_zernike_opd_rms_nm", [None])[-1] if "m1_zernike_opd_rms_nm" in residuals else None),
        "Residual M1 Zernike Coefficient Errors (nm)": ", ".join(
            map(str, onp.asarray(residuals.get("m1_aperture.coefficients", [onp.array([])])[-1]).tolist())
        ) if "m1_aperture.coefficients" in residuals else "",
        "Residual M2 Zernike Coefficient Errors (nm)": ", ".join(
            map(str, onp.asarray(residuals.get("m2_aperture.coefficients", [onp.array([])])[-1]).tolist())
        ) if "m2_aperture.coefficients" in residuals else "",
    }

    # --- Eigenmode fields (conditional) ---
    if "use_eigen" in misc:
        row["Use Eigenmodes"] = excel_friendly(misc.get("use_eigen"))
    if "truncate_k" in misc:
        row["Eigenmode Truncation"] = excel_friendly(misc.get("truncate_k"))
    if "whiten_basis" in misc:
        row["Whiten Eigenbasis"] = excel_friendly(misc.get("whiten_basis"))
    if "eigen_coefficients" in true_vals:
        row["True Eigenmode Coefficients"] = ", ".join(
            map(str, onp.asarray(true_vals["eigen_coefficients"]).tolist())
        )
    if "eigen_coefficients" in initial_vals:
        row["Initial Eigenmode Coefficients"] = ", ".join(
            map(str, onp.asarray(initial_vals["eigen_coefficients"]).tolist())
        )
    if "eigen_coefficients" in history:
        row["Found Eigenmode Coefficients"] = ", ".join(
            map(str, onp.asarray(history["eigen_coefficients"][-1]).tolist())
        )
    if "eigen_coefficients" in residuals:
        row["Residual Eigenmode Coefficients"] = ", ".join(
            map(str, onp.asarray(residuals["eigen_coefficients"][-1]).tolist())
        )

    return row


# ----------------------------- Excel writer ----------------------------------
def save_results(
    save_path: str,
    save_name: str,
    *,
    true_vals: Dict[str, Any],
    initial_vals: Dict[str, Any],
    history: Dict[str, Any],
    residuals: Dict[str, Any],
    data_params: Any,
    model_initial_params: Any,
    data: Union[onp.ndarray, Any],
    misc: Dict[str, Any],
    sheet_name: str = "Results",
    overwrite: bool = False,
    coerce_numeric: bool = True,
    create_dirs: bool = True,
) -> int:
    """Write a single results row to an Excel workbook, with auto row detection.

    Behavior:
        - If `overwrite=True`, creates/replaces the file and writes a fresh sheet
          with a header and one data row.
        - If `overwrite=False` (default):
            * If the file/sheet exists, appends one row after the current last row.
            * If not, creates a new file/sheet with header + one row.

    Args:
        save_path: Directory to write the file into (will be created if needed).
        save_name: Workbook filename (e.g., "AstrometryRun_2025-08-15.xlsx").
        true_vals, initial_vals, history, residuals, data_params, model_initial_params, data:
            See `build_results_row` for the meaning of these inputs.
        misc: Mapping of additional run metadata (rng_seed, sweep indices, etc.).
        sheet_name: Sheet/tab name (default "Results").
        overwrite: If True, replace the entire file; if False, append if possible.
        coerce_numeric: If True, attempt to convert object columns that are numeric
            to numeric dtype before writing (helps Excel treat them as numbers).
        create_dirs: If True, create `save_path` if it doesn’t exist.

    Returns:
        The 1-based Excel row index that was written (i.e., the last used row).
        Example: If a new file is created, header is row 1 and your data row is row 2,
        so this function returns 2.

    Raises:
        ImportError: If the openpyxl engine is not available when appending.
    """
    if create_dirs:
        os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, save_name)
    row_dict = build_results_row(
        true_vals=true_vals,
        initial_vals=initial_vals,
        history=history,
        residuals=residuals,
        data_params=data_params,
        model_initial_params=model_initial_params,
        data=data,
        misc=misc,
    )

    df = pd.DataFrame([row_dict])

    if coerce_numeric:
        # Best-effort: cast object columns that look numeric to numeric dtype.
        # (We use errors="ignore" to avoid touching intended strings like "1, 2, 3".)
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(df[col], errors="ignore")

    # Overwrite or first-time write
    if overwrite or not os.path.exists(file_path):
        df.to_excel(file_path, index=False, sheet_name=sheet_name)
        return 2  # header row (1) + one data row (2)

    # Append to existing workbook/sheet
    try:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            if sheet_name in writer.book.sheetnames:
                ws = writer.book[sheet_name]
                startrow = ws.max_row  # openpyxl is 1-based; pandas handles startrow correctly
                header = False
            else:
                startrow = 0
                header = True

            df.to_excel(writer, index=False, header=header, sheet_name=sheet_name, startrow=startrow)
            last_row = 2 if header else startrow + 1
    except ValueError as e:
        # E.g., engine not found or incompatible openpyxl version
        raise ImportError(
            "Failed to append to Excel file. Ensure 'openpyxl' is installed and up to date."
        ) from e

    return last_row


# Utilities to allow logging during optimization steps
def _to_serializable(x):
    """Make JAX/NumPy scalars/arrays JSON-friendly."""
    if x is None:
        return None
    x = jax.device_get(x)
    if onp.isscalar(x):
        # floats/ints/bools
        return x.item() if hasattr(x, "item") else float(x)
    return onp.asarray(x).tolist()

def _pack_any(pytree, params_pure, initial_model_params, pack_params_fn):
    """
    Pack grads/updates for PURE runs; returns (vec, labels).
    Expects leaves keyed by pure param names in `params_pure`.
    """
    vec, labels = pack_params_fn(pytree, params_pure, initial_model_params)
    return onp.asarray(vec).ravel(), list(labels)

def _pack_eigen_leaf(pytree_or_updates):
    """
    Extract eigen vector from a ModelParams-like pytree where the only leaf is
    'eigen_coefficients'. Returns a flat numpy vector.
    """
    # Handle either ModelParams-like (with .params) or raw dict
    if hasattr(pytree_or_updates, "params"):
        arr = pytree_or_updates.params.get("eigen_coefficients", None)
    elif isinstance(pytree_or_updates, dict):
        arr = pytree_or_updates.get("eigen_coefficients", None)
    else:
        # Some optax transforms may already hand you a bare array
        arr = pytree_or_updates
    if arr is None:
        return None
    return onp.asarray(jax.device_get(arr)).ravel()

def log_step_jsonl(
    filepath: str,
    step_idx: int,
    *,
    loss,
    raw_grads,
    scaled_grads,
    updates,
    lr_model=None,                 # ModelParams-like or dict or scalar
    # Pure-space packing (always supply these)
    params_pure,
    initial_model_params,
    pack_params_fn,
    # Optional eigen back-projection (only used if we're in eigen mode)
    eig_B=None,                    # jax/np array (P x K)
    pure_labels=None,              # list[str], order used by `pack_params` for pure
    extras: dict = None            # anything else to stash
):
    """
    Append a single JSON line with loss, grads, updates, learning rates.
    Handles both PURE and EIGEN runs automatically.

    If in EIGEN mode and eig_B is provided, also logs 'grad_pure' (back-projected).
    """
    # Detect eigen vs pure by the presence of the eigen leaf
    is_eigen = hasattr(raw_grads, "params") and ("eigen_coefficients" in raw_grads.params)

    if is_eigen:
        # --- EIGEN SPACE ---
        g_raw   = _pack_eigen_leaf(raw_grads)        # (K,)
        g_scale = _pack_eigen_leaf(scaled_grads)     # (K,)
        upd     = _pack_eigen_leaf(updates)          # (K,) from optax
        # labels for eigen coeffs
        labels  = [f"c[{i}]" for i in range(g_raw.size)] if g_raw is not None else []

        # learning rates (optional)
        if lr_model is not None:
            lr_vec = _pack_eigen_leaf(lr_model)
        else:
            lr_vec = None

        # Optional: back-project eigen gradients to pure space for comparability
        grad_pure = None
        pure_lbls = None
        if eig_B is not None and g_raw is not None:
            grad_pure = onp.asarray(eig_B) @ onp.asarray(g_raw)  # (P,)
            pure_lbls = list(pure_labels) if pure_labels is not None else None

        record = {
            "t": step_idx,
            "time": time.time(),
            "mode": "eigen",
            "loss": _to_serializable(loss),
            "labels": labels,
            "grad_raw": _to_serializable(g_raw),
            "grad_scaled": _to_serializable(g_scale),
            "update": _to_serializable(upd),
            "learning_rate": _to_serializable(lr_vec),
        }
        if grad_pure is not None:
            record["grad_pure"] = _to_serializable(grad_pure)
            if pure_lbls is not None:
                record["pure_labels"] = pure_lbls

    else:
        # --- PURE SPACE ---
        g_raw, labels   = _pack_any(raw_grads,   params_pure, initial_model_params, pack_params_fn)
        g_scale, _      = _pack_any(scaled_grads, params_pure, initial_model_params, pack_params_fn)
        upd, _          = _pack_any(updates,      params_pure, initial_model_params, pack_params_fn)
        # learning rates (optional)
        if lr_model is not None:
            try:
                lr_vec, _  = _pack_any(lr_model, params_pure, initial_model_params, pack_params_fn)
            except Exception:
                lr_vec = None
        else:
            lr_vec = None

        record = {
            "t": step_idx,
            "time": time.time(),
            "mode": "pure",
            "loss": _to_serializable(loss),
            "labels": labels,
            "grad_raw": _to_serializable(g_raw),
            "grad_scaled": _to_serializable(g_scale),
            "update": _to_serializable(upd),
            "learning_rate": _to_serializable(lr_vec),
        }

    # optional extras (serializable)
    if extras:
        record["extras"] = {k: _to_serializable(v) for k, v in extras.items()}

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record) + "\n")
# -------- /Logging utilities --------
