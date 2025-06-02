# utils.py
import jax.numpy as np
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


def plot_parameter_history0(names, histories, figsize=(8, 5), display=True, save=False,
                           save_dir="../Results", save_name=None):
    plt.figure(figsize=figsize)
    if isinstance(names, str):
        names = [names]
        histories = [histories]
    for name, history in zip(names, histories):
        plt.plot(history, label=name)
    plt.title(f"Optimization History")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()

    # Save the plot if requested
    if save:
        if save_name is None: # Generate a new save_name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"parameter_history_{timestamp}.png"
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), save_dir))
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)

    # Show the plot if requested
    if display:
        plt.show()
    else:
        plt.close()


def plot_parameter_history(
        names,
        histories,
        true_vals=None,
        figsize=(8, 5),
        display=True,
        save=False,
        save_dir="../Results",
        save_name=None,
        log_scale=False,
        ax=None,
        plot_residuals=False,
        title=None,
        custom_labels=None,
        scale_factors=None,
        unit_labels=None
):
    """
    Plots the optimization history or residuals of one or more parameters.

    Parameters
    ----------
    names : str or list of str
        Parameter name(s).
    histories : list or list of lists
        Optimization history per parameter.
    true_vals : list or float, optional
        Ground truth value(s) for overlay or residual calculation.
    figsize : tuple, optional
        Figure size if creating new figure.
    display : bool, optional
        Whether to display the plot.
    save : bool, optional
        Whether to save the plot.
    save_dir : str, optional
        Directory to save the plot to.
    save_name : str, optional
        Filename to use when saving.
    log_scale : bool, optional
        If True, use log scale on y-axis.
    ax : matplotlib.axes.Axes, optional
        If provided, plot into this axis instead of creating a new figure.
    plot_residuals : bool, optional
        If True, plot (estimate - truth) residuals instead of raw values.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    import datetime
    import os
    import jax.numpy as np

    # Prepare inputs
    if isinstance(names, str):
        names = [names]
        histories = [histories]
        if true_vals is not None:
            true_vals = [true_vals]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i, (name, history) in enumerate(zip(names, histories)):
        y_data = np.array(history)

        # Subtract true value if residuals requested
        if plot_residuals and true_vals is not None:
            truth = true_vals[i]
            y_data = y_data - truth

        # Apply scaling if specified
        key = custom_labels[i] if custom_labels else name
        if scale_factors and key in scale_factors:
            y_data = y_data * scale_factors[key]

        # Label logic
        if custom_labels is not None:
            label = custom_labels[i]
        else:
            label = f"Residual {name}" if plot_residuals else f"Recovered {name}"

        ax.plot(y_data, label=label)

        if not plot_residuals and true_vals is not None:
            truth = true_vals[i]
            if np.ndim(truth) == 0:
                ax.axhline(truth, linestyle="--", color="k", alpha=0.6, label=f"True {name}")
            else:
                ax.plot([0, len(history)], [truth, truth], linestyle="--", color="k", alpha=0.6)

    if plot_residuals:
        ax.axhline(0, linestyle="--", color="gray", alpha=0.5)

    if log_scale:
        ax.set_yscale("log")

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Parameter Residuals" if plot_residuals else "Parameter History")

    ax.set_xlabel("Iteration")
    # Determine y-axis unit label
    unit = None
    if unit_labels:
        # Use first matching unit based on parameter names
        units = [unit_labels.get(name) for name in names if name in unit_labels]
        unique_units = list(set(u for u in units if u is not None))
        if len(unique_units) == 1:
            unit = unique_units[0]

    ylabel = "Residual" if plot_residuals else "Value"
    if unit:
        ylabel += f" ({unit})"

    ax.set_ylabel(ylabel)

    ax.legend()

    # Save the plot if requested
    if save:
        if save_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"parameter_history_{timestamp}.png"
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), save_dir))
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, save_name), dpi=300)

    if display:
        plt.show()
    else:
        plt.close()

    return fig

def plot_parameter_history_grid(param_histories, true_vals=None, log_scale=False, cols=3, figsize=(15, 10)):
    import math

    param_names = list(param_histories.keys())
    n_params = len(param_names)
    rows = math.ceil(n_params / cols)
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for i, param in enumerate(param_names):
        r, c = divmod(i, cols)
        ax = axs[r][c]
        history = param_histories[param]
        true_val = true_vals[param] if true_vals and param in true_vals else None
        plot_parameter_history(param, history, true_vals=true_val, log_scale=log_scale, ax=ax, display=False)

    # Hide unused axes
    for i in range(n_params, rows * cols):
        fig.delaxes(axs[i // cols][i % cols])

    plt.tight_layout()
    plt.show()
    return fig


def plot_psf_comparison(
        data,
        model,
        var=None,
        extent=None,
        model_label="Model PSF",
        display=True,
        save=False,
        save_dir="../Results",
        save_name=None,
        cmap="inferno",
        diverging_cmap="seismic",
        dpi=300,
        suptitle=True,
        font_size=12
):
    """
    Plots a 2x2 comparison of data, model PSF, residual, and Z-score.

    Parameters
    ----------
    data : ndarray or object with .model()
        Observed data array or a dLux optical model object.
    model : ndarray or object with .model()
        Model PSF array or a dLux optical model object.
    var : ndarray or None
        Variance image (same shape as `data`). If None, it is estimated as
        np.maximum(data, 0) after evaluating the data model.
    extent : list or None
        Axis extent for imshow (e.g., [-1, 1, -1, 1] in arcseconds).
        If None, pixel coordinates are used.
    model_label : str
        Title for the model PSF subplot.
    display : bool, optional
        Whether to display the plot.
    save : bool, optional
        Whether to save the figure to disk.
    save_dir : str, optional
        Directory to save the figure. Ignored unless `save=True`.
    save_name : str, optional
        Name of the saved file. Defaults to "psf_comparison.png".
    cmap : str
        Colormap for the data and model PSF plots.
    diverging_cmap : str
        Colormap for the residual and Z-score plots.
    dpi : int
        Dots per inch when saving the figure.
    suptitle : bool or str
        If True, displays the reduced chi-squared (χ²/ν) as the figure title.
        If a string is provided, it is used as a custom title.
    font_size : int
        Font size for all axis labels and titles.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting matplotlib figure object.
    """

    # Evaluate model and data if needed
    psf_data = data.model() if hasattr(data, "model") else data
    psf_model = model.model() if hasattr(model, "model") else model

    # Handle default variance
    if var is None:
        var = np.maximum(psf_data, 0)

    # Compute residuals and Z-score
    resid = psf_data - psf_model
    z_score = resid / np.sqrt(var)
    v_res = np.nanmax(np.abs(resid))
    v_z = np.nanmax(np.abs(z_score))

    # Chi² per degree of freedom
    try:
        dof = psf_data.size - sum(np.size(v) for v in getattr(model, "params", {}).values())
    except Exception:
        dof = psf_data.size
    chi2 = np.nansum(z_score**2) / dof if dof > 0 else np.nan

    # Create figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    if suptitle:
        title_text = r"$\chi^2_\nu$: {:.3f}".format(chi2) if suptitle is True else suptitle
        fig.suptitle(title_text, fontsize=font_size + 2)

    images = [
        ("Data", psf_data, cmap, None, None),
        (model_label, psf_model, cmap, None, None),
        ("Residuals", resid, diverging_cmap, -v_res, v_res),
        ("Z-Score", z_score, diverging_cmap, -v_z, v_z)
    ]

    for ax, (title, image, cm, vmin, vmax) in zip(axs.flat, images):
        if extent is None:
            im = ax.imshow(image, cmap=cm, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(image, cmap=cm, extent=extent, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=font_size)
        ax.set_xlabel("X (arcsec)" if extent is not None else "X (px)", fontsize=font_size)
        ax.set_ylabel("Y (arcsec)" if extent is not None else "Y (px)", fontsize=font_size)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        if title in ["Residuals", "Z-Score"]:
            cbar.set_label("Z-score" if "Z" in title else "Δ Photons")
        else:
            cbar.set_label("Photons")

    fig.tight_layout(rect=[0, 0.03, 1, 0.95] if suptitle else None)

    # Save the figure if requested
    if save:
        os.makedirs(save_dir, exist_ok=True)
        full_name = os.path.join(save_dir, save_name or "psf_comparison.png")
        fig.savefig(full_name, dpi=dpi)

    # Show or close
    if display:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_opd_surface(opd, title, figsize=(8, 6), mask=None, extent=None, cmap="inferno", vmin=None, vmax=None,
                     save=False, display=True, save_dir="../Results", save_name=None):
    plt.figure(figsize=figsize)
    if mask is not None:
        opd = opd * mask
    plt.imshow(1e9 * opd, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    plt.colorbar(label="OPD (nm)")
    plt.title(title)
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.tight_layout()
    # Save the plot if requested
    if save:
        if save_name is None: # Generate a new save_name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"surface_opd_{timestamp}.png"
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), save_dir))
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)

    # Show the plot if requested
    if display:
        plt.show()
    else:
        plt.close()
