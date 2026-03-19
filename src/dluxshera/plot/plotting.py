"""Plotting helpers used across the refactor-era stack.

The utilities in this module follow a consistent IO policy:

* Accept optional Matplotlib axes/figures; create new ones when omitted.
* Return ``(fig, ax)`` or ``(fig, axes)`` to let callers further customise.
* Never call ``plt.show()`` implicitly; callers control display via ``show``.
* Support optional saving through a single ``save_path`` argument.

These helpers keep the spirit of the earlier plotting routines (PSF and
parameter history visualisation, colourbar alignment) while being predictable
for headless test environments and notebook workflows alike.
"""

from pathlib import Path
from typing import List, Mapping, MutableSequence, Optional, Sequence, Tuple, Union

import math

import matplotlib

matplotlib.use("Agg", force=True)

import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dluxshera.utils.utils import get_param_scale_and_unit

ArrayLike = Union[onp.ndarray, np.ndarray]
PanelPaths = MutableSequence[Path]

__all__ = [
    "apply_plot_defaults",
    "get_default_cmaps",
    "make_nan_cmaps",
    "merge_cbar",
    "plot_parameter_history",
    "plot_parameter_history_grid",
    "plot_psf_single",
    "plot_psf_comparison",
    "plot_opd_surface",
    "choose_subplot_grid",
    "plot_parameter_sweeps",
    "plot_fim",
    "plot_eigenvalue_spectrum",
    "plot_signals_panels",
    "plot_signals_grid",
]


def apply_plot_defaults(
    *,
    font_family: str = "serif",
    image_origin: str = "lower",
    figure_dpi: int = 120,
) -> None:
    """Apply default matplotlib rcParams used in the refactor scripts."""

    plt.rcParams["font.family"] = font_family
    plt.rcParams["image.origin"] = image_origin
    plt.rcParams["figure.dpi"] = figure_dpi


def make_nan_cmaps(
    names: Sequence[str],
    *,
    bad_color: str = "k",
    bad_alpha: float = 0.5,
) -> Mapping[str, matplotlib.colors.Colormap]:
    """Create colormaps with NaNs rendered as a specified color.

    Parameters
    ----------
    names : sequence[str]
        Names of Matplotlib colormaps to clone.
    bad_color : str
        Color to use for NaNs/invalid entries.
    bad_alpha : float
        Alpha value to use for NaNs/invalid entries.

    Returns
    -------
    Mapping[str, matplotlib.colors.Colormap]
        Mapping of colormap names to NaN-safe colormap objects.
    """

    cmaps = {}
    for name in names:
        cmap = matplotlib.colormaps[name].copy()
        cmap.set_bad(bad_color, bad_alpha)
        cmaps[name] = cmap
        registered_name = f"{name}_nan"
        matplotlib.colormaps.register(cmap, name=registered_name, force=True)
        cmaps[registered_name] = cmap
    return cmaps


def get_default_cmaps(
    *,
    bad_color: str = "k",
    bad_alpha: float = 0.5,
) -> Mapping[str, matplotlib.colors.Colormap]:
    """Return the default plotting colormaps with NaN handling."""

    return make_nan_cmaps(
        ["inferno", "viridis", "seismic", "coolwarm"],
        bad_color=bad_color,
        bad_alpha=bad_alpha,
    )


def _normalise_histories(
    names: Optional[Union[str, Sequence[str]]],
    histories: Union[Sequence[ArrayLike], Mapping[str, ArrayLike]],
    true_vals: Optional[Union[Sequence[float], Mapping[str, float]]],
) -> Tuple[List[str], List[ArrayLike], Optional[List[float]]]:
    """Normalise flexible inputs for history plotting."""

    if isinstance(histories, Mapping) and names is None:
        names = list(histories.keys())
        history_list = [histories[n] for n in names]
    else:
        if isinstance(names, str):
            names = [names]
            history_list = [histories]
        else:
            assert names is not None, "Parameter names must be provided."
            history_list = list(histories)

    names_list = list(names)

    if true_vals is None:
        true_list = None
    elif isinstance(true_vals, Mapping):
        true_list = [true_vals.get(name) for name in names_list]
    else:
        true_list = list(true_vals)

    return names_list, history_list, true_list


def _maybe_save(fig, save_path: Optional[Union[str, Path]], dpi: int = 300) -> None:
    if save_path is None:
        return
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)




def _plot_lines_on_ax(
    ax,
    x,
    ys,
    labels,
    title: str,
    ylabel: str,
) -> None:
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)
    ax.axhline(0, linestyle="--", color="k", alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()


def _plot_lines(x, ys, labels, title: str, ylabel: str, save_path: Path) -> None:
    fig, ax = plt.subplots()
    _plot_lines_on_ax(ax, x, ys, labels, title, ylabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def _iter_signal_panels(
    signals: Mapping[str, ArrayLike],
    x,
    *,
    title_prefix: Optional[str] = None,
    include_zernike_rms: bool = False,
):
    def title(base: str) -> str:
        return f"{title_prefix} — {base}" if title_prefix else base

    astrometry = [
        ("source.x_error_uas", "Δx"),
        ("source.y_error_uas", "Δy"),
    ]
    if all(key in signals for key, _ in astrometry):
        ys = [signals[key] for key, _ in astrometry]
        labels = [label for _, label in astrometry]
        yield {
            "title": title("Astrometry residuals (µas)"),
            "ylabel": "Residual (µas)",
            "ys": ys,
            "labels": labels,
            "filename": "astrometry_residuals_uas.png",
        }

    if "source.separation_error_uas" in signals:
        yield {
            "title": title("Separation residual (µas)"),
            "ylabel": "Residual (µas)",
            "ys": [signals["source.separation_error_uas"]],
            "labels": ["Δρ"],
            "filename": "separation_residual_uas.png",
        }

    if "source.position_angle_error_as" in signals:
        yield {
            "title": title("Position angle residual (arcsec)"),
            "ylabel": "Residual (arcsec)",
            "ys": [signals["source.position_angle_error_as"]],
            "labels": ["ΔPA"],
            "filename": "position_angle_residual_as.png",
        }

    if "optics.plate_scale_error_ppm" in signals:
        yield {
            "title": title("Plate scale error (ppm)"),
            "ylabel": "Error (ppm)",
            "ys": [signals["optics.plate_scale_error_ppm"]],
            "labels": ["Plate scale"],
            "filename": "plate_scale_error_ppm.png",
        }

    if "source.raw_flux_error_ppm" in signals:
        flux_err = signals["source.raw_flux_error_ppm"]
        if flux_err.ndim == 2 and flux_err.shape[1] >= 2:
            ys = [flux_err[:, 0], flux_err[:, 1]]
            labels = ["Star A", "Star B"]
        else:
            ys = [flux_err]
            labels = ["Raw flux"]
        yield {
            "title": title("Raw flux error (ppm)"),
            "ylabel": "Error (ppm)",
            "ys": ys,
            "labels": labels,
            "filename": "raw_flux_error_ppm.png",
        }

    if "optics.primary.zernike_error_nm" in signals:
        zerr = signals["optics.primary.zernike_error_nm"]
        if zerr.ndim == 2 and zerr.shape[1] > 0:
            ys = [zerr[:, i] for i in range(zerr.shape[1])]
            labels = [f"M1 Z{i + 4}" for i in range(zerr.shape[1])]
            yield {
                "title": title("M1 Zernike component residuals (nm)"),
                "ylabel": "Residual (nm)",
                "ys": ys,
                "labels": labels,
                "filename": "m1_zernike_components_nm.png",
            }

    if include_zernike_rms and "primary.zernike_rms_nm" in signals:
        yield {
            "title": title("M1 Zernike RMS (nm)"),
            "ylabel": "RMS Error (nm)",
            "ys": [signals["primary.zernike_rms_nm"]],
            "labels": ["Zernike RMS"],
            "filename": "m1_zernike_rms_nm.png",
        }

    if "optics.secondary.zernike_error_nm" in signals:
        zerr = signals["optics.secondary.zernike_error_nm"]
        if zerr.ndim == 2 and zerr.shape[1] > 0:
            ys = [zerr[:, i] for i in range(zerr.shape[1])]
            labels = [f"M2 Z{i + 4}" for i in range(zerr.shape[1])]
            yield {
                "title": title("M2 Zernike component residuals (nm)"),
                "ylabel": "Residual (nm)",
                "ys": ys,
                "labels": labels,
                "filename": "m2_zernike_components_nm.png",
            }

    if include_zernike_rms and "secondary.zernike_rms_nm" in signals:
        yield {
            "title": title("M2 Zernike RMS (nm)"),
            "ylabel": "RMS Error (nm)",
            "ys": [signals["secondary.zernike_rms_nm"]],
            "labels": ["Zernike RMS"],
            "filename": "m2_zernike_rms_nm.png",
        }


def merge_cbar(ax):
    """Append a slim colourbar axis flush with ``ax`` and return it."""

    return make_axes_locatable(ax).append_axes("right", size="5%", pad=0.0)


def plot_parameter_history(
    names,
    histories,
    true_vals=None,
    figsize=(8, 5),
    log_scale=False,
    ax=None,
    plot_residuals=False,
    title=None,
    custom_labels=None,
    scale_factors=None,
    unit_labels=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = True,
):
    """
    Plot optimisation history or residuals for one or more parameters.

    Parameters
    ----------
    names : str | sequence[str] | None
        Parameter names. If ``None`` and ``histories`` is a mapping, keys are
        used as names.
    histories : sequence | mapping
        Per-parameter history arrays. When a mapping is provided and
        ``names`` is ``None``, the mapping keys define the plotting order.
    true_vals : sequence | mapping | None
        Ground truth values for overlay or residual calculation.
    figsize : tuple
        Figure size when creating a new figure.
    log_scale : bool
        Plot the y-axis on a logarithmic scale.
    ax : matplotlib.axes.Axes | None
        Target axes. When ``None``, a new ``(fig, ax)`` is created.
    plot_residuals : bool
        Plot ``estimate - truth`` if ``true_vals`` are provided.
    title : str | None
        Custom title for the axes.
    custom_labels : sequence[str] | None
        Labels to use instead of derived defaults.
    scale_factors : mapping[str, float] | None
        Optional per-parameter scale factors applied to histories.
    unit_labels : mapping[str, str] | None
        Optional units per parameter to append to the y-label.
    save_path : str | Path | None
        If provided, save the figure to this path.
    show : bool
        Whether to call ``plt.show()`` at the end.
    close : bool
        Close the figure when ``show`` is ``False`` to avoid leaked figures in
        tests.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes containing the plot.
    """

    names_list, history_list, true_list = _normalise_histories(names, histories, true_vals)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for i, (name, history) in enumerate(zip(names_list, history_list)):
        y_data = np.array(history)

        truth = true_list[i] if true_list is not None else None
        if plot_residuals and truth is not None:
            y_data = y_data - truth

        key = custom_labels[i] if custom_labels else name
        if scale_factors and key in scale_factors:
            y_data = y_data * scale_factors[key]

        label = custom_labels[i] if custom_labels is not None else (
            f"Residual {name}" if plot_residuals else f"Recovered {name}"
        )
        ax.plot(y_data, label=label)

        if not plot_residuals and truth is not None:
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
    unit = None
    if unit_labels:
        units = [unit_labels.get(name) for name in names_list if name in unit_labels]
        unique_units = list({u for u in units if u is not None})
        if len(unique_units) == 1:
            unit = unique_units[0]

    ylabel = "Residual" if plot_residuals else "Value"
    if unit:
        ylabel += f" ({unit})"
    ax.set_ylabel(ylabel)

    ax.legend()

    _maybe_save(fig, save_path)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, ax


def plot_signals_panels(
    signals: Mapping[str, ArrayLike],
    out_dir: Path,
    *,
    title_prefix: Optional[str] = None,
    include_zernike_rms: bool = False,
) -> PanelPaths:
    """
    Render standard diagnostic panels from Signals.

    Parameters
    ----------
    signals:
        Mapping from signal names to numpy arrays.
    out_dir:
        Run directory; plots are written to ``out_dir``.
    title_prefix:
        Optional prefix applied to each panel title.
    include_zernike_rms:
        Whether to include M1/M2 Zernike RMS panels (default False).

    Returns
    -------
    list[Path]
        Paths to saved PNG files.
    """


    x = onp.arange(next(iter(signals.values())).shape[0])
    saved: PanelPaths = []

    for panel in _iter_signal_panels(
        signals,
        x,
        title_prefix=title_prefix,
        include_zernike_rms=include_zernike_rms,
    ):
        path = Path(out_dir / panel["filename"] )
        _plot_lines(x, panel["ys"], panel["labels"], panel["title"], panel["ylabel"], path)
        saved.append(path)

    return saved


def plot_signals_grid(
    signals: Mapping[str, ArrayLike],
    out_dir: Optional[Union[str, Path]] = None,
    *,
    title_prefix: Optional[str] = None,
    include_zernike_rms: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    show: bool = False,
    close: bool = True,
):
    """
    Render standard diagnostic panels into a grid of subplots.

    Parameters
    ----------
    signals:
        Mapping from signal names to numpy arrays.
    out_dir:
        Optional output directory; plots are written to ``out_dir`` when provided
    title_prefix:
        Optional prefix applied to each panel title.
    include_zernike_rms:
        Whether to include M1/M2 Zernike RMS panels (default False).
    figsize:
        Optional explicit figure size.
    show:
        Whether to call ``plt.show()`` at the end.
    close:
        Close the figure when ``show`` is ``False`` to avoid leaked figures.

    Returns
    -------
    fig, axes
        Matplotlib figure and axes array containing the grid.
    """


    x = onp.arange(next(iter(signals.values())).shape[0])
    panels = list(
        _iter_signal_panels(
            signals,
            x,
            title_prefix=title_prefix,
            include_zernike_rms=include_zernike_rms,
        )
    )
    if not panels:
        raise ValueError("No panels available for the provided signals mapping.")

    rows, cols = choose_subplot_grid(len(panels))
    if figsize is None:
        figsize = (5 * cols, 3.5 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for ax, panel in zip(axes_flat, panels):
        _plot_lines_on_ax(ax, x, panel["ys"], panel["labels"], panel["title"], panel["ylabel"])

    for ax in axes_flat[len(panels):]:
        fig.delaxes(ax)

    fig.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir / "signals_grid.png" )

    _maybe_save(fig, out_dir)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, axes


def plot_parameter_history_grid(
    param_histories,
    true_vals=None,
    log_scale=False,
    cols=3,
    figsize=(15, 10),
    plot_residuals: bool = False,
    unit_labels=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = True,
):
    """Plot a grid of parameter histories using :func:`plot_parameter_history`.

    ``unit_labels`` mirrors the single-plot helper, allowing refactor-era unit
    mappings to flow into each panel.
    """

    param_names = list(param_histories.keys())
    n_params = len(param_names)
    rows = math.ceil(n_params / cols)
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for i, param in enumerate(param_names):
        r, c = divmod(i, cols)
        ax = axs[r][c]
        history = param_histories[param]
        if isinstance(true_vals, Mapping):
            truth = true_vals.get(param)
        elif true_vals is not None:
            truth = true_vals[i]
        else:
            truth = None
        plot_parameter_history(
            param,
            history,
            true_vals=[truth] if truth is not None else None,
            log_scale=log_scale,
            ax=ax,
            plot_residuals=plot_residuals,
            unit_labels=unit_labels,
            show=False,
            close=False,
        )

    for i in range(n_params, rows * cols):
        fig.delaxes(axs[i // cols][i % cols])

    fig.tight_layout()

    _maybe_save(fig, save_path)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, axs


def plot_psf_single(
    psf,
    extent=None,
    title="PSF",
    ax=None,
    cmap="inferno",
    dpi=300,
    font_size=12,
    normalise=True,
    stretch="sqrt",  # "linear", "sqrt", or "log"
    vmin=None,
    vmax=None,
    cbar_label=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = True,
):
    """
    Plot a single PSF image.

    Parameters
    ----------
    psf : ndarray or object with .model()
        PSF image or a dLux optical model object.
    extent : list or None
        Axis extent for imshow (e.g., [-1, 1, -1, 1] in arcseconds).
        If None, pixel coordinates are used.
    title : str
        Title for the PSF subplot.
    ax : matplotlib.axes.Axes | None
        Optional target axes.
    cmap : str
        Colormap for the PSF plot.
    dpi : int
        Dots per inch when saving the figure.
    font_size : int
        Font size for axis labels and titles.
    normalise : bool
        If True, normalise the PSF by its maximum value.
    stretch : {"linear", "sqrt", "log"}
        Intensity stretch applied before plotting.\n
        Supported: "linear", "sqrt", "log"\n
        "sqrt" is often nice for PSF visualisation.
    vmin, vmax : float or None
        Optional explicit limits for imshow. If None, determined from the
        stretched image.
    cbar_label : str or None
        Label for the colorbar. If None, a sensible default is chosen.
    save_path : str | Path | None
        If provided, save the figure to this path.
    show : bool
        Whether to display the figure via ``plt.show()``.
    close : bool
        Close the figure when ``show`` is ``False``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting matplotlib figure object.
    ax : matplotlib.axes.Axes
        The axis containing the PSF plot.
    """

    # Evaluate PSF if a model object is passed
    psf_img = psf.model() if hasattr(psf, "model") else psf
    psf_img = onp.array(psf_img)

    # Normalise if requested
    if normalise:
        max_val = onp.nanmax(psf_img)
        if max_val > 0:
            psf_img = psf_img / max_val

    # Apply intensity stretch
    if stretch.lower() == "linear":
        img_to_plot = psf_img
        default_cbar = "Normalised Photons" if normalise else "Photons"

    elif stretch.lower() == "sqrt":
        img_to_plot = onp.sqrt(onp.maximum(psf_img, 0))
        default_cbar = r"$\sqrt{\mathrm{Photons}}$ (normalised)" if normalise else r"$\sqrt{\mathrm{Photons}}$"

    elif stretch.lower() == "log":
        # Avoid log of zero/negative
        eps = 1e-12 * onp.nanmax(psf_img) if onp.nanmax(psf_img) > 0 else 1e-12
        img_to_plot = onp.log10(onp.clip(psf_img, eps, None))
        default_cbar = r"$\log_{10}$ Photons (normalised)" if normalise else r"$\log_{10}$ Photons"
    else:
        raise ValueError(f"Unknown stretch '{stretch}'. Use 'linear', 'sqrt', or 'log'.")

    # Determine vmin/vmax if not set
    if vmin is None:
        vmin = onp.nanmin(img_to_plot)
    if vmax is None:
        vmax = onp.nanmax(img_to_plot)

    # Fall back colorbar label
    if cbar_label is None:
        cbar_label = default_cbar

    # Create figure and axis
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    else:
        fig = ax.figure

    if extent is None:
        im = ax.imshow(img_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xlabel("X (px)", fontsize=font_size)
        ax.set_ylabel("Y (px)", fontsize=font_size)
    else:
        im = ax.imshow(img_to_plot, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
        ax.set_xlabel("X (arcsec)", fontsize=font_size)
        ax.set_ylabel("Y (arcsec)", fontsize=font_size)

    ax.set_title(title, fontsize=font_size)

    # Colorbar using your merge_cbar helper
    cax = merge_cbar(ax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=font_size)

    fig.tight_layout()

    _maybe_save(fig, save_path, dpi=dpi)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, ax


def plot_psf_comparison(
    data,
    model,
    var=None,
    extent=None,
    model_label="Model PSF",
    cmap="inferno",
    diverging_cmap="seismic",
    dpi=300,
    suptitle=True,
    font_size=12,
    axes=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = True,
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
    axes : array-like of matplotlib.axes.Axes | None
        Optional pre-created 2x2 axes grid.
    save_path : str | Path | None
        If provided, save the figure to this path.
    show : bool
        Whether to display the figure via ``plt.show()``.
    close : bool
        Close the figure when ``show`` is ``False``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The resulting matplotlib figure object.
    axes : ndarray
        The 2x2 array of axes used for plotting.
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
    if axes is None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    else:
        axs = onp.array(axes)
        fig = axs.ravel()[0].figure
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

    _maybe_save(fig, save_path, dpi=dpi)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, axs


def plot_opd_surface(
    opd,
    title,
    figsize=(8, 6),
    mask=None,
    extent=None,
    cmap="inferno",
    vmin=None,
    vmax=None,
    ax=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = True,
):
    """Plot a surface of optical path difference (OPD) in nanometres."""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if mask is not None:
        opd = opd * mask
    im = ax.imshow(1e9 * opd, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    cax = merge_cbar(ax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("OPD (nm)")
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    fig.tight_layout()

    _maybe_save(fig, save_path)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, ax


def choose_subplot_grid(n):
    """Choose a reasonable ``(rows, cols)`` layout for ``n`` subplots."""
    if n <= 3:
        return int(n), 1
    elif n <= 6:
        rows = int(np.ceil(n / 2))
        return rows, 2
    else:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        return rows, cols


def plot_fim(
    fim: np.ndarray,
    labels: Sequence[str],
    log_scale: bool = True,
    vmin=None,
    vmax=None,
    cmap: str = "viridis",
    cbar_label: Optional[str] = None,
    figsize=(8, 6),
    eps: float = 1e-20,
    ax=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = True,
):
    """
    Plot a Fisher Information Matrix (FIM) heatmap.

    Parameters
    ----------
    fim : ndarray
        Fisher information matrix to plot.
    labels : sequence[str]
        Labels for the matrix axes.
    log_scale : bool
        Whether to apply the legacy log-scaling (log10(abs(fim) + eps)).
    vmin, vmax : float or None
        Optional explicit color scaling for imshow.
    cmap : str
        Colormap for the heatmap.
    cbar_label : str | None
        Optional label for the colorbar, overrides automatic labels if provided.
    figsize : tuple
        Figure size when creating a new figure.
    eps : float
        Small value added before log scaling to avoid log(0).
    ax : matplotlib.axes.Axes | None
        Optional target axes.
    save_path : str | Path | None
        If provided, save the figure to this path.
    show : bool
        Whether to display the figure via ``plt.show()``.
    close : bool
        Close the figure when ``show`` is ``False``.

    Returns
    -------
    fig, ax
        Matplotlib figure and axes containing the plot.
    """

    fim_array = onp.array(fim)
    if log_scale:
        data = onp.log10(onp.abs(fim_array) + eps)
        if cbar_label is None:
            cbar_label = r"$\log_{10}(|\mathrm{FIM}| + \epsilon)$"
    else:
        data = fim_array
        if cbar_label is None:
            cbar_label = "FIM"

    if vmin is None:
        vmin = onp.nanmin(data)
    if vmax is None:
        vmax = onp.nanmax(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    tick_positions = onp.arange(len(labels))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    cax = merge_cbar(ax)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(cbar_label)

    fig.tight_layout()

    _maybe_save(fig, save_path)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, ax


def plot_eigenvalue_spectrum(
    eigvals: ArrayLike,
    eigvecs: Optional[ArrayLike] = None,
    labels: Optional[Sequence[str]] = None,
    truncate_k: Optional[int] = None,
    label_top_k: Optional[int] = None,
    label_fontsize: int = 9,
    label_boxes: bool = True,
    alternate_updn: bool = True,
    ax=None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    close: bool = True,
):
    """Plot eigenvalue spectrum with optional dominant-parameter annotations."""

    eigvals_arr = onp.asarray(eigvals)
    x = onp.arange(len(eigvals_arr))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.semilogy(x, eigvals_arr, marker="o")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.set_xlabel("Eigenmode Index")
    ax.set_ylabel("Eigenvalue (log scale)")

    if eigvecs is not None and labels is not None:
        eigvecs_arr = onp.asarray(eigvecs)
        dom_idx = onp.argmax(onp.abs(eigvecs_arr), axis=0)
        label_limit = len(x) if label_top_k is None else min(len(x), int(label_top_k))
        bbox = dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8) if label_boxes else None

        for k in range(label_limit):
            dy = 10
            if alternate_updn:
                dy = 12 if k % 2 == 0 else -14
            ax.annotate(
                labels[int(dom_idx[k])],
                xy=(x[k], eigvals_arr[k]),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va="bottom" if dy >= 0 else "top",
                fontsize=label_fontsize,
                bbox=bbox,
            )

    if truncate_k is not None:
        ax.axvline(truncate_k - 0.5, linestyle="--", color="k", alpha=0.6)
        _, y_max = ax.get_ylim()
        ax.text(
            truncate_k - 0.5,
            y_max * 0.9,
            f"k={truncate_k}",
            ha="right",
            va="top",
            fontsize=label_fontsize,
        )

    fig.tight_layout()
    _maybe_save(fig, save_path)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig, ax


# --- Legacy Functions ---
def plot_parameter_sweeps(
    df,
    highlight_min=True,
    sharey=False,
    show: bool = False,
    close: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi=300,
    font_size=12,
    reference_params=None,
    true_loss=None,
):
    """Plot loss vs. Δvalue for each parameter sweep in a grid layout."""

    from matplotlib.ticker import MaxNLocator

    all_params = df["parameter"].unique()
    n = len(all_params)
    rows, cols = choose_subplot_grid(n)

    fig, axes = plt.subplots(int(rows), int(cols), figsize=(5.5 * cols, 4 * rows), sharey=sharey)
    axes = onp.atleast_1d(axes).flatten()
    df = df.copy()  # Prevent modifying original

    # Compute delta values
    if reference_params is not None:
        df_params = df["parameter"].unique()

        try:
            path_map = reference_params.get_param_path_map()
            inv_path_map = {v: k for k, v in path_map.items()}
        except AttributeError:
            path_map = {}
            inv_path_map = {}

        for param in df_params:
            ref_key = param
            if ref_key not in reference_params.params:
                ref_key = path_map.get(param, inv_path_map.get(param, param))

            try:
                ref_val = reference_params.get(ref_key)
            except (KeyError, ValueError):
                print(f"⚠️  Warning: Could not resolve reference for parameter '{param}'. Skipping delta.")
                continue

            if np.ndim(ref_val) > 0:
                for i in range(len(ref_val)):
                    mask = (df["parameter"] == param) & (df["index"] == i)
                    df.loc[mask, "delta"] = df.loc[mask, "value"] - float(ref_val[i])
            else:
                mask = (df["parameter"] == param)
                df.loc[mask, "delta"] = df.loc[mask, "value"] - float(ref_val)
    else:
        df["delta"] = df["value"]

    model_path_to_noll = {}
    if reference_params is not None:
        try:
            path_map = reference_params.get_param_path_map()
            for param_key, model_path in path_map.items():
                if param_key.endswith("zernike_amp"):
                    noll_key = param_key.replace("zernike_amp", "zernike_noll")
                    try:
                        model_path_to_noll[model_path] = reference_params.get(noll_key)
                    except KeyError:
                        continue
        except AttributeError:
            pass

    for i, param in enumerate(all_params):
        scale, unit = get_param_scale_and_unit(param)
        ax = axes[i]
        sub = df[df["parameter"] == param]

        if sub["index"].isna().all():
            ax.plot(sub["delta"] * scale, sub["loss"], label=param, color="black")
            if highlight_min:
                min_idx = sub["loss"].idxmin()
                ax.axvline(sub.loc[min_idx, "delta"] * scale, color="red", linestyle="--", label="Min loss")
        else:
            for idx, group in sub.groupby("index"):
                if param in model_path_to_noll:
                    try:
                        noll = int(model_path_to_noll[param][int(idx)])
                        label = f"Noll {noll}"
                    except (IndexError, ValueError, TypeError):
                        label = f"{param}[{int(idx)}]"
                else:
                    label = f"{param}[{int(idx)}]"

                ax.plot(group["delta"] * scale, group["loss"], label=label)
                if highlight_min:
                    min_idx = group["loss"].idxmin()
                    ax.axvline(group.loc[min_idx, "delta"] * scale, color="red", linestyle=":", alpha=0.5)

        if true_loss is not None:
            ax.axhline(true_loss, color="gray", linestyle="--", alpha=0.6, label="True loss")

        ax.set_title(param, fontsize=font_size)
        if reference_params is not None:
            ax.set_xlabel(f"Δ{param} [{unit}]", fontsize=font_size)
        else:
            ax.set_xlabel(f"{param} [{unit}]", fontsize=font_size)
        ax.set_ylabel("Loss", fontsize=font_size)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(labelsize=font_size - 2)

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=font_size - 2)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()

    _maybe_save(fig, save_path, dpi=dpi)

    if show:
        plt.show()
    elif close:
        plt.close(fig)

    return fig
