#!/usr/bin/env python3
"""Render standalone SHERA diffractive-pupil OPD and single-star PSF figures.

This experimental script mirrors the grating construction used by
``examples/scripts/generate_target_grating_portraits.py`` without importing the
example as a library.  The combined DP plus grating OPD array plotted here is
the same array injected into the resolved SHERA optics configuration.

Run from the repository root:

    PYTHONPATH=src python work/experiments/render_shera_dp_psf_figures.py
"""

from __future__ import annotations

from copy import deepcopy
from math import pi
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import warnings

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dluxshera-matplotlib"),
)

import matplotlib

matplotlib.use("Agg", force=True)

import dLux.utils as dlu
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.patches import Circle

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.plot.plotting import apply_plot_defaults, get_default_cmaps, merge_cbar
from dluxshera.systems.base import SheraBinder, compose_forward_spec
from dluxshera.utils.single_star_calibration import (
    prepare_alpha_cen_a_single_star_system_config,
)
from dluxshera.utils.utils import default_diffractive_pupil_path, scale_array


# -------------------------------------------------------------------------
# User-adjustable settings
# -------------------------------------------------------------------------

SYSTEM_PRESET = "SHERA_FLIGHT_3P_SIMPLE"

PSF_NPIX = 128
PSF_OVERSAMPLE = 1
PUPIL_NPIX = 2048
EXPOSURE_TIME_S = 0.05
N_LAMBDA = 11

PSF_STRETCH = "sqrt"  # Supported: "log", "sqrt".
DISPLAY_PMIN = 1.0
DISPLAY_PMAX = 99.9

SHOW_DIAMETER_CIRCLE = True
CIRCLE_DIAMETER_AS = 10.0
CIRCLE_COLOR = "red"
CIRCLE_LINESTYLE = "--"
CIRCLE_LINEWIDTH = 1.7

DP_ENABLED = True
GRATING_ENABLED = True
GRATING_PHASE_AMPLITUDE_RAD = pi / 16.0
GRATING_FREQUENCY = 128.0
GRATING_ANGLE_DEG = 45.0
GRATING_PHASE_FLIP = True
GRATING_MASK_THRESHOLD = 0.5
BINARY_MASK = True
LEGACY_DP_CENTERING = True

OUTPUT_DIR = Path("Results/figures/shera_dp_psf")
OPD_FIGURE_NAME = "shera_diffractive_pupil_opd.png"
PSF_FIGURE_NAME = "shera_single_star_psf.png"
MODEL_OPD_NAME = "shera_combined_dp_grating_opd.npy"
SAVE_PDF = False
FIGURE_DPI = 600

OPD_CMAP_NAME = "inferno_nan"
PSF_CMAP_NAME = "inferno"

FONT_FAMILY = "sans-serif"
TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 13
COLORBAR_LABEL_FONT_SIZE = 15
COLORBAR_TICK_LABEL_FONT_SIZE = 12


# -------------------------------------------------------------------------
# Small plotting / coordinate helpers
# -------------------------------------------------------------------------


def image_extent_from_diameter(pupil_diameter_m: float) -> np.ndarray:
    """Return ``imshow`` extent in centimetres for a square pupil image."""

    half_width_cm = float(pupil_diameter_m) * 100.0 / 2.0
    return half_width_cm * np.array([-1.0, 1.0, -1.0, 1.0])


def psf_extent_as(
    image: np.ndarray,
    plate_scale_as_per_pix: float,
    oversample: int,
) -> np.ndarray:
    """Return ``imshow`` extent in arcseconds for a square rendered PSF image."""

    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError(f"Expected a square 2-D PSF image; got shape {image.shape}.")
    pixel_scale_as_per_pix = float(plate_scale_as_per_pix) / float(oversample)
    half_width_as = float(image.shape[-1]) * pixel_scale_as_per_pix / 2.0
    return half_width_as * np.array([-1.0, 1.0, -1.0, 1.0])


def _resolve_display_limits(
    image: np.ndarray,
    *,
    stretch: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    """Return percentile-based display limits matching target portrait plots."""

    finite_values = np.asarray(image, dtype=float)[np.isfinite(image)]
    if finite_values.size == 0:
        raise ValueError("No finite values found in image.")

    if stretch == "log":
        positive = finite_values[finite_values > 0.0]
        if positive.size == 0:
            raise ValueError("Log display requires positive image values.")
        auto_vmin = float(np.percentile(positive, DISPLAY_PMIN))
        auto_vmax = float(np.percentile(positive, DISPLAY_PMAX))
        auto_vmin = max(auto_vmin, float(np.min(positive)), np.finfo(float).tiny)
    else:
        auto_vmin = float(np.percentile(finite_values, DISPLAY_PMIN))
        auto_vmax = float(np.percentile(finite_values, DISPLAY_PMAX))

    resolved_vmin = auto_vmin if vmin is None else float(vmin)
    resolved_vmax = auto_vmax if vmax is None else float(vmax)
    if stretch == "log" and resolved_vmin <= 0.0:
        raise ValueError("Log display requires vmin > 0.")
    if resolved_vmax <= resolved_vmin:
        raise ValueError("Display limits must satisfy vmax > vmin.")
    return resolved_vmin, resolved_vmax


def _imshow_psf_stretched(
    ax: plt.Axes,
    display_psf: np.ndarray,
    *,
    stretch: str,
    extent_as: np.ndarray,
) -> Any:
    """Render a normalized PSF with a compact selectable nonlinear stretch."""

    vmin, vmax = _resolve_display_limits(display_psf, stretch=stretch)
    if stretch == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif stretch == "sqrt":
        norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)
    else:
        raise ValueError(f"Unsupported PSF_STRETCH={stretch!r}; use 'log' or 'sqrt'.")
    return ax.imshow(
        display_psf,
        origin="lower",
        cmap=PSF_CMAP_NAME,
        norm=norm,
        extent=extent_as,
    )


def _opd_display_limits_nm(
    dp_payload: Mapping[str, Any],
    support: np.ndarray,
) -> tuple[float, float]:
    """Return OPD limits using the portrait diagnostic's shared-range rule."""

    supported = np.asarray(support, dtype=bool)
    arrays_nm = [
        np.asarray(dp_payload["dp_mask_opd_m"], dtype=float) * 1e9,
        np.asarray(dp_payload["grating_opd_m"], dtype=float) * 1e9,
        np.asarray(dp_payload["combined_opd_m"], dtype=float) * 1e9,
    ]
    values = np.concatenate([arr[supported] for arr in arrays_nm])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1.0e-9
    return vmin, vmax


def _extract_m1_support(binder: SheraBinder) -> np.ndarray:
    """Return the actual primary pupil support used by the built optics."""

    try:
        transmission = binder.telescope.optics.m1_aperture.transmission
    except AttributeError:
        transmission = binder.telescope.optics.aperture.transmission
    return np.asarray(transmission, dtype=float) > 0.0


def _apply_axis_typography(ax: plt.Axes) -> None:
    """Apply report-tunable typography to one Matplotlib axis."""

    ax.title.set_fontfamily(FONT_FAMILY)
    ax.title.set_fontsize(TITLE_FONT_SIZE)
    ax.xaxis.label.set_fontfamily(FONT_FAMILY)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontfamily(FONT_FAMILY)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily(FONT_FAMILY)


def _apply_colorbar_typography(cbar: Any) -> None:
    """Apply report-tunable typography to one Matplotlib colorbar."""

    cbar.ax.yaxis.label.set_fontfamily(FONT_FAMILY)
    cbar.ax.yaxis.label.set_fontsize(COLORBAR_LABEL_FONT_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_LABEL_FONT_SIZE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)


def _neutralize_detector_downsample_layers(system_cfg: dict[str, Any]) -> None:
    """Keep the optics oversampled PSF grid in the rendered inspection image."""

    detector_cfg = system_cfg.get("detector", {})
    if not isinstance(detector_cfg, dict):
        return
    layers = detector_cfg.get("layers", [])
    if not isinstance(layers, list):
        return
    for layer_cfg in layers:
        if isinstance(layer_cfg, dict) and layer_cfg.get("kind") == "Downsample":
            layer_cfg["kernel_size"] = 1.0


# -------------------------------------------------------------------------
# Grating construction mirrored from generate_target_grating_portraits.py
# -------------------------------------------------------------------------


def _build_phase_flipped_grating_opd(
    *,
    binary_mask: np.ndarray,
    amplitude_opd_m: float,
    frequency: float,
    angle_deg: float,
    phase_flip: bool,
) -> np.ndarray:
    """Return the two-axis sinusoidal grating OPD used by target portraits."""

    n = binary_mask.shape[-1]
    x = np.linspace(-0.5, 0.5, n)
    y = np.linspace(-0.5, 0.5, n)
    xx, yy = np.meshgrid(x, y)
    theta = np.deg2rad(angle_deg)
    x_rot = xx * np.cos(theta) + yy * np.sin(theta)
    y_rot = -xx * np.sin(theta) + yy * np.cos(theta)
    phase = binary_mask * pi if phase_flip else np.zeros_like(binary_mask)
    return amplitude_opd_m * (
        np.sin(2.0 * pi * frequency * x_rot - phase)
        + np.sin(2.0 * pi * frequency * y_rot - phase)
    )


def _build_grating_dp_opd_payload(
    *,
    pupil_npix: int,
    wavelength_m: float,
    dp_enabled: bool,
    grating_phase_amplitude_rad: float,
    grating_enabled: bool,
    grating_frequency: float,
    grating_angle_deg: float,
    grating_phase_flip: bool,
    grating_mask_threshold: float,
    binary_mask: bool,
    legacy_dp_centering: bool,
    dp_path: Path | None,
) -> dict[str, Any]:
    """Return a combined DP plus grating OPD payload in metres.

    This mirrors the current target-grating portrait construction.  When the
    payload is injected into optics with ``dp_design_wavelength_m=None``, the
    model consumes the array directly as an OPD map in metres.
    """

    source_dp_path = Path(dp_path) if dp_path is not None else Path(default_diffractive_pupil_path())
    if dp_enabled:
        dp_raw = np.asarray(np.load(source_dp_path), dtype=float)
        if dp_raw.shape[-2:] != (pupil_npix, pupil_npix):
            dp_raw = np.asarray(scale_array(jnp.asarray(dp_raw), pupil_npix, order=1), dtype=float)
        dp_phase_mask = (dp_raw >= grating_mask_threshold).astype(float) if binary_mask else dp_raw
    else:
        dp_phase_mask = np.zeros((pupil_npix, pupil_npix), dtype=float)

    if legacy_dp_centering:
        dp_mask_opd_m = np.asarray(dlu.phase2opd(dp_phase_mask * pi, wavelength_m), dtype=float) - (
            wavelength_m / 4.0
        )
    else:
        dp_mask_opd_m = np.asarray(dlu.phase2opd(dp_phase_mask * pi, wavelength_m), dtype=float)

    amplitude_opd_m = float(np.asarray(dlu.phase2opd(grating_phase_amplitude_rad, wavelength_m)))
    if dp_enabled and grating_enabled:
        grating_opd_m = _build_phase_flipped_grating_opd(
            binary_mask=dp_phase_mask,
            amplitude_opd_m=amplitude_opd_m,
            frequency=grating_frequency,
            angle_deg=grating_angle_deg,
            phase_flip=grating_phase_flip,
        )
    else:
        grating_opd_m = np.zeros_like(dp_mask_opd_m)

    return {
        "source_dp_path": source_dp_path,
        "dp_phase_mask": dp_phase_mask,
        "dp_mask_opd_m": dp_mask_opd_m,
        "grating_opd_m": grating_opd_m,
        "combined_opd_m": dp_mask_opd_m + grating_opd_m,
        "amplitude_opd_m": amplitude_opd_m,
    }


# -------------------------------------------------------------------------
# Model construction
# -------------------------------------------------------------------------


def _resolve_base_system_config(system_preset: str) -> dict[str, Any]:
    """Resolve a system preset through the public config machinery."""

    user_cfg = load_user_config(
        config_path=None,
        system_preset=system_preset,
        experiment_preset=None,
    )
    resolved_cfg = resolve_config(user_cfg)
    if "system" not in resolved_cfg:
        raise ValueError("Resolved config does not include a 'system' block.")
    return resolved_cfg["system"]


def build_figure_system() -> tuple[dict[str, Any], ParameterStore, SheraBinder, dict[str, Any]]:
    """Build the centered Alpha Cen A single-star system and DP OPD payload."""

    if int(PSF_OVERSAMPLE) <= 0:
        raise ValueError("PSF_OVERSAMPLE must be a positive integer.")

    base_system_cfg = _resolve_base_system_config(SYSTEM_PRESET)
    optics_cfg = base_system_cfg.get("optics", {})
    if not isinstance(optics_cfg, dict):
        raise ValueError("Expected system.optics to be a mapping.")

    optics_cfg["pupil_npix"] = int(PUPIL_NPIX)
    pupil_npix = int(optics_cfg["pupil_npix"])
    wavelength_m = float(
        optics_cfg.get("dp_design_wavelength_m")
        or optics_cfg.get("wavelength_m")
        or 550e-9
    )
    dp_path_cfg = optics_cfg.get("diffractive_pupil_path", optics_cfg.get("dp_path"))
    dp_payload = _build_grating_dp_opd_payload(
        pupil_npix=pupil_npix,
        wavelength_m=wavelength_m,
        dp_enabled=DP_ENABLED,
        grating_phase_amplitude_rad=GRATING_PHASE_AMPLITUDE_RAD,
        grating_enabled=GRATING_ENABLED,
        grating_frequency=GRATING_FREQUENCY,
        grating_angle_deg=GRATING_ANGLE_DEG,
        grating_phase_flip=GRATING_PHASE_FLIP,
        grating_mask_threshold=GRATING_MASK_THRESHOLD,
        binary_mask=BINARY_MASK,
        legacy_dp_centering=LEGACY_DP_CENTERING,
        dp_path=Path(dp_path_cfg) if isinstance(dp_path_cfg, str) else None,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_opd_path = OUTPUT_DIR / MODEL_OPD_NAME
    np.save(model_opd_path, np.asarray(dp_payload["combined_opd_m"], dtype=float))
    dp_payload["model_opd_path"] = model_opd_path

    system_cfg = prepare_alpha_cen_a_single_star_system_config(
        base_system_cfg,
        exposure_time_s=EXPOSURE_TIME_S,
        n_lambda=N_LAMBDA,
    )
    system_cfg = deepcopy(system_cfg)
    source_cfg = system_cfg["source"]
    source_cfg["x_position_as"] = 0.0
    source_cfg["y_position_as"] = 0.0

    optics_cfg = system_cfg["optics"]
    optics_cfg["psf_npix"] = int(PSF_NPIX)
    optics_cfg["oversample"] = int(PSF_OVERSAMPLE)
    optics_cfg["pupil_npix"] = int(PUPIL_NPIX)
    optics_cfg["diffractive_pupil_path"] = str(model_opd_path)
    optics_cfg["dp_path"] = str(model_opd_path)
    optics_cfg["dp_design_wavelength_m"] = None
    _neutralize_detector_downsample_layers(system_cfg)

    forward_spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(forward_spec)
    store = store.refresh_derived(forward_spec)
    binder = SheraBinder(system_cfg, forward_spec, store)
    return system_cfg, store, binder, dp_payload


def render_noiseless_psf(binder: SheraBinder, store: ParameterStore) -> np.ndarray:
    """Render the noiseless detector-level model image from the binder."""

    return np.asarray(binder.model(binder.strip_structural(store)), dtype=float)


# -------------------------------------------------------------------------
# Figure generation
# -------------------------------------------------------------------------


def build_dp_opd_plot(
    *,
    dp_payload: Mapping[str, Any],
    support: np.ndarray,
    pupil_diameter_m: float,
) -> tuple[plt.Figure, plt.Axes]:
    """Build the physical-coordinate SHERA diffractive-pupil OPD figure."""

    cmaps = get_default_cmaps(bad_color="0.5", bad_alpha=1.0, register=False)
    cmap = cmaps[OPD_CMAP_NAME]
    extent_cm = image_extent_from_diameter(pupil_diameter_m)
    combined_opd_nm = np.asarray(dp_payload["combined_opd_m"], dtype=float) * 1e9
    display_opd_nm = np.where(np.asarray(support, dtype=bool), combined_opd_nm, np.nan)
    vmin, vmax = _opd_display_limits_nm(dp_payload, support)

    fig, ax = plt.subplots(figsize=(6.4, 5.8))
    im = ax.imshow(
        display_opd_nm,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent_cm,
    )
    ax.set_title("SHERA Diffractive Pupil OPD")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_aspect("equal")
    _apply_axis_typography(ax)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("OPD [nm]")
    _apply_colorbar_typography(cbar)
    fig.tight_layout()
    return fig, ax


def build_single_star_psf_plot(
    *,
    psf: np.ndarray,
    plate_scale_as_per_pix: float,
    oversample: int,
    show_diameter_circle: bool,
    stretch: str = PSF_STRETCH,
) -> tuple[plt.Figure, plt.Axes]:
    """Build the centered Alpha Cen A single-star PSF figure."""

    max_value = float(np.nanmax(psf))
    if not np.isfinite(max_value) or max_value <= 0.0:
        raise ValueError("PSF maximum must be positive and finite.")
    display_psf = np.asarray(psf, dtype=float) / max_value
    extent_as = psf_extent_as(display_psf, plate_scale_as_per_pix, oversample)

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    im = _imshow_psf_stretched(
        ax,
        display_psf,
        stretch=stretch,
        extent_as=extent_as,
    )
    if show_diameter_circle:
        radius_as = 0.5 * float(CIRCLE_DIAMETER_AS)
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius_as,
                edgecolor=CIRCLE_COLOR,
                facecolor="none",
                linestyle=CIRCLE_LINESTYLE,
                linewidth=CIRCLE_LINEWIDTH,
            )
        )

    ax.set_title("SHERA PSF")
    ax.set_xlabel("X (arcsec)")
    ax.set_ylabel("Y (arcsec)")
    ax.set_aspect("equal")
    _apply_axis_typography(ax)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label(f"Normalized Intensity ({stretch})")
    _apply_colorbar_typography(cbar)
    fig.tight_layout()
    return fig, ax


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save one figure as PNG, and optionally PDF next to it."""

    fig.savefig(path, dpi=FIGURE_DPI)
    if SAVE_PDF:
        fig.savefig(path.with_suffix(".pdf"), dpi=FIGURE_DPI)


def _print_diagnostics(
    *,
    system_cfg: Mapping[str, Any],
    store: ParameterStore,
    psf: np.ndarray,
    support: np.ndarray,
    dp_payload: Mapping[str, Any],
    opd_path: Path,
    psf_path: Path,
) -> None:
    """Print concise smoke-run diagnostics for manual figure inspection."""

    optics_cfg = system_cfg["optics"]
    source_cfg = system_cfg["source"]
    pupil_diameter_m = float(optics_cfg["m1_diameter_m"])
    pupil_extent_cm = image_extent_from_diameter(pupil_diameter_m)
    plate_scale_as_per_pix = float(np.asarray(store.get("optics.plate_scale_as_per_pix")))
    oversample = int(np.asarray(store.get("optics.oversample")))
    display_pixel_scale_as_per_pix = plate_scale_as_per_pix / float(oversample)
    psf_extent = psf_extent_as(psf, plate_scale_as_per_pix, oversample)
    combined_opd = np.asarray(dp_payload["combined_opd_m"], dtype=float)
    display_opd = np.where(support, combined_opd, np.nan)
    injected_opd = np.asarray(np.load(Path(str(optics_cfg["dp_path"]))), dtype=float)
    circle_radius_as = 0.5 * float(CIRCLE_DIAMETER_AS)

    print("SHERA DP + PSF figure smoke diagnostics")
    print(f"  system_preset: {SYSTEM_PRESET}")
    print(f"  m1_diameter_m: {pupil_diameter_m:.6f}")
    print(f"  pupil_extent_cm: {pupil_extent_cm.tolist()}")
    print(f"  pupil_npix: {int(optics_cfg['pupil_npix'])}")
    print(f"  combined_opd_shape: {combined_opd.shape}")
    print(f"  outside_pupil_nan_for_display: {bool(np.isnan(display_opd[~support]).all())}")
    print("  opd_colorbar_label: OPD [nm]")
    print(f"  psf_shape: {psf.shape}")
    print(f"  base_psf_npix: {PSF_NPIX}")
    print(f"  resolved_optics_psf_npix: {int(optics_cfg['psf_npix'])}")
    print(f"  optics_oversample: {oversample}")
    print(f"  plate_scale_as_per_pix: {plate_scale_as_per_pix:.12g}")
    print(f"  display_pixel_scale_as_per_pix: {display_pixel_scale_as_per_pix:.12g}")
    print(f"  psf_extent_as: {psf_extent.tolist()}")
    print(
        "  source_center_as: "
        f"({float(source_cfg['x_position_as']):.6g}, {float(source_cfg['y_position_as']):.6g})"
    )
    print("  noiseless: True")
    print(f"  same_opd_in_plot_and_model: {bool(np.array_equal(combined_opd, injected_opd))}")
    print(f"  circle_enabled: {SHOW_DIAMETER_CIRCLE}")
    print(f"  circle_radius_as: {circle_radius_as:.6g}")
    print(
        "  grating: "
        f"enabled={GRATING_ENABLED}, phase_amplitude_rad={GRATING_PHASE_AMPLITUDE_RAD:.12g}, "
        f"amplitude_opd_nm={float(dp_payload['amplitude_opd_m']) * 1e9:.12g}, "
        f"frequency={GRATING_FREQUENCY:.12g}, angle_deg={GRATING_ANGLE_DEG:.12g}, "
        f"phase_flip={GRATING_PHASE_FLIP}, mask_threshold={GRATING_MASK_THRESHOLD}, "
        f"binary_mask={BINARY_MASK}, legacy_dp_centering={LEGACY_DP_CENTERING}"
    )
    print(f"  opd_path: {opd_path}")
    print(f"  psf_path: {psf_path}")
    print(f"  model_opd_path: {dp_payload['model_opd_path']}")


def main() -> None:
    """Render and save the two standalone figure files."""

    apply_plot_defaults(font_family=FONT_FAMILY, figure_dpi=120)
    get_default_cmaps(bad_color="0.5", bad_alpha=1.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system_cfg, store, binder, dp_payload = build_figure_system()
    support = _extract_m1_support(binder)
    psf = render_noiseless_psf(binder, store)

    pupil_diameter_m = float(system_cfg["optics"]["m1_diameter_m"])
    plate_scale_as_per_pix = float(np.asarray(store.get("optics.plate_scale_as_per_pix")))
    oversample = int(np.asarray(store.get("optics.oversample")))

    opd_fig, _ = build_dp_opd_plot(
        dp_payload=dp_payload,
        support=support,
        pupil_diameter_m=pupil_diameter_m,
    )
    psf_fig, _ = build_single_star_psf_plot(
        psf=psf,
        plate_scale_as_per_pix=plate_scale_as_per_pix,
        oversample=oversample,
        show_diameter_circle=SHOW_DIAMETER_CIRCLE,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        no_circle_fig, no_circle_ax = build_single_star_psf_plot(
            psf=psf,
            plate_scale_as_per_pix=plate_scale_as_per_pix,
            oversample=oversample,
            show_diameter_circle=False,
            stretch=PSF_STRETCH,
        )
    if no_circle_ax.patches:
        raise RuntimeError("Circle toggle smoke failed: patches remain when disabled.")
    plt.close(no_circle_fig)

    # Smoke the alternate supported stretch without saving an extra file.
    alternate_stretch = "sqrt" if PSF_STRETCH == "log" else "log"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        test_fig, _ = build_single_star_psf_plot(
            psf=psf,
            plate_scale_as_per_pix=plate_scale_as_per_pix,
            oversample=oversample,
            show_diameter_circle=False,
            stretch=alternate_stretch,
        )
    plt.close(test_fig)

    opd_path = OUTPUT_DIR / OPD_FIGURE_NAME
    psf_path = OUTPUT_DIR / PSF_FIGURE_NAME
    _save_figure(opd_fig, opd_path)
    _save_figure(psf_fig, psf_path)
    plt.close(opd_fig)
    plt.close(psf_fig)

    _print_diagnostics(
        system_cfg=system_cfg,
        store=store,
        psf=psf,
        support=support,
        dp_payload=dp_payload,
        opd_path=opd_path,
        psf_path=psf_path,
    )


if __name__ == "__main__":
    main()
