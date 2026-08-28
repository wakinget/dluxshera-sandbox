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
from matplotlib import font_manager
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter

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

PSF_NPIX = 160
PSF_OVERSAMPLE = 1
PUPIL_NPIX = 2048
EXPOSURE_TIME_S = 0.05
N_LAMBDA = 11

PSF_STRETCH = "sqrt"  # Supported: "log", "sqrt".
PSF_COLORBAR_TICKS = (0.0, 0.1, 0.2, 0.5, 0.8, 1.0)
DISPLAY_PMIN = 1.0
DISPLAY_PMAX = 99.9

SHOW_DIAMETER_CIRCLE = False
CIRCLE_DIAMETER_AS = 10.0
CIRCLE_COLOR = "red"
CIRCLE_LINESTYLE = "--"
CIRCLE_LINEWIDTH = 1.7

DP_ENABLED = True
GRATING_ENABLED = True

# These controls replace only the modeled M1 amplitude transmission; M2
# geometry and all DP/grating OPD values remain on the selected preset path.
SECONDARY_OBSCURATION_ENABLED = True
CUSTOM_OBSCURATION = True
CUSTOM_N_STRUTS = 4
CUSTOM_STRUT_ROTATION_DEG = 0.0
CUSTOM_STRUT_WIDTH_M = 5.0e-3
CUSTOM_CENTRAL_OBSCURATION_DIAMETER_M = 0.055

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
SANS_SERIF_FONT_PREFERENCE = (
    "Helvetica",
    "Arial",
    "Liberation Sans",
    "DejaVu Sans",
)
TITLE_FONT_SIZE = 28
TITLES_BOLD = True
AXIS_LABEL_FONT_SIZE = 28
TICK_LABEL_FONT_SIZE = 22
COLORBAR_LABEL_FONT_SIZE = 22
COLORBAR_TICK_LABEL_FONT_SIZE = 20


# -------------------------------------------------------------------------
# Small plotting / coordinate helpers
# -------------------------------------------------------------------------


def _configure_font_preferences() -> str:
    """Prefer Helvetica/Arial-style fonts for Matplotlib sans-serif text."""

    active_preference = list(SANS_SERIF_FONT_PREFERENCE)
    if TITLES_BOLD:
        bold_capable = [
            family
            for family in active_preference
            if _font_family_has_bold_face(family)
        ]
        if bold_capable:
            preferred_bold_family = bold_capable[0]
            active_preference = [
                preferred_bold_family,
                *[
                    family
                    for family in active_preference
                    if family != preferred_bold_family
                ],
            ]

    plt.rcParams["font.family"] = FONT_FAMILY
    plt.rcParams["font.sans-serif"] = active_preference
    normal_font_path = font_manager.findfont(
        font_manager.FontProperties(family=[FONT_FAMILY], weight="normal"),
        fallback_to_default=True,
    )
    bold_font_path = font_manager.findfont(
        font_manager.FontProperties(family=[FONT_FAMILY], weight="bold"),
        fallback_to_default=True,
    )
    normal_font = font_manager.FontProperties(fname=normal_font_path).get_name()
    bold_font = font_manager.FontProperties(fname=bold_font_path).get_name()
    return f"normal={normal_font}, bold={bold_font}"


def _font_family_has_bold_face(family: str) -> bool:
    """Return whether Matplotlib knows an explicit bold face for a font family."""

    return any(
        font.name == family and font.style == "normal" and int(font.weight) >= 600
        for font in font_manager.fontManager.ttflist
    )


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

    if stretch == "log":
        vmin, vmax = _resolve_display_limits(display_psf, stretch=stretch)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    elif stretch == "sqrt":
        vmin, vmax = 0.0, 1.0
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


def _extract_aperture_transmission(binder: SheraBinder, name: str) -> np.ndarray:
    """Return one aperture transmission from the built optics."""

    try:
        transmission = getattr(binder.telescope.optics, name).transmission
    except AttributeError:
        transmission = getattr(binder, name).transmission
    return np.asarray(transmission, dtype=float)


def _extract_m1_transmission(binder: SheraBinder) -> np.ndarray:
    """Return the actual primary pupil transmission used by the built optics."""

    try:
        return _extract_aperture_transmission(binder, "m1_aperture")
    except AttributeError:
        return np.asarray(binder.telescope.optics.aperture.transmission, dtype=float)


def _build_m1_transmission(
    *,
    npix: int,
    m1_diameter_m: float,
    central_obscuration_diameter_m: float,
    n_struts: int,
    strut_width_m: float,
    strut_rotation_deg: float,
) -> np.ndarray:
    """Return an M1 transmission using the canonical three-plane recipe."""

    pupil_oversample = 2
    m1_diameter_m = float(m1_diameter_m)
    central_obscuration_diameter_m = float(central_obscuration_diameter_m)
    n_struts = int(n_struts)
    strut_width_m = float(strut_width_m)
    strut_rotation_deg = float(strut_rotation_deg)

    coords = dlu.pixel_coords(pupil_oversample * int(npix), m1_diameter_m)
    components = [dlu.circle(coords, m1_diameter_m / 2.0)]

    if central_obscuration_diameter_m > 0.0:
        components.append(
            dlu.circle(coords, central_obscuration_diameter_m / 2.0, invert=True)
        )

    if n_struts > 0 and strut_width_m > 0.0:
        strut_angles = (
            jnp.linspace(0.0, 360.0, n_struts + 1)[:-1] + strut_rotation_deg
        )
        components.append(dlu.spider(coords, strut_width_m, strut_angles))

    return np.asarray(dlu.combine(components, pupil_oversample), dtype=float)


def _validate_m1_obscuration_geometry(
    *,
    m1_diameter_m: float,
    central_obscuration_diameter_m: float,
    n_struts: int,
    strut_width_m: float,
) -> None:
    """Validate the selected experimental M1 obscuration geometry."""

    if int(n_struts) != n_struts or int(n_struts) < 0:
        raise ValueError("CUSTOM_N_STRUTS must be a non-negative integer.")
    if float(strut_width_m) < 0.0:
        raise ValueError("CUSTOM_STRUT_WIDTH_M must be non-negative.")
    if float(central_obscuration_diameter_m) < 0.0:
        raise ValueError("CUSTOM_CENTRAL_OBSCURATION_DIAMETER_M must be non-negative.")
    if float(central_obscuration_diameter_m) > float(m1_diameter_m):
        raise ValueError(
            "CUSTOM_CENTRAL_OBSCURATION_DIAMETER_M must not exceed m1_diameter_m."
        )
    if float(strut_width_m) > float(m1_diameter_m):
        raise ValueError("CUSTOM_STRUT_WIDTH_M must not exceed m1_diameter_m.")


def _resolve_selected_m1_transmission(
    binder: SheraBinder,
    *,
    system_cfg: Mapping[str, Any],
    secondary_obscuration_enabled: bool,
    custom_obscuration: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the M1 transmission selected for both the model and DP plot.

    The selected transmission is installed on the modeled M1 aperture before
    the PSF is rendered. The DP portrait support is derived from the installed
    modeled transmission so the pupil plot and PSF always share one aperture.
    """

    optics_cfg = system_cfg["optics"]
    m1_diameter_m = float(optics_cfg["m1_diameter_m"])
    m2_diameter_m = float(optics_cfg["m2_diameter_m"])
    m1_transmission = _extract_m1_transmission(binder)
    info: dict[str, Any] = {
        "secondary_obscuration_enabled": bool(secondary_obscuration_enabled),
        "custom_obscuration": bool(custom_obscuration),
        "nominal_n_struts": int(optics_cfg["n_struts"]),
        "nominal_strut_rotation_deg": float(optics_cfg["strut_rotation_deg"]),
        "nominal_strut_width_m": float(optics_cfg["strut_width_m"]),
        "nominal_central_obscuration_diameter_m": m2_diameter_m,
        "m1_diameter_m": m1_diameter_m,
        "m2_diameter_m": m2_diameter_m,
        "nominal_m1_transmission_shape": tuple(m1_transmission.shape),
    }
    optics = getattr(getattr(binder, "telescope", None), "optics", None)
    if optics is not None and hasattr(optics, "wf_npixels"):
        info["optics_wf_npixels"] = int(np.asarray(optics.wf_npixels))

    if not secondary_obscuration_enabled:
        info.update(
            {
                "pupil_mode": "clear",
                "selected_n_struts": 0,
                "selected_strut_rotation_deg": 0.0,
                "selected_strut_width_m": 0.0,
                "selected_central_obscuration_diameter_m": 0.0,
            }
        )
        return (
            _build_m1_transmission(
                npix=m1_transmission.shape[-1],
                m1_diameter_m=m1_diameter_m,
                central_obscuration_diameter_m=0.0,
                n_struts=0,
                strut_width_m=0.0,
                strut_rotation_deg=0.0,
            ),
            info,
        )

    if custom_obscuration:
        _validate_m1_obscuration_geometry(
            m1_diameter_m=m1_diameter_m,
            central_obscuration_diameter_m=CUSTOM_CENTRAL_OBSCURATION_DIAMETER_M,
            n_struts=CUSTOM_N_STRUTS,
            strut_width_m=CUSTOM_STRUT_WIDTH_M,
        )
        info.update(
            {
                "pupil_mode": "custom",
                "selected_n_struts": int(CUSTOM_N_STRUTS),
                "selected_strut_rotation_deg": float(CUSTOM_STRUT_ROTATION_DEG),
                "selected_strut_width_m": float(CUSTOM_STRUT_WIDTH_M),
                "selected_central_obscuration_diameter_m": float(
                    CUSTOM_CENTRAL_OBSCURATION_DIAMETER_M
                ),
            }
        )
        return (
            _build_m1_transmission(
                npix=m1_transmission.shape[-1],
                m1_diameter_m=m1_diameter_m,
                central_obscuration_diameter_m=(
                    CUSTOM_CENTRAL_OBSCURATION_DIAMETER_M
                ),
                n_struts=CUSTOM_N_STRUTS,
                strut_width_m=CUSTOM_STRUT_WIDTH_M,
                strut_rotation_deg=CUSTOM_STRUT_ROTATION_DEG,
            ),
            info,
        )

    info.update(
        {
            "pupil_mode": "nominal",
            "selected_n_struts": int(optics_cfg["n_struts"]),
            "selected_strut_rotation_deg": float(optics_cfg["strut_rotation_deg"]),
            "selected_strut_width_m": float(optics_cfg["strut_width_m"]),
            "selected_central_obscuration_diameter_m": m2_diameter_m,
        }
    )
    return m1_transmission, info


def _with_modeled_m1_transmission(
    binder: SheraBinder,
    transmission: np.ndarray,
) -> SheraBinder:
    """Return a binder whose modeled M1 amplitude transmission is replaced."""

    updated = binder.__class__.__new__(binder.__class__)
    updated.cfg = binder.cfg
    updated.forward_spec = binder.forward_spec
    updated.base_forward_store = binder.base_forward_store
    updated.structural_hash = binder.structural_hash
    updated.telescope = binder.telescope.set(
        "optics.m1_aperture.transmission",
        jnp.asarray(transmission),
    )
    return updated


def _resolve_and_apply_m1_transmission(
    binder: SheraBinder,
    *,
    system_cfg: Mapping[str, Any],
    secondary_obscuration_enabled: bool,
    custom_obscuration: bool,
) -> tuple[SheraBinder, np.ndarray, dict[str, Any]]:
    """Install the selected modeled M1 transmission and return its support."""

    selected_transmission, info = _resolve_selected_m1_transmission(
        binder,
        system_cfg=system_cfg,
        secondary_obscuration_enabled=secondary_obscuration_enabled,
        custom_obscuration=custom_obscuration,
    )
    if selected_transmission.shape != _extract_m1_transmission(binder).shape:
        raise ValueError(
            "Selected M1 transmission shape must match nominal M1 transmission "
            f"shape; got {selected_transmission.shape} and "
            f"{_extract_m1_transmission(binder).shape}."
        )

    if info["pupil_mode"] == "nominal":
        modeled_binder = binder
    else:
        modeled_binder = _with_modeled_m1_transmission(binder, selected_transmission)

    modeled_transmission = _extract_m1_transmission(modeled_binder)
    info["final_m1_transmission_shape"] = tuple(modeled_transmission.shape)
    info["final_m1_support_fraction"] = float(np.mean(modeled_transmission > 0.0))
    info["plot_support_from_modeled_m1_transmission"] = True
    return modeled_binder, modeled_transmission > 0.0, info


def _apply_axis_typography(ax: plt.Axes) -> None:
    """Apply report-tunable typography to one Matplotlib axis."""

    ax.title.set_fontfamily(FONT_FAMILY)
    ax.title.set_fontsize(TITLE_FONT_SIZE)
    ax.title.set_fontweight("bold" if TITLES_BOLD else "normal")
    ax.xaxis.label.set_fontfamily(FONT_FAMILY)
    ax.xaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.xaxis.label.set_fontweight("bold" if TITLES_BOLD else "normal")
    ax.yaxis.label.set_fontfamily(FONT_FAMILY)
    ax.yaxis.label.set_fontsize(AXIS_LABEL_FONT_SIZE)
    ax.yaxis.label.set_fontweight("bold" if TITLES_BOLD else "normal")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily(FONT_FAMILY)


def _apply_colorbar_typography(cbar: Any) -> None:
    """Apply report-tunable typography to one Matplotlib colorbar."""

    cbar.ax.yaxis.label.set_fontfamily(FONT_FAMILY)
    cbar.ax.yaxis.label.set_fontsize(COLORBAR_LABEL_FONT_SIZE)
    cbar.ax.yaxis.label.set_fontweight("bold" if TITLES_BOLD else "normal")
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
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Y [cm]")
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
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
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_aspect("equal")
    _apply_axis_typography(ax)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label(f"Normalized Intensity [{stretch}]")
    if stretch == "sqrt":
        cbar.set_ticks(PSF_COLORBAR_TICKS)
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
    support_info: Mapping[str, Any],
    nominal_m1_transmission: np.ndarray,
    final_m1_transmission: np.ndarray,
    nominal_m2_transmission: np.ndarray,
    final_m2_transmission: np.ndarray,
    dp_payload: Mapping[str, Any],
    opd_path: Path,
    psf_path: Path,
    resolved_sans_serif_font: str,
) -> None:
    """Print concise smoke-run diagnostics for manual figure inspection."""

    optics_cfg = system_cfg["optics"]
    source_cfg = system_cfg["source"]
    pupil_diameter_m = float(optics_cfg["m1_diameter_m"])
    pupil_extent_cm = image_extent_from_diameter(pupil_diameter_m)
    plate_scale_as_per_pix = float(
        np.asarray(store.get("optics.plate_scale_as_per_pix"))
    )
    oversample = int(np.asarray(store.get("optics.oversample")))
    display_pixel_scale_as_per_pix = plate_scale_as_per_pix / float(oversample)
    psf_extent = psf_extent_as(psf, plate_scale_as_per_pix, oversample)
    combined_opd = np.asarray(dp_payload["combined_opd_m"], dtype=float)
    display_opd = np.where(support, combined_opd, np.nan)
    injected_opd = np.asarray(np.load(Path(str(optics_cfg["dp_path"]))), dtype=float)
    circle_radius_as = 0.5 * float(CIRCLE_DIAMETER_AS)
    display_psf = np.asarray(psf, dtype=float) / float(np.nanmax(psf))
    if PSF_STRETCH == "sqrt":
        psf_display_vmin = 0.0
        psf_display_vmax = 1.0
        psf_colorbar_ticks = list(PSF_COLORBAR_TICKS)
    else:
        psf_display_vmin, psf_display_vmax = _resolve_display_limits(
            display_psf,
            stretch=PSF_STRETCH,
        )
        psf_colorbar_ticks = None

    print("SHERA DP + PSF figure smoke diagnostics")
    print(f"  system_preset: {SYSTEM_PRESET}")
    print(f"  resolved_sans_serif_font: {resolved_sans_serif_font}")
    print(f"  m1_diameter_m: {pupil_diameter_m:.6f}")
    print(f"  m2_diameter_m: {float(optics_cfg['m2_diameter_m']):.6f}")
    print(f"  pupil_extent_cm: {pupil_extent_cm.tolist()}")
    print(f"  pupil_npix: {int(optics_cfg['pupil_npix'])}")
    print(f"  combined_opd_shape: {combined_opd.shape}")
    print(
        "  secondary_obscuration_enabled: "
        f"{bool(support_info['secondary_obscuration_enabled'])}"
    )
    print(f"  custom_obscuration: {bool(support_info['custom_obscuration'])}")
    print(f"  pupil_mode: {support_info['pupil_mode']}")
    print(f"  nominal_n_struts: {support_info['nominal_n_struts']}")
    print(
        "  nominal_strut_rotation_deg: "
        f"{support_info['nominal_strut_rotation_deg']:.6g}"
    )
    print(f"  nominal_strut_width_m: {support_info['nominal_strut_width_m']:.6g}")
    print(
        "  nominal_central_obscuration_diameter_m: "
        f"{support_info['nominal_central_obscuration_diameter_m']:.6g}"
    )
    print(f"  selected_n_struts: {support_info['selected_n_struts']}")
    print(
        "  selected_strut_rotation_deg: "
        f"{support_info['selected_strut_rotation_deg']:.6g}"
    )
    print(f"  selected_strut_width_m: {support_info['selected_strut_width_m']:.6g}")
    print(
        "  selected_central_obscuration_diameter_m: "
        f"{support_info['selected_central_obscuration_diameter_m']:.6g}"
    )
    print(
        "  nominal_m1_transmission_shape: "
        f"{support_info['nominal_m1_transmission_shape']}"
    )
    print(
        "  final_m1_transmission_shape: "
        f"{support_info['final_m1_transmission_shape']}"
    )
    print(f"  m2_transmission_shape: {nominal_m2_transmission.shape}")
    if "optics_wf_npixels" in support_info:
        print(f"  optics_wf_npixels: {support_info['optics_wf_npixels']}")
    print(f"  pupil_plot_support_shape: {support.shape}")
    print(
        "  final_m1_support_fraction: "
        f"{support_info['final_m1_support_fraction']:.6g}"
    )
    print(
        "  plotting_support_from_modeled_m1_transmission: "
        f"{bool(support_info['plot_support_from_modeled_m1_transmission'])}"
    )
    print(
        "  nominal_m1_transmission_retained: "
        f"{bool(np.array_equal(nominal_m1_transmission, final_m1_transmission))}"
    )
    print(
        "  m2_transmission_unchanged: "
        f"{bool(np.array_equal(nominal_m2_transmission, final_m2_transmission))}"
    )
    print(
        "  outside_pupil_nan_for_display: "
        f"{bool(np.isnan(display_opd[~support]).all())}"
    )
    print("  opd_colorbar_label: OPD [nm]")
    print(f"  psf_shape: {psf.shape}")
    print(f"  base_psf_npix: {PSF_NPIX}")
    print(f"  resolved_optics_psf_npix: {int(optics_cfg['psf_npix'])}")
    print(f"  optics_oversample: {oversample}")
    print(f"  plate_scale_as_per_pix: {plate_scale_as_per_pix:.12g}")
    print(f"  display_pixel_scale_as_per_pix: {display_pixel_scale_as_per_pix:.12g}")
    print(f"  psf_extent_as: {psf_extent.tolist()}")
    print(f"  psf_stretch: {PSF_STRETCH}")
    print(f"  psf_display_vmin: {psf_display_vmin:.12g}")
    print(f"  psf_display_vmax: {psf_display_vmax:.12g}")
    print(f"  psf_colorbar_ticks: {psf_colorbar_ticks}")
    print(
        "  source_center_as: "
        f"({float(source_cfg['x_position_as']):.6g}, {float(source_cfg['y_position_as']):.6g})"
    )
    print("  noiseless: True")
    print(
        "  same_opd_in_plot_and_model: "
        f"{bool(np.array_equal(combined_opd, injected_opd))}"
    )
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
    resolved_sans_serif_font = _configure_font_preferences()
    get_default_cmaps(bad_color="0.5", bad_alpha=1.0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system_cfg, store, binder, dp_payload = build_figure_system()
    pupil_diameter_m = float(system_cfg["optics"]["m1_diameter_m"])
    nominal_m1_transmission = _extract_m1_transmission(binder)
    nominal_m2_transmission = _extract_aperture_transmission(binder, "m2_aperture")
    binder, support, support_info = _resolve_and_apply_m1_transmission(
        binder,
        system_cfg=system_cfg,
        secondary_obscuration_enabled=SECONDARY_OBSCURATION_ENABLED,
        custom_obscuration=CUSTOM_OBSCURATION,
    )
    final_m1_transmission = _extract_m1_transmission(binder)
    final_m2_transmission = _extract_aperture_transmission(binder, "m2_aperture")
    if support.shape != np.asarray(dp_payload["combined_opd_m"]).shape:
        raise ValueError(
            "DP plotting support shape must match combined OPD shape; "
            f"got support={support.shape}, opd={np.asarray(dp_payload['combined_opd_m']).shape}."
        )
    if not np.array_equal(support, final_m1_transmission > 0.0):
        raise RuntimeError("DP plotting support must match the modeled M1 support.")
    psf = render_noiseless_psf(binder, store)

    plate_scale_as_per_pix = float(
        np.asarray(store.get("optics.plate_scale_as_per_pix"))
    )
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
        support_info=support_info,
        nominal_m1_transmission=nominal_m1_transmission,
        final_m1_transmission=final_m1_transmission,
        nominal_m2_transmission=nominal_m2_transmission,
        final_m2_transmission=final_m2_transmission,
        dp_payload=dp_payload,
        opd_path=opd_path,
        psf_path=psf_path,
        resolved_sans_serif_font=resolved_sans_serif_font,
    )


if __name__ == "__main__":
    main()
