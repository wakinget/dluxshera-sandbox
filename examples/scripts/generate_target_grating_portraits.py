"""Render target portraits with a phase-flipped sinusoidal grating on the DP layer.

Smoke commands
--------------
PYTHONPATH=src python examples/scripts/generate_target_grating_portraits.py \
  --targets ALPHA_CEN \
  --psf-npix 512 \
  --dry-run

PYTHONPATH=src python examples/scripts/generate_target_grating_portraits.py \
  --targets ALPHA_CEN \
  --include-alpha-cen-a-single-star \
  --psf-npix 512 \
  --dry-run

PYTHONPATH=src python examples/scripts/generate_target_grating_portraits.py \
  --targets ALPHA_CEN \
  --psf-npix 512 \
  --run-name alpha_cen_grating_smoke

PYTHONPATH=src python examples/scripts/generate_target_grating_portraits.py \
  --targets all \
  --psf-npix 512 \
  --run-name target_suite_grating_portraits
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import datetime as dt
import json
from math import ceil, pi
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import dLux.utils as dlu
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dluxshera.components.detectors import DetectorSpec, GSENSE2020BSI_SPEC, HWK4123_SPEC
from dluxshera.components.sources import TARGET_SPECS, compute_source_flux_diagnostics
from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.spec import ParamSpec
from dluxshera.params.store import ParameterStore
from dluxshera.plot.obs_subblock import apply_intensity_stretch
from dluxshera.plot.plotting import get_default_cmaps, merge_cbar
from dluxshera.systems.base import SheraBinder, compose_forward_spec
from dluxshera.utils.noise import apply_observation_noise
from dluxshera.utils.utils import default_diffractive_pupil_path, scale_array

SYSTEM_PRESET = "SHERA_FLIGHT_3P"
DEFAULT_RESULTS_DIR = Path("Results/target_grating_portraits")
DEFAULT_STRETCH = "asinh"
DEFAULT_PSF_NPIX = 512
DEFAULT_PUPIL_NPIX = 2048
DEFAULT_EXPOSURE_TIME_S = 0.05
DEFAULT_N_LAMBDA = 11
DISPLAY_PMIN = 1.0
DISPLAY_PMAX = 99.9
TARGET_AUTHORITY_OVERRIDE_KEYS = (
    "contrast",
    "log_flux_total",
    "position_angle_deg",
    "separation_as",
    "vmag_a",
    "vmag_b",
)
ALPHA_CEN_A_SINGLE_KEY = "ALPHA_CEN_A_SINGLE"
ALPHA_CEN_A_SINGLE_DISPLAY_NAME = "Alpha Cen A (single star placeholder)"
ALPHA_CEN_A_PLACEHOLDER_NOTE = (
    "Placeholder calibration-star stand-in centered at (0, 0). "
    "Flux is seeded from Alpha Cen component A; spectral shape falls back to "
    "the current single_star flat spectrum until single-star SED weights are exposed."
)


def _timestamp_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _created_at() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        return np.asarray(value).tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _asinh_stretch(image: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        raise ValueError("vmax must be larger than vmin.")
    scaled = (np.asarray(image, dtype=float) - vmin) / (vmax - vmin)
    clipped = np.clip(scaled, 0.0, 1.0)
    return np.arcsinh(10.0 * clipped) / np.arcsinh(10.0)


def _imshow_stretched(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    stretch: str,
    vmin: float,
    vmax: float,
    extent: np.ndarray | None = None,
) -> Any:
    if stretch == "log":
        norm = LogNorm(vmin=float(vmin), vmax=float(vmax))
        return ax.imshow(image, origin="lower", cmap="inferno", norm=norm, extent=extent)
    if stretch == "asinh":
        stretched = _asinh_stretch(image, vmin=float(vmin), vmax=float(vmax))
    else:
        stretched = apply_intensity_stretch(
            image,
            vmin=float(vmin),
            vmax=float(vmax),
            stretch=stretch,
        )
    return ax.imshow(stretched, origin="lower", cmap="inferno", vmin=0.0, vmax=1.0, extent=extent)


def _parse_targets(raw_targets: str) -> list[str]:
    token = raw_targets.strip()
    if token.lower() == "all":
        return sorted(TARGET_SPECS)
    keys = [part.strip().upper() for part in token.split(",") if part.strip()]
    if not keys:
        raise ValueError("No target keys were provided.")
    unknown = [key for key in keys if key not in TARGET_SPECS]
    if unknown:
        known = ", ".join(sorted(TARGET_SPECS))
        raise ValueError(f"Unknown target key(s): {', '.join(unknown)}. Available: {known}.")
    return keys


def _resolve_image_display_limits(
    image: np.ndarray,
    *,
    stretch: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    finite_values = image[np.isfinite(image)]
    if finite_values.size == 0:
        raise ValueError("No finite values found in rendered image.")
    if stretch == "log":
        positive = finite_values[finite_values > 0.0]
        if positive.size == 0:
            raise ValueError("Log display requires positive values.")
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


def _store_float_or_none(store: ParameterStore, key: str) -> float | None:
    value = store.get(key, default=None)
    if value is None:
        return None
    return float(np.asarray(value))


def _source_summary(store: ParameterStore, *, source_kind: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "source_kind": source_kind,
        "position_angle_deg": _store_float_or_none(store, "source.position_angle_deg"),
        "x_position_as": _store_float_or_none(store, "source.x_position_as"),
        "y_position_as": _store_float_or_none(store, "source.y_position_as"),
        "log_flux_total": float(np.asarray(store.get("source.log_flux_total"))),
        "exposure_time_s": float(np.asarray(store.get("source.exposure_time_s"))),
        "wavelength_m": float(np.asarray(store.get("source.wavelength_m"))),
        "bandwidth_m": float(np.asarray(store.get("source.bandwidth_m"))),
        "n_lambda": int(np.asarray(store.get("source.n_lambda"))),
        "plate_scale_as_per_pix": float(np.asarray(store.get("optics.plate_scale_as_per_pix"))),
        "psf_npix": int(np.asarray(store.get("optics.psf_npix"))),
    }
    if "source.separation_as" in store:
        summary["separation_as"] = _store_float_or_none(store, "source.separation_as")
    if "source.contrast" in store:
        summary["contrast"] = _store_float_or_none(store, "source.contrast")
    return summary


def _format_panel_title(
    *,
    display_name: str,
    entry_key: str,
    summary: Mapping[str, Any],
) -> str:
    source_kind = str(summary.get("source_kind", "binary_target"))
    if source_kind == "single_star":
        return (
            f"{display_name}\n"
            f"single_star, x={float(summary.get('x_position_as') or 0.0):.2f} as, "
            f"y={float(summary.get('y_position_as') or 0.0):.2f} as\n"
            f"logF={float(summary['log_flux_total']):.2f}"
        )
    return (
        f"{display_name} ({entry_key})\n"
        f"sep={float(summary['separation_as']):.1f} as, "
        f"PA={float(summary['position_angle_deg']):.1f} deg\n"
        f"contrast={float(summary['contrast']):.2g}, "
        f"logF={float(summary['log_flux_total']):.2f}"
    )


def _image_extent_as(store: ParameterStore, image: np.ndarray) -> np.ndarray:
    psf_npix = int(np.asarray(store.get("optics.psf_npix", default=image.shape[-1])))
    plate_scale_as_per_pix = float(np.asarray(store.get("optics.plate_scale_as_per_pix")))
    return psf_npix * plate_scale_as_per_pix / 2.0 * np.array([-1.0, 1.0, -1.0, 1.0])


def _build_phase_flipped_grating_opd(
    *,
    binary_mask: np.ndarray,
    amplitude_opd_m: float,
    frequency: float,
    angle_deg: float,
    phase_flip: bool,
) -> np.ndarray:
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
    source_dp_path = Path(dp_path) if dp_path is not None else Path(default_diffractive_pupil_path())
    if dp_enabled:
        dp_raw = np.asarray(np.load(source_dp_path), dtype=float)
        if dp_raw.shape[-2:] != (pupil_npix, pupil_npix):
            dp_raw = np.asarray(scale_array(jnp.asarray(dp_raw), pupil_npix, order=1), dtype=float)
        dp_phase_mask = (dp_raw >= grating_mask_threshold).astype(float) if binary_mask else dp_raw
    else:
        dp_phase_mask = np.zeros((pupil_npix, pupil_npix), dtype=float)

    if legacy_dp_centering:
        # Center the modern normalized-phase mapping around zero OPD:
        # P in {0, 1} -> phase in {0, pi}, then subtract pi/2 equivalent.
        # This yields binary OPD levels at {-lambda/4, +lambda/4}.
        dp_mask_opd_m = np.asarray(dlu.phase2opd(dp_phase_mask * pi, wavelength_m), dtype=float) - (
            wavelength_m / 4.0
        )
    else:
        # Current three-plane optics convention maps normalized phase P in [0, 1]
        # to OPD via phase = P*pi and phase2opd(phase, dp_design_wavelength_m).
        dp_mask_opd_m = np.asarray(dlu.phase2opd(dp_phase_mask * pi, wavelength_m), dtype=float)

    amp_opd_m = float(np.asarray(dlu.phase2opd(grating_phase_amplitude_rad, wavelength_m)))
    if dp_enabled and grating_enabled:
        grating_opd_m = _build_phase_flipped_grating_opd(
            binary_mask=dp_phase_mask,
            amplitude_opd_m=amp_opd_m,
            frequency=grating_frequency,
            angle_deg=grating_angle_deg,
            phase_flip=grating_phase_flip,
        )
    else:
        grating_opd_m = np.zeros_like(dp_mask_opd_m)
    combined_opd_m = dp_mask_opd_m + grating_opd_m
    aperture_support = (dp_phase_mask > 0.0).astype(float)

    return {
        "source_dp_path": source_dp_path,
        "dp_phase_mask": dp_phase_mask,
        "dp_mask_opd_m": dp_mask_opd_m,
        "grating_opd_m": grating_opd_m,
        "combined_opd_m": combined_opd_m,
        "amplitude_opd_m": amp_opd_m,
        "dp_enabled": dp_enabled,
        "grating_enabled": grating_enabled,
        "aperture_support": aperture_support,
        "legacy_dp_centering": legacy_dp_centering,
    }


def _prepare_target_system_cfg(
    base_system_cfg: Mapping[str, Any],
    *,
    target_key: str,
    psf_npix: int | None,
    dp_opd_m: np.ndarray,
) -> dict[str, Any]:
    system_cfg = deepcopy(dict(base_system_cfg))
    source_cfg = system_cfg.setdefault("source", {})
    optics_cfg = system_cfg.setdefault("optics", {})
    if not isinstance(source_cfg, dict) or not isinstance(optics_cfg, dict):
        raise ValueError("Expected system.source and system.optics mappings.")
    source_cfg["kind"] = "binary_target"
    source_cfg["target"] = target_key
    for key in TARGET_AUTHORITY_OVERRIDE_KEYS:
        source_cfg.pop(key, None)
    if psf_npix is not None:
        optics_cfg["psf_npix"] = int(psf_npix)
    optics_cfg["diffractive_pupil_path"] = dp_opd_m
    optics_cfg["dp_path"] = dp_opd_m
    optics_cfg["dp_design_wavelength_m"] = None
    return system_cfg


def _prepare_alpha_cen_a_single_star_system_cfg(
    base_system_cfg: Mapping[str, Any],
    *,
    psf_npix: int | None,
    dp_opd_m: np.ndarray,
) -> dict[str, Any]:
    """Return a centered Alpha Cen A-like ``single_star`` placeholder config.

    Alpha Cen A's component flux is derived through the legacy Alpha Cen
    binary-target path, then exposed as the public single-star
    ``source.log_flux_total`` parameter. The current ``single_star`` builder
    uses a flat wavelength-weight spectrum, so the manifest marks the spectral
    shape as an approximation.
    """

    alpha_system_cfg = _prepare_target_system_cfg(
        base_system_cfg,
        target_key="ALPHA_CEN",
        psf_npix=psf_npix,
        dp_opd_m=dp_opd_m,
    )
    alpha_spec, alpha_store = _build_forward_store(
        alpha_system_cfg,
        normalize_total_flux=False,
    )
    del alpha_spec
    alpha_flux_diag = compute_source_flux_diagnostics("binary_target", alpha_store)
    alpha_a_flux = float(np.asarray(alpha_flux_diag["component_fluxes"]["primary"]))
    if not np.isfinite(alpha_a_flux) or alpha_a_flux <= 0.0:
        raise ValueError("Alpha Cen A component flux must be positive and finite.")

    system_cfg = deepcopy(dict(base_system_cfg))
    source_cfg = system_cfg.setdefault("source", {})
    optics_cfg = system_cfg.setdefault("optics", {})
    if not isinstance(source_cfg, dict) or not isinstance(optics_cfg, dict):
        raise ValueError("Expected system.source and system.optics mappings.")

    wavelength_m = source_cfg.get("wavelength_m", alpha_store.get("source.wavelength_m"))
    bandwidth_m = source_cfg.get("bandwidth_m", alpha_store.get("source.bandwidth_m"))
    n_lambda = source_cfg.get("n_lambda", alpha_store.get("source.n_lambda"))
    exposure_time_s = source_cfg.get("exposure_time_s", alpha_store.get("source.exposure_time_s"))

    system_cfg["source"] = {
        "kind": "single_star",
        "wavelength_m": float(np.asarray(wavelength_m)),
        "bandwidth_m": float(np.asarray(bandwidth_m)),
        "n_lambda": int(np.asarray(n_lambda)),
        "exposure_time_s": float(np.asarray(exposure_time_s)),
        "x_position_as": 0.0,
        "y_position_as": 0.0,
        "position_angle_deg": 0.0,
        "log_flux_total": float(np.log10(alpha_a_flux)),
    }
    if psf_npix is not None:
        optics_cfg["psf_npix"] = int(psf_npix)
    optics_cfg["diffractive_pupil_path"] = dp_opd_m
    optics_cfg["dp_path"] = dp_opd_m
    optics_cfg["dp_design_wavelength_m"] = None
    return system_cfg


def _build_forward_store(
    system_cfg: Mapping[str, Any],
    *,
    normalize_total_flux: bool,
) -> tuple[ParamSpec, ParameterStore]:
    forward_spec = compose_forward_spec({"system": system_cfg})
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    if normalize_total_flux:
        store = store.replace({"source.log_flux_total": 0.0})
    return forward_spec, store


def _render_image(
    system_cfg: Mapping[str, Any],
    forward_spec: ParamSpec,
    store: ParameterStore,
) -> np.ndarray:
    binder = SheraBinder(system_cfg, forward_spec, store)
    return np.asarray(binder.model(binder.strip_structural(store)))


def _resolve_detector_spec_from_system_cfg(system_cfg: Mapping[str, Any]) -> DetectorSpec | None:
    detector_cfg = system_cfg.get("detector", {})
    if not isinstance(detector_cfg, Mapping):
        return None
    model_name = detector_cfg.get("model")
    if model_name == GSENSE2020BSI_SPEC.model_name:
        return GSENSE2020BSI_SPEC
    if model_name == HWK4123_SPEC.model_name:
        return HWK4123_SPEC
    return None


def _extract_primary_aperture_support(
    system_cfg: Mapping[str, Any],
    forward_spec: ParamSpec,
    store: ParameterStore,
) -> np.ndarray | None:
    binder = SheraBinder(system_cfg, forward_spec, store)
    try:
        transmission = np.asarray(binder.m1_aperture.transmission, dtype=float)
    except Exception:
        return None
    return transmission


def _resolve_system_cfg(*, config_path: Path | None, system_preset: str | None) -> dict[str, Any]:
    user_cfg = load_user_config(
        config_path=config_path,
        system_preset=system_preset,
        experiment_preset=None,
    )
    resolved = resolve_config(user_cfg)
    if "system" not in resolved:
        raise ValueError("Resolved config does not include a 'system' block.")
    return resolved["system"]


def _save_dp_diagnostic(path: Path, payload: Mapping[str, Any]) -> None:
    nm = 1e9
    dp_nm = np.asarray(payload["dp_mask_opd_m"]) * nm
    gr_nm = np.asarray(payload["grating_opd_m"]) * nm
    combined_nm = np.asarray(payload["combined_opd_m"]) * nm
    support = np.asarray(payload["aperture_support"]) > 0
    cmap_name = str(payload.get("diagnostic_cmap", "inferno"))
    cmaps = get_default_cmaps(bad_color="0.5", bad_alpha=1.0, register=False)
    cmap = cmaps.get(cmap_name, cmaps["inferno"])
    values = np.concatenate([dp_nm[support], gr_nm[support], combined_nm[support]])
    if values.size > 0 and np.isfinite(values).any():
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = [
        ("DP OPD [nm]", dp_nm),
        ("Phase-Flipped Grating OPD [nm]", gr_nm),
        ("Combined DP + Grating OPD [nm]", combined_nm),
    ]
    for ax, (title, data) in zip(axes, panels):
        masked = np.where(support, data, np.nan)
        im = ax.imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        cbar.set_label("OPD [nm]")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_target_portrait(
    *,
    path: Path,
    image: np.ndarray,
    entry_key: str,
    display_name: str,
    summary: Mapping[str, Any],
    stretch: str,
    vmin: float,
    vmax: float,
    extent_as: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    im = _imshow_stretched(ax, image, stretch=stretch, vmin=vmin, vmax=vmax, extent=extent_as)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_title(_format_panel_title(display_name=display_name, entry_key=entry_key, summary=summary), fontsize=10)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    if stretch == "log":
        cbar.set_label("PSF intensity (log scale)")
    else:
        cbar.set_label(f"{stretch} PSF intensity (normalized)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_montage(
    *,
    path: Path,
    entry_keys: Sequence[str],
    display_names: Sequence[str],
    images: Sequence[np.ndarray],
    summaries: Sequence[Mapping[str, float | int]],
    extents: Sequence[np.ndarray],
    display_limits: Sequence[tuple[float, float]],
    stretch: str,
) -> None:
    ncols = 3
    nrows = ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, key, display_name, image, summary, extent, limits in zip(
        axes, entry_keys, display_names, images, summaries, extents, display_limits
    ):
        panel_vmin, panel_vmax = limits
        _imshow_stretched(
            ax,
            image,
            stretch=stretch,
            vmin=panel_vmin,
            vmax=panel_vmax,
            extent=extent,
        )
        ax.set_xlabel("X [arcsec]", fontsize=8)
        ax.set_ylabel("Y [arcsec]", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_title(_format_panel_title(display_name=display_name, entry_key=key, summary=summary), fontsize=9)
    for ax in axes[len(images):]:
        ax.axis("off")

    fig.suptitle(f"Target Grating Portrait Suite ({stretch} stretch)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render curated target portraits with a phase-flipped sinusoidal DP grating.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--system-preset", default=SYSTEM_PRESET)
    parser.add_argument("--targets", default="all")
    parser.add_argument(
        "--include-alpha-cen-a-single-star",
        action="store_true",
        help=(
            "Append a centered single_star placeholder seeded from Alpha Cen A "
            "component flux. This is a calibration-demo visual smoke, not a "
            "real calibration-star registry entry."
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--psf-npix", type=int, default=DEFAULT_PSF_NPIX)
    parser.add_argument("--pupil-npix", type=int, default=DEFAULT_PUPIL_NPIX)
    parser.add_argument("--exposure-time-s", type=float, default=DEFAULT_EXPOSURE_TIME_S)
    parser.add_argument("--n-lambda", type=int, default=DEFAULT_N_LAMBDA)
    parser.add_argument("--normalize-total-flux", action="store_true")
    parser.add_argument("--stretch", choices=("asinh", "log", "linear", "sqrt"), default=DEFAULT_STRETCH)
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--grating-phase-amplitude-rad", type=float, default=float(pi / 16.0))
    parser.add_argument("--dp-enabled", dest="dp_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grating-enabled", dest="grating_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grating-frequency", type=float, default=128.0)
    parser.add_argument("--grating-angle-deg", type=float, default=45.0)
    parser.add_argument("--grating-phase-flip", dest="grating_phase_flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grating-mask-threshold", type=float, default=0.5)
    parser.add_argument("--binary-mask", dest="binary_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--legacy-dp-centering", dest="legacy_dp_centering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dp-diagnostic-cmap",
        choices=("inferno", "seismic", "coolwarm", "viridis"),
        default="inferno",
        help="Colormap for DP diagnostic panels (NaN-safe variants are applied).",
    )
    parser.add_argument("--observation-noise", dest="observation_noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--noise-photon", dest="noise_photon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise-read", dest="noise_read", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-dark-current", dest="noise_dark_current", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--noise-bright-threshold", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.target_keys = _parse_targets(args.targets)
    except ValueError as exc:
        parser.error(str(exc))
    if args.psf_npix <= 0:
        parser.error("--psf-npix must be positive.")
    if args.pupil_npix <= 0:
        parser.error("--pupil-npix must be positive.")
    if not np.isfinite(args.exposure_time_s) or args.exposure_time_s <= 0.0:
        parser.error("--exposure-time-s must be a positive finite value.")
    if args.n_lambda <= 0:
        parser.error("--n-lambda must be positive.")
    if args.vmin is not None and not np.isfinite(args.vmin):
        parser.error("--vmin must be finite.")
    if args.vmax is not None and not np.isfinite(args.vmax):
        parser.error("--vmax must be finite.")
    if args.vmin is not None and args.vmax is not None and args.vmax <= args.vmin:
        parser.error("--vmax must be larger than --vmin.")
    if args.stretch == "log" and args.vmin is not None and args.vmin <= 0.0:
        parser.error("--vmin must be > 0 for --stretch log.")
    if not np.isfinite(args.noise_bright_threshold) or args.noise_bright_threshold <= 0.0:
        parser.error("--noise-bright-threshold must be a positive finite value.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    timestamp = _timestamp_tag()
    created_at = _created_at()
    run_name = args.run_name or f"target_grating_{timestamp}"
    outdir = Path(args.results_dir).expanduser() / run_name
    targets_dir = outdir / "targets"
    artifacts = {
        "manifest_json": outdir / "manifest.json",
        "dry_run_plan_json": outdir / "dry_run_plan.json",
        "dp_diagnostic_png": outdir / "dp_grating_diagnostic.png",
        "montage_png": outdir / "target_grating_montage.png",
        "targets_dir": targets_dir,
    }

    base_system_cfg = _resolve_system_cfg(config_path=args.config, system_preset=args.system_preset)
    optics_cfg = base_system_cfg.get("optics", {})
    source_cfg = base_system_cfg.get("source", {})
    if not isinstance(optics_cfg, dict):
        raise ValueError("Expected system.optics mapping.")
    if not isinstance(source_cfg, dict):
        raise ValueError("Expected system.source mapping.")
    optics_cfg["pupil_npix"] = int(args.pupil_npix)
    source_cfg["exposure_time_s"] = float(args.exposure_time_s)
    source_cfg["n_lambda"] = int(args.n_lambda)
    pupil_npix = int(optics_cfg["pupil_npix"])
    wavelength_m = float(optics_cfg.get("dp_design_wavelength_m") or optics_cfg.get("wavelength_m") or 550e-9)
    dp_path_cfg = optics_cfg.get("diffractive_pupil_path", optics_cfg.get("dp_path"))
    dp_payload = _build_grating_dp_opd_payload(
        pupil_npix=pupil_npix,
        wavelength_m=wavelength_m,
        dp_enabled=bool(args.dp_enabled),
        grating_phase_amplitude_rad=float(args.grating_phase_amplitude_rad),
        grating_enabled=bool(args.grating_enabled),
        grating_frequency=float(args.grating_frequency),
        grating_angle_deg=float(args.grating_angle_deg),
        grating_phase_flip=bool(args.grating_phase_flip),
        grating_mask_threshold=float(args.grating_mask_threshold),
        binary_mask=bool(args.binary_mask),
        legacy_dp_centering=bool(args.legacy_dp_centering),
        dp_path=Path(dp_path_cfg) if isinstance(dp_path_cfg, str) else None,
    )

    plan = {
        "schema": "target_grating_portraits.v1",
        "created_at": created_at,
        "generator": Path(__file__).as_posix(),
        "config_path": None if args.config is None else str(args.config),
        "system_preset": args.system_preset,
        "targets": list(args.target_keys),
        "single_star_placeholders_requested": (
            [ALPHA_CEN_A_SINGLE_KEY] if args.include_alpha_cen_a_single_star else []
        ),
        "outdir": str(outdir),
        "artifacts": {k: str(v) for k, v in artifacts.items()},
        "grating": {
            "phase_amplitude_rad": float(args.grating_phase_amplitude_rad),
            "amplitude_opd_m": float(dp_payload["amplitude_opd_m"]),
            "amplitude_opd_nm": float(dp_payload["amplitude_opd_m"]) * 1e9,
            "dp_enabled": bool(args.dp_enabled),
            "enabled": bool(args.grating_enabled),
            "frequency_cycles_per_aperture": float(args.grating_frequency),
            "angle_deg": float(args.grating_angle_deg),
            "phase_flip": bool(args.grating_phase_flip),
            "mask_threshold": float(args.grating_mask_threshold),
            "binary_mask": bool(args.binary_mask),
            "legacy_dp_centering": bool(args.legacy_dp_centering),
            "phase2opd_wavelength_m": wavelength_m,
        },
        "optics_overrides": {
            "psf_npix": int(args.psf_npix),
            "pupil_npix": int(args.pupil_npix),
        },
        "source_overrides": {
            "exposure_time_s": float(args.exposure_time_s),
            "n_lambda": int(args.n_lambda),
        },
        "single_star_placeholder_note": ALPHA_CEN_A_PLACEHOLDER_NOTE,
        "observation_noise": {
            "enabled": bool(args.observation_noise),
            "seed": int(args.noise_seed),
            "photon_noise": bool(args.noise_photon),
            "read_noise": bool(args.noise_read),
            "dark_current": bool(args.noise_dark_current),
            "bright_threshold": float(args.noise_bright_threshold),
        },
        "dp_construction": {
            "source_dp_path": str(dp_payload["source_dp_path"]),
            "pupil_array_shape": list(np.asarray(dp_payload["combined_opd_m"]).shape),
            "aperture_mask_used_for_diagnostics": "dp_binary_support",
            "diagnostic_cmap": str(args.dp_diagnostic_cmap),
            "temporary_dp_file_used": False,
        },
    }
    if args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
        artifacts["dry_run_plan_json"].write_text(json.dumps(_jsonable(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Dry-run plan written: {artifacts['dry_run_plan_json']}")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    rendered_images: list[np.ndarray] = []
    target_summaries: list[dict[str, Any]] = []
    extents: list[np.ndarray] = []
    target_artifacts: dict[str, str] = {}
    entry_keys: list[str] = []
    display_names: list[str] = []
    single_star_placeholders: dict[str, dict[str, Any]] = {}
    detector_spec = _resolve_detector_spec_from_system_cfg(base_system_cfg)
    base_noise_key = jr.PRNGKey(int(args.noise_seed))

    render_requests: list[dict[str, Any]] = [
        {"entry_kind": "binary_target", "entry_key": target_key}
        for target_key in args.target_keys
    ]
    if args.include_alpha_cen_a_single_star:
        render_requests.append(
            {
                "entry_kind": "single_star_placeholder",
                "entry_key": ALPHA_CEN_A_SINGLE_KEY,
            }
        )

    for idx, request in enumerate(render_requests):
        entry_kind = str(request["entry_kind"])
        entry_key = str(request["entry_key"])
        if entry_kind == "single_star_placeholder":
            system_cfg = _prepare_alpha_cen_a_single_star_system_cfg(
                base_system_cfg,
                psf_npix=args.psf_npix,
                dp_opd_m=np.asarray(dp_payload["combined_opd_m"], dtype=float),
            )
            display_name = ALPHA_CEN_A_SINGLE_DISPLAY_NAME
            source_kind = "single_star"
        else:
            target_key = entry_key
            system_cfg = _prepare_target_system_cfg(
                base_system_cfg,
                target_key=target_key,
                psf_npix=args.psf_npix,
                dp_opd_m=np.asarray(dp_payload["combined_opd_m"], dtype=float),
            )
            display_name = TARGET_SPECS[target_key].display_name
            source_kind = "binary_target"

        forward_spec, store = _build_forward_store(system_cfg, normalize_total_flux=bool(args.normalize_total_flux))
        if plan["dp_construction"]["aperture_mask_used_for_diagnostics"] == "dp_binary_support":
            m1_support = _extract_primary_aperture_support(system_cfg, forward_spec, store)
            if m1_support is not None:
                dp_payload["aperture_support"] = np.asarray(m1_support, dtype=float)
                plan["dp_construction"]["aperture_mask_used_for_diagnostics"] = "binder.optics.m1_aperture.transmission"
        dp_payload["diagnostic_cmap"] = str(args.dp_diagnostic_cmap)
        image = _render_image(system_cfg, forward_spec, store)
        if args.observation_noise:
            noise_key = jr.fold_in(base_noise_key, int(idx))
            noisy, _ = apply_observation_noise(
                jnp.asarray(image),
                noise_cfg={
                    "enabled": True,
                    "photon_noise": bool(args.noise_photon),
                    "read_noise": bool(args.noise_read),
                    "dark_current": bool(args.noise_dark_current),
                },
                rng_key=noise_key,
                bright_threshold=float(args.noise_bright_threshold),
                detector_spec=detector_spec,
                exposure_time_s=float(args.exposure_time_s),
            )
            image = np.asarray(noisy, dtype=float)
        summary = _source_summary(store, source_kind=source_kind)
        extent = _image_extent_as(store, image)
        rendered_images.append(image)
        entry_summary = {
            "entry_key": entry_key,
            "entry_kind": entry_kind,
            "display_name": display_name,
            "source_kind": source_kind,
            "source": summary,
            "image_shape": list(image.shape),
            "image_extent_as": extent.tolist(),
            "image_sum": float(np.sum(image)),
            "image_max": float(np.max(image)),
        }
        if entry_kind == "binary_target":
            entry_summary["target_key"] = entry_key
            entry_summary["target_display_name"] = display_name
        else:
            entry_summary.update(
                {
                    "photometry_source": "ALPHA_CEN component A placeholder",
                    "photometry_source_target": "ALPHA_CEN",
                    "photometry_source_component": "A",
                    "centered_at_as": [0.0, 0.0],
                    "spectral_shape": {
                        "intended": "Alpha Cen A component spectral shape",
                        "used": "flat single_star spectrum on the configured wavelength grid",
                        "note": "Current single_star contract does not expose custom spectral weights.",
                    },
                    "note": ALPHA_CEN_A_PLACEHOLDER_NOTE,
                }
            )
            single_star_placeholders[entry_key] = entry_summary
        target_summaries.append(entry_summary)
        extents.append(extent)
        entry_keys.append(entry_key)
        display_names.append(display_name)

    _save_dp_diagnostic(artifacts["dp_diagnostic_png"], dp_payload)

    per_target_limits = [
        _resolve_image_display_limits(
            image,
            stretch=args.stretch,
            vmin=args.vmin,
            vmax=args.vmax,
        )
        for image in rendered_images
    ]

    for key, display_name, image, summary, extent, limits in zip(
        entry_keys,
        display_names,
        rendered_images,
        target_summaries,
        extents,
        per_target_limits,
    ):
        panel_vmin, panel_vmax = limits
        portrait_path = targets_dir / f"{key.lower()}_portrait.png"
        _save_target_portrait(
            path=portrait_path,
            image=image,
            entry_key=key,
            display_name=display_name,
            summary=summary["source"],
            stretch=args.stretch,
            vmin=panel_vmin,
            vmax=panel_vmax,
            extent_as=extent,
        )
        summary["display"] = {
            "vmin": float(panel_vmin),
            "vmax": float(panel_vmax),
        }
        target_artifacts[key] = portrait_path.relative_to(outdir).as_posix()

    _save_montage(
        path=artifacts["montage_png"],
        entry_keys=entry_keys,
        display_names=display_names,
        images=rendered_images,
        summaries=[item["source"] for item in target_summaries],
        extents=extents,
        display_limits=per_target_limits,
        stretch=args.stretch,
    )

    manifest = {
        **plan,
        "display": {
            "stretch": args.stretch,
            "requested_vmin": None if args.vmin is None else float(args.vmin),
            "requested_vmax": None if args.vmax is None else float(args.vmax),
            "per_target_limits": [
                {"target_key": key, "vmin": float(vmin), "vmax": float(vmax)}
                for key, (vmin, vmax) in zip(entry_keys, per_target_limits)
            ],
        },
        "artifacts": {
            "dp_diagnostic_png": artifacts["dp_diagnostic_png"].relative_to(outdir).as_posix(),
            "montage_png": artifacts["montage_png"].relative_to(outdir).as_posix(),
            "manifest_json": artifacts["manifest_json"].name,
            "target_portraits": target_artifacts,
        },
        "targets_resolved": target_summaries,
        "single_star_placeholders": single_star_placeholders,
    }
    artifacts["manifest_json"].write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {artifacts['manifest_json']}")


if __name__ == "__main__":
    main()
