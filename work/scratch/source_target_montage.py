"""Render a curated-target source comparison montage through the current stack.

Usage
-----
    python work/scratch/source_target_montage.py
    python work/scratch/source_target_montage.py --normalize-total-flux
    python work/scratch/source_target_montage.py --targets ALPHA_CEN --include-alpha-cen-a-single-star

Output
------
    work/scratch/Results/source_target_montage_<timestamp>.png

Notes
-----
By default this script uses the per-target seeded photometry so panel
brightness reflects the curated target catalogue. Use
``--normalize-total-flux`` to normalize all targets to a common total flux of
1 when you want to emphasize geometry, contrast, and chromatic weighting
instead of absolute brightness.
Display rendering uses a configurable stretch, with ``log`` as the default to
compress dynamic range across the curated target list.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import datetime as dt
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from dluxshera.components.sources import TARGET_SPECS, TargetSpec, compute_source_flux_diagnostics
from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.plot.obs_subblock import apply_intensity_stretch
from dluxshera.plot.plotting import merge_cbar
from dluxshera.systems.base import SheraBinder, compose_forward_spec

SYSTEM_PRESET = "SHERA_FLIGHT_3P"
DEFAULT_STRETCH = "log"
DEFAULT_VMIN = 1e3
DEFAULT_VMAX = 1e9
DEFAULT_NORMALIZED_VMIN = 1e-7
DEFAULT_NORMALIZED_VMAX = 1e-3
DEFAULT_PSF_NPIX: int | None = None
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_PATH = Path(f"work/scratch/Results/source_target_montage_{TIMESTAMP}.png")
TARGET_AUTHORITY_OVERRIDE_KEYS = (
    "contrast",
    "log_flux_total",
    "position_angle_deg",
    "separation_as",
    "vmag_a",
    "vmag_b",
)
ALPHA_CEN_A_SINGLE_KEY = "ALPHA_CEN_A_SINGLE"
ALPHA_CEN_A_SINGLE_SPEC = TargetSpec(
    key=ALPHA_CEN_A_SINGLE_KEY,
    display_name="Alpha Cen A (single star placeholder)",
    component_labels=("A", "single"),
    notes="Placeholder centered single_star source seeded from Alpha Cen A component flux.",
)


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


def _resolve_display_limits(
    *,
    normalize_total_flux: bool,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """Return shared display limits for the requested montage mode."""

    if normalize_total_flux:
        resolved_vmin = DEFAULT_NORMALIZED_VMIN if vmin is None else float(vmin)
        resolved_vmax = DEFAULT_NORMALIZED_VMAX if vmax is None else float(vmax)
    else:
        resolved_vmin = DEFAULT_VMIN if vmin is None else float(vmin)
        resolved_vmax = DEFAULT_VMAX if vmax is None else float(vmax)
    return resolved_vmin, resolved_vmax


def _parse_args() -> argparse.Namespace:
    """Parse CLI options for the source-target montage workflow."""

    parser = argparse.ArgumentParser(
        description="Render a curated target-source montage through the current Shera model stack.",
    )
    parser.add_argument(
        "--system-preset",
        default=SYSTEM_PRESET,
        help="System preset to load before per-target source overrides are applied.",
    )
    parser.add_argument(
        "--targets",
        default="all",
        help="Comma-separated target keys, or 'all'.",
    )
    parser.add_argument(
        "--include-alpha-cen-a-single-star",
        action="store_true",
        help="Append the centered Alpha Cen A-like single_star placeholder.",
    )
    parser.add_argument(
        "--normalize-total-flux",
        action="store_true",
        help="Normalize all targets to total flux = 1 (source.log_flux_total = 0).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to the output montage image.",
    )
    parser.add_argument(
        "--stretch",
        choices=("linear", "sqrt", "log"),
        default=DEFAULT_STRETCH,
        help="Display stretch applied to the rendered PSF montage.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help=(
            "Shared lower display bound used across all panels. "
            "When omitted, the script uses mode-specific defaults."
        ),
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help=(
            "Shared upper display bound used across all panels. "
            "When omitted, the script uses mode-specific defaults."
        ),
    )
    parser.add_argument(
        "--psf-npix",
        type=int,
        default=DEFAULT_PSF_NPIX,
        help=(
            "Optional override for optics.psf_npix used for all targets in this montage run. "
            "If omitted, the preset/system value is used."
        ),
    )
    args = parser.parse_args()
    try:
        args.target_keys = _parse_targets(args.targets)
    except ValueError as exc:
        parser.error(str(exc))
    args.vmin, args.vmax = _resolve_display_limits(
        normalize_total_flux=args.normalize_total_flux,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    if not np.isfinite(args.vmin) or not np.isfinite(args.vmax):
        parser.error("--vmin and --vmax must be finite.")
    if args.vmax <= args.vmin:
        parser.error("--vmax must be larger than --vmin.")
    if args.stretch == "log" and args.vmin <= 0.0:
        parser.error("--vmin must be > 0 for --stretch log.")
    if args.psf_npix is not None and args.psf_npix <= 0:
        parser.error("--psf-npix must be a positive integer.")
    return args


def _build_target_image(
    base_system_cfg: dict,
    *,
    target_key: str,
    normalize_total_flux: bool,
    psf_npix: int | None,
) -> tuple[np.ndarray, ParameterStore]:
    """Build one target-specific PSF image plus its seeded forward store."""

    system_cfg = deepcopy(base_system_cfg)
    source_cfg = system_cfg.setdefault("source", {})
    if not isinstance(source_cfg, dict):
        raise ValueError("Expected 'system.source' to be a mapping.")
    optics_cfg = system_cfg.setdefault("optics", {})
    if not isinstance(optics_cfg, dict):
        raise ValueError("Expected 'system.optics' to be a mapping.")

    source_cfg["kind"] = "binary_target"
    source_cfg["target"] = target_key
    for key in TARGET_AUTHORITY_OVERRIDE_KEYS:
        source_cfg.pop(key, None)
    if psf_npix is not None:
        optics_cfg["psf_npix"] = int(psf_npix)

    forward_spec = compose_forward_spec({"system": system_cfg})
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    if normalize_total_flux:
        store = store.replace({"source.log_flux_total": 0.0})

    binder = SheraBinder(system_cfg, forward_spec, store)
    image = np.asarray(binder.model(binder.strip_structural(store)))
    return image, store


def _build_alpha_cen_a_single_star_image(
    base_system_cfg: dict,
    *,
    normalize_total_flux: bool,
    psf_npix: int | None,
) -> tuple[np.ndarray, ParameterStore]:
    """Build the centered Alpha Cen A-like single-star placeholder image."""

    _, alpha_store = _build_target_image(
        base_system_cfg,
        target_key="ALPHA_CEN",
        normalize_total_flux=False,
        psf_npix=psf_npix,
    )
    alpha_flux_diag = compute_source_flux_diagnostics("binary_target", alpha_store)
    alpha_a_flux = float(np.asarray(alpha_flux_diag["component_fluxes"]["primary"]))
    if not np.isfinite(alpha_a_flux) or alpha_a_flux <= 0.0:
        raise ValueError("Alpha Cen A component flux must be positive and finite.")

    system_cfg = deepcopy(base_system_cfg)
    source_cfg = system_cfg.setdefault("source", {})
    optics_cfg = system_cfg.setdefault("optics", {})
    if not isinstance(source_cfg, dict) or not isinstance(optics_cfg, dict):
        raise ValueError("Expected system.source and system.optics mappings.")
    system_cfg["source"] = {
        "kind": "single_star",
        "wavelength_m": float(np.asarray(alpha_store.get("source.wavelength_m"))),
        "bandwidth_m": float(np.asarray(alpha_store.get("source.bandwidth_m"))),
        "n_lambda": int(np.asarray(alpha_store.get("source.n_lambda"))),
        "exposure_time_s": float(np.asarray(alpha_store.get("source.exposure_time_s"))),
        "x_position_as": 0.0,
        "y_position_as": 0.0,
        "position_angle_deg": 0.0,
        "log_flux_total": float(np.log10(alpha_a_flux)),
    }
    if psf_npix is not None:
        optics_cfg["psf_npix"] = int(psf_npix)

    forward_spec = compose_forward_spec({"system": system_cfg})
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    if normalize_total_flux:
        store = store.replace({"source.log_flux_total": 0.0})
    binder = SheraBinder(system_cfg, forward_spec, store)
    image = np.asarray(binder.model(binder.strip_structural(store)))
    return image, store


def _panel_title(spec: TargetSpec, store: ParameterStore) -> str:
    source_kind = "single_star" if spec.key == ALPHA_CEN_A_SINGLE_KEY else "binary_target"
    if source_kind == "single_star":
        return (
            f"{spec.display_name}\n"
            f"single_star, x={float(store.get('source.x_position_as')):.1f} as, "
            f"y={float(store.get('source.y_position_as')):.1f} as\n"
            f"logF={float(store.get('source.log_flux_total')):.2f}"
        )
    return (
        f"{spec.display_name}\n"
        f"sep={float(store.get('source.separation_as')):.1f} as, "
        f"PA={float(store.get('source.position_angle_deg')):.1f} deg\n"
        f"contrast={float(store.get('source.contrast')):.2f}, "
        f"logF={float(store.get('source.log_flux_total')):.2f}"
    )


def main() -> None:
    args = _parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    user_cfg = load_user_config(
        config_path=None,
        system_preset=args.system_preset,
        experiment_preset=None,
    )
    resolved = resolve_config(user_cfg)
    system_cfg = resolved["system"]

    target_keys = list(args.target_keys)
    images: list[np.ndarray] = []
    stores: list[ParameterStore] = []
    specs = [TARGET_SPECS[key] for key in target_keys]

    for key in target_keys:
        image, store = _build_target_image(
            system_cfg,
            target_key=key,
            normalize_total_flux=args.normalize_total_flux,
            psf_npix=args.psf_npix,
        )
        images.append(image)
        stores.append(store)

    if args.include_alpha_cen_a_single_star:
        image, store = _build_alpha_cen_a_single_star_image(
            system_cfg,
            normalize_total_flux=args.normalize_total_flux,
            psf_npix=args.psf_npix,
        )
        images.append(image)
        stores.append(store)
        specs.append(ALPHA_CEN_A_SINGLE_SPEC)

    n_panels = len(images)
    ncols = 3
    nrows = ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, image, spec, store in zip(axes, images, specs, stores):
        psf_npix = int(store.get("optics.psf_npix", default=image.shape[-1]))
        plate_scale_as_per_pix = float(store.get("optics.plate_scale_as_per_pix"))
        psf_extent_as = (
            psf_npix * plate_scale_as_per_pix / 2.0 * np.array([-1.0, 1.0, -1.0, 1.0])
        )

        if args.stretch == "log":
            # Shared LogNorm across panels supports direct target-to-target comparison.
            norm = LogNorm(vmin=float(args.vmin), vmax=float(args.vmax))
            im = ax.imshow(image, origin="lower", cmap="inferno", norm=norm, extent=psf_extent_as)
        else:
            stretched = apply_intensity_stretch(
                image,
                vmin=float(args.vmin),
                vmax=float(args.vmax),
                stretch=args.stretch,
            )
            im = ax.imshow(
                stretched,
                origin="lower",
                cmap="inferno",
                vmin=0.0,
                vmax=1.0,
                extent=psf_extent_as,
            )
        ax.set_xlabel("X [arcsec]", fontsize=8)
        ax.set_ylabel("Y [arcsec]", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)
        ax.set_title(_panel_title(spec, store), fontsize=9)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        cbar.ax.tick_params(labelsize=7)

    for ax in axes[n_panels:]:
        ax.axis("off")

    flux_mode_note = (
        "all targets normalized to total flux = 1"
        if args.normalize_total_flux
        else "per-target seeded photometry"
    )
    fig.suptitle(
        f"SHERA Target Montage ({args.system_preset}, {flux_mode_note}, stretch={args.stretch})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=220)
    print(f"Saved target montage to {args.output_path}")


if __name__ == "__main__":
    main()
