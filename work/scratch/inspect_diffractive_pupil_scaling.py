"""Quick inspector for diffractive pupil file scaling and OPD interpretation.

Usage
-----
PYTHONPATH=src python work/scratch/inspect_diffractive_pupil_scaling.py
PYTHONPATH=src python work/scratch/inspect_diffractive_pupil_scaling.py --wavelength-nm 550
PYTHONPATH=src python work/scratch/inspect_diffractive_pupil_scaling.py --save-path work/scratch/Results/dp_scaling.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dLux.utils as dlu
import matplotlib.pyplot as plt
import numpy as np

from dluxshera.utils.utils import default_diffractive_pupil_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect default diffractive pupil native scaling and OPD conversions.",
    )
    parser.add_argument(
        "--dp-path",
        type=Path,
        default=None,
        help="Optional path to a diffractive pupil .npy file. Defaults to canonical package path.",
    )
    parser.add_argument(
        "--wavelength-nm",
        type=float,
        default=550.0,
        help="Reference wavelength in nm for phase/OPD conversions.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, no file is written.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive display (useful for remote/headless runs).",
    )
    args = parser.parse_args()
    if not np.isfinite(args.wavelength_nm) or args.wavelength_nm <= 0.0:
        parser.error("--wavelength-nm must be a positive finite value.")
    return args


def _main() -> None:
    args = _parse_args()

    dp_path = args.dp_path if args.dp_path is not None else Path(default_diffractive_pupil_path())
    dp_native = np.asarray(np.load(dp_path), dtype=float)
    wavelength_m = float(args.wavelength_nm) * 1e-9

    # Interpretation A (current three-plane convention):
    #   native P in [0, 1] -> phase = P*pi -> opd = phase2opd(phase, lambda)
    phase_modern_rad = dp_native * np.pi
    opd_modern_m = np.asarray(dlu.phase2opd(phase_modern_rad, wavelength_m), dtype=float)

    # Interpretation B (legacy centered convention used for diagnostic writing):
    #   phase = P*pi/2, then center by -lambda/4.
    phase_legacy_rad = dp_native * (np.pi / 2.0)
    opd_legacy_centered_m = np.asarray(
        dlu.phase2opd(phase_legacy_rad, wavelength_m) - wavelength_m / 4.0,
        dtype=float,
    )

    # If the file was interpreted as mirror surface sag, reflected OPD is 2x sag.
    sag_assumed_m = opd_modern_m
    opd_reflection_m = 2.0 * sag_assumed_m

    print(f"DP path: {dp_path}")
    print(f"Array shape: {dp_native.shape}, dtype={dp_native.dtype}")
    print(
        f"Native range: min={np.nanmin(dp_native):.6g}, "
        f"max={np.nanmax(dp_native):.6g}, mean={np.nanmean(dp_native):.6g}"
    )
    print(f"Reference wavelength: {args.wavelength_nm:.3f} nm")
    print(
        "Modern OPD range [nm]: "
        f"{1e9*np.nanmin(opd_modern_m):.6g} to {1e9*np.nanmax(opd_modern_m):.6g}"
    )
    print(
        "Legacy centered OPD range [nm]: "
        f"{1e9*np.nanmin(opd_legacy_centered_m):.6g} to {1e9*np.nanmax(opd_legacy_centered_m):.6g}"
    )
    print(
        "Reflected OPD (2x surface sag assumption) range [nm]: "
        f"{1e9*np.nanmin(opd_reflection_m):.6g} to {1e9*np.nanmax(opd_reflection_m):.6g}"
    )

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("Native DP values", dp_native, "viridis", None),
        ("Modern phase [rad] (P*pi)", phase_modern_rad, "viridis", None),
        ("Modern OPD [nm]", opd_modern_m * 1e9, "inferno", None),
        ("Legacy phase [rad] (P*pi/2)", phase_legacy_rad, "viridis", None),
        ("Legacy centered OPD [nm]", opd_legacy_centered_m * 1e9, "inferno", None),
        ("Reflected OPD [nm] (2x sag assumption)", opd_reflection_m * 1e9, "inferno", None),
    ]

    for ax, (title, data, cmap, limits) in zip(axes.ravel(), panels):
        if limits is None:
            im = ax.imshow(data, origin="lower", cmap=cmap)
        else:
            im = ax.imshow(data, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Diffractive Pupil Scaling Inspector ({dp_path.name}, lambda={args.wavelength_nm:.1f} nm)",
        fontsize=12,
    )
    fig.tight_layout()

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=200)
        print(f"Wrote figure: {args.save_path}")

    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    _main()
