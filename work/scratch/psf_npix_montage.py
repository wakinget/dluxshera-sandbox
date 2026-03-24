"""Render a small structural sweep of system.optics.psf_npix.

Usage:
    venv/bin/python work/scratch/psf_npix_montage.py

Outputs:
    Results/psf_npix_montage.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import SheraBinder, compose_forward_spec

SYSTEM_PRESET = "SHERA_FLIGHT_3P"
PSF_NPIX_VALUES = [256, 224, 192, 160]


def _render_psf_for_psf_npix(psf_npix: int):
    """Resolve config + rebuild binder for one structural psf_npix value."""

    user_cfg = load_user_config(
        config_path=None,
        system_preset=SYSTEM_PRESET,
        experiment_preset=None,
    )
    system_cfg_user = user_cfg.setdefault("system", {})
    if not isinstance(system_cfg_user, dict):
        raise ValueError("Expected top-level 'system' mapping in user config.")
    optics_cfg = system_cfg_user.setdefault("optics", {})
    if not isinstance(optics_cfg, dict):
        raise ValueError("Expected 'system.optics' mapping in user config.")
    optics_cfg["psf_npix"] = int(psf_npix)

    cfg = resolve_config(user_cfg)
    system_cfg = cfg["system"]

    # psf_npix is structural, so rebuild spec/store/binder per value.
    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = base_store.refresh_derived(forward_spec)
    binder = SheraBinder(system_cfg, forward_spec, base_store)

    return binder.model(binder.strip_structural(base_store))


def main() -> None:
    outdir = Path("Results")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "psf_npix_montage.png"

    images = []
    for psf_npix in PSF_NPIX_VALUES:
        print(f"Rendering PSF for system.optics.psf_npix={psf_npix}")
        images.append(_render_psf_for_psf_npix(psf_npix))

    ncols = 2
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    axes = axes.flatten()

    vmin = min(float(img.min()) for img in images)
    vmax = max(float(img.max()) for img in images)

    for ax, img, psf_npix in zip(axes, images, PSF_NPIX_VALUES):
        im = ax.imshow(img, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"psf_npix={psf_npix}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(images) :]:
        ax.axis("off")

    fig.suptitle(f"{SYSTEM_PRESET} PSF vs system.optics.psf_npix", fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"Saved psf_npix montage to {outfile}")


if __name__ == "__main__":
    main()
