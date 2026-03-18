"""Render a small contrast sweep to verify source.contrast affects the PSF.

Usage:
    venv/bin/python work/scratch/contrast_montage.py

Outputs:
    Results/contrast_montage.png
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.systems.base import compose_forward_spec, SheraBinder
from dluxshera.params.store import ParameterStore


def main() -> None:
    outdir = Path("Results")
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "contrast_montage.png"

    # Build system from preset (no experiment preset needed).
    user_cfg = load_user_config(
        config_path=None,
        system_preset="SHERA_TESTBED_3P",
        experiment_preset=None,
    )
    cfg = resolve_config(user_cfg)
    system_cfg = cfg["system"]

    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = base_store.refresh_derived(forward_spec)
    print(base_store)
    binder = SheraBinder(system_cfg, forward_spec, base_store)

    contrasts = [1.0, 2.0, 3.0, 5.0]
    images = []
    for c in contrasts:
        delta = binder.strip_structural(
            base_store.replace({"source.contrast": jnp.asarray(c)})
        )
        print(delta)
        img = binder.model(delta)
        images.append(img)

    # Build montage
    ncols = 2
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8, 8))
    axes = axes.flatten()

    vmin = min(float(img.min()) for img in images)
    vmax = max(float(img.max()) for img in images)

    for ax, img, c in zip(axes, images, contrasts):
        im = ax.imshow(img, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(f"contrast={c}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any unused axes
    for ax in axes[len(images) :]:
        ax.axis("off")

    fig.suptitle("SHERA_TESTBED_3P PSF vs source.contrast", fontsize=12)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print(f"Saved contrast montage to {outfile}")


if __name__ == "__main__":
    main()
