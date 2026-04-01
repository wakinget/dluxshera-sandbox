"""Render a curated-target source comparison montage through the current stack.

Usage
-----
    python work/scratch/source_target_montage.py
    python work/scratch/source_target_montage.py --fixed-log-flux-total 8.0

Output
------
    work/scratch/Results/source_target_montage.png

Notes
-----
By default this script uses the per-target seeded ``source.log_flux_total`` so
panel brightness reflects the curated target photometry. Use
``--fixed-log-flux-total`` to normalize all targets to a common total flux when
you want to emphasize geometry, contrast, and chromatic weighting instead.
Display rendering uses a configurable stretch, with ``log`` as the default to
compress dynamic range across the curated target list.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dluxshera.components.sources import TARGET_SPECS
from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.plot.obs_subblock import apply_intensity_stretch
from dluxshera.plot.plotting import merge_cbar
from dluxshera.systems.base import SheraBinder, compose_forward_spec

SYSTEM_PRESET = "SHERA_TESTBED_3P"
DEFAULT_FIXED_LOG_FLUX_TOTAL: float | None = None
DEFAULT_STRETCH = "log"
OUTPUT_PATH = Path("work/scratch/Results/source_target_montage.png")
TARGET_AUTHORITY_OVERRIDE_KEYS = (
    "contrast",
    "log_flux_total",
    "position_angle_deg",
    "separation_as",
    "vmag_a",
    "vmag_b",
)


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
        "--fixed-log-flux-total",
        type=float,
        default=DEFAULT_FIXED_LOG_FLUX_TOTAL,
        help=(
            "Optional fixed source.log_flux_total applied to every target. "
            "Omit this flag to use the per-target seeded brightness."
        ),
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
    return parser.parse_args()


def _build_target_image(
    base_system_cfg: dict,
    *,
    target_key: str,
    fixed_log_flux_total: float | None,
) -> tuple[np.ndarray, ParameterStore]:
    """Build one target-specific PSF image plus its seeded forward store."""

    system_cfg = deepcopy(base_system_cfg)
    source_cfg = system_cfg.setdefault("source", {})
    if not isinstance(source_cfg, dict):
        raise ValueError("Expected 'system.source' to be a mapping.")

    source_cfg["kind"] = "binary_target"
    source_cfg["target"] = target_key
    for key in TARGET_AUTHORITY_OVERRIDE_KEYS:
        source_cfg.pop(key, None)

    forward_spec = compose_forward_spec({"system": system_cfg})
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    if fixed_log_flux_total is not None:
        store = store.replace({"source.log_flux_total": float(fixed_log_flux_total)})

    binder = SheraBinder(system_cfg, forward_spec, store)
    image = np.asarray(binder.model(binder.strip_structural(store)))
    return image, store


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

    target_keys = sorted(TARGET_SPECS.keys())
    images: list[np.ndarray] = []
    stores: list[ParameterStore] = []
    specs = [TARGET_SPECS[key] for key in target_keys]

    for key in target_keys:
        image, store = _build_target_image(
            system_cfg,
            target_key=key,
            fixed_log_flux_total=args.fixed_log_flux_total,
        )
        images.append(image)
        stores.append(store)

    n_panels = len(images)
    ncols = 3
    nrows = ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    vmin = min(float(img.min()) for img in images)
    vmax = max(float(img.max()) for img in images)

    for ax, image, spec, store in zip(axes, images, specs, stores):
        stretched = apply_intensity_stretch(
            image,
            vmin=vmin,
            vmax=vmax,
            stretch=args.stretch,
        )
        im = ax.imshow(stretched, origin="lower", cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        title = (
            f"{spec.display_name} ({spec.key})\n"
            f"sep={float(store.get('source.separation_as')):.3f} as  "
            f"contrast={float(store.get('source.contrast')):.3f}  "
            f"logF={float(store.get('source.log_flux_total')):.2f}\n"
            f"{spec.spectral_type_a or '?'} + {spec.spectral_type_b or '?'}"
        )
        ax.set_title(title, fontsize=9)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label(args.stretch, fontsize=7)

    for ax in axes[n_panels:]:
        ax.axis("off")

    fixed_flux_note = (
        f"fixed log_flux_total={args.fixed_log_flux_total:.2f}"
        if args.fixed_log_flux_total is not None
        else "per-target seeded log_flux_total"
    )
    fig.suptitle(
        f"Curated Target Source Montage ({args.system_preset}, {fixed_flux_note}, stretch={args.stretch})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(args.output_path, dpi=220)
    print(f"Saved target montage to {args.output_path}")


if __name__ == "__main__":
    main()
