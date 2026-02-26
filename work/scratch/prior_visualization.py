"""Prior sampling + visualization workflow for SHERA three-plane astrometry.

This script builds a canonical-style SHERA binder around a nominal (truth-like)
parameter store, defines independent priors for inferred parameters, draws many
samples, and produces two diagnostic plots:

1) ``prior_histograms.png``
   Histogram grid for every plotted parameter:
   - scalar inferred parameters (e.g., separation, position, plate scale, etc.)
   - each primary and secondary Zernike coefficient individually
   - derived ``binary.raw_fluxes`` components (A/B)
   - ``binary.log_flux_total`` and unscaled ``binary.total_flux = 10**log_flux_total``

2) ``psf_mosaic.png``
   Mosaic of sampled PSFs (default 3x4) with panel titles showing per-sample NLL
   against noiseless nominal data.

Inputs (CLI)
- config selection (testbed/flight), fast mode toggle, RNG seed
- number of prior samples, number of PSFs in the mosaic
- optional JSON patch to override/merge prior settings
- output directory + save/show plotting toggles

Outputs (written to ``outdir``)
- ``prior_histograms.png`` (if ``--save-plots``)
- ``psf_mosaic.png`` (if ``--save-plots``)
- ``samples.npz`` (all arrays used for plotting + selected mosaic losses/indices)
- ``meta.json`` (run metadata, infer keys, resolved priors, nominal loss)

Notes
- NLL construction follows the canonical recipe style using
  ``make_binder_nll_fn(..., noise_model='gaussian', reduce='sum')``.
- Per request, variance is set as ``data_var = data`` exactly (no variance floor).
- The script is headless-friendly by default; use ``--show-plot`` for interactive
  display at the end of execution.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from dluxshera.inference.optimization import make_binder_nll_fn
from dluxshera.inference.prior import PriorSpec
from dluxshera.params.packing import pack_params
from dluxshera.params.spec import build_inference_spec_basic, make_inference_subspec
from dluxshera.params.store import ParameterStore, strip_structural
from dluxshera.plot.plotting import apply_plot_defaults, choose_subplot_grid, get_default_cmaps
from dluxshera.systems.three_plane import (
    SHERA_FLIGHT_CONFIG,
    SHERA_TESTBED_CONFIG,
    SheraBinder,
    build_forward_spec_from_config,
)

INFER_KEYS = (
    "binary.separation_as",
    "binary.position_angle_deg",
    "binary.x_position_as",
    "binary.y_position_as",
    "binary.log_flux_total",
    "binary.contrast",
    "system.plate_scale_as_per_pix",
    "primary.zernike_coeffs_nm",
    "secondary.zernike_coeffs_nm",
)


def _resolve_outdir(user_outdir: str | None) -> Path:
    """Resolve/create output directory.

    If ``user_outdir`` is provided, it is used as-is. Otherwise we create a
    timestamped directory under ``Results/``.
    """
    if user_outdir is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return Path(f"Results/prior_viz_{timestamp}")
    return Path(user_outdir)


def _deep_update_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``patch`` into ``base`` and return a new mapping.

    This is used for ``--prior-json`` so users can override only selected prior
    entries (e.g., one sigma value) without copying the full defaults.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_jsonable(value: Any) -> Any:
    """Convert arrays/JAX scalars to JSON-serializable Python objects."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return np.asarray(value).tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _default_prior_info(truth_store: ParameterStore) -> dict[str, dict[str, Any]]:
    """Return canonical-style prior mapping centered on ``truth_store``."""
    return {
        "binary.separation_as": {"sigma": 1e-3, "dist": "Normal"},
        "binary.position_angle_deg": {"sigma": 1.67e-2, "dist": "Uniform"},
        "binary.x_position_as": {"sigma": 1e-2, "dist": "Normal"},
        "binary.y_position_as": {"sigma": 1e-2, "dist": "Normal"},
        "binary.log_flux_total": {"sigma": 4.3e-3, "dist": "Normal"},
        "binary.contrast": {"sigma": 6e-3, "dist": "LogNormal"},
        "system.plate_scale_as_per_pix": {"sigma": 4.3e-3, "dist": "LogNormal"},
        "primary.zernike_coeffs_nm": {
            "sigma": np.full_like(np.asarray(truth_store.get("primary.zernike_coeffs_nm")), 2.0),
            "dist": "Normal",
        },
        "secondary.zernike_coeffs_nm": {
            "sigma": np.full_like(np.asarray(truth_store.get("secondary.zernike_coeffs_nm")), 2.0),
            "dist": "Normal",
        },
    }


def _flatten_samples_for_plotting(sample_stores: list[ParameterStore], derived_spec) -> dict[str, np.ndarray]:
    """Flatten sampled stores into a columnar dictionary for histogram plotting.

    Parameters
    ----------
    sample_stores
        List of sampled stores (already anchored near the nominal point).
    derived_spec
        Spec used with ``refresh_derived`` so derived parameters (notably
        ``binary.raw_fluxes``) are present.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from plot-label key to 1D sample array.
    """
    accum: dict[str, list[float]] = {}

    def push(name: str, value: float) -> None:
        accum.setdefault(name, []).append(float(value))

    for store in sample_stores:
        enriched = store.refresh_derived(derived_spec)

        # Scalars directly in infer keys.
        for key in INFER_KEYS:
            value = np.asarray(enriched.get(key))
            if value.ndim == 0:
                push(key, float(value))
            else:
                for i, coeff in enumerate(value):
                    push(f"{key}[{i}]", float(coeff))

        # Derived flux quantities requested in task.
        raw_fluxes = np.asarray(enriched.get("binary.raw_fluxes"))
        push("binary.raw_fluxes[0]", float(raw_fluxes[0]))
        push("binary.raw_fluxes[1]", float(raw_fluxes[1]))

        log_flux_total = float(np.asarray(enriched.get("binary.log_flux_total")))
        push("binary.log_flux_total", log_flux_total)
        push("binary.total_flux", float(10.0 ** log_flux_total))

    return {name: np.asarray(vals) for name, vals in accum.items()}


def _build_truth_centers(truth_store: ParameterStore, derived_spec) -> dict[str, float]:
    """Compute center/reference values for each plotted parameter."""
    centers: dict[str, float] = {}
    truth_enriched = truth_store.refresh_derived(derived_spec)

    for key in INFER_KEYS:
        value = np.asarray(truth_enriched.get(key))
        if value.ndim == 0:
            centers[key] = float(value)
        else:
            for i, coeff in enumerate(value):
                centers[f"{key}[{i}]"] = float(coeff)

    raw_fluxes = np.asarray(truth_enriched.get("binary.raw_fluxes"))
    centers["binary.raw_fluxes[0]"] = float(raw_fluxes[0])
    centers["binary.raw_fluxes[1]"] = float(raw_fluxes[1])
    log_flux_total = float(np.asarray(truth_enriched.get("binary.log_flux_total")))
    centers["binary.log_flux_total"] = log_flux_total
    centers["binary.total_flux"] = float(10.0 ** log_flux_total)

    return centers


def _plot_hist_grid(
    arrays: dict[str, np.ndarray],
    centers: dict[str, float],
    *,
    save_path: Path | None,
    show: bool,
) -> None:
    """Plot a single histogram-grid figure for all sampled parameters."""
    names = sorted(arrays.keys())
    rows, cols = choose_subplot_grid(len(names))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.8 * rows), squeeze=False)
    fwhm_factor = 2.0 * math.sqrt(2.0 * math.log(2.0))

    for idx, name in enumerate(names):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        values = np.asarray(arrays[name], dtype=float)

        ax.hist(values, bins=40, density=True, alpha=0.72, color="tab:blue", edgecolor="k", linewidth=0.25)

        # Optional Gaussian overlay fitted from sample moments.
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        if std > 0:
            x = np.linspace(np.min(values), np.max(values), 300)
            pdf = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2.0 * np.pi))
            ax.plot(x, pdf, color="tab:orange", lw=1.3, label="Gaussian fit")

        if name in centers:
            ax.axvline(centers[name], color="crimson", lw=1.2, linestyle="--", label="Center")

        fwhm = fwhm_factor * std
        ax.set_title(f"{name}\nμ={mean:.3g}, σ={std:.3g}, FWHM={fwhm:.3g}", fontsize=8)

        if r == rows - 1:
            ax.set_xlabel(name, fontsize=8)
        if c == 0:
            ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

        if idx == 0:
            ax.legend(fontsize=7, loc="best")

    # Hide any unused axes.
    for idx in range(len(names), rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250)
    if show:
        return
    plt.close(fig)


def _plot_psf_mosaic(
    binder: SheraBinder,
    sample_stores: list[ParameterStore],
    inference_subspec,
    nll_loss_fn,
    mosaic_count: int,
    *,
    save_path: Path | None,
    show: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a PSF mosaic and return selected indices + corresponding losses."""
    n_sel = min(mosaic_count, len(sample_stores))
    indices = np.arange(n_sel, dtype=int)
    stores = [sample_stores[i] for i in indices]

    rows, cols = choose_subplot_grid(n_sel)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)

    extent_as = (
        binder.cfg.psf_npix
        * float(binder.base_forward_store.get("system.plate_scale_as_per_pix"))
        / 2.0
        * np.array([-1.0, 1.0, -1.0, 1.0])
    )

    psfs: list[np.ndarray] = []
    losses: list[float] = []
    for store in stores:
        store_model = strip_structural(store, structural_keys=binder.structural_store_keys())
        psf = np.asarray(binder.model(store_model))
        theta = pack_params(inference_subspec, store_model)
        loss = float(nll_loss_fn(theta))
        psfs.append(psf)
        losses.append(loss)

    # Use a single global scale for comparability across panels.
    vmax = max(float(np.max(psf)) for psf in psfs) if psfs else 1.0

    for idx, (i_sample, psf, loss) in enumerate(zip(indices, psfs, losses)):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        ax.imshow(psf, origin="lower", extent=extent_as, cmap="inferno", vmin=0.0, vmax=vmax)
        ax.set_title(f"idx={i_sample} | loss={loss:.3e}", fontsize=8)

        if r == rows - 1:
            ax.set_xlabel("arcsec")
        if c == 0:
            ax.set_ylabel("arcsec")
        ax.tick_params(labelsize=7)

    for idx in range(n_sel, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=250)
    if not show:
        plt.close(fig)

    return indices, np.asarray(losses)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prior sampling + histogram/PSF visualization utility.")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--mosaic-count", type=int, default=12)
    parser.add_argument("--fast", action="store_true", help="Reduce wavelength/zernike settings for faster runs.")
    parser.add_argument("--show-plot", action="store_true", help="Keep figures open and call plt.show() at end.")
    parser.add_argument("--save-plots", dest="save_plots", action="store_true")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    parser.set_defaults(save_plots=True)
    parser.add_argument("--config", choices=("testbed", "flight"), default="testbed")
    parser.add_argument("--prior-json", type=str, default=None, help="Optional JSON file merged into default prior_info.")
    return parser.parse_args()


def main() -> None:
    """Entrypoint for prior sampling, diagnostics, and artifact writing."""
    args = _parse_args()

    jax.config.update("jax_enable_x64", True)
    apply_plot_defaults()
    _ = get_default_cmaps()
    plt.rcParams["image.cmap"] = "inferno_nan"

    outdir = _resolve_outdir(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SHERA_TESTBED_CONFIG if args.config == "testbed" else SHERA_FLIGHT_CONFIG
    cfg = cfg.replace(
        primary_noll_indices=tuple(range(4, 12)),
        secondary_noll_indices=tuple(range(4, 12)),
    )
    if args.fast:
        cfg = cfg.replace(
            n_lambda=1,
            primary_noll_indices=tuple(range(4, 9)),
            secondary_noll_indices=tuple(range(4, 9)),
        )

    forward_spec = build_forward_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic(cfg)
    inference_subspec = make_inference_subspec(base_spec=inference_spec, infer_keys=INFER_KEYS, cfg=cfg)

    truth_store = ParameterStore.from_spec_defaults(forward_spec)
    truth_store = truth_store.replace(
        {
            "binary.separation_as": 10.0,
            "binary.position_angle_deg": 90.0,
            "binary.x_position_as": 0.0,
            "binary.y_position_as": 0.0,
            "imaging.exposure_time_s": 1800.0,
        }
    )
    truth_store = truth_store.refresh_derived(forward_spec)

    binder = SheraBinder(cfg, forward_spec, truth_store)

    data = binder.model()
    data_var = data

    nll_loss_fn, _theta0 = make_binder_nll_fn(
        binder=binder,
        infer_keys=INFER_KEYS,
        data=data,
        var=data_var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=truth_store,
    )

    theta_true = pack_params(inference_subspec, truth_store)
    loss_true = float(nll_loss_fn(theta_true))

    prior_info = _default_prior_info(truth_store)
    if args.prior_json is not None:
        with Path(args.prior_json).open("r", encoding="utf-8") as f:
            prior_patch = json.load(f)
        prior_info = _deep_update_dict(prior_info, prior_patch)

    prior_spec = PriorSpec.from_info(truth_store, prior_info)

    rng_key = jr.PRNGKey(args.seed)
    sample_stores: list[ParameterStore] = []
    for _ in range(args.n_samples):
        rng_key, split_key = jr.split(rng_key)
        sampled_delta = prior_spec.sample(rng_key=split_key, keys=INFER_KEYS)
        sample_store = truth_store.replace(sampled_delta.as_dict()).refresh_derived(inference_spec)
        sample_stores.append(sample_store)

    flat_arrays = _flatten_samples_for_plotting(sample_stores, inference_spec)
    centers = _build_truth_centers(truth_store, inference_spec)

    hist_path = outdir / "prior_histograms.png" if args.save_plots else None
    _plot_hist_grid(flat_arrays, centers, save_path=hist_path, show=args.show_plot)

    mosaic_path = outdir / "psf_mosaic.png" if args.save_plots else None
    mosaic_indices, mosaic_losses = _plot_psf_mosaic(
        binder=binder,
        sample_stores=sample_stores,
        inference_subspec=inference_subspec,
        nll_loss_fn=nll_loss_fn,
        mosaic_count=args.mosaic_count,
        save_path=mosaic_path,
        show=args.show_plot,
    )

    np.savez(
        outdir / "samples.npz",
        **{name: arr for name, arr in flat_arrays.items()},
        mosaic_indices=mosaic_indices,
        mosaic_losses=mosaic_losses,
    )

    meta = {
        "timestamp": dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "mosaic_count": args.mosaic_count,
        "config": args.config,
        "fast": bool(args.fast),
        "infer_keys": list(INFER_KEYS),
        "prior_info": _to_jsonable(prior_info),
        "loss_true": loss_true,
        "outdir": str(outdir),
    }
    with (outdir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved outputs to: {outdir}")
    print(f"Nominal loss (theta_true): {loss_true:.6e}")
    print(f"Recorded arrays: {len(flat_arrays)}")

    if args.show_plot:
        plt.show()


if __name__ == "__main__":
    main()
