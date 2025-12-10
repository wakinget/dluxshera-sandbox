"""Notebook-style canonical astrometry demo using the refactored Shera stack.

Run the script directly for a narrated walk-through of the refactor workflow:

```
python -m Examples.scripts.run_canonical_astrometry_demo [--fast] [--save-plots] [--add-noise]
```

- ``--fast``: reduce optimisation steps for a quick smoke run.
- ``--save-plots``: write PSF and loss-curve figures to ``Results/CanonicalAstrometryDemo``.
- ``--add-noise``: inject light Gaussian noise into the synthetic PSF before inference.

The tutorial covers building a three-plane Shera model, generating synthetic
binary-star PSFs, and recovering astrometry and wavefront parameters with a
binder/SystemGraph loss under tight priors. Read the file top-to-bottom to see
each step inline, from configuring the optics to running gradient descent.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Tuple

import jax.numpy as jnp
import numpy as np

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.optimization import make_binder_image_nll_fn, run_simple_gd
from dluxshera.optics.config import SheraThreePlaneConfig
from dluxshera.params.packing import unpack_params as store_unpack_params
from dluxshera.params.spec import (
    ParamKey,
    ParamSpec,
    build_forward_model_spec_from_config,
    build_inference_spec_basic,
)
from dluxshera.params.store import ParameterStore
from dluxshera.params.transforms import resolve_derived
import dluxshera.params.shera_threeplane_transforms  # Registers default derived transforms

# Default output directory (gitignored via Results/)
DEFAULT_RESULTS_DIR = Path("Results/CanonicalAstrometryDemo")


@dataclass
class DemoData:
    """Container for key demo artifacts returned to callers and tests."""

    cfg: SheraThreePlaneConfig
    forward_spec: ParamSpec
    inference_spec: ParamSpec
    truth_store: ParameterStore
    binder: SheraThreePlaneBinder
    truth_psf: np.ndarray
    variant_psf: np.ndarray


# ---------------------------------------------------------------------------
# Utility helpers used for readable output
# ---------------------------------------------------------------------------


def format_parameter_summary(
    keys: Iterable[ParamKey],
    truth_store: ParameterStore,
    init_store: ParameterStore,
    final_store: ParameterStore,
) -> str:
    lines = ["Parameter summary (truth → init → recovered):"]
    for key in keys:
        truth_val = truth_store.get(key)
        init_val = init_store.get(key)
        final_val = final_store.get(key)
        lines.append(f"  - {key}: {truth_val} → {init_val} → {final_val}")
    return "\n".join(lines)


def main(fast: bool = False, save_plots: bool = False, add_noise: bool = False) -> DemoData:
    """Execute the canonical astrometry demo from data generation to recovery.

    The steps mirror a tutorial: build configs/specs, instantiate truth
    parameters, synthesize PSFs, define inference knobs and priors, perturb the
    initial guess, and optimise with a binder-based loss. Optional switches
    allow quicker runs (``fast``), saving plots (``save_plots``), and adding
    synthetic noise (``add_noise``).
    """

    rng = np.random.default_rng(0)

    # ------------------------------------------------------------------
    # 1. Build Shera config and forward/inference specs
    # ------------------------------------------------------------------
    print("Step 1: Building SheraThreePlaneConfig and ParamSpecs...")
    cfg = SheraThreePlaneConfig(
        design_name="shera_testbed",
        pupil_npix=128,
        psf_npix=128,
        oversample=2,
        primary_noll_indices=(4, 5, 6, 7),
        secondary_noll_indices=(4, 5, 6, 7),
    )
    forward_spec = build_forward_model_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic()

    # ------------------------------------------------------------------
    # 2. Construct truth ParameterStore and resolve derived parameters
    # ------------------------------------------------------------------
    print("Step 2: Constructing truth ParameterStore and derived parameters...")
    forward_store = ParameterStore.from_spec_defaults(forward_spec)
    forward_store = forward_store.replace(
        {
            "binary.separation_as": 0.55,
            "binary.position_angle_deg": 120.0,
            "binary.x_position": 0.03,
            "binary.y_position": -0.02,
            # Photon budget controls
            "binary.spectral_flux_density": 7e9,
            "imaging.exposure_time_s": 8.0,
            "imaging.throughput": 0.85,
        }
    )

    # Compute derived parameters (plate scale, total flux) from the forward store
    plate_scale = resolve_derived("system.plate_scale_as_per_pix", forward_store)
    log_flux = resolve_derived("binary.log_flux_total", forward_store)

    # Build the inference-space truth store and inject astrometric / wavefront values
    truth_store = ParameterStore.from_spec_defaults(inference_spec)
    primary_zernikes = 5.0 * rng.standard_normal(len(cfg.primary_noll_indices))
    secondary_zernikes = np.zeros(len(cfg.secondary_noll_indices))
    truth_store = truth_store.replace(
        {
            "binary.separation_as": forward_store.get("binary.separation_as"),
            "binary.position_angle_deg": forward_store.get("binary.position_angle_deg"),
            "binary.x_position": forward_store.get("binary.x_position"),
            "binary.y_position": forward_store.get("binary.y_position"),
            "binary.log_flux_total": log_flux,
            "binary.contrast": 3.2,
            "system.plate_scale_as_per_pix": plate_scale,
            "primary.zernike_coeffs": primary_zernikes,
            "secondary.zernike_coeffs": secondary_zernikes,
        }
    )

    # ------------------------------------------------------------------
    # 3. Generate synthetic PSFs and show how parameter updates regenerate data
    # ------------------------------------------------------------------
    print("Step 3: Generating synthetic PSFs from the refactor stack...")
    binder = SheraThreePlaneBinder(cfg, inference_spec, truth_store)
    truth_psf = np.array(binder.forward(truth_store))

    if add_noise:
        truth_psf = truth_psf + rng.normal(scale=0.005, size=truth_psf.shape)

    # Demonstrate that changing parameters regenerates a different PSF
    variant_store = truth_store.replace(
        {
            "binary.separation_as": truth_store.get("binary.separation_as") + 0.03,
            "primary.zernike_coeffs": truth_store.get("primary.zernike_coeffs") + 2.0,
        }
    )
    variant_psf = np.array(binder.forward(variant_store))

    # ------------------------------------------------------------------
    # 4. Define inference keys and tight priors around the truth
    # ------------------------------------------------------------------
    print("Step 4: Defining inference keys and tight priors around the truth...")
    infer_keys: Tuple[ParamKey, ...] = (
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.x_position",
        "binary.y_position",
        "binary.log_flux_total",
        "binary.contrast",
        "system.plate_scale_as_per_pix",
        "primary.zernike_coeffs",
    )
    priors: Mapping[ParamKey, object] = {
        "binary.separation_as": 0.01,
        "binary.position_angle_deg": 0.5,
        "binary.x_position": 0.005,
        "binary.y_position": 0.005,
        "binary.log_flux_total": 0.05,
        "binary.contrast": 0.05,
        "system.plate_scale_as_per_pix": 0.002,
        "primary.zernike_coeffs": np.full_like(truth_store.get("primary.zernike_coeffs"), 1.0),
    }

    # ------------------------------------------------------------------
    # 5. Initialise inference store by perturbing the truth using priors
    # ------------------------------------------------------------------
    print("Step 5: Initialising inference store with prior-perturbed truth values...")
    updates = {}
    for key in infer_keys:
        sigma = priors[key]
        true_val = truth_store.get(key)
        if isinstance(true_val, np.ndarray):
            updates[key] = true_val + rng.normal(scale=sigma, size=true_val.shape)
        else:
            updates[key] = float(true_val) + float(rng.normal(scale=sigma, size=()))
    init_store = truth_store.replace(updates)

    # ------------------------------------------------------------------
    # 6. Build binder-based loss with priors and run gradient descent
    # ------------------------------------------------------------------
    print("Step 6: Building loss and running binder/SystemGraph-based gradient descent...")
    var_image = np.ones_like(truth_psf) * 0.01
    sub_spec = inference_spec.subset(infer_keys)
    loss_nll, theta0 = make_binder_image_nll_fn(
        cfg,
        inference_spec,
        init_store,
        infer_keys,
        truth_psf,
        var_image,
        noise_model="gaussian",
        reduce="sum",
        use_system_graph=not fast,
    )

    def loss_with_prior(theta: np.ndarray) -> np.ndarray:
        store_theta = store_unpack_params(sub_spec, theta, init_store)
        penalty = jnp.array(0.0)
        for key in infer_keys:
            sigma = priors[key]
            value = store_theta.get(key)
            truth_val = truth_store.get(key)
            if isinstance(value, np.ndarray):
                penalty += jnp.sum((value - truth_val) ** 2 / (2.0 * sigma**2))
            else:
                penalty += (value - truth_val) ** 2 / (2.0 * sigma**2)
        loss_val = loss_nll(theta)
        return jnp.asarray(loss_val).sum() + jnp.asarray(penalty).sum()

    num_steps = 20 if fast else 120
    theta_final, history = run_simple_gd(
        loss_with_prior,
        theta0,
        learning_rate=1e-2,
        num_steps=num_steps,
    )
    final_store = store_unpack_params(sub_spec, theta_final, init_store)

    # ------------------------------------------------------------------
    # 7. Summarize recovered parameters and optionally save plots
    # ------------------------------------------------------------------
    print("Step 7: Summarising recovered parameters and plotting (optional)...")
    summary = format_parameter_summary(infer_keys, truth_store, init_store, final_store)
    print(summary)
    print("Loss curve (first 5 steps):", np.array(history["loss"])[:5])

    if save_plots:
        output_dir = DEFAULT_RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(truth_psf, cmap="inferno")
        ax[0].set_title("Truth PSF")
        ax[1].imshow(variant_psf, cmap="inferno")
        ax[1].set_title("Variant PSF")
        fig.tight_layout()
        fig.savefig(output_dir / "psf_comparison.png", dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(np.array(history["loss"]))
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (data + prior)")
        fig.tight_layout()
        fig.savefig(output_dir / "loss_curve.png", dpi=200)
        plt.close(fig)

    return DemoData(
        cfg=cfg,
        forward_spec=forward_spec,
        inference_spec=inference_spec,
        truth_store=truth_store,
        binder=binder,
        truth_psf=truth_psf,
        variant_psf=variant_psf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the canonical astrometry demo")
    parser.add_argument("--fast", action="store_true", help="Use fewer optimisation steps")
    parser.add_argument("--save-plots", action="store_true", help="Save demo plots")
    parser.add_argument("--add-noise", action="store_true", help="Add Gaussian noise to the synthetic PSF")
    args = parser.parse_args()

    main(fast=args.fast, save_plots=args.save_plots, add_noise=args.add_noise)
