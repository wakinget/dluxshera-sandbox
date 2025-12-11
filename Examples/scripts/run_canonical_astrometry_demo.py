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

import jax
import jax.numpy as jnp
import numpy as np

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.inference import run_shera_image_gd_eigen
from dluxshera.inference.prior import PriorSpec
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

# Define the path to the default diffractive pupil file
_PACKAGE_ROOT = Path(dluxshera.__file__).resolve().parent
DEFAULT_DP_PATH = _PACKAGE_ROOT / "data" / "diffractive_pupil.npy"
# Diffractive Pupil design files are saved under src/dluxshera/data/
# Currently, we only have a single design file

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
    # Construct a custom config to demonstrate all available inputs
    cfg = SheraThreePlaneConfig(
        design_name="shera_testbed_custom", # User-defined name
        # Grid Sampling
        pupil_npix=256,
        psf_npix=256,
        oversample=2,
        # Wavelength Sampling
        wavelength_m=550e-9,
        bandwidth_m=110e-9,
        n_lambda=3,
        # System Geometry (three-plane layout)
        m1_diameter_m=0.09,
        m2_diameter_m=0.025,
        m1_focal_length_m=0.35796,
        m2_focal_length_m=-0.041935,
        m1_m2_separation_m=0.320,
        pixel_pitch_m=6.5e-6,
        # Aperture Features
        n_struts=4,
        strut_width_m=0.002,
        strut_rotation_deg=45.0,
        # Diffractive Pupil Mask
        diffractive_pupil_path=str(DEFAULT_DP_PATH),
        dp_design_wavelength_m=550e-9,
        # Zernike Basis (Noll indices)
        primary_noll_indices=(4, 5, 6, 7, 8, 9, 10, 11),
        secondary_noll_indices=(4, 5, 6, 7, 8, 9, 10, 11),
    )
    # Optionally, the user can import pre-defined configs:
    # from dluxshera.optics.config import SHERA_TESTBED_CONFIG, SHERA_FLIGHT_CONFIG
    # cfg = SHERA_TESTBED_CONFIG

    # Construct ParamSpec objects - The spec describes how we intend to use the parameters
    # e.g. `system.plate_scale_as_per_pix` as a derived parameter vs a primitive inference knob
    forward_spec = build_forward_model_spec_from_config(cfg) # One Spec for the forward model
    inference_spec = build_inference_spec_basic() # Another Spec for the inference model

    # ------------------------------------------------------------------
    # 2. Construct truth ParameterStore and resolve derived parameters
    # ------------------------------------------------------------------
    print("Step 2: Constructing truth ParameterStore and derived parameters...")
    # The Spec describes what parameters we intend to use and how we use them
    # The Store actually holds real values for the parameters
    # Here, we initialize a store from default values defined in the Spec
    forward_store = ParameterStore.from_spec_defaults(forward_spec)
    # We may now update any parameters to set up our initial data
    forward_store = forward_store.replace(
        {
            "binary.separation_as": 9.5,
            "binary.position_angle_deg": 80.0,
            "binary.x_position_as": 0.50,
            "binary.y_position_as": -0.25,
            # Photon budget controls
            "binary.spectral_flux_density": 1.7e17,
            "imaging.exposure_time_s": 1.0,
            "imaging.throughput": 0.8,
        }
    )

    # Compute derived parameters (plate scale, total flux) from the forward store
    plate_scale = resolve_derived("system.plate_scale_as_per_pix", forward_store)
    log_flux = resolve_derived("binary.log_flux_total", forward_store)

    forward_store = forward_store.replace(
        {
            "binary.log_flux_total": log_flux,
            "system.plate_scale_as_per_pix": plate_scale,
            "binary.contrast": 3.2,
        }
    )

    # Build the inference-space truth store and inject astrometric / wavefront values
    truth_store = ParameterStore.from_spec_defaults(inference_spec)
    primary_zernikes = 5.0 * rng.standard_normal(len(cfg.primary_noll_indices))
    secondary_zernikes = np.zeros(len(cfg.secondary_noll_indices))
    # ParameterStores are immutable, meaning we must use the .replace() method to provide updates
    truth_store = truth_store.replace(
        {
            "binary.separation_as": forward_store.get("binary.separation_as"),
            "binary.position_angle_deg": forward_store.get("binary.position_angle_deg"),
            "binary.x_position_as": forward_store.get("binary.x_position_as"),
            "binary.y_position_as": forward_store.get("binary.y_position_as"),
            "binary.log_flux_total": forward_store.get("binary.log_flux_total"),
            "binary.contrast": forward_store.get("binary.contrast"),
            "system.plate_scale_as_per_pix": forward_store.get("system.plate_scale_as_per_pix"),
            "primary.zernike_coeffs": primary_zernikes,
            "secondary.zernike_coeffs": secondary_zernikes,
        }
    )

    # ------------------------------------------------------------------
    # 3. Generate synthetic PSFs and show how parameter updates regenerate data
    # ------------------------------------------------------------------
    print("Step 3: Generating synthetic PSFs from the refactor stack...")
    binder = SheraThreePlaneBinder(cfg, forward_spec, forward_store)
    truth_psf = np.array(binder.model())
    # binder.forward(store) merges the given store into the internal base store,
    # then builds a new source based on the store, composes a dl.Telescope object
    # using the new source, the internal optics, and the internal detector, then
    # calls telescope.model() to return the PSF image.
    # I'm worried that the provided store only updates the source object, and
    # doesn't affect either the optics or the detector. `build_alpha_cen_source` is
    # the only thing that uses eff_store within the .forward() method.
    # For instance, if truth_store attempts to update system.plate_scale_as_per_pix,
    # it appears as though this update will fall through the cracks and won't affect
    # the output psf image

    if add_noise:
        truth_psf = truth_psf + rng.normal(scale=0.005, size=truth_psf.shape)

    # Demonstrate that changing parameters regenerates a different PSF
    variant_delta = ParameterStore.from_dict(
        {
            "binary.separation_as": forward_store.get("binary.separation_as") + 1.0,
            "primary.zernike_coeffs": forward_store.get("primary.zernike_coeffs") + 2.0,
        }
    )
    variant_psf = np.array(binder.model(variant_delta))

    # ------------------------------------------------------------------
    # 4. Define inference keys and tight priors around the truth
    # ------------------------------------------------------------------
    print("Step 4: Defining inference keys and tight priors around the truth...")
    infer_keys: Tuple[ParamKey, ...] = (
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.x_position_as",
        "binary.y_position_as",
        "binary.log_flux_total",
        "binary.contrast",
        "system.plate_scale_as_per_pix",
        "primary.zernike_coeffs",
    )
    priors: Mapping[ParamKey, object] = {
        "binary.separation_as": 0.01,
        "binary.position_angle_deg": 0.5,
        "binary.x_position_as": 0.005,
        "binary.y_position_as": 0.005,
        "binary.log_flux_total": 0.05,
        "binary.contrast": 0.05,
        "system.plate_scale_as_per_pix": 0.002,
        "primary.zernike_coeffs": np.full_like(truth_store.get("primary.zernike_coeffs"), 1.0),
    }
    prior_spec = PriorSpec.from_sigmas(truth_store, priors)

    # ------------------------------------------------------------------
    # 5. Initialise inference store by perturbing the truth using priors
    # ------------------------------------------------------------------
    print("Step 5: Initialising inference store with prior-perturbed truth values...")
    jitter_key = jax.random.PRNGKey(rng.integers(0, 1_000_000))
    init_store = prior_spec.sample_near(truth_store, jitter_key, keys=infer_keys)
    updates = {key: init_store.get(key) for key in infer_keys}
    init_forward_store = forward_store.replace(updates)

    # ------------------------------------------------------------------
    # 6. Build binder-based loss with priors and run gradient descent
    # ------------------------------------------------------------------
    print("Step 6: Building loss and running binder/SystemGraph-based gradient descent...")
    var_image = np.ones_like(truth_psf) * 0.01
    sub_spec = forward_spec.subset(infer_keys)
    # 6a) Build a Binder-based data term:
    #     theta → ParameterStore delta → binder.model(delta) → image → Gaussian NLL
    loss_nll, theta0 = make_binder_image_nll_fn(
        cfg,
        forward_spec,
        init_forward_store,
        infer_keys,
        truth_psf,
        var_image,
        noise_model="gaussian",
        reduce="sum",
        use_system_graph=not fast,
    )

    # 6b) Wrap with a simple Gaussian prior penalty to form a MAP loss
    def gaussian_prior_penalty(store_theta: ParameterStore) -> jnp.ndarray:
        return prior_spec.quadratic_penalty(store_theta, center_store=truth_store, keys=infer_keys)

    def loss_with_prior(theta: np.ndarray) -> np.ndarray:
        store_theta = store_unpack_params(sub_spec, theta, init_forward_store)
        data_term = loss_nll(theta)
        prior_term = gaussian_prior_penalty(store_theta)
        return jnp.asarray(data_term).sum() + jnp.asarray(prior_term).sum()

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

    # ------------------------------------------------------------------
    # 8. Eigenmode-based optimisation using EigenThetaMap
    # ------------------------------------------------------------------
    print("Step 8: Computing curvature/FIM and building EigenThetaMap...")
    eigen_steps = 10 if fast else 60
    eigen_results = run_shera_image_gd_eigen(
        cfg=cfg,
        inference_spec=inference_spec,
        base_store=init_store,
        infer_keys=infer_keys,
        data=truth_psf,
        var=var_image,
        noise_model="gaussian",
        num_steps=eigen_steps,
        learning_rate=None,
        truncate=None,
        whiten=True,
        theta_ref=theta0,
        use_system_graph=not fast,
    )

    eigen_store = store_unpack_params(sub_spec, eigen_results.theta_final, init_store)

    print("Step 9: Running gradient descent in eigenmode coordinates...")
    print("  -> initial z-norm:", float(np.linalg.norm(np.array(eigen_results.z_history[0]))))
    print("  -> final z-norm:", float(np.linalg.norm(np.array(eigen_results.z_final))))

    print("Step 10: Comparing pure-θ vs eigenmode recovered parameters...")
    print("Parameter summary (truth → pure-θ → eigenmode):")
    for key in infer_keys:
        truth_val = truth_store.get(key)
        gd_val = final_store.get(key)
        eigen_val = eigen_store.get(key)
        print(f"  - {key}: {truth_val} → {gd_val} → {eigen_val}")

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
        ax.plot(np.array(history["loss"]), label="Pure θ")
        ax.plot(np.array(eigen_results.loss_history), label="Eigenmode θ")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (data + prior)")
        ax.legend()
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
