"""
Canonical astrometry retrieval recipe (Shera three-plane).

This script is the primary, end-to-end onboarding example for the dLuxShera workflow.
It is designed to be read like a Matlab script from top to bottom.
Open this in your editor and run it.

What this recipe demonstrates
- Building/choosing a three-plane Shera configuration and applying small overrides.
- Constructing ParameterSpecs:
    - a forward spec describing the simulation parameters ("forward_spec")
    - an inference spec describing the solved-for parameters ("inference_spec")
- Initializing a ParameterStore (values) and populating derived parameters
  (e.g., plate scale computed from focal lengths + pixel pitch via registered
  transforms).
- Building a SheraThreePlaneBinder to bind parameters to optics/source/detector.
- Generating synthetic data (optionally with noise).
- Defining inference keys + priors, sampling an initial state from priors.
- Defining the loss (typically NLL; MAP variants also available).
- Running a single optimization loop and saving/plotting results using the
  repository’s built-in artifacts + plotting utilities.

Eigenmode re-parameterization (recommended default)
This recipe supports an optional eigenmode parameterization of the inference
variables. When enabled, the optimization runs in an eigen-basis derived from
curvature information (e.g., Fisher Information Matrix), which can improve
conditioning and convergence. You can disable eigenmodes to run in the “raw”
parameter basis for clarity or debugging.

How to use
1) Scan the configuration + “options” block near the top (runtime, noise, eigen).
2) Run the script top-to-bottom.
3) Inspect the printed summary and saved artifacts/plots in the run directory.

Outputs
- A run directory containing saved artifacts (e.g., parameters, metrics, traces,
  and optionally checkpoints), plus summary figures produced by the built-in
  plotting utilities.

Notes
- This recipe is intentionally explicit to avoid any hidden helper layers.
  If you want to adapt the workflow, copy this file and edit the explicit steps.
- For deeper background, see docs on Params/Stores, inference/losses, eigenmodes,
  and optimization artifacts in the repository documentation.
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from dluxshera.inference.optimization import (
    EigenThetaMap,
    fim_theta,
    generate_fim_labels,
    make_binder_nll_fn,
    map_labels_to_keys,
    run_shera_gd,
)
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.signals import build_signals
from dluxshera.params.packing import (
    build_eigen_index_map,
    build_index_map,
    pack_params,
    unpack_params as store_unpack_params,
)
from dluxshera.params.spec import build_inference_spec_basic, make_inference_subspec
from dluxshera.params.store import ParameterStore, strip_structural
from dluxshera.plot.plotting import (
    apply_plot_defaults,
    get_default_cmaps,
    plot_fim,
    plot_parameter_history,
    plot_psf_comparison,
    plot_signals_grid,
)
from dluxshera.plot.printing import print_optimization_summary
from dluxshera.systems.three_plane import (
    SheraThreePlaneConfig,
    SHERA_TESTBED_CONFIG,
    SHERA_FLIGHT_CONFIG,
    SheraThreePlaneBinder,
    build_forward_spec_from_config,
)

# ----------------------------
# User-facing toggles (edit me)
# ----------------------------
JAX_ENABLE_X64 = True
RNG_SEED = 42
FAST_MODE = False
ADD_NOISE = False
SAVE_PLOTS = True

# Telescope Config Selection (9cm testbed vs 22cm flight design)
# Options: None, SHERA_TESTBED_CONFIG / SHERA_FLIGHT_CONFIG
CONFIG = SHERA_TESTBED_CONFIG

# Eigenmode settings
USE_EIGEN = True           # Enables re-parameterization
WHITEN_BASIS = True        # If True, scales each eigenvector by 1/sqrt(lambda)
TRUNCATE_K = None          # int or None; keep top-k eigenmodes when set
TRUNCATE_BY_EIGVAL = None  # float or None; only used when TRUNCATE_K is None

# Inference settings
N_ITER = 60
FAST_ITER = 30
BASE_LR = 0.5

INFER_KEYS = (
    "binary.separation_as",
    "binary.position_angle_deg",
    "binary.x_position_as",
    "binary.y_position_as",
    "binary.log_flux_total",
    "binary.contrast",
    "system.plate_scale_as_per_pix",
    "primary.zernike_coeffs_nm",
    "secondary.zernike_coeffs_nm",  # Optionally comment this one out
)

# Directories
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path(REPO_ROOT / f"Results/canonical_astrometry_recipe_{TIMESTAMP}")

# Plotting defaults
_ = get_default_cmaps()
apply_plot_defaults()
plt.rcParams["image.cmap"] = "inferno_nan"


def main(
    *,
    config: SheraThreePlaneConfig | None = CONFIG,
    fast: bool = FAST_MODE,
    save_plots: bool = SAVE_PLOTS,
    add_noise: bool = ADD_NOISE,
    results_dir: Path | None = None,
    use_eigen: bool = USE_EIGEN,
    whiten_basis: bool = WHITEN_BASIS,
    truncate_k: int | None = TRUNCATE_K,
    truncate_by_eigval: float | None = TRUNCATE_BY_EIGVAL,
) -> None:
    """Run the canonical astrometry recipe."""
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    rng_key = jr.PRNGKey(RNG_SEED)
    results_dir = results_dir or DEFAULT_RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Simulation...")
    print("Creating Config, Spec, Store, and Binder...")
    print("Eigenmode configuration:")
    print(f"  use_eigen={use_eigen}")
    print(f"  whiten_basis={whiten_basis}")
    print(f"  truncate_k={truncate_k}")
    print(f"  truncate_by_eigval={truncate_by_eigval}")

    t0_script = time.time()

    cfg = config or SHERA_TESTBED_CONFIG
    cfg = cfg.replace(
        primary_noll_indices=tuple(range(4, 12)),
        secondary_noll_indices=tuple(range(4, 12)),)
    if fast:
        cfg = cfg.replace(n_lambda=1,
            primary_noll_indices=tuple(range(4, 9)),
            secondary_noll_indices=tuple(range(4, 9)))

    forward_spec = build_forward_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic(cfg)

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

    binder = SheraThreePlaneBinder(cfg, forward_spec, truth_store)

    print("Generating synthetic data...")
    data = binder.model()

    if add_noise:
        rng_key, split_key = jr.split(rng_key)
        if np.min(data) > 100:
            data = np.sqrt(data) * jr.normal(split_key, data.shape) + data
        else:
            data = jr.poisson(split_key, data)

    data_var = data

    print("Configuring Inference...")
    inference_subspec = make_inference_subspec(
        base_spec=inference_spec,
        infer_keys=INFER_KEYS,
        cfg=cfg,
    )

    prior_info = {
        "binary.separation_as":          {"sigma": 1e-4, "dist": "Normal"},
        "binary.position_angle_deg":     {"sigma": 1e-3, "dist": "Uniform"},
        "binary.x_position_as":          {"sigma": 1e-3, "dist": "Normal"},
        "binary.y_position_as":          {"sigma": 1e-3, "dist": "Normal"},
        "binary.log_flux_total":         {"sigma": 1e-3, "dist": "LogNormal"},
        "binary.contrast":               {"sigma": 1e-3, "dist": "LogNormal"},
        "system.plate_scale_as_per_pix": {"sigma": 1e-5, "dist": "LogNormal"},
        "primary.zernike_coeffs_nm": {
            "sigma": np.full_like(truth_store.get("primary.zernike_coeffs_nm"),5),
            "dist": "Normal",
        },
        "secondary.zernike_coeffs_nm": {
            "sigma": np.full_like(truth_store.get("secondary.zernike_coeffs_nm"),5),
            "dist": "Normal",
        },
    }
    prior_spec = PriorSpec.from_info(truth_store, prior_info)

    print("Drawing starting point from priors...")
    rng_key, split_key = jr.split(rng_key)
    init_store = prior_spec.sample(rng_key=split_key, keys=INFER_KEYS)
    init_psf = binder.model(
        strip_structural(init_store, structural_keys=binder.structural_store_keys())
    )

    print("Building the loss function...")
    nll_loss_fn, theta0 = make_binder_nll_fn(
        binder=binder,
        infer_keys=INFER_KEYS,
        data=data,
        var=data_var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=init_store,
    )
    fim_labels = generate_fim_labels(INFER_KEYS, cfg=cfg, store=init_store)

    def map_loss_fn(theta: np.ndarray) -> np.ndarray:
        store_theta = store_unpack_params(inference_subspec, theta, init_store)
        nll_loss = nll_loss_fn(theta)
        prior_gaussian_loss = prior_spec.quadratic_penalty(
            store_theta,
            center_store=truth_store,
            keys=INFER_KEYS,
        )
        return nll_loss + prior_gaussian_loss

    loss_fn = nll_loss_fn

    theta_true = pack_params(inference_subspec, truth_store)
    loss_true = loss_fn(theta_true)
    loss0 = loss_fn(theta0)

    print("Computing Fisher Information Matrix (FIM) for preconditioning...")
    theta_ref = theta_true
    F = fim_theta(nll_loss_fn, theta_ref)
    if save_plots:
        plot_fim(
            F,
            fim_labels,
            save_path=results_dir / "fim.png",
            vmin=4,
            vmax=14,
            show=False,
        )

    fim_diag = jnp.diag(F)

    if use_eigen:
        if truncate_k is not None and truncate_by_eigval is not None:
            print(
                "truncate_k is set; ignoring truncate_by_eigval="
                f"{truncate_by_eigval}."
            )

        eigen_map_full = EigenThetaMap.from_fim(F, theta_ref, whiten=whiten_basis)
        eigvals_full = (
            np.asarray(eigen_map_full.eigvals)
            if eigen_map_full.eigvals is not None
            else None
        )

        if truncate_k is not None:
            k = int(truncate_k)
        elif truncate_by_eigval is not None and eigvals_full is not None:
            k = int(np.sum(eigvals_full >= truncate_by_eigval))
        else:
            k = None

        if k is not None:
            if k <= 0:
                print("truncate_by_eigval removed all modes; keeping top-1.")
                k = 1
            eigen_map = EigenThetaMap.from_fim(
                F,
                theta_ref,
                truncate=k,
                whiten=whiten_basis,
            )
        else:
            eigen_map = eigen_map_full

        eigvals_kept = (
            np.asarray(eigen_map.eigvals)
            if eigen_map.eigvals is not None
            else np.array([])
        )
        k_kept = eigen_map.dim_eigen
        if eigvals_kept.size > 0:
            min_eval = float(np.min(eigvals_kept))
            max_eval = float(np.max(eigvals_kept))
        else:
            min_eval = float("nan")
            max_eval = float("nan")

        print("\nEigenThetaMap summary:")
        print(f"  N total dims: {eigen_map.dim_theta}")
        print(f"  k kept dims : {k_kept}")
        print(f"  eigenvalues : min={min_eval:.3e}, max={max_eval:.3e}")
        print(f"  whiten_basis: {whiten_basis}")

        z0 = eigen_map.z_from_theta(theta0)
        if whiten_basis:
            lr_vec = np.ones_like(z0)
            curvature_vec = np.ones_like(z0)
        else:
            lr_vec = 1.0 / (eigvals_kept + 1e-12)
            curvature_vec = eigvals_kept

        index_map = build_eigen_index_map(eigen_map)
        loss_opt = lambda z: loss_fn(eigen_map.theta_from_z(z))
        theta0_opt = z0
        theta_space = "eigen"
        precond_meta = {
            "lr_vec": lr_vec,
            "whiten_basis": whiten_basis,
            "truncate_k": truncate_k,
            "truncate_by_eigval": truncate_by_eigval,
        }
    else:
        index_map = build_index_map(inference_subspec, init_store, theta=theta0)
        lr_vec = 1.0 / (np.asarray(fim_diag) + 1e-12)
        curvature_vec = fim_diag
        loss_opt = loss_fn
        theta0_opt = theta0
        theta_space = "primitive"
        precond_meta = {"lr_vec": lr_vec}

    print(
        "FIM diag: min={:.3e}, max={:.3e}".format(
            float(jnp.min(fim_diag)),
            float(jnp.max(fim_diag)),
        )
    )
    print(
        "LR vec : min={:.3e}, max={:.3e}".format(
            float(jnp.min(lr_vec)),
            float(jnp.max(lr_vec)),
        )
    )

    print("\nRefactored curvature and learning rates (via index_map):")
    for entry in index_map["entries"]:
        name = entry["name"]
        start = entry["start"]
        stop = entry["stop"]
        shape = entry.get("shape", ())

        n = stop - start

        if n == 1:
            print(
                f"  {name:40s} : "
                f"curv={curvature_vec[start]:.3e}  lr={lr_vec[start]:.3e}"
            )
        else:
            print(f"  {name:40s} : shape={shape}")
            for i, (c, l) in enumerate(
                zip(curvature_vec[start:stop], lr_vec[start:stop])
            ):
                print(
                    f"    {name}[{i:02d}] : "
                    f"curv={c:.3e}  lr={l:.3e}"
                )

    print("Running preconditioned gradient descent...")
    n_iter = FAST_ITER if fast else N_ITER
    theta_final_opt, trace = run_shera_gd(
        loss_fn=loss_opt,
        theta0=theta0_opt,
        index_map=index_map,
        learning_rate=BASE_LR,
        lr_vec=lr_vec,
        num_steps=n_iter,
        runs_dir=results_dir,
        return_artifacts=False,
        theta_space=theta_space,
        curvature=curvature_vec,
        precond=precond_meta,
    )

    if use_eigen:
        theta_final = eigen_map.theta_from_z(theta_final_opt)
    else:
        theta_final = theta_final_opt

    final_store = store_unpack_params(inference_subspec, theta_final, init_store)
    final_psf = binder.model(
        strip_structural(final_store, structural_keys=binder.structural_store_keys())
    )

    labels_by_key = map_labels_to_keys(
        INFER_KEYS,
        fim_labels,
        store=init_store if use_eigen else None,
        index_map=None if use_eigen else index_map,
    )

    print("\n==============================")
    if use_eigen:
        print("Eigenmode Gradient Descent Summary")
    else:
        print("FIM-preconditioned Gradient Descent Summary")
    print("==============================")
    print(f"n_iter = {n_iter}")
    print(f"loss(true theta) = {loss_true:.8g}")
    print(f"loss(init theta0) = {loss0:.8g}")
    print(f"loss(final theta)       = {float(loss_fn(theta_final)):.8g}")
    print("")

    summary_true = {k: truth_store.get(k) for k in INFER_KEYS}
    summary_init = {k: init_store.get(k) for k in INFER_KEYS}
    summary_final = {k: final_store.get(k) for k in INFER_KEYS}
    print_optimization_summary(
        summary_true,
        summary_init,
        summary_final,
        labels=labels_by_key,
    )

    if save_plots:
        print("Plotting outputs...")
        psf_extent_as = (
            binder.cfg.psf_npix
            * binder.base_forward_store.get("system.plate_scale_as_per_pix")
            / 2
            * np.array([-1, 1, -1, 1])
        )

        plot_psf_comparison(
            data=data,
            model=init_psf,
            var=data_var,
            extent=psf_extent_as,
            model_label="Initial Model",
            save_path=results_dir / "initial_psf_comparison.png",
        )

        plot_psf_comparison(
            data=data,
            model=final_psf,
            var=data_var,
            extent=psf_extent_as,
            model_label="Final Model",
            save_path=results_dir / "final_psf_comparison.png",
        )

        losses = np.asarray(trace["loss"])
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes = axes.flatten()
        plot_parameter_history(
            names=("Loss",),
            histories=(losses,),
            true_vals=(float(loss_true),),
            ax=axes[0],
            title="Optimization Loss History",
            show=False,
            close=False,
        )
        window = min(10, n_iter)
        axes[1].plot(np.arange(n_iter - window, n_iter) + 1, losses[-window:])
        axes[1].set_title(f"Last {window} Iterations, Final= {losses[-1]:.3f}")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].axhline(loss_true, linestyle="--", color="k", alpha=0.6)
        final_delta = np.abs(losses[-1] - loss_true)
        if final_delta != 0:
            axes[1].set_ylim(loss_true - 3 * final_delta, loss_true + 3 * final_delta)
        fig.tight_layout()
        fig.savefig(results_dir / "loss_history.png", dpi=300)
        plt.close()

        if use_eigen:
            decoder = lambda z: store_unpack_params(
                inference_subspec,
                eigen_map.theta_from_z(z),
                init_store,
            ).refresh_derived(inference_spec)
        else:
            decoder = lambda theta: store_unpack_params(
                inference_subspec,
                theta,
                init_store,
            ).refresh_derived(inference_spec)

        signals = build_signals(
            trace,
            meta={},
            decoder=decoder,
            truth=truth_store,
            signal_set="intro",
        )
        plot_signals_grid(
            signals,
            results_dir,
            include_zernike_rms=False,
            figsize=(15, 10),
            show=False,
        )

    t1_script = time.time()
    print("Script finished in %.3f sec" % (t1_script - t0_script))


if __name__ == "__main__":
    main()
