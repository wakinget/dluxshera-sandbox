"""Canonical astrometry retrieval recipe.

Configuration model
-------------------
This recipe uses the canonical nested config schema and resolves configuration
in two layers:

1. Preset seeds:
   - system presets contain only top-level ``system``.
   - experiment presets contain only top-level ``experiment``.
2. Optional prescription file:
   - passed via ``--prescription`` (``--config`` remains an alias).
   - may contain ``experiment`` only, or ``experiment`` plus an optional
     top-level ``system`` block.
   - values in the prescription deep-merge over the preset seeds before
     ``resolve_config()`` validates and resolves each block.

The built-in ``CANONICAL_ASTROMETRY`` preset is intentionally experiment-only.
An example full prescription lives next to this recipe at
``examples/recipes/canonical_astrometry_prescription.yaml``.

Execution flow
--------------
1. Load preset seeds plus optional prescription YAML/JSON.
2. Resolve with ``resolve_config`` (preset merge + validation).
3. Build the binder from the resolved ``system`` block.
4. Drive inference settings from the resolved ``experiment`` block.

What this recipe demonstrates
-----------------------------
- Choosing a three-plane Shera configuration and applying overrides.
- Constructing a forward ParameterSpec and an inference subspec from
  ``experiment.infer_keys``.
- Initializing a ParameterStore from the resolved system truth values and
  refreshing derived parameters.
- Building a SheraBinder that dispatches source/optics/detector by kind.
- Generating synthetic data with optional observation noise.
- Defining inference priors and initialization from the resolved experiment
  config.
- Running a single optimization loop and saving plots/artifacts.

Eigenmode re-parameterization
-----------------------------
This recipe supports an optional eigenmode parameterization of the inference
variables. When enabled, optimization runs in an eigen-basis derived from the
Fisher information matrix. ``experiment.eigenmodes`` provides the config
defaults; CLI flags can still disable eigenmode optimization explicitly.
"""
from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path
from typing import Any

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
    diagnose_first_step,
)
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.run_artifacts import build_param_summary, patch_summary
from dluxshera.inference.signals import build_signals
from dluxshera.config.io import load_user_config
from dluxshera.config.numeric import coerce_numeric_value, normalize_optimizer_kwargs
from dluxshera.config.resolver import resolve_config
from dluxshera.params.packing import (
    build_eigen_index_map,
    build_index_map,
    pack_params,
    unpack_params as store_unpack_params,
)
from dluxshera.params.store import ParameterStore
from dluxshera.utils.dtype_diagnostics import print_dtype_audit
from dluxshera.utils.noise import apply_observation_noise
from dluxshera.plot.plotting import (
    apply_plot_defaults,
    get_default_cmaps,
    plot_eigenvalue_spectrum,
    plot_fim,
    plot_parameter_history,
    plot_psf_comparison,
    plot_signals_grid,
)
from dluxshera.plot.printing import print_optimization_summary
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec

##############################
# MAIN SIMULATION PARAMETERS #
##############################

PRESCRIPTION = "examples/recipes/canonical_astrometry_prescription.yaml"

JAX_ENABLE_X64 = True
FAST_MODE = False
ADD_NOISE = False
SAVE_PLOTS = True
PLOT_EIGEN_SPECTRUM = True

# Eigenmode settings
USE_EIGEN = True           # Enables re-parameterization
WHITEN_BASIS = True        # If True, scales each eigenvector by 1/sqrt(lambda)
TRUNCATE_K = None          # int or None; keep top-k eigenmodes when set
TRUNCATE_BY_EIGVAL = None  # float or None; only used when TRUNCATE_K is None

DEFAULT_SEED = 42
DEFAULT_N_ITER = 50

DEFAULT_FAST_ITER = 30
DEFAULT_BASE_LR = 0.5

# User may comment out any keys they wish not to include in the optimization
DEFAULT_INFER_KEYS = (
    "source.separation_as",
    "source.position_angle_deg",
    "source.x_position_as",
    "source.y_position_as",
    "source.log_flux_total",
    "source.contrast",
    "optics.plate_scale_as_per_pix",
    "optics.primary.zernike_coeffs_nm",
    "optics.secondary.zernike_coeffs_nm",
)

# Presets
DEFAULT_SYSTEM_PRESET = "SHERA_FLIGHT_3P" # System presets describe the source, optics, + detector
DEFAULT_EXPERIMENT_PRESET = "CANONICAL_ASTROMETRY" # Experiment presets describe what to do + default settings

# Directories
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path(REPO_ROOT / f"Results/canonical_astrometry" / TIMESTAMP)


# Plotting defaults
_ = get_default_cmaps()
apply_plot_defaults()
plt.rcParams["image.cmap"] = "inferno_nan"


def main(
    *,
    prescription_path: Path | None = PRESCRIPTION,
    system_preset: str = DEFAULT_SYSTEM_PRESET,
    experiment_preset: str = DEFAULT_EXPERIMENT_PRESET,
    fast: bool = FAST_MODE,
    results_dir: Path | None = None,
    use_eigen: bool | None = None,
    whiten_basis: bool | None = None,
    truncate_k: int | None = None,
    truncate_by_eigval: float | None = None,
) -> None:
    """Run the canonical astrometry recipe."""
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    user_cfg = load_user_config(
        config_path=prescription_path,
        system_preset=system_preset,
        experiment_preset=experiment_preset,
    )
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    experiment_cfg = resolved_cfg.get("experiment")

    if system_cfg is None:
        raise ValueError("canonical_astrometry requires a 'system' block ...")
    if experiment_cfg is None:
        raise ValueError("canonical_astrometry requires an 'experiment' block ...")

    resolved_prescription = _repo_relative_path(prescription_path)
    print(
        f"Resolved prescription path: {resolved_prescription or prescription_path}"
        if prescription_path is not None
        else "Resolved prescription path: None (using preset seeds only)"
    )
    print(f"System preset: {_selected_preset(user_cfg, 'system') or 'custom'}")
    print(f"Experiment preset: {_selected_preset(user_cfg, 'experiment') or 'custom'}")

    # --- Optional explicit override: replace detector.layers within config ---
    # Demonstrates how we might manually change the detector layers
    # detector_cfg = system_cfg.get("detector", {}) # Copy the default detector config
    # detector_cfg["layers"] = [{"name": "downsample", "kind": "Downsample", "factor": 3}]
    # detector_cfg["layers"] = [  # This example defines two named layers
    #     {"name": "downsample", "kind": "Downsample", "factor": 3},
    #     {"name": "jitter", "kind": "ApplyJitter", "sigma": 1.0e-1},
    #     {
    #         "name": "diffusion",
    #         "kind": "ApplyConvolution",
    #         "kernel": {
    #             "kind": "gaussian",
    #             "sigma_x": 0.30,
    #             "sigma_y": 0.20,
    #             "theta_deg": 15.0,
    #             "kernel_size": 9,
    #             "units": "detector_pix",
    #         },
    #     },
    # ]
    # system_cfg["detector"] = detector_cfg # Insert into the system config
    # -------------------------------------------------------------

    experiment = _validate_experiment(experiment_cfg)
    eigen_cfg = experiment["eigenmodes"]
    infer_keys = tuple(experiment["infer_keys"])
    rng_key = jr.PRNGKey(int(experiment["seed"]))
    save_plots = bool(experiment["outputs"]["plots"])
    noise_cfg = experiment["noise"]
    optimizer_cfg = experiment["optimizer"]

    use_eigen = eigen_cfg["enable"] if use_eigen is None else bool(use_eigen)
    whiten_basis = eigen_cfg["whiten"] if whiten_basis is None else bool(whiten_basis)
    truncate_k = eigen_cfg["truncate_k"] if truncate_k is None else truncate_k
    truncate_by_eigval = (
        eigen_cfg["truncate_by_eigval"]
        if truncate_by_eigval is None
        else truncate_by_eigval
    )

    results_dir, results_dir_source = _resolve_results_dir(
        cli_results_dir=results_dir,
        experiment_cfg=experiment,
        prescription_path=prescription_path,
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    forward_spec = compose_forward_spec(system_cfg)
    truth_store = ParameterStore.from_spec_defaults(forward_spec)
    truth_store = truth_store.refresh_derived(forward_spec)

    print("Starting Simulation...")
    print("Creating Config, Spec, Store, and Binder...")
    print(f"Resolved results_dir: {results_dir} ({results_dir_source})")
    print("Eigenmode configuration:")
    print(f"  use_eigen={use_eigen}")
    print(f"  whiten_basis={whiten_basis}")
    print(f"  truncate_k={truncate_k}")
    print(f"  truncate_by_eigval={truncate_by_eigval}")

    t0_script = time.time()

    if fast:
        print("FAST_MODE enabled: using reduced iteration count.")

    binder = SheraBinder(system_cfg, forward_spec, truth_store)

    print("Generating synthetic data...")
    data_psf = binder.model()

    if noise_cfg["enabled"]:
        rng_key, noise_key = jr.split(rng_key)
    else:
        noise_key = rng_key
    data, data_var = apply_observation_noise(
        data_psf,
        noise_cfg=noise_cfg,
        rng_key=noise_key,
        detector_spec=getattr(binder.detector, "spec", None),
        exposure_time_s=truth_store.get("source.exposure_time_s", default=None),
    )

    print("Configuring Inference...")
    # Phase 5 migration note: inference layout is now defined directly from
    # the forward spec. This file demonstrates the new pattern:
    #   inference_subspec = forward_spec.subset(INFER_KEYS)
    # Pack/unpack operate on this subspec, and derived-labeled keys remain
    # inferable directly (store-wins, no forced runtime recomputation).
    inference_subspec = forward_spec.subset(infer_keys)

    prior_info = experiment["priors"] or _default_prior_info(truth_store)
    prior_spec = PriorSpec.from_info(truth_store, prior_info)

    init_mode = experiment["init"]["sampling"]
    print(f"Initialization mode: {init_mode!r}")
    if init_mode == "prior":
        print("Drawing starting point from priors...")
        rng_key, split_key = jr.split(rng_key)
        prior_sample = prior_spec.sample(rng_key=split_key, keys=infer_keys)
        # Seed structural defaults from the truth store, then apply sampled infer keys
        init_store = truth_store.replace(prior_sample.as_dict())
    elif init_mode in {"explicit", "truth"}:
        print("Using truth store as initialization.")
        init_store = truth_store
    else:
        raise ValueError(
            f"Unsupported experiment.init.mode={init_mode!r}. "
            "Supported modes: 'prior_sample', 'truth'."
        )
    # Apply the randomly drawn perturbations to the model and produce an image
    # Use binder.strip_structural() so structural keys are removed using
    # the binder's contract-driven structural policy before binder.model().
    init_psf = binder.model(
        binder.strip_structural(init_store)
    )

    print("Building the loss function...")
    nll_loss_fn, theta0 = make_binder_nll_fn(
        binder=binder,
        infer_keys=infer_keys,
        data=data,
        var=data_var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=init_store,
    )
    fim_labels = generate_fim_labels(infer_keys, cfg=system_cfg, store=init_store)

    def map_loss_fn(theta: np.ndarray) -> np.ndarray:
        store_theta = store_unpack_params(inference_subspec, theta, init_store)
        nll_loss = nll_loss_fn(theta)
        prior_gaussian_loss = prior_spec.quadratic_penalty(
            store_theta,
            center_store=truth_store,
            keys=infer_keys,
        )
        return nll_loss + prior_gaussian_loss

    loss_kind = optimizer_cfg["loss"]
    loss_fn = map_loss_fn if loss_kind == "map" else nll_loss_fn

    theta_true = pack_params(inference_subspec, truth_store)
    loss_true = loss_fn(theta_true)
    loss0 = loss_fn(theta0)
    print_dtype_audit(
        "canonical_astrometry data_and_loss",
        {
            "data_psf": data_psf,
            "data": data,
            "data_var": data_var,
            "init_psf": init_psf,
            "theta_true": theta_true,
            "theta0": theta0,
            "loss_true": loss_true,
            "loss0": loss0,
        },
    )

    print("Computing Fisher Information Matrix (FIM) for preconditioning...")
    fim_point = theta_true
    F = fim_theta(nll_loss_fn, fim_point)
    fim_labels = generate_fim_labels(infer_keys, cfg=system_cfg, store=init_store)
    if save_plots:
        plot_fim(
            F,
            fim_labels,
            save_path=results_dir / "fim.png",
            vmin=4,
            vmax=14,
            show=False,
        )

    if PLOT_EIGEN_SPECTRUM:
        eigvals, eigvecs = np.linalg.eigh(np.asarray(F))
        sort_idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sort_idx]
        eigvecs = eigvecs[:, sort_idx]

        if truncate_k is not None:
            spectrum_truncate_k = int(truncate_k)
        elif truncate_by_eigval is not None:
            spectrum_truncate_k = int(np.sum(eigvals >= truncate_by_eigval))
            if spectrum_truncate_k <= 0:
                spectrum_truncate_k = 1
        else:
            spectrum_truncate_k = None

        plot_eigenvalue_spectrum(
            eigvals,
            eigvecs,
            labels=fim_labels,
            truncate_k=spectrum_truncate_k,
            label_boxes=False,
            save_path=results_dir / "eigenvalue_spectrum.png" if save_plots else None,
            show=False,
        )

    fim_diag = jnp.diag(F)

    if use_eigen:
        if truncate_k is not None and truncate_by_eigval is not None:
            print(
                "truncate_k is set; ignoring truncate_by_eigval="
                f"{truncate_by_eigval}."
            )

        # NOTE: theta_ref is the origin for the eigen coefficients (z). Truncation
        # zeroes discarded components *relative to theta_ref*. If we set theta_ref
        # to the truth, truncation snaps discarded directions back to truth and
        # makes severe truncation look unrealistically powerful. Using the initial
        # guess freezes discarded directions at their initial offsets, which is
        # the intended pedagogical behavior for this recipe.
        theta_ref = theta0
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
            curvature_vec = np.maximum(eigvals_kept, 1e-8)
            lr_vec = 1.0 / (curvature_vec + 1e-12)

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
        curvature_vec = np.maximum(np.asarray(fim_diag), 1e-8)
        lr_vec = 1.0 / (curvature_vec + 1e-12)
        loss_opt = loss_fn
        theta0_opt = theta0
        theta_space = "primitive"
        precond_meta = {"lr_vec": lr_vec}

    print_dtype_audit(
        "canonical_astrometry optimizer",
        {
            "F": F,
            "fim_diag": fim_diag,
            "theta0_opt": theta0_opt,
            "curvature_vec": curvature_vec,
            "lr_vec": lr_vec,
        },
    )

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

    labels_by_key = map_labels_to_keys(
        infer_keys,
        fim_labels,
        store=init_store if use_eigen else None,
        index_map=None if use_eigen else index_map,
    )

    print("Running preconditioned gradient descent...")
    if optimizer_cfg["kind"] not in {"sgd", "adam"}:
        raise ValueError(
            f"Unsupported experiment.optimizer.kind={optimizer_cfg['kind']!r}. "
            "Supported kinds: 'sgd'/'adam'."
        )
    n_iter = int(optimizer_cfg["n_iter_fast"] if fast else optimizer_cfg["n_iter"])
    metric_payload = {
        "theta_ref": np.asarray(theta0_opt),
        "metric_diag": np.asarray(curvature_vec),
        "lr_scale": np.asarray(lr_vec),
    }

    run_diagnosis = False
    if run_diagnosis:
        diag = diagnose_first_step(
            loss_fn=loss_opt,
            theta0=theta0_opt,
            learning_rate=float(optimizer_cfg["base_lr"]),
            lr_vec=lr_vec,
            optimizer_kind="sgd",
            index_map=index_map,
            verbose=True,
        )
        print("First-step diagnostic:")
        print(
            f"  loss0={diag['loss0']:.6g} finite={diag['loss0_finite']} | "
            f"grad_finite={diag['grad0_finite']} | theta1_finite={diag['theta1_finite']} | "
            f"loss1={diag['loss1']:.6g} finite={diag['loss1_finite']}"
        )
        print(
            f"  grad0 min/max={diag['grad0_min']:.3e}/{diag['grad0_max']:.3e} | "
            f"delta min/max={diag['delta_min']:.3e}/{diag['delta_max']:.3e}"
        )
        if lr_vec is not None:
            print(
                f"  lr_vec min/max={diag['lr_vec_min']:.3e}/{diag['lr_vec_max']:.3e}"
            )
        if diag.get("top_grad"):
            topg = ", ".join(f"{i}:{v:.2e}" for i, v in diag["top_grad"])
            print(f"  top |grad|: {topg}")
        if diag.get("top_delta"):
            topl = ", ".join(f"{i}:{v:.2e}" for i, v in diag["top_delta"])
            print(f"  top |delta|: {topl}")

        diag_unscaled = diagnose_first_step(
            loss_fn=loss_opt,
            theta0=theta0_opt,
            learning_rate=float(optimizer_cfg["base_lr"]),
            lr_vec=None,
            optimizer_kind="sgd",
        )
        diag_tiny = diagnose_first_step(
            loss_fn=loss_opt,
            theta0=theta0_opt,
            learning_rate=float(optimizer_cfg["base_lr"]) * 1e-3,
            lr_vec=None,
            optimizer_kind="sgd",
        )
        print(
            "Variant first-step diagnostics: "
            f"unscaled loss1_finite={diag_unscaled['loss1_finite']} "
            f"theta1_finite={diag_unscaled['theta1_finite']}; "
            f"tiny loss1_finite={diag_tiny['loss1_finite']} "
            f"theta1_finite={diag_tiny['theta1_finite']}"
        )

    theta_final_opt, trace = run_shera_gd(
        loss_fn=loss_opt,
        theta0=theta0_opt,
        index_map=index_map,
        learning_rate=float(optimizer_cfg["base_lr"]),
        lr_vec=lr_vec,
        num_steps=n_iter,
        optimizer_kind=str(optimizer_cfg["kind"]),
        optimizer_kwargs=dict(optimizer_cfg["kwargs"]),
        run_dir=results_dir,
        return_artifacts=False,
        theta_space=theta_space,
        metric=metric_payload,
        extra_meta={
            "optimizer": {
                "preconditioning": precond_meta,
                "kind": optimizer_cfg["kind"],
                "kwargs": dict(optimizer_cfg["kwargs"]),
                "loss": optimizer_cfg["loss"],
            },
            "theta": {"labels_by_key": labels_by_key},
        },
    )

    if use_eigen:
        theta_final = eigen_map.theta_from_z(theta_final_opt)
    else:
        theta_final = theta_final_opt

    final_store = store_unpack_params(inference_subspec, theta_final, init_store)
    final_psf = binder.model(
        binder.strip_structural(final_store)
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

    summary_true = {k: truth_store.get(k) for k in infer_keys}
    summary_init = {k: init_store.get(k) for k in infer_keys}
    summary_final = {k: final_store.get(k) for k in infer_keys}
    param_summary = build_param_summary(summary_init, summary_final, truth=summary_true)
    patch_summary(results_dir, {"param_summary": param_summary})
    print_optimization_summary(
        summary_true,
        summary_init,
        summary_final,
        labels=labels_by_key,
    )

    if save_plots:
        print("Plotting outputs...")
        psf_extent_as = (
            binder.base_forward_store.get("optics.psf_npix")
            * binder.base_forward_store.get("optics.plate_scale_as_per_pix")
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
            ).refresh_derived(forward_spec)
        else:
            decoder = lambda theta: store_unpack_params(
                inference_subspec,
                theta,
                init_store,
            ).refresh_derived(forward_spec)

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


def _require_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"experiment.{key} must be a mapping")
    return value


def _optional_dict(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"experiment.{key} must be a mapping when provided")
    return dict(value)


def _require_bool(parent: dict[str, Any], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean for {key!r}, got {type(value).__name__}")
    return value


def _optional_bool(parent: dict[str, Any], key: str, default: bool) -> bool:
    value = parent.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Expected boolean for {key!r}, got {type(value).__name__}")
    return value


def _optional_bool_alias(
    parent: dict[str, Any],
    primary_key: str,
    alias_keys: tuple[str, ...],
    default: bool,
) -> bool:
    if primary_key in parent:
        value = parent[primary_key]
    else:
        value = default
        for alias in alias_keys:
            if alias in parent:
                value = parent[alias]
                break
    if not isinstance(value, bool):
        raise ValueError(
            f"Expected boolean for {primary_key!r}, got {type(value).__name__}"
        )
    return value


def _require_int(parent: dict[str, Any], key: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for {key!r}, got {type(value).__name__}")
    return value


def _optional_int(parent: dict[str, Any], key: str, default: int) -> int:
    value = parent.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"Expected integer for {key!r}, got {type(value).__name__}")
    return value


def _require_number(parent: dict[str, Any], key: str) -> float:
    value = parent.get(key)
    return float(coerce_numeric_value(value, path=f"experiment.{key}"))


def _optional_number(parent: dict[str, Any], key: str, default: float) -> float:
    value = parent.get(key, default)
    return float(coerce_numeric_value(value, path=f"experiment.{key}"))


def _require_str(parent: dict[str, Any], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Expected string for {key!r}, got {type(value).__name__}")
    return value


def _optional_str(parent: dict[str, Any], key: str, default: str) -> str:
    value = parent.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"Expected string for {key!r}, got {type(value).__name__}")
    return value


def _require_str_list(parent: dict[str, Any], key: str) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{key!r} must be a list of strings")
    return tuple(value)


def _normalize_init_sampling(raw_value: str) -> str:
    if raw_value in {"prior", "prior_sample"}:
        return "prior"
    if raw_value in {"explicit", "truth"}:
        return "explicit"
    raise ValueError(
        f"Unsupported experiment.init sampling mode {raw_value!r}; "
        "expected 'prior', 'prior_sample', 'explicit', or 'truth'."
    )


def _default_prior_info(truth_store: ParameterStore) -> dict[str, Any]:
    return {
        "source.separation_as": {"sigma": 1e-3, "dist": "Normal"},
        "source.position_angle_deg": {
            "sigma": 1.67e-2,
            "dist": "Uniform",
        },
        "source.x_position_as": {"sigma": 1e-2, "dist": "Normal"},
        "source.y_position_as": {"sigma": 1e-2, "dist": "Normal"},
        "source.log_flux_total": {"sigma": 4.3e-3, "dist": "Normal"},
        "source.contrast": {"sigma": 6e-3, "dist": "LogNormal"},
        "optics.plate_scale_as_per_pix": {"sigma": 4.3e-3, "dist": "LogNormal"},
        "optics.primary.zernike_coeffs_nm": {
            "sigma": np.full_like(
                truth_store.get("optics.primary.zernike_coeffs_nm"),
                2,
            ),
            "dist": "Normal",
        },
        "optics.secondary.zernike_coeffs_nm": {
            "sigma": np.full_like(
                truth_store.get("optics.secondary.zernike_coeffs_nm"),
                2,
            ),
            "dist": "Normal",
        },
    }


def _selected_preset(user_cfg: dict[str, Any], block_name: str) -> str | None:
    block = user_cfg.get(block_name)
    if not isinstance(block, dict):
        return None
    preset = block.get("preset")
    if isinstance(preset, str) and preset:
        return preset
    return None


def _repo_relative_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _resolve_results_dir(
    *,
    cli_results_dir: Path | None,
    experiment_cfg: dict[str, Any],
    prescription_path: Path | None,
) -> tuple[Path, str]:
    if cli_results_dir is not None:
        return Path(cli_results_dir), "--results-dir"

    outputs_cfg = experiment_cfg.get("outputs", {})
    outdir_value = outputs_cfg.get("outdir")
    if outdir_value is None:
        return DEFAULT_RESULTS_DIR, "default timestamped directory"

    outdir_path = Path(outdir_value).expanduser()
    if not outdir_path.is_absolute() and prescription_path is not None:
        outdir_path = (prescription_path.parent / outdir_path).resolve()
        return outdir_path, "experiment.outputs.outdir (relative to prescription)"
    return outdir_path, "experiment.outputs.outdir"


def _validate_experiment(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict):
        raise ValueError("experiment config must be a mapping")

    if "experiment" in cfg:
        experiment_cfg = cfg["experiment"]
        if not isinstance(experiment_cfg, dict):
            raise ValueError("cfg['experiment'] must be a mapping")
    else:
        experiment_cfg = cfg

    seed = _require_int(experiment_cfg, "seed")
    infer_keys = _require_str_list(experiment_cfg, "infer_keys")

    noise_cfg = _optional_dict(experiment_cfg, "noise")
    init_cfg = _optional_dict(experiment_cfg, "init")
    optimizer_cfg = _optional_dict(experiment_cfg, "optimizer")
    outputs_cfg = _optional_dict(experiment_cfg, "outputs")
    priors_cfg = _optional_dict(experiment_cfg, "priors")

    eigen_cfg_raw = experiment_cfg.get("eigenmodes", experiment_cfg.get("eigen"))
    if eigen_cfg_raw is None:
        eigen_cfg = {}
    elif isinstance(eigen_cfg_raw, dict):
        eigen_cfg = dict(eigen_cfg_raw)
    else:
        raise ValueError("experiment.eigenmodes must be a mapping when provided")

    init_sampling_raw = init_cfg.get("sampling", init_cfg.get("mode", "prior"))
    if not isinstance(init_sampling_raw, str):
        raise ValueError("experiment.init.sampling must be a string")
    init_sampling = _normalize_init_sampling(init_sampling_raw)

    optimizer_kind = _optional_str(optimizer_cfg, "kind", "sgd")
    if optimizer_kind not in {"sgd", "adam"}:
        raise ValueError(
            f"Unsupported experiment.optimizer.kind {optimizer_kind!r}; "
            "expected 'sgd' or 'adam'."
        )
    loss_kind = _optional_str(optimizer_cfg, "loss", "nll")
    if loss_kind not in {"nll", "map"}:
        raise ValueError(
            f"Unsupported experiment.optimizer.loss {loss_kind!r}; "
            "expected 'nll' or 'map'."
        )

    plots_enabled = outputs_cfg.get("plots", outputs_cfg.get("save_plots", SAVE_PLOTS))
    if not isinstance(plots_enabled, bool):
        raise ValueError("experiment.outputs.plots must be a boolean")
    outdir_value = outputs_cfg.get("outdir")
    if outdir_value is not None and not isinstance(outdir_value, str):
        raise ValueError("experiment.outputs.outdir must be a string or null")

    truncate_k_value = eigen_cfg.get("truncate_k", TRUNCATE_K)
    if truncate_k_value is not None and not isinstance(truncate_k_value, int):
        raise ValueError("experiment.eigenmodes.truncate_k must be an integer or null")
    truncate_by_eigval_value = eigen_cfg.get(
        "truncate_by_eigval",
        TRUNCATE_BY_EIGVAL,
    )
    if truncate_by_eigval_value is not None and not isinstance(
        truncate_by_eigval_value,
        (int, float),
    ):
        raise ValueError(
            "experiment.eigenmodes.truncate_by_eigval must be a number or null"
        )

    return {
        "seed": seed,
        "infer_keys": infer_keys,
        "noise": {
            "enabled": _optional_bool(noise_cfg, "enabled", ADD_NOISE),
            "photon_noise": _optional_bool(noise_cfg, "photon_noise", True),
            "read_noise": _optional_bool(noise_cfg, "read_noise", False),
            "dark_current": _optional_bool(noise_cfg, "dark_current", False),
        },
        "init": {"sampling": init_sampling},
        "optimizer": {
            "kind": optimizer_kind,
            "loss": loss_kind,
            "n_iter": _optional_int(optimizer_cfg, "n_iter", DEFAULT_N_ITER),
            "n_iter_fast": _optional_int(
                optimizer_cfg,
                "n_iter_fast",
                DEFAULT_FAST_ITER,
            ),
            "base_lr": _optional_number(optimizer_cfg, "base_lr", DEFAULT_BASE_LR),
            "kwargs": normalize_optimizer_kwargs(
                optimizer_kind,
                optimizer_cfg.get("kwargs", {}),
                path="experiment.optimizer.kwargs",
            ),
        },
        "outputs": {
            "plots": plots_enabled,
            "outdir": outdir_value,
        },
        "priors": priors_cfg,
        "eigenmodes": {
            "enable": _optional_bool_alias(
                eigen_cfg,
                "enable",
                ("use_eigen",),
                USE_EIGEN,
            ),
            "whiten": _optional_bool_alias(
                eigen_cfg,
                "whiten",
                ("whiten_basis",),
                WHITEN_BASIS,
            ),
            "truncate_k": truncate_k_value,
            "truncate_by_eigval": (
                float(truncate_by_eigval_value)
                if truncate_by_eigval_value is not None
                else None
            ),
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Canonical astrometry recipe with preset seeds plus optional prescription overrides"
    )
    parser.add_argument(
        "--config",
        "--prescription",
        dest="prescription",
        type=Path,
        default=None,
        help=(
            "Path to YAML/JSON prescription. A prescription may contain "
            "top-level experiment only, or experiment plus an optional system "
            "block. Values deep-merge over --system-preset/--experiment-preset."
        ),
    )
    parser.add_argument("--system-preset", type=str, default=DEFAULT_SYSTEM_PRESET)
    parser.add_argument("--experiment-preset", type=str, default=DEFAULT_EXPERIMENT_PRESET)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--fast", action="store_true", help="Use reduced optimization iterations.")
    parser.add_argument(
        "--no-eigen",
        dest="use_eigen",
        action="store_false",
        default=None,
        help="Disable eigenmode optimization (overrides experiment.eigenmodes.enable).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        prescription_path=args.prescription,
        system_preset=args.system_preset,
        experiment_preset=args.experiment_preset,
        fast=bool(args.fast),
        results_dir=args.results_dir,
        use_eigen=args.use_eigen,
    )
