"""
Two-plane astrometry retrieval recipe (Shera two-plane).

This script mirrors ``canonical_astrometry.py`` but uses the Shera two-plane
optical system. It follows the migrated config/spec/store/binder pattern:
1) Load/resolve config (preset + overrides).
2) Build the forward spec from the resolved system config.
3) Initialize a ParameterStore from spec defaults, apply truth tweaks, and
   refresh derived parameters.
4) Build a SheraBinder from the config/spec/store.
5) Generate synthetic data, set up inference keys/priors, and run optimization
   (primitive or eigen parameterization).
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

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.optimization import (
    EigenThetaMap,
    fim_theta,
    generate_fim_labels,
    make_binder_nll_fn,
    map_labels_to_keys,
    run_shera_gd,
)
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.run_artifacts import build_param_summary, patch_summary
from dluxshera.inference.signals import build_signals
from dluxshera.params.packing import (
    build_eigen_index_map,
    build_index_map,
    pack_params,
    unpack_params as store_unpack_params,
)
from dluxshera.params.store import ParameterStore
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

JAX_ENABLE_X64 = True
FAST_MODE = False
ADD_NOISE = False
SAVE_PLOTS = True
PLOT_EIGEN_SPECTRUM = True

# Eigenmode settings
USE_EIGEN = True
WHITEN_BASIS = True
TRUNCATE_K = None
TRUNCATE_BY_EIGVAL = None

DEFAULT_SEED = 42
DEFAULT_N_ITER = 60
DEFAULT_FAST_ITER = 30
DEFAULT_BASE_LR = 0.5

# Inference keys (two-plane, no secondary mirror zernikes)
DEFAULT_INFER_KEYS = (
    "source.separation_as",
    "source.position_angle_deg",
    "source.x_position_as",
    "source.y_position_as",
    "source.log_flux_total",
    "source.contrast",
    "optics.plate_scale_as_per_pix",
    "optics.primary.zernike_coeffs_nm",
)

# Presets
DEFAULT_SYSTEM_PRESET = "SHERA_TESTBED_2P"          # two-plane system preset
DEFAULT_EXPERIMENT_PRESET = "CANONICAL_ASTROMETRY"  # experiment preset shared with canonical

# Directories
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path(REPO_ROOT / f"Results/twoplane_astrometry" / TIMESTAMP)

# Plotting defaults
_ = get_default_cmaps()
apply_plot_defaults()
plt.rcParams["image.cmap"] = "inferno_nan"


def main(
    *,
    config_path: Path | None = None,
    system_preset: str = DEFAULT_SYSTEM_PRESET,
    experiment_preset: str = DEFAULT_EXPERIMENT_PRESET,
    fast: bool = FAST_MODE,
    results_dir: Path | None = None,
    use_eigen: bool = USE_EIGEN,
    whiten_basis: bool = WHITEN_BASIS,
    truncate_k: int | None = TRUNCATE_K,
    truncate_by_eigval: float | None = TRUNCATE_BY_EIGVAL,
) -> None:
    """Run the two-plane astrometry recipe."""
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    user_cfg = load_user_config(
        config_path=config_path,
        system_preset=system_preset,
        experiment_preset=experiment_preset,
    )
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    experiment_cfg = resolved_cfg.get("experiment")

    if system_cfg is None:
        raise ValueError("twoplane_astrometry requires a 'system' block in the config.")
    if experiment_cfg is None:
        raise ValueError("twoplane_astrometry requires an 'experiment' block in the config.")

    experiment = _validate_experiment(experiment_cfg)
    infer_keys = tuple(experiment["infer_keys"])
    rng_key = jr.PRNGKey(int(experiment["seed"]))
    save_plots = bool(experiment["save_plots"])
    add_noise = bool(experiment["add_noise"])

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

    system_cfg = dict(system_cfg)  # shallow copy for tweaks
    optics_cfg = dict(system_cfg.get("optics", {}))
    if fast:
        print("FAST_MODE enabled: using reduced wavelengths and Zernike set.")
        # two-plane: primary only
        source_cfg = dict(system_cfg.get("source", {}))
        source_cfg["n_lambda"] = 1
        system_cfg["source"] = source_cfg
        optics_cfg["primary_noll_indices"] = list(range(4, 12))
    system_cfg["optics"] = optics_cfg

    forward_spec = compose_forward_spec({"system": system_cfg})
    truth_store = ParameterStore.from_spec_defaults(forward_spec)
    truth_store = truth_store.replace(
        {
            "source.separation_as": 10.0,
            "source.position_angle_deg": 90.0,
            "source.x_position_as": 0.0,
            "source.y_position_as": 0.0,
            "source.exposure_time_s": 1800.0,
        }
    )
    truth_store = truth_store.refresh_derived(forward_spec)

    binder = SheraBinder(system_cfg, forward_spec, truth_store)

    print("Generating synthetic data...")
    data_psf = binder.model()

    if add_noise:
        rng_key, split_key = jr.split(rng_key)
        if np.min(data_psf) > 100:
            data = np.sqrt(data_psf) * jr.normal(split_key, data_psf.shape) + data_psf
        else:
            data = jr.poisson(split_key, data_psf).astype(data_psf.dtype)
    else:
        data = data_psf

    data_var = jnp.maximum(data_psf, 1.0)

    print("Configuring Inference...")
    missing = [k for k in infer_keys if k not in forward_spec]
    if missing:
        print(
            "Warning: dropping inference keys not present in the two-plane forward spec: "
            + ", ".join(missing)
        )
    infer_keys = tuple(k for k in infer_keys if k in forward_spec)

    inference_subspec = forward_spec.subset(infer_keys)

    prior_info = {
        "source.separation_as": {"sigma": 1e-3, "dist": "Normal"},
        "source.position_angle_deg": {"sigma": 1.67e-2, "dist": "Uniform"},
        "source.x_position_as": {"sigma": 1e-2, "dist": "Normal"},
        "source.y_position_as": {"sigma": 1e-2, "dist": "Normal"},
        "source.log_flux_total": {"sigma": 4.3e-3, "dist": "Normal"},
        "source.contrast": {"sigma": 6e-3, "dist": "LogNormal"},
        "optics.plate_scale_as_per_pix": {"sigma": 4.3e-3, "dist": "LogNormal"},
        "optics.primary.zernike_coeffs_nm": {
            "sigma": np.full_like(truth_store.get("optics.primary.zernike_coeffs_nm"), 5),
            "dist": "Normal",
        },
    }
    prior_spec = PriorSpec.from_info(truth_store, prior_info)

    init_mode = experiment["init_mode"]
    print(f"Initialization mode: {init_mode!r}")
    if init_mode == "prior_sample":
        rng_key, split_key = jr.split(rng_key)
        prior_sample = prior_spec.sample(rng_key=split_key, keys=infer_keys)
        init_store = truth_store.replace(prior_sample.as_dict())
    elif init_mode == "truth":
        init_store = truth_store
    else:
        raise ValueError(
            f"Unsupported experiment.init.mode={init_mode!r}. "
            "Supported modes: 'prior_sample', 'truth'."
        )

    init_psf = binder.model(binder.strip_structural(init_store))

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

    loss_fn = nll_loss_fn
    theta_true = pack_params(inference_subspec, truth_store)
    loss_true = loss_fn(theta_true)
    loss0 = loss_fn(theta0)

    print("Computing Fisher Information Matrix (FIM) for preconditioning...")
    fim_point = theta_true
    F = fim_theta(nll_loss_fn, fim_point)
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
        eigvals_plot, eigvecs_plot = np.linalg.eigh(np.asarray(F))
        sort_idx = np.argsort(eigvals_plot)[::-1]
        eigvals_plot = eigvals_plot[sort_idx]
        eigvecs_plot = eigvecs_plot[:, sort_idx]
        plot_eigenvalue_spectrum(
            eigvals_plot,
            eigvecs_plot,
            labels=fim_labels,
            truncate_k=TRUNCATE_K,
            label_boxes=False,
            save_path=results_dir / "eigenvalue_spectrum.png" if save_plots else None,
            show=False,
        )

    fim_diag = jnp.diag(F)

    if use_eigen:
        if truncate_k is not None and truncate_by_eigval is not None:
            print("truncate_k is set; ignoring truncate_by_eigval={truncate_by_eigval}.")

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

    labels_by_key = map_labels_to_keys(
        infer_keys,
        fim_labels,
        store=init_store if use_eigen else None,
        index_map=None if use_eigen else index_map,
    )

    print("Running preconditioned gradient descent...")
    optimizer_cfg = experiment["optimizer"]
    if optimizer_cfg["kind"] != "gd":
        raise ValueError(
            f"Unsupported experiment.optimizer.kind={optimizer_cfg['kind']!r}. "
            "Only 'gd' is currently implemented in this recipe."
        )
    n_iter = int(optimizer_cfg["n_iter_fast"] if fast else optimizer_cfg["n_iter"])
    metric_payload = {
        "theta_ref": np.asarray(theta0_opt),
        "metric_diag": np.asarray(curvature_vec),
        "lr_scale": np.asarray(lr_vec),
    }
    theta_final_opt, trace = run_shera_gd(
        loss_fn=loss_opt,
        theta0=theta0_opt,
        index_map=index_map,
        learning_rate=float(optimizer_cfg["base_lr"]),
        lr_vec=lr_vec,
        num_steps=n_iter,
        run_dir=results_dir,
        return_artifacts=False,
        theta_space=theta_space,
        metric=metric_payload,
        extra_meta={
            "optimizer": {"preconditioning": precond_meta},
            "theta": {"labels_by_key": labels_by_key},
        },
    )

    if use_eigen:
        theta_final = eigen_map.theta_from_z(theta_final_opt)
    else:
        theta_final = theta_final_opt

    final_store = store_unpack_params(inference_subspec, theta_final, init_store)
    final_psf = binder.model(binder.strip_structural(final_store))

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


def _validate_experiment(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = experiment_cfg.get("optimizer", {})
    outputs_cfg = experiment_cfg.get("outputs", {})

    return {
        "seed": int(experiment_cfg.get("seed", DEFAULT_SEED)),
        "infer_keys": tuple(experiment_cfg.get("infer_keys", DEFAULT_INFER_KEYS)),
        "add_noise": bool(experiment_cfg.get("add_noise", ADD_NOISE)),
        "save_plots": bool(outputs_cfg.get("save_plots", SAVE_PLOTS)),
        "optimizer": {
            "kind": optimizer_cfg.get("kind", "gd"),
            "n_iter": int(optimizer_cfg.get("n_iter", DEFAULT_N_ITER)),
            "n_iter_fast": int(optimizer_cfg.get("n_iter_fast", DEFAULT_FAST_ITER)),
            "base_lr": float(optimizer_cfg.get("base_lr", DEFAULT_BASE_LR)),
        },
        "init_mode": experiment_cfg.get("init", {}).get("mode", "prior_sample"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-plane astrometry strict-schema recipe")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML/JSON config file (must include strict top-level system/experiment blocks).",
    )
    parser.add_argument("--system-preset", type=str, default=DEFAULT_SYSTEM_PRESET)
    parser.add_argument("--experiment-preset", type=str, default=DEFAULT_EXPERIMENT_PRESET)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--fast", action="store_true", help="Use reduced optimization iterations.")
    parser.add_argument("--no-eigen", action="store_true", help="Disable eigenmode optimization.")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    main(
        config_path=args.config,
        system_preset=args.system_preset,
        experiment_preset=args.experiment_preset,
        fast=bool(args.fast),
        results_dir=args.results_dir,
        use_eigen=not bool(args.no_eigen),
    )
