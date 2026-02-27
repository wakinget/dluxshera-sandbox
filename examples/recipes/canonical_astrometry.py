"""
Canonical astrometry retrieval recipe.

Recent Migration reference
------------------------------------------------------
This recipe is the exemplar for the config-schema migration:
1) Load YAML/JSON config from disk.
2) Resolve with `resolve_config` (preset + deep-merge + validation).
3) Build the binder from resolved `system` config.
4) Drive inference settings from resolved `experiment` config.

Use this script as the pattern when migrating other recipes.

This script is the primary, end-to-end onboarding example for the dLuxShera workflow.
It is designed to be read like a Matlab script from top to bottom.
You can open this in your editor and run it.

What this recipe demonstrates
- Building/choosing a three-plane Shera configuration and applying overrides.
- Constructing ParameterSpecs:
    - a forward spec describing the simulation parameters ("forward_spec")
    - a subspec selected from the forward spec via infer_keys
- Initializing a ParameterStore (values) and populating derived parameters
  (e.g., plate scale computed from focal lengths + pixel pitch via registered
  transforms).
- Building a SheraBinder that dispatches source/optics/detector by kind.
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

import argparse
import datetime
import json
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
)
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.run_artifacts import build_param_summary, patch_summary
from dluxshera.inference.signals import build_signals
from dluxshera.config.resolver import resolve_config
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
ADD_NOISE = True
SAVE_PLOTS = True
PLOT_EIGEN_SPECTRUM = True

# Eigenmode settings
USE_EIGEN = True           # Enables re-parameterization
WHITEN_BASIS = True        # If True, scales each eigenvector by 1/sqrt(lambda)
TRUNCATE_K = None          # int or None; keep top-k eigenmodes when set
TRUNCATE_BY_EIGVAL = None  # float or None; only used when TRUNCATE_K is None

DEFAULT_SEED = 42
DEFAULT_N_ITER = 50


class CanonicalSystemConfig:
    """Lightweight config adapter for canonical binder/spec construction."""

    def __init__(self, resolved: dict[str, Any]):
        self.system = resolved

        source = resolved.get("source", {})
        optics = resolved.get("optics", {})
        detector = resolved.get("detector", {})

        for block in (source, optics):
            for key, value in block.items():
                setattr(self, key, value)

        if "dp_path" in optics and not hasattr(self, "diffractive_pupil_path"):
            self.diffractive_pupil_path = optics["dp_path"]

        if "model" in detector:
            self.detector_model = detector["model"]

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
DEFAULT_SYSTEM_PRESET = "SHERA_TESTBED_3P" # System presets describe the system (source, optics, detector)
DEFAULT_EXPERIMENT_PRESET = "CANONICAL_ASTROMETRY" # Experiment presets describe the experiment

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
    """Run the canonical astrometry recipe."""
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    user_cfg = _build_user_config(
        config_path=config_path,
        system_preset=system_preset,
        experiment_preset=experiment_preset,
    )
    cfg = resolve_config(user_cfg)
    if "experiment" not in cfg:
        raise ValueError(
            "canonical_astrometry requires an 'experiment' block in config; "
            "provide experiment.preset (or a full experiment config)."
        )
    if "system" not in cfg:
        raise ValueError(
            "canonical_astrometry requires a 'system' block in config; "
            "provide system.preset (or a full system config)."
        )

    system_cfg = CanonicalSystemConfig(cfg["system"])
    forward_spec = compose_forward_spec(system_cfg)

    experiment = _resolve_experiment_options(cfg["experiment"])
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

    if fast:
        print("FAST_MODE enabled: using reduced iteration count.")

    truth_store = ParameterStore.from_spec_defaults(forward_spec)
    truth_store = truth_store.replace(
        {
            "source.separation_as": 10.0,
            "source.position_angle_deg": 90.0,
            "source.x_position_as": 0.0,
            "source.y_position_as": 0.0,
            "source.exposure_time_s": 1800.,
        }
    )
    truth_store = truth_store.refresh_derived(forward_spec)

    binder = SheraBinder(system_cfg, forward_spec, truth_store)

    print("Generating synthetic data...")
    data_psf = binder.model()

    # Optionally add noise to the data
    if add_noise:
        rng_key, split_key = jr.split(rng_key)
        if np.min(data_psf) > 100:
            # Use gaussian approximation to shot noise if image is bright enough
            data = np.sqrt(data_psf) * jr.normal(split_key, data_psf.shape) + data_psf
        else:
            # Otherwise use poisson statistics
            data = jr.poisson(split_key, data_psf).astype(data_psf.dtype)
            # Casting back to float is important for the optimization
    else: # No noise
        data = data_psf

    # Define data variance = data_psf -> Shot noise dominated
    # Add a minimum variance floor to avoid any division by zero
    data_var = jnp.maximum(data_psf, 1.0)

    print("Configuring Inference...")
    # Phase 5 migration note: inference layout is now defined directly from
    # the forward spec. This file demonstrates the new pattern:
    #   inference_subspec = forward_spec.subset(INFER_KEYS)
    # Pack/unpack operate on this subspec, and derived-labeled keys remain
    # inferable directly (store-wins, no forced runtime recomputation).
    inference_subspec = forward_spec.subset(infer_keys)

    # prior_info defines our initial knowledge of each parameter,
    # and determines the amplitude of the random perturbation applied to the model
    prior_info = {
        "source.separation_as":          {"sigma": 1e-3, "dist": "Normal"},
        "source.position_angle_deg":     {"sigma": 1.67e-2, "dist": "Uniform"}, # 1.67e-2 deg = 1 arcmin
        "source.x_position_as":          {"sigma": 1e-2, "dist": "Normal"},
        "source.y_position_as":          {"sigma": 1e-2, "dist": "Normal"},
        "source.log_flux_total":         {"sigma": 4.3e-3, "dist": "Normal"}, # 4.3e-3 log-flux -> 1% flux cal
        "source.contrast":               {"sigma": 6e-3, "dist": "LogNormal"}, # 6e-3 log-contrast -> indep. 1% star cal
        "optics.plate_scale_as_per_pix": {"sigma": 4.3e-3, "dist": "LogNormal"}, # 4.3e-3 log-platescale -> 1% cal
        "optics.primary.zernike_coeffs_nm": {
            "sigma": np.full_like(truth_store.get("optics.primary.zernike_coeffs_nm"),2),
            "dist": "Normal",
        },
        "optics.secondary.zernike_coeffs_nm": {
            "sigma": np.full_like(truth_store.get("optics.secondary.zernike_coeffs_nm"),2),
            "dist": "Normal",
        },
    }
    prior_spec = PriorSpec.from_info(truth_store, prior_info)

    init_mode = experiment["init_mode"]
    print(f"Initialization mode: {init_mode!r}")
    if init_mode == "prior_sample":
        print("Drawing starting point from priors...")
        rng_key, split_key = jr.split(rng_key)
        init_store = prior_spec.sample(rng_key=split_key, keys=infer_keys)
    elif init_mode == "truth":
        print("Using truth store as initialization (debug mode).")
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

    loss_fn = nll_loss_fn

    theta_true = pack_params(inference_subspec, truth_store)
    loss_true = loss_fn(theta_true)
    loss0 = loss_fn(theta0)

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


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            loaded = json.load(handle)
        elif suffix in {".yaml", ".yml"}:
            import yaml

            loaded = yaml.safe_load(handle)
        else:
            raise ValueError("Config file must be YAML (.yaml/.yml) or JSON (.json).")
    if not isinstance(loaded, dict):
        raise ValueError("Config file must deserialize to a mapping/dict.")
    return loaded


def _build_user_config(
    *,
    config_path: Path | None,
    system_preset: str,
    experiment_preset: str,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "system": {"preset": system_preset},
        "experiment": {"preset": experiment_preset},
    }
    if config_path is None:
        return defaults

    loaded = _load_config_file(config_path)
    if not isinstance(loaded.get("system"), dict) or not isinstance(loaded.get("experiment"), dict):
        raise ValueError("Config file must define top-level 'system' and 'experiment' mappings.")
    return _deep_merge(defaults, loaded)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_experiment_options(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = experiment_cfg.get("optimizer", {})
    outputs_cfg = experiment_cfg.get("outputs", {})

    # TODO: tighten validation once the experiment schema is finalized.
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
    parser = argparse.ArgumentParser(description="Canonical astrometry strict-schema recipe")
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
