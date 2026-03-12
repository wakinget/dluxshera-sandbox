"""
Canonical astrometry Monte Carlo (multi-start) recipe.

This script mirrors ``canonical_astrometry.py`` but runs a Monte Carlo sweep:
synthetic data and priors are fixed once, and each run draws a fresh
initialization from those priors.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

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
from dluxshera.inference.run_artifacts import (
    _now_iso_local_ms,
    build_param_summary,
    patch_summary,
)
from dluxshera.inference.sweeps import write_sweep_csv
from dluxshera.params.packing import (
    build_eigen_index_map,
    build_index_map,
    pack_params,
    unpack_params as store_unpack_params,
)
from dluxshera.params.store import ParameterStore
from dluxshera.plot.plotting import apply_plot_defaults, get_default_cmaps, plot_fim
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec

##############################
# MAIN SIMULATION PARAMETERS #
##############################

JAX_ENABLE_X64 = True
FAST_MODE = False
ADD_NOISE = False
SAVE_PLOTS = False

# Monte Carlo settings
N_RUNS = 5
RUN_ID_PREFIX = "mc"

DEFAULT_SEED = 42
DEFAULT_N_ITER = 60
DEFAULT_FAST_ITER = 30
DEFAULT_BASE_LR = 0.5

# Eigenmode settings
USE_EIGEN = True
WHITEN_BASIS = True
TRUNCATE_K = None
TRUNCATE_BY_EIGVAL = None

# Inference keys (may be overridden by experiment config)
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
DEFAULT_SYSTEM_PRESET = "SHERA_TESTBED_3P"
DEFAULT_EXPERIMENT_PRESET = "CANONICAL_ASTROMETRY"

# Directories
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = Path(REPO_ROOT / f"Results/canonical_monte_carlo" / TIMESTAMP)

# Plotting defaults
_ = get_default_cmaps()
apply_plot_defaults()
plt.rcParams["image.cmap"] = "inferno_nan"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, jnp.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


def _mc_run_id(index: int, prefix: str = RUN_ID_PREFIX) -> str:
    return f"{prefix}_{index:04d}"


def _repo_relative_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, REPO_ROOT)).as_posix()


def _summarize_prior_info(prior_info: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, entry in prior_info.items():
        sigma = entry.get("sigma") if isinstance(entry, Mapping) else None
        if isinstance(sigma, (np.ndarray, jnp.ndarray)):
            sigma_summary = {"shape": list(sigma.shape), "dtype": str(sigma.dtype)}
        else:
            sigma_summary = sigma
        summary[key] = {"dist": entry.get("dist"), "sigma": sigma_summary}
    return summary


def _maybe_warn_missing_artifacts(run_dir: Path) -> None:
    required = ["meta.json", "summary.json", "trace.npz"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        print(f"WARNING: run artifacts missing in {run_dir}: {', '.join(missing)}")


def _git_info() -> dict[str, str | None]:
    commit = None
    status = None
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
        status = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT)
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return {"commit": commit, "status": status}


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
    n_runs: int | None = None,
    run_id_prefix: str | None = None,
    add_noise: bool | None = None,
    save_plots: bool | None = None,
) -> None:
    """Run the canonical Monte Carlo astrometry recipe."""
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
        raise ValueError("canonical_monte_carlo requires a 'system' block in the config.")
    if experiment_cfg is None:
        raise ValueError("canonical_monte_carlo requires an 'experiment' block in the config.")

    experiment = _validate_experiment(experiment_cfg)
    if n_runs is not None:
        experiment["n_runs"] = int(n_runs)
    if run_id_prefix is not None:
        experiment["run_id_prefix"] = str(run_id_prefix)
    if add_noise is not None:
        experiment["add_noise"] = bool(add_noise)
    if save_plots is not None:
        experiment["save_plots"] = bool(save_plots)

    infer_keys = tuple(experiment["infer_keys"])
    rng_key = jr.PRNGKey(int(experiment["seed"]))
    save_plots = bool(experiment["save_plots"])
    add_noise = bool(experiment["add_noise"])
    n_runs = int(experiment["n_runs"])
    run_id_prefix = str(experiment["run_id_prefix"])

    results_dir = results_dir or DEFAULT_RESULTS_DIR
    runs_dir = results_dir / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Monte Carlo Simulation...")
    print("Creating Config, Spec, Store, and Binder...")
    print("Eigenmode configuration:")
    print(f"  use_eigen={use_eigen}")
    print(f"  whiten_basis={whiten_basis}")
    print(f"  truncate_k={truncate_k}")
    print(f"  truncate_by_eigval={truncate_by_eigval}")

    t0_script = time.time()

    system_cfg = deepcopy(system_cfg)
    if fast:
        print("FAST_MODE enabled: reducing wavelengths and Zernike indices.")
        source_cfg = system_cfg.get("source", {})
        if isinstance(source_cfg, Mapping):
            source_cfg["n_lambda"] = 1
            system_cfg["source"] = source_cfg
        optics_cfg = system_cfg.get("optics", {})
        if isinstance(optics_cfg, Mapping):
            optics_cfg["primary_noll_indices"] = list(range(4, 9))
            optics_cfg["secondary_noll_indices"] = list(range(4, 9))
            system_cfg["optics"] = optics_cfg

    forward_spec = compose_forward_spec(system_cfg)
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

    np.savez(
        results_dir / "data.npz",
        noise=add_noise,
        data_psf=np.asarray(data_psf),
        data=np.asarray(data),
        data_var=np.asarray(data_var),
    )

    print("Configuring Inference...")
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
            "sigma": np.full_like(truth_store.get("optics.primary.zernike_coeffs_nm"), 2),
            "dist": "Normal",
        },
        "optics.secondary.zernike_coeffs_nm": {
            "sigma": np.full_like(truth_store.get("optics.secondary.zernike_coeffs_nm"), 2),
            "dist": "Normal",
        },
    }
    prior_spec = PriorSpec.from_info(truth_store, prior_info)

    experiment_created_at = _now_iso_local_ms()
    manifest = {
        "script": "canonical_monte_carlo.py",
        "git": _git_info(),
        "created_at": experiment_created_at,
        "n_runs": n_runs,
        "rng_seed": int(experiment["seed"]),
        "fast": fast,
        "add_noise": add_noise,
        "use_eigen": use_eigen,
        "whiten_basis": whiten_basis,
        "truncate_k": truncate_k,
        "truncate_by_eigval": truncate_by_eigval,
        "infer_keys": list(infer_keys),
        "prior_info": _summarize_prior_info(prior_info),
        "dp_path": _repo_relative_path(system_cfg.get("optics", {}).get("dp_path")),
        "config": resolved_cfg,
        "experiment": experiment,
    }
    _write_json(results_dir / "manifest.json", _coerce_jsonable(manifest))

    print("Building the loss function...")
    nll_loss_fn, _ = make_binder_nll_fn(
        binder=binder,
        infer_keys=infer_keys,
        data=data,
        var=data_var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=truth_store,
    )
    fim_labels = generate_fim_labels(infer_keys, cfg=system_cfg, store=truth_store)

    theta_true = pack_params(inference_subspec, truth_store)
    loss_true = float(nll_loss_fn(theta_true))

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

    fim_diag = jnp.diag(F)

    if use_eigen:
        theta_space = "eigen"
        precond_meta_base = {
            "method": "eigen",
            "whiten_basis": whiten_basis,
            "truncate_k": truncate_k,
            "truncate_by_eigval": truncate_by_eigval,
        }
    else:
        theta_space = "primitive"
        precond_meta_base = {"method": "fim_diag"}

    print(
        "FIM diag: min={:.3e}, max={:.3e}".format(
            float(jnp.min(fim_diag)),
            float(jnp.max(fim_diag)),
        )
    )

    optimizer_cfg = experiment["optimizer"]
    if optimizer_cfg["kind"] != "gd":
        raise ValueError(
            f"Unsupported experiment.optimizer.kind={optimizer_cfg['kind']!r}. Only 'gd' is implemented."
        )
    n_iter = int(optimizer_cfg["n_iter_fast"] if fast else optimizer_cfg["n_iter"])
    base_lr = float(optimizer_cfg["base_lr"])

    print(f"\nRunning {n_runs} Monte Carlo optimizations...")

    success_count = 0
    failure_count = 0

    for run_index in range(n_runs):
        rng_key, split_key = jr.split(rng_key)
        run_id = _mc_run_id(run_index, prefix=run_id_prefix)
        print(f"\n--- Run {run_index + 1}/{n_runs} ({run_id}) ---")

        init_mode = experiment["init_mode"]
        if init_mode == "prior_sample":
            prior_sample = prior_spec.sample(rng_key=split_key, keys=infer_keys)
            init_store = truth_store.replace(prior_sample.as_dict())
        elif init_mode == "truth":
            init_store = truth_store
        else:
            raise ValueError(
                f"Unsupported experiment.init.mode={init_mode!r}. Supported: 'prior_sample', 'truth'."
            )
        init_psf = binder.model(binder.strip_structural(init_store))

        _, theta0 = make_binder_nll_fn(
            binder=binder,
            infer_keys=infer_keys,
            data=data,
            var=data_var,
            noise_model="gaussian",
            reduce="sum",
            theta0_store=init_store,
        )

        eigen_map = None
        curvature_vec = fim_diag
        lr_vec = None
        theta0_opt = theta0
        loss_opt = nll_loss_fn
        index_map = None

        if use_eigen:
            if truncate_k is not None and truncate_by_eigval is not None:
                print(
                    f"truncate_k is set; ignoring truncate_by_eigval={truncate_by_eigval}."
                )

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
                if eigen_map is not None and eigen_map.eigvals is not None
                else np.array([])
            )
            k_kept = eigen_map.dim_eigen if eigen_map is not None else 0
            if eigvals_kept.size > 0:
                min_eval = float(np.min(eigvals_kept))
                max_eval = float(np.max(eigvals_kept))
            else:
                min_eval = float("nan")
                max_eval = float("nan")

            print("\nEigenThetaMap summary:")
            print(f"  N total dims: {eigen_map.dim_theta if eigen_map else 0}")
            print(f"  k kept dims : {k_kept}")
            print(f"  eigenvalues : min={min_eval:.3e}, max={max_eval:.3e}")
            print(f"  whiten_basis: {whiten_basis}")

            if eigen_map is not None:
                z0 = eigen_map.z_from_theta(theta0)
                if whiten_basis:
                    lr_vec = np.ones_like(z0)
                    curvature_vec = np.ones_like(z0)
                else:
                    lr_vec = 1.0 / (eigvals_kept + 1e-12)
                    curvature_vec = eigvals_kept

                index_map = build_eigen_index_map(eigen_map)
                loss_opt = lambda z: nll_loss_fn(eigen_map.theta_from_z(z))
                theta0_opt = z0
        else:
            index_map = build_index_map(inference_subspec, init_store, theta=theta0)
            lr_vec = 1.0 / (np.asarray(fim_diag) + 1e-12)
            curvature_vec = fim_diag
            loss_opt = nll_loss_fn
            theta0_opt = theta0

        lr_vec = np.asarray(lr_vec) if lr_vec is not None else None
        metric_payload = {
            "theta_ref": np.asarray(theta0_opt),
            "metric_diag": np.asarray(curvature_vec),
            "lr_scale": np.asarray(lr_vec) if lr_vec is not None else None,
        }
        precond_meta = {**precond_meta_base, "lr_vec": metric_payload["lr_scale"]}

        labels_by_key = map_labels_to_keys(
            infer_keys,
            fim_labels,
            store=init_store if use_eigen else None,
            index_map=None if use_eigen else index_map,
        )

        print("Running preconditioned gradient descent...")
        theta_final_opt, trace, artifacts = run_shera_gd(
            loss_fn=loss_opt,
            theta0=theta0_opt,
            index_map=index_map,
            learning_rate=base_lr,
            lr_vec=lr_vec,
            num_steps=n_iter,
            runs_dir=runs_dir,
            run_id=run_id,
            return_artifacts=True,
            theta_space=theta_space,
            metric=metric_payload,
            extra_meta={
                "optimizer": {"preconditioning": precond_meta},
                "theta": {"labels_by_key": labels_by_key},
                "mc": {
                    "index": run_index,
                    "seed": int(experiment["seed"]),
                    "run_id": run_id,
                    "fast": fast,
                    "add_noise": add_noise,
                    "use_eigen": use_eigen,
                },
            },
        )

        if use_eigen and eigen_map is not None:
            theta_final = eigen_map.theta_from_z(theta_final_opt)
        else:
            theta_final = theta_final_opt

        final_store = store_unpack_params(inference_subspec, theta_final, init_store)
        final_psf = binder.model(binder.strip_structural(final_store))

        loss_init = float(nll_loss_fn(theta0))
        loss_final = float(nll_loss_fn(theta_final))
        improvement_ratio = loss_init / loss_final if loss_final != 0 else float("nan")

        if artifacts is not None:
            run_dir = Path(artifacts["run_dir"]) if artifacts.get("run_dir") else None
            if run_dir is not None:
                truth_dict = {key: truth_store.get(key) for key in infer_keys}
                init_dict = {key: init_store.get(key) for key in infer_keys}
                final_dict = {key: final_store.get(key) for key in infer_keys}
                param_summary = build_param_summary(init_dict, final_dict, truth=truth_dict)
                patch_summary(
                    run_dir,
                    {
                        "param_summary": param_summary,
                        "loss_true": loss_true,
                        "improvement_ratio": improvement_ratio,
                    },
                )
                _maybe_warn_missing_artifacts(run_dir)

        print(
            "Run summary: loss(true)={:.6g}, loss(init)={:.6g}, loss(final)={:.6g}".format(
                loss_true,
                loss_init,
                loss_final,
            )
        )

        if np.isfinite(loss_final):
            success_count += 1
        else:
            failure_count += 1

    print("\nWriting sweep CSV...")
    sweep_count = write_sweep_csv(
        runs_dir=runs_dir,
        out_csv=results_dir / "sweep.csv",
        include_meta_fields=(
            "mc.index",
            "mc.run_id",
            "mc.seed",
            "mc.fast",
            "mc.add_noise",
            "mc.use_eigen",
        ),
    )

    experiment_summary = {
        "created_at": experiment_created_at,
        "n_runs": n_runs,
        "success_count": success_count,
        "failure_count": failure_count,
        "sweep_rows": sweep_count,
        "results_dir": str(results_dir),
    }
    _write_json(results_dir / "experiment_summary.json", experiment_summary)

    t1_script = time.time()
    print("\nExperiment complete.")
    print(f"Success: {success_count}, Failures: {failure_count}")
    print(f"Results directory: {results_dir}")
    print("Script finished in %.3f sec" % (t1_script - t0_script))


def _validate_experiment(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    optimizer_cfg = experiment_cfg.get("optimizer", {})
    outputs_cfg = experiment_cfg.get("outputs", {})
    mc_cfg = experiment_cfg.get("monte_carlo", {})

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
        "n_runs": int(mc_cfg.get("n_runs", N_RUNS)),
        "run_id_prefix": str(mc_cfg.get("run_id_prefix", RUN_ID_PREFIX)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical Monte Carlo astrometry recipe")
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
    parser.add_argument("--n-runs", type=int, default=None, help="Override number of Monte Carlo runs.")
    parser.add_argument("--run-id-prefix", type=str, default=None, help="Prefix for run IDs.")
    parser.add_argument("--add-noise", action="store_true", help="Force noise injection (overrides config).")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot saving (overrides config).")
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
        n_runs=args.n_runs,
        run_id_prefix=args.run_id_prefix,
        add_noise=bool(args.add_noise) if args.add_noise else None,
        save_plots=False if args.no_plots else None,
    )
