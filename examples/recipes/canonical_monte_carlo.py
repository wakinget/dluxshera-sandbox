"""
Canonical astrometry Monte Carlo (multi-start) recipe.

This script mirrors ``examples/recipes/canonical_astrometry.py`` but runs a
multi-start Monte Carlo experiment: the synthetic data and priors are fixed
once, and each run draws a new random initialization from the same priors.

The primary goal is to demonstrate reproducible artifact layouts across many
runs (``runs/<run_id>/...``) while keeping the canonical forward-modeling steps
explicit and easy to follow.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

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
from dluxshera.params.spec import build_inference_spec_basic, make_inference_subspec
from dluxshera.params.store import ParameterStore, strip_structural
from dluxshera.plot.plotting import (
    apply_plot_defaults,
    get_default_cmaps,
    plot_fim,
)
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
SAVE_PLOTS = False

# Monte Carlo settings
N_RUNS = 5
N_ITER = 60
FAST_ITER = 30
BASE_LR = 0.5
RUN_ID_PREFIX = "mc"

# Telescope Config Selection (9cm testbed vs 22cm flight design)
# Options: None, SHERA_TESTBED_CONFIG / SHERA_FLIGHT_CONFIG
CONFIG = SHERA_TESTBED_CONFIG

# Eigenmode settings
USE_EIGEN = True           # Enables re-parameterization
WHITEN_BASIS = True        # If True, scales each eigenvector by 1/sqrt(lambda)
TRUNCATE_K = None          # int or None; keep top-k eigenmodes when set
TRUNCATE_BY_EIGVAL = None  # float or None; only used when TRUNCATE_K is None

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

REPO_ROOT = Path(__file__).resolve().parents[2]


# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
def _timestamp_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, jnp.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    return value


# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
def _mc_run_id(index: int, prefix: str = RUN_ID_PREFIX) -> str:
    return f"{prefix}_{index:04d}"


# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
def _repo_relative_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, REPO_ROOT)).as_posix()


# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
def _summarize_prior_info(prior_info: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, entry in prior_info.items():
        sigma = entry.get("sigma") if isinstance(entry, Mapping) else None
        if isinstance(sigma, np.ndarray):
            sigma_summary = {"shape": list(sigma.shape), "dtype": str(sigma.dtype)}
        elif isinstance(sigma, jnp.ndarray):
            sigma_summary = {"shape": list(sigma.shape), "dtype": str(sigma.dtype)}
        else:
            sigma_summary = sigma
        summary[key] = {"dist": entry.get("dist"), "sigma": sigma_summary}
    return summary


# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
def _maybe_warn_missing_artifacts(run_dir: Path) -> None:
    required = ["meta.json", "summary.json", "trace.npz"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        print(
            f"WARNING: run artifacts missing in {run_dir}: {', '.join(missing)}"
        )


# NOTE: Local helper (candidate for future migration to dluxshera.inference.*)
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
    return {
        "commit": commit,
        "status": status,
    }


def main(
    *,
    config: SheraThreePlaneConfig | None = CONFIG,
    fast: bool = FAST_MODE,
    save_plots: bool = SAVE_PLOTS,
    add_noise: bool = ADD_NOISE,
    results_dir: Path | None = None,
    n_runs: int = N_RUNS,
    rng_seed: int = RNG_SEED,
    use_eigen: bool = USE_EIGEN,
    whiten_basis: bool = WHITEN_BASIS,
    truncate_k: int | None = TRUNCATE_K,
    truncate_by_eigval: float | None = TRUNCATE_BY_EIGVAL,
) -> None:
    """Run the canonical Monte Carlo astrometry recipe."""
    jax.config.update("jax_enable_x64", JAX_ENABLE_X64)

    rng_key = jr.PRNGKey(rng_seed)

    results_dir = results_dir or (
        REPO_ROOT
        / "Results"
        / "canonical_monte_carlo"
        / _timestamp_tag()
    )
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

    cfg = config or SHERA_TESTBED_CONFIG
    cfg = cfg.replace(
        primary_noll_indices=tuple(range(4, 12)),
        secondary_noll_indices=tuple(range(4, 12)),
    )
    if fast:
        cfg = cfg.replace(
            n_lambda=1,
            primary_noll_indices=tuple(range(4, 9)),
            secondary_noll_indices=tuple(range(4, 9)),
        )

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

    np.savez(results_dir / "data.npz", data=np.asarray(data), data_var=np.asarray(data_var))

    print("Configuring Inference...")
    inference_subspec = make_inference_subspec(
        base_spec=inference_spec,
        infer_keys=INFER_KEYS,
        cfg=cfg,
    )

    prior_info = {
        "binary.separation_as": {"sigma": 1e-4, "dist": "Normal"},
        "binary.position_angle_deg": {"sigma": 1e-3, "dist": "Uniform"},
        "binary.x_position_as": {"sigma": 1e-3, "dist": "Normal"},
        "binary.y_position_as": {"sigma": 1e-3, "dist": "Normal"},
        "binary.log_flux_total": {"sigma": 1e-3, "dist": "LogNormal"},
        "binary.contrast": {"sigma": 1e-3, "dist": "LogNormal"},
        "system.plate_scale_as_per_pix": {"sigma": 1e-5, "dist": "LogNormal"},
        "primary.zernike_coeffs_nm": {
            "sigma": np.full_like(truth_store.get("primary.zernike_coeffs_nm"), 5),
            "dist": "Normal",
        },
        "secondary.zernike_coeffs_nm": {
            "sigma": np.full_like(truth_store.get("secondary.zernike_coeffs_nm"), 5),
            "dist": "Normal",
        },
    }
    prior_spec = PriorSpec.from_info(truth_store, prior_info)

    experiment_created_at = _now_iso_local_ms()
    config_payload = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg
    if isinstance(config_payload, dict) and "diffractive_pupil_path" in config_payload:
        config_payload = {
            **config_payload,
            "diffractive_pupil_path": _repo_relative_path(
                config_payload.get("diffractive_pupil_path")
            ),
        }

    manifest = {
        "script": "canonical_monte_carlo.py",
        "git": _git_info(),
        "created_at": experiment_created_at,
        "n_runs": n_runs,
        "rng_seed": rng_seed,
        "fast": fast,
        "add_noise": add_noise,
        "use_eigen": use_eigen,
        "whiten_basis": whiten_basis,
        "truncate_k": truncate_k,
        "truncate_by_eigval": truncate_by_eigval,
        "infer_keys": list(INFER_KEYS),
        "prior_info": _summarize_prior_info(prior_info),
        "diffractive_pupil_path": _repo_relative_path(cfg.diffractive_pupil_path),
        "config": config_payload,
    }
    _write_json(results_dir / "manifest.json", _coerce_jsonable(manifest))

    _ = get_default_cmaps()
    apply_plot_defaults()
    plt.rcParams["image.cmap"] = "inferno_nan"

    print("Building the loss function...")
    nll_loss_fn, _ = make_binder_nll_fn(
        binder=binder,
        infer_keys=INFER_KEYS,
        data=data,
        var=data_var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=truth_store,
    )
    fim_labels = generate_fim_labels(INFER_KEYS, cfg=cfg, store=truth_store)

    loss_fn = nll_loss_fn

    theta_true = pack_params(inference_subspec, truth_store)
    loss_true = float(loss_fn(theta_true))

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
        eigen_map = None
        theta_space = "primitive"
        precond_meta_base = {"method": "fim_diag"}

    print(
        "FIM diag: min={:.3e}, max={:.3e}".format(
            float(jnp.min(fim_diag)),
            float(jnp.max(fim_diag)),
        )
    )

    n_iter = FAST_ITER if fast else N_ITER

    print(f"\nRunning {n_runs} Monte Carlo optimizations...")

    success_count = 0
    failure_count = 0

    for run_index in range(n_runs):
        rng_key, split_key = jr.split(rng_key)
        run_id = _mc_run_id(run_index)
        print(f"\n--- Run {run_index + 1}/{n_runs} ({run_id}) ---")

        init_store = prior_spec.sample(rng_key=split_key, keys=INFER_KEYS)
        init_psf = binder.model(
            strip_structural(init_store, structural_keys=binder.structural_store_keys())
        )

        _, theta0 = make_binder_nll_fn(
            binder=binder,
            infer_keys=INFER_KEYS,
            data=data,
            var=data_var,
            noise_model="gaussian",
            reduce="sum",
            theta0_store=init_store,
        )

        if use_eigen:
            if truncate_k is not None and truncate_by_eigval is not None:
                print(
                    "truncate_k is set; ignoring truncate_by_eigval="
                    f"{truncate_by_eigval}."
                )

            # NOTE: theta_ref is the origin for the eigen coefficients (z).
            # Truncation zeroes discarded components *relative to theta_ref*.
            # If we set theta_ref to the truth, truncation snaps discarded
            # directions back to truth and makes severe truncation look
            # unrealistically powerful. Using the initial guess freezes
            # discarded directions at their initial offsets.
            theta_ref = theta0
            eigen_map_full = EigenThetaMap.from_fim(
                F,
                theta_ref,
                whiten=whiten_basis,
            )
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
        else:
            eigen_map = None

        if use_eigen and eigen_map is not None:
            z0 = eigen_map.z_from_theta(theta0)
            eigvals_kept = (
                np.asarray(eigen_map.eigvals)
                if eigen_map.eigvals is not None
                else np.array([])
            )
            if whiten_basis:
                lr_vec = np.ones_like(z0)
                curvature_vec = np.ones_like(z0)
            else:
                lr_vec = 1.0 / (eigvals_kept + 1e-12)
                curvature_vec = eigvals_kept

            index_map = build_eigen_index_map(eigen_map)
            loss_opt = lambda z: loss_fn(eigen_map.theta_from_z(z))
            theta0_opt = z0
            metric_payload = {
                "theta_ref": np.asarray(theta0_opt),
                "metric_diag": np.asarray(curvature_vec),
                "lr_scale": np.asarray(lr_vec),
            }
            precond_meta = {
                **precond_meta_base,
                "lr_vec": np.asarray(lr_vec),
            }
        else:
            index_map = build_index_map(inference_subspec, init_store, theta=theta0)
            lr_vec = 1.0 / (np.asarray(fim_diag) + 1e-12)
            curvature_vec = fim_diag
            loss_opt = loss_fn
            theta0_opt = theta0
            metric_payload = {
                "theta_ref": np.asarray(theta0_opt),
                "metric_diag": np.asarray(curvature_vec),
                "lr_scale": np.asarray(lr_vec),
            }
            precond_meta = {
                **precond_meta_base,
                "lr_vec": np.asarray(lr_vec),
            }

        labels_by_key = map_labels_to_keys(
            INFER_KEYS,
            fim_labels,
            store=init_store if use_eigen else None,
            index_map=None if use_eigen else index_map,
        )

        print("Running preconditioned gradient descent...")
        theta_final_opt, trace, artifacts = run_shera_gd(
            loss_fn=loss_opt,
            theta0=theta0_opt,
            index_map=index_map,
            learning_rate=BASE_LR,
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
                    "seed": int(rng_seed),
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
        final_psf = binder.model(
            strip_structural(final_store, structural_keys=binder.structural_store_keys())
        )

        loss_init = float(loss_fn(theta0))
        loss_final = float(loss_fn(theta_final))
        improvement_ratio = loss_init / loss_final if loss_final != 0 else float("nan")

        if artifacts is not None:
            run_dir = Path(artifacts["run_dir"]) if artifacts.get("run_dir") else None
            if run_dir is not None:
                truth_dict = {key: truth_store.get(key) for key in INFER_KEYS}
                init_dict = {key: init_store.get(key) for key in INFER_KEYS}
                final_dict = {key: final_store.get(key) for key in INFER_KEYS}
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


if __name__ == "__main__":
    main()
