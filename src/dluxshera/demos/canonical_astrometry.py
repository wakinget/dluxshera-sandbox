"""Canonical three-plane astrometry demo packaged for reuse.

The functions below mirror the notebook-style demo while keeping the logic
explicit and linear. Import the helpers in notebooks or tests, or invoke
:func:`main` from a thin wrapper script under ``examples/``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.inference import run_shera_image_gd_eigen
from dluxshera.inference.optimization import make_binder_image_nll_fn, run_simple_gd
from dluxshera.inference.prior import PriorSpec
from dluxshera.optics.config import SheraThreePlaneConfig
from dluxshera.params.packing import unpack_params as store_unpack_params
from dluxshera.params.spec import (
    ParamKey,
    ParamSpec,
    build_forward_model_spec_from_config,
    build_inference_spec_basic,
)
from dluxshera.params.store import ParameterStore
from dluxshera.plot.plotting import plot_parameter_history_grid, plot_psf_comparison, plot_psf_single

DEFAULT_RESULTS_DIR = Path("Results/CanonicalAstrometryDemo")
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DP_PATH = _PACKAGE_ROOT / "data" / "diffractive_pupil.npy"


@dataclass
class DemoData:
    """Container for key demo artifacts returned to callers and tests."""

    cfg: SheraThreePlaneConfig
    forward_spec: ParamSpec
    inference_spec: ParamSpec
    truth_store: ParameterStore
    binder: Optional[SheraThreePlaneBinder]
    truth_psf: np.ndarray
    variant_psf: np.ndarray
    init_psf: Optional[np.ndarray] = None
    gd_psf: Optional[np.ndarray] = None
    eigen_psf: Optional[np.ndarray] = None


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


def _evaluate_psf(binder: SheraThreePlaneBinder, store: Optional[ParameterStore] = None) -> np.ndarray:
    return np.array(binder.model(store))


def _build_param_history(
    theta_sequence: Sequence[np.ndarray],
    base_store: ParameterStore,
    sub_spec: ParamSpec,
    *,
    zernike_indices: Sequence[int] = (0, 1),
) -> Mapping[str, np.ndarray]:
    keys = (
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.x_position_as",
        "binary.y_position_as",
        "binary.contrast",
    )
    histories = {key: [] for key in keys}
    for idx in zernike_indices:
        histories[f"primary.zernike_coeffs_nm[{idx}]"] = []

    for theta_vec in theta_sequence:
        store = store_unpack_params(sub_spec, theta_vec, base_store)
        for key in keys:
            histories[key].append(np.array(store.get(key)))

        zernikes = np.array(store.get("primary.zernike_coeffs_nm"))
        for idx in zernike_indices:
            if idx < len(zernikes):
                histories[f"primary.zernike_coeffs_nm[{idx}]"].append(zernikes[idx])

    return {k: np.stack(v) for k, v in histories.items()}


def build_system(
    *, fast: bool, rng: np.random.Generator, make_binder: bool = True
) -> tuple[SheraThreePlaneConfig, ParamSpec, ParamSpec, ParameterStore, ParameterStore, Optional[SheraThreePlaneBinder]]:
    print("Step 1: Building SheraThreePlaneConfig and ParamSpecs...")
    pupil_npix = 16 if fast else 256
    psf_npix = 16 if fast else 256
    n_lambda = 1 if fast else 3

    cfg = SheraThreePlaneConfig(
        design_name="shera_testbed_custom",
        pupil_npix=pupil_npix,
        psf_npix=psf_npix,
        oversample=2,
        wavelength_m=550e-9,
        bandwidth_m=110e-9,
        n_lambda=n_lambda,
        m1_diameter_m=0.09,
        m2_diameter_m=0.025,
        m1_focal_length_m=0.35796,
        m2_focal_length_m=-0.041935,
        m1_m2_separation_m=0.320,
        pixel_pitch_m=6.5e-6,
        n_struts=4,
        strut_width_m=0.002,
        strut_rotation_deg=45.0,
        diffractive_pupil_path=str(DEFAULT_DP_PATH),
        dp_design_wavelength_m=550e-9,
        primary_noll_indices=(4, 5, 6, 7, 8, 9, 10, 11),
        secondary_noll_indices=(4, 5, 6, 7, 8, 9, 10, 11),
    )

    forward_spec = build_forward_model_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic()

    print("Step 2: Constructing truth ParameterStore and derived parameters...")
    forward_store = ParameterStore.from_spec_defaults(forward_spec)
    forward_store = forward_store.replace(
        {
            "binary.separation_as": 9.5,
            "binary.position_angle_deg": 80.0,
            "binary.x_position_as": 0.50,
            "binary.y_position_as": -0.25,
            "binary.spectral_flux_density": 1.7e17,
            "imaging.exposure_time_s": 1.0,
            "imaging.throughput": 0.8,
            "binary.contrast": 3.2,
        }
    )

    forward_store = forward_store.refresh_derived(forward_spec)

    truth_store = ParameterStore.from_spec_defaults(inference_spec)
    primary_zernikes = 5.0 * rng.standard_normal(len(cfg.primary_noll_indices))
    secondary_zernikes = np.zeros(len(cfg.secondary_noll_indices))
    truth_store = truth_store.replace(
        {
            "binary.separation_as": forward_store.get("binary.separation_as"),
            "binary.position_angle_deg": forward_store.get("binary.position_angle_deg"),
            "binary.x_position_as": forward_store.get("binary.x_position_as"),
            "binary.y_position_as": forward_store.get("binary.y_position_as"),
            "binary.log_flux_total": forward_store.get("binary.log_flux_total"),
            "binary.contrast": forward_store.get("binary.contrast"),
            "system.plate_scale_as_per_pix": forward_store.get("system.plate_scale_as_per_pix"),
            "primary.zernike_coeffs_nm": primary_zernikes,
            "secondary.zernike_coeffs_nm": secondary_zernikes,
        }
    )

    binder = SheraThreePlaneBinder(cfg, forward_spec, forward_store) if make_binder else None
    return cfg, forward_spec, inference_spec, forward_store, truth_store, binder


def simulate_data(
    binder: SheraThreePlaneBinder,
    forward_store: ParameterStore,
    *,
    add_noise: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    print("Step 3: Generating synthetic PSFs from the refactor stack...")
    truth_psf = _evaluate_psf(binder)
    if add_noise:
        truth_psf = truth_psf + rng.normal(scale=0.005, size=truth_psf.shape)

    variant_delta = ParameterStore.from_dict(
        {
            "binary.separation_as": forward_store.get("binary.separation_as") + 1.0,
            "primary.zernike_coeffs_nm": np.asarray(forward_store.get("primary.zernike_coeffs_nm"))
            + 2.0,
        }
    )
    variant_psf = _evaluate_psf(binder, variant_delta)
    return truth_psf, variant_psf


def make_inference_setup(
    *,
    truth_store: ParameterStore,
    forward_store: ParameterStore,
    forward_spec: ParamSpec,
    binder: SheraThreePlaneBinder,
    truth_psf: np.ndarray,
    rng: np.random.Generator,
    fast: bool,
) -> tuple[Tuple[ParamKey, ...], ParameterStore, ParamSpec, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], np.ndarray]:
    print("Step 4: Defining inference keys and tight priors around the truth...")
    infer_keys: Tuple[ParamKey, ...] = (
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.x_position_as",
        "binary.y_position_as",
        "binary.log_flux_total",
        "binary.contrast",
        "system.plate_scale_as_per_pix",
        "primary.zernike_coeffs_nm",
    )
    priors: Mapping[ParamKey, object] = {
        "binary.separation_as": 0.01,
        "binary.position_angle_deg": 0.5,
        "binary.x_position_as": 0.005,
        "binary.y_position_as": 0.005,
        "binary.log_flux_total": 0.05,
        "binary.contrast": 0.05,
        "system.plate_scale_as_per_pix": 0.002,
        "primary.zernike_coeffs_nm": np.full_like(
            truth_store.get("primary.zernike_coeffs_nm"), 1.0
        ),
    }
    prior_spec = PriorSpec.from_sigmas(truth_store, priors)

    print("Step 5: Initialising inference store with prior-perturbed truth values...")
    jitter_key = jax.random.PRNGKey(rng.integers(0, 1_000_000))
    init_store = prior_spec.sample_near(truth_store, jitter_key, keys=infer_keys)
    updates = {key: init_store.get(key) for key in infer_keys}
    init_forward_store = forward_store.replace(updates)
    init_psf = _evaluate_psf(binder, init_forward_store)

    var_image = np.ones_like(truth_psf) * 0.01
    sub_spec = forward_spec.subset(infer_keys)

    print("Step 6: Building loss and running binder/SystemGraph-based gradient descent...")
    loss_nll, theta0 = make_binder_image_nll_fn(
        cfg=binder.cfg,
        forward_spec=forward_spec,
        base_forward_store=init_forward_store,
        infer_keys=infer_keys,
        data=truth_psf,
        var=var_image,
        noise_model="gaussian",
        reduce="sum",
        use_system_graph=not fast,
    )

    def gaussian_prior_penalty(store_theta: ParameterStore) -> jnp.ndarray:
        return prior_spec.quadratic_penalty(store_theta, center_store=truth_store, keys=infer_keys)

    def loss_with_prior(theta: np.ndarray) -> np.ndarray:
        store_theta = store_unpack_params(sub_spec, theta, init_forward_store)
        data_term = loss_nll(theta)
        prior_term = gaussian_prior_penalty(store_theta)
        return jnp.asarray(data_term).sum() + jnp.asarray(prior_term).sum()

    return infer_keys, init_store, sub_spec, var_image, init_psf, loss_with_prior, theta0


def run_gradient_descent(
    *,
    loss_fn,
    theta0: np.ndarray,
    sub_spec: ParamSpec,
    init_store: ParameterStore,
    binder: SheraThreePlaneBinder,
    infer_keys: Sequence[ParamKey],
    fast: bool,
) -> tuple[ParameterStore, Mapping[str, np.ndarray], np.ndarray, np.ndarray]:
    if fast:
        theta_history = [np.array(theta0)]
        param_history_gd = _build_param_history(theta_history, init_store, sub_spec)
        gd_psf = _evaluate_psf(binder, init_store)
        return init_store, param_history_gd, gd_psf, np.array([])

    theta_final, history = run_simple_gd(
        loss_fn,
        theta0,
        learning_rate=1e-2,
        num_steps=120,
    )
    final_store = store_unpack_params(sub_spec, theta_final, init_store)

    theta_history = [np.array(theta0)]
    if "theta" in history:
        theta_history.extend([np.array(t) for t in history["theta"]])
    param_history_gd = _build_param_history(theta_history, init_store, sub_spec)
    gd_psf = _evaluate_psf(binder, final_store)
    return final_store, param_history_gd, gd_psf, np.array(history.get("loss", []))


def run_eigen_optimization(
    *,
    fast: bool,
    cfg: SheraThreePlaneConfig,
    inference_spec: ParamSpec,
    base_store: ParameterStore,
    truth_store: ParameterStore,
    gd_store: ParameterStore,
    infer_keys: Sequence[ParamKey],
    truth_psf: np.ndarray,
    var_image: np.ndarray,
    theta0: np.ndarray,
    binder: SheraThreePlaneBinder,
) -> tuple[Optional[ParameterStore], Optional[Mapping[str, np.ndarray]], Optional[np.ndarray]]:
    print("Step 8: Computing curvature/FIM and building EigenThetaMap...")
    if fast:
        return None, None, None

    eigen_results = run_shera_image_gd_eigen(
        cfg=cfg,
        inference_spec=inference_spec,
        base_store=base_store,
        infer_keys=infer_keys,
        data=truth_psf,
        var=var_image,
        noise_model="gaussian",
        num_steps=60,
        learning_rate=None,
        truncate=None,
        whiten=True,
        theta_ref=theta0,
        use_system_graph=True,
    )

    sub_spec = inference_spec.subset(infer_keys)
    eigen_store = store_unpack_params(sub_spec, eigen_results.theta_final, base_store)
    theta_history_eigen = [np.array(theta0)]
    if eigen_results.theta_history.size:
        theta_history_eigen.extend([np.array(t) for t in eigen_results.theta_history])
    param_history_eigen = _build_param_history(theta_history_eigen, base_store, sub_spec)
    eigen_psf = _evaluate_psf(binder, eigen_store)

    print("Step 9: Running gradient descent in eigenmode coordinates...")
    print("  -> initial z-norm:", float(np.linalg.norm(np.array(eigen_results.z_history[0]))))
    print("  -> final z-norm:", float(np.linalg.norm(np.array(eigen_results.z_final))))

    print("Step 10: Comparing pure-θ vs eigenmode recovered parameters...")
    comparison = format_parameter_summary(infer_keys, truth_store, gd_store, eigen_store)
    print(comparison)

    return eigen_store, param_history_eigen, eigen_psf


def plot_results(
    *,
    output_dir: Path,
    make_plots: bool,
    truth_psf: np.ndarray,
    init_psf: np.ndarray,
    gd_psf: np.ndarray,
    eigen_psf: Optional[np.ndarray],
    var_image: np.ndarray,
    truth_store: ParameterStore,
    param_history_gd: Mapping[str, np.ndarray],
    param_history_eigen: Optional[Mapping[str, np.ndarray]],
):
    if not make_plots:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_psf_single(
        truth_psf,
        title="Truth PSF",
        save_path=output_dir / "psf_truth.png",
    )
    plot_psf_comparison(
        data=truth_psf,
        model=init_psf,
        var=var_image,
        model_label="Initial Model",
        save_path=output_dir / "psf_comparison_init.png",
    )
    plot_psf_comparison(
        data=truth_psf,
        model=gd_psf,
        var=var_image,
        model_label="Pure θ Model",
        save_path=output_dir / "psf_comparison_gd.png",
    )
    if eigen_psf is not None:
        plot_psf_comparison(
            data=truth_psf,
            model=eigen_psf,
            var=var_image,
            model_label="Eigen Model",
            save_path=output_dir / "psf_comparison_eigen.png",
        )

    true_values = {
        key: truth_store.get(key)
        for key in (
            "binary.separation_as",
            "binary.position_angle_deg",
            "binary.x_position_as",
            "binary.y_position_as",
            "binary.contrast",
        )
    }
    true_values.update(
        {
            f"primary.zernike_coeffs_nm[{i}]": truth_store.get("primary.zernike_coeffs_nm")[i]
            for i in (0, 1)
        }
    )

    plot_parameter_history_grid(
        param_history_gd,
        true_vals=true_values,
        save_path=output_dir / "parameter_history_gd.png",
    )
    if param_history_eigen is not None:
        plot_parameter_history_grid(
            param_history_eigen,
            true_vals=true_values,
            save_path=output_dir / "parameter_history_eigen.png",
        )


def main(
    fast: bool = False,
    save_plots: bool = False,
    add_noise: bool = False,
    save_plots_dir: Optional[Path] = None,
) -> DemoData:
    if fast:
        jax.config.update("jax_disable_jit", True)

    rng = np.random.default_rng(0)
    make_plots = save_plots or save_plots_dir is not None
    output_dir = Path(save_plots_dir) if save_plots_dir is not None else DEFAULT_RESULTS_DIR

    cfg, forward_spec, inference_spec, forward_store, truth_store, binder = build_system(
        fast=fast, rng=rng, make_binder=not fast
    )

    if fast:
        truth_psf = rng.random((8, 8))
        variant_psf = truth_psf + rng.normal(scale=0.01, size=truth_psf.shape)
        init_psf = truth_psf.copy()
        var_image = np.ones_like(truth_psf) * 0.01
        param_history_gd = {
            "binary.separation_as": np.array([truth_store.get("binary.separation_as")]),
            "binary.position_angle_deg": np.array([truth_store.get("binary.position_angle_deg")]),
            "binary.x_position_as": np.array([truth_store.get("binary.x_position_as")]),
            "binary.y_position_as": np.array([truth_store.get("binary.y_position_as")]),
            "binary.contrast": np.array([truth_store.get("binary.contrast")]),
            "primary.zernike_coeffs_nm[0]": np.array(
                [truth_store.get("primary.zernike_coeffs_nm")[0]]
            ),
            "primary.zernike_coeffs_nm[1]": np.array(
                [truth_store.get("primary.zernike_coeffs_nm")[1]]
            ),
        }

        plot_results(
            output_dir=output_dir,
            make_plots=make_plots,
            truth_psf=truth_psf,
            init_psf=init_psf,
            gd_psf=init_psf,
            eigen_psf=None,
            var_image=var_image,
            truth_store=truth_store,
            param_history_gd=param_history_gd,
            param_history_eigen=None,
        )

        return DemoData(
            cfg=cfg,
            forward_spec=forward_spec,
            inference_spec=inference_spec,
            truth_store=truth_store,
            binder=binder,
            truth_psf=truth_psf,
            variant_psf=variant_psf,
            init_psf=init_psf,
            gd_psf=init_psf,
            eigen_psf=None,
        )

    truth_psf, variant_psf = simulate_data(binder, forward_store, add_noise=add_noise, rng=rng)

    (
        infer_keys,
        init_store,
        sub_spec,
        var_image,
        init_psf,
        loss_with_prior,
        theta0,
    ) = make_inference_setup(
        truth_store=truth_store,
        forward_store=forward_store,
        forward_spec=forward_spec,
        binder=binder,
        truth_psf=truth_psf,
        rng=rng,
        fast=fast,
    )

    final_store, param_history_gd, gd_psf, loss_history = run_gradient_descent(
        loss_fn=loss_with_prior,
        theta0=theta0,
        sub_spec=sub_spec,
        init_store=init_store,
        binder=binder,
        infer_keys=infer_keys,
        fast=fast,
    )

    print("Step 7: Summarising recovered parameters and plotting (optional)...")
    summary = format_parameter_summary(infer_keys, truth_store, init_store, final_store)
    print(summary)
    print("Loss curve (first 5 steps):", loss_history[:5])

    eigen_store, param_history_eigen, eigen_psf = run_eigen_optimization(
        fast=fast,
        cfg=cfg,
        inference_spec=inference_spec,
        base_store=init_store,
        truth_store=truth_store,
        gd_store=final_store,
        infer_keys=infer_keys,
        truth_psf=truth_psf,
        var_image=var_image,
        theta0=theta0,
        binder=binder,
    )

    plot_results(
        output_dir=output_dir,
        make_plots=make_plots,
        truth_psf=truth_psf,
        init_psf=init_psf,
        gd_psf=gd_psf,
        eigen_psf=eigen_psf,
        var_image=var_image,
        truth_store=truth_store,
        param_history_gd=param_history_gd,
        param_history_eigen=param_history_eigen,
    )

    return DemoData(
        cfg=cfg,
        forward_spec=forward_spec,
        inference_spec=inference_spec,
        truth_store=truth_store,
        binder=binder,
        truth_psf=truth_psf,
        variant_psf=variant_psf,
        init_psf=init_psf,
        gd_psf=gd_psf,
        eigen_psf=eigen_psf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the canonical astrometry demo")
    parser.add_argument("--fast", action="store_true", help="Use fewer optimisation steps")
    parser.add_argument("--save-plots", action="store_true", help="Save demo plots")
    parser.add_argument("--save-plots-dir", type=Path, default=None, help="Directory to save plots (implies --save-plots)")
    parser.add_argument("--add-noise", action="store_true", help="Add Gaussian noise to the synthetic PSF")
    args = parser.parse_args()

    main(
        fast=args.fast,
        save_plots=args.save_plots or args.save_plots_dir is not None,
        add_noise=args.add_noise,
        save_plots_dir=args.save_plots_dir,
    )
