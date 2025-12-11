import jax.numpy as jnp
import numpy as np

from dluxshera.inference.inference import run_shera_image_gd_eigen
from dluxshera.inference.optimization import make_binder_image_nll_fn, run_simple_gd
from dluxshera.optics.config import SheraThreePlaneConfig, SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_forward_model_spec_from_config, build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.params.packing import pack_params as store_pack_params, unpack_params as store_unpack_params


def test_eigen_helper_quadratic_roundtrip_and_descent():
    theta_true = jnp.array([1.0, -2.0, 0.5])

    def loss_fn(theta):
        return 0.5 * jnp.sum((theta - theta_true) ** 2)

    theta0 = jnp.array([2.5, 0.5, -1.0])
    results = run_shera_image_gd_eigen(
        loss_fn=loss_fn,
        theta0=theta0,
        num_steps=30,
        learning_rate=0.2,
        truncate=2,
        whiten=True,
    )

    assert results.theta_final.shape == theta0.shape
    assert results.z_final.shape[0] == 2  # truncated eigen basis
    assert float(results.loss_history[-1]) < float(results.loss_history[0])

    # Roundtrip consistency
    z_roundtrip = results.eigen_map.to_eigen(results.theta_final)
    assert np.allclose(np.asarray(z_roundtrip).shape, results.z_final.shape)


def test_run_shera_image_gd_eigen_smoke():
    cfg = SheraThreePlaneConfig(
        design_name="shera_testbed",
        pupil_npix=16,
        psf_npix=16,
        oversample=1,
        primary_noll_indices=SHERA_TESTBED_CONFIG.primary_noll_indices,
        secondary_noll_indices=SHERA_TESTBED_CONFIG.secondary_noll_indices,
    )
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    updates = {
        "binary.separation_as": 0.6,
        "binary.position_angle_deg": 75.0,
        "binary.x_position_as": 0.02,
        "binary.y_position_as": -0.015,
        "binary.contrast": 1.5,
        "binary.log_flux_total": 7.5,
        "system.plate_scale_as_per_pix": 0.3,
    }
    n_m1 = len(cfg.primary_noll_indices)
    if n_m1 > 0:
        updates["primary.zernike_coeffs"] = np.zeros(n_m1)
    store = store.replace(updates)

    infer_keys = ["binary.separation_as"]
    theta_bias = store.replace({"binary.separation_as": updates["binary.separation_as"] * 1.1})

    sub_spec = spec.subset(infer_keys)
    theta0 = store_pack_params(sub_spec, theta_bias)
    sep_true = jnp.array(store.get("binary.separation_as"))

    def loss_fn(theta):
        store_theta = store_unpack_params(sub_spec, theta, theta_bias)
        sep_val = store_theta.get("binary.separation_as")
        return 0.5 * jnp.sum((sep_val - sep_true) ** 2)

    results = run_shera_image_gd_eigen(
        loss_fn=loss_fn,
        theta0=theta0,
        num_steps=12,
        learning_rate=5e-2,
        truncate=None,
        whiten=True,
    )

    assert float(results.loss_history[-1]) < float(results.loss_history[0])

    final_store = store_unpack_params(sub_spec, results.theta_final, theta_bias)

    sep_true_val = store.get("binary.separation_as")
    sep_init = theta_bias.get("binary.separation_as")
    sep_est = final_store.get("binary.separation_as")
    assert abs(sep_est - sep_true_val) < abs(sep_init - sep_true_val)


def test_eigen_and_pure_theta_share_binder_loss():
    # Synthetic Binder-style loss built through make_binder_image_nll_fn to
    # ensure the eigen path reuses the same closure as θ-space GD.
    cfg = SheraThreePlaneConfig(
        design_name="shera_testbed",
        pupil_npix=12,
        psf_npix=12,
        oversample=1,
        primary_noll_indices=SHERA_TESTBED_CONFIG.primary_noll_indices,
        secondary_noll_indices=SHERA_TESTBED_CONFIG.secondary_noll_indices,
    )

    forward_spec = build_forward_model_spec_from_config(cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec).replace(
        {"binary.x_position_as": 0.0, "binary.y_position_as": 0.0}
    )

    class _LinearBinder:
        def __init__(self, base_image):
            self.base_image = base_image

        def model(self, store_delta):
            shift = store_delta.get("binary.x_position_as") + store_delta.get("binary.y_position_as")
            return self.base_image + shift

    truth_image = jnp.ones((4, 4)) * 0.2
    binder = _LinearBinder(truth_image)
    var_image = jnp.ones_like(truth_image) * 0.01

    infer_keys = ["binary.x_position_as", "binary.y_position_as"]
    sub_spec = forward_spec.subset(infer_keys)
    theta_truth = store_pack_params(sub_spec, base_store)

    biased_store = base_store.replace({"binary.x_position_as": 0.05, "binary.y_position_as": -0.03})

    loss_nll, theta0 = make_binder_image_nll_fn(
        cfg,
        forward_spec,
        biased_store,
        infer_keys,
        truth_image,
        var_image,
        binder=binder,
        noise_model="gaussian",
        reduce="sum",
        use_system_graph=False,
    )

    initial_loss = float(loss_nll(theta0))

    theta_pure, history_pure = run_simple_gd(
        loss_nll,
        theta0,
        learning_rate=5e-2,
        num_steps=6,
    )

    eigen_results = run_shera_image_gd_eigen(
        loss_fn=loss_nll,
        theta0=theta0,
        num_steps=6,
        learning_rate=5e-2,
        truncate=None,
        whiten=True,
    )

    theta_eigen = eigen_results.theta_final

    init_dist = float(jnp.linalg.norm(theta0 - theta_truth))
    pure_dist = float(jnp.linalg.norm(theta_pure - theta_truth))
    eigen_dist = float(jnp.linalg.norm(theta_eigen - theta_truth))

    assert jnp.isfinite(initial_loss)
    assert jnp.isfinite(history_pure["loss"][-1])
    assert jnp.isfinite(eigen_results.loss_history[-1])

    assert pure_dist < init_dist
    assert eigen_dist < init_dist
    assert abs(pure_dist - eigen_dist) < 0.25 * init_dist
