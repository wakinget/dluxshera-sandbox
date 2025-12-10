import jax.numpy as jnp
import numpy as np

from dluxshera.inference.inference import run_shera_image_gd_eigen
from dluxshera.optics.config import SheraThreePlaneConfig, SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.core.binder import SheraThreePlaneBinder
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
        "binary.x_position": 0.02,
        "binary.y_position": -0.015,
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
