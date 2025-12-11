# tests/test_image_nll_bridge.py
import jax
import jax.numpy as np

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.core.builder import build_shera_threeplane_model
from dluxshera.inference.optimization import make_image_nll_fn, run_image_gd


def _make_store_for_smoke(cfg):
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    updates = {
        "binary.separation_as": 10.0,
        "binary.position_angle_deg": 90.0,
        "binary.x_position_as": 0.0,
        "binary.y_position_as": 0.0,
        "binary.contrast": 3.0,
        "binary.log_flux_total": 8.0,
        "system.plate_scale_as_per_pix": 0.355,
    }

    n_m1 = len(cfg.primary_noll_indices)
    n_m2 = len(cfg.secondary_noll_indices)
    if n_m1 > 0:
        updates["primary.zernike_coeffs"] = np.zeros(n_m1)
    if n_m2 > 0:
        updates["secondary.zernike_coeffs"] = np.zeros(n_m2)

    store = store.replace(updates)
    return spec, store


def test_make_image_nll_fn_smoke_gaussian():
    cfg = SHERA_TESTBED_CONFIG
    spec, store = _make_store_for_smoke(cfg)

    # synthetic data from the same model
    model = build_shera_threeplane_model(cfg, spec, store)
    data = model.model()
    var = np.ones_like(data)

    infer_keys = [
        "binary.separation_as",
        "binary.x_position_as",
        "binary.y_position_as",
    ]

    loss_fn, theta0 = make_image_nll_fn(
        cfg,
        spec,
        store,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
        build_model_fn=build_shera_threeplane_model,
    )

    loss0 = loss_fn(theta0)
    assert np.isfinite(loss0)

    grad_fn = jax.grad(loss_fn)
    g0 = grad_fn(theta0)
    assert g0.shape == theta0.shape


def test_make_binder_image_nll_fn_smoke_gaussian():
    cfg = SHERA_TESTBED_CONFIG
    spec, store = _make_store_for_smoke(cfg)

    # Synthetic data from the same config/store, but via the Binder path
    from dluxshera.core.binder import SheraThreePlaneBinder
    binder = SheraThreePlaneBinder(cfg, spec, store)
    data = binder.forward(store)
    var = np.ones_like(data)

    infer_keys = [
        "binary.separation_as",
        "binary.x_position_as",
        "binary.y_position_as",
    ]

    from dluxshera.inference.optimization import make_binder_image_nll_fn

    loss_fn, theta0 = make_binder_image_nll_fn(
        cfg,
        spec,
        store,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
    )

    loss0 = loss_fn(theta0)
    assert np.isfinite(loss0)

    g0 = jax.grad(loss_fn)(theta0)
    assert g0.shape == theta0.shape


def test_run_image_gd_separation_smoke():
    cfg = SHERA_TESTBED_CONFIG
    spec, store_true = _make_store_for_smoke(cfg)

    # 1) Generate synthetic data from the "truth" store
    model_true = build_shera_threeplane_model(cfg, spec, store_true)
    data = model_true.model()
    var = np.ones_like(data)

    # 2) Start from a slightly wrong separation
    sep_true = store_true.get("binary.separation_as")
    store_init = store_true.replace({"binary.separation_as": sep_true * 1.1})

    infer_keys = ["binary.separation_as"]

    theta_final, store_final, history = run_image_gd(
        cfg,
        spec,
        store_init,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
        learning_rate=1e-1,
        num_steps=20,
    )

    # Loss should go down
    assert float(history["loss"][-1]) < float(history["loss"][0])

    # Separation should move closer to the truth
    sep_init = store_init.get("binary.separation_as")
    sep_est = store_final.get("binary.separation_as")
    assert abs(sep_est - sep_true) < abs(sep_init - sep_true)