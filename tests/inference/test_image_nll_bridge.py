# tests/test_image_nll_bridge.py
import jax
import jax.numpy as np

from dluxshera.inference.optimization import make_image_nll_fn, run_image_gd


def test_make_image_nll_fn_smoke_gaussian(
    shera_smoke_cfg,
    shera_smoke_inference,
    shera_smoke_model_data,
    shera_smoke_infer_keys,
):
    inference_spec, inference_store = shera_smoke_inference
    data, var = shera_smoke_model_data

    loss_fn, theta0 = make_image_nll_fn(
        shera_smoke_cfg,
        inference_spec,
        inference_store,
        shera_smoke_infer_keys,
        data,
        var,
        noise_model="gaussian",
        build_model_fn=None,
    )

    loss0 = loss_fn(theta0)
    assert np.isfinite(loss0)

    grad_fn = jax.grad(loss_fn)
    g0 = grad_fn(theta0)
    assert g0.shape == theta0.shape


def test_make_binder_image_nll_fn_smoke_gaussian(
    shera_smoke_cfg,
    shera_smoke_forward,
    shera_smoke_binder_data,
    shera_smoke_infer_keys,
):
    forward_spec, forward_store = shera_smoke_forward
    binder, data, var = shera_smoke_binder_data

    from dluxshera.inference.optimization import make_binder_image_nll_fn

    loss_fn, theta0 = make_binder_image_nll_fn(
        shera_smoke_cfg,
        forward_spec,
        forward_store,
        shera_smoke_infer_keys,
        data,
        var,
        noise_model="gaussian",
        binder=binder,
    )

    loss0 = loss_fn(theta0)
    assert np.isfinite(loss0)

    g0 = jax.grad(loss_fn)(theta0)
    assert g0.shape == theta0.shape


def test_run_image_gd_separation_smoke(
    shera_smoke_cfg,
    shera_smoke_forward,
    shera_smoke_model_data,
):
    forward_spec, store_true = shera_smoke_forward
    data, var = shera_smoke_model_data

    # 2) Start from a slightly wrong separation
    sep_true = store_true.get("binary.separation_as")
    store_init = store_true.replace({"binary.separation_as": sep_true * 1.1})

    infer_keys = ["binary.separation_as"]

    theta_final, store_final, history = run_image_gd(
        shera_smoke_cfg,
        forward_spec,
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
