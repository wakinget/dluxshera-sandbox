# tests/test_image_nll_bridge.py
import jax
import jax.numpy as np
import pytest

from dluxshera.inference.optimization import run_image_gd


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


@pytest.mark.slow
def test_run_image_gd_separation_smoke(
    shera_smoke_cfg,
    shera_smoke_forward,
    shera_smoke_binder_data,
):
    forward_spec, store_true = shera_smoke_forward
    _, data, var = shera_smoke_binder_data

    # 2) Start from a slightly wrong separation
    sep_true = store_true.get("source.separation_as")
    store_init = store_true.replace({"source.separation_as": sep_true * 1.1})

    infer_keys = ["source.separation_as"]

    theta_final, store_final, history = run_image_gd(
        shera_smoke_cfg,
        forward_spec,
        store_init,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
        learning_rate=1e-1,
        num_steps=10,
    )

    # Loss should go down
    assert float(history["loss"][-1]) < float(history["loss"][0])

    # Separation should move closer to the truth
    sep_init = store_init.get("source.separation_as")
    sep_est = store_final.get("source.separation_as")
    assert abs(sep_est - sep_true) < abs(sep_init - sep_true)
