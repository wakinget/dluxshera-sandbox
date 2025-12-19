from __future__ import annotations

import jax
import jax.numpy as jnp

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.optimization import make_binder_nll_fn
from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.packing import pack_params
from dluxshera.params.spec import build_forward_model_spec_from_config
from dluxshera.params.store import ParameterStore


def test_theta0_store_override_keeps_binder_base_alignment():
    jax.config.update("jax_enable_x64", True)

    cfg = SHERA_TESTBED_CONFIG
    forward_spec = build_forward_model_spec_from_config(cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = base_store.refresh_derived(forward_spec)

    binder = SheraThreePlaneBinder(cfg, forward_spec, base_store, use_system_graph=False)
    data = binder.model()
    var = jnp.ones_like(data)

    infer_keys = (
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.x_position_as",
    )

    theta0_store = base_store.replace(
        {
            "binary.separation_as": 5.0,
            "binary.position_angle_deg": 45.0,
            "binary.x_position_as": 0.5,
        }
    )

    loss_fn, theta0, predict_fn = make_binder_nll_fn(
        binder=binder,
        infer_keys=infer_keys,
        data=data,
        var=var,
        theta0_store=theta0_store,
        return_predict_fn=True,
    )

    sub_spec = forward_spec.subset(infer_keys)
    expected_theta0 = pack_params(sub_spec, theta0_store)
    assert jnp.allclose(theta0, expected_theta0)

    theta_true = pack_params(sub_spec, base_store)
    pred_true = predict_fn(theta_true)
    assert jnp.max(jnp.abs(pred_true - data)) < 1e-7

    loss_true = loss_fn(theta_true)
    assert jnp.isfinite(loss_true)
