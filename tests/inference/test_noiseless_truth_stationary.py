from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.optimization import make_binder_nll_fn
from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.packing import pack_params
from dluxshera.params.spec import (
    build_forward_model_spec_from_config,
    make_inference_subspec,
)
from dluxshera.params.store import ParameterStore


@pytest.mark.slow
def test_noiseless_truth_is_stationary_for_gaussian_nll():
    jax.config.update("jax_enable_x64", True)

    cfg = SHERA_TESTBED_CONFIG
    forward_spec = build_forward_model_spec_from_config(cfg)

    # Build a deterministic "truth" store and corresponding Binder/data.
    truth_store = ParameterStore.from_spec_defaults(forward_spec)
    truth_store = truth_store.replace(
        {
            "binary.separation_as": 10.0,
            "binary.position_angle_deg": 90.0,
            "binary.x_position_as": 0.0,
            "binary.y_position_as": 0.0,
            "imaging.exposure_time_s": 1.0,
        }
    )
    truth_store = truth_store.refresh_derived(forward_spec)

    binder_truth = SheraThreePlaneBinder(
        cfg, forward_spec, truth_store, use_system_graph=False
    )
    data = binder_truth.model()
    var = jnp.ones_like(data)

    infer_keys = (
        "binary.separation_as",
        "binary.position_angle_deg",
        "binary.x_position_as",
        "binary.y_position_as",
        "binary.log_flux_total",
        "binary.contrast",
        "system.plate_scale_as_per_pix",
        "primary.zernike_coeffs_nm",
    )
    inference_subspec = make_inference_subspec(
        base_spec=forward_spec, infer_keys=infer_keys, cfg=cfg
    )
    theta_true = pack_params(inference_subspec, truth_store)

    # Deliberately perturb a non-inference key in an alternate store. Supplying
    # this via theta0_store must not affect the unpack base used by the binder
    # inside the loss.
    mismatched_base = truth_store.replace(
        {"imaging.exposure_time_s": truth_store.get("imaging.exposure_time_s") * 2.0}
    )

    loss_fn, _theta0, predict_fn = make_binder_nll_fn(
        binder=binder_truth,
        infer_keys=infer_keys,
        data=data,
        var=var,
        noise_model="gaussian",
        reduce="sum",
        theta0_store=mismatched_base,
        return_predict_fn=True,
    )

    pred_true = predict_fn(theta_true)
    resid = pred_true - data
    grad_true = jax.grad(loss_fn)(theta_true)

    max_abs_resid = float(jnp.max(jnp.abs(resid)))
    max_abs_grad = float(jnp.max(jnp.abs(grad_true)))

    assert max_abs_resid < 1e-7, max_abs_resid
    assert max_abs_grad < 1e-5, max_abs_grad
