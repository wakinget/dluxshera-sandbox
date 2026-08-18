from __future__ import annotations

import jax.numpy as jnp

from dluxshera.inference.optimization import (
    diagnose_first_step,
    make_binder_nll_fn,
    run_shera_gd,
)


def test_first_step_finite_smoke(shera_smoke_binder_data, shera_smoke_infer_keys):
    binder, data, var = shera_smoke_binder_data
    loss_fn, theta0 = make_binder_nll_fn(
        binder=binder,
        infer_keys=shera_smoke_infer_keys,
        data=data,
        var=var,
        noise_model="gaussian",
        reduce="sum",
    )

    diag = diagnose_first_step(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=1e-2,
    )

    assert diag["loss0_finite"]
    assert diag["grad0_finite"]
    assert diag["theta1_finite"]
    assert diag["loss1_finite"]

    theta_final, history = run_shera_gd(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=1e-2,
        num_steps=1,
        return_artifacts=False,
    )

    assert jnp.all(jnp.isfinite(theta_final))
    assert jnp.all(jnp.isfinite(history["loss"]))
