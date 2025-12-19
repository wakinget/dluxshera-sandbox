# tests/test_fim_theta.py
import jax
import jax.numpy as np
import numpy as onp

from dluxshera.inference.optimization import (
    make_binder_image_nll_fn,
    fim_theta,
    fim_theta_shera,
)


def test_fim_theta_shape_and_symmetry(
    shera_smoke_cfg,
    shera_smoke_forward,
    shera_smoke_binder_data,
    shera_smoke_infer_keys,
):
    spec, store = shera_smoke_forward
    _, data, var = shera_smoke_binder_data

    # Canonical Binder-based loss
    loss_fn, theta0 = make_binder_image_nll_fn(
        shera_smoke_cfg,
        spec,
        store,
        shera_smoke_infer_keys,
        data,
        var,
        noise_model="gaussian",
    )

    # Direct θ-space FIM
    F = fim_theta(loss_fn, theta0)

    # Basic checks
    assert F.shape == (theta0.size, theta0.size)

    # Symmetry
    assert np.allclose(F, F.T, atol=1e-5)

    # PSD (up to numerical tolerance): eigenvalues >= -eps
    evals = onp.linalg.eigvalsh(onp.asarray(F))
    assert evals.min() >= -1e-5


def test_fim_theta_shera_wrapper_consistency(
    shera_smoke_cfg,
    shera_smoke_forward,
    shera_smoke_binder_data,
):
    cfg = shera_smoke_cfg
    spec, store = shera_smoke_forward

    infer_keys = ["binary.separation_as", "binary.x_position_as"]

    _, data, var = shera_smoke_binder_data

    # Wrapper-based FIM
    F_wrapped, theta0_wrapped = fim_theta_shera(
        cfg,
        spec,
        store,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
    )

    # Manual construction should match
    loss_fn_manual, theta0_manual = make_binder_image_nll_fn(
        cfg,
        spec,
        store,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
    )
    F_manual = fim_theta(loss_fn_manual, theta0_manual)

    assert np.allclose(theta0_wrapped, theta0_manual)
    assert np.allclose(F_wrapped, F_manual, atol=1e-6)
