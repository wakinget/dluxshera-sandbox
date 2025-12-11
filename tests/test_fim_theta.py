# tests/test_fim_theta.py
import jax
import jax.numpy as np
import numpy as onp

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.inference.optimization import (
    make_binder_image_nll_fn,
    fim_theta,
    fim_theta_shera,
)
from dluxshera.core.binder import SheraThreePlaneBinder


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


def test_fim_theta_shape_and_symmetry():
    cfg = SHERA_TESTBED_CONFIG
    spec, store = _make_store_for_smoke(cfg)

    infer_keys = [
        "binary.separation_as",
        "binary.x_position_as",
        "binary.y_position_as",
    ]

    # Synthetic data via Binder path
    binder = SheraThreePlaneBinder(cfg, spec, store)
    data = binder.forward(store)
    var = np.ones_like(data)

    # Canonical Binder-based loss
    loss_fn, theta0 = make_binder_image_nll_fn(
        cfg,
        spec,
        store,
        infer_keys,
        data,
        var,
        noise_model="gaussian",
    )

    # Direct θ-space FIM
    F = fim_theta(loss_fn, theta0)

    # Basic checks
    n = theta0.size
    assert F.shape == (n, n)

    # Symmetry
    assert np.allclose(F, F.T, atol=1e-5)

    # PSD (up to numerical tolerance): eigenvalues >= -eps
    evals = onp.linalg.eigvalsh(onp.asarray(F))
    assert evals.min() >= -1e-5


def test_fim_theta_shera_wrapper_consistency():
    cfg = SHERA_TESTBED_CONFIG
    spec, store = _make_store_for_smoke(cfg)

    infer_keys = ["binary.separation_as", "binary.x_position_as"]

    binder = SheraThreePlaneBinder(cfg, spec, store)
    data = binder.forward(store)
    var = np.ones_like(data)

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
