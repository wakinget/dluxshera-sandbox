import jax
import jax.numpy as np

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.core.builder import build_shera_threeplane_model
from dluxshera.inference.optimization import make_image_nll_fn


def _make_store_for_smoke(cfg):
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    updates = {
        "binary.separation_as": 10.0,
        "binary.position_angle_deg": 45.0,
        "binary.x_position": 0.0,
        "binary.y_position": 0.0,
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
        "binary.x_position",
        "binary.y_position",
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
