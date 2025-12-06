# tests/test_inference_api.py
import jax.numpy as np

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.core.builder import build_shera_threeplane_model
from dluxshera.inference.inference import run_shera_image_gd_basic


def _make_store_for_smoke(cfg):
    """Helper to build a simple 'truth' store, mirroring test_image_nll_bridge."""
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    updates = {
        "binary.separation_as": 10.0,
        "binary.position_angle_deg": 90.0,
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


def test_run_shera_image_gd_basic_separation_smoke():
    cfg = SHERA_TESTBED_CONFIG
    spec, store_true = _make_store_for_smoke(cfg)

    # 1) Generate synthetic data from the "truth" store
    model_true = build_shera_threeplane_model(cfg, spec, store_true)
    data = model_true.model()
    var = np.ones_like(data)

    # 2) Start from a slightly wrong separation via init_overrides
    sep_true = store_true.get("binary.separation_as")
    sep_init = sep_true * 1.1
    init_overrides = {"binary.separation_as": sep_init}

    # 3) Call the high-level API
    theta_final, store_final, history = run_shera_image_gd_basic(
        data,
        var,
        cfg=cfg,
        infer_keys=("binary.separation_as",),
        init_overrides=init_overrides,
        noise_model="gaussian",
        learning_rate=1e-1,
        num_steps=20,
    )

    # Basic sanity: theta_final shape matches number of infer_keys
    assert theta_final.shape == (1,)

    # Loss should go down
    loss_start = float(history["loss"][0])
    loss_end = float(history["loss"][-1])
    assert loss_end < loss_start

    # Separation should move closer to the truth than the initial guess
    sep_est = store_final.get("binary.separation_as")
    assert abs(sep_est - sep_true) < abs(sep_init - sep_true)
