# tests/test_inference_api.py
import jax.numpy as np
import pytest

from dluxshera.inference.inference import run_shera_image_gd_basic


@pytest.mark.slow
def test_run_shera_image_gd_basic_separation_smoke(
    shera_smoke_cfg,
    shera_smoke_inference,
    shera_smoke_model_data,
):
    _, store_true = shera_smoke_inference
    data, var = shera_smoke_model_data

    # 2) Start from a slightly wrong separation via init_overrides
    sep_true = store_true.get("binary.separation_as")
    sep_init = sep_true * 1.1
    init_overrides = {"binary.separation_as": sep_init}

    # 3) Call the high-level API
    theta_final, store_final, history = run_shera_image_gd_basic(
        data,
        var,
        cfg=shera_smoke_cfg,
        infer_keys=("binary.separation_as",),
        init_overrides=init_overrides,
        noise_model="gaussian",
        learning_rate=1e-1,
        num_steps=10,
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
