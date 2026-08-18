from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dluxshera.systems import SheraBinder
from tests.conftest import make_forward_store


def _single_star_cfg(shera_smoke_cfg):
    return shera_smoke_cfg.replace(
        system={
            "source": {
                "kind": "single_star",
                "wavelength_m": shera_smoke_cfg.wavelength_m,
                "bandwidth_m": shera_smoke_cfg.bandwidth_m,
                "n_lambda": shera_smoke_cfg.n_lambda,
                "exposure_time_s": 0.05,
                "x_position_as": 0.0,
                "y_position_as": 0.0,
                "position_angle_deg": 0.0,
                "log_flux_total": 6.0,
            },
            "optics": {"kind": "three_plane"},
            "detector": {
                "kind": "layered",
                "model": shera_smoke_cfg.detector_model,
                "layers": shera_smoke_cfg.detector_layers,
            },
        }
    )


def _binder_and_store(cfg):
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)
    return binder, forward_store


def test_single_star_position_angle_is_inert_for_rendered_image(shera_smoke_cfg):
    cfg = _single_star_cfg(shera_smoke_cfg)
    binder, store = _binder_and_store(cfg)
    base = binder.model(binder.strip_structural(store.replace({"source.position_angle_deg": 0.0})))
    rolled = binder.model(
        binder.strip_structural(store.replace({"source.position_angle_deg": 0.1}))
    )

    assert base.shape == rolled.shape
    assert jnp.all(jnp.isfinite(base))
    assert jnp.all(jnp.isfinite(rolled))
    np.testing.assert_allclose(
        float(jnp.sum(base)),
        float(jnp.sum(rolled)),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.asarray(rolled),
        np.asarray(base),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_single_star_position_angle_has_negligible_loss_sensitivity(shera_smoke_cfg):
    cfg = _single_star_cfg(shera_smoke_cfg)
    binder, store = _binder_and_store(cfg)
    truth_pa = 0.1
    truth_store = binder.strip_structural(store.replace({"source.position_angle_deg": truth_pa}))
    data = binder.model(truth_store)

    zero_model = binder.model(binder.strip_structural(store.replace({"source.position_angle_deg": 0.0})))
    matched_model = binder.model(truth_store)
    zero_loss = float(jnp.sum(jnp.square(zero_model - data)))
    matched_loss = float(jnp.sum(jnp.square(matched_model - data)))

    np.testing.assert_allclose(matched_loss, zero_loss, rtol=1.0e-7, atol=1.0e-7)


def test_single_star_position_angle_gradient_is_near_zero(shera_smoke_cfg):
    jax.config.update("jax_enable_x64", True)
    cfg = _single_star_cfg(shera_smoke_cfg)
    binder, store = _binder_and_store(cfg)
    truth_pa = 0.15
    truth_store = binder.strip_structural(store.replace({"source.position_angle_deg": truth_pa}))
    data = binder.model(truth_store)

    def loss_fn(pa_deg: jnp.ndarray) -> jnp.ndarray:
        pa_store = binder.strip_structural(store.replace({"source.position_angle_deg": pa_deg}))
        model = binder.model(pa_store)
        return jnp.sum(jnp.square(model - data))

    grad_at_zero = float(jax.grad(loss_fn)(jnp.asarray(0.0, dtype=float)))
    assert abs(grad_at_zero) <= 1.0e-9
