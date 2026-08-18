# tests/test_binder_smoke.py

import jax.numpy as jnp
import numpy as np
import pytest

from dluxshera.systems import SheraBinder
from dluxshera.systems import SheraBinder
from dluxshera.systems import SheraBinder
from dluxshera.systems.two_plane import SheraTwoPlaneConfig
from tests.conftest import make_forward_store


@pytest.mark.slow
def test_shera_threeplane_binder_smoke(shera_smoke_cfg):
    cfg = shera_smoke_cfg
    forward_spec, forward_store = make_forward_store(cfg)

    binder = SheraBinder(cfg, forward_spec, forward_store)

    img = binder.model()

    assert img.ndim == 2  # simple shape sanity check
    assert jnp.all(jnp.isfinite(img))


def test_shera_twoplane_binder_smoke():
    cfg = SheraTwoPlaneConfig()
    forward_spec, forward_store = make_forward_store(cfg)

    binder = SheraBinder(cfg, forward_spec, forward_store)

    img = binder.model()

    assert img.ndim == 2
    assert jnp.all(jnp.isfinite(img))


def test_base_binder_dispatch_threeplane_smoke(shera_smoke_cfg):
    cfg = shera_smoke_cfg.replace(
        system={
            "source": {"kind": "binary"},
            "optics": {"kind": "three_plane"},
            "detector": {
                "kind": "layered",
                "model": shera_smoke_cfg.detector_model,
                "layers": shera_smoke_cfg.detector_layers,
            },
        }
    )
    forward_spec, forward_store = make_forward_store(cfg)

    binder = SheraBinder(cfg, forward_spec, forward_store)

    img = binder.model()

    assert img.ndim == 2
    assert jnp.all(jnp.isfinite(img))


def test_single_star_binder_renders_and_log_flux_scales(shera_smoke_cfg):
    cfg = shera_smoke_cfg.replace(
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
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)

    image = binder.model()
    brighter_store = binder.strip_structural(
        forward_store.replace({"source.log_flux_total": 7.0})
    )
    brighter = binder.model(brighter_store)

    assert image.ndim == 2
    assert jnp.all(jnp.isfinite(image))
    assert float(jnp.sum(image)) > 0.0
    assert float(jnp.sum(brighter)) > 0.0
    np.testing.assert_allclose(
        float(jnp.sum(brighter) / jnp.sum(image)),
        10.0,
        rtol=5.0e-2,
    )


def test_binary_target_binder_regression_smoke(shera_smoke_cfg):
    cfg = shera_smoke_cfg.replace(
        system={
            "source": {
                "kind": "binary_target",
                "target": "ALPHA_CEN",
                "wavelength_m": shera_smoke_cfg.wavelength_m,
                "bandwidth_m": shera_smoke_cfg.bandwidth_m,
                "n_lambda": shera_smoke_cfg.n_lambda,
                "exposure_time_s": 0.05,
            },
            "optics": {"kind": "three_plane"},
            "detector": {
                "kind": "layered",
                "model": shera_smoke_cfg.detector_model,
                "layers": shera_smoke_cfg.detector_layers,
            },
        }
    )
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraBinder(cfg, forward_spec, forward_store)

    image = binder.model(binder.strip_structural(forward_store))

    assert cfg.system["source"]["kind"] == "binary_target"
    assert "source.raw_fluxes" in forward_store
    assert image.ndim == 2
    assert jnp.all(jnp.isfinite(image))
    assert float(jnp.sum(image)) > 0.0
