from dataclasses import replace

import jax.numpy as jnp
import pytest

from dluxshera.core.binder import SheraThreePlaneBinder, SheraTwoPlaneBinder
from dluxshera.graph.system_graph import build_threeplane_system_graph, build_shera_twoplane_system_graph
from dluxshera.optics.config import SHERA_TESTBED_CONFIG, SheraTwoPlaneConfig
from tests.conftest import inference_store_from_forward, make_forward_store


def _make_forward_and_inference_stores(cfg):
    n_m1 = len(cfg.primary_noll_indices)
    n_m2 = len(cfg.secondary_noll_indices)

    updates = {
        "binary.separation_as": 5.0,
        "binary.position_angle_deg": 10.0,
        "binary.x_position_as": 0.0,
        "binary.y_position_as": 0.0,
        "binary.contrast": 2.0,
    }

    if n_m1 > 0:
        updates["primary.zernike_coeffs_nm"] = jnp.zeros(n_m1, dtype=jnp.float32)
    if n_m2 > 0:
        updates["secondary.zernike_coeffs_nm"] = jnp.zeros(n_m2, dtype=jnp.float32)

    forward_spec, forward_store = make_forward_store(cfg, updates=updates)
    inference_spec, inference_store = inference_store_from_forward(forward_store)
    return forward_spec, forward_store, inference_spec, inference_store


@pytest.mark.slow
def test_system_graph_forward_matches_legacy_model():
    cfg = SHERA_TESTBED_CONFIG.replace(n_lambda=1)
    forward_spec, forward_store, inference_spec, inference_store = _make_forward_and_inference_stores(cfg)

    graph = build_threeplane_system_graph(cfg, forward_spec, forward_store)
    graph_psf = graph.evaluate()

    binder = SheraThreePlaneBinder(cfg, forward_spec, forward_store)
    binder_psf = binder.model()

    assert graph_psf.shape == binder_psf.shape == (cfg.psf_npix, cfg.psf_npix)
    assert graph_psf.dtype == binder_psf.dtype
    assert jnp.all(jnp.isfinite(graph_psf))
    assert jnp.allclose(graph_psf, binder_psf)


def test_twoplane_system_graph_matches_binder():
    cfg = SheraTwoPlaneConfig(n_lambda=1)
    forward_spec, forward_store = make_forward_store(cfg)

    graph = build_shera_twoplane_system_graph(cfg, forward_spec, forward_store)
    graph_psf = graph.evaluate()

    binder = SheraTwoPlaneBinder(cfg, forward_spec, forward_store)
    binder_psf = binder.model()

    assert graph_psf.shape == binder_psf.shape == (cfg.psf_npix, cfg.psf_npix)
    assert graph_psf.dtype == binder_psf.dtype
    assert jnp.all(jnp.isfinite(graph_psf))
    assert jnp.allclose(graph_psf, binder_psf)


def test_graph_outputs_mapping_is_shared():
    cfg = SheraTwoPlaneConfig(n_lambda=1)
    forward_spec, forward_store = make_forward_store(cfg)
    graph = build_shera_twoplane_system_graph(cfg, forward_spec, forward_store)

    outputs = graph.evaluate(outputs=("psf", "copy"))

    assert set(outputs.keys()) == {"psf", "copy"}
    assert jnp.allclose(outputs["psf"], outputs["copy"])
