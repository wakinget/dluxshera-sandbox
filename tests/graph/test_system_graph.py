from dataclasses import replace

import jax.numpy as jnp

from dluxshera.graph.system_graph import build_threeplane_system_graph
from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.core.builder import build_shera_threeplane_model


def _make_inference_store(cfg):
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    n_m1 = len(cfg.primary_noll_indices)
    n_m2 = len(cfg.secondary_noll_indices)

    updates = {
        "binary.separation_as": 5.0,
        "binary.position_angle_deg": 10.0,
        "binary.x_position": 0.0,
        "binary.y_position": 0.0,
        "binary.contrast": 2.0,
        "binary.log_flux_total": 7.5,
        "system.plate_scale_as_per_pix": 0.355,
    }

    if n_m1 > 0:
        updates["primary.zernike_coeffs"] = jnp.zeros(n_m1, dtype=jnp.float32)
    if n_m2 > 0:
        updates["secondary.zernike_coeffs"] = jnp.zeros(n_m2, dtype=jnp.float32)

    store = store.replace(updates)
    return spec, store


def test_system_graph_forward_matches_legacy_model():
    cfg = replace(SHERA_TESTBED_CONFIG, n_lambda=1)
    spec, store = _make_inference_store(cfg)

    graph = build_threeplane_system_graph(cfg, spec, store)
    graph_psf = graph.forward(store)

    legacy_model = build_shera_threeplane_model(cfg, spec, store)
    legacy_psf = legacy_model.model()

    assert graph_psf.shape == legacy_psf.shape == (cfg.psf_npix, cfg.psf_npix)
    assert graph_psf.dtype == legacy_psf.dtype
    assert jnp.all(jnp.isfinite(graph_psf))
    assert jnp.allclose(graph_psf, legacy_psf)
