# tests/test_model_builder.py

import jax.numpy as jnp
import pytest

from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore

from dluxshera.legacy.builders import build_legacy_shera_threeplane_model
from dluxshera.legacy.modeling import SheraThreePlane_Model



def _make_inference_store(cfg):
    """
    Build a minimal, self-consistent inference ParameterStore suitable
    for constructing a SheraThreePlane_Model.

    We keep the values simple but non-zero so that the model has
    something sensible to do.
    """
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    # Zernike coeffs: zeros with lengths matching the config’s Noll indices
    n_m1 = len(cfg.primary_noll_indices)
    n_m2 = len(cfg.secondary_noll_indices)

    updates = {
        # Binary astrometry / photometry
        "source.separation_as": 10.0,
        "source.position_angle_deg": 45.0,
        "source.x_position_as": 0.0,
        "source.y_position_as": 0.0,
        "source.contrast": 3.0,
        "source.log_flux_total": 8.0,
        # System plate scale (not used by the legacy model construction yet,
        # but included for completeness)
        "optics.plate_scale_as_per_pix": 0.355,
    }

    if n_m1 > 0:
        updates["optics.primary.zernike_coeffs_nm"] = jnp.zeros(n_m1, dtype=jnp.float32)
    if n_m2 > 0:
        updates["optics.secondary.zernike_coeffs_nm"] = jnp.zeros(n_m2, dtype=jnp.float32)

    store = store.replace(updates)
    return spec, store


@pytest.mark.slow
def test_build_shera_threeplane_model_smoke(shera_smoke_cfg):
    """
    Smoke test for the (cfg, spec, store) → SheraThreePlane_Model bridge.

    Checks that:
      * we can construct a SheraThreePlane_Model without error,
      * a forward pass produces a PSF with the expected shape, and
      * key astrometric parameters are wired through into the source.
    """
    cfg = shera_smoke_cfg
    spec, store = _make_inference_store(cfg)

    model = build_legacy_shera_threeplane_model(cfg, spec, store)

    # Basic type check
    assert isinstance(model, SheraThreePlane_Model)

    # Forward pass: should produce a PSF with expected spatial dimensions
    psf = model.model()
    assert psf.shape[-2:] == (cfg.psf_npix, cfg.psf_npix)

    # Astrometry wiring: the model’s source should see the same values
    src = model.source

    # These assertions deliberately only check *consistency* with the store;
    # they don't assume any particular unit convention beyond “passed through”.
    sep_store = float(store.get("source.separation_as"))
    pa_store = float(store.get("source.position_angle_deg"))
    x_store = float(store.get("source.x_position_as"))
    y_store = float(store.get("source.y_position_as"))
    contrast_store = float(store.get("source.contrast"))
    log_flux_store = float(store.get("source.log_flux_total"))

    # AlphaCen fields are typically JAX arrays; cast to float for comparison.
    assert pytest.approx(float(src.separation)) == sep_store
    assert pytest.approx(float(src.position_angle)) == pa_store
    assert pytest.approx(float(src.x_position)) == x_store
    assert pytest.approx(float(src.y_position)) == y_store
    assert pytest.approx(float(src.contrast)) == contrast_store
    assert pytest.approx(float(src.log_flux)) == log_flux_store
