# tests/test_universe_builder.py

import jax.numpy as jnp

import dLuxToliman as dlT

from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.core.universe import build_alpha_cen_source


def test_build_alpha_cen_source_roundtrip():
    # Start from the basic inference spec and default store
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    # Override with some non-trivial values
    updates = {
        "binary.separation_as": 9.5,
        "binary.position_angle_deg": 75.0,
        "binary.x_position": 2.0,
        "binary.y_position": -1.5,
        "binary.log_flux_total": 7.5,
        "binary.contrast": 2.8,
    }
    store = store.replace(updates)

    # Build the source – for now we just pick n_wavels=3
    source = build_alpha_cen_source(store, n_wavels=3)

    # Sanity check type
    assert isinstance(source, dlT.AlphaCen)

    # Fetch the same parameters back from the AlphaCen object.
    # These are the canonical Zodiax paths used in the dLux tutorials.
    paths = [
        "x_position",
        "y_position",
        "separation",
        "position_angle",
        "log_flux",
        "contrast",
    ]
    x, y, r, theta, log_flux, contrast = source.get(paths)

    assert jnp.allclose(x, updates["binary.x_position"])
    assert jnp.allclose(y, updates["binary.y_position"])
    assert jnp.allclose(r, updates["binary.separation_as"])
    assert jnp.allclose(theta, updates["binary.position_angle_deg"])
    assert jnp.allclose(log_flux, updates["binary.log_flux_total"])
    assert jnp.allclose(contrast, updates["binary.contrast"])
