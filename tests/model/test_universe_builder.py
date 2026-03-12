# tests/test_universe_builder.py

import jax.numpy as jnp

import dLuxToliman as dlT

from dluxshera.params.store import ParameterStore
from dluxshera.builders.source import build_alpha_cen_source
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG


def test_build_alpha_cen_source_roundtrip():
    cfg = SHERA_TESTBED_CONFIG.replace(n_lambda=3)

    store = ParameterStore.from_dict(
        {
            "source.wavelength_m": cfg.wavelength_m,
            "source.bandwidth_m": cfg.bandwidth_m,
            "source.n_lambda": cfg.n_lambda,
            "source.separation_as": 9.5,
            "source.position_angle_deg": 75.0,
            "source.x_position_as": 2.0,
            "source.y_position_as": -1.5,
            "source.log_flux_total": 7.5,
            "source.contrast": 2.8,
        }
    )

    source = build_alpha_cen_source(store, cfg=cfg)

    # Sanity check type
    assert isinstance(source, dlT.AlphaCen)

    # Fetch the same parameters back from the AlphaCen object.
    paths = [
        "x_position",
        "y_position",
        "separation",
        "position_angle",
        "log_flux",
        "contrast",
    ]
    x, y, r, theta, log_flux, contrast = source.get(paths)

    assert jnp.allclose(x, store.get("source.x_position_as"))
    assert jnp.allclose(y, store.get("source.y_position_as"))
    assert jnp.allclose(r, store.get("source.separation_as"))
    assert jnp.allclose(theta, store.get("source.position_angle_deg"))
    assert jnp.allclose(log_flux, store.get("source.log_flux_total"))
    assert jnp.allclose(contrast, store.get("source.contrast"))

    center_nm = float(store.get("source.wavelength_m")) * 1e9
    bandwidth_nm = float(store.get("source.bandwidth_m")) * 1e9
    expected_bandpass = (center_nm - bandwidth_nm / 2.0, center_nm + bandwidth_nm / 2.0)

    assert tuple(float(v) for v in source.bandpass) == expected_bandpass
    assert len(source.wavelengths) == int(store.get("source.n_lambda"))


def test_build_alpha_cen_source_defaults_xy_to_zero():
    cfg = SHERA_TESTBED_CONFIG
    store = ParameterStore.from_dict(
        {
            "source.wavelength_m": cfg.wavelength_m,
            "source.bandwidth_m": cfg.bandwidth_m,
            "source.n_lambda": cfg.n_lambda,
            "source.separation_as": 1.0,
            "source.position_angle_deg": 2.0,
            "source.log_flux_total": 7.0,
            "source.contrast": 3.0,
        }
    )

    source = build_alpha_cen_source(store, cfg=cfg)
    assert float(source.x_position) == 0.0
    assert float(source.y_position) == 0.0
