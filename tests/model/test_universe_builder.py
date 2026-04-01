# tests/test_universe_builder.py

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import dLuxToliman as dlT

from dluxshera.params.store import ParameterStore
from dluxshera.builders.source import (
    build_alpha_cen_source,
    build_binary_target_source,
    load_normalized_sed_weights,
)
from dluxshera.components.sources import TARGET_SPECS
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

    assert isinstance(source, dlT.AlphaCen)

    x, y, r, theta, log_flux, contrast = source.get(
        ["x_position", "y_position", "separation", "position_angle", "log_flux", "contrast"]
    )

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


def test_build_binary_target_source_seeds_nominal_target_values():
    cfg = SHERA_TESTBED_CONFIG.replace(
        system={
            "source": {
                "kind": "binary_target",
                "target": "ALPHA_CEN",
            }
        }
    )

    store = ParameterStore.from_dict(
        {
            "source.wavelength_m": cfg.wavelength_m,
            "source.bandwidth_m": cfg.bandwidth_m,
            "source.n_lambda": cfg.n_lambda,
        }
    )

    source = build_binary_target_source(store, cfg=cfg)

    assert float(source.separation) == pytest.approx(9.765)
    assert float(source.position_angle) == pytest.approx(14.508)
    assert float(source.contrast) > 0.0
    assert np.isfinite(float(source.log_flux))
    assert float(source.log_flux) > 0.0


def test_build_binary_target_source_store_overrides_win_over_target_defaults():
    cfg = SHERA_TESTBED_CONFIG.replace(
        system={
            "source": {
                "kind": "binary_target",
                "target": "ALPHA_CEN",
                "separation_as": 111.0,
            }
        }
    )

    store = ParameterStore.from_dict(
        {
            "source.wavelength_m": cfg.wavelength_m,
            "source.bandwidth_m": cfg.bandwidth_m,
            "source.n_lambda": cfg.n_lambda,
            "source.separation_as": 7.3,
            "source.position_angle_deg": 42.0,
            "source.contrast": 9.0,
            "source.log_flux_total": 7.0,
        }
    )

    source = build_binary_target_source(store, cfg=cfg)

    assert float(source.separation) == 7.3
    assert float(source.position_angle) == 42.0
    assert float(source.contrast) == 9.0


def test_load_normalized_sed_weights_interpolates_nm_to_m(tmp_path: Path):
    sed_file = tmp_path / "toy.dat"
    sed_file.write_text("500 1.0 0.0\n550 2.0 0.0\n600 3.0 0.0\n")

    # meter grid should map to [500, 550, 600] nm entries
    model_grid_m = np.array([500e-9, 550e-9, 600e-9])
    weights = load_normalized_sed_weights(sed_file, wavelength_grid_m=model_grid_m)

    assert np.isclose(np.sum(weights), 1.0)
    expected = np.array([500.0 * 1.0, 550.0 * 2.0, 600.0 * 3.0], dtype=float)
    expected /= np.sum(expected)
    assert np.allclose(weights, expected)


def test_load_normalized_sed_weights_errors_on_degenerate_interpolation(tmp_path: Path):
    sed_file = tmp_path / "degenerate.dat"
    sed_file.write_text("500 0.0 0.0\n550 0.0 0.0\n")

    with pytest.raises(ValueError, match="zero-sum photon spectral flux"):
        load_normalized_sed_weights(sed_file, wavelength_grid_m=np.array([520e-9, 530e-9]))


def test_build_binary_target_source_uses_uniform_weights_without_target_seds():
    cfg = SHERA_TESTBED_CONFIG.replace(system={"source": {"kind": "binary_target"}}, n_lambda=4)
    store = ParameterStore.from_dict(
        {
            "source.wavelength_m": cfg.wavelength_m,
            "source.bandwidth_m": cfg.bandwidth_m,
            "source.n_lambda": cfg.n_lambda,
            "source.separation_as": 10.0,
            "source.position_angle_deg": 90.0,
            "source.log_flux_total": 7.0,
            "source.contrast": 3.0,
        }
    )

    source = build_binary_target_source(store, cfg=cfg)
    expected = np.full((2, 4), 0.25)
    assert np.allclose(np.asarray(source.weights), expected)


@pytest.mark.parametrize("target_key", ["ALPHA_CEN", "61_CYG"])
def test_build_binary_target_source_uses_curated_target_sed_weights(target_key: str):
    cfg = SHERA_TESTBED_CONFIG.replace(
        system={"source": {"kind": "binary_target", "target": target_key}},
        n_lambda=4,
    )
    store = ParameterStore.from_dict(
        {
            "source.wavelength_m": cfg.wavelength_m,
            "source.bandwidth_m": cfg.bandwidth_m,
            "source.n_lambda": cfg.n_lambda,
        }
    )

    source = build_binary_target_source(store, cfg=cfg)
    weights = np.asarray(source.weights)

    assert weights.shape == (2, 4)
    assert np.allclose(np.sum(weights, axis=1), np.ones(2))
    assert not np.allclose(weights, np.full((2, 4), 0.25))
    assert float(source.contrast) > 0.0
    assert np.isfinite(float(source.log_flux))
    assert float(source.log_flux) > 0.0


def test_build_binary_target_source_falls_back_when_curated_sed_files_are_missing(monkeypatch):
    alpha = TARGET_SPECS["ALPHA_CEN"]
    monkeypatch.setitem(
        TARGET_SPECS,
        "ALPHA_CEN",
        replace(
            alpha,
            sed_a_file="does_not_exist_a.dat",
            sed_b_file="does_not_exist_b.dat",
        ),
    )

    cfg = SHERA_TESTBED_CONFIG.replace(
        system={"source": {"kind": "binary_target", "target": "ALPHA_CEN"}},
        n_lambda=4,
    )
    store = ParameterStore.from_dict(
        {
            "source.wavelength_m": cfg.wavelength_m,
            "source.bandwidth_m": cfg.bandwidth_m,
            "source.n_lambda": cfg.n_lambda,
        }
    )

    source = build_binary_target_source(store, cfg=cfg)
    assert np.allclose(np.asarray(source.weights), np.full((2, 4), 0.25))
    expected_contrast = 10 ** (0.4 * (alpha.vmag_b - alpha.vmag_a))
    assert float(source.contrast) == pytest.approx(expected_contrast)
    assert np.isfinite(float(source.log_flux))
    assert float(source.log_flux) > 0.0
