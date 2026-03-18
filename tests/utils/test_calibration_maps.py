from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from astropy.io import fits

from dluxshera.utils.calibration_maps import (
    MapMetadata,
    generate_baseline_maps,
    realize_fpa_offsets,
    write_fits,
)
from dluxshera.builders.detector import _load_array


def test_baseline_maps_zero_one_default():
    dx, dy, prf = generate_baseline_maps(4, 5)

    assert dx.shape == (4, 5)
    assert dy.shape == (4, 5)
    assert prf.shape == (4, 5)
    assert np.allclose(dx, 0.0)
    assert np.allclose(dy, 0.0)
    assert np.allclose(prf, 1.0)


def test_baseline_noise_reproducible_with_seed():
    dx1, dy1, prf1 = generate_baseline_maps(3, 3, noise_amplitude=0.1, seed=123)
    dx2, dy2, prf2 = generate_baseline_maps(3, 3, noise_amplitude=0.1, seed=123)

    assert np.allclose(dx1, dx2)
    assert np.allclose(dy1, dy2)
    assert np.allclose(prf1, prf2)


def test_realize_fpa_tiling_no_noise():
    fixed_row = np.array([0.1])
    fixed_col = np.array([0.3, 0.4])

    dx, dy = realize_fpa_offsets(
        3,
        5,
        fixed_row=fixed_row,
        fixed_col=fixed_col,
        sig_offset=0.0,
        seed=42,
    )

    expected_row = np.tile(np.array([[0.1], [0.1], [0.1]]), (1, 5))
    expected_col = np.tile(np.array([[0.3, 0.4, 0.3, 0.4, 0.3]]), (3, 1))

    assert np.allclose(dx, expected_col)
    assert np.allclose(dy, expected_row)


def test_write_fits_metadata_and_roundtrip(tmp_path):
    arr = np.ones((2, 2), dtype=float)
    path = tmp_path / "test_map.fits"
    meta = MapMetadata(map_type="DX", mode="baseline", seed=7, noise_amplitude=0.0)

    write_fits(arr, path, metadata=meta)

    data = fits.getdata(path)
    hdr = fits.getheader(path)

    assert np.allclose(data, arr)
    assert hdr["MAPTYPE"] == "DX"
    assert hdr["MODE"] == "baseline"
    assert hdr["SEED"] == 7
    assert hdr["SHAPE0"] == 2
    assert hdr["SHAPE1"] == 2


def test_load_array_reads_fits(tmp_path):
    arr = np.arange(4, dtype=float).reshape(2, 2)
    path = tmp_path / "loader.fits"
    fits.writeto(path, arr)

    loaded = _load_array(path)

    assert isinstance(loaded, jnp.ndarray)
    assert loaded.shape == (2, 2)
    assert np.allclose(np.array(loaded), arr)


def test_load_array_reads_npy(tmp_path):
    arr = np.arange(9, dtype=float).reshape(3, 3)
    path = tmp_path / "loader.npy"
    np.save(path, arr)

    loaded = _load_array(path)

    assert isinstance(loaded, jnp.ndarray)
    assert loaded.shape == (3, 3)
    assert np.allclose(np.array(loaded), arr)
