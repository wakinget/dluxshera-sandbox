from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "examples/scripts/generate_calibration_maps.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_calibration_maps_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_preview_image_writes_png(tmp_path):
    module = _load_script_module()
    preview_path = tmp_path / "preview_test.png"

    result = module.save_preview_image(
        [
            ("dx", np.zeros((4, 4), dtype=float)),
            ("dy", np.ones((4, 4), dtype=float)),
            ("prf", np.full((4, 4), 0.5, dtype=float)),
        ],
        preview_path,
        basename="test",
        mode="baseline",
    )

    assert result == preview_path
    assert preview_path.exists()
    assert preview_path.stat().st_size > 0


def test_generate_baseline_maps_jax_reproducible_with_seed():
    module = _load_script_module()

    dx1, dy1, prf1, seed1 = module.generate_baseline_maps_jax(4, 4, noise_amplitude=0.1, seed=123)
    dx2, dy2, prf2, seed2 = module.generate_baseline_maps_jax(4, 4, noise_amplitude=0.1, seed=123)

    assert seed1 == 123
    assert seed2 == 123
    assert np.allclose(dx1, dx2)
    assert np.allclose(dy1, dy2)
    assert np.allclose(prf1, prf2)


def test_realize_fpa_offsets_jax_reproducible_with_seed():
    module = _load_script_module()
    fixed_row = np.array([0.1], dtype=float)
    fixed_col = np.array([-0.01, 0.01], dtype=float)

    dx1, dy1, seed1 = module.realize_fpa_offsets_jax(
        4,
        5,
        fixed_row=fixed_row,
        fixed_col=fixed_col,
        sig_offset=0.01,
        seed=321,
    )
    dx2, dy2, seed2 = module.realize_fpa_offsets_jax(
        4,
        5,
        fixed_row=fixed_row,
        fixed_col=fixed_col,
        sig_offset=0.01,
        seed=321,
    )

    assert seed1 == 321
    assert seed2 == 321
    assert np.allclose(dx1, dx2)
    assert np.allclose(dy1, dy2)


def test_realize_fpa_offsets_jax_repeats_short_patterns_like_matlab():
    module = _load_script_module()

    dx, dy, _ = module.realize_fpa_offsets_jax(
        3,
        5,
        fixed_row=np.array([0.0], dtype=float),
        fixed_col=np.array([-0.01, 0.01], dtype=float),
        sig_offset=0.0,
        seed=42,
    )

    expected_dx = np.tile(np.array([[-0.01, 0.01, -0.01, 0.01, -0.01]], dtype=float), (3, 1))
    expected_dy = np.zeros((3, 5), dtype=float)

    assert np.allclose(dx, expected_dx)
    assert np.allclose(dy, expected_dy)


def test_generate_baseline_maps_jax_generates_seed_when_needed():
    module = _load_script_module()

    _, _, _, used_seed = module.generate_baseline_maps_jax(2, 2, noise_amplitude=0.1, seed=None)

    assert used_seed is not None
    assert isinstance(used_seed, int)
