from types import SimpleNamespace

import jax
import numpy as np
import pytest

from dluxshera.builders.detector import build_detector
from dluxshera.components.detectors import GSENSE2020BSI_SPEC, HWK4123_SPEC, SheraDetector


def test_build_detector_has_explicit_v1_layer_order():
    cfg = SimpleNamespace(psf_npix=8, oversample=2)

    detector, _contract = build_detector(cfg)

    assert list(detector.layers.keys()) == [
        "downsample",
        "pixel_offsets",
        "pixel_response",
        "jitter",
    ]


def test_build_detector_conditions_larger_maps_with_center_crop_and_warns(tmp_path):
    dx = jax.numpy.arange(36, dtype=float).reshape(6, 6)
    dy = jax.numpy.zeros((6, 6), dtype=float)
    dx_path = tmp_path / "dx.npy"
    dy_path = tmp_path / "dy.npy"
    np.save(dx_path, dx)
    np.save(dy_path, dy)

    cfg = SimpleNamespace(
        psf_npix=4,
        oversample=1,
        ppu_dx_path=str(dx_path),
        ppu_dy_path=str(dy_path),
    )

    with pytest.warns(UserWarning, match="policy center-crop"):
        detector, _contract = build_detector(cfg)

    assert detector.layers["pixel_offsets"].dx_map.shape == (4, 4)
    assert detector.layers["pixel_offsets"].dy_map.shape == (4, 4)


def test_build_detector_conditions_smaller_maps_with_reflect_pad_and_warns(tmp_path):
    dx = jax.numpy.array([[1.0, 2.0], [3.0, 4.0]])
    dy = jax.numpy.zeros((2, 2), dtype=float)
    dx_path = tmp_path / "dx_small.npy"
    dy_path = tmp_path / "dy_small.npy"
    np.save(dx_path, dx)
    np.save(dy_path, dy)

    cfg = SimpleNamespace(
        psf_npix=4,
        oversample=1,
        ppu_dx_path=str(dx_path),
        ppu_dy_path=str(dy_path),
    )

    with pytest.warns(UserWarning, match=r"policy center-pad\+reflect"):
        detector, _contract = build_detector(cfg)

    assert detector.layers["pixel_offsets"].dx_map.shape == (4, 4)
    assert detector.layers["pixel_offsets"].dy_map.shape == (4, 4)


def test_build_detector_uses_zero_offset_maps_when_unset():
    cfg = SimpleNamespace(psf_npix=6, oversample=1)

    detector, _contract = build_detector(cfg)

    dx = detector.layers["pixel_offsets"].dx_map
    dy = detector.layers["pixel_offsets"].dy_map
    assert dx.shape == (6, 6)
    assert dy.shape == (6, 6)
    assert float(dx.sum()) == 0.0
    assert float(dy.sum()) == 0.0


def test_build_detector_returns_shera_detector_with_spec_access():
    cfg = SimpleNamespace(psf_npix=8, oversample=1, detector_model="HWK4123")

    detector, _contract = build_detector(cfg)

    assert isinstance(detector, SheraDetector)
    assert detector.spec == HWK4123_SPEC
    assert detector.spec.model_name == "HWK4123"


def test_detector_spec_is_not_part_of_pytree_leaves():
    cfg = SimpleNamespace(psf_npix=8, oversample=1, detector_model="GSENSE2020BSI")

    detector, _contract = build_detector(cfg)

    leaves = jax.tree_util.tree_leaves(detector)
    assert detector.spec == GSENSE2020BSI_SPEC
    assert detector.spec not in leaves


def test_build_detector_rejects_unknown_model_name():
    cfg = SimpleNamespace(psf_npix=8, oversample=1, detector_model="UNKNOWN")

    with pytest.raises(ValueError, match="Unknown detector_model"):
        build_detector(cfg)
