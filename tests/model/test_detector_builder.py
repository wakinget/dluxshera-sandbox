from types import SimpleNamespace

import pytest

from dluxshera.builders.detector import build_detector


def test_build_detector_has_explicit_v1_layer_order():
    cfg = SimpleNamespace(psf_npix=8, oversample=2)

    detector = build_detector(cfg)

    assert list(detector.layers.keys()) == [
        "downsample",
        "pixel_offsets",
        "pixel_response",
        "jitter",
    ]


def test_build_detector_conditions_larger_maps_with_center_crop_and_warns():
    cfg = SimpleNamespace(
        psf_npix=4,
        oversample=1,
        dx_map=[[0, 1, 2, 3, 4, 5]] * 6,
        dy_map=[[0, 0, 0, 0, 0, 0]] * 6,
    )

    with pytest.warns(UserWarning, match="policy center-crop"):
        detector = build_detector(cfg)

    assert detector.layers["pixel_offsets"].dx_map.shape == (4, 4)
    assert detector.layers["pixel_offsets"].dy_map.shape == (4, 4)


def test_build_detector_conditions_smaller_maps_with_reflect_pad_and_warns():
    cfg = SimpleNamespace(
        psf_npix=4,
        oversample=1,
        dx_map=[[1, 2], [3, 4]],
        dy_map=[[0, 0], [0, 0]],
    )

    with pytest.warns(UserWarning, match=r"policy center-pad\+reflect"):
        detector = build_detector(cfg)

    assert detector.layers["pixel_offsets"].dx_map.shape == (4, 4)
    assert detector.layers["pixel_offsets"].dy_map.shape == (4, 4)


def test_build_detector_uses_zero_offset_maps_when_unset():
    cfg = SimpleNamespace(psf_npix=6, oversample=1)

    detector = build_detector(cfg)

    dx = detector.layers["pixel_offsets"].dx_map
    dy = detector.layers["pixel_offsets"].dy_map
    assert dx.shape == (6, 6)
    assert dy.shape == (6, 6)
    assert float(dx.sum()) == 0.0
    assert float(dy.sum()) == 0.0
