import jax
import numpy as np
import pytest

from dluxshera.builders.detector import build_detector, build_detector_contract
from dluxshera.components.detectors import GSENSE2020BSI_SPEC, HWK4123_SPEC, SheraDetector


def _layer(name: str, kind: str, **kwargs):
    return {"name": name, "kind": kind, **kwargs}


def test_build_detector_accepts_repeated_layer_kinds_and_preserves_order():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {
                "layers": [
                    _layer("pixel_response_a", "ApplyPixelResponse"),
                    _layer("jitter_a", "ApplyJitter", sigma=1e-12, kernel_size=3),
                    _layer("pixel_response_b", "ApplyPixelResponse"),
                    _layer("jitter_b", "ApplyJitter", sigma=2e-12, kernel_size=5),
                ]
            },
        }
    }

    detector, _contract = build_detector(cfg)

    assert list(detector.layers.keys()) == [
        "pixel_response_a",
        "jitter_a",
        "pixel_response_b",
        "jitter_b",
    ]
    assert float(detector.layers["jitter_a"].sigma) == 1e-12
    assert float(detector.layers["jitter_b"].sigma) == 2e-12
    assert int(detector.layers["jitter_b"].kernel_size) == 5


def test_detector_contract_uses_name_scoped_layer_keys_for_repeated_kinds():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {
                "layers": [
                    _layer("jitter_a", "ApplyJitter", sigma=1e-12, kernel_size=3),
                    _layer("jitter_b", "ApplyJitter", sigma=2e-12, kernel_size=5),
                ]
            },
        }
    }

    detector_contract = build_detector_contract(cfg)

    assert "detector.layers.jitter_a.sigma" in detector_contract
    assert "detector.layers.jitter_b.sigma" in detector_contract
    assert "detector.jitter.sigma" not in detector_contract


def test_build_detector_duplicate_layer_names_raise_clear_error():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {
                "layers": [
                    _layer("shared", "ApplyPixelResponse"),
                    _layer("shared", "ApplyJitter"),
                ]
            },
        }
    }

    with pytest.raises(ValueError, match="Duplicate detector layer name 'shared'"):
        build_detector(cfg)


def test_build_detector_missing_layer_name_raises_clear_error():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {"layers": [{"kind": "ApplyPixelResponse"}]},
        }
    }

    with pytest.raises(ValueError, match=r"system\.detector\.layers\[0\]\.name"):
        build_detector(cfg)


def test_build_detector_missing_layer_kind_raises_clear_error():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {"layers": [{"name": "pixel_response"}]},
        }
    }

    with pytest.raises(ValueError, match=r"system\.detector\.layers\[0\]\.kind"):
        build_detector(cfg)


def test_build_detector_conditions_larger_maps_with_center_crop_and_warns(tmp_path):
    dx = jax.numpy.arange(36, dtype=float).reshape(6, 6)
    dy = jax.numpy.zeros((6, 6), dtype=float)
    dx_path = tmp_path / "dx.npy"
    dy_path = tmp_path / "dy.npy"
    np.save(dx_path, dx)
    np.save(dy_path, dy)

    cfg = {
        "system": {
            "optics": {"psf_npix": 4},
            "detector": {
                "layers": [
                    _layer(
                        "pixel_offsets",
                        "ApplyPixelOffsets",
                        dx_path=str(dx_path),
                        dy_path=str(dy_path),
                    ),
                ]
            },
        }
    }

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

    cfg = {
        "system": {
            "optics": {"psf_npix": 4},
            "detector": {
                "layers": [
                    _layer(
                        "pixel_offsets",
                        "ApplyPixelOffsets",
                        dx_path=str(dx_path),
                        dy_path=str(dy_path),
                    ),
                ]
            },
        }
    }

    with pytest.warns(UserWarning, match=r"policy center-pad\+reflect"):
        detector, _contract = build_detector(cfg)

    assert detector.layers["pixel_offsets"].dx_map.shape == (4, 4)
    assert detector.layers["pixel_offsets"].dy_map.shape == (4, 4)


def test_build_detector_uses_zero_offset_maps_when_unset():
    cfg = {
        "system": {
            "optics": {"psf_npix": 6},
            "detector": {"layers": [_layer("pixel_offsets", "ApplyPixelOffsets")]},
        }
    }

    detector, _contract = build_detector(cfg)

    dx = detector.layers["pixel_offsets"].dx_map
    dy = detector.layers["pixel_offsets"].dy_map
    assert dx.shape == (6, 6)
    assert dy.shape == (6, 6)
    assert float(dx.sum()) == 0.0
    assert float(dy.sum()) == 0.0


def test_build_detector_returns_shera_detector_with_spec_access():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {
                "model": "HWK4123",
                "layers": [
                    _layer("downsample", "Downsample", kernel_size=1),
                    _layer("jitter", "ApplyJitter"),
                ],
            },
        }
    }

    detector, _contract = build_detector(cfg)

    assert isinstance(detector, SheraDetector)
    assert detector.spec == HWK4123_SPEC
    assert detector.spec.model_name == "HWK4123"


def test_detector_spec_is_not_part_of_pytree_leaves():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {
                "model": "GSENSE2020BSI",
                "layers": [
                    _layer("downsample", "Downsample", kernel_size=1),
                    _layer("pixel_response", "ApplyPixelResponse"),
                ],
            },
        }
    }

    detector, _contract = build_detector(cfg)

    leaves = jax.tree_util.tree_leaves(detector)
    assert detector.spec == GSENSE2020BSI_SPEC
    assert detector.spec not in leaves


def test_build_detector_rejects_unknown_model_name():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {
                "model": "UNKNOWN",
                "layers": [_layer("downsample", "Downsample", kernel_size=1)],
            },
        }
    }

    with pytest.raises(ValueError, match="Unknown detector_model"):
        build_detector(cfg)


def test_build_detector_uses_layers_pipeline_order_and_no_implicit_downsample():
    cfg = {
        "system": {
            "optics": {"psf_npix": 8},
            "detector": {
                "model": "HWK4123",
                "layers": [
                    _layer("pixel_response", "ApplyPixelResponse"),
                    _layer("jitter", "ApplyJitter", sigma=2e-12, kernel_size=5),
                ],
            },
        }
    }

    detector, _contract = build_detector(cfg)

    assert list(detector.layers.keys()) == ["pixel_response", "jitter"]
    assert detector.spec == HWK4123_SPEC
    assert float(detector.layers["jitter"].sigma) == 2e-12
    assert int(detector.layers["jitter"].kernel_size) == 5


def test_build_detector_warns_when_only_dx_path_provided(tmp_path):
    dx = np.ones((5, 5), dtype=float)
    dx_path = tmp_path / "dx.npy"
    np.save(dx_path, dx)

    cfg = {
        "system": {
            "optics": {"psf_npix": 5},
            "detector": {
                "layers": [
                    _layer("pixel_offsets", "ApplyPixelOffsets", dx_path=str(dx_path)),
                ]
            },
        }
    }

    with pytest.warns(UserWarning, match="dy_path missing; defaulting dy_map to zeros"):
        detector, _contract = build_detector(cfg)

    dy = detector.layers["pixel_offsets"].dy_map
    assert float(dy.sum()) == 0.0


def test_build_detector_warns_when_only_dy_path_provided(tmp_path):
    dy = np.ones((5, 5), dtype=float)
    dy_path = tmp_path / "dy.npy"
    np.save(dy_path, dy)

    cfg = {
        "system": {
            "optics": {"psf_npix": 5},
            "detector": {
                "layers": [
                    _layer("pixel_offsets", "ApplyPixelOffsets", dy_path=str(dy_path)),
                ]
            },
        }
    }

    with pytest.warns(UserWarning, match="dx_path missing; defaulting dx_map to zeros"):
        detector, _contract = build_detector(cfg)

    dx = detector.layers["pixel_offsets"].dx_map
    assert float(dx.sum()) == 0.0
