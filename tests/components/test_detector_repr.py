from dluxshera.components.detectors import SheraDetector, GSENSE2020BSI_SPEC
from dluxshera.builders.detector import build_detector
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG


def test_detector_repr_includes_spec():
    cfg = SHERA_TESTBED_CONFIG
    detector_cfg = {
        "system": {
            "optics": {"psf_npix": cfg.psf_npix},
            "detector": {
                "model": cfg.detector_model,
                "layers": cfg.detector_layers,
            },
        }
    }
    detector, _contract = build_detector(detector_cfg)

    rendered = repr(detector)

    assert "SheraDetector(" in rendered
    assert "spec=" in rendered
    assert "DetectorSpec" in rendered
    assert "layers={" in rendered
