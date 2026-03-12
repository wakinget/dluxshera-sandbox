from dluxshera.components.detectors import SheraDetector, GSENSE2020BSI_SPEC
from dluxshera.builders.detector import _build_legacy_detector_layers
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG


def test_detector_repr_includes_spec():
    # Build detector via legacy helper to get layers quickly
    cfg = SHERA_TESTBED_CONFIG
    psf_npix = cfg.psf_npix
    layers = _build_legacy_detector_layers(cfg, target_shape=(psf_npix, psf_npix))
    detector = SheraDetector(layers=layers, spec=GSENSE2020BSI_SPEC)

    rendered = repr(detector)

    assert "SheraDetector(" in rendered
    assert "spec=" in rendered
    assert "DetectorSpec" in rendered
    assert "layers={" in rendered

