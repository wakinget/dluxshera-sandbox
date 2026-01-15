"""Detector builder responsibilities (detector assembly and runtime wiring)."""

from __future__ import annotations

import dLux as dl


DETECTOR_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()


def build_detector(cfg) -> dl.LayeredDetector:
    """Construct the baseline detector for a Shera system."""

    return dl.LayeredDetector(layers=[("downsample", dl.Downsample(cfg.oversample))])


__all__ = [
    "DETECTOR_RUNTIME_BINDINGS",
    "build_detector",
]
