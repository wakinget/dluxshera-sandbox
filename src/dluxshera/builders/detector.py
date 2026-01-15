"""Detector builder responsibilities (detector assembly and runtime wiring)."""

from __future__ import annotations

import dLux as dl


DETECTOR_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()


def build_detector(cfg) -> dl.LayeredDetector:
    """Construct the baseline detector for a Shera system."""

    return dl.LayeredDetector(layers=[("downsample", dl.Downsample(cfg.oversample))])


def apply_runtime_bindings(
    detector: dl.LayeredDetector,
    store,
    bindings: tuple[tuple[str, str], ...] = DETECTOR_RUNTIME_BINDINGS,
) -> dl.LayeredDetector:
    """Apply runtime ParameterStore overrides onto a cached detector."""

    if store is None:
        return detector

    for store_key, set_path in bindings:
        val = store.get(store_key, default=None)
        if val is None:
            continue
        detector = detector.set(set_path, val)
    return detector


__all__ = [
    "DETECTOR_RUNTIME_BINDINGS",
    "apply_runtime_bindings",
    "build_detector",
]
