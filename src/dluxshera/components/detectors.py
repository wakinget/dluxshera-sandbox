"""Shera-specific detector components and metadata specs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import dLux as dl


@dataclass(frozen=True)
class DetectorSpec:
    """Static Shera detector metadata for a physical sensor model."""

    model_name: Optional[str] = None
    pixel_pitch_m: Optional[float] = None
    array_size: Optional[tuple[int, int]] = None
    read_noise: Optional[float] = None
    dark_current: Optional[float] = None
    full_well: Optional[float] = None
    qe: Optional[float] = None
    adc_bits: Optional[int] = None
    shutter_type: Optional[str] = None
    power_draw: Optional[float] = None


GSENSE2020BSI_SPEC = DetectorSpec(
    model_name="GSENSE2020BSI",
    pixel_pitch_m=6.5e-6,
    array_size=(2048, 2048),
    read_noise=1.6,
    dark_current=0.07,
    full_well=55e3,
    qe=0.95,
    adc_bits=12,
    shutter_type="Rolling",
    power_draw=1.2,
)

HWK4123_SPEC = DetectorSpec(
    model_name="HWK4123",
    pixel_pitch_m=4.6e-6,
    array_size=(4096, 2300),
    read_noise=0.5,
    dark_current=2.0,
    full_well=7e3,
    qe=0.85,
    adc_bits=12,
    shutter_type="Rolling",
    power_draw=1.8,
)


class SheraDetector(dl.LayeredDetector):
    """Layered detector with non-pytree Shera detector metadata exposed as ``.spec``."""

    def __init__(self, layers, spec: DetectorSpec):
        super().__init__(layers=layers)
        object.__setattr__(self, "_spec", spec)

    @property
    def spec(self) -> DetectorSpec:
        """Return detector metadata that is intentionally excluded from pytree leaves."""
        return self._spec


__all__ = [
    "DetectorSpec",
    "GSENSE2020BSI_SPEC",
    "HWK4123_SPEC",
    "SheraDetector",
]
