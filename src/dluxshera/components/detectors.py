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

    def __repr__(self) -> str:  # pragma: no cover - simple formatting
        fields = [
            f"model_name={self.model_name!r}",
            f"pixel_pitch_m={self.pixel_pitch_m!r}",
            f"array_size={self.array_size!r}",
            f"read_noise={self.read_noise!r}",
            f"dark_current={self.dark_current!r}",
            f"full_well={self.full_well!r}",
            f"qe={self.qe!r}",
            f"adc_bits={self.adc_bits!r}",
            f"shutter_type={self.shutter_type!r}",
            f"power_draw={self.power_draw!r}",
        ]
        inner = ",\n  ".join(fields)
        return f"DetectorSpec(\n  {inner}\n)"


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

    def __repr__(self) -> str:  # pragma: no cover - formatting exercised in tests
        lines = ["SheraDetector("]
        lines.append("  spec=")
        spec_repr = repr(self.spec)
        lines.extend(f"    {line}" for line in spec_repr.splitlines())
        lines.append("  layers={")
        for name, layer in self.layers.items():
            lines.append(f"    {name}: {layer!r}")
        lines.append("  }")
        lines.append(")")
        return "\n".join(lines)


__all__ = [
    "DetectorSpec",
    "GSENSE2020BSI_SPEC",
    "HWK4123_SPEC",
    "SheraDetector",
]
