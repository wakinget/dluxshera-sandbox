"""Legacy optics package exports.

Prefer importing system configs/binders from ``dluxshera.systems``.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseConfig",
    "SheraThreePlaneConfig",
    "SheraTwoPlaneConfig",
    "SHERA_TESTBED_CONFIG",
    "SHERA_FLIGHT_CONFIG",
    "SheraThreePlaneOptics",
    "SheraTwoPlaneOptics",
]


def __getattr__(name: str) -> Any:
    if name == "BaseConfig":
        from ..systems.base import BaseConfig

        return BaseConfig
    if name in {"SheraThreePlaneConfig", "SHERA_TESTBED_CONFIG", "SHERA_FLIGHT_CONFIG"}:
        from ..systems.three_plane import (
            SHERA_FLIGHT_CONFIG,
            SHERA_TESTBED_CONFIG,
            SheraThreePlaneConfig,
        )

        return {
            "SheraThreePlaneConfig": SheraThreePlaneConfig,
            "SHERA_TESTBED_CONFIG": SHERA_TESTBED_CONFIG,
            "SHERA_FLIGHT_CONFIG": SHERA_FLIGHT_CONFIG,
        }[name]
    if name == "SheraTwoPlaneConfig":
        from ..systems.two_plane import SheraTwoPlaneConfig

        return SheraTwoPlaneConfig
    if name in {"SheraThreePlaneOptics", "SheraTwoPlaneOptics"}:
        from .optical_systems import SheraThreePlaneOptics, SheraTwoPlaneOptics

        return {
            "SheraThreePlaneOptics": SheraThreePlaneOptics,
            "SheraTwoPlaneOptics": SheraTwoPlaneOptics,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
