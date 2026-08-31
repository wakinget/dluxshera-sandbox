from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .catalog import SampleCatalog

__all__ = ["IntensityScaler", "fit_intensity_scaler"]


@dataclass(frozen=True)
class IntensityScaler:
    """Apply one train-derived amplitude-preserving image scale."""

    mode: str = "raw"
    scale: float = 1.0
    sample_count: int = 0
    statistic: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"raw", "global_max_abs", "global_p99_abs"}:
            raise ValueError("mode must be 'raw', 'global_max_abs', or 'global_p99_abs'.")
        if not np.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("scale must be finite and > 0.")

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return ``image`` divided by the fixed scalar scale."""
        arr = np.asarray(image, dtype=np.float32)
        if self.mode == "raw":
            return np.array(arr, copy=True)
        return (arr / float(self.scale)).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready image scaling provenance."""
        return {
            "mode": self.mode,
            "scale": float(self.scale),
            "sample_count": int(self.sample_count),
            "statistic": self.statistic,
            "amplitude_preserving": True,
            "per_image_normalization": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "IntensityScaler":
        """Build a scaler from a serialized mapping."""
        if payload is None:
            return cls()
        return cls(
            mode=str(payload.get("mode", "raw")),
            scale=float(payload.get("scale", 1.0)),
            sample_count=int(payload.get("sample_count", 0)),
            statistic=payload.get("statistic"),
        )


def fit_intensity_scaler(
    catalog: SampleCatalog,
    sample_indices: Iterable[int],
    *,
    mode: str = "global_max_abs",
    max_samples: int | None = 512,
    cache_size: int = 4,
) -> IntensityScaler:
    """Fit one scalar image normalization using only selected training samples."""
    if mode == "raw":
        return IntensityScaler(mode="raw", scale=1.0, sample_count=0, statistic=None)
    indices = [int(idx) for idx in sample_indices]
    if max_samples is not None:
        indices = indices[: int(max_samples)]
    if not indices:
        raise ValueError("fit_intensity_scaler requires at least one training sample.")
    if mode not in {"global_max_abs", "global_p99_abs"}:
        raise ValueError("Unsupported intensity scaler mode.")
    values: list[float] = []
    with catalog.image_reader(cache_size=cache_size) as reader:
        if mode == "global_max_abs":
            for idx in indices:
                values.append(float(np.max(np.abs(reader.get(int(catalog.array_indices[idx]))))))
            scale = max(values)
            statistic = "max(abs(image)) over selected training samples"
        else:
            for idx in indices:
                values.extend(
                    np.abs(reader.get(int(catalog.array_indices[idx]))).ravel().astype(float).tolist()
                )
            scale = float(np.percentile(np.asarray(values, dtype=np.float64), 99.0))
            statistic = "99th percentile abs(pixel) over selected training samples"
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Derived image intensity scale is not finite and positive.")
    return IntensityScaler(
        mode=mode,
        scale=float(scale),
        sample_count=len(indices),
        statistic=statistic,
    )
