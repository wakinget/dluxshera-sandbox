from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from dluxshera.datasets.schema import json_ready, read_json, write_json

from .catalog import SampleCatalog

__all__ = [
    "IntensityScaler",
    "fit_intensity_scaler",
    "intensity_scaler_content_sha256",
    "load_intensity_scaler",
    "write_intensity_scaler",
]


@dataclass(frozen=True)
class IntensityScaler:
    """Apply one train-derived amplitude-preserving image scale.

    ``IntensityScaler`` stores a single scalar normalization derived from a
    selected population, usually training images.  It deliberately avoids
    per-image normalization so absolute image amplitudes and image differences
    remain meaningful for downstream models.

    Parameters
    ----------
    mode:
        Scaling mode.  ``"raw"`` returns copied images unchanged,
        ``"global_max_abs"`` divides by the maximum absolute selected pixel, and
        ``"global_p99_abs"`` divides by the 99th percentile absolute selected
        pixel value.
    scale:
        Positive scalar divisor.
    sample_count:
        Number of selected samples used to fit the scale, if known.
    statistic:
        Human-readable description of the fitted statistic.
    source_population:
        Optional provenance describing the selected fitting population.
    """

    mode: str = "raw"
    scale: float = 1.0
    sample_count: int = 0
    statistic: str | None = None
    source_population: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"raw", "global_max_abs", "global_p99_abs"}:
            raise ValueError("mode must be 'raw', 'global_max_abs', or 'global_p99_abs'.")
        if not np.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("scale must be finite and > 0.")

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Return ``image`` divided by the fixed scalar scale.

        Parameters
        ----------
        image:
            Array-like image.

        Returns
        -------
        numpy.ndarray
            ``float32`` image copy for ``mode="raw"`` or scaled ``float32``
            image for fitted modes.
        """
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
            "source_population": json_ready(dict(self.source_population or {})),
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
            source_population=dict(payload.get("source_population", {}) or {}),
        )


def fit_intensity_scaler(
    catalog: SampleCatalog,
    sample_indices: Iterable[int],
    *,
    mode: str = "global_max_abs",
    max_samples: int | None = 512,
    cache_size: int = 4,
) -> IntensityScaler:
    """Fit one scalar image normalization from selected catalog samples.

    Parameters
    ----------
    catalog:
        Prepared sample catalog used to open the sharded image reader.
    sample_indices:
        Catalog row indices selected for fitting.  For leakage control this
        should usually be training-only.
    mode:
        Scaling mode.  ``"raw"`` returns an identity scaler without reading
        images.
    max_samples:
        Optional prefix cap applied to ``sample_indices`` before reading images.
    cache_size:
        Shard-reader cache size used while fitting.

    Returns
    -------
    IntensityScaler
        Fitted amplitude-preserving scaler.

    Raises
    ------
    ValueError
        If no selected samples are available for a fitted mode or the derived
        scale is not finite and positive.
    """
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
        source_population={
            "selection": "all_provided_sample_indices"
            if max_samples is None
            else "provided_sample_indices_prefix_after_optional_max_samples",
            "max_samples": max_samples,
            "catalog_artifact_id": catalog.artifact_id,
            "prepared_dataset_hash": catalog.prepared_dataset_hash,
        },
    )


def intensity_scaler_content_sha256(scaler: IntensityScaler | Mapping[str, Any]) -> str:
    """Return a stable content hash for a serialized intensity scaler."""
    payload = scaler.to_dict() if isinstance(scaler, IntensityScaler) else dict(scaler)
    stable = dict(payload)
    stable.pop("generated_at", None)
    stable.pop("content_identity", None)
    raw = json.dumps(json_ready(stable), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def write_intensity_scaler(
    path: Path,
    scaler: IntensityScaler,
    *,
    artifact_id: str = "SCALER-ML-v1",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a compact scaler artifact with a stable content identity."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it.")
    payload = {
        "schema_version": "dluxshera_ml_intensity_scaler/1",
        "artifact_id": str(artifact_id),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **scaler.to_dict(),
    }
    payload["content_identity"] = {
        "algorithm": "sha256/json-canonical/intensity-scaler-v1",
        "sha256": intensity_scaler_content_sha256(payload),
        "excludes": ["generated_at", "content_identity"],
    }
    write_json(path, payload)
    return payload


def load_intensity_scaler(
    path: Path,
    *,
    expected_content_sha256: str | None = None,
) -> IntensityScaler:
    """Load and validate a frozen intensity scaler artifact."""
    payload = read_json(Path(path))
    if payload.get("schema_version") != "dluxshera_ml_intensity_scaler/1":
        raise ValueError(f"Unsupported intensity scaler schema {payload.get('schema_version')!r}.")
    content_identity = payload.get("content_identity", {})
    actual = intensity_scaler_content_sha256(payload)
    if isinstance(content_identity, Mapping) and content_identity.get("sha256"):
        if str(content_identity["sha256"]) != actual:
            raise ValueError(
                "Intensity scaler content_identity.sha256 does not match scaler content "
                f"({content_identity['sha256']} != {actual})."
            )
    if expected_content_sha256 and str(expected_content_sha256) != actual:
        raise ValueError(
            "Intensity scaler hash does not match expected content identity "
            f"({expected_content_sha256} != {actual})."
        )
    return IntensityScaler.from_dict(payload)
