from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

__all__ = ["NoiseConfig", "apply_pair_noise"]


@dataclass(frozen=True)
class NoiseConfig:
    """Configure optional observation-noise augmentation for image pairs."""

    enabled: bool = False
    apply_to: str = "observation"
    photon_noise: bool = True
    read_noise: bool = False
    read_noise_sigma: float | None = None
    seed: int = 0
    training_dynamic: bool = True
    negative_policy: str = "raise"

    def __post_init__(self) -> None:
        if self.apply_to not in {"model", "observation", "both"}:
            raise ValueError("apply_to must be 'model', 'observation', or 'both'.")
        if self.negative_policy not in {"raise", "clip"}:
            raise ValueError("negative_policy must be 'raise' or 'clip'.")
        if self.read_noise and self.read_noise_sigma is None:
            raise ValueError("read_noise_sigma is required when read_noise=True.")
        if self.read_noise_sigma is not None and float(self.read_noise_sigma) < 0.0:
            raise ValueError("read_noise_sigma must be >= 0 when provided.")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready noise provenance."""
        return {
            "enabled": bool(self.enabled),
            "apply_to": self.apply_to,
            "photon_noise": bool(self.photon_noise),
            "read_noise": bool(self.read_noise),
            "read_noise_sigma": self.read_noise_sigma,
            "seed": int(self.seed),
            "training_dynamic": bool(self.training_dynamic),
            "negative_policy": self.negative_policy,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "NoiseConfig":
        """Build a noise config from an optional mapping."""
        if payload is None:
            return cls()
        return cls(
            enabled=bool(payload.get("enabled", False)),
            apply_to=str(payload.get("apply_to", "observation")),
            photon_noise=bool(payload.get("photon_noise", True)),
            read_noise=bool(payload.get("read_noise", False)),
            read_noise_sigma=payload.get("read_noise_sigma"),
            seed=int(payload.get("seed", 0)),
            training_dynamic=bool(payload.get("training_dynamic", True)),
            negative_policy=str(payload.get("negative_policy", "raise")),
        )


def _record_seed(base_seed: int, record_id: str | None, offset: int) -> int:
    payload = json.dumps([int(base_seed), record_id or "", int(offset)], separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def _apply_noise(image: np.ndarray, config: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
    noisy = np.asarray(image, dtype=np.float32).copy()
    if config.negative_policy == "raise" and np.any(noisy < 0.0):
        raise ValueError("Photon-noise inputs must be non-negative; use negative_policy='clip' intentionally.")
    if config.negative_policy == "clip":
        np.maximum(noisy, 0.0, out=noisy)
    if config.photon_noise:
        noisy = rng.poisson(noisy).astype(np.float32)
    if config.read_noise:
        noisy += rng.normal(0.0, float(config.read_noise_sigma), size=noisy.shape).astype(np.float32)
    return noisy


def apply_pair_noise(
    image_a: np.ndarray,
    image_b: np.ndarray,
    config: NoiseConfig | Mapping[str, Any] | None = None,
    *,
    pair_record_id: str | None = None,
    dynamic_seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply optional asymmetric observation noise to an ``(A, B)`` image pair.

    Disabled mode returns exact copies of the input values.  The default enabled
    policy treats ``B`` as the observation image and leaves ``A`` noiseless.
    """
    cfg = config if isinstance(config, NoiseConfig) else NoiseConfig.from_dict(config)
    a = np.asarray(image_a, dtype=np.float32)
    b = np.asarray(image_b, dtype=np.float32)
    if not cfg.enabled:
        return np.array(a, copy=True), np.array(b, copy=True)
    base_seed = _record_seed(cfg.seed, pair_record_id, dynamic_seed_offset)
    if cfg.apply_to in {"model", "both"}:
        a = _apply_noise(a, cfg, np.random.default_rng(base_seed + 17))
    else:
        a = np.array(a, copy=True)
    if cfg.apply_to in {"observation", "both"}:
        b = _apply_noise(b, cfg, np.random.default_rng(base_seed + 31))
    else:
        b = np.array(b, copy=True)
    return a, b
