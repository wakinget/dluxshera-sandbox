from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from dluxshera.components.detectors import GSENSE2020BSI_SPEC, HWK4123_SPEC, DetectorSpec
from dluxshera.utils.noise import apply_observation_noise, make_subkey

__all__ = ["NoiseConfig", "apply_pair_noise", "noise_config_identity", "pair_noise_side_seeds"]

DETECTOR_SPECS = {
    "GSENSE2020BSI": GSENSE2020BSI_SPEC,
    "HWK4123": HWK4123_SPEC,
}


@dataclass(frozen=True)
class NoiseConfig:
    """Configure optional observation-noise augmentation for image pairs."""

    enabled: bool = False
    apply_to: str = "observation"
    noise_model: str = "legacy_numpy"
    photon_noise: bool = True
    read_noise: bool = False
    read_noise_sigma: float | None = None
    dark_current: bool = False
    detector_model: str | None = None
    exposure_time_s: float | None = None
    bright_threshold: float = 100.0
    seed: int = 0
    training_dynamic: bool = True
    negative_policy: str = "raise"

    def __post_init__(self) -> None:
        if self.apply_to not in {"model", "observation", "both"}:
            raise ValueError("apply_to must be 'model', 'observation', or 'both'.")
        if self.noise_model not in {"legacy_numpy", "shera_observation"}:
            raise ValueError("noise_model must be 'legacy_numpy' or 'shera_observation'.")
        if self.negative_policy not in {"raise", "clip"}:
            raise ValueError("negative_policy must be 'raise' or 'clip'.")
        if self.noise_model == "legacy_numpy" and self.read_noise and self.read_noise_sigma is None:
            raise ValueError("read_noise_sigma is required when read_noise=True.")
        if self.read_noise_sigma is not None and float(self.read_noise_sigma) < 0.0:
            raise ValueError("read_noise_sigma must be >= 0 when provided.")
        if self.dark_current and self.exposure_time_s is None:
            raise ValueError("exposure_time_s is required when dark_current=True.")
        if self.exposure_time_s is not None and float(self.exposure_time_s) <= 0.0:
            raise ValueError("exposure_time_s must be > 0 when provided.")
        if self.detector_model is not None and str(self.detector_model) not in DETECTOR_SPECS:
            raise ValueError(f"Unsupported detector_model {self.detector_model!r}.")
        if float(self.bright_threshold) < 0.0:
            raise ValueError("bright_threshold must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready noise provenance."""
        return {
            "enabled": bool(self.enabled),
            "apply_to": self.apply_to,
            "noise_model": self.noise_model,
            "photon_noise": bool(self.photon_noise),
            "read_noise": bool(self.read_noise),
            "read_noise_sigma": self.read_noise_sigma,
            "dark_current": bool(self.dark_current),
            "detector_model": self.detector_model,
            "exposure_time_s": self.exposure_time_s,
            "bright_threshold": float(self.bright_threshold),
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
            noise_model=str(payload.get("noise_model", "legacy_numpy")),
            photon_noise=bool(payload.get("photon_noise", True)),
            read_noise=bool(payload.get("read_noise", False)),
            read_noise_sigma=payload.get("read_noise_sigma"),
            dark_current=bool(payload.get("dark_current", False)),
            detector_model=payload.get("detector_model"),
            exposure_time_s=payload.get("exposure_time_s"),
            bright_threshold=float(payload.get("bright_threshold", 100.0)),
            seed=int(payload.get("seed", 0)),
            training_dynamic=bool(payload.get("training_dynamic", True)),
            negative_policy=str(payload.get("negative_policy", "raise")),
        )


def _record_seed(base_seed: int, record_id: str | None, offset: int) -> int:
    payload = json.dumps([int(base_seed), record_id or "", int(offset)], separators=(",", ":"))
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (2**32)


def pair_noise_side_seeds(
    config: NoiseConfig | Mapping[str, Any] | None,
    *,
    pair_record_id: str | None,
    dynamic_seed_offset: int = 0,
) -> dict[str, int | None]:
    cfg = config if isinstance(config, NoiseConfig) else NoiseConfig.from_dict(config)
    if not cfg.enabled:
        return {"base_seed": None, "image_a_seed": None, "image_b_seed": None}
    base_seed = _record_seed(cfg.seed, pair_record_id, dynamic_seed_offset)
    return {
        "base_seed": int(base_seed),
        "image_a_seed": int(base_seed + 17) if cfg.apply_to in {"model", "both"} else None,
        "image_b_seed": int(base_seed + 31) if cfg.apply_to in {"observation", "both"} else None,
    }


def noise_config_identity(config: NoiseConfig | Mapping[str, Any] | None) -> dict[str, Any]:
    cfg = config if isinstance(config, NoiseConfig) else NoiseConfig.from_dict(config)
    return {
        "schema_version": "dluxshera_ml_noise_config/1",
        **cfg.to_dict(),
        "seed_policy": "sha256(base_seed,pair_record_id,dynamic_seed_offset) with side-specific offsets",
        "physical_ordering": "noise is applied to count-space image arrays before IntensityScaler.transform",
    }


def _detector_spec(config: NoiseConfig) -> DetectorSpec | None:
    if config.detector_model is None:
        return None
    return DETECTOR_SPECS[str(config.detector_model)]


def _apply_legacy_numpy_noise(image: np.ndarray, config: NoiseConfig, rng: np.random.Generator) -> np.ndarray:
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


def _apply_shera_observation_noise(image: np.ndarray, config: NoiseConfig, *, seed: int) -> np.ndarray:
    clean = np.asarray(image, dtype=np.float32)
    if config.negative_policy == "raise" and np.any(clean < 0.0):
        raise ValueError("Observation-noise inputs must be non-negative; use negative_policy='clip' intentionally.")
    if config.negative_policy == "clip":
        clean = np.maximum(clean, 0.0).astype(np.float32)
    noisy, _ = apply_observation_noise(
        jnp.asarray(clean),
        noise_cfg={
            "enabled": True,
            "photon_noise": bool(config.photon_noise),
            "read_noise": bool(config.read_noise),
            "dark_current": bool(config.dark_current),
        },
        rng_key=make_subkey(int(seed), "ml_pair_noise"),
        bright_threshold=float(config.bright_threshold),
        detector_spec=_detector_spec(config),
        exposure_time_s=config.exposure_time_s,
    )
    return np.asarray(noisy, dtype=np.float32)


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
    seeds = pair_noise_side_seeds(
        cfg,
        pair_record_id=pair_record_id,
        dynamic_seed_offset=dynamic_seed_offset,
    )
    if cfg.apply_to in {"model", "both"}:
        a = (
            _apply_shera_observation_noise(a, cfg, seed=int(seeds["image_a_seed"]))
            if cfg.noise_model == "shera_observation"
            else _apply_legacy_numpy_noise(a, cfg, np.random.default_rng(int(seeds["image_a_seed"])))
        )
    else:
        a = np.array(a, copy=True)
    if cfg.apply_to in {"observation", "both"}:
        b = (
            _apply_shera_observation_noise(b, cfg, seed=int(seeds["image_b_seed"]))
            if cfg.noise_model == "shera_observation"
            else _apply_legacy_numpy_noise(b, cfg, np.random.default_rng(int(seeds["image_b_seed"])))
        )
    else:
        b = np.array(b, copy=True)
    return a, b
