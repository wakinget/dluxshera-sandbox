from __future__ import annotations

from functools import lru_cache

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from dluxshera.components.detectors import GSENSE2020BSI_SPEC, HWK4123_SPEC, DetectorSpec
from dluxshera.utils.noise import apply_observation_noise, make_subkey

__all__ = [
    "NUMPY_BIT_GENERATOR",
    "NoiseConfig",
    "apply_pair_noise",
    "apply_pair_noise_batch",
    "apply_pair_noise_numpy",
    "effective_noise_implementation",
    "noise_config_identity",
    "numpy_noise_provenance",
    "numpy_supports",
    "pair_noise_side_seeds",
]

DETECTOR_SPECS = {
    "GSENSE2020BSI": GSENSE2020BSI_SPEC,
    "HWK4123": HWK4123_SPEC,
}

#: BitGenerator used by the NumPy photon-noise strategy.  Named explicitly
#: rather than relying on ``np.random.default_rng``, whose default algorithm is
#: a documented-but-changeable NumPy implementation detail; pinning it here is
#: what makes the recorded provenance meaningful.
NUMPY_BIT_GENERATOR = "PCG64"


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


def _numpy_generator(seed: int) -> np.random.Generator:
    """Return a ``PCG64`` generator for one logical noise draw.

    The integer seed is the one :func:`pair_noise_side_seeds` already derives,
    so the NumPy strategy inherits the existing logical seed schedule intact.
    NumPy routes an integer through :class:`numpy.random.SeedSequence`, which
    hashes it before seeding the state; that is why the schedule's small
    side-specific offsets (``+17`` / ``+31``) yield uncorrelated streams rather
    than near-identical ones.

    No global or per-process RNG state is consulted: every draw is a function
    of its own derived seed alone.  That is what makes the realization
    independent of batch size, worker count, and worker assignment.
    """
    return np.random.Generator(np.random.PCG64(int(seed)))


def _apply_shera_observation_noise_numpy(
    image: np.ndarray, config: NoiseConfig, *, seed: int
) -> np.ndarray:
    """NumPy counterpart of :func:`_apply_shera_observation_noise`.

    This samples the *same probability model* as the JAX reference -- including
    the identical whole-image-minimum bright-image branch -- from a different
    random-number stream.  It is deliberately not bitwise comparable to the
    reference; see :func:`numpy_noise_provenance`.
    """
    clean = np.asarray(image, dtype=np.float32)
    if config.negative_policy == "raise" and np.any(clean < 0.0):
        raise ValueError("Observation-noise inputs must be non-negative; use negative_policy='clip' intentionally.")
    if config.negative_policy == "clip":
        clean = np.maximum(clean, 0.0).astype(np.float32)
    if not config.photon_noise:
        return np.array(clean, copy=True)
    rng = _numpy_generator(seed)
    # Same predicate as src/dluxshera/utils/noise.py: the Gaussian branch is
    # selected only when every pixel exceeds the threshold.
    if float(np.min(clean)) > float(config.bright_threshold):
        noisy = np.sqrt(clean) * rng.standard_normal(clean.shape, dtype=np.float32) + clean
        return np.asarray(noisy, dtype=np.float32)
    return rng.poisson(clean).astype(np.float32)


def numpy_supports(config: NoiseConfig) -> bool:
    """Return ``True`` when the NumPy strategy covers ``config`` exactly.

    The NumPy sampler implements the SHERA photon-noise model only.  Read
    noise, dark current, the ``legacy_numpy`` model, and disabled noise are not
    reimplemented; those configurations fall back to the reference path rather
    than being approximated by a different detector model.
    """
    return bool(
        config.enabled
        and config.noise_model == "shera_observation"
        and not config.read_noise
        and not config.dark_current
    )


def effective_noise_implementation(config: NoiseConfig, requested: str) -> str:
    """Return the strategy that will actually run for ``config``.

    Provenance must record what executed, not what was asked for.  A requested
    strategy that cannot cover a configuration silently degrades to
    ``"reference"`` inside the sampler; this makes that visible to callers.
    """
    if requested == "numpy" and not numpy_supports(config):
        return "reference"
    return str(requested)


def numpy_noise_provenance() -> dict[str, Any]:
    """Return RNG provenance for the NumPy photon-noise strategy."""
    return {
        "bit_generator": NUMPY_BIT_GENERATOR,
        "seeding": "numpy.random.SeedSequence(derived_integer_seed) per logical draw",
        "numpy_version": str(np.__version__),
        "seed_schedule": "identical to the reference: pair_noise_side_seeds()",
        "distribution": "Poisson(lambda) per pixel; Gaussian only when min(image) > bright_threshold",
        "equivalence": (
            "same photon-noise probability model as the JAX reference, different "
            "RNG algorithm and different realized samples; not bitwise comparable"
        ),
        "determinism_scope": (
            "deterministic for a given logical (pair record, epoch, side, view) "
            "within this NumPy version and build; bitwise stability across NumPy "
            "versions or platforms is not claimed"
        ),
    }


def apply_pair_noise_numpy(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    config: NoiseConfig,
    records_offsets: list[tuple[str, int]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Apply SHERA photon noise with deterministic per-record NumPy RNGs.

    Each noisy image is drawn from its own ``PCG64`` generator seeded by the
    existing derived integer seed, so the realization depends only on the
    logical ``(pair record, epoch offset, side, view)`` identity -- never on
    batch size, batch composition, worker count, or worker assignment.  Input
    and output order match :func:`apply_pair_noise_batch` and the reference.

    Unsupported configurations use the reference path unchanged; see
    :func:`numpy_supports`.
    """
    if len(pairs) != len(records_offsets):
        raise ValueError("pairs and seed records must have matching lengths")
    if not numpy_supports(config):
        return [apply_pair_noise(a, b, config, pair_record_id=rid, dynamic_seed_offset=offset)
                for (a, b), (rid, offset) in zip(pairs, records_offsets)]
    result = []
    for (image_a, image_b), (rid, offset) in zip(pairs, records_offsets):
        seeds = pair_noise_side_seeds(config, pair_record_id=rid, dynamic_seed_offset=offset)
        sides = []
        for image, name in ((image_a, "image_a_seed"), (image_b, "image_b_seed")):
            seed = seeds[name]
            if seed is None:
                sides.append(np.asarray(image, dtype=np.float32).copy())
            else:
                sides.append(
                    _apply_shera_observation_noise_numpy(image, config, seed=int(seed))
                )
        result.append((sides[0], sides[1]))
    return result


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


# One compiled function per worker, cached by JAX for image/batch shape.
# Bright/read/dark paths intentionally retain the reference operations: XLA
# fusion of Gaussian arithmetic can change rounding relative to eager JAX.
def _poisson_batch_impl(images, keys):
    import jax
    import jax.random as jr
    return jax.vmap(lambda x, k: jr.poisson(jr.split(k)[1], x).astype(x.dtype))(images, keys)


@lru_cache(maxsize=1)
def _poisson_batch_kernel():
    import jax
    return jax.jit(_poisson_batch_impl, backend="cpu")


def apply_pair_noise_batch(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    config: NoiseConfig,
    records_offsets: list[tuple[str, int]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Exact reference seed schedule, batching only ordinary SHERA photon draws.

    Input/output order is unchanged. Unsupported configurations and individual
    bright images use the reference path. No scaler or dataset cache lives here.
    """
    import jax

    if config.enabled and config.noise_model == "shera_observation" and jax.default_backend() != "cpu":
        raise ValueError("Exact batched SHERA noise requires JAX CPU; select reference on other backends.")
    if len(pairs) != len(records_offsets):
        raise ValueError("pairs and seed records must have matching lengths")
    if not (config.enabled and config.noise_model == "shera_observation"
            and config.photon_noise and not config.read_noise and not config.dark_current
            and jax.config.jax_default_prng_impl == "threefry2x32"):
        return [apply_pair_noise(a, b, config, pair_record_id=rid, dynamic_seed_offset=offset)
                for (a, b), (rid, offset) in zip(pairs, records_offsets)]
    result = [[np.array(a, dtype=np.float32, copy=True), np.array(b, dtype=np.float32, copy=True)]
              for a, b in pairs]
    groups = {}
    for i, (rid, offset) in enumerate(records_offsets):
        seeds = pair_noise_side_seeds(config, pair_record_id=rid, dynamic_seed_offset=offset)
        for side, name in enumerate(("image_a_seed", "image_b_seed")):
            seed = seeds[name]
            if seed is None:
                continue
            x = result[i][side]
            if config.negative_policy == "raise" and np.any(x < 0):
                raise ValueError("Observation-noise inputs must be non-negative; use negative_policy='clip' intentionally.")
            if config.negative_policy == "clip":
                x = np.maximum(x, 0.0)
            if np.min(x) > config.bright_threshold:
                result[i][side] = _apply_shera_observation_noise(x, config, seed=int(seed))
            else:
                groups.setdefault(x.shape, []).append((i, side, x, int(seed)))
    for group in groups.values():
        # Pad mixed-brightness groups to a stable batch shape. Padding has its
        # own key and is discarded; no logical sample key depends on position.
        width = len(pairs) * (2 if config.apply_to == "both" else 1)
        images = np.stack([entry[2] for entry in group] +
                          [np.zeros_like(group[0][2])] * (width - len(group)))
        keys = jnp.stack([make_subkey(entry[3], "ml_pair_noise") for entry in group] +
                         [make_subkey(0, "ml_pair_noise")] * (width - len(group)))
        noisy = np.asarray(_poisson_batch_kernel()(images, keys), dtype=np.float32)
        for row, (i, side, _, _) in zip(noisy, group):
            result[i][side] = row
    return [tuple(pair) for pair in result]
