"""Lightweight perturbation and observation-noise utilities.

This module intentionally stays small: it centralises seeded subkey derivation,
simple additive perturbations for file-backed calibration maps, and the
prescribed Monte Carlo observation-noise path. It is not a general noise
framework; additional models (e.g., 1/f) can hook into the stubs below later.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Tuple

import jax.numpy as jnp
import jax.random as jr

Array = jnp.ndarray


# ---------------------------------------------------------------------------
# Seed / subkey helpers
# ---------------------------------------------------------------------------
def _hash_to_uint32(token: Any) -> int:
    """Deterministic 32-bit hash for fold-in tokens."""
    digest = hashlib.sha1(str(token).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0xFFFFFFFF


def make_subkey(base_seed: int, token: Any) -> jr.KeyArray:
    """Derive a PRNGKey from a base seed + stable token."""
    base_key = jr.PRNGKey(int(base_seed))
    return jr.fold_in(base_key, _hash_to_uint32(token))


def make_subseed(base_seed: int, token: Any) -> int:
    """Return an integer seed derived from a base seed + token."""
    subkey = make_subkey(base_seed, token)
    return int(jnp.asarray(subkey)[0])


# ---------------------------------------------------------------------------
# Calibration perturbations
# ---------------------------------------------------------------------------
def _extract_scale(config: Mapping[str, Any]) -> float | None:
    """Pick a perturbation scale from supported aliases."""
    for key in ("scale", "sigma", "rms"):
        if config.get(key) is not None:
            return float(config[key])
    return None


def perturb_array(
    arr: Array,
    *,
    model: str,
    scale: float,
    rng_key: jr.KeyArray,
) -> Array:
    """Apply additive perturbation to a calibration array."""
    if scale == 0.0:
        return arr
    if model == "gaussian":
        return arr + scale * jr.normal(rng_key, arr.shape, dtype=arr.dtype)
    if model == "uniform":
        return arr + scale * jr.uniform(
            rng_key,
            arr.shape,
            minval=-1.0,
            maxval=1.0,
            dtype=arr.dtype,
        )
    raise ValueError(f"Unsupported knowledge_error model: {model!r}")


def apply_knowledge_error(
    arr: Array,
    *,
    knowledge_cfg: Mapping[str, Any] | None,
    base_seed: int | None,
    token: Any,
) -> Tuple[Array, int | None]:
    """Optionally perturb ``arr`` according to knowledge_error config.

    Returns the perturbed array and the integer seed used (or None if no-op).
    """
    if not knowledge_cfg:
        return arr, None

    model = knowledge_cfg.get("model")
    if model is None:
        return arr, None

    scale = _extract_scale(knowledge_cfg)
    if scale is None or scale == 0.0:
        return arr, None

    seed = knowledge_cfg.get("seed", base_seed)
    if seed is None:
        # Without a seed we cannot guarantee reproducibility; skip.
        return arr, None

    rng_key = make_subkey(int(seed), token)
    perturbed = perturb_array(arr, model=str(model), scale=float(scale), rng_key=rng_key)
    return perturbed, int(jnp.asarray(rng_key)[0])


# ---------------------------------------------------------------------------
# Observation noise
# ---------------------------------------------------------------------------
def apply_observation_noise(
    image: Array,
    *,
    noise_cfg: Mapping[str, Any] | None,
    rng_key: jr.KeyArray,
    bright_threshold: float = 100.0,
) -> tuple[Array, Array]:
    """Add observation noise according to ``noise_cfg`` and return (data, var).

    Supported fields:
      - enabled / add_noise : bool
      - photon_noise : bool (defaults to True when enabled)
      - read_noise, dark_current : accepted but currently ignored (documented)
    """
    if noise_cfg is None:
        noise_cfg = {}

    enabled = noise_cfg.get("enabled")
    if enabled is None:
        enabled = noise_cfg.get("add_noise")
    if not enabled:
        var = jnp.maximum(image, 1.0)
        return image, var

    photon_noise = noise_cfg.get("photon_noise", True)
    read_noise = noise_cfg.get("read_noise", False)
    dark_current = noise_cfg.get("dark_current", False)

    if photon_noise:
        rng_key, subkey = jr.split(rng_key)
        if jnp.min(image) > bright_threshold:
            noisy = jnp.sqrt(image) * jr.normal(subkey, image.shape) + image
        else:
            noisy = jr.poisson(subkey, image).astype(image.dtype)
    else:
        noisy = image

    # Placeholder handling for read_noise/dark_current: not implemented yet.
    if read_noise or dark_current:
        # Explicitly document in trace rather than silently dropping.
        pass

    var = jnp.maximum(image, 1.0)
    return noisy, var


__all__ = [
    "apply_knowledge_error",
    "apply_observation_noise",
    "make_subkey",
    "make_subseed",
    "perturb_array",
]
