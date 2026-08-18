"""Lightweight perturbation and observation-noise utilities.

This module intentionally stays small: it centralises seeded subkey derivation,
simple additive perturbations for file-backed calibration maps, and the
prescribed Monte Carlo observation-noise path. It is not a general noise
framework; additional models (e.g., 1/f) can hook into the stubs below later.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Literal
from typing import Any, Mapping, Tuple, Optional

import jax.numpy as jnp
import jax.random as jr

from ..components.detectors import DetectorSpec

Array = jnp.ndarray

_MISSING = object()


@dataclass(frozen=True)
class NormalizedSubblockNoiseConfig:
    original: Any
    enabled: bool
    legacy_noise_mode: Literal["enabled", "disabled", "inherit"]
    shot_noise: bool
    photon_noise: bool
    read_noise: bool
    dark_current: bool
    use_detector_read_noise: bool
    read_noise_electrons: float | None
    read_noise_source: str
    use_detector_dark_current: bool
    dark_current_e_per_s: float | None
    dark_current_source: str
    variance_floor: float | Literal["auto"] | None
    variance_floor_source: str
    write_variance: bool
    use_render_variance: bool | Literal["auto"]
    use_render_variance_resolved: bool
    use_render_variance_source: str
    seed_policy: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render_template_noise_block(self) -> dict[str, Any] | None:
        if self.legacy_noise_mode == "inherit" and not isinstance(self.original, Mapping):
            return None
        return {
            "enabled": bool(self.enabled),
            "photon_noise": bool(self.photon_noise),
            "shot_noise": bool(self.shot_noise),
            "read_noise": bool(self.read_noise),
            "dark_current": bool(self.dark_current),
            "read_noise_electrons": self.read_noise_electrons,
            "dark_current_e_per_s": self.dark_current_e_per_s,
            "write_variance": bool(self.write_variance),
            "seed_policy": self.seed_policy,
        }


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _coalesce_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def normalize_noise_request(noise: Any) -> dict[str, Any]:
    """Normalize legacy scalar or structured render-noise requests.

    The returned mapping is the canonical review schema. Legacy
    ``enabled``/``disabled``/``inherit`` values remain accepted; ``enabled``
    means shot noise is requested while read noise and dark current are left
    off unless the structured request explicitly enables them.
    """

    defaults = {
        "enabled": False,
        "shot_noise": False,
        "read_noise": False,
        "dark_current": False,
        "use_detector_read_noise": True,
        "read_noise_electrons": None,
        "use_detector_dark_current": True,
        "dark_current_e_per_s": None,
        "variance_floor": None,
        "write_variance": True,
        "seed_policy": "from_subblock_noise_seed",
        "legacy_mode": None,
    }
    if isinstance(noise, Mapping):
        out = dict(defaults)
        out.update(dict(noise))
        enabled = _bool(out.get("enabled"), False)
        out["enabled"] = enabled
        requested_shot = noise.get("shot_noise", noise.get("photon_noise", enabled))
        out["shot_noise"] = _bool(
            requested_shot,
            enabled,
        ) if enabled else False
        out["photon_noise"] = bool(out["shot_noise"])
        out["read_noise"] = _bool(out.get("read_noise"), False) if enabled else False
        out["dark_current"] = _bool(out.get("dark_current"), False) if enabled else False
        out["use_detector_read_noise"] = _bool(out.get("use_detector_read_noise"), True)
        out["use_detector_dark_current"] = _bool(out.get("use_detector_dark_current"), True)
        out["write_variance"] = _bool(out.get("write_variance"), True)
        return out

    mode = str(noise if noise is not None else "disabled").strip().lower()
    out = dict(defaults)
    out["legacy_mode"] = mode
    if mode in {"enabled", "enable", "true", "on", "yes", "inherit"}:
        out["enabled"] = True
        out["shot_noise"] = True
        out["photon_noise"] = True
    elif mode in {"disabled", "disable", "false", "off", "none", "no", ""}:
        out["enabled"] = False
        out["shot_noise"] = False
        out["photon_noise"] = False
    elif mode in {"read", "read_noise", "shot_read", "photon_read"}:
        out["enabled"] = True
        out["shot_noise"] = mode != "read"
        out["photon_noise"] = out["shot_noise"]
        out["read_noise"] = True
    else:
        out["enabled"] = True
        out["shot_noise"] = True
        out["photon_noise"] = True
    return out


def detector_spec_for_model(model: str | None) -> DetectorSpec:
    from ..components.detectors import GSENSE2020BSI_SPEC, HWK4123_SPEC

    return {
        "GSENSE2020BSI": GSENSE2020BSI_SPEC,
        "HWK4123": HWK4123_SPEC,
    }.get(str(model), GSENSE2020BSI_SPEC)


def _detector_block(system_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    system = _coalesce_mapping(system_cfg)
    if isinstance(system.get("system"), Mapping):
        system = dict(system["system"])
    return _coalesce_mapping(system.get("detector"))


def _first_numeric(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    for key in keys:
        value = mapping.get(key, _MISSING)
        if value is _MISSING or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number, key
    return None, None


def _first_layer_numeric(detector: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    layers = detector.get("layers", [])
    if not isinstance(layers, list):
        return None, None
    for idx, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            continue
        value, key = _first_numeric(layer, keys)
        if value is not None:
            name = layer.get("name", f"layer_{idx}")
            return value, f"detector.layers[{idx}]({name}).{key}"
    return None, None


def resolve_detector_noise_spec(
    system_cfg: Mapping[str, Any] | None,
    noise_cfg: Mapping[str, Any] | Any,
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Resolve render-noise amplitudes and provenance for a system config."""

    normalized = normalize_noise_request(noise_cfg)
    detector = _detector_block(system_cfg)
    spec = detector_spec_for_model(detector.get("model"))
    warnings_out: list[str] = []

    def resolve_read() -> tuple[float | None, str]:
        if normalized.get("read_noise_electrons") is not None:
            return float(normalized["read_noise_electrons"]), "config_override"
        if not _bool(normalized.get("use_detector_read_noise"), True):
            return None, "missing"
        value, key = _first_numeric(detector, ("read_noise_electrons", "read_noise_e", "read_noise"))
        if value is not None:
            return value, f"detector_spec:{key}"
        value, key = _first_layer_numeric(detector, ("read_noise_electrons", "read_noise_e", "read_noise"))
        if value is not None:
            return value, f"detector_layer:{key}"
        if spec.read_noise is not None:
            return float(spec.read_noise), "detector_spec"
        return None, "missing"

    def resolve_dark() -> tuple[float | None, str]:
        if normalized.get("dark_current_e_per_s") is not None:
            return float(normalized["dark_current_e_per_s"]), "config_override"
        if not _bool(normalized.get("use_detector_dark_current"), True):
            return None, "missing"
        value, key = _first_numeric(detector, ("dark_current_e_per_s", "dark_current"))
        if value is not None:
            return value, f"detector_spec:{key}"
        value, key = _first_layer_numeric(detector, ("dark_current_e_per_s", "dark_current"))
        if value is not None:
            return value, f"detector_layer:{key}"
        if spec.dark_current is not None:
            return float(spec.dark_current), "detector_spec"
        return None, "missing"

    read_noise, read_source = resolve_read()
    dark_current, dark_source = resolve_dark()

    exposure = None
    system = _coalesce_mapping(system_cfg)
    if isinstance(system.get("system"), Mapping):
        system = dict(system["system"])
    source = _coalesce_mapping(system.get("source"))
    if source.get("exposure_time_s") is not None:
        exposure = float(source["exposure_time_s"])
    elif normalized.get("exposure_time_s") is not None:
        exposure = float(normalized["exposure_time_s"])

    if normalized["enabled"] and normalized["read_noise"] and read_noise is None:
        warnings_out.append("read_noise=true but no read-noise amplitude was found in config or detector spec.")
    if normalized["enabled"] and normalized["dark_current"] and dark_current is None:
        warnings_out.append("dark_current=true but no dark-current amplitude was found in config or detector spec.")
    if normalized["enabled"] and normalized["dark_current"] and exposure is None:
        warnings_out.append("dark_current=true but exposure_time_s was not found for dark-current scaling.")

    if strict and warnings_out:
        raise ValueError("; ".join(warnings_out))

    return {
        "noise_request_normalized": normalized,
        "detector_model": detector.get("model", spec.model_name),
        "read_noise_electrons": read_noise,
        "read_noise_source": read_source,
        "dark_current_e_per_s": dark_current,
        "dark_current_source": dark_source,
        "exposure_time_s": exposure,
        "shot_noise_enabled": bool(normalized["enabled"] and normalized["shot_noise"]),
        "read_noise_enabled": bool(normalized["enabled"] and normalized["read_noise"]),
        "dark_current_enabled": bool(normalized["enabled"] and normalized["dark_current"]),
        "warnings": warnings_out,
    }


def normalize_subblock_noise_config(
    subblock_cfg: Mapping[str, Any],
    *,
    detector_cfg: Mapping[str, Any] | None = None,
    exposure_time_s: float | None = None,
    strict: bool = False,
) -> NormalizedSubblockNoiseConfig:
    """Normalize campaign subblock noise controls for render/inference paths."""

    subblock = dict(subblock_cfg or {})
    original = subblock.get("noise", "disabled")
    noise_model = subblock.get("noise_model")
    if (
        not isinstance(original, Mapping)
        and str(original).strip().lower() == "inherit"
        and isinstance(noise_model, Mapping)
        and isinstance(noise_model.get("original_request"), Mapping)
    ):
        original = noise_model["original_request"]
    structured = isinstance(original, Mapping)
    legacy_raw = str(original if not structured else "").strip().lower()
    warnings_out: list[str] = []
    if structured:
        normalized = normalize_noise_request(original)
        legacy_mode: Literal["enabled", "disabled", "inherit"] = "inherit"
    elif legacy_raw in {"inherit", ""}:
        normalized = normalize_noise_request("inherit")
        legacy_mode = "inherit"
    elif legacy_raw in {"enabled", "enable", "true", "on", "yes"}:
        normalized = normalize_noise_request("enabled")
        legacy_mode = "enabled"
    else:
        normalized = normalize_noise_request("disabled")
        legacy_mode = "disabled"

    info = resolve_detector_noise_spec(
        detector_cfg,
        {**normalized, "exposure_time_s": exposure_time_s},
        strict=False,
    )
    warnings_out.extend(str(item) for item in info.get("warnings", []))
    read_noise = info.get("read_noise_electrons")
    dark_current = info.get("dark_current_e_per_s")
    read_source = str(info.get("read_noise_source", "missing"))
    dark_source = str(info.get("dark_current_source", "missing"))

    enabled = bool(normalized.get("enabled", False))
    shot_noise = bool(enabled and normalized.get("shot_noise", False))
    read_enabled = bool(enabled and normalized.get("read_noise", False))
    dark_enabled = bool(enabled and normalized.get("dark_current", False))
    if not enabled:
        read_noise = None
        dark_current = None
        read_source = "disabled"
        dark_source = "disabled"
    elif not read_enabled:
        read_source = "disabled"
        read_noise = None
    elif read_noise is None:
        read_source = "missing"
    if not enabled or not dark_enabled:
        dark_source = "disabled"
        dark_current = None
    elif dark_current is None:
        dark_source = "missing"

    explicit_floor = None
    floor_source = "default"
    noise_floor = normalized.get("variance_floor")
    legacy_floor = subblock.get("variance_floor")
    if noise_floor is not None:
        explicit_floor = noise_floor
        floor_source = "experiment.subblocks.noise.variance_floor"
        if legacy_floor is not None and legacy_floor != noise_floor:
            warnings_out.append(
                "experiment.subblocks.variance_floor is deprecated and disagrees with "
                "experiment.subblocks.noise.variance_floor; using the nested canonical value."
            )
    elif legacy_floor is not None:
        explicit_floor = legacy_floor
        floor_source = "experiment.subblocks.variance_floor"
        warnings_out.append(
            "experiment.subblocks.variance_floor is deprecated; use "
            "experiment.subblocks.noise.variance_floor."
        )
    variance_floor: float | Literal["auto"] | None
    if explicit_floor == "auto":
        if read_enabled and read_noise is None:
            warnings_out.append("variance_floor=auto requires resolved read_noise_electrons.")
            variance_floor = "auto"
        elif dark_enabled and (dark_current is None or exposure_time_s is None):
            warnings_out.append("variance_floor=auto requires dark_current_e_per_s and exposure_time_s.")
            variance_floor = "auto"
        else:
            read_term = float(read_noise or 0.0) ** 2 if read_enabled else 0.0
            dark_term = float(dark_current or 0.0) * float(exposure_time_s or 0.0) if dark_enabled else 0.0
            variance_floor = float(read_term + dark_term)
            floor_source = "auto"
    elif explicit_floor is None:
        variance_floor = None
    else:
        variance_floor = float(explicit_floor)

    use_render_raw = subblock.get("use_render_variance", normalized.get("use_render_variance", "auto"))
    if isinstance(use_render_raw, str) and use_render_raw.strip().lower() == "auto":
        use_render_variance: bool | Literal["auto"] = "auto"
        use_render_resolved = bool(enabled and normalized.get("write_variance", True))
        use_render_source = "auto:write_variance" if use_render_resolved else "auto:false"
    else:
        use_render_variance = _bool(use_render_raw, False)
        use_render_resolved = bool(use_render_variance)
        use_render_source = "explicit"

    if read_enabled and read_noise is None:
        warnings_out.append("read_noise=true but no read-noise amplitude was resolved.")
    if dark_enabled and dark_current is None:
        warnings_out.append("dark_current=true but no dark-current amplitude was resolved.")
    if dark_enabled and exposure_time_s is None:
        warnings_out.append("dark_current=true but exposure_time_s is missing.")
    if strict and warnings_out:
        raise ValueError("; ".join(warnings_out))

    return NormalizedSubblockNoiseConfig(
        original=original,
        enabled=enabled,
        legacy_noise_mode=legacy_mode,
        shot_noise=shot_noise,
        photon_noise=shot_noise,
        read_noise=read_enabled,
        dark_current=dark_enabled,
        use_detector_read_noise=bool(normalized.get("use_detector_read_noise", True)),
        read_noise_electrons=None if read_noise is None else float(read_noise),
        read_noise_source=read_source,
        use_detector_dark_current=bool(normalized.get("use_detector_dark_current", True)),
        dark_current_e_per_s=None if dark_current is None else float(dark_current),
        dark_current_source=dark_source,
        variance_floor=variance_floor,
        variance_floor_source=floor_source,
        write_variance=bool(normalized.get("write_variance", True)),
        use_render_variance=use_render_variance,
        use_render_variance_resolved=use_render_resolved,
        use_render_variance_source=use_render_source,
        seed_policy=str(normalized.get("seed_policy", "from_subblock_noise_seed")),
        warnings=tuple(dict.fromkeys(warnings_out)),
    )


def expected_noise_variance(
    image: Array,
    *,
    noise_cfg: Mapping[str, Any] | Any,
    detector_noise: Mapping[str, Any] | None = None,
    detector_spec: DetectorSpec | None = None,
    exposure_time_s: float | None = None,
    variance_floor: float | None = None,
) -> Array:
    """Compute render-noise variance for enabled terms.

    ``variance_floor`` is optional and should only be passed for inference
    likelihood diagnostics. Render variance itself is returned without a floor.
    """

    normalized = normalize_noise_request(noise_cfg)
    variance = jnp.zeros_like(image)
    if not normalized["enabled"]:
        return variance if variance_floor is None else jnp.maximum(variance, float(variance_floor))
    if normalized["shot_noise"]:
        variance = variance + jnp.maximum(image, 0.0)

    info = dict(detector_noise or {})
    read = info.get("read_noise_electrons")
    dark = info.get("dark_current_e_per_s")
    exposure = info.get("exposure_time_s", exposure_time_s)
    if read is None and detector_spec is not None:
        read = detector_spec.read_noise
    if dark is None and detector_spec is not None:
        dark = detector_spec.dark_current

    if normalized["read_noise"] and read is not None:
        variance = variance + float(read) ** 2
    if normalized["dark_current"] and dark is not None and exposure is not None:
        variance = variance + max(float(dark) * float(exposure), 0.0)
    if variance_floor is not None:
        variance = jnp.maximum(variance, float(variance_floor))
    return variance


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
    clip_min = knowledge_cfg.get("clip_min")
    clip_max = knowledge_cfg.get("clip_max")
    if clip_min is not None or clip_max is not None:
        min_value = -jnp.inf if clip_min is None else float(clip_min)
        max_value = jnp.inf if clip_max is None else float(clip_max)
        perturbed = jnp.clip(perturbed, min_value, max_value)
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
    detector_spec: Optional[DetectorSpec] = None,
    exposure_time_s: Optional[float] = None,
) -> tuple[Array, Array]:
    """Add observation noise according to ``noise_cfg`` and return (data, var).

    Supported fields:
      - enabled / add_noise : bool
      - photon_noise : bool (defaults to True when enabled)
      - read_noise : bool (uses detector_spec.read_noise as Gaussian sigma)
      - dark_current : bool (Gaussian sigma derived from detector_spec.dark_current and exposure_time_s)
    """
    if noise_cfg is None:
        noise_cfg = {}

    noise_cfg = normalize_noise_request(noise_cfg)

    enabled = noise_cfg.get("enabled")
    if enabled is None:
        enabled = noise_cfg.get("add_noise")
    if not enabled:
        var = jnp.zeros_like(image)
        return image, var

    photon_noise = noise_cfg.get("shot_noise", noise_cfg.get("photon_noise", True))
    read_noise_enabled = noise_cfg.get("read_noise", False)
    dark_current_enabled = noise_cfg.get("dark_current", False)

    total_var = jnp.zeros_like(image)

    if photon_noise:
        rng_key, subkey = jr.split(rng_key)
        shot_mean = jnp.maximum(image, 0.0)
        if jnp.min(image) > bright_threshold:
            noisy = jnp.sqrt(image) * jr.normal(subkey, image.shape) + image
        else:
            noisy = jr.poisson(subkey, shot_mean).astype(image.dtype)
        total_var = total_var + shot_mean
    else:
        noisy = image

    if read_noise_enabled:
        if detector_spec is None or detector_spec.read_noise is None:
            raise ValueError("read_noise enabled but detector spec is missing read_noise.")
        sigma_read = float(detector_spec.read_noise)
        rng_key, subkey = jr.split(rng_key)
        noisy = noisy + sigma_read * jr.normal(subkey, noisy.shape, dtype=noisy.dtype)
        total_var = total_var + sigma_read**2

    if dark_current_enabled:
        if detector_spec is None or detector_spec.dark_current is None:
            raise ValueError("dark_current enabled but detector spec is missing dark_current.")
        if exposure_time_s is None:
            raise ValueError("dark_current enabled but exposure_time_s is not provided.")
        dc_rate = float(detector_spec.dark_current)
        dc_var = max(dc_rate * float(exposure_time_s), 0.0)
        rng_key, subkey = jr.split(rng_key)
        noisy = noisy + jnp.sqrt(dc_var) * jr.normal(subkey, noisy.shape, dtype=noisy.dtype)
        total_var = total_var + dc_var

    return noisy, total_var


__all__ = [
    "apply_knowledge_error",
    "apply_observation_noise",
    "detector_spec_for_model",
    "expected_noise_variance",
    "make_subkey",
    "make_subseed",
    "normalize_noise_request",
    "perturb_array",
    "resolve_detector_noise_spec",
]
