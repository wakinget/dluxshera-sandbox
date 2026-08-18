"""Trace-construction helpers for observation sub-block rendering.

This helper layer generates explicit per-frame trace rows consumed by
``examples/recipes/observation_subblock.py``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np

from .obs_subblock_keys import (
    canonical_obs_subblock_varying_keys,
    parse_obs_subblock_varying_keys,
    validate_supported_obs_subblock_key_addresses,
)


SUPPORTED_TRACE_EFFECT_KINDS: tuple[str, ...] = (
    "constant_offset",
    "linear_drift",
    "random_walk",
    "iid_jitter",
    "explicit",
)

@dataclass(frozen=True)
class ObsSubblockKeyTracePlan:
    """Per-key trace plan for one varying key."""

    key: str
    base: float | None
    effects: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ObsSubblockTraceBuildPlan:
    """Normalized trace-generation plan."""

    n_frames: int
    dt_s: float
    seed: int | None
    varying_keys: tuple[str, ...]
    key_plans: Mapping[str, ObsSubblockKeyTracePlan]

    @property
    def time_s(self) -> np.ndarray:
        """Return frame times as ``0, dt, 2*dt, ...``."""

        return np.arange(self.n_frames, dtype=float) * float(self.dt_s)


def _require_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping/dict.")
    return dict(value)


def _require_int(value: Any, *, path: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    return int(value)


def _require_finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path} must be finite.")
    return parsed


def _normalize_explicit_values(
    *,
    values: Any,
    n_frames: int,
    path: str,
) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"{path} must be a list.")
    if len(values) != n_frames:
        raise ValueError(
            f"{path} length must equal n_frames ({n_frames}), got {len(values)}."
        )
    return [
        _require_finite_float(item, path=f"{path}[{idx}]")
        for idx, item in enumerate(values)
    ]


def _normalize_effect(
    *,
    payload: Mapping[str, Any],
    key: str,
    effect_index: int,
    n_frames: int,
) -> dict[str, Any]:
    item = _require_mapping(payload, path=f"plan.{key}.effects[{effect_index}]")
    kind_value = item.get("kind")
    if not isinstance(kind_value, str) or not kind_value.strip():
        raise ValueError(
            f"plan.{key}.effects[{effect_index}].kind must be a non-empty string."
        )
    kind = kind_value.strip()
    if kind not in SUPPORTED_TRACE_EFFECT_KINDS:
        raise ValueError(
            f"plan.{key}.effects[{effect_index}].kind must be one of "
            f"{SUPPORTED_TRACE_EFFECT_KINDS}, got {kind!r}."
        )

    path_prefix = f"plan.{key}.effects[{effect_index}]"
    if kind == "constant_offset":
        return {
            "kind": kind,
            "offset": _require_finite_float(item.get("offset"), path=f"{path_prefix}.offset"),
        }
    if kind == "linear_drift":
        return {
            "kind": kind,
            "start": _require_finite_float(item.get("start"), path=f"{path_prefix}.start"),
            "rate_per_s": _require_finite_float(
                item.get("rate_per_s"), path=f"{path_prefix}.rate_per_s"
            ),
        }
    if kind == "random_walk":
        sigma_step = _require_finite_float(
            item.get("sigma_step"),
            path=f"{path_prefix}.sigma_step",
        )
        if sigma_step < 0.0:
            raise ValueError(f"{path_prefix}.sigma_step must be >= 0.")
        return {
            "kind": kind,
            "start": _require_finite_float(item.get("start"), path=f"{path_prefix}.start"),
            "sigma_step": sigma_step,
        }
    if kind == "iid_jitter":
        sigma = _require_finite_float(item.get("sigma"), path=f"{path_prefix}.sigma")
        if sigma < 0.0:
            raise ValueError(f"{path_prefix}.sigma must be >= 0.")
        return {
            "kind": kind,
            "center": _require_finite_float(item.get("center"), path=f"{path_prefix}.center"),
            "sigma": sigma,
        }

    return {
        "kind": kind,
        "values": _normalize_explicit_values(
            values=item.get("values"),
            n_frames=n_frames,
            path=f"{path_prefix}.values",
        ),
    }


def _normalize_trace_plan_entry(
    *,
    key: str,
    payload: Mapping[str, Any],
    n_frames: int,
) -> ObsSubblockKeyTracePlan:
    item = _require_mapping(payload, path=f"plan.{key}")
    base_value = item.get("base")
    base = (
        None
        if base_value is None
        else _require_finite_float(base_value, path=f"plan.{key}.base")
    )
    effects_raw = item.get("effects", [])
    if effects_raw is None:
        effects_raw = []
    if not isinstance(effects_raw, list):
        raise ValueError(f"plan.{key}.effects must be a list when provided.")
    effects = tuple(
        _normalize_effect(
            payload=effect_payload,
            key=key,
            effect_index=idx,
            n_frames=n_frames,
        )
        for idx, effect_payload in enumerate(effects_raw)
    )
    return ObsSubblockKeyTracePlan(key=key, base=base, effects=effects)


def _normalize_general_plan(
    cfg: dict[str, Any],
    *,
    n_frames: int,
) -> tuple[tuple[str, ...], dict[str, ObsSubblockKeyTracePlan]]:
    plan_cfg = _require_mapping(
        cfg.get("plan"),
        path="experiment.trace.plan",
    )
    varying_keys_value = cfg.get("varying_keys")
    if varying_keys_value is None:
        requested_keys = list(plan_cfg.keys())
    else:
        if not isinstance(varying_keys_value, list):
            raise ValueError(
                "experiment.trace.varying_keys must be a "
                "list[str] when provided."
            )
        requested_keys = list(varying_keys_value)

    addresses = parse_obs_subblock_varying_keys(requested_keys)
    validate_supported_obs_subblock_key_addresses(addresses)
    varying_keys = canonical_obs_subblock_varying_keys(addresses)
    if not varying_keys:
        raise ValueError("At least one varying key is required.")

    missing = [key for key in varying_keys if key not in plan_cfg]
    if missing:
        raise ValueError(
            "experiment.trace.plan is missing entries for: "
            + ", ".join(missing)
        )
    extra = sorted(key for key in plan_cfg if key not in varying_keys)
    if extra:
        raise ValueError(
            "experiment.trace.plan has keys not declared in "
            "varying_keys: " + ", ".join(extra)
        )

    key_plans = {
        key: _normalize_trace_plan_entry(
            key=key,
            payload=_require_mapping(plan_cfg[key], path=f"plan.{key}"),
            n_frames=n_frames,
        )
        for key in varying_keys
    }
    return varying_keys, key_plans


def build_obs_subblock_trace_plan(
    trace_cfg: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> ObsSubblockTraceBuildPlan:
    """Validate and normalize subblock trace-generation config."""

    cfg = _require_mapping(trace_cfg, path="experiment.trace")
    n_frames = _require_int(
        cfg.get("n_frames"),
        path="experiment.trace.n_frames",
    )
    if n_frames < 1:
        raise ValueError("experiment.trace.n_frames must be >= 1.")

    dt_s = _require_finite_float(
        cfg.get("dt_s"),
        path="experiment.trace.dt_s",
    )
    if dt_s <= 0.0:
        raise ValueError("experiment.trace.dt_s must be > 0.")

    seed_value = cfg.get("seed", seed)
    normalized_seed: int | None
    if seed_value is None:
        normalized_seed = None
    else:
        if not isinstance(seed_value, int):
            raise ValueError(
                "experiment.trace.seed must be an integer when provided."
            )
        normalized_seed = int(seed_value)

    if "plan" in cfg:
        varying_keys, key_plans = _normalize_general_plan(cfg, n_frames=n_frames)
    else:
        raise ValueError(
            "experiment.trace must define 'plan'."
        )

    return ObsSubblockTraceBuildPlan(
        n_frames=n_frames,
        dt_s=float(dt_s),
        seed=normalized_seed,
        varying_keys=varying_keys,
        key_plans=key_plans,
    )


def resolve_obs_subblock_trace_anchors(
    plan: ObsSubblockTraceBuildPlan,
    *,
    nominal_anchors: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Resolve per-key anchor values from explicit base or nominal anchors."""

    anchors: dict[str, float] = {}
    for key in plan.varying_keys:
        key_plan = plan.key_plans[key]
        if key_plan.base is not None:
            anchors[key] = float(key_plan.base)
            continue
        if nominal_anchors is not None and key in nominal_anchors:
            anchors[key] = _require_finite_float(
                nominal_anchors[key],
                path=f"nominal_anchors.{key}",
            )
            continue
        raise ValueError(
            f"plan.{key}.base is required when no nominal anchor is available."
        )
    return anchors


def _derive_effect_seed(
    *,
    seed: int | None,
    key: str,
    effect_index: int,
    effect_kind: str,
) -> int | None:
    if seed is None:
        return None
    payload = f"{seed}|{key}|{effect_index}|{effect_kind}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def _generate_effect_series(
    *,
    effect: Mapping[str, Any],
    n_frames: int,
    time_s: np.ndarray,
    seed: int | None,
    key: str,
    effect_index: int,
) -> np.ndarray:
    kind = str(effect["kind"])
    if kind == "constant_offset":
        return np.full(n_frames, float(effect["offset"]), dtype=float)
    if kind == "linear_drift":
        return float(effect["start"]) + (float(effect["rate_per_s"]) * time_s)

    if kind == "random_walk":
        rng = np.random.default_rng(
            _derive_effect_seed(
                seed=seed,
                key=key,
                effect_index=effect_index,
                effect_kind=kind,
            )
        )
        values = np.empty(n_frames, dtype=float)
        values[0] = float(effect["start"])
        if n_frames > 1:
            sigma_step = float(effect["sigma_step"])
            steps = rng.normal(loc=0.0, scale=sigma_step, size=n_frames - 1)
            values[1:] = float(effect["start"]) + np.cumsum(steps)
        return values

    if kind == "iid_jitter":
        rng = np.random.default_rng(
            _derive_effect_seed(
                seed=seed,
                key=key,
                effect_index=effect_index,
                effect_kind=kind,
            )
        )
        return rng.normal(
            loc=float(effect["center"]),
            scale=float(effect["sigma"]),
            size=n_frames,
        )

    return np.asarray(effect["values"], dtype=float)


def generate_obs_subblock_trace_rows(
    plan: ObsSubblockTraceBuildPlan,
    *,
    anchors: Mapping[str, Any],
) -> list[dict[str, float | int]]:
    """Generate explicit trace rows from plan + resolved anchors."""

    time_s = plan.time_s
    n_frames = int(plan.n_frames)
    series_by_key: dict[str, np.ndarray] = {}
    for key in plan.varying_keys:
        anchor_value = _require_finite_float(anchors.get(key), path=f"anchors.{key}")
        key_plan = plan.key_plans[key]
        series = np.full(n_frames, anchor_value, dtype=float)
        for effect_index, effect in enumerate(key_plan.effects):
            series = series + _generate_effect_series(
                effect=effect,
                n_frames=n_frames,
                time_s=time_s,
                seed=plan.seed,
                key=key,
                effect_index=effect_index,
            )
        if not np.isfinite(series).all():
            raise ValueError(f"Generated non-finite values for key {key!r}.")
        series_by_key[key] = series

    rows: list[dict[str, float | int]] = []
    for frame_index, frame_time in enumerate(time_s):
        row: dict[str, float | int] = {
            "frame_index": int(frame_index),
            "time_s": float(frame_time),
        }
        for key in plan.varying_keys:
            row[key] = float(series_by_key[key][frame_index])
        rows.append(row)
    return rows


__all__ = [
    "ObsSubblockKeyTracePlan",
    "ObsSubblockTraceBuildPlan",
    "SUPPORTED_TRACE_EFFECT_KINDS",
    "build_obs_subblock_trace_plan",
    "generate_obs_subblock_trace_rows",
    "resolve_obs_subblock_trace_anchors",
]
