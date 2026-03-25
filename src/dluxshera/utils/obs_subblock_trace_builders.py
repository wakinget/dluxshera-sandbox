"""Trace-construction helpers for observation sub-block rendering.

This helper layer generates canonical explicit per-frame traces that are
consumed by ``examples/recipes/observation_subblock.py``. It intentionally
keeps motion construction separate from rendering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np

from .obs_subblock_trace import APPLIED_V1_VARYING_KEYS


SUPPORTED_TRACE_MODES: tuple[str, ...] = (
    "explicit",
    "linear_drift",
    "random_walk",
    "iid_jitter",
)


@dataclass(frozen=True)
class ObsSubblockTraceBuildPlan:
    """Normalized trace-generation plan.

    Parameters
    ----------
    n_frames : int
        Number of frames to generate.
    dt_s : float
        Frame cadence in seconds.
    seed : int | None
        Optional deterministic seed for stochastic modes.
    key_specs : dict[str, dict[str, Any]]
        Per-key normalized generation specs for
        ``source.x_position_as``, ``source.y_position_as``,
        and ``source.position_angle_deg``.
    """

    n_frames: int
    dt_s: float
    seed: int | None
    key_specs: dict[str, dict[str, Any]]

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


def _normalize_explicit_spec(
    payload: dict[str, Any],
    *,
    key: str,
    n_frames: int,
) -> dict[str, Any]:
    values = payload.get("values")
    if not isinstance(values, list):
        raise ValueError(f"{key}: explicit mode requires a list field 'values'.")
    if len(values) != n_frames:
        raise ValueError(
            f"{key}: explicit.values length must equal n_frames ({n_frames}), "
            f"got {len(values)}."
        )
    normalized = [
        _require_finite_float(item, path=f"{key}.values[{idx}]")
        for idx, item in enumerate(values)
    ]
    return {"mode": "explicit", "values": normalized}


def _normalize_linear_drift_spec(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    start = _require_finite_float(payload.get("start"), path=f"{key}.start")
    rate_per_s = _require_finite_float(
        payload.get("rate_per_s"), path=f"{key}.rate_per_s"
    )
    return {"mode": "linear_drift", "start": start, "rate_per_s": rate_per_s}


def _normalize_random_walk_spec(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    start = _require_finite_float(payload.get("start"), path=f"{key}.start")
    sigma_step = _require_finite_float(payload.get("sigma_step"), path=f"{key}.sigma_step")
    if sigma_step < 0.0:
        raise ValueError(f"{key}.sigma_step must be >= 0.")
    return {"mode": "random_walk", "start": start, "sigma_step": sigma_step}


def _normalize_iid_jitter_spec(payload: dict[str, Any], *, key: str) -> dict[str, Any]:
    center = _require_finite_float(payload.get("center"), path=f"{key}.center")
    sigma = _require_finite_float(payload.get("sigma"), path=f"{key}.sigma")
    if sigma < 0.0:
        raise ValueError(f"{key}.sigma must be >= 0.")
    return {"mode": "iid_jitter", "center": center, "sigma": sigma}


def _normalize_key_spec(
    key: str,
    *,
    payload: dict[str, Any],
    n_frames: int,
) -> dict[str, Any]:
    mode_value = payload.get("mode")
    if not isinstance(mode_value, str) or not mode_value.strip():
        raise ValueError(f"{key}.mode must be a non-empty string.")
    mode = mode_value.strip()
    if mode not in SUPPORTED_TRACE_MODES:
        raise ValueError(
            f"{key}.mode must be one of {SUPPORTED_TRACE_MODES}, got {mode!r}."
        )
    if mode == "explicit":
        return _normalize_explicit_spec(payload, key=key, n_frames=n_frames)
    if mode == "linear_drift":
        return _normalize_linear_drift_spec(payload, key=key)
    if mode == "random_walk":
        return _normalize_random_walk_spec(payload, key=key)
    return _normalize_iid_jitter_spec(payload, key=key)


def build_obs_subblock_trace_plan(
    trace_cfg: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> ObsSubblockTraceBuildPlan:
    """Validate and normalize a trace-generation config block."""

    cfg = _require_mapping(trace_cfg, path="experiment.observation_subblock_trace")
    n_frames = _require_int(cfg.get("n_frames"), path="experiment.observation_subblock_trace.n_frames")
    if n_frames < 1:
        raise ValueError("experiment.observation_subblock_trace.n_frames must be >= 1.")

    dt_s = _require_finite_float(
        cfg.get("dt_s"),
        path="experiment.observation_subblock_trace.dt_s",
    )
    if dt_s <= 0.0:
        raise ValueError("experiment.observation_subblock_trace.dt_s must be > 0.")

    key_specs_value = cfg.get("keys")
    key_specs_raw = _require_mapping(
        key_specs_value,
        path="experiment.observation_subblock_trace.keys",
    )
    unknown_keys = sorted(set(key_specs_raw) - set(APPLIED_V1_VARYING_KEYS))
    if unknown_keys:
        raise ValueError(
            "experiment.observation_subblock_trace.keys contains unsupported keys: "
            + ", ".join(unknown_keys)
        )
    missing_keys = [key for key in APPLIED_V1_VARYING_KEYS if key not in key_specs_raw]
    if missing_keys:
        raise ValueError(
            "experiment.observation_subblock_trace.keys is missing required v1 keys: "
            + ", ".join(missing_keys)
        )

    seed_value = cfg.get("seed", seed)
    normalized_seed: int | None
    if seed_value is None:
        normalized_seed = None
    else:
        if not isinstance(seed_value, int):
            raise ValueError(
                "experiment.observation_subblock_trace.seed must be an integer when provided."
            )
        normalized_seed = int(seed_value)

    normalized_key_specs = {
        key: _normalize_key_spec(
            key,
            payload=_require_mapping(
                key_specs_raw[key],
                path=f"experiment.observation_subblock_trace.keys.{key}",
            ),
            n_frames=n_frames,
        )
        for key in APPLIED_V1_VARYING_KEYS
    }
    return ObsSubblockTraceBuildPlan(
        n_frames=n_frames,
        dt_s=float(dt_s),
        seed=normalized_seed,
        key_specs=normalized_key_specs,
    )


def _spawn_key_rngs(seed: int | None) -> dict[str, np.random.Generator]:
    if seed is None:
        return {
            key: np.random.default_rng()
            for key in APPLIED_V1_VARYING_KEYS
        }
    seed_seq = np.random.SeedSequence(seed)
    child_seqs = seed_seq.spawn(len(APPLIED_V1_VARYING_KEYS))
    return {
        key: np.random.default_rng(child_seq)
        for key, child_seq in zip(APPLIED_V1_VARYING_KEYS, child_seqs)
    }


def _generate_key_values(
    *,
    mode_spec: dict[str, Any],
    time_s: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    mode = str(mode_spec["mode"])
    n_frames = int(time_s.shape[0])
    if mode == "explicit":
        return np.asarray(mode_spec["values"], dtype=float)
    if mode == "linear_drift":
        start = float(mode_spec["start"])
        rate_per_s = float(mode_spec["rate_per_s"])
        return start + rate_per_s * time_s
    if mode == "random_walk":
        start = float(mode_spec["start"])
        sigma_step = float(mode_spec["sigma_step"])
        values = np.empty(n_frames, dtype=float)
        values[0] = start
        if n_frames > 1:
            steps = rng.normal(loc=0.0, scale=sigma_step, size=n_frames - 1)
            values[1:] = start + np.cumsum(steps)
        return values

    center = float(mode_spec["center"])
    sigma = float(mode_spec["sigma"])
    return rng.normal(loc=center, scale=sigma, size=n_frames)


def generate_obs_subblock_trace_rows(
    plan: ObsSubblockTraceBuildPlan,
) -> list[dict[str, float | int]]:
    """Generate canonical explicit-trace rows from a normalized plan."""

    time_s = plan.time_s
    rngs = _spawn_key_rngs(plan.seed)
    values_by_key = {
        key: _generate_key_values(
            mode_spec=plan.key_specs[key],
            time_s=time_s,
            rng=rngs[key],
        )
        for key in APPLIED_V1_VARYING_KEYS
    }

    for key, values in values_by_key.items():
        if values.shape != time_s.shape:
            raise RuntimeError(
                f"Generated values for {key} have shape {values.shape}, expected {time_s.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"Generated non-finite values for {key}.")

    rows: list[dict[str, float | int]] = []
    for frame_index, frame_time in enumerate(time_s):
        row: dict[str, float | int] = {
            "frame_index": int(frame_index),
            "time_s": float(frame_time),
        }
        for key in APPLIED_V1_VARYING_KEYS:
            row[key] = float(values_by_key[key][frame_index])
        rows.append(row)
    return rows


__all__ = [
    "ObsSubblockTraceBuildPlan",
    "SUPPORTED_TRACE_MODES",
    "build_obs_subblock_trace_plan",
    "generate_obs_subblock_trace_rows",
]
