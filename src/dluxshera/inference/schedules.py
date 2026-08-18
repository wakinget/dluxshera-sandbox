"""Scalar learning-rate schedule helpers for inference optimizers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from ..config.numeric import coerce_numeric_value

__all__ = [
    "build_scalar_lr_schedule",
    "build_schedule_factor_history",
    "validate_optimizer_schedule_config",
]


def _coerce_schedule_int(
    value: Any,
    *,
    path: str,
    minimum: int | None = None,
) -> int:
    numeric = coerce_numeric_value(value, path=path, finite_only=True)
    integer = int(numeric)
    if float(integer) != float(numeric):
        raise ValueError(f"{path} must be an integer.")
    if minimum is not None and integer < minimum:
        raise ValueError(f"{path} must be >= {minimum}.")
    return integer


def _coerce_schedule_bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a bool.")
    return value


def _coerce_factor(
    value: Any,
    *,
    path: str,
    positive: bool = False,
    min_zero: bool = False,
    max_one: bool = False,
) -> float:
    factor = float(
        coerce_numeric_value(
            value,
            path=path,
            must_be_positive=positive,
            must_be_nonnegative=min_zero,
            finite_only=True,
        )
    )
    if max_one and factor > 1.0:
        raise ValueError(f"{path} must be <= 1.0.")
    return factor


def _validate_known_fields(
    payload: Mapping[str, Any],
    *,
    path: str,
    allowed: set[str],
) -> None:
    for key in payload:
        if key not in allowed:
            supported = ", ".join(sorted(allowed))
            raise ValueError(
                f"{path}.{key} is not supported. Supported fields: {supported}."
            )


def _validate_boundaries(value: Any, *, path: str) -> list[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a sequence of positive integers.")
    boundaries: list[int] = []
    for index, item in enumerate(value):
        boundary = _coerce_schedule_int(
            item,
            path=f"{path}[{index}]",
            minimum=1,
        )
        if boundaries and boundary <= boundaries[-1]:
            raise ValueError(f"{path} must be strictly increasing.")
        boundaries.append(boundary)
    return boundaries


def _validate_factor_list(
    value: Any,
    *,
    path: str,
) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{path} must be a sequence of positive floats.")
    return [
        _coerce_factor(item, path=f"{path}[{index}]", positive=True)
        for index, item in enumerate(value)
    ]


def validate_optimizer_schedule_config(
    schedule_cfg: Mapping[str, Any] | None,
    *,
    n_iter: int,
    path: str = "optimizer.schedule",
) -> dict[str, Any] | None:
    """Validate and normalize one optional scalar LR schedule config."""

    if n_iter <= 0:
        raise ValueError("n_iter must be > 0.")
    if schedule_cfg is None:
        return None
    if not isinstance(schedule_cfg, Mapping):
        raise ValueError(f"{path} must be a mapping/dict when provided.")

    kind_raw = schedule_cfg.get("kind")
    if not isinstance(kind_raw, str) or not kind_raw.strip():
        raise ValueError(f"{path}.kind must be a non-empty string.")
    kind = kind_raw.strip().lower()

    if kind == "constant":
        _validate_known_fields(schedule_cfg, path=path, allowed={"kind", "factor"})
        return {
            "kind": kind,
            "factor": _coerce_factor(
                schedule_cfg.get("factor", 1.0),
                path=f"{path}.factor",
                positive=True,
            ),
        }

    if kind == "linear_warmup":
        _validate_known_fields(
            schedule_cfg,
            path=path,
            allowed={"kind", "warmup_steps", "start_factor"},
        )
        return {
            "kind": kind,
            "warmup_steps": _coerce_schedule_int(
                schedule_cfg.get("warmup_steps"),
                path=f"{path}.warmup_steps",
                minimum=1,
            ),
            "start_factor": _coerce_factor(
                schedule_cfg.get("start_factor"),
                path=f"{path}.start_factor",
                positive=True,
                max_one=True,
            ),
        }

    if kind == "piecewise_constant":
        _validate_known_fields(
            schedule_cfg,
            path=path,
            allowed={"kind", "boundaries", "factors"},
        )
        boundaries = _validate_boundaries(
            schedule_cfg.get("boundaries"),
            path=f"{path}.boundaries",
        )
        factors = _validate_factor_list(
            schedule_cfg.get("factors"),
            path=f"{path}.factors",
        )
        if len(factors) != len(boundaries) + 1:
            raise ValueError(
                f"{path}.factors must have length len(boundaries) + 1."
            )
        return {
            "kind": kind,
            "boundaries": boundaries,
            "factors": factors,
        }

    if kind == "exponential_decay":
        _validate_known_fields(
            schedule_cfg,
            path=path,
            allowed={"kind", "decay_rate", "transition_steps", "staircase"},
        )
        return {
            "kind": kind,
            "decay_rate": _coerce_factor(
                schedule_cfg.get("decay_rate"),
                path=f"{path}.decay_rate",
                positive=True,
            ),
            "transition_steps": _coerce_schedule_int(
                schedule_cfg.get("transition_steps"),
                path=f"{path}.transition_steps",
                minimum=1,
            ),
            "staircase": _coerce_schedule_bool(
                schedule_cfg.get("staircase", False),
                path=f"{path}.staircase",
            ),
        }

    if kind == "cosine_decay":
        _validate_known_fields(
            schedule_cfg,
            path=path,
            allowed={"kind", "min_factor"},
        )
        return {
            "kind": kind,
            "min_factor": _coerce_factor(
                schedule_cfg.get("min_factor", 0.0),
                path=f"{path}.min_factor",
                min_zero=True,
                max_one=True,
            ),
        }

    if kind == "linear_warmup_cosine_decay":
        _validate_known_fields(
            schedule_cfg,
            path=path,
            allowed={"kind", "warmup_steps", "start_factor", "min_factor"},
        )
        warmup_steps = _coerce_schedule_int(
            schedule_cfg.get("warmup_steps"),
            path=f"{path}.warmup_steps",
            minimum=1,
        )
        if warmup_steps >= n_iter:
            raise ValueError(
                f"{path}.warmup_steps must be < n_iter ({n_iter}) for "
                "linear_warmup_cosine_decay."
            )
        return {
            "kind": kind,
            "warmup_steps": warmup_steps,
            "start_factor": _coerce_factor(
                schedule_cfg.get("start_factor", 0.0),
                path=f"{path}.start_factor",
                min_zero=True,
                max_one=True,
            ),
            "min_factor": _coerce_factor(
                schedule_cfg.get("min_factor", 0.0),
                path=f"{path}.min_factor",
                min_zero=True,
                max_one=True,
            ),
        }

    raise ValueError(
        f"Unsupported {path}.kind {kind!r}. Expected one of: constant, "
        "linear_warmup, piecewise_constant, exponential_decay, cosine_decay, "
        "linear_warmup_cosine_decay."
    )


def build_schedule_factor_history(
    schedule_cfg: Mapping[str, Any] | None,
    *,
    n_iter: int,
    path: str = "optimizer.schedule",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a per-step schedule factor history for zero-based optimizer steps."""

    if n_iter <= 0:
        raise ValueError("n_iter must be > 0.")

    normalized = validate_optimizer_schedule_config(
        schedule_cfg,
        n_iter=n_iter,
        path=path,
    )
    steps = np.arange(int(n_iter), dtype=float)

    if normalized is None:
        factor_history = np.ones((int(n_iter),), dtype=float)
        metadata = {
            "enabled": False,
            "configured": False,
            "kind": "none",
            "normalized_config": None,
            "n_iter": int(n_iter),
        }
    else:
        kind = str(normalized["kind"])
        if kind == "constant":
            factor_history = np.full(
                (int(n_iter),),
                float(normalized["factor"]),
                dtype=float,
            )
        elif kind == "linear_warmup":
            warmup_steps = int(normalized["warmup_steps"])
            start_factor = float(normalized["start_factor"])
            factor_history = np.where(
                steps >= warmup_steps,
                1.0,
                start_factor + (1.0 - start_factor) * steps / warmup_steps,
            )
        elif kind == "piecewise_constant":
            boundaries = np.asarray(normalized["boundaries"], dtype=int)
            factors = np.asarray(normalized["factors"], dtype=float)
            segment_index = np.searchsorted(boundaries, steps.astype(int), side="right")
            factor_history = factors[segment_index]
        elif kind == "exponential_decay":
            transition_steps = float(normalized["transition_steps"])
            if bool(normalized["staircase"]):
                exponent = np.floor(steps / transition_steps)
            else:
                exponent = steps / transition_steps
            factor_history = float(normalized["decay_rate"]) ** exponent
        elif kind == "cosine_decay":
            min_factor = float(normalized["min_factor"])
            progress = np.minimum(steps / max(int(n_iter) - 1, 1), 1.0)
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            factor_history = min_factor + (1.0 - min_factor) * cosine
            if int(n_iter) <= 1:
                factor_history = np.ones((1,), dtype=float)
        elif kind == "linear_warmup_cosine_decay":
            warmup_steps = int(normalized["warmup_steps"])
            start_factor = float(normalized["start_factor"])
            min_factor = float(normalized["min_factor"])
            factor_history = np.ones((int(n_iter),), dtype=float)
            warmup_mask = steps < warmup_steps
            factor_history[warmup_mask] = (
                start_factor + (1.0 - start_factor) * steps[warmup_mask] / warmup_steps
            )
            decay_steps = np.maximum(steps - warmup_steps, 0.0)
            decay_denominator = max(int(n_iter) - 1 - warmup_steps, 1)
            decay_progress = np.minimum(decay_steps / decay_denominator, 1.0)
            decay_cosine = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
            factor_history[~warmup_mask] = (
                min_factor + (1.0 - min_factor) * decay_cosine[~warmup_mask]
            )
        else:  # pragma: no cover - guarded by validate_optimizer_schedule_config
            raise AssertionError(f"Unhandled schedule kind {kind!r}.")

        metadata = {
            "enabled": True,
            "configured": True,
            "kind": kind,
            "normalized_config": normalized,
            "n_iter": int(n_iter),
        }

    metadata.update(
        {
            "factor_min": float(np.min(factor_history)),
            "factor_max": float(np.max(factor_history)),
            "first_factor": float(factor_history[0]),
            "last_factor": float(factor_history[-1]),
        }
    )
    return np.asarray(factor_history, dtype=float), metadata


def build_scalar_lr_schedule(
    schedule_cfg: Mapping[str, Any] | None,
    *,
    n_iter: int,
    path: str = "optimizer.schedule",
) -> tuple[Callable[[int], float], dict[str, Any]]:
    """Build ``schedule_fn(step)`` from config and return JSON-friendly metadata."""

    factor_history, metadata = build_schedule_factor_history(
        schedule_cfg,
        n_iter=n_iter,
        path=path,
    )
    factor_history = np.asarray(factor_history, dtype=float)
    last_index = int(factor_history.size - 1)

    def schedule_fn(step: int) -> float:
        clipped = min(max(int(step), 0), last_index)
        return float(factor_history[clipped])

    return schedule_fn, metadata
