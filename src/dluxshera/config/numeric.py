"""Field-aware numeric coercion helpers for config validation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

__all__ = [
    "coerce_numeric_value",
    "coerce_numeric_mapping",
    "normalize_optimizer_kwargs",
]


_NUMERIC_EXPECTATION = (
    "expected a numeric value (int/float or numeric string such as '1e-3')"
)


def coerce_numeric_value(
    value: Any,
    *,
    path: str,
    allow_none: bool = False,
    allow_str: bool = True,
    must_be_positive: bool = False,
    must_be_nonnegative: bool = False,
    finite_only: bool = True,
) -> float | None:
    """Coerce one expected numeric config field to ``float``.

    This helper is intentionally opt-in and field-aware. It exists for config
    fields that are already known to be numeric, so free-text strings elsewhere
    in a config are never walked or converted globally.
    """

    if must_be_positive and must_be_nonnegative:
        raise ValueError("Only one of must_be_positive/must_be_nonnegative may be set.")

    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{path}: {_NUMERIC_EXPECTATION}; got null.")

    if isinstance(value, bool):
        raise ValueError(f"{path}: {_NUMERIC_EXPECTATION}; got bool.")

    if isinstance(value, Real):
        numeric = float(value)
    elif allow_str and isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{path}: {_NUMERIC_EXPECTATION}; got an empty string.")
        try:
            numeric = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{path}: {_NUMERIC_EXPECTATION}; got {value!r}.") from exc
    else:
        raise ValueError(
            f"{path}: {_NUMERIC_EXPECTATION}; got {type(value).__name__}."
        )

    if finite_only and not math.isfinite(numeric):
        raise ValueError(f"{path}: expected a finite numeric value; got {numeric!r}.")
    if must_be_positive and numeric <= 0.0:
        raise ValueError(f"{path} must be > 0.")
    if must_be_nonnegative and numeric < 0.0:
        raise ValueError(f"{path} must be >= 0.")
    return numeric


def coerce_numeric_mapping(
    payload: Mapping[str, Any] | None,
    *,
    path: str,
    allow_none: bool = True,
    allow_str: bool = True,
    finite_only: bool = True,
) -> dict[str, float]:
    """Validate a string-keyed numeric mapping and normalize values to float."""

    if payload is None:
        if allow_none:
            return {}
        raise ValueError(f"{path} must be a mapping/dict.")
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must be a mapping/dict when provided.")

    coerced: dict[str, float] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"{path} keys must be non-empty strings.")
        key = raw_key.strip()
        coerced[key] = float(
            coerce_numeric_value(
                raw_value,
                path=f"{path}.{key}",
                allow_str=allow_str,
                finite_only=finite_only,
            )
        )
    return coerced


def _coerce_optional_bool(value: Any, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a bool.")
    return value


def normalize_optimizer_kwargs(
    kind: str,
    kwargs: Mapping[str, Any] | None,
    *,
    path: str = "optimizer.kwargs",
) -> dict[str, Any]:
    """Normalize kwargs for supported Optax optimizers.

    Only known numeric optimizer fields are coerced. Known nonnumeric fields
    are validated lightly, and unknown kwargs are rejected so typoed fields do
    not reach Optax unchanged.
    """

    if kwargs is None:
        return {}
    if not isinstance(kwargs, Mapping):
        raise ValueError(f"{path} must be a mapping/dict.")

    optimizer_kind = str(kind).strip().lower()
    if optimizer_kind == "sgd":
        numeric_fields = {"momentum"}
        bool_fields = {"nesterov"}
        passthrough_fields = {"accumulator_dtype"}
    elif optimizer_kind == "adam":
        numeric_fields = {"b1", "b2", "eps", "eps_root"}
        bool_fields = {"nesterov"}
        passthrough_fields = {"mu_dtype"}
    else:
        raise ValueError(
            f"Unsupported optimizer kind {kind!r}; expected 'sgd' or 'adam'."
        )

    supported = numeric_fields | bool_fields | passthrough_fields
    normalized: dict[str, Any] = {}
    for raw_key, raw_value in kwargs.items():
        if not isinstance(raw_key, str) or not raw_key.strip():
            raise ValueError(f"{path} keys must be non-empty strings.")
        key = raw_key.strip()
        field_path = f"{path}.{key}"
        if key not in supported:
            allowed = ", ".join(sorted(supported))
            raise ValueError(
                f"{field_path} is not supported for optimizer kind "
                f"{optimizer_kind!r}. Supported kwargs: {allowed}."
            )
        if key in numeric_fields:
            normalized[key] = coerce_numeric_value(raw_value, path=field_path)
        elif key in bool_fields:
            normalized[key] = _coerce_optional_bool(raw_value, path=field_path)
        else:
            if isinstance(raw_value, bool):
                raise ValueError(f"{field_path} must not be a bool.")
            if isinstance(raw_value, str) and not raw_value.strip():
                raise ValueError(f"{field_path} must be non-empty when provided.")
            normalized[key] = raw_value
    return normalized
