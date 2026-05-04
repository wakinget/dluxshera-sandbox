"""Shared key-address and override helpers for observation sub-block workflows.

This module centralizes how observation-subblock keys are parsed, validated,
and applied so the trace-builder and renderer stay aligned.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..params.transforms import (
    compute_source_raw_fluxes_from_log_flux_total_and_contrast,
)


OBS_SUBBLOCK_V1_DEFAULT_VARYING_KEYS: tuple[str, ...] = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)

OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS: tuple[str, ...] = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
    "source.separation_as",
    "source.contrast",
    "source.log_flux_total",
    "optics.plate_scale_as_per_pix",
)

OBS_SUBBLOCK_SUPPORTED_INDEXED_KEYS: tuple[str, ...] = (
    "optics.primary.zernike_coeffs_nm",
    "optics.secondary.zernike_coeffs_nm",
)

OBS_SUBBLOCK_SUPPORTED_BASE_KEYS: tuple[str, ...] = (
    *OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS,
    *OBS_SUBBLOCK_SUPPORTED_INDEXED_KEYS,
)

_KEY_ADDRESS_PATTERN = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_.]*)(?:\[(?P<idx>\d+)\])?$")


@dataclass(frozen=True)
class ObsSubblockKeyAddress:
    """Canonical parsed key address for observation-subblock varying keys."""

    base_key: str
    index: int | None = None

    @property
    def canonical(self) -> str:
        """Return normalized key syntax string."""

        if self.index is None:
            return self.base_key
        return f"{self.base_key}[{self.index}]"


def parse_obs_subblock_key_address(raw_key: str) -> ObsSubblockKeyAddress:
    """Parse scalar or indexed key syntax into a canonical address."""

    text = raw_key.strip()
    if not text:
        raise ValueError("Observation-subblock varying key cannot be blank.")

    match = _KEY_ADDRESS_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(
            "Invalid observation-subblock key syntax. "
            "Use 'a.b.c' or 'a.b.c[index]'."
        )

    base_key = match.group("base")
    idx_text = match.group("idx")
    if idx_text is None:
        return ObsSubblockKeyAddress(base_key=base_key, index=None)
    return ObsSubblockKeyAddress(base_key=base_key, index=int(idx_text))


def parse_obs_subblock_varying_keys(
    keys: Sequence[str],
) -> tuple[ObsSubblockKeyAddress, ...]:
    """Parse and deduplicate varying-key strings while preserving order."""

    parsed: list[ObsSubblockKeyAddress] = []
    seen: set[str] = set()
    for idx, raw_key in enumerate(keys):
        if not isinstance(raw_key, str):
            raise ValueError(
                "Observation-subblock varying keys must be strings, "
                f"got {type(raw_key).__name__} at index {idx}."
            )
        address = parse_obs_subblock_key_address(raw_key)
        canonical = address.canonical
        if canonical in seen:
            raise ValueError(f"Duplicate observation-subblock varying key: {canonical}.")
        seen.add(canonical)
        parsed.append(address)
    return tuple(parsed)


def canonical_obs_subblock_varying_keys(
    addresses: Sequence[ObsSubblockKeyAddress],
) -> tuple[str, ...]:
    """Return canonical key strings from parsed addresses."""

    return tuple(address.canonical for address in addresses)


def _coerce_finite_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{path} must be numeric.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{path} must be finite.")
    return parsed


def _vector_length_from_store(
    *,
    base_key: str,
    store: Any | None,
) -> int | None:
    if store is None:
        return None
    try:
        value = store.get(base_key)
    except KeyError:
        return None
    array_value = np.asarray(value)
    if array_value.ndim != 1:
        return None
    return int(array_value.shape[0])


def validate_supported_obs_subblock_key_addresses(
    addresses: Sequence[ObsSubblockKeyAddress],
    *,
    forward_spec: Any | None = None,
    reference_store: Any | None = None,
) -> None:
    """Validate key addresses against supported policy and optional spec/store."""

    for address in addresses:
        base_key = address.base_key
        if base_key not in OBS_SUBBLOCK_SUPPORTED_BASE_KEYS:
            raise ValueError(
                "Unsupported observation-subblock varying key "
                f"{address.canonical!r}. Supported keys are: "
                + ", ".join(OBS_SUBBLOCK_SUPPORTED_BASE_KEYS)
            )

        if base_key in OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS and address.index is not None:
            raise ValueError(
                f"Key {address.canonical!r} uses indexed syntax, but {base_key!r} "
                "is scalar-only."
            )
        if base_key in OBS_SUBBLOCK_SUPPORTED_INDEXED_KEYS and address.index is None:
            raise ValueError(
                f"Key {base_key!r} is vector-valued; use indexed syntax like "
                f"{base_key}[0]."
            )

        if forward_spec is not None:
            if base_key not in forward_spec:
                raise ValueError(
                    f"Key {base_key!r} is not present in the resolved forward spec."
                )
            field = forward_spec.get(base_key)
            if getattr(field, "structural", False):
                raise ValueError(
                    f"Key {base_key!r} is structural and cannot vary per frame."
                )
            kind = getattr(field, "kind", None)
            if kind not in {"primitive", "derived"}:
                raise ValueError(
                    f"Key {base_key!r} has unsupported kind {kind!r}; "
                    "only primitive/derived keys are supported."
                )

        if address.index is not None:
            vector_len = _vector_length_from_store(
                base_key=base_key,
                store=reference_store,
            )
            if vector_len is not None and address.index >= vector_len:
                raise ValueError(
                    f"Key {address.canonical!r} index is out of bounds "
                    f"(length={vector_len})."
                )


def get_obs_subblock_store_value(
    store: Any,
    *,
    address: ObsSubblockKeyAddress,
) -> float:
    """Return a scalar store value for scalar or indexed key addresses."""

    value = store.get(address.base_key)
    array_value = np.asarray(value)
    if address.index is None:
        if array_value.ndim != 0:
            raise ValueError(
                f"Key {address.base_key!r} is vector-valued; use indexed syntax."
            )
        return float(array_value)

    if array_value.ndim != 1:
        raise ValueError(
            f"Key {address.base_key!r} is not 1D vector-valued and cannot be indexed."
        )
    if address.index >= int(array_value.shape[0]):
        raise ValueError(
            f"Key {address.canonical!r} index is out of bounds "
            f"(length={array_value.shape[0]})."
        )
    return float(array_value[address.index])


def get_obs_subblock_mapping_value(
    mapping: Mapping[str, Any] | None,
    *,
    address: ObsSubblockKeyAddress,
) -> float | None:
    """Return a scalar mapping value for scalar or indexed key addresses.

    Missing paths return ``None`` so callers can fall back to resolved-store
    defaults. Present-but-incompatible values raise explicit errors.
    """

    if mapping is None:
        return None
    current: Any = mapping
    for part in address.base_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]

    if isinstance(current, bool):
        raise ValueError(f"Key {address.base_key!r} must be numeric.")

    array_value = np.asarray(current)
    if address.index is None:
        if array_value.ndim != 0:
            raise ValueError(
                f"Key {address.base_key!r} is vector-valued; use indexed syntax."
            )
        return float(array_value)

    if array_value.ndim != 1:
        raise ValueError(
            f"Key {address.base_key!r} is not 1D vector-valued and cannot be indexed."
        )
    if address.index >= int(array_value.shape[0]):
        raise ValueError(
            f"Key {address.canonical!r} index is out of bounds "
            f"(length={array_value.shape[0]})."
        )
    return float(array_value[address.index])


def set_obs_subblock_mapping_value(
    mapping: MutableMapping[str, Any],
    *,
    address: ObsSubblockKeyAddress,
    value: Any,
    reference_vector: Sequence[Any] | np.ndarray | None = None,
) -> None:
    """Set one scalar or indexed candidate value inside a nested mapping.

    Indexed updates write the full vector back into the mapping so preset-backed
    system configs can be overridden by one explicit component at a time.
    """

    if not isinstance(mapping, MutableMapping):
        raise ValueError("Target mapping must be a mutable mapping/dict.")

    scalar_value = _coerce_finite_float(value, path=address.canonical)
    current = mapping
    parts = address.base_key.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            current[part] = {}
            child = current[part]
        if not isinstance(child, MutableMapping):
            raise ValueError(
                f"Cannot set key {address.canonical!r}; path component {part!r} "
                "is not a mapping."
            )
        current = child

    leaf = parts[-1]
    if address.index is None:
        current[leaf] = float(scalar_value)
        return

    existing = current.get(leaf)
    if existing is None:
        if reference_vector is None:
            raise ValueError(
                f"Cannot set indexed key {address.canonical!r}; base vector "
                f"{address.base_key!r} is missing and no reference vector was provided."
            )
        vector_value = np.asarray(reference_vector, dtype=float).copy()
    else:
        vector_value = np.asarray(existing, dtype=float).copy()
    if vector_value.ndim != 1:
        raise ValueError(
            f"Key {address.base_key!r} is not 1D vector-valued and cannot be indexed."
        )
    if address.index >= int(vector_value.shape[0]):
        raise ValueError(
            f"Key {address.canonical!r} index is out of bounds "
            f"(length={vector_value.shape[0]})."
        )
    vector_value[address.index] = float(scalar_value)
    current[leaf] = vector_value.tolist()


def collect_obs_subblock_anchor_values(
    store: Any,
    *,
    addresses: Iterable[ObsSubblockKeyAddress],
) -> dict[str, float]:
    """Collect scalar anchor values from a store for given key addresses."""

    anchors: dict[str, float] = {}
    for address in addresses:
        anchors[address.canonical] = get_obs_subblock_store_value(
            store,
            address=address,
        )
    return anchors


def split_obs_subblock_frame_overrides(
    *,
    base_store: Any,
    forward_spec: Any,
    addresses: Sequence[ObsSubblockKeyAddress],
    values_by_key: Mapping[str, Any],
    value_path_prefix: str = "trace",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split frame overrides into primitive and derived replacement payloads."""

    primitive_overrides: dict[str, Any] = {}
    derived_overrides: dict[str, Any] = {}

    for address in addresses:
        canonical = address.canonical
        if canonical not in values_by_key:
            raise ValueError(f"{value_path_prefix} is missing required key {canonical!r}.")
        value = _coerce_finite_float(
            values_by_key[canonical],
            path=f"{value_path_prefix}.{canonical}",
        )

        field = forward_spec.get(address.base_key)
        kind = field.kind
        if kind == "primitive":
            target = primitive_overrides
        elif kind == "derived":
            target = derived_overrides
        else:
            raise ValueError(
                f"Key {address.base_key!r} has unsupported kind {kind!r}."
            )

        if address.index is None:
            target[address.base_key] = value
            continue

        if address.base_key in target:
            vector_value = np.asarray(target[address.base_key], dtype=float).copy()
        else:
            vector_value = np.asarray(base_store.get(address.base_key), dtype=float).copy()
        if vector_value.ndim != 1:
            raise ValueError(
                f"Key {address.base_key!r} is not 1D vector-valued and cannot be indexed."
            )
        if address.index >= int(vector_value.shape[0]):
            raise ValueError(
                f"Key {address.canonical!r} index is out of bounds "
                f"(length={vector_value.shape[0]})."
            )
        vector_value[address.index] = value
        target[address.base_key] = vector_value

    return primitive_overrides, derived_overrides


def apply_obs_subblock_overrides_preserving_derived(
    store: Any,
    *,
    forward_spec: Any,
    primitive_overrides: Mapping[str, Any] | None = None,
    derived_overrides: Mapping[str, Any] | None = None,
) -> Any:
    """Apply primitive overrides, refresh derived values, then re-apply derived."""

    updated = store
    if primitive_overrides:
        updated = updated.replace(dict(primitive_overrides))
    updated = updated.refresh_derived(forward_spec)
    if derived_overrides:
        updated = updated.replace(dict(derived_overrides))
    return updated


def apply_jax_safe_source_photometry_update(
    store: Any,
    *,
    log_flux_total: Any | None = None,
    contrast: Any | None = None,
    forward_spec: Any | None = None,
) -> Any:
    """Update active source photometry values without traced derived refresh.

    Notes
    -----
    ``source.log_flux_total`` may be derived in the full forward spec, but in a
    local inference context it can still be an authoritative active variable.
    This helper preserves that active-value semantics and updates the dependent
    ``source.raw_fluxes`` term with JAX-safe array operations instead of
    calling a full ``refresh_derived(...)`` inside autodiff.
    """

    if forward_spec is not None:
        for key in ("source.log_flux_total", "source.contrast", "source.raw_fluxes"):
            if key not in forward_spec:
                raise ValueError(
                    f"JAX-safe source photometry update requires {key!r} in the forward spec."
                )

    effective_log_flux = (
        store.get("source.log_flux_total")
        if log_flux_total is None
        else log_flux_total
    )
    effective_contrast = (
        store.get("source.contrast")
        if contrast is None
        else contrast
    )
    effective_log_flux = jnp.asarray(effective_log_flux, dtype=float)
    effective_contrast = jnp.asarray(effective_contrast, dtype=float)
    raw_fluxes = compute_source_raw_fluxes_from_log_flux_total_and_contrast(
        effective_log_flux,
        effective_contrast,
    )
    return store.replace(
        {
            "source.log_flux_total": effective_log_flux,
            "source.contrast": effective_contrast,
            "source.raw_fluxes": raw_fluxes,
        }
    )


def apply_obs_subblock_runtime_overrides_without_refresh(
    store: Any,
    *,
    overrides_flat: Mapping[str, Any],
    forward_spec: Any,
) -> Any:
    """Apply active runtime overrides without full traced derived refresh.

    This mirrors the canonical inference semantics more closely than
    ``apply_obs_subblock_overrides_preserving_derived`` for local autodiff:
    active values are authoritative overlays on a resolved base store, and only
    the minimal dependent quantities needed by the runtime model are refreshed
    explicitly with JAX-safe operations.
    """

    primitive_overrides, derived_overrides, unknown = (
        partition_obs_subblock_overrides_by_kind(
            overrides_flat,
            forward_spec=forward_spec,
        )
    )
    if unknown:
        raise ValueError(
            "Runtime overrides contain unknown or unsupported keys: "
            + ", ".join(sorted(unknown))
        )

    updated = store.replace({**primitive_overrides, **derived_overrides})
    if (
        "source.log_flux_total" in overrides_flat
        or "source.contrast" in overrides_flat
    ):
        updated = apply_jax_safe_source_photometry_update(
            updated,
            log_flux_total=overrides_flat.get("source.log_flux_total"),
            contrast=overrides_flat.get("source.contrast"),
            forward_spec=forward_spec,
        )
    return updated


def partition_obs_subblock_overrides_by_kind(
    overrides_flat: Mapping[str, Any],
    *,
    forward_spec: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Split flattened dotted-key overrides into primitive/derived/unknown."""

    primitive_overrides: dict[str, Any] = {}
    derived_overrides: dict[str, Any] = {}
    unknown_overrides: dict[str, Any] = {}
    for key, value in overrides_flat.items():
        if key not in forward_spec:
            unknown_overrides[key] = value
            continue
        kind = forward_spec.get(key).kind
        if kind == "primitive":
            primitive_overrides[key] = value
        elif kind == "derived":
            derived_overrides[key] = value
        else:
            unknown_overrides[key] = value
    return primitive_overrides, derived_overrides, unknown_overrides


__all__ = [
    "OBS_SUBBLOCK_SUPPORTED_BASE_KEYS",
    "OBS_SUBBLOCK_SUPPORTED_INDEXED_KEYS",
    "OBS_SUBBLOCK_SUPPORTED_SCALAR_KEYS",
    "OBS_SUBBLOCK_V1_DEFAULT_VARYING_KEYS",
    "ObsSubblockKeyAddress",
    "apply_obs_subblock_overrides_preserving_derived",
    "apply_obs_subblock_runtime_overrides_without_refresh",
    "apply_jax_safe_source_photometry_update",
    "canonical_obs_subblock_varying_keys",
    "collect_obs_subblock_anchor_values",
    "get_obs_subblock_mapping_value",
    "get_obs_subblock_store_value",
    "parse_obs_subblock_key_address",
    "parse_obs_subblock_varying_keys",
    "partition_obs_subblock_overrides_by_kind",
    "set_obs_subblock_mapping_value",
    "split_obs_subblock_frame_overrides",
    "validate_supported_obs_subblock_key_addresses",
]
