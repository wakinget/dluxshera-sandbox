from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from astropy.io import fits

from dluxshera.params.spec import ParamSpec
from dluxshera.params.store import ParameterStore

__all__ = [
    "RenderResult",
    "apply_absolute_vector_to_store",
    "apply_sample_deltas_to_store",
    "refresh_preserving_derived_keys",
    "render_image",
    "set_scalar_label",
    "write_fits",
]


class ScalarParameterLike(Protocol):
    """Expose the scalarized parameter fields used by dataset render plans."""

    label: str
    base_key: str
    component_index: int | None
    nominal_value: float


@dataclass(frozen=True)
class RenderResult:
    """Hold one deterministic binder render as a NumPy image."""

    image: np.ndarray
    image_shape: tuple[int, ...]
    dtype: str


def write_fits(
    *, output_path: Path, image: np.ndarray, header_data: Mapping[str, Any]
) -> None:
    """Write one V3-compatible FITS image.

    This is the shared FITS output primitive used by raw SHERA ML renderers.
    Detailed scientific metadata belongs in the JSON sidecar; the FITS header
    stays compact and human-readable.
    """

    header = fits.Header()
    for key, value in header_data.items():
        if value is None:
            continue
        if isinstance(value, tuple) and len(value) == 2:
            card_value, comment = value
            header.set(str(key).upper(), card_value, comment=str(comment))
        else:
            header.set(str(key).upper(), value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=image, header=header).writeto(output_path, overwrite=True)


def set_scalar_label(
    store: ParameterStore,
    target: ScalarParameterLike,
    value: float,
) -> ParameterStore:
    """Set a scalarized store label, preserving indexed vector placement."""

    if target.component_index is None:
        return store.replace({target.base_key: value})
    current = np.asarray(store.get(target.base_key), dtype=float).copy().reshape(-1)
    current[int(target.component_index)] = value
    original_shape = np.asarray(store.get(target.base_key)).shape
    return store.replace({target.base_key: current.reshape(original_shape)})


def refresh_preserving_derived_keys(
    store: ParameterStore,
    *,
    preserved_keys: Iterable[str],
    spec: ParamSpec,
) -> ParameterStore:
    """Refresh derived values while keeping explicitly controlled derived keys.

    V3 allowed controlled axes such as plate scale to remain as applied values
    even though they are represented as derived fields in the forward spec.
    Keep that behavior centralized so V3 and V4 renders share the same
    primitive/derived boundary.
    """

    preserved_values: dict[str, Any] = {}
    for key in preserved_keys:
        if key not in spec or spec.get(key).kind != "derived":
            continue
        try:
            preserved_values[key] = store.get(key)
        except KeyError:
            continue
    refreshed = store.refresh_derived(spec)
    if preserved_values:
        refreshed = refreshed.replace(preserved_values)
    return refreshed


def apply_sample_deltas_to_store(
    *,
    base_store: ParameterStore,
    theta_delta: Mapping[str, Any],
    registration_nuisance_values: Mapping[str, Any],
    parameters_by_label: Mapping[str, ScalarParameterLike],
    forward_spec: ParamSpec,
    registration_nuisance_keys: Iterable[str],
) -> ParameterStore:
    """Apply V3-style controlled deltas and registration nuisance deltas.

    Science controls are applied as ``nominal_value + delta`` for each
    scalarized V3 label. Registration nuisance values are physical deltas
    added to the current source registration coordinates. This preserves the
    historical V3 nuisance semantics used to build the V4 nuisance bank.
    """

    store = base_store
    for label, delta in dict(theta_delta).items():
        target = parameters_by_label[str(label)]
        store = set_scalar_label(store, target, target.nominal_value + float(delta))
    for key, delta in dict(registration_nuisance_values).items():
        key = str(key)
        if key not in forward_spec:
            continue
        current = float(np.asarray(store.get(key)))
        store = store.replace({key: current + float(delta)})
    preserve = {
        target.base_key for target in parameters_by_label.values()
    } | {str(key) for key in registration_nuisance_keys}
    return refresh_preserving_derived_keys(
        store,
        preserved_keys=preserve,
        spec=forward_spec,
    )


def apply_absolute_vector_to_store(
    *,
    base_store: ParameterStore,
    labels: Iterable[str],
    physical_values: Iterable[Any],
    forward_spec: ParamSpec,
    component_indices: Mapping[str, int | None],
    nuisance_labels: Iterable[str] = (),
    nuisance_physical_deltas: Iterable[Any] = (),
) -> ParameterStore:
    """Apply one V4 absolute science vector plus V3-style nuisance deltas.

    Science labels are absolute physical values in the frozen V4 vector-space
    order. Nuisance values remain registration deltas and are therefore added
    to the science-applied coordinates instead of replacing them.
    """

    store = base_store
    preserve: set[str] = set()
    labels = tuple(labels)
    physical_values = tuple(physical_values)
    nuisance_labels = tuple(nuisance_labels)
    nuisance_physical_deltas = tuple(nuisance_physical_deltas)
    if len(labels) != len(physical_values):
        raise ValueError(
            "labels and physical_values must have matching lengths: "
            f"{len(labels)} != {len(physical_values)}."
        )
    if len(nuisance_labels) != len(nuisance_physical_deltas):
        raise ValueError(
            "nuisance_labels and nuisance_physical_deltas must have matching lengths: "
            f"{len(nuisance_labels)} != {len(nuisance_physical_deltas)}."
        )
    for label, value in zip(labels, physical_values):
        label = str(label)
        base_key = label.split("[", 1)[0]
        if base_key not in forward_spec:
            raise KeyError(f"Science label {label!r} maps to unknown store key {base_key!r}.")
        component_index = component_indices.get(label)
        if "[" in label and component_index is None:
            raise ValueError(
                f"Indexed science label {label!r} is missing a vector-space component_index."
            )
        target = _ScalarTarget(
            label=label,
            base_key=base_key,
            component_index=component_index,
            nominal_value=0.0,
        )
        store = set_scalar_label(store, target, float(value))
        preserve.add(base_key)

    for label, delta in zip(nuisance_labels, nuisance_physical_deltas):
        label = str(label)
        if label not in forward_spec:
            raise KeyError(f"Nuisance label {label!r} is not present in the forward spec.")
        current = float(np.asarray(store.get(label)))
        store = store.replace({label: current + float(delta)})
        preserve.add(label)

    return refresh_preserving_derived_keys(
        store,
        preserved_keys=preserve,
        spec=forward_spec,
    )


def render_image(*, binder: Any, applied_store: ParameterStore) -> RenderResult:
    """Evaluate the existing SHERA binder forward path and return a NumPy image."""

    image = binder.model(binder.strip_structural(applied_store))
    image_np = np.asarray(image)
    return RenderResult(
        image=image_np,
        image_shape=tuple(int(value) for value in image_np.shape),
        dtype=str(image_np.dtype),
    )


@dataclass(frozen=True)
class _ScalarTarget:
    label: str
    base_key: str
    component_index: int | None
    nominal_value: float
