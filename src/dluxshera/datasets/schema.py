from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "VectorComponentSpec",
    "VectorSpaceSpec",
    "json_ready",
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
]


def json_ready(value: Any) -> Any:
    """Return a JSON-serializable representation of common scientific values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write a stable UTF-8 JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Stream rows to a JSONL file without materializing them all."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_ready(dict(row)), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield object rows from a JSONL file with clear line-number errors."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{path} line {line_number} is not a JSON object.")
            yield dict(payload)


@dataclass(frozen=True)
class VectorComponentSpec:
    """Describe one ordered component in a vector-valued parameter space.

    Use this for metadata attached to any named vector basis: SHERA physical
    parameters, nuisance coordinates, Fisher-scaled coordinates, future
    eigenmodes, or detector state vectors.  The class intentionally stores
    component identity and metadata only; it does not encode model, optimizer,
    or training semantics.
    """

    label: str
    index: int | None = None
    source_key: str | None = None
    component_index: int | None = None
    display_label: str | None = None
    unit: str | None = None
    group: str | None = None
    reference_value: float | int | None = None
    scale: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.label).strip():
            raise ValueError("VectorComponentSpec.label must be non-empty.")
        if self.index is not None and int(self.index) < 0:
            raise ValueError("VectorComponentSpec.index must be >= 0 when provided.")
        if self.component_index is not None and int(self.component_index) < 0:
            raise ValueError(
                "VectorComponentSpec.component_index must be >= 0 when provided."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "label": self.label,
            "index": self.index,
            "source_key": self.source_key,
            "component_index": self.component_index,
            "display_label": self.display_label,
            "unit": self.unit,
            "group": self.group,
            "reference_value": self.reference_value,
            "scale": json_ready(self.scale),
            "metadata": json_ready(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VectorComponentSpec":
        """Build a component spec from a serialized representation."""
        return cls(
            label=str(payload["label"]),
            index=None if payload.get("index") is None else int(payload["index"]),
            source_key=(
                None if payload.get("source_key") is None else str(payload["source_key"])
            ),
            component_index=(
                None
                if payload.get("component_index") is None
                else int(payload["component_index"])
            ),
            display_label=(
                None
                if payload.get("display_label") is None
                else str(payload["display_label"])
            ),
            unit=None if payload.get("unit") is None else str(payload["unit"]),
            group=None if payload.get("group") is None else str(payload["group"]),
            reference_value=payload.get("reference_value"),
            scale=payload.get("scale"),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class VectorSpaceSpec:
    """Describe an ordered vector basis with named components.

    ``VectorSpaceSpec`` is the reusable metadata contract for vectorized state.
    It preserves ordering, validates dimensionality, and round-trips through
    JSON without assuming what the vector means scientifically.
    """

    name: str
    components: tuple[VectorComponentSpec, ...]
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "vector_space_spec/1"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("VectorSpaceSpec.name must be non-empty.")
        if not self.components:
            raise ValueError("VectorSpaceSpec.components must not be empty.")
        labels: set[str] = set()
        for idx, component in enumerate(self.components):
            if component.label in labels:
                raise ValueError(f"Duplicate vector component label {component.label!r}.")
            labels.add(component.label)
            if component.index is not None and int(component.index) != idx:
                raise ValueError(
                    f"Component {component.label!r} has index {component.index}, expected {idx}."
                )

    @property
    def dimension(self) -> int:
        """Return vector dimension."""
        return len(self.components)

    @property
    def labels(self) -> tuple[str, ...]:
        """Return ordered component labels."""
        return tuple(component.label for component in self.components)

    def component_index(self, label: str) -> int:
        """Return the positional index for a component label."""
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise KeyError(f"Unknown vector component label {label!r}.") from exc

    def validate_vector(self, vector: Any, *, name: str = "vector") -> np.ndarray:
        """Return a 1D array after validating dimensional compatibility."""
        arr = np.asarray(vector)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be a 1D vector, got shape {arr.shape}.")
        if arr.shape[0] != self.dimension:
            raise ValueError(
                f"{name} has dimension {arr.shape[0]}, expected {self.dimension} for {self.name!r}."
            )
        return arr

    def zeros(self, *, dtype: Any = float) -> np.ndarray:
        """Return a zero vector compatible with this space."""
        return np.zeros((self.dimension,), dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "dimension": self.dimension,
            "components": [component.to_dict() for component in self.components],
            "metadata": json_ready(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "VectorSpaceSpec":
        """Build a vector-space spec from serialized metadata."""
        components = tuple(
            VectorComponentSpec.from_dict(component)
            for component in payload.get("components", [])
        )
        return cls(
            name=str(payload["name"]),
            components=components,
            description=(
                None
                if payload.get("description") is None
                else str(payload["description"])
            ),
            metadata=dict(payload.get("metadata", {}) or {}),
            schema_version=str(payload.get("schema_version", "vector_space_spec/1")),
        )

    @classmethod
    def from_labels(
        cls,
        name: str,
        labels: Iterable[str],
        *,
        description: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VectorSpaceSpec":
        """Construct a vector-space spec from ordered labels."""
        components = tuple(
            VectorComponentSpec(label=str(label), index=idx)
            for idx, label in enumerate(labels)
        )
        return cls(
            name=name,
            components=components,
            description=description,
            metadata=dict(metadata or {}),
        )
