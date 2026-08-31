from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import VectorSpaceSpec, json_ready

__all__ = [
    "CompositeTransform",
    "CoordinateTransform",
    "DiagonalScaleTransform",
    "LinearTransform",
]


def _space_ref(space: VectorSpaceSpec) -> dict[str, Any]:
    return {"name": space.name, "dimension": space.dimension, "labels": list(space.labels)}


@dataclass(frozen=True)
class CoordinateTransform:
    """Base value object for deterministic transforms between vector spaces."""

    source_space: VectorSpaceSpec
    destination_space: VectorSpaceSpec
    name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __call__(self, vector: Any) -> np.ndarray:
        """Apply the forward transform."""
        return self.forward(vector)

    def forward(self, vector: Any) -> np.ndarray:
        """Transform a vector from source to destination coordinates."""
        raise NotImplementedError("Subclasses must implement forward().")

    def inverse(self, vector: Any) -> np.ndarray:
        """Transform a vector from destination back to source coordinates."""
        raise NotImplementedError(f"{self.name} does not provide an inverse transform.")

    def to_dict(self) -> dict[str, Any]:
        """Return common serialized transform provenance."""
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "source_space": _space_ref(self.source_space),
            "destination_space": _space_ref(self.destination_space),
            "metadata": json_ready(dict(self.metadata)),
        }


@dataclass(frozen=True)
class DiagonalScaleTransform(CoordinateTransform):
    """Apply independent per-component scale factors.

    Forward mode divides by ``scales`` by default, matching the common
    Fisher-diagonal convention ``z = delta / sigma``.  Set
    ``forward_mode='multiply'`` for the opposite convention.
    """

    scales: Sequence[float] = field(default_factory=tuple)
    forward_mode: str = "divide"

    def __post_init__(self) -> None:
        if self.source_space.dimension != self.destination_space.dimension:
            raise ValueError("DiagonalScaleTransform source and destination dimensions differ.")
        scales = np.asarray(self.scales, dtype=float)
        if scales.shape != (self.source_space.dimension,):
            raise ValueError(
                f"scales must have shape ({self.source_space.dimension},), got {scales.shape}."
            )
        if not np.all(np.isfinite(scales)) or np.any(scales == 0):
            raise ValueError("scales must be finite and non-zero.")
        if self.forward_mode not in {"divide", "multiply"}:
            raise ValueError("forward_mode must be 'divide' or 'multiply'.")

    def forward(self, vector: Any) -> np.ndarray:
        arr = self.source_space.validate_vector(vector, name="source vector")
        scales = np.asarray(self.scales, dtype=float)
        out = arr / scales if self.forward_mode == "divide" else arr * scales
        return self.destination_space.validate_vector(out, name="destination vector")

    def inverse(self, vector: Any) -> np.ndarray:
        arr = self.destination_space.validate_vector(vector, name="destination vector")
        scales = np.asarray(self.scales, dtype=float)
        out = arr * scales if self.forward_mode == "divide" else arr / scales
        return self.source_space.validate_vector(out, name="source vector")

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update({"scales": json_ready(np.asarray(self.scales)), "forward_mode": self.forward_mode})
        return payload


@dataclass(frozen=True)
class LinearTransform(CoordinateTransform):
    """Apply a dense linear basis transform between compatible vector spaces."""

    matrix: Any = None
    inverse_matrix: Any | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float)
        expected = (self.destination_space.dimension, self.source_space.dimension)
        if matrix.shape != expected:
            raise ValueError(f"matrix must have shape {expected}, got {matrix.shape}.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("matrix must contain only finite values.")
        if self.inverse_matrix is not None:
            inv = np.asarray(self.inverse_matrix, dtype=float)
            inv_expected = (self.source_space.dimension, self.destination_space.dimension)
            if inv.shape != inv_expected:
                raise ValueError(
                    f"inverse_matrix must have shape {inv_expected}, got {inv.shape}."
                )
            if not np.all(np.isfinite(inv)):
                raise ValueError("inverse_matrix must contain only finite values.")

    def forward(self, vector: Any) -> np.ndarray:
        arr = self.source_space.validate_vector(vector, name="source vector")
        out = np.asarray(self.matrix, dtype=float) @ arr
        return self.destination_space.validate_vector(out, name="destination vector")

    def inverse(self, vector: Any) -> np.ndarray:
        arr = self.destination_space.validate_vector(vector, name="destination vector")
        inv = self.inverse_matrix
        if inv is None:
            matrix = np.asarray(self.matrix, dtype=float)
            if matrix.shape[0] != matrix.shape[1]:
                raise NotImplementedError(
                    f"{self.name} has no explicit inverse and matrix is not square."
                )
            inv = np.linalg.inv(matrix)
        out = np.asarray(inv, dtype=float) @ arr
        return self.source_space.validate_vector(out, name="source vector")

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["matrix"] = json_ready(np.asarray(self.matrix))
        if self.inverse_matrix is not None:
            payload["inverse_matrix"] = json_ready(np.asarray(self.inverse_matrix))
        return payload


@dataclass(frozen=True)
class CompositeTransform(CoordinateTransform):
    """Compose multiple transforms into one source-to-destination transform."""

    transforms: tuple[CoordinateTransform, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.transforms:
            raise ValueError("CompositeTransform requires at least one transform.")
        if self.transforms[0].source_space != self.source_space:
            raise ValueError("First transform source space does not match composite source.")
        if self.transforms[-1].destination_space != self.destination_space:
            raise ValueError("Last transform destination space does not match composite destination.")
        for prev, nxt in zip(self.transforms[:-1], self.transforms[1:]):
            if prev.destination_space != nxt.source_space:
                raise ValueError(
                    f"Transform {prev.name!r} destination does not match {nxt.name!r} source."
                )

    def forward(self, vector: Any) -> np.ndarray:
        out = self.source_space.validate_vector(vector, name="source vector")
        for transform in self.transforms:
            out = transform.forward(out)
        return self.destination_space.validate_vector(out, name="destination vector")

    def inverse(self, vector: Any) -> np.ndarray:
        out = self.destination_space.validate_vector(vector, name="destination vector")
        for transform in reversed(self.transforms):
            out = transform.inverse(out)
        return self.source_space.validate_vector(out, name="source vector")

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload["transforms"] = [transform.to_dict() for transform in self.transforms]
        return payload
