from __future__ import annotations

from .arrays import ArrayShardReader, ArrayShardStore, ShardRecord
from .schema import VectorComponentSpec, VectorSpaceSpec
from .splitting import GroupedSplitResult, assign_grouped_split
from .transforms import (
    CompositeTransform,
    CoordinateTransform,
    DiagonalScaleTransform,
    LinearTransform,
)
from .validation import ArrayComparisonResult, compare_arrays

__all__ = [
    "ArrayComparisonResult",
    "ArrayShardReader",
    "ArrayShardStore",
    "CompositeTransform",
    "CoordinateTransform",
    "DiagonalScaleTransform",
    "GroupedSplitResult",
    "LinearTransform",
    "ShardRecord",
    "VectorComponentSpec",
    "VectorSpaceSpec",
    "assign_grouped_split",
    "compare_arrays",
]
