from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def describe_dtype(value: Any) -> str:
    """Return a compact dtype/shape summary for debug logging."""

    if value is None:
        return "None"

    dtype = getattr(value, "dtype", None)
    shape = getattr(value, "shape", None)
    if dtype is None or shape is None:
        arr = np.asarray(value)
        dtype = arr.dtype
        shape = arr.shape

    shape_tuple = tuple(int(dim) for dim in shape)
    return f"type={type(value).__name__} dtype={dtype} shape={shape_tuple}"


def print_dtype_audit(title: str, entries: Mapping[str, Any]) -> None:
    """Print one compact dtype audit block."""

    print(f"Dtype audit: {title}")
    for name, value in entries.items():
        print(f"  {name}: {describe_dtype(value)}")


__all__ = ["describe_dtype", "print_dtype_audit"]
