from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import json_ready

__all__ = ["ArrayComparisonResult", "compare_arrays"]


@dataclass(frozen=True)
class ArrayComparisonResult:
    """Store structured metrics for comparing a candidate array to a reference."""

    shape_match: bool
    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    reference_dtype: str
    candidate_dtype: str
    finite_reference_count: int
    finite_candidate_count: int
    finite_pair_count: int
    nonfinite_reference_count: int
    nonfinite_candidate_count: int
    max_abs_error: float | None
    mean_abs_error: float | None
    rms_error: float | None
    reference_norm: float | None
    difference_norm: float | None
    relative_l2_error: float | None
    max_relative_error: float | None
    rms_relative_error: float | None
    reference_sum: float | None
    candidate_sum: float | None
    sum_difference: float | None
    relative_sum_difference: float | None
    safe_denominator_policy: str
    relative_denominator_floor: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result."""
        return json_ready(self.__dict__)


def _none_result(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    policy: str,
    denominator_floor: float | None,
) -> ArrayComparisonResult:
    return ArrayComparisonResult(
        shape_match=False,
        reference_shape=tuple(int(v) for v in reference.shape),
        candidate_shape=tuple(int(v) for v in candidate.shape),
        reference_dtype=str(reference.dtype),
        candidate_dtype=str(candidate.dtype),
        finite_reference_count=int(np.isfinite(reference).sum()),
        finite_candidate_count=int(np.isfinite(candidate).sum()),
        finite_pair_count=0,
        nonfinite_reference_count=int(reference.size - np.isfinite(reference).sum()),
        nonfinite_candidate_count=int(candidate.size - np.isfinite(candidate).sum()),
        max_abs_error=None,
        mean_abs_error=None,
        rms_error=None,
        reference_norm=None,
        difference_norm=None,
        relative_l2_error=None,
        max_relative_error=None,
        rms_relative_error=None,
        reference_sum=None,
        candidate_sum=None,
        sum_difference=None,
        relative_sum_difference=None,
        safe_denominator_policy=policy,
        relative_denominator_floor=denominator_floor,
    )


def _resolve_denominator_floor(reference: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        floor = float(requested)
        if floor < 0 or not np.isfinite(floor):
            raise ValueError("relative_denominator_floor must be finite and >= 0.")
        return floor
    finite = reference[np.isfinite(reference)]
    if finite.size == 0:
        return 1.0
    scale = float(np.sqrt(np.mean(np.square(finite.astype(np.float64)))))
    return max(scale * 1.0e-12, np.finfo(np.float64).eps)


def compare_arrays(
    reference: Any,
    candidate: Any,
    *,
    safe_denominator_policy: str = "floor",
    relative_denominator_floor: float | None = None,
) -> ArrayComparisonResult:
    """Compare two arrays with robust absolute and relative metrics.

    Relative pixel metrics use an explicit denominator policy.  The default
    ``floor`` policy divides by ``max(abs(reference), floor)``, where an
    automatic floor is derived from the reference RMS.  This avoids reporting
    unbounded relative errors solely because a reference pixel is exactly zero.
    """
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if np.iscomplexobj(ref) or np.iscomplexobj(cand):
        raise TypeError(
            "compare_arrays does not support complex-valued arrays; compare real-valued "
            "components explicitly before calling this helper."
        )
    if ref.shape != cand.shape:
        return _none_result(
            ref,
            cand,
            policy=safe_denominator_policy,
            denominator_floor=relative_denominator_floor,
        )
    if safe_denominator_policy != "floor":
        raise ValueError("Only safe_denominator_policy='floor' is supported.")

    ref64 = ref.astype(np.float64, copy=False)
    cand64 = cand.astype(np.float64, copy=False)
    finite_ref = np.isfinite(ref64)
    finite_cand = np.isfinite(cand64)
    finite_pair = finite_ref & finite_cand
    finite_pair_count = int(finite_pair.sum())
    floor = _resolve_denominator_floor(ref64, relative_denominator_floor)

    if finite_pair_count == 0:
        max_abs = mean_abs = rms = ref_norm = diff_norm = rel_l2 = None
        max_rel = rms_rel = ref_sum = cand_sum = sum_diff = rel_sum = None
    else:
        ref_f = ref64[finite_pair]
        cand_f = cand64[finite_pair]
        diff = cand_f - ref_f
        abs_diff = np.abs(diff)
        max_abs = float(np.max(abs_diff))
        mean_abs = float(np.mean(abs_diff))
        rms = float(np.sqrt(np.mean(np.square(diff))))
        ref_norm = float(np.linalg.norm(ref_f))
        diff_norm = float(np.linalg.norm(diff))
        rel_l2 = None if ref_norm == 0.0 else float(diff_norm / ref_norm)
        denom = np.maximum(np.abs(ref_f), floor)
        rel = np.abs(diff) / denom
        max_rel = float(np.max(rel))
        rms_rel = float(np.sqrt(np.mean(np.square(rel))))
        ref_sum = float(np.sum(ref_f))
        cand_sum = float(np.sum(cand_f))
        sum_diff = float(cand_sum - ref_sum)
        sum_floor = max(abs(ref_sum), floor * max(1, finite_pair_count))
        rel_sum = float(sum_diff / sum_floor)

    return ArrayComparisonResult(
        shape_match=True,
        reference_shape=tuple(int(v) for v in ref.shape),
        candidate_shape=tuple(int(v) for v in cand.shape),
        reference_dtype=str(ref.dtype),
        candidate_dtype=str(cand.dtype),
        finite_reference_count=int(finite_ref.sum()),
        finite_candidate_count=int(finite_cand.sum()),
        finite_pair_count=finite_pair_count,
        nonfinite_reference_count=int(ref.size - finite_ref.sum()),
        nonfinite_candidate_count=int(cand.size - finite_cand.sum()),
        max_abs_error=max_abs,
        mean_abs_error=mean_abs,
        rms_error=rms,
        reference_norm=ref_norm,
        difference_norm=diff_norm,
        relative_l2_error=rel_l2,
        max_relative_error=max_rel,
        rms_relative_error=rms_rel,
        reference_sum=ref_sum,
        candidate_sum=cand_sum,
        sum_difference=sum_diff,
        relative_sum_difference=rel_sum,
        safe_denominator_policy=safe_denominator_policy,
        relative_denominator_floor=floor,
    )
