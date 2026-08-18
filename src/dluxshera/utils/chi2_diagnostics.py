"""Shared chi-squared diagnostics for image and cube comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CHI2_METRIC_NOTES = (
    "Chi-squared uses sum((data - model)^2 / variance) over finite pixels with "
    "positive finite variance. Reduced chi-squared divides by the contributing "
    "pixel count and does not subtract fitted parameters from the degrees of "
    "freedom."
)


def _jsonable_float(value: float) -> float | None:
    value_float = float(value)
    return value_float if np.isfinite(value_float) else None


def _jsonable_float_list(values: np.ndarray) -> list[float | None]:
    return [_jsonable_float(value) for value in np.asarray(values, dtype=float).ravel()]


@dataclass(frozen=True)
class ChiSquaredCubeSummary:
    """Frame-wise and block-wise chi-squared diagnostics for one data/model cube."""

    per_frame_chi2: np.ndarray
    per_frame_reduced_chi2: np.ndarray
    per_frame_dof_pixels: np.ndarray
    block_sum_chi2: float
    block_reduced_chi2: float
    block_mean_reduced_chi2: float
    block_dof_pixels: int

    def block_payload(self) -> dict[str, Any]:
        """Return the compact block-level scalar summary."""

        return {
            "block_sum_chi2": _jsonable_float(self.block_sum_chi2),
            "block_reduced_chi2": _jsonable_float(self.block_reduced_chi2),
            "block_mean_reduced_chi2": _jsonable_float(self.block_mean_reduced_chi2),
            "block_dof_pixels": int(self.block_dof_pixels),
        }

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-friendly full payload including per-frame arrays."""

        return {
            "per_frame_chi2": _jsonable_float_list(self.per_frame_chi2),
            "per_frame_reduced_chi2": _jsonable_float_list(self.per_frame_reduced_chi2),
            "per_frame_dof_pixels": [
                int(value)
                for value in np.asarray(self.per_frame_dof_pixels, dtype=int).ravel()
            ],
            **self.block_payload(),
        }


def summarize_framewise_chi2(
    data_cube: Any,
    model_cube: Any,
    *,
    variance_cube: Any,
) -> ChiSquaredCubeSummary:
    """Return per-frame and block-level chi-squared summaries for a 3D cube.

    The caller is expected to pass the same variance cube already used by the
    corresponding Gaussian image objective, including any clipping or flooring.
    Non-finite inputs or non-positive variance are excluded from the diagnostic
    dof count rather than raising.
    """

    data_arr = np.asarray(data_cube, dtype=float)
    model_arr = np.asarray(model_cube, dtype=float)
    var_arr = np.asarray(variance_cube, dtype=float)
    if data_arr.shape != model_arr.shape or data_arr.shape != var_arr.shape:
        raise ValueError(
            "data_cube, model_cube, and variance_cube must have the same shape."
        )
    if data_arr.ndim != 3:
        raise ValueError(
            "data_cube, model_cube, and variance_cube must be 3D arrays with "
            "shape (n_frame, y, x)."
        )

    n_frame = int(data_arr.shape[0])
    valid_mask = (
        np.isfinite(data_arr)
        & np.isfinite(model_arr)
        & np.isfinite(var_arr)
        & (var_arr > 0.0)
    )
    residual_sq = np.square(data_arr - model_arr)
    chi2_terms = np.zeros_like(residual_sq, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(residual_sq, var_arr, out=chi2_terms, where=valid_mask)

    pixels_per_frame = int(np.prod(data_arr.shape[1:], dtype=int))
    flat_shape = (n_frame, pixels_per_frame)
    per_frame_chi2 = np.sum(chi2_terms.reshape(flat_shape), axis=1, dtype=float)
    per_frame_dof_pixels = np.sum(valid_mask.reshape(flat_shape), axis=1, dtype=int)
    per_frame_reduced_chi2 = np.full(n_frame, np.nan, dtype=float)
    np.divide(
        per_frame_chi2,
        per_frame_dof_pixels,
        out=per_frame_reduced_chi2,
        where=per_frame_dof_pixels > 0,
    )

    block_sum_chi2 = float(np.sum(per_frame_chi2, dtype=float))
    block_dof_pixels = int(np.sum(per_frame_dof_pixels, dtype=int))
    block_reduced_chi2 = (
        float(block_sum_chi2 / block_dof_pixels)
        if block_dof_pixels > 0
        else float("nan")
    )
    finite_reduced = per_frame_reduced_chi2[np.isfinite(per_frame_reduced_chi2)]
    block_mean_reduced_chi2 = (
        float(np.mean(finite_reduced)) if finite_reduced.size else float("nan")
    )

    return ChiSquaredCubeSummary(
        per_frame_chi2=per_frame_chi2,
        per_frame_reduced_chi2=per_frame_reduced_chi2,
        per_frame_dof_pixels=per_frame_dof_pixels,
        block_sum_chi2=block_sum_chi2,
        block_reduced_chi2=block_reduced_chi2,
        block_mean_reduced_chi2=block_mean_reduced_chi2,
        block_dof_pixels=block_dof_pixels,
    )


def reduced_chi2_between_images(
    data_image: Any,
    model_image: Any,
    *,
    variance_image: Any,
) -> float:
    """Return reduced chi-squared for one image pair."""

    data_arr = np.asarray(data_image, dtype=float)
    model_arr = np.asarray(model_image, dtype=float)
    var_arr = np.asarray(variance_image, dtype=float)
    if data_arr.shape != model_arr.shape or data_arr.shape != var_arr.shape:
        raise ValueError(
            "data_image, model_image, and variance_image must have the same shape."
        )
    if data_arr.ndim != 2:
        raise ValueError(
            "data_image, model_image, and variance_image must be 2D arrays."
        )

    summary = summarize_framewise_chi2(
        data_arr[None, ...],
        model_arr[None, ...],
        variance_cube=var_arr[None, ...],
    )
    return float(summary.block_reduced_chi2)


__all__ = [
    "CHI2_METRIC_NOTES",
    "ChiSquaredCubeSummary",
    "reduced_chi2_between_images",
    "summarize_framewise_chi2",
]
