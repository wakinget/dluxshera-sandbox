from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .catalog import SampleCatalog

__all__ = ["compute_regression_metrics", "metrics_by_group", "transform_z_to_physical"]


def transform_z_to_physical(z_delta: np.ndarray, fisher_sigmas: Sequence[float]) -> np.ndarray:
    """Transform Fisher-scaled corrections to native physical units."""
    return np.asarray(z_delta, dtype=np.float64) * np.asarray(fisher_sigmas, dtype=np.float64)


def _safe_cosine(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    dot = np.sum(pred * truth, axis=1)
    out = np.full((pred.shape[0],), np.nan, dtype=np.float64)
    mask = denom > 1.0e-12
    out[mask] = dot[mask] / denom[mask]
    return out


def _safe_norm_ratio(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    truth_norm = np.linalg.norm(truth, axis=1)
    out = np.full((pred.shape[0],), np.nan, dtype=np.float64)
    mask = truth_norm > 1.0e-12
    out[mask] = np.linalg.norm(pred[mask], axis=1) / truth_norm[mask]
    return out


def _nanmean(values: np.ndarray) -> float | None:
    if values.size == 0 or np.all(np.isnan(values)):
        return None
    return float(np.nanmean(values))


def _basic_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    labels: Sequence[str],
    prefix: str,
) -> dict[str, Any]:
    error = y_pred - y_true
    rmse = np.sqrt(np.mean(error**2, axis=0))
    mae = np.mean(np.abs(error), axis=0)
    vector_error_norm = np.linalg.norm(error, axis=1)
    return {
        f"{prefix}_overall_rmse": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_mean_vector_error_norm": float(np.mean(vector_error_norm)),
        f"{prefix}_median_vector_error_norm": float(np.median(vector_error_norm)),
        f"{prefix}_per_parameter_rmse": {
            str(label): float(value) for label, value in zip(labels, rmse)
        },
        f"{prefix}_per_parameter_mae": {
            str(label): float(value) for label, value in zip(labels, mae)
        },
    }


def compute_regression_metrics(
    y_pred_z: np.ndarray,
    y_true_z: np.ndarray,
    *,
    catalog: SampleCatalog | None = None,
    fisher_sigmas: Sequence[float] | None = None,
    parameter_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute Fisher-scaled and physical-unit pairwise correction metrics."""
    pred = np.asarray(y_pred_z, dtype=np.float64)
    truth = np.asarray(y_true_z, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match truth shape {truth.shape}.")
    if pred.ndim != 2:
        raise ValueError("Predictions and truth must be 2D arrays.")
    if catalog is not None:
        fisher_sigmas = catalog.fisher_sigmas
        parameter_labels = catalog.parameter_labels
    if parameter_labels is None:
        parameter_labels = tuple(f"z[{idx}]" for idx in range(pred.shape[1]))
    if len(parameter_labels) != pred.shape[1]:
        raise ValueError("parameter_labels length must match prediction dimension.")
    metrics = _basic_metrics(pred, truth, labels=parameter_labels, prefix="fisher")
    metrics["fisher_alignment_cosine_mean"] = _nanmean(_safe_cosine(pred, truth))
    metrics["fisher_correction_norm_ratio_mean"] = _nanmean(_safe_norm_ratio(pred, truth))
    metrics["sample_count"] = int(pred.shape[0])
    if fisher_sigmas is not None:
        pred_phys = transform_z_to_physical(pred, fisher_sigmas)
        truth_phys = transform_z_to_physical(truth, fisher_sigmas)
        metrics.update(
            _basic_metrics(
                pred_phys,
                truth_phys,
                labels=parameter_labels,
                prefix="physical",
            )
        )
    return metrics


def metrics_by_group(
    y_pred_z: np.ndarray,
    y_true_z: np.ndarray,
    groups: Sequence[str],
    *,
    catalog: SampleCatalog | None = None,
    fisher_sigmas: Sequence[float] | None = None,
    parameter_labels: Sequence[str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    """Compute regression metrics independently for each group label."""
    pred = np.asarray(y_pred_z)
    truth = np.asarray(y_true_z)
    if pred.shape[0] != len(groups):
        raise ValueError("groups length must match prediction row count.")
    out: dict[str, Mapping[str, Any]] = {}
    for group in sorted(set(str(v) for v in groups)):
        mask = np.asarray([str(v) == group for v in groups], dtype=bool)
        out[group] = compute_regression_metrics(
            pred[mask],
            truth[mask],
            catalog=catalog,
            fisher_sigmas=fisher_sigmas,
            parameter_labels=parameter_labels,
        )
    return out
