from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .catalog import SampleCatalog

__all__ = [
    "compute_capture_metrics",
    "compute_eigenmode_metrics",
    "compute_regression_metrics",
    "metrics_by_group",
    "transform_z_to_physical",
]


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


def _quantiles(values: np.ndarray, probs: Sequence[float]) -> dict[str, float | None]:
    if values.size == 0:
        return {f"q{int(p * 100):02d}": None for p in probs}
    return {f"q{int(p * 100):02d}": float(np.quantile(values, p)) for p in probs}


def compute_capture_metrics(
    y_pred_z: np.ndarray,
    y_true_z: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Compute correction/capture diagnostics for radial-distance studies."""
    pred = np.asarray(y_pred_z, dtype=np.float64)
    truth = np.asarray(y_true_z, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("Predictions and truth must be same-shape 2D arrays.")
    residual = truth - pred
    d0 = np.linalg.norm(truth, axis=1)
    d1 = np.linalg.norm(residual, axis=1)
    rho = d1 / np.maximum(d0, float(epsilon))
    alignment = _safe_cosine(pred, truth)
    mse = float(np.mean((pred - truth) ** 2)) if pred.size else 0.0
    baseline_mse = float(np.mean(truth**2)) if truth.size else 0.0
    thresholds = (100.0, 250.0, 500.0, 1000.0, 2000.0)
    return {
        "schema_version": "dluxshera_ml_capture_metrics/1",
        "sample_count": int(pred.shape[0]),
        "fisher_rmse": float(np.sqrt(mse)),
        "mse_skill": None if baseline_mse <= 0.0 else float(1.0 - mse / baseline_mse),
        "correction_vector_alignment_mean": _nanmean(alignment),
        "d0_mean": float(np.mean(d0)) if d0.size else None,
        "d0_median": float(np.median(d0)) if d0.size else None,
        "d1_mean": float(np.mean(d1)) if d1.size else None,
        "d1_median": float(np.median(d1)) if d1.size else None,
        "rho_mean": float(np.mean(rho)) if rho.size else None,
        "rho_median": float(np.median(rho)) if rho.size else None,
        "rho_quantiles": _quantiles(rho, (0.1, 0.25, 0.5, 0.75, 0.9)),
        "fraction_rho_lt_1": float(np.mean(rho < 1.0)) if rho.size else None,
        "remaining_distance_threshold_fractions": {
            str(int(threshold)): float(np.mean(d1 <= threshold)) if d1.size else None
            for threshold in thresholds
        },
    }


def compute_eigenmode_metrics(
    y_pred_z: np.ndarray,
    y_true_z: np.ndarray,
    *,
    eigenvectors: np.ndarray,
    eigenvalues: Sequence[float],
) -> dict[str, Any]:
    """Compute validation diagnostics after projecting z-coordinate errors into eigenmodes."""
    pred = np.asarray(y_pred_z, dtype=np.float64)
    truth = np.asarray(y_true_z, dtype=np.float64)
    vectors = np.asarray(eigenvectors, dtype=np.float64)
    values = np.asarray(eigenvalues, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 2:
        raise ValueError("Predictions and truth must be same-shape 2D arrays.")
    if vectors.shape != (pred.shape[1], pred.shape[1]):
        raise ValueError("eigenvectors must have shape (science_dim, science_dim).")
    if values.shape != (pred.shape[1],):
        raise ValueError("eigenvalues length must match science dimension.")
    coeff = (pred - truth) @ vectors
    per_mode_mse = np.mean(coeff**2, axis=0) if coeff.size else np.zeros((pred.shape[1],), dtype=float)
    per_mode_rmse = np.sqrt(per_mode_mse)
    n = int(per_mode_rmse.shape[0])
    third = max(n // 3, 1)

    def group_payload(indices: np.ndarray) -> dict[str, Any]:
        if indices.size == 0:
            return {"mode_count": 0, "rmse": None, "mse": None}
        mse = float(np.mean(per_mode_mse[indices]))
        return {
            "mode_count": int(indices.size),
            "first_mode_index": int(indices[0]),
            "last_mode_index": int(indices[-1]),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "eigenvalue_min": float(np.min(values[indices])),
            "eigenvalue_max": float(np.max(values[indices])),
        }

    strong = np.arange(0, third, dtype=int)
    weak = np.arange(max(n - third, 0), n, dtype=int)
    middle = np.arange(third, max(n - third, third), dtype=int)
    if middle.size == 0:
        middle = np.arange(0, n, dtype=int)
    return {
        "schema_version": "dluxshera_ml_eigenmode_metrics/1",
        "sample_count": int(pred.shape[0]),
        "coordinate_convention": "error_z_eigen = Q^T (pred_z - true_z), with eigenvectors stored as columns",
        "per_mode_rmse": {str(idx): float(value) for idx, value in enumerate(per_mode_rmse)},
        "mode_groups": {
            "strong": group_payload(strong),
            "middle": group_payload(middle),
            "weak": group_payload(weak),
        },
    }


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
