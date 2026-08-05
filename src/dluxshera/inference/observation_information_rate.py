"""Diagnostics for prior-whitened observation information rates.

The helpers in this module operate on small dense NumPy arrays.  They do not
know about any campaign directory layout and never mutate input matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment

__all__ = [
    "DEGENERACY_RTOL",
    "PSD_ATOL",
    "PSD_RTOL",
    "DriftScenario",
    "EigenSpectrum",
    "FitDiagnostics",
    "ThresholdCrossing",
    "canonical_projected_gain",
    "canonical_projected_gains",
    "check_projected_gain_monotonicity",
    "covariance_square_root",
    "detect_degeneracy_groups",
    "deterministic_sign_vectors",
    "drift_scenario",
    "effective_rank",
    "fit_information_rate",
    "information_replacement_timescale",
    "label_physical_group",
    "matrix_psd_diagnostics",
    "mode_composition",
    "mode_overlap_assignment",
    "observability_category",
    "posterior_marginal_sigma",
    "subspace_overlap_diagnostics",
    "symmetric_eigendecomposition",
    "threshold_crossings",
    "whiten_information",
]


PSD_ATOL = 1.0e-10
"""Absolute tolerance for treating tiny negative eigenvalues as numerical."""

PSD_RTOL = 1.0e-8
"""Relative PSD tolerance scaled by the largest absolute eigenvalue."""

DEGENERACY_RTOL = 1.0e-3
"""Relative eigenvalue-gap tolerance for canonical degeneracy groups."""

EIGEN_EPS = 1.0e-15


@dataclass(frozen=True)
class EigenSpectrum:
    """Prior-whitened symmetric eigenspectrum.

    Attributes
    ----------
    eigenvalues : ndarray, shape (n,)
        Eigenvalues ordered from largest to smallest.
    eigenvectors : ndarray, shape (n, n)
        Columns are eigenvectors in the same order as ``eigenvalues``.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray


@dataclass(frozen=True)
class ThresholdCrossing:
    """First time a gain curve crosses one requested threshold."""

    threshold: float
    crossed: bool
    prefix_index: int | None
    crossing_time_s: float
    interpolated_time_s: float
    method: str


@dataclass(frozen=True)
class FitDiagnostics:
    """Linear information-rate fit diagnostics for one gain curve."""

    through_origin_slope: float
    ordinary_slope: float
    ordinary_intercept: float
    r_squared: float
    max_fractional_departure: float


@dataclass(frozen=True)
class DriftScenario:
    """Illustrative scalar random-walk tracking scenario for one mode."""

    process_variance_rate: float
    rms_drift_per_sqrt_s: float
    one_prior_sigma_drift_timescale_s: float
    steady_state_variance: float
    steady_state_sigma: float
    status: str


def _as_square_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _as_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def covariance_square_root(covariance: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return the symmetric covariance square root.

    Parameters
    ----------
    covariance : array_like, shape (n, n)
        Positive-definite prior covariance in physical parameter units.

    Returns
    -------
    ndarray, shape (n, n)
        Symmetric matrix ``W`` satisfying ``W @ W.T == covariance``.
    """

    cov = 0.5 * (_as_square_matrix(covariance, name="covariance") + np.asarray(covariance, dtype=float).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    scale = max(float(np.max(np.abs(eigvals))), 1.0)
    negative_tol = PSD_ATOL + PSD_RTOL * scale
    if np.any(eigvals <= 0.0):
        if np.min(eigvals) >= -negative_tol:
            raise ValueError("covariance must be positive definite; a near-zero eigenvalue was found.")
        raise ValueError("covariance must be positive definite.")
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def whiten_information(
    information: Sequence[Sequence[float]] | np.ndarray,
    covariance_sqrt: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return prior-whitened information gain ``W.T @ S @ W``.

    Parameters
    ----------
    information : array_like, shape (n, n)
        Physical-basis information matrix with units inverse covariance.
    covariance_sqrt : array_like, shape (n, n)
        Prior covariance square root ``W``.
    """

    info = 0.5 * (_as_square_matrix(information, name="information") + np.asarray(information, dtype=float).T)
    w = _as_square_matrix(covariance_sqrt, name="covariance_sqrt")
    if info.shape != w.shape:
        raise ValueError("information and covariance_sqrt shapes must match.")
    gain = w.T @ info @ w
    return 0.5 * (gain + gain.T)


def deterministic_sign_vectors(vectors: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Apply a deterministic sign convention to eigenvector columns.

    The largest-absolute loading in each column is made positive.  NumPy's
    first-index ``argmax`` supplies the deterministic label-order tie-breaker.
    """

    vecs = np.asarray(vectors, dtype=float).copy()
    if vecs.ndim != 2:
        raise ValueError("vectors must be a 2D array.")
    if not np.all(np.isfinite(vecs)):
        raise ValueError("vectors contains non-finite values.")
    for j in range(vecs.shape[1]):
        i = int(np.argmax(np.abs(vecs[:, j])))
        if vecs[i, j] < 0.0:
            vecs[:, j] *= -1.0
    return vecs


def symmetric_eigendecomposition(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    descending: bool = True,
    deterministic_signs: bool = True,
) -> EigenSpectrum:
    """Eigendecompose a finite symmetric matrix.

    Parameters
    ----------
    matrix : array_like, shape (n, n)
        Symmetric matrix.  It is symmetrized for numerical diagnostics.
    descending : bool, default=True
        Return eigenvalues from largest to smallest when true.
    deterministic_signs : bool, default=True
        Apply the canonical sign convention to returned eigenvectors.
    """

    sym = 0.5 * (_as_square_matrix(matrix, name="matrix") + np.asarray(matrix, dtype=float).T)
    values, vectors = np.linalg.eigh(sym)
    order = np.argsort(values)
    if descending:
        order = order[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if deterministic_signs:
        vectors = deterministic_sign_vectors(vectors)
    return EigenSpectrum(values, vectors)


def matrix_psd_diagnostics(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    psd_atol: float = PSD_ATOL,
    psd_rtol: float = PSD_RTOL,
) -> dict[str, Any]:
    """Return symmetry and PSD diagnostics for a square matrix.

    Tiny negative eigenvalues are marked as clipped for spectral diagnostics.
    Material negative eigenvalues are flagged but the input matrix is not
    modified.
    """

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return {"finite": False, "shape_ok": False, "materially_indefinite": True}
    finite = bool(np.all(np.isfinite(arr)))
    if not finite:
        return {"finite": False, "shape_ok": True, "materially_indefinite": True}
    sym = 0.5 * (arr + arr.T)
    residual = float(np.linalg.norm(arr - arr.T, ord="fro") / max(float(np.linalg.norm(sym, ord="fro")), EIGEN_EPS))
    eig = np.linalg.eigvalsh(sym)
    max_abs = max(float(np.max(np.abs(eig))) if eig.size else 0.0, 1.0)
    tol = float(psd_atol + psd_rtol * max_abs)
    negative = eig < 0.0
    material = eig < -tol
    clipped = negative & ~material
    return {
        "finite": True,
        "shape_ok": True,
        "symmetry_residual": residual,
        "minimum_eigenvalue": float(np.min(eig)),
        "maximum_eigenvalue": float(np.max(eig)),
        "negative_eigenvalue_count": int(np.count_nonzero(negative)),
        "clipped_eigenvalue_count": int(np.count_nonzero(clipped)),
        "psd_tolerance": tol,
        "materially_indefinite": bool(np.any(material)),
        "clipping_status": "materially_indefinite"
        if np.any(material)
        else ("clipped_tiny_negative" if np.any(clipped) else "not_clipped"),
    }


def detect_degeneracy_groups(
    eigenvalues: Sequence[float] | np.ndarray,
    *,
    rtol: float = DEGENERACY_RTOL,
    epsilon: float = EIGEN_EPS,
) -> tuple[tuple[int, ...], ...]:
    """Return contiguous canonical eigenmode groups with nearly equal rates."""

    vals = _as_vector(eigenvalues, name="eigenvalues")
    groups: list[list[int]] = []
    current = [0]
    for i in range(len(vals) - 1):
        denom = max(abs(float(vals[i])), abs(float(vals[i + 1])), epsilon)
        gap = abs(float(vals[i] - vals[i + 1])) / denom
        if gap <= rtol:
            current.append(i + 1)
        else:
            groups.append(current)
            current = [i + 1]
    groups.append(current)
    return tuple(tuple(group) for group in groups)


def mode_overlap_assignment(
    reference_vectors: Sequence[Sequence[float]] | np.ndarray,
    current_eigenvalues: Sequence[float] | np.ndarray,
    current_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Align current eigenvectors to canonical eigenvectors by maximum overlap.

    Parameters
    ----------
    reference_vectors, current_vectors : array_like, shape (n, n)
        Eigenvector matrices with modes stored in columns.
    current_eigenvalues : array_like, shape (n,)
        Current eigenvalues associated with ``current_vectors``.
    """

    ref = np.asarray(reference_vectors, dtype=float)
    cur = np.asarray(current_vectors, dtype=float)
    vals = _as_vector(current_eigenvalues, name="current_eigenvalues")
    if ref.shape != cur.shape or ref.ndim != 2 or vals.shape != (ref.shape[1],):
        raise ValueError("reference/current eigenspectra shapes must match.")
    signed = ref.T @ cur
    cost = -np.abs(signed)
    row_ind, col_ind = linear_sum_assignment(cost)
    aligned_vals = np.empty_like(vals)
    aligned_vecs = np.empty_like(cur)
    rows: list[dict[str, Any]] = []
    for ref_i, cur_j in sorted(zip(row_ind, col_ind), key=lambda item: int(item[0])):
        overlap = float(signed[ref_i, cur_j])
        sign = 1.0 if overlap >= 0.0 else -1.0
        aligned_vals[ref_i] = vals[cur_j]
        aligned_vecs[:, ref_i] = sign * cur[:, cur_j]
        rows.append(
            {
                "canonical_mode_id": int(ref_i),
                "assigned_mode": int(cur_j),
                "absolute_overlap": float(abs(overlap)),
                "signed_overlap": float(abs(overlap)),
                "assignment_status": "ok",
            }
        )
    return aligned_vals, aligned_vecs, rows


def subspace_overlap_diagnostics(
    reference_vectors: Sequence[Sequence[float]] | np.ndarray,
    current_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> dict[str, float]:
    """Return principal-angle diagnostics between two column subspaces."""

    ref = np.asarray(reference_vectors, dtype=float)
    cur = np.asarray(current_vectors, dtype=float)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape != cur.shape:
        raise ValueError("subspace arrays must have matching 2D shapes.")
    singular_values = np.linalg.svd(ref.T @ cur, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = subspace_angles(ref, cur)
    return {
        "minimum_subspace_singular_value": float(np.min(singular_values)),
        "mean_subspace_singular_value": float(np.mean(singular_values)),
        "maximum_principal_angle_deg": float(np.max(angles) * 180.0 / np.pi) if angles.size else 0.0,
    }


def canonical_projected_gain(
    gain_matrix: Sequence[Sequence[float]] | np.ndarray,
    canonical_vector: Sequence[float] | np.ndarray,
) -> float:
    """Return ``v.T @ G @ v`` for one canonical whitened mode."""

    gain = _as_square_matrix(gain_matrix, name="gain_matrix")
    vector = _as_vector(canonical_vector, name="canonical_vector")
    if gain.shape != (vector.size, vector.size):
        raise ValueError("gain_matrix shape must match canonical_vector length.")
    return float(vector @ (0.5 * (gain + gain.T)) @ vector)


def canonical_projected_gains(
    gain_matrix: Sequence[Sequence[float]] | np.ndarray,
    canonical_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return projected gains for all canonical eigenvector columns."""

    gain = 0.5 * (_as_square_matrix(gain_matrix, name="gain_matrix") + np.asarray(gain_matrix, dtype=float).T)
    vectors = np.asarray(canonical_vectors, dtype=float)
    if vectors.ndim != 2 or gain.shape != (vectors.shape[0], vectors.shape[0]):
        raise ValueError("canonical_vectors must have shape (n, m) for an (n, n) gain matrix.")
    return np.einsum("ik,ij,jk->k", vectors, gain, vectors)


def check_projected_gain_monotonicity(
    gains: Sequence[float] | np.ndarray,
    *,
    atol: float = 1.0e-10,
) -> dict[str, Any]:
    """Diagnose whether a projected cumulative gain curve is non-decreasing."""

    values = _as_vector(gains, name="gains")
    if values.size < 2:
        return {"monotonic": True, "minimum_delta": 0.0, "violation_count": 0}
    delta = np.diff(values)
    violations = delta < -float(atol)
    return {
        "monotonic": bool(not np.any(violations)),
        "minimum_delta": float(np.min(delta)),
        "violation_count": int(np.count_nonzero(violations)),
    }


def threshold_crossings(
    times_s: Sequence[float] | np.ndarray,
    gains: Sequence[float] | np.ndarray,
    thresholds: Sequence[float],
) -> tuple[ThresholdCrossing, ...]:
    """Find exact and linearly interpolated first threshold crossings."""

    times = _as_vector(times_s, name="times_s")
    values = _as_vector(gains, name="gains")
    if times.shape != values.shape:
        raise ValueError("times_s and gains must have matching shapes.")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("times_s must be non-decreasing.")
    out: list[ThresholdCrossing] = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        reached = np.flatnonzero(values >= threshold)
        if reached.size == 0:
            out.append(ThresholdCrossing(threshold, False, None, np.nan, np.nan, "not_crossed"))
            continue
        idx = int(reached[0])
        if idx == 0:
            interp = float(times[0])
            method = "first_prefix"
        else:
            t0, t1 = float(times[idx - 1]), float(times[idx])
            g0, g1 = float(values[idx - 1]), float(values[idx])
            if np.isclose(g1, g0):
                interp = t1
                method = "integer_prefix"
            else:
                frac = np.clip((threshold - g0) / (g1 - g0), 0.0, 1.0)
                interp = t0 + frac * (t1 - t0)
                method = "linear_interpolation"
        out.append(ThresholdCrossing(threshold, True, idx, float(times[idx]), interp, method))
    return tuple(out)


def fit_information_rate(
    times_s: Sequence[float] | np.ndarray,
    gains: Sequence[float] | np.ndarray,
) -> FitDiagnostics:
    """Fit cumulative projected gain versus elapsed time."""

    times = _as_vector(times_s, name="times_s")
    values = _as_vector(gains, name="gains")
    if times.shape != values.shape:
        raise ValueError("times_s and gains must have matching shapes.")
    denom = float(times @ times)
    through = float((times @ values) / denom) if denom > 0.0 else np.nan
    if times.size >= 2:
        ordinary_slope, ordinary_intercept = np.polyfit(times, values, deg=1)
        pred = ordinary_slope * times + ordinary_intercept
        ss_res = float(np.sum((values - pred) ** 2))
        ss_tot = float(np.sum((values - np.mean(values)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    else:
        ordinary_slope = through
        ordinary_intercept = 0.0
        r2 = np.nan
    through_pred = through * times if np.isfinite(through) else np.full_like(values, np.nan)
    scale = np.maximum(np.abs(values), EIGEN_EPS)
    departures = np.abs(values - through_pred) / scale
    return FitDiagnostics(
        through_origin_slope=through,
        ordinary_slope=float(ordinary_slope),
        ordinary_intercept=float(ordinary_intercept),
        r_squared=float(r2),
        max_fractional_departure=float(np.max(departures)) if departures.size else np.nan,
    )


def effective_rank(eigenvalues: Sequence[float] | np.ndarray) -> float:
    """Return entropy effective rank for non-negative eigenvalues."""

    vals = np.clip(_as_vector(eigenvalues, name="eigenvalues"), 0.0, None)
    total = float(np.sum(vals))
    if total <= 0.0:
        return 0.0
    p = vals / total
    p = p[p > 0.0]
    return float(np.exp(-np.sum(p * np.log(p))))


def label_physical_group(label: str) -> str:
    """Map a canonical theta label to a coarse physical group."""

    text = str(label)
    if text.startswith("source."):
        return "source"
    if text == "optics.plate_scale_as_per_pix":
        return "plate_scale"
    if "optics.primary.zernike_coeffs_nm" in text:
        return "m1_zernike"
    if "optics.secondary.zernike_coeffs_nm" in text:
        return "m2_zernike"
    return "other"


def mode_composition(
    labels: Sequence[str],
    covariance_sqrt: Sequence[Sequence[float]] | np.ndarray,
    canonical_vectors: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return label-level and mode-level physical composition rows.

    ``canonical_vectors`` are prior-whitened mode columns.  Composition fractions
    use squared whitened coefficients.  Physical one-prior-sigma directions are
    ``W @ v`` in native parameter units.
    """

    label_tuple = tuple(str(label) for label in labels)
    w = _as_square_matrix(covariance_sqrt, name="covariance_sqrt")
    vectors = np.asarray(canonical_vectors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(label_tuple) or w.shape != (len(label_tuple), len(label_tuple)):
        raise ValueError("labels, covariance_sqrt, and canonical_vectors shapes are incompatible.")
    physical = w @ vectors
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for mode_id in range(vectors.shape[1]):
        coeff = vectors[:, mode_id]
        frac = np.square(coeff) / max(float(np.sum(np.square(coeff))), EIGEN_EPS)
        order = np.argsort(-np.abs(coeff), kind="mergesort")
        group_norms: dict[str, float] = {}
        for rank, index in enumerate(order, start=1):
            group = label_physical_group(label_tuple[int(index)])
            group_norms[group] = group_norms.get(group, 0.0) + float(frac[int(index)])
            rows.append(
                {
                    "canonical_mode_id": int(mode_id),
                    "theta_label": label_tuple[int(index)],
                    "whitened_coefficient": float(coeff[int(index)]),
                    "absolute_whitened_coefficient": float(abs(coeff[int(index)])),
                    "squared_composition_fraction": float(frac[int(index)]),
                    "physical_one_sigma_direction_coefficient": float(physical[int(index), mode_id]),
                    "physical_group": group,
                    "rank_within_mode": int(rank),
                }
            )
        dominant_group = sorted(group_norms.items(), key=lambda item: (-item[1], item[0]))[0][0]
        participation = 1.0 / max(float(np.sum(frac**2)), EIGEN_EPS)
        summaries.append(
            {
                "canonical_mode_id": int(mode_id),
                "dominant_labels": ";".join(label_tuple[int(i)] for i in order[:5]),
                "source_group_squared_norm": float(group_norms.get("source", 0.0)),
                "plate_scale_squared_norm": float(group_norms.get("plate_scale", 0.0)),
                "m1_zernike_squared_norm": float(group_norms.get("m1_zernike", 0.0)),
                "m2_zernike_squared_norm": float(group_norms.get("m2_zernike", 0.0)),
                "other_squared_norm": float(group_norms.get("other", 0.0)),
                "dominant_physical_group": dominant_group,
                "dominant_mirror": "M1"
                if group_norms.get("m1_zernike", 0.0) >= group_norms.get("m2_zernike", 0.0)
                and group_norms.get("m1_zernike", 0.0) > 0.0
                else ("M2" if group_norms.get("m2_zernike", 0.0) > 0.0 else ""),
                "participation_ratio": float(participation),
            }
        )
    return rows, summaries


def posterior_marginal_sigma(
    prior_precision: Sequence[Sequence[float]] | np.ndarray,
    information: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return marginal sigma from ``prior_precision + information``."""

    prior = 0.5 * (_as_square_matrix(prior_precision, name="prior_precision") + np.asarray(prior_precision, dtype=float).T)
    info = 0.5 * (_as_square_matrix(information, name="information") + np.asarray(information, dtype=float).T)
    if prior.shape != info.shape:
        raise ValueError("prior_precision and information shapes must match.")
    covariance = np.linalg.pinv(prior + info, hermitian=True)
    return np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def information_replacement_timescale(rate: float) -> float:
    """Return ``1 / rate`` for positive prior-relative gain rates."""

    r = float(rate)
    return 1.0 / r if np.isfinite(r) and r > 0.0 else np.nan


def observability_category(rate: float, *, threshold: float = 1.0) -> str:
    """Categorize when a mode reaches one requested gain threshold."""

    r = float(rate)
    if not np.isfinite(r) or r <= 0.0:
        return "unresolved"
    t = threshold / r
    if t <= 1.0:
        return "subblock_scale"
    if t <= 30.0:
        return "window_scale"
    if t <= 300.0:
        return "five_minute_scale"
    if t <= 1800.0:
        return "thirty_minute_projected"
    return "multi_observation_or_external"


def drift_scenario(rate: float, target_sigma_fraction: float) -> DriftScenario:
    """Return illustrative random-walk drift quantities for one mode.

    The scalar model is ``dP/dt = q - r * P**2``.  For target steady-state
    standard deviation fraction ``f``, ``q_max = r * f**4``.
    """

    r = float(rate)
    f = float(target_sigma_fraction)
    if not np.isfinite(r) or r <= 0.0 or not np.isfinite(f) or f <= 0.0:
        return DriftScenario(np.nan, np.nan, np.nan, np.nan, np.nan, "non_positive_rate_or_fraction")
    q = r * f**4
    variance = np.sqrt(q / r)
    sigma = (q / r) ** 0.25
    return DriftScenario(
        process_variance_rate=float(q),
        rms_drift_per_sqrt_s=float(np.sqrt(q)),
        one_prior_sigma_drift_timescale_s=float(1.0 / q) if q > 0.0 else np.nan,
        steady_state_variance=float(variance),
        steady_state_sigma=float(sigma),
        status="ok",
    )
