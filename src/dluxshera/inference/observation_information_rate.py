"""Diagnostics for prior-whitened observation information rates.

The helpers in this module operate on small dense NumPy arrays.  They do not
know about any campaign directory layout and never mutate input matrices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment

__all__ = [
    "DEGENERACY_RTOL",
    "PSD_ATOL",
    "PSD_RTOL",
    "QUASI_DEGENERACY_RTOL",
    "DriftScenario",
    "EigenSpectrum",
    "FitDiagnostics",
    "ModeAssignment",
    "PSDProjectionResult",
    "SequentialInformationGateUpdate",
    "ThresholdCrossing",
    "canonical_projected_gain",
    "canonical_projected_gains",
    "canonical_physical_directions",
    "check_projected_gain_monotonicity",
    "covariance_square_root",
    "deduplicate_warnings",
    "detect_degeneracy_groups",
    "detect_quasi_degeneracy_groups",
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
    "precision_normalized_projected_gain",
    "precision_normalized_projected_gains",
    "project_information_to_psd",
    "resolve_unique_mode_assignments",
    "simulate_sequential_information_gate",
    "subspace_overlap_diagnostics",
    "symmetric_eigendecomposition",
    "threshold_crossings",
    "update_precision_with_information",
    "whiten_information",
]


PSD_ATOL = 1.0e-10
"""Absolute tolerance for treating tiny negative eigenvalues as numerical."""

PSD_RTOL = 1.0e-8
"""Relative PSD tolerance scaled by the largest absolute eigenvalue."""

DEGENERACY_RTOL = 1.0e-3
"""Relative eigenvalue-gap tolerance for canonical degeneracy groups."""

QUASI_DEGENERACY_RTOL = 1.0e-2
"""Relative eigenvalue-gap tolerance for quasi-degenerate mode groups."""

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


@dataclass(frozen=True)
class SequentialInformationGateUpdate:
    """One closure in a covariance-only sequential information gate.

    The update records precision/covariance diagnostics before and after adding
    a buffered physical-basis information matrix.  It intentionally contains no
    score, posterior mean, innovation, or reference-trajectory fields.
    """

    update_index: int
    start_index: int
    end_index: int
    block_length: int
    block_duration_s: float
    cumulative_elapsed_s: float
    selected_mode_ids: tuple[int, ...]
    gains: tuple[float, ...]
    precision_norms_before: tuple[float, ...]
    absolute_buffered_information: tuple[float, ...]
    controlling_mode_id: int
    minimum_gain: float
    maximum_gain: float
    information_trace: float
    precision_trace_before: float
    precision_trace_after: float
    covariance_trace_before: float
    covariance_trace_after: float
    closure_reason: str
    triggered_naturally: bool
    maximum_latency_reached: bool
    historical_window_boundary_flush: bool
    end_of_scope_flush: bool
    information_only_status: str


@dataclass(frozen=True)
class ModeAssignment:
    """Resolved unique canonical mode assignment for one physical concept."""

    concept: str
    canonical_mode_id: int | None
    squared_loading: float
    assignment_rank: int
    next_best_mode_id: int | None
    next_best_squared_loading: float
    assignment_status: str


@dataclass(frozen=True)
class PSDProjectionResult:
    """Information-matrix PSD projection result and diagnostics."""

    matrix: np.ndarray
    diagnostics: dict[str, Any]


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


def _validate_spd_precision(precision: np.ndarray, *, name: str) -> np.ndarray:
    prec = 0.5 * (_as_square_matrix(precision, name=name) + np.asarray(precision, dtype=float).T)
    try:
        np.linalg.cholesky(prec)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite.") from exc
    return prec


def _inverse_spd(matrix: np.ndarray, *, name: str) -> np.ndarray:
    mat = _validate_spd_precision(matrix, name=name)
    identity = np.eye(mat.shape[0])
    try:
        chol = np.linalg.cholesky(mat)
        y = np.linalg.solve(chol, identity)
        inv = np.linalg.solve(chol.T, y)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite.") from exc
    return 0.5 * (inv + inv.T)


def canonical_physical_directions(
    covariance_sqrt: Sequence[Sequence[float]] | np.ndarray,
    canonical_vectors: Sequence[Sequence[float]] | np.ndarray,
    prior_precision: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return fixed physical canonical directions normalized by ``P0``.

    Parameters
    ----------
    covariance_sqrt : array_like, shape (n, n)
        Symmetric square root ``W0`` of the initial covariance.
    canonical_vectors : array_like, shape (n, m)
        Prior-whitened canonical eigenvectors stored in columns.
    prior_precision : array_like, shape (n, n)
        Initial prior precision ``P0`` used to normalize each physical
        direction so that ``d_k.T @ P0 @ d_k == 1``.

    Returns
    -------
    ndarray, shape (n, m)
        Physical directions ``d_k`` in fixed native parameter coordinates.
    """

    w = _as_square_matrix(covariance_sqrt, name="covariance_sqrt")
    vectors = np.asarray(canonical_vectors, dtype=float)
    precision = _validate_spd_precision(np.asarray(prior_precision, dtype=float), name="prior_precision")
    if vectors.ndim != 2 or w.shape[1] != vectors.shape[0] or precision.shape != (w.shape[0], w.shape[0]):
        raise ValueError("covariance_sqrt, canonical_vectors, and prior_precision shapes are incompatible.")
    physical = w @ vectors
    out = physical.copy()
    for mode_id in range(out.shape[1]):
        denom = float(out[:, mode_id] @ precision @ out[:, mode_id])
        if not np.isfinite(denom) or denom <= 0.0:
            raise ValueError("canonical physical direction has non-positive prior precision norm.")
        out[:, mode_id] /= np.sqrt(denom)
    return out


def precision_normalized_projected_gain(
    precision: Sequence[Sequence[float]] | np.ndarray,
    information: Sequence[Sequence[float]] | np.ndarray,
    physical_direction: Sequence[float] | np.ndarray,
) -> float:
    """Return current-prior-relative gain for one fixed physical direction.

    The gain is ``(d.T @ S @ d) / (d.T @ P_current @ d)``.  This is equivalent
    to projecting the buffered information after normalizing ``d`` in the
    current precision metric.
    """

    prec = _validate_spd_precision(np.asarray(precision, dtype=float), name="precision")
    info = 0.5 * (_as_square_matrix(information, name="information") + np.asarray(information, dtype=float).T)
    direction = _as_vector(physical_direction, name="physical_direction")
    if prec.shape != info.shape or prec.shape != (direction.size, direction.size):
        raise ValueError("precision, information, and physical_direction shapes are incompatible.")
    denom = float(direction @ prec @ direction)
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError("current precision norm must be positive.")
    return float(direction @ info @ direction / denom)


def precision_normalized_projected_gains(
    precision: Sequence[Sequence[float]] | np.ndarray,
    information: Sequence[Sequence[float]] | np.ndarray,
    physical_directions: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return current-precision-normalized gains for direction columns."""

    prec = _validate_spd_precision(np.asarray(precision, dtype=float), name="precision")
    info = 0.5 * (_as_square_matrix(information, name="information") + np.asarray(information, dtype=float).T)
    directions = np.asarray(physical_directions, dtype=float)
    if directions.ndim != 2 or prec.shape != info.shape or prec.shape != (directions.shape[0], directions.shape[0]):
        raise ValueError("precision, information, and physical_directions shapes are incompatible.")
    precision_norms = np.einsum("ik,ij,jk->k", directions, prec, directions)
    if np.any(~np.isfinite(precision_norms)) or np.any(precision_norms <= 0.0):
        raise ValueError("all current precision norms must be positive finite values.")
    absolute = np.einsum("ik,ij,jk->k", directions, info, directions)
    return absolute / precision_norms


def update_precision_with_information(
    precision: Sequence[Sequence[float]] | np.ndarray,
    information: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return ``precision + information`` after validation and symmetrization."""

    prec = _validate_spd_precision(np.asarray(precision, dtype=float), name="precision")
    info = 0.5 * (_as_square_matrix(information, name="information") + np.asarray(information, dtype=float).T)
    if prec.shape != info.shape:
        raise ValueError("precision and information shapes must match.")
    updated = 0.5 * (prec + info + (prec + info).T)
    return _validate_spd_precision(updated, name="updated_precision")


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


def project_information_to_psd(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    expected_dimension: int | None = None,
    psd_atol: float = PSD_ATOL,
    psd_rtol: float = PSD_RTOL,
) -> PSDProjectionResult:
    """Project one finite symmetric information matrix to PSD.

    This helper applies the same PSD tolerance policy used by
    :func:`matrix_psd_diagnostics`.  Materially negative eigenvalues remain
    errors.  Tolerated tiny-negative eigenvalues are set to zero and the matrix
    is reconstructed in the original physical basis.  The input is never
    mutated.

    Parameters
    ----------
    matrix : array_like, shape (n, n)
        Physical-basis information matrix.
    expected_dimension : int, optional
        Required dimension when supplied.
    psd_atol, psd_rtol : float
        Absolute and relative PSD tolerances.

    Returns
    -------
    PSDProjectionResult
        Projected matrix plus raw and projected diagnostics.

    Raises
    ------
    ValueError
        If the matrix is non-finite, has an invalid shape, or is materially
        indefinite under the configured PSD tolerance.
    """

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("information matrix must be a square matrix.")
    if expected_dimension is not None and arr.shape != (int(expected_dimension), int(expected_dimension)):
        raise ValueError(
            "information matrix shape does not match expected dimension: "
            f"expected {(int(expected_dimension), int(expected_dimension))}, got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("information matrix contains non-finite values.")
    sym = 0.5 * (arr + arr.T)
    raw_eigenvalues, raw_eigenvectors = np.linalg.eigh(sym)
    max_abs = max(float(np.max(np.abs(raw_eigenvalues))) if raw_eigenvalues.size else 0.0, 1.0)
    tolerance = float(psd_atol + psd_rtol * max_abs)
    raw_negative = raw_eigenvalues < 0.0
    material = raw_eigenvalues < -tolerance
    if np.any(material):
        raise ValueError(
            "information matrix is materially indefinite: "
            f"minimum eigenvalue {float(np.min(raw_eigenvalues)):.12g} is below "
            f"the PSD tolerance {-tolerance:.12g}."
        )
    projected_eigenvalues = np.maximum(raw_eigenvalues, 0.0)
    clipped = raw_negative
    if np.any(clipped):
        projected = (raw_eigenvectors * projected_eigenvalues) @ raw_eigenvectors.T
    else:
        projected = sym.copy()
    projected = 0.5 * (projected + projected.T)
    if not np.all(np.isfinite(projected)):
        raise ValueError("PSD projection produced non-finite values.")
    projected_eigenvalues_check = np.linalg.eigvalsh(projected)
    projected_negative = projected_eigenvalues_check < 0.0
    projected_material = projected_eigenvalues_check < -tolerance
    if np.any(projected_material):
        raise ValueError("PSD projection produced a materially negative eigenvalue.")
    delta = projected - sym
    sym_norm = max(float(np.linalg.norm(sym, ord="fro")), EIGEN_EPS)
    status = "clipped_tiny_negative" if np.any(clipped) else "not_needed"
    diagnostics = {
        "raw_minimum_eigenvalue": float(np.min(raw_eigenvalues)),
        "raw_maximum_eigenvalue": float(np.max(raw_eigenvalues)),
        "raw_negative_eigenvalue_count": int(np.count_nonzero(raw_negative)),
        "psd_tolerance": tolerance,
        "psd_projection_applied": bool(np.any(clipped)),
        "psd_projection_clipped_eigenvalue_count": int(np.count_nonzero(clipped)),
        "projected_minimum_eigenvalue": float(np.min(projected_eigenvalues_check)),
        "projected_negative_eigenvalue_count": int(np.count_nonzero(projected_negative)),
        "projection_frobenius_delta": float(np.linalg.norm(delta, ord="fro")),
        "projection_relative_frobenius_delta": float(np.linalg.norm(delta, ord="fro") / sym_norm),
        "projection_max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "projection_status": status,
    }
    return PSDProjectionResult(matrix=projected, diagnostics=diagnostics)


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


def detect_quasi_degeneracy_groups(
    eigenvalues: Sequence[float] | np.ndarray,
    *,
    quasi_rtol: float = QUASI_DEGENERACY_RTOL,
    strict_rtol: float = DEGENERACY_RTOL,
    epsilon: float = EIGEN_EPS,
) -> tuple[tuple[int, ...], ...]:
    """Return quasi-degenerate groups without changing strict classification.

    Parameters
    ----------
    eigenvalues : array_like, shape (n,)
        Canonical eigenvalues ordered consistently with canonical mode IDs.
    quasi_rtol : float, default=``QUASI_DEGENERACY_RTOL``
        Relative adjacent-gap tolerance for cautionary quasi groups.
    strict_rtol : float, default=``DEGENERACY_RTOL``
        Formal degeneracy tolerance.  ``quasi_rtol`` must be larger.

    Returns
    -------
    tuple of tuple of int
        Contiguous groups under the quasi tolerance.  Singleton groups are kept
        so callers can build complete mode-to-group maps deterministically.
    """

    if not np.isfinite(quasi_rtol) or not np.isfinite(strict_rtol) or quasi_rtol <= strict_rtol:
        raise ValueError("quasi_rtol must be finite and greater than strict_rtol.")
    return detect_degeneracy_groups(eigenvalues, rtol=float(quasi_rtol), epsilon=epsilon)


def resolve_unique_mode_assignments(
    loading_fractions: Mapping[int, Mapping[str, float]],
    requested_concepts: Sequence[str],
    *,
    weak_threshold: float = 0.25,
    ambiguity_ratio: float = 0.9,
) -> tuple[ModeAssignment, ...]:
    """Resolve unique canonical modes for requested physical concepts.

    ``loading_fractions`` maps canonical mode IDs to squared whitened loading
    fractions by physical label.  A deterministic maximum-weight bipartite
    assignment is used.  Mode ID tie-breaks are encoded as tiny cost offsets,
    preserving stable behavior when weights are exactly equal.
    """

    concepts = tuple(str(item) for item in requested_concepts)
    if not concepts:
        raise ValueError("requested_concepts must be non-empty.")
    modes = tuple(sorted(int(mode) for mode in loading_fractions))
    if not modes:
        raise ValueError("loading_fractions must contain at least one mode.")
    for concept in concepts:
        if not any(concept in fractions for fractions in loading_fractions.values()):
            raise KeyError(f"Missing physical loading label {concept!r}.")
    weights = np.zeros((len(concepts), len(modes)), dtype=float)
    for i, concept in enumerate(concepts):
        for j, mode in enumerate(modes):
            weights[i, j] = float(loading_fractions[mode].get(concept, 0.0))
    if len(modes) < len(concepts):
        raise ValueError("not enough canonical modes for a unique assignment.")
    tie = np.asarray(modes, dtype=float)[None, :] * 1.0e-12
    row_ind, col_ind = linear_sum_assignment(-(weights - tie))
    assigned_by_row = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    assignments: list[ModeAssignment] = []
    for i, concept in enumerate(concepts):
        col = assigned_by_row.get(i)
        if col is None:
            assignments.append(ModeAssignment(concept, None, 0.0, 0, None, 0.0, "unassigned"))
            continue
        order = sorted(range(len(modes)), key=lambda j: (-weights[i, j], modes[j]))
        best_mode = modes[col]
        best_weight = float(weights[i, col])
        rank = int(order.index(col) + 1)
        next_col = next((j for j in order if j != col), None)
        next_mode = modes[next_col] if next_col is not None else None
        next_weight = float(weights[i, next_col]) if next_col is not None else np.nan
        status_parts = ["ok"]
        if best_weight < weak_threshold:
            status_parts.append("weak_assignment")
        if next_col is not None and next_weight >= ambiguity_ratio * max(best_weight, EIGEN_EPS):
            status_parts.append("ambiguous_assignment")
        assignments.append(
            ModeAssignment(
                concept=concept,
                canonical_mode_id=int(best_mode),
                squared_loading=best_weight,
                assignment_rank=rank,
                next_best_mode_id=None if next_mode is None else int(next_mode),
                next_best_squared_loading=next_weight,
                assignment_status=";".join(status_parts),
            )
        )
    return tuple(assignments)


def simulate_sequential_information_gate(
    prior_precision: Sequence[Sequence[float]] | np.ndarray,
    information_matrices: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    durations_s: Sequence[float] | np.ndarray,
    physical_directions: Sequence[Sequence[float]] | np.ndarray,
    selected_mode_ids: Sequence[int],
    *,
    gain_threshold: float,
    minimum_subblocks: int,
    maximum_subblocks: int,
    boundary_after: Sequence[bool] | None = None,
) -> tuple[SequentialInformationGateUpdate, ...]:
    """Simulate one covariance-only sequential information gate policy.

    The simulation adds accepted subblock information matrices in order, closes
    buffered blocks by information threshold, maximum latency, historical
    boundary, or final end-of-scope flush, and updates precision as
    ``P_after = P_before + S_buffer``.  It never updates a mean or score.
    """

    precision = _validate_spd_precision(np.asarray(prior_precision, dtype=float), name="prior_precision")
    infos = np.asarray(information_matrices, dtype=float)
    durations = _as_vector(durations_s, name="durations_s")
    directions = np.asarray(physical_directions, dtype=float)
    modes = tuple(int(mode) for mode in selected_mode_ids)
    threshold = float(gain_threshold)
    min_count = int(minimum_subblocks)
    max_count = int(maximum_subblocks)
    if infos.ndim != 3 or infos.shape[1] != infos.shape[2] or infos.shape[1:] != precision.shape:
        raise ValueError("information_matrices must have shape (n_blocks, n, n) matching prior_precision.")
    if durations.shape != (infos.shape[0],):
        raise ValueError("durations_s length must match information_matrices.")
    if directions.ndim != 2 or directions.shape[0] != precision.shape[0]:
        raise ValueError("physical_directions must have shape (n, n_modes).")
    if any(mode < 0 or mode >= directions.shape[1] for mode in modes):
        raise ValueError("selected_mode_ids contains an out-of-range mode.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("gain_threshold must be positive and finite.")
    if min_count < 1 or max_count < min_count:
        raise ValueError("minimum/maximum subblocks are invalid.")
    boundaries = tuple(bool(v) for v in (boundary_after if boundary_after is not None else [False] * infos.shape[0]))
    if len(boundaries) != infos.shape[0]:
        raise ValueError("boundary_after length must match information_matrices.")

    selected_dirs = directions[:, modes]
    buffer = np.zeros_like(precision)
    start_index = 0
    buffer_duration = 0.0
    elapsed = 0.0
    updates: list[SequentialInformationGateUpdate] = []

    for idx, raw_info in enumerate(infos):
        info = 0.5 * (_as_square_matrix(raw_info, name="information_matrix") + np.asarray(raw_info, dtype=float).T)
        buffer += info
        buffer = 0.5 * (buffer + buffer.T)
        buffer_duration += float(durations[idx])
        elapsed += float(durations[idx])
        count = idx - start_index + 1
        gains = precision_normalized_projected_gains(precision, buffer, selected_dirs)
        natural = bool(count >= min_count and np.all(gains >= threshold))
        maxed = bool(count >= max_count)
        boundary = bool(boundaries[idx])
        end = bool(idx == infos.shape[0] - 1)
        reason = ""
        if natural:
            reason = "natural_information_trigger"
        elif maxed:
            reason = "maximum_latency"
        elif boundary:
            reason = "historical_window_boundary"
        elif end:
            reason = "end_of_scope"
        if not reason:
            continue

        precision_before = precision.copy()
        covariance_before = _inverse_spd(precision_before, name="precision_before")
        precision_norms = np.einsum("ik,ij,jk->k", selected_dirs, precision_before, selected_dirs)
        absolute = np.einsum("ik,ij,jk->k", selected_dirs, buffer, selected_dirs)
        controlling_pos = int(np.argmin(gains)) if gains.size else 0
        precision = update_precision_with_information(precision_before, buffer)
        covariance_after = _inverse_spd(precision, name="precision_after")
        updates.append(
            SequentialInformationGateUpdate(
                update_index=len(updates),
                start_index=int(start_index),
                end_index=int(idx),
                block_length=int(count),
                block_duration_s=float(buffer_duration),
                cumulative_elapsed_s=float(elapsed),
                selected_mode_ids=modes,
                gains=tuple(float(v) for v in gains),
                precision_norms_before=tuple(float(v) for v in precision_norms),
                absolute_buffered_information=tuple(float(v) for v in absolute),
                controlling_mode_id=int(modes[controlling_pos]) if modes else -1,
                minimum_gain=float(np.min(gains)) if gains.size else np.nan,
                maximum_gain=float(np.max(gains)) if gains.size else np.nan,
                information_trace=float(np.trace(buffer)),
                precision_trace_before=float(np.trace(precision_before)),
                precision_trace_after=float(np.trace(precision)),
                covariance_trace_before=float(np.trace(covariance_before)),
                covariance_trace_after=float(np.trace(covariance_after)),
                closure_reason=reason,
                triggered_naturally=natural,
                maximum_latency_reached=maxed,
                historical_window_boundary_flush=bool(boundary and not natural and not maxed),
                end_of_scope_flush=bool(end and not natural and not maxed and not boundary),
                information_only_status="covariance_only_frozen_factor",
            )
        )
        buffer = np.zeros_like(precision)
        buffer_duration = 0.0
        start_index = idx + 1
    return tuple(updates)


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


def _warning_normalized_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _warning_normalized_value(value.tolist())
    if isinstance(value, np.generic):
        return _warning_normalized_value(value.item())
    if isinstance(value, Mapping):
        return {str(k): _warning_normalized_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_warning_normalized_value(item) for item in value]
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    return str(value) if value.__class__.__name__ == "Path" else value


def deduplicate_warnings(
    warnings: Sequence[Mapping[str, Any]],
    *,
    merge_contexts: bool = True,
) -> list[dict[str, Any]]:
    """Return warnings with exact semantic duplicates collapsed.

    The first occurrence order is preserved.  The deduplication key is the
    warning content after JSON-compatible normalization, excluding ``context``
    when ``merge_contexts`` is true.  Differing contexts are then preserved in a
    deterministic ``contexts`` list.
    """

    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    contexts: dict[int, list[Any]] = {}
    for raw in warnings:
        warning = dict(raw)
        key_payload = dict(warning)
        context = key_payload.pop("context", None) if merge_contexts else None
        key = json.dumps(_warning_normalized_value(key_payload), sort_keys=True, separators=(",", ":"))
        if key in seen:
            idx = seen[key]
            if merge_contexts and context not in (None, ""):
                contexts.setdefault(idx, [])
                if context not in contexts[idx]:
                    contexts[idx].append(context)
                    out[idx]["contexts"] = list(contexts[idx])
            continue
        seen[key] = len(out)
        normalized = _warning_normalized_value(warning)
        if not isinstance(normalized, dict):
            normalized = {"message": str(normalized)}
        out.append(dict(normalized))
        if merge_contexts:
            initial_contexts = []
            if context not in (None, ""):
                initial_contexts.append(context)
            if "context" in warning and "context" not in out[-1] and context not in (None, ""):
                out[-1]["context"] = context
            contexts[len(out) - 1] = initial_contexts
    return out
