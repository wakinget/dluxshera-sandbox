"""Assemble structured curvature for independent-frame Schur summaries.

This module contains the reusable math used by
``examples/scripts/run_obs_subblock_study.py`` when exporting image-backed
Schur summaries without differentiating a dense packed ``[Theta, phi]`` vector.
The current scope is deliberately narrow: independent frame objectives with
frame-local nuisance variables and no shared active sub-block state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dluxshera.inference.observation_belief import MatrixDiagnostics
from dluxshera.inference.structured_preconditioning import (
    SubblockReduce,
    subblock_reduce_weight,
)

__all__ = [
    "FrameLocalQuadraticBlocks",
    "IndependentFrameThetaPhiQuadraticBlocks",
    "StructuredSchurReductionResult",
    "build_independent_frame_theta_phi_quadratic_blocks",
    "compare_structured_and_dense_schur_outputs",
    "build_residual_prior_temporal_curvature",
    "materialize_structured_schur_sidecar_blocks",
    "schur_reduce_independent_frame_blocks",
]


@dataclass(frozen=True)
class FrameLocalQuadraticBlocks:
    """Store one weighted frame-local quadratic over ``[Theta, phi_i]``.

    These blocks are produced by
    :func:`build_independent_frame_theta_phi_quadratic_blocks` and consumed by
    :func:`schur_reduce_independent_frame_blocks`. They represent one frame's
    contribution to an independent-frame sub-block objective after applying the
    ``sum`` or ``mean`` sub-block reduction weight.

    Parameters
    ----------
    frame_index :
        Zero-based frame index.
    theta_dim :
        Number of observation-level ``Theta`` variables.
    frame_phi_dim :
        Number of frame-local fast variables for this frame.
    local_gradient :
        Weighted gradient in local ``[Theta, phi_i]`` order.
    local_hessian :
        Weighted Hessian in local ``[Theta, phi_i]`` order.
    g_theta, g_phi :
        Partitioned local gradient blocks.
    h_tt, h_tphi, h_phiphi :
        Partitioned local curvature blocks.

    Notes
    -----
    This class is specific to independent-frame Schur summary export. It is not
    the optimizer preconditioning state, although both paths use the same
    sub-block reduction convention.
    """

    frame_index: int
    theta_dim: int
    frame_phi_dim: int
    local_gradient: np.ndarray
    local_hessian: np.ndarray
    g_theta: np.ndarray
    g_phi: np.ndarray
    h_tt: np.ndarray
    h_tphi: np.ndarray
    h_phiphi: np.ndarray


@dataclass(frozen=True)
class IndependentFrameThetaPhiQuadraticBlocks:
    """Represent structured independent-frame curvature for Schur export.

    The represented global objective is a sum of frame terms, each defined over
    the shared observation-level vector ``Theta`` and that frame's local
    nuisance vector ``phi_i``. This avoids differentiating a dense packed
    ``[Theta, phi_0, ..., phi_n]`` vector while retaining the full local blocks
    needed for Schur reduction.

    Parameters
    ----------
    kind :
        Curvature representation identifier.
    theta_dim :
        Number of observation-level variables.
    frame_phi_dim :
        Number of fast variables per frame.
    n_frame :
        Number of independent frame terms.
    subblock_reduce :
        Either ``"sum"`` or ``"mean"``.
    reduce_weight :
        Scalar weight applied to each frame term.
    theta_ref :
        Observation-level reference vector.
    frame_phi_ref :
        Reference frame-local nuisance matrix with shape
        ``(n_frame, frame_phi_dim)``.
    blocks :
        Weighted per-frame local quadratic blocks.

    Notes
    -----
    The current production caller is
    ``examples/scripts/run_obs_subblock_study.py`` in ``schur_summary`` mode.
    It uses this class for registration-only independent-frame summaries with
    no shared active sub-block state.
    """

    kind: str
    theta_dim: int
    frame_phi_dim: int
    n_frame: int
    subblock_reduce: SubblockReduce
    reduce_weight: float
    theta_ref: np.ndarray
    frame_phi_ref: np.ndarray
    blocks: tuple[FrameLocalQuadraticBlocks, ...]

    @property
    def phi_dim(self) -> int:
        """Return the total packed fast-state dimension."""

        return int(self.n_frame * self.frame_phi_dim)

    @property
    def combined_dim(self) -> int:
        """Return the transitional sidecar dimension ``theta_dim + phi_dim``."""

        return int(self.theta_dim + self.phi_dim)

    def to_debug_payload(self, *, include_blocks: bool = False) -> dict[str, Any]:
        """Return JSON-friendly metadata for structured Schur diagnostics."""

        payload: dict[str, Any] = {
            "kind": self.kind,
            "theta_dim": int(self.theta_dim),
            "frame_phi_dim": int(self.frame_phi_dim),
            "n_frame": int(self.n_frame),
            "phi_dim": int(self.phi_dim),
            "combined_dim": int(self.combined_dim),
            "subblock_reduce": self.subblock_reduce,
            "reduce_weight": float(self.reduce_weight),
        }
        if include_blocks:
            payload["blocks"] = [
                {
                    "frame_index": int(block.frame_index),
                    "local_gradient": block.local_gradient.tolist(),
                    "local_hessian": block.local_hessian.tolist(),
                    "h_phiphi_diagnostics": _matrix_diagnostics(
                        block.h_phiphi
                    ).to_dict(),
                }
                for block in self.blocks
            ]
        return payload


@dataclass(frozen=True)
class StructuredSchurReductionResult:
    """Store a framewise Schur reduction for independent-frame blocks.

    The result is mathematically equivalent to reducing the block-diagonal
    packed nuisance Hessian, but it solves each frame's ``H_phiphi_i`` block
    independently. This avoids dense autodiff over the global packed vector and
    avoids using a global dense ``H_pp`` solve for the reduced summary.

    Parameters
    ----------
    reduced_information :
        Schur-reduced ``Theta`` information matrix.
    reduced_score :
        Schur-reduced objective gradient at ``theta_ref``.
    damping :
        Non-negative diagonal damping added to each frame-local ``H_phiphi_i``
        before solves.
    solve_method :
        ``"solve"`` when all frame blocks used direct solves, ``"pinv"`` if any
        frame required a pseudo-inverse.
    used_pseudoinverse :
        Whether any frame-local solve used a pseudo-inverse.
    reduced_diagnostics :
        Matrix diagnostics for ``reduced_information``.
    per_frame_diagnostics :
        Diagnostics for the frame-local Schur terms and nuisance blocks.
    """

    reduced_information: np.ndarray
    reduced_score: np.ndarray
    damping: float
    solve_method: str
    used_pseudoinverse: bool
    reduced_diagnostics: MatrixDiagnostics
    per_frame_diagnostics: tuple[dict[str, Any], ...]

    def to_diagnostics_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic snapshot."""

        return {
            "h_pp_solve_method": self.solve_method,
            "used_pseudoinverse": bool(self.used_pseudoinverse),
            "damping": float(self.damping),
            "reduced_information": self.reduced_diagnostics.to_dict(),
            "per_frame": [dict(item) for item in self.per_frame_diagnostics],
            "structured_curvature_used": True,
        }


def _as_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values.")
    return vector


def _as_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _as_square_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = _as_matrix(values, name=name)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return 0.5 * (matrix + matrix.T)


def _ols_projection_matrix(centered_time: np.ndarray) -> np.ndarray:
    """Return projection onto span([1, centered_time])."""

    t = np.asarray(centered_time, dtype=float)
    if t.ndim != 1 or t.size <= 0:
        raise ValueError("centered_time must be a non-empty 1D vector.")
    design = np.stack((np.ones_like(t), t), axis=1)
    gram = design.T @ design
    gram_inv = np.linalg.pinv(gram)
    proj = design @ gram_inv @ design.T
    return 0.5 * (proj + proj.T)


def build_residual_prior_temporal_curvature(
    *,
    frame_times_s: np.ndarray,
    frame_keys: Sequence[str],
    residual_sigmas_by_key: dict[str, float],
    reduce: str,
    subblock_reduce: str,
) -> np.ndarray:
    """Build expanded fast-state temporal-prior curvature for residual-prior model.

    The penalty per key is ``0.5 / sigma^2 * ||(I - P) phi||^2`` with ``P`` as
    the OLS projection onto intercept+slope in centered time.
    """

    times = np.asarray(frame_times_s, dtype=float)
    if times.ndim != 1 or times.size <= 0:
        raise ValueError("frame_times_s must be a non-empty 1D vector.")
    keys = tuple(str(key) for key in frame_keys)
    if not keys:
        raise ValueError("frame_keys must be non-empty.")
    for key in keys:
        if key not in residual_sigmas_by_key:
            raise ValueError(f"Missing residual-prior sigma for frame key {key!r}.")
        sigma = float(residual_sigmas_by_key[key])
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError(f"Residual-prior sigma for {key!r} must be positive and finite.")
    resolved_reduce = str(reduce)
    if resolved_reduce == "match_subblock_reduce":
        resolved_reduce = str(subblock_reduce)
    if resolved_reduce not in {"sum", "mean"}:
        raise ValueError("reduce must be one of: sum, mean, match_subblock_reduce.")

    n_frame = int(times.size)
    frame_width = int(len(keys))
    centered = times - float(np.mean(times))
    proj = _ols_projection_matrix(centered)
    residual_op = np.eye(n_frame, dtype=float) - proj
    residual_op = 0.5 * (residual_op + residual_op.T)
    scale = 1.0 if resolved_reduce == "sum" else 1.0 / float(n_frame)
    h_pp = np.zeros((n_frame * frame_width, n_frame * frame_width), dtype=float)
    for key_index, key in enumerate(keys):
        sigma = float(residual_sigmas_by_key[key])
        block = scale * residual_op / (sigma * sigma)
        index = np.arange(n_frame, dtype=int) * frame_width + int(key_index)
        h_pp[np.ix_(index, index)] = block
    return 0.5 * (h_pp + h_pp.T)


def _matrix_diagnostics(matrix: np.ndarray) -> MatrixDiagnostics:
    matrix = _as_square_matrix(matrix, name="matrix")
    if matrix.size == 0:
        return MatrixDiagnostics(
            rank_estimate=0,
            min_eigenvalue=0.0,
            max_eigenvalue=0.0,
            condition_number=1.0,
            trace=0.0,
            frobenius_norm=0.0,
        )
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = (
        np.finfo(float).eps
        * max(matrix.shape)
        * max(float(np.max(np.abs(eigenvalues))), 1.0)
    )
    positive = eigenvalues[eigenvalues > tolerance]
    condition_number = (
        float("inf")
        if positive.size == 0
        else float(np.max(positive) / np.min(positive))
    )
    return MatrixDiagnostics(
        rank_estimate=int(np.count_nonzero(np.abs(eigenvalues) > tolerance)),
        min_eigenvalue=float(np.min(eigenvalues)),
        max_eigenvalue=float(np.max(eigenvalues)),
        condition_number=condition_number,
        trace=float(np.trace(matrix)),
        frobenius_norm=float(np.linalg.norm(matrix)),
    )


def build_independent_frame_theta_phi_quadratic_blocks(
    *,
    frame_loss_fn: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    theta_ref: Sequence[float] | np.ndarray,
    frame_phi_ref: Sequence[Sequence[float]] | np.ndarray,
    subblock_reduce: SubblockReduce = "sum",
    kind: str = "structured_independent_frames",
) -> IndependentFrameThetaPhiQuadraticBlocks:
    """Build per-frame ``[Theta, phi_i]`` quadratic blocks.

    ``frame_loss_fn(theta_values, frame_phi_values, frame_index)`` must return
    the unreduced scalar data/objective term for one frame. This helper applies
    the requested sub-block reduction weight to both gradients and Hessians so
    the represented blocks match a global ``sum`` or ``mean`` objective.

    Parameters
    ----------
    frame_loss_fn :
        Callable for one independent frame term before sub-block reduction.
    theta_ref :
        Observation-level reference vector with shape ``(theta_dim,)``.
    frame_phi_ref :
        Frame-local reference matrix with shape ``(n_frame, frame_phi_dim)``.
    subblock_reduce :
        ``"sum"`` uses weight ``1``; ``"mean"`` uses weight ``1 / n_frame``.
    kind :
        Identifier stored in diagnostics.

    Returns
    -------
    IndependentFrameThetaPhiQuadraticBlocks
        Weighted per-frame gradient and curvature blocks.

    Raises
    ------
    ValueError
        Raised for invalid shapes, non-finite references, or unsupported
        ``subblock_reduce`` values.

    Notes
    -----
    This helper materializes only frame-local Hessians of shape
    ``(theta_dim + frame_phi_dim, theta_dim + frame_phi_dim)``. It does not
    materialize the full packed global Hessian.
    """

    theta = _as_vector(np.asarray(theta_ref, dtype=float), name="theta_ref")
    frame_phi = _as_matrix(np.asarray(frame_phi_ref, dtype=float), name="frame_phi_ref")
    n_frame = int(frame_phi.shape[0])
    frame_phi_dim = int(frame_phi.shape[1])
    if n_frame <= 0:
        raise ValueError("frame_phi_ref must include at least one frame.")
    if frame_phi_dim <= 0:
        raise ValueError("frame_phi_ref must include at least one fast variable.")

    theta_jax = jnp.asarray(theta)
    frame_phi_jax = jnp.asarray(frame_phi)
    theta_dim = int(theta.size)
    weight = subblock_reduce_weight(n_frame, subblock_reduce)
    blocks: list[FrameLocalQuadraticBlocks] = []

    for frame_index in range(n_frame):
        phi_ref = frame_phi_jax[frame_index]
        local_ref = jnp.concatenate((theta_jax, phi_ref), axis=0)

        def _local_loss(local_values: jnp.ndarray) -> jnp.ndarray:
            theta_values = local_values[:theta_dim]
            phi_values = local_values[theta_dim:]
            return frame_loss_fn(theta_values, phi_values, frame_index)

        local_gradient = _as_vector(
            np.asarray(jax.grad(_local_loss)(local_ref), dtype=float) * weight,
            name="local_gradient",
        )
        local_hessian = (
            _as_square_matrix(
                np.asarray(jax.hessian(_local_loss)(local_ref), dtype=float),
                name="local_hessian",
            )
            * weight
        )
        expected_dim = theta_dim + frame_phi_dim
        if local_gradient.shape != (expected_dim,):
            raise ValueError("Local gradient shape does not match theta/phi dimensions.")
        if local_hessian.shape != (expected_dim, expected_dim):
            raise ValueError("Local Hessian shape does not match theta/phi dimensions.")

        blocks.append(
            FrameLocalQuadraticBlocks(
                frame_index=frame_index,
                theta_dim=theta_dim,
                frame_phi_dim=frame_phi_dim,
                local_gradient=local_gradient,
                local_hessian=local_hessian,
                g_theta=local_gradient[:theta_dim],
                g_phi=local_gradient[theta_dim:],
                h_tt=local_hessian[:theta_dim, :theta_dim],
                h_tphi=local_hessian[:theta_dim, theta_dim:],
                h_phiphi=local_hessian[theta_dim:, theta_dim:],
            )
        )

    return IndependentFrameThetaPhiQuadraticBlocks(
        kind=str(kind),
        theta_dim=theta_dim,
        frame_phi_dim=frame_phi_dim,
        n_frame=n_frame,
        subblock_reduce=subblock_reduce,
        reduce_weight=float(weight),
        theta_ref=theta,
        frame_phi_ref=frame_phi,
        blocks=tuple(blocks),
    )


def _solve_frame_system(
    h_phiphi: np.ndarray,
    rhs: np.ndarray,
    *,
    damping: float,
    rcond: float | None,
) -> tuple[np.ndarray, str, bool]:
    matrix = _as_square_matrix(h_phiphi, name="h_phiphi")
    if damping < 0.0:
        raise ValueError("damping must be non-negative.")
    if damping > 0.0:
        matrix = matrix + float(damping) * np.eye(matrix.shape[0], dtype=float)
    rhs_array = np.asarray(rhs, dtype=float)
    try:
        solution = np.linalg.solve(matrix, rhs_array)
        return solution, "solve", False
    except np.linalg.LinAlgError:
        pinv = np.linalg.pinv(
            matrix,
            rcond=np.finfo(float).eps if rcond is None else float(rcond),
            hermitian=True,
        )
        return pinv @ rhs_array, "pinv", True


def schur_reduce_independent_frame_blocks(
    blocks: IndependentFrameThetaPhiQuadraticBlocks,
    *,
    damping: float = 0.0,
    rcond: float | None = None,
    frame_indices: Sequence[int] | None = None,
    frame_scale: float = 1.0,
) -> StructuredSchurReductionResult:
    """Schur-reduce independent frame-local nuisance blocks.

    This computes
    ``sum_i H_tt_i - H_tphi_i solve(H_phiphi_i, H_phitheta_i)`` and
    ``sum_i g_theta_i - H_tphi_i solve(H_phiphi_i, g_phi_i)`` using one small
    solve per frame. ``H_phiphi_i`` receives the same diagonal ``damping`` that
    the dense Schur path applies to the packed nuisance block.

    Parameters
    ----------
    blocks :
        Weighted independent-frame local quadratic blocks.
    damping :
        Non-negative diagonal damping added to each ``H_phiphi_i`` solve.
    rcond :
        Optional pseudo-inverse tolerance.

    Returns
    -------
    StructuredSchurReductionResult
        Reduced information, reduced score, and diagnostics.

    Raises
    ------
    ValueError
        Raised when damping is negative or block shapes are inconsistent.
    """

    if float(damping) < 0.0:
        raise ValueError("damping must be non-negative.")
    if not np.isfinite(float(frame_scale)) or float(frame_scale) <= 0.0:
        raise ValueError("frame_scale must be a positive finite float.")

    selected_indices: set[int] | None = None
    if frame_indices is not None:
        selected_indices = {int(index) for index in frame_indices}
        if not selected_indices:
            raise ValueError("frame_indices must include at least one frame.")
        valid_indices = {int(block.frame_index) for block in blocks.blocks}
        missing = sorted(selected_indices - valid_indices)
        if missing:
            raise ValueError(f"frame_indices contains unavailable frames: {missing}")

    reduced_information = np.zeros(
        (blocks.theta_dim, blocks.theta_dim),
        dtype=float,
    )
    reduced_score = np.zeros((blocks.theta_dim,), dtype=float)
    per_frame: list[dict[str, Any]] = []
    used_pinv = False
    solve_methods: list[str] = []

    included_count = 0
    for block in blocks.blocks:
        if selected_indices is not None and int(block.frame_index) not in selected_indices:
            continue
        included_count += 1
        solved_hpt, solve_method_a, used_pinv_a = _solve_frame_system(
            block.h_phiphi,
            block.h_tphi.T,
            damping=float(damping),
            rcond=rcond,
        )
        solved_g, solve_method_b, used_pinv_b = _solve_frame_system(
            block.h_phiphi,
            block.g_phi,
            damping=float(damping),
            rcond=rcond,
        )
        frame_reduced_info = (block.h_tt - block.h_tphi @ solved_hpt) * float(
            frame_scale
        )
        frame_reduced_score = (block.g_theta - block.h_tphi @ solved_g) * float(
            frame_scale
        )
        reduced_information += frame_reduced_info
        reduced_score += frame_reduced_score
        used_frame_pinv = bool(used_pinv_a or used_pinv_b)
        used_pinv = bool(used_pinv or used_frame_pinv)
        solve_methods.extend((solve_method_a, solve_method_b))
        hpp_solve = block.h_phiphi
        if damping > 0.0:
            hpp_solve = hpp_solve + float(damping) * np.eye(
                block.frame_phi_dim,
                dtype=float,
            )
        per_frame.append(
            {
                "frame_index": int(block.frame_index),
                "solve_method": "pinv" if used_frame_pinv else "solve",
                "used_pseudoinverse": used_frame_pinv,
                "h_phiphi": _matrix_diagnostics(hpp_solve).to_dict(),
                "frame_reduced_information": _matrix_diagnostics(
                    frame_reduced_info
                ).to_dict(),
                "frame_reduced_score_norm": float(np.linalg.norm(frame_reduced_score)),
            }
        )

    reduced_information = 0.5 * (reduced_information + reduced_information.T)
    if included_count == 0:
        raise ValueError("No frame blocks were included in the Schur reduction.")
    solve_method = "pinv" if "pinv" in solve_methods else "solve"
    return StructuredSchurReductionResult(
        reduced_information=reduced_information,
        reduced_score=reduced_score,
        damping=float(damping),
        solve_method=solve_method,
        used_pseudoinverse=used_pinv,
        reduced_diagnostics=_matrix_diagnostics(reduced_information),
        per_frame_diagnostics=tuple(per_frame),
    )


def materialize_structured_schur_sidecar_blocks(
    blocks: IndependentFrameThetaPhiQuadraticBlocks,
    *,
    frame_indices: Sequence[int] | None = None,
    frame_scale: float = 1.0,
) -> dict[str, np.ndarray]:
    """Materialize dense sidecar blocks for loader-compatible artifacts.

    Use this transitional helper when writing the existing
    ``subblock_summary_matrices.npz`` contract. It materializes ``H_tt``,
    ``H_tp``, block-diagonal ``H_pp``, ``g_theta``, and packed ``g_phi`` from
    structured per-frame blocks. It does not materialize or differentiate the
    full packed global Hessian.

    Parameters
    ----------
    blocks :
        Structured independent-frame quadratic blocks.

    Returns
    -------
    dict
        Dense sidecar arrays keyed by ``h_tt``, ``h_tp``, ``h_pp``,
        ``g_theta``, and ``g_phi``.
    """

    h_tt = np.zeros((blocks.theta_dim, blocks.theta_dim), dtype=float)
    h_tp = np.zeros((blocks.theta_dim, blocks.phi_dim), dtype=float)
    h_pp = np.zeros((blocks.phi_dim, blocks.phi_dim), dtype=float)
    g_theta = np.zeros((blocks.theta_dim,), dtype=float)
    g_phi = np.zeros((blocks.phi_dim,), dtype=float)

    if not np.isfinite(float(frame_scale)) or float(frame_scale) <= 0.0:
        raise ValueError("frame_scale must be a positive finite float.")
    selected_indices: set[int] | None = None
    if frame_indices is not None:
        selected_indices = {int(index) for index in frame_indices}
        if not selected_indices:
            raise ValueError("frame_indices must include at least one frame.")

    for block in blocks.blocks:
        if selected_indices is not None and int(block.frame_index) not in selected_indices:
            continue
        start = block.frame_index * blocks.frame_phi_dim
        stop = start + blocks.frame_phi_dim
        h_tt += block.h_tt * float(frame_scale)
        h_tp[:, start:stop] = block.h_tphi * float(frame_scale)
        h_pp[start:stop, start:stop] = block.h_phiphi * float(frame_scale)
        g_theta += block.g_theta * float(frame_scale)
        g_phi[start:stop] = block.g_phi * float(frame_scale)

    return {
        "h_tt": 0.5 * (h_tt + h_tt.T),
        "h_tp": h_tp,
        "h_pp": 0.5 * (h_pp + h_pp.T),
        "g_theta": g_theta,
        "g_phi": g_phi,
    }


def compare_structured_and_dense_schur_outputs(
    *,
    structured_information: np.ndarray,
    structured_score: np.ndarray,
    dense_information: np.ndarray,
    dense_score: np.ndarray,
) -> dict[str, float]:
    """Return compact dense-vs-structured Schur comparison metrics.

    Use this in small validation runs where both curvature paths are computed.
    The helper reports absolute and relative differences for the reduced
    information matrix and reduced score vector.

    Parameters
    ----------
    structured_information, dense_information :
        Reduced ``Theta`` information matrices to compare.
    structured_score, dense_score :
        Reduced score vectors to compare.

    Returns
    -------
    dict
        JSON-friendly absolute and relative difference metrics for the matrix
        and score outputs.

    Raises
    ------
    ValueError
        Raised when matrix or vector inputs have invalid shapes or contain
        non-finite values.

    Notes
    -----
    This helper does not run either curvature path. It is a reporting utility
    for callers that intentionally computed both outputs, usually with a small
    dense validation case below the dense-dimension guard.
    """

    structured_information = _as_square_matrix(
        structured_information,
        name="structured_information",
    )
    dense_information = _as_square_matrix(dense_information, name="dense_information")
    structured_score = _as_vector(structured_score, name="structured_score")
    dense_score = _as_vector(dense_score, name="dense_score")
    info_delta = structured_information - dense_information
    score_delta = structured_score - dense_score
    return {
        "reduced_information_max_abs_delta": float(np.max(np.abs(info_delta))),
        "reduced_information_frobenius_delta": float(np.linalg.norm(info_delta)),
        "reduced_information_relative_frobenius_delta": float(
            np.linalg.norm(info_delta) / max(np.linalg.norm(dense_information), 1.0e-30)
        ),
        "reduced_score_max_abs_delta": float(np.max(np.abs(score_delta))),
        "reduced_score_norm_delta": float(np.linalg.norm(score_delta)),
        "reduced_score_relative_norm_delta": float(
            np.linalg.norm(score_delta) / max(np.linalg.norm(dense_score), 1.0e-30)
        ),
    }
