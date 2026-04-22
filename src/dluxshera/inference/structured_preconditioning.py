from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np


SubblockReduce = Literal["sum", "mean"]


@dataclass(frozen=True)
class FrameSharedCurvatureBlock:
    """One frame's local contribution to a structured sub-block curvature.

    The local coordinate order is always ``[frame_values, shared_values]``.
    Decomposing each local Hessian this way gives the global independent-frame
    arrowhead structure:

    - frame blocks live only on their own frame diagonal
    - shared blocks accumulate into one global shared block
    - coupling blocks connect one frame block to the shared block
    """

    frame_index: int
    frame_dim: int
    shared_dim: int
    local_fim: np.ndarray
    frame_block: np.ndarray
    coupling_block: np.ndarray
    shared_block: np.ndarray


@dataclass(frozen=True)
class StructuredCurvatureBlocks:
    """Block representation of independent-frame sub-block curvature."""

    kind: str
    frame_dim: int
    shared_dim: int
    n_frame: int
    subblock_reduce: SubblockReduce
    reduce_weight: float
    blocks: tuple[FrameSharedCurvatureBlock, ...]

    @property
    def theta_size(self) -> int:
        return int(self.n_frame * self.frame_dim + self.shared_dim)

    @property
    def frame_only(self) -> bool:
        return self.shared_dim == 0

    def curvature_diag(self) -> np.ndarray:
        """Return the packed diagonal of the represented global curvature."""

        frame_diag = np.empty((self.n_frame, self.frame_dim), dtype=float)
        shared_diag = np.zeros((self.shared_dim,), dtype=float)
        for block in self.blocks:
            if self.frame_dim:
                frame_diag[block.frame_index, :] = np.diag(block.frame_block)
            if self.shared_dim:
                shared_diag += np.diag(block.shared_block)

        packed = frame_diag.reshape((self.n_frame * self.frame_dim,))
        if self.shared_dim:
            packed = np.concatenate((packed, shared_diag), axis=0)
        if packed.shape != (self.theta_size,):
            raise ValueError("Structured curvature diagonal does not match theta size.")
        return packed

    def trace(self) -> float:
        return float(np.sum(self.curvature_diag()))

    def assemble_dense(self) -> np.ndarray:
        """Materialize the represented global matrix for tests/debugging only."""

        dense = np.zeros((self.theta_size, self.theta_size), dtype=float)
        frame_size = self.n_frame * self.frame_dim
        shared_slice = slice(frame_size, frame_size + self.shared_dim)

        for block in self.blocks:
            frame_start = block.frame_index * self.frame_dim
            frame_stop = frame_start + self.frame_dim
            frame_slice = slice(frame_start, frame_stop)

            if self.frame_dim:
                dense[frame_slice, frame_slice] = block.frame_block
            if self.shared_dim:
                dense[shared_slice, shared_slice] += block.shared_block
            if self.frame_dim and self.shared_dim:
                dense[frame_slice, shared_slice] = block.coupling_block
                dense[shared_slice, frame_slice] = block.coupling_block.T

        return 0.5 * (dense + dense.T)

    def to_debug_payload(self, *, include_blocks: bool = True) -> dict[str, object]:
        """Return JSON-friendly metadata for structured curvature diagnostics."""

        payload: dict[str, object] = {
            "kind": self.kind,
            "frame_dim": int(self.frame_dim),
            "shared_dim": int(self.shared_dim),
            "n_frame": int(self.n_frame),
            "theta_size": int(self.theta_size),
            "subblock_reduce": self.subblock_reduce,
            "reduce_weight": float(self.reduce_weight),
            "frame_only": bool(self.frame_only),
            "curvature_trace": float(self.trace()),
        }
        if include_blocks:
            payload["blocks"] = [
                {
                    "frame_index": int(block.frame_index),
                    "local_fim": np.asarray(block.local_fim, dtype=float).tolist(),
                    "frame_block": np.asarray(block.frame_block, dtype=float).tolist(),
                    "coupling_block": np.asarray(block.coupling_block, dtype=float).tolist(),
                    "shared_block": np.asarray(block.shared_block, dtype=float).tolist(),
                }
                for block in self.blocks
            ]
        return payload


def subblock_reduce_weight(n_frame: int, reduce: SubblockReduce) -> float:
    """Return the scalar by which each frame term enters the block objective."""

    n_frame = int(n_frame)
    if n_frame <= 0:
        raise ValueError("n_frame must be positive.")
    if reduce == "sum":
        return 1.0
    if reduce == "mean":
        return 1.0 / float(n_frame)
    raise ValueError("subblock_reduce must be 'sum' or 'mean'.")


def _as_symmetric_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Curvature block must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Curvature block contains non-finite values.")
    return 0.5 * (matrix + matrix.T)


def build_independent_frame_curvature_blocks(
    *,
    frame_loss_fn: Callable[[jnp.ndarray, jnp.ndarray, int], jnp.ndarray],
    frame_theta_ref: jnp.ndarray,
    shared_theta_ref: jnp.ndarray,
    subblock_reduce: SubblockReduce = "sum",
    kind: str | None = None,
) -> StructuredCurvatureBlocks:
    """Build independent-frame curvature blocks without a dense global Hessian.

    ``frame_loss_fn(frame_values, shared_values, frame_index)`` must return the
    scalar data term for one frame before sub-block reduction. This helper
    applies the ``sum``/``mean`` sub-block weight to each local Hessian so the
    represented blocks match the curvature of the global data term.
    """

    frame_theta = jnp.asarray(frame_theta_ref)
    shared_theta = jnp.asarray(shared_theta_ref)
    if frame_theta.ndim != 2:
        raise ValueError("frame_theta_ref must have shape (n_frame, frame_dim).")
    if shared_theta.ndim != 1:
        raise ValueError("shared_theta_ref must be a 1D vector.")

    n_frame = int(frame_theta.shape[0])
    frame_dim = int(frame_theta.shape[1])
    shared_dim = int(shared_theta.size)
    weight = subblock_reduce_weight(n_frame, subblock_reduce)
    blocks: list[FrameSharedCurvatureBlock] = []

    for frame_index in range(n_frame):
        frame_ref = frame_theta[frame_index]

        if shared_dim:
            local_ref = jnp.concatenate((frame_ref, shared_theta), axis=0)

            def _local_loss(local_values: jnp.ndarray) -> jnp.ndarray:
                frame_values = local_values[:frame_dim]
                shared_values = local_values[frame_dim:]
                return frame_loss_fn(frame_values, shared_values, frame_index)

            local_fim = _as_symmetric_matrix(
                np.asarray(jax.hessian(_local_loss)(local_ref), dtype=float) * weight
            )
        else:
            local_fim = _as_symmetric_matrix(
                np.asarray(
                    jax.hessian(
                        lambda frame_values: frame_loss_fn(
                            frame_values,
                            shared_theta,
                            frame_index,
                        )
                    )(frame_ref),
                    dtype=float,
                )
                * weight
            )

        expected_dim = frame_dim + shared_dim
        if local_fim.shape != (expected_dim, expected_dim):
            raise ValueError(
                "Local curvature block shape does not match frame/shared dimensions."
            )

        frame_block = local_fim[:frame_dim, :frame_dim]
        coupling_block = local_fim[:frame_dim, frame_dim:]
        shared_block = local_fim[frame_dim:, frame_dim:]
        blocks.append(
            FrameSharedCurvatureBlock(
                frame_index=frame_index,
                frame_dim=frame_dim,
                shared_dim=shared_dim,
                local_fim=local_fim,
                frame_block=frame_block,
                coupling_block=coupling_block,
                shared_block=shared_block,
            )
        )

    if kind is None:
        kind = "frame_block" if shared_dim == 0 else "frame_shared_structured"

    return StructuredCurvatureBlocks(
        kind=str(kind),
        frame_dim=frame_dim,
        shared_dim=shared_dim,
        n_frame=n_frame,
        subblock_reduce=subblock_reduce,
        reduce_weight=weight,
        blocks=tuple(blocks),
    )


def build_diagonal_preconditioner_from_curvature_diag(
    curvature_diag: np.ndarray,
    *,
    curvature_floor: float = 1e-8,
    eps: float = 1e-12,
    lr_clip: Optional[tuple[float, float]] = None,
) -> dict[str, np.ndarray | dict[str, object]]:
    """Build a packed diagonal preconditioner from structured curvature diag."""

    fim_diag = np.asarray(curvature_diag, dtype=float)
    if fim_diag.ndim != 1:
        raise ValueError("curvature_diag must be a 1D vector.")
    if not np.all(np.isfinite(fim_diag)):
        raise ValueError("curvature_diag contains non-finite values.")

    curvature_floor = float(curvature_floor)
    eps = float(eps)
    if curvature_floor < 0.0:
        raise ValueError("curvature_floor must be non-negative.")
    if eps < 0.0:
        raise ValueError("eps must be non-negative.")

    curvature_floored_count = int(np.count_nonzero(fim_diag < curvature_floor))
    curvature_vec = np.maximum(fim_diag, curvature_floor)
    lr_vec_unclipped = np.reciprocal(curvature_vec + eps)
    lr_vec = np.array(lr_vec_unclipped, copy=True)
    lr_clip_applied_count = 0
    if lr_clip is not None:
        lr_min, lr_max = lr_clip
        lr_min = float(lr_min)
        lr_max = float(lr_max)
        if lr_min <= 0.0:
            raise ValueError("lr_clip lower bound must be positive.")
        if lr_max < lr_min:
            raise ValueError("lr_clip upper bound must be >= lower bound.")
        lr_clip_applied_count = int(
            np.count_nonzero((lr_vec < lr_min) | (lr_vec > lr_max))
        )
        lr_vec = np.clip(lr_vec, lr_min, lr_max)

    for name, arr in {
        "fim_diag": fim_diag,
        "curvature_vec": curvature_vec,
        "lr_vec_unclipped": lr_vec_unclipped,
        "lr_vec": lr_vec,
    }.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite values encountered in {name}.")
    if np.any(lr_vec <= 0.0):
        raise ValueError("Preconditioning vector must be strictly positive.")

    return {
        "fim_diag": fim_diag,
        "curvature_vec": curvature_vec,
        "lr_vec_unclipped": lr_vec_unclipped,
        "lr_vec": lr_vec,
        "config": {
            "curvature_floor": curvature_floor,
            "curvature_floored_count": curvature_floored_count,
            "eps": eps,
            "lr_clip": None if lr_clip is None else [float(lr_clip[0]), float(lr_clip[1])],
            "lr_clip_applied_count": lr_clip_applied_count,
        },
    }
