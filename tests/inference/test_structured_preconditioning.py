from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dluxshera.inference.structured_preconditioning import (
    build_diagonal_preconditioner_from_curvature_diag,
    build_independent_frame_curvature_blocks,
)


def test_frame_only_curvature_blocks_match_local_hessians():
    local_curvatures = (
        jnp.asarray([[4.0, 0.5], [0.5, 2.0]]),
        jnp.asarray([[3.0, -0.25], [-0.25, 5.0]]),
    )

    def frame_loss(frame_values, shared_values, frame_index):
        del shared_values
        curvature = local_curvatures[frame_index]
        return 0.5 * frame_values @ curvature @ frame_values

    blocks = build_independent_frame_curvature_blocks(
        frame_loss_fn=frame_loss,
        frame_theta_ref=jnp.zeros((2, 2)),
        shared_theta_ref=jnp.asarray([]),
        subblock_reduce="sum",
    )

    assert blocks.kind == "frame_block"
    assert blocks.frame_only is True
    assert blocks.theta_size == 4
    assert len(blocks.blocks) == 2
    np.testing.assert_allclose(blocks.blocks[0].frame_block, local_curvatures[0])
    np.testing.assert_allclose(blocks.blocks[1].frame_block, local_curvatures[1])
    np.testing.assert_allclose(blocks.curvature_diag(), [4.0, 2.0, 3.0, 5.0])


def test_packed_lr_vector_assembly_from_frame_blocks():
    def frame_loss(frame_values, shared_values, frame_index):
        del shared_values
        curvature = jnp.diag(jnp.asarray([2.0 + frame_index, 8.0 + frame_index]))
        return 0.5 * frame_values @ curvature @ frame_values

    blocks = build_independent_frame_curvature_blocks(
        frame_loss_fn=frame_loss,
        frame_theta_ref=jnp.zeros((3, 2)),
        shared_theta_ref=jnp.asarray([]),
        subblock_reduce="sum",
    )
    precond = build_diagonal_preconditioner_from_curvature_diag(
        blocks.curvature_diag(),
        curvature_floor=0.0,
        eps=0.0,
        lr_clip=(0.2, 1.0),
    )

    np.testing.assert_allclose(
        precond["fim_diag"],
        [2.0, 8.0, 3.0, 9.0, 4.0, 10.0],
    )
    np.testing.assert_allclose(
        precond["lr_vec_unclipped"],
        [0.5, 0.125, 1.0 / 3.0, 1.0 / 9.0, 0.25, 0.1],
    )
    np.testing.assert_allclose(
        precond["lr_vec"],
        [0.5, 0.2, 1.0 / 3.0, 0.2, 0.25, 0.2],
    )


def test_frame_only_structured_blocks_match_dense_tiny_hessian():
    local_curvatures = (
        jnp.asarray([[4.0, 1.0], [1.0, 2.0]]),
        jnp.asarray([[3.0, 0.25], [0.25, 6.0]]),
    )

    def frame_loss(frame_values, shared_values, frame_index):
        del shared_values
        curvature = local_curvatures[frame_index]
        return 0.5 * frame_values @ curvature @ frame_values

    blocks = build_independent_frame_curvature_blocks(
        frame_loss_fn=frame_loss,
        frame_theta_ref=jnp.zeros((2, 2)),
        shared_theta_ref=jnp.asarray([]),
        subblock_reduce="mean",
    )

    def dense_loss(theta):
        frames = theta.reshape((2, 2))
        return 0.5 * sum(
            frame_loss(frames[index], jnp.asarray([]), index) for index in range(2)
        )

    dense_hessian = np.asarray(jax.hessian(dense_loss)(jnp.zeros(4)), dtype=float)
    np.testing.assert_allclose(blocks.assemble_dense(), dense_hessian)
    np.testing.assert_allclose(blocks.curvature_diag(), np.diag(dense_hessian))


def test_frame_shared_structured_blocks_match_arrowhead_dense_hessian():
    local_curvatures = (
        jnp.asarray(
            [
                [4.0, 0.5, -0.25],
                [0.5, 3.0, 0.1],
                [-0.25, 0.1, 2.0],
            ]
        ),
        jnp.asarray(
            [
                [6.0, -0.75, 0.2],
                [-0.75, 5.0, 0.3],
                [0.2, 0.3, 7.0],
            ]
        ),
    )

    def frame_loss(frame_values, shared_values, frame_index):
        local = jnp.concatenate((frame_values, shared_values), axis=0)
        curvature = local_curvatures[frame_index]
        return 0.5 * local @ curvature @ local

    blocks = build_independent_frame_curvature_blocks(
        frame_loss_fn=frame_loss,
        frame_theta_ref=jnp.zeros((2, 1)),
        shared_theta_ref=jnp.zeros(2),
        subblock_reduce="sum",
        kind="frame_shared_structured",
    )

    def dense_loss(theta):
        frame_values = theta[:2].reshape((2, 1))
        shared_values = theta[2:]
        return sum(
            frame_loss(frame_values[index], shared_values, index)
            for index in range(2)
        )

    dense_hessian = np.asarray(jax.hessian(dense_loss)(jnp.zeros(4)), dtype=float)

    assert blocks.kind == "frame_shared_structured"
    assert blocks.frame_only is False
    assert blocks.shared_dim == 2
    assert blocks.blocks[0].coupling_block.shape == (1, 2)
    assert np.any(np.abs(blocks.blocks[0].coupling_block) > 0.0)
    np.testing.assert_allclose(blocks.assemble_dense(), dense_hessian)
    np.testing.assert_allclose(blocks.curvature_diag(), np.diag(dense_hessian))
