from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dluxshera.inference.observation_summary import (
    build_combined_local_parameter_layout,
    partition_local_curvature,
    schur_reduce_local_quadratic,
)
from dluxshera.inference.structured_curvature import (
    build_independent_frame_theta_phi_quadratic_blocks,
    materialize_structured_schur_sidecar_blocks,
    schur_reduce_independent_frame_blocks,
)


def _frame_quadratic_case(*, n_frames: int, reduce: str):
    theta_dim = 2
    frame_phi_dim = 2
    theta_ref = np.array([0.3, -0.2], dtype=float)
    frame_phi_ref = np.arange(n_frames * frame_phi_dim, dtype=float).reshape(
        n_frames,
        frame_phi_dim,
    ) * 0.1
    local_hessians = []
    local_gradients = []
    for frame_index in range(n_frames):
        h_tt = np.array(
            [
                [4.0 + frame_index, 0.2],
                [0.2, 3.0 + 0.5 * frame_index],
            ],
            dtype=float,
        )
        h_tphi = np.array(
            [
                [0.5 + 0.1 * frame_index, -0.15],
                [0.25, -0.3 - 0.05 * frame_index],
            ],
            dtype=float,
        )
        h_pp = np.array(
            [
                [2.5 + frame_index, 0.1],
                [0.1, 2.0 + 0.25 * frame_index],
            ],
            dtype=float,
        )
        hessian = np.block([[h_tt, h_tphi], [h_tphi.T, h_pp]])
        local_hessians.append(jnp.asarray(0.5 * (hessian + hessian.T)))
        local_gradients.append(
            jnp.asarray(
                [
                    -0.2 + 0.1 * frame_index,
                    0.3 - 0.05 * frame_index,
                    0.1 + 0.02 * frame_index,
                    -0.15,
                ]
            )
        )

    def frame_loss(theta_values, frame_phi_values, frame_index):
        local = jnp.concatenate((theta_values, frame_phi_values), axis=0)
        ref = jnp.concatenate(
            (
                jnp.asarray(theta_ref),
                jnp.asarray(frame_phi_ref[frame_index]),
            ),
            axis=0,
        )
        delta = local - ref
        return (
            local_gradients[frame_index] @ delta
            + 0.5 * delta @ local_hessians[frame_index] @ delta
        )

    structured_blocks = build_independent_frame_theta_phi_quadratic_blocks(
        frame_loss_fn=frame_loss,
        theta_ref=theta_ref,
        frame_phi_ref=frame_phi_ref,
        subblock_reduce=reduce,  # type: ignore[arg-type]
    )
    structured = schur_reduce_independent_frame_blocks(
        structured_blocks,
        damping=1.0e-8,
    )
    sidecar = materialize_structured_schur_sidecar_blocks(structured_blocks)
    combined_gradient = np.concatenate((sidecar["g_theta"], sidecar["g_phi"]))
    combined_curvature = np.zeros(
        (
            structured_blocks.combined_dim,
            structured_blocks.combined_dim,
        ),
        dtype=float,
    )
    theta_slice = slice(0, theta_dim)
    phi_slice = slice(theta_dim, theta_dim + structured_blocks.phi_dim)
    combined_curvature[theta_slice, theta_slice] = sidecar["h_tt"]
    combined_curvature[theta_slice, phi_slice] = sidecar["h_tp"]
    combined_curvature[phi_slice, theta_slice] = sidecar["h_tp"].T
    combined_curvature[phi_slice, phi_slice] = sidecar["h_pp"]
    layout = build_combined_local_parameter_layout(
        tuple(f"theta.{index}" for index in range(theta_dim)),
        tuple(f"phi.{index}" for index in range(structured_blocks.phi_dim)),
    )
    dense_blocks = partition_local_curvature(
        layout=layout,
        combined_gradient=combined_gradient,
        combined_curvature=combined_curvature,
    )
    dense = schur_reduce_local_quadratic(
        blocks=dense_blocks,
        damping=1.0e-8,
    )
    return structured, dense


@pytest.mark.parametrize("n_frames", [1, 3])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_structured_schur_reduction_matches_dense_assembled_reduction(
    n_frames: int,
    reduce: str,
):
    structured, dense = _frame_quadratic_case(n_frames=n_frames, reduce=reduce)

    np.testing.assert_allclose(
        structured.reduced_information,
        dense.reduced_information,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        structured.reduced_score,
        dense.reduced_score,
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    assert structured.solve_method == "solve"
    assert structured.used_pseudoinverse is False
