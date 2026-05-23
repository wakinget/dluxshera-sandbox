from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dluxshera.inference.observation_belief import (
    ObservationBeliefState,
    SubblockSummary,
    update_observation_belief,
)
from dluxshera.inference.observation_summary import (
    ImageBackedSubblockSummaryArtifact,
    build_combined_local_parameter_layout,
    get_summary_information_accounting,
    get_summary_information_scale,
    load_subblock_summary,
    partition_local_curvature,
    schur_reduce_local_quadratic,
    validate_summary_information_scale,
)


def test_build_combined_local_parameter_layout_is_deterministic():
    layout = build_combined_local_parameter_layout(
        ("theta.source.separation_as", "theta.source.contrast"),
        ("phi.frame[0].source.x_position_as", "phi.frame[0].source.y_position_as"),
    )

    assert layout.theta_slice == slice(0, 2)
    assert layout.phi_slice == slice(2, 4)
    assert layout.combined_labels == (
        "theta.source.separation_as",
        "theta.source.contrast",
        "phi.frame[0].source.x_position_as",
        "phi.frame[0].source.y_position_as",
    )


def _summary_payload_with_scale(scale: str | None) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "image_backed_subblock_summary.v1",
        "subblock_id": "test",
    }
    if scale is not None:
        payload["information_accounting"] = {
            "summary_information_scale": scale,
            "summary_frame_reduce": "sum",
            "summary_subblock_reduce": "sum" if scale == "summed_likelihood" else "mean",
        }
    return payload


def test_summary_information_scale_validation_requires_summed_artifacts():
    payload = _summary_payload_with_scale("summed_likelihood")

    assert get_summary_information_scale(payload) == "summed_likelihood"
    assert get_summary_information_accounting(payload)["summary_subblock_reduce"] == "sum"
    validation = validate_summary_information_scale(payload)

    assert validation["accepted_summary_information_scale"] == "summed_likelihood"
    assert validation["summary_scale_policy"] == "require_summed"
    assert validation["override_used"] is False


@pytest.mark.parametrize(
    ("payloads", "match"),
    [
        ([_summary_payload_with_scale("optimizer")], "optimizer"),
        ([_summary_payload_with_scale(None)], "information-accounting"),
        (
            [
                _summary_payload_with_scale("summed_likelihood"),
                _summary_payload_with_scale("optimizer"),
            ],
            "optimizer",
        ),
    ],
)
def test_summary_information_scale_validation_rejects_untrusted_units(
    payloads: list[dict[str, object]],
    match: str,
):
    with pytest.raises(ValueError, match=match):
        validate_summary_information_scale(payloads)


def test_summary_information_scale_validation_records_explicit_override():
    validation = validate_summary_information_scale(
        [_summary_payload_with_scale("optimizer"), _summary_payload_with_scale(None)],
        policy="allow_optimizer",
        summary_paths=("optimizer.json", "legacy.json"),
    )

    assert validation["accepted_summary_information_scale"] == "mixed"
    assert validation["override_used"] is True
    assert [item["summary_information_scale"] for item in validation["input_summaries"]] == [
        "optimizer",
        "unknown",
    ]
    assert validation["warnings"]


def test_partition_local_curvature_returns_expected_shapes():
    layout = build_combined_local_parameter_layout(
        ("theta.a", "theta.b"),
        ("phi.c", "phi.d", "phi.e"),
    )
    gradient = np.arange(5.0)
    curvature = np.arange(25.0, dtype=float).reshape(5, 5)
    curvature = 0.5 * (curvature + curvature.T)

    blocks = partition_local_curvature(
        layout=layout,
        combined_gradient=gradient,
        combined_curvature=curvature,
    )

    assert blocks.h_tt.shape == (2, 2)
    assert blocks.h_tp.shape == (2, 3)
    assert blocks.h_pp.shape == (3, 3)
    assert blocks.g_theta.shape == (2,)
    assert blocks.g_phi.shape == (3,)


def test_schur_reduced_information_and_score_match_dense_reference():
    layout = build_combined_local_parameter_layout(("theta.a",), ("phi.b", "phi.c"))
    gradient = np.array([1.5, -2.0, 0.5])
    curvature = np.array(
        [
            [6.0, 1.0, -0.5],
            [1.0, 4.0, 0.2],
            [-0.5, 0.2, 3.0],
        ]
    )
    blocks = partition_local_curvature(
        layout=layout,
        combined_gradient=gradient,
        combined_curvature=curvature,
    )

    reduced = schur_reduce_local_quadratic(blocks=blocks)
    solved_h_pt = np.linalg.solve(blocks.h_pp, blocks.h_tp.T)
    solved_g_phi = np.linalg.solve(blocks.h_pp, blocks.g_phi)
    expected_info = blocks.h_tt - blocks.h_tp @ solved_h_pt
    expected_score = blocks.g_theta - blocks.h_tp @ solved_g_phi

    np.testing.assert_allclose(reduced.reduced_information, expected_info)
    np.testing.assert_allclose(reduced.reduced_score, expected_score)


def test_schur_reduction_with_damping_reports_diagnostics():
    layout = build_combined_local_parameter_layout(("theta.a",), ("phi.b",))
    gradient = np.array([0.0, 1.0])
    curvature = np.array([[2.0, 1.0], [1.0, 0.0]])
    blocks = partition_local_curvature(
        layout=layout,
        combined_gradient=gradient,
        combined_curvature=curvature,
    )

    reduced = schur_reduce_local_quadratic(
        blocks=blocks,
        damping=1.0,
    )

    assert reduced.h_pp_solve_method == "solve"
    assert reduced.schur_result.damping == pytest.approx(1.0)
    assert "h_pp" in reduced.to_diagnostics_dict()


def test_image_backed_summary_writer_and_loader_round_trip(tmp_path: Path):
    layout = build_combined_local_parameter_layout(("theta.a",), ("phi.b",))
    gradient = np.array([0.5, -0.25])
    curvature = np.array([[4.0, 1.0], [1.0, 3.0]])
    blocks = partition_local_curvature(
        layout=layout,
        combined_gradient=gradient,
        combined_curvature=curvature,
    )
    reduced = schur_reduce_local_quadratic(blocks=blocks)
    summary = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=("theta_a",),
        theta_ref=np.array([2.0]),
        reduced_information=reduced.reduced_information,
        reduced_score=reduced.reduced_score,
        summary_kind="image_backed_schur",
    )
    artifact = ImageBackedSubblockSummaryArtifact(
        summary=summary,
        layout=layout,
        theta_ref=np.array([2.0]),
        phi_ref=np.array([3.0]),
        reduced=reduced,
        metadata={"case_root": "/tmp/case"},
        combined_gradient=gradient,
        combined_curvature=curvature,
    )

    summary_path = tmp_path / "subblock_summary.json"
    matrix_path = tmp_path / "subblock_summary_matrices.npz"
    artifact.write(summary_json_path=summary_path, matrix_npz_path=matrix_path)

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["matrix_artifact_path"] == matrix_path.name
    assert matrix_path.exists()

    loaded = load_subblock_summary(summary_path)
    assert loaded.subblock_id == "subblock_000000"
    np.testing.assert_allclose(loaded.theta_ref, np.array([2.0]))
    np.testing.assert_allclose(loaded.reduced_information, reduced.reduced_information)
    np.testing.assert_allclose(loaded.reduced_score, reduced.reduced_score)


def test_observation_belief_accumulator_consumes_loaded_real_summary(tmp_path: Path):
    layout = build_combined_local_parameter_layout(("theta.a",), ("phi.b",))
    gradient = np.array([-2.0, 0.0])
    curvature = np.array([[4.0, 0.0], [0.0, 1.0]])
    blocks = partition_local_curvature(
        layout=layout,
        combined_gradient=gradient,
        combined_curvature=curvature,
    )
    reduced = schur_reduce_local_quadratic(blocks=blocks)
    summary = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=("source.separation_as",),
        theta_ref=np.array([0.0]),
        reduced_information=reduced.reduced_information,
        reduced_score=reduced.reduced_score,
        summary_kind="image_backed_schur",
    )
    artifact = ImageBackedSubblockSummaryArtifact(
        summary=summary,
        layout=layout,
        theta_ref=np.array([0.0]),
        phi_ref=np.array([0.0]),
        reduced=reduced,
    )

    summary_path = tmp_path / "subblock_summary.json"
    matrix_path = tmp_path / "subblock_summary_matrices.npz"
    artifact.write(summary_json_path=summary_path, matrix_npz_path=matrix_path)
    loaded = load_subblock_summary(summary_path)

    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=("source.separation_as",),
        mean=np.array([0.0]),
        sigma=np.array([10.0]),
    )
    result = update_observation_belief(prior, [loaded])

    assert result.posterior.mean[0] > 0.0
    assert result.posterior.precision[0, 0] > prior.precision[0, 0]
