from __future__ import annotations

import numpy as np
import pytest

from dluxshera.inference.observation_belief import (
    ObservationBeliefState,
    ObservationThetaLayout,
    ObservationUpdatePolicy,
    SubblockSummary,
    build_observation_eigenbasis,
    build_prior_whitened_information_gain_matrix,
    build_system_observation_theta_layout,
    infer_indexed_parameter_indices,
    schur_reduce_information,
    update_observation_belief,
    update_observation_belief_with_policy,
)


@pytest.mark.parametrize(
    ("config", "expected_labels"),
    [
        (
            {
                "theta_layout": {
                    "source": {
                        "separation_as": True,
                        "log_flux_total": True,
                        "contrast": True,
                    },
                    "optics": {
                        "plate_scale_as_per_pix": False,
                        "primary_zernikes": {"enabled": False, "indices": []},
                        "secondary_zernikes": {"enabled": False, "indices": []},
                    },
                }
            },
            (
                "source.separation_as",
                "source.log_flux_total",
                "source.contrast",
            ),
        ),
        (
            {
                "theta_layout": {
                    "source": {
                        "separation_as": True,
                        "log_flux_total": False,
                        "contrast": False,
                    },
                    "optics": {
                        "plate_scale_as_per_pix": True,
                        "primary_zernikes": {"enabled": False, "indices": []},
                        "secondary_zernikes": {"enabled": False, "indices": []},
                    },
                }
            },
            (
                "source.separation_as",
                "optics.plate_scale_as_per_pix",
            ),
        ),
        (
            {
                "theta_layout": {
                    "source": {
                        "separation_as": True,
                        "log_flux_total": True,
                        "contrast": True,
                    },
                    "optics": {
                        "plate_scale_as_per_pix": True,
                        "primary_zernikes": {"enabled": True, "indices": [0, 2]},
                        "secondary_zernikes": {"enabled": True, "indices": [0, 2]},
                    },
                }
            },
            (
                "source.separation_as",
                "source.log_flux_total",
                "source.contrast",
                "optics.plate_scale_as_per_pix",
                "optics.primary.zernike_coeffs_nm[0]",
                "optics.primary.zernike_coeffs_nm[2]",
                "optics.secondary.zernike_coeffs_nm[0]",
                "optics.secondary.zernike_coeffs_nm[2]",
            ),
        ),
    ],
)
def test_observation_theta_layout_expands_expected_labels(config, expected_labels):
    layout = ObservationThetaLayout.from_config(config)

    assert layout.labels == expected_labels
    assert layout.size == len(expected_labels)
    np.testing.assert_allclose(layout.validate_vector(np.zeros(layout.size)), 0.0)
    np.testing.assert_allclose(layout.validate_matrix(np.eye(layout.size)), np.eye(layout.size))


def test_system_observation_theta_layout_uses_store_vector_lengths_and_masks():
    store = {
        "optics.primary.zernike_coeffs_nm": np.zeros(4),
        "optics.secondary.zernike_coeffs_nm": np.zeros(3),
        "optics.primary_noll_indices": np.array([4, 5, 6, 7]),
        "optics.secondary_noll_indices": np.array([4, 5, 6]),
    }

    assert infer_indexed_parameter_indices(
        store,
        "optics.primary.zernike_coeffs_nm",
    ) == (0, 1, 2, 3)

    layout, metadata = build_system_observation_theta_layout(
        store,
        config={
            "source": {
                "separation_as": True,
                "log_flux_total": False,
                "contrast": True,
            },
            "optics": {
                "plate_scale_as_per_pix": False,
                "primary_zernikes": {
                    "enabled": True,
                    "indices": "from_system",
                    "include": [0, 2, 3],
                    "exclude": [2],
                },
                "secondary_zernikes": {
                    "enabled": True,
                    "indices": "from_system",
                    "include": None,
                    "exclude": [1],
                },
            },
        },
    )

    assert layout.labels == (
        "source.separation_as",
        "source.contrast",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.primary.zernike_coeffs_nm[3]",
        "optics.secondary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[2]",
    )
    assert metadata["primary_zernike_indices"] == [0, 3]
    assert metadata["secondary_zernike_indices"] == [0, 2]
    assert metadata["primary_zernike_noll_indices"] == [4, 5, 6, 7]


def test_system_observation_theta_layout_rejects_bad_masks():
    store = {"optics.primary.zernike_coeffs_nm": np.zeros(2)}

    with pytest.raises(ValueError, match="duplicates"):
        build_system_observation_theta_layout(
            store,
            config={
                "optics": {
                    "primary_zernikes": {
                        "enabled": True,
                        "indices": "from_system",
                        "include": [0, 0],
                    },
                    "secondary_zernikes": {"enabled": False},
                }
            },
        )

    with pytest.raises(ValueError, match="outside the resolved system"):
        build_system_observation_theta_layout(
            store,
            config={
                "optics": {
                    "primary_zernikes": {
                        "enabled": True,
                        "indices": "from_system",
                        "include": [5],
                    },
                    "secondary_zernikes": {"enabled": False},
                }
            },
        )


def test_schur_reduce_information_matches_dense_reference():
    h_tt = np.array([[5.0, 1.5], [1.5, 4.0]])
    h_tp = np.array([[1.0, 0.5], [2.0, -1.0]])
    h_pp = np.array([[3.0, 0.2], [0.2, 2.5]])

    result = schur_reduce_information(h_tt, h_tp, h_pp)
    expected = h_tt - h_tp @ np.linalg.solve(h_pp, h_tp.T)
    expected = 0.5 * (expected + expected.T)

    np.testing.assert_allclose(result.reduced_information, expected)
    assert result.solve_method == "solve"
    assert result.used_pseudoinverse is False


def test_update_observation_belief_recovers_1d_gaussian_posterior():
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=("source.separation_as",),
        mean=np.array([0.0]),
        sigma=np.array([2.0]),
    )
    summary = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=("source.separation_as",),
        theta_ref=np.array([0.0]),
        reduced_information=np.array([[4.0]]),
        reduced_score=np.array([-12.0]),
    )

    result = update_observation_belief(prior, [summary])
    expected_precision = 4.0 + 0.25
    expected_mean = 12.0 / expected_precision

    np.testing.assert_allclose(result.posterior.precision, np.array([[expected_precision]]))
    np.testing.assert_allclose(result.posterior.mean, np.array([expected_mean]))
    np.testing.assert_allclose(
        result.posterior.covariance,
        np.array([[1.0 / expected_precision]]),
    )


def test_update_observation_belief_handles_different_theta_refs_consistently():
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=("source.separation_as",),
        mean=np.array([0.0]),
        sigma=np.array([10.0]),
    )
    summary_ref_0 = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=prior.theta_labels,
        theta_ref=np.array([0.0]),
        reduced_information=np.array([[2.0]]),
        reduced_score=np.array([-3.0]),
    )
    summary_ref_1 = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000001",
        theta_labels=prior.theta_labels,
        theta_ref=np.array([1.0]),
        reduced_information=np.array([[2.0]]),
        reduced_score=np.array([-1.0]),
    )
    summary_same_info = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000002",
        theta_labels=prior.theta_labels,
        theta_ref=np.array([0.0]),
        reduced_information=np.array([[2.0]]),
        reduced_score=np.array([-3.0]),
    )

    mixed_refs = update_observation_belief(prior, [summary_ref_0, summary_ref_1])
    common_refs = update_observation_belief(prior, [summary_ref_0, summary_same_info])

    np.testing.assert_allclose(mixed_refs.posterior.mean, common_refs.posterior.mean)
    np.testing.assert_allclose(mixed_refs.posterior.precision, common_refs.posterior.precision)


def test_posterior_covariance_shrinks_with_multiple_information_summaries():
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=("source.separation_as", "source.log_flux_total"),
        mean=np.array([0.0, 0.0]),
        sigma=np.array([3.0, 4.0]),
    )
    theta_true = np.array([0.5, -0.25])
    reduced_information_1 = np.array([[1.0, 0.2], [0.2, 0.8]])
    reduced_information_2 = np.array([[0.7, -0.1], [-0.1, 1.3]])

    summary_1 = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=prior.theta_labels,
        theta_ref=np.array([0.0, 0.0]),
        reduced_information=reduced_information_1,
        reduced_score=reduced_information_1 @ (np.array([0.0, 0.0]) - theta_true),
    )
    summary_2 = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000001",
        theta_labels=prior.theta_labels,
        theta_ref=np.array([0.2, -0.1]),
        reduced_information=reduced_information_2,
        reduced_score=reduced_information_2 @ (np.array([0.2, -0.1]) - theta_true),
    )

    result_1 = update_observation_belief(prior, [summary_1])
    result_2 = update_observation_belief(prior, [summary_1, summary_2])

    assert np.all(np.diag(result_1.posterior.covariance) < np.diag(prior.covariance))
    assert np.all(np.diag(result_2.posterior.covariance) < np.diag(result_1.posterior.covariance))


def test_build_observation_eigenbasis_identifies_weak_zernike_mode():
    labels = (
        "source.separation_as",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[0]",
    )
    precision = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 5.0, 4.99],
            [0.0, 4.99, 5.0],
        ]
    )

    basis = build_observation_eigenbasis(
        precision,
        labels,
        eig_floor_rel=0.05,
    )

    assert basis.weak_mode_mask[-1]
    assert basis.eigenvalues[-1] < 0.1
    contributors = basis.mode_contributors(2, top_k=2)
    assert {label for label, _ in contributors} == {
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[0]",
    }


def test_build_prior_whitened_information_gain_matrix_matches_diagonal_whitening():
    information = np.array([[4.0, 1.5], [1.5, 9.0]])
    prior_sigma = np.array([0.5, 2.0])

    gain = build_prior_whitened_information_gain_matrix(information, prior_sigma)
    expected = np.diag(prior_sigma) @ information @ np.diag(prior_sigma)

    np.testing.assert_allclose(gain, expected)


def test_prior_whitened_gain_eigenbasis_identifies_strongest_gain_direction():
    labels = (
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
    )
    information = np.diag([0.2, 3.5, 0.1])
    prior_sigma = np.array([1.0, 2.0, 1.0])

    gain = build_prior_whitened_information_gain_matrix(information, prior_sigma)
    basis = build_observation_eigenbasis(gain, labels)

    assert basis.eigenvalues[0] == pytest.approx(14.0)
    assert basis.mode_contributors(0, top_k=1)[0][0] == "source.log_flux_total"


def test_observation_eigenbasis_rows_report_raw_and_floored_fields():
    labels = (
        "source.separation_as",
        "source.log_flux_total",
    )
    precision = np.array([[2.0, 0.0], [0.0, 1.0e-8]])

    basis = build_observation_eigenbasis(
        precision,
        labels,
        eig_floor_rel=0.1,
    )
    rows = basis.to_rows(top_k=2)

    assert rows[0]["raw_eigenvalue"] == pytest.approx(2.0)
    assert rows[0]["floored_eigenvalue"] == pytest.approx(2.0)
    assert rows[0]["was_floored"] is False
    assert rows[1]["raw_eigenvalue"] == pytest.approx(1.0e-8)
    assert rows[1]["floored_eigenvalue"] == pytest.approx(0.2)
    assert rows[1]["was_floored"] is True
    assert "raw_sigma_along_mode" in rows[1]
    assert "floored_sigma_along_mode" in rows[1]


def _synthetic_policy_summary(
    *,
    labels: tuple[str, ...],
    information: np.ndarray,
    theta_target: np.ndarray,
    theta_ref: np.ndarray | None = None,
) -> SubblockSummary:
    if theta_ref is None:
        theta_ref = np.zeros(len(labels), dtype=float)
    return SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=labels,
        theta_ref=theta_ref,
        reduced_information=information,
        reduced_score=information @ (theta_ref - theta_target),
    )


def test_policy_physical_full_matches_existing_update_for_diagonal_case():
    labels = ("source.separation_as", "source.log_flux_total")
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=np.array([0.0, 0.0]),
        sigma=np.array([2.0, 3.0]),
    )
    summary = _synthetic_policy_summary(
        labels=labels,
        information=np.diag([4.0, 2.0]),
        theta_target=np.array([1.0, -0.5]),
    )

    legacy = update_observation_belief(prior, [summary])
    policy_result = update_observation_belief_with_policy(
        prior,
        [summary],
        policy=ObservationUpdatePolicy(update_mode="physical_full"),
    )

    np.testing.assert_allclose(policy_result.posterior.mean, legacy.posterior.mean)
    np.testing.assert_allclose(
        policy_result.posterior.precision,
        legacy.posterior.precision,
    )
    np.testing.assert_allclose(
        policy_result.physical_update_full,
        legacy.posterior.mean - prior.mean,
    )


def test_policy_eigen_full_matches_physical_full_for_coupled_matrix():
    labels = (
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
    )
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=np.array([0.1, -0.2, 0.3]),
        sigma=np.array([2.0, 1.5, 3.0]),
    )
    information = np.array(
        [
            [5.0, 1.2, -0.4],
            [1.2, 4.0, 0.8],
            [-0.4, 0.8, 3.0],
        ]
    )
    summary = _synthetic_policy_summary(
        labels=labels,
        information=information,
        theta_target=np.array([0.7, -0.1, 0.9]),
    )

    physical = update_observation_belief_with_policy(
        prior,
        [summary],
        policy={"update_mode": "physical_full"},
    )
    eigen = update_observation_belief_with_policy(
        prior,
        [summary],
        policy={"update_mode": "eigen_full", "basis_source": "posterior_precision"},
    )

    np.testing.assert_allclose(
        eigen.posterior.mean,
        physical.posterior.mean,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        eigen.physical_update_applied,
        physical.physical_update_applied,
        atol=1e-12,
    )
    assert eigen.diagnostics["n_modes_kept"] == len(labels)


def test_policy_eigen_truncated_removes_weak_mode_component():
    labels = ("source.separation_as", "source.log_flux_total")
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=np.zeros(2),
        sigma=np.array([100.0, 100.0]),
    )
    summary = _synthetic_policy_summary(
        labels=labels,
        information=np.diag([10.0, 0.01]),
        theta_target=np.array([2.0, 3.0]),
    )

    result = update_observation_belief_with_policy(
        prior,
        [summary],
        policy={
            "update_mode": "eigen_truncated",
            "basis_source": "accumulated_information",
            "gate_source": "accumulated_information",
            "whiten": False,
            "eig_floor_abs": 1.0,
        },
    )

    assert result.kept_mode_mask.tolist() == [True, False]
    assert result.posterior.mean[0] == pytest.approx(result.posterior_full.mean[0])
    assert result.posterior.mean[1] == pytest.approx(prior.mean[1])
    assert result.diagnostics["n_modes_kept"] == 1


def test_policy_eigen_damped_reduces_update_norm_relative_to_eigen_full():
    labels = ("source.separation_as", "source.log_flux_total")
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=np.zeros(2),
        sigma=np.array([2.0, 2.0]),
    )
    information = np.array([[3.0, 0.4], [0.4, 1.5]])
    summary = _synthetic_policy_summary(
        labels=labels,
        information=information,
        theta_target=np.array([1.0, -2.0]),
    )

    full = update_observation_belief_with_policy(
        prior,
        [summary],
        policy={"update_mode": "eigen_full", "whiten": False},
    )
    damped = update_observation_belief_with_policy(
        prior,
        [summary],
        policy={
            "update_mode": "eigen_damped",
            "whiten": False,
            "damping_mode": "information",
            "damping_value": 5.0,
        },
    )

    assert np.linalg.norm(damped.physical_update_applied) < np.linalg.norm(
        full.physical_update_applied
    )
    assert np.all(damped.damping_factors < 1.0)


@pytest.mark.parametrize(
    "policy_kwargs, match",
    [
        ({"update_mode": "bad"}, "update_mode"),
        ({"basis_source": "bad"}, "basis_source"),
        ({"gate_source": "bad"}, "gate_source"),
        ({"eig_floor_abs": -1.0}, "non-negative"),
        ({"eig_floor_rel": -1.0}, "non-negative"),
        ({"min_kept_modes": 3, "max_kept_modes": 2}, "cannot exceed"),
    ],
)
def test_observation_update_policy_rejects_invalid_configuration(
    policy_kwargs,
    match,
):
    with pytest.raises(ValueError, match=match):
        ObservationUpdatePolicy(**policy_kwargs)


def test_policy_rejects_impossible_kept_mode_count_for_layout():
    labels = ("source.separation_as", "source.log_flux_total")
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=np.zeros(2),
        sigma=np.ones(2),
    )
    summary = _synthetic_policy_summary(
        labels=labels,
        information=np.eye(2),
        theta_target=np.ones(2),
    )

    with pytest.raises(ValueError, match="min_kept_modes"):
        update_observation_belief_with_policy(
            prior,
            [summary],
            policy={
                "update_mode": "eigen_truncated",
                "min_kept_modes": 3,
            },
        )


def test_policy_preserves_label_order_and_vector_shapes():
    labels = (
        "source.contrast",
        "source.separation_as",
        "optics.plate_scale_as_per_pix",
    )
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=np.array([0.0, 1.0, -1.0]),
        sigma=np.array([1.0, 2.0, 3.0]),
    )
    summary = _synthetic_policy_summary(
        labels=(labels[2], labels[0], labels[1]),
        information=np.diag([2.0, 3.0, 4.0]),
        theta_ref=np.array([-1.0, 0.0, 1.0]),
        theta_target=np.array([-0.5, 0.4, 1.5]),
    )

    result = update_observation_belief_with_policy(
        prior,
        [summary],
        policy={"update_mode": "eigen_full", "whiten": False},
    )

    assert result.posterior.theta_labels == labels
    assert result.physical_update_full.shape == (len(labels),)
    assert result.physical_update_applied.shape == (len(labels),)
    assert result.eigen_update_full.shape == (len(labels),)
    assert result.eigenvectors.shape == (len(labels), len(labels))


def test_policy_whitening_path_uses_non_uniform_prior_sigma():
    labels = ("source.separation_as", "source.log_flux_total")
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=np.zeros(2),
        sigma=np.array([0.5, 4.0]),
    )
    information = np.diag([4.0, 1.0])
    summary = _synthetic_policy_summary(
        labels=labels,
        information=information,
        theta_target=np.array([1.0, 1.0]),
    )

    result = update_observation_belief_with_policy(
        prior,
        [summary],
        policy={
            "update_mode": "eigen_full",
            "basis_source": "accumulated_information",
            "gate_source": "accumulated_information",
            "whiten": True,
        },
    )

    expected_gain = build_prior_whitened_information_gain_matrix(
        information,
        prior.sigma(),
    )
    np.testing.assert_allclose(result.eigenvalues, np.array([16.0, 1.0]))
    np.testing.assert_allclose(
        np.linalg.eigvalsh(expected_gain)[::-1],
        result.eigenvalues,
    )
    np.testing.assert_allclose(result.posterior.mean, result.posterior_full.mean)
    assert result.diagnostics["eigenvalue_basis"] == "prior_whitened"
