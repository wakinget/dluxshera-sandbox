from __future__ import annotations

import numpy as np
import pytest

from dluxshera.inference.observation_belief import (
    ObservationBeliefState,
    ObservationThetaLayout,
    SubblockSummary,
    build_observation_eigenbasis,
    build_prior_whitened_information_gain_matrix,
    schur_reduce_information,
    update_observation_belief,
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
