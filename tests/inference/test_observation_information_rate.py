import numpy as np
import pytest

from dluxshera.inference.observation_information_rate import (
    canonical_projected_gain,
    canonical_projected_gains,
    check_projected_gain_monotonicity,
    covariance_square_root,
    detect_degeneracy_groups,
    deterministic_sign_vectors,
    drift_scenario,
    effective_rank,
    fit_information_rate,
    label_physical_group,
    matrix_psd_diagnostics,
    mode_composition,
    mode_overlap_assignment,
    observability_category,
    posterior_marginal_sigma,
    subspace_overlap_diagnostics,
    symmetric_eigendecomposition,
    threshold_crossings,
    whiten_information,
)


def test_whitening_diagonal_correlated_identity_and_invalid_covariance():
    info = np.diag([2.0, 5.0])
    w = covariance_square_root(np.diag([4.0, 9.0]))
    assert np.allclose(w @ w.T, np.diag([4.0, 9.0]))
    assert np.allclose(whiten_information(info, w), np.diag([8.0, 45.0]))

    cov = np.array([[4.0, 1.0], [1.0, 2.0]])
    w = covariance_square_root(cov)
    assert np.allclose(w @ w.T, cov)
    assert np.allclose(whiten_information(np.eye(2), np.eye(2)), np.eye(2))

    with pytest.raises(ValueError):
        covariance_square_root(np.array([[1.0, 2.0], [2.0, 1.0]]))


def test_known_diagonal_spectrum_thresholds_and_variance_contraction():
    rate = np.diag([3.0, 1.0, 0.25])
    spec = symmetric_eigendecomposition(rate)
    assert spec.eigenvalues.tolist() == pytest.approx([3.0, 1.0, 0.25])
    gain_10s = canonical_projected_gains(rate * 10.0, spec.eigenvectors)
    assert gain_10s.tolist() == pytest.approx([30.0, 10.0, 2.5])
    assert (1.0 / (1.0 + gain_10s)).tolist() == pytest.approx([1 / 31, 1 / 11, 1 / 3.5])
    crossings = threshold_crossings([1, 2, 3, 4], [0.2, 0.5, 1.0, 2.0], [0.5, 1.5, 3.0])
    assert crossings[0].crossed
    assert crossings[0].crossing_time_s == pytest.approx(2.0)
    assert crossings[1].interpolated_time_s == pytest.approx(3.5)
    assert not crossings[2].crossed


def test_sign_invariance_is_deterministic():
    vectors = np.array([[-0.1, 0.8], [0.9, -0.2]])
    signed = deterministic_sign_vectors(vectors)
    assert signed[1, 0] > 0.0
    assert signed[0, 1] > 0.0
    flipped = deterministic_sign_vectors(-vectors)
    assert np.allclose(flipped, signed)


def test_mode_permutation_alignment():
    ref = np.eye(3)
    current_vectors = ref[:, [2, 0, 1]]
    current_values = np.array([30.0, 10.0, 20.0])
    values, vectors, rows = mode_overlap_assignment(ref, current_values, current_vectors)
    assert values.tolist() == pytest.approx([10.0, 20.0, 30.0])
    assert np.allclose(vectors, ref)
    assert [row["assigned_mode"] for row in rows] == [1, 2, 0]


def test_eigenvalue_crossing_projected_gains_remain_stable():
    canonical = np.eye(2)
    increments = [np.diag([2.0, 0.1]), np.diag([0.1, 5.0])]
    cumulative = np.zeros((2, 2))
    curves = []
    leading_modes = []
    for inc in increments:
        cumulative += inc
        spec = symmetric_eigendecomposition(cumulative)
        leading_modes.append(int(np.argmax(np.abs(spec.eigenvectors[:, 0]))))
        curves.append(canonical_projected_gains(cumulative, canonical))
    assert leading_modes == [0, 1]
    assert curves[0].tolist() == pytest.approx([2.0, 0.1])
    assert curves[1].tolist() == pytest.approx([2.1, 5.1])


def test_degenerate_rotation_subspace_overlap():
    vals = np.array([4.0, 4.0 * (1.0 - 1.0e-4), 1.0])
    groups = detect_degeneracy_groups(vals, rtol=1.0e-3)
    assert groups[0] == (0, 1)
    angle = np.pi / 5.0
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    ref = np.eye(3)[:, :2]
    cur = np.eye(3)[:, :2] @ rot
    diag = subspace_overlap_diagnostics(ref, cur)
    assert diag["minimum_subspace_singular_value"] == pytest.approx(1.0)
    assert diag["maximum_principal_angle_deg"] == pytest.approx(0.0, abs=1.0e-10)


def test_projected_gain_monotonicity_for_psd_increments():
    v = np.array([1.0, 0.0])
    cumulative = np.zeros((2, 2))
    gains = []
    for inc in [np.diag([0.2, 1.0]), np.array([[0.3, 0.1], [0.1, 0.4]])]:
        cumulative += inc
        gains.append(canonical_projected_gain(cumulative, v))
    assert check_projected_gain_monotonicity(gains)["monotonic"]


def test_information_rate_fit_linear_and_nonlinear():
    linear = fit_information_rate([1, 2, 3], [2, 4, 6])
    assert linear.through_origin_slope == pytest.approx(2.0)
    assert linear.r_squared == pytest.approx(1.0)
    nonlinear = fit_information_rate([1, 2, 3], [1, 5, 6])
    assert nonlinear.max_fractional_departure > 0.0


def test_physical_composition_group_norms_and_dominant_labels():
    labels = [
        "source.separation_as",
        "optics.plate_scale_as_per_pix",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[0]",
        "custom.term",
    ]
    vectors = np.eye(5)
    rows, summaries = mode_composition(labels, np.diag([2, 3, 4, 5, 6]), vectors)
    assert {label_physical_group(label) for label in labels} == {
        "source",
        "plate_scale",
        "m1_zernike",
        "m2_zernike",
        "other",
    }
    first = summaries[0]
    assert first["dominant_labels"] == "source.separation_as;optics.plate_scale_as_per_pix;optics.primary.zernike_coeffs_nm[0];optics.secondary.zernike_coeffs_nm[0];custom.term"
    for summary in summaries:
        total = (
            summary["source_group_squared_norm"]
            + summary["plate_scale_squared_norm"]
            + summary["m1_zernike_squared_norm"]
            + summary["m2_zernike_squared_norm"]
            + summary["other_squared_norm"]
        )
        assert total == pytest.approx(1.0)
    assert len(rows) == 25


def test_psd_handling_tiny_and_material_negative():
    tiny = matrix_psd_diagnostics(np.diag([1.0, -1.0e-12]))
    assert tiny["clipped_eigenvalue_count"] == 1
    assert not tiny["materially_indefinite"]
    material = matrix_psd_diagnostics(np.diag([1.0, -1.0e-3]))
    assert material["materially_indefinite"]


def test_drift_formulas_and_observability():
    out = drift_scenario(2.0, 0.5)
    assert out.process_variance_rate == pytest.approx(2.0 * 0.5**4)
    assert out.rms_drift_per_sqrt_s == pytest.approx(np.sqrt(2.0 * 0.5**4))
    assert out.steady_state_variance == pytest.approx(0.5**2)
    assert out.steady_state_sigma == pytest.approx(0.5)
    assert observability_category(2.0) == "subblock_scale"
    assert observability_category(1.0 / 100.0) == "five_minute_scale"


def test_duration_weighting_and_posterior_sigma():
    rates = [np.diag([1.0, 2.0]), np.diag([3.0, 4.0])]
    durations = [2.0, 6.0]
    pooled = sum(rate * dt for rate, dt in zip(rates, durations)) / sum(durations)
    assert np.allclose(pooled, np.diag([2.5, 3.5]))
    sigma = posterior_marginal_sigma(np.eye(2), pooled * 4.0)
    assert sigma.tolist() == pytest.approx((1.0 / np.sqrt([11.0, 15.0])).tolist())
