import numpy as np
import pytest

from dluxshera.inference.observation_information_rate import (
    canonical_projected_gain,
    canonical_projected_gains,
    canonical_physical_directions,
    check_projected_gain_monotonicity,
    covariance_square_root,
    deduplicate_warnings,
    detect_degeneracy_groups,
    detect_quasi_degeneracy_groups,
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
    precision_normalized_projected_gain,
    precision_normalized_projected_gains,
    resolve_unique_mode_assignments,
    simulate_sequential_information_gate,
    subspace_overlap_diagnostics,
    symmetric_eigendecomposition,
    threshold_crossings,
    update_precision_with_information,
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


def test_canonical_physical_directions_identity_correlated_and_no_mutation():
    cov = np.array([[2.0, 0.4], [0.4, 1.0]])
    precision = np.linalg.inv(cov)
    w = covariance_square_root(cov)
    vectors = np.eye(2)
    w_before = w.copy()
    vectors_before = vectors.copy()
    directions = canonical_physical_directions(w, vectors, precision)
    norms = np.einsum("ik,ij,jk->k", directions, precision, directions)
    assert norms.tolist() == pytest.approx([1.0, 1.0])
    assert np.allclose(w, w_before)
    assert np.allclose(vectors, vectors_before)
    ident = canonical_physical_directions(np.eye(2), np.eye(2), np.eye(2))
    assert np.allclose(ident, np.eye(2))


def test_precision_normalized_gain_scalar_and_initial_whitened_equivalence():
    precision = np.array([[1.0]])
    info = np.array([[2.5]])
    direction = np.array([1.0])
    assert precision_normalized_projected_gain(precision, info, direction) == pytest.approx(2.5)
    updated = update_precision_with_information(precision, info)
    assert updated[0, 0] == pytest.approx(3.5)
    assert precision_normalized_projected_gain(updated, info, direction) == pytest.approx(2.5 / 3.5)
    cov = np.array([[4.0, 0.5], [0.5, 1.0]])
    p0 = np.linalg.inv(cov)
    w = covariance_square_root(cov)
    canonical = np.eye(2)
    dirs = canonical_physical_directions(w, canonical, p0)
    s = np.array([[0.2, 0.03], [0.03, 0.1]])
    whitened = whiten_information(s, w)
    assert precision_normalized_projected_gains(p0, s, dirs).tolist() == pytest.approx(canonical_projected_gains(whitened, canonical).tolist())


def test_sequential_scalar_block_growth_maximum_minimum_and_flushes():
    p0 = np.array([[1.0]])
    infos = np.asarray([[[0.5]]] * 12)
    durations = np.ones(12)
    dirs = np.array([[1.0]])
    updates = simulate_sequential_information_gate(
        p0,
        infos,
        durations,
        dirs,
        [0],
        gain_threshold=1.0,
        minimum_subblocks=1,
        maximum_subblocks=10,
    )
    lengths = [u.block_length for u in updates]
    assert lengths[0] == 2
    assert lengths[1] > lengths[0]
    final_precision = p0[0, 0] + np.sum(infos)
    assert final_precision == pytest.approx(7.0)
    assert updates[-1].closure_reason == "end_of_scope"

    maxed = simulate_sequential_information_gate(
        p0,
        infos[:5],
        durations[:5],
        dirs,
        [0],
        gain_threshold=100.0,
        minimum_subblocks=1,
        maximum_subblocks=3,
    )
    assert maxed[0].closure_reason == "maximum_latency"
    assert maxed[0].minimum_gain < 100.0

    minimum = simulate_sequential_information_gate(
        p0,
        np.asarray([[[10.0]]] * 4),
        durations[:4],
        dirs,
        [0],
        gain_threshold=1.0,
        minimum_subblocks=3,
        maximum_subblocks=4,
    )
    assert minimum[0].block_length == 3
    assert minimum[0].closure_reason == "natural_information_trigger"

    boundary = simulate_sequential_information_gate(
        p0,
        infos[:5],
        durations[:5],
        dirs,
        [0],
        gain_threshold=10.0,
        minimum_subblocks=1,
        maximum_subblocks=10,
        boundary_after=[False, True, False, False, False],
    )
    assert boundary[0].closure_reason == "historical_window_boundary"
    assert boundary[0].historical_window_boundary_flush


def test_sequential_multivariate_correlated_final_information_invariance():
    cov = np.array([[1.5, 0.2], [0.2, 0.8]])
    p0 = np.linalg.inv(cov)
    w = covariance_square_root(cov)
    dirs = canonical_physical_directions(w, np.eye(2), p0)
    infos = np.asarray(
        [
            [[0.4, 0.05], [0.05, 0.2]],
            [[0.3, 0.01], [0.01, 0.25]],
            [[0.2, 0.02], [0.02, 0.3]],
        ]
    )
    updates = simulate_sequential_information_gate(p0, infos, [1, 1, 1], dirs, [0, 1], gain_threshold=0.2, minimum_subblocks=1, maximum_subblocks=2)
    precision = p0.copy()
    for update in updates:
        precision = precision + np.sum(infos[update.start_index : update.end_index + 1], axis=0)
    assert np.allclose(precision, p0 + np.sum(infos, axis=0))
    assert np.all(np.linalg.eigvalsh(precision) > 0.0)
    assert np.all(np.linalg.eigvalsh(np.linalg.inv(p0) - np.linalg.inv(precision)) >= -1e-12)


def test_unique_mode_assignment_named_loadings_and_weak_status():
    loadings = {
        0: {"a": 0.8, "b": 0.75},
        1: {"a": 0.7, "b": 0.2},
        2: {"a": 0.1, "b": 0.4},
    }
    assignments = resolve_unique_mode_assignments(loadings, ["a", "b"])
    assert [a.canonical_mode_id for a in assignments] == [1, 0]
    weak = resolve_unique_mode_assignments({0: {"x": 0.1}, 1: {"x": 0.05}}, ["x"])
    assert "weak_assignment" in weak[0].assignment_status
    with pytest.raises(KeyError):
        resolve_unique_mode_assignments(loadings, ["missing"])


def test_strict_versus_quasi_degeneracy_groups():
    vals = np.array([10.0, 9.995, 8.0, 7.94, 1.0])
    strict = detect_degeneracy_groups(vals, rtol=1e-3)
    quasi = detect_quasi_degeneracy_groups(vals, quasi_rtol=1e-2, strict_rtol=1e-3)
    assert strict[0] == (0, 1)
    assert (2, 3) not in strict
    assert (2, 3) in quasi


def test_warning_deduplication_order_contexts_and_distinct_values():
    warnings = [
        {"status": "x", "message": "same", "value": np.float64(1.0), "context": "a"},
        {"status": "x", "message": "same", "value": 1.0, "context": "b"},
        {"status": "x", "message": "same", "value": 2.0, "context": "a"},
    ]
    deduped = deduplicate_warnings(warnings)
    assert len(deduped) == 2
    assert deduped[0]["status"] == "x"
    assert deduped[0]["contexts"] == ["a", "b"]
    assert deduped[1]["value"] == 2.0
