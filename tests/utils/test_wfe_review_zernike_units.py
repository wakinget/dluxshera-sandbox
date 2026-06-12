from __future__ import annotations

import numpy as np

from dluxshera.utils.high_order_wfe import (
    fit_zernike_coefficients_nm,
    make_pupil_mask,
    reconstruct_zernike_opd_nm,
)


def _assert_coeffs_close(actual: dict[str, float], expected: dict[str, float], atol: float = 1.0e-10) -> None:
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert abs(actual[key] - value) < atol


def test_pure_z4_signal_recovers_nm_coefficient() -> None:
    mask = make_pupil_mask((48, 48))
    opd = reconstruct_zernike_opd_nm({"Z4": 5.0}, mask.shape, mask=mask)

    coeffs = fit_zernike_coefficients_nm(opd, mask, [4])

    _assert_coeffs_close(coeffs, {"Z4": 5.0})


def test_pure_z5_signal_recovers_negative_nm_coefficient() -> None:
    mask = make_pupil_mask((48, 48))
    opd = reconstruct_zernike_opd_nm({"Z5": -2.0}, mask.shape, mask=mask)

    coeffs = fit_zernike_coefficients_nm(opd, mask, [5])

    _assert_coeffs_close(coeffs, {"Z5": -2.0})


def test_combined_z4_to_z11_coefficients_recover_within_tolerance() -> None:
    mask = make_pupil_mask((56, 56))
    expected = {f"Z{i}": float(i - 7) * 0.75 for i in range(4, 12)}
    opd = reconstruct_zernike_opd_nm(expected, mask.shape, mask=mask)

    coeffs = fit_zernike_coefficients_nm(opd, mask, range(4, 12))

    _assert_coeffs_close(coeffs, expected)


def test_meter_input_returns_nanometre_coefficients() -> None:
    mask = make_pupil_mask((48, 48))
    opd_nm = reconstruct_zernike_opd_nm({"Z4": 5.0, "Z5": -2.0}, mask.shape, mask=mask)

    coeffs = fit_zernike_coefficients_nm(opd_nm * 1.0e-9, mask, [4, 5], input_unit="m")

    _assert_coeffs_close(coeffs, {"Z4": 5.0, "Z5": -2.0})


def test_residual_projection_is_near_zero_after_subtracting_fit() -> None:
    mask = make_pupil_mask((56, 56))
    noll = list(range(4, 12))
    expected = {f"Z{i}": float(i - 7) * 0.5 for i in noll}
    opd = reconstruct_zernike_opd_nm(expected, mask.shape, mask=mask)
    coeffs = fit_zernike_coefficients_nm(opd, mask, noll)
    residual = opd - reconstruct_zernike_opd_nm(coeffs, mask.shape, mask=mask)

    leakage = fit_zernike_coefficients_nm(residual, mask, noll)

    assert max(abs(value) for value in leakage.values()) < 1.0e-10
    assert float(np.sqrt(np.mean(residual[mask] ** 2))) < 1.0e-10
