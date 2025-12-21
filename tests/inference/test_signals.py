import numpy as np

from dluxshera.inference.signals import build_signals
from dluxshera.params.store import ParameterStore
from dluxshera.params.transforms import TRANSFORMS


def test_binary_raw_flux_transform_matches_alpha_cen_formula():
    store = ParameterStore.from_dict(
        {
            "binary.log_flux_total": 8.0,
            "binary.contrast": 3.37,
        }
    )

    fluxes = TRANSFORMS.compute("binary.raw_fluxes", store)

    total_flux = 10 ** 8.0
    expected_b = total_flux / (1 + 3.37)
    expected_a = 3.37 * expected_b

    np.testing.assert_allclose(fluxes, np.array([expected_a, expected_b]))


def test_build_intro_signals_scaling_and_shapes():
    T = 5
    D = 3
    theta = np.linspace(0.0, 0.4, T * D).reshape(T, D)
    trace = {"theta": theta, "loss": np.linspace(0.0, 1.0, T)}

    truth = {
        "binary.x_position_as": np.zeros(T),
        "binary.y_position_as": np.zeros(T),
        "binary.separation_as": np.ones(T),
        "system.plate_scale_as_per_pix": np.full(T, 0.1),
        "primary.zernike_coeffs_nm": np.array([0.05, 0.1, 0.15]),
        "binary.raw_fluxes": np.array([10.0, 5.0]),
    }

    def decoder(theta_row: np.ndarray):
        return {
            "binary.x_position_as": theta_row[0],
            "binary.y_position_as": -theta_row[0],
            "binary.separation_as": 1.0 + theta_row[1],
            "system.plate_scale_as_per_pix": 0.1 + 0.01 * theta_row[2],
            "primary.zernike_coeffs_nm": theta_row + 0.1,
            "binary.raw_fluxes": np.array([10.0, 5.0]) + theta_row[0],
        }

    signals = build_signals(trace, meta={}, decoder=decoder, truth=truth)

    assert signals["binary.x_error_uas"].shape == (T,)
    assert signals["binary.y_error_uas"].shape == (T,)
    assert signals["binary.separation_error_uas"].shape == (T,)
    assert signals["system.plate_scale_error_ppm"].shape == (T,)
    assert signals["binary.raw_flux_error_ppm"].shape == (T, 2)
    assert signals["primary.zernike_error_nm"].shape == (T, 3)
    assert signals["primary.zernike_rms_nm"].shape == (T,)

    expected_x_error = theta[:, 0] * 1e6
    expected_y_error = -theta[:, 0] * 1e6
    expected_sep_error = theta[:, 1] * 1e6
    expected_plate_scale_error = 1e6 * (0.01 * theta[:, 2]) / 0.1
    expected_flux_error = 1e6 * (theta[:, 0][:, None] / np.array([10.0, 5.0]))
    expected_zern_error = (theta + 0.1) - truth["primary.zernike_coeffs_nm"]
    expected_zern_rms = np.sqrt(np.mean(expected_zern_error**2, axis=-1))

    np.testing.assert_allclose(signals["binary.x_error_uas"], expected_x_error)
    np.testing.assert_allclose(signals["binary.y_error_uas"], expected_y_error)
    np.testing.assert_allclose(signals["binary.separation_error_uas"], expected_sep_error)
    np.testing.assert_allclose(
        signals["system.plate_scale_error_ppm"], expected_plate_scale_error
    )
    np.testing.assert_allclose(signals["binary.raw_flux_error_ppm"], expected_flux_error)
    np.testing.assert_allclose(signals["primary.zernike_error_nm"], expected_zern_error)
    np.testing.assert_allclose(signals["primary.zernike_rms_nm"], expected_zern_rms)


def test_build_signals_handles_missing_truth_with_nans():
    theta = np.zeros((3, 2))
    trace = {"theta": theta, "loss": np.zeros(3)}

    def decoder(_theta):
        return {
            "binary.x_position_as": 0.0,
            "binary.y_position_as": 0.0,
            "binary.separation_as": 1.0,
            "system.plate_scale_as_per_pix": 0.1,
            "primary.zernike_coeffs_nm": np.zeros(2),
            "binary.raw_fluxes": np.ones(2),
        }

    signals = build_signals(trace, meta={}, decoder=decoder, truth=None)

    assert np.isnan(signals["binary.x_error_uas"]).all()
    assert np.isnan(signals["binary.raw_flux_error_ppm"]).all()
    assert np.isnan(signals["primary.zernike_rms_nm"]).all()
