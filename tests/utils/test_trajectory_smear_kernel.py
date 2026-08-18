from __future__ import annotations

import pytest

from dluxshera.utils.obs_subblock_trajectory import SubblockTrajectory
from dluxshera.utils.trajectory_smear import SmearConfig, subblock_constant_line_kernel_from_fit


def _block(slope_x: float, slope_y: float) -> SubblockTrajectory:
    return SubblockTrajectory(
        subblock_index=0,
        frame_times_s=[],
        time_relative_s=[],
        truth={},
        prediction={},
        residual={},
        fit_coefficients={
            "source.x_position_as": (0.0, slope_x),
            "source.y_position_as": (0.0, slope_y),
        },
        diagnostics={},
    )


def _cfg(*, exposure_time_s: float = 0.05, plate_scale_as_per_pix: float = 0.01) -> SmearConfig:
    return SmearConfig(
        enabled=True,
        exposure_time_s=exposure_time_s,
        plate_scale_as_per_pix=plate_scale_as_per_pix,
        truth_sigma_perp_detector_pix=0.25,
        truth_kernel_size=9,
        render_mode="subblock_constant_layer",
    )


def test_subblock_constant_smear_length_scales_with_exposure_time() -> None:
    short = subblock_constant_line_kernel_from_fit(_block(2.0, 0.0), _cfg(exposure_time_s=0.05))
    long = subblock_constant_line_kernel_from_fit(_block(2.0, 0.0), _cfg(exposure_time_s=0.10))

    assert long["length"] == pytest.approx(2.0 * short["length"])


def test_subblock_constant_smear_length_divides_by_plate_scale() -> None:
    fine = subblock_constant_line_kernel_from_fit(_block(2.0, 0.0), _cfg(plate_scale_as_per_pix=0.01))
    coarse = subblock_constant_line_kernel_from_fit(_block(2.0, 0.0), _cfg(plate_scale_as_per_pix=0.02))

    assert coarse["length"] == pytest.approx(0.5 * fine["length"])


@pytest.mark.parametrize(
    ("slope_x", "slope_y", "theta"),
    [
        (1.0, 1.0, 45.0),
        (-1.0, 1.0, 135.0),
        (-1.0, -1.0, -135.0),
        (1.0, -1.0, -45.0),
    ],
)
def test_subblock_constant_smear_theta_uses_atan2_quadrant(
    slope_x: float,
    slope_y: float,
    theta: float,
) -> None:
    kernel = subblock_constant_line_kernel_from_fit(_block(slope_x, slope_y), _cfg())

    assert kernel["theta_deg"] == pytest.approx(theta)
