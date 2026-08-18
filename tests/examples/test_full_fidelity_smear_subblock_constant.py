from __future__ import annotations

import numpy as np
import pytest

from dluxshera.utils.obs_subblock_trajectory import SubblockTrajectory
from dluxshera.utils.trajectory_smear import (
    SmearConfig,
    parse_smear_config,
    subblock_constant_line_kernel_from_fit,
)


def _block(slope_x: float, slope_y: float) -> SubblockTrajectory:
    times = np.array([0.0, 1.0, 2.0])
    return SubblockTrajectory(
        subblock_index=0,
        frame_times_s=times,
        time_relative_s=times,
        truth={},
        prediction={},
        residual={},
        fit_coefficients={
            "source.x_position_as": (0.0, slope_x),
            "source.y_position_as": (0.0, slope_y),
        },
        diagnostics={},
    )


def _cfg() -> SmearConfig:
    return SmearConfig(
        enabled=True,
        exposure_time_s=0.05,
        plate_scale_as_per_pix=0.01,
        truth_sigma_perp_detector_pix=0.25,
        truth_kernel_size=9,
        render_mode="subblock_constant_layer",
    )


@pytest.mark.parametrize(
    ("slope_x", "slope_y", "theta"),
    [(2.0, 0.0, 0.0), (0.0, 2.0, 90.0), (2.0, 2.0, 45.0)],
)
def test_subblock_constant_kernel_uses_fit_slope_times_one_exposure(
    slope_x: float,
    slope_y: float,
    theta: float,
) -> None:
    kernel = subblock_constant_line_kernel_from_fit(_block(slope_x, slope_y), _cfg())

    assert kernel["length"] == pytest.approx(np.hypot(slope_x, slope_y) * 0.05 / 0.01)
    assert kernel["theta_deg"] == pytest.approx(theta)
    assert kernel["units"] == "detector_pix"
    assert kernel["source"] == "subblock_linear_fit_one_frame_exposure"


def test_per_frame_smear_mode_fails_as_future() -> None:
    with pytest.raises(ValueError, match="future/deferred"):
        parse_smear_config(
            {"smear": {"enabled": True, "render": {"mode": "per_frame"}}},
            exposure_time_s=0.05,
            plate_scale_as_per_pix=0.01,
        )
