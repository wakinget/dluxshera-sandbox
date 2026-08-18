from __future__ import annotations

import numpy as np

from dluxshera.utils.chi2_diagnostics import (
    reduced_chi2_between_images,
    summarize_framewise_chi2,
)


def test_summarize_framewise_chi2_known_cube():
    data_cube = np.array(
        [
            [[2.0, 4.0], [6.0, 8.0]],
            [[1.0, 3.0], [5.0, 7.0]],
        ],
        dtype=float,
    )
    model_cube = data_cube - 1.0
    variance_cube = np.full_like(data_cube, 2.0)

    summary = summarize_framewise_chi2(
        data_cube,
        model_cube,
        variance_cube=variance_cube,
    )

    np.testing.assert_allclose(summary.per_frame_chi2, [2.0, 2.0])
    np.testing.assert_allclose(summary.per_frame_reduced_chi2, [0.5, 0.5])
    np.testing.assert_array_equal(summary.per_frame_dof_pixels, [4, 4])
    assert summary.block_sum_chi2 == 4.0
    assert summary.block_reduced_chi2 == 0.5
    assert summary.block_mean_reduced_chi2 == 0.5
    assert summary.block_dof_pixels == 8


def test_reduced_chi2_between_images_matches_valid_pixels_only():
    data_image = np.array([[2.0, 4.0], [6.0, np.nan]], dtype=float)
    model_image = np.array([[1.0, 3.0], [5.0, 0.0]], dtype=float)
    variance_image = np.array([[2.0, 0.0], [2.0, 2.0]], dtype=float)

    reduced = reduced_chi2_between_images(
        data_image,
        model_image,
        variance_image=variance_image,
    )

    assert reduced == 0.5
