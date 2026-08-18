from pathlib import Path

import numpy as np

from dluxshera.plot.plotting import plot_pixel_offset_maps, plot_pixel_response_maps


def test_plot_pixel_offset_maps_writes_png(tmp_path: Path):
    data_dx = np.zeros((4, 4))
    data_dy = np.ones((4, 4))
    infer_dx = np.full((4, 4), 0.5)
    infer_dy = np.full((4, 4), -0.5)

    save_path = tmp_path / "pixel_offset_maps.png"

    plot_pixel_offset_maps(
        data_dx,
        data_dy,
        infer_dx,
        infer_dy,
        save_path=save_path,
        show=False,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_pixel_response_maps_writes_png(tmp_path: Path):
    data_prf = np.ones((4, 4))
    infer_prf = np.full((4, 4), 0.8)

    save_path = tmp_path / "pixel_response_maps.png"

    plot_pixel_response_maps(
        data_prf,
        infer_prf,
        save_path=save_path,
        show=False,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
