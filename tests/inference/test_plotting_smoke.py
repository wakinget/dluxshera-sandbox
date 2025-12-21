from pathlib import Path

import numpy as np

from dluxshera.inference.plotting import plot_signals_panels


def test_plot_signals_panels_creates_pngs(tmp_path: Path):
    signals = {
        "binary.x_error_uas": np.linspace(-1.0, 1.0, 5),
        "binary.y_error_uas": np.linspace(1.0, -1.0, 5),
        "binary.separation_error_uas": np.zeros(5),
        "system.plate_scale_error_ppm": np.ones(5),
        "binary.raw_flux_error_ppm": np.zeros((5, 2)),
        "primary.zernike_rms_nm": np.linspace(0.0, 0.2, 5),
        "primary.zernike_error_nm": np.zeros((5, 3)),
    }

    paths = plot_signals_panels(signals, tmp_path, panel_set="intro", title_prefix="test")

    assert paths, "No plots were generated"
    for path in paths:
        assert path.exists(), f"Missing plot: {path}"
        assert path.stat().st_size > 0, f"Plot is empty: {path}"
