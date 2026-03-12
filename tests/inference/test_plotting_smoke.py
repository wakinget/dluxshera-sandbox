from pathlib import Path

import numpy as np

from dluxshera.plot.plotting import plot_signals_panels


def test_plot_signals_panels_creates_pngs(tmp_path: Path):
    signals = {
        "source.x_error_uas": np.linspace(-1.0, 1.0, 5),
        "source.y_error_uas": np.linspace(1.0, -1.0, 5),
        "source.separation_error_uas": np.zeros(5),
        "optics.plate_scale_error_ppm": np.ones(5),
        "source.raw_flux_error_ppm": np.zeros((5, 2)),
        "primary.zernike_rms_nm": np.linspace(0.0, 0.2, 5),
        "optics.primary.zernike_error_nm": np.zeros((5, 3)),
    }

    paths = plot_signals_panels(signals, tmp_path, title_prefix="test")

    assert paths, "No plots were generated"
    for path in paths:
        assert path.exists(), f"Missing plot: {path}"
        assert path.stat().st_size > 0, f"Plot is empty: {path}"
