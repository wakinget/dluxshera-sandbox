from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dluxshera.plot.plotting import plot_signals_grid, plot_signals_panels


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


def test_plot_signals_grid_with_final_values_creates_png(tmp_path: Path):
    signals = {
        "source.x_error_uas": np.linspace(-1.0, 1.0, 5),
        "source.y_error_uas": np.linspace(1.0, -1.0, 5),
        "source.separation_error_uas": np.zeros(5),
        "optics.plate_scale_error_ppm": np.ones(5),
        "source.raw_flux_error_ppm": np.zeros((5, 2)),
        "optics.primary.zernike_error_nm": np.zeros((5, 3)),
    }

    fig, _ = plot_signals_grid(
        signals,
        tmp_path,
        show_final_values=True,
        show=False,
        close=False,
    )
    out = tmp_path / "signals_grid.png"

    ax0 = fig.axes[0]
    assert ax0.get_title() == "Astrometry residuals (µas)"
    legend = ax0.get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert any(label.startswith("Δx=") for label in legend_labels)
    assert any(label.startswith("Δy=") for label in legend_labels)

    assert fig is not None
    assert out.exists(), f"Missing plot: {out}"
    assert out.stat().st_size > 0, f"Plot is empty: {out}"
    plt.close(fig)
