import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from dluxshera.plot.plotting import (
    choose_subplot_grid,
    plot_parameter_history,
    plot_parameter_history_grid,
    plot_parameter_sweeps,
    plot_psf_comparison,
    plot_psf_single,
)


@pytest.fixture
def small_psf():
    x = np.linspace(-1, 1, 16)
    xx, yy = np.meshgrid(x, x)
    return np.exp(-(xx**2 + yy**2))


def test_plot_psf_single_returns_fig_ax(small_psf, tmp_path):
    fig, ax = plot_psf_single(small_psf, show=False, close=False)
    assert fig is not None
    assert ax is not None

    save_path = tmp_path / "psf_single.png"
    plot_psf_single(small_psf, save_path=save_path, show=False)
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_plot_psf_comparison_grid(small_psf):
    noisy = small_psf + 0.01
    fig, axes = plot_psf_comparison(noisy, small_psf, show=False, close=False)
    assert axes.shape == (2, 2)
    titles = [ax.get_title() for ax in axes.flat]
    assert "Residuals" in titles
    assert "Z-Score" in titles
    fig.clf()


def test_plot_parameter_history_basic(tmp_path):
    names = ["alpha", "beta"]
    histories = [np.arange(5), np.arange(5) * 2]
    fig, ax = plot_parameter_history(names, histories, true_vals=[0, 1], show=False, close=False)
    assert fig is not None
    assert ax is not None

    save_path = tmp_path / "history.png"
    plot_parameter_history(names, histories, save_path=save_path, show=False)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    fig.clf()


def test_plot_parameter_history_grid(tmp_path):
    histories = {"a": np.arange(4), "b": np.arange(4) + 1, "c": np.arange(4) + 2}
    fig, axes = plot_parameter_history_grid(histories, show=False, close=False)
    assert axes.shape[0] * axes.shape[1] >= len(histories)

    save_path = tmp_path / "grid.png"
    plot_parameter_history_grid(histories, save_path=save_path, show=False)
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    fig.clf()


def test_choose_subplot_grid_shapes():
    assert choose_subplot_grid(1) == (1, 1)
    assert choose_subplot_grid(2) == (2, 1)
    rows, cols = choose_subplot_grid(5)
    assert rows * cols >= 5


def test_plot_parameter_sweeps(tmp_path):
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pytest.skip("pandas not available")

    df = pd.DataFrame(
        {
            "parameter": ["x", "x", "y"],
            "index": [np.nan, np.nan, np.nan],
            "value": [0.0, 1.0, 0.5],
            "loss": [1.0, 0.5, 0.75],
        }
    )
    fig = plot_parameter_sweeps(df, show=False, save_path=tmp_path / "sweeps.png")
    assert fig is not None
    assert (tmp_path / "sweeps.png").exists()
    assert (tmp_path / "sweeps.png").stat().st_size > 0
