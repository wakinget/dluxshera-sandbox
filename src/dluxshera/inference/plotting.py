from __future__ import annotations

"""
Lightweight plotting helpers for Signals.

Panels are saved as PNGs under ``<run_dir>/plots`` and are safe for headless
environments (matplotlib Agg backend).
"""

from pathlib import Path
from typing import Mapping, MutableSequence, Optional

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

PanelPaths = MutableSequence[Path]


def _ensure_plots_dir(out_dir: Path) -> Path:
    plots_dir = Path(out_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def _plot_lines(x, ys, labels, title: str, ylabel: str, save_path: Path) -> None:
    fig, ax = plt.subplots()
    for y, label in zip(ys, labels):
        ax.plot(x, y, label=label)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_signals_panels(
    signals: Mapping[str, np.ndarray],
    out_dir: Path,
    *,
    panel_set: str = "intro",
    title_prefix: Optional[str] = None,
) -> PanelPaths:
    """
    Render standard diagnostic panels from Signals.

    Parameters
    ----------
    signals:
        Mapping from signal names to numpy arrays.
    out_dir:
        Run directory; plots are written to ``out_dir / 'plots'``.
    panel_set:
        Panel recipe. Only ``"intro"`` is supported.
    title_prefix:
        Optional prefix applied to each panel title.

    Returns
    -------
    list[Path]
        Paths to saved PNG files.
    """

    if panel_set != "intro":
        raise ValueError(f"Unsupported panel_set {panel_set!r} (expected 'intro').")

    plots_dir = _ensure_plots_dir(Path(out_dir))
    x = np.arange(next(iter(signals.values())).shape[0])

    def title(base: str) -> str:
        return f"{title_prefix} — {base}" if title_prefix else base

    saved: PanelPaths = []

    # Panel 1: Astrometry residuals
    astrometry = [
        ("binary.x_error_uas", "Δx"),
        ("binary.y_error_uas", "Δy"),
    ]
    if all(key in signals for key, _ in astrometry):
        ys = [signals[key] for key, _ in astrometry]
        labels = [label for _, label in astrometry]
        path = plots_dir / "astrometry_residuals_uas.png"
        _plot_lines(x, ys, labels, title("Astrometry residuals (µas)"), "Residual (µas)", path)
        saved.append(path)

    # Panel 2: Separation
    if "binary.separation_error_uas" in signals:
        path = plots_dir / "separation_residual_uas.png"
        _plot_lines(
            x,
            [signals["binary.separation_error_uas"]],
            ["Δρ"],
            title("Separation residual (µas)"),
            "Residual (µas)",
            path,
        )
        saved.append(path)

    # Panel 3: Plate scale
    if "system.plate_scale_error_ppm" in signals:
        path = plots_dir / "plate_scale_error_ppm.png"
        _plot_lines(
            x,
            [signals["system.plate_scale_error_ppm"]],
            ["Plate scale"],
            title("Plate scale error (ppm)"),
            "Error (ppm)",
            path,
        )
        saved.append(path)

    # Panel 4: Raw flux error
    if "binary.raw_flux_error_ppm" in signals:
        flux_err = signals["binary.raw_flux_error_ppm"]
        if flux_err.ndim == 2 and flux_err.shape[1] >= 2:
            ys = [flux_err[:, 0], flux_err[:, 1]]
            labels = ["Star A", "Star B"]
        else:
            ys = [flux_err]
            labels = ["Raw flux"]
        path = plots_dir / "raw_flux_error_ppm.png"
        _plot_lines(
            x,
            ys,
            labels,
            title("Raw flux error (ppm)"),
            "Error (ppm)",
            path,
        )
        saved.append(path)

    # Panel 5: Zernike RMS
    if "primary.zernike_rms_nm" in signals:
        path = plots_dir / "zernike_rms_nm.png"
        _plot_lines(
            x,
            [signals["primary.zernike_rms_nm"]],
            ["Zernike RMS"],
            title("Zernike RMS (nm)"),
            "RMS (nm)",
            path,
        )
        saved.append(path)

    # Panel 6: Optional component residuals
    if "primary.zernike_error_nm" in signals:
        zerr = signals["primary.zernike_error_nm"]
        if zerr.ndim == 2 and zerr.shape[1] > 0:
            k = min(zerr.shape[1], 6)
            ys = [zerr[:, i] for i in range(k)]
            labels = [f"Z{i}" for i in range(k)]
            path = plots_dir / "zernike_components_nm.png"
            _plot_lines(
                x,
                ys,
                labels,
                title("Zernike component residuals (nm)"),
                "Residual (nm)",
                path,
            )
            saved.append(path)

    return saved


__all__ = ["plot_signals_panels"]
