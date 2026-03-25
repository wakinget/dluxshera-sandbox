"""Observation sub-block quick-look plotting and preview helpers.

These helpers are intentionally lightweight and focus on inspection of rendered
observation sub-block artifacts (FITS cubes + optional trace metadata).

The module keeps rendering concerns out of recipe generation code, and follows
repo plotting policy for reusable plotting helpers:

- figure/axes are accepted optionally and created only when omitted,
- figures are returned for caller-side saving/customization,
- ``plt.show()`` is never called implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw

from dluxshera.utils.obs_subblock_trace import ObsSubblockTrace

__all__ = [
    "apply_intensity_stretch",
    "compute_cube_display_limits",
    "make_obs_subblock_summary_figure",
    "make_obs_subblock_trace_summary_figure",
    "preview_frame_indices",
    "write_obs_subblock_preview_gif",
    "write_obs_subblock_preview_mp4",
]


def compute_cube_display_limits(
    cube: np.ndarray,
    *,
    pmin: float = 1.0,
    pmax: float = 99.0,
) -> tuple[float, float]:
    """Return robust global display limits for an image cube.

    Parameters
    ----------
    cube : np.ndarray
        Frame cube with shape ``(n_frame, ny, nx)``.
    pmin, pmax : float
        Percentile bounds used for robust global scaling.

    Returns
    -------
    tuple[float, float]
        ``(vmin, vmax)`` display limits.
    """

    if cube.ndim != 3:
        raise ValueError(f"Expected cube with shape (n_frame, ny, nx), got {cube.shape}.")
    if not np.isfinite(cube).any():
        raise ValueError("Cube contains no finite values.")
    if not (0.0 <= pmin < pmax <= 100.0):
        raise ValueError("Percentiles must satisfy 0 <= pmin < pmax <= 100.")

    finite_values = cube[np.isfinite(cube)]
    vmin = float(np.percentile(finite_values, pmin))
    vmax = float(np.percentile(finite_values, pmax))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Computed display limits are not finite.")
    if vmax <= vmin:
        vmax = vmin + 1e-12

    return vmin, vmax


def apply_intensity_stretch(
    image: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    stretch: str = "linear",
) -> np.ndarray:
    """Scale an image into ``[0, 1]`` using a configured intensity stretch.

    ``linear`` and ``sqrt`` support general numeric data. ``log`` is limited to
    non-negative display ranges and raises a ``ValueError`` otherwise.
    """

    if vmax <= vmin:
        raise ValueError("vmax must be larger than vmin.")

    scaled = (np.asarray(image, dtype=float) - vmin) / (vmax - vmin)
    clipped = np.clip(scaled, 0.0, 1.0)

    mode = stretch.lower()
    if mode == "linear":
        return clipped
    if mode == "sqrt":
        return np.sqrt(clipped)
    if mode == "log":
        if vmin < 0:
            raise ValueError(
                "log stretch requires non-negative display range (vmin >= 0)."
            )
        # Stable map from [0, 1] -> [0, 1] with stronger low-end contrast.
        return np.log10(1.0 + 9.0 * clipped) / np.log10(10.0)

    raise ValueError(f"Unknown stretch '{stretch}'. Use 'linear', 'sqrt', or 'log'.")


def preview_frame_indices(n_frame: int, *, stride: int = 1) -> np.ndarray:
    """Return frame indices used for preview products."""

    if n_frame <= 0:
        raise ValueError("n_frame must be positive.")
    if stride <= 0:
        raise ValueError("stride must be a positive integer.")
    return np.arange(0, n_frame, stride, dtype=int)


def _trace_rows(trace: ObsSubblockTrace | Sequence[Mapping[str, float]] | None):
    if trace is None:
        return None
    if isinstance(trace, ObsSubblockTrace):
        return trace.rows
    return tuple(trace)


def _preview_text(
    *,
    frame_index: int,
    trace_rows: Sequence[Mapping[str, float]] | None,
) -> str:
    lines = [f"frame={frame_index}"]
    if trace_rows is None:
        return " | ".join(lines)

    row = trace_rows[frame_index]
    if "time_s" in row:
        lines.append(f"t={float(row['time_s']):.3f}s")
    if "source.x_position_as" in row and "source.y_position_as" in row:
        lines.append(
            "x={:.5f} as, y={:.5f} as".format(
                float(row["source.x_position_as"]),
                float(row["source.y_position_as"]),
            )
        )
    if "source.position_angle_deg" in row:
        lines.append(f"PA={float(row['source.position_angle_deg']):.3f} deg")

    return " | ".join(lines)


def _preview_rgb_frames(
    cube: np.ndarray,
    *,
    trace: ObsSubblockTrace | Sequence[Mapping[str, float]] | None = None,
    stride: int = 1,
    pmin: float = 1.0,
    pmax: float = 99.0,
    stretch: str = "linear",
    cmap: str = "inferno",
) -> list[np.ndarray]:
    indices = preview_frame_indices(cube.shape[0], stride=stride)
    vmin, vmax = compute_cube_display_limits(cube, pmin=pmin, pmax=pmax)
    cmap_obj = matplotlib.colormaps.get_cmap(cmap)
    rows = _trace_rows(trace)

    frames: list[np.ndarray] = []
    for frame_index in indices:
        stretched = apply_intensity_stretch(
            cube[frame_index],
            vmin=vmin,
            vmax=vmax,
            stretch=stretch,
        )
        rgba = cmap_obj(stretched)
        rgb = np.asarray((255.0 * rgba[..., :3]).astype(np.uint8))

        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.rectangle([(0, 0), (image.width, 18)], fill=(0, 0, 0))
        draw.text((4, 3), _preview_text(frame_index=frame_index, trace_rows=rows), fill=(255, 255, 255))
        frames.append(np.asarray(image))

    return frames


def write_obs_subblock_preview_gif(
    cube: np.ndarray,
    *,
    output_path: Path,
    trace: ObsSubblockTrace | Sequence[Mapping[str, float]] | None = None,
    stride: int = 1,
    pmin: float = 1.0,
    pmax: float = 99.0,
    stretch: str = "linear",
    cmap: str = "inferno",
    fps: int = 10,
) -> Path:
    """Write a quick-look GIF preview for a cube."""

    frames = _preview_rgb_frames(
        cube,
        trace=trace,
        stride=stride,
        pmin=pmin,
        pmax=pmax,
        stretch=stretch,
        cmap=cmap,
    )
    images = [Image.fromarray(frame) for frame in frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=max(int(round(1000.0 / max(fps, 1))), 1),
        loop=0,
    )
    return output_path


def write_obs_subblock_preview_mp4(
    cube: np.ndarray,
    *,
    output_path: Path,
    trace: ObsSubblockTrace | Sequence[Mapping[str, float]] | None = None,
    stride: int = 1,
    pmin: float = 1.0,
    pmax: float = 99.0,
    stretch: str = "linear",
    cmap: str = "inferno",
    fps: int = 10,
) -> Path:
    """Write a quick-look MP4 preview when ``imageio``/FFmpeg are available."""

    try:
        import imageio.v3 as iio
    except Exception as exc:  # pragma: no cover - dependency is optional.
        raise RuntimeError("MP4 export requires imageio[ffmpeg].") from exc

    frames = _preview_rgb_frames(
        cube,
        trace=trace,
        stride=stride,
        pmin=pmin,
        pmax=pmax,
        stretch=stretch,
        cmap=cmap,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, np.stack(frames, axis=0), fps=fps)
    return output_path


def make_obs_subblock_summary_figure(
    cube: np.ndarray,
    *,
    pmin: float = 1.0,
    pmax: float = 99.0,
    stretch: str = "linear",
    cmap: str = "inferno",
    fig=None,
    axes=None,
    title: str | None = None,
):
    """Build a 2x3 static summary panel for a sub-block cube."""

    if cube.ndim != 3:
        raise ValueError(f"Expected cube with shape (n_frame, ny, nx), got {cube.shape}.")

    if axes is None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    else:
        fig = axes[0][0].figure if fig is None else fig

    idx_mid = cube.shape[0] // 2
    panels = [
        ("first frame", cube[0]),
        ("middle frame", cube[idx_mid]),
        ("last frame", cube[-1]),
        ("mean image", np.mean(cube, axis=0)),
        ("std image", np.std(cube, axis=0)),
        ("max-min image", np.max(cube, axis=0) - np.min(cube, axis=0)),
    ]

    base_vmin, base_vmax = compute_cube_display_limits(cube, pmin=pmin, pmax=pmax)
    norm = Normalize(vmin=0.0, vmax=1.0)

    for ax, (panel_title, panel_data) in zip(axes.ravel(), panels):
        if panel_title in {"std image", "max-min image"}:
            finite_data = panel_data[np.isfinite(panel_data)]
            pvmin = float(np.percentile(finite_data, pmin))
            pvmax = float(np.percentile(finite_data, pmax))
            if pvmax <= pvmin:
                pvmax = pvmin + 1e-12
        else:
            pvmin, pvmax = base_vmin, base_vmax

        stretched = apply_intensity_stretch(
            panel_data,
            vmin=pvmin,
            vmax=pvmax,
            stretch=stretch,
        )
        im = ax.imshow(stretched, cmap=cmap, norm=norm)
        ax.set_title(panel_title)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def make_obs_subblock_trace_summary_figure(
    trace: ObsSubblockTrace | Sequence[Mapping[str, float]],
    *,
    fig=None,
    axes=None,
    title: str | None = None,
):
    """Build a 2x2 trace-summary panel for observation-subblock frame truth."""

    rows = _trace_rows(trace)
    if rows is None or len(rows) == 0:
        raise ValueError("Trace summary requires at least one row.")

    times = np.asarray([float(row["time_s"]) for row in rows], dtype=float)
    x = np.asarray([float(row["source.x_position_as"]) for row in rows], dtype=float)
    y = np.asarray([float(row["source.y_position_as"]) for row in rows], dtype=float)
    pa = np.asarray([float(row["source.position_angle_deg"]) for row in rows], dtype=float)

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
    else:
        fig = axes[0][0].figure if fig is None else fig

    ax_xt, ax_yt, ax_pat, ax_xy = axes.ravel()
    ax_xt.plot(times, x, color="tab:blue")
    ax_xt.set_title("source.x_position_as vs time")
    ax_xt.set_xlabel("time (s)")
    ax_xt.set_ylabel("x (arcsec)")

    ax_yt.plot(times, y, color="tab:orange")
    ax_yt.set_title("source.y_position_as vs time")
    ax_yt.set_xlabel("time (s)")
    ax_yt.set_ylabel("y (arcsec)")

    ax_pat.plot(times, pa, color="tab:green")
    ax_pat.set_title("source.position_angle_deg vs time")
    ax_pat.set_xlabel("time (s)")
    ax_pat.set_ylabel("PA (deg)")

    scatter = ax_xy.scatter(x, y, c=times, cmap="viridis", s=20)
    ax_xy.plot(x, y, color="0.6", linewidth=1.0, alpha=0.8)
    ax_xy.scatter([x[0]], [y[0]], marker="o", color="lime", edgecolor="k", s=60, label="start")
    ax_xy.scatter([x[-1]], [y[-1]], marker="X", color="red", edgecolor="k", s=70, label="end")
    ax_xy.set_title("(x, y) path colored by time")
    ax_xy.set_xlabel("x (arcsec)")
    ax_xy.set_ylabel("y (arcsec)")
    ax_xy.legend(loc="best")
    fig.colorbar(scatter, ax=ax_xy, fraction=0.046, pad=0.04, label="time (s)")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
