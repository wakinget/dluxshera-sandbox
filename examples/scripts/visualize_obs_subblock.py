"""Generate quick-look visualizations for an observation sub-block cube.

This script reads an observation cube FITS file (typically ``*_cube.fits``) and
produces visual diagnostics to quickly sanity-check simulated or measured data.

What this includes
------------------
By default, outputs are written under ``<cube-parent>/quicklook/``:

- ``preview.gif``: animated frame preview (unless ``--no-gif``)
- ``summary.png``: static cube summary panel (unless ``--no-summary``)
- ``trace_summary.png``: trace-only summary plot (when trace is available and
  ``--no-trace-summary`` is not set)
- ``preview.mp4``: optional MP4 animation (only when ``--mp4`` is used)

Trace and manifest handling
---------------------------
- ``--trace`` explicitly provides the frame-truth CSV.
- If ``--manifest`` is omitted, the script looks for ``manifest.json`` beside
  the cube.
- If ``--trace`` is omitted and a manifest is available, trace discovery is:
  1. ``manifest["artifacts"]["frame_truth_csv"]``
  2. ``manifest["trace"]["path"]``
- Manifest metadata is also used to enrich figure titles (for example,
  generator and system preset).

Usage examples
--------------
Basic quick-look (GIF + summary PNG):

    python examples/scripts/visualize_obs_subblock.py \\
        --cube Results/run_001/observation_cube.fits

Use manifest-assisted trace discovery:

    python examples/scripts/visualize_obs_subblock.py \\
        --cube Results/run_001/observation_cube.fits \\
        --manifest Results/run_001/manifest.json

Explicit trace CSV, custom output directory, and MP4 export:

    python examples/scripts/visualize_obs_subblock.py \\
        --cube Results/run_001/observation_cube.fits \\
        --trace Results/run_001/frame_truth.csv \\
        --outdir Results/run_001/quicklook_custom \\
        --mp4 --stride 2 --stretch sqrt --pmin 0.5 --pmax 99.5 --fps 12

CLI options
-----------
- ``--cube PATH``: required cube FITS input.
- ``--trace PATH``: optional frame-truth CSV input.
- ``--manifest PATH``: optional manifest JSON for metadata and trace inference.
- ``--outdir PATH``: optional output directory override.
- ``--no-gif``: skip ``preview.gif``.
- ``--mp4``: also attempt ``preview.mp4`` export.
- ``--no-summary``: skip ``summary.png``.
- ``--no-trace-summary``: skip ``trace_summary.png``.
- ``--stride N``: use every Nth frame for animations.
- ``--stretch {linear,sqrt,log}``: display stretch used for rendering.
- ``--pmin VALUE``: lower display percentile.
- ``--pmax VALUE``: upper display percentile.
- ``--fps N``: animation frame rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from dluxshera.plot.obs_subblock import (
    make_obs_subblock_summary_figure,
    make_obs_subblock_trace_summary_figure,
    write_obs_subblock_preview_gif,
    write_obs_subblock_preview_mp4,
)
from dluxshera.utils.obs_subblock_trace import ObsSubblockTrace, load_obs_subblock_trace_csv


DEFAULT_OUTDIR_NAME = "quicklook"


class _CliHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Keep multiline epilog formatting and include default values in help."""


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Manifest must decode to a JSON object.")
    return payload


def _resolve_manifest_artifact(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    artifact_key: str,
) -> Path | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(artifact_key)
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _infer_trace_path(
    *,
    trace_path: Path | None,
    manifest: dict[str, Any] | None,
    manifest_path: Path | None,
) -> Path | None:
    if trace_path is not None:
        return trace_path.resolve()
    if manifest is None or manifest_path is None:
        return None

    candidate = _resolve_manifest_artifact(
        manifest,
        manifest_path=manifest_path,
        artifact_key="frame_truth_csv",
    )
    if candidate is not None and candidate.exists():
        return candidate

    trace_payload = manifest.get("trace")
    if isinstance(trace_payload, dict):
        trace_value = trace_payload.get("path")
        if isinstance(trace_value, str) and trace_value.strip():
            trace_candidate = Path(trace_value)
            if not trace_candidate.is_absolute():
                trace_candidate = (manifest_path.parent / trace_candidate).resolve()
            if trace_candidate.exists():
                return trace_candidate

    return None


def _title_prefix(
    *,
    manifest: dict[str, Any] | None,
    cube_shape: tuple[int, ...],
    trace: ObsSubblockTrace | None,
) -> str:
    parts = [f"frames={cube_shape[0]}"]
    if trace is not None and trace.time_start_s is not None and trace.time_stop_s is not None:
        parts.append(f"t=[{trace.time_start_s:.3f}, {trace.time_stop_s:.3f}] s")

    if manifest is not None:
        generator = manifest.get("generator")
        if isinstance(generator, str) and generator.strip():
            parts.append(f"generator={generator}")
        system = manifest.get("system")
        if isinstance(system, dict):
            preset = system.get("preset")
            if isinstance(preset, str) and preset.strip():
                parts.append(f"preset={preset}")

    return " | ".join(parts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate quick-look artifacts for an observation sub-block cube.",
        formatter_class=_CliHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/scripts/visualize_obs_subblock.py --cube Results/run/cube.fits\n"
            "  python examples/scripts/visualize_obs_subblock.py --cube Results/run/cube.fits --manifest Results/run/manifest.json\n"
            "  python examples/scripts/visualize_obs_subblock.py --cube Results/run/cube.fits --trace Results/run/frame_truth.csv --mp4\n\n"
            "Manifest resolution order:\n"
            "  1) --manifest\n"
            "  2) sibling manifest.json beside --cube\n\n"
            "Trace resolution order:\n"
            "  1) --trace\n"
            "  2) manifest artifacts.frame_truth_csv\n"
            "  3) manifest trace.path\n\n"
            f"Default output directory:\n  <cube-parent>/{DEFAULT_OUTDIR_NAME}\n\n"
            "Generated files:\n"
            "  preview.gif        (unless --no-gif)\n"
            "  summary.png        (unless --no-summary)\n"
            "  trace_summary.png  (if trace is available and --no-trace-summary is not set)\n"
            "  preview.mp4        (only when --mp4 is requested)"
        ),
    )
    parser.add_argument(
        "--cube",
        type=Path,
        required=True,
        help="Path to observation cube FITS file (typically *_cube.fits).",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Optional frame-truth CSV path. If omitted, inferred from the resolved manifest when possible.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest JSON used for trace discovery and title metadata. Defaults to sibling manifest.json beside --cube when present.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=f"Output directory. Defaults to <cube-parent>/{DEFAULT_OUTDIR_NAME}.",
    )

    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Skip preview GIF generation (preview.gif).",
    )
    parser.add_argument(
        "--mp4",
        action="store_true",
        help="Also attempt MP4 export (preview.mp4); depends on animation backend support.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip static summary figure generation (summary.png).",
    )
    parser.add_argument(
        "--no-trace-summary",
        action="store_true",
        help="Skip trace summary figure generation (trace_summary.png).",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame for animation outputs.",
    )
    parser.add_argument(
        "--stretch",
        choices=["linear", "sqrt", "log"],
        default="linear",
        help="Intensity stretch used for rendering.",
    )
    parser.add_argument(
        "--pmin",
        type=float,
        default=1.0,
        help="Lower percentile for global display scaling.",
    )
    parser.add_argument(
        "--pmax",
        type=float,
        default=99.0,
        help="Upper percentile for global display scaling.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Animation frame rate in frames-per-second.",
    )
    return parser


def generate_obs_subblock_quicklook(
    *,
    cube_path: Path,
    trace_path: Path | None = None,
    manifest_path: Path | None = None,
    outdir: Path | None = None,
    write_gif: bool = True,
    write_mp4: bool = False,
    write_summary: bool = True,
    write_trace_summary: bool = True,
    stride: int = 1,
    stretch: str = "linear",
    pmin: float = 1.0,
    pmax: float = 99.0,
    fps: int = 10,
) -> dict[str, Any]:
    """Generate quick-look artifacts for one observation sub-block cube."""

    cube_path = cube_path.resolve()
    if not cube_path.exists():
        raise FileNotFoundError(f"Cube FITS not found: {cube_path}")

    manifest_path = manifest_path.resolve() if manifest_path is not None else None
    if manifest_path is None:
        sibling_manifest = cube_path.parent / "manifest.json"
        if sibling_manifest.exists():
            manifest_path = sibling_manifest.resolve()
    manifest = _load_manifest(manifest_path)

    trace_path = _infer_trace_path(
        trace_path=trace_path.resolve() if trace_path is not None else None,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    trace_varying_keys: list[str] | None = None
    if manifest is not None:
        varying_keys_value = (
            manifest.get("applied_varying_keys")
            if manifest.get("applied_varying_keys") is not None
            else manifest.get("varying_keys")
        )
        if isinstance(varying_keys_value, list) and all(
            isinstance(item, str) for item in varying_keys_value
        ):
            trace_varying_keys = list(varying_keys_value)
    trace = (
        load_obs_subblock_trace_csv(
            trace_path,
            required_varying_keys=trace_varying_keys,
        )
        if trace_path is not None
        else None
    )

    outdir = outdir.resolve() if outdir is not None else cube_path.parent / DEFAULT_OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data)

    title_prefix = _title_prefix(manifest=manifest, cube_shape=cube.shape, trace=trace)
    artifacts: dict[str, str] = {}

    if write_gif:
        gif_path = outdir / "preview.gif"
        write_obs_subblock_preview_gif(
            cube,
            output_path=gif_path,
            trace=trace,
            stride=stride,
            pmin=pmin,
            pmax=pmax,
            stretch=stretch,
            fps=fps,
        )
        print(f"Wrote: {gif_path}")
        artifacts["preview_gif"] = str(gif_path)

    if write_mp4:
        mp4_path = outdir / "preview.mp4"
        try:
            write_obs_subblock_preview_mp4(
                cube,
                output_path=mp4_path,
                trace=trace,
                stride=stride,
                pmin=pmin,
                pmax=pmax,
                stretch=stretch,
                fps=fps,
            )
            print(f"Wrote: {mp4_path}")
            artifacts["preview_mp4"] = str(mp4_path)
        except RuntimeError as exc:
            print(f"Skipping MP4 export: {exc}")

    if write_summary:
        fig, _ = make_obs_subblock_summary_figure(
            cube,
            pmin=pmin,
            pmax=pmax,
            stretch=stretch,
            title=title_prefix,
        )
        summary_path = outdir / "summary.png"
        fig.savefig(summary_path, dpi=180)
        plt.close(fig)
        print(f"Wrote: {summary_path}")
        artifacts["summary_png"] = str(summary_path)

    if trace is not None and write_trace_summary:
        fig, _ = make_obs_subblock_trace_summary_figure(trace, title=title_prefix)
        trace_summary_path = outdir / "trace_summary.png"
        fig.savefig(trace_summary_path, dpi=180)
        plt.close(fig)
        print(f"Wrote: {trace_summary_path}")
        artifacts["trace_summary_png"] = str(trace_summary_path)

    return {
        "cube_path": str(cube_path),
        "manifest_path": None if manifest_path is None else str(manifest_path),
        "trace_path": None if trace_path is None else str(trace_path),
        "output_dir": str(outdir),
        "artifacts": artifacts,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)

    return generate_obs_subblock_quicklook(
        cube_path=args.cube,
        trace_path=args.trace,
        manifest_path=args.manifest,
        outdir=args.outdir,
        write_gif=not bool(args.no_gif),
        write_mp4=bool(args.mp4),
        write_summary=not bool(args.no_summary),
        write_trace_summary=not bool(args.no_trace_summary),
        stride=int(args.stride),
        stretch=str(args.stretch),
        pmin=float(args.pmin),
        pmax=float(args.pmax),
        fps=int(args.fps),
    )


if __name__ == "__main__":
    main()
