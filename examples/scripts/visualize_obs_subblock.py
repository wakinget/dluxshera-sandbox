"""Quick-look visualizer for observation sub-block artifacts."""

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
        description="Generate quick-look artifacts for an observation sub-block cube."
    )
    parser.add_argument("--cube", type=Path, required=True, help="Path to *_cube.fits")
    parser.add_argument("--trace", type=Path, default=None, help="Optional path to frame-truth CSV")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional path to manifest.json")
    parser.add_argument("--outdir", type=Path, default=None, help="Optional output directory override")

    parser.add_argument("--no-gif", action="store_true", help="Disable preview GIF generation")
    parser.add_argument("--mp4", action="store_true", help="Also attempt MP4 export")
    parser.add_argument("--no-summary", action="store_true", help="Disable static summary panel")
    parser.add_argument(
        "--no-trace-summary",
        action="store_true",
        help="Disable trace summary figure generation",
    )

    parser.add_argument("--stride", type=int, default=1, help="Frame stride for animation outputs")
    parser.add_argument("--stretch", choices=["linear", "sqrt", "log"], default="linear")
    parser.add_argument("--pmin", type=float, default=1.0, help="Lower percentile for global scaling")
    parser.add_argument("--pmax", type=float, default=99.0, help="Upper percentile for global scaling")
    parser.add_argument("--fps", type=int, default=10, help="Animation frames-per-second")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    cube_path = args.cube.resolve()
    if not cube_path.exists():
        raise FileNotFoundError(f"Cube FITS not found: {cube_path}")

    manifest_path = args.manifest.resolve() if args.manifest is not None else None
    manifest = _load_manifest(manifest_path)

    trace_path = _infer_trace_path(
        trace_path=args.trace,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    trace = load_obs_subblock_trace_csv(trace_path) if trace_path is not None else None

    outdir = args.outdir.resolve() if args.outdir is not None else cube_path.parent / DEFAULT_OUTDIR_NAME
    outdir.mkdir(parents=True, exist_ok=True)

    with fits.open(cube_path) as hdul:
        cube = np.asarray(hdul[0].data)

    title_prefix = _title_prefix(manifest=manifest, cube_shape=cube.shape, trace=trace)

    if not args.no_gif:
        gif_path = outdir / "preview.gif"
        write_obs_subblock_preview_gif(
            cube,
            output_path=gif_path,
            trace=trace,
            stride=args.stride,
            pmin=args.pmin,
            pmax=args.pmax,
            stretch=args.stretch,
            fps=args.fps,
        )
        print(f"Wrote: {gif_path}")

    if args.mp4:
        mp4_path = outdir / "preview.mp4"
        try:
            write_obs_subblock_preview_mp4(
                cube,
                output_path=mp4_path,
                trace=trace,
                stride=args.stride,
                pmin=args.pmin,
                pmax=args.pmax,
                stretch=args.stretch,
                fps=args.fps,
            )
            print(f"Wrote: {mp4_path}")
        except RuntimeError as exc:
            print(f"Skipping MP4 export: {exc}")

    if not args.no_summary:
        fig, _ = make_obs_subblock_summary_figure(
            cube,
            pmin=args.pmin,
            pmax=args.pmax,
            stretch=args.stretch,
            title=title_prefix,
        )
        summary_path = outdir / "summary.png"
        fig.savefig(summary_path, dpi=180)
        plt.close(fig)
        print(f"Wrote: {summary_path}")

    if trace is not None and not args.no_trace_summary:
        fig, _ = make_obs_subblock_trace_summary_figure(trace, title=title_prefix)
        trace_summary_path = outdir / "trace_summary.png"
        fig.savefig(trace_summary_path, dpi=180)
        plt.close(fig)
        print(f"Wrote: {trace_summary_path}")


if __name__ == "__main__":
    main()
