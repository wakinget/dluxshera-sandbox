from __future__ import annotations

"""Generate detector calibration maps (FITS primary format).

Supports two modes:
- baseline: zeros dx/dy, ones PRF, optional Gaussian noise
- realize-fpa: MATLAB-like tiled fixed_row/fixed_col offsets plus noise
"""

import argparse
from pathlib import Path

import numpy as np

from dluxshera.utils.calibration_maps import (
    generate_baseline_maps,
    realize_fpa_offsets,
    write_baseline_maps,
    write_realize_fpa_maps,
)


DEFAULT_OFFSETS_DIR = Path("src/dluxshera/data/pixel_offsets")
DEFAULT_PRF_DIR = Path("src/dluxshera/data/pixel_response")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detector calibration maps (FITS).")
    parser.add_argument("--mode", choices=["baseline", "realize-fpa"], default="baseline")

    shape = parser.add_argument_group("shape")
    shape.add_argument("--npix", type=int, help="Square detector size; overrides nrows/ncols if set.")
    shape.add_argument("--nrows", type=int, default=512, help="Number of detector rows (default: 512).")
    shape.add_argument("--ncols", type=int, default=512, help="Number of detector cols (default: 512).")

    noise = parser.add_argument_group("noise")
    noise.add_argument("--noise-amplitude", type=float, default=0.0, help="Gaussian noise std dev (default: 0).")
    noise.add_argument("--seed", type=int, help="Random seed for reproducibility.")

    parser.add_argument(
        "--offsets-dir",
        type=Path,
        default=DEFAULT_OFFSETS_DIR,
        help="Output directory for dx/dy maps (default: src/dluxshera/data/pixel_offsets).",
    )
    parser.add_argument(
        "--prf-dir",
        type=Path,
        default=DEFAULT_PRF_DIR,
        help="Output directory for PRF map (default: src/dluxshera/data/pixel_response).",
    )
    parser.add_argument("--basename", default=None, help="Base filename stem (default depends on mode).")
    parser.add_argument("--write-npy", action="store_true", help="Also write .npy sidecars.")

    realize = parser.add_argument_group("realize-fpa")
    realize.add_argument("--fixed-row", type=float, nargs="+", help="Fixed_row vector (length = ncols).")
    realize.add_argument("--fixed-col", type=float, nargs="+", help="Fixed_col vector (length = nrows).")
    realize.add_argument("--sig-offset", type=float, help="Std dev for iid offsets (required for realize-fpa).")
    realize.add_argument("--sig-diff", type=float, help="Optional sig_diff (metadata only).")

    return parser.parse_args()


def resolve_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.npix is not None:
        return args.npix, args.npix
    return args.nrows, args.ncols


def main() -> None:
    args = parse_args()
    nrows, ncols = resolve_shape(args)
    basename = args.basename or ("baseline" if args.mode == "baseline" else "realize_fpa")

    if args.mode == "baseline":
        dx, dy, prf = generate_baseline_maps(
            nrows,
            ncols,
            noise_amplitude=args.noise_amplitude,
            seed=args.seed,
        )

        dx_path, dy_path, prf_path = write_baseline_maps(
            dx,
            dy,
            prf,
            offsets_dir=args.offsets_dir,
            prf_dir=args.prf_dir,
            noise_amplitude=args.noise_amplitude,
            seed=args.seed,
            basename=basename,
            write_npy=bool(args.write_npy),
        )
        print(f"Wrote baseline maps:\n  dx: {dx_path}\n  dy: {dy_path}\n  prf: {prf_path}")
        return

    # realize-fpa mode
    if args.fixed_row is None or args.fixed_col is None or args.sig_offset is None:
        raise SystemExit("realize-fpa mode requires --fixed-row, --fixed-col, and --sig-offset")

    fixed_row = np.asarray(args.fixed_row, dtype=float)
    fixed_col = np.asarray(args.fixed_col, dtype=float)

    dx, dy = realize_fpa_offsets(
        nrows,
        ncols,
        fixed_row=fixed_row,
        fixed_col=fixed_col,
        sig_offset=args.sig_offset,
        sig_diff=args.sig_diff,
        seed=args.seed,
    )

    dx_path, dy_path = write_realize_fpa_maps(
        dx,
        dy,
        offsets_dir=args.offsets_dir,
        noise_amplitude=args.noise_amplitude,
        seed=args.seed,
        fixed_row=fixed_row,
        fixed_col=fixed_col,
        sig_offset=args.sig_offset,
        sig_diff=args.sig_diff,
        basename=basename,
        write_npy=bool(args.write_npy),
    )

    print(f"Wrote realize_fpa offsets:\n  dx: {dx_path}\n  dy: {dy_path}")


if __name__ == "__main__":
    main()

