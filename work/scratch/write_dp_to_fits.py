from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

from dluxshera.utils.utils import default_diffractive_pupil_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export the default dLuxShera diffractive pupil mask to a FITS file as phase (radians)."
        )
    )
    parser.add_argument(
        "--dp-path",
        type=str,
        default=None,
        help="Path to diffractive_pupil.npy (defaults to dluxshera default).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="Results/diffractive_pupil_phase.fits",
        help="Output FITS path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite output file if it already exists.",
    )
    return parser


def _relative_path(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def main() -> None:
    args = _build_parser().parse_args()

    dp_path = Path(args.dp_path) if args.dp_path else Path(default_diffractive_pupil_path())
    if not dp_path.exists():
        raise FileNotFoundError(f"Diffractive pupil file not found: {dp_path}")

    mask = np.load(dp_path)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={mask.shape}")

    phase = (np.asarray(mask, dtype=np.float64) * np.pi).astype(np.float64)

    repo_root = Path(__file__).resolve().parents[2]
    out_path = Path(args.out)

    header = fits.Header()
    header["BUNIT"] = ("rad", "Phase units")
    header["MIN"] = (float(np.nanmin(phase)), "Min phase value (rad)")
    header["MAX"] = (float(np.nanmax(phase)), "Max phase value (rad)")
    header["NY"] = (int(phase.shape[0]), "Array size (rows)")
    header["NX"] = (int(phase.shape[1]), "Array size (cols)")
    header["DP_SRC"] = (
        _relative_path(dp_path, repo_root),
        "Source file path",
    )
    header.add_comment("Original was stored as mask in [0,1].")
    header.add_comment("This FITS scales to rad: phase = mask * pi.")
    header.add_comment("For OPD: opd_m = phase / (2*pi) * lambda_m.")

    hdu = fits.PrimaryHDU(data=phase, header=header)
    hdu.writeto(out_path, overwrite=bool(args.overwrite))

    print("Wrote diffractive pupil FITS:")
    print(f"  input : {dp_path}")
    print(f"  output: {out_path}")
    print(f"  shape : {phase.shape}")
    print(f"  dtype : {phase.dtype}")
    print(f"  min/max (rad): {np.nanmin(phase)} / {np.nanmax(phase)}")


if __name__ == "__main__":
    main()
