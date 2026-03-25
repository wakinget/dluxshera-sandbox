#!/usr/bin/env python3
"""
Generate a static two-ROI pixel-response mask for binary-star studies.

This script resolves a system preset, reads source geometry from the resolved
ParameterStore, builds two circular ROIs in arcsecond coordinates, and writes:

- a float32 FITS mask that can be used as a detector ``pixel_response`` map
- a PNG quick-look preview for visual inspection

Typical use cases:
- proof-of-concept static detector cropping experiments
- generating reusable mask files under ``src/dluxshera/data/pixel_response/``
- validating that ROI size and source geometry are aligned before larger sweeps

Coordinate convention for source position angle (PA):
- ``PA = 90 deg`` gives left-right (horizontal) separation
- ``PA = 0 deg`` gives up-down (vertical) separation

For complete CLI usage, examples, valid value ranges, and option descriptions:

    python examples/scripts/generate_binary_rois.py --help

If the image field of view is too small for the selected geometry/ROI size, the
resulting mask can be entirely zero.

By default the final mask is clipped to hard binary edges (0/1). Use
``--no-clip-to-binary`` to keep anti-aliased edge values from the aperture
construction/downsampling step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from astropy.io import fits
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


try:
    import dLux.utils as dlu
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
    dlu = None
    _DLUX_IMPORT_ERROR = exc
else:
    _DLUX_IMPORT_ERROR = None

from dluxshera.config.resolver import resolve_system_config
from dluxshera.systems.base import compose_forward_spec
from dluxshera.params.store import ParameterStore


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTDIR = REPO_ROOT / "src" / "dluxshera" / "data" / "pixel_response"
BUILTIN_PRESET_DIR = REPO_ROOT / "src" / "dluxshera" / "data" / "system_presets"


class _CliHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Keep multiline epilog formatting and include default values in help."""


def _discover_builtin_system_presets() -> tuple[str, ...]:
    if not BUILTIN_PRESET_DIR.exists():
        return tuple()
    return tuple(sorted(path.stem for path in BUILTIN_PRESET_DIR.glob("*.yaml") if path.stem))


BUILTIN_SYSTEM_PRESETS = _discover_builtin_system_presets()


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if value < 0.0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def _build_parser() -> argparse.ArgumentParser:
    if BUILTIN_SYSTEM_PRESETS:
        preset_help = (
            "System preset name resolved via dluxshera config resolver. "
            f"Known built-ins: {', '.join(BUILTIN_SYSTEM_PRESETS)}."
        )
    else:
        preset_help = "System preset name resolved via dluxshera config resolver."

    parser = argparse.ArgumentParser(
        description="Generate a static two-ROI pixel-response mask.",
        formatter_class=_CliHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python examples/scripts/generate_binary_rois.py\n"
            "  python examples/scripts/generate_binary_rois.py \\\n"
            "      --system-preset SHERA_TESTBED_3P --npix 192 --oversample 8 --roi-diameter-as 6.0\n"
            "  python examples/scripts/generate_binary_rois.py \\\n"
            "      --separation-as 8.0 --position-angle-deg 35.0 --x-position-as 0.2 --y-position-as -0.1 \\\n"
            "      --outfile Results/roi_mask_custom.fits --preview Results/roi_mask_custom.png\n\n"
            "Valid value ranges:\n"
            "  --npix > 0\n"
            "  --oversample > 0\n"
            "  --roi-diameter-as > 0\n"
            "  --separation-as >= 0 (when provided)\n\n"
            "Binary clipping options:\n"
            "  --clip-to-binary     Force mask values to 0 or 1 using threshold 0.5 (default).\n"
            "  --no-clip-to-binary  Keep anti-aliased edge values.\n\n"
            "Output notes:\n"
            "  Parent directories for --outfile and --preview are created automatically.\n"
            "  The default output filenames are static examples and may not encode override values.\n"
            "  If ROI circles fall outside the image field of view, the resulting mask can be all zeros."
        ),
    )

    core = parser.add_argument_group("Core options")
    core.add_argument(
        "--system-preset",
        type=str,
        default="SHERA_FLIGHT_3P",
        help=preset_help,
    )
    core.add_argument(
        "--npix",
        type=_positive_int,
        default=256,
        help="Output mask image size in pixels (square: npix x npix).",
    )
    core.add_argument(
        "--oversample",
        type=_positive_int,
        default=4,
        help="Aperture oversampling factor used for anti-aliased circle edges.",
    )
    core.add_argument(
        "--roi-diameter-as",
        type=_positive_float,
        default=10.0,
        help="Circular ROI diameter in arcseconds, applied to each star ROI.",
    )
    core.add_argument(
        "--clip-to-binary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Force final mask values to hard binary {0,1} using threshold 0.5. "
            "Use --no-clip-to-binary to keep soft anti-aliased edges."
        ),
    )

    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--outfile",
        type=Path,
        default=DEFAULT_OUTDIR / "roi_mask_flight_sep10as_pa90_diam10as_256.fits",
        help="Path to output FITS mask file.",
    )
    output_group.add_argument(
        "--preview",
        type=Path,
        default=DEFAULT_OUTDIR / "roi_mask_flight_sep10as_pa90_diam10as_256.png",
        help="Path to output PNG preview file.",
    )

    overrides = parser.add_argument_group("Optional source-geometry overrides")
    overrides.add_argument(
        "--separation-as",
        type=_nonnegative_float,
        default=None,
        help="Override source.separation_as from the resolved preset [arcsec].",
    )
    overrides.add_argument(
        "--position-angle-deg",
        type=float,
        default=None,
        help="Override source.position_angle_deg from the resolved preset [deg].",
    )
    overrides.add_argument(
        "--x-position-as",
        type=float,
        default=None,
        help="Override source.x_position_as (binary midpoint x) [arcsec].",
    )
    overrides.add_argument(
        "--y-position-as",
        type=float,
        default=None,
        help="Override source.y_position_as (binary midpoint y) [arcsec].",
    )
    return parser


def _resolve_store(system_preset: str) -> tuple[dict, object, ParameterStore]:
    system_cfg = resolve_system_config({"preset": system_preset})
    forward_spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    return system_cfg, forward_spec, store


def _apply_optional_overrides(store: ParameterStore, forward_spec, args) -> ParameterStore:
    overrides = {}
    if args.separation_as is not None:
        overrides["source.separation_as"] = float(args.separation_as)
    if args.position_angle_deg is not None:
        overrides["source.position_angle_deg"] = float(args.position_angle_deg)
    if args.x_position_as is not None:
        overrides["source.x_position_as"] = float(args.x_position_as)
    if args.y_position_as is not None:
        overrides["source.y_position_as"] = float(args.y_position_as)

    if overrides:
        store = store.replace(overrides).refresh_derived(forward_spec)
    return store


def _star_centers_as(
    x0_as: float,
    y0_as: float,
    separation_as: float,
    position_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the two star centers in arcseconds.

    Convention used here:
    - PA = 90 deg gives horizontal alignment
    - PA = 0 deg gives vertical alignment
    """
    pa_rad = np.deg2rad(position_angle_deg)
    dx = 0.5 * separation_as * np.sin(pa_rad)
    dy = 0.5 * separation_as * np.cos(pa_rad)

    c1 = np.array([x0_as - dx, y0_as - dy], dtype=float)
    c2 = np.array([x0_as + dx, y0_as + dy], dtype=float)
    return c1, c2


def build_two_circle_mask(
    *,
    npix: int,
    oversample: int,
    plate_scale_as_per_pix: float,
    c1_as: np.ndarray,
    c2_as: np.ndarray,
    roi_radius_as: float,
    clip_to_binary: bool = True,
    binary_threshold: float = 0.5,
) -> np.ndarray:
    """
    Build an anti-aliased union mask from two circular ROIs.

    The coordinate grid is defined in arcseconds so the circle radius and
    centers can be specified directly in arcsecond units.
    """
    if dlu is None:
        raise ModuleNotFoundError(
            "dLux is required to build ROI masks. Install project dependencies "
            "before running this script."
        ) from _DLUX_IMPORT_ERROR

    extent_as = npix * plate_scale_as_per_pix
    coords = dlu.pixel_coords(npix * oversample, extent_as)

    roi1 = np.asarray(dlu.circle(dlu.translate_coords(coords, c1_as), roi_radius_as), dtype=np.float32)
    roi2 = np.asarray(dlu.circle(dlu.translate_coords(coords, c2_as), roi_radius_as), dtype=np.float32)

    # We want the union of both circles; dLux `combine` performs multiplicative
    # composition and would yield an empty mask for disjoint circles.
    mask_hi = np.maximum(roi1, roi2)
    mask = dlu.downsample(mask_hi, oversample)
    mask = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if clip_to_binary:
        mask = np.where(mask > float(binary_threshold), 1.0, 0.0).astype(np.float32)
    return mask


def write_mask_fits(
    mask: np.ndarray,
    outfile: Path,
    *,
    system_preset: str,
    npix: int,
    oversample: int,
    plate_scale_as_per_pix: float,
    separation_as: float,
    position_angle_deg: float,
    x_position_as: float,
    y_position_as: float,
    roi_diameter_as: float,
    c1_as: np.ndarray,
    c2_as: np.ndarray,
    clip_to_binary: bool,
) -> None:
    outfile.parent.mkdir(parents=True, exist_ok=True)

    hdu = fits.PrimaryHDU(mask.astype(np.float32))
    hdr = hdu.header
    hdr["MASKTYPE"] = ("2ROI", "Two-circle static pixel-response mask")
    hdr["SYSPRES"] = (system_preset, "Resolved system preset")
    hdr["NPIX"] = (int(npix), "Mask image size in pixels")
    hdr["OVERSAMP"] = (int(oversample), "Aperture oversampling factor")
    hdr["PLTSCALE"] = (float(plate_scale_as_per_pix), "Arcsec per pixel")
    hdr["SEPAS"] = (float(separation_as), "Binary separation [arcsec]")
    hdr["PADEG"] = (float(position_angle_deg), "Binary position angle [deg]")
    hdr["X0AS"] = (float(x_position_as), "Binary midpoint x [arcsec]")
    hdr["Y0AS"] = (float(y_position_as), "Binary midpoint y [arcsec]")
    hdr["ROIDIAM"] = (float(roi_diameter_as), "ROI diameter [arcsec]")
    hdr["ROIRAD"] = (float(0.5 * roi_diameter_as), "ROI radius [arcsec]")
    hdr["C1XAS"] = (float(c1_as[0]), "Star 1 x center [arcsec]")
    hdr["C1YAS"] = (float(c1_as[1]), "Star 1 y center [arcsec]")
    hdr["C2XAS"] = (float(c2_as[0]), "Star 2 x center [arcsec]")
    hdr["C2YAS"] = (float(c2_as[1]), "Star 2 y center [arcsec]")
    hdr["CLIPBIN"] = (bool(clip_to_binary), "Mask binarized to {0,1}")
    hdr["BINTHR"] = (0.5, "Binary threshold applied when CLIPBIN=True")

    hdu.writeto(outfile, overwrite=True)


def write_preview_png(
    mask: np.ndarray,
    preview_path: Path,
    *,
    plate_scale_as_per_pix: float,
) -> None:
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    extent_as = 0.5 * mask.shape[0] * plate_scale_as_per_pix
    extent = (-extent_as, extent_as, -extent_as, extent_as)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(mask, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Mask transmission")
    ax.set_title("Static 2-ROI pixel-response mask")
    ax.set_xlabel("x [arcsec]")
    ax.set_ylabel("y [arcsec]")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()

    system_cfg, forward_spec, store = _resolve_store(args.system_preset)
    store = _apply_optional_overrides(store, forward_spec, args)

    plate_scale = float(store.get("optics.plate_scale_as_per_pix"))
    separation_as = float(store.get("source.separation_as"))
    position_angle_deg = float(store.get("source.position_angle_deg"))
    x0_as = float(store.get("source.x_position_as"))
    y0_as = float(store.get("source.y_position_as"))

    roi_diameter_as = float(args.roi_diameter_as)
    roi_radius_as = 0.5 * roi_diameter_as

    c1_as, c2_as = _star_centers_as(
        x0_as=x0_as,
        y0_as=y0_as,
        separation_as=separation_as,
        position_angle_deg=position_angle_deg,
    )

    mask = build_two_circle_mask(
        npix=args.npix,
        oversample=args.oversample,
        plate_scale_as_per_pix=plate_scale,
        c1_as=c1_as,
        c2_as=c2_as,
        roi_radius_as=roi_radius_as,
        clip_to_binary=bool(args.clip_to_binary),
    )

    write_mask_fits(
        mask,
        args.outfile,
        system_preset=args.system_preset,
        npix=args.npix,
        oversample=args.oversample,
        plate_scale_as_per_pix=plate_scale,
        separation_as=separation_as,
        position_angle_deg=position_angle_deg,
        x_position_as=x0_as,
        y_position_as=y0_as,
        roi_diameter_as=roi_diameter_as,
        c1_as=c1_as,
        c2_as=c2_as,
        clip_to_binary=bool(args.clip_to_binary),
    )

    write_preview_png(
        mask,
        args.preview,
        plate_scale_as_per_pix=plate_scale,
    )

    roi_radius_pix = roi_radius_as / plate_scale
    nonzero_pixels = int(np.count_nonzero(mask))
    print("Wrote FITS mask:", args.outfile)
    print("Wrote preview PNG:", args.preview)
    print(f"plate_scale_as_per_pix = {plate_scale:.6f}")
    print(f"separation_as          = {separation_as:.6f}")
    print(f"position_angle_deg     = {position_angle_deg:.6f}")
    print(f"midpoint_as            = ({x0_as:.6f}, {y0_as:.6f})")
    print(f"star_1_center_as       = ({c1_as[0]:.6f}, {c1_as[1]:.6f})")
    print(f"star_2_center_as       = ({c2_as[0]:.6f}, {c2_as[1]:.6f})")
    print(f"roi_radius_as          = {roi_radius_as:.6f}")
    print(f"roi_radius_pix         = {roi_radius_pix:.3f}")
    print(f"clip_to_binary         = {bool(args.clip_to_binary)}")
    print(f"mask_shape             = {mask.shape}")
    print(f"mask_nonzero           = {nonzero_pixels}")
    print(f"mask_min/max           = ({mask.min():.3f}, {mask.max():.3f})")
    if nonzero_pixels == 0:
        print(
            "WARNING: mask is all zeros. The ROI circles may be outside the image field of view "
            "for the current npix/plate-scale/geometry settings."
        )


if __name__ == "__main__":
    main()
