from __future__ import annotations

"""Generate detector calibration maps (FITS primary format).

This script produces calibration maps for the detector builder's
``pixel_offsets`` and ``pixel_response`` layers. It supports two distinct
workflows:

``baseline``
    Generates a full set of baseline maps for a detector of shape
    ``(nrows, ncols)``:

    - ``dx``: all zeros, optionally perturbed by iid Gaussian noise
    - ``dy``: all zeros, optionally perturbed by iid Gaussian noise
    - ``prf``: all ones, optionally perturbed by iid Gaussian noise

    The outputs are written as:

    - ``<offsets-dir>/dx_<basename>.fits``
    - ``<offsets-dir>/dy_<basename>.fits``
    - ``<prf-dir>/prf_<basename>.fits``
    - optionally ``<offsets-dir>/preview_<basename>.png`` when
      ``--save-preview`` is used

``realize-fpa``
    Generates detector pixel-offset maps only, matching the behavior of the
    MATLAB-style ``realize_fpa`` implementation used as the reference for this
    port. It does not generate a pixel-response / PRF map.

    Inputs:

    - ``fixed_row``: repeating row-pattern vector applied down detector rows,
      then copied across all columns and added to ``dy``
    - ``fixed_col``: repeating column-pattern vector applied across detector
      columns, then copied down all rows and added to ``dx``
    - ``sig_offset``: standard deviation of the iid Gaussian offset noise added
      independently to both ``dx`` and ``dy``
    - ``sig_diff``: accepted for metadata parity with MATLAB; currently not used
      in the map generation itself

    The outputs are written as:

    - ``<offsets-dir>/dx_<basename>.fits``
    - ``<offsets-dir>/dy_<basename>.fits``
    - optionally ``<offsets-dir>/preview_<basename>.png`` when
      ``--save-preview`` is used

Shape rules
-----------
Use ``--npix N`` for an ``N x N`` detector, or provide ``--nrows`` and
``--ncols`` explicitly. In ``realize-fpa`` mode, ``--fixed-row`` and
``--fixed-col`` may contain any positive number of values. Each vector is
treated as a repeating pattern and is repeated or truncated as needed to fill
the detector shape, matching the MATLAB implementation.

CLI notes
---------
- ``--mode baseline`` is the default.
- ``--basename`` controls the filename stem after the ``dx_`` / ``dy_`` /
  ``prf_`` prefixes.
- ``--write-npy`` writes NumPy sidecars alongside the FITS files.
- ``--save-preview`` writes a PNG quick-look image of the generated map(s).
- ``--seed`` controls the JAX PRNG base seed used for any random realization.
  This includes baseline Gaussian noise and the iid ``sig_offset`` term in
  ``realize-fpa`` mode.
- ``--noise-amplitude`` applies only to ``baseline`` mode.
- ``--prf-dir`` is used only in ``baseline`` mode because ``realize-fpa`` does
  not emit a PRF map.

Examples
--------
Generate baseline maps for a 512 x 512 detector:

    python examples/scripts/generate_calibration_maps.py --mode baseline --npix 512 --seed 42

Generate realize-fpa offset maps for a 3 x 2 detector:

    python examples/scripts/generate_calibration_maps.py \\
        --mode realize-fpa \\
        --nrows 3 --ncols 2 \\
        --fixed-row 0.1 0.2 \\
        --fixed-col 0.3 0.4 0.5 \\
        --sig-offset 0.01 \\
        --seed 42
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dluxshera.utils.calibration_maps import (
    write_baseline_maps,
    write_realize_fpa_maps,
)


DEFAULT_OFFSETS_DIR = Path("src/dluxshera/data/pixel_offsets")
DEFAULT_PRF_DIR = Path("src/dluxshera/data/pixel_response")


def resolve_generation_seed(seed: int | None, *, requires_random: bool) -> int | None:
    """Return the seed to use for generation, creating one when needed."""

    if seed is not None:
        return int(seed)
    if not requires_random:
        return None
    return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])


def generate_baseline_maps_jax(
    nrows: int,
    ncols: int,
    *,
    noise_amplitude: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
    """Generate baseline calibration maps using JAX PRNG splits."""

    used_seed = resolve_generation_seed(seed, requires_random=noise_amplitude != 0.0)

    dx = jnp.zeros((nrows, ncols), dtype=float)
    dy = jnp.zeros((nrows, ncols), dtype=float)
    prf = jnp.ones((nrows, ncols), dtype=float)

    if noise_amplitude != 0.0:
        base_key = jr.PRNGKey(int(used_seed))
        dx_key, dy_key, prf_key = jr.split(base_key, 3)
        dx = dx + noise_amplitude * jr.normal(dx_key, dx.shape, dtype=dx.dtype)
        dy = dy + noise_amplitude * jr.normal(dy_key, dy.shape, dtype=dy.dtype)
        prf = prf + noise_amplitude * jr.normal(prf_key, prf.shape, dtype=prf.dtype)

    return np.asarray(dx), np.asarray(dy), np.asarray(prf), used_seed


def realize_fpa_offsets_jax(
    nrows: int,
    ncols: int,
    *,
    fixed_row: np.ndarray,
    fixed_col: np.ndarray,
    sig_offset: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int | None]:
    """Generate MATLAB-style realize-fpa dx/dy offsets using JAX PRNG splits."""

    fixed_row = np.asarray(fixed_row, dtype=float).reshape(-1)
    fixed_col = np.asarray(fixed_col, dtype=float).reshape(-1)

    if fixed_row.shape[0] == 0:
        raise ValueError("fixed_row must contain at least one value.")
    if fixed_col.shape[0] == 0:
        raise ValueError("fixed_col must contain at least one value.")

    nfixed_row = fixed_row.shape[0]
    row_pattern = jnp.tile(
        jnp.asarray(fixed_row).reshape(-1, 1),
        (int(np.ceil(nrows / nfixed_row)), 1),
    )[:nrows]
    row_err = jnp.tile(row_pattern, (1, ncols))

    nfixed_col = fixed_col.shape[0]
    col_pattern = jnp.tile(
        jnp.asarray(fixed_col).reshape(1, -1),
        (1, int(np.ceil(ncols / nfixed_col))),
    )[:, :ncols]
    col_err = jnp.tile(col_pattern, (nrows, 1))
    used_seed = resolve_generation_seed(seed, requires_random=sig_offset != 0.0)

    if sig_offset != 0.0:
        base_key = jr.PRNGKey(int(used_seed))
        dx_key, dy_key = jr.split(base_key, 2)
        iid_dx = sig_offset * jr.normal(dx_key, (nrows, ncols), dtype=row_err.dtype)
        iid_dy = sig_offset * jr.normal(dy_key, (nrows, ncols), dtype=col_err.dtype)
    else:
        iid_dx = jnp.zeros((nrows, ncols), dtype=col_err.dtype)
        iid_dy = jnp.zeros((nrows, ncols), dtype=row_err.dtype)

    dx = col_err + iid_dx
    dy = row_err + iid_dy
    return np.asarray(dx), np.asarray(dy), used_seed


def save_preview_image(
    arrays: list[tuple[str, np.ndarray]],
    path: Path,
    *,
    basename: str,
    mode: str,
) -> Path:
    """Save a quick-look PNG preview of the generated calibration maps."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(arrays), figsize=(5 * len(arrays), 4), squeeze=False)
    fig.suptitle(f"Calibration map preview: {basename} ({mode})")

    for ax, (title, array) in zip(axes[0], arrays):
        image = ax.imshow(np.asarray(array), origin="lower", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate detector calibration maps (FITS).")
    parser.add_argument("--mode", choices=["baseline", "realize-fpa"], default="baseline")

    shape = parser.add_argument_group("shape")
    shape.add_argument("--npix", type=int, help="Square detector size; overrides nrows/ncols if set.")
    shape.add_argument("--nrows", type=int, default=512, help="Number of detector rows (default: 512).")
    shape.add_argument("--ncols", type=int, default=512, help="Number of detector cols (default: 512).")

    noise = parser.add_argument_group("noise")
    noise.add_argument(
        "--noise-amplitude",
        type=float,
        default=0.0,
        help="Gaussian noise std dev for baseline mode only (default: 0).",
    )
    noise.add_argument(
        "--seed",
        type=int,
        help="Base JAX PRNG seed for reproducible noise realizations. If omitted, a seed is generated when needed.",
    )

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
        help="Output directory for PRF map in baseline mode (default: src/dluxshera/data/pixel_response).",
    )
    parser.add_argument("--basename", default=None, help="Base filename stem (default depends on mode).")
    parser.add_argument("--write-npy", action="store_true", help="Also write .npy sidecars.")
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Also save a PNG quick-look image to <offsets-dir>/preview_<basename>.png.",
    )

    realize = parser.add_argument_group("realize-fpa")
    realize.add_argument(
        "--fixed-row",
        type=float,
        nargs="+",
        help="Repeating row-pattern vector applied down detector rows and added to dy.",
    )
    realize.add_argument(
        "--fixed-col",
        type=float,
        nargs="+",
        help="Repeating column-pattern vector applied across detector columns and added to dx.",
    )
    realize.add_argument(
        "--sig-offset",
        type=float,
        help="Std dev of the iid Gaussian offset noise added to dx and dy (required for realize-fpa).",
    )
    realize.add_argument("--sig-diff", type=float, help="Optional sig_diff value stored as metadata only.")

    return parser.parse_args()


def resolve_shape(args: argparse.Namespace) -> tuple[int, int]:
    if args.npix is not None:
        return args.npix, args.npix
    return args.nrows, args.ncols


def main() -> None:
    args = parse_args()
    nrows, ncols = resolve_shape(args)
    basename = args.basename or ("baseline" if args.mode == "baseline" else "realize_fpa")
    preview_path = args.offsets_dir / f"preview_{basename}.png"

    if args.mode == "baseline":
        dx, dy, prf, used_seed = generate_baseline_maps_jax(
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
            seed=used_seed,
            basename=basename,
            write_npy=bool(args.write_npy),
        )
        preview_line = ""
        if args.save_preview:
            saved_preview = save_preview_image(
                [("dx", dx), ("dy", dy), ("prf", prf)],
                preview_path,
                basename=basename,
                mode="baseline",
            )
            preview_line = f"\n  preview: {saved_preview}"
        seed_line = f"\n  seed: {used_seed}" if used_seed is not None else ""
        print(
            f"Wrote baseline maps:\n  dx: {dx_path}\n  dy: {dy_path}\n  prf: {prf_path}{preview_line}{seed_line}"
        )
        return

    # realize-fpa mode
    if args.fixed_row is None or args.fixed_col is None or args.sig_offset is None:
        raise SystemExit("realize-fpa mode requires --fixed-row, --fixed-col, and --sig-offset")

    fixed_row = np.asarray(args.fixed_row, dtype=float)
    fixed_col = np.asarray(args.fixed_col, dtype=float)

    dx, dy, used_seed = realize_fpa_offsets_jax(
        nrows,
        ncols,
        fixed_row=fixed_row,
        fixed_col=fixed_col,
        sig_offset=args.sig_offset,
        seed=args.seed,
    )

    dx_path, dy_path = write_realize_fpa_maps(
        dx,
        dy,
        offsets_dir=args.offsets_dir,
        noise_amplitude=args.sig_offset,
        seed=used_seed,
        fixed_row=fixed_row,
        fixed_col=fixed_col,
        sig_offset=args.sig_offset,
        sig_diff=args.sig_diff,
        basename=basename,
        write_npy=bool(args.write_npy),
    )
    preview_line = ""
    if args.save_preview:
        saved_preview = save_preview_image(
            [("dx", dx), ("dy", dy)],
            preview_path,
            basename=basename,
            mode="realize_fpa",
        )
        preview_line = f"\n  preview: {saved_preview}"
    seed_line = f"\n  seed: {used_seed}" if used_seed is not None else ""
    print(f"Wrote realize_fpa offsets:\n  dx: {dx_path}\n  dy: {dy_path}{preview_line}{seed_line}")


if __name__ == "__main__":
    main()
