"""Utilities for generating synthetic detector calibration maps and writing FITS.

This module provides lightweight helpers to create baseline pixel-offset and
pixel-response maps, add optional Gaussian noise, and emit FITS artifacts with
minimal but useful metadata. FITS is treated as the primary user-facing
artifact; NumPy sidecars are optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


@dataclass
class MapMetadata:
    """Metadata to embed in FITS headers."""

    map_type: str
    units: str = "pixel"
    generator: str = "dluxshera.calibration_maps"
    mode: str | None = None
    seed: int | None = None
    noise_amplitude: float | None = None
    note: str | None = None
    detector_model: str | None = None
    extra: dict[str, Any] | None = None


def _rng(seed: int | None):
    """Return a NumPy generator (seeded or unseeded)."""

    return np.random.default_rng(seed)


def generate_baseline_maps(
    nrows: int,
    ncols: int,
    noise_amplitude: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate baseline dx, dy (zeros) and PRF (ones) maps.

    Optionally adds iid Gaussian noise with the provided amplitude (std dev).
    """

    dx = np.zeros((nrows, ncols), dtype=float)
    dy = np.zeros((nrows, ncols), dtype=float)
    prf = np.ones((nrows, ncols), dtype=float)

    if noise_amplitude != 0.0:
        rng = _rng(seed)
        noise_dx = noise_amplitude * rng.standard_normal(size=dx.shape)
        noise_dy = noise_amplitude * rng.standard_normal(size=dy.shape)
        noise_prf = noise_amplitude * rng.standard_normal(size=prf.shape)
        dx = dx + noise_dx
        dy = dy + noise_dy
        prf = prf + noise_prf

    return dx, dy, prf


def realize_fpa_offsets(
    nrows: int,
    ncols: int,
    *,
    fixed_row: np.ndarray,
    fixed_col: np.ndarray,
    sig_offset: float,
    sig_diff: float | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Replicate the MATLAB-style realize_fpa offset generation.

    - ``fixed_row`` is a repeating row-pattern applied down the detector rows
      and then copied across all columns into ``dy``.
    - ``fixed_col`` is a repeating column-pattern applied across detector
      columns and then copied down all rows into ``dx``.
    - iid Gaussian noise with std ``sig_offset`` is added to both dx and dy.
    - ``sig_diff`` is currently metadata-only; kept for parity with MATLAB.
    """

    fixed_row = np.asarray(fixed_row, dtype=float).reshape(-1)
    fixed_col = np.asarray(fixed_col, dtype=float).reshape(-1)

    if fixed_row.shape[0] == 0:
        raise ValueError("fixed_row must contain at least one value.")
    if fixed_col.shape[0] == 0:
        raise ValueError("fixed_col must contain at least one value.")

    nfixed_row = fixed_row.shape[0]
    row_pattern = np.tile(fixed_row.reshape(-1, 1), (int(np.ceil(nrows / nfixed_row)), 1))[:nrows]
    row_err = np.tile(row_pattern, (1, ncols))

    nfixed_col = fixed_col.shape[0]
    col_pattern = np.tile(fixed_col.reshape(1, -1), (1, int(np.ceil(ncols / nfixed_col))))[:, :ncols]
    col_err = np.tile(col_pattern, (nrows, 1))

    rng = _rng(seed) if sig_offset else None
    iid_dx = sig_offset * rng.standard_normal(size=(nrows, ncols)) if rng is not None else 0.0
    iid_dy = sig_offset * rng.standard_normal(size=(nrows, ncols)) if rng is not None else 0.0

    offset_dx = iid_dx + col_err
    offset_dy = iid_dy + row_err

    return offset_dx, offset_dy


def write_fits(
    array: np.ndarray,
    path: Path,
    *,
    metadata: MapMetadata,
    dtype=np.float32,
    write_npy: bool = False,
) -> Path:
    """Write an array to FITS with minimal metadata.

    Parameters
    ----------
    array : np.ndarray
        Array data to write.
    path : Path
        Destination path for the FITS file.
    metadata : MapMetadata
        Structured metadata embedded in the FITS header.
    dtype : type, optional
        Output dtype (default float32).
    write_npy : bool, optional
        Also write a NumPy sidecar (.npy) if True.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(array, dtype=dtype)
    hdu = fits.PrimaryHDU(data=data)
    hdr = hdu.header

    hdr["MAPTYPE"] = metadata.map_type
    hdr["SHAPE0"] = data.shape[0]
    hdr["SHAPE1"] = data.shape[1]
    hdr["UNITS"] = metadata.units
    hdr["CREATOR"] = metadata.generator
    hdr["DATE"] = datetime.now(timezone.utc).isoformat()

    if metadata.mode:
        hdr["MODE"] = metadata.mode
    if metadata.seed is not None:
        hdr["SEED"] = int(metadata.seed)
    if metadata.noise_amplitude is not None:
        hdr["NOISEAMP"] = float(metadata.noise_amplitude)
    if metadata.detector_model:
        hdr["DETMODEL"] = metadata.detector_model
    if metadata.note:
        hdr["NOTE"] = metadata.note[:68]
    if metadata.extra:
        for key, value in metadata.extra.items():
            # FITS keys must be <= 8 chars; truncate cautiously.
            if isinstance(value, (list, tuple, np.ndarray)):
                value_to_store = str(np.asarray(value).tolist())
            else:
                value_to_store = value
            hdr[str(key)[:8].upper()] = value_to_store

    fits.HDUList([hdu]).writeto(path, overwrite=True)

    if write_npy:
        np.save(path.with_suffix(".npy"), data)

    return path


def write_baseline_maps(
    dx: np.ndarray,
    dy: np.ndarray,
    prf: np.ndarray,
    *,
    offsets_dir: Path,
    prf_dir: Path,
    noise_amplitude: float = 0.0,
    seed: int | None = None,
    basename: str = "baseline",
    write_npy: bool = False,
) -> tuple[Path, Path, Path]:
    """Write baseline dx, dy, and PRF maps to FITS (and optional .npy)."""

    dx_path = Path(offsets_dir) / f"dx_{basename}.fits"
    dy_path = Path(offsets_dir) / f"dy_{basename}.fits"
    prf_path = Path(prf_dir) / f"prf_{basename}.fits"

    common_meta = dict(
        noise_amplitude=noise_amplitude,
        seed=seed,
        mode="baseline",
    )

    write_fits(
        dx,
        dx_path,
        metadata=MapMetadata(map_type="DX", **common_meta),
        write_npy=write_npy,
    )
    write_fits(
        dy,
        dy_path,
        metadata=MapMetadata(map_type="DY", **common_meta),
        write_npy=write_npy,
    )
    write_fits(
        prf,
        prf_path,
        metadata=MapMetadata(map_type="PRF", units="dimensionless", **common_meta),
        write_npy=write_npy,
    )

    return dx_path, dy_path, prf_path


def write_realize_fpa_maps(
    dx: np.ndarray,
    dy: np.ndarray,
    *,
    offsets_dir: Path,
    noise_amplitude: float,
    seed: int | None,
    fixed_row: np.ndarray,
    fixed_col: np.ndarray,
    sig_offset: float,
    sig_diff: float | None = None,
    basename: str = "realize_fpa",
    write_npy: bool = False,
) -> tuple[Path, Path]:
    """Write realize_fpa-generated dx/dy maps to FITS."""

    dx_path = Path(offsets_dir) / f"dx_{basename}.fits"
    dy_path = Path(offsets_dir) / f"dy_{basename}.fits"

    extra_meta = {
        "FIXROW": ",".join(map(str, np.asarray(fixed_row, dtype=float).tolist())),
        "FIXCOL": ",".join(map(str, np.asarray(fixed_col, dtype=float).tolist())),
        "SIGOFSET": sig_offset,
    }
    if sig_diff is not None:
        extra_meta["SIGDIFF"] = sig_diff

    meta_kwargs = dict(
        noise_amplitude=noise_amplitude,
        seed=seed,
        mode="realize_fpa",
        extra=extra_meta,
    )

    write_fits(
        dx,
        dx_path,
        metadata=MapMetadata(map_type="DX", **meta_kwargs),
        write_npy=write_npy,
    )
    write_fits(
        dy,
        dy_path,
        metadata=MapMetadata(map_type="DY", **meta_kwargs),
        write_npy=write_npy,
    )

    return dx_path, dy_path
