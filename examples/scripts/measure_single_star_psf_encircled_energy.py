#!/usr/bin/env python3
"""Measure finite-FOV encircled energy for a centered Alpha Cen A-like PSF.

Example
-------
PYTHONPATH=src python examples/scripts/measure_single_star_psf_encircled_energy.py \
  --system-preset SHERA_FLIGHT_3P \
  --psf-npix 512 \
  --aperture-diameters-as 9 10 \
  --outdir Results/psf_encircled_energy/alpha_cen_a_512
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from math import pi
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from generate_target_grating_portraits import (
    ALPHA_CEN_A_PLACEHOLDER_NOTE,
    DEFAULT_EXPOSURE_TIME_S,
    DEFAULT_N_LAMBDA,
    DEFAULT_PSF_NPIX,
    DEFAULT_PUPIL_NPIX,
    SYSTEM_PRESET,
    _build_forward_store,
    _build_grating_dp_opd_payload,
    _image_extent_as,
    _prepare_alpha_cen_a_single_star_system_cfg,
    _render_image,
    _resolve_system_cfg,
)

try:
    from astropy.io import fits

    _HAVE_FITS = True
except ModuleNotFoundError:
    fits = None
    _HAVE_FITS = False


DEFAULT_RESULTS_ROOT = Path("Results/psf_encircled_energy")
DEFAULT_APERTURE_DIAMETERS_AS = (9.0, 10.0)
EEF_LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)
SCHEMA_VERSION = "single_star_psf_encircled_energy.v1"


def _timestamp_tag() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _created_at() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render centered Alpha Cen A-like single-star PSFs with and without "
            "the phase-flipped sinusoidal DP grating, then measure finite-FOV "
            "encircled energy."
        ),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--system-preset", default=SYSTEM_PRESET)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--psf-npix", type=int, default=DEFAULT_PSF_NPIX)
    parser.add_argument("--pupil-npix", type=int, default=DEFAULT_PUPIL_NPIX)
    parser.add_argument("--exposure-time-s", type=float, default=DEFAULT_EXPOSURE_TIME_S)
    parser.add_argument("--n-lambda", type=int, default=DEFAULT_N_LAMBDA)
    parser.add_argument(
        "--aperture-diameters-as",
        type=float,
        nargs="+",
        default=list(DEFAULT_APERTURE_DIAMETERS_AS),
    )
    parser.add_argument("--aperture-oversample", type=int, default=8)
    parser.add_argument("--eef-samples", type=int, default=512)
    parser.add_argument("--quicklooks", dest="quicklooks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grating-phase-amplitude-rad", type=float, default=float(pi / 16.0))
    parser.add_argument("--grating-frequency", type=float, default=128.0)
    parser.add_argument("--grating-angle-deg", type=float, default=45.0)
    parser.add_argument("--grating-phase-flip", dest="grating_phase_flip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grating-mask-threshold", type=float, default=0.5)
    parser.add_argument("--binary-mask", dest="binary_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--legacy-dp-centering", dest="legacy_dp_centering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dp-enabled", dest="dp_enabled", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    if args.psf_npix <= 0:
        parser.error("--psf-npix must be positive.")
    if args.pupil_npix <= 0:
        parser.error("--pupil-npix must be positive.")
    if args.n_lambda <= 0:
        parser.error("--n-lambda must be positive.")
    if not np.isfinite(args.exposure_time_s) or args.exposure_time_s <= 0.0:
        parser.error("--exposure-time-s must be a positive finite value.")
    if args.aperture_oversample <= 0:
        parser.error("--aperture-oversample must be positive.")
    if args.eef_samples <= 1:
        parser.error("--eef-samples must be greater than 1.")
    if not args.aperture_diameters_as:
        parser.error("At least one aperture diameter is required.")
    for diameter in args.aperture_diameters_as:
        if not np.isfinite(diameter) or diameter <= 0.0:
            parser.error("--aperture-diameters-as values must be positive and finite.")
    return args


def aperture_weights(
    shape: tuple[int, int],
    pixel_scale_as: float,
    radius_as: float,
    oversample: int = 8,
) -> np.ndarray:
    ny, nx = shape
    y0 = (ny - 1) / 2.0
    x0 = (nx - 1) / 2.0

    yy, xx = np.indices((ny, nx), dtype=float)
    weights = np.zeros((ny, nx), dtype=float)

    offsets = (np.arange(oversample) + 0.5) / oversample - 0.5
    for dy in offsets:
        for dx in offsets:
            x_as = (xx + dx - x0) * pixel_scale_as
            y_as = (yy + dy - y0) * pixel_scale_as
            weights += (x_as**2 + y_as**2) <= radius_as**2

    weights /= float(oversample * oversample)
    return weights


def _pixel_scale_from_extent(extent_as: np.ndarray, shape: tuple[int, int]) -> float:
    ny, nx = shape
    x_scale = float(abs(extent_as[1] - extent_as[0]) / nx)
    y_scale = float(abs(extent_as[3] - extent_as[2]) / ny)
    if not np.isclose(x_scale, y_scale, rtol=1e-6, atol=0.0):
        raise ValueError(f"Non-square pixels are not supported: x={x_scale}, y={y_scale}.")
    return x_scale


def _measure_apertures(
    image: np.ndarray,
    *,
    case: str,
    pixel_scale_as: float,
    aperture_diameters_as: Sequence[float],
    oversample: int,
) -> list[dict[str, Any]]:
    total = float(np.sum(image))
    rows: list[dict[str, Any]] = []
    for diameter_as in aperture_diameters_as:
        radius_as = float(diameter_as) / 2.0
        weights = aperture_weights(image.shape, pixel_scale_as, radius_as, oversample=oversample)
        core = float(np.sum(image * weights))
        outside = float(total - core)
        core_fraction = float(core / total) if total != 0.0 else np.nan
        outside_fraction = float(outside / total) if total != 0.0 else np.nan
        rows.append(
            {
                "case": case,
                "aperture_diameter_as": float(diameter_as),
                "aperture_radius_as": radius_as,
                "core_energy": core,
                "outside_energy": outside,
                "total_captured_energy": total,
                "core_fraction_captured_fov": core_fraction,
                "outside_fraction_captured_fov": outside_fraction,
                "aperture_oversample": int(oversample),
            }
        )
    return rows


def _center_radius_grid(shape: tuple[int, int], pixel_scale_as: float) -> np.ndarray:
    ny, nx = shape
    y0 = (ny - 1) / 2.0
    x0 = (nx - 1) / 2.0
    yy, xx = np.indices((ny, nx), dtype=float)
    return np.hypot(xx - x0, yy - y0) * pixel_scale_as


def _compute_eef(
    image: np.ndarray,
    *,
    case: str,
    pixel_scale_as: float,
    usable_half_fov_as: float,
    n_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    radius_grid = _center_radius_grid(image.shape, pixel_scale_as)
    flat_radius = radius_grid.ravel()
    flat_image = np.asarray(image, dtype=float).ravel()
    order = np.argsort(flat_radius)
    sorted_radius = flat_radius[order]
    sorted_energy = flat_image[order]
    cumulative = np.cumsum(sorted_energy)
    total = float(cumulative[-1]) if cumulative.size else 0.0

    radii = np.linspace(0.0, float(usable_half_fov_as), int(n_samples))
    curve_rows: list[dict[str, Any]] = []
    if total == 0.0 or not np.isfinite(total):
        for radius_as in radii:
            curve_rows.append({"case": case, "radius_as": float(radius_as), "eef_captured_fov": np.nan})
        return curve_rows, {f"r{level * 100:g}_as": None for level in EEF_LEVELS}

    sample_energy = np.interp(radii, sorted_radius, cumulative, left=0.0, right=total)
    for radius_as, energy in zip(radii, sample_energy):
        curve_rows.append(
            {
                "case": case,
                "radius_as": float(radius_as),
                "encircled_energy": float(energy),
                "eef_captured_fov": float(energy / total),
                "total_captured_energy": total,
            }
        )

    levels: dict[str, float | None] = {}
    fraction = cumulative / total
    usable = sorted_radius <= float(usable_half_fov_as)
    usable_radius = sorted_radius[usable]
    usable_fraction = fraction[usable]
    for level in EEF_LEVELS:
        key = f"r{level * 100:g}_as"
        if usable_fraction.size == 0 or usable_fraction[-1] < level:
            levels[key] = None
        else:
            levels[key] = float(np.interp(level, usable_fraction, usable_radius))
    return curve_rows, levels


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_image(path_without_suffix: Path, image: np.ndarray, metadata: Mapping[str, Any]) -> Path:
    if _HAVE_FITS:
        output_path = path_without_suffix.with_suffix(".fits")
        header = fits.Header()
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) and len(str(key)) <= 8:
                header.set(str(key).upper(), value)
        fits.PrimaryHDU(data=np.asarray(image), header=header).writeto(output_path, overwrite=True)
        return output_path
    output_path = path_without_suffix.with_suffix(".npy")
    np.save(output_path, np.asarray(image))
    return output_path


def _save_quicklook(path: Path, image: np.ndarray, *, title: str, extent_as: np.ndarray) -> None:
    positive = image[np.isfinite(image) & (image > 0.0)]
    if positive.size == 0:
        vmin, vmax = 1.0, 2.0
    else:
        vmin = max(float(np.percentile(positive, 10.0)), np.finfo(float).tiny)
        vmax = float(np.percentile(positive, 99.9))
        if vmax <= vmin:
            vmax = vmin * 1.001
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    im = ax.imshow(image, origin="lower", cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent_as)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="PSF intensity (log)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _save_eef_plot(path: Path, curve_rows: Sequence[Mapping[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    cases = sorted({str(row["case"]) for row in curve_rows})
    for case in cases:
        rows = [row for row in curve_rows if row["case"] == case]
        ax.plot(
            [float(row["radius_as"]) for row in rows],
            [float(row["eef_captured_fov"]) for row in rows],
            label=case,
        )
    ax.set_xlabel("Radius [arcsec]")
    ax.set_ylabel("EEF relative to captured finite-FOV energy")
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _comparison_rows(summary_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_diameter: dict[float, dict[str, Mapping[str, Any]]] = {}
    for row in summary_rows:
        by_diameter.setdefault(float(row["aperture_diameter_as"]), {})[str(row["case"])] = row

    rows: list[dict[str, Any]] = []
    for diameter_as, case_rows in sorted(by_diameter.items()):
        dp_only = case_rows.get("dp_only")
        grating = case_rows.get("dp_plus_grating")
        if dp_only is None or grating is None:
            continue
        no_grating_core = float(dp_only["core_fraction_captured_fov"])
        grating_core = float(grating["core_fraction_captured_fov"])
        no_grating_outside = float(dp_only["outside_fraction_captured_fov"])
        grating_outside = float(grating["outside_fraction_captured_fov"])
        rows.append(
            {
                "aperture_diameter_as": diameter_as,
                "grating_core_fraction_delta": grating_core - no_grating_core,
                "grating_outside_fraction_delta": grating_outside - no_grating_outside,
                "grating_relative_core_loss": (
                    (no_grating_core - grating_core) / no_grating_core
                    if no_grating_core != 0.0
                    else np.nan
                ),
            }
        )
    return rows


def _attach_comparison_columns(
    summary_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    comparison_by_diameter = {
        float(row["aperture_diameter_as"]): row
        for row in comparison_rows
    }
    annotated: list[dict[str, Any]] = []
    for row in summary_rows:
        output_row = dict(row)
        comparison = comparison_by_diameter.get(float(row["aperture_diameter_as"]))
        if comparison is not None and row["case"] == "dp_plus_grating":
            output_row.update(
                {
                    "grating_core_fraction_delta": comparison["grating_core_fraction_delta"],
                    "grating_outside_fraction_delta": comparison["grating_outside_fraction_delta"],
                    "grating_relative_core_loss": comparison["grating_relative_core_loss"],
                }
            )
        else:
            output_row.update(
                {
                    "grating_core_fraction_delta": "",
                    "grating_outside_fraction_delta": "",
                    "grating_relative_core_loss": "",
                }
            )
        annotated.append(output_row)
    return annotated


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    timestamp = _timestamp_tag()
    created_at = _created_at()
    outdir = Path(args.outdir).expanduser() if args.outdir is not None else DEFAULT_RESULTS_ROOT / timestamp
    outdir.mkdir(parents=True, exist_ok=True)

    base_system_cfg = _resolve_system_cfg(config_path=args.config, system_preset=args.system_preset)
    optics_cfg = base_system_cfg.get("optics", {})
    source_cfg = base_system_cfg.get("source", {})
    if not isinstance(optics_cfg, dict) or not isinstance(source_cfg, dict):
        raise ValueError("Expected resolved system.optics and system.source mappings.")
    optics_cfg["pupil_npix"] = int(args.pupil_npix)
    source_cfg["exposure_time_s"] = float(args.exposure_time_s)
    source_cfg["n_lambda"] = int(args.n_lambda)

    wavelength_m = float(optics_cfg.get("dp_design_wavelength_m") or optics_cfg.get("wavelength_m") or 550e-9)
    dp_path_cfg = optics_cfg.get("diffractive_pupil_path", optics_cfg.get("dp_path"))
    grating_payload = _build_grating_dp_opd_payload(
        pupil_npix=int(args.pupil_npix),
        wavelength_m=wavelength_m,
        dp_enabled=bool(args.dp_enabled),
        grating_phase_amplitude_rad=float(args.grating_phase_amplitude_rad),
        grating_enabled=True,
        grating_frequency=float(args.grating_frequency),
        grating_angle_deg=float(args.grating_angle_deg),
        grating_phase_flip=bool(args.grating_phase_flip),
        grating_mask_threshold=float(args.grating_mask_threshold),
        binary_mask=bool(args.binary_mask),
        legacy_dp_centering=bool(args.legacy_dp_centering),
        dp_path=Path(dp_path_cfg) if isinstance(dp_path_cfg, str) else None,
    )
    dp_only_payload = _build_grating_dp_opd_payload(
        pupil_npix=int(args.pupil_npix),
        wavelength_m=wavelength_m,
        dp_enabled=bool(args.dp_enabled),
        grating_phase_amplitude_rad=float(args.grating_phase_amplitude_rad),
        grating_enabled=False,
        grating_frequency=float(args.grating_frequency),
        grating_angle_deg=float(args.grating_angle_deg),
        grating_phase_flip=bool(args.grating_phase_flip),
        grating_mask_threshold=float(args.grating_mask_threshold),
        binary_mask=bool(args.binary_mask),
        legacy_dp_centering=bool(args.legacy_dp_centering),
        dp_path=Path(dp_path_cfg) if isinstance(dp_path_cfg, str) else None,
    )

    cases = {
        "dp_only": np.asarray(dp_only_payload["combined_opd_m"], dtype=float),
        "dp_plus_grating": np.asarray(grating_payload["combined_opd_m"], dtype=float),
    }
    dp_opd_paths: dict[str, Path] = {}
    for case_name, dp_opd_m in cases.items():
        dp_opd_path = outdir / f"{case_name}_dp_opd_m.npy"
        np.save(dp_opd_path, dp_opd_m)
        dp_opd_paths[case_name] = dp_opd_path

    images: dict[str, np.ndarray] = {}
    extents: dict[str, np.ndarray] = {}
    image_paths: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []
    eef_rows: list[dict[str, Any]] = []
    eef_radii_by_case: dict[str, dict[str, float | None]] = {}

    for case_name, dp_opd_m in cases.items():
        system_cfg = _prepare_alpha_cen_a_single_star_system_cfg(
            base_system_cfg,
            psf_npix=int(args.psf_npix),
            dp_opd_m=dp_opd_m,
        )
        case_optics_cfg = system_cfg.setdefault("optics", {})
        if not isinstance(case_optics_cfg, dict):
            raise ValueError("Expected prepared system.optics mapping.")
        case_dp_path = str(dp_opd_paths[case_name])
        case_optics_cfg["diffractive_pupil_path"] = case_dp_path
        case_optics_cfg["dp_path"] = case_dp_path
        case_optics_cfg["dp_design_wavelength_m"] = None
        forward_spec, store = _build_forward_store(system_cfg, normalize_total_flux=False)
        image = np.asarray(_render_image(system_cfg, forward_spec, store), dtype=float)
        extent = _image_extent_as(store, image)
        pixel_scale_as = _pixel_scale_from_extent(extent, image.shape)
        half_fov_as = float(min(abs(extent[0]), abs(extent[1]), abs(extent[2]), abs(extent[3])))

        images[case_name] = image
        extents[case_name] = extent
        summary_rows.extend(
            _measure_apertures(
                image,
                case=case_name,
                pixel_scale_as=pixel_scale_as,
                aperture_diameters_as=args.aperture_diameters_as,
                oversample=int(args.aperture_oversample),
            )
        )
        case_eef_rows, case_eef_radii = _compute_eef(
            image,
            case=case_name,
            pixel_scale_as=pixel_scale_as,
            usable_half_fov_as=half_fov_as,
            n_samples=int(args.eef_samples),
        )
        eef_rows.extend(case_eef_rows)
        eef_radii_by_case[case_name] = case_eef_radii
        image_path = _write_image(
            outdir / case_name,
            image,
            {
                "CASE": case_name,
                "PSFNPIX": int(args.psf_npix),
                "PIXSCALE": pixel_scale_as,
                "HALFFOV": half_fov_as,
            },
        )
        image_paths[case_name] = image_path.relative_to(outdir).as_posix()

    comparison = _comparison_rows(summary_rows)
    annotated_summary_rows = _attach_comparison_columns(summary_rows, comparison)
    summary_payload = {
        "schema": SCHEMA_VERSION,
        "created_at": created_at,
        "normalization_note": "Fractions are relative to total captured image energy over the rendered finite FOV.",
        "measurements": annotated_summary_rows,
        "comparisons": comparison,
        "eef_enclosing_radii_as": eef_radii_by_case,
    }

    _write_csv(outdir / "encircled_energy_summary.csv", annotated_summary_rows)
    _write_csv(outdir / "eef_curve.csv", eef_rows)
    (outdir / "encircled_energy_summary.json").write_text(
        json.dumps(_jsonable(summary_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    first_case = "dp_only"
    first_image = images[first_case]
    first_extent = extents[first_case]
    pixel_scale_as = _pixel_scale_from_extent(first_extent, first_image.shape)
    half_fov_as = float(min(abs(first_extent[0]), abs(first_extent[1]), abs(first_extent[2]), abs(first_extent[3])))
    warnings = []
    if half_fov_as < 60.0:
        warnings.append(
            "Rendered half FOV is less than 60 arcsec; rerun with larger --psf-npix "
            "or adjusted plate scale if outer spectral streak capture is required."
        )
    warnings.append(
        "A single run cannot assess total captured energy convergence; compare this "
        "manifest across larger --psf-npix runs to check FOV/cropping behavior."
    )

    artifacts: dict[str, Any] = {
        "encircled_energy_summary_csv": "encircled_energy_summary.csv",
        "encircled_energy_summary_json": "encircled_energy_summary.json",
        "eef_curve_csv": "eef_curve.csv",
        "images": image_paths,
        "dp_opd_maps": {
            case_name: path.relative_to(outdir).as_posix()
            for case_name, path in dp_opd_paths.items()
        },
    }
    if args.quicklooks:
        _save_quicklook(outdir / "dp_only_log.png", images["dp_only"], title="DP only", extent_as=extents["dp_only"])
        _save_quicklook(
            outdir / "dp_plus_grating_log.png",
            images["dp_plus_grating"],
            title="DP plus phase-flipped sinusoidal grating",
            extent_as=extents["dp_plus_grating"],
        )
        _save_eef_plot(outdir / "eef_curve.png", eef_rows)
        artifacts["quicklooks"] = [
            "dp_only_log.png",
            "dp_plus_grating_log.png",
            "eef_curve.png",
        ]

    manifest = {
        "schema": SCHEMA_VERSION,
        "created_at": created_at,
        "generator": Path(__file__).as_posix(),
        "system_preset": args.system_preset,
        "config_path": None if args.config is None else str(args.config),
        "outdir": str(outdir),
        "psf_npix": int(args.psf_npix),
        "pupil_npix": int(args.pupil_npix),
        "image_shape": list(first_image.shape),
        "pixel_scale_as_per_pix": float(pixel_scale_as),
        "image_extent_as": first_extent.tolist(),
        "half_fov_as": float(half_fov_as),
        "aperture_diameters_as": [float(x) for x in args.aperture_diameters_as],
        "aperture_oversample": int(args.aperture_oversample),
        "normalization": {
            "fractions_relative_to": "captured finite-FOV image energy",
            "raw_summed_image_totals_preserved": True,
        },
        "source": {
            "kind": "single_star",
            "placeholder": "Alpha Cen A component A flux",
            "centered_at_as": [0.0, 0.0],
            "photometry_note": ALPHA_CEN_A_PLACEHOLDER_NOTE,
        },
        "source_overrides": {
            "exposure_time_s": float(args.exposure_time_s),
            "n_lambda": int(args.n_lambda),
        },
        "grating_parameters": {
            "phase_amplitude_rad": float(args.grating_phase_amplitude_rad),
            "amplitude_opd_m": float(grating_payload["amplitude_opd_m"]),
            "amplitude_opd_nm": float(grating_payload["amplitude_opd_m"]) * 1e9,
            "dp_enabled": bool(args.dp_enabled),
            "enabled_for_cases": {"dp_only": False, "dp_plus_grating": True},
            "frequency_cycles_per_aperture": float(args.grating_frequency),
            "angle_deg": float(args.grating_angle_deg),
            "phase_flip": bool(args.grating_phase_flip),
            "mask_threshold": float(args.grating_mask_threshold),
            "binary_mask": bool(args.binary_mask),
            "legacy_dp_centering": bool(args.legacy_dp_centering),
            "phase2opd_wavelength_m": wavelength_m,
            "source_dp_path": str(grating_payload["source_dp_path"]),
        },
        "case_image_totals": {
            case_name: {
                "total_captured_energy": float(np.sum(image)),
                "image_max": float(np.max(image)),
                "image_min": float(np.min(image)),
            }
            for case_name, image in images.items()
        },
        "eef_enclosing_radii_as": eef_radii_by_case,
        "artifacts": artifacts,
        "warnings": warnings,
    }
    (outdir / "manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote PSF encircled-energy artifacts: {outdir}")
    print(f"Half FOV: {half_fov_as:.3f} arcsec; pixel scale: {pixel_scale_as:.6g} arcsec/pix")
    for warning in warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
