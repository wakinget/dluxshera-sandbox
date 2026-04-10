"""Render one curated binary target as portrait-style artifacts.

This scratch utility is the single-target variant of
``work/scratch/source_target_montage.py``. It preserves the same registry-backed
binary-target build path, but writes a small artifact set for one selected
target instead of a multi-panel montage.

Usage
-----
    python work/scratch/target_portrait.py --target ALPHA_CEN
    python work/scratch/target_portrait.py --target 61_CYG --stretch sqrt
    python work/scratch/target_portrait.py --target XI_BOO --normalize-total-flux
    python work/scratch/target_portrait.py \
        --target ALPHA_CEN \
        --config work/scratch/target_portrait_example.yaml

Use ``--config`` for structural system edits such as ``optics.psf_npix`` or
``detector.layers``. The config file should contain a top-level ``system`` block
and may optionally include an unused ``experiment`` block.

Output
------
By default artifacts are written under:

    work/scratch/Results/target_portrait/<target>_<timestamp>/

Each run writes the raw PSF FITS array, a clean portrait PNG, an annotated PNG
with arcsecond axes and metadata, and a compact ``manifest.json``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import unicodedata

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm

from dluxshera.components.sources import TARGET_SPECS, TargetSpec
from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.spec import ParamSpec
from dluxshera.params.store import ParameterStore
from dluxshera.plot.obs_subblock import apply_intensity_stretch
from dluxshera.systems.base import SheraBinder, compose_forward_spec

SYSTEM_PRESET = "SHERA_FLIGHT_3P"
DEFAULT_STRETCH = "log"
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "Results" / "target_portrait"
DISPLAY_PMIN = 1.0
DISPLAY_PMAX = 100.0
CMAP = "inferno"
TARGET_AUTHORITY_OVERRIDE_KEYS = (
    "contrast",
    "log_flux_total",
    "position_angle_deg",
    "separation_as",
    "vmag_a",
    "vmag_b",
)


@dataclass(frozen=True)
class TargetBuildResult:
    """Rendered target payload produced by the registry-backed build path."""

    image: np.ndarray
    store: ParameterStore
    system_cfg: dict[str, Any]
    forward_spec: ParamSpec


def _timestamp_tag(*, now: dt.datetime | None = None) -> str:
    current = now or dt.datetime.now()
    return current.strftime("%Y%m%d-%H%M%S")


def _created_at(*, now: dt.datetime | None = None) -> str:
    current = now or dt.datetime.now()
    return current.isoformat(timespec="seconds")


def _validate_target_key(raw_target: str) -> str:
    """Return the canonical target key, or raise a clear validation error."""

    target_key = raw_target.strip().upper()
    if target_key in TARGET_SPECS:
        return target_key
    known = ", ".join(sorted(TARGET_SPECS))
    raise ValueError(f"Unknown target {raw_target!r}. Available target keys: {known}.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for the single-target portrait workflow."""

    parser = argparse.ArgumentParser(
        description="Render one curated binary target through the current Shera model stack.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Curated target registry key from TARGET_SPECS.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML/JSON config with top-level system structural overrides.",
    )
    parser.add_argument(
        "--system-preset",
        default=SYSTEM_PRESET,
        help="System preset to load before target-specific source overrides are applied.",
    )
    parser.add_argument(
        "--normalize-total-flux",
        action="store_true",
        help="Normalize the target to total flux = 1 (source.log_flux_total = 0).",
    )
    parser.add_argument(
        "--stretch",
        choices=("linear", "sqrt", "log"),
        default=DEFAULT_STRETCH,
        help="Display stretch applied to PNG artifacts.",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Lower display bound. When omitted, a robust per-image bound is used.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Upper display bound. When omitted, a robust per-image bound is used.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Root directory for target portrait runs.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run subdirectory name. Defaults to <target>_<timestamp>.",
    )

    args = parser.parse_args(argv)
    try:
        args.target = _validate_target_key(args.target)
    except ValueError as exc:
        parser.error(str(exc))

    if args.vmin is not None and not np.isfinite(args.vmin):
        parser.error("--vmin must be finite.")
    if args.vmax is not None and not np.isfinite(args.vmax):
        parser.error("--vmax must be finite.")
    if args.vmin is not None and args.vmax is not None and args.vmax <= args.vmin:
        parser.error("--vmax must be larger than --vmin.")
    if args.stretch == "log" and args.vmin is not None and args.vmin <= 0.0:
        parser.error("--vmin must be > 0 for --stretch log.")

    return args


def _resolve_system_cfg(
    *,
    config_path: Path | None,
    system_preset: str | None,
) -> dict[str, Any]:
    """Load and resolve the top-level system config used for rendering."""

    user_cfg = load_user_config(
        config_path=config_path,
        system_preset=system_preset,
        experiment_preset=None,
    )
    resolved = resolve_config(user_cfg)
    if "system" not in resolved:
        raise ValueError("Target portrait rendering requires a resolved top-level 'system' block.")
    return resolved["system"]


def _resolve_outdir(
    *,
    results_dir: Path,
    run_name: str | None,
    target_key: str,
    timestamp: str,
) -> Path:
    """Return the output directory for one portrait run."""

    resolved_run_name = run_name if run_name else f"{target_key}_{timestamp}"
    return Path(results_dir).expanduser() / resolved_run_name


def _artifact_paths(
    *,
    outdir: Path,
    target_key: str,
    timestamp: str,
) -> dict[str, Path]:
    prefix = f"{target_key}_{timestamp}"
    return {
        "psf_fits": outdir / f"{prefix}_psf.fits",
        "clean_png": outdir / f"{prefix}_portrait.png",
        "annotated_png": outdir / f"{prefix}_annotated.png",
        "manifest_json": outdir / "manifest.json",
    }


def _resolve_display_limits(
    image: np.ndarray,
    *,
    stretch: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float, float]:
    """Return validated display limits, using robust per-image defaults as needed."""

    values = np.asarray(image, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Rendered image contains no finite values.")

    if stretch == "log":
        positive = finite[finite > 0.0]
        if positive.size == 0:
            raise ValueError("Log display requires at least one positive finite image value.")
        auto_vmin = float(np.percentile(positive, DISPLAY_PMIN))
        auto_vmax = float(np.percentile(positive, DISPLAY_PMAX))
        positive_floor = float(np.min(positive))
        if auto_vmin <= 0.0:
            auto_vmin = max(positive_floor, np.finfo(float).tiny)
    else:
        auto_vmin = float(np.percentile(finite, DISPLAY_PMIN))
        auto_vmax = float(np.percentile(finite, DISPLAY_PMAX))

    resolved_vmin = auto_vmin if vmin is None else float(vmin)
    resolved_vmax = auto_vmax if vmax is None else float(vmax)

    if not np.isfinite(resolved_vmin) or not np.isfinite(resolved_vmax):
        raise ValueError("Display limits must be finite.")
    if stretch == "log" and resolved_vmin <= 0.0:
        raise ValueError("Log display requires vmin > 0.")
    if resolved_vmax <= resolved_vmin:
        if vmin is None and vmax is None:
            delta = max(abs(resolved_vmin) * 1e-6, 1e-12)
            resolved_vmax = resolved_vmin + delta
        else:
            raise ValueError("Display limits must satisfy vmax > vmin.")

    return resolved_vmin, resolved_vmax


def _prepare_target_system_cfg(
    base_system_cfg: Mapping[str, Any],
    *,
    target_key: str,
) -> dict[str, Any]:
    """Clone a system config and apply registry-authoritative target settings."""

    system_cfg = deepcopy(dict(base_system_cfg))
    source_cfg = system_cfg.setdefault("source", {})
    if not isinstance(source_cfg, dict):
        raise ValueError("Expected 'system.source' to be a mapping.")

    source_cfg["kind"] = "binary_target"
    source_cfg["target"] = target_key
    for key in TARGET_AUTHORITY_OVERRIDE_KEYS:
        source_cfg.pop(key, None)

    return system_cfg


def _build_forward_store(
    system_cfg: Mapping[str, Any],
    *,
    normalize_total_flux: bool,
) -> tuple[ParamSpec, ParameterStore]:
    """Build the forward spec and seeded store for one prepared target config."""

    forward_spec = compose_forward_spec({"system": system_cfg})
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    if normalize_total_flux:
        store = store.replace({"source.log_flux_total": 0.0})
    return forward_spec, store


def _render_image(
    system_cfg: Mapping[str, Any],
    forward_spec: ParamSpec,
    store: ParameterStore,
) -> np.ndarray:
    """Render the raw PSF image through the Shera binder."""

    binder = SheraBinder(system_cfg, forward_spec, store)
    return np.asarray(binder.model(binder.strip_structural(store)))


def _build_target_image(
    base_system_cfg: Mapping[str, Any],
    *,
    target_key: str,
    normalize_total_flux: bool,
) -> TargetBuildResult:
    """Build one target-specific PSF image plus its seeded forward store."""

    system_cfg = _prepare_target_system_cfg(
        base_system_cfg,
        target_key=target_key,
    )
    forward_spec, store = _build_forward_store(
        system_cfg,
        normalize_total_flux=normalize_total_flux,
    )
    image = _render_image(system_cfg, forward_spec, store)
    return TargetBuildResult(
        image=image,
        store=store,
        system_cfg=system_cfg,
        forward_spec=forward_spec,
    )


def _as_scalar(value: Any) -> Any:
    """Convert numpy/JAX scalar-like values to JSON/formatting-friendly scalars."""

    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.tolist()


def _store_float(store: ParameterStore, key: str) -> float:
    return float(_as_scalar(store.get(key)))


def _store_int(store: ParameterStore, key: str, *, default: Any | None = None) -> int:
    value = store.get(key, default) if default is not None else store.get(key)
    return int(_as_scalar(value))


def _source_summary(store: ParameterStore) -> dict[str, float | int]:
    """Return a compact summary of resolved source/store values."""

    return {
        "separation_as": _store_float(store, "source.separation_as"),
        "position_angle_deg": _store_float(store, "source.position_angle_deg"),
        "contrast": _store_float(store, "source.contrast"),
        "log_flux_total": _store_float(store, "source.log_flux_total"),
        "plate_scale_as_per_pix": _store_float(store, "optics.plate_scale_as_per_pix"),
        "psf_npix": _store_int(store, "optics.psf_npix"),
    }


def _image_extent_as(store: ParameterStore, image: np.ndarray) -> np.ndarray:
    """Return the image extent in arcseconds using montage-compatible logic."""

    psf_npix = int(store.get("optics.psf_npix", default=image.shape[-1]))
    plate_scale_as_per_pix = _store_float(store, "optics.plate_scale_as_per_pix")
    return psf_npix * plate_scale_as_per_pix / 2.0 * np.array([-1.0, 1.0, -1.0, 1.0])


def _imshow_stretched(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    stretch: str,
    vmin: float,
    vmax: float,
    extent: np.ndarray | None = None,
) -> Any:
    """Draw an image using the requested display stretch."""

    if stretch == "log":
        norm = LogNorm(vmin=float(vmin), vmax=float(vmax))
        return ax.imshow(image, origin="lower", cmap=CMAP, norm=norm, extent=extent)

    stretched = apply_intensity_stretch(
        image,
        vmin=float(vmin),
        vmax=float(vmax),
        stretch=stretch,
    )
    return ax.imshow(
        stretched,
        origin="lower",
        cmap=CMAP,
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )


def _fits_ascii(value: str) -> str:
    """Return a FITS-header-safe ASCII representation of a display string."""

    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    printable = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in ascii_text)
    return " ".join(printable.split()) or "UNKNOWN"


def _write_fits(
    *,
    output_path: Path,
    image: np.ndarray,
    target_key: str,
    spec: TargetSpec,
    summary: Mapping[str, float | int],
) -> None:
    """Write the un-stretched raw PSF array to FITS."""

    header = fits.Header()
    header.set("TARGET", target_key)
    header.set("NAME", _fits_ascii(spec.display_name), "ASCII-normalized target name")
    header.set("SEP_AS", float(summary["separation_as"]))
    header.set("PA_DEG", float(summary["position_angle_deg"]))
    header.set("CONTR", float(summary["contrast"]))
    header.set("LOGFLUX", float(summary["log_flux_total"]))
    header.set("PSFNPIX", int(summary["psf_npix"]))
    header.set("PLTSCALE", float(summary["plate_scale_as_per_pix"]))
    fits.PrimaryHDU(data=np.asarray(image), header=header).writeto(output_path, overwrite=True)


def _save_clean_portrait(
    *,
    output_path: Path,
    image: np.ndarray,
    stretch: str,
    vmin: float,
    vmax: float,
) -> None:
    """Save a presentation-ready image with no axes or overlays."""

    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    _imshow_stretched(ax, image, stretch=stretch, vmin=vmin, vmax=vmax)
    ax.set_axis_off()
    fig.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def _format_annotated_title(
    *,
    spec: TargetSpec,
    target_key: str,
    summary: Mapping[str, float | int],
    normalize_total_flux: bool,
    stretch: str,
) -> str:
    flux_mode = "normalized total flux" if normalize_total_flux else "curated total flux"
    return (
        f"{spec.display_name} ({target_key})\n"
        f"sep={float(summary['separation_as']):.3f} as, "
        f"PA={float(summary['position_angle_deg']):.3f} deg, "
        f"contrast={float(summary['contrast']):.3g}, "
        f"logF={float(summary['log_flux_total']):.3f}\n"
        f"{flux_mode}, stretch={stretch}"
    )


def _save_annotated_plot(
    *,
    output_path: Path,
    image: np.ndarray,
    store: ParameterStore,
    target_key: str,
    spec: TargetSpec,
    summary: Mapping[str, float | int],
    normalize_total_flux: bool,
    stretch: str,
    vmin: float,
    vmax: float,
) -> None:
    """Save an inspection plot with arcsecond axes, colorbar, and metadata title."""

    extent = _image_extent_as(store, image)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    im = _imshow_stretched(ax, image, stretch=stretch, vmin=vmin, vmax=vmax, extent=extent)
    ax.set_xlabel("X [arcsec]")
    ax.set_ylabel("Y [arcsec]")
    ax.set_title(
        _format_annotated_title(
            spec=spec,
            target_key=target_key,
            summary=summary,
            normalize_total_flux=normalize_total_flux,
            stretch=stretch,
        ),
        fontsize=10,
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PSF intensity" if stretch == "log" else f"{stretch} stretched intensity")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_manifest(
    *,
    output_path: Path,
    outdir: Path,
    created_at: str,
    target_key: str,
    spec: TargetSpec,
    config_path: Path | None,
    system_preset: str,
    resolved_system_preset: str | None,
    normalize_total_flux: bool,
    stretch: str,
    vmin: float,
    vmax: float,
    summary: Mapping[str, float | int],
    artifacts: Mapping[str, Path],
) -> dict[str, Any]:
    """Write and return a compact JSON manifest for the portrait run."""

    manifest: dict[str, Any] = {
        "created_at": created_at,
        "script": Path(__file__).as_posix(),
        "target_key": target_key,
        "target_display_name": spec.display_name,
        "config_path": None if config_path is None else str(config_path),
        "requested_system_preset": system_preset,
        "resolved_system_preset": resolved_system_preset,
        "normalize_total_flux": bool(normalize_total_flux),
        "stretch": stretch,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "artifacts": {
            name: Path(path).resolve().relative_to(outdir.resolve()).as_posix()
            for name, path in artifacts.items()
        },
        "resolved": dict(summary),
    }
    output_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_artifacts(
    *,
    outdir: Path,
    target_key: str,
    spec: TargetSpec,
    image: np.ndarray,
    store: ParameterStore,
    config_path: Path | None,
    system_preset: str,
    resolved_system_preset: str | None,
    normalize_total_flux: bool,
    stretch: str,
    vmin: float,
    vmax: float,
    timestamp: str,
    created_at: str,
) -> dict[str, Path]:
    """Write FITS, PNGs, and manifest for one target portrait run."""

    outdir.mkdir(parents=True, exist_ok=True)
    paths = _artifact_paths(outdir=outdir, target_key=target_key, timestamp=timestamp)
    summary = _source_summary(store)

    _write_fits(
        output_path=paths["psf_fits"],
        image=image,
        target_key=target_key,
        spec=spec,
        summary=summary,
    )
    _save_clean_portrait(
        output_path=paths["clean_png"],
        image=image,
        stretch=stretch,
        vmin=vmin,
        vmax=vmax,
    )
    _save_annotated_plot(
        output_path=paths["annotated_png"],
        image=image,
        store=store,
        target_key=target_key,
        spec=spec,
        summary=summary,
        normalize_total_flux=normalize_total_flux,
        stretch=stretch,
        vmin=vmin,
        vmax=vmax,
    )
    _write_manifest(
        output_path=paths["manifest_json"],
        outdir=outdir,
        created_at=created_at,
        target_key=target_key,
        spec=spec,
        config_path=config_path,
        system_preset=system_preset,
        resolved_system_preset=resolved_system_preset,
        normalize_total_flux=normalize_total_flux,
        stretch=stretch,
        vmin=vmin,
        vmax=vmax,
        summary=summary,
        artifacts=paths,
    )
    return paths


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    timestamp = _timestamp_tag()
    created_at = _created_at()
    outdir = _resolve_outdir(
        results_dir=args.results_dir,
        run_name=args.run_name,
        target_key=args.target,
        timestamp=timestamp,
    )

    system_cfg = _resolve_system_cfg(
        config_path=args.config,
        system_preset=args.system_preset,
    )
    resolved_system_preset = system_cfg.get("preset")

    result = _build_target_image(
        system_cfg,
        target_key=args.target,
        normalize_total_flux=args.normalize_total_flux,
    )
    vmin, vmax = _resolve_display_limits(
        result.image,
        stretch=args.stretch,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    spec = TARGET_SPECS[args.target]
    paths = _write_artifacts(
        outdir=outdir,
        target_key=args.target,
        spec=spec,
        image=result.image,
        store=result.store,
        config_path=args.config,
        system_preset=args.system_preset,
        resolved_system_preset=(
            str(resolved_system_preset) if resolved_system_preset is not None else None
        ),
        normalize_total_flux=args.normalize_total_flux,
        stretch=args.stretch,
        vmin=vmin,
        vmax=vmax,
        timestamp=timestamp,
        created_at=created_at,
    )

    summary = _source_summary(result.store)
    print(f"Saved target portrait artifacts to {outdir}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(
        "  target: "
        f"{spec.display_name} ({args.target}), "
        f"sep={summary['separation_as']:.3f} as, "
        f"PA={summary['position_angle_deg']:.3f} deg, "
        f"contrast={summary['contrast']:.3g}, "
        f"logF={summary['log_flux_total']:.3f}"
    )


if __name__ == "__main__":
    main()
