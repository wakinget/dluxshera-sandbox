#!/usr/bin/env python3
"""Export light-themed poster figures for the full-fidelity review notebook."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "dluxshera-matplotlib"),
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from dluxshera.config.io import load_config_file
from dluxshera.params.store import ParameterStore
from dluxshera.plot.plotting import apply_plot_defaults, get_default_cmaps
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils import full_fidelity_review as review
from dluxshera.utils.spectral_response import (
    load_response_curve_csv,
    resolve_response_curve_path,
)


DEFAULT_CONFIG = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_review.yaml"
)
DEFAULT_OUTDIR = Path("Results/poster_figures/full_fidelity_resolved_system_review")
DEFAULT_SED_FILES = {
    "Alpha Cen A": Path("src/dluxshera/data/target_seds/alfCenA_SED.dat"),
    "Alpha Cen B": Path("src/dluxshera/data/target_seds/alfCenB_SED.dat"),
}


@contextmanager
def poster_light_theme(*, dpi: int):
    """Temporarily apply explicit white-background, dark-foreground rcParams."""

    with plt.rc_context():
        matplotlib.rcdefaults()
        plt.style.use("default")
        _ = get_default_cmaps(bad_color="0.65", bad_alpha=1.0)
        apply_plot_defaults(figure_dpi=dpi)
        plt.rcParams.update(
            {
                "figure.dpi": dpi,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "savefig.edgecolor": "white",
                "savefig.transparent": False,
                "text.color": "black",
                "axes.labelcolor": "black",
                "axes.edgecolor": "black",
                "axes.titlecolor": "black",
                "xtick.color": "black",
                "ytick.color": "black",
                "grid.color": "0.85",
                "grid.linewidth": 0.8,
                "legend.facecolor": "white",
                "legend.edgecolor": "0.75",
                "legend.framealpha": 1.0,
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "image.origin": "lower",
                "image.cmap": "inferno_nan",
            }
        )
        yield


def _repo_root() -> Path:
    return review.repo_root(Path(__file__))


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    root = _repo_root()
    candidate = root / p
    return candidate.resolve()


def _formats(value: str) -> list[str]:
    aliases = {"tif": "tiff", "jpg": "jpeg"}
    formats = [item.strip().lower().lstrip(".") for item in value.split(",")]
    formats = [aliases.get(item, item) for item in formats if item]
    unique = list(dict.fromkeys(formats))
    if "png" in unique:
        unique.remove("png")
    return ["png", *unique]


def _figsize(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Figure size must be '<width>,<height>' in inches.")
    try:
        width, height = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Figure size entries must be numeric.") from exc
    if width <= 0.0 or height <= 0.0:
        raise argparse.ArgumentTypeError("Figure size entries must be positive.")
    return width, height


def _save_figure(fig: plt.Figure, outdir: Path, stem: str, formats: list[str], *, dpi: int) -> list[str]:
    paths: list[str] = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(
            path,
            dpi=dpi,
            facecolor="white",
            edgecolor="white",
            transparent=False,
            bbox_inches="tight",
        )
        paths.append(str(path))
    plt.close(fig)
    return paths


def _save_map_figure(
    *,
    data_nm: np.ndarray,
    mask: np.ndarray | None,
    extent: np.ndarray,
    axis_label: str,
    outdir: Path,
    stem: str,
    title: str,
    formats: list[str],
    dpi: int,
    figsize: tuple[float, float],
    cmap_name: str = "RdBu_r",
    symmetric_limits: bool = True,
) -> list[str]:
    masked = review.masked_for_imshow(data_nm, mask) if mask is not None else np.asarray(data_nm, dtype=float)
    finite = masked[np.isfinite(masked)]
    if finite.size == 0:
        vmin, vmax = -1.0, 1.0
    elif symmetric_limits:
        vmin, vmax = review.symmetric_nan_limits(masked, percentile=99.0)
    else:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if not vmax > vmin:
            vmin, vmax = vmin - 0.5, vmax + 0.5
    cmap = review.cmap_with_bad(cmap_name, bad="0.65")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    _style_axis(ax, grid=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("OPD (nm)", color="black")
    cbar.ax.tick_params(colors="black")
    cbar.outline.set_edgecolor("black")
    fig.tight_layout()
    return _save_figure(fig, outdir, stem, formats, dpi=dpi)


def _style_axis(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.tick_params(colors="black", which="both")
    for spine in ax.spines.values():
        spine.set_color("black")
    if grid:
        ax.grid(True, color="0.85", linewidth=0.8, alpha=0.8)


def _component_spec(config: dict[str, Any], name: str) -> dict[str, Any]:
    exp = config.get("experiment", config)
    spectral = exp.get("spectral_model", {})
    truth = spectral.get("truth", {})
    components = truth.get("components", {})
    spec = components.get(name)
    if not isinstance(spec, dict):
        raise KeyError(f"Missing spectral_model.truth.components.{name} in config.")
    return dict(spec)


def _plot_response_curve(
    *,
    spec: dict[str, Any],
    outdir: Path,
    stem: str,
    ylabel: str,
    color: str,
    wavelength_limits_nm: tuple[float, float],
    figsize: tuple[float, float],
    formats: list[str],
    dpi: int,
) -> tuple[list[str], Path]:
    path = resolve_response_curve_path(spec["path"])
    wavelengths_m, response = load_response_curve_csv(
        path,
        wavelength_column=str(spec.get("wavelength_column", "wavelength")),
        response_column=str(spec.get("response_column", "response")),
        wavelength_unit=str(spec.get("wavelength_unit", "nm")),
        response_scale=float(spec.get("response_scale", 1.0)),
        clip_negative=bool(spec.get("clip_negative", False)),
        allow_above_one=bool(spec.get("allow_above_one", False)),
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(wavelengths_m * 1e9, response, color=color, linewidth=2.0)
    ax.set_xlim(*wavelength_limits_nm)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    _style_axis(ax)
    fig.tight_layout()
    return _save_figure(fig, outdir, stem, formats, dpi=dpi), path


def _load_raw_sed(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(path, ndmin=2)
    if table.shape[1] < 2:
        raise ValueError(f"SED file must contain wavelength and flux columns: {path}")
    order = np.argsort(table[:, 0])
    return np.asarray(table[order, 0], dtype=float), np.asarray(table[order, 1], dtype=float)


def _plot_raw_seds(
    *,
    outdir: Path,
    wavelength_limits_nm: tuple[float, float],
    figsize: tuple[float, float],
    formats: list[str],
    dpi: int,
) -> tuple[list[str], dict[str, str]]:
    fig, ax = plt.subplots(figsize=figsize)
    colors = {"Alpha Cen A": "#1f77b4", "Alpha Cen B": "#d62728"}
    source_paths: dict[str, str] = {}
    for label, rel_path in DEFAULT_SED_FILES.items():
        path = _resolve_repo_path(rel_path)
        wavelength_nm, flux = _load_raw_sed(path)
        in_view = (wavelength_nm >= wavelength_limits_nm[0]) & (wavelength_nm <= wavelength_limits_nm[1])
        ax.plot(
            wavelength_nm[in_view],
            flux[in_view],
            label=label,
            color=colors[label],
            linewidth=2.0,
        )
        source_paths[label] = str(path)
    ax.set_xlim(*wavelength_limits_nm)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Raw SED flux density (W m$^{-2}$ nm$^{-1}$)")
    legend = ax.legend(loc="best", frameon=True)
    for text in legend.get_texts():
        text.set_color("black")
    _style_axis(ax)
    fig.tight_layout()
    return _save_figure(fig, outdir, "alpha_cen_raw_sed", formats, dpi=dpi), source_paths


def _extract_layer_array(layer: Any, attr: str) -> np.ndarray | None:
    if layer is None or not hasattr(layer, attr):
        return None
    arr = np.asarray(getattr(layer, attr), dtype=float)
    return arr if arr.ndim == 2 else None


def _store_get(store: ParameterStore, key: str) -> Any:
    try:
        return store.get(key)
    except Exception:
        return None


def resolved_pupil_extent(
    binder: SheraBinder,
    store: ParameterStore,
    system_cfg: dict[str, Any],
) -> tuple[np.ndarray, str, list[str], dict[str, Any]]:
    """Return imshow extent and axis label for the resolved M1 pupil plane."""

    warnings: list[str] = []
    source = "binder.base_forward_store optics.m1_diameter_m"
    diameter_m = _store_get(getattr(binder, "base_forward_store", store), "optics.m1_diameter_m")
    if diameter_m is None:
        source = "ParameterStore optics.m1_diameter_m"
        diameter_m = _store_get(store, "optics.m1_diameter_m")
    if diameter_m is None:
        source = "system_cfg optics.m1_diameter_m"
        optics_cfg = system_cfg.get("optics", {}) if isinstance(system_cfg.get("optics"), dict) else {}
        diameter_m = optics_cfg.get("m1_diameter_m")

    try:
        diameter = float(diameter_m)
    except (TypeError, ValueError):
        diameter = np.nan

    if np.isfinite(diameter) and diameter > 0.0:
        extent = 0.5 * diameter * np.array([-1.0, 1.0, -1.0, 1.0])
        return extent, "M1 pupil coordinate (m)", warnings, {
            "extent": extent.tolist(),
            "units": "m",
            "axis_label": "M1 pupil coordinate (m)",
            "source": source,
            "m1_diameter_m": diameter,
        }

    warnings.append(
        "Could not resolve optics.m1_diameter_m from binder/store/system config; "
        "using normalized pupil coordinates."
    )
    extent = np.array([-1.0, 1.0, -1.0, 1.0])
    return extent, "Normalized pupil coordinate", warnings, {
        "extent": extent.tolist(),
        "units": "normalized",
        "axis_label": "Normalized pupil coordinate",
        "source": "fallback",
    }


def resolved_mirror_extent(
    store: ParameterStore,
    system_cfg: dict[str, Any],
    mirror: str,
) -> tuple[np.ndarray, str, list[str], dict[str, Any]]:
    """Return a physical imshow extent for a primary or secondary WFE map."""

    key = "optics.m1_diameter_m" if mirror == "primary" else "optics.m2_diameter_m"
    label = "M1 pupil coordinate (m)" if mirror == "primary" else "M2 pupil coordinate (m)"
    warnings: list[str] = []
    source = f"ParameterStore {key}"
    diameter_m = _store_get(store, key)
    if diameter_m is None:
        source = f"system_cfg {key}"
        optics_cfg = system_cfg.get("optics", {}) if isinstance(system_cfg.get("optics"), dict) else {}
        diameter_m = optics_cfg.get(key.split(".")[-1])
    try:
        diameter = float(diameter_m)
    except (TypeError, ValueError):
        diameter = np.nan
    if np.isfinite(diameter) and diameter > 0.0:
        extent = 0.5 * diameter * np.array([-1.0, 1.0, -1.0, 1.0])
        return extent, label, warnings, {
            "extent": extent.tolist(),
            "units": "m",
            "axis_label": label,
            "source": source,
            "diameter_m": diameter,
        }
    warnings.append(f"Could not resolve {key}; using normalized {mirror} pupil coordinates.")
    extent = np.array([-1.0, 1.0, -1.0, 1.0])
    return extent, "Normalized pupil coordinate", warnings, {
        "extent": extent.tolist(),
        "units": "normalized",
        "axis_label": "Normalized pupil coordinate",
        "source": "fallback",
    }


def _build_resolved_context(
    config: dict[str, Any],
    outdir: Path,
) -> dict[str, Any]:
    split_ctx = review.build_model_split_from_smoke(
        config,
        outdir / "_model_split_for_poster",
        run_label="poster_figures",
        write_artifacts=True,
    )
    system_cfg = split_ctx["truth_system_cfg"]
    spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    binder = SheraBinder(system_cfg, spec, store)
    return {
        "split_ctx": split_ctx,
        "split": split_ctx["model_split"],
        "system_cfg": system_cfg,
        "spec": spec,
        "store": store,
        "binder": binder,
    }


def _build_opd_from_resolved_model(
    resolved_ctx: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any], list[str]]:
    system_cfg = resolved_ctx["system_cfg"]
    store = resolved_ctx["store"]
    binder = resolved_ctx["binder"]
    optics = binder.telescope.optics
    aperture = _extract_layer_array(getattr(optics, "m1_aperture", None), "transmission")
    opd_m = _extract_layer_array(getattr(optics, "dp", None), "opd")
    if aperture is None or opd_m is None:
        raise RuntimeError("Could not extract m1_aperture.transmission and dp.opd from resolved optics.")
    mask = aperture > 0.0
    extent, axis_label, warnings, extent_metadata = resolved_pupil_extent(binder, store, system_cfg)
    metadata = {
        "system_preset": system_cfg.get("preset"),
        "optics_kind": (system_cfg.get("optics") or {}).get("kind"),
        "dp_path": (system_cfg.get("optics") or {}).get("dp_path"),
        "dp_design_wavelength_m": (system_cfg.get("optics") or {}).get("dp_design_wavelength_m"),
        "aperture_mask": {
            "source": "resolved optics layer",
            "layer": "m1_aperture",
            "attribute": "transmission",
            "shape": list(aperture.shape),
            "valid_fraction": float(np.mean(mask)),
        },
        "diffractive_pupil_opd": {
            "source": "resolved optics layer",
            "layer": "dp",
            "attribute": "opd",
            "input_path": (system_cfg.get("optics") or {}).get("dp_path"),
            "display_unit": "nm",
            "shape": list(opd_m.shape),
        },
        "pupil_extent": extent_metadata,
    }
    return opd_m * 1e9, mask, extent, axis_label, metadata, warnings


def _plot_diffractive_pupil_opd(
    *,
    resolved_ctx: dict[str, Any],
    outdir: Path,
    cmap_name: str,
    figsize: tuple[float, float],
    formats: list[str],
    dpi: int,
) -> tuple[list[str], dict[str, Any], list[str]]:
    opd_nm, mask, extent, axis_label, metadata, warnings = _build_opd_from_resolved_model(resolved_ctx)
    masked = review.masked_for_imshow(opd_nm, mask)
    finite = masked[np.isfinite(masked)]
    vmin = float(np.nanmin(finite)) if finite.size else 0.0
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    if not vmax > vmin:
        vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite))
    cmap = review.cmap_with_bad(cmap_name, bad="0.65")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax.set_aspect("equal")
    ax.set_title("Diffractive Pupil OPD")
    ax.set_xlabel(axis_label)
    ax.set_ylabel(axis_label)
    _style_axis(ax, grid=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("OPD (nm)", color="black")
    cbar.ax.tick_params(colors="black")
    cbar.outline.set_edgecolor("black")
    metadata["color_limits_nm"] = [vmin, vmax]
    metadata["colormap"] = cmap_name
    fig.tight_layout()
    return _save_figure(fig, outdir, "diffractive_pupil_opd", formats, dpi=dpi), metadata, warnings


WFE_MAP_PANELS = [
    ("raw_ptt_removed_truth_opd_nm", "full_truth_opd_nm", "Raw PTT-removed truth OPD"),
    ("low_order_truth_reconstruction_nm", "low_order_truth_reconstruction_nm", "Low-order Zernike reconstruction"),
    ("truth_high_order_residual_opd_nm", "high_order_truth_opd_nm", "Truth high-order residual OPD"),
    ("knowledge_error_high_order_residual_opd_nm", "high_order_error_opd_nm", "High-order knowledge-error residual OPD"),
    ("inference_high_order_opd_nm", "inference_high_order_opd_nm", "Inference high-order OPD"),
    ("inference_sum_residual_nm", "inference_sum_residual_nm", "Inference - truth residual - error"),
]


def _plot_wfe_review_maps(
    *,
    resolved_ctx: dict[str, Any],
    outdir: Path,
    formats: list[str],
    dpi: int,
    figsize: tuple[float, float],
) -> tuple[dict[str, list[str]], dict[str, Any], list[str]]:
    split = resolved_ctx["split"]
    store = resolved_ctx["store"]
    system_cfg = resolved_ctx["system_cfg"]
    wfe_summary = review.summarize_wfe_artifacts(split)
    exports: dict[str, list[str]] = {}
    metadata: dict[str, Any] = {
        "enabled": bool(wfe_summary.get("enabled", False)),
        "mirrors": {},
    }
    warnings = list(wfe_summary.get("warnings", []))
    if not wfe_summary.get("enabled", False):
        return exports, metadata, warnings

    for mirror, item in wfe_summary.get("mirrors", {}).items():
        mirror_extent, axis_label, extent_warnings, extent_metadata = resolved_mirror_extent(store, system_cfg, mirror)
        warnings.extend(f"{mirror}: {warning}" for warning in extent_warnings)
        mask = np.asarray(item.get("mask"), dtype=bool) if item.get("mask") is not None else None
        mirror_meta: dict[str, Any] = {
            "extent": extent_metadata,
            "mask_source": "review.summarize_wfe_artifacts(...)[mirror]['mask']",
            "panels": {},
        }
        for key, suffix, title in WFE_MAP_PANELS:
            if key not in item:
                warnings.append(f"{mirror}: WFE panel {key} is unavailable; skipped.")
                continue
            arr = np.asarray(item[key], dtype=float)
            stem = f"{mirror}_{suffix}"
            files = _save_map_figure(
                data_nm=arr,
                mask=mask,
                extent=mirror_extent,
                axis_label=axis_label,
                outdir=outdir,
                stem=stem,
                title=f"{mirror.capitalize()} {title}",
                formats=formats,
                dpi=dpi,
                figsize=figsize,
                cmap_name="RdBu_r",
                symmetric_limits=True,
            )
            exports[stem] = files
            mirror_meta["panels"][stem] = {
                "summary_key": key,
                "title": title,
                "shape": list(arr.shape),
                "display_unit": "nm",
            }
        metadata["mirrors"][mirror] = mirror_meta
    return exports, metadata, warnings


def _collect_source_artifacts(outdir: Path) -> dict[str, list[str]]:
    root = outdir / "_model_split_for_poster"
    patterns = {
        "high_order_wfe_fits": "model_split/high_order_wfe/maps/*.fits",
        "high_order_wfe_manifests": "model_split/high_order_wfe/maps/*.json",
        "high_order_wfe_config_maps": "model_split/high_order_wfe/config_maps/*",
    }
    artifacts: dict[str, list[str]] = {}
    for label, pattern in patterns.items():
        artifacts[label] = [str(path) for path in sorted(root.glob(pattern)) if path.is_file()]
    return artifacts


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def export_figures(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_repo_path(args.config)
    outdir = _resolve_repo_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figure_dir = outdir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    config = dict(load_config_file(config_path))
    formats = _formats(args.formats)
    wavelength_limits_nm = (float(args.wavelength_min_nm), float(args.wavelength_max_nm))
    line_figsize = tuple(args.line_figsize)
    opd_figsize = tuple(args.opd_figsize)

    manifest: dict[str, Any] = {
        "config_path": str(config_path),
        "output_directory": str(outdir),
        "figure_directory": str(figure_dir),
        "wavelength_limits_nm": list(wavelength_limits_nm),
        "dpi": int(args.dpi),
        "formats": formats,
        "figure_sizes_in": {
            "line": list(line_figsize),
            "opd": list(opd_figsize),
        },
        "plotting_conventions": {
            "source": "examples/recipes/canonical_astrometry.py",
            "patterns": [
                "get_default_cmaps()",
                "apply_plot_defaults()",
                "plt.rcParams['image.cmap'] = 'inferno_nan'",
                "resolved physical imshow extents from ParameterStore/system config",
            ],
        },
        "figure_exports": {},
        "source_data_paths": {},
        "source_artifacts": {},
        "git_commit": _git_commit(),
        "warnings": [],
    }

    with poster_light_theme(dpi=int(args.dpi)):
        resolved_ctx: dict[str, Any] | None = None
        m2_spec = _component_spec(config, "m2_filter_response")
        files, source = _plot_response_curve(
            spec=m2_spec,
            outdir=figure_dir,
            stem="m2_filter_response",
            ylabel="M2 reflectance / response",
            color="#2a6f97",
            wavelength_limits_nm=wavelength_limits_nm,
            figsize=line_figsize,
            formats=formats,
            dpi=int(args.dpi),
        )
        manifest["figure_exports"]["m2_filter_response"] = files
        manifest["source_data_paths"]["m2_filter_response"] = str(source)

        qe_spec = _component_spec(config, "detector_qe")
        files, source = _plot_response_curve(
            spec=qe_spec,
            outdir=figure_dir,
            stem="detector_qe",
            ylabel="Detector QE / response",
            color="#7f4f24",
            wavelength_limits_nm=wavelength_limits_nm,
            figsize=line_figsize,
            formats=formats,
            dpi=int(args.dpi),
        )
        manifest["figure_exports"]["detector_qe"] = files
        manifest["source_data_paths"]["detector_qe"] = str(source)

        files, sed_paths = _plot_raw_seds(
            outdir=figure_dir,
            wavelength_limits_nm=wavelength_limits_nm,
            figsize=line_figsize,
            formats=formats,
            dpi=int(args.dpi),
        )
        manifest["figure_exports"]["alpha_cen_raw_sed"] = files
        manifest["source_data_paths"]["raw_target_seds"] = sed_paths

        try:
            resolved_ctx = _build_resolved_context(config, outdir)
            files, metadata, warnings = _plot_diffractive_pupil_opd(
                resolved_ctx=resolved_ctx,
                outdir=figure_dir,
                cmap_name=str(args.dp_opd_cmap),
                figsize=opd_figsize,
                formats=formats,
                dpi=int(args.dpi),
            )
            manifest["figure_exports"]["diffractive_pupil_opd"] = files
            manifest["source_data_paths"]["diffractive_pupil_opd"] = metadata
            manifest["warnings"].extend(warnings)
        except Exception as exc:
            manifest["warnings"].append(f"Skipped diffractive_pupil_opd: {exc}")

        if resolved_ctx is not None:
            try:
                files_by_panel, metadata, warnings = _plot_wfe_review_maps(
                    resolved_ctx=resolved_ctx,
                    outdir=figure_dir,
                    formats=formats,
                    dpi=int(args.dpi),
                    figsize=opd_figsize,
                )
                manifest["figure_exports"].update(files_by_panel)
                manifest["source_data_paths"]["high_order_wfe_review_maps"] = metadata
                manifest["warnings"].extend(warnings)
            except Exception as exc:
                manifest["warnings"].append(f"Skipped high-order WFE review map exports: {exc}")
            manifest["source_artifacts"] = _collect_source_artifacts(outdir)

    manifest_path = outdir / "poster_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    for group, files in manifest["figure_exports"].items():
        print(f"{group}: {', '.join(Path(path).name for path in files)}")
    for warning in manifest["warnings"]:
        print(f"WARNING: {warning}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export light-themed poster figures for the full-fidelity resolved-system review.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Review YAML config path.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Directory for generated poster figures.")
    parser.add_argument("--wavelength-min-nm", type=float, default=450.0, help="Minimum plotted wavelength in nm.")
    parser.add_argument("--wavelength-max-nm", type=float, default=650.0, help="Maximum plotted wavelength in nm.")
    parser.add_argument("--dpi", type=int, default=450, help="Raster export DPI.")
    parser.add_argument("--formats", default="png", help="Comma-separated formats for line plots; png is always included.")
    parser.add_argument("--dp-opd-cmap", default="inferno", help="Matplotlib colormap for the diffractive-pupil OPD map.")
    parser.add_argument("--line-figsize", type=_figsize, default=(6.0, 3.5), help="Line-plot figure size as '<width>,<height>' inches.")
    parser.add_argument("--opd-figsize", type=_figsize, default=(5.3, 4.8), help="OPD figure size as '<width>,<height>' inches.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True, help="Overwrite existing outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.overwrite:
        outdir = _resolve_repo_path(args.outdir)
        manifest = outdir / "poster_figure_manifest.json"
        if manifest.exists():
            raise FileExistsError(f"Refusing to overwrite existing poster exports: {manifest}")
    export_figures(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
