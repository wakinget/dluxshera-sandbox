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
from dluxshera.plot.plotting import (
    apply_plot_defaults,
    get_default_cmaps,
    plot_pixel_offset_maps,
    plot_pixel_response_maps,
)
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
SUPPORTED_FIGURE_FORMATS = {"png", "tiff", "pdf", "jpeg"}
EXPORT_SECTIONS = (
    "spectral",
    "sed",
    "dp_opd",
    "high_order_wfe",
    "trajectory",
    "detector_calibration",
)


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
    if not formats:
        raise argparse.ArgumentTypeError("--formats must include at least one format.")
    unsupported = sorted(set(formats) - SUPPORTED_FIGURE_FORMATS)
    if unsupported:
        allowed = ", ".join(sorted(SUPPORTED_FIGURE_FORMATS | set(aliases)))
        raise argparse.ArgumentTypeError(
            f"Unsupported figure format(s): {', '.join(unsupported)}. Allowed formats: {allowed}."
        )
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
    outdir.mkdir(parents=True, exist_ok=True)
    _style_figure(fig)
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        path.parent.mkdir(parents=True, exist_ok=True)
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


def _safe_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value)).strip("_").lower()


def _style_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor("white")
    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.tick_params(colors="black", which="both")
        for spine in ax.spines.values():
            spine.set_color("black")
        ax.title.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("0.75")
            for text in legend.get_texts():
                text.set_color("black")


def _record_exports(manifest: dict[str, Any], category: str, name: str, files: list[str]) -> None:
    manifest["figure_exports"][name] = files
    manifest["figure_exports_by_category"].setdefault(category, {})[name] = files


def _mark_completed(manifest: dict[str, Any], category: str) -> None:
    if category not in manifest["export_sections_completed"]:
        manifest["export_sections_completed"].append(category)


def _mark_skipped(manifest: dict[str, Any], category: str, reason: str) -> None:
    manifest["export_sections_skipped"][category] = reason
    manifest["warnings"].append(f"Skipped {category}: {reason}")


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


def _plot_spectral_review_figures(
    *,
    resolved_ctx: dict[str, Any],
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> tuple[dict[str, list[str]], dict[str, Any], list[str]]:
    split_ctx = resolved_ctx["split_ctx"]
    translated = split_ctx["translated_config"]
    base_system = split_ctx["base_system_cfg"]
    truth_system = split_ctx["truth_system_cfg"]
    inference_system = split_ctx["inference_system_cfg"]
    tables = review.spectral_review_tables(base_system, truth_system, inference_system)
    responses = review.response_curve_review(translated["experiment"].get("spectral_model"))
    truth_rows = list(tables.get("truth", []))
    inference_rows = list(tables.get("inference", []))
    exports: dict[str, list[str]] = {}
    warnings: list[str] = []

    def rows_for_component(rows: list[dict[str, Any]], component: str) -> list[dict[str, Any]]:
        return sorted((row for row in rows if row.get("component") == component), key=lambda row: row["wavelength_nm"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (label, response) in zip(axes[0], responses.items()):
        if response["available"]:
            ax.plot(response["wavelengths_nm"], response["response"], color="#2a6f97", linewidth=2.0)
        else:
            warnings.append(f"{label} response unavailable: {response.get('error')}")
        ax.set_title(f"{label}: {'active' if response['enabled'] else 'available but not active'}")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Response")
        _style_axis(ax)

    for role, rows, ax in [("truth", truth_rows, axes[1, 0]), ("inference", inference_rows, axes[1, 1])]:
        components = sorted({str(row["component"]) for row in rows})
        for component in components:
            group = rows_for_component(rows, component)
            ax.plot(
                [row["wavelength_nm"] for row in group],
                [row["weight"] for row in group],
                marker="o",
                linewidth=1.8,
                label=component,
            )
        ax.set_title(f"Effective {role} component weights")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized sample weight")
        _style_axis(ax)
        if components:
            ax.legend()
    fig.tight_layout()
    exports["spectral_response_and_weights"] = _save_figure(
        fig, outdir, "spectral_response_and_weights", formats, dpi=dpi
    )

    components = sorted({str(row["component"]) for row in truth_rows + inference_rows})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for component in components:
        truth = rows_for_component(truth_rows, component)
        inference = rows_for_component(inference_rows, component)
        if truth:
            axes[0].plot(
                [row["wavelength_nm"] for row in truth],
                [row["weight"] for row in truth],
                marker="o",
                linewidth=1.8,
                label=f"truth {component}",
            )
        if inference:
            axes[0].plot(
                [row["wavelength_nm"] for row in inference],
                [row["weight"] for row in inference],
                marker="s",
                linestyle="--",
                linewidth=1.8,
                label=f"inference {component}",
            )
    if {"primary", "secondary"} <= set(components):
        for role, rows, marker in [("truth", truth_rows, "o"), ("inference", inference_rows, "s")]:
            primary = rows_for_component(rows, "primary")
            secondary = rows_for_component(rows, "secondary")
            if len(primary) == len(secondary) and primary:
                axes[1].plot(
                    [row["wavelength_nm"] for row in primary],
                    [p["weight"] - s["weight"] for p, s in zip(primary, secondary)],
                    marker=marker,
                    linewidth=1.8,
                    label=f"{role} primary-secondary",
                )
    axes[0].set_title("Truth vs inference effective spectral response")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Weight")
    axes[1].set_title("Component weight difference")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Primary - secondary")
    for ax in axes:
        _style_axis(ax)
        ax.legend()
    fig.tight_layout()
    exports["spectral_truth_vs_inference_weights"] = _save_figure(
        fig, outdir, "spectral_truth_vs_inference_weights", formats, dpi=dpi
    )

    metadata = {
        "tables": {
            "truth_rows": len(truth_rows),
            "inference_rows": len(inference_rows),
        },
        "response_paths": {label: response.get("path") for label, response in responses.items()},
    }
    return exports, metadata, warnings


def _plot_detector_calibration_figures(
    *,
    resolved_ctx: dict[str, Any],
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> tuple[dict[str, list[str]], dict[str, Any], list[str]]:
    truth_system = resolved_ctx["split_ctx"]["truth_system_cfg"]
    inference_system = resolved_ctx["split_ctx"]["inference_system_cfg"]
    truth_maps = review.load_detector_calibration_maps(truth_system)
    inference_maps = review.load_detector_calibration_maps(inference_system)
    exports: dict[str, list[str]] = {}
    warnings: list[str] = []
    metadata: dict[str, Any] = {
        "truth_maps": {name: list(np.asarray(arr).shape) for name, arr in truth_maps.items()},
        "inference_maps": {name: list(np.asarray(arr).shape) for name, arr in inference_maps.items()},
    }
    if not truth_maps:
        return exports, metadata, ["No detector calibration maps were loaded from the truth system."]

    n = len(truth_maps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, (name, arr) in zip(axes.flat, truth_maps.items()):
        im = ax.imshow(np.asarray(arr, dtype=float), origin="lower", cmap=review.cmap_with_bad("viridis", bad="0.65"))
        ax.set_title(name)
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        _style_axis(ax, grid=False)
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(colors="black")
        cbar.outline.set_edgecolor("black")
    fig.tight_layout()
    exports["detector_calibration_maps"] = _save_figure(
        fig, outdir, "detector_calibration_maps", formats, dpi=dpi
    )

    def first_map(suffix: str, maps: dict[str, np.ndarray]) -> np.ndarray | None:
        for name, arr in maps.items():
            if name.endswith(suffix):
                return np.asarray(arr, dtype=float)
        return None

    truth_dx = first_map(".dx_path", truth_maps)
    truth_dy = first_map(".dy_path", truth_maps)
    infer_dx = first_map(".dx_path", inference_maps)
    infer_dy = first_map(".dy_path", inference_maps)
    if all(arr is not None for arr in (truth_dx, truth_dy, infer_dx, infer_dy)):
        try:
            fig, _ = plot_pixel_offset_maps(
                truth_dx,
                truth_dy,
                infer_dx,
                infer_dy,
                cmap="viridis_nan",
                show=False,
                close=False,
            )
            exports["detector_pixel_offsets"] = _save_figure(
                fig, outdir, "detector_pixel_offsets", formats, dpi=dpi
            )
        except Exception as exc:
            warnings.append(f"Detector pixel offset comparison skipped: {exc}")
    else:
        warnings.append("Detector pixel offset comparison skipped: dx/dy maps not available for both truth and inference.")

    truth_prf = first_map(".prf_path", truth_maps)
    if truth_prf is None:
        truth_prf = first_map(".flat_path", truth_maps)
    infer_prf = first_map(".prf_path", inference_maps)
    if infer_prf is None:
        infer_prf = first_map(".flat_path", inference_maps)
    if truth_prf is not None and infer_prf is not None:
        try:
            fig, _ = plot_pixel_response_maps(
                truth_prf,
                infer_prf,
                cmap="viridis_nan",
                show=False,
                close=False,
            )
            exports["detector_pixel_response"] = _save_figure(
                fig, outdir, "detector_pixel_response", formats, dpi=dpi
            )
        except Exception as exc:
            warnings.append(f"Detector pixel response comparison skipped: {exc}")
    else:
        warnings.append("Detector pixel response comparison skipped: prf/flat maps not available for both truth and inference.")

    return exports, metadata, warnings


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
    ("raw_ptt_removed_truth_opd_nm", "full_truth_opd_nm", "Raw PTT-removed truth OPD", "full_truth"),
    ("low_order_truth_reconstruction_nm", "low_order_truth_reconstruction_nm", "Low-order Zernike reconstruction", None),
    ("truth_high_order_residual_opd_nm", "high_order_truth_opd_nm", "Truth high-order residual OPD", "high_order_truth"),
    (
        "knowledge_error_high_order_residual_opd_nm",
        "high_order_error_opd_nm",
        "High-order knowledge-error residual OPD",
        "high_order_error",
    ),
    ("inference_high_order_opd_nm", "inference_high_order_opd_nm", "Inference high-order OPD", "high_order_knowledge"),
    ("inference_sum_residual_nm", "inference_sum_residual_nm", "Inference - truth residual - error", None),
]


def _split_artifact_path(split: Any, filename: str) -> str | None:
    for path in getattr(split, "artifact_paths", {}).values():
        candidate = Path(str(path))
        if candidate.name == filename:
            return str(candidate)
    return None


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
            "mask_fits_source": _split_artifact_path(split, f"{mirror}_mask.fits"),
            "panels": {},
        }
        for key, suffix, title, source_name in WFE_MAP_PANELS:
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
                "source_fits": (
                    _split_artifact_path(split, f"{mirror}_{source_name}_opd_nm.fits")
                    if source_name is not None
                    else None
                ),
                "source_kind": "model_split_fits" if source_name is not None else "derived_from_review_summary",
            }
        metadata["mirrors"][mirror] = mirror_meta
    return exports, metadata, warnings


def _trajectory_ylabel(key: str) -> str:
    if key.endswith("_as"):
        return f"{key} (arcsec)"
    if key.endswith("_deg"):
        return f"{key} (deg)"
    return str(key)


def _plot_trajectory_figures(
    *,
    translated_config: dict[str, Any],
    outdir: Path,
    formats: list[str],
    dpi: int,
) -> tuple[dict[str, list[str]], dict[str, Any], list[str]]:
    trajectory_review = review.load_trajectory_for_review(translated_config)
    hp_review = review.make_high_pass_trajectory_diagnostic(trajectory_review, timescale_s=15.0)
    exports: dict[str, list[str]] = {}
    warnings = list(trajectory_review.get("warnings", []))
    metadata: dict[str, Any] = {
        "available": bool(trajectory_review.get("available", False)),
        "summary": trajectory_review.get("summary", trajectory_review),
        "high_pass_diagnostic": {
            key: value for key, value in hp_review.items() if key != "series"
        },
    }
    if not trajectory_review.get("available"):
        reason = str(trajectory_review.get("reason", "trajectory unavailable"))
        return exports, metadata, [reason]

    component_figures = review.plot_trajectory_review_components(trajectory_review)
    traj = trajectory_review["trajectory"]
    keys = list(traj.values)
    for idx, fig in enumerate(component_figures, start=1):
        key = keys[idx - 1] if idx - 1 < len(keys) else f"component_{idx:02d}"
        stem = f"trajectory_{_safe_stem(key)}_components"
        exports[stem] = _save_figure(fig, outdir, stem, formats, dpi=dpi)

    filter_rows = review.trajectory_filter_provenance_table(trajectory_review)
    if filter_rows:
        labels = [str(row["key"]) for row in filter_rows]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 4))
        width = 0.25
        ax.bar(x - width, [row["raw_rms"] for row in filter_rows], width=width, label="raw")
        ax.bar(x, [row["filtered_rms"] for row in filter_rows], width=width, label="filtered")
        ax.bar(x + width, [row["removed_rms"] for row in filter_rows], width=width, label="removed")
        ax.set_title("Trajectory RMS by raw, filtered, and removed components")
        ax.set_ylabel("RMS")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.legend()
        _style_axis(ax)
        fig.tight_layout()
        exports["trajectory_rms_summary"] = _save_figure(
            fig, outdir, "trajectory_rms_summary", formats, dpi=dpi
        )

    filter_prov = trajectory_review.get("summary", {}).get("filter", {})
    if filter_prov.get("frequency_response"):
        response = filter_prov["frequency_response"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(response.get("frequency_hz", []), response.get("gain", []), color="#2a6f97", linewidth=2.0)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Gain")
        ax.set_title("Configured Bessel filter response")
        _style_axis(ax)
        fig.tight_layout()
        exports["trajectory_filter_response"] = _save_figure(
            fig, outdir, "trajectory_filter_response", formats, dpi=dpi
        )

    time_s = np.asarray(traj.time_s, dtype=float)
    if time_s.size > 1:
        dt = float(np.median(np.diff(time_s)))
        raw_values = traj.unfiltered_values or traj.values
        fig, axes = plt.subplots(len(traj.values), 1, figsize=(11, 3 * len(traj.values)), squeeze=False)
        for ax, key in zip(axes[:, 0], traj.values):
            raw = np.asarray(raw_values[key], dtype=float)
            filt = np.asarray(traj.values[key], dtype=float)
            freq = np.fft.rfftfreq(raw.size, d=dt)
            ax.semilogy(freq[1:], np.abs(np.fft.rfft(raw - raw.mean()))[1:] ** 2, label="raw")
            ax.semilogy(freq[1:], np.abs(np.fft.rfft(filt - filt.mean()))[1:] ** 2, label="filtered")
            ax.set_title(f"FFT power {key}")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Power")
            ax.legend()
            _style_axis(ax)
        fig.tight_layout()
        exports["trajectory_fft_power"] = _save_figure(
            fig, outdir, "trajectory_fft_power", formats, dpi=dpi
        )

    if hp_review.get("available") and hp_review.get("series"):
        fig, axes = plt.subplots(len(hp_review["series"]), 1, figsize=(12, 3 * len(hp_review["series"])), squeeze=False)
        for ax, key in zip(axes[:, 0], hp_review["series"]):
            series = hp_review["series"][key]
            ax.plot(time_s, series["raw"], label="raw", alpha=0.55)
            ax.plot(time_s, series["low_pass"], label="15 s moving average")
            ax.plot(time_s, series["high_pass"], label="diagnostic residual")
            ax.set_title(f"Moving-average diagnostic {key}; residual RMS={series['rms_high_pass']:.4g}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(_trajectory_ylabel(key))
            ax.legend()
            _style_axis(ax)
        fig.tight_layout()
        exports["trajectory_moving_average_diagnostic"] = _save_figure(
            fig, outdir, "trajectory_moving_average_diagnostic", formats, dpi=dpi
        )

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
    artifacts["all"] = sorted(
        path for group, paths in artifacts.items() if group != "all" for path in paths
    )
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
        "figure_exports_by_category": {section: {} for section in EXPORT_SECTIONS},
        "export_sections_requested": list(EXPORT_SECTIONS),
        "export_sections_completed": [],
        "export_sections_skipped": {},
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
        _record_exports(manifest, "spectral", "m2_filter_response", files)
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
        _record_exports(manifest, "spectral", "detector_qe", files)
        manifest["source_data_paths"]["detector_qe"] = str(source)
        _mark_completed(manifest, "spectral")

        files, sed_paths = _plot_raw_seds(
            outdir=figure_dir,
            wavelength_limits_nm=wavelength_limits_nm,
            figsize=line_figsize,
            formats=formats,
            dpi=int(args.dpi),
        )
        _record_exports(manifest, "sed", "alpha_cen_raw_sed", files)
        manifest["source_data_paths"]["raw_target_seds"] = sed_paths
        _mark_completed(manifest, "sed")

        try:
            resolved_ctx = _build_resolved_context(config, outdir)
        except Exception as exc:
            _mark_skipped(manifest, "dp_opd", f"resolved-system context unavailable: {exc}")
            _mark_skipped(manifest, "high_order_wfe", f"resolved-system context unavailable: {exc}")
            _mark_skipped(manifest, "trajectory", f"translated review config unavailable: {exc}")
            _mark_skipped(manifest, "detector_calibration", f"resolved-system context unavailable: {exc}")

        if resolved_ctx is not None:
            try:
                files_by_panel, metadata, warnings = _plot_spectral_review_figures(
                    resolved_ctx=resolved_ctx,
                    outdir=figure_dir,
                    formats=formats,
                    dpi=int(args.dpi),
                )
                for name, files in files_by_panel.items():
                    _record_exports(manifest, "spectral", name, files)
                manifest["source_data_paths"]["spectral_review"] = metadata
                manifest["warnings"].extend(warnings)
                if files_by_panel:
                    _mark_completed(manifest, "spectral")
            except Exception as exc:
                manifest["warnings"].append(f"Skipped spectral review figure exports: {exc}")

            try:
                files, metadata, warnings = _plot_diffractive_pupil_opd(
                    resolved_ctx=resolved_ctx,
                    outdir=figure_dir,
                    cmap_name=str(args.dp_opd_cmap),
                    figsize=opd_figsize,
                    formats=formats,
                    dpi=int(args.dpi),
                )
                _record_exports(manifest, "dp_opd", "diffractive_pupil_opd", files)
                manifest["source_data_paths"]["diffractive_pupil_opd"] = metadata
                manifest["warnings"].extend(warnings)
                _mark_completed(manifest, "dp_opd")
            except Exception as exc:
                _mark_skipped(manifest, "dp_opd", str(exc))

            try:
                files_by_panel, metadata, warnings = _plot_wfe_review_maps(
                    resolved_ctx=resolved_ctx,
                    outdir=figure_dir,
                    formats=formats,
                    dpi=int(args.dpi),
                    figsize=opd_figsize,
                )
                for name, files in files_by_panel.items():
                    _record_exports(manifest, "high_order_wfe", name, files)
                manifest["source_data_paths"]["high_order_wfe_review_maps"] = metadata
                manifest["warnings"].extend(warnings)
                if files_by_panel:
                    _mark_completed(manifest, "high_order_wfe")
                else:
                    _mark_skipped(manifest, "high_order_wfe", "no high-order WFE review maps were available")
            except Exception as exc:
                _mark_skipped(manifest, "high_order_wfe", str(exc))

            try:
                files_by_panel, metadata, warnings = _plot_trajectory_figures(
                    translated_config=resolved_ctx["split_ctx"]["translated_config"],
                    outdir=figure_dir,
                    formats=formats,
                    dpi=int(args.dpi),
                )
                for name, files in files_by_panel.items():
                    _record_exports(manifest, "trajectory", name, files)
                manifest["source_data_paths"]["trajectory_review"] = metadata
                manifest["warnings"].extend(warnings)
                if files_by_panel:
                    _mark_completed(manifest, "trajectory")
                else:
                    _mark_skipped(manifest, "trajectory", "; ".join(warnings) or "no trajectory figures were available")
            except Exception as exc:
                _mark_skipped(manifest, "trajectory", str(exc))

            try:
                files_by_panel, metadata, warnings = _plot_detector_calibration_figures(
                    resolved_ctx=resolved_ctx,
                    outdir=figure_dir,
                    formats=formats,
                    dpi=int(args.dpi),
                )
                for name, files in files_by_panel.items():
                    _record_exports(manifest, "detector_calibration", name, files)
                manifest["source_data_paths"]["detector_calibration"] = metadata
                manifest["warnings"].extend(warnings)
                if files_by_panel:
                    _mark_completed(manifest, "detector_calibration")
                else:
                    _mark_skipped(manifest, "detector_calibration", "; ".join(warnings) or "no detector calibration figures were available")
            except Exception as exc:
                _mark_skipped(manifest, "detector_calibration", str(exc))

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
