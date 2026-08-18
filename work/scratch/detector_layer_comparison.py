"""Compare detector-layer PSFs with explicit pairwise diagnostic figures.

Usage:
    python work/scratch/detector_layer_comparison.py

Outputs:
    Results/detector_layer_comparison_<timestamp>/
"""

from __future__ import annotations

import copy
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.plot.plotting import apply_plot_defaults, get_default_cmaps
from dluxshera.systems.base import SheraBinder, compose_forward_spec

SYSTEM_PRESET = "SHERA_FLIGHT_3P"
TIMESTAMP = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "Results" / f"detector_layer_comparison_{TIMESTAMP}"

PIXEL_OFFSETS_DX_PATH = "src/dluxshera/data/pixel_offsets/dx_baseline.fits"
PIXEL_OFFSETS_DY_PATH = "src/dluxshera/data/pixel_offsets/dy_baseline.fits"
PIXEL_RESPONSE_PATH = "src/dluxshera/data/pixel_response/prf_baseline.fits"


def _downsample_layer() -> dict[str, Any]:
    return {
        "name": "downsample",
        "kind": "Downsample",
        "kernel_size": 3,
    }


def _pixel_offsets_layer() -> dict[str, Any]:
    return {
        "name": "pixel_offsets",
        "kind": "ApplyPixelOffsets",
        "dx_path": PIXEL_OFFSETS_DX_PATH,
        "dy_path": PIXEL_OFFSETS_DY_PATH,
    }


def _pixel_response_layer() -> dict[str, Any]:
    return {
        "name": "pixel_response",
        "kind": "ApplyPixelResponse",
        "prf_path": PIXEL_RESPONSE_PATH,
    }


def _pixel_mtf_layer() -> dict[str, Any]:
    return {
        "name": "pixel_mtf",
        "kind": "ApplyConvolution",
        "kernel": {
            "kind": "box",
            "width_x": 1.0,
            "width_y": 1.0,
            "kernel_size": 3,
            "units": "detector_pix",
        },
    }


def _diffusion_layer() -> dict[str, Any]:
    return {
        "name": "diffusion",
        "kind": "ApplyConvolution",
        "kernel": {
            "kind": "gaussian",
            "sigma_x": 0.15,
            "sigma_y": 0.15,
            "theta_deg": 0.0,
            "kernel_size": 9,
            "units": "detector_pix",
        },
    }


def _smear_layer() -> dict[str, Any]:
    return {
        "name": "smear",
        "kind": "ApplyConvolution",
        "kernel": {
            "kind": "line",
            "length": 0.5,
            "theta_deg": 45.0,
            "sigma_perp": 0.05,
            "kernel_size": 11,
            "units": "detector_pix",
        },
    }


# Keep the detector stacks explicit and close to the top of the file so this
# remains easy to modify for exploratory detector-model comparisons.
DETECTOR_LAYER_SETS: dict[str, list[dict[str, Any]]] = {
    "minimal": [
        _downsample_layer(),
    ],
    "pixel_offsets_only": [
        _downsample_layer(),
        _pixel_offsets_layer(),
    ],
    "pixel_response_only": [
        _downsample_layer(),
        _pixel_response_layer(),
    ],
    "pixel_offsets_plus_response": [
        _downsample_layer(),
        _pixel_offsets_layer(),
        _pixel_response_layer(),
    ],
    # Pixel MTF integrates one detector pixel on the oversampled PSF; the
    # pixel-offset warp remains to sample/interpolate onto the detector grid.
    "pixel_mtf_only": [
        _pixel_mtf_layer(),
        _pixel_offsets_layer(),
    ],
    "pixel_mtf_plus_diffusion": [
        _pixel_mtf_layer(),
        _diffusion_layer(),
        _pixel_offsets_layer(),
    ],
    "pixel_mtf_plus_smear": [
        _pixel_mtf_layer(),
        _smear_layer(),
        _pixel_offsets_layer(),
    ],
}


@dataclass(frozen=True)
class ComparisonPair:
    comparison_name: str
    reference_name: str
    question: str

    @property
    def stem(self) -> str:
        return f"{self.comparison_name}__vs__{self.reference_name}"


# Keep the comparison list explicit rather than inferring references from a
# generic scheme. Each entry corresponds to one physical question.
COMPARISON_PAIRS: list[ComparisonPair] = [
    ComparisonPair(
        comparison_name="pixel_offsets_only",
        reference_name="minimal",
        question="Legacy offsets-only total effect vs minimal legacy baseline",
    ),
    ComparisonPair(
        comparison_name="pixel_response_only",
        reference_name="minimal",
        question="Legacy response-only total effect vs minimal legacy baseline",
    ),
    ComparisonPair(
        comparison_name="pixel_offsets_plus_response",
        reference_name="minimal",
        question="Legacy offsets + response total effect vs minimal legacy baseline",
    ),
    ComparisonPair(
        comparison_name="pixel_mtf_only",
        reference_name="minimal",
        question="Convolutional pixel-MTF total effect vs minimal legacy baseline",
    ),
    ComparisonPair(
        comparison_name="pixel_mtf_plus_diffusion",
        reference_name="pixel_mtf_only",
        question="Incremental diffusion effect beyond the pixel-MTF convolutional baseline",
    ),
    ComparisonPair(
        comparison_name="pixel_mtf_plus_smear",
        reference_name="pixel_mtf_only",
        question="Incremental smear effect beyond the pixel-MTF convolutional baseline",
    ),
    ComparisonPair(
        comparison_name="pixel_mtf_plus_diffusion",
        reference_name="minimal",
        question="Pixel-MTF + diffusion total effect vs minimal legacy baseline",
    ),
    ComparisonPair(
        comparison_name="pixel_mtf_plus_smear",
        reference_name="minimal",
        question="Pixel-MTF + smear total effect vs minimal legacy baseline",
    ),
]


def _load_base_system_config(system_preset: str = SYSTEM_PRESET) -> dict[str, Any]:
    user_cfg = load_user_config(
        config_path=None,
        system_preset=system_preset,
        experiment_preset=None,
    )
    cfg = resolve_config(user_cfg)
    system_cfg = cfg.get("system")
    if not isinstance(system_cfg, dict):
        raise ValueError("Resolved config did not contain a top-level 'system' mapping.")

    source_cfg = system_cfg.get("source", {})
    if not isinstance(source_cfg, dict) or source_cfg.get("exposure_time_s") is None:
        raise ValueError(
            f"{system_preset} must define system.source.exposure_time_s for binder construction."
        )

    return copy.deepcopy(system_cfg)


def _replace_detector_layers(
    system_cfg: dict[str, Any],
    detector_layers: list[dict[str, Any]],
) -> dict[str, Any]:
    system_copy = copy.deepcopy(system_cfg)
    detector_cfg = dict(system_copy.get("detector", {}) or {})
    detector_cfg["layers"] = copy.deepcopy(detector_layers)
    system_copy["detector"] = detector_cfg
    return system_copy


def _render_psf(system_cfg: dict[str, Any]) -> np.ndarray:
    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec)
    base_store = base_store.refresh_derived(forward_spec)
    binder = SheraBinder(system_cfg, forward_spec, base_store)
    return np.asarray(binder.model(binder.strip_structural(base_store)), dtype=float)


def _render_all_layer_sets(base_system_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    rendered_psfs: dict[str, np.ndarray] = {}
    for layer_set_name, detector_layers in DETECTOR_LAYER_SETS.items():
        print(f"Rendering {layer_set_name} with {len(detector_layers)} detector layer(s)")
        system_cfg = _replace_detector_layers(base_system_cfg, detector_layers)
        rendered_psfs[layer_set_name] = _render_psf(system_cfg)
        print(
            f"  shape={rendered_psfs[layer_set_name].shape}, "
            f"total_flux={np.sum(rendered_psfs[layer_set_name]):.6e}"
        )
    return rendered_psfs


def _compute_centroid(image: np.ndarray) -> tuple[float, float]:
    total_flux = float(np.sum(image))
    if not np.isfinite(total_flux) or total_flux == 0.0:
        return np.nan, np.nan

    yy, xx = np.indices(image.shape, dtype=float)
    x_centroid = float(np.sum(image * xx) / total_flux)
    y_centroid = float(np.sum(image * yy) / total_flux)
    return x_centroid, y_centroid


def _compute_diagnostics(
    *,
    pair: ComparisonPair,
    reference_psf: np.ndarray,
    comparison_psf: np.ndarray,
) -> dict[str, Any]:
    if reference_psf.shape != comparison_psf.shape:
        raise ValueError(
            "Reference and comparison PSFs must have identical shapes for comparison; "
            f"got {reference_psf.shape} vs {comparison_psf.shape} for {pair.stem}."
        )

    residual = comparison_psf - reference_psf
    reference_flux = float(np.sum(reference_psf))
    comparison_flux = float(np.sum(comparison_psf))
    flux_ratio = comparison_flux / reference_flux if reference_flux != 0.0 else np.nan
    max_abs_residual = float(np.max(np.abs(residual)))
    rms_residual = float(np.sqrt(np.mean(residual**2)))

    reference_cx, reference_cy = _compute_centroid(reference_psf)
    comparison_cx, comparison_cy = _compute_centroid(comparison_psf)
    centroid_shift_x = comparison_cx - reference_cx
    centroid_shift_y = comparison_cy - reference_cy
    centroid_shift_r = float(np.hypot(centroid_shift_x, centroid_shift_y))

    return {
        "comparison_name": pair.comparison_name,
        "reference_name": pair.reference_name,
        "question": pair.question,
        "reference_flux": reference_flux,
        "comparison_flux": comparison_flux,
        "flux_ratio": flux_ratio,
        "max_abs_residual": max_abs_residual,
        "rms_residual": rms_residual,
        "reference_centroid_x_pix": reference_cx,
        "reference_centroid_y_pix": reference_cy,
        "comparison_centroid_x_pix": comparison_cx,
        "comparison_centroid_y_pix": comparison_cy,
        "centroid_shift_x_pix": centroid_shift_x,
        "centroid_shift_y_pix": centroid_shift_y,
        "centroid_shift_r_pix": centroid_shift_r,
    }


def _format_float(value: float, *, fmt: str) -> str:
    if not np.isfinite(value):
        return "nan"
    return format(value, fmt)


def _save_comparison_figure(
    *,
    pair: ComparisonPair,
    reference_psf: np.ndarray,
    comparison_psf: np.ndarray,
    diagnostics: dict[str, Any],
    outdir: Path,
) -> Path:
    residual = comparison_psf - reference_psf
    image_vmax = float(max(np.max(reference_psf), np.max(comparison_psf)))
    diff_abs_max = float(np.max(np.abs(residual)))

    if not np.isfinite(image_vmax) or image_vmax <= 0.0:
        image_vmax = 1.0
    if not np.isfinite(diff_abs_max) or diff_abs_max <= 0.0:
        diff_abs_max = 1.0e-12

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)

    ref_im = axes[0].imshow(
        reference_psf,
        origin="lower",
        cmap="inferno_nan",
        vmin=0.0,
        vmax=image_vmax,
    )
    cmp_im = axes[1].imshow(
        comparison_psf,
        origin="lower",
        cmap="inferno_nan",
        vmin=0.0,
        vmax=image_vmax,
    )
    diff_im = axes[2].imshow(
        residual,
        origin="lower",
        cmap="coolwarm_nan",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-diff_abs_max, vmax=diff_abs_max),
    )

    axes[0].set_title(f"Reference\n{pair.reference_name}")
    axes[1].set_title(f"Comparison\n{pair.comparison_name}")
    axes[2].set_title("Residual\ncomparison - reference")

    for ax in axes:
        ax.set_xlabel("x [pix]")
        ax.set_ylabel("y [pix]")

    fig.colorbar(ref_im, ax=axes[:2], fraction=0.046, pad=0.04, label="PSF intensity")
    fig.colorbar(diff_im, ax=axes[2], fraction=0.046, pad=0.04, label="Residual")

    diagnostics_line = (
        "flux ratio="
        f"{_format_float(float(diagnostics['flux_ratio']), fmt='.6f')}  |  "
        "max|resid|="
        f"{_format_float(float(diagnostics['max_abs_residual']), fmt='.3e')}  |  "
        "RMS="
        f"{_format_float(float(diagnostics['rms_residual']), fmt='.3e')}  |  "
        "centroid shift="
        f"{_format_float(float(diagnostics['centroid_shift_r_pix']), fmt='.3e')} pix"
    )
    fig.suptitle(
        f"{pair.comparison_name} vs {pair.reference_name}\n"
        f"{pair.question}\n"
        f"{diagnostics_line}",
        fontsize=11,
    )

    outfile = outdir / f"{pair.stem}.png"
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile


def _write_diagnostics_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = (
        "comparison_name",
        "reference_name",
        "question",
        "figure_filename",
        "reference_flux",
        "comparison_flux",
        "flux_ratio",
        "max_abs_residual",
        "rms_residual",
        "reference_centroid_x_pix",
        "reference_centroid_y_pix",
        "comparison_centroid_x_pix",
        "comparison_centroid_y_pix",
        "centroid_shift_x_pix",
        "centroid_shift_y_pix",
        "centroid_shift_r_pix",
    )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _validate_top_level_definitions() -> None:
    known_layer_sets = set(DETECTOR_LAYER_SETS)
    missing_layer_sets = sorted(
        {
            name
            for pair in COMPARISON_PAIRS
            for name in (pair.reference_name, pair.comparison_name)
            if name not in known_layer_sets
        }
    )
    if missing_layer_sets:
        raise ValueError(
            "Comparison pairs reference unknown detector layer sets: "
            + ", ".join(missing_layer_sets)
        )


def main() -> None:
    _validate_top_level_definitions()

    _ = get_default_cmaps()
    apply_plot_defaults()
    plt.rcParams["image.cmap"] = "inferno_nan"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"System preset: {SYSTEM_PRESET}")
    print(f"Output directory: {RESULTS_DIR}")

    base_system_cfg = _load_base_system_config()
    rendered_psfs = _render_all_layer_sets(base_system_cfg)

    diagnostics_rows: list[dict[str, Any]] = []
    for pair in COMPARISON_PAIRS:
        reference_psf = rendered_psfs[pair.reference_name]
        comparison_psf = rendered_psfs[pair.comparison_name]
        diagnostics = _compute_diagnostics(
            pair=pair,
            reference_psf=reference_psf,
            comparison_psf=comparison_psf,
        )
        figure_path = _save_comparison_figure(
            pair=pair,
            reference_psf=reference_psf,
            comparison_psf=comparison_psf,
            diagnostics=diagnostics,
            outdir=RESULTS_DIR,
        )
        diagnostics["figure_filename"] = figure_path.name
        diagnostics_rows.append(diagnostics)

        print(f"[{pair.stem}]")
        print(f"  question: {pair.question}")
        print(f"  figure: {figure_path}")
        print(
            "  fluxes: "
            f"reference={diagnostics['reference_flux']:.6e}, "
            f"comparison={diagnostics['comparison_flux']:.6e}, "
            f"ratio={diagnostics['flux_ratio']:.6f}"
        )
        print(
            "  residuals: "
            f"max_abs={diagnostics['max_abs_residual']:.6e}, "
            f"rms={diagnostics['rms_residual']:.6e}"
        )
        print(
            "  centroids: "
            f"reference=({diagnostics['reference_centroid_x_pix']:.4f}, "
            f"{diagnostics['reference_centroid_y_pix']:.4f}), "
            f"comparison=({diagnostics['comparison_centroid_x_pix']:.4f}, "
            f"{diagnostics['comparison_centroid_y_pix']:.4f}), "
            f"shift=({diagnostics['centroid_shift_x_pix']:.4e}, "
            f"{diagnostics['centroid_shift_y_pix']:.4e}), "
            f"|shift|={diagnostics['centroid_shift_r_pix']:.4e}"
        )

    diagnostics_csv = RESULTS_DIR / "comparison_diagnostics.csv"
    _write_diagnostics_csv(diagnostics_rows, diagnostics_csv)
    print(f"Saved diagnostics CSV to {diagnostics_csv}")


if __name__ == "__main__":
    main()
