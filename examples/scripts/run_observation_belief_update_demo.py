"""Run a synthetic observation-level belief update demo from reduced summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dluxshera-matplotlib"),
)

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from dluxshera.inference.observation_belief import (
    ObservationBeliefState,
    ObservationThetaLayout,
    SubblockSummary,
    build_observation_eigenbasis,
    update_observation_belief,
)
from dluxshera.utils.obs_subblock_io import now_iso_local_ms, timestamp_tag


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "observation_belief_demo"
DEFAULT_ZERNIKE_INDICES = (0, 1, 2, 3, 4, 5)
DEMO_SCHEMA_VERSION = "observation_belief_demo.v1"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any] | Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(label).strip().lower())
    slug = slug.strip("_")
    return slug or "unnamed"


def _display_label(label: str) -> str:
    translations = {
        "source.separation_as": "Separation",
        "source.log_flux_total": "Log Flux",
        "source.contrast": "Contrast",
        "optics.plate_scale_as_per_pix": "Plate Scale",
    }
    if label in translations:
        return translations[label]

    match = re.fullmatch(
        r"optics\.(primary|secondary)\.zernike_coeffs_nm\[(\d+)\]",
        label,
    )
    if match is None:
        return label
    optic = "M1" if match.group(1) == "primary" else "M2"
    return f"{optic} Z{int(match.group(2))}"


def _parameter_unit(label: str) -> str:
    if label == "source.separation_as":
        return "arcsec"
    if label == "source.log_flux_total":
        return "log flux"
    if label == "source.contrast":
        return "dimensionless"
    if label == "optics.plate_scale_as_per_pix":
        return "arcsec / pixel"
    if "zernike_coeffs_nm" in label:
        return "nm"
    return "arb"


def parse_zernike_indices(raw: str | Sequence[int] | None) -> tuple[int, ...]:
    """Parse one comma-separated Zernike index list."""

    if raw is None:
        return DEFAULT_ZERNIKE_INDICES
    if isinstance(raw, str):
        tokens = [piece.strip() for piece in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    indices: list[int] = []
    for token in tokens:
        if not token:
            continue
        indices.append(int(token))
    if not indices:
        raise ValueError("zernike indices must contain at least one integer.")
    if len(set(indices)) != len(indices):
        raise ValueError("zernike indices must not contain duplicates.")
    return tuple(indices)


def build_demo_theta_layout_config(
    *,
    enable_zernikes: bool = True,
    zernike_indices: Sequence[int] = DEFAULT_ZERNIKE_INDICES,
    include_plate_scale: bool = True,
) -> dict[str, Any]:
    """Build the narrow resolved config used by the synthetic demo."""

    indices = tuple(int(index) for index in zernike_indices)
    return {
        "theta_layout": {
            "source": {
                "separation_as": True,
                "log_flux_total": True,
                "contrast": True,
            },
            "optics": {
                "plate_scale_as_per_pix": bool(include_plate_scale),
                "primary_zernikes": {
                    "enabled": bool(enable_zernikes),
                    "indices": list(indices),
                },
                "secondary_zernikes": {
                    "enabled": bool(enable_zernikes),
                    "indices": list(indices),
                },
            },
        }
    }


def build_default_prior(
    labels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return narrow default prior means and sigmas for the demo layout."""

    mean = np.zeros((len(labels),), dtype=float)
    sigma = np.zeros((len(labels),), dtype=float)
    for index, label in enumerate(labels):
        if label == "source.separation_as":
            mean[index] = 10.0
            sigma[index] = 0.40
        elif label == "source.log_flux_total":
            mean[index] = 12.0
            sigma[index] = 0.20
        elif label == "source.contrast":
            mean[index] = 3.0
            sigma[index] = 0.30
        elif label == "optics.plate_scale_as_per_pix":
            mean[index] = 0.030
            sigma[index] = 0.0015
        elif "zernike_coeffs_nm" in label:
            mean[index] = 0.0
            sigma[index] = 12.0
        else:
            mean[index] = 0.0
            sigma[index] = 1.0
    return mean, sigma


def build_synthetic_truth(
    *,
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one synthetic truth offset from the prior mean."""

    offset = rng.normal(loc=0.0, scale=0.55 * prior_sigma)
    return prior_mean + offset


def _matching_zernike_pairs(labels: Sequence[str]) -> list[tuple[int, int, int]]:
    pattern = re.compile(r"optics\.(primary|secondary)\.zernike_coeffs_nm\[(\d+)\]")
    primary: dict[int, int] = {}
    secondary: dict[int, int] = {}
    for index, label in enumerate(labels):
        match = pattern.fullmatch(label)
        if match is None:
            continue
        family = match.group(1)
        z_index = int(match.group(2))
        if family == "primary":
            primary[z_index] = index
        else:
            secondary[z_index] = index
    return [
        (primary[z_index], secondary[z_index], z_index)
        for z_index in sorted(set(primary).intersection(secondary))
    ]


def build_synthetic_reduced_information(
    *,
    layout: ObservationThetaLayout,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build one positive-semidefinite reduced information matrix.

    The synthetic design intentionally constrains the common mode of matching
    M1/M2 Zernike pairs much more strongly than the differential mode. That
    leaves a weak observation-level direction for the eigen diagnostics to
    expose while keeping the posterior well behaved under the prior.
    """

    labels = layout.labels
    label_to_index = {label: index for index, label in enumerate(labels)}
    theta_size = layout.size
    rows: list[np.ndarray] = []

    def _append_row(vector: np.ndarray, weight: float) -> None:
        scaled = np.asarray(vector, dtype=float) * np.sqrt(float(weight))
        rows.append(scaled)

    separation_index = label_to_index.get("source.separation_as")
    log_flux_index = label_to_index.get("source.log_flux_total")
    contrast_index = label_to_index.get("source.contrast")
    plate_scale_index = label_to_index.get("optics.plate_scale_as_per_pix")

    scalar_weights = {
        "source.separation_as": 2.2,
        "source.log_flux_total": 2.8,
        "source.contrast": 2.0,
        "optics.plate_scale_as_per_pix": 1.6,
    }
    for label, base_weight in scalar_weights.items():
        if label not in label_to_index:
            continue
        vector = np.zeros((theta_size,), dtype=float)
        vector[label_to_index[label]] = 1.0
        if label == "source.separation_as" and plate_scale_index is not None:
            vector[plate_scale_index] = 0.25
        if label == "source.contrast" and log_flux_index is not None:
            vector[log_flux_index] = 0.20
        weight = base_weight * (0.85 + 0.30 * rng.random())
        _append_row(vector, weight)

    scalar_indices = [
        index
        for index, label in enumerate(labels)
        if "zernike_coeffs_nm" not in label
    ]
    if len(scalar_indices) >= 2:
        for _ in range(2):
            vector = np.zeros((theta_size,), dtype=float)
            vector[np.asarray(scalar_indices, dtype=int)] = rng.normal(
                loc=0.0,
                scale=0.35,
                size=len(scalar_indices),
            )
            _append_row(vector, 0.85 * (0.85 + 0.30 * rng.random()))

    for primary_index, secondary_index, _ in _matching_zernike_pairs(labels):
        common = np.zeros((theta_size,), dtype=float)
        common[primary_index] = 1.0 / np.sqrt(2.0)
        common[secondary_index] = 1.0 / np.sqrt(2.0)
        _append_row(common, 0.18 * (0.85 + 0.30 * rng.random()))

        differential = np.zeros((theta_size,), dtype=float)
        differential[primary_index] = 1.0 / np.sqrt(2.0)
        differential[secondary_index] = -1.0 / np.sqrt(2.0)
        _append_row(differential, 0.006 * (0.85 + 0.30 * rng.random()))

        if separation_index is not None:
            coupled = np.zeros((theta_size,), dtype=float)
            coupled[separation_index] = 0.35
            coupled[primary_index] = 0.25
            coupled[secondary_index] = 0.25
            _append_row(coupled, 0.030 * (0.85 + 0.30 * rng.random()))

        if contrast_index is not None:
            contrast_mix = np.zeros((theta_size,), dtype=float)
            contrast_mix[contrast_index] = 0.25
            contrast_mix[primary_index] = -0.18
            contrast_mix[secondary_index] = -0.18
            _append_row(contrast_mix, 0.015 * (0.85 + 0.30 * rng.random()))

    if not rows:
        raise ValueError("Synthetic information builder produced no rows.")

    design = np.vstack(rows)
    design *= np.sqrt(0.85 + 0.30 * rng.random())
    information = design.T @ design
    return 0.5 * (information + information.T)


def generate_synthetic_subblock_summaries(
    *,
    layout: ObservationThetaLayout,
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
    theta_true: np.ndarray,
    n_subblocks: int,
    rng: np.random.Generator,
) -> list[SubblockSummary]:
    """Generate synthetic reduced summaries consistent with one truth vector."""

    if n_subblocks <= 0:
        raise ValueError("n_subblocks must be positive.")

    summaries: list[SubblockSummary] = []
    for block_index in range(n_subblocks):
        theta_ref = prior_mean + rng.normal(loc=0.0, scale=0.25 * prior_sigma)
        reduced_information = build_synthetic_reduced_information(layout=layout, rng=rng)
        exact_score = reduced_information @ (theta_ref - theta_true)
        noise_scale = 0.05 * np.maximum(np.abs(exact_score), 1.0e-8)
        score_noise = rng.normal(loc=0.0, scale=noise_scale)
        reduced_score = exact_score + score_noise
        summary = SubblockSummary.from_reduced_form(
            subblock_id=f"subblock_{block_index:06d}",
            theta_labels=layout.labels,
            theta_ref=theta_ref,
            reduced_information=reduced_information,
            reduced_score=reduced_score,
            summary_kind="synthetic_schur",
            diagnostics={
                "theta_ref_offset_norm": float(np.linalg.norm(theta_ref - prior_mean)),
                "truth_offset_norm": float(np.linalg.norm(theta_true - prior_mean)),
                "score_noise_norm": float(np.linalg.norm(score_noise)),
            },
        )
        summaries.append(summary)
    return summaries


def build_posterior_table_rows(
    *,
    labels: Sequence[str],
    prior_mean: np.ndarray,
    posterior_mean: np.ndarray,
    truth: np.ndarray,
    prior_sigma: np.ndarray,
    posterior_sigma: np.ndarray,
) -> list[dict[str, Any]]:
    """Return one row per parameter for the posterior CSV."""

    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "label": label,
                "display_label": _display_label(label),
                "unit": _parameter_unit(label),
                "prior_mean": float(prior_mean[index]),
                "posterior_mean": float(posterior_mean[index]),
                "truth": float(truth[index]),
                "prior_sigma": float(prior_sigma[index]),
                "posterior_sigma": float(posterior_sigma[index]),
                "posterior_error": float(posterior_mean[index] - truth[index]),
            }
        )
    return rows


def build_cumulative_update_rows(
    *,
    labels: Sequence[str],
    cumulative_steps: Sequence[Any],
    truth: np.ndarray,
) -> list[dict[str, Any]]:
    """Return one row per cumulative update step."""

    rows: list[dict[str, Any]] = []
    slugs = [_slugify_label(label) for label in labels]
    for step in cumulative_steps:
        sigma = step.sigma()
        row: dict[str, Any] = {
            "n_subblocks": int(step.n_subblocks),
            "subblock_id": step.subblock_id,
            "condition_number": float(step.diagnostics.condition_number),
            "rank_estimate": int(step.diagnostics.rank_estimate),
            "min_eigenvalue": float(step.diagnostics.min_eigenvalue),
            "max_eigenvalue": float(step.diagnostics.max_eigenvalue),
        }
        for index, slug in enumerate(slugs):
            row[f"posterior_sigma__{slug}"] = float(sigma[index])
            row[f"posterior_error__{slug}"] = float(step.mean[index] - truth[index])
        rows.append(row)
    return rows


def _plot_posterior_sigma_history(
    *,
    path: Path,
    labels: Sequence[str],
    cumulative_steps: Sequence[Any],
) -> None:
    x = np.asarray([step.n_subblocks for step in cumulative_steps], dtype=int)
    sigma_matrix = np.vstack([step.sigma() for step in cumulative_steps])
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, label in enumerate(labels):
        ax.semilogy(x, sigma_matrix[:, index], marker="o", label=_display_label(label))
    ax.set_xlabel("Accumulated Subblocks")
    ax.set_ylabel("Posterior Sigma")
    ax.set_title("Posterior Sigma vs Accumulated Subblocks")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_posterior_error_history(
    *,
    path: Path,
    labels: Sequence[str],
    cumulative_steps: Sequence[Any],
    truth: np.ndarray,
) -> None:
    x = np.asarray([step.n_subblocks for step in cumulative_steps], dtype=int)
    error_matrix = np.vstack([np.abs(step.mean - truth) for step in cumulative_steps])
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, label in enumerate(labels):
        ax.semilogy(
            x,
            np.clip(error_matrix[:, index], 1.0e-12, None),
            marker="o",
            label=_display_label(label),
        )
    ax.set_xlabel("Accumulated Subblocks")
    ax.set_ylabel("Absolute Posterior Error")
    ax.set_title("Posterior Error vs Accumulated Subblocks")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_precision_eigenvalue_spectrum(
    *,
    path: Path,
    eigenvalues: np.ndarray,
) -> None:
    x = np.arange(1, eigenvalues.size + 1, dtype=int)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x, np.clip(eigenvalues, 1.0e-12, None), marker="o")
    ax.set_xlabel("Mode Index (Strongest to Weakest)")
    ax.set_ylabel("Precision Eigenvalue")
    ax.set_title("Posterior Precision Eigenvalue Spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _truth_map(labels: Sequence[str], values: np.ndarray) -> dict[str, float]:
    return {label: float(values[index]) for index, label in enumerate(labels)}


def run_observation_belief_update_demo(
    *,
    results_dir: Path | str = DEFAULT_RESULTS_ROOT,
    run_name: str | None = None,
    n_subblocks: int = 8,
    seed: int = 42,
    enable_zernikes: bool = True,
    zernike_indices: Sequence[int] = DEFAULT_ZERNIKE_INDICES,
    include_plate_scale: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the synthetic observation-level belief update demo."""

    resolved_run_name = run_name or f"observation_belief_demo_{timestamp_tag()}"
    run_dir = Path(results_dir).resolve() / resolved_run_name
    zernike_indices = parse_zernike_indices(zernike_indices)
    layout_config = build_demo_theta_layout_config(
        enable_zernikes=enable_zernikes,
        zernike_indices=zernike_indices,
        include_plate_scale=include_plate_scale,
    )
    layout = ObservationThetaLayout.from_config(layout_config)
    rng = np.random.default_rng(int(seed))

    prior_mean, prior_sigma = build_default_prior(layout.labels)
    theta_true = build_synthetic_truth(
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
        rng=rng,
    )
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=layout.labels,
        mean=prior_mean,
        sigma=prior_sigma,
        metadata={
            "generator": "run_observation_belief_update_demo.py",
            "seed": int(seed),
        },
    )
    summaries = generate_synthetic_subblock_summaries(
        layout=layout,
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
        theta_true=theta_true,
        n_subblocks=int(n_subblocks),
        rng=rng,
    )
    update_result = update_observation_belief(prior, summaries)
    eigenbasis = build_observation_eigenbasis(
        update_result.posterior.precision,
        layout.labels,
        eig_floor_abs=1.0e-10,
        eig_floor_rel=1.0e-2,
    )

    posterior_sigma = update_result.posterior.sigma()
    posterior_rows = build_posterior_table_rows(
        labels=layout.labels,
        prior_mean=prior_mean,
        posterior_mean=update_result.posterior.mean,
        truth=theta_true,
        prior_sigma=prior_sigma,
        posterior_sigma=posterior_sigma,
    )
    cumulative_rows = build_cumulative_update_rows(
        labels=layout.labels,
        cumulative_steps=update_result.cumulative_steps,
        truth=theta_true,
    )
    eigenmode_rows = eigenbasis.to_rows(top_k=4)

    config_payload = {
        "schema_version": DEMO_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "generator": "run_observation_belief_update_demo.py",
        "results_dir": str(Path(results_dir).resolve()),
        "run_name": resolved_run_name,
        "n_subblocks": int(n_subblocks),
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "theta_layout": layout_config["theta_layout"],
        "resolved_layout": layout.to_dict(),
    }
    truth_payload = {
        "labels": list(layout.labels),
        "prior_mean": _truth_map(layout.labels, prior_mean),
        "prior_sigma": _truth_map(layout.labels, prior_sigma),
        "truth": _truth_map(layout.labels, theta_true),
    }
    summary_payload = {
        "schema_version": DEMO_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "generator": "run_observation_belief_update_demo.py",
        "dry_run": bool(dry_run),
        "run_dir": str(run_dir),
        "seed": int(seed),
        "n_subblocks": int(n_subblocks),
        "theta_layout": layout.to_dict(),
        "prior": {
            "mean": _truth_map(layout.labels, prior_mean),
            "sigma": _truth_map(layout.labels, prior_sigma),
        },
        "truth": _truth_map(layout.labels, theta_true),
        "posterior": {
            "mean": _truth_map(layout.labels, update_result.posterior.mean),
            "sigma": _truth_map(layout.labels, posterior_sigma),
            "precision_diagnostics": update_result.posterior.metadata[
                "posterior_precision_diagnostics"
            ],
        },
        "update": {
            "damping": float(update_result.metadata["damping"]),
            "n_summaries": int(update_result.metadata["n_summaries"]),
            "solve_method": str(update_result.metadata["solve_method"]),
            "contributing_subblocks": [summary.subblock_id for summary in summaries],
        },
        "eigenbasis": {
            "condition_number": float(eigenbasis.condition_number),
            "weak_mode_count": int(np.count_nonzero(eigenbasis.weak_mode_mask)),
            "largest_eigenvalue": float(eigenbasis.eigenvalues[0]),
            "smallest_eigenvalue": float(eigenbasis.eigenvalues[-1]),
        },
    }

    artifacts: dict[str, str] = {}
    if dry_run:
        return {
            "dry_run": True,
            "run_dir": str(run_dir),
            "summary": summary_payload,
            "artifacts": artifacts,
        }

    _ensure_dir(run_dir)
    summary_dir = run_dir / "synthetic_subblock_summaries"
    _ensure_dir(summary_dir)

    config_path = run_dir / "config_resolved.json"
    truth_path = run_dir / "synthetic_truth.json"
    update_summary_path = run_dir / "observation_update_summary.json"
    posterior_table_path = run_dir / "posterior_table.csv"
    eigenmode_table_path = run_dir / "eigenmode_table.csv"
    cumulative_table_path = run_dir / "cumulative_update_table.csv"
    sigma_plot_path = run_dir / "posterior_sigma_vs_n_subblocks.png"
    error_plot_path = run_dir / "posterior_error_vs_n_subblocks.png"
    eigen_plot_path = run_dir / "precision_eigenvalue_spectrum.png"

    _write_json(config_path, config_payload)
    _write_json(truth_path, truth_payload)
    _write_json(update_summary_path, summary_payload)
    _write_csv_rows(posterior_table_path, posterior_rows)
    _write_csv_rows(eigenmode_table_path, eigenmode_rows)
    _write_csv_rows(cumulative_table_path, cumulative_rows)

    for summary in summaries:
        summary_json_path = summary_dir / f"{summary.subblock_id}_summary.json"
        summary_npz_path = summary_dir / f"{summary.subblock_id}_matrices.npz"
        _write_json(summary_json_path, summary.to_dict(include_arrays=True))
        np.savez_compressed(
            summary_npz_path,
            theta_ref=summary.theta_ref,
            reduced_information=summary.reduced_information,
            reduced_score=summary.reduced_score,
        )

    _plot_posterior_sigma_history(
        path=sigma_plot_path,
        labels=layout.labels,
        cumulative_steps=update_result.cumulative_steps,
    )
    _plot_posterior_error_history(
        path=error_plot_path,
        labels=layout.labels,
        cumulative_steps=update_result.cumulative_steps,
        truth=theta_true,
    )
    _plot_precision_eigenvalue_spectrum(
        path=eigen_plot_path,
        eigenvalues=eigenbasis.eigenvalues,
    )

    artifacts.update(
        {
            "config_resolved_json": str(config_path),
            "synthetic_truth_json": str(truth_path),
            "observation_update_summary_json": str(update_summary_path),
            "posterior_table_csv": str(posterior_table_path),
            "eigenmode_table_csv": str(eigenmode_table_path),
            "cumulative_update_table_csv": str(cumulative_table_path),
            "posterior_sigma_vs_n_subblocks_png": str(sigma_plot_path),
            "posterior_error_vs_n_subblocks_png": str(error_plot_path),
            "precision_eigenvalue_spectrum_png": str(eigen_plot_path),
            "synthetic_subblock_summaries_dir": str(summary_dir),
        }
    )
    return {
        "dry_run": False,
        "run_dir": str(run_dir),
        "summary": summary_payload,
        "artifacts": artifacts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the synthetic demo."""

    parser = argparse.ArgumentParser(
        description="Run a synthetic observation-level belief update demo.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory where the demo run directory will be created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name. Defaults to a timestamped demo name.",
    )
    parser.add_argument(
        "--n-subblocks",
        type=int,
        default=8,
        help="Number of synthetic reduced summaries to accumulate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic RNG seed for the synthetic demo.",
    )
    parser.add_argument(
        "--enable-zernikes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable matching primary/secondary Zernike groups in the layout.",
    )
    parser.add_argument(
        "--zernike-indices",
        type=str,
        default=",".join(str(index) for index in DEFAULT_ZERNIKE_INDICES),
        help="Comma-separated Zernike indices used when Zernikes are enabled.",
    )
    parser.add_argument(
        "--include-plate-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include optics.plate_scale_as_per_pix in the observation layout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the synthetic update without writing artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments and run the synthetic belief update demo."""

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_observation_belief_update_demo(
        results_dir=args.results_dir,
        run_name=args.run_name,
        n_subblocks=args.n_subblocks,
        seed=args.seed,
        enable_zernikes=bool(args.enable_zernikes),
        zernike_indices=parse_zernike_indices(args.zernike_indices),
        include_plate_scale=bool(args.include_plate_scale),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
