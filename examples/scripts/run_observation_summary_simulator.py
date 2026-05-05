"""Forecast observation-level uncertainty from existing summary artifacts.

This script is a narrow simulator layer above the image-backed sub-block
summary handoff:

``sub-block solve -> SubblockSummary handoff -> observation-level forecast``.

It consumes one or more existing ``subblock_summary.json`` artifacts, tiles
those summaries deterministically in ``replicate`` mode, and evaluates how the
posterior uncertainty on ``source.separation_as`` changes as more synthetic
summary contributions are accumulated. Replicate mode is an accumulation sanity
check only. It does not add score noise, bootstrap real Monte Carlo outputs, or
sample matrix entries.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
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
    ObservationUpdateResult,
    SubblockSummary,
    accumulate_summary_information,
    update_observation_belief,
)
from dluxshera.inference.observation_forecast import (
    PriorContext,
    build_default_prior_sigma,
    require_identical_summary_theta_labels,
    resolve_prior_context_for_summaries,
)
from dluxshera.inference.observation_summary import (
    load_subblock_summary,
    load_subblock_summary_artifact_payload,
)
from dluxshera.utils.obs_subblock_io import now_iso_local_ms, timestamp_tag


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "observation_summary_simulator"
SIMULATOR_SCHEMA_VERSION = "observation_summary_simulator.v1"
SEPARATION_LABEL = "source.separation_as"
PRIOR_SIGMA_HELPER_PATH = "dluxshera.inference.observation_forecast.build_default_prior_sigma"
SMALL_PROVENANCE_LIMIT = 100


@dataclass(frozen=True)
class ReplicatedSummaryBatch:
    """Store one deterministic replicate-mode contribution batch.

    The simulator calls this after loading and validating real summaries and
    before invoking :func:`update_observation_belief`. The summaries are reused
    by object reference to preserve each source summary's original
    ``theta_ref``, ``reduced_information``, ``reduced_score``, diagnostics, and
    metadata.

    Parameters
    ----------
    summaries :
        Tiled summary objects in deterministic contribution order.
    source_indices :
        Zero-based input-summary index for each synthetic contribution.
    provenance :
        JSON-friendly compact provenance payload for the batch.

    Notes
    -----
    This is a local script helper. If future forecast modes need the same
    provenance contract, it is a candidate for migration into a small shared
    ``dluxshera.inference.observation_forecast`` module.
    """

    summaries: tuple[SubblockSummary, ...]
    source_indices: tuple[int, ...]
    provenance: dict[str, Any]


def _ensure_dir(path: Path) -> None:
    """Create one output directory for simulator artifact writers.

    Called only after dry-run handling has returned. This helper should remain
    local until multiple forecast scripts need identical filesystem behavior.

    Parameters
    ----------
    path :
        Directory path to create.
    """

    path.mkdir(parents=True, exist_ok=True)


def _json_ready(value: Any) -> Any:
    """Return a JSON-safe representation of simulator payload values.

    The manifest writer calls this to normalize NumPy scalars, arrays, paths,
    and non-finite floating-point diagnostics before ``json.dump``. It is local
    glue for this script's artifact contract and can stay local unless a shared
    forecast artifact writer appears.

    Parameters
    ----------
    value :
        Arbitrary nested payload value.

    Returns
    -------
    Any
        JSON-friendly nested value.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        if np.isnan(value):
            return "nan"
        return "inf" if value > 0.0 else "-inf"
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any] | Sequence[Any] | Any) -> None:
    """Write one simulator JSON artifact with stable indentation.

    Called by the non-dry-run simulator path for ``manifest.json``. This helper
    is local script IO glue and may be migrated later if forecast artifacts
    become shared across scripts.

    Parameters
    ----------
    path :
        Artifact path.
    payload :
        JSON-compatible payload.
    """

    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write CSV rows while preserving first-seen field order.

    The simulator calls this for all tabular forecast artifacts. It accepts an
    empty sequence so callers can still materialize an empty diagnostic table
    during narrow debugging runs.

    Parameters
    ----------
    path :
        CSV artifact path.
    rows :
        Sequence of row mappings.
    """

    _ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
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


def parse_n_subblocks_grid(raw: str | Sequence[int]) -> tuple[int, ...]:
    """Parse and validate the requested accumulated sub-block counts.

    Called by the CLI and tests before any update work is run. The parser keeps
    user order fixed because forecast tables and plots should reflect the
    explicit requested grid.

    Parameters
    ----------
    raw :
        Comma-separated string such as ``"1,3,10"`` or an integer sequence.

    Returns
    -------
    tuple of int
        Positive, duplicate-free accumulated sub-block counts.

    Raises
    ------
    ValueError
        Raised when no counts are provided, a token is not an integer, a count
        is non-positive, or a count is duplicated.
    """

    if isinstance(raw, str):
        tokens = [piece.strip() for piece in raw.split(",")]
    else:
        tokens = [str(value).strip() for value in raw]

    values: list[int] = []
    for token in tokens:
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(
                f"n_subblocks grid contains non-integer token {token!r}."
            ) from exc
        if value <= 0:
            raise ValueError("n_subblocks values must be positive integers.")
        values.append(value)

    if not values:
        raise ValueError("n_subblocks grid must contain at least one value.")
    if len(set(values)) != len(values):
        raise ValueError("n_subblocks grid must not contain duplicate values.")
    return tuple(values)


def _parameter_units(label: str) -> str:
    """Return the physical units used by simulator tables.

    Called while building posterior and sigma-history rows. It mirrors the
    current observation-belief demo conventions without importing a private
    helper from that script.

    Parameters
    ----------
    label :
        Observation-level parameter label.

    Returns
    -------
    str
        Human-readable unit string.
    """

    if label == SEPARATION_LABEL:
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


def _compute_matrix_diagnostics(matrix: np.ndarray) -> dict[str, float | int]:
    """Compute compact symmetric-matrix diagnostics for forecast tables.

    Called for accumulated summary information and posterior precision
    matrices at each requested ``n_subblocks``. This local helper intentionally
    matches the existing diagnostic field names used by
    ``ObservationBeliefState`` metadata.

    Parameters
    ----------
    matrix :
        Square matrix in the observation-level parameter basis.

    Returns
    -------
    dict
        Rank, eigenvalue, condition, trace, and Frobenius-norm diagnostics.

    Raises
    ------
    ValueError
        Raised if ``matrix`` is not finite and square.
    """

    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix diagnostics require a square matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix diagnostics require finite matrix entries.")
    array = 0.5 * (array + array.T)
    if array.size == 0:
        return {
            "rank_estimate": 0,
            "min_eigenvalue": 0.0,
            "max_eigenvalue": 0.0,
            "condition_number": 1.0,
            "trace": 0.0,
            "frobenius_norm": 0.0,
        }

    eigenvalues = np.linalg.eigvalsh(array)
    tolerance = (
        np.finfo(float).eps
        * max(array.shape)
        * max(float(np.max(np.abs(eigenvalues))), 1.0)
    )
    active = np.abs(eigenvalues) > tolerance
    positive = eigenvalues[eigenvalues > tolerance]
    condition_number = (
        float("inf")
        if positive.size == 0
        else float(np.max(positive) / np.min(positive))
    )
    return {
        "rank_estimate": int(np.count_nonzero(active)),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "condition_number": condition_number,
        "trace": float(np.trace(array)),
        "frobenius_norm": float(np.linalg.norm(array)),
    }


def require_identical_theta_labels(
    summaries: Sequence[SubblockSummary],
) -> tuple[str, ...]:
    """Return shared ``theta_labels`` using the library validation helper.

    The simulator calls this immediately after loading summaries. It remains as
    script-local compatibility glue for tests and readers, while the shared
    validation behavior now lives in
    :func:`dluxshera.inference.observation_forecast.require_identical_summary_theta_labels`.

    Parameters
    ----------
    summaries :
        Loaded summary artifacts.

    Returns
    -------
    tuple of str
        Shared ordered theta labels.

    Raises
    ------
    ValueError
        Raised by the shared helper when no summaries are provided or any
        summary has a different ordered ``theta_labels`` tuple.

    Notes
    -----
    Replicate mode intentionally does not perform label union or reordering, so
    all input summaries must already share the same ordered observation-level
    basis.
    """

    return require_identical_summary_theta_labels(summaries)


def validate_required_forecast_labels(theta_labels: Sequence[str]) -> None:
    """Validate labels required by the first summary forecast metric.

    Called immediately after source summary labels are resolved and before
    prior construction or update loops. This remains script-local because the
    first simulator is specifically defined around the separation sigma
    forecast; a later shared forecast module may generalize metric validation.

    Parameters
    ----------
    theta_labels :
        Ordered observation-level labels.

    Raises
    ------
    ValueError
        Raised when ``source.separation_as`` is absent.
    """

    if SEPARATION_LABEL not in tuple(theta_labels):
        raise ValueError(
            "run_observation_summary_simulator currently requires "
            "source.separation_as in theta_labels because the primary forecast "
            "metric is separation posterior sigma in microarcseconds."
        )


def load_summary_artifacts(
    summary_paths: Sequence[Path | str],
) -> tuple[tuple[SubblockSummary, ...], tuple[dict[str, Any], ...], tuple[Path, ...]]:
    """Load and validate summary artifacts for the simulator.

    Called by ``run_observation_summary_simulator`` before prior resolution.
    The summaries are loaded through :func:`load_subblock_summary`, while raw
    payloads are retained for manifest provenance.

    Parameters
    ----------
    summary_paths :
        One or more ``subblock_summary.json`` paths.

    Returns
    -------
    summaries, payloads, paths :
        Loaded summaries, raw JSON payloads, and resolved paths.

    Raises
    ------
    ValueError
        Raised when no paths are provided or ordered ``theta_labels`` differ.
    """

    paths = tuple(Path(path).resolve() for path in summary_paths)
    if not paths:
        raise ValueError("--summary-json requires at least one artifact path.")
    summaries = tuple(load_subblock_summary(path) for path in paths)
    require_identical_theta_labels(summaries)
    payloads = tuple(load_subblock_summary_artifact_payload(path) for path in paths)
    return summaries, payloads, paths


def summarize_source_indices(
    source_indices: Sequence[int],
    *,
    n_input_summaries: int,
) -> dict[str, Any]:
    """Build compact deterministic-replication provenance.

    Called by ``replicate_summaries`` and by the manifest builder. Small batches
    include the full per-slot source-index list; large batches record the first
    and last slots plus counts by source index.

    Parameters
    ----------
    source_indices :
        Zero-based source index for each synthetic contribution.
    n_input_summaries :
        Number of loaded source summaries.

    Returns
    -------
    dict
        JSON-friendly provenance summary.
    """

    values = tuple(int(index) for index in source_indices)
    counts = [
        {
            "source_summary_index": int(index),
            "count": int(values.count(index)),
        }
        for index in range(int(n_input_summaries))
    ]
    payload: dict[str, Any] = {
        "n_contributions": int(len(values)),
        "n_input_summaries": int(n_input_summaries),
        "counts_by_source_summary_index": counts,
        "source_index_sequence_is_full": bool(len(values) <= SMALL_PROVENANCE_LIMIT),
    }
    if len(values) <= SMALL_PROVENANCE_LIMIT:
        payload["source_index_sequence"] = list(values)
    else:
        payload["source_index_sequence_head"] = list(values[:20])
        payload["source_index_sequence_tail"] = list(values[-20:])
    return payload


def replicate_summaries(
    summaries: Sequence[SubblockSummary],
    *,
    n_subblocks: int,
) -> ReplicatedSummaryBatch:
    """Tile source summaries into exactly ``n_subblocks`` contributions.

    The simulator calls this once for each requested forecast grid point. With
    one input summary, the same summary is repeated. With multiple summaries,
    inputs are tiled in their original order and truncated to the requested
    length. No arrays or score vectors are modified.

    Parameters
    ----------
    summaries :
        Loaded source summaries with identical ``theta_labels``.
    n_subblocks :
        Number of synthetic contributions to produce.

    Returns
    -------
    ReplicatedSummaryBatch
        Tiled summaries plus source-index provenance.

    Raises
    ------
    ValueError
        Raised when no summaries are provided, labels do not match, or
        ``n_subblocks`` is non-positive.

    Notes
    -----
    This helper is deterministic by construction. It should remain local until
    additional synthesis modes justify a shared forecast-synthesis module.
    """

    if n_subblocks <= 0:
        raise ValueError("n_subblocks must be positive.")
    source_summaries = tuple(summaries)
    require_identical_theta_labels(source_summaries)
    n_sources = len(source_summaries)
    source_indices = tuple(index % n_sources for index in range(int(n_subblocks)))
    tiled = tuple(source_summaries[index] for index in source_indices)
    provenance = summarize_source_indices(
        source_indices,
        n_input_summaries=n_sources,
    )
    provenance["ordering"] = "tile_input_order_then_truncate"
    return ReplicatedSummaryBatch(
        summaries=tiled,
        source_indices=source_indices,
        provenance=provenance,
    )


def build_artifact_paths(run_dir: Path) -> dict[str, str]:
    """Return planned output artifact paths for one simulator run.

    Called before dry-run returns so users can inspect the planned layout
    without creating files. The helper is local because the artifact set is
    specific to this first summary simulator.

    Parameters
    ----------
    run_dir :
        Resolved run directory.

    Returns
    -------
    dict
        Artifact name to absolute path.
    """

    return {
        "manifest_json": str(run_dir / "manifest.json"),
        "forecast_results_csv": str(run_dir / "forecast_results.csv"),
        "posterior_table_by_n_subblocks_csv": str(
            run_dir / "posterior_table_by_n_subblocks.csv"
        ),
        "cumulative_sigma_history_csv": str(
            run_dir / "cumulative_sigma_history.csv"
        ),
        "information_diagnostics_csv": str(
            run_dir / "information_diagnostics.csv"
        ),
        "separation_sigma_vs_n_subblocks_png": str(
            run_dir / "separation_sigma_vs_n_subblocks.png"
        ),
        "prior_normalized_sigma_vs_n_subblocks_png": str(
            run_dir / "prior_normalized_sigma_vs_n_subblocks.png"
        ),
    }


def _theta_value_map(labels: Sequence[str], values: np.ndarray) -> dict[str, float]:
    """Return a label-keyed float map for manifest prior and posterior payloads."""

    return {label: float(values[index]) for index, label in enumerate(labels)}


def build_prior_state(
    *,
    theta_labels: Sequence[str],
    prior_context: PriorContext,
    prior_sigma_scale: float,
) -> tuple[ObservationBeliefState, np.ndarray]:
    """Build the diagonal prior used by each simulator forecast update.

    Called once after real-summary prior resolution. The mean comes from the
    same policy as the observation belief demo where practical, and the sigma
    vector is the demo's default prior sigma multiplied by
    ``prior_sigma_scale``.

    Parameters
    ----------
    theta_labels :
        Ordered observation-level labels.
    prior_context :
        Resolved prior mean and provenance.
    prior_sigma_scale :
        Positive scalar multiplier for the default prior sigma vector.

    Returns
    -------
    prior, prior_sigma :
        Information-form prior state and the scaled sigma vector.

    Raises
    ------
    ValueError
        Raised when the sigma scale is not positive.
    """

    if float(prior_sigma_scale) <= 0.0:
        raise ValueError("prior_sigma_scale must be strictly positive.")
    prior_mean = np.asarray(prior_context.prior_mean, dtype=float)
    prior_sigma = (
        build_default_prior_sigma(theta_labels) * float(prior_sigma_scale)
    )
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=tuple(theta_labels),
        mean=prior_mean,
        sigma=prior_sigma,
        metadata={
            "generator": "run_observation_summary_simulator.py",
            "prior_mean_source": str(prior_context.prior_mean_source),
            "prior_mean_provenance": dict(prior_context.provenance),
            "prior_sigma_policy": {
                "base": PRIOR_SIGMA_HELPER_PATH,
                "scale": float(prior_sigma_scale),
            },
        },
    )
    return prior, prior_sigma


def build_forecast_row(
    *,
    n_subblocks: int,
    mode: str,
    n_input_summaries: int,
    theta_labels: Sequence[str],
    prior_sigma: np.ndarray,
    update_result: ObservationUpdateResult,
) -> dict[str, Any]:
    """Return one row for ``forecast_results.csv``.

    Called after each requested ``update_observation_belief`` run. The row
    includes the first forecast metric, ``source.separation_as`` posterior
    sigma in arcseconds and microarcseconds, plus posterior precision
    diagnostics.

    Parameters
    ----------
    n_subblocks :
        Number of accumulated synthetic contributions in the update.
    mode :
        Synthesis mode. Only ``replicate`` is active in this script.
    n_input_summaries :
        Number of loaded source summaries.
    theta_labels :
        Ordered observation-level labels.
    prior_sigma :
        Prior sigma vector in the same order as ``theta_labels``.
    update_result :
        Posterior update result for this grid point.

    Returns
    -------
    dict
        CSV-ready forecast row.
    """

    posterior_sigma = update_result.posterior.sigma()
    diagnostics = dict(
        update_result.posterior.metadata.get("posterior_precision_diagnostics", {})
    )
    separation_index = (
        tuple(theta_labels).index(SEPARATION_LABEL)
        if SEPARATION_LABEL in theta_labels
        else None
    )
    separation_sigma_as = (
        None if separation_index is None else float(posterior_sigma[separation_index])
    )
    prior_sigma_value = (
        None if separation_index is None else float(prior_sigma[separation_index])
    )
    ratio = (
        None
        if separation_sigma_as is None or prior_sigma_value is None
        else float(separation_sigma_as / prior_sigma_value)
    )
    return {
        "n_subblocks": int(n_subblocks),
        "mode": str(mode),
        "n_input_summaries": int(n_input_summaries),
        "theta_dim": int(len(theta_labels)),
        "separation_label_found": bool(separation_index is not None),
        "separation_posterior_sigma_as": separation_sigma_as,
        "separation_posterior_sigma_uas": (
            None if separation_sigma_as is None else float(1.0e6 * separation_sigma_as)
        ),
        "separation_posterior_sigma_over_prior_sigma": ratio,
        "posterior_precision_rank_estimate": diagnostics.get("rank_estimate"),
        "posterior_precision_min_eigenvalue": diagnostics.get("min_eigenvalue"),
        "posterior_precision_max_eigenvalue": diagnostics.get("max_eigenvalue"),
        "posterior_precision_condition_number": diagnostics.get("condition_number"),
        "posterior_precision_trace": diagnostics.get("trace"),
        "posterior_precision_frobenius_norm": diagnostics.get("frobenius_norm"),
        "solve_method": update_result.metadata.get(
            "solve_method",
            update_result.posterior.metadata.get("solve_method"),
        ),
    }


def build_posterior_table_rows(
    *,
    n_subblocks: int,
    theta_labels: Sequence[str],
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
    update_result: ObservationUpdateResult,
) -> list[dict[str, Any]]:
    """Return long-form posterior rows for one forecast grid point.

    Called for ``posterior_table_by_n_subblocks.csv``. It reports every
    parameter in the fixed observation basis so downstream plotting can filter
    by ``label`` without decoding wide column names.

    Parameters
    ----------
    n_subblocks :
        Number of accumulated synthetic contributions.
    theta_labels :
        Ordered observation-level labels.
    prior_mean, prior_sigma :
        Prior mean and sigma vectors.
    update_result :
        Posterior update result.

    Returns
    -------
    list of dict
        One row per label.
    """

    posterior_mean = update_result.posterior.mean
    posterior_sigma = update_result.posterior.sigma()
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(theta_labels):
        rows.append(
            {
                "n_subblocks": int(n_subblocks),
                "label": str(label),
                "prior_mean": float(prior_mean[index]),
                "posterior_mean": float(posterior_mean[index]),
                "prior_sigma": float(prior_sigma[index]),
                "posterior_sigma": float(posterior_sigma[index]),
                "posterior_sigma_over_prior_sigma": float(
                    posterior_sigma[index] / prior_sigma[index]
                ),
                "units": _parameter_units(str(label)),
            }
        )
    return rows


def build_cumulative_sigma_history_rows(
    *,
    n_subblocks: int,
    theta_labels: Sequence[str],
    prior_sigma: np.ndarray,
    update_result: ObservationUpdateResult,
) -> list[dict[str, Any]]:
    """Return independent long-form sigma history rows.

    Called for ``cumulative_sigma_history.csv``. The rows duplicate selected
    posterior-table fields, but the file is intentionally easy to load directly
    for plotting posterior sigma histories.

    Parameters
    ----------
    n_subblocks :
        Number of accumulated synthetic contributions.
    theta_labels :
        Ordered observation-level labels.
    prior_sigma :
        Prior sigma vector.
    update_result :
        Posterior update result.

    Returns
    -------
    list of dict
        One row per label for this requested accumulation count.
    """

    posterior_sigma = update_result.posterior.sigma()
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(theta_labels):
        units = _parameter_units(str(label))
        row = {
            "n_subblocks": int(n_subblocks),
            "label": str(label),
            "posterior_sigma": float(posterior_sigma[index]),
            "prior_sigma": float(prior_sigma[index]),
            "posterior_sigma_over_prior_sigma": float(
                posterior_sigma[index] / prior_sigma[index]
            ),
            "units": units,
        }
        if label == SEPARATION_LABEL:
            row["posterior_sigma_uas"] = float(1.0e6 * posterior_sigma[index])
        rows.append(row)
    return rows


def build_information_diagnostic_row(
    *,
    n_subblocks: int,
    theta_labels: Sequence[str],
    summaries: Sequence[SubblockSummary],
    update_result: ObservationUpdateResult,
) -> dict[str, Any]:
    """Return matrix diagnostics for one requested accumulation count.

    Called after each update to populate ``information_diagnostics.csv``. It
    records both the accumulated summary information and posterior precision so
    reviewers can separate evidence growth from prior regularization.

    Parameters
    ----------
    n_subblocks :
        Number of accumulated synthetic contributions.
    theta_labels :
        Ordered observation-level labels.
    summaries :
        Synthetic contribution list passed to the update.
    update_result :
        Posterior update result.

    Returns
    -------
    dict
        CSV-ready diagnostic row.
    """

    accumulated = accumulate_summary_information(theta_labels, summaries)
    accumulated_diag = _compute_matrix_diagnostics(accumulated)
    posterior_diag = _compute_matrix_diagnostics(update_result.posterior.precision)
    row: dict[str, Any] = {
        "n_subblocks": int(n_subblocks),
        "theta_dim": int(len(theta_labels)),
        "solve_method": update_result.metadata.get("solve_method"),
    }
    for prefix, diagnostics in (
        ("accumulated_information", accumulated_diag),
        ("posterior_precision", posterior_diag),
    ):
        for key, value in diagnostics.items():
            row[f"{prefix}_{key}"] = value
    return row


def _plot_separation_sigma(
    *,
    path: Path,
    forecast_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write the separation sigma forecast plot.

    Called by the non-dry-run path after all forecast rows are built. The plot
    uses a headless Matplotlib backend and never calls ``plt.show()``.

    Parameters
    ----------
    path :
        PNG output path.
    forecast_rows :
        Rows containing ``n_subblocks`` and
        ``separation_posterior_sigma_uas``.
    """

    x = np.asarray([int(row["n_subblocks"]) for row in forecast_rows], dtype=int)
    y = np.asarray(
        [float(row["separation_posterior_sigma_uas"]) for row in forecast_rows],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x, np.clip(y, 1.0e-12, None), marker="o")
    ax.set_xlabel("Accumulated Subblocks")
    ax.set_ylabel("source.separation_as Posterior Sigma (microarcsec)")
    ax.set_title("Separation Sigma vs Accumulated Subblocks")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_prior_normalized_sigma(
    *,
    path: Path,
    theta_labels: Sequence[str],
    posterior_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write the optional prior-normalized sigma history plot.

    Called by the non-dry-run simulator path. It plots
    ``posterior_sigma / prior_sigma`` for every observation-level parameter.

    Parameters
    ----------
    path :
        PNG output path.
    theta_labels :
        Ordered labels to plot.
    posterior_rows :
        Long-form posterior table rows.
    """

    by_label: dict[str, list[tuple[int, float]]] = {label: [] for label in theta_labels}
    for row in posterior_rows:
        by_label[str(row["label"])].append(
            (
                int(row["n_subblocks"]),
                float(row["posterior_sigma_over_prior_sigma"]),
            )
        )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label in theta_labels:
        pairs = sorted(by_label[label], key=lambda item: item[0])
        if not pairs:
            continue
        x = np.asarray([item[0] for item in pairs], dtype=int)
        y = np.asarray([item[1] for item in pairs], dtype=float)
        ax.semilogx(x, y, marker="o", label=label)
    ax.set_xlabel("Accumulated Subblocks")
    ax.set_ylabel("Posterior Sigma / Prior Sigma")
    ax.set_title("Prior-Normalized Sigma vs Accumulated Subblocks")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_observation_summary_simulator(
    *,
    summary_paths: Sequence[Path | str],
    mode: str = "replicate",
    n_subblocks_grid: str | Sequence[int] = (1, 3, 10, 30, 100, 300, 1800),
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    run_name: str | None = None,
    config_path: Path | None = None,
    system_preset: str | None = None,
    prior_source: str = "auto",
    prior_sigma_scale: float = 1.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the deterministic observation summary forecast simulator.

    This is the script's programmatic entry point. It loads real or test
    summary artifacts, resolves the prior mean using the current real-summary
    policy, synthesizes contribution batches by deterministic replication, and
    runs :func:`update_observation_belief` for each requested accumulation
    count.

    Parameters
    ----------
    summary_paths :
        One or more ``subblock_summary.json`` artifacts.
    mode :
        Synthesis mode. Only ``"replicate"`` is supported in this first
        simulator.
    n_subblocks_grid :
        Comma-separated string or integer sequence of accumulation counts.
    results_root :
        Directory under which the run directory is created.
    run_name :
        Optional run directory name. A timestamped name is used when omitted.
    config_path, system_preset :
        Optional explicit prior-context overrides. When provided, they take
        precedence over summary ``theta_ref`` values.
    prior_source :
        Prior mean policy: ``auto``, ``summary_theta_ref``,
        ``resolved_system``, or ``default_system``.
    prior_sigma_scale :
        Positive multiplier applied to the default prior sigma vector.
    dry_run :
        Resolve inputs, prior context, planned paths, and grid without running
        updates or writing artifacts.

    Returns
    -------
    dict
        Run summary, manifest payload, and artifact paths.

    Raises
    ------
    ValueError
        Raised for unsupported modes, invalid grids, invalid prior sigma scale,
        missing summaries, or mismatched ``theta_labels``.
    """

    if mode != "replicate":
        raise ValueError(
            "Unsupported observation summary simulator mode "
            f"{mode!r}. The only supported mode is 'replicate'."
        )
    grid = parse_n_subblocks_grid(n_subblocks_grid)
    resolved_run_name = run_name or f"observation_summary_simulator_{timestamp_tag()}"
    run_dir = Path(results_root).resolve() / resolved_run_name
    planned_artifacts = build_artifact_paths(run_dir)

    summaries, payloads, resolved_summary_paths = load_summary_artifacts(summary_paths)
    theta_labels = require_identical_theta_labels(summaries)
    validate_required_forecast_labels(theta_labels)
    prior_context = resolve_prior_context_for_summaries(
        summaries,
        summary_paths=resolved_summary_paths,
        explicit_config_path=config_path,
        explicit_system_preset=system_preset,
        prior_source=str(prior_source),
        allow_summary_theta_ref_default=True,
    )
    prior, prior_sigma = build_prior_state(
        theta_labels=theta_labels,
        prior_context=prior_context,
        prior_sigma_scale=float(prior_sigma_scale),
    )

    prior_mean = np.asarray(prior_context.prior_mean, dtype=float)
    input_summary_payload = [
        {
            "source_summary_index": int(index),
            "summary_json_path": str(resolved_summary_paths[index]),
            "schema_version": payloads[index].get("schema_version"),
            "subblock_id": summaries[index].subblock_id,
            "summary_kind": summaries[index].summary_kind,
            "theta_ref": _theta_value_map(theta_labels, summaries[index].theta_ref),
            "diagnostics": dict(summaries[index].diagnostics),
            "artifact_metadata": payloads[index].get("metadata"),
            "artifact_prior_context": payloads[index].get("prior_context"),
        }
        for index in range(len(summaries))
    ]
    manifest: dict[str, Any] = {
        "schema_version": SIMULATOR_SCHEMA_VERSION,
        "generator": "run_observation_summary_simulator.py",
        "created_at": now_iso_local_ms(),
        "mode": str(mode),
        "run_name": resolved_run_name,
        "run_dir": str(run_dir),
        "dry_run": bool(dry_run),
        "input_summary_paths": [str(path) for path in resolved_summary_paths],
        "input_summaries": input_summary_payload,
        "theta_labels": list(theta_labels),
        "theta_dim": int(len(theta_labels)),
        "n_subblocks_grid": list(grid),
        "prior_source_requested": str(prior_source),
        "prior": {
            "mean_source": str(prior_context.prior_mean_source),
            "mean": _theta_value_map(theta_labels, prior_mean),
            "sigma": _theta_value_map(theta_labels, prior_sigma),
            "provenance": dict(prior_context.provenance),
            "warnings": list(prior_context.warnings),
        },
        "prior_sigma_policy": {
            "base": PRIOR_SIGMA_HELPER_PATH,
            "scale": float(prior_sigma_scale),
        },
        "output_artifact_paths": dict(planned_artifacts),
        "replicate_mode": {
            "ordering": "tile_input_order_then_truncate",
            "limitations": (
                "Deterministic replicate mode preserves source summary score "
                "vectors and matrices exactly. It is an accumulation sanity "
                "check, not a realism model, and does not add score noise, "
                "bootstrap summaries, or sample matrix entries."
            ),
            "future_modes_not_implemented": [
                "fixed_information_score_noise",
                "bootstrap_real_summaries",
                "trajectory_conditioned_summaries",
            ],
        },
    }

    if dry_run:
        return {
            "dry_run": True,
            "run_dir": str(run_dir),
            "planned_artifacts": planned_artifacts,
            "artifacts": {},
            "manifest": manifest,
            "theta_labels": list(theta_labels),
            "n_subblocks_grid": list(grid),
            "prior_mean_source": str(prior_context.prior_mean_source),
        }

    forecast_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    sigma_history_rows: list[dict[str, Any]] = []
    information_rows: list[dict[str, Any]] = []
    replicate_provenance_by_n: dict[str, Any] = {}

    for n_subblocks in grid:
        replicated = replicate_summaries(summaries, n_subblocks=int(n_subblocks))
        update_result = update_observation_belief(prior, replicated.summaries)
        forecast_rows.append(
            build_forecast_row(
                n_subblocks=int(n_subblocks),
                mode=mode,
                n_input_summaries=len(summaries),
                theta_labels=theta_labels,
                prior_sigma=prior_sigma,
                update_result=update_result,
            )
        )
        posterior_rows.extend(
            build_posterior_table_rows(
                n_subblocks=int(n_subblocks),
                theta_labels=theta_labels,
                prior_mean=prior_mean,
                prior_sigma=prior_sigma,
                update_result=update_result,
            )
        )
        sigma_history_rows.extend(
            build_cumulative_sigma_history_rows(
                n_subblocks=int(n_subblocks),
                theta_labels=theta_labels,
                prior_sigma=prior_sigma,
                update_result=update_result,
            )
        )
        information_rows.append(
            build_information_diagnostic_row(
                n_subblocks=int(n_subblocks),
                theta_labels=theta_labels,
                summaries=replicated.summaries,
                update_result=update_result,
            )
        )
        replicate_provenance_by_n[str(n_subblocks)] = replicated.provenance

    manifest["replicate_mode"]["provenance_by_n_subblocks"] = (
        replicate_provenance_by_n
    )
    manifest["forecast_summary"] = {
        "separation_label": SEPARATION_LABEL,
        "separation_units": "microarcsec",
        "n_rows": int(len(forecast_rows)),
    }

    _ensure_dir(run_dir)
    _write_json(Path(planned_artifacts["manifest_json"]), manifest)
    _write_csv_rows(Path(planned_artifacts["forecast_results_csv"]), forecast_rows)
    _write_csv_rows(
        Path(planned_artifacts["posterior_table_by_n_subblocks_csv"]),
        posterior_rows,
    )
    _write_csv_rows(
        Path(planned_artifacts["cumulative_sigma_history_csv"]),
        sigma_history_rows,
    )
    _write_csv_rows(
        Path(planned_artifacts["information_diagnostics_csv"]),
        information_rows,
    )
    _plot_separation_sigma(
        path=Path(planned_artifacts["separation_sigma_vs_n_subblocks_png"]),
        forecast_rows=forecast_rows,
    )
    _plot_prior_normalized_sigma(
        path=Path(planned_artifacts["prior_normalized_sigma_vs_n_subblocks_png"]),
        theta_labels=theta_labels,
        posterior_rows=posterior_rows,
    )

    return {
        "dry_run": False,
        "run_dir": str(run_dir),
        "planned_artifacts": planned_artifacts,
        "artifacts": dict(planned_artifacts),
        "manifest": manifest,
        "forecast_rows": forecast_rows,
        "theta_labels": list(theta_labels),
        "n_subblocks_grid": list(grid),
        "prior_mean_source": str(prior_context.prior_mean_source),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the summary simulator.

    The parser is called by ``main`` and kept separate so tests can inspect CLI
    behavior without running an update.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Forecast observation-level uncertainty by deterministic replication "
            "of existing SubblockSummary artifacts."
        ),
    )
    parser.add_argument(
        "--summary-json",
        nargs="+",
        type=Path,
        required=True,
        help="One or more image-backed subblock_summary.json artifacts.",
    )
    parser.add_argument(
        "--mode",
        default="replicate",
        choices=("replicate",),
        help="Summary synthesis mode. Only deterministic replicate mode is supported.",
    )
    parser.add_argument(
        "--n-subblocks",
        default="1,3,10,30,100,300,1800",
        help="Comma-separated accumulated sub-block counts to evaluate.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory for simulator outputs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run directory name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML/JSON config with a top-level system block. This is "
            "an explicit prior-context override."
        ),
    )
    parser.add_argument(
        "--system-preset",
        type=str,
        default=None,
        help=(
            "Optional system preset used as an explicit prior-context override."
        ),
    )
    parser.add_argument(
        "--prior-source",
        choices=("auto", "summary_theta_ref", "resolved_system", "default_system"),
        default="auto",
        help=(
            "Prior-mean initialization policy. Explicit config or system preset "
            "takes precedence over this setting."
        ),
    )
    parser.add_argument(
        "--prior-sigma-scale",
        type=float,
        default=1.0,
        help="Scalar multiplier applied to the default prior sigma vector.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve inputs and planned paths without running updates or writing plots.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Parse CLI arguments and run the observation summary simulator.

    Parameters
    ----------
    argv :
        Optional argument sequence. ``None`` uses ``sys.argv``.

    Returns
    -------
    dict
        Programmatic simulator result.
    """

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        grid = parse_n_subblocks_grid(args.n_subblocks)
    except ValueError as exc:
        parser.error(str(exc))
    return run_observation_summary_simulator(
        summary_paths=tuple(args.summary_json),
        mode=str(args.mode),
        n_subblocks_grid=grid,
        results_root=args.results_root,
        run_name=args.run_name,
        config_path=args.config,
        system_preset=args.system_preset,
        prior_source=str(args.prior_source),
        prior_sigma_scale=float(args.prior_sigma_scale),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
