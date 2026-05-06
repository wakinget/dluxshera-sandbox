"""Forecast observation-level uncertainty from existing summary artifacts.

This script is a narrow simulator layer above the image-backed sub-block
summary handoff:

``sub-block solve -> SubblockSummary handoff -> observation-level forecast``.

It consumes one or more existing ``subblock_summary.json`` artifacts and
evaluates how the posterior uncertainty on ``source.separation_as`` changes as
more synthetic summary contributions are accumulated.

``replicate`` mode tiles complete summaries deterministically, preserving both
the reduced information matrix ``S`` and reduced score vector ``g``. It is an
accumulation sanity check and repeats one score fluctuation coherently.

``fixed_information_score_noise`` mode tiles the same reduced information
matrices and linearization points, but draws independent synthetic score
vectors with covariance ``alpha * S`` around the score expected for a chosen
truth vector. It is the first stochastic summary simulator mode and is not yet
calibrated by real summary Monte Carlo. Bootstrap and trajectory-conditioned
summary modes remain future work.
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
MODE_REPLICATE = "replicate"
MODE_FIXED_INFORMATION_SCORE_NOISE = "fixed_information_score_noise"
SUPPORTED_MODES = (MODE_REPLICATE, MODE_FIXED_INFORMATION_SCORE_NOISE)
TRUTH_MODE_THETA_REF = "theta-ref"
TRUTH_MODE_PRIOR_MEAN = "prior-mean"
TRUTH_MODE_EXPLICIT = "explicit"
TRUTH_MODE_OFFSET = "offset"
SUPPORTED_TRUTH_MODES = (
    TRUTH_MODE_THETA_REF,
    TRUTH_MODE_PRIOR_MEAN,
    TRUTH_MODE_EXPLICIT,
    TRUTH_MODE_OFFSET,
)
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


@dataclass(frozen=True)
class ScoreNoiseSummaryBatch:
    """Store one nested-prefix stochastic summary sequence.

    The simulator builds this once per stochastic trial, then passes prefixes of
    ``summaries`` to :func:`update_observation_belief` for each requested
    ``n_subblocks`` value. The helper is local to this script for now; if the
    fixed-information score-noise model becomes part of a broader forecast API,
    this container and its synthesis helper are candidates for migration into
    ``dluxshera.inference.observation_forecast``.

    Parameters
    ----------
    summaries :
        Synthetic summaries in deterministic slot order. Each summary preserves
        the source template ``theta_ref`` and ``reduced_information`` but
        replaces ``reduced_score``.
    source_indices :
        Zero-based source-template index for each synthetic contribution.
    diagnostics :
        One compact diagnostic row per synthetic contribution.
    provenance :
        JSON-friendly batch provenance.
    """

    summaries: tuple[SubblockSummary, ...]
    source_indices: tuple[int, ...]
    diagnostics: tuple[dict[str, Any], ...]
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


def parse_simulator_mode(raw: str) -> str:
    """Validate one simulator synthesis mode string.

    Called by the CLI and tests before any summaries are loaded. The mode names
    are script-level orchestration choices, while the underlying belief update
    remains independent of synthesis mode.

    Parameters
    ----------
    raw :
        Requested mode string.

    Returns
    -------
    str
        Normalized supported mode.

    Raises
    ------
    ValueError
        Raised when the mode is unsupported.
    """

    mode = str(raw).strip()
    if mode not in SUPPORTED_MODES:
        supported = ", ".join(SUPPORTED_MODES)
        raise ValueError(
            f"Unsupported observation summary simulator mode {mode!r}. "
            f"Supported modes: {supported}."
        )
    return mode


def parse_truth_offset_mapping(raw: str | Mapping[str, float] | None) -> dict[str, float]:
    """Parse label offsets used by ``truth-mode=offset``.

    The parser is local simulator CLI glue. Offsets are interpreted in the
    physical units of each observation-level label and are added to the first
    input summary ``theta_ref`` when constructing ``theta_true``.

    Parameters
    ----------
    raw :
        Comma-separated ``LABEL=VALUE`` string, mapping, empty string, or
        ``None``.

    Returns
    -------
    dict
        Label to physical-unit offset.

    Raises
    ------
    ValueError
        Raised when an item is malformed or a value is not finite.
    """

    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        stripped = str(raw).strip()
        if not stripped:
            return {}
        pairs = []
        for token in stripped.split(","):
            piece = token.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise ValueError(
                    "truth offset entries must have form LABEL=VALUE; "
                    f"got {piece!r}."
                )
            key, value = piece.split("=", 1)
            pairs.append((key.strip(), value.strip()))
        items = pairs

    offsets: dict[str, float] = {}
    for raw_label, raw_value in items:
        label = str(raw_label).strip()
        if not label:
            raise ValueError("truth offset labels must be non-empty.")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"truth offset for {label!r} must be numeric; got {raw_value!r}."
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"truth offset for {label!r} must be finite.")
        offsets[label] = value
    return offsets


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


def sample_score_noise_from_information(
    information: np.ndarray,
    rng: np.random.Generator,
    *,
    alpha: float,
    eig_floor_abs: float,
    eig_floor_rel: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Draw PSD-safe Gaussian score noise from a reduced information matrix.

    ``fixed_information_score_noise`` calls this once per synthetic subblock.
    The score convention is local quadratic gradient noise: for a summary with
    reduced information ``S``, the synthetic perturbation is
    ``epsilon ~ Normal(0, alpha * S)``. The helper symmetrizes ``S``, samples in
    its eigenbasis, and floors small eigenvalues before taking square roots.

    Parameters
    ----------
    information :
        Square reduced information matrix ``S`` with shape ``(theta_dim,
        theta_dim)``.
    rng :
        NumPy random generator used for reproducible score draws.
    alpha :
        Non-negative score-noise covariance multiplier. ``0`` returns exactly
        zero noise.
    eig_floor_abs, eig_floor_rel :
        Absolute and relative eigenvalue floors. The effective floor is
        ``max(eig_floor_abs, eig_floor_rel * max(abs(lambda)))``.

    Returns
    -------
    noise, diagnostics :
        Score-noise vector and JSON-friendly sampling diagnostics.

    Raises
    ------
    ValueError
        Raised for invalid shapes, non-finite entries, negative alpha/floors, or
        significantly negative information eigenvalues.

    Notes
    -----
    This helper is local to the simulator. If stochastic summary synthesis moves
    into the library API, this PSD-safe sampler is a candidate for migration to
    ``dluxshera.inference.observation_forecast``.
    """

    matrix = np.asarray(information, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("score-noise sampling requires a square information matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("score-noise sampling requires finite information entries.")
    if float(alpha) < 0.0:
        raise ValueError("score_noise_alpha must be non-negative.")
    if float(eig_floor_abs) < 0.0:
        raise ValueError("score_noise_eig_floor_abs must be non-negative.")
    if float(eig_floor_rel) < 0.0:
        raise ValueError("score_noise_eig_floor_rel must be non-negative.")

    sym = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(sym)
    max_abs = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    floor = max(float(eig_floor_abs), float(eig_floor_rel) * max_abs)
    negative_tolerance = max(
        floor,
        np.finfo(float).eps * max(sym.shape) * max(max_abs, 1.0),
    )
    min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    if min_eigenvalue < -negative_tolerance:
        raise ValueError(
            "Reduced information matrix has significantly negative eigenvalues "
            f"for score-noise sampling: min_eigenvalue={min_eigenvalue:g}, "
            f"tolerance={negative_tolerance:g}."
        )

    clipped = np.maximum(eigenvalues, floor)
    diagnostics = {
        "sampling_method": "eigen_psd_floor",
        "alpha": float(alpha),
        "min_eigenvalue_raw": min_eigenvalue,
        "max_eigenvalue_raw": float(np.max(eigenvalues)) if eigenvalues.size else 0.0,
        "eig_floor_used": float(floor),
        "n_eigenvalues_below_floor": int(np.count_nonzero(eigenvalues < floor)),
        "negative_eigenvalue_tolerance": float(negative_tolerance),
    }
    if float(alpha) == 0.0:
        return np.zeros((matrix.shape[0],), dtype=float), diagnostics

    z = rng.normal(size=matrix.shape[0])
    noise = eigenvectors @ (np.sqrt(float(alpha) * clipped) * z)
    return np.asarray(noise, dtype=float), diagnostics


def construct_truth_vector(
    *,
    theta_labels: Sequence[str],
    summaries: Sequence[SubblockSummary],
    prior_context: PriorContext,
    truth_mode: str,
    truth_json_path: Path | None = None,
    truth_offset: str | Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Construct the physical truth vector for stochastic score synthesis.

    ``fixed_information_score_noise`` calls this after prior resolution and
    before drawing scores. The returned ``theta_true`` is used in the score
    convention ``g_expected = S @ (theta_ref - theta_true)`` for every
    synthetic summary. Shapes are one-dimensional and aligned with
    ``theta_labels``.

    Parameters
    ----------
    theta_labels :
        Ordered observation-level labels.
    summaries :
        Loaded source summaries. ``theta-ref`` and ``offset`` modes use the
        first summary ``theta_ref`` as their base.
    prior_context :
        Resolved prior context. ``prior-mean`` mode uses
        ``prior_context.prior_mean``.
    truth_mode :
        One of ``theta-ref``, ``prior-mean``, ``explicit``, or ``offset``.
    truth_json_path :
        Required for ``explicit`` mode. JSON object mapping every label to a
        finite numeric value.
    truth_offset :
        Optional offset mapping for ``offset`` mode.

    Returns
    -------
    theta_true, provenance :
        Truth vector and JSON-friendly provenance.

    Raises
    ------
    ValueError
        Raised for unsupported modes, missing explicit labels, unknown offset
        labels, or non-finite values.
    """

    labels = tuple(str(label) for label in theta_labels)
    if not summaries:
        raise ValueError("truth construction requires at least one summary.")
    mode = str(truth_mode).strip()
    if mode not in SUPPORTED_TRUTH_MODES:
        supported = ", ".join(SUPPORTED_TRUTH_MODES)
        raise ValueError(f"Unsupported truth_mode {mode!r}. Supported modes: {supported}.")

    provenance: dict[str, Any] = {"truth_mode": mode}
    if mode == TRUTH_MODE_THETA_REF:
        theta_true = np.asarray(summaries[0].theta_ref, dtype=float)
        provenance["base"] = "first_summary_theta_ref"
        provenance["source_template_subblock_id"] = summaries[0].subblock_id
    elif mode == TRUTH_MODE_PRIOR_MEAN:
        theta_true = np.asarray(prior_context.prior_mean, dtype=float)
        provenance["base"] = "prior_context.prior_mean"
        provenance["prior_mean_source"] = str(prior_context.prior_mean_source)
    elif mode == TRUTH_MODE_EXPLICIT:
        if truth_json_path is None:
            raise ValueError("truth-mode=explicit requires --truth-json.")
        with Path(truth_json_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("--truth-json must contain a JSON object.")
        missing = [label for label in labels if label not in payload]
        if missing:
            raise ValueError(
                "--truth-json must provide every theta label; missing: "
                + ", ".join(missing)
            )
        theta_true = np.asarray([float(payload[label]) for label in labels], dtype=float)
        provenance["truth_json_path"] = str(Path(truth_json_path).resolve())
    else:
        offsets = parse_truth_offset_mapping(truth_offset)
        unknown = sorted(label for label in offsets if label not in labels)
        if unknown:
            raise ValueError(
                "truth offsets contain labels outside theta_labels: "
                + ", ".join(unknown)
            )
        theta_true = np.asarray(summaries[0].theta_ref, dtype=float).copy()
        for index, label in enumerate(labels):
            theta_true[index] += float(offsets.get(label, 0.0))
        provenance["base"] = "first_summary_theta_ref"
        provenance["offsets"] = dict(offsets)
        provenance["source_template_subblock_id"] = summaries[0].subblock_id

    if theta_true.shape != (len(labels),):
        raise ValueError("truth vector shape must match theta_labels.")
    if not np.all(np.isfinite(theta_true)):
        raise ValueError("truth vector contains non-finite values.")
    provenance["truth"] = _theta_value_map(labels, theta_true)
    return theta_true, provenance


def synthesize_score_noise_summaries(
    templates: Sequence[SubblockSummary],
    *,
    n_subblocks: int,
    theta_true: np.ndarray,
    rng: np.random.Generator,
    trial_id: int,
    score_noise_alpha: float,
    eig_floor_abs: float,
    eig_floor_rel: float,
) -> ScoreNoiseSummaryBatch:
    """Build a nested-prefix batch with fixed information and noisy scores.

    ``fixed_information_score_noise`` calls this once per trial with
    ``n_subblocks=max(n_subblocks_grid)``. Template indices tile in deterministic
    input order, so requested grid points can use cumulative prefixes of the
    same synthetic observation sequence.

    The score convention is
    ``g_expected = S @ (theta_ref - theta_true)`` and
    ``g_synth = g_expected + epsilon`` with
    ``epsilon ~ Normal(0, alpha * S)`` sampled by
    :func:`sample_score_noise_from_information`. ``theta_true`` is a
    one-dimensional physical-basis vector aligned with the shared
    ``theta_labels``.

    Parameters
    ----------
    templates :
        Real source summaries with identical ordered ``theta_labels``.
    n_subblocks :
        Number of synthetic contributions to synthesize.
    theta_true :
        Physical truth vector aligned with the template labels.
    rng :
        Trial-local random generator.
    trial_id :
        Zero-based trial id for subblock ids and diagnostics.
    score_noise_alpha, eig_floor_abs, eig_floor_rel :
        Score-noise covariance multiplier and PSD eigenvalue-floor controls.

    Returns
    -------
    ScoreNoiseSummaryBatch
        Synthetic summaries, source indices, per-slot diagnostics, and compact
        provenance.

    Raises
    ------
    ValueError
        Raised when inputs are empty, shapes are inconsistent, or the sampler
        rejects an information matrix.
    """

    if n_subblocks <= 0:
        raise ValueError("n_subblocks must be positive.")
    source_summaries = tuple(templates)
    theta_labels = require_identical_theta_labels(source_summaries)
    truth = np.asarray(theta_true, dtype=float)
    if truth.shape != (len(theta_labels),):
        raise ValueError("theta_true shape must match template theta_labels.")
    if not np.all(np.isfinite(truth)):
        raise ValueError("theta_true must contain finite values.")

    n_sources = len(source_summaries)
    source_indices = tuple(index % n_sources for index in range(int(n_subblocks)))
    synthetic: list[SubblockSummary] = []
    diagnostics_rows: list[dict[str, Any]] = []
    for slot, source_index in enumerate(source_indices):
        template = source_summaries[source_index]
        information = np.asarray(template.reduced_information, dtype=float)
        theta_ref = np.asarray(template.theta_ref, dtype=float)
        g_expected = information @ (theta_ref - truth)
        epsilon, sampling_diag = sample_score_noise_from_information(
            information,
            rng,
            alpha=float(score_noise_alpha),
            eig_floor_abs=float(eig_floor_abs),
            eig_floor_rel=float(eig_floor_rel),
        )
        g_synth = g_expected + epsilon
        truth_offset = theta_ref - truth
        diagnostics = {
            "synthesis_mode": MODE_FIXED_INFORMATION_SCORE_NOISE,
            "source_template_subblock_id": template.subblock_id,
            "source_template_index": int(source_index),
            "trial_id": int(trial_id),
            "slot": int(slot),
            "score_noise_alpha": float(score_noise_alpha),
            "score_noise_norm": float(np.linalg.norm(epsilon)),
            "expected_score_norm": float(np.linalg.norm(g_expected)),
            "synthesized_score_norm": float(np.linalg.norm(g_synth)),
            "truth_offset_norm": float(np.linalg.norm(truth_offset)),
            **sampling_diag,
        }
        synthetic.append(
            SubblockSummary.from_reduced_form(
                subblock_id=(
                    f"{template.subblock_id}__score_noise_"
                    f"trial{int(trial_id):04d}_slot{int(slot):06d}"
                ),
                theta_labels=template.theta_labels,
                theta_ref=theta_ref,
                reduced_information=information,
                reduced_score=g_synth,
                summary_kind="synthetic_fixed_information_score_noise",
                diagnostics=diagnostics,
            )
        )
        diagnostics_rows.append(diagnostics)

    provenance = summarize_source_indices(
        source_indices,
        n_input_summaries=n_sources,
    )
    provenance.update(
        {
            "ordering": "tile_input_order_then_truncate",
            "nested_prefix_policy": "synthesize_max_grid_once_per_trial_then_slice_prefixes",
            "trial_id": int(trial_id),
        }
    )
    return ScoreNoiseSummaryBatch(
        summaries=tuple(synthetic),
        source_indices=source_indices,
        diagnostics=tuple(diagnostics_rows),
        provenance=provenance,
    )


def build_artifact_paths(run_dir: Path, *, mode: str = MODE_REPLICATE) -> dict[str, str]:
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

    if mode == MODE_REPLICATE:
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
    artifacts = {
        "manifest_json": str(run_dir / "manifest.json"),
        "forecast_results_csv": str(run_dir / "forecast_results.csv"),
        "information_diagnostics_csv": str(
            run_dir / "information_diagnostics.csv"
        ),
    }
    if mode == MODE_FIXED_INFORMATION_SCORE_NOISE:
        artifacts.update(
            {
                "trial_forecast_results_csv": str(
                    run_dir / "trial_forecast_results.csv"
                ),
                "trial_posterior_table_csv": str(
                    run_dir / "trial_posterior_table.csv"
                ),
                "stochastic_synthesis_diagnostics_csv": str(
                    run_dir / "stochastic_synthesis_diagnostics.csv"
                ),
                "separation_error_vs_n_subblocks_png": str(
                    run_dir / "separation_error_vs_n_subblocks.png"
                ),
            }
        )
    return artifacts


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


def build_trial_forecast_row(
    *,
    trial_id: int,
    n_subblocks: int,
    mode: str,
    n_input_summaries: int,
    theta_labels: Sequence[str],
    prior_sigma: np.ndarray,
    update_result: ObservationUpdateResult,
    theta_true: np.ndarray,
) -> dict[str, Any]:
    """Return one stochastic trial forecast row.

    Called by ``fixed_information_score_noise`` for every trial and requested
    accumulated subblock count. It extends the deterministic forecast sigma
    contract with truth, posterior mean, and separation error fields. Vectors
    are one-dimensional and aligned with ``theta_labels``.

    Parameters
    ----------
    trial_id, n_subblocks :
        Trial and nested-prefix accumulation identifiers.
    mode :
        Synthesis mode, expected to be ``fixed_information_score_noise``.
    n_input_summaries :
        Number of real template summaries.
    theta_labels :
        Ordered observation-level labels.
    prior_sigma :
        Prior sigma vector in the same basis.
    update_result :
        Posterior update result for this prefix.
    theta_true :
        Physical truth vector used to synthesize scores.

    Returns
    -------
    dict
        CSV-ready trial forecast row.
    """

    base = build_forecast_row(
        n_subblocks=n_subblocks,
        mode=mode,
        n_input_summaries=n_input_summaries,
        theta_labels=theta_labels,
        prior_sigma=prior_sigma,
        update_result=update_result,
    )
    labels = tuple(theta_labels)
    sep_index = labels.index(SEPARATION_LABEL)
    posterior_mean = np.asarray(update_result.posterior.mean, dtype=float)
    separation_truth = float(theta_true[sep_index])
    separation_posterior_mean = float(posterior_mean[sep_index])
    separation_error = separation_posterior_mean - separation_truth
    base.update(
        {
            "trial_id": int(trial_id),
            "separation_truth_as": separation_truth,
            "separation_posterior_mean_as": separation_posterior_mean,
            "separation_error_as": float(separation_error),
            "separation_error_uas": float(1.0e6 * separation_error),
            "separation_abs_error_uas": float(1.0e6 * abs(separation_error)),
        }
    )
    return base


def build_trial_posterior_table_rows(
    *,
    trial_id: int,
    n_subblocks: int,
    theta_labels: Sequence[str],
    prior_mean: np.ndarray,
    prior_sigma: np.ndarray,
    update_result: ObservationUpdateResult,
    theta_true: np.ndarray,
) -> list[dict[str, Any]]:
    """Return long-form posterior rows for one stochastic trial prefix.

    Called by ``fixed_information_score_noise`` to populate
    ``trial_posterior_table.csv``. Truth and posterior error are included for
    every parameter; ``posterior_error_uas`` is populated for
    ``source.separation_as`` only.

    Parameters
    ----------
    trial_id, n_subblocks :
        Trial and nested-prefix accumulation identifiers.
    theta_labels :
        Ordered observation-level labels.
    prior_mean, prior_sigma :
        Prior vectors in the same basis.
    update_result :
        Posterior update result.
    theta_true :
        Truth vector used for score synthesis.

    Returns
    -------
    list of dict
        One row per label.
    """

    posterior_mean = np.asarray(update_result.posterior.mean, dtype=float)
    posterior_sigma = update_result.posterior.sigma()
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(theta_labels):
        error = float(posterior_mean[index] - theta_true[index])
        row = {
            "trial_id": int(trial_id),
            "n_subblocks": int(n_subblocks),
            "label": str(label),
            "truth": float(theta_true[index]),
            "prior_mean": float(prior_mean[index]),
            "posterior_mean": float(posterior_mean[index]),
            "posterior_error": error,
            "prior_sigma": float(prior_sigma[index]),
            "posterior_sigma": float(posterior_sigma[index]),
            "posterior_sigma_over_prior_sigma": float(
                posterior_sigma[index] / prior_sigma[index]
            ),
            "unit": _parameter_units(str(label)),
        }
        if label == SEPARATION_LABEL:
            row["posterior_error_uas"] = float(1.0e6 * error)
        rows.append(row)
    return rows


def aggregate_stochastic_forecast_rows(
    trial_rows: Sequence[Mapping[str, Any]],
    *,
    n_trials: int,
) -> list[dict[str, Any]]:
    """Aggregate stochastic trial forecast rows by ``n_subblocks``.

    ``fixed_information_score_noise`` calls this after all trial updates. It
    summarizes empirical separation errors and posterior sigma distributions
    while preserving one aggregate row per requested accumulation count.

    Parameters
    ----------
    trial_rows :
        Rows produced by :func:`build_trial_forecast_row`.
    n_trials :
        Number of stochastic trials requested.

    Returns
    -------
    list of dict
        Aggregate ``forecast_results.csv`` rows for stochastic mode.
    """

    by_n: dict[int, list[Mapping[str, Any]]] = {}
    for row in trial_rows:
        by_n.setdefault(int(row["n_subblocks"]), []).append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for n_subblocks in sorted(by_n):
        rows = by_n[n_subblocks]
        errors = np.asarray([float(row["separation_error_uas"]) for row in rows])
        abs_errors = np.abs(errors)
        sigmas = np.asarray(
            [float(row["separation_posterior_sigma_uas"]) for row in rows],
            dtype=float,
        )
        median_sigma = float(np.median(sigmas))
        rms = float(np.sqrt(np.mean(np.square(errors))))
        first = rows[0]
        aggregate_rows.append(
            {
                "n_trials": int(n_trials),
                "n_subblocks": int(n_subblocks),
                "mode": MODE_FIXED_INFORMATION_SCORE_NOISE,
                "separation_truth_as": first.get("separation_truth_as"),
                "separation_bias_uas": float(np.mean(errors)),
                "separation_rms_error_uas": rms,
                "separation_mean_abs_error_uas": float(np.mean(abs_errors)),
                "separation_error_p16_uas": float(np.percentile(errors, 16)),
                "separation_error_p50_uas": float(np.percentile(errors, 50)),
                "separation_error_p84_uas": float(np.percentile(errors, 84)),
                "separation_abs_error_p50_uas": float(np.percentile(abs_errors, 50)),
                "separation_abs_error_p84_uas": float(np.percentile(abs_errors, 84)),
                "separation_posterior_sigma_mean_uas": float(np.mean(sigmas)),
                "separation_posterior_sigma_median_uas": median_sigma,
                "separation_posterior_sigma_p16_uas": float(np.percentile(sigmas, 16)),
                "separation_posterior_sigma_p84_uas": float(np.percentile(sigmas, 84)),
                "calibration_ratio_rms_over_median_sigma": (
                    None if median_sigma == 0.0 else float(rms / median_sigma)
                ),
                "posterior_precision_condition_number_mean": float(
                    np.mean(
                        [
                            float(row["posterior_precision_condition_number"])
                            for row in rows
                        ]
                    )
                ),
                "posterior_precision_trace_mean": float(
                    np.mean([float(row["posterior_precision_trace"]) for row in rows])
                ),
            }
        )
    return aggregate_rows


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


def _plot_stochastic_separation_error(
    *,
    path: Path,
    aggregate_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write the stochastic separation error calibration plot.

    Called only by ``fixed_information_score_noise`` after aggregate rows are
    built. The plot compares empirical RMS separation error against the median
    posterior sigma for each nested-prefix accumulation count.

    Parameters
    ----------
    path :
        PNG output path.
    aggregate_rows :
        Stochastic aggregate forecast rows.
    """

    x = np.asarray([int(row["n_subblocks"]) for row in aggregate_rows], dtype=int)
    rms = np.asarray(
        [float(row["separation_rms_error_uas"]) for row in aggregate_rows],
        dtype=float,
    )
    sigma = np.asarray(
        [
            float(row["separation_posterior_sigma_median_uas"])
            for row in aggregate_rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(x, np.clip(rms, 1.0e-12, None), marker="o", label="RMS error")
    ax.loglog(
        x,
        np.clip(sigma, 1.0e-12, None),
        marker="s",
        label="Median posterior sigma",
    )
    ax.set_xlabel("Accumulated Subblocks")
    ax.set_ylabel("source.separation_as (microarcsec)")
    ax.set_title("Stochastic Separation Error vs Posterior Sigma")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_observation_summary_simulator(
    *,
    summary_paths: Sequence[Path | str],
    mode: str = MODE_REPLICATE,
    n_subblocks_grid: str | Sequence[int] = (1, 3, 10, 30, 100, 300, 1800),
    results_root: Path | str = DEFAULT_RESULTS_ROOT,
    run_name: str | None = None,
    config_path: Path | None = None,
    system_preset: str | None = None,
    prior_source: str = "auto",
    prior_sigma_scale: float = 1.0,
    n_trials: int = 100,
    seed: int = 42,
    score_noise_alpha: float = 1.0,
    score_noise_eig_floor_abs: float = 0.0,
    score_noise_eig_floor_rel: float = 1.0e-12,
    truth_mode: str = TRUTH_MODE_THETA_REF,
    truth_json_path: Path | None = None,
    truth_offset: str | Mapping[str, float] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the observation summary forecast simulator.

    This is the script's programmatic entry point. It loads real or test
    summary artifacts, resolves the prior mean using the current real-summary
    policy, synthesizes contribution batches, and runs
    :func:`update_observation_belief` for each requested accumulation count.

    Parameters
    ----------
    summary_paths :
        One or more ``subblock_summary.json`` artifacts.
    mode :
        Synthesis mode: deterministic ``replicate`` or stochastic
        ``fixed_information_score_noise``.
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
    n_trials, seed :
        Stochastic trial count and base seed. Used only by
        ``fixed_information_score_noise``.
    score_noise_alpha, score_noise_eig_floor_abs, score_noise_eig_floor_rel :
        Score-noise covariance multiplier and PSD eigenvalue-floor controls.
    truth_mode, truth_json_path, truth_offset :
        Truth-vector policy and optional explicit/offset inputs for stochastic
        score synthesis.
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

    mode = parse_simulator_mode(mode)
    if int(n_trials) <= 0:
        raise ValueError("n_trials must be positive.")
    if float(score_noise_alpha) < 0.0:
        raise ValueError("score_noise_alpha must be non-negative.")
    if float(score_noise_eig_floor_abs) < 0.0:
        raise ValueError("score_noise_eig_floor_abs must be non-negative.")
    if float(score_noise_eig_floor_rel) < 0.0:
        raise ValueError("score_noise_eig_floor_rel must be non-negative.")
    grid = parse_n_subblocks_grid(n_subblocks_grid)
    resolved_run_name = run_name or f"observation_summary_simulator_{timestamp_tag()}"
    run_dir = Path(results_root).resolve() / resolved_run_name
    planned_artifacts = build_artifact_paths(run_dir, mode=mode)

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
    theta_true: np.ndarray | None = None
    truth_provenance: dict[str, Any] | None = None
    if mode == MODE_FIXED_INFORMATION_SCORE_NOISE:
        theta_true, truth_provenance = construct_truth_vector(
            theta_labels=theta_labels,
            summaries=summaries,
            prior_context=prior_context,
            truth_mode=truth_mode,
            truth_json_path=truth_json_path,
            truth_offset=truth_offset,
        )
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
                "bootstrap_real_summaries",
                "trajectory_conditioned_summaries",
            ],
        },
    }
    if mode == MODE_FIXED_INFORMATION_SCORE_NOISE:
        assert truth_provenance is not None
        manifest["stochastic_mode"] = {
            "mode": MODE_FIXED_INFORMATION_SCORE_NOISE,
            "n_trials": int(n_trials),
            "seed": int(seed),
            "score_noise_alpha": float(score_noise_alpha),
            "score_noise_eig_floor_abs": float(score_noise_eig_floor_abs),
            "score_noise_eig_floor_rel": float(score_noise_eig_floor_rel),
            "sampling_method": "eigen_psd_floor",
            "nested_prefix_policy": (
                "synthesize_max_grid_once_per_trial_then_slice_prefixes"
            ),
            "truth": truth_provenance,
            "limitations": (
                "Synthetic fixed-information score-noise mode assumes stable "
                "reduced_information and Fisher-consistent score covariance "
                "alpha * S. Alpha is not yet calibrated by real summary Monte "
                "Carlo."
            ),
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

    if mode == MODE_FIXED_INFORMATION_SCORE_NOISE:
        assert theta_true is not None
        max_n_subblocks = int(max(grid))
        trial_forecast_rows: list[dict[str, Any]] = []
        trial_posterior_rows: list[dict[str, Any]] = []
        information_rows: list[dict[str, Any]] = []
        synthesis_diag_rows: list[dict[str, Any]] = []
        trial_provenance: dict[str, Any] = {}

        for trial_id in range(int(n_trials)):
            seed_sequence = np.random.SeedSequence(int(seed), spawn_key=(trial_id,))
            rng = np.random.default_rng(seed_sequence)
            batch = synthesize_score_noise_summaries(
                summaries,
                n_subblocks=max_n_subblocks,
                theta_true=theta_true,
                rng=rng,
                trial_id=trial_id,
                score_noise_alpha=float(score_noise_alpha),
                eig_floor_abs=float(score_noise_eig_floor_abs),
                eig_floor_rel=float(score_noise_eig_floor_rel),
            )
            trial_provenance[str(trial_id)] = batch.provenance
            score_noise_norms = np.asarray(
                [row["score_noise_norm"] for row in batch.diagnostics],
                dtype=float,
            )
            expected_norms = np.asarray(
                [row["expected_score_norm"] for row in batch.diagnostics],
                dtype=float,
            )
            synthesized_norms = np.asarray(
                [row["synthesized_score_norm"] for row in batch.diagnostics],
                dtype=float,
            )
            eig_floors = np.asarray(
                [row["eig_floor_used"] for row in batch.diagnostics],
                dtype=float,
            )
            below_floor = np.asarray(
                [row["n_eigenvalues_below_floor"] for row in batch.diagnostics],
                dtype=float,
            )
            synthesis_diag_rows.append(
                {
                    "trial_id": int(trial_id),
                    "max_n_subblocks": max_n_subblocks,
                    "score_noise_alpha": float(score_noise_alpha),
                    "mean_score_noise_norm": float(np.mean(score_noise_norms)),
                    "median_score_noise_norm": float(np.median(score_noise_norms)),
                    "max_score_noise_norm": float(np.max(score_noise_norms)),
                    "mean_expected_score_norm": float(np.mean(expected_norms)),
                    "mean_synthesized_score_norm": float(np.mean(synthesized_norms)),
                    "mean_eig_floor_used": float(np.mean(eig_floors)),
                    "max_eig_floor_used": float(np.max(eig_floors)),
                    "mean_n_eigenvalues_below_floor": float(np.mean(below_floor)),
                    "template_counts": json.dumps(
                        batch.provenance["counts_by_source_summary_index"]
                    ),
                }
            )

            for n_subblocks in grid:
                prefix = batch.summaries[: int(n_subblocks)]
                update_result = update_observation_belief(prior, prefix)
                trial_forecast_rows.append(
                    build_trial_forecast_row(
                        trial_id=trial_id,
                        n_subblocks=int(n_subblocks),
                        mode=mode,
                        n_input_summaries=len(summaries),
                        theta_labels=theta_labels,
                        prior_sigma=prior_sigma,
                        update_result=update_result,
                        theta_true=theta_true,
                    )
                )
                trial_posterior_rows.extend(
                    build_trial_posterior_table_rows(
                        trial_id=trial_id,
                        n_subblocks=int(n_subblocks),
                        theta_labels=theta_labels,
                        prior_mean=prior_mean,
                        prior_sigma=prior_sigma,
                        update_result=update_result,
                        theta_true=theta_true,
                    )
                )
                info_row = build_information_diagnostic_row(
                    n_subblocks=int(n_subblocks),
                    theta_labels=theta_labels,
                    summaries=prefix,
                    update_result=update_result,
                )
                info_row["trial_id"] = int(trial_id)
                information_rows.append(info_row)

        forecast_rows = aggregate_stochastic_forecast_rows(
            trial_forecast_rows,
            n_trials=int(n_trials),
        )
        manifest["stochastic_mode"]["trial_provenance"] = trial_provenance
        manifest["forecast_summary"] = {
            "separation_label": SEPARATION_LABEL,
            "separation_units": "microarcsec",
            "n_trials": int(n_trials),
            "n_aggregate_rows": int(len(forecast_rows)),
            "n_trial_rows": int(len(trial_forecast_rows)),
        }

        _ensure_dir(run_dir)
        _write_json(Path(planned_artifacts["manifest_json"]), manifest)
        _write_csv_rows(Path(planned_artifacts["forecast_results_csv"]), forecast_rows)
        _write_csv_rows(
            Path(planned_artifacts["trial_forecast_results_csv"]),
            trial_forecast_rows,
        )
        _write_csv_rows(
            Path(planned_artifacts["trial_posterior_table_csv"]),
            trial_posterior_rows,
        )
        _write_csv_rows(
            Path(planned_artifacts["information_diagnostics_csv"]),
            information_rows,
        )
        _write_csv_rows(
            Path(planned_artifacts["stochastic_synthesis_diagnostics_csv"]),
            synthesis_diag_rows,
        )
        _plot_stochastic_separation_error(
            path=Path(planned_artifacts["separation_error_vs_n_subblocks_png"]),
            aggregate_rows=forecast_rows,
        )

        return {
            "dry_run": False,
            "run_dir": str(run_dir),
            "planned_artifacts": planned_artifacts,
            "artifacts": dict(planned_artifacts),
            "manifest": manifest,
            "forecast_rows": forecast_rows,
            "trial_forecast_rows": trial_forecast_rows,
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
            "Forecast observation-level uncertainty by synthesizing accumulated "
            "SubblockSummary contributions."
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
        default=MODE_REPLICATE,
        choices=SUPPORTED_MODES,
        help=(
            "Summary synthesis mode: deterministic replicate or stochastic "
            "fixed-information score-noise synthesis."
        ),
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
        "--n-trials",
        type=int,
        default=100,
        help=(
            "Number of independent stochastic trials for "
            "fixed_information_score_noise."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for stochastic score-noise draws.",
    )
    parser.add_argument(
        "--score-noise-alpha",
        type=float,
        default=1.0,
        help="Non-negative multiplier on score-noise covariance alpha * S.",
    )
    parser.add_argument(
        "--score-noise-eig-floor-abs",
        type=float,
        default=0.0,
        help="Absolute eigenvalue floor for score-noise sampling.",
    )
    parser.add_argument(
        "--score-noise-eig-floor-rel",
        type=float,
        default=1.0e-12,
        help="Relative eigenvalue floor for score-noise sampling.",
    )
    parser.add_argument(
        "--truth-mode",
        choices=SUPPORTED_TRUTH_MODES,
        default=TRUTH_MODE_THETA_REF,
        help="Truth vector policy for fixed_information_score_noise.",
    )
    parser.add_argument(
        "--truth-json",
        type=Path,
        default=None,
        help="JSON label-to-value mapping required when --truth-mode=explicit.",
    )
    parser.add_argument(
        "--truth-offset",
        type=str,
        default=None,
        help=(
            "Comma-separated LABEL=VALUE offsets for --truth-mode=offset, in "
            "physical units."
        ),
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
        n_trials=int(args.n_trials),
        seed=int(args.seed),
        score_noise_alpha=float(args.score_noise_alpha),
        score_noise_eig_floor_abs=float(args.score_noise_eig_floor_abs),
        score_noise_eig_floor_rel=float(args.score_noise_eig_floor_rel),
        truth_mode=str(args.truth_mode),
        truth_json_path=args.truth_json,
        truth_offset=args.truth_offset,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
