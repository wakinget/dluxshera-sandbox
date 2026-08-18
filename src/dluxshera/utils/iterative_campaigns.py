"""Small helpers for physical-basis iterative campaign updates."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

POSTERIOR_MEAN_COLUMNS = ("posterior_mean", "mean", "posterior", "value")
SEPARATION_LABEL = "source.separation_as"

__all__ = [
    "POSTERIOR_MEAN_COLUMNS",
    "SEPARATION_LABEL",
    "PhysicalUpdateComponents",
    "apply_physical_reference_update",
    "label_offsets_to_vector",
    "physical_update_components",
    "posterior_float",
    "posterior_label",
    "posterior_offsets_from_rows",
    "posterior_rows_by_label",
    "separation_update_diagnostics",
    "vector_update_diagnostics",
]


@dataclass(frozen=True)
class PhysicalUpdateComponents:
    """Vector components for one physical-basis iterative update.

    Attributes
    ----------
    labels
        Physical parameter labels defining vector order.
    reference
        Current reference error vector, stored as offsets relative to truth.
    posterior
        Posterior error vector, stored as offsets relative to truth.
    next_reference
        Next reference error vector after applying the configured gain.
    posterior_update
        Full posterior shift, ``posterior - reference``.
    applied_update
        Actual reference shift, ``next_reference - reference``.
    ideal_update
        Truth-directed correction, ``-reference``.
    """

    labels: tuple[str, ...]
    reference: np.ndarray
    posterior: np.ndarray
    next_reference: np.ndarray
    posterior_update: np.ndarray
    applied_update: np.ndarray
    ideal_update: np.ndarray


def posterior_label(row: Mapping[str, Any]) -> str:
    """Return the physical parameter label from a posterior table row.

    Parameters
    ----------
    row
        Mapping from CSV/JSON row column names to values. The first non-empty
        label candidate among ``theta_label``, ``parameter``, ``label``, and
        ``name`` is returned.

    Returns
    -------
    str
        Resolved label, or an empty string when no candidate is present.
    """

    for key in ("theta_label", "parameter", "label", "name"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def posterior_float(row: Mapping[str, Any], candidates: Sequence[str]) -> float:
    """Read the first float-compatible value from a posterior row.

    Parameters
    ----------
    row
        Mapping-valued row.
    candidates
        Column names to try in order.

    Returns
    -------
    float
        First successfully parsed value, or ``NaN`` when all candidates are
        absent, empty, or non-numeric.
    """

    for key in candidates:
        value = row.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return float("nan")


def posterior_rows_by_label(path: Path) -> dict[str, dict[str, str]]:
    """Load a CSV posterior table indexed by physical parameter label.

    Parameters
    ----------
    path
        CSV file containing a posterior table.

    Returns
    -------
    dict[str, dict[str, str]]
        Rows keyed by physical parameter label. Missing files return an empty
        mapping so aggregate-only callers can report missing outputs cleanly.
    """

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {posterior_label(row): row for row in rows if posterior_label(row)}


def apply_physical_reference_update(
    *,
    current_offsets: Mapping[str, float],
    posterior_rows_by_label: Mapping[str, Mapping[str, Any]],
    truth_by_label: Mapping[str, float],
    update_gain: float,
    posterior_mean_columns: Sequence[str] = POSTERIOR_MEAN_COLUMNS,
) -> dict[str, float]:
    """Apply the physical-basis reference update.

    The update is
    ``theta_ref_next = theta_ref_current + gain * (posterior_mean - theta_ref_current)``.
    Offsets are stored relative to truth. Labels absent from ``truth_by_label``
    are ignored. Non-finite posterior means leave the current offset unchanged.

    Parameters
    ----------
    current_offsets
        Current reference offsets relative to truth.
    posterior_rows_by_label
        Posterior table rows keyed by physical label.
    truth_by_label
        Truth values keyed by physical label.
    update_gain
        Finite scalar gain applied to the posterior shift.
    posterior_mean_columns
        Candidate columns for the absolute posterior mean.

    Returns
    -------
    dict[str, float]
        Next reference offsets relative to truth.
    """

    if not math.isfinite(float(update_gain)):
        raise ValueError("update_gain must be finite.")
    next_offsets = {str(label): float(value) for label, value in current_offsets.items()}
    for label, posterior_row in posterior_rows_by_label.items():
        if label not in truth_by_label:
            continue
        posterior_mean = posterior_float(posterior_row, posterior_mean_columns)
        if not math.isfinite(posterior_mean):
            continue
        truth = float(truth_by_label[label])
        theta_ref_current = truth + float(current_offsets.get(label, 0.0))
        theta_ref_next = theta_ref_current + float(update_gain) * (posterior_mean - theta_ref_current)
        next_offsets[label] = float(theta_ref_next - truth)
    return next_offsets


def posterior_offsets_from_rows(
    *,
    labels: Sequence[str],
    posterior_rows_by_label: Mapping[str, Mapping[str, Any]],
    truth_by_label: Mapping[str, float],
    fallback_offsets: Mapping[str, float] | None = None,
    posterior_mean_columns: Sequence[str] = POSTERIOR_MEAN_COLUMNS,
) -> tuple[dict[str, float], dict[str, str]]:
    """Convert absolute posterior means to offsets relative to truth.

    Parameters
    ----------
    labels
        Physical labels to include in output order.
    posterior_rows_by_label
        Posterior rows keyed by label.
    truth_by_label
        Truth values keyed by label.
    fallback_offsets
        Optional offsets to use when a posterior row is missing or non-finite.
    posterior_mean_columns
        Candidate columns for absolute posterior mean.

    Returns
    -------
    tuple[dict[str, float], dict[str, str]]
        Posterior offsets and per-label status values: ``ok``,
        ``missing_posterior_row``, ``missing_truth``, or
        ``nonfinite_posterior_mean``.
    """

    fallback = fallback_offsets or {}
    offsets: dict[str, float] = {}
    status: dict[str, str] = {}
    for label in tuple(str(item) for item in labels):
        row = posterior_rows_by_label.get(label)
        if row is None:
            offsets[label] = float(fallback.get(label, 0.0))
            status[label] = "missing_posterior_row"
            continue
        if label not in truth_by_label:
            offsets[label] = float(fallback.get(label, 0.0))
            status[label] = "missing_truth"
            continue
        posterior_mean = posterior_float(row, posterior_mean_columns)
        if not math.isfinite(posterior_mean):
            offsets[label] = float(fallback.get(label, 0.0))
            status[label] = "nonfinite_posterior_mean"
            continue
        offsets[label] = float(posterior_mean - float(truth_by_label[label]))
        status[label] = "ok"
    return offsets, status


def label_offsets_to_vector(
    offsets: Mapping[str, float],
    labels: Sequence[str],
    *,
    default: float = 0.0,
) -> np.ndarray:
    """Convert label-indexed offsets to a vector in ``labels`` order."""

    return np.asarray([float(offsets.get(label, default)) for label in labels], dtype=float)


def physical_update_components(
    *,
    labels: Sequence[str],
    current_offsets: Mapping[str, float],
    posterior_offsets: Mapping[str, float],
    next_offsets: Mapping[str, float] | None = None,
) -> PhysicalUpdateComponents:
    """Build vector components for posterior and applied update diagnostics.

    Parameters
    ----------
    labels
        Physical labels defining vector order.
    current_offsets
        Current reference offsets relative to truth.
    posterior_offsets
        Posterior offsets relative to truth.
    next_offsets
        Next reference offsets. If omitted, the next reference is treated as
        the posterior vector, matching a unit-gain update.

    Returns
    -------
    PhysicalUpdateComponents
        Reference, posterior, next-reference, posterior update, applied update,
        and ideal truth-directed update vectors.
    """

    label_order = tuple(str(label) for label in labels)
    reference = label_offsets_to_vector(current_offsets, label_order)
    posterior = label_offsets_to_vector(posterior_offsets, label_order)
    next_reference = (
        label_offsets_to_vector(next_offsets, label_order)
        if next_offsets is not None
        else posterior.copy()
    )
    return PhysicalUpdateComponents(
        labels=label_order,
        reference=reference,
        posterior=posterior,
        next_reference=next_reference,
        posterior_update=posterior - reference,
        applied_update=next_reference - reference,
        ideal_update=-reference,
    )


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) <= 1.0e-30:
        return float("nan")
    return float(numerator / denominator)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _projection_gain(update: np.ndarray, ideal: np.ndarray) -> float:
    denom = float(np.dot(ideal, ideal))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(update, ideal) / denom)


def vector_update_diagnostics(
    *,
    labels: Sequence[str],
    current_offsets: Mapping[str, float],
    posterior_offsets: Mapping[str, float],
    next_offsets: Mapping[str, float] | None = None,
    previous_residual_norm: float | None = None,
    previous_next_reference_norm: float | None = None,
) -> dict[str, Any]:
    """Return compact vector diagnostics for one physical-basis update.

    Backward-compatible fields ``update_norm``, ``update_cosine_with_ideal``,
    and ``vector_gain`` describe the posterior update, not the damped/applied
    update. New ``posterior_*`` and ``applied_*`` fields distinguish these
    quantities when ``update_gain != 1``.
    """

    components = physical_update_components(
        labels=labels,
        current_offsets=current_offsets,
        posterior_offsets=posterior_offsets,
        next_offsets=next_offsets,
    )
    bias_norm = float(np.linalg.norm(components.reference))
    posterior_norm = float(np.linalg.norm(components.posterior))
    next_norm = float(np.linalg.norm(components.next_reference))
    posterior_update_norm = float(np.linalg.norm(components.posterior_update))
    applied_update_norm = float(np.linalg.norm(components.applied_update))
    return {
        "reference_error_norm_before": bias_norm,
        "posterior_error_norm_after": posterior_norm,
        "posterior_update_norm": posterior_update_norm,
        "posterior_update_cosine_with_ideal": _cosine(components.posterior_update, components.ideal_update),
        "posterior_vector_gain": _projection_gain(components.posterior_update, components.ideal_update),
        "applied_update_norm": applied_update_norm,
        "applied_update_cosine_with_ideal": _cosine(components.applied_update, components.ideal_update),
        "applied_vector_gain": _projection_gain(components.applied_update, components.ideal_update),
        "next_reference_error_norm": next_norm,
        "next_reference_error_norm_over_bias_norm": _safe_fraction(next_norm, bias_norm),
        "residual_norm_over_bias_norm": _safe_fraction(posterior_norm, bias_norm),
        "residual_norm_decreased_from_previous_window": ""
        if previous_residual_norm is None
        else bool(posterior_norm < float(previous_residual_norm)),
        "next_reference_residual_decreased_from_previous_window": ""
        if previous_next_reference_norm is None
        else bool(next_norm < float(previous_next_reference_norm)),
        # Backward-compatible aliases: these describe the posterior update.
        "update_norm": posterior_update_norm,
        "update_cosine_with_ideal": _cosine(components.posterior_update, components.ideal_update),
        "vector_gain": _projection_gain(components.posterior_update, components.ideal_update),
    }


def separation_update_diagnostics(
    *,
    current_offsets: Mapping[str, float],
    posterior_offsets: Mapping[str, float],
    next_offsets: Mapping[str, float],
    label: str = SEPARATION_LABEL,
) -> dict[str, Any]:
    """Return separation-specific diagnostics in microarcseconds.

    Parameters
    ----------
    current_offsets, posterior_offsets, next_offsets
        Offset mappings relative to truth.
    label
        Separation label. Defaults to ``source.separation_as``.

    Returns
    -------
    dict[str, Any]
        Microarcsecond error/update fields and sign/improvement flags.
    """

    current = float(current_offsets.get(label, 0.0))
    posterior = float(posterior_offsets.get(label, current))
    next_reference = float(next_offsets.get(label, current))
    posterior_update = posterior - current
    applied_update = next_reference - current
    ideal = -current
    sign = ""
    if abs(applied_update) > 0.0 and abs(ideal) > 0.0:
        sign = bool(applied_update * ideal > 0.0)
    return {
        "separation_reference_error_before_microas": current * 1.0e6,
        "separation_posterior_error_after_microas": posterior * 1.0e6,
        "separation_next_reference_error_microas": next_reference * 1.0e6,
        "separation_posterior_update_microas": posterior_update * 1.0e6,
        "separation_applied_update_microas": applied_update * 1.0e6,
        "separation_update_sign_toward_truth": sign,
        "separation_next_reference_improved": bool(abs(next_reference) < abs(current)),
        # Backward-compatible aliases.
        "separation_update_microas": posterior_update * 1.0e6,
        "separation_update_cosine_or_sign": sign,
    }
