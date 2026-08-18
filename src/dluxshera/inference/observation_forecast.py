"""Shared forecast helpers for observation-level summary accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.observation_belief import SubblockSummary
from dluxshera.inference.observation_summary import (
    load_subblock_summary_artifact_payload,
)
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_keys import (
    get_obs_subblock_store_value,
    parse_obs_subblock_key_address,
)

__all__ = [
    "DEFAULT_SYSTEM_PRESET",
    "PriorContext",
    "build_default_prior_sigma",
    "build_prior_mean_from_store",
    "build_prior_store_from_system",
    "build_prior_store_from_system_cfg",
    "require_identical_summary_theta_labels",
    "resolve_prior_context_for_summaries",
    "summarize_summary_theta_ref_compatibility",
]


DEFAULT_SYSTEM_PRESET = "SHERA_FLIGHT_3P"


@dataclass(frozen=True)
class PriorContext:
    """Store resolved prior initialization context for observation updates.

    Use this object when an observation-level update or forecast needs a prior
    mean derived from either an explicit system context, image-backed summary
    linearization points, serialized summary system metadata, or a fallback
    system preset. The object keeps the resolved labels and mean together with
    provenance so downstream artifacts can explain which policy branch was used.

    Parameters
    ----------
    theta_labels :
        Ordered observation-level labels aligned with ``prior_mean``.
    prior_mean :
        Prior mean vector in the physical observation-level basis.
    prior_mean_source :
        Short policy name such as ``"summary_theta_ref"`` or
        ``"explicit_prior_config"``.
    provenance :
        JSON-friendly policy and input metadata.
    warnings :
        User-facing warnings produced during resolution.

    Notes
    -----
    ``PriorContext`` is shared by the observation belief demo and the summary
    simulator. Keep it small so future forecast modes can reuse it without
    depending on example-script implementation details.
    """

    theta_labels: tuple[str, ...]
    prior_mean: np.ndarray
    prior_mean_source: str
    provenance: dict[str, Any]
    warnings: tuple[str, ...] = ()


def require_identical_summary_theta_labels(
    summaries: Sequence[SubblockSummary],
) -> tuple[str, ...]:
    """Return shared ``theta_labels`` after exact ordered-label validation.

    Use this before aggregating or tiling real ``SubblockSummary`` artifacts in
    a single observation-level basis. The current real-summary update and
    forecast paths intentionally require identical ordered labels rather than
    attempting union, subset, or reordering behavior.

    Parameters
    ----------
    summaries :
        One or more loaded summary artifacts.

    Returns
    -------
    tuple of str
        Shared ordered ``theta_labels`` tuple.

    Raises
    ------
    ValueError
        Raised when no summaries are provided or when any input summary has
        different ordered labels.

    Notes
    -----
    Deterministic replicate ordering and summary theta-ref compatibility are
    meaningful only after this invariant has been established.
    """

    if not summaries:
        raise ValueError("At least one SubblockSummary artifact is required.")
    reference = tuple(str(label) for label in summaries[0].theta_labels)
    for index, summary in enumerate(summaries[1:], start=1):
        labels = tuple(str(label) for label in summary.theta_labels)
        if labels != reference:
            raise ValueError(
                "Real-summary observation updates currently require identical "
                "theta_labels across all summaries. "
                f"Summary 0 has {reference}, summary {index} has {labels}."
            )
    return reference


def summarize_summary_theta_ref_compatibility(
    summaries: Sequence[SubblockSummary],
) -> dict[str, Any]:
    """Summarize agreement across summary linearization points.

    Use this during real-summary prior resolution to record whether all input
    summaries share the same ``theta_ref`` values. The summary simulator and
    belief demo preserve each summary's own linearization point in the update,
    but the default prior mean uses the first summary ``theta_ref`` when that
    policy branch is selected.

    Parameters
    ----------
    summaries :
        Loaded summaries with identical ordered ``theta_labels``.

    Returns
    -------
    dict
        JSON-friendly compatibility payload containing the reference policy,
        first summary ``theta_ref``, per-label maximum absolute spread, and any
        warnings.

    Raises
    ------
    ValueError
        Raised when labels differ, no summaries are provided, or any
        ``theta_ref`` entry is non-finite.

    Notes
    -----
    The comparison tolerance is intentionally strict at ``1e-12`` because this
    diagnostic is provenance, not an optimization convergence test.
    """

    labels = require_identical_summary_theta_labels(summaries)
    theta_ref_matrix = np.vstack(
        [np.asarray(summary.theta_ref, dtype=float) for summary in summaries]
    )
    if not np.all(np.isfinite(theta_ref_matrix)):
        raise ValueError("All real summaries must provide finite theta_ref values.")

    first_theta_ref = theta_ref_matrix[0]
    max_abs_spread = np.max(np.abs(theta_ref_matrix - first_theta_ref), axis=0)
    warnings: list[str] = []
    if np.any(max_abs_spread > 1.0e-12):
        warnings.append(
            "Input summaries do not share identical theta_ref values. "
            "Defaulting the prior mean to the first summary theta_ref while "
            "preserving each summary's own linearization point in the update."
        )

    return {
        "theta_labels": list(labels),
        "reference_policy": "first_summary_theta_ref",
        "all_equal_within_tolerance": bool(np.all(max_abs_spread <= 1.0e-12)),
        "max_abs_spread_by_label": {
            label: float(max_abs_spread[index]) for index, label in enumerate(labels)
        },
        "first_summary_theta_ref": {
            label: float(first_theta_ref[index]) for index, label in enumerate(labels)
        },
        "warnings": warnings,
    }


def build_prior_store_from_system(
    *,
    config_path: Path | None = None,
    system_preset: str | None = DEFAULT_SYSTEM_PRESET,
) -> tuple[ParameterStore, Any, dict[str, Any]]:
    """Resolve one system config and return a refreshed forward store.

    Use this when an observation-level prior mean must be derived from an
    explicit config path, an explicit system preset, or the default system
    preset fallback. The returned provenance is intentionally compact and
    suitable for simulator and demo JSON artifacts.

    Parameters
    ----------
    config_path :
        Optional YAML/JSON user config path with a top-level ``system`` block.
    system_preset :
        Optional named system preset passed to the config resolver.

    Returns
    -------
    store, forward_spec, provenance :
        Refreshed parameter store, composed forward spec, and JSON-friendly
        source metadata.

    Raises
    ------
    ValueError
        Raised when config resolution does not produce a mapping-valued
        ``system`` block.
    """

    user_cfg = load_user_config(
        config_path=config_path.resolve() if config_path is not None else None,
        system_preset=system_preset,
        experiment_preset=None,
    )
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    if not isinstance(system_cfg, Mapping):
        raise ValueError(
            "Observation prior context resolution requires a resolved 'system' block."
        )

    forward_spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(
        forward_spec
    )
    provenance = {
        "prior_mean_source": "resolved_system_store",
        "system_preset": None if system_preset is None else str(system_preset),
        "system_config_path": None
        if config_path is None
        else str(config_path.resolve()),
        "resolved_system": {
            "source_kind": system_cfg.get("source", {}).get("kind"),
            "source_target": system_cfg.get("source", {}).get("target"),
            "optics_kind": system_cfg.get("optics", {}).get("kind"),
            "detector_model": system_cfg.get("detector", {}).get("model"),
        },
    }
    return store, forward_spec, provenance


def build_prior_mean_from_store(
    labels: Sequence[str],
    *,
    store: ParameterStore,
) -> np.ndarray:
    """Return an observation-level prior mean vector from a store.

    Use this after resolving or reconstructing a system ``ParameterStore``.
    Labels are parsed with the observation sub-block key helpers so scalar and
    indexed labels, including Zernike coefficient labels, are resolved through
    the same address semantics as the summary exporter.

    Parameters
    ----------
    labels :
        Ordered observation-level labels.
    store :
        Refreshed parameter store.

    Returns
    -------
    numpy.ndarray
        Prior mean vector aligned with ``labels``.

    Raises
    ------
    ValueError
        Raised when a label cannot be resolved from the store.
    """

    mean = np.zeros((len(labels),), dtype=float)
    for index, label in enumerate(labels):
        address = parse_obs_subblock_key_address(label)
        try:
            mean[index] = get_obs_subblock_store_value(store, address=address)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Unable to resolve prior mean for observation label {label!r} "
                "from the resolved system store."
            ) from exc
    return mean


def build_prior_store_from_system_cfg(
    *,
    system_cfg: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> tuple[ParameterStore, Any, dict[str, Any]]:
    """Build a refreshed forward store from a resolved system mapping.

    Use this for real-summary artifacts that serialize a resolved system
    context. This lets a forecast or update reconstruct prior means without
    requiring the original CLI config path.

    Parameters
    ----------
    system_cfg :
        Mapping shaped like a resolved ``system`` config.
    provenance :
        Optional provenance entries merged into the returned payload.

    Returns
    -------
    store, forward_spec, provenance :
        Refreshed parameter store, composed forward spec, and JSON-friendly
        source metadata.
    """

    forward_spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(
        forward_spec
    )
    resolved_provenance = {
        "prior_mean_source": "resolved_summary_system_store",
        "system_preset": system_cfg.get("preset"),
        "system_config_path": None,
        "resolved_system": {
            "source_kind": system_cfg.get("source", {}).get("kind"),
            "source_target": system_cfg.get("source", {}).get("target"),
            "optics_kind": system_cfg.get("optics", {}).get("kind"),
            "detector_model": system_cfg.get("detector", {}).get("model"),
        },
    }
    if provenance:
        resolved_provenance.update(dict(provenance))
    return store, forward_spec, resolved_provenance


def build_default_prior_sigma(labels: Sequence[str]) -> np.ndarray:
    """Return default diagonal prior sigmas for observation-level labels.

    Use this default policy for the current observation belief demo and summary
    simulator until a calibrated observation-level prior model replaces it. The
    values preserve the existing demo policy and are intentionally simple.

    Parameters
    ----------
    labels :
        Ordered observation-level labels.

    Returns
    -------
    numpy.ndarray
        Positive finite sigma vector aligned with ``labels``.

    Notes
    -----
    Units follow each physical label: arcseconds for separation, log flux for
    ``source.log_flux_total``, dimensionless for contrast, arcsec/pixel for
    plate scale, and nanometres for Zernike coefficients.
    """

    sigma = np.zeros((len(labels),), dtype=float)
    for index, label in enumerate(labels):
        if label == "source.separation_as":
            sigma[index] = 0.1
        elif label == "source.log_flux_total":
            sigma[index] = 0.10
        elif label == "source.contrast":
            sigma[index] = 0.10
        elif label == "optics.plate_scale_as_per_pix":
            sigma[index] = 0.001
        elif "zernike_coeffs_nm" in label:
            sigma[index] = 3.0
        else:
            sigma[index] = 1.0
    return sigma


def resolve_prior_context_for_summaries(
    summaries: Sequence[SubblockSummary],
    *,
    summary_paths: Sequence[Path],
    explicit_config_path: Path | None = None,
    explicit_system_preset: str | None = None,
    prior_source: str = "auto",
    allow_summary_theta_ref_default: bool = True,
) -> PriorContext:
    """Resolve the prior mean context for one real-summary batch.

    Use this shared policy whenever image-backed ``SubblockSummary`` artifacts
    are consumed by an observation-level update or forecast. It centralizes the
    provenance-sensitive prior mean resolution previously owned by the example
    demo script.

    Resolution order:

    1. explicit ``config`` or ``system_preset`` override;
    2. summary ``theta_ref`` values when requested or when ``prior_source`` is
       ``"auto"`` and ``allow_summary_theta_ref_default`` is true;
    3. resolved system context serialized in the first summary artifact when
       requested or when available under ``"auto"``;
    4. default system preset fallback with a clear provenance warning.

    Parameters
    ----------
    summaries :
        Loaded summaries with identical ordered ``theta_labels``.
    summary_paths :
        Paths to the corresponding ``subblock_summary.json`` artifacts.
    explicit_config_path :
        Optional config override. Takes precedence over summary ``theta_ref``.
    explicit_system_preset :
        Optional system preset override. Takes precedence over summary
        ``theta_ref``.
    prior_source :
        One of ``"auto"``, ``"summary_theta_ref"``, ``"resolved_system"``, or
        ``"default_system"``.
    allow_summary_theta_ref_default :
        Whether ``"auto"`` is allowed to use the first summary ``theta_ref``.

    Returns
    -------
    PriorContext
        Prior mean, labels, source name, provenance, and warnings.

    Raises
    ------
    ValueError
        Raised when no summaries are provided, labels differ, theta refs are
        non-finite, or ``prior_source`` is unsupported.

    Notes
    -----
    The update itself still preserves each summary's own ``theta_ref``. This
    helper only selects the prior mean initialization context.
    """

    if prior_source not in {
        "auto",
        "summary_theta_ref",
        "resolved_system",
        "default_system",
    }:
        raise ValueError(
            "prior_source must be one of auto, summary_theta_ref, "
            "resolved_system, or default_system."
        )
    if not summaries:
        raise ValueError(
            "resolve_prior_context_for_summaries requires at least one summary."
        )

    labels = require_identical_summary_theta_labels(summaries)
    compatibility = summarize_summary_theta_ref_compatibility(summaries)
    payloads = [load_subblock_summary_artifact_payload(path) for path in summary_paths]
    warnings = list(compatibility["warnings"])
    explicit_override_requested = (
        explicit_config_path is not None or explicit_system_preset is not None
    )

    if explicit_override_requested:
        prior_store, _, provenance = build_prior_store_from_system(
            config_path=explicit_config_path,
            system_preset=explicit_system_preset,
        )
        provenance.update(
            {
                "summary_paths": [str(path) for path in summary_paths],
                "summary_theta_ref_compatibility": compatibility,
            }
        )
        source_name = (
            "explicit_prior_config"
            if explicit_config_path is not None
            else "explicit_prior_system"
        )
        provenance["prior_mean_source"] = source_name
        return PriorContext(
            theta_labels=labels,
            prior_mean=build_prior_mean_from_store(labels, store=prior_store),
            prior_mean_source=source_name,
            provenance=provenance,
            warnings=tuple(warnings),
        )

    if prior_source == "summary_theta_ref" or (
        prior_source == "auto" and allow_summary_theta_ref_default
    ):
        prior_mean = np.asarray(summaries[0].theta_ref, dtype=float)
        if not np.all(np.isfinite(prior_mean)):
            raise ValueError(
                "Real-summary default prior initialization requires finite "
                "theta_ref values."
            )
        provenance = {
            "prior_mean_source": "summary_theta_ref",
            "summary_paths": [str(path) for path in summary_paths],
            "summary_theta_ref_compatibility": compatibility,
            "recommended_prior_context": (
                payloads[0].get("prior_context") if payloads else None
            ),
        }
        return PriorContext(
            theta_labels=labels,
            prior_mean=prior_mean,
            prior_mean_source="summary_theta_ref",
            provenance=provenance,
            warnings=tuple(warnings),
        )

    if prior_source in {"resolved_system", "auto"} and payloads:
        first_system = payloads[0].get("system")
        resolved_system_cfg = (
            first_system.get("resolved_config")
            if isinstance(first_system, Mapping)
            else None
        )
        if isinstance(resolved_system_cfg, Mapping):
            prior_store, _, provenance = build_prior_store_from_system_cfg(
                system_cfg=resolved_system_cfg,
                provenance={
                    "prior_mean_source": "resolved_summary_system_store",
                    "summary_paths": [str(path) for path in summary_paths],
                    "summary_theta_ref_compatibility": compatibility,
                },
            )
            return PriorContext(
                theta_labels=labels,
                prior_mean=build_prior_mean_from_store(labels, store=prior_store),
                prior_mean_source="resolved_summary_system_store",
                provenance=provenance,
                warnings=tuple(warnings),
            )

    if prior_source == "default_system":
        prior_store, _, provenance = build_prior_store_from_system(
            config_path=None,
            system_preset=DEFAULT_SYSTEM_PRESET,
        )
        provenance.update(
            {
                "summary_paths": [str(path) for path in summary_paths],
                "summary_theta_ref_compatibility": compatibility,
            }
        )
        source_name = "default_system_preset_fallback"
        provenance["prior_mean_source"] = source_name
        return PriorContext(
            theta_labels=labels,
            prior_mean=build_prior_mean_from_store(labels, store=prior_store),
            prior_mean_source=source_name,
            provenance=provenance,
            warnings=tuple(warnings),
        )

    prior_store, _, provenance = build_prior_store_from_system(
        config_path=None,
        system_preset=DEFAULT_SYSTEM_PRESET,
    )
    warnings.append(
        "Real-summary prior initialization fell back to the default system preset. "
        "This may be stale for image-backed summaries that were generated from "
        "a different effective render or inference context."
    )
    provenance.update(
        {
            "prior_mean_source": "default_system_preset_fallback",
            "summary_paths": [str(path) for path in summary_paths],
            "summary_theta_ref_compatibility": compatibility,
        }
    )
    return PriorContext(
        theta_labels=labels,
        prior_mean=build_prior_mean_from_store(labels, store=prior_store),
        prior_mean_source="default_system_preset_fallback",
        provenance=provenance,
        warnings=tuple(warnings),
    )
