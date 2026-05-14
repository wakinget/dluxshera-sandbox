"""Run a small observation-level Zernike bias campaign.

This script orchestrates image-backed Schur-summary sub-block exports with the
existing ``run_obs_subblock_study.py --mode schur_summary`` entrypoint, then
accumulates the summaries into one observation-level belief update per bias
case. The native state stays in the physical parameter basis; eigenmodes are
diagnostic transforms of the accumulated or posterior precision matrix.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "dluxshera-matplotlib"),
)

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _HAVE_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    _HAVE_MATPLOTLIB = False
import numpy as np

from dluxshera.config.io import load_config_file
from dluxshera.config.resolver import resolve_config
from dluxshera.inference.observation_belief import (
    MatrixDiagnostics,
    ObservationBeliefState,
    ObservationThetaLayout,
    SubblockSummary,
    accumulate_summary_information,
    build_observation_eigenbasis,
    build_prior_whitened_information_gain_matrix,
    build_system_observation_theta_layout,
    update_observation_belief,
)
from dluxshera.inference.observation_forecast import (
    DEFAULT_SYSTEM_PRESET,
    build_default_prior_sigma,
    build_prior_mean_from_store,
    resolve_prior_context_for_summaries,
)
from dluxshera.inference.observation_summary import (
    load_subblock_summary,
    load_subblock_summary_artifact_payload,
)
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_io import now_iso_local_ms, timestamp_tag
from dluxshera.utils.obs_subblock_keys import (
    parse_obs_subblock_key_address,
)
from dluxshera.utils.noise import make_subseed


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "observation_bias_campaign"
SUBBLOCK_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_study.py"
CAMPAIGN_SCHEMA_VERSION = "observation_bias_campaign.v1"
DEFAULT_LOCAL_ELIMINATED_KEYS = (
    "source.x_position_as",
    "source.y_position_as",
    "source.position_angle_deg",
)
SUPPORTED_SEED_POLICIES = (
    "different_jitter_different_noise",
    "same_jitter_different_noise",
    "different_jitter_same_noise",
)
SUPPORTED_EIGEN_SOURCES = (
    "accumulated_information",
    "posterior_precision",
)
SUPPORTED_FORECAST_MODES = (
    "replicate",
    "fixed_information_score_noise",
)
SUPPORTED_REFERENCE_DIAGNOSTICS_PROFILES = ("none", "basic", "review", "full")
SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES = ("warn", "mask", "reject")
SUPPORTED_SCHUR_FRAME_QUALITY_MISSING = ("allow_all", "error")
SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS = ("original", "kept")
SUBBLOCK_OPTION_FLAG_MAP = {
    "exposure_time_s": "--exposure-time-s",
    "reference_diagnostics_profile": "--reference-diagnostics-profile",
    "schur_frame_quality_policy": "--schur-frame-quality-policy",
    "schur_frame_chi2_threshold": "--schur-frame-chi2-threshold",
    "schur_frame_quality_missing": "--schur-frame-quality-missing",
    "schur_frame_mask_denominator": "--schur-frame-mask-denominator",
    "schur_frame_mask_min_good_frames": "--schur-frame-mask-min-good-frames",
    "reference_optimizer_kind": "--reference-optimizer-kind",
    "reference_base_lr": "--reference-base-lr",
    "reference_n_iter": "--reference-n-iter",
    "reference_schedule_kind": "--reference-schedule-kind",
    "reference_schedule_warmup_steps": "--reference-schedule-warmup-steps",
    "reference_schedule_start_factor": "--reference-schedule-start-factor",
    "reference_preconditioning_method": "--reference-preconditioning-method",
    "reference_preconditioning_reference": "--reference-preconditioning-reference",
    "reference_preconditioning_damping": "--reference-preconditioning-damping",
    "reference_preconditioning_eig_floor_rel": "--reference-preconditioning-eig-floor-rel",
    "reference_preconditioning_eig_floor_abs": "--reference-preconditioning-eig-floor-abs",
    "reference_preconditioning_lr_clip": "--reference-preconditioning-lr-clip",
    "variance_floor": "--variance-floor",
}


@dataclass(frozen=True)
class BiasCase:
    case_name: str
    theta_reference_offsets: dict[str, float]
    case_origin: str = "explicit"
    prior_sigma_by_label: dict[str, float] | None = None
    prior_draw_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CampaignPlan:
    run_root: Path
    layout: ObservationThetaLayout
    layout_metadata: dict[str, Any]
    prior_truth: np.ndarray
    cases: tuple[BiasCase, ...]
    subblock_commands: dict[str, list[list[str]]]
    summary_paths: dict[str, list[Path]]
    subblock_plans: dict[str, list[dict[str, Any]]]
    prior_draw_rows_by_case: dict[str, list[dict[str, Any]]]
    config: dict[str, Any]
    partition: dict[str, Any]
    case_generation: dict[str, Any]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
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
        writer.writerows(rows)


def _matrix_diagnostics(matrix: np.ndarray) -> MatrixDiagnostics:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    if matrix.size == 0:
        return MatrixDiagnostics(0, 0.0, 0.0, 1.0, 0.0, 0.0)
    eigenvalues = np.linalg.eigvalsh(matrix)
    tol = np.finfo(float).eps * max(matrix.shape) * max(float(np.max(np.abs(eigenvalues))), 1.0)
    positive = eigenvalues[eigenvalues > tol]
    condition = float("inf") if positive.size == 0 else float(np.max(positive) / np.min(positive))
    return MatrixDiagnostics(
        rank_estimate=int(np.count_nonzero(np.abs(eigenvalues) > tol)),
        min_eigenvalue=float(np.min(eigenvalues)),
        max_eigenvalue=float(np.max(eigenvalues)),
        condition_number=condition,
        trace=float(np.trace(matrix)),
        frobenius_norm=float(np.linalg.norm(matrix)),
    )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "unnamed"


def _label_group(label: str) -> str:
    if label.startswith("source."):
        return "source"
    if label == "optics.plate_scale_as_per_pix":
        return "optics.plate_scale"
    if label.startswith("optics.primary.zernike_coeffs_nm"):
        return "optics.primary_zernikes"
    if label.startswith("optics.secondary.zernike_coeffs_nm"):
        return "optics.secondary_zernikes"
    return "other"


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


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or abs(denominator) <= 1.0e-30:
        return float("nan")
    return float(numerator / denominator)


def _load_campaign_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"experiment": _default_experiment_config()}
    payload = load_config_file(path.resolve())
    experiment = payload.get("experiment", payload)
    if not isinstance(experiment, Mapping):
        raise ValueError("Campaign config must contain a mapping-valued experiment block.")
    if "experiment" not in payload:
        payload = {"experiment": dict(experiment)}
    return dict(payload)


def _default_experiment_config() -> dict[str, Any]:
    return {
        "kind": "observation_bias_campaign",
        "seed": 42,
        "run_name": "full_zernike_bias_smoke",
        "subblocks": {
            "n_subblocks": 3,
            "n_frames": 3,
            "noise": "disabled",
            "phi_ref": "truth_when_available",
            "schur_curvature_method": "auto",
            "max_dense_dim": 40,
            "schur_damping": 1.0e-8,
        },
        "seeding": {
            "seed_policy": "different_jitter_different_noise",
            "base_seed": 42,
        },
        "observation_theta": {
            "source": {
                "separation_as": True,
                "log_flux_total": True,
                "contrast": True,
            },
            "optics": {
                "plate_scale_as_per_pix": True,
                "primary_zernikes": {
                    "enabled": True,
                    "indices": "from_system",
                    "include": None,
                    "exclude": [],
                },
                "secondary_zernikes": {
                    "enabled": True,
                    "indices": "from_system",
                    "include": None,
                    "exclude": [],
                },
            },
        },
        "eigenbasis": {
            "enabled": True,
            "sources": ["accumulated_information", "posterior_precision"],
            "whiten": True,
            "eig_floor_abs": 0.0,
            "eig_floor_rel": 1.0e-12,
            "top_k_contributors": 8,
        },
        "bias_cases": [
            {"case_name": "zero_bias_full_zernike", "theta_reference_offsets": {}},
        ],
        "prior_draws": {
            "enabled": False,
            "n_cases": 3,
            "center": "truth",
            "distribution": "normal",
            "draw_seed": 12345,
            "case_name_template": "prior_draw_{draw_index:03d}",
            "sigmas": {
                "source.separation_as": {"kind": "absolute", "sigma": 1.0e-5, "unit": "arcsec"},
                "source.log_flux_total": {"kind": "absolute", "sigma": 1.0e-5, "unit": "log_flux"},
                "source.contrast": {"kind": "fractional", "sigma": 1.0e-5},
                "optics.plate_scale_as_per_pix": {"kind": "fractional", "sigma": 1.0e-5},
                "optics.primary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 1.0e-1, "unit": "nm"},
                "optics.secondary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 1.0e-1, "unit": "nm"},
            },
        },
        "forecast": {
            "enabled": True,
            "modes": ["replicate", "fixed_information_score_noise"],
            "n_subblocks_grid": [1, 3, 5, 10, 30, 100, 300, 1000, 1800],
            "subblock_duration_s": 1.0,
            "single_observation_n_subblocks": 1800,
            "replicate": {"enabled": True},
            "fixed_information_score_noise": {
                "enabled": True,
                "n_trials": 100,
                "seed": 2026,
                "score_noise_alpha": 1.0,
                "score_noise_eig_floor_abs": 0.0,
                "score_noise_eig_floor_rel": 1.0e-12,
                "truth_mode": "campaign_truth",
            },
            "plots": True,
        },
    }


def _experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    experiment = config.get("experiment", config)
    if not isinstance(experiment, Mapping):
        raise ValueError("Campaign experiment config must be a mapping.")
    return dict(experiment)


def _resolve_system_store(
    *,
    config: Mapping[str, Any],
    system_preset: str | None,
) -> tuple[ParameterStore, dict[str, Any], dict[str, Any]]:
    user_cfg = dict(config)
    preset = DEFAULT_SYSTEM_PRESET if system_preset is None else system_preset
    if "system" not in user_cfg:
        user_cfg = {**user_cfg, "system": {"preset": preset}}
    elif system_preset is not None:
        system_cfg = dict(user_cfg["system"])
        system_cfg["preset"] = system_preset
        user_cfg["system"] = system_cfg
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    if not isinstance(system_cfg, Mapping):
        raise ValueError("Campaign system resolution requires a resolved system block.")
    forward_spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(forward_spec)
    provenance = {
        "system_preset": system_cfg.get("preset", preset),
        "source_kind": system_cfg.get("source", {}).get("kind"),
        "source_target": system_cfg.get("source", {}).get("target"),
        "optics_kind": system_cfg.get("optics", {}).get("kind"),
        "detector_model": system_cfg.get("detector", {}).get("model"),
    }
    return store, dict(system_cfg), provenance


def _validate_partition_config(partition_cfg: Mapping[str, Any] | None, layout: ObservationThetaLayout) -> dict[str, Any]:
    cfg = dict(partition_cfg or {})
    local_eliminated = tuple(cfg.get("local_eliminated_keys", DEFAULT_LOCAL_ELIMINATED_KEYS))
    if local_eliminated != DEFAULT_LOCAL_ELIMINATED_KEYS:
        raise ValueError(
            "For this campaign, state_partition.local_eliminated_keys must match "
            + ", ".join(DEFAULT_LOCAL_ELIMINATED_KEYS)
            + "."
        )
    shared_active = tuple(cfg.get("subblock_shared_active_keys", ()) or ())
    if shared_active:
        raise NotImplementedError(
            "state_partition.subblock_shared_active_keys is accepted for future "
            "work but is not implemented by this registration-only campaign."
        )
    return {
        "observation_theta_keys": list(layout.labels),
        "local_eliminated_keys": list(local_eliminated),
        "subblock_shared_active_keys": list(shared_active),
        "report_only_keys": list(cfg.get("report_only_keys", ()) or ()),
    }


def _parse_bias_cases(
    experiment_cfg: Mapping[str, Any],
    *,
    layout: ObservationThetaLayout,
    layout_metadata: Mapping[str, Any],
) -> tuple[BiasCase, ...]:
    label_set = set(layout.labels)
    raw_cases = experiment_cfg.get("bias_cases", [])
    cases: list[BiasCase] = []
    if raw_cases is not None:
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise ValueError("experiment.bias_cases must be a list.")
        for raw_case in raw_cases:
            if not isinstance(raw_case, Mapping):
                raise ValueError("Each bias case must be a mapping.")
            name = str(raw_case.get("case_name", "")).strip()
            if not name:
                raise ValueError("Each bias case requires a non-empty case_name.")
            offsets_raw = raw_case.get("theta_reference_offsets", {}) or {}
            if not isinstance(offsets_raw, Mapping):
                raise ValueError(f"Bias case {name!r} offsets must be a mapping.")
            offsets: dict[str, float] = {}
            for raw_key, raw_value in offsets_raw.items():
                address = parse_obs_subblock_key_address(str(raw_key))
                key = address.canonical
                if key not in label_set:
                    raise ValueError(
                        f"Bias case {name!r} references {key!r}, which is not in "
                        "the resolved observation theta layout."
                    )
                offsets[key] = float(raw_value)
            prior_sigmas_raw = raw_case.get("prior_sigmas")
            prior_sigma_by_label: dict[str, float] | None = None
            if prior_sigmas_raw is not None:
                if not isinstance(prior_sigmas_raw, Mapping):
                    raise ValueError(f"Bias case {name!r} prior_sigmas must be a mapping.")
                prior_sigma_by_label = {}
                for label, value in prior_sigmas_raw.items():
                    canonical = parse_obs_subblock_key_address(str(label)).canonical
                    if canonical not in label_set:
                        raise ValueError(
                            f"Bias case {name!r} prior_sigmas references {canonical!r}, "
                            "which is not in the resolved observation theta layout."
                        )
                    sigma = float(value)
                    if sigma <= 0.0 or not math.isfinite(sigma):
                        raise ValueError(f"Bias case {name!r} prior sigma must be positive.")
                    prior_sigma_by_label[canonical] = sigma
            cases.append(
                BiasCase(
                    case_name=name,
                    theta_reference_offsets=offsets,
                    case_origin="explicit",
                    prior_sigma_by_label=prior_sigma_by_label,
                )
            )

    auto_cfg = experiment_cfg.get("auto_cases", {}) or {}
    if isinstance(auto_cfg, Mapping) and bool(auto_cfg.get("enabled", False)):
        cases.extend(
            _generate_auto_cases(
                layout=layout,
                layout_metadata=layout_metadata,
                amplitude_nm=float(auto_cfg.get("zernike_pair_amplitude_nm", 5.0)),
                include_zero_bias=bool(auto_cfg.get("include_zero_bias", True)),
                include_matched=bool(auto_cfg.get("include_matched_pairs", True)),
                include_differential=bool(auto_cfg.get("include_differential_pairs", True)),
                max_cases=int(auto_cfg.get("max_cases", 6)),
            )
        )

    seen: set[str] = set()
    unique_cases: list[BiasCase] = []
    for case in cases:
        if case.case_name in seen:
            raise ValueError(f"Duplicate bias case name: {case.case_name}.")
        seen.add(case.case_name)
        unique_cases.append(case)
    return tuple(unique_cases)


def _resolve_case_generation(
    experiment_cfg: Mapping[str, Any],
    *,
    parsed_cases: Sequence[BiasCase],
    prior_draw_cases: Sequence[BiasCase],
) -> tuple[tuple[BiasCase, ...], dict[str, Any]]:
    raw_cfg = experiment_cfg.get("case_generation", {}) or {}
    if not isinstance(raw_cfg, Mapping):
        raise ValueError("experiment.case_generation must be a mapping when provided.")
    include_key_present = "include_implicit_zero_bias" in raw_cfg
    include_implicit = bool(raw_cfg.get("include_implicit_zero_bias", True))
    explicit_present = any(case.case_origin == "explicit" for case in parsed_cases)
    auto_present = any(case.case_origin.startswith("auto") for case in parsed_cases)
    has_configured_cases = bool(parsed_cases)
    has_prior_draw_cases = bool(prior_draw_cases)

    add_implicit = False
    zero_bias_status = "not_added"
    if explicit_present or auto_present:
        if include_key_present and include_implicit:
            add_implicit = True
            zero_bias_status = "implicit_requested_with_configured_cases"
        else:
            zero_bias_status = "not_added_configured_cases_present"
    elif has_prior_draw_cases:
        if include_implicit:
            add_implicit = True
            zero_bias_status = "implicit_default_with_prior_draws"
        else:
            zero_bias_status = "disabled"
    else:
        if include_implicit:
            add_implicit = True
            zero_bias_status = "implicit_default_no_cases"
        else:
            raise ValueError(
                "No bias cases or prior draws were configured, and "
                "case_generation.include_implicit_zero_bias is false."
            )

    cases: list[BiasCase] = []
    if add_implicit:
        cases.append(
            BiasCase("zero_bias_full_zernike", {}, case_origin="implicit_zero_bias")
        )
    cases.extend(parsed_cases)
    cases.extend(prior_draw_cases)
    names = [case.case_name for case in cases]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate case names across configured, implicit, and prior-draw cases.")
    metadata = {
        "include_implicit_zero_bias": include_implicit,
        "include_implicit_zero_bias_configured": include_key_present,
        "zero_bias_case_status": zero_bias_status,
        "n_configured_cases": len(parsed_cases),
        "n_prior_draw_cases": len(prior_draw_cases),
        "n_total_cases": len(cases),
    }
    return tuple(cases), metadata


def _generate_auto_cases(
    *,
    layout: ObservationThetaLayout,
    layout_metadata: Mapping[str, Any],
    amplitude_nm: float,
    include_zero_bias: bool,
    include_matched: bool,
    include_differential: bool,
    max_cases: int,
) -> list[BiasCase]:
    cases: list[BiasCase] = []
    if include_zero_bias:
        cases.append(BiasCase("zero_bias_full_zernike", {}))
    primary = set(int(index) for index in layout_metadata.get("primary_zernike_indices", []))
    secondary = set(int(index) for index in layout_metadata.get("secondary_zernike_indices", []))
    for index in sorted(primary & secondary):
        m1 = f"optics.primary.zernike_coeffs_nm[{index}]"
        m2 = f"optics.secondary.zernike_coeffs_nm[{index}]"
        if m1 not in layout.labels or m2 not in layout.labels:
            continue
        if include_matched:
            cases.append(
                BiasCase(
                    f"matched_zernike_pair_{index}_plus",
                    {m1: float(amplitude_nm), m2: float(amplitude_nm)},
                    case_origin="auto_matched_pair",
                )
            )
        if include_differential:
            cases.append(
                BiasCase(
                    f"differential_zernike_pair_{index}_plus",
                    {m1: float(amplitude_nm), m2: -float(amplitude_nm)},
                    case_origin="auto_differential_pair",
                )
            )
        if len(cases) >= max_cases:
            break
    return cases[:max_cases]


def _apply_cli_overrides(experiment_cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(json.dumps(experiment_cfg))
    subblocks = dict(cfg.get("subblocks", {}) or {})
    seeding = dict(cfg.get("seeding", {}) or {})
    for arg_name, key in (
        ("n_subblocks", "n_subblocks"),
        ("n_frames", "n_frames"),
        ("noise", "noise"),
        ("phi_ref", "phi_ref"),
        ("max_dense_dim", "max_dense_dim"),
        ("schur_curvature_method", "schur_curvature_method"),
    ):
        value = getattr(args, arg_name, None)
        if value is not None:
            subblocks[key] = value
    cfg["subblocks"] = subblocks
    if getattr(args, "seed_policy", None) is not None:
        seeding["seed_policy"] = args.seed_policy
    if getattr(args, "base_seed", None) is not None:
        seeding["base_seed"] = int(args.base_seed)
    if seeding:
        cfg["seeding"] = seeding
    if args.run_name is not None:
        cfg["run_name"] = args.run_name
    return cfg


def _resolve_seeding_config(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    seeding_cfg = dict(experiment_cfg.get("seeding", {}) or {})
    policy = str(
        seeding_cfg.get(
            "seed_policy",
            experiment_cfg.get("subblocks", {}).get(
                "seed_policy", "different_jitter_different_noise"
            ),
        )
    )
    if policy not in SUPPORTED_SEED_POLICIES:
        raise ValueError(f"Unsupported seed_policy: {policy}")
    base_seed = int(
        seeding_cfg.get(
            "base_seed",
            experiment_cfg.get("subblocks", {}).get(
                "base_seed",
                experiment_cfg.get("seed", 42),
            ),
        )
    )
    return {"seed_policy": policy, "base_seed": base_seed}


def _derive_subblock_seeds(
    *,
    run_name: str,
    case_name: str,
    subblock_index: int,
    seed_policy: str,
    base_seed: int,
) -> dict[str, int]:
    if seed_policy not in SUPPORTED_SEED_POLICIES:
        raise ValueError(f"Unsupported seed_policy: {seed_policy}")
    token = f"{run_name}.{case_name}.subblock_{subblock_index:03d}"
    subblock_seed = make_subseed(int(base_seed), token)
    if seed_policy == "same_jitter_different_noise":
        trace_seed = make_subseed(int(base_seed), f"{run_name}.{case_name}.shared_trace")
        noise_seed = make_subseed(int(base_seed), f"{token}.noise")
    elif seed_policy == "different_jitter_same_noise":
        trace_seed = make_subseed(int(base_seed), f"{token}.trace")
        noise_seed = make_subseed(int(base_seed), f"{run_name}.{case_name}.shared_noise")
    else:
        trace_seed = make_subseed(int(base_seed), f"{token}.trace")
        noise_seed = make_subseed(int(base_seed), f"{token}.noise")
    return {
        "seed_policy": seed_policy,
        "base_seed": int(base_seed),
        "subblock_seed": int(subblock_seed),
        "trace_seed": int(trace_seed),
        "noise_seed": int(noise_seed),
    }


def _resolve_eigen_sources(eigen_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    if "sources" in eigen_cfg:
        raw_sources = eigen_cfg.get("sources")
        if not isinstance(raw_sources, Sequence) or isinstance(raw_sources, (str, bytes)):
            raise ValueError("eigenbasis.sources must be a list of source names.")
        sources = tuple(str(item) for item in raw_sources)
    elif "source_matrix" in eigen_cfg:
        sources = (str(eigen_cfg.get("source_matrix")),)
    else:
        sources = ("accumulated_information",)
    if not sources:
        raise ValueError("At least one eigenbasis source is required.")
    invalid = [name for name in sources if name not in SUPPORTED_EIGEN_SOURCES]
    if invalid:
        raise ValueError(
            "Unsupported eigenbasis source(s): " + ", ".join(sorted(set(invalid)))
        )
    if len(set(sources)) != len(sources):
        raise ValueError("eigenbasis sources must not contain duplicates.")
    return sources


def _match_wildcard_pattern(pattern: str, label: str) -> bool:
    if "[*]" not in pattern:
        return pattern == label
    escaped = re.escape(pattern).replace(r"\[\*\]", r"\[\d+\]")
    return re.fullmatch(escaped, label) is not None


def _resolve_prior_draw_sigmas(
    *,
    labels: Sequence[str],
    truth_by_label: Mapping[str, float],
    sigmas_cfg: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    if not isinstance(sigmas_cfg, Mapping):
        raise ValueError("prior_draws.sigmas must be a mapping.")
    resolved: dict[str, float] = {}
    meta: dict[str, dict[str, Any]] = {}
    for raw_rule_label, raw_rule in sigmas_cfg.items():
        rule_label = str(raw_rule_label).strip()
        if not rule_label:
            raise ValueError("prior_draw sigma rule labels must be non-empty.")
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"Sigma rule {rule_label!r} must be a mapping.")
        kind = str(raw_rule.get("kind", "absolute"))
        if kind not in {"absolute", "fractional"}:
            raise ValueError(f"Unsupported sigma kind {kind!r} for rule {rule_label!r}.")
        sigma_value = float(raw_rule.get("sigma", 0.0))
        if sigma_value <= 0.0 or not math.isfinite(sigma_value):
            raise ValueError(f"Sigma for rule {rule_label!r} must be positive and finite.")
        unit = None if raw_rule.get("unit") is None else str(raw_rule.get("unit"))
        matching_labels = [label for label in labels if _match_wildcard_pattern(rule_label, label)]
        if not matching_labels:
            raise ValueError(f"Sigma rule {rule_label!r} matched no theta labels.")
        for label in matching_labels:
            if kind == "absolute":
                sigma = sigma_value
            else:
                truth = float(truth_by_label[label])
                sigma = abs(truth) * sigma_value
            if sigma <= 0.0 or not math.isfinite(sigma):
                raise ValueError(
                    f"Resolved sigma for label {label!r} from rule {rule_label!r} "
                    "must be positive and finite."
                )
            resolved[label] = float(sigma)
            meta[label] = {
                "sigma_kind": kind,
                "sigma_source_rule": rule_label,
                "unit": _parameter_unit(label) if unit is None else unit,
                "sigma_config_value": sigma_value,
            }
    missing = [label for label in labels if label not in resolved]
    if missing:
        raise ValueError(
            "prior_draws.sigmas did not resolve all observation labels; missing: "
            + ", ".join(missing)
        )
    return resolved, meta


def _generate_prior_draw_cases(
    *,
    experiment_cfg: Mapping[str, Any],
    labels: Sequence[str],
    truth_by_label: Mapping[str, float],
) -> tuple[list[BiasCase], dict[str, list[dict[str, Any]]]]:
    raw_cfg = experiment_cfg.get("prior_draws", {}) or {}
    if not isinstance(raw_cfg, Mapping) or not bool(raw_cfg.get("enabled", False)):
        return [], {}
    center = str(raw_cfg.get("center", "truth"))
    if center != "truth":
        raise ValueError("prior_draws.center currently supports only 'truth'.")
    distribution = str(raw_cfg.get("distribution", "normal"))
    if distribution != "normal":
        raise ValueError("prior_draws.distribution currently supports only 'normal'.")
    n_cases = int(raw_cfg.get("n_cases", 0))
    if n_cases <= 0:
        raise ValueError("prior_draws.n_cases must be positive when enabled.")
    draw_seed = int(raw_cfg.get("draw_seed", 12345))
    case_name_template = str(raw_cfg.get("case_name_template", "prior_draw_{draw_index:03d}"))
    sigmas_cfg = raw_cfg.get("sigmas", {})
    sigma_by_label, sigma_meta = _resolve_prior_draw_sigmas(
        labels=labels,
        truth_by_label=truth_by_label,
        sigmas_cfg=sigmas_cfg,
    )
    label_order = tuple(labels)
    sigma_vector = np.asarray([sigma_by_label[label] for label in label_order], dtype=float)
    rng = np.random.default_rng(draw_seed)
    cases: list[BiasCase] = []
    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    for draw_index in range(n_cases):
        z_vector = rng.normal(loc=0.0, scale=1.0, size=len(label_order))
        delta = z_vector * sigma_vector
        case_name = case_name_template.format(draw_index=draw_index)
        offsets = {label: float(delta[i]) for i, label in enumerate(label_order)}
        case = BiasCase(
            case_name=case_name,
            theta_reference_offsets=offsets,
            case_origin="prior_draw",
            prior_sigma_by_label={label: float(sigma_by_label[label]) for label in label_order},
            prior_draw_metadata={
                "draw_seed": int(draw_seed),
                "draw_index": int(draw_index),
                "distribution": distribution,
                "center": center,
            },
        )
        cases.append(case)
        rows: list[dict[str, Any]] = []
        for i, label in enumerate(label_order):
            truth_value = float(truth_by_label[label])
            offset = float(delta[i])
            sigma = float(sigma_vector[i])
            z_value = float(z_vector[i])
            rows.append(
                {
                    "case_name": case_name,
                    "theta_label": label,
                    "truth_value": truth_value,
                    "prior_mean": truth_value + offset,
                    "reference_value": truth_value + offset,
                    "prior_sigma": sigma,
                    "draw_z": z_value,
                    "theta_reference_offset": offset,
                    "sigma_kind": sigma_meta[label]["sigma_kind"],
                    "sigma_source_rule": sigma_meta[label]["sigma_source_rule"],
                    "unit": sigma_meta[label]["unit"],
                    "draw_seed": int(draw_seed),
                    "draw_index": int(draw_index),
                }
            )
        rows_by_case[case_name] = rows
    return cases, rows_by_case


def _validate_choice(value: Any, *, key: str, choices: Sequence[str]) -> str:
    text = str(value)
    if text not in choices:
        raise ValueError(
            f"experiment.subblocks.{key} must be one of "
            + ", ".join(choices)
            + "."
        )
    return text


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def resolve_subblock_command_options(
    subblock_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize optional flags forwarded to the subblock runner."""

    options: dict[str, Any] = {}
    if subblock_cfg.get("exposure_time_s") is not None:
        exposure_time_s = float(subblock_cfg["exposure_time_s"])
        if exposure_time_s <= 0.0 or not math.isfinite(exposure_time_s):
            raise ValueError("experiment.subblocks.exposure_time_s must be positive.")
        options["exposure_time_s"] = exposure_time_s
    if subblock_cfg.get("reference_diagnostics_profile") is not None:
        options["reference_diagnostics_profile"] = _validate_choice(
            subblock_cfg["reference_diagnostics_profile"],
            key="reference_diagnostics_profile",
            choices=SUPPORTED_REFERENCE_DIAGNOSTICS_PROFILES,
        )
    if subblock_cfg.get("schur_frame_quality_policy") is not None:
        options["schur_frame_quality_policy"] = _validate_choice(
            subblock_cfg["schur_frame_quality_policy"],
            key="schur_frame_quality_policy",
            choices=SUPPORTED_SCHUR_FRAME_QUALITY_POLICIES,
        )
    if subblock_cfg.get("schur_frame_chi2_threshold") is not None:
        threshold = float(subblock_cfg["schur_frame_chi2_threshold"])
        if threshold <= 0.0 or not math.isfinite(threshold):
            raise ValueError("experiment.subblocks.schur_frame_chi2_threshold must be positive.")
        options["schur_frame_chi2_threshold"] = threshold
    if subblock_cfg.get("schur_frame_quality_missing") is not None:
        options["schur_frame_quality_missing"] = _validate_choice(
            subblock_cfg["schur_frame_quality_missing"],
            key="schur_frame_quality_missing",
            choices=SUPPORTED_SCHUR_FRAME_QUALITY_MISSING,
        )
    if subblock_cfg.get("schur_frame_mask_denominator") is not None:
        options["schur_frame_mask_denominator"] = _validate_choice(
            subblock_cfg["schur_frame_mask_denominator"],
            key="schur_frame_mask_denominator",
            choices=SUPPORTED_SCHUR_FRAME_MASK_DENOMINATORS,
        )
    if subblock_cfg.get("schur_frame_mask_min_good_frames") is not None:
        min_good = int(subblock_cfg["schur_frame_mask_min_good_frames"])
        if min_good < 1:
            raise ValueError(
                "experiment.subblocks.schur_frame_mask_min_good_frames must be >= 1."
            )
        options["schur_frame_mask_min_good_frames"] = min_good

    optional_keys = (
        "reference_optimizer_kind",
        "reference_base_lr",
        "reference_n_iter",
        "reference_schedule_kind",
        "reference_schedule_warmup_steps",
        "reference_schedule_start_factor",
        "reference_preconditioning_method",
        "reference_preconditioning_reference",
        "reference_preconditioning_damping",
        "reference_preconditioning_eig_floor_rel",
        "reference_preconditioning_eig_floor_abs",
        "reference_preconditioning_lr_clip",
        "variance_floor",
    )
    for key in optional_keys:
        if subblock_cfg.get(key) is not None:
            value = subblock_cfg[key]
            if key == "reference_n_iter" or key == "reference_schedule_warmup_steps":
                value = int(value)
            elif key in {
                "reference_base_lr",
                "reference_schedule_start_factor",
                "reference_preconditioning_damping",
                "reference_preconditioning_eig_floor_rel",
                "reference_preconditioning_eig_floor_abs",
                "variance_floor",
            }:
                value = float(value)
            options[key] = value
    if subblock_cfg.get("reference_preconditioning_enabled") is not None:
        options["reference_preconditioning_enabled"] = bool(
            subblock_cfg["reference_preconditioning_enabled"]
        )
    forwarded_flags = [
        SUBBLOCK_OPTION_FLAG_MAP[key]
        for key in SUBBLOCK_OPTION_FLAG_MAP
        if key in options
    ]
    if "reference_preconditioning_enabled" in options:
        forwarded_flags.append(
            "--reference-preconditioning-enabled"
            if options["reference_preconditioning_enabled"]
            else "--reference-preconditioning-disabled"
        )
    options["forwarded_flags"] = forwarded_flags
    return options


def build_subblock_command(
    *,
    case_root_parent: Path,
    case_subblock_name: str,
    theta_labels: Sequence[str],
    offsets: Mapping[str, float],
    subblock_cfg: Mapping[str, Any],
    trace_seed: int | None = None,
    noise_seed: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(SUBBLOCK_SCRIPT),
        "--results-root",
        str(case_root_parent),
        "--case-name",
        case_subblock_name,
        "--mode",
        "schur_summary",
        "--n-frames",
        str(int(subblock_cfg.get("n_frames", 3))),
        "--noise",
        str(subblock_cfg.get("noise", "disabled")),
        "--theta-keys",
        ",".join(str(label) for label in theta_labels),
        "--phi-ref",
        str(subblock_cfg.get("phi_ref", "truth_when_available")),
        "--schur-curvature-method",
        str(subblock_cfg.get("schur_curvature_method", "auto")),
        "--max-dense-dim",
        str(int(subblock_cfg.get("max_dense_dim", 40))),
        "--schur-damping",
        str(float(subblock_cfg.get("schur_damping", 1.0e-8))),
    ]
    for key in sorted(offsets):
        command.extend(["--theta-reference-offset", f"{key}={float(offsets[key])}"])
    if trace_seed is not None:
        command.extend(["--trace-seed", str(int(trace_seed))])
    if noise_seed is not None:
        command.extend(["--render-seed", str(int(noise_seed))])
    trace_jitter_cfg = dict(subblock_cfg.get("trace_jitter", {}) or {})
    if trace_jitter_cfg.get("x_sigma_as") is not None:
        command.extend(
            ["--trace-jitter-x-sigma-as", str(float(trace_jitter_cfg["x_sigma_as"]))]
        )
    if trace_jitter_cfg.get("y_sigma_as") is not None:
        command.extend(
            ["--trace-jitter-y-sigma-as", str(float(trace_jitter_cfg["y_sigma_as"]))]
        )
    if trace_jitter_cfg.get("pa_sigma_deg") is not None:
        command.extend(
            [
                "--trace-jitter-pa-sigma-deg",
                str(float(trace_jitter_cfg["pa_sigma_deg"])),
            ]
        )
    command_options = resolve_subblock_command_options(subblock_cfg)
    for key, flag in SUBBLOCK_OPTION_FLAG_MAP.items():
        if key in command_options:
            command.extend([flag, _format_cli_value(command_options[key])])
    if "reference_preconditioning_enabled" in command_options:
        command.append(
            "--reference-preconditioning-enabled"
            if command_options["reference_preconditioning_enabled"]
            else "--reference-preconditioning-disabled"
        )
    return command


def build_campaign_plan(
    *,
    config_path: Path | None,
    results_root: Path,
    run_name: str | None,
    system_preset: str | None,
    args: argparse.Namespace | None = None,
) -> CampaignPlan:
    config = _load_campaign_config(config_path)
    experiment_cfg = _experiment_config(config)
    if args is not None:
        experiment_cfg = _apply_cli_overrides(experiment_cfg, args)
    if run_name is not None:
        experiment_cfg["run_name"] = run_name
    resolved_run_name = str(experiment_cfg.get("run_name") or f"observation_bias_campaign_{timestamp_tag()}")
    run_root = Path(results_root).resolve() / resolved_run_name
    effective_system_preset = system_preset or experiment_cfg.get("system_preset")
    system_store, system_cfg, system_provenance = _resolve_system_store(
        config=config,
        system_preset=effective_system_preset,
    )
    observation_theta_cfg = experiment_cfg.get("observation_theta", {}) or {}
    layout, layout_metadata = build_system_observation_theta_layout(
        system_store,
        config=observation_theta_cfg,
    )
    layout_metadata["system"] = system_provenance
    layout_metadata["resolved_system"] = system_cfg
    partition = _validate_partition_config(experiment_cfg.get("state_partition"), layout)
    prior_truth = build_prior_mean_from_store(layout.labels, store=system_store)
    configured_cases = _parse_bias_cases(
        experiment_cfg,
        layout=layout,
        layout_metadata=layout_metadata,
    )
    truth_by_label = {
        label: float(prior_truth[index]) for index, label in enumerate(layout.labels)
    }
    prior_draw_cases, prior_draw_rows_by_case = _generate_prior_draw_cases(
        experiment_cfg=experiment_cfg,
        labels=layout.labels,
        truth_by_label=truth_by_label,
    )
    cases, case_generation = _resolve_case_generation(
        experiment_cfg,
        parsed_cases=configured_cases,
        prior_draw_cases=prior_draw_cases,
    )
    subblock_cfg = dict(experiment_cfg.get("subblocks", {}) or {})
    subblock_command_options = resolve_subblock_command_options(subblock_cfg)
    seeding_cfg = _resolve_seeding_config(experiment_cfg)
    n_subblocks = int(subblock_cfg.get("n_subblocks", 3))
    if n_subblocks <= 0:
        raise ValueError("subblocks.n_subblocks must be positive.")
    subblock_root = run_root / "subblock_runs"
    commands: dict[str, list[list[str]]] = {}
    summary_paths: dict[str, list[Path]] = {}
    subblock_plans: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        commands[case.case_name] = []
        summary_paths[case.case_name] = []
        subblock_plans[case.case_name] = []
        for subblock_index in range(n_subblocks):
            subblock_name = f"{case.case_name}/subblock_{subblock_index:03d}"
            seeds = _derive_subblock_seeds(
                run_name=resolved_run_name,
                case_name=case.case_name,
                subblock_index=subblock_index,
                seed_policy=str(seeding_cfg["seed_policy"]),
                base_seed=int(seeding_cfg["base_seed"]),
            )
            summary_path = (
                subblock_root
                / subblock_name
                / "study"
                / "schur_summary"
                / "subblock_summary.json"
            )
            commands[case.case_name].append(
                build_subblock_command(
                    case_root_parent=subblock_root,
                    case_subblock_name=subblock_name,
                    theta_labels=layout.labels,
                    offsets=case.theta_reference_offsets,
                    subblock_cfg=subblock_cfg,
                    trace_seed=int(seeds["trace_seed"]),
                    noise_seed=int(seeds["noise_seed"]),
                )
            )
            summary_paths[case.case_name].append(summary_path)
            plan_options = {
                key: value
                for key, value in subblock_command_options.items()
                if key != "forwarded_flags"
            }
            subblock_plans[case.case_name].append(
                {
                    "case_name": case.case_name,
                    "subblock_index": int(subblock_index),
                    "subblock_name": subblock_name,
                    "summary_path": str(summary_path),
                    "case_origin": case.case_origin,
                    "n_frames": int(subblock_cfg.get("n_frames", 3)),
                    "noise": str(subblock_cfg.get("noise", "disabled")),
                    "phi_ref": str(subblock_cfg.get("phi_ref", "truth_when_available")),
                    "schur_curvature_method": str(
                        subblock_cfg.get("schur_curvature_method", "auto")
                    ),
                    "max_dense_dim": int(subblock_cfg.get("max_dense_dim", 40)),
                    "schur_damping": float(subblock_cfg.get("schur_damping", 1.0e-8)),
                    "forwarded_flags": ",".join(
                        str(flag) for flag in subblock_command_options["forwarded_flags"]
                    ),
                    **plan_options,
                    **seeds,
                }
            )
    resolved_config = {
        **config,
        "experiment": experiment_cfg,
        "system": system_cfg,
    }
    return CampaignPlan(
        run_root=run_root,
        layout=layout,
        layout_metadata=layout_metadata,
        prior_truth=prior_truth,
        cases=cases,
        subblock_commands=commands,
        summary_paths=summary_paths,
        subblock_plans=subblock_plans,
        prior_draw_rows_by_case=prior_draw_rows_by_case,
        config=resolved_config,
        partition=partition,
        case_generation=case_generation,
    )


def _plan_payload(plan: CampaignPlan) -> dict[str, Any]:
    subblock_cfg = plan.config["experiment"].get("subblocks", {})
    subblock_command_options = resolve_subblock_command_options(subblock_cfg)
    n_phi = 3 * int(subblock_cfg.get("n_frames", 3))
    combined_dim = int(plan.layout.size + n_phi)
    max_dense_dim = int(subblock_cfg.get("max_dense_dim", 40))
    eigen_cfg = dict(plan.config["experiment"].get("eigenbasis", {}) or {})
    forecast_cfg = _forecast_config(plan.config["experiment"])
    actual_n_summaries = int(plan.config["experiment"].get("subblocks", {}).get("n_subblocks", 3))
    forecast_grid = parse_forecast_grid(
        forecast_cfg.get("n_subblocks_grid"),
        actual_n_summaries=actual_n_summaries,
    )
    return {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "run_root": str(plan.run_root),
        "theta_layout": plan.layout.to_dict(),
        "layout_metadata": plan.layout_metadata,
        "state_partition": plan.partition,
        "case_generation": dict(plan.case_generation),
        "subblock_command_options": dict(subblock_command_options),
        "seeding": _resolve_seeding_config(plan.config["experiment"]),
        "eigenbasis": {
            **eigen_cfg,
            "resolved_sources": list(_resolve_eigen_sources(eigen_cfg)),
        },
        "forecast": {
            "enabled": bool(forecast_cfg.get("enabled", False)),
            "modes": list(forecast_cfg.get("modes", [])),
            "n_subblocks_grid": list(forecast_grid),
            "single_observation_n_subblocks": int(
                forecast_cfg.get("single_observation_n_subblocks", 1800)
            ),
            "subblock_duration_s": float(forecast_cfg.get("subblock_duration_s", 1.0)),
            "case_output_dirs": {
                case.case_name: str(plan.run_root / "cases" / case.case_name / "forecast")
                for case in plan.cases
            },
            "limitations": {
                "replicate": (
                    "Repeats actual reduced information and score summaries; "
                    "not an independent shot-noise realization model."
                ),
                "fixed_information_score_noise": (
                    "Keeps template information fixed and samples score noise "
                    "with covariance alpha * S."
                ),
            },
        },
        "bias_cases": [
            {
                "case_name": case.case_name,
                "theta_reference_offsets": dict(case.theta_reference_offsets),
                "case_origin": case.case_origin,
                "has_case_prior_sigma": bool(case.prior_sigma_by_label),
                "is_prior_draw": bool(case.case_origin == "prior_draw"),
            }
            for case in plan.cases
        ],
        "prior_draw_rows_by_case": {
            case_name: list(rows)
            for case_name, rows in plan.prior_draw_rows_by_case.items()
        },
        "dimension_estimate": {
            "n_theta": int(plan.layout.size),
            "n_phi": int(n_phi),
            "combined_dim": int(combined_dim),
            "max_dense_dim": int(max_dense_dim),
            "dense_schur_allowed": bool(combined_dim <= max_dense_dim),
            "structured_schur_recommended": bool(combined_dim > max_dense_dim),
        },
        "subblock_commands": {
            name: [" ".join(command) for command in commands]
            for name, commands in plan.subblock_commands.items()
        },
        "summary_paths": {
            name: [str(path) for path in paths]
            for name, paths in plan.summary_paths.items()
        },
        "subblock_plan": {
            name: list(items) for name, items in plan.subblock_plans.items()
        },
    }


def _run_command(command: Sequence[str], *, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env),
    )


def execute_subblocks(
    plan: CampaignPlan,
    *,
    resume: bool,
    max_workers: int,
    fail_fast: bool,
    quiet: bool,
) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    jobs: list[tuple[str, Path, list[str]]] = []
    for case_name, commands in plan.subblock_commands.items():
        for command, summary_path in zip(commands, plan.summary_paths[case_name], strict=True):
            if resume and summary_path.exists():
                continue
            jobs.append((case_name, summary_path, command))
    if not jobs:
        return

    def _job(item: tuple[str, Path, list[str]]) -> tuple[str, Path, subprocess.CompletedProcess[str]]:
        case_name, summary_path, command = item
        result = _run_command(command, env=env)
        return case_name, summary_path, result

    if max_workers <= 1:
        for job in jobs:
            try:
                case_name, summary_path, _ = _job(job)
                if not quiet:
                    print(f"[observation_bias_campaign] completed {case_name}: {summary_path}", flush=True)
            except subprocess.CalledProcessError as exc:
                if fail_fast:
                    raise RuntimeError(exc.stderr or str(exc)) from exc
                print(exc.stderr or str(exc), file=sys.stderr, flush=True)
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            try:
                case_name, summary_path, _ = future.result()
                if not quiet:
                    print(f"[observation_bias_campaign] completed {case_name}: {summary_path}", flush=True)
            except subprocess.CalledProcessError as exc:
                if fail_fast:
                    raise RuntimeError(exc.stderr or str(exc)) from exc
                print(exc.stderr or str(exc), file=sys.stderr, flush=True)


def _flatten_subblock_plan_rows(plan: CampaignPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name, items in plan.subblock_plans.items():
        for item, command in zip(items, plan.subblock_commands[case_name], strict=True):
            rows.append(
                {
                    **item,
                    "command": " ".join(command),
                }
            )
    return rows


def _flatten_prior_draw_rows(plan: CampaignPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_name, case_rows in plan.prior_draw_rows_by_case.items():
        for row in case_rows:
            rows.append(dict(row))
    return rows


def _bias_case_rows(plan: CampaignPlan) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in plan.cases:
        rows.append(
            {
                "case_name": case.case_name,
                "case_origin": case.case_origin,
                "has_case_prior_sigma": bool(case.prior_sigma_by_label),
                "has_prior_draw_metadata": bool(case.prior_draw_metadata),
                "n_theta_offsets": len(case.theta_reference_offsets),
            }
        )
    return rows


def _load_case_summaries(paths: Sequence[Path]) -> list[SubblockSummary]:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing subblock summaries: " + ", ".join(str(path) for path in missing)
        )
    summaries = [load_subblock_summary(path) for path in paths]
    labels = tuple(summaries[0].theta_labels)
    for index, summary in enumerate(summaries[1:], start=1):
        if tuple(summary.theta_labels) != labels:
            raise ValueError(
                f"Summary {index} labels differ from the first summary for this case."
            )
    return summaries


def _truth_reference_maps(
    *,
    plan: CampaignPlan,
    case: BiasCase,
    summaries: Sequence[SubblockSummary],
    summary_paths: Sequence[Path],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    labels = tuple(plan.layout.labels)
    truth = {label: float(plan.prior_truth[index]) for index, label in enumerate(labels)}
    reference = {
        label: float(summaries[0].theta_ref[index]) for index, label in enumerate(labels)
    }
    offsets = {label: 0.0 for label in labels}
    offsets.update({key: float(value) for key, value in case.theta_reference_offsets.items()})
    for path in summary_paths:
        try:
            payload = load_subblock_summary_artifact_payload(path)
        except Exception:
            continue
        override_payload = payload.get("metadata", {}).get("theta_reference_overrides")
        if not isinstance(override_payload, Mapping):
            override_payload = payload.get("theta_reference_overrides")
        items = override_payload.get("items", []) if isinstance(override_payload, Mapping) else []
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            key = str(item.get("key", ""))
            if key not in labels:
                continue
            if item.get("truth_value") is not None:
                truth[key] = float(item["truth_value"])
            elif item.get("reference_base_value") is not None:
                truth[key] = float(item["reference_base_value"])
            if item.get("reference_value") is not None:
                reference[key] = float(item["reference_value"])
            if item.get("offset") is not None:
                offsets[key] = float(item["offset"])
    for key, offset in offsets.items():
        if abs(offset) > 0.0 and key in reference:
            truth.setdefault(key, float(reference[key] - offset))
    return truth, reference, offsets


def _posterior_rows(
    *,
    case_name: str,
    labels: Sequence[str],
    truth: Mapping[str, float],
    reference: Mapping[str, float],
    offsets: Mapping[str, float],
    posterior_mean: np.ndarray,
    posterior_sigma: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        truth_value = float(truth[label])
        reference_value = float(reference[label])
        posterior_value = float(posterior_mean[index])
        posterior_error = posterior_value - truth_value
        reference_error = reference_value - truth_value
        posterior_shift = posterior_value - reference_value
        rows.append(
            {
                "case_name": case_name,
                "theta_label": label,
                "truth_value": truth_value,
                "reference_value": reference_value,
                "theta_reference_offset": float(offsets.get(label, 0.0)),
                "posterior_mean": posterior_value,
                "posterior_shift": posterior_shift,
                "posterior_error": posterior_error,
                "posterior_sigma": float(posterior_sigma[index]),
                "posterior_error_over_sigma": _safe_fraction(posterior_error, float(posterior_sigma[index])),
                "correction_fraction": _safe_fraction(posterior_shift, truth_value - reference_value),
                "residual_fraction": _safe_fraction(posterior_error, reference_error),
                "label_group": _label_group(label),
                "unit": _parameter_unit(label),
            }
        )
    return rows


def _science_row(
    *,
    case_name: str,
    labels: Sequence[str],
    truth: Mapping[str, float],
    reference: Mapping[str, float],
    posterior_mean: np.ndarray,
    posterior_sigma: np.ndarray,
) -> dict[str, Any]:
    label = "source.separation_as"
    index = labels.index(label)
    truth_value = float(truth[label])
    reference_value = float(reference[label])
    posterior_value = float(posterior_mean[index])
    error = posterior_value - truth_value
    sigma = float(posterior_sigma[index])
    prior_error = reference_value - truth_value
    return {
        "case_name": case_name,
        "truth_separation_as": truth_value,
        "reference_separation_as": reference_value,
        "posterior_separation_as": posterior_value,
        "posterior_separation_shift_as": posterior_value - reference_value,
        "posterior_separation_error_as": error,
        "posterior_separation_error_microas": error * 1.0e6,
        "posterior_separation_sigma_as": sigma,
        "posterior_separation_sigma_microas": sigma * 1.0e6,
        "posterior_separation_error_over_sigma": _safe_fraction(error, sigma),
        "separation_correction_fraction": _safe_fraction(posterior_value - reference_value, truth_value - reference_value),
        "moves_separation_toward_truth": bool(abs(error) < abs(prior_error)),
    }


def _top_contributors(
    *,
    basis: Any,
    mode_index: int,
    top_k: int,
) -> list[tuple[str, float]]:
    return basis.mode_contributors(mode_index, top_k=max(1, top_k))


def _eigen_rows(
    *,
    case_name: str,
    source_name: str,
    labels: Sequence[str],
    matrix: np.ndarray,
    prior_sigma: np.ndarray,
    reference_error: np.ndarray,
    posterior_error: np.ndarray,
    posterior_shift: np.ndarray,
    eigen_cfg: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    Any,
]:
    whiten = bool(eigen_cfg.get("whiten", True))
    source_matrix = (
        build_prior_whitened_information_gain_matrix(matrix, prior_sigma)
        if whiten
        else np.asarray(matrix, dtype=float)
    )
    basis = build_observation_eigenbasis(
        source_matrix,
        labels,
        eig_floor_abs=float(eigen_cfg.get("eig_floor_abs", 0.0)),
        eig_floor_rel=float(eigen_cfg.get("eig_floor_rel", 1.0e-12)),
    )
    top_k = int(eigen_cfg.get("top_k_contributors", 8))
    raw_sigmas = basis.raw_sigma_along_modes()
    eigenvalue_rows: list[dict[str, Any]] = []
    contributor_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    projected_reference_error = reference_error / prior_sigma if whiten else reference_error
    projected_posterior_error = posterior_error / prior_sigma if whiten else posterior_error
    projected_posterior_shift = posterior_shift / prior_sigma if whiten else posterior_shift
    prior_proj = basis.physical_delta_to_eigen(projected_reference_error)
    posterior_proj = basis.physical_delta_to_eigen(projected_posterior_error)
    shift_proj = basis.physical_delta_to_eigen(projected_posterior_shift)
    for mode_index, eigenvalue in enumerate(basis.eigenvalues):
        eigenvalue_rows.append(
            {
                "case_name": case_name,
                "mode_index": int(mode_index),
                "eigenvalue": float(eigenvalue),
                "sqrt_eigenvalue": float(math.sqrt(max(float(eigenvalue), 0.0))),
                "sigma_equivalent": float(raw_sigmas[mode_index]),
                "is_weak": bool(basis.weak_mode_mask[mode_index]),
                "is_retained": bool(not basis.was_floored()[mode_index]),
            }
        )
        contributors = _top_contributors(basis=basis, mode_index=mode_index, top_k=top_k)
        for rank, (label, coeff) in enumerate(contributors, start=1):
            contributor_rows.append(
                {
                    "case_name": case_name,
                    "mode_index": int(mode_index),
                    "rank": int(rank),
                    "theta_label": label,
                    "coefficient": float(coeff),
                    "abs_coefficient": float(abs(coeff)),
                    "label_group": _label_group(label),
                    "unit": _parameter_unit(label),
                }
            )
        top_summary = "; ".join(f"{label}:{coeff:+.4f}" for label, coeff in contributors[:3])
        projection_rows.append(
            {
                "case_name": case_name,
                "mode_index": int(mode_index),
                "eigenvalue": float(eigenvalue),
                "prior_error_projection": float(prior_proj[mode_index]),
                "posterior_error_projection": float(posterior_proj[mode_index]),
                "posterior_shift_projection": float(shift_proj[mode_index]),
                "correction_fraction_projection": _safe_fraction(
                    float(shift_proj[mode_index]),
                    -float(prior_proj[mode_index]),
                ),
                "top_contributor_summary": top_summary,
            }
        )
    weak_mode_rows: list[dict[str, Any]] = []
    for mode_index, eigenvalue in enumerate(basis.eigenvalues):
        if not bool(basis.weak_mode_mask[mode_index]):
            continue
        vector = basis.eigenvectors[:, int(mode_index)]
        abs_vector = np.abs(vector)
        primary_weight = float(
            np.sum(
                abs_vector[
                    [
                        i
                        for i, label in enumerate(labels)
                        if label.startswith("optics.primary.zernike_coeffs_nm")
                    ]
                ]
            )
        )
        secondary_weight = float(
            np.sum(
                abs_vector[
                    [
                        i
                        for i, label in enumerate(labels)
                        if label.startswith("optics.secondary.zernike_coeffs_nm")
                    ]
                ]
            )
        )
        source_scalar_weight = float(
            np.sum(
                abs_vector[
                    [i for i, label in enumerate(labels) if label.startswith("source.")]
                ]
            )
        )
        plate_scale_weight = float(
            np.sum(
                abs_vector[
                    [
                        i
                        for i, label in enumerate(labels)
                        if label == "optics.plate_scale_as_per_pix"
                    ]
                ]
            )
        )
        separation_weight = float(
            np.sum(
                abs_vector[
                    [i for i, label in enumerate(labels) if label == "source.separation_as"]
                ]
            )
        )
        group_weights = {
            "primary_zernikes": primary_weight,
            "secondary_zernikes": secondary_weight,
            "source_scalars": source_scalar_weight,
            "plate_scale": plate_scale_weight,
        }
        dominant_group = max(group_weights, key=lambda item: group_weights[item])
        notes = ""
        if dominant_group in {"primary_zernikes", "secondary_zernikes"}:
            notes = "M1/M2-dominated weak mode candidate"
        weak_mode_rows.append(
            {
                "case_name": case_name,
                "source_matrix": source_name,
                "mode_index": int(mode_index),
                "eigenvalue": float(eigenvalue),
                "sigma_equivalent": float(raw_sigmas[mode_index]),
                "top_contributor_summary": projection_rows[mode_index]["top_contributor_summary"],
                "primary_zernike_weight": primary_weight,
                "secondary_zernike_weight": secondary_weight,
                "source_scalar_weight": source_scalar_weight,
                "plate_scale_weight": plate_scale_weight,
                "separation_weight": separation_weight,
                "dominant_group": dominant_group,
                "notes": notes,
            }
        )
    summary = {
        "case_name": case_name,
        "source_matrix": source_name,
        "source_matrix_whitened": bool(whiten),
        "condition_number": float(basis.condition_number),
        "eig_floor_abs": float(basis.eig_floor_abs),
        "eig_floor_rel": float(basis.eig_floor_rel),
        "n_modes": int(len(labels)),
        "n_weak_modes": int(np.count_nonzero(basis.weak_mode_mask)),
    }
    return (
        eigenvalue_rows,
        contributor_rows,
        projection_rows,
        weak_mode_rows,
        summary,
        basis,
    )


def _plot_eigenvalue_spectrum(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAVE_MATPLOTLIB or plt is None:
        return
    if not rows:
        return
    x = np.asarray([int(row["mode_index"]) + 1 for row in rows], dtype=int)
    y = np.asarray([float(row["eigenvalue"]) for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(x, np.clip(y, 1.0e-30, None), marker="o")
    ax.set_xlabel("Mode Index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_group_error(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAVE_MATPLOTLIB or plt is None:
        return
    if not rows:
        return
    groups = sorted(set(str(row["label_group"]) for row in rows))
    values = [
        np.nanmax([
            abs(float(row["posterior_error_over_sigma"]))
            for row in rows
            if row["label_group"] == group and math.isfinite(float(row["posterior_error_over_sigma"]))
        ] or [0.0])
        for group in groups
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(groups, values)
    ax.set_ylabel("max |posterior error / sigma|")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _forecast_config(experiment_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(experiment_cfg.get("forecast", {}) or {})
    if "enabled" not in cfg:
        cfg["enabled"] = False
    if "modes" not in cfg:
        cfg["modes"] = ["replicate", "fixed_information_score_noise"]
    cfg.setdefault("n_subblocks_grid", [1, 3, 5, 10, 30, 100, 300, 1000, 1800])
    cfg.setdefault("subblock_duration_s", 1.0)
    cfg.setdefault("single_observation_n_subblocks", 1800)
    cfg.setdefault("plots", True)
    cfg.setdefault("replicate", {"enabled": True})
    cfg.setdefault(
        "fixed_information_score_noise",
        {
            "enabled": True,
            "n_trials": 100,
            "seed": 2026,
            "score_noise_alpha": 1.0,
            "score_noise_eig_floor_abs": 0.0,
            "score_noise_eig_floor_rel": 1.0e-12,
            "truth_mode": "campaign_truth",
        },
    )
    modes = tuple(str(mode) for mode in cfg.get("modes", ()))
    invalid = [mode for mode in modes if mode not in SUPPORTED_FORECAST_MODES]
    if invalid:
        raise ValueError("Unsupported forecast mode(s): " + ", ".join(sorted(invalid)))
    cfg["modes"] = list(modes)
    return cfg


def parse_forecast_grid(
    raw: Sequence[int] | None,
    *,
    actual_n_summaries: int,
) -> tuple[int, ...]:
    values = list(raw or [1, 3, 5, 10, 30, 100, 300, 1000, 1800])
    values.append(int(actual_n_summaries))
    parsed: list[int] = []
    for value in values:
        count = int(value)
        if count < 1:
            raise ValueError("forecast n_subblocks values must be >= 1.")
        parsed.append(count)
    return tuple(sorted(set(parsed)))


def replicate_summaries_for_count(
    summaries: Sequence[SubblockSummary],
    n_subblocks: int,
) -> tuple[SubblockSummary, ...]:
    if not summaries:
        raise ValueError("At least one summary is required for forecast replication.")
    if int(n_subblocks) < 1:
        raise ValueError("n_subblocks must be >= 1.")
    return tuple(summaries[index % len(summaries)] for index in range(int(n_subblocks)))


def _sample_score_noise(
    information: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float,
    eig_floor_abs: float,
    eig_floor_rel: float,
) -> np.ndarray:
    if alpha < 0.0:
        raise ValueError("score_noise_alpha must be non-negative.")
    matrix = 0.5 * (np.asarray(information, dtype=float) + np.asarray(information, dtype=float).T)
    if alpha == 0.0:
        return np.zeros((matrix.shape[0],), dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    max_eig = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    floor = max(float(eig_floor_abs), float(eig_floor_rel) * max(max_eig, 0.0))
    effective = np.clip(eigenvalues, floor, None)
    standard = rng.normal(loc=0.0, scale=1.0, size=effective.shape)
    return eigenvectors @ (np.sqrt(alpha * effective) * standard)


def synthesize_score_noise_summaries(
    summaries: Sequence[SubblockSummary],
    *,
    n_subblocks: int,
    theta_true: np.ndarray,
    rng: np.random.Generator,
    alpha: float,
    eig_floor_abs: float,
    eig_floor_rel: float,
) -> tuple[SubblockSummary, ...]:
    templates = replicate_summaries_for_count(summaries, n_subblocks)
    synthetic: list[SubblockSummary] = []
    truth = np.asarray(theta_true, dtype=float)
    for index, template in enumerate(templates):
        expected = template.reduced_information @ (template.theta_ref - truth)
        score = expected + _sample_score_noise(
            template.reduced_information,
            rng=rng,
            alpha=float(alpha),
            eig_floor_abs=float(eig_floor_abs),
            eig_floor_rel=float(eig_floor_rel),
        )
        synthetic.append(
            SubblockSummary.from_reduced_form(
                subblock_id=f"forecast_{index:06d}_{template.subblock_id}",
                theta_labels=template.theta_labels,
                theta_ref=template.theta_ref,
                reduced_information=template.reduced_information,
                reduced_score=score,
                summary_kind=f"{template.summary_kind}_score_noise_forecast",
                diagnostics={
                    "template_subblock_id": template.subblock_id,
                    "forecast_sequence_index": int(index),
                    "score_noise_alpha": float(alpha),
                },
            )
        )
    return tuple(synthetic)


def _forecast_duration_fields(
    *,
    n_subblocks: int,
    actual_n_summaries: int,
    subblock_duration_s: float,
    single_observation_n_subblocks: int,
) -> dict[str, Any]:
    duration_s = float(n_subblocks) * float(subblock_duration_s)
    return {
        "n_actual_summaries": int(actual_n_summaries),
        "n_subblocks": int(n_subblocks),
        "equivalent_duration_s": duration_s,
        "equivalent_duration_min": duration_s / 60.0,
        "is_actual_summary_count": bool(int(n_subblocks) == int(actual_n_summaries)),
        "is_single_observation_target": bool(
            int(n_subblocks) == int(single_observation_n_subblocks)
        ),
    }


def build_forecast_information_row(
    *,
    case_name: str,
    forecast_mode: str,
    labels: Sequence[str],
    summaries: Sequence[SubblockSummary],
    n_subblocks: int,
    actual_n_summaries: int,
    subblock_duration_s: float,
    single_observation_n_subblocks: int,
) -> dict[str, Any]:
    info = accumulate_summary_information(labels, summaries)
    diagnostics = _matrix_diagnostics(info)
    return {
        "case_name": case_name,
        "forecast_mode": forecast_mode,
        **_forecast_duration_fields(
            n_subblocks=n_subblocks,
            actual_n_summaries=actual_n_summaries,
            subblock_duration_s=subblock_duration_s,
            single_observation_n_subblocks=single_observation_n_subblocks,
        ),
        "accumulated_information_rank_estimate": int(diagnostics.rank_estimate),
        "accumulated_information_min_eigenvalue": float(diagnostics.min_eigenvalue),
        "accumulated_information_condition_number": float(diagnostics.condition_number),
        "accumulated_information_trace": float(diagnostics.trace),
    }


def build_forecast_science_row(
    *,
    case_name: str,
    forecast_mode: str,
    labels: Sequence[str],
    update_result: Any,
    accumulated_information: np.ndarray,
    prior_sigma: np.ndarray,
    theta_true: np.ndarray,
    n_subblocks: int,
    actual_n_summaries: int,
    subblock_duration_s: float,
    single_observation_n_subblocks: int,
    prior_sigma_source: str,
    n_trials: int | None = None,
) -> dict[str, Any]:
    label = "source.separation_as"
    index = labels.index(label) if label in labels else None
    posterior_sigma = update_result.posterior.sigma()
    posterior_diag = _matrix_diagnostics(update_result.posterior.precision)
    info_diag = _matrix_diagnostics(accumulated_information)
    row: dict[str, Any] = {
        "case_name": case_name,
        "forecast_mode": forecast_mode,
        **_forecast_duration_fields(
            n_subblocks=n_subblocks,
            actual_n_summaries=actual_n_summaries,
            subblock_duration_s=subblock_duration_s,
            single_observation_n_subblocks=single_observation_n_subblocks,
        ),
        "theta_dim": int(len(labels)),
        "separation_label_found": bool(index is not None),
        "posterior_precision_rank_estimate": int(posterior_diag.rank_estimate),
        "posterior_precision_min_eigenvalue": float(posterior_diag.min_eigenvalue),
        "posterior_precision_condition_number": float(posterior_diag.condition_number),
        "accumulated_information_rank_estimate": int(info_diag.rank_estimate),
        "accumulated_information_condition_number": float(info_diag.condition_number),
        "prior_sigma_source": str(prior_sigma_source),
    }
    if n_trials is not None:
        row["n_trials"] = int(n_trials)
    if index is None:
        row.update(
            {
                "separation_posterior_sigma_as": float("nan"),
                "separation_posterior_sigma_microas": float("nan"),
                "separation_posterior_error_as": float("nan"),
                "separation_posterior_error_microas": float("nan"),
                "separation_posterior_error_over_sigma": float("nan"),
                "separation_posterior_sigma_over_prior_sigma": float("nan"),
            }
        )
        return row
    sigma = float(posterior_sigma[index])
    error = float(update_result.posterior.mean[index] - theta_true[index])
    row.update(
        {
            "separation_posterior_sigma_as": sigma,
            "separation_posterior_sigma_microas": sigma * 1.0e6,
            "separation_posterior_error_as": error,
            "separation_posterior_error_microas": error * 1.0e6,
            "separation_posterior_error_over_sigma": _safe_fraction(error, sigma),
            "separation_posterior_sigma_over_prior_sigma": _safe_fraction(
                sigma,
                float(prior_sigma[index]),
            ),
        }
    )
    return row


def build_forecast_posterior_rows(
    *,
    case_name: str,
    forecast_mode: str,
    labels: Sequence[str],
    update_result: Any,
    theta_true: np.ndarray,
    prior_sigma: np.ndarray,
    n_subblocks: int,
    trial_index: int | None = None,
) -> list[dict[str, Any]]:
    sigma = update_result.posterior.sigma()
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        error = float(update_result.posterior.mean[index] - theta_true[index])
        row: dict[str, Any] = {
            "case_name": case_name,
            "forecast_mode": forecast_mode,
            "n_subblocks": int(n_subblocks),
            "theta_label": label,
            "truth_value": float(theta_true[index]),
            "posterior_mean": float(update_result.posterior.mean[index]),
            "posterior_error": error,
            "posterior_sigma": float(sigma[index]),
            "posterior_error_over_sigma": _safe_fraction(error, float(sigma[index])),
            "posterior_sigma_over_prior_sigma": _safe_fraction(
                float(sigma[index]),
                float(prior_sigma[index]),
            ),
            "label_group": _label_group(label),
            "unit": _parameter_unit(label),
        }
        if trial_index is not None:
            row["trial_index"] = int(trial_index)
        rows.append(row)
    return rows


def _plot_forecast_line(
    *,
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    y_key: str,
    ylabel: str,
    actual_n_summaries: int,
    single_observation_n_subblocks: int,
) -> None:
    if not _HAVE_MATPLOTLIB or plt is None or not rows:
        return
    x = np.asarray([int(row["n_subblocks"]) for row in rows], dtype=int)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=float)
    order = np.argsort(x)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x[order], y[order], marker="o")
    ax.axvline(int(actual_n_summaries), color="0.4", linestyle="--", linewidth=1)
    ax.axvline(int(single_observation_n_subblocks), color="0.2", linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Subblocks")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_forecast_manifest(
    *,
    path: Path,
    case_name: str,
    mode: str,
    summaries: Sequence[SubblockSummary],
    summary_paths: Sequence[Path],
    prior: ObservationBeliefState,
    prior_sigma: np.ndarray,
    theta_true: np.ndarray,
    forecast_cfg: Mapping[str, Any],
    grid: Sequence[int],
    extra: Mapping[str, Any] | None = None,
) -> None:
    limitations = {
        "replicate": (
            "This mode repeats the actual summary matrices and score vectors. "
            "It is an information accumulation sanity check and not an independent "
            "shot-noise realization model."
        ),
        "fixed_information_score_noise": (
            "This mode keeps the observed/template information matrices fixed and "
            "draws independent score noise with covariance alpha * S. Alpha is "
            "Fisher-consistent when alpha=1 but is not yet empirically calibrated "
            "unless a calibration artifact is supplied."
        ),
    }
    payload = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "case_name": case_name,
        "forecast_mode": mode,
        "limitations": limitations.get(mode, ""),
        "input_summary_paths": [str(path) for path in summary_paths],
        "n_actual_summaries": int(len(summaries)),
        "input_summary_theta_refs": [summary.theta_ref.tolist() for summary in summaries],
        "input_summary_matrix_diagnostics": [
            _matrix_diagnostics(summary.reduced_information).to_dict()
            for summary in summaries
        ],
        "prior": prior.to_dict(),
        "prior_sigma": np.asarray(prior_sigma, dtype=float).tolist(),
        "truth_vector": np.asarray(theta_true, dtype=float).tolist(),
        "forecast_grid": [int(value) for value in grid],
        "subblock_duration_s": float(forecast_cfg.get("subblock_duration_s", 1.0)),
        "single_observation_n_subblocks": int(
            forecast_cfg.get("single_observation_n_subblocks", 1800)
        ),
    }
    if extra:
        payload.update(dict(extra))
    _write_json(path, payload)


def run_case_forecast(
    *,
    case_name: str,
    case_root: Path,
    summaries: Sequence[SubblockSummary],
    summary_paths: Sequence[Path],
    prior: ObservationBeliefState,
    prior_sigma: np.ndarray,
    theta_true: np.ndarray,
    theta_reference: np.ndarray,
    labels: Sequence[str],
    forecast_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = _forecast_config({"forecast": forecast_cfg})
    if not bool(cfg.get("enabled", False)):
        return {
            "forecast_results": [],
            "forecast_trial_results": [],
            "forecast_information_diagnostics": [],
            "forecast_summary": {"enabled": False},
        }
    grid = parse_forecast_grid(
        cfg.get("n_subblocks_grid"),
        actual_n_summaries=len(summaries),
    )
    subblock_duration_s = float(cfg.get("subblock_duration_s", 1.0))
    single_observation_n_subblocks = int(cfg.get("single_observation_n_subblocks", 1800))
    prior_sigma_source = str(prior.metadata.get("prior_sigma_source", "unknown"))
    forecast_root = case_root / "forecast"
    all_results: list[dict[str, Any]] = []
    all_trial_results: list[dict[str, Any]] = []
    all_trial_posterior_rows: list[dict[str, Any]] = []
    all_information_rows: list[dict[str, Any]] = []
    modes = tuple(str(mode) for mode in cfg.get("modes", ()))

    if "replicate" in modes and bool(dict(cfg.get("replicate", {}) or {}).get("enabled", True)):
        mode = "replicate"
        mode_root = forecast_root / mode
        rows: list[dict[str, Any]] = []
        posterior_rows: list[dict[str, Any]] = []
        info_rows: list[dict[str, Any]] = []
        for count in grid:
            sequence = replicate_summaries_for_count(summaries, count)
            update = update_observation_belief(prior, sequence)
            accumulated = accumulate_summary_information(labels, sequence)
            row = build_forecast_science_row(
                case_name=case_name,
                forecast_mode=mode,
                labels=labels,
                update_result=update,
                accumulated_information=accumulated,
                prior_sigma=prior_sigma,
                theta_true=theta_true,
                n_subblocks=count,
                actual_n_summaries=len(summaries),
                subblock_duration_s=subblock_duration_s,
                single_observation_n_subblocks=single_observation_n_subblocks,
                prior_sigma_source=prior_sigma_source,
            )
            rows.append(row)
            posterior_rows.extend(
                build_forecast_posterior_rows(
                    case_name=case_name,
                    forecast_mode=mode,
                    labels=labels,
                    update_result=update,
                    theta_true=theta_true,
                    prior_sigma=prior_sigma,
                    n_subblocks=count,
                )
            )
            info_rows.append(
                build_forecast_information_row(
                    case_name=case_name,
                    forecast_mode=mode,
                    labels=labels,
                    summaries=sequence,
                    n_subblocks=count,
                    actual_n_summaries=len(summaries),
                    subblock_duration_s=subblock_duration_s,
                    single_observation_n_subblocks=single_observation_n_subblocks,
                )
            )
        _write_csv_rows(mode_root / "forecast_results.csv", rows)
        _write_csv_rows(mode_root / "posterior_by_label_by_n_subblocks.csv", posterior_rows)
        _write_csv_rows(mode_root / "information_diagnostics.csv", info_rows)
        _write_forecast_manifest(
            path=mode_root / "manifest.json",
            case_name=case_name,
            mode=mode,
            summaries=summaries,
            summary_paths=summary_paths,
            prior=prior,
            prior_sigma=prior_sigma,
            theta_true=theta_true,
            forecast_cfg=cfg,
            grid=grid,
            extra={"theta_reference": np.asarray(theta_reference, dtype=float).tolist()},
        )
        if bool(cfg.get("plots", True)):
            _plot_forecast_line(
                path=mode_root / "separation_sigma_vs_n_subblocks.png",
                rows=rows,
                y_key="separation_posterior_sigma_microas",
                ylabel="Separation sigma (microas)",
                actual_n_summaries=len(summaries),
                single_observation_n_subblocks=single_observation_n_subblocks,
            )
            _plot_forecast_line(
                path=mode_root / "prior_normalized_sigma_vs_n_subblocks.png",
                rows=rows,
                y_key="separation_posterior_sigma_over_prior_sigma",
                ylabel="Separation sigma / prior sigma",
                actual_n_summaries=len(summaries),
                single_observation_n_subblocks=single_observation_n_subblocks,
            )
        all_results.extend(rows)
        all_information_rows.extend(info_rows)

    noise_cfg = dict(cfg.get("fixed_information_score_noise", {}) or {})
    if "fixed_information_score_noise" in modes and bool(noise_cfg.get("enabled", True)):
        mode = "fixed_information_score_noise"
        mode_root = forecast_root / mode
        n_trials = int(noise_cfg.get("n_trials", 100))
        if n_trials <= 0:
            raise ValueError("fixed_information_score_noise.n_trials must be positive.")
        seed = int(noise_cfg.get("seed", 2026))
        alpha = float(noise_cfg.get("score_noise_alpha", 1.0))
        if alpha < 0.0:
            raise ValueError("fixed_information_score_noise.score_noise_alpha must be non-negative.")
        eig_floor_abs = float(noise_cfg.get("score_noise_eig_floor_abs", 0.0))
        eig_floor_rel = float(noise_cfg.get("score_noise_eig_floor_rel", 1.0e-12))
        max_count = max(grid)
        trial_rows: list[dict[str, Any]] = []
        trial_posterior_rows: list[dict[str, Any]] = []
        synthesis_rows: list[dict[str, Any]] = []
        information_rows: list[dict[str, Any]] = []
        for trial_index in range(n_trials):
            rng = np.random.default_rng(make_subseed(seed, f"{case_name}.{mode}.{trial_index}"))
            sequence = synthesize_score_noise_summaries(
                summaries,
                n_subblocks=max_count,
                theta_true=theta_true,
                rng=rng,
                alpha=alpha,
                eig_floor_abs=eig_floor_abs,
                eig_floor_rel=eig_floor_rel,
            )
            synthesis_rows.append(
                {
                    "case_name": case_name,
                    "forecast_mode": mode,
                    "trial_index": int(trial_index),
                    "seed": int(seed),
                    "n_generated_summaries": int(len(sequence)),
                    "score_noise_alpha": float(alpha),
                    "score_noise_eig_floor_abs": float(eig_floor_abs),
                    "score_noise_eig_floor_rel": float(eig_floor_rel),
                }
            )
            for count in grid:
                prefix = sequence[:count]
                update = update_observation_belief(prior, prefix)
                accumulated = accumulate_summary_information(labels, prefix)
                row = build_forecast_science_row(
                    case_name=case_name,
                    forecast_mode=mode,
                    labels=labels,
                    update_result=update,
                    accumulated_information=accumulated,
                    prior_sigma=prior_sigma,
                    theta_true=theta_true,
                    n_subblocks=count,
                    actual_n_summaries=len(summaries),
                    subblock_duration_s=subblock_duration_s,
                    single_observation_n_subblocks=single_observation_n_subblocks,
                    prior_sigma_source=prior_sigma_source,
                    n_trials=n_trials,
                )
                row["trial_index"] = int(trial_index)
                trial_rows.append(row)
                trial_posterior_rows.extend(
                    build_forecast_posterior_rows(
                        case_name=case_name,
                        forecast_mode=mode,
                        labels=labels,
                        update_result=update,
                        theta_true=theta_true,
                        prior_sigma=prior_sigma,
                        n_subblocks=count,
                        trial_index=trial_index,
                    )
                )
                if trial_index == 0:
                    information_rows.append(
                        build_forecast_information_row(
                            case_name=case_name,
                            forecast_mode=mode,
                            labels=labels,
                            summaries=prefix,
                            n_subblocks=count,
                            actual_n_summaries=len(summaries),
                            subblock_duration_s=subblock_duration_s,
                            single_observation_n_subblocks=single_observation_n_subblocks,
                        )
                    )
        aggregate_rows: list[dict[str, Any]] = []
        for count in grid:
            subset = [row for row in trial_rows if int(row["n_subblocks"]) == int(count)]
            errors = np.asarray(
                [float(row["separation_posterior_error_microas"]) for row in subset],
                dtype=float,
            )
            sigmas = np.asarray(
                [float(row["separation_posterior_sigma_microas"]) for row in subset],
                dtype=float,
            )
            base = dict(subset[0])
            base.pop("trial_index", None)
            base.update(
                {
                    "n_trials": int(n_trials),
                    "separation_error_mean_microas": float(np.mean(errors)),
                    "separation_error_std_microas": float(np.std(errors, ddof=0)),
                    "separation_error_rms_microas": float(np.sqrt(np.mean(np.square(errors)))),
                    "separation_error_p16_microas": float(np.percentile(errors, 16)),
                    "separation_error_p50_microas": float(np.percentile(errors, 50)),
                    "separation_error_p84_microas": float(np.percentile(errors, 84)),
                    "separation_sigma_mean_microas": float(np.mean(sigmas)),
                    "separation_sigma_p16_microas": float(np.percentile(sigmas, 16)),
                    "separation_sigma_p50_microas": float(np.percentile(sigmas, 50)),
                    "separation_sigma_p84_microas": float(np.percentile(sigmas, 84)),
                }
            )
            aggregate_rows.append(base)
        _write_csv_rows(mode_root / "forecast_results.csv", aggregate_rows)
        _write_csv_rows(mode_root / "trial_forecast_results.csv", trial_rows)
        _write_csv_rows(mode_root / "trial_posterior_by_label.csv", trial_posterior_rows)
        _write_csv_rows(mode_root / "information_diagnostics.csv", information_rows)
        _write_csv_rows(mode_root / "stochastic_synthesis_diagnostics.csv", synthesis_rows)
        _write_forecast_manifest(
            path=mode_root / "manifest.json",
            case_name=case_name,
            mode=mode,
            summaries=summaries,
            summary_paths=summary_paths,
            prior=prior,
            prior_sigma=prior_sigma,
            theta_true=theta_true,
            forecast_cfg=cfg,
            grid=grid,
            extra={
                "score_noise_settings": dict(noise_cfg),
                "theta_reference": np.asarray(theta_reference, dtype=float).tolist(),
            },
        )
        if bool(cfg.get("plots", True)):
            _plot_forecast_line(
                path=mode_root / "separation_sigma_vs_n_subblocks.png",
                rows=aggregate_rows,
                y_key="separation_sigma_p50_microas",
                ylabel="Median separation sigma (microas)",
                actual_n_summaries=len(summaries),
                single_observation_n_subblocks=single_observation_n_subblocks,
            )
            _plot_forecast_line(
                path=mode_root / "separation_error_vs_n_subblocks.png",
                rows=aggregate_rows,
                y_key="separation_error_rms_microas",
                ylabel="Separation error RMS (microas)",
                actual_n_summaries=len(summaries),
                single_observation_n_subblocks=single_observation_n_subblocks,
            )
        all_results.extend(aggregate_rows)
        all_trial_results.extend(trial_rows)
        all_trial_posterior_rows.extend(trial_posterior_rows)
        all_information_rows.extend(information_rows)

    summary = {
        "enabled": True,
        "case_name": case_name,
        "forecast_grid": list(grid),
        "modes": list(modes),
        "forecast_root": str(forecast_root),
        "n_actual_summaries": int(len(summaries)),
        "single_observation_n_subblocks": int(single_observation_n_subblocks),
        "subblock_duration_s": float(subblock_duration_s),
    }
    _write_json(forecast_root / "forecast_summary.json", summary)
    return {
        "forecast_results": all_results,
        "forecast_trial_results": all_trial_results,
        "forecast_trial_posterior_rows": all_trial_posterior_rows,
        "forecast_information_diagnostics": all_information_rows,
        "forecast_summary": summary,
    }


def aggregate_case(
    *,
    plan: CampaignPlan,
    case: BiasCase,
    prior_source: str,
) -> dict[str, Any]:
    case_root = plan.run_root / "cases" / case.case_name
    _ensure_dir(case_root)
    summary_paths = plan.summary_paths[case.case_name]
    summaries = _load_case_summaries(summary_paths)
    summary_reference_mean = np.asarray(summaries[0].theta_ref, dtype=float)
    use_case_prior = bool(case.prior_sigma_by_label)
    if use_case_prior:
        prior_mean = summary_reference_mean.copy()
        prior_mean_source = (
            "prior_draw_theta_ref"
            if case.case_origin == "prior_draw"
            else "case_specific_theta_ref"
        )
        prior_mean_provenance = {
            "policy": "case_specific_prior",
            "case_origin": case.case_origin,
            "summary_paths": [str(path) for path in summary_paths],
        }
        if case.prior_draw_metadata is not None:
            prior_mean_provenance["prior_draw_metadata"] = dict(case.prior_draw_metadata)
        prior_warnings: list[str] = []
    else:
        prior_context = resolve_prior_context_for_summaries(
            summaries,
            summary_paths=summary_paths,
            prior_source=prior_source,
            allow_summary_theta_ref_default=True,
        )
        prior_mean = np.asarray(prior_context.prior_mean, dtype=float)
        prior_mean_source = prior_context.prior_mean_source
        prior_mean_provenance = dict(prior_context.provenance)
        prior_warnings = list(prior_context.warnings)
    if case.prior_sigma_by_label:
        sigma_defaults = build_default_prior_sigma(plan.layout.labels)
        prior_sigma = np.asarray(sigma_defaults, dtype=float)
        for i, label in enumerate(plan.layout.labels):
            if label in case.prior_sigma_by_label:
                prior_sigma[i] = float(case.prior_sigma_by_label[label])
        if np.any(prior_sigma <= 0.0) or not np.all(np.isfinite(prior_sigma)):
            raise ValueError(
                f"Case {case.case_name!r} resolved non-positive or non-finite prior sigma."
            )
        prior_sigma_source = (
            "prior_draw_config" if case.case_origin == "prior_draw" else "case_prior_sigmas"
        )
    else:
        prior_sigma = build_default_prior_sigma(plan.layout.labels)
        prior_sigma_source = "build_default_prior_sigma"
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=plan.layout.labels,
        mean=prior_mean,
        sigma=prior_sigma,
        metadata={
            "prior_mean_source": prior_mean_source,
            "prior_mean_provenance": prior_mean_provenance,
            "prior_sigma_source": prior_sigma_source,
            "case_origin": case.case_origin,
        },
    )
    update = update_observation_belief(prior, summaries)
    accumulated = accumulate_summary_information(plan.layout.labels, summaries)
    posterior_sigma = update.posterior.sigma()
    truth, reference, offsets = _truth_reference_maps(
        plan=plan,
        case=case,
        summaries=summaries,
        summary_paths=summary_paths,
    )
    posterior_rows = _posterior_rows(
        case_name=case.case_name,
        labels=plan.layout.labels,
        truth=truth,
        reference=reference,
        offsets=offsets,
        posterior_mean=update.posterior.mean,
        posterior_sigma=posterior_sigma,
    )
    science = _science_row(
        case_name=case.case_name,
        labels=plan.layout.labels,
        truth=truth,
        reference=reference,
        posterior_mean=update.posterior.mean,
        posterior_sigma=posterior_sigma,
    )
    _write_csv_rows(
        case_root / "summary_paths.csv",
        [{"case_name": case.case_name, "summary_path": str(path)} for path in summary_paths],
    )
    _write_csv_rows(case_root / "posterior_by_label.csv", posterior_rows)
    _write_csv_rows(case_root / "science_summary.csv", [science])
    eigen_cfg = dict(plan.config["experiment"].get("eigenbasis", {}) or {})
    eigen_sources = _resolve_eigen_sources(eigen_cfg)
    eigenvalue_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    contributor_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    projection_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    weak_rows_by_source: dict[str, list[dict[str, Any]]] = {}
    eigen_summary_by_source: dict[str, dict[str, Any]] = {}
    if bool(eigen_cfg.get("enabled", True)):
        reference_error = np.asarray([reference[label] - truth[label] for label in plan.layout.labels], dtype=float)
        posterior_error = np.asarray([update.posterior.mean[index] - truth[label] for index, label in enumerate(plan.layout.labels)], dtype=float)
        posterior_shift = np.asarray([update.posterior.mean[index] - reference[label] for index, label in enumerate(plan.layout.labels)], dtype=float)
        for source_name in eigen_sources:
            matrix = (
                update.posterior.precision
                if source_name == "posterior_precision"
                else accumulated
            )
            (
                eigen_rows,
                contributor_rows,
                projection_rows,
                weak_rows,
                eigen_summary,
                _,
            ) = _eigen_rows(
                case_name=case.case_name,
                source_name=source_name,
                labels=plan.layout.labels,
                matrix=matrix,
                prior_sigma=prior_sigma,
                reference_error=reference_error,
                posterior_error=posterior_error,
                posterior_shift=posterior_shift,
                eigen_cfg=eigen_cfg,
            )
            eigenvalue_rows_by_source[source_name] = eigen_rows
            contributor_rows_by_source[source_name] = contributor_rows
            projection_rows_by_source[source_name] = projection_rows
            weak_rows_by_source[source_name] = weak_rows
            eigen_summary_by_source[source_name] = eigen_summary
            if len(eigen_sources) == 1:
                _write_csv_rows(case_root / "eigenvalues.csv", eigen_rows)
                _write_csv_rows(case_root / "eigenmode_contributors.csv", contributor_rows)
                _write_csv_rows(case_root / "eigenmode_projection.csv", projection_rows)
                _write_csv_rows(case_root / "weak_mode_summary.csv", weak_rows)
                _write_json(case_root / "eigenbasis_summary.json", eigen_summary)
            _write_csv_rows(
                case_root / f"eigenvalues_{source_name}.csv",
                eigen_rows,
            )
            _write_csv_rows(
                case_root / f"eigenmode_contributors_{source_name}.csv",
                contributor_rows,
            )
            _write_csv_rows(
                case_root / f"eigenmode_projection_{source_name}.csv",
                projection_rows,
            )
            _write_csv_rows(
                case_root / f"weak_mode_summary_{source_name}.csv",
                weak_rows,
            )
            _write_json(
                case_root / f"eigenbasis_summary_{source_name}.json",
                eigen_summary,
            )
            _plot_eigenvalue_spectrum(
                case_root / f"eigenvalue_spectrum_{source_name}.png",
                eigen_rows,
            )
        _plot_group_error(case_root / f"posterior_error_over_sigma_by_group_{_slugify(case.case_name)}.png", posterior_rows)
    matrix_diagnostics = {
        "accumulated_information": _matrix_diagnostics(accumulated).to_dict(),
        "posterior_precision": _matrix_diagnostics(update.posterior.precision).to_dict(),
        "prior_precision": _matrix_diagnostics(prior.precision).to_dict(),
    }
    _write_json(case_root / "matrix_diagnostics.json", matrix_diagnostics)
    update_summary = {
        "case_name": case.case_name,
        "n_summaries": len(summaries),
        "prior": prior.to_dict(),
        "posterior": update.posterior.to_dict(),
        "information_vector": update.information_vector.tolist(),
        "prior_context": {
            "prior_mean_source": prior_mean_source,
            "provenance": prior_mean_provenance,
            "warnings": prior_warnings,
            "prior_sigma_source": prior_sigma_source,
        },
    }
    _write_json(case_root / "observation_update_summary.json", update_summary)
    case_manifest = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "case_name": case.case_name,
        "case_root": str(case_root),
        "case_origin": case.case_origin,
        "seeding": _resolve_seeding_config(plan.config["experiment"]),
        "case_generation": dict(plan.case_generation),
        "subblock_command_options": resolve_subblock_command_options(
            plan.config["experiment"].get("subblocks", {}) or {}
        ),
        "theta_reference_offsets": dict(case.theta_reference_offsets),
        "case_prior_sigma_by_label": (
            None if case.prior_sigma_by_label is None else dict(case.prior_sigma_by_label)
        ),
        "prior_draw_metadata": (
            None if case.prior_draw_metadata is None else dict(case.prior_draw_metadata)
        ),
        "prior_draw_rows": list(plan.prior_draw_rows_by_case.get(case.case_name, [])),
        "theta_layout": plan.layout.to_dict(),
        "layout_metadata": plan.layout_metadata,
        "state_partition": plan.partition,
        "summary_paths": [str(path) for path in summary_paths],
        "subblock_plan": list(plan.subblock_plans.get(case.case_name, [])),
        "planned_commands": [" ".join(command) for command in plan.subblock_commands[case.case_name]],
        "outputs": {
            "posterior_by_label_csv": str((case_root / "posterior_by_label.csv").resolve()),
            "science_summary_csv": str((case_root / "science_summary.csv").resolve()),
            "eigen_sources": list(eigen_sources),
            "subblock_plan_csv": str((case_root / "subblock_plan.csv").resolve()),
        },
    }
    _write_csv_rows(
        case_root / "subblock_plan.csv",
        list(plan.subblock_plans.get(case.case_name, [])),
    )
    _write_csv_rows(
        case_root / "prior_draws.csv",
        list(plan.prior_draw_rows_by_case.get(case.case_name, [])),
    )
    _write_json(case_root / "case_manifest.json", case_manifest)
    theta_true_vector = np.asarray([truth[label] for label in plan.layout.labels], dtype=float)
    theta_reference_vector = np.asarray(
        [reference[label] for label in plan.layout.labels],
        dtype=float,
    )
    forecast_payload = run_case_forecast(
        case_name=case.case_name,
        case_root=case_root,
        summaries=summaries,
        summary_paths=summary_paths,
        prior=prior,
        prior_sigma=prior_sigma,
        theta_true=theta_true_vector,
        theta_reference=theta_reference_vector,
        labels=plan.layout.labels,
        forecast_cfg=plan.config["experiment"].get("forecast", {}) or {},
    )
    return {
        "case_name": case.case_name,
        "case_root": str(case_root),
        "posterior_rows": posterior_rows,
        "science": science,
        "eigen_sources": list(eigen_sources),
        "eigenvalues_by_source": eigenvalue_rows_by_source,
        "eigenmode_contributors_by_source": contributor_rows_by_source,
        "eigenmode_projection_by_source": projection_rows_by_source,
        "weak_mode_summary_by_source": weak_rows_by_source,
        "matrix_diagnostics": matrix_diagnostics,
        "eigenbasis_summary_by_source": eigen_summary_by_source,
        "forecast_results": forecast_payload["forecast_results"],
        "forecast_trial_results": forecast_payload["forecast_trial_results"],
        "forecast_information_diagnostics": forecast_payload[
            "forecast_information_diagnostics"
        ],
        "forecast_summary": forecast_payload["forecast_summary"],
    }


def _plot_campaign_bar(path: Path, rows: Sequence[Mapping[str, Any]], *, key: str, ylabel: str) -> None:
    if not _HAVE_MATPLOTLIB or plt is None:
        return
    if not rows:
        return
    names = [str(row["case_name"]) for row in rows]
    values = [float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(names)), 5))
    ax.bar(names, values)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _campaign_forecast_summary(
    *,
    forecast_rows: Sequence[Mapping[str, Any]],
    forecast_trial_rows: Sequence[Mapping[str, Any]],
    forecast_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = _forecast_config({"forecast": forecast_cfg})
    target = int(cfg.get("single_observation_n_subblocks", 1800))
    target_rows = [row for row in forecast_rows if int(row.get("n_subblocks", -1)) == target]
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "single_observation_n_subblocks": int(target),
        "subblock_duration_s": float(cfg.get("subblock_duration_s", 1.0)),
        "single_observation_target_rows": [
            {
                "case_name": row.get("case_name"),
                "forecast_mode": row.get("forecast_mode"),
                "n_subblocks": int(row.get("n_subblocks", target)),
                "equivalent_duration_min": float(row.get("equivalent_duration_min", 0.0)),
                "n_actual_summaries": int(row.get("n_actual_summaries", 0)),
                "separation_posterior_sigma_microas": row.get(
                    "separation_posterior_sigma_microas"
                ),
                "separation_posterior_error_microas": row.get(
                    "separation_posterior_error_microas"
                ),
                "separation_error_rms_microas": row.get(
                    "separation_error_rms_microas"
                ),
                "prior_sigma_source": row.get("prior_sigma_source"),
            }
            for row in target_rows
        ],
        "n_forecast_rows": int(len(forecast_rows)),
        "n_trial_rows": int(len(forecast_trial_rows)),
    }


def _plot_forecast_campaign_bars(
    *,
    run_root: Path,
    forecast_rows: Sequence[Mapping[str, Any]],
    forecast_cfg: Mapping[str, Any],
) -> None:
    if not _HAVE_MATPLOTLIB or plt is None or not forecast_rows:
        return
    cfg = _forecast_config({"forecast": forecast_cfg})
    target = int(cfg.get("single_observation_n_subblocks", 1800))
    rows = [row for row in forecast_rows if int(row.get("n_subblocks", -1)) == target]
    if not rows:
        return

    def _bar(path: Path, key_candidates: Sequence[str], ylabel: str) -> None:
        labels: list[str] = []
        values: list[float] = []
        for row in rows:
            key = next((candidate for candidate in key_candidates if candidate in row), None)
            if key is None:
                continue
            value = row.get(key)
            if value is None:
                continue
            labels.append(f"{row['case_name']}\n{row['forecast_mode']}")
            values.append(float(value))
        if not values:
            return
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(values)), 5))
        ax.bar(labels, values)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    _bar(
        run_root / "separation_sigma_forecast_by_case.png",
        ("separation_sigma_p50_microas", "separation_posterior_sigma_microas"),
        "Separation sigma at target (microas)",
    )
    _bar(
        run_root / "separation_error_forecast_by_case.png",
        ("separation_error_rms_microas", "separation_posterior_error_microas"),
        "Separation error at target (microas)",
    )


def aggregate_campaign(
    plan: CampaignPlan,
    *,
    prior_source: str,
) -> dict[str, Any]:
    all_posterior_rows: list[dict[str, Any]] = []
    all_science_rows: list[dict[str, Any]] = []
    all_eigenvalues_by_source: dict[str, list[dict[str, Any]]] = {}
    all_contributors_by_source: dict[str, list[dict[str, Any]]] = {}
    all_projection_by_source: dict[str, list[dict[str, Any]]] = {}
    all_weak_by_source: dict[str, list[dict[str, Any]]] = {}
    all_forecast_rows: list[dict[str, Any]] = []
    all_forecast_trial_rows: list[dict[str, Any]] = []
    all_forecast_information_rows: list[dict[str, Any]] = []
    all_subblock_plan_rows = _flatten_subblock_plan_rows(plan)
    case_payloads: list[dict[str, Any]] = []
    for case in plan.cases:
        payload = aggregate_case(plan=plan, case=case, prior_source=prior_source)
        case_payloads.append(payload)
        all_posterior_rows.extend(payload["posterior_rows"])
        all_science_rows.append(payload["science"])
        for source in payload["eigen_sources"]:
            all_eigenvalues_by_source.setdefault(source, []).extend(
                payload["eigenvalues_by_source"].get(source, [])
            )
            all_contributors_by_source.setdefault(source, []).extend(
                payload["eigenmode_contributors_by_source"].get(source, [])
            )
            all_projection_by_source.setdefault(source, []).extend(
                payload["eigenmode_projection_by_source"].get(source, [])
            )
            all_weak_by_source.setdefault(source, []).extend(
                payload["weak_mode_summary_by_source"].get(source, [])
            )
        all_forecast_rows.extend(payload.get("forecast_results", []))
        all_forecast_trial_rows.extend(payload.get("forecast_trial_results", []))
        all_forecast_information_rows.extend(
            payload.get("forecast_information_diagnostics", [])
        )
    _write_csv_rows(plan.run_root / "posterior_by_label.csv", all_posterior_rows)
    _write_csv_rows(plan.run_root / "science_summary.csv", all_science_rows)
    _write_csv_rows(plan.run_root / "subblock_plan.csv", all_subblock_plan_rows)
    _write_csv_rows(plan.run_root / "forecast_results.csv", all_forecast_rows)
    _write_csv_rows(plan.run_root / "forecast_trial_results.csv", all_forecast_trial_rows)
    _write_csv_rows(
        plan.run_root / "forecast_information_diagnostics.csv",
        all_forecast_information_rows,
    )
    sources = _resolve_eigen_sources(plan.config["experiment"].get("eigenbasis", {}) or {})
    for source in sources:
        eigen_rows = all_eigenvalues_by_source.get(source, [])
        contributor_rows = all_contributors_by_source.get(source, [])
        projection_rows = all_projection_by_source.get(source, [])
        weak_rows = all_weak_by_source.get(source, [])
        _write_csv_rows(plan.run_root / f"eigenvalues_{source}.csv", eigen_rows)
        _write_csv_rows(
            plan.run_root / f"eigenmode_contributors_{source}.csv",
            contributor_rows,
        )
        _write_csv_rows(
            plan.run_root / f"eigenmode_projection_{source}.csv",
            projection_rows,
        )
        _write_csv_rows(plan.run_root / f"weak_mode_summary_{source}.csv", weak_rows)
        if len(sources) == 1:
            _write_csv_rows(plan.run_root / "eigenvalues.csv", eigen_rows)
            _write_csv_rows(
                plan.run_root / "eigenmode_contributors.csv",
                contributor_rows,
            )
            _write_csv_rows(
                plan.run_root / "eigenmode_projection.csv",
                projection_rows,
            )
            _write_csv_rows(plan.run_root / "weak_mode_summary.csv", weak_rows)
    _plot_campaign_bar(
        plan.run_root / "separation_error_by_case.png",
        all_science_rows,
        key="posterior_separation_error_microas",
        ylabel="Posterior separation error (microas)",
    )
    _plot_campaign_bar(
        plan.run_root / "separation_sigma_by_case.png",
        all_science_rows,
        key="posterior_separation_sigma_microas",
        ylabel="Posterior separation sigma (microas)",
    )
    _plot_forecast_campaign_bars(
        run_root=plan.run_root,
        forecast_rows=all_forecast_rows,
        forecast_cfg=plan.config["experiment"].get("forecast", {}) or {},
    )
    forecast_summary = _campaign_forecast_summary(
        forecast_rows=all_forecast_rows,
        forecast_trial_rows=all_forecast_trial_rows,
        forecast_cfg=plan.config["experiment"].get("forecast", {}) or {},
    )
    _write_json(plan.run_root / "forecast_summary.json", forecast_summary)
    summary = {
        "schema_version": CAMPAIGN_SCHEMA_VERSION,
        "run_root": str(plan.run_root),
        "n_cases": len(plan.cases),
        "n_theta": plan.layout.size,
        "eigen_sources": list(sources),
        "forecast": forecast_summary,
        "case_outputs": [
            {
                "case_name": payload["case_name"],
                "case_root": payload["case_root"],
                "science": payload["science"],
            }
            for payload in case_payloads
        ],
    }
    _write_json(plan.run_root / "campaign_summary.json", summary)
    return summary


def run_observation_bias_campaign(
    *,
    config_path: Path | None,
    results_root: Path,
    run_name: str | None = None,
    dry_run: bool = False,
    aggregate_only: bool = False,
    resume: bool = False,
    max_workers: int = 1,
    fail_fast: bool = True,
    quiet: bool = False,
    system_preset: str | None = None,
    prior_source: str = "summary_theta_ref",
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    plan = build_campaign_plan(
        config_path=config_path,
        results_root=results_root,
        run_name=run_name,
        system_preset=system_preset,
        args=args,
    )
    _ensure_dir(plan.run_root)
    _write_json(plan.run_root / "campaign_plan.json", _plan_payload(plan))
    _write_json(plan.run_root / "resolved_config.json", plan.config)
    _write_csv_rows(plan.run_root / "subblock_plan.csv", _flatten_subblock_plan_rows(plan))
    _write_csv_rows(plan.run_root / "bias_cases.csv", _bias_case_rows(plan))
    _write_csv_rows(plan.run_root / "prior_draws.csv", _flatten_prior_draw_rows(plan))
    if dry_run:
        payload = _plan_payload(plan)
        if not quiet:
            print(json.dumps(payload, indent=2), flush=True)
        return payload
    if not aggregate_only:
        execute_subblocks(
            plan,
            resume=resume,
            max_workers=max(1, int(max_workers)),
            fail_fast=fail_fast,
            quiet=quiet,
        )
    return aggregate_campaign(plan, prior_source=prior_source)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an observation-level Zernike bias campaign.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--n-subblocks", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--noise", choices=("inherit", "enabled", "disabled"), default=None)
    parser.add_argument("--phi-ref", choices=("truth_when_available", "recovered"), default=None)
    parser.add_argument("--max-dense-dim", type=int, default=None)
    parser.add_argument("--schur-curvature-method", choices=("auto", "dense", "structured_independent_frames"), default=None)
    parser.add_argument("--seed-policy", choices=SUPPORTED_SEED_POLICIES, default=None)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--system-preset", default=None)
    parser.add_argument(
        "--prior-source",
        choices=("auto", "summary_theta_ref", "resolved_system", "default_system"),
        default="summary_theta_ref",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    run_observation_bias_campaign(
        config_path=args.config,
        results_root=args.results_root,
        run_name=args.run_name,
        dry_run=bool(args.dry_run),
        aggregate_only=bool(args.aggregate_only),
        resume=bool(args.resume),
        max_workers=int(args.max_workers),
        fail_fast=bool(args.fail_fast),
        quiet=bool(args.quiet),
        system_preset=args.system_preset,
        prior_source=str(args.prior_source),
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
