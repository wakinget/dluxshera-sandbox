"""Run an image-backed single-star calibration observation demo.

The demo observes a centered ``single_star`` calibration target, solves each
short sub-block for frame-local registration only, and accumulates Schur-reduced
information about slow calibration parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    build_prior_mean_from_store,
)
from dluxshera.inference.observation_summary import load_subblock_summary
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.noise import make_subseed
from dluxshera.utils.obs_subblock_io import now_iso_local_ms, timestamp_tag
from dluxshera.utils.obs_subblock_keys import parse_obs_subblock_key_address
from dluxshera.utils.subprocess_diagnostics import run_subprocess_with_diagnostics
from dluxshera.utils.single_star_calibration import (
    ALPHA_CEN_A_PLACEHOLDER_NOTE,
    prepare_alpha_cen_a_single_star_system_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBBLOCK_SCRIPT = REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_study.py"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "single_star_calibration_demo"
SCHEMA_VERSION = "single_star_calibration_demo.v1"
ACTIVE_FRAME_KEYS = (
    "source.x_position_as",
    "source.y_position_as",
)
SUPPORTED_SEED_POLICIES = (
    "different_jitter_different_noise",
    "same_jitter_different_noise",
    "different_jitter_same_noise",
)
DEFAULT_SINGLE_STAR_SCHUR_METHOD = "structured_independent_frames"


@dataclass(frozen=True)
class CalibrationCase:
    case_name: str
    theta_reference_offsets: dict[str, float]
    case_origin: str
    prior_sigma_by_label: dict[str, float] | None = None


@dataclass(frozen=True)
class CalibrationPlan:
    run_root: Path
    layout: ObservationThetaLayout
    layout_metadata: dict[str, Any]
    truth_vector: np.ndarray
    system_cfg: dict[str, Any]
    config: dict[str, Any]
    cases: tuple[CalibrationCase, ...]
    summary_paths: dict[str, list[Path]]
    subblock_commands: dict[str, list[list[str]]]
    subblock_rows: list[dict[str, Any]]
    prior_draw_rows: list[dict[str, Any]]
    status_rows: list[dict[str, Any]]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _ensure_dir(path.parent)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _default_experiment_config() -> dict[str, Any]:
    return {
        "kind": "single_star_calibration_demo",
        "seed": 42,
        "run_name": "single_star_calibration_smoke",
        "calibration_source": {
            "mode": "alpha_cen_a_placeholder",
            "source_kind": "single_star",
            "x_position_as": 0.0,
            "y_position_as": 0.0,
            "position_angle_deg": 0.0,
            "photometry_note": ALPHA_CEN_A_PLACEHOLDER_NOTE,
        },
        "subblocks": {
            "n_subblocks": 3,
            "n_frames": 20,
            "noise": "enabled",
            "phi_ref": "recovered",
            "schur_curvature_method": "structured_independent_frames",
            "max_dense_dim": 40,
            "schur_damping": 1.0e-8,
            "exposure_time_s": 0.05,
            "reference_diagnostics_profile": "basic",
            "reference_optimizer_kind": "sgd",
            "reference_base_lr": 0.7,
            "reference_n_iter": 80,
            "reference_schedule_kind": "linear_warmup",
            "reference_schedule_warmup_steps": 10,
            "reference_schedule_start_factor": 0.125,
            "trace_jitter": {
                "x_sigma_as": 1.0e-3,
                "y_sigma_as": 1.0e-3,
                "pa_sigma_deg": 1.0e-4,
                "pa_mode": "omitted",
            },
        },
        "seeding": {
            "seed_policy": "different_jitter_different_noise",
            "base_seed": 42,
        },
        "local_eliminated_keys": list(ACTIVE_FRAME_KEYS),
        "observation_theta": {
            "source": {"log_flux_total": True},
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
        "prior": {
            "sigma": {
                "source.log_flux_total": {
                    "kind": "absolute",
                    "sigma": 1.0e-5,
                    "unit": "log_flux",
                },
                "optics.plate_scale_as_per_pix": {
                    "kind": "fractional",
                    "sigma": 1.0e-5,
                },
                "optics.primary.zernike_coeffs_nm[*]": {
                    "kind": "absolute",
                    "sigma": 1.0,
                    "unit": "nm",
                },
                "optics.secondary.zernike_coeffs_nm[*]": {
                    "kind": "absolute",
                    "sigma": 1.0,
                    "unit": "nm",
                },
            }
        },
        "case_generation": {
            "mode": "prior_draw",
            "n_cases": 1,
            "seed": 123,
            "draw_scale": 0.5,
            "include_zero_bias_case": True,
        },
        "history_prefixes": [1, 2, 3, 5, 10, 30, 100, 300, 1000, 1800],
        "eigenbasis": {
            "enabled": True,
            "sources": ["accumulated_information", "posterior_precision"],
            "whiten": True,
            "eig_floor_abs": 0.0,
            "eig_floor_rel": 1.0e-12,
            "top_k_contributors": 8,
        },
        "forecast": {
            "enabled": True,
            "modes": ["replicate", "fixed_information_score_noise"],
            "n_subblocks_grid": [1, 3, 5, 10, 30, 100, 300, 1000, 1800],
            "subblock_duration_s": 1.0,
            "single_observation_n_subblocks": 1800,
            "fixed_information_score_noise": {
                "enabled": True,
                "n_trials": 100,
                "seed": 2026,
                "score_noise_alpha": 1.0,
                "score_noise_eig_floor_rel": 1.0e-12,
                "truth_mode": "campaign_truth",
            },
            "plots": True,
        },
    }


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"experiment": _default_experiment_config()}
    payload = load_config_file(path.resolve())
    experiment = payload.get("experiment", payload)
    if not isinstance(experiment, Mapping):
        raise ValueError("Calibration config must contain a mapping experiment block.")
    return {"experiment": dict(experiment)}


def _deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace | None) -> dict[str, Any]:
    out = _deepcopy_json(cfg)
    if args is None:
        return out
    if args.run_name is not None:
        out["run_name"] = args.run_name
    subblocks = dict(out.get("subblocks", {}) or {})
    for attr, key in (
        ("n_subblocks", "n_subblocks"),
        ("n_frames", "n_frames"),
        ("noise", "noise"),
        ("phi_ref", "phi_ref"),
        ("reference_diagnostics_profile", "reference_diagnostics_profile"),
        ("schur_curvature_method", "schur_curvature_method"),
        ("max_dense_dim", "max_dense_dim"),
    ):
        value = getattr(args, attr, None)
        if value is not None:
            subblocks[key] = value
    out["subblocks"] = subblocks
    theta = dict(out.get("observation_theta", {}) or {})
    source = dict(theta.get("source", {}) or {})
    optics = dict(theta.get("optics", {}) or {})
    if args.include_log_flux is not None:
        source["log_flux_total"] = bool(args.include_log_flux)
    if args.include_plate_scale is not None:
        optics["plate_scale_as_per_pix"] = bool(args.include_plate_scale)
    if args.zernike_indices is not None:
        indices = [int(part) for part in str(args.zernike_indices).split(",") if part.strip()]
        for key in ("primary_zernikes", "secondary_zernikes"):
            group = dict(optics.get(key, {}) or {})
            group["enabled"] = bool(indices)
            group["indices"] = indices
            optics[key] = group
    theta["source"] = source
    theta["optics"] = optics
    out["observation_theta"] = theta
    return out


def _schur_settings_for_single_star_config(
    subblock_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    requested = str(subblock_cfg.get("schur_curvature_method") or DEFAULT_SINGLE_STAR_SCHUR_METHOD)
    requested_norm = requested.strip().lower()
    route_source = "user_request" if subblock_cfg.get("schur_curvature_method") is not None else "single_star_default_structured"
    effective = requested
    if requested_norm == "auto":
        effective = DEFAULT_SINGLE_STAR_SCHUR_METHOD
        route_source = "auto_prefers_structured_independent_frames"
    dense_like = requested_norm == "dense"
    return {
        "schur_curvature_method_requested": requested,
        "schur_curvature_method_effective": effective,
        "schur_route_source": route_source,
        "max_dense_dim": int(subblock_cfg.get("max_dense_dim", 40)),
        "structured_curvature_expected": str(effective).startswith("structured_"),
        "validate_structured_against_dense": bool(subblock_cfg.get("validate_structured_against_dense", False)),
        "dense_route_requested": dense_like,
    }


def _resolve_single_star_system(
    *,
    experiment_cfg: Mapping[str, Any],
    system_preset: str | None,
) -> tuple[ParameterStore, dict[str, Any], dict[str, Any]]:
    preset = system_preset or str(experiment_cfg.get("system_preset", DEFAULT_SYSTEM_PRESET))
    base_resolved = resolve_config({"system": {"preset": preset}})
    base_system = base_resolved["system"]
    subblocks = dict(experiment_cfg.get("subblocks", {}) or {})
    system_cfg = prepare_alpha_cen_a_single_star_system_config(
        base_system,
        exposure_time_s=float(subblocks.get("exposure_time_s", 0.05)),
        n_lambda=int(experiment_cfg.get("calibration_source", {}).get("n_lambda", 11)),
    )
    source_overrides = dict(experiment_cfg.get("calibration_source", {}) or {})
    source_cfg = system_cfg["source"]
    for key in ("x_position_as", "y_position_as", "position_angle_deg"):
        if key in source_overrides:
            source_cfg[key] = float(source_overrides[key])
    spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    provenance = {
        "system_preset": preset,
        "source_kind": "single_star",
        "calibration_source_mode": "alpha_cen_a_placeholder",
        "photometry_source": "ALPHA_CEN component A placeholder",
        "photometry_is_placeholder": True,
        "calibration_star_registry_used": False,
        "photometry_note": ALPHA_CEN_A_PLACEHOLDER_NOTE,
    }
    return store, system_cfg, provenance


def _parameter_unit(label: str) -> str:
    if label == "source.log_flux_total":
        return "log flux"
    if label == "optics.plate_scale_as_per_pix":
        return "arcsec / pixel"
    if "zernike_coeffs_nm" in label:
        return "nm"
    return "arb"


def _parameter_group(label: str) -> str:
    if label == "source.log_flux_total":
        return "source.log_flux_total"
    if label == "optics.plate_scale_as_per_pix":
        return "optics.plate_scale"
    if label.startswith("optics.primary.zernike_coeffs_nm"):
        return "M1 Zernike"
    if label.startswith("optics.secondary.zernike_coeffs_nm"):
        return "M2 Zernike"
    return "other"


def _safe_fraction(num: float, den: float) -> float:
    if not math.isfinite(den) or abs(den) <= 1.0e-30:
        return float("nan")
    return float(num / den)


def select_active_truth_comparison_keys(
    csv_columns: Sequence[str],
    *,
    active_frame_keys: Sequence[str] = ACTIVE_FRAME_KEYS,
) -> list[str]:
    """Return solved frame keys present in a truth-comparison CSV schema."""

    columns = set(str(name) for name in csv_columns)
    selected: list[str] = []
    for key in active_frame_keys:
        if (
            f"{key}_truth" in columns
            and f"{key}_recovered" in columns
            and f"{key}_residual" in columns
        ):
            selected.append(str(key))
    return selected


def _match_rule(rule: str, label: str) -> bool:
    if "[*]" not in rule:
        return rule == label
    prefix = rule.replace("[*]", "[")
    return label.startswith(prefix) and label.endswith("]")


def resolve_prior_sigmas(
    labels: Sequence[str],
    truth_by_label: Mapping[str, float],
    sigma_cfg: Mapping[str, Any],
) -> dict[str, float]:
    """Resolve absolute/fractional prior sigma rules for calibration labels."""

    out: dict[str, float] = {}
    for raw_rule, raw_value in sigma_cfg.items():
        if not isinstance(raw_value, Mapping):
            raise ValueError(f"prior.sigma.{raw_rule} must be a mapping.")
        kind = str(raw_value.get("kind", "absolute"))
        configured = float(raw_value.get("sigma", 0.0))
        if configured <= 0.0:
            raise ValueError(f"prior sigma rule {raw_rule!r} must be positive.")
        for label in labels:
            if not _match_rule(str(raw_rule), label):
                continue
            if kind == "absolute":
                sigma = configured
            elif kind == "fractional":
                sigma = abs(float(truth_by_label[label])) * configured
            else:
                raise ValueError(f"Unsupported prior sigma kind: {kind}")
            if sigma <= 0.0 or not math.isfinite(sigma):
                raise ValueError(f"Resolved prior sigma for {label!r} is invalid.")
            out[label] = float(sigma)
    missing = [label for label in labels if label not in out]
    if missing:
        raise ValueError("Missing prior sigma rules for: " + ", ".join(missing))
    return out


def generate_calibration_cases(
    *,
    experiment_cfg: Mapping[str, Any],
    labels: Sequence[str],
    truth_by_label: Mapping[str, float],
) -> tuple[tuple[CalibrationCase, ...], list[dict[str, Any]]]:
    """Generate zero-bias, explicit, or prior-draw calibration cases."""

    prior_sigma = resolve_prior_sigmas(
        labels,
        truth_by_label,
        dict(experiment_cfg.get("prior", {}).get("sigma", {}) or {}),
    )
    cfg = dict(experiment_cfg.get("case_generation", {}) or {})
    mode = str(cfg.get("mode", "prior_draw"))
    cases: list[CalibrationCase] = []
    draw_rows: list[dict[str, Any]] = []
    if bool(cfg.get("include_zero_bias_case", mode == "zero_bias")):
        cases.append(
            CalibrationCase(
                "zero_bias",
                {},
                "zero_bias",
                prior_sigma_by_label={label: prior_sigma[label] for label in labels},
            )
        )
    if mode == "zero_bias":
        return tuple(cases), draw_rows
    if mode == "explicit":
        for item in cfg.get("cases", []) or []:
            name = str(item.get("case_name", "")).strip()
            if not name:
                raise ValueError("Explicit calibration cases require case_name.")
            offsets = {}
            for raw_label, raw_offset in (item.get("theta_reference_offsets", {}) or {}).items():
                label = parse_obs_subblock_key_address(str(raw_label)).canonical
                if label not in labels:
                    raise ValueError(f"Explicit case references non-theta label {label!r}.")
                offsets[label] = float(raw_offset)
            cases.append(
                CalibrationCase(
                    name,
                    offsets,
                    "explicit",
                    prior_sigma_by_label={label: prior_sigma[label] for label in labels},
                )
            )
        return tuple(cases), draw_rows
    if mode != "prior_draw":
        raise ValueError("case_generation.mode must be prior_draw, explicit, or zero_bias.")
    rng = np.random.default_rng(int(cfg.get("seed", 42)))
    draw_scale = float(cfg.get("draw_scale", 1.0))
    n_cases = int(cfg.get("n_cases", 1))
    for draw_index in range(n_cases):
        z = rng.normal(size=len(labels))
        offsets = {
            label: float(z[index] * prior_sigma[label] * draw_scale)
            for index, label in enumerate(labels)
        }
        name = f"prior_draw_{draw_index:03d}"
        cases.append(
            CalibrationCase(
                name,
                offsets,
                "prior_draw",
                prior_sigma_by_label={label: prior_sigma[label] for label in labels},
            )
        )
        for index, label in enumerate(labels):
            draw_rows.append(
                {
                    "case_name": name,
                    "theta_label": label,
                    "truth_value": float(truth_by_label[label]),
                    "prior_mean": float(truth_by_label[label] + offsets[label]),
                    "reference_value": float(truth_by_label[label] + offsets[label]),
                    "prior_sigma": float(prior_sigma[label]),
                    "draw_z": float(z[index]),
                    "draw_scale": draw_scale,
                    "theta_reference_offset": float(offsets[label]),
                    "unit": _parameter_unit(label),
                    "draw_seed": int(cfg.get("seed", 42)),
                    "draw_index": int(draw_index),
                }
            )
    return tuple(cases), draw_rows


def _derive_subblock_seeds(
    *,
    run_name: str,
    case_name: str,
    subblock_index: int,
    seed_policy: str,
    base_seed: int,
) -> dict[str, int]:
    token = f"{run_name}.{case_name}.subblock_{subblock_index:03d}"
    if seed_policy == "same_jitter_different_noise":
        trace_seed = make_subseed(base_seed, f"{run_name}.{case_name}.shared_trace")
        noise_seed = make_subseed(base_seed, f"{token}.noise")
    elif seed_policy == "different_jitter_same_noise":
        trace_seed = make_subseed(base_seed, f"{token}.trace")
        noise_seed = make_subseed(base_seed, f"{run_name}.{case_name}.shared_noise")
    else:
        trace_seed = make_subseed(base_seed, f"{token}.trace")
        noise_seed = make_subseed(base_seed, f"{token}.noise")
    return {"trace_seed": int(trace_seed), "noise_seed": int(noise_seed)}


def _trace_key_policy(experiment_cfg: Mapping[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    """Resolve single-star trace policy for optional inert PA diagnostics."""

    trace_cfg = dict(experiment_cfg.get("subblocks", {}).get("trace_jitter", {}) or {})
    pa_mode = str(trace_cfg.get("pa_mode", "omitted")).strip().lower()
    if pa_mode not in {"omitted", "inert_diagnostic"}:
        raise ValueError("subblocks.trace_jitter.pa_mode must be 'omitted' or 'inert_diagnostic'.")
    trace_keys = list(ACTIVE_FRAME_KEYS)
    inactive_truth_keys: list[str] = []
    if pa_mode == "inert_diagnostic":
        trace_keys.append("source.position_angle_deg")
        inactive_truth_keys.append("source.position_angle_deg")
    pa_policy = {
        "status": "inactive",
        "mode": pa_mode,
        "reason": (
            "Single-star PA is not solved; DP PSF orientation is treated as fixed "
            "instrument geometry in this calibration workflow."
        ),
    }
    return trace_keys, inactive_truth_keys, pa_policy


def _template_payloads(
    system_cfg: Mapping[str, Any],
    *,
    experiment_cfg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    trace_keys, _inactive_truth_keys, _pa_policy = _trace_key_policy(experiment_cfg)
    trace_plan: dict[str, Any] = {
        "source.x_position_as": {
            "base": 0.0,
            "effects": [{"kind": "iid_jitter", "center": 0.0, "sigma": 0.001}],
        },
        "source.y_position_as": {
            "base": 0.0,
            "effects": [{"kind": "iid_jitter", "center": 0.0, "sigma": 0.001}],
        },
    }
    if "source.position_angle_deg" in trace_keys:
        trace_plan["source.position_angle_deg"] = {
            "base": 0.0,
            "effects": [{"kind": "iid_jitter", "center": 0.0, "sigma": 0.0001}],
        }
    return {
        "trace": {
            "system": system_cfg,
            "experiment": {
                "kind": "subblock_trace_generation",
                "seed": 42,
                "trace": {
                    "n_frames": 20,
                    "dt_s": 0.05,
                    "varying_keys": list(trace_keys),
                    "plan": trace_plan,
                },
                "outputs": {"outdir": "trace", "file_prefix": "subblock_trace"},
            },
        },
        "render": {
            "system": system_cfg,
            "experiment": {
                "kind": "subblock_generation",
                "seed": 42,
                "subblock": {
                    "varying_keys": list(trace_keys),
                    "trace": {"format": "csv", "path": "frame_truth.csv"},
                },
                "noise": {"enabled": False, "photon_noise": True, "read_noise": False},
                "outputs": {"outdir": "render", "file_prefix": "obs_subblock"},
            },
        },
        "inference": {
            "system": system_cfg,
            "experiment": {
                "kind": "subblock_inference",
                "inference": {
                    "data": {
                        "cube": "cube.fits",
                        "truth_trace": "frame_truth.csv",
                        "manifest": "manifest.json",
                    },
                    "active": {
                        "frame_keys": list(ACTIVE_FRAME_KEYS),
                        "shared_keys": [],
                    },
                    "init": {
                        "frame": {
                            "mode": "shared_guess",
                            "values": {
                                "source.x_position_as": 0.0,
                                "source.y_position_as": 0.0,
                            },
                        },
                        "shared": {},
                    },
                    "priors": {"frame": {}, "shared": {}},
                    "temporal": {"frame_model": {"kind": "independent"}},
                    "objective": {
                        "kind": "nll",
                        "frame_reduce": "sum",
                        "subblock_reduce": "mean",
                        "noise_model": {
                            "kind": "gaussian",
                            "variance_model": "data",
                            "variance_floor": 1.0,
                        },
                    },
                    "optimizer": {
                        "kind": "sgd",
                        "base_lr": 0.9,
                        "n_iter": 100,
                        "preconditioning": {"enabled": True, "method": "auto", "reference": "initial"},
                    },
                    "diagnostics": {"plots": True, "compare_to_truth_when_available": True},
                }
            },
        },
    }


def write_single_star_templates(
    run_root: Path,
    system_cfg: Mapping[str, Any],
    *,
    experiment_cfg: Mapping[str, Any],
) -> dict[str, Path]:
    """Write run-local JSON templates consumed by ``run_obs_subblock_study.py``."""

    template_root = run_root / "templates"
    payloads = _template_payloads(system_cfg, experiment_cfg=experiment_cfg)
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = template_root / f"{name}_template.json"
        _write_json(path, payload)
        paths[name] = path
    return paths


def _theta_scalar_keys(labels: Sequence[str]) -> list[str]:
    return [
        label
        for label in labels
        if not label.startswith("optics.primary.zernike_coeffs_nm")
        and not label.startswith("optics.secondary.zernike_coeffs_nm")
    ]


def _zernike_indices_arg(layout_metadata: Mapping[str, Any]) -> str:
    primary = set(int(v) for v in layout_metadata.get("primary_zernike_indices", []))
    secondary = set(int(v) for v in layout_metadata.get("secondary_zernike_indices", []))
    indices = sorted(primary | secondary)
    return ",".join(str(value) for value in indices)


def _single_star_observation_theta_config(raw_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = _deepcopy_json(dict(raw_cfg))
    source_cfg = dict(cfg.get("source", {}) or {})
    source_cfg["separation_as"] = False
    source_cfg["contrast"] = False
    source_cfg["log_flux_total"] = bool(source_cfg.get("log_flux_total", True))
    cfg["source"] = source_cfg
    return cfg


def build_subblock_command(
    *,
    case_root_parent: Path,
    case_subblock_name: str,
    template_paths: Mapping[str, Path],
    theta_labels: Sequence[str],
    layout_metadata: Mapping[str, Any],
    offsets: Mapping[str, float],
    subblock_cfg: Mapping[str, Any],
    trace_seed: int,
    noise_seed: int,
) -> list[str]:
    """Build the image-backed Schur-summary command for one calibration block."""

    command = [
        sys.executable,
        str(SUBBLOCK_SCRIPT),
        "--results-root",
        str(case_root_parent),
        "--case-name",
        case_subblock_name,
        "--mode",
        "schur_summary",
        "--trace-template",
        str(template_paths["trace"]),
        "--render-template",
        str(template_paths["render"]),
        "--inference-template",
        str(template_paths["inference"]),
        "--n-frames",
        str(int(subblock_cfg.get("n_frames", 20))),
        "--noise",
        str(subblock_cfg.get("noise", "enabled")),
        "--theta-keys",
        ",".join(_theta_scalar_keys(theta_labels)),
        "--enable-zernikes",
        "--zernike-indices",
        _zernike_indices_arg(layout_metadata),
        "--phi-ref",
        str(subblock_cfg.get("phi_ref", "recovered")),
        "--schur-curvature-method",
        str(subblock_cfg.get("schur_curvature_method", DEFAULT_SINGLE_STAR_SCHUR_METHOD)),
        "--max-dense-dim",
        str(int(subblock_cfg.get("max_dense_dim", 40))),
        "--schur-damping",
        str(float(subblock_cfg.get("schur_damping", 1.0e-8))),
        "--trace-seed",
        str(int(trace_seed)),
        "--render-seed",
        str(int(noise_seed)),
    ]
    if subblock_cfg.get("exposure_time_s") is not None:
        command.extend(["--exposure-time-s", str(float(subblock_cfg["exposure_time_s"]))])
    for flag_key, flag in (
        ("reference_diagnostics_profile", "--reference-diagnostics-profile"),
        ("reference_optimizer_kind", "--reference-optimizer-kind"),
        ("reference_base_lr", "--reference-base-lr"),
        ("reference_n_iter", "--reference-n-iter"),
        ("reference_schedule_kind", "--reference-schedule-kind"),
        ("reference_schedule_warmup_steps", "--reference-schedule-warmup-steps"),
        ("reference_schedule_start_factor", "--reference-schedule-start-factor"),
        ("schur_frame_quality_policy", "--schur-frame-quality-policy"),
        ("schur_frame_chi2_threshold", "--schur-frame-chi2-threshold"),
        ("schur_frame_quality_missing", "--schur-frame-quality-missing"),
        ("schur_frame_mask_denominator", "--schur-frame-mask-denominator"),
        ("schur_frame_mask_min_good_frames", "--schur-frame-mask-min-good-frames"),
    ):
        if subblock_cfg.get(flag_key) is not None:
            command.extend([flag, str(subblock_cfg[flag_key])])
    jitter = dict(subblock_cfg.get("trace_jitter", {}) or {})
    if jitter.get("x_sigma_as") is not None:
        command.extend(["--trace-jitter-x-sigma-as", str(float(jitter["x_sigma_as"]))])
    if jitter.get("y_sigma_as") is not None:
        command.extend(["--trace-jitter-y-sigma-as", str(float(jitter["y_sigma_as"]))])
    if (
        str(jitter.get("pa_mode", "omitted")).strip().lower() == "inert_diagnostic"
        and jitter.get("pa_sigma_deg") is not None
    ):
        command.extend(["--trace-jitter-pa-sigma-deg", str(float(jitter["pa_sigma_deg"]))])
    for label, offset in sorted(offsets.items()):
        command.extend(["--theta-reference-offset", f"{label}={float(offset)}"])
    if bool(subblock_cfg.get("memory_diagnostics", False)):
        command.append("--memory-diagnostics")
    return command


def build_calibration_plan(
    *,
    config_path: Path | None,
    results_root: Path,
    run_name: str | None,
    system_preset: str | None,
    args: argparse.Namespace | None = None,
) -> CalibrationPlan:
    config = _load_config(config_path)
    experiment_cfg = _apply_cli_overrides(dict(config["experiment"]), args)
    if run_name is not None:
        experiment_cfg["run_name"] = run_name
    resolved_run_name = str(experiment_cfg.get("run_name") or f"single_star_calibration_{timestamp_tag()}")
    run_root = Path(results_root).resolve() / resolved_run_name
    store, system_cfg, source_provenance = _resolve_single_star_system(
        experiment_cfg=experiment_cfg,
        system_preset=system_preset,
    )
    theta_cfg = _single_star_observation_theta_config(
        experiment_cfg.get("observation_theta", {}) or {}
    )
    experiment_cfg["observation_theta"] = theta_cfg
    layout, metadata = build_system_observation_theta_layout(
        store,
        config=theta_cfg,
    )
    forbidden = {"source.separation_as", "source.contrast", "source.position_angle_deg"}
    if forbidden & set(layout.labels):
        raise ValueError("Single-star calibration theta must not include separation or contrast.")
    metadata["system"] = source_provenance
    metadata["resolved_system"] = system_cfg
    truth_vector = build_prior_mean_from_store(layout.labels, store=store)
    truth_by_label = {label: float(truth_vector[i]) for i, label in enumerate(layout.labels)}
    cases, prior_draw_rows = generate_calibration_cases(
        experiment_cfg=experiment_cfg,
        labels=layout.labels,
        truth_by_label=truth_by_label,
    )
    template_paths = write_single_star_templates(
        run_root,
        system_cfg,
        experiment_cfg=experiment_cfg,
    )
    subblock_cfg = dict(experiment_cfg.get("subblocks", {}) or {})
    schur_settings = _schur_settings_for_single_star_config(subblock_cfg)
    subblock_cfg["schur_curvature_method"] = schur_settings["schur_curvature_method_requested"]
    if args is not None and bool(getattr(args, "memory_diagnostics", False)):
        subblock_cfg["memory_diagnostics"] = True
    seeding = dict(experiment_cfg.get("seeding", {}) or {})
    seed_policy = str(seeding.get("seed_policy", "different_jitter_different_noise"))
    if seed_policy not in SUPPORTED_SEED_POLICIES:
        raise ValueError(f"Unsupported seed_policy: {seed_policy}")
    base_seed = int(seeding.get("base_seed", experiment_cfg.get("seed", 42)))
    n_subblocks = int(subblock_cfg.get("n_subblocks", 3))
    subblock_root = run_root / "subblock_runs"
    commands: dict[str, list[list[str]]] = {}
    summary_paths: dict[str, list[Path]] = {}
    rows: list[dict[str, Any]] = []
    for case in cases:
        commands[case.case_name] = []
        summary_paths[case.case_name] = []
        for subblock_index in range(n_subblocks):
            seeds = _derive_subblock_seeds(
                run_name=resolved_run_name,
                case_name=case.case_name,
                subblock_index=subblock_index,
                seed_policy=seed_policy,
                base_seed=base_seed,
            )
            subblock_name = f"{case.case_name}/subblock_{subblock_index:06d}"
            summary_path = (
                subblock_root / subblock_name / "study" / "schur_summary" / "subblock_summary.json"
            )
            command = build_subblock_command(
                case_root_parent=subblock_root,
                case_subblock_name=subblock_name,
                template_paths=template_paths,
                theta_labels=layout.labels,
                layout_metadata=metadata,
                offsets=case.theta_reference_offsets,
                subblock_cfg=subblock_cfg,
                trace_seed=seeds["trace_seed"],
                noise_seed=seeds["noise_seed"],
            )
            commands[case.case_name].append(command)
            summary_paths[case.case_name].append(summary_path)
            rows.append(
                {
                    "case_name": case.case_name,
                    "case_origin": case.case_origin,
                    "subblock_index": int(subblock_index),
                    "subblock_name": subblock_name,
                    "summary_path": str(summary_path),
                    "n_frames": int(subblock_cfg.get("n_frames", 20)),
                    "noise": str(subblock_cfg.get("noise", "enabled")),
                    "phi_ref": str(subblock_cfg.get("phi_ref", "recovered")),
                    "schur_curvature_method_requested": schur_settings["schur_curvature_method_requested"],
                    "schur_curvature_method_effective": schur_settings["schur_curvature_method_effective"],
                    "schur_route_source": schur_settings["schur_route_source"],
                    "max_dense_dim": schur_settings["max_dense_dim"],
                    "structured_curvature_used": schur_settings["structured_curvature_expected"],
                    "dense_global_hessian_materialized": "",
                    "trace_seed": seeds["trace_seed"],
                    "noise_seed": seeds["noise_seed"],
                    "command": " ".join(command),
                    "parent_diagnostics_json": str((subblock_root / subblock_name / "study" / "subprocess_diagnostics.json")),
                    "subprocess_diagnostics_path": str((subblock_root / subblock_name / "study" / "subprocess_diagnostics.json")),
                    "schur_diagnostics_path": str((subblock_root / subblock_name / "study" / "schur_summary" / "schur_diagnostics.json")),
                    "schur_memory_audit_path": str((subblock_root / subblock_name / "study" / "schur_summary" / "schur_summary_memory_audit.json")),
                }
            )
    resolved_config = {"experiment": experiment_cfg, "system": system_cfg}
    return CalibrationPlan(
        run_root=run_root,
        layout=layout,
        layout_metadata=metadata,
        truth_vector=truth_vector,
        system_cfg=system_cfg,
        config=resolved_config,
        cases=cases,
        summary_paths=summary_paths,
        subblock_commands=commands,
        subblock_rows=rows,
        prior_draw_rows=prior_draw_rows,
        status_rows=[],
    )


def _plan_payload(plan: CalibrationPlan) -> dict[str, Any]:
    forecast = dict(plan.config["experiment"].get("forecast", {}) or {})
    trace_keys, inactive_truth_keys, pa_policy = _trace_key_policy(
        plan.config["experiment"]
    )
    n_frames = int(plan.config["experiment"].get("subblocks", {}).get("n_frames", 20))
    frame_phi_dim = len(ACTIVE_FRAME_KEYS)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "run_root": str(plan.run_root),
        "source_kind": "single_star",
        "calibration_source_mode": "alpha_cen_a_placeholder",
        "photometry_note": ALPHA_CEN_A_PLACEHOLDER_NOTE,
        "local_eliminated_keys": list(ACTIVE_FRAME_KEYS),
        "active_frame_keys": list(ACTIVE_FRAME_KEYS),
        "trace_varying_keys": list(trace_keys),
        "inactive_truth_keys": list(inactive_truth_keys),
        "single_star_pa_policy": pa_policy,
        "observation_theta_labels": list(plan.layout.labels),
        "theta_layout": plan.layout.to_dict(),
        "layout_metadata": plan.layout_metadata,
        "n_cases": len(plan.cases),
        "n_subblocks": int(plan.config["experiment"].get("subblocks", {}).get("n_subblocks", 3)),
        "n_frames": n_frames,
        "dimension_estimate": {
            "frame_phi_dim": int(frame_phi_dim),
            "n_phi": int(frame_phi_dim * n_frames),
        },
        "summary_paths": {
            case: [str(path) for path in paths]
            for case, paths in plan.summary_paths.items()
        },
        "intended_subblock_commands": {
            case: [" ".join(cmd) for cmd in commands]
            for case, commands in plan.subblock_commands.items()
        },
        "forecast_grid": list(forecast.get("n_subblocks_grid", [])),
        "notes": [
            "Calibration star photometry is an Alpha Cen A component placeholder.",
            "Single-star frame solve uses X/Y only; PA is optional inert truth-trace diagnostics.",
            "Forecast-to-1800-subblocks is extrapolation, not a real image-backed 30-minute run.",
        ],
    }


def write_plan_artifacts(plan: CalibrationPlan) -> None:
    _write_json(plan.run_root / "campaign_plan.json", _plan_payload(plan))
    _write_json(plan.run_root / "resolved_config.json", plan.config)
    _write_csv(plan.run_root / "subblock_plan.csv", plan.subblock_rows)
    _write_csv(
        plan.run_root / "calibration_cases.csv",
        [
            {
                "case_name": case.case_name,
                "case_origin": case.case_origin,
                "n_theta_offsets": len(case.theta_reference_offsets),
            }
            for case in plan.cases
        ],
    )
    _write_csv(plan.run_root / "prior_draws.csv", plan.prior_draw_rows)


def _load_case_summaries(paths: Sequence[Path]) -> list[SubblockSummary]:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing subblock summaries: " + ", ".join(str(path) for path in missing))
    summaries = [load_subblock_summary(path) for path in paths]
    labels = tuple(summaries[0].theta_labels)
    for summary in summaries[1:]:
        if tuple(summary.theta_labels) != labels:
            raise ValueError("Subblock summary theta labels differ within case.")
    return summaries


def _prior_for_case(plan: CalibrationPlan, case: CalibrationCase, summaries: Sequence[SubblockSummary]) -> tuple[ObservationBeliefState, np.ndarray, np.ndarray, np.ndarray]:
    labels = tuple(plan.layout.labels)
    truth = np.asarray(plan.truth_vector, dtype=float)
    offsets = np.asarray([float(case.theta_reference_offsets.get(label, 0.0)) for label in labels])
    reference = truth + offsets
    if summaries:
        reference = np.asarray(summaries[0].theta_ref, dtype=float)
    if case.prior_sigma_by_label is None:
        sigma_cfg = dict(plan.config["experiment"].get("prior", {}).get("sigma", {}) or {})
        sigma_by_label = resolve_prior_sigmas(
            labels,
            {label: float(truth[i]) for i, label in enumerate(labels)},
            sigma_cfg,
        )
    else:
        sigma_by_label = case.prior_sigma_by_label
    sigma = np.asarray([sigma_by_label[label] for label in labels], dtype=float)
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=labels,
        mean=reference,
        sigma=sigma,
        metadata={"prior_sigma_source": "calibration_config", "case_origin": case.case_origin},
    )
    return prior, truth, reference, sigma


def _posterior_metric_rows(
    *,
    case_name: str,
    labels: Sequence[str],
    truth: np.ndarray,
    reference: np.ndarray,
    prior_sigma: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_sigma: np.ndarray,
    history_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    one_nm_cross: dict[str, tuple[int, float]] = {}
    for row in history_rows:
        label = str(row["theta_label"])
        if "zernike_coeffs_nm" in label and float(row["posterior_sigma"]) <= 1.0 and label not in one_nm_cross:
            one_nm_cross[label] = (int(row["n_subblocks"]), float(row["calibration_time_s"]))
    for index, label in enumerate(labels):
        truth_value = float(truth[index])
        reference_value = float(reference[index])
        posterior_value = float(posterior_mean[index])
        prior_error = reference_value - truth_value
        posterior_error = posterior_value - truth_value
        posterior_shift = posterior_value - reference_value
        is_zernike = "zernike_coeffs_nm" in label
        crossed = one_nm_cross.get(label)
        rows.append(
            {
                "case_name": case_name,
                "theta_label": label,
                "parameter_group": _parameter_group(label),
                "unit": _parameter_unit(label),
                "truth_value": truth_value,
                "prior_mean": reference_value,
                "reference_value": reference_value,
                "injected_bias": prior_error,
                "posterior_mean": posterior_value,
                "posterior_shift": posterior_shift,
                "posterior_error": posterior_error,
                "prior_sigma": float(prior_sigma[index]),
                "posterior_sigma": float(posterior_sigma[index]),
                "posterior_error_over_sigma": _safe_fraction(posterior_error, float(posterior_sigma[index])),
                "correction_fraction": _safe_fraction(posterior_shift, -prior_error),
                "residual_fraction": _safe_fraction(posterior_error, prior_error),
                "moves_toward_truth": bool(abs(posterior_error) < abs(prior_error)),
                "crosses_1nm_threshold": (None if not is_zernike else bool(crossed is not None)),
                "time_to_1nm_s": (None if crossed is None else crossed[1]),
                "n_subblocks_to_1nm": (None if crossed is None else crossed[0]),
            }
        )
    return rows


def _history_rows(
    *,
    case_name: str,
    plan: CalibrationPlan,
    prior: ObservationBeliefState,
    summaries: Sequence[SubblockSummary],
    truth: np.ndarray,
    reference: np.ndarray,
) -> list[dict[str, Any]]:
    prefixes = [
        int(v)
        for v in plan.config["experiment"].get("history_prefixes", [1, 2, 3, 5, 10, 30, 100, 300, 1000, 1800])
        if int(v) <= len(summaries)
    ]
    if len(summaries) not in prefixes:
        prefixes.append(len(summaries))
    subblock_duration = float(plan.config["experiment"].get("forecast", {}).get("subblock_duration_s", 1.0))
    rows: list[dict[str, Any]] = []
    for count in sorted(set(prefixes)):
        update = update_observation_belief(prior, summaries[:count])
        sigma = update.posterior.sigma()
        for index, label in enumerate(plan.layout.labels):
            posterior_error = float(update.posterior.mean[index] - truth[index])
            prior_error = float(reference[index] - truth[index])
            rows.append(
                {
                    "case_name": case_name,
                    "n_subblocks": int(count),
                    "calibration_time_s": float(count * subblock_duration),
                    "theta_label": label,
                    "posterior_mean": float(update.posterior.mean[index]),
                    "posterior_sigma": float(sigma[index]),
                    "posterior_error": posterior_error,
                    "posterior_error_over_sigma": _safe_fraction(posterior_error, float(sigma[index])),
                    "correction_fraction": _safe_fraction(
                        float(update.posterior.mean[index] - reference[index]),
                        -prior_error,
                    ),
                    "residual_fraction": _safe_fraction(posterior_error, prior_error),
                }
            )
    return rows


def _write_eigenmodes(
    *,
    case_root: Path,
    source_name: str,
    labels: Sequence[str],
    matrix: np.ndarray,
    prior_sigma: np.ndarray,
    eigen_cfg: Mapping[str, Any],
) -> None:
    if bool(eigen_cfg.get("whiten", True)):
        source_matrix = build_prior_whitened_information_gain_matrix(matrix, prior_sigma)
    else:
        source_matrix = np.asarray(matrix, dtype=float)
    basis = build_observation_eigenbasis(
        source_matrix,
        labels,
        eig_floor_abs=float(eigen_cfg.get("eig_floor_abs", 0.0)),
        eig_floor_rel=float(eigen_cfg.get("eig_floor_rel", 1.0e-12)),
    )
    top_k = int(eigen_cfg.get("top_k_contributors", 8))
    rows: list[dict[str, Any]] = []
    for mode_index, eigenvalue in enumerate(basis.eigenvalues):
        contributors = basis.mode_contributors(mode_index, top_k=top_k)
        rows.append(
            {
                "mode_index": int(mode_index),
                "eigenvalue": float(eigenvalue),
                "prior_whitened_eigenvalue": float(eigenvalue) if bool(eigen_cfg.get("whiten", True)) else None,
                "dominant_labels": "; ".join(label for label, _ in contributors[:3]),
                "dominant_components": "; ".join(f"{coeff:+.4f}" for _, coeff in contributors[:3]),
                "participation": float(1.0 / np.sum(np.square(basis.eigenvectors[:, mode_index]) ** 2)),
            }
        )
    _write_csv(case_root / f"eigenmodes_{source_name}.csv", rows)


def _replicate_summaries(
    summaries: Sequence[SubblockSummary],
    count: int,
) -> tuple[SubblockSummary, ...]:
    if not summaries:
        raise ValueError("Forecast requires at least one summary.")
    return tuple(summaries[index % len(summaries)] for index in range(int(count)))


def _sample_score_noise(
    information: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha: float,
    eig_floor_rel: float,
) -> np.ndarray:
    matrix = 0.5 * (np.asarray(information, dtype=float) + np.asarray(information, dtype=float).T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0, 1.0)
    effective = np.clip(eigenvalues, float(eig_floor_rel) * scale, None)
    return eigenvectors @ (np.sqrt(float(alpha) * effective) * rng.normal(size=effective.shape))


def _write_forecast_outputs(
    *,
    case_root: Path,
    case_name: str,
    plan: CalibrationPlan,
    prior: ObservationBeliefState,
    summaries: Sequence[SubblockSummary],
    truth: np.ndarray,
    reference: np.ndarray,
    prior_sigma: np.ndarray,
) -> dict[str, Any]:
    cfg = dict(plan.config["experiment"].get("forecast", {}) or {})
    if not bool(cfg.get("enabled", False)):
        summary = {"enabled": False}
        _write_json(case_root / "forecast_summary.json", summary)
        return summary
    modes = tuple(str(mode) for mode in cfg.get("modes", ("replicate",)))
    grid = sorted(
        {
            int(value)
            for value in cfg.get("n_subblocks_grid", [1, 3, 5, 10, 30, 100, 300, 1000, 1800])
            if int(value) >= 1
        }
        | {len(summaries)}
    )
    subblock_duration = float(cfg.get("subblock_duration_s", 1.0))
    by_parameter: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []

    def append_rows(mode: str, count: int, update: Any, trial_index: int | None = None) -> None:
        sigma = update.posterior.sigma()
        for index, label in enumerate(plan.layout.labels):
            posterior_error = float(update.posterior.mean[index] - truth[index])
            prior_error = float(reference[index] - truth[index])
            row = {
                "case_name": case_name,
                "forecast_mode": mode,
                "n_subblocks": int(count),
                "calibration_time_s": float(count * subblock_duration),
                "theta_label": label,
                "parameter_group": _parameter_group(label),
                "unit": _parameter_unit(label),
                "posterior_mean": float(update.posterior.mean[index]),
                "posterior_sigma": float(sigma[index]),
                "posterior_sigma_over_prior_sigma": _safe_fraction(float(sigma[index]), float(prior_sigma[index])),
                "posterior_error": posterior_error,
                "posterior_error_over_sigma": _safe_fraction(posterior_error, float(sigma[index])),
                "correction_fraction": _safe_fraction(
                    float(update.posterior.mean[index] - reference[index]),
                    -prior_error,
                ),
            }
            if trial_index is None:
                by_parameter.append(row)
            else:
                row["trial_index"] = int(trial_index)
                trial_rows.append(row)

    if "replicate" in modes:
        for count in grid:
            update = update_observation_belief(prior, _replicate_summaries(summaries, count))
            append_rows("replicate", count, update)

    if "fixed_information_score_noise" in modes:
        noise_cfg = dict(cfg.get("fixed_information_score_noise", {}) or {})
        if bool(noise_cfg.get("enabled", True)):
            n_trials = int(noise_cfg.get("n_trials", 100))
            seed = int(noise_cfg.get("seed", 2026))
            alpha = float(noise_cfg.get("score_noise_alpha", 1.0))
            eig_floor_rel = float(noise_cfg.get("score_noise_eig_floor_rel", 1.0e-12))
            max_count = max(grid)
            for trial_index in range(n_trials):
                rng = np.random.default_rng(make_subseed(seed, f"{case_name}.forecast.{trial_index}"))
                synthetic: list[SubblockSummary] = []
                for seq_index, template in enumerate(_replicate_summaries(summaries, max_count)):
                    info = np.asarray(template.reduced_information, dtype=float)
                    expected = info @ (np.asarray(template.theta_ref, dtype=float) - truth)
                    score = expected + _sample_score_noise(
                        info,
                        rng=rng,
                        alpha=alpha,
                        eig_floor_rel=eig_floor_rel,
                    )
                    synthetic.append(
                        SubblockSummary.from_reduced_form(
                            subblock_id=f"forecast_{trial_index:03d}_{seq_index:06d}",
                            theta_labels=template.theta_labels,
                            theta_ref=template.theta_ref,
                            reduced_information=info,
                            reduced_score=score,
                            summary_kind="fixed_information_score_noise_forecast",
                        )
                    )
                for count in grid:
                    update = update_observation_belief(prior, synthetic[:count])
                    append_rows("fixed_information_score_noise", count, update, trial_index)

    _write_csv(case_root / "forecast_by_parameter.csv", by_parameter)
    _write_csv(case_root / "forecast_trials.csv", trial_rows)
    summary = {
        "enabled": True,
        "modes": list(modes),
        "forecast_grid": grid,
        "subblock_duration_s": subblock_duration,
        "limitations": {
            "replicate": "Repeats available real summaries deterministically.",
            "fixed_information_score_noise": "Keeps information fixed and samples score noise with covariance alpha * S.",
        },
    }
    _write_json(case_root / "forecast_summary.json", summary)
    return summary


def _plot_case(case_root: Path, history: Sequence[Mapping[str, Any]], posterior_rows: Sequence[Mapping[str, Any]]) -> None:
    if not _HAVE_MATPLOTLIB or plt is None:
        return
    plot_root = case_root / "plots"
    _ensure_dir(plot_root)
    z_rows = [row for row in history if "zernike_coeffs_nm" in str(row["theta_label"])]
    if z_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label in sorted({str(row["theta_label"]) for row in z_rows}):
            subset = [row for row in z_rows if row["theta_label"] == label]
            ax.plot(
                [float(row["calibration_time_s"]) for row in subset],
                [float(row["posterior_sigma"]) for row in subset],
                marker="o",
                label=label.replace("optics.", ""),
            )
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel("Calibration Time (s)")
        ax.set_ylabel("Posterior Sigma (nm)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_root / "posterior_sigma_nm_vs_time_zernikes.png", dpi=160)
        plt.close(fig)
    if posterior_rows:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [str(row["theta_label"]) for row in posterior_rows]
        values = [float(row["posterior_sigma"]) for row in posterior_rows]
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("Posterior Sigma")
        fig.tight_layout()
        fig.savefig(plot_root / "final_posterior_sigma_by_parameter.png", dpi=160)
        plt.close(fig)


def aggregate_case(plan: CalibrationPlan, case: CalibrationCase) -> dict[str, Any]:
    case_root = plan.run_root / "cases" / case.case_name
    summaries = _load_case_summaries(plan.summary_paths[case.case_name])
    prior, truth, reference, prior_sigma = _prior_for_case(plan, case, summaries)
    update = update_observation_belief(prior, summaries)
    accumulated = accumulate_summary_information(plan.layout.labels, summaries)
    history = _history_rows(
        case_name=case.case_name,
        plan=plan,
        prior=prior,
        summaries=summaries,
        truth=truth,
        reference=reference,
    )
    posterior_rows = _posterior_metric_rows(
        case_name=case.case_name,
        labels=plan.layout.labels,
        truth=truth,
        reference=reference,
        prior_sigma=prior_sigma,
        posterior_mean=update.posterior.mean,
        posterior_sigma=update.posterior.sigma(),
        history_rows=history,
    )
    _write_csv(case_root / "posterior_by_parameter.csv", posterior_rows)
    _write_csv(case_root / "posterior_history.csv", history)
    _write_json(
        case_root / "case_summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "case_name": case.case_name,
            "n_summaries": len(summaries),
            "posterior": update.posterior.to_dict(),
            "prior": prior.to_dict(),
            "summary_paths": [str(path) for path in plan.summary_paths[case.case_name]],
        },
    )
    eigen_cfg = dict(plan.config["experiment"].get("eigenbasis", {}) or {})
    if bool(eigen_cfg.get("enabled", True)):
        for source_name in eigen_cfg.get("sources", ["accumulated_information", "posterior_precision"]):
            matrix = update.posterior.precision if source_name == "posterior_precision" else accumulated
            _write_eigenmodes(
                case_root=case_root,
                source_name=str(source_name),
                labels=plan.layout.labels,
                matrix=matrix,
                prior_sigma=prior_sigma,
                eigen_cfg=eigen_cfg,
            )
    forecast_summary = _write_forecast_outputs(
        case_root=case_root,
        case_name=case.case_name,
        plan=plan,
        prior=prior,
        summaries=summaries,
        truth=truth,
        reference=reference,
        prior_sigma=prior_sigma,
    )
    _plot_case(case_root, history, posterior_rows)
    return {
        "case_name": case.case_name,
        "case_root": str(case_root),
        "n_summaries": len(summaries),
        "posterior_by_parameter_csv": str(case_root / "posterior_by_parameter.csv"),
        "forecast_summary": forecast_summary,
    }


def aggregate_campaign(plan: CalibrationPlan) -> dict[str, Any]:
    summaries = [aggregate_case(plan, case) for case in plan.cases]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(plan.run_root),
        "cases": summaries,
        "memory_failure_summary_csv": str(plan.run_root / "memory_failure_summary.csv"),
    }
    status_path = plan.run_root / "subblock_status.csv"
    if status_path.exists():
        rows = list(csv.DictReader(status_path.open("r", encoding="utf-8")))
        _write_csv(
            plan.run_root / "memory_failure_summary.csv",
            [
                {
                    "case_name": row.get("case_name"),
                    "summary_path": row.get("summary_path"),
                    "diagnostics_json": row.get("diagnostics_json"),
                    "failure_class": row.get("failure_class"),
                    "peak_total_rss_mb": row.get("peak_total_rss_mb"),
                }
                for row in rows
            ],
        )
    _write_json(plan.run_root / "campaign_summary.json", payload)
    return payload


def execute_subblocks(
    plan: CalibrationPlan,
    *,
    resume: bool,
    max_workers: int,
    fail_fast: bool,
    quiet: bool,
    memory_diagnostics: bool,
    resource_time: bool,
) -> None:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    jobs: list[tuple[str, Path, list[str]]] = []
    for case_name, commands in plan.subblock_commands.items():
        for command, summary_path in zip(commands, plan.summary_paths[case_name], strict=True):
            if resume and summary_path.exists():
                try:
                    load_subblock_summary(summary_path)
                    continue
                except Exception:
                    if not fail_fast:
                        pass
            jobs.append((case_name, summary_path, command))
    if not jobs:
        return

    status_rows: list[dict[str, Any]] = []

    def run_job(job: tuple[str, Path, list[str]]) -> tuple[str, Path, dict[str, Any]]:
        case_name, summary_path, command = job
        case_root = summary_path.parent.parent
        diagnostics_json = case_root / "subprocess_diagnostics.json"
        diag = run_subprocess_with_diagnostics(
            command=command,
            cwd=REPO_ROOT,
            env=env,
            stdout_log=case_root / "subprocess.stdout.log",
            stderr_log=case_root / "subprocess.stderr.log",
            diagnostics_json=diagnostics_json,
            memory_diagnostics=memory_diagnostics,
            resource_time=resource_time,
        )
        if int(diag.return_code) != 0:
            raise RuntimeError(
                f"Subprocess failed ({diag.return_code}) for {summary_path}: {diagnostics_json}"
            )
        schur_diag_path = summary_path.parent / "schur_diagnostics.json"
        schur_diag: dict[str, Any] = {}
        if schur_diag_path.exists():
            try:
                schur_diag = json.loads(schur_diag_path.read_text(encoding="utf-8"))
            except Exception:
                schur_diag = {}
        return case_name, summary_path, {
            "diagnostics_json": str(diagnostics_json),
            "subprocess_diagnostics_path": str(diagnostics_json),
            "schur_diagnostics_path": str(schur_diag_path),
            "schur_memory_audit_path": str(summary_path.parent / "schur_summary_memory_audit.json"),
            "return_code": diag.return_code,
            "failure_class": diag.failure_class,
            "peak_total_rss_mb": diag.memory_sampler.get("peak_total_rss_mb"),
            "schur_curvature_method_requested": schur_diag.get("schur_curvature_method_requested"),
            "schur_curvature_method_effective": schur_diag.get("schur_curvature_method_effective"),
            "structured_curvature_used": schur_diag.get("structured_curvature_used"),
            "structured_supported_layout": schur_diag.get("structured_supported_layout"),
            "dense_global_hessian_materialized": schur_diag.get("dense_global_hessian_materialized"),
            "dense_hessian_allowed": schur_diag.get("dense_hessian_allowed"),
            "dense_hessian_skipped_reason": schur_diag.get("dense_hessian_skipped_reason"),
            "dense_hessian_materialized_reason": schur_diag.get("dense_hessian_materialized_reason"),
            "validate_structured_against_dense": schur_diag.get("validate_structured_against_dense"),
            "max_dense_dim": schur_diag.get("max_dense_dim"),
            "combined_dim": schur_diag.get("combined_dim"),
            "n_theta": schur_diag.get("n_theta"),
            "n_phi": schur_diag.get("n_phi"),
            "n_frames": schur_diag.get("n_frames"),
            "frame_keys": schur_diag.get("frame_keys"),
            "theta_keys": schur_diag.get("theta_keys"),
        }

    if max_workers <= 1:
        for job in jobs:
            try:
                case_name, summary_path, row = run_job(job)
                status_rows.append({"case_name": case_name, "summary_path": str(summary_path), **row})
                if not quiet:
                    print(f"[single_star_calibration] completed {case_name}: {summary_path}", flush=True)
            except Exception as exc:
                if fail_fast:
                    raise RuntimeError(str(exc)) from exc
                print(str(exc), file=sys.stderr, flush=True)
        _write_csv(plan.run_root / "subblock_status.csv", status_rows)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_job = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            try:
                case_name, summary_path, row = future.result()
                status_rows.append({"case_name": case_name, "summary_path": str(summary_path), **row})
                if not quiet:
                    print(f"[single_star_calibration] completed {case_name}: {summary_path}", flush=True)
            except Exception as exc:
                if fail_fast:
                    raise RuntimeError(str(exc)) from exc
                print(str(exc), file=sys.stderr, flush=True)
    _write_csv(plan.run_root / "subblock_status.csv", status_rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the single-star calibration observation demo.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=True)
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--system-preset", default=None)
    parser.add_argument("--n-subblocks", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--noise", choices=("enabled", "disabled", "inherit"), default=None)
    parser.add_argument("--phi-ref", choices=("recovered", "truth_when_available", "init"), default=None)
    parser.add_argument("--zernike-indices", default=None)
    parser.add_argument("--include-plate-scale", dest="include_plate_scale", action="store_true", default=None)
    parser.add_argument("--no-include-plate-scale", dest="include_plate_scale", action="store_false")
    parser.add_argument("--include-log-flux", dest="include_log_flux", action="store_true", default=None)
    parser.add_argument("--no-include-log-flux", dest="include_log_flux", action="store_false")
    parser.add_argument(
        "--reference-diagnostics-profile",
        choices=("none", "basic", "review", "full"),
        default=None,
    )
    parser.add_argument("--memory-diagnostics", action="store_true")
    parser.add_argument("--schur-curvature-method", default=None)
    parser.add_argument("--max-dense-dim", type=int, default=None)
    parser.add_argument("--memory-progress-tail-lines", type=int, default=3)
    parser.add_argument("--resource-time", dest="resource_time", action="store_true", default=None)
    parser.add_argument("--no-resource-time", dest="resource_time", action="store_false")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    plan = build_calibration_plan(
        config_path=args.config,
        results_root=args.results_root,
        run_name=args.run_name,
        system_preset=args.system_preset,
        args=args,
    )
    write_plan_artifacts(plan)
    if args.dry_run:
        if not args.quiet:
            print(f"Dry-run plan written to {plan.run_root}")
        return _plan_payload(plan)
    if not args.aggregate_only:
        execute_subblocks(
            plan,
            resume=bool(args.resume),
            max_workers=max(1, int(args.max_workers)),
            fail_fast=bool(args.fail_fast),
            quiet=bool(args.quiet),
            memory_diagnostics=bool(args.memory_diagnostics),
            resource_time=True if args.resource_time is None else bool(args.resource_time),
        )
    summary = aggregate_campaign(plan)
    if not args.quiet:
        print(f"Calibration summary written to {plan.run_root / 'campaign_summary.json'}")
    return summary


if __name__ == "__main__":
    main()
