"""Thin full-fidelity binary iterative smoke wrapper.

This wrapper intentionally delegates execution to run_observation_bias_campaign.py.
It only translates the narrow smoke schema into the existing campaign schema and
keeps the Data/Inference model split auditable through the shared split helper
used by that campaign.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from dluxshera.config.io import load_config_file

from run_observation_bias_campaign import (  # type: ignore
    DEFAULT_RESULTS_ROOT,
    run_observation_bias_campaign,
)


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _full_fidelity_to_observation_bias(config: Mapping[str, Any], *, run_name: str | None) -> dict[str, Any]:
    experiment = _as_mapping(config.get("experiment", config), name="experiment")
    if str(experiment.get("kind", "")) == "observation_bias_campaign":
        out = {"experiment": dict(experiment)}
        if run_name is not None:
            out["experiment"]["run_name"] = run_name
        return out
    if str(experiment.get("kind")) != "full_fidelity_binary_iterative_smoke":
        raise ValueError(
            "Full-fidelity wrapper expects experiment.kind='full_fidelity_binary_iterative_smoke' "
            "or an already translated observation_bias_campaign config."
        )

    subblocks = _as_mapping(experiment.get("subblocks"), name="experiment.subblocks")
    iterative = _as_mapping(experiment.get("iterative"), name="experiment.iterative")
    source_kind = str(experiment.get("source_kind", "binary_target"))
    target = experiment.get("target", "ALPHA_CEN")
    n_cases = int(experiment.get("n_cases", 1))

    observation_theta = _as_mapping(experiment.get("observation_theta"), name="experiment.observation_theta") or {
        "source": {"separation_as": True, "log_flux_total": True, "contrast": True},
        "optics": {
            "plate_scale_as_per_pix": True,
            "primary_zernikes": {"enabled": True, "indices": [0]},
            "secondary_zernikes": {"enabled": True, "indices": [0]},
        },
    }
    prior_draws = _as_mapping(experiment.get("prior_draws"), name="experiment.prior_draws") or {
        "enabled": True,
        "n_cases": n_cases,
        "center": "truth",
        "distribution": "normal",
        "draw_seed": int(experiment.get("seed", 42)) + 123,
        "case_name_template": "full_fidelity_smoke_draw_{draw_index:03d}",
        "sigmas": {
            "source.separation_as": {"kind": "absolute", "sigma": 2.0e-6, "unit": "arcsec"},
            "source.log_flux_total": {"kind": "absolute", "sigma": 1.0e-5, "unit": "log_flux"},
            "source.contrast": {"kind": "fractional", "sigma": 1.0e-5},
            "optics.plate_scale_as_per_pix": {"kind": "fractional", "sigma": 1.0e-5},
            "optics.primary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 0.1, "unit": "nm"},
            "optics.secondary.zernike_coeffs_nm[*]": {"kind": "absolute", "sigma": 0.1, "unit": "nm"},
        },
    }

    translated = {
        "experiment": {
            "kind": "observation_bias_campaign",
            "source_campaign_kind": "full_fidelity_binary_iterative_smoke",
            "schema_version": "full_fidelity_binary_iterative_smoke.translated.v1",
            "seed": int(experiment.get("seed", 42)),
            "run_name": run_name or experiment.get("run_name", "full_fidelity_binary_iterative_smoke"),
            "system": {
                "preset": experiment.get("system_preset", "SHERA_FLIGHT_3P"),
                "source": {"kind": source_kind, "target": target},
            },
            "spectral_model": _as_mapping(experiment.get("spectral_model"), name="experiment.spectral_model"),
            "high_order_wfe": _as_mapping(experiment.get("high_order_wfe"), name="experiment.high_order_wfe"),
            "subblocks": subblocks,
            "iterative": iterative,
            "seeding": _as_mapping(experiment.get("seeding"), name="experiment.seeding") or {
                "seed_policy": "different_jitter_different_noise",
                "base_seed": int(experiment.get("seed", 42)),
            },
            "observation_theta": observation_theta,
            "bias_cases": [],
            "case_generation": {"include_implicit_zero_bias": False},
            "prior_draws": prior_draws,
            "truth_realization": _as_mapping(experiment.get("truth_realization"), name="experiment.truth_realization") or {"enabled": False},
            "eigenbasis": _as_mapping(experiment.get("eigenbasis"), name="experiment.eigenbasis") or {"enabled": False},
            "forecast": _as_mapping(experiment.get("forecast"), name="experiment.forecast") or {"enabled": False, "plots": False},
            "full_fidelity_smoke_contract": {
                "data_render_uses": "truth_system_cfg",
                "inference_reference_uses": "inference_system_cfg",
                "deferred": [
                    "production_scale_campaign",
                    "dynamic_crop_roi_origin",
                    "per_frame_dynamic_smear_kernels",
                    "high_order_map_pixel_inference",
                    "full_bayesian_recursive_filter",
                ],
            },
        }
    }
    return translated


def _write_translated_config(config: Mapping[str, Any], *, run_name: str | None) -> Path:
    payload = _full_fidelity_to_observation_bias(config, run_name=run_name)
    root = Path(tempfile.mkdtemp(prefix="dluxshera_full_fidelity_smoke_"))
    path = root / "translated_observation_bias_config.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_full_fidelity_binary_iterative_campaign(
    *,
    config_path: Path,
    results_root: Path,
    run_name: str | None,
    dry_run: bool,
    aggregate_only: bool,
    resume: bool,
    max_workers: int,
    fail_fast: bool,
    quiet: bool,
    resource_time: bool | str | None,
) -> dict[str, Any]:
    raw = load_config_file(config_path)
    translated_path = _write_translated_config(raw, run_name=run_name)
    return run_observation_bias_campaign(
        config_path=translated_path,
        results_root=results_root,
        run_name=run_name,
        dry_run=dry_run,
        aggregate_only=aggregate_only,
        resume=resume,
        max_workers=max_workers,
        fail_fast=fail_fast,
        quiet=quiet,
        resource_time=resource_time,
        args=argparse.Namespace(
            aggregate_only=aggregate_only,
            resume=resume,
            run_name=run_name,
            n_subblocks=None,
            n_frames=None,
            trace_source_mode=None,
            trajectory_csv=None,
            trajectory_start_s=None,
            trajectory_duration_s=None,
            trajectory_n_subblocks=None,
            trajectory_frame_dt_s=None,
            trajectory_output_keys=None,
            trajectory_plan=None,
            noise=None,
            phi_ref=None,
            max_dense_dim=None,
            schur_curvature_method=None,
            summary_information_scale=None,
            seed_policy=None,
            base_seed=None,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full-fidelity binary iterative smoke wrapper.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--resource-time",
        dest="resource_time",
        nargs="?",
        const="enabled",
        choices=("auto", "enabled", "gnu", "disabled"),
        default=None,
    )
    parser.add_argument("--no-resource-time", dest="resource_time", action="store_const", const="disabled")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_full_fidelity_binary_iterative_campaign(
        config_path=args.config,
        results_root=args.results_root,
        run_name=args.run_name,
        dry_run=bool(args.dry_run),
        aggregate_only=bool(args.aggregate_only),
        resume=bool(args.resume),
        max_workers=int(args.max_workers),
        fail_fast=bool(args.fail_fast),
        quiet=bool(args.quiet),
        resource_time=args.resource_time,
    )


if __name__ == "__main__":
    main()
