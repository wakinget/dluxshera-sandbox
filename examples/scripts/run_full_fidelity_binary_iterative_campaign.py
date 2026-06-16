"""Run executable full-fidelity binary iterative campaigns.

This is a thin wrapper, not a new campaign framework. It accepts the
canonical ``full_fidelity_binary_iterative`` schema, translates it into the
existing ``observation_bias_campaign`` schema, and delegates execution to
``run_observation_bias_campaign.py``. Deprecated review/smoke aliases are
accepted temporarily and normalized immediately. Already translated
``observation_bias_campaign`` configs are also accepted for replay/debugging.

The future ``full_fidelity_algorithm_campaign`` schema skeleton is intentionally
not accepted here. Use dry-run first when reviewing this wrapper because dry-run
writes the translated campaign plan and model-split artifacts without launching
sub-block inference.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping

from dluxshera.config.io import load_config_file
from dluxshera.utils.full_fidelity_defaults import DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
from dluxshera.utils.noise import normalize_noise_request

from run_observation_bias_campaign import (  # type: ignore
    DEFAULT_RESULTS_ROOT,
    run_observation_bias_campaign,
)


CANONICAL_FULL_FIDELITY_KIND = "full_fidelity_binary_iterative"
DEPRECATED_FULL_FIDELITY_ALIASES = (
    "full_fidelity_binary_iterative_smoke",
    "full_fidelity_binary_iterative_review",
)
FULL_FIDELITY_EXECUTABLE_KINDS = (
    CANONICAL_FULL_FIDELITY_KIND,
    *DEPRECATED_FULL_FIDELITY_ALIASES,
)
ACCEPTED_CONFIG_KINDS = (*FULL_FIDELITY_EXECUTABLE_KINDS, "observation_bias_campaign")
FUTURE_SKELETON_KIND = "full_fidelity_algorithm_campaign"
FUTURE_ONLY_SMOKE_BLOCKS = (
    "detector",
    "observation",
    "trajectory",
    "smear",
    "optics",
    "noise",
    "active_state",
    "iterative_update",
    "knockdowns",
    "outputs",
)


def _as_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _warn(message: str, *, emit: bool) -> None:
    if emit:
        warnings.warn(message, UserWarning, stacklevel=3)


def validate_full_fidelity_smoke_config(
    config: Mapping[str, Any],
    *,
    emit_warnings: bool = False,
) -> list[str]:
    """Return actionable smoke-wrapper warnings without rejecting loose fields."""

    warnings_out: list[str] = []
    experiment = _as_mapping(config.get("experiment", config), name="experiment")
    kind = str(experiment.get("kind", ""))

    def add(message: str) -> None:
        warnings_out.append(message)
        _warn(message, emit=emit_warnings)

    if kind == FUTURE_SKELETON_KIND:
        add(
            "experiment.kind='full_fidelity_algorithm_campaign' is the future schema skeleton "
            "and is not executable by this wrapper. Use "
            "experiment.kind='full_fidelity_binary_iterative' for executable review- "
            "or smoke-scale campaign configs."
        )
        return warnings_out
    if kind not in ACCEPTED_CONFIG_KINDS:
        add(
            "Unsupported experiment.kind for the full-fidelity wrapper. Accepted kinds are "
            "'full_fidelity_binary_iterative' and 'observation_bias_campaign'."
        )
        return warnings_out
    if kind == "observation_bias_campaign":
        return warnings_out

    spectral = _as_mapping(experiment.get("spectral_model"), name="experiment.spectral_model")
    if "fast" in spectral:
        if bool(spectral.get("fast")):
            add(
                "spectral_model.fast is consumed by the model-split helper and clamps effective "
                "spectral grids to truth<=7 and inference<=5 wavelengths. It is not a substitute "
                "for explicit n_lambda, wavelength range, or response-component settings."
            )
        else:
            add(
                "spectral_model.fast=false leaves explicit spectral grid settings in control; "
                "runtime still depends on n_lambda, wavelength range, and enabled response components."
            )
    if kind in {"full_fidelity_binary_iterative", "full_fidelity_binary_iterative_review"} and "fast" in spectral:
        add(
            "spectral_model.fast is smoke-only and should be absent from the review config; "
            "use explicit spectral grids instead."
        )

    future_blocks = [key for key in FUTURE_ONLY_SMOKE_BLOCKS if key in experiment]
    if future_blocks:
        add(
            "Future-schema blocks are present in the smoke config but are not consumed by the "
            f"smoke wrapper: {', '.join(future_blocks)}. Do not copy skeleton blocks into the "
            "executable smoke config unless wrapper support is added."
        )

    subblocks = _as_mapping(experiment.get("subblocks"), name="experiment.subblocks")
    if isinstance(subblocks.get("noise"), Mapping):
        add(
            "subblocks.noise uses the structured review schema. The current subblock runner "
            "receives only a legacy enabled/disabled noise flag; individual shot/read/dark "
            "terms are recorded in provenance but not separately controlled yet."
        )
    trajectory_processing = subblocks.get("trajectory_processing")
    smear = (
        trajectory_processing.get("smear", {})
        if isinstance(trajectory_processing, Mapping)
        else {}
    )
    render = smear.get("render", {}) if isinstance(smear, Mapping) else {}
    smear_enabled = bool(smear.get("enabled", False)) if isinstance(smear, Mapping) else False
    render_mode = (
        str(render.get("mode", "metadata_only" if smear_enabled else "none"))
        if isinstance(render, Mapping)
        else "none"
    )
    if render_mode == "per_frame":
        add(
            "subblocks.trajectory_processing.smear.render.mode='per_frame' is reserved for "
            "future per-frame template/runtime detector-layer parameterization."
        )

    if isinstance(experiment.get("detector"), Mapping):
        detector = experiment["detector"]
        for key in ("pixel_offsets", "flat_field"):
            if key in detector:
                add(
                    f"detector.{key} is present but detector calibration decks are deferred and "
                    "not wired into the executable smoke wrapper."
                )

    observation_theta = _as_mapping(experiment.get("observation_theta"), name="experiment.observation_theta")
    optics_theta = observation_theta.get("optics", {}) if isinstance(observation_theta.get("optics"), Mapping) else {}
    high_order_requests = [key for key in ("high_order_wfe", "high_order_map", "high_order_map_pixels") if key in optics_theta]
    if high_order_requests:
        add(
            "observation_theta requests high-order map pixels "
            f"({', '.join(high_order_requests)}), but only source scalars, plate scale, and "
            "low-order Zernike coefficients are optimizer-visible today."
        )
    return warnings_out


def _normalize_subblock_noise_for_observation_bias(subblocks: Mapping[str, Any]) -> dict[str, Any]:
    """Map structured review noise requests onto the current legacy runner flag.

    The subblock runner currently accepts only ``--noise enabled|disabled|inherit``.
    Keep the detailed per-term request in metadata so audits/review notebooks can
    report which terms are requested and which are not separately controllable.
    """

    out = dict(subblocks)
    noise = out.get("noise")
    if isinstance(noise, Mapping):
        noise_cfg = normalize_noise_request(noise)
        enabled = bool(noise_cfg.get("enabled", False))
        out["noise_model"] = {
            "schema_version": "structured_noise_request.v1",
            "original_request": dict(noise),
            "requested": noise_cfg,
            "normalized": noise_cfg,
            "render_template_terms": {
                "enabled": enabled,
                "photon_noise": bool(noise_cfg.get("shot_noise", False)),
                "shot_noise": bool(noise_cfg.get("shot_noise", False)),
                "read_noise": bool(noise_cfg.get("read_noise", False)),
                "dark_current": bool(noise_cfg.get("dark_current", False)),
                "write_variance": bool(noise_cfg.get("write_variance", True)),
                "variance_floor": noise_cfg.get("variance_floor"),
            },
            "legacy_runner_flag": "enabled" if enabled else "disabled",
            "separate_term_control": False,
            "warnings": [
                "Structured shot/read/dark-current settings are translated to the "
                "legacy subblock --noise enabled/disabled flag; separate per-term "
                "runner controls are not implemented yet."
            ],
        }
        out["noise"] = "enabled" if enabled else "disabled"
    else:
        out["noise_model"] = {
            "schema_version": "structured_noise_request.v1",
            "original_request": noise,
            "requested": normalize_noise_request(noise),
            "normalized": normalize_noise_request(noise),
            "legacy_runner_flag": str(noise or "disabled"),
            "separate_term_control": False,
            "warnings": [],
        }
    return out


def _full_fidelity_to_observation_bias(config: Mapping[str, Any], *, run_name: str | None) -> dict[str, Any]:
    experiment = _as_mapping(config.get("experiment", config), name="experiment")
    if str(experiment.get("kind", "")) == "observation_bias_campaign":
        out = {"experiment": dict(experiment)}
        if run_name is not None:
            out["experiment"]["run_name"] = run_name
        return out
    kind = str(experiment.get("kind"))
    if kind not in FULL_FIDELITY_EXECUTABLE_KINDS:
        if str(experiment.get("kind")) == FUTURE_SKELETON_KIND:
            raise ValueError(
                "experiment.kind='full_fidelity_algorithm_campaign' is a non-executable "
                "schema/design skeleton. Use experiment.kind='full_fidelity_binary_iterative' "
                "with this wrapper, or implement a future runner for that schema."
            )
        raise ValueError(
            "Full-fidelity wrapper expects experiment.kind='full_fidelity_binary_iterative' "
            "or an already translated "
            "observation_bias_campaign config. "
            "experiment.kind='full_fidelity_algorithm_campaign' is not accepted."
        )
    source_alias = kind if kind in DEPRECATED_FULL_FIDELITY_ALIASES else None
    if source_alias is not None:
        warnings.warn(
            f"experiment.kind={source_alias!r} is deprecated; use "
            f"{CANONICAL_FULL_FIDELITY_KIND!r}.",
            DeprecationWarning,
            stacklevel=2,
        )
    canonical_kind = CANONICAL_FULL_FIDELITY_KIND

    subblocks = _normalize_subblock_noise_for_observation_bias(
        _as_mapping(experiment.get("subblocks"), name="experiment.subblocks")
    )
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
            "source_campaign_kind": canonical_kind,
            **({"source_campaign_alias": source_alias} if source_alias else {}),
            "schema_version": f"{canonical_kind}.translated.v1",
            "seed": int(experiment.get("seed", 42)),
            "run_name": run_name or experiment.get("run_name", canonical_kind),
            "system": {
                "preset": experiment.get("system_preset", DEFAULT_FULL_FIDELITY_SYSTEM_PRESET),
                "source": {"kind": source_kind, "target": target},
            },
            "detector_overrides": _as_mapping(experiment.get("detector_overrides"), name="experiment.detector_overrides"),
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
    validate_full_fidelity_smoke_config(raw, emit_warnings=True)
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
    parser = argparse.ArgumentParser(
        description=(
            "Translate and run the thin full-fidelity binary iterative wrapper. "
            "Accepted experiment.kind values are 'full_fidelity_binary_iterative_review', "
            "'full_fidelity_binary_iterative_smoke', and 'observation_bias_campaign'; "
            "the future 'full_fidelity_algorithm_campaign' skeleton is not executable here. "
            "Start with --dry-run."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Review/smoke YAML or already translated observation-bias config.")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT, help="Directory that will contain the campaign run root.")
    parser.add_argument("--run-name", default=None, help="Override experiment.run_name.")
    parser.add_argument("--dry-run", action="store_true", help="Write translated plan/artifacts without running sub-block inference.")
    parser.add_argument("--aggregate-only", action="store_true", help="Replay aggregation from an existing dry-run/execution run root.")
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
