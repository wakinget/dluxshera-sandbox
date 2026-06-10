"""Prepare trajectory-driven observation subblock campaign artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from dluxshera.config.io import load_config_file
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.campaigns import write_csv_rows, write_json, write_shell_command
from dluxshera.utils.obs_subblock_cli import append_reference_optimizer_flags
from dluxshera.utils.obs_subblock_io import now_iso_local_ms
from dluxshera.utils.obs_subblock_trajectory import (
    DEFAULT_OUTPUT_KEYS,
    TRAJECTORY_NOTES,
    prepare_airbus_subblocks,
    write_subblock_artifacts,
)
from dluxshera.utils.trajectory_smear import (
    inject_subblock_smear_layer,
    parse_smear_config,
    write_smear_sidecars,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "trajectory_subblock_campaign"
DEFAULT_AIRBUS_CSV = (
    REPO_ROOT / "src" / "dluxshera" / "data" / "airbus_data" / "Thirty_Min_Observation_Window.csv"
)
DEFAULT_RENDER_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_template"
    / "subblock_generation_prescription.yaml"
)
DEFAULT_INFERENCE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_inference_template"
    / "subblock_inference_prescription.yaml"
)


def _load_case_runner_module():
    module_path = REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_case.py"
    spec = importlib.util.spec_from_file_location(
        "trajectory_campaign_case_runner",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load case runner at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_local_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _patch_inference_config(
    *,
    template_path: Path,
    config_dir: Path,
    case_root: Path,
    frame_truth_path: Path,
    starting_guess_path: Path,
    active_frame_keys: Sequence[str],
    output_keys: Sequence[str],
    source_kind: str,
    system_preset: str,
    phi_ref: str,
    reference_optimizer: dict[str, Any],
) -> dict[str, Any]:
    case_module = _load_case_runner_module()
    cfg = load_config_file(template_path)
    system_cfg = case_module._ensure_mapping(cfg, "system", path="root")
    system_cfg["preset"] = system_preset
    if source_kind == "single_star":
        source_cfg = case_module._ensure_mapping(system_cfg, "source", path="system")
        source_cfg["target"] = "SINGLE_STAR"

    experiment_cfg = case_module._ensure_mapping(cfg, "experiment", path="root")
    outputs_cfg = case_module._ensure_mapping(experiment_cfg, "outputs", path="experiment")
    outputs_cfg["outdir"] = str(case_root)
    inference_cfg = case_module._ensure_mapping(experiment_cfg, "inference", path="experiment")
    data_cfg = case_module._ensure_mapping(inference_cfg, "data", path="experiment.inference")
    data_cfg["cube"] = "__CASE_RENDER_CUBE_PATH_UNRESOLVED__"
    data_cfg["truth_trace"] = case_module._path_for_config(
        frame_truth_path.resolve(),
        config_dir=config_dir,
    )

    active_cfg = case_module._ensure_mapping(inference_cfg, "active", path="experiment.inference")
    active_cfg["frame_keys"] = list(active_frame_keys)
    active_cfg.setdefault("shared_keys", [])

    init_cfg = case_module._ensure_mapping(inference_cfg, "init", path="experiment.inference")
    frame_init_cfg = case_module._ensure_mapping(init_cfg, "frame", path="experiment.inference.init")
    frame_init_cfg.clear()
    frame_init_cfg["mode"] = "starting_guess_csv"
    frame_init_cfg["path"] = case_module._path_for_config(
        starting_guess_path.resolve(),
        config_dir=config_dir,
    )
    frame_init_cfg["columns"] = {
        key: f"{key}_linear_fit" for key in active_frame_keys if key in output_keys
    }
    init_cfg.setdefault("shared", {})

    optimizer_cfg = case_module._ensure_mapping(
        inference_cfg,
        "optimizer",
        path="experiment.inference",
    )
    if reference_optimizer:
        if "kind" in reference_optimizer:
            optimizer_cfg["kind"] = reference_optimizer["kind"]
        if "base_lr" in reference_optimizer:
            optimizer_cfg["base_lr"] = float(reference_optimizer["base_lr"])
        if "n_iter" in reference_optimizer:
            optimizer_cfg["n_iter"] = int(reference_optimizer["n_iter"])
        if "schedule" in reference_optimizer:
            optimizer_cfg["schedule"] = dict(reference_optimizer["schedule"])
        if "preconditioning" in reference_optimizer:
            optimizer_cfg["preconditioning"] = dict(reference_optimizer["preconditioning"])

    diagnostics_cfg = case_module._ensure_mapping(
        inference_cfg,
        "diagnostics",
        path="experiment.inference",
    )
    diagnostics_cfg["compare_to_truth_when_available"] = phi_ref == "truth_when_available"
    return cfg


def _build_child_command(
    *,
    subblock_dir: Path,
    frame_truth_path: Path,
    starting_guess_path: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "examples" / "scripts" / "run_obs_subblock_study.py"),
        "--mode",
        "schur_summary",
        "--case-root",
        str(subblock_dir),
        "--external-frame-truth-csv",
        str(frame_truth_path),
        "--starting-guess-csv",
        str(starting_guess_path),
        "--starting-guess-mode",
        "starting_guess_csv",
        "--phi-ref",
        str(args.phi_ref),
        "--summary-information-scale",
        "summed_likelihood",
        "--schur-curvature-method",
        "auto",
        "--max-dense-dim",
        str(args.max_dense_dim),
        "--schur-frame-quality-policy",
        "mask",
        "--schur-frame-chi2-threshold",
        "5.0",
        "--schur-frame-mask-denominator",
        "original",
        "--noise",
        str(args.noise),
        "--n-frames",
        str(args.n_frames_per_subblock),
        "--dt-s",
        str(args.frame_dt_s),
    ]
    if args.render_template is not None:
        command.extend(["--render-template", str(args.render_template)])
    if args.inference_template is not None:
        command.extend(["--inference-template", str(args.inference_template)])
    optimizer_flags = {
        "reference_optimizer_kind": args.reference_optimizer.get("kind"),
        "reference_base_lr": args.reference_optimizer.get("base_lr"),
        "reference_n_iter": args.reference_optimizer.get("n_iter"),
        "reference_schedule": args.reference_optimizer.get("schedule"),
    }
    preconditioning = args.reference_optimizer.get("preconditioning")
    if isinstance(preconditioning, dict):
        optimizer_flags.update(
            {
                "reference_preconditioning_enabled": preconditioning.get("enabled"),
                "reference_preconditioning_method": preconditioning.get("method"),
                "reference_preconditioning_reference": preconditioning.get("reference"),
            }
        )
    append_reference_optimizer_flags(command, optimizer_flags)
    return command


def _default_active_frame_keys(source_kind: str, output_keys: Sequence[str]) -> tuple[str, ...]:
    if source_kind == "single_star":
        requested = ("source.x_position_as", "source.y_position_as")
    else:
        requested = (
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        )
    return tuple(key for key in requested if key in output_keys)


def _parse_output_keys(raw: str) -> tuple[str, ...]:
    keys = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not keys:
        raise ValueError("At least one trajectory output key is required.")
    return keys


def _plate_scale_from_system_preset(system_preset: str) -> float | None:
    try:
        resolved = resolve_config({"system": {"preset": system_preset}})
        system_cfg = resolved.get("system", {})
        spec = compose_forward_spec(system_cfg)
        store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
        return float(store.get("optics.plate_scale_as_per_pix"))
    except Exception:
        return None


def _trajectory_processing_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    if args.trajectory_processing_config is not None:
        loaded = load_config_file(args.trajectory_processing_config)
        if "trajectory_processing" in loaded:
            cfg = dict(loaded["trajectory_processing"] or {})
        elif "smear" in loaded:
            cfg = {"smear": dict(loaded["smear"] or {})}
        else:
            cfg = dict(loaded or {})
    if bool(args.smear_enabled):
        smear = dict(cfg.get("smear", {}) or {})
        smear["enabled"] = True
        render = dict(smear.get("render", {}) or {})
        if args.smear_render_mode is not None:
            render["mode"] = args.smear_render_mode
        if render:
            smear["render"] = render
        cfg["smear"] = smear
    return cfg


def run_trajectory_subblock_campaign(args: argparse.Namespace) -> dict[str, Any]:
    run_root = (args.results_root / args.run_name).resolve()
    subblocks_root = run_root / "subblocks"
    output_keys = _parse_output_keys(args.output_keys)
    active_frame_keys = tuple(args.active_frame_keys or _default_active_frame_keys(args.source_kind, output_keys))
    trajectory_processing_cfg = _trajectory_processing_config(args)
    smear_cfg = parse_smear_config(
        trajectory_processing_cfg,
        exposure_time_s=float(args.frame_dt_s),
        plate_scale_as_per_pix=_plate_scale_from_system_preset(args.system_preset),
    )

    trajectory, frame_times, blocks = prepare_airbus_subblocks(
        path=args.trajectory_csv,
        start_s=float(args.start_s),
        duration_s=args.duration_s,
        n_subblocks=args.n_subblocks,
        sample_dt_s=float(args.sample_dt_s),
        frame_dt_s=float(args.frame_dt_s),
        subblock_duration_s=float(args.subblock_duration_s),
        n_frames_per_subblock=int(args.n_frames_per_subblock),
        output_keys=output_keys,
        fit_keys=active_frame_keys,
        time_mode="inferred_uniform",
        time_start_s=0.0,
    )

    run_root.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "experiment": {
            "kind": "trajectory_subblock_campaign",
            "run_name": args.run_name,
            "seed": args.seed,
        },
        "trajectory": {
            "source": {
                "kind": "airbus_csv",
                "path": str(args.trajectory_csv.resolve()),
                "sample_dt_s": args.sample_dt_s,
                "time": {"mode": "inferred_uniform", "start_s": 0.0},
            },
            "window": {
                "start_s": args.start_s,
                "duration_s": args.duration_s,
                "n_subblocks": args.n_subblocks,
            },
            "sampling": {
                "frame_dt_s": args.frame_dt_s,
                "subblock_duration_s": args.subblock_duration_s,
                "n_frames_per_subblock": args.n_frames_per_subblock,
                "interpolation": "linear",
            },
            "output_keys": list(output_keys),
            "starting_guess": {
                "kind": "per_subblock_linear_fit",
                "fit_keys": list(active_frame_keys),
                "time_origin": "subblock_start",
            },
        },
        "trajectory_processing": trajectory_processing_cfg,
        "subblock": {
            "source_kind": args.source_kind,
            "system_preset": args.system_preset,
            "noise": args.noise,
            "phi_ref": args.phi_ref,
            "summary_information_scale": "summed_likelihood",
        },
        "execution": {
            "dry_run": bool(args.dry_run),
            "run_children": bool(args.run_children),
            "max_workers": int(args.max_workers),
        },
    }
    write_json(run_root / "resolved_config.json", resolved_config)

    ingest_summary = {
        "schema_version": "trajectory_ingest_summary.v1",
        "source_kind": trajectory.raw.source_kind,
        "source_path": str(trajectory.raw.source_path),
        "raw_sample_count": trajectory.raw.sample_count,
        "raw_time_span_s": list(trajectory.raw.span),
        "selected_time_span_s": [float(frame_times[0]), float(frame_times[-1])],
        "mapping": trajectory.mapping,
        "output_keys": list(output_keys),
        "smear": {
            "enabled": bool(smear_cfg.enabled),
            "render_mode": smear_cfg.render_mode if smear_cfg.enabled else "disabled",
            "inference_mode": smear_cfg.inference_mode if smear_cfg.enabled else "disabled",
        },
    }
    write_json(run_root / "trajectory_ingest_summary.json", ingest_summary)

    case_module = _load_case_runner_module()
    subblock_rows: list[dict[str, Any]] = []
    child_results: list[dict[str, Any]] = []
    for block in blocks:
        subblock_dir = subblocks_root / f"subblock_{block.subblock_index:06d}"
        artifacts = write_subblock_artifacts(block, outdir=subblock_dir, output_keys=output_keys)
        frame_truth_path = artifacts["frame_truth_csv"]
        starting_guess_path = artifacts["starting_guess_prediction_csv"]
        smear_artifacts: dict[str, Any] = {}
        if smear_cfg.enabled:
            smear_artifacts = write_smear_sidecars(
                outdir=subblock_dir,
                trajectory=trajectory,
                block=block,
                cfg=smear_cfg,
                processing_context={"caller": "run_trajectory_subblock_campaign"},
            )

        trace_config = {
            "schema_version": "trajectory_external_frame_truth.v1",
            "kind": "external_frame_truth_csv",
            "frame_truth_csv": str(frame_truth_path),
            "source": "trajectory_subblock_campaign",
            "smear_truth_csv": str(smear_artifacts.get("smear_truth_csv", "")),
            "smear_model_csv": str(smear_artifacts.get("smear_model_csv", "")),
        }
        _write_local_json(subblock_dir / "trace_config.json", trace_config)

        render_cfg = case_module.build_render_case_config(
            template_path=args.render_template,
            config_dir=subblock_dir,
            case_root=subblock_dir,
            trace_input=case_module.ResolvedInput(frame_truth_path, "trajectory_frame_truth"),
            exposure_time_s=args.frame_dt_s,
            noise_mode=args.noise,
            render_seed=args.seed + block.subblock_index,
        )
        render_cfg.setdefault("system", {})["preset"] = args.system_preset
        if smear_cfg.enabled and smear_cfg.render_mode == "subblock_constant_layer":
            render_cfg = inject_subblock_smear_layer(
                render_cfg,
                cfg=smear_cfg,
                representative_kernel=smear_artifacts["representative_kernel"],
            )
        _write_local_json(subblock_dir / "render_config.json", render_cfg)

        inference_cfg = _patch_inference_config(
            template_path=args.inference_template,
            config_dir=subblock_dir,
            case_root=subblock_dir,
            frame_truth_path=frame_truth_path,
            starting_guess_path=starting_guess_path,
            active_frame_keys=active_frame_keys,
            output_keys=output_keys,
            source_kind=args.source_kind,
            system_preset=args.system_preset,
            phi_ref=args.phi_ref,
            reference_optimizer=args.reference_optimizer,
        )
        _write_local_json(subblock_dir / "inference_config.json", inference_cfg)

        command = _build_child_command(
            subblock_dir=subblock_dir,
            frame_truth_path=frame_truth_path,
            starting_guess_path=starting_guess_path,
            args=args,
        )
        command_path = subblock_dir / "command.sh"
        write_shell_command(command_path, command, env_prefix={"PYTHONPATH": "src"})
        expected_summary = subblock_dir / "study" / "schur_summary" / "subblock_summary.json"

        row: dict[str, Any] = {
            "subblock_index": block.subblock_index,
            "time_start_s": block.time_start_s,
            "time_end_s": block.time_end_s,
            "n_frames": block.n_frames,
            "frame_truth_path": str(frame_truth_path),
            "starting_guess_prediction_path": str(starting_guess_path),
            "render_config_path": str((subblock_dir / "render_config.json").resolve()),
            "inference_config_path": str((subblock_dir / "inference_config.json").resolve()),
            "command_path": str(command_path.resolve()),
            "expected_summary_json": str(expected_summary.resolve()),
            "status": "planned",
            "smear_enabled": bool(smear_cfg.enabled),
            "smear_truth_csv": str(smear_artifacts.get("smear_truth_csv", "")),
            "smear_model_csv": str(smear_artifacts.get("smear_model_csv", "")),
            "smear_provenance_json": str(smear_artifacts.get("smear_provenance_json", "")),
            "smear_truth_length_pix_median": smear_artifacts.get("smear_truth_length_pix_median", ""),
            "smear_truth_length_pix_max": smear_artifacts.get("smear_truth_length_pix_max", ""),
            "smear_model_length_pix_median": smear_artifacts.get("smear_model_length_pix_median", ""),
            "smear_model_policy": smear_cfg.inference_mode if smear_cfg.enabled else "",
            "smear_render_mode": smear_cfg.render_mode if smear_cfg.enabled else "",
            "smear_layer_name": smear_cfg.render_layer_name if smear_cfg.enabled else "",
        }
        for key, diag in block.diagnostics.items():
            row[f"rms_{key}_residual"] = diag["rms_residual"]
            row[f"max_abs_{key}_residual"] = diag["max_abs_residual"]
        subblock_rows.append(row)

        if args.run_children and not args.dry_run:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": "src"},
                check=False,
                text=True,
                capture_output=True,
            )
            row["status"] = "completed" if completed.returncode == 0 else "failed"
            child_results.append(
                {
                    "subblock_index": block.subblock_index,
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                }
            )
            if completed.returncode != 0:
                break

    write_csv_rows(run_root / "subblock_plan.csv", subblock_rows)
    smear_summary = {
        "schema_version": "trajectory_smear_summary.v1",
        "enabled": bool(smear_cfg.enabled),
        "render_mode": smear_cfg.render_mode if smear_cfg.enabled else "disabled",
        "inference_mode": smear_cfg.inference_mode if smear_cfg.enabled else "disabled",
        "subblocks": [
            {
                "subblock_index": row["subblock_index"],
                "smear_truth_csv": row.get("smear_truth_csv", ""),
                "smear_model_csv": row.get("smear_model_csv", ""),
                "smear_provenance_json": row.get("smear_provenance_json", ""),
                "smear_truth_length_pix_median": row.get("smear_truth_length_pix_median", ""),
                "smear_truth_length_pix_max": row.get("smear_truth_length_pix_max", ""),
            }
            for row in subblock_rows
        ],
    }
    write_json(run_root / "smear_summary.json", smear_summary)
    campaign_plan = {
        "schema_version": "trajectory_subblock_campaign_plan.v1",
        "created_at": now_iso_local_ms(),
        "run_root": str(run_root),
        "source_trajectory_path": str(args.trajectory_csv.resolve()),
        "source_kind": "airbus_csv",
        "raw_sample_count": trajectory.raw.sample_count,
        "raw_time_span_s": list(trajectory.raw.span),
        "selected_time_span_s": [float(frame_times[0]), float(frame_times[-1])],
        "interpolation_method": "linear",
        "frame_dt_s": args.frame_dt_s,
        "subblock_duration_s": args.subblock_duration_s,
        "n_frames_per_subblock": args.n_frames_per_subblock,
        "n_subblocks": len(blocks),
        "output_keys": list(output_keys),
        "active_frame_keys": list(active_frame_keys),
        "source_mode": args.source_kind,
        "subblocks": subblock_rows,
        "smear": smear_summary,
        "notes": list(TRAJECTORY_NOTES),
        "child_results": child_results,
    }
    write_json(run_root / "campaign_plan.json", campaign_plan)
    return campaign_plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Airbus trajectory-driven observation subblock artifacts."
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--trajectory-csv", type=Path, default=DEFAULT_AIRBUS_CSV)
    parser.add_argument("--sample-dt-s", type=float, default=0.1)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=None)
    parser.add_argument("--n-subblocks", type=int, default=None)
    parser.add_argument("--frame-dt-s", type=float, default=0.05)
    parser.add_argument("--subblock-duration-s", type=float, default=1.0)
    parser.add_argument("--n-frames-per-subblock", type=int, default=20)
    parser.add_argument(
        "--output-keys",
        default=",".join(DEFAULT_OUTPUT_KEYS),
        help="Comma-separated canonical trajectory output keys.",
    )
    parser.add_argument(
        "--active-frame-key",
        dest="active_frame_keys",
        action="append",
        default=None,
        help="Repeatable active frame key override. Defaults depend on source-kind.",
    )
    parser.add_argument("--source-kind", choices=("binary", "single_star"), default="binary")
    parser.add_argument("--system-preset", default="SHERA_FLIGHT_3P")
    parser.add_argument("--noise", choices=("inherit", "enabled", "disabled"), default="enabled")
    parser.add_argument(
        "--phi-ref",
        choices=("recovered", "truth_when_available", "truth", "init"),
        default="recovered",
    )
    parser.add_argument("--max-dense-dim", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-template", type=Path, default=DEFAULT_RENDER_TEMPLATE)
    parser.add_argument("--inference-template", type=Path, default=DEFAULT_INFERENCE_TEMPLATE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-children", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--trajectory-processing-config", type=Path, default=None)
    parser.add_argument("--smear-enabled", action="store_true")
    parser.add_argument(
        "--smear-render-mode",
        choices=("metadata_only", "subblock_constant_layer"),
        default=None,
    )
    parser.set_defaults(
        reference_optimizer={
            "kind": "sgd",
            "base_lr": 0.7,
            "n_iter": 80,
            "schedule": {"kind": "linear_warmup", "warmup_steps": 10, "start_factor": 0.125},
            "preconditioning": {"enabled": True, "method": "auto", "reference": "initial"},
        }
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    if args.duration_s is None and args.n_subblocks is None:
        args.duration_s = 2.0
    plan = run_trajectory_subblock_campaign(args)
    print(f"Run root: {plan['run_root']}")
    print(f"Subblocks: {plan['n_subblocks']}")
    print(f"Campaign plan: {Path(plan['run_root']) / 'campaign_plan.json'}")
    return plan


if __name__ == "__main__":
    main()
