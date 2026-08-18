"""Trace-source planning helpers shared by campaign wrappers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .obs_subblock_trajectory import (
    DEFAULT_OUTPUT_KEYS,
    TRAJECTORY_NOTES,
    parse_trajectory_offsets_config,
    prepare_airbus_subblocks,
    write_subblock_artifacts,
    write_trajectory_filter_artifacts,
)
from .trajectory_filters import parse_trajectory_filter_config
from .trajectory_smear import (
    parse_smear_config,
    write_smear_sidecars,
)


TRACE_SOURCE_MODE_IID = "iid_jitter"
TRACE_SOURCE_MODE_TRAJECTORY = "trajectory"
TRACE_SOURCE_MODE_EXTERNAL_PLAN = "external_plan"
SUPPORTED_TRACE_SOURCE_MODES = (
    TRACE_SOURCE_MODE_IID,
    TRACE_SOURCE_MODE_TRAJECTORY,
    TRACE_SOURCE_MODE_EXTERNAL_PLAN,
)


@dataclass(frozen=True)
class PreparedTraceSubblock:
    subblock_index: int
    frame_truth_path: Path | None
    starting_guess_prediction_path: Path | None
    smear_truth_path: Path | None
    smear_model_path: Path | None
    smear_provenance_path: Path | None
    trace_source_mode: str
    time_start_s: float | None
    time_end_s: float | None
    n_frames: int
    output_keys: tuple[str, ...]
    active_frame_keys: tuple[str, ...]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class PreparedTraceSourcePlan:
    mode: str
    run_root: Path
    source_kind: str
    output_keys: tuple[str, ...]
    active_frame_keys: tuple[str, ...]
    subblocks: tuple[PreparedTraceSubblock, ...]
    summary: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def _resolve_path(value: Any, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty path string.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _as_key_tuple(value: Any, *, default: Sequence[str], field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple(str(key) for key in default)
    if isinstance(value, str):
        keys = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Sequence):
        keys = tuple(str(item).strip() for item in value if str(item).strip())
    else:
        raise ValueError(f"{field_name} must be a list[str] or comma-separated string.")
    if not keys:
        raise ValueError(f"{field_name} must contain at least one key.")
    return keys


def _iid_plan(
    *,
    run_root: Path,
    source_kind: str,
    active_frame_keys: Sequence[str],
    n_subblocks: int,
    n_frames_per_subblock: int,
    default_output_keys: Sequence[str],
    trace_source_cfg: Mapping[str, Any],
) -> PreparedTraceSourcePlan:
    output_keys = _as_key_tuple(
        trace_source_cfg.get("output_keys"),
        default=default_output_keys,
        field_name="subblocks.trace_source.output_keys",
    )
    rows: list[dict[str, Any]] = []
    subblocks: list[PreparedTraceSubblock] = []
    for index in range(int(n_subblocks)):
        row = {
            "trace_source_mode": TRACE_SOURCE_MODE_IID,
            "frame_truth_path": "",
            "starting_guess_prediction_path": "",
            "trajectory_source_kind": "",
            "trajectory_source_path": "",
            "trajectory_subblock_index": int(index),
            "trajectory_output_keys": ",".join(output_keys),
            "trajectory_active_frame_keys": ",".join(active_frame_keys),
        }
        rows.append(row)
        subblocks.append(
            PreparedTraceSubblock(
                subblock_index=index,
                frame_truth_path=None,
                starting_guess_prediction_path=None,
                smear_truth_path=None,
                smear_model_path=None,
                smear_provenance_path=None,
                trace_source_mode=TRACE_SOURCE_MODE_IID,
                time_start_s=None,
                time_end_s=None,
                n_frames=int(n_frames_per_subblock),
                output_keys=output_keys,
                active_frame_keys=tuple(active_frame_keys),
                diagnostics={},
                provenance=dict(row),
            )
        )
    return PreparedTraceSourcePlan(
        mode=TRACE_SOURCE_MODE_IID,
        run_root=run_root,
        source_kind=source_kind,
        output_keys=output_keys,
        active_frame_keys=tuple(active_frame_keys),
        subblocks=tuple(subblocks),
        summary={
            "mode": TRACE_SOURCE_MODE_IID,
            "source_kind": source_kind,
            "legacy_trace_jitter_preserved": True,
            "notes": ["Existing IID jitter trace-template behavior is preserved."],
        },
        rows=tuple(rows),
    )


def _trajectory_plan(
    *,
    trace_source_cfg: Mapping[str, Any],
    run_root: Path,
    artifact_root: Path,
    source_kind: str,
    active_frame_keys: Sequence[str],
    n_subblocks: int,
    n_frames_per_subblock: int,
    frame_dt_s: float,
    subblock_duration_s: float,
    default_output_keys: Sequence[str],
    reuse_existing: bool,
    trajectory_processing_cfg: Mapping[str, Any] | None,
    plate_scale_as_per_pix: float | None,
) -> PreparedTraceSourcePlan:
    source_cfg = dict(trace_source_cfg.get("source", {}) or {})
    source_kind_value = str(source_cfg.get("kind", "airbus_csv"))
    source_format = str(source_cfg.get("format", "airbus_xyz_arcsec"))
    if source_kind_value == "csv" and source_format != "airbus_xyz_arcsec":
        raise ValueError(
            "subblocks.trace_source.source.kind='csv' currently supports only "
            "format='airbus_xyz_arcsec'."
        )
    if source_kind_value not in {"airbus_csv", "csv"}:
        raise ValueError(
            "Only subblocks.trace_source.source.kind='airbus_csv' or "
            "kind='csv' with format='airbus_xyz_arcsec' is supported."
        )
    source_path = _resolve_path(
        source_cfg.get(
            "path",
            "src/dluxshera/data/airbus_data/Thirty_Min_Observation_Window.csv",
        ),
        field_name="subblocks.trace_source.source.path",
    )
    sample_dt_s = float(source_cfg.get("sample_dt_s", 0.1))
    window_cfg = dict(trace_source_cfg.get("window", {}) or {})
    sampling_cfg = dict(trace_source_cfg.get("sampling", {}) or {})
    processing_cfg = dict(trace_source_cfg.get("processing", {}) or {})
    legacy_processing_cfg = dict(trajectory_processing_cfg or {})
    filter_cfg = dict(
        processing_cfg.get("filter")
        or legacy_processing_cfg.get("filter")
        or legacy_processing_cfg.get("high_pass_filter")
        or {}
    )
    offsets_cfg = dict(
        processing_cfg.get("offsets")
        if isinstance(processing_cfg.get("offsets"), Mapping)
        else legacy_processing_cfg.get("offsets")
        if isinstance(legacy_processing_cfg.get("offsets"), Mapping)
        else {}
    )
    filter_spec = parse_trajectory_filter_config(filter_cfg)
    start_s = float(window_cfg.get("start_s", 0.0))
    requested_n_subblocks = int(window_cfg.get("n_subblocks", n_subblocks))
    if requested_n_subblocks != int(n_subblocks):
        raise ValueError(
            "subblocks.trace_source.window.n_subblocks must match subblocks.n_subblocks "
            "for integrated campaign trajectory mode."
        )
    output_keys = _as_key_tuple(
        trace_source_cfg.get("output_keys"),
        default=default_output_keys,
        field_name="subblocks.trace_source.output_keys",
    )
    parsed_offsets = parse_trajectory_offsets_config(
        offsets_cfg,
        available_keys=output_keys,
    )
    starting_guess_cfg = dict(trace_source_cfg.get("starting_guess", {}) or {})
    fit_keys = _as_key_tuple(
        starting_guess_cfg.get("fit_keys"),
        default=active_frame_keys,
        field_name="subblocks.trace_source.starting_guess.fit_keys",
    )
    missing_active = [key for key in active_frame_keys if key not in output_keys]
    if missing_active:
        raise ValueError(
            "Trace-source output_keys must include all active frame keys; missing: "
            + ", ".join(missing_active)
        )

    resolved_frame_dt_s = float(sampling_cfg.get("frame_dt_s", frame_dt_s))
    resolved_subblock_duration_s = float(
        sampling_cfg.get("subblock_duration_s", subblock_duration_s)
    )
    resolved_n_frames = int(
        sampling_cfg.get("n_frames_per_subblock", n_frames_per_subblock)
    )
    if resolved_n_frames != int(n_frames_per_subblock):
        raise ValueError(
            "trace_source sampling.n_frames_per_subblock must match subblocks.n_frames."
        )
    if abs(resolved_frame_dt_s - float(frame_dt_s)) > 1.0e-12:
        raise ValueError("trace_source sampling.frame_dt_s must match subblock frame cadence.")
    smear_cfg = parse_smear_config(
        trajectory_processing_cfg,
        exposure_time_s=float(frame_dt_s),
        plate_scale_as_per_pix=plate_scale_as_per_pix,
    )

    trajectory_root = artifact_root
    if reuse_existing:
        missing: list[Path] = []
        if filter_spec.enabled and filter_spec.kind != "none":
            for filename in (
                "trajectory_raw.csv",
                "trajectory_filtered.csv",
                "trajectory_filter_provenance.json",
                "trajectory_filter_summary.csv",
            ):
                path = trajectory_root / filename
                if not path.exists():
                    missing.append(path)
        if parsed_offsets:
            for filename in (
                "trajectory_offset_provenance.json",
                "trajectory_offset_summary.csv",
            ):
                path = trajectory_root / filename
                if not path.exists():
                    missing.append(path)
        for index in range(int(n_subblocks)):
            for filename in ("frame_truth.csv", "starting_guess_prediction.csv"):
                path = trajectory_root / f"subblock_{index:06d}" / filename
                if not path.exists():
                    missing.append(path)
            if smear_cfg.enabled:
                for filename in (
                    "frame_smear_truth.csv",
                    "frame_smear_model.csv",
                    "smear_provenance.json",
                ):
                    path = trajectory_root / f"subblock_{index:06d}" / filename
                    if not path.exists():
                        missing.append(path)
        if missing:
            raise FileNotFoundError(
                "Stored trajectory trace-source artifacts are missing: "
                + ", ".join(str(path) for path in missing[:5])
            )

    trajectory, frame_times, blocks = prepare_airbus_subblocks(
        path=source_path,
        start_s=start_s,
        duration_s=window_cfg.get("duration_s"),
        n_subblocks=requested_n_subblocks,
        sample_dt_s=sample_dt_s,
        frame_dt_s=resolved_frame_dt_s,
        subblock_duration_s=resolved_subblock_duration_s,
        n_frames_per_subblock=resolved_n_frames,
        output_keys=output_keys,
        fit_keys=fit_keys,
        interpolation=str(sampling_cfg.get("interpolation", "linear")),
        filter_config=filter_cfg,
        offsets_config=parsed_offsets,
    )
    filter_artifacts: dict[str, Any] = {}
    offsets_enabled = bool(trajectory.offset_provenance and trajectory.offset_provenance.get("enabled"))
    if (filter_spec.enabled and filter_spec.kind != "none") or offsets_enabled:
        if not reuse_existing:
            filter_artifacts = write_trajectory_filter_artifacts(
                outdir=trajectory_root,
                trajectory=trajectory,
            )
        else:
            filter_artifacts = {
                "trajectory_raw_csv": (trajectory_root / "trajectory_raw.csv").resolve(),
                "trajectory_filtered_csv": (trajectory_root / "trajectory_filtered.csv").resolve(),
                "trajectory_filter_provenance_json": (
                    trajectory_root / "trajectory_filter_provenance.json"
                ).resolve(),
                "trajectory_filter_summary_csv": (
                    trajectory_root / "trajectory_filter_summary.csv"
                ).resolve(),
                "trajectory_filter_diagnostic_png": (
                    trajectory_root / "trajectory_filter_diagnostic.png"
                ).resolve(),
            }
            if offsets_enabled:
                filter_artifacts.update(
                    {
                        "trajectory_offset_provenance_json": (
                            trajectory_root / "trajectory_offset_provenance.json"
                        ).resolve(),
                        "trajectory_offset_summary_csv": (
                            trajectory_root / "trajectory_offset_summary.csv"
                        ).resolve(),
                    }
                )

    rows: list[dict[str, Any]] = []
    subblocks: list[PreparedTraceSubblock] = []
    for block in blocks:
        outdir = trajectory_root / f"subblock_{block.subblock_index:06d}"
        paths = {
            "frame_truth_csv": (outdir / "frame_truth.csv").resolve(),
            "starting_guess_prediction_csv": (
                outdir / "starting_guess_prediction.csv"
            ).resolve(),
        }
        smear_artifacts: dict[str, Any] = {}
        if not reuse_existing:
            paths = write_subblock_artifacts(block, outdir=outdir, output_keys=output_keys)
            if smear_cfg.enabled:
                smear_artifacts = write_smear_sidecars(
                    outdir=outdir,
                    trajectory=trajectory,
                    block=block,
                    cfg=smear_cfg,
                    processing_context={
                        "caller": "campaign_trace_sources",
                        "input_frame_truth_csv": str(paths["frame_truth_csv"]),
                    },
                )
        elif smear_cfg.enabled:
            smear_artifacts = {
                "smear_truth_csv": (outdir / "frame_smear_truth.csv").resolve(),
                "smear_model_csv": (outdir / "frame_smear_model.csv").resolve(),
                "smear_provenance_json": (outdir / "smear_provenance.json").resolve(),
            }
        diagnostics: dict[str, Any] = {}
        row: dict[str, Any] = {
            "trace_source_mode": TRACE_SOURCE_MODE_TRAJECTORY,
            "frame_truth_path": str(paths["frame_truth_csv"]),
            "starting_guess_prediction_path": str(paths["starting_guess_prediction_csv"]),
            "trajectory_source_kind": source_kind_value,
            "trajectory_source_format": source_format,
            "trajectory_source_path": str(source_path),
            "trajectory_window_start_s": float(frame_times[0]),
            "trajectory_window_end_s": float(frame_times[-1]),
            "trajectory_subblock_index": int(block.subblock_index),
            "trajectory_time_start_s": block.time_start_s,
            "trajectory_time_end_s": block.time_end_s,
            "trajectory_output_keys": ",".join(output_keys),
            "trajectory_active_frame_keys": ",".join(active_frame_keys),
            "trajectory_window_policy": "shared_across_cases",
            "smear_enabled": bool(smear_cfg.enabled),
            "smear_truth_csv": str(smear_artifacts.get("smear_truth_csv", "")),
            "smear_model_csv": str(smear_artifacts.get("smear_model_csv", "")),
            "smear_provenance_json": str(smear_artifacts.get("smear_provenance_json", "")),
            "smear_representative_kernel_json": (
                json.dumps(smear_artifacts.get("representative_kernel", {}), sort_keys=True)
                if smear_artifacts.get("representative_kernel")
                else ""
            ),
            "smear_truth_length_pix_median": smear_artifacts.get("smear_truth_length_pix_median", ""),
            "smear_truth_length_pix_max": smear_artifacts.get("smear_truth_length_pix_max", ""),
            "smear_model_length_pix_median": smear_artifacts.get("smear_model_length_pix_median", ""),
            "smear_model_policy": smear_cfg.inference_mode if smear_cfg.enabled else "",
            "smear_render_mode": smear_cfg.render_mode if smear_cfg.enabled else "",
            "smear_layer_name": smear_cfg.render_layer_name if smear_cfg.enabled else "",
            "trajectory_filter_enabled": bool(filter_spec.enabled and filter_spec.kind != "none"),
            "trajectory_filter_kind": filter_spec.kind if filter_spec.enabled else "",
            "trajectory_filter_method": filter_spec.method if filter_spec.enabled else "",
            "trajectory_filter_provenance_json": str(
                filter_artifacts.get("trajectory_filter_provenance_json", "")
            ),
            "trajectory_filtered_csv": str(filter_artifacts.get("trajectory_filtered_csv", "")),
            "trajectory_offsets_json": json.dumps(parsed_offsets, sort_keys=True),
            "trajectory_offsets_enabled": bool(offsets_enabled),
            "trajectory_offset_provenance_json": str(
                filter_artifacts.get("trajectory_offset_provenance_json", "")
            ),
            "trajectory_offset_summary_csv": str(
                filter_artifacts.get("trajectory_offset_summary_csv", "")
            ),
        }
        for key, diag in block.diagnostics.items():
            diagnostics[key] = dict(diag)
            row[f"rms_{key}_residual"] = diag["rms_residual"]
            row[f"max_abs_{key}_residual"] = diag["max_abs_residual"]
        rows.append(row)
        subblocks.append(
            PreparedTraceSubblock(
                subblock_index=int(block.subblock_index),
                frame_truth_path=paths["frame_truth_csv"],
                starting_guess_prediction_path=paths["starting_guess_prediction_csv"],
                smear_truth_path=smear_artifacts.get("smear_truth_csv"),
                smear_model_path=smear_artifacts.get("smear_model_csv"),
                smear_provenance_path=smear_artifacts.get("smear_provenance_json"),
                trace_source_mode=TRACE_SOURCE_MODE_TRAJECTORY,
                time_start_s=block.time_start_s,
                time_end_s=block.time_end_s,
                n_frames=block.n_frames,
                output_keys=output_keys,
                active_frame_keys=tuple(active_frame_keys),
                diagnostics=diagnostics,
                provenance=dict(row),
            )
        )

    return PreparedTraceSourcePlan(
        mode=TRACE_SOURCE_MODE_TRAJECTORY,
        run_root=run_root,
        source_kind=source_kind,
        output_keys=output_keys,
        active_frame_keys=tuple(active_frame_keys),
        subblocks=tuple(subblocks),
        summary={
            "mode": TRACE_SOURCE_MODE_TRAJECTORY,
            "source_kind": source_kind,
            "trajectory_source_kind": source_kind_value,
            "trajectory_source_format": source_format,
            "trajectory_source_path": str(source_path),
            "raw_sample_dt_s": sample_dt_s,
            "selected_time_span_s": [float(frame_times[0]), float(frame_times[-1])],
            "frame_dt_s": resolved_frame_dt_s,
            "subblock_duration_s": resolved_subblock_duration_s,
            "n_frames_per_subblock": resolved_n_frames,
            "n_subblocks": int(n_subblocks),
            "output_keys": list(output_keys),
            "active_frame_keys": list(active_frame_keys),
            "trajectory_window_policy": "shared_across_cases",
            "smear": {
                "enabled": bool(smear_cfg.enabled),
                "render_mode": smear_cfg.render_mode if smear_cfg.enabled else "disabled",
                "inference_mode": smear_cfg.inference_mode if smear_cfg.enabled else "disabled",
            },
            "filter": {
                "enabled": bool(filter_spec.enabled and filter_spec.kind != "none"),
                "kind": filter_spec.kind,
                "method": filter_spec.method,
                "order": filter_spec.order,
                "zero_phase": filter_spec.zero_phase,
                "apply_stage": filter_spec.apply_stage,
                "provenance_json": str(filter_artifacts.get("trajectory_filter_provenance_json", "")),
                "summary_csv": str(filter_artifacts.get("trajectory_filter_summary_csv", "")),
            },
            "offsets": {
                "enabled": bool(offsets_enabled),
                "requested_offsets": dict(parsed_offsets),
                "stage": (
                    None
                    if trajectory.offset_provenance is None
                    else trajectory.offset_provenance.get("stage")
                ),
                "provenance_json": str(filter_artifacts.get("trajectory_offset_provenance_json", "")),
                "summary_csv": str(filter_artifacts.get("trajectory_offset_summary_csv", "")),
            },
            "notes": list(TRAJECTORY_NOTES),
        },
        rows=tuple(rows),
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _external_plan(
    *,
    trace_source_cfg: Mapping[str, Any],
    run_root: Path,
    source_kind: str,
    active_frame_keys: Sequence[str],
    n_subblocks: int,
    n_frames_per_subblock: int,
    frame_dt_s: float,
    default_output_keys: Sequence[str],
    trajectory_processing_cfg: Mapping[str, Any] | None,
    plate_scale_as_per_pix: float | None,
) -> PreparedTraceSourcePlan:
    campaign_plan_path = _resolve_path(
        trace_source_cfg.get("campaign_plan") or trace_source_cfg.get("trajectory_plan"),
        field_name="subblocks.trace_source.campaign_plan",
    )
    subblock_plan_value = trace_source_cfg.get("subblock_plan")
    if subblock_plan_value is None:
        subblock_plan_path = campaign_plan_path.parent / "subblock_plan.csv"
    else:
        subblock_plan_path = _resolve_path(
            subblock_plan_value,
            field_name="subblocks.trace_source.subblock_plan",
        )
    rows_in = _read_csv_rows(subblock_plan_path)
    if len(rows_in) < int(n_subblocks):
        raise ValueError("External trace-source plan does not contain enough subblocks.")
    output_keys = _as_key_tuple(
        trace_source_cfg.get("output_keys"),
        default=default_output_keys,
        field_name="subblocks.trace_source.output_keys",
    )
    smear_cfg = parse_smear_config(
        trajectory_processing_cfg,
        exposure_time_s=float(frame_dt_s),
        plate_scale_as_per_pix=plate_scale_as_per_pix,
    )
    rows: list[dict[str, Any]] = []
    subblocks: list[PreparedTraceSubblock] = []
    for index, source_row in enumerate(rows_in[: int(n_subblocks)]):
        frame_truth = Path(str(source_row.get("frame_truth_path", ""))).expanduser()
        guess = Path(str(source_row.get("starting_guess_prediction_path", ""))).expanduser()
        if not frame_truth.is_absolute():
            frame_truth = (subblock_plan_path.parent / frame_truth).resolve()
        if not guess.is_absolute():
            guess = (subblock_plan_path.parent / guess).resolve()
        smear_truth: Path | None = None
        smear_model: Path | None = None
        smear_provenance: Path | None = None
        if smear_cfg.enabled:
            for key, target in (
                ("smear_truth_csv", "truth"),
                ("smear_model_csv", "model"),
                ("smear_provenance_json", "provenance"),
            ):
                raw = str(source_row.get(key, "")).strip()
                if not raw:
                    raise ValueError(f"External trace-source plan is missing required {key}.")
                path = Path(raw).expanduser()
                if not path.is_absolute():
                    path = (subblock_plan_path.parent / path).resolve()
                if target == "truth":
                    smear_truth = path
                elif target == "model":
                    smear_model = path
                else:
                    smear_provenance = path
        missing = [
            path
            for path in (frame_truth, guess, smear_truth, smear_model, smear_provenance)
            if path is not None and not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "External trace-source artifact missing: "
                + ", ".join(str(path) for path in missing)
            )
        row = {
            **source_row,
            "trace_source_mode": TRACE_SOURCE_MODE_EXTERNAL_PLAN,
            "frame_truth_path": str(frame_truth),
            "starting_guess_prediction_path": str(guess),
            "smear_enabled": bool(smear_cfg.enabled),
            "smear_truth_csv": str(smear_truth or source_row.get("smear_truth_csv", "")),
            "smear_model_csv": str(smear_model or source_row.get("smear_model_csv", "")),
            "smear_provenance_json": str(
                smear_provenance or source_row.get("smear_provenance_json", "")
            ),
            "trajectory_subblock_index": int(source_row.get("subblock_index", index)),
            "trajectory_active_frame_keys": ",".join(active_frame_keys),
        }
        rows.append(row)
        subblocks.append(
            PreparedTraceSubblock(
                subblock_index=index,
                frame_truth_path=frame_truth,
                starting_guess_prediction_path=guess,
                smear_truth_path=smear_truth,
                smear_model_path=smear_model,
                smear_provenance_path=smear_provenance,
                trace_source_mode=TRACE_SOURCE_MODE_EXTERNAL_PLAN,
                time_start_s=(
                    float(source_row["time_start_s"])
                    if source_row.get("time_start_s")
                    else None
                ),
                time_end_s=(
                    float(source_row["time_end_s"]) if source_row.get("time_end_s") else None
                ),
                n_frames=int(source_row.get("n_frames") or n_frames_per_subblock),
                output_keys=output_keys,
                active_frame_keys=tuple(active_frame_keys),
                diagnostics={},
                provenance=dict(row),
            )
        )
    return PreparedTraceSourcePlan(
        mode=TRACE_SOURCE_MODE_EXTERNAL_PLAN,
        run_root=run_root,
        source_kind=source_kind,
        output_keys=output_keys,
        active_frame_keys=tuple(active_frame_keys),
        subblocks=tuple(subblocks),
        summary={
            "mode": TRACE_SOURCE_MODE_EXTERNAL_PLAN,
            "source_kind": source_kind,
            "campaign_plan": str(campaign_plan_path),
            "subblock_plan": str(subblock_plan_path),
            "n_subblocks": int(n_subblocks),
            "n_frames_per_subblock": int(n_frames_per_subblock),
            "frame_dt_s": float(frame_dt_s),
            "output_keys": list(output_keys),
            "active_frame_keys": list(active_frame_keys),
            "notes": list(TRAJECTORY_NOTES),
            "smear": {
                "enabled": bool(smear_cfg.enabled),
                "render_mode": smear_cfg.render_mode if smear_cfg.enabled else "disabled",
                "inference_mode": smear_cfg.inference_mode if smear_cfg.enabled else "disabled",
            },
        },
        rows=tuple(rows),
    )


def prepare_campaign_trace_source(
    *,
    trace_source_cfg: Mapping[str, Any] | None,
    run_root: Path,
    artifact_root: Path | None = None,
    source_kind: str,
    active_frame_keys: Sequence[str],
    n_subblocks: int,
    n_frames_per_subblock: int,
    frame_dt_s: float,
    subblock_duration_s: float,
    default_output_keys: Sequence[str] = DEFAULT_OUTPUT_KEYS,
    reuse_existing: bool = False,
    trajectory_processing_cfg: Mapping[str, Any] | None = None,
    plate_scale_as_per_pix: float | None = None,
) -> PreparedTraceSourcePlan:
    """Prepare a materialized trace source plan for campaign wrappers."""

    cfg = dict(trace_source_cfg or {})
    mode = str(cfg.get("mode", TRACE_SOURCE_MODE_IID)).strip().lower()
    if mode not in SUPPORTED_TRACE_SOURCE_MODES:
        raise ValueError(
            "subblocks.trace_source.mode must be one of: "
            + ", ".join(SUPPORTED_TRACE_SOURCE_MODES)
        )
    run_root = run_root.resolve()
    artifact_root = (artifact_root or (run_root / "trajectory")).resolve()
    if mode == TRACE_SOURCE_MODE_IID:
        return _iid_plan(
            run_root=run_root,
            source_kind=source_kind,
            active_frame_keys=active_frame_keys,
            n_subblocks=n_subblocks,
            n_frames_per_subblock=n_frames_per_subblock,
            default_output_keys=default_output_keys,
            trace_source_cfg=cfg,
        )
    if mode == TRACE_SOURCE_MODE_TRAJECTORY:
        return _trajectory_plan(
            trace_source_cfg=cfg,
            run_root=run_root,
            artifact_root=artifact_root,
            source_kind=source_kind,
            active_frame_keys=active_frame_keys,
            n_subblocks=n_subblocks,
            n_frames_per_subblock=n_frames_per_subblock,
            frame_dt_s=frame_dt_s,
            subblock_duration_s=subblock_duration_s,
            default_output_keys=default_output_keys,
            reuse_existing=reuse_existing,
            trajectory_processing_cfg=trajectory_processing_cfg,
            plate_scale_as_per_pix=plate_scale_as_per_pix,
        )
    return _external_plan(
        trace_source_cfg=cfg,
        run_root=run_root,
        source_kind=source_kind,
        active_frame_keys=active_frame_keys,
        n_subblocks=n_subblocks,
        n_frames_per_subblock=n_frames_per_subblock,
        frame_dt_s=frame_dt_s,
        default_output_keys=default_output_keys,
        trajectory_processing_cfg=trajectory_processing_cfg,
        plate_scale_as_per_pix=plate_scale_as_per_pix,
    )


def trace_subblock_command_flags(subblock: PreparedTraceSubblock) -> list[str]:
    """Return run_obs_subblock_study flags for materialized trace-source files."""

    if subblock.trace_source_mode == TRACE_SOURCE_MODE_IID:
        return []
    if subblock.frame_truth_path is None or subblock.starting_guess_prediction_path is None:
        raise ValueError("Materialized trace source is missing explicit CSV paths.")
    return [
        "--external-frame-truth-csv",
        str(subblock.frame_truth_path),
        "--starting-guess-csv",
        str(subblock.starting_guess_prediction_path),
        "--starting-guess-mode",
        "starting_guess_csv",
    ]


def validate_stored_trace_source_artifacts(stored_plan: Mapping[str, Any]) -> None:
    """Validate materialized trace-source paths recorded in a stored campaign plan."""

    trace_source = stored_plan.get("trace_source")
    if not isinstance(trace_source, Mapping):
        return
    mode = str(trace_source.get("mode", TRACE_SOURCE_MODE_IID))
    if mode == TRACE_SOURCE_MODE_IID:
        return

    subblock_plan = stored_plan.get("subblock_plan")
    rows: list[Mapping[str, Any]] = []
    if isinstance(subblock_plan, Mapping):
        for case_rows in subblock_plan.values():
            if isinstance(case_rows, list):
                rows.extend(row for row in case_rows if isinstance(row, Mapping))
    elif isinstance(subblock_plan, list):
        rows.extend(row for row in subblock_plan if isinstance(row, Mapping))

    missing: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        row_mode = str(row.get("trace_source_mode", mode))
        if row_mode == TRACE_SOURCE_MODE_IID:
            continue
        keys = ["frame_truth_path", "starting_guess_prediction_path"]
        if str(row.get("smear_enabled", "")).lower() in {"true", "1"}:
            keys.extend(["smear_truth_csv", "smear_model_csv", "smear_provenance_json"])
        for key in keys:
            value = row.get(key)
            if not isinstance(value, str) or not value.strip():
                missing.append(Path(f"<missing {key}>"))
                continue
            if value in seen:
                continue
            seen.add(value)
            path = Path(value).expanduser()
            if not path.exists():
                missing.append(path)
    if missing:
        raise FileNotFoundError(
            "Stored trace-source artifact is missing: "
            + ", ".join(str(path) for path in missing[:10])
        )


__all__ = [
    "PreparedTraceSourcePlan",
    "PreparedTraceSubblock",
    "SUPPORTED_TRACE_SOURCE_MODES",
    "TRACE_SOURCE_MODE_EXTERNAL_PLAN",
    "TRACE_SOURCE_MODE_IID",
    "TRACE_SOURCE_MODE_TRAJECTORY",
    "prepare_campaign_trace_source",
    "trace_subblock_command_flags",
    "validate_stored_trace_source_artifacts",
]
