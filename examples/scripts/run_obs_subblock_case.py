"""Run one lightweight repeated-study case for the observation sub-block flow.

This script keeps the existing staged workflow explicit:

1. trace generation
2. sub-block rendering
3. optional quick-look diagnostics
4. inference

It is intentionally narrow. The script writes case-local config copies,
handles the repetitive stage-to-stage path plumbing, and runs selected stages
under one stable case root:

    <case_root>/
      trace/
      render/
      render/quicklook/
      inference/
      trace_config.json
      render_config.json
      inference_config.json
      case_summary.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from dluxshera.config.io import load_config_file
from dluxshera.utils.obs_subblock_io import now_iso_local_ms


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results"
DEFAULT_TRACE_TEMPLATE = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "observation_subblock_trace_template"
    / "subblock_trace_prescription.yaml"
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

CASE_SCHEMA_VERSION = "obs_subblock_case_summary.v1"

STAGE_TRACE = "trace"
STAGE_RENDER = "render"
STAGE_QUICKLOOK = "quicklook"
STAGE_INFERENCE = "inference"
STAGE_ORDER = (
    STAGE_TRACE,
    STAGE_RENDER,
    STAGE_QUICKLOOK,
    STAGE_INFERENCE,
)
STAGE_ALIASES = {
    "trace": STAGE_TRACE,
    "render": STAGE_RENDER,
    "quicklook": STAGE_QUICKLOOK,
    "quick-look": STAGE_QUICKLOOK,
    "quick_look": STAGE_QUICKLOOK,
    "infer": STAGE_INFERENCE,
    "inference": STAGE_INFERENCE,
    "all": "all",
}

TRACE_RUN_LABEL = STAGE_TRACE
RENDER_RUN_LABEL = STAGE_RENDER
INFERENCE_RUN_LABEL = STAGE_INFERENCE

UNRESOLVED_RENDER_TRACE = "__CASE_TRACE_PATH_UNRESOLVED__"
UNRESOLVED_INFERENCE_CUBE = "__CASE_RENDER_CUBE_PATH_UNRESOLVED__"


@dataclass(frozen=True)
class CaseLayout:
    """Stable on-disk layout for one repeated-study case."""

    case_root: Path
    trace_dir: Path
    render_dir: Path
    quicklook_dir: Path
    inference_dir: Path
    trace_config_path: Path
    render_config_path: Path
    inference_config_path: Path
    summary_path: Path


@dataclass(frozen=True)
class ResolvedInput:
    """Resolved path plus provenance note for one case-stage input."""

    path: Path | None
    source: str | None


@dataclass(frozen=True)
class RenderInputs:
    """Resolved render outputs used by quick-look and inference stages."""

    cube: ResolvedInput
    truth_trace: ResolvedInput
    manifest: ResolvedInput


TraceRunner = Callable[[Path, Path, bool], dict[str, Any]]
RenderRunner = Callable[[Path, Path, bool], dict[str, Any]]
InferenceRunner = Callable[[Path, Path, bool], dict[str, Any]]
QuicklookRunner = Callable[[Path, Path | None, Path | None, Path], dict[str, Any]]


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_trace_runner(config_path: Path, case_root: Path, dry_run: bool) -> dict[str, Any]:
    module = _load_module(
        REPO_ROOT / "examples" / "recipes" / "subblock_trace_generation.py",
        "obs_subblock_case_trace_recipe",
    )
    return module.generate_subblock_trace_generation(
        config_path=config_path,
        results_dir=case_root,
        run_name=TRACE_RUN_LABEL,
        dry_run=dry_run,
    )


def _default_render_runner(config_path: Path, case_root: Path, dry_run: bool) -> dict[str, Any]:
    module = _load_module(
        REPO_ROOT / "examples" / "recipes" / "observation_subblock.py",
        "obs_subblock_case_render_recipe",
    )
    return module.generate_obs_subblock(
        config_path=config_path,
        results_dir=case_root,
        run_name=RENDER_RUN_LABEL,
        dry_run=dry_run,
        show_progress=False,
    )


def _default_inference_runner(
    config_path: Path,
    case_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    module = _load_module(
        REPO_ROOT / "examples" / "recipes" / "observation_subblock_inference.py",
        "obs_subblock_case_inference_recipe",
    )
    argv = [
        "--config",
        str(config_path),
        "--results-dir",
        str(case_root),
        "--run-name",
        INFERENCE_RUN_LABEL,
        "--no-progress",
    ]
    if dry_run:
        argv.append("--dry-run")
    return module.main(argv)


def _default_quicklook_runner(
    cube_path: Path,
    manifest_path: Path | None,
    trace_path: Path | None,
    outdir: Path,
) -> dict[str, Any]:
    module = _load_module(
        REPO_ROOT / "examples" / "scripts" / "visualize_obs_subblock.py",
        "obs_subblock_case_quicklook_script",
    )
    return module.generate_obs_subblock_quicklook(
        cube_path=cube_path,
        manifest_path=manifest_path,
        trace_path=trace_path,
        outdir=outdir,
    )


def parse_stage_selection(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse and canonicalize selected stages."""

    if raw is None:
        return STAGE_ORDER

    if isinstance(raw, str):
        tokens = [part.strip() for part in raw.split(",")]
    else:
        tokens = [str(part).strip() for part in raw]

    requested: set[str] = set()
    for token in tokens:
        if not token:
            continue
        canonical = STAGE_ALIASES.get(token.lower())
        if canonical is None:
            raise ValueError(
                f"Unsupported stage {token!r}. Expected one of: {', '.join(STAGE_ORDER)}."
            )
        if canonical == "all":
            return STAGE_ORDER
        requested.add(canonical)

    if not requested:
        return STAGE_ORDER
    return tuple(stage for stage in STAGE_ORDER if stage in requested)


def resolve_case_root(
    *,
    case_root: Path | None,
    case_name: Path | None,
    results_root: Path,
) -> Path:
    """Resolve the case root from either an explicit path or name."""

    if case_root is not None:
        return case_root.expanduser().resolve()
    if case_name is None:
        raise ValueError("Either case_root or case_name must be provided.")
    if case_name.is_absolute():
        raise ValueError("--case-name must be a relative path. Use --case-root instead.")
    return (results_root.expanduser() / case_name).resolve()


def build_case_layout(case_root: Path) -> CaseLayout:
    """Return the stable directory and config layout for one case."""

    return CaseLayout(
        case_root=case_root,
        trace_dir=case_root / STAGE_TRACE,
        render_dir=case_root / STAGE_RENDER,
        quicklook_dir=case_root / STAGE_RENDER / STAGE_QUICKLOOK,
        inference_dir=case_root / STAGE_INFERENCE,
        trace_config_path=case_root / "trace_config.json",
        render_config_path=case_root / "render_config.json",
        inference_config_path=case_root / "inference_config.json",
        summary_path=case_root / "case_summary.json",
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)


def _ensure_mapping(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if value is None:
        parent[key] = {}
        return parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict.")
    return value


def _path_for_config(target: Path, *, config_dir: Path) -> str:
    relative = os.path.relpath(target.resolve(), config_dir.resolve())
    return Path(relative).as_posix()


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must decode to a JSON object: {path}")
    return payload


def _resolve_manifest_artifact(
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    artifact_key: str,
) -> Path | None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(artifact_key)
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    return candidate if candidate.exists() else None


def _resolve_trace_path_from_manifest(manifest_path: Path) -> Path | None:
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return None
    artifact_path = _resolve_manifest_artifact(
        manifest_path,
        manifest,
        artifact_key="trace_csv",
    )
    if artifact_path is not None:
        return artifact_path
    trace_payload = manifest.get("trace")
    if isinstance(trace_payload, dict):
        value = trace_payload.get("path")
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (manifest_path.parent / candidate).resolve()
            if candidate.exists():
                return candidate
    return None


def _resolve_render_truth_from_manifest(manifest_path: Path) -> Path | None:
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        return None
    artifact_path = _resolve_manifest_artifact(
        manifest_path,
        manifest,
        artifact_key="frame_truth_csv",
    )
    if artifact_path is not None:
        return artifact_path
    trace_payload = manifest.get("trace")
    if isinstance(trace_payload, dict):
        value = trace_payload.get("path")
        if isinstance(value, str) and value.strip():
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = (manifest_path.parent / candidate).resolve()
            if candidate.exists():
                return candidate
    return None


def _latest_matching(directory: Path, pattern: str) -> Path | None:
    matches = [path for path in directory.glob(pattern) if path.is_file()]
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _discover_case_trace_input(layout: CaseLayout) -> ResolvedInput:
    manifest_path = layout.trace_dir / "manifest.json"
    trace_path = _resolve_trace_path_from_manifest(manifest_path)
    if trace_path is not None:
        return ResolvedInput(trace_path, "case_trace_manifest")
    latest = _latest_matching(layout.trace_dir, "*_frame_truth.csv")
    if latest is not None:
        return ResolvedInput(latest.resolve(), "case_trace_latest_csv")
    return ResolvedInput(None, None)


def _discover_case_render_inputs(layout: CaseLayout) -> RenderInputs:
    manifest_path = layout.render_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    cube_path = None
    truth_trace_path = None
    manifest_input = ResolvedInput(None, None)
    cube_source: str | None = None
    truth_source: str | None = None

    if manifest is not None:
        cube_path = _resolve_manifest_artifact(
            manifest_path,
            manifest,
            artifact_key="cube_fits",
        )
        truth_trace_path = _resolve_render_truth_from_manifest(manifest_path)
        manifest_input = ResolvedInput(manifest_path.resolve(), "case_render_manifest")
        if cube_path is not None:
            cube_source = "case_render_manifest"
        if truth_trace_path is not None:
            truth_source = "case_render_manifest"

    if cube_path is None:
        cube_path = _latest_matching(layout.render_dir, "*_cube.fits")
        if cube_path is not None:
            cube_source = "case_render_latest_cube"
    if truth_trace_path is None:
        truth_trace_path = _latest_matching(layout.render_dir, "*_frame_truth.csv")
        if truth_trace_path is not None:
            truth_source = "case_render_latest_truth_csv"

    return RenderInputs(
        cube=ResolvedInput(
            None if cube_path is None else cube_path.resolve(),
            cube_source,
        ),
        truth_trace=ResolvedInput(
            None if truth_trace_path is None else truth_trace_path.resolve(),
            truth_source,
        ),
        manifest=manifest_input,
    )


def _resolved_from_path(path: Path | None, source: str | None) -> ResolvedInput:
    if path is None:
        return ResolvedInput(None, None if source is None else source)
    return ResolvedInput(path.resolve(), source)


def _overlay_render_inputs(
    base: RenderInputs,
    *,
    cube_path: Path | None,
    truth_trace_path: Path | None,
    manifest_path: Path | None,
) -> RenderInputs:
    cube_input = base.cube
    truth_input = base.truth_trace
    manifest_input = base.manifest

    if manifest_path is not None:
        manifest_input = _resolved_from_path(manifest_path, "manifest_override")
    if cube_path is not None:
        cube_input = _resolved_from_path(cube_path, "cube_override")
    if truth_trace_path is not None:
        truth_input = _resolved_from_path(truth_trace_path, "truth_trace_override")

    if manifest_input.path is not None and cube_input.path is None:
        manifest = _load_manifest(manifest_input.path)
        if manifest is not None:
            derived_cube = _resolve_manifest_artifact(
                manifest_input.path,
                manifest,
                artifact_key="cube_fits",
            )
            if derived_cube is not None:
                cube_input = ResolvedInput(derived_cube.resolve(), manifest_input.source)

    if cube_input.path is not None and manifest_input.path is None:
        sibling_manifest = cube_input.path.parent / "manifest.json"
        if sibling_manifest.exists():
            manifest_input = ResolvedInput(sibling_manifest.resolve(), "cube_sibling_manifest")

    if manifest_input.path is not None and truth_input.path is None:
        derived_truth = _resolve_render_truth_from_manifest(manifest_input.path)
        if derived_truth is not None:
            truth_input = ResolvedInput(derived_truth.resolve(), manifest_input.source)

    return RenderInputs(
        cube=cube_input,
        truth_trace=truth_input,
        manifest=manifest_input,
    )


def _artifact_path(result: dict[str, Any], key: str) -> Path | None:
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        return None
    value = artifacts.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return Path(value).resolve()


def _apply_exposure_override(cfg: dict[str, Any], exposure_time_s: float | None) -> None:
    if exposure_time_s is None:
        return
    system = _ensure_mapping(cfg, "system", path="root")
    source_cfg = _ensure_mapping(system, "source", path="system")
    source_cfg["exposure_time_s"] = float(exposure_time_s)

    experiment_cfg = cfg.get("experiment")
    if isinstance(experiment_cfg, dict):
        truth_cfg = experiment_cfg.get("truth")
        if isinstance(truth_cfg, dict):
            truth_source_cfg = _ensure_mapping(
                truth_cfg,
                "source",
                path="experiment.truth",
            )
            truth_source_cfg["exposure_time_s"] = float(exposure_time_s)


def build_trace_case_config(
    *,
    template_path: Path,
    case_root: Path,
    n_frames: int | None,
    dt_s: float | None,
    exposure_time_s: float | None,
) -> dict[str, Any]:
    """Build the case-local trace config."""

    cfg = load_config_file(template_path)
    experiment_cfg = _ensure_mapping(cfg, "experiment", path="root")
    outputs_cfg = _ensure_mapping(experiment_cfg, "outputs", path="experiment")
    outputs_cfg["outdir"] = str(case_root)
    if n_frames is not None or dt_s is not None:
        trace_cfg = _ensure_mapping(experiment_cfg, "trace", path="experiment")
        if n_frames is not None:
            trace_cfg["n_frames"] = int(n_frames)
        if dt_s is not None:
            trace_cfg["dt_s"] = float(dt_s)
    _apply_exposure_override(cfg, exposure_time_s)
    return cfg


def build_render_case_config(
    *,
    template_path: Path,
    config_dir: Path,
    case_root: Path,
    trace_input: ResolvedInput,
    exposure_time_s: float | None,
    noise_mode: str,
) -> dict[str, Any]:
    """Build the case-local render config."""

    cfg = load_config_file(template_path)
    experiment_cfg = _ensure_mapping(cfg, "experiment", path="root")
    outputs_cfg = _ensure_mapping(experiment_cfg, "outputs", path="experiment")
    outputs_cfg["outdir"] = str(case_root)

    subblock_cfg = _ensure_mapping(experiment_cfg, "subblock", path="experiment")
    trace_cfg = _ensure_mapping(subblock_cfg, "trace", path="experiment.subblock")
    trace_cfg["path"] = (
        UNRESOLVED_RENDER_TRACE
        if trace_input.path is None
        else _path_for_config(trace_input.path, config_dir=config_dir)
    )

    if noise_mode != "inherit":
        noise_cfg = _ensure_mapping(experiment_cfg, "noise", path="experiment")
        noise_cfg["enabled"] = noise_mode == "enabled"

    _apply_exposure_override(cfg, exposure_time_s)
    return cfg


def build_inference_case_config(
    *,
    template_path: Path,
    config_dir: Path,
    case_root: Path,
    render_inputs: RenderInputs,
    exposure_time_s: float | None,
) -> dict[str, Any]:
    """Build the case-local inference config."""

    cfg = load_config_file(template_path)
    experiment_cfg = _ensure_mapping(cfg, "experiment", path="root")
    outputs_cfg = _ensure_mapping(experiment_cfg, "outputs", path="experiment")
    outputs_cfg["outdir"] = str(case_root)

    inference_cfg = _ensure_mapping(experiment_cfg, "inference", path="experiment")
    data_cfg = _ensure_mapping(inference_cfg, "data", path="experiment.inference")
    data_cfg["cube"] = (
        UNRESOLVED_INFERENCE_CUBE
        if render_inputs.cube.path is None
        else _path_for_config(render_inputs.cube.path, config_dir=config_dir)
    )

    if render_inputs.truth_trace.path is None:
        data_cfg.pop("truth_trace", None)
    else:
        data_cfg["truth_trace"] = _path_for_config(
            render_inputs.truth_trace.path,
            config_dir=config_dir,
        )

    if render_inputs.manifest.path is None:
        data_cfg.pop("manifest", None)
    else:
        data_cfg["manifest"] = _path_for_config(
            render_inputs.manifest.path,
            config_dir=config_dir,
        )

    _apply_exposure_override(cfg, exposure_time_s)
    return cfg


def _quicklook_plan(
    *,
    cube_path: Path,
    manifest_path: Path | None,
    trace_path: Path | None,
    outdir: Path,
) -> dict[str, Any]:
    artifacts: dict[str, str] = {
        "preview_gif": str((outdir / "preview.gif").resolve()),
        "summary_png": str((outdir / "summary.png").resolve()),
    }
    if trace_path is not None:
        artifacts["trace_summary_png"] = str((outdir / "trace_summary.png").resolve())
    return {
        "dry_run": True,
        "cube_path": str(cube_path.resolve()),
        "manifest_path": None if manifest_path is None else str(manifest_path.resolve()),
        "trace_path": None if trace_path is None else str(trace_path.resolve()),
        "output_dir": str(outdir.resolve()),
        "artifacts": artifacts,
    }


def _input_payload(resolved: ResolvedInput) -> dict[str, Any]:
    return {
        "path": None if resolved.path is None else str(resolved.path),
        "source": resolved.source,
    }


def run_case_workflow(
    *,
    case_root: Path,
    stages: str | Sequence[str],
    trace_template: Path = DEFAULT_TRACE_TEMPLATE,
    render_template: Path = DEFAULT_RENDER_TEMPLATE,
    inference_template: Path = DEFAULT_INFERENCE_TEMPLATE,
    n_frames: int | None = None,
    dt_s: float | None = None,
    exposure_time_s: float | None = None,
    noise_mode: str = "inherit",
    trace_path_override: Path | None = None,
    cube_path_override: Path | None = None,
    truth_trace_path_override: Path | None = None,
    manifest_path_override: Path | None = None,
    dry_run: bool = False,
    trace_runner: TraceRunner | None = None,
    render_runner: RenderRunner | None = None,
    inference_runner: InferenceRunner | None = None,
    quicklook_runner: QuicklookRunner | None = None,
) -> dict[str, Any]:
    """Run one case with case-local configs and stable stage layout."""

    selected_stages = parse_stage_selection(stages)
    if noise_mode not in {"inherit", "enabled", "disabled"}:
        raise ValueError("noise_mode must be one of: inherit, enabled, disabled.")

    layout = build_case_layout(case_root.resolve())
    layout.case_root.mkdir(parents=True, exist_ok=True)

    trace_runner = trace_runner or _default_trace_runner
    render_runner = render_runner or _default_render_runner
    inference_runner = inference_runner or _default_inference_runner
    quicklook_runner = quicklook_runner or _default_quicklook_runner

    trace_template = trace_template.resolve()
    render_template = render_template.resolve()
    inference_template = inference_template.resolve()

    summary: dict[str, Any] = {
        "schema_version": CASE_SCHEMA_VERSION,
        "created_at": now_iso_local_ms(),
        "case_root": str(layout.case_root),
        "stages_requested": list(selected_stages),
        "dry_run": bool(dry_run),
        "templates": {
            "trace": str(trace_template),
            "render": str(render_template),
            "inference": str(inference_template),
        },
        "layout": {
            "trace_dir": str(layout.trace_dir),
            "render_dir": str(layout.render_dir),
            "quicklook_dir": str(layout.quicklook_dir),
            "inference_dir": str(layout.inference_dir),
        },
        "overrides": {
            "n_frames": n_frames,
            "dt_s": dt_s,
            "exposure_time_s": exposure_time_s,
            "noise_mode": noise_mode,
            "trace_path_override": None
            if trace_path_override is None
            else str(trace_path_override.resolve()),
            "cube_path_override": None
            if cube_path_override is None
            else str(cube_path_override.resolve()),
            "truth_trace_path_override": None
            if truth_trace_path_override is None
            else str(truth_trace_path_override.resolve()),
            "manifest_path_override": None
            if manifest_path_override is None
            else str(manifest_path_override.resolve()),
        },
        "generated_configs": {},
        "resolved_inputs": {},
        "stages": {},
    }

    trace_cfg = build_trace_case_config(
        template_path=trace_template,
        case_root=layout.case_root,
        n_frames=n_frames,
        dt_s=dt_s,
        exposure_time_s=exposure_time_s,
    )
    _write_json(layout.trace_config_path, trace_cfg)
    summary["generated_configs"]["trace"] = str(layout.trace_config_path)

    trace_result: dict[str, Any] | None = None
    trace_input = _discover_case_trace_input(layout)
    if trace_path_override is not None:
        trace_input = ResolvedInput(trace_path_override.resolve(), "trace_override")
    if STAGE_TRACE in selected_stages:
        trace_result = trace_runner(layout.trace_config_path, layout.case_root, dry_run)
        summary["stages"][STAGE_TRACE] = trace_result
    if trace_result is not None:
        trace_path = _artifact_path(trace_result, "trace_csv")
        if trace_path is not None:
            trace_input = ResolvedInput(trace_path, "trace_stage")
    summary["resolved_inputs"]["trace"] = _input_payload(trace_input)

    render_cfg = build_render_case_config(
        template_path=render_template,
        config_dir=layout.case_root,
        case_root=layout.case_root,
        trace_input=trace_input,
        exposure_time_s=exposure_time_s,
        noise_mode=noise_mode,
    )
    _write_json(layout.render_config_path, render_cfg)
    summary["generated_configs"]["render"] = str(layout.render_config_path)

    render_inputs = _discover_case_render_inputs(layout)
    render_inputs = _overlay_render_inputs(
        render_inputs,
        cube_path=cube_path_override,
        truth_trace_path=truth_trace_path_override,
        manifest_path=manifest_path_override,
    )
    render_result: dict[str, Any] | None = None
    if STAGE_RENDER in selected_stages:
        if trace_input.path is None:
            raise ValueError(
                "Render stage requires a trace CSV. Run the trace stage first or "
                "provide --trace-path."
            )
        render_result = render_runner(layout.render_config_path, layout.case_root, dry_run)
        summary["stages"][STAGE_RENDER] = render_result
    if render_result is not None:
        render_inputs = RenderInputs(
            cube=ResolvedInput(_artifact_path(render_result, "cube_fits"), "render_stage"),
            truth_trace=ResolvedInput(
                _artifact_path(render_result, "frame_truth_csv"),
                "render_stage",
            ),
            manifest=ResolvedInput(
                _artifact_path(render_result, "manifest_json"),
                "render_stage",
            ),
        )
    summary["resolved_inputs"]["render"] = {
        "cube": _input_payload(render_inputs.cube),
        "truth_trace": _input_payload(render_inputs.truth_trace),
        "manifest": _input_payload(render_inputs.manifest),
    }

    inference_cfg = build_inference_case_config(
        template_path=inference_template,
        config_dir=layout.case_root,
        case_root=layout.case_root,
        render_inputs=render_inputs,
        exposure_time_s=exposure_time_s,
    )
    _write_json(layout.inference_config_path, inference_cfg)
    summary["generated_configs"]["inference"] = str(layout.inference_config_path)

    if STAGE_QUICKLOOK in selected_stages:
        if render_inputs.cube.path is None:
            raise ValueError(
                "Quick-look stage requires a rendered cube. Run the render stage "
                "first or provide --cube-path."
            )
        if dry_run:
            quicklook_result = _quicklook_plan(
                cube_path=render_inputs.cube.path,
                manifest_path=render_inputs.manifest.path,
                trace_path=render_inputs.truth_trace.path,
                outdir=layout.quicklook_dir,
            )
        else:
            quicklook_result = quicklook_runner(
                render_inputs.cube.path,
                render_inputs.manifest.path,
                render_inputs.truth_trace.path,
                layout.quicklook_dir,
            )
        summary["stages"][STAGE_QUICKLOOK] = quicklook_result

    if STAGE_INFERENCE in selected_stages:
        if render_inputs.cube.path is None:
            raise ValueError(
                "Inference stage requires a rendered cube. Run the render stage "
                "first or provide --cube-path."
            )
        inference_result = inference_runner(
            layout.inference_config_path,
            layout.case_root,
            dry_run,
        )
        summary["stages"][STAGE_INFERENCE] = inference_result

    summary["updated_at"] = now_iso_local_ms()
    summary["summary_path"] = str(layout.summary_path)
    _write_json(layout.summary_path, summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one lightweight repeated-study observation sub-block case."
    )
    case_group = parser.add_mutually_exclusive_group(required=True)
    case_group.add_argument(
        "--case-root",
        type=Path,
        default=None,
        help="Explicit case root directory.",
    )
    case_group.add_argument(
        "--case-name",
        type=Path,
        default=None,
        help="Relative case path under --results-root.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Parent directory used with --case-name.",
    )
    parser.add_argument(
        "--stages",
        default="trace,render,quicklook,inference",
        help="Comma-separated stage list. Supported: trace, render, quicklook, inference.",
    )
    parser.add_argument(
        "--trace-template",
        type=Path,
        default=DEFAULT_TRACE_TEMPLATE,
        help="Trace template YAML/JSON path.",
    )
    parser.add_argument(
        "--render-template",
        type=Path,
        default=DEFAULT_RENDER_TEMPLATE,
        help="Render template YAML/JSON path.",
    )
    parser.add_argument(
        "--inference-template",
        type=Path,
        default=DEFAULT_INFERENCE_TEMPLATE,
        help="Inference template YAML/JSON path.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Optional override for experiment.trace.n_frames in the trace config.",
    )
    parser.add_argument(
        "--dt-s",
        type=float,
        default=None,
        help="Optional override for experiment.trace.dt_s in the trace config.",
    )
    parser.add_argument(
        "--exposure-time-s",
        type=float,
        default=None,
        help="Optional override for system.source.exposure_time_s across case configs.",
    )
    parser.add_argument(
        "--noise",
        choices=("inherit", "enabled", "disabled"),
        default="inherit",
        help="Optional override for experiment.noise.enabled in the render config.",
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help="Optional trace CSV override used by the render stage.",
    )
    parser.add_argument(
        "--cube-path",
        type=Path,
        default=None,
        help="Optional cube FITS override used by quick-look or inference stages.",
    )
    parser.add_argument(
        "--truth-trace-path",
        type=Path,
        default=None,
        help="Optional truth-trace override used by inference or quick-look stages.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional manifest override used by quick-look or inference stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write case-local configs and summary, but validate only for selected stages.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    case_root = resolve_case_root(
        case_root=args.case_root,
        case_name=args.case_name,
        results_root=args.results_root,
    )
    summary = run_case_workflow(
        case_root=case_root,
        stages=parse_stage_selection(args.stages),
        trace_template=args.trace_template,
        render_template=args.render_template,
        inference_template=args.inference_template,
        n_frames=args.n_frames,
        dt_s=args.dt_s,
        exposure_time_s=args.exposure_time_s,
        noise_mode=args.noise,
        trace_path_override=args.trace_path,
        cube_path_override=args.cube_path,
        truth_trace_path_override=args.truth_trace_path,
        manifest_path_override=args.manifest_path,
        dry_run=bool(args.dry_run),
    )
    print(f"Case root: {case_root}")
    print(f"Stages: {', '.join(summary['stages_requested'])}")
    print(f"Summary: {summary['summary_path']}")
    return summary


if __name__ == "__main__":
    main()
