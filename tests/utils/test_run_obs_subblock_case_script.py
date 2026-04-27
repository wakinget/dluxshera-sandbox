from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_obs_subblock_case.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_obs_subblock_case_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace_template() -> dict:
    return {
        "system": {
            "preset": "SHERA_TESTBED_3P",
            "source": {"target": "ALPHA_CEN", "exposure_time_s": 0.05},
        },
        "experiment": {
            "kind": "subblock_trace_generation",
            "trace": {
                "n_frames": 3,
                "dt_s": 0.05,
                "varying_keys": ["source.x_position_as"],
                "plan": {
                    "source.x_position_as": {
                        "base": 0.0,
                        "effects": [{"kind": "constant_offset", "offset": 0.0}],
                    }
                },
            },
            "outputs": {"outdir": "Results/default", "file_prefix": "subblock_trace"},
        },
    }


def _render_template() -> dict:
    return {
        "system": {
            "preset": "SHERA_TESTBED_3P",
            "source": {"exposure_time_s": 0.05},
        },
        "experiment": {
            "kind": "subblock_generation",
            "subblock": {
                "trace": {"format": "csv", "path": "placeholder_trace.csv"},
            },
            "noise": {"enabled": False},
            "outputs": {"outdir": "Results/default", "file_prefix": "obs_subblock"},
        },
    }


def _inference_template() -> dict:
    return {
        "system": {
            "preset": "SHERA_TESTBED_3P",
            "source": {"exposure_time_s": 0.05},
        },
        "experiment": {
            "kind": "subblock_inference",
            "inference": {
                "data": {
                    "cube": "placeholder_cube.fits",
                    "truth_trace": "placeholder_truth.csv",
                    "manifest": "placeholder_manifest.json",
                }
            },
            "outputs": {
                "outdir": "Results/default",
                "file_prefix": "subblock_inference",
            },
        },
    }


def _write_templates(tmp_path: Path) -> tuple[Path, Path, Path]:
    trace_path = tmp_path / "trace_template.json"
    render_path = tmp_path / "render_template.json"
    inference_path = tmp_path / "inference_template.json"
    _write_json(trace_path, _trace_template())
    _write_json(render_path, _render_template())
    _write_json(inference_path, _inference_template())
    return trace_path, render_path, inference_path


def test_parse_stage_selection_orders_and_normalizes_aliases():
    module = _load_script_module()

    assert module.parse_stage_selection("infer,trace,quick-look") == (
        "trace",
        "quicklook",
        "inference",
    )
    assert module.parse_stage_selection(["render", "trace"]) == ("trace", "render")
    assert module.parse_stage_selection("all") == (
        "trace",
        "render",
        "quicklook",
        "inference",
    )


def test_full_case_flow_writes_configs_and_runs_in_stage_order(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "study_a" / "case_001"

    calls: list[str] = []

    def trace_runner(config_path: Path, root: Path, dry_run: bool) -> dict:
        calls.append("trace")
        return {
            "dry_run": dry_run,
            "artifacts": {
                "trace_csv": str(
                    root / "trace" / "subblock_trace_20260423-120000_frame_truth.csv"
                ),
                "manifest_json": str(root / "trace" / "manifest.json"),
            },
        }

    def render_runner(config_path: Path, root: Path, dry_run: bool) -> dict:
        calls.append("render")
        return {
            "dry_run": dry_run,
            "artifacts": {
                "cube_fits": str(root / "render" / "obs_subblock_20260423-120001_cube.fits"),
                "frame_truth_csv": str(
                    root / "render" / "obs_subblock_20260423-120001_frame_truth.csv"
                ),
                "manifest_json": str(root / "render" / "manifest.json"),
            },
        }

    def quicklook_runner(
        cube_path: Path,
        manifest_path: Path | None,
        trace_path: Path | None,
        outdir: Path,
    ) -> dict:
        calls.append("quicklook")
        return {
            "output_dir": str(outdir),
            "artifacts": {
                "preview_gif": str(outdir / "preview.gif"),
                "summary_png": str(outdir / "summary.png"),
                "trace_summary_png": str(outdir / "trace_summary.png"),
            },
        }

    def inference_runner(config_path: Path, root: Path, dry_run: bool) -> dict:
        calls.append("inference")
        return {
            "dry_run": dry_run,
            "artifacts": {
                "manifest_json": str(root / "inference" / "manifest.json"),
            },
        }

    summary = module.run_case_workflow(
        case_root=case_root,
        stages=("inference", "trace", "quicklook", "render"),
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        n_frames=12,
        dt_s=0.2,
        exposure_time_s=0.4,
        noise_mode="enabled",
        dry_run=False,
        trace_runner=trace_runner,
        render_runner=render_runner,
        quicklook_runner=quicklook_runner,
        inference_runner=inference_runner,
    )

    assert calls == ["trace", "render", "quicklook", "inference"]

    trace_cfg = _read_json(case_root / "trace_config.json")
    assert trace_cfg["experiment"]["outputs"]["outdir"] == str(case_root)
    assert trace_cfg["experiment"]["trace"]["n_frames"] == 12
    assert trace_cfg["experiment"]["trace"]["dt_s"] == 0.2
    assert trace_cfg["system"]["source"]["exposure_time_s"] == 0.4

    render_cfg = _read_json(case_root / "render_config.json")
    assert render_cfg["experiment"]["outputs"]["outdir"] == str(case_root)
    assert render_cfg["experiment"]["subblock"]["trace"]["path"] == (
        "trace/subblock_trace_20260423-120000_frame_truth.csv"
    )
    assert render_cfg["experiment"]["noise"]["enabled"] is True
    assert render_cfg["system"]["source"]["exposure_time_s"] == 0.4

    inference_cfg = _read_json(case_root / "inference_config.json")
    assert inference_cfg["experiment"]["outputs"]["outdir"] == str(case_root)
    assert inference_cfg["experiment"]["inference"]["data"]["cube"] == (
        "render/obs_subblock_20260423-120001_cube.fits"
    )
    assert inference_cfg["experiment"]["inference"]["data"]["truth_trace"] == (
        "render/obs_subblock_20260423-120001_frame_truth.csv"
    )
    assert inference_cfg["experiment"]["inference"]["data"]["manifest"] == (
        "render/manifest.json"
    )
    assert inference_cfg["system"]["source"]["exposure_time_s"] == 0.4

    saved_summary = _read_json(case_root / "case_summary.json")
    assert summary["summary_path"] == str(case_root / "case_summary.json")
    assert saved_summary["resolved_inputs"]["trace"]["source"] == "trace_stage"
    assert saved_summary["resolved_inputs"]["render"]["cube"]["source"] == "render_stage"


def test_render_only_reuses_existing_case_trace_manifest(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_render_only"

    trace_dir = case_root / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_csv = trace_dir / "existing_trace.csv"
    trace_csv.write_text("frame_index,time_s,source.x_position_as\n0,0.0,0.0\n", encoding="utf-8")
    _write_json(
        trace_dir / "manifest.json",
        {
            "artifacts": {"trace_csv": trace_csv.name},
        },
    )

    calls: list[str] = []

    def render_runner(config_path: Path, root: Path, dry_run: bool) -> dict:
        calls.append("render")
        return {
            "dry_run": dry_run,
            "artifacts": {
                "cube_fits": str(root / "render" / "obs_subblock_cube.fits"),
                "frame_truth_csv": str(root / "render" / "obs_subblock_truth.csv"),
                "manifest_json": str(root / "render" / "manifest.json"),
            },
        }

    summary = module.run_case_workflow(
        case_root=case_root,
        stages=("render",),
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        render_runner=render_runner,
        inference_runner=lambda *_args, **_kwargs: {},
        quicklook_runner=lambda *_args, **_kwargs: {},
    )

    assert calls == ["render"]
    render_cfg = _read_json(case_root / "render_config.json")
    assert render_cfg["experiment"]["subblock"]["trace"]["path"] == "trace/existing_trace.csv"
    assert summary["resolved_inputs"]["trace"]["source"] == "case_trace_manifest"


def test_quicklook_only_uses_existing_render_manifest_and_case_quicklook_dir(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_quicklook"

    render_dir = case_root / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    cube_path = render_dir / "obs_subblock_cube.fits"
    cube_path.write_bytes(b"cube")
    truth_path = render_dir / "obs_subblock_truth.csv"
    truth_path.write_text("frame_index,time_s,source.x_position_as\n0,0.0,0.0\n", encoding="utf-8")
    _write_json(
        render_dir / "manifest.json",
        {
            "artifacts": {
                "cube_fits": cube_path.name,
                "frame_truth_csv": truth_path.name,
            }
        },
    )

    recorded: dict[str, str] = {}

    def quicklook_runner(
        cube_arg: Path,
        manifest_arg: Path | None,
        trace_arg: Path | None,
        outdir_arg: Path,
    ) -> dict:
        recorded["cube"] = str(cube_arg)
        recorded["manifest"] = None if manifest_arg is None else str(manifest_arg)
        recorded["trace"] = None if trace_arg is None else str(trace_arg)
        recorded["outdir"] = str(outdir_arg)
        return {
            "output_dir": str(outdir_arg),
            "artifacts": {"summary_png": str(outdir_arg / "summary.png")},
        }

    module.run_case_workflow(
        case_root=case_root,
        stages=("quicklook",),
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        inference_runner=lambda *_args, **_kwargs: {},
        quicklook_runner=quicklook_runner,
    )

    assert recorded["cube"] == str(cube_path.resolve())
    assert recorded["manifest"] == str((render_dir / "manifest.json").resolve())
    assert recorded["trace"] == str(truth_path.resolve())
    assert recorded["outdir"] == str((case_root / "render" / "quicklook").resolve())


def test_inference_only_uses_explicit_render_overrides(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_inference_override"

    external_dir = tmp_path / "external_inputs"
    external_dir.mkdir(parents=True, exist_ok=True)
    cube_path = external_dir / "external_cube.fits"
    cube_path.write_bytes(b"cube")
    truth_path = external_dir / "external_truth.csv"
    truth_path.write_text("frame_index,time_s,source.x_position_as\n0,0.0,0.0\n", encoding="utf-8")
    manifest_path = external_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "artifacts": {
                "cube_fits": cube_path.name,
                "frame_truth_csv": truth_path.name,
            }
        },
    )

    calls: list[str] = []

    def inference_runner(config_path: Path, root: Path, dry_run: bool) -> dict:
        calls.append("inference")
        return {"dry_run": dry_run, "artifacts": {"manifest_json": str(root / "inference" / "manifest.json")}}

    summary = module.run_case_workflow(
        case_root=case_root,
        stages=("inference",),
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        cube_path_override=cube_path,
        truth_trace_path_override=truth_path,
        manifest_path_override=manifest_path,
        inference_runner=inference_runner,
    )

    assert calls == ["inference"]
    inference_cfg = _read_json(case_root / "inference_config.json")
    assert inference_cfg["experiment"]["inference"]["data"]["cube"] == (
        "../../external_inputs/external_cube.fits"
    )
    assert inference_cfg["experiment"]["inference"]["data"]["truth_trace"] == (
        "../../external_inputs/external_truth.csv"
    )
    assert inference_cfg["experiment"]["inference"]["data"]["manifest"] == (
        "../../external_inputs/manifest.json"
    )
    assert summary["resolved_inputs"]["render"]["cube"]["source"] == "cube_override"
    assert summary["resolved_inputs"]["render"]["truth_trace"]["source"] == "truth_trace_override"
    assert summary["resolved_inputs"]["render"]["manifest"]["source"] == "manifest_override"


def test_missing_render_inputs_raise_clear_error(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_missing_inputs"

    try:
        module.run_case_workflow(
            case_root=case_root,
            stages=("render",),
            trace_template=trace_template,
            render_template=render_template,
            inference_template=inference_template,
            render_runner=lambda *_args, **_kwargs: {},
        )
    except ValueError as exc:
        assert "Render stage requires a trace CSV" in str(exc)
    else:
        raise AssertionError("Expected missing trace input to raise ValueError.")
