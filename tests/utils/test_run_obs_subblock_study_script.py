from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_obs_subblock_study.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_obs_subblock_study_script",
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
            "optics": {"plate_scale_as_per_pix": 0.01},
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
            "optics": {"plate_scale_as_per_pix": 0.01},
        },
        "experiment": {
            "kind": "subblock_generation",
            "truth": {"optics": {"plate_scale_as_per_pix": 0.01}},
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
            "optics": {"plate_scale_as_per_pix": 0.01},
        },
        "experiment": {
            "kind": "subblock_inference",
            "inference": {
                "data": {
                    "cube": "placeholder_cube.fits",
                    "truth_trace": "placeholder_truth.csv",
                    "manifest": "placeholder_manifest.json",
                },
                "validate": {
                    "require_contiguous_frame_index": True,
                    "require_monotonic_time": True,
                },
                "active": {
                    "frame_keys": [
                        "source.x_position_as",
                        "source.y_position_as",
                        "source.position_angle_deg",
                    ],
                    "shared_keys": [],
                },
                "init": {
                    "frame": {
                        "mode": "shared_guess",
                        "values": {
                            "source.x_position_as": 0.0,
                            "source.y_position_as": 0.0,
                            "source.position_angle_deg": 0.0,
                        },
                    },
                    "shared": {},
                },
                "priors": {"frame": {}, "shared": {}},
                "temporal": {"frame_model": {"kind": "independent"}},
                "objective": {
                    "kind": "nll",
                    "frame_reduce": "sum",
                    "subblock_reduce": "sum",
                    "noise_model": {"kind": "gaussian", "variance_model": "data"},
                },
                "optimizer": {
                    "kind": "sgd",
                    "base_lr": 0.5,
                    "n_iter": 2,
                    "preconditioning": {"enabled": False},
                },
                "diagnostics": {
                    "plots": True,
                    "compare_to_truth_when_available": True,
                },
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


def _write_case_render_artifacts(
    case_root: Path,
    *,
    truth_value: float = 0.01,
) -> tuple[Path, Path, Path]:
    render_dir = case_root / "render"
    render_dir.mkdir(parents=True, exist_ok=True)
    cube_path = render_dir / "obs_subblock_cube.fits"
    cube_path.write_bytes(b"cube")
    truth_path = render_dir / "obs_subblock_truth.csv"
    truth_path.write_text(
        "frame_index,time_s,source.x_position_as,source.y_position_as\n"
        "0,0.0,0.0,0.0\n"
        "1,0.1,0.1,-0.1\n",
        encoding="utf-8",
    )
    manifest_path = render_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "artifacts": {
                "cube_fits": cube_path.name,
                "frame_truth_csv": truth_path.name,
            },
            "shared_truth": {"optics": {"plate_scale_as_per_pix": truth_value}},
            "system": {
                "resolved_config": {"optics": {"plate_scale_as_per_pix": truth_value}}
            },
        },
    )
    return cube_path, truth_path, manifest_path


def _resolve_relative(config_path: Path, value: str) -> Path:
    return (config_path.parent / value).resolve()


@dataclass(frozen=True)
class _ResolvedInput:
    path: Path | None
    source: str | None = None


@dataclass(frozen=True)
class _RenderInputs:
    cube: _ResolvedInput
    truth_trace: _ResolvedInput
    manifest: _ResolvedInput


def test_parse_helpers_validate_mode_candidate_and_grid():
    module = _load_script_module()

    assert module.parse_study_mode("profile_objective") == "profile_objective"
    assert module.parse_scalar_candidate_parameter("optics.plate_scale_as_per_pix") == (
        "optics.plate_scale_as_per_pix"
    )
    assert module.parse_scalar_grid("0.99,1.0,1.01") == (0.99, 1.0, 1.01)

    with pytest.raises(ValueError, match="Unsupported study mode"):
        module.parse_study_mode("profile")

    with pytest.raises(ValueError, match="scalar candidate"):
        module.parse_scalar_candidate_parameter("optics.primary.zernike_coeffs_nm[0]")

    with pytest.raises(ValueError, match="at least one value"):
        module.parse_scalar_grid("")


def test_derive_scalar_information_metrics_handles_nonpositive_marginal_info():
    module = _load_script_module()

    valid = module.derive_scalar_information_metrics(f_pp=4.0, i_marg=1.0)
    assert valid["sigma_cond"] == pytest.approx(0.5)
    assert valid["sigma_marg"] == pytest.approx(1.0)
    assert valid["absorption_fraction"] == pytest.approx(0.75)
    assert valid["marginalization_status"] == "ok"
    assert valid["valid_marginal_sigma"] is True

    zero_marg = module.derive_scalar_information_metrics(f_pp=4.0, i_marg=0.0)
    assert zero_marg["sigma_marg"] == float("inf")
    assert zero_marg["valid_marginal_sigma"] is False
    assert zero_marg["marginalization_status"] == "zero_marginal_information"

    negative_marg = module.derive_scalar_information_metrics(f_pp=4.0, i_marg=-1.0)
    assert negative_marg["sigma_marg"] is None
    assert negative_marg["valid_marginal_sigma"] is False
    assert negative_marg["absorption_fraction"] == pytest.approx(1.25)
    assert negative_marg["marginalization_status"] == "negative_marginal_information"


def test_fisher_only_dry_run_writes_shared_candidate_config(tmp_path: Path):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_fisher"
    cube_path, truth_path, manifest_path = _write_case_render_artifacts(
        case_root,
        truth_value=0.0125,
    )

    summary = module.run_obs_subblock_study(
        mode="fisher_only",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        candidate_key="optics.plate_scale_as_per_pix",
        truth_value=0.0125,
        dry_run=True,
    )

    summary_path = case_root / "study" / "fisher_only" / "summary.json"
    fisher_config_path = case_root / "study" / "fisher_only" / "fisher" / "inference_config.json"
    fisher_cfg = _read_json(fisher_config_path)

    assert summary["summary_path"] == str(summary_path.resolve())
    assert summary["case_prep_stages_executed"] == []
    assert summary["rendered_truth_value"] == 0.0125
    assert fisher_cfg["experiment"]["inference"]["active"]["shared_keys"] == [
        "optics.plate_scale_as_per_pix"
    ]
    assert fisher_cfg["experiment"]["inference"]["init"]["shared"] == {
        "optics.plate_scale_as_per_pix": 0.0125
    }
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["cube"],
    ) == cube_path.resolve()
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["truth_trace"],
    ) == truth_path.resolve()
    assert _resolve_relative(
        fisher_config_path,
        fisher_cfg["experiment"]["inference"]["data"]["manifest"],
    ) == manifest_path.resolve()


def test_profile_objective_reuses_existing_render_outputs_and_writes_curve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_profile"
    cube_path, _truth_path, _manifest_path = _write_case_render_artifacts(case_root)

    def fake_inference_runner(config_path: Path, run_root: Path, dry_run: bool) -> dict:
        assert dry_run is False
        cfg = _read_json(config_path)
        assumed_value = float(cfg["system"]["optics"]["plate_scale_as_per_pix"])
        output_dir = run_root / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {
            "initial_loss": 10.0,
            "final_loss": abs(assumed_value - 1.0),
            "output_dir": str(output_dir),
            "artifacts": {
                "manifest_json": str((output_dir / "manifest.json").resolve()),
            },
        }

    monkeypatch.setattr(module, "_default_inference_runner", fake_inference_runner)

    summary = module.run_obs_subblock_study(
        mode="profile_objective",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        candidate_key="optics.plate_scale_as_per_pix",
        scan_values=(0.99, 1.0, 1.01),
        dry_run=False,
    )

    profile_summary = summary["profile_summary"]
    curve_path = Path(profile_summary["curve_csv"])
    rows = list(csv.DictReader(curve_path.open("r", encoding="utf-8", newline="")))
    first_config_path = Path(rows[0]["config_path"])
    first_cfg = _read_json(first_config_path)

    assert summary["case_prep_stages_executed"] == []
    assert profile_summary["best_run"]["assumed_value"] == 1.0
    assert profile_summary["summary_path"] == str(
        (case_root / "study" / "profile_objective" / "summary.json").resolve()
    )
    assert len(rows) == 3
    assert [float(row["assumed_value"]) for row in rows] == [0.99, 1.0, 1.01]
    assert _resolve_relative(
        first_config_path,
        first_cfg["experiment"]["inference"]["data"]["cube"],
    ) == cube_path.resolve()


def test_nuisance_absorption_writes_bias_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_absorption"
    _write_case_render_artifacts(case_root, truth_value=0.011)

    def fake_inference_runner(config_path: Path, run_root: Path, dry_run: bool) -> dict:
        assert dry_run is False
        output_dir = run_root / "inference"
        output_dir.mkdir(parents=True, exist_ok=True)
        truth_csv = output_dir / "truth_comparison.csv"
        with truth_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "source.x_position_as_residual",
                    "source.y_position_as_residual",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source.x_position_as_residual": "1.0",
                    "source.y_position_as_residual": "-1.0",
                }
            )
            writer.writerow(
                {
                    "source.x_position_as_residual": "3.0",
                    "source.y_position_as_residual": "-3.0",
                }
            )
        return {
            "initial_loss": 6.0,
            "final_loss": 2.0,
            "output_dir": str(output_dir),
            "frame_keys": [
                "source.x_position_as",
                "source.y_position_as",
            ],
            "artifacts": {
                "truth_comparison_csv": str(truth_csv.resolve()),
            },
        }

    monkeypatch.setattr(module, "_default_inference_runner", fake_inference_runner)

    summary = module.run_obs_subblock_study(
        mode="nuisance_absorption",
        case_root=case_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        candidate_key="optics.plate_scale_as_per_pix",
        assumed_value=0.009,
        dry_run=False,
    )

    bias_summary = summary["nuisance_absorption_summary"]["bias_summary"]
    assert summary["rendered_truth_value"] == 0.011
    assert bias_summary["frame_count"] == 2
    assert bias_summary["per_key"]["source.x_position_as"]["mean_bias"] == 2.0
    assert bias_summary["per_key"]["source.y_position_as"]["max_abs_residual"] == 3.0
    assert bias_summary["overall"]["rms_residual"] == pytest.approx(
        (5.0**0.5),
        rel=1.0e-9,
    )


def test_profile_objective_errors_when_rendered_cube_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_missing_cube"

    def fake_prepare_case_render_artifacts(**_kwargs) -> dict:
        return {
            "render_inputs": _RenderInputs(
                cube=_ResolvedInput(None),
                truth_trace=_ResolvedInput(None),
                manifest=_ResolvedInput(None),
            ),
            "case_prep_summary": None,
            "stages_executed": [],
        }

    monkeypatch.setattr(
        module,
        "_prepare_case_render_artifacts",
        fake_prepare_case_render_artifacts,
    )

    with pytest.raises(ValueError, match="requires a rendered cube"):
        module.run_obs_subblock_study(
            mode="profile_objective",
            case_root=case_root,
            trace_template=trace_template,
            render_template=render_template,
            inference_template=inference_template,
            candidate_key="optics.plate_scale_as_per_pix",
            scan_values=(1.0,),
            dry_run=False,
        )


def test_full_case_mode_delegates_to_case_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_templates(tmp_path)
    case_root = tmp_path / "Results" / "case_full"
    recorded: dict[str, object] = {}

    class _FakeCaseModule:
        @staticmethod
        def run_case_workflow(**kwargs) -> dict:
            recorded.update(kwargs)
            return {
                "summary_path": str((case_root / "case_summary.json").resolve()),
                "stages_requested": ["trace", "render"],
            }

    monkeypatch.setattr(module, "_load_case_runner_module", lambda: _FakeCaseModule)

    summary = module.run_obs_subblock_study(
        mode="full_case",
        case_root=case_root,
        case_stages="trace,render",
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        dry_run=True,
    )

    assert recorded["case_root"] == case_root
    assert recorded["stages"] == "trace,render"
    assert summary["case_summary_path"] == str((case_root / "case_summary.json").resolve())
    assert summary["case_stages_requested"] == ["trace", "render"]
