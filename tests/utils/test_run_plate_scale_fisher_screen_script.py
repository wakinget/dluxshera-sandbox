from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_plate_scale_fisher_screen.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_plate_scale_fisher_screen_script",
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


def _write_minimal_templates(tmp_path: Path) -> tuple[Path, Path, Path]:
    trace_path = tmp_path / "trace_template.json"
    render_path = tmp_path / "render_template.json"
    inference_path = tmp_path / "inference_template.json"
    _write_json(trace_path, {"system": {"source": {"target": "ALPHA_CEN"}}})
    _write_json(
        render_path,
        {
            "system": {
                "source": {"target": "ALPHA_CEN"},
                "optics": {"plate_scale_as_per_pix": 0.006},
            },
            "experiment": {
                "kind": "subblock_generation",
                "noise": {"enabled": False, "photon_noise": True},
            },
        },
    )
    _write_json(
        inference_path,
        {
            "system": {
                "source": {"target": "ALPHA_CEN"},
                "optics": {"plate_scale_as_per_pix": 0.006},
            }
        },
    )
    return trace_path, render_path, inference_path


def test_build_plate_scale_fisher_case_specs_expands_expected_8_cases(tmp_path: Path):
    module = _load_script_module()

    cases = module.build_plate_scale_fisher_case_specs(study_root=tmp_path)

    assert len(cases) == 8
    assert {case.target_name for case in cases} == {"ALPHA_CEN"}
    assert {case.frame_count for case in cases} == {1, 5, 20, 50}
    assert {case.noise_mode for case in cases} == {"noiseless", "shot_noise_only"}


def test_write_plate_scale_fisher_artifacts_writes_csv_json_and_plots(tmp_path: Path):
    module = _load_script_module()

    rows = [
        {
            "target": "ALPHA_CEN",
            "candidate": "optics.plate_scale_as_per_pix",
            "study_mode": "fisher_only",
            "frame_count": 1,
            "noise_mode": "noiseless",
            "truth_value": 0.006,
            "case_name": "alpha_cen_n001_noiseless",
            "case_root": str(tmp_path / "cases" / "alpha_cen_n001_noiseless"),
            "case_status": "ok",
            "f_pp": 4.0,
            "i_marg": 1.0,
            "sigma_cond": 0.5,
            "sigma_marg": 1.0,
            "absorption_fraction": 0.75,
            "f_pp_is_finite": True,
            "i_marg_is_finite": True,
            "valid_conditional_sigma": True,
            "valid_marginal_sigma": True,
            "marginalization_status": "ok",
            "nuisance_block_status": "ok",
        },
        {
            "target": "ALPHA_CEN",
            "candidate": "optics.plate_scale_as_per_pix",
            "study_mode": "fisher_only",
            "frame_count": 5,
            "noise_mode": "shot_noise_only",
            "truth_value": 0.006,
            "case_name": "alpha_cen_n005_shot_noise_only",
            "case_root": str(tmp_path / "cases" / "alpha_cen_n005_shot_noise_only"),
            "case_status": "ok",
            "f_pp": 9.0,
            "i_marg": 4.0,
            "sigma_cond": 1.0 / 3.0,
            "sigma_marg": 0.5,
            "absorption_fraction": 5.0 / 9.0,
            "f_pp_is_finite": True,
            "i_marg_is_finite": True,
            "valid_conditional_sigma": True,
            "valid_marginal_sigma": True,
            "marginalization_status": "ok",
            "nuisance_block_status": "ok",
        },
    ]
    case_summaries = [
        {"summary_path": str(tmp_path / "cases" / "case_a" / "summary.json")},
        {"summary_path": str(tmp_path / "cases" / "case_b" / "summary.json")},
    ]

    summary = module.write_plate_scale_fisher_artifacts(
        study_root=tmp_path,
        rows=rows,
        case_summaries=case_summaries,
        truth_value=0.006,
        target_name="ALPHA_CEN",
        frame_counts=(1, 5),
        noise_modes=("noiseless", "shot_noise_only"),
        dry_run=False,
    )

    csv_path = tmp_path / "plate_scale_fisher_summary.csv"
    json_path = tmp_path / "plate_scale_fisher_summary.json"
    loaded_json = _read_json(json_path)
    csv_rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))

    assert summary["artifacts"]["aggregate_csv"] == str(csv_path.resolve())
    assert loaded_json["target"] == "ALPHA_CEN"
    assert loaded_json["candidate"] == "optics.plate_scale_as_per_pix"
    assert len(csv_rows) == 2
    assert "sigma_marg" in csv_rows[0]
    assert "target" in csv_rows[0]
    assert (tmp_path / "sigma_marg_vs_frame_count.png").exists()
    assert (tmp_path / "absorption_fraction_vs_frame_count.png").exists()
    assert (tmp_path / "sigma_cond_vs_frame_count.png").exists()


def test_run_plate_scale_fisher_screen_aggregates_stubbed_case_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_minimal_templates(tmp_path)

    class _FakeStudyModule:
        @staticmethod
        def run_obs_subblock_study(**kwargs) -> dict:
            case_root = Path(kwargs["case_root"]).resolve()
            frame_count = int(kwargs["n_frames"])
            noise_mode = "shot_noise_only" if kwargs["noise_mode"] == "enabled" else "noiseless"
            summary_path = case_root / "study" / "fisher_only" / "summary.json"
            fisher_summary_path = case_root / "study" / "fisher_only" / "fisher_summary.json"
            fisher_blocks_path = case_root / "study" / "fisher_only" / "fisher_blocks.npz"
            fisher_summary_path.parent.mkdir(parents=True, exist_ok=True)
            fisher_blocks_path.write_bytes(b"npz")

            if noise_mode == "noiseless":
                f_pp = float(frame_count * 20.0)
                i_marg = float(frame_count * 5.0)
            else:
                f_pp = float(frame_count * 8.0)
                i_marg = float(frame_count * 2.0)

            fisher_summary = {
                "mode": "fisher_only",
                "candidate_parameter": "optics.plate_scale_as_per_pix",
                "target_name": "ALPHA_CEN",
                "truth_value": float(kwargs["truth_value"]),
                "candidate_reference_value": float(kwargs["truth_value"]),
                "noise_mode": noise_mode,
                "frame_count": frame_count,
                "frame_keys": [
                    "source.x_position_as",
                    "source.y_position_as",
                    "source.position_angle_deg",
                ],
                "f_pp": f_pp,
                "i_marg": i_marg,
                "sigma_cond": 1.0 / math.sqrt(f_pp),
                "sigma_marg": 1.0 / math.sqrt(i_marg),
                "absorption_fraction": 1.0 - (i_marg / f_pp),
                "f_pp_is_finite": True,
                "i_marg_is_finite": True,
                "valid_conditional_sigma": True,
                "valid_marginal_sigma": True,
                "marginalization_status": "ok",
                "nuisance_block_status": "ok",
                "artifacts": {
                    "fisher_summary_json": str(fisher_summary_path.resolve()),
                    "fisher_blocks_npz": str(fisher_blocks_path.resolve()),
                },
            }
            _write_json(fisher_summary_path, fisher_summary)
            summary = {
                "summary_path": str(summary_path.resolve()),
                "dry_run": False,
                "fisher_summary": fisher_summary,
            }
            _write_json(summary_path, summary)
            return summary

    monkeypatch.setattr(module, "_load_study_module", lambda: _FakeStudyModule)
    monkeypatch.setattr(
        module,
        "resolve_plate_scale_truth_and_target",
        lambda **_kwargs: (0.006, "ALPHA_CEN"),
    )

    summary = module.run_plate_scale_fisher_screen(
        study_root=tmp_path / "study",
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        frame_counts=(1, 5),
        noise_modes=("noiseless", "shot_noise_only"),
        dry_run=False,
    )

    aggregate_json = _read_json(tmp_path / "study" / "plate_scale_fisher_summary.json")
    aggregate_csv_rows = list(
        csv.DictReader(
            (tmp_path / "study" / "plate_scale_fisher_summary.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            )
        )
    )

    assert summary["case_count"] == 4
    assert summary["successful_case_count"] == 4
    assert aggregate_json["target"] == "ALPHA_CEN"
    assert aggregate_json["candidate"] == "optics.plate_scale_as_per_pix"
    assert {row["noise_mode"] for row in aggregate_csv_rows} == {
        "noiseless",
        "shot_noise_only",
    }
    assert all(row["target"] == "ALPHA_CEN" for row in aggregate_csv_rows)
    assert all(row["case_status"] == "ok" for row in aggregate_csv_rows)
