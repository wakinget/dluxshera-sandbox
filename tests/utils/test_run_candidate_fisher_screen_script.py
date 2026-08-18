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
    / "run_candidate_fisher_screen.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_candidate_fisher_screen_script",
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


def _write_existing_case_outputs(
    case_root: Path,
    *,
    candidate_key: str,
    candidate_base_key: str,
    candidate_index: int | None,
    truth_value: float,
    frame_count: int,
    noise_mode: str,
    stale_case_root: Path,
) -> None:
    study_root = case_root / "study" / "fisher_only"
    summary_path = study_root / "summary.json"
    fisher_summary_path = study_root / "fisher_summary.json"
    fisher_blocks_path = study_root / "fisher_blocks.npz"
    candidate_sensitivity_path = study_root / "candidate_sensitivity.json"
    noise_audit_path = study_root / "noise_audit.json"
    stale_study_root = stale_case_root / "study" / "fisher_only"

    fisher_blocks_path.parent.mkdir(parents=True, exist_ok=True)
    fisher_blocks_path.write_bytes(b"npz")
    _write_json(candidate_sensitivity_path, {"ok": True})
    _write_json(noise_audit_path, {"ok": True})

    fisher_summary = {
        "mode": "fisher_only",
        "candidate_parameter": candidate_key,
        "candidate_base_key": candidate_base_key,
        "candidate_index": candidate_index,
        "target_name": "ALPHA_CEN",
        "truth_value": truth_value,
        "candidate_reference_value": truth_value,
        "noise_mode": noise_mode,
        "frame_count": frame_count,
        "frame_keys": [
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ],
        "fisher_method": "dense_full_theta_hessian",
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
        "noise_audit": {
            "variance_model": "provided_cube",
            "variance_source": "provided_cube",
            "cube_stats": {"mean": 10.0},
            "variance_stats": {"mean": 12.0},
            "data_as_variance_stats": {"mean": 8.0},
            "render_variance_stats": {"mean": 12.0},
            "data_variance_floor_clipped_count": 0,
            "variance_mean_over_cube_mean": 1.2,
            "data_variance_mean_over_cube_mean": 0.8,
            "render_variance_mean_over_cube_mean": 1.2,
        },
        "artifacts": {
            "fisher_summary_json": str((stale_study_root / "fisher_summary.json").resolve()),
            "fisher_blocks_npz": str((stale_study_root / "fisher_blocks.npz").resolve()),
            "candidate_sensitivity_json": str(
                (stale_study_root / "candidate_sensitivity.json").resolve()
            ),
            "noise_audit_json": str((stale_study_root / "noise_audit.json").resolve()),
        },
    }
    _write_json(fisher_summary_path, fisher_summary)
    _write_json(
        summary_path,
        {
            "summary_path": str((stale_study_root / "summary.json").resolve()),
            "case_root": str(stale_case_root.resolve()),
            "study_root": str(stale_study_root.resolve()),
            "dry_run": False,
            "fisher_summary": fisher_summary,
        },
    )


def test_build_candidate_fisher_case_specs_expands_expected_matrix(tmp_path: Path):
    module = _load_script_module()

    cases = module.build_candidate_fisher_case_specs(study_root=tmp_path)

    assert len(cases) == len(module.DEFAULT_FRAME_COUNTS) * len(
        module.SUPPORTED_NOISE_MODES
    )
    assert {case.target_name for case in cases} == {"ALPHA_CEN"}
    assert {case.frame_count for case in cases} == set(module.DEFAULT_FRAME_COUNTS)
    assert {case.noise_mode for case in cases} == {"noiseless", "shot_noise_only"}
    assert all("optics_plate_scale_as_per_pix" in case.case_name for case in cases)


def test_build_candidate_fisher_case_specs_includes_candidate_slug_for_nondefault(tmp_path: Path):
    module = _load_script_module()

    cases = module.build_candidate_fisher_case_specs(
        study_root=tmp_path,
        candidate_key="optics.primary.zernike_coeffs_nm[3]",
        frame_counts=(1,),
        noise_modes=("noiseless",),
        target_name="ALPHA_CEN",
    )

    assert len(cases) == 1
    assert cases[0].candidate_key == "optics.primary.zernike_coeffs_nm[3]"
    assert cases[0].candidate_base_key == "optics.primary.zernike_coeffs_nm"
    assert cases[0].candidate_index == 3
    assert "optics_primary_zernike_coeffs_nm_i3" in cases[0].case_name


def test_default_study_root_derivation_is_generic_for_default_candidate():
    module = _load_script_module()

    study_root = module._derive_study_root(
        candidate_key="optics.plate_scale_as_per_pix",
        target_name="ALPHA_CEN",
    )

    assert study_root == (
        Path(__file__).resolve().parents[2]
        / "Results"
        / "optics_plate_scale_as_per_pix_fisher_alpha_cen"
    )


def test_study_root_and_artifact_prefix_are_generic_for_indexed_candidate():
    module = _load_script_module()
    candidate_key = "optics.primary.zernike_coeffs_nm[3]"

    study_root = module._derive_study_root(
        candidate_key=candidate_key,
        target_name="ALPHA_CEN",
    )

    assert study_root == (
        Path(__file__).resolve().parents[2]
        / "Results"
        / "optics_primary_zernike_coeffs_nm_i3_fisher_alpha_cen"
    )
    assert module._artifact_prefix(candidate_key) == "optics_primary_zernike_coeffs_nm_i3_fisher"


def test_plot_title_uses_standard_multiline_format():
    module = _load_script_module()

    assert module._plot_title(
        "optics.primary.zernike_coeffs_nm[3]",
        "Marginalized Sigma",
    ) == (
        "Fisher Screening\n"
        "optics.primary.zernike_coeffs_nm[3]\n"
        "Marginalized Sigma"
    )


def test_parse_frame_counts_and_noise_modes_accept_subset_strings():
    module = _load_script_module()

    assert module.parse_frame_counts("1,5") == (1, 5)
    assert module.parse_noise_modes("noiseless") == ("noiseless",)


def test_write_candidate_fisher_artifacts_writes_csv_json_and_plots(tmp_path: Path):
    module = _load_script_module()

    rows = [
        {
            "target": "ALPHA_CEN",
            "candidate": "optics.plate_scale_as_per_pix",
            "candidate_base_key": "optics.plate_scale_as_per_pix",
            "candidate_index": None,
            "study_mode": "fisher_only",
            "frame_count": 1,
            "noise_mode": "noiseless",
            "truth_value": 0.006,
            "case_name": "alpha_cen_optics_plate_scale_as_per_pix_n001_noiseless",
            "case_root": str(tmp_path / "cases" / "alpha_cen_optics_plate_scale_as_per_pix_n001_noiseless"),
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
            "candidate_base_key": "optics.plate_scale_as_per_pix",
            "candidate_index": None,
            "study_mode": "fisher_only",
            "frame_count": 5,
            "noise_mode": "shot_noise_only",
            "truth_value": 0.006,
            "case_name": "alpha_cen_optics_plate_scale_as_per_pix_n005_shot_noise_only",
            "case_root": str(tmp_path / "cases" / "alpha_cen_optics_plate_scale_as_per_pix_n005_shot_noise_only"),
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
    noise_audit_rows = [
        {
            "target": "ALPHA_CEN",
            "candidate": "optics.plate_scale_as_per_pix",
            "candidate_base_key": "optics.plate_scale_as_per_pix",
            "candidate_index": None,
            "frame_count": 1,
            "noise_mode": "noiseless",
            "case_status": "ok",
            "variance_model": "provided_cube",
            "variance_source": "provided_cube",
            "cube_mean": 10.0,
            "variance_mean": 12.0,
            "data_as_variance_mean": 10.0,
            "render_variance_mean": 12.0,
            "f_pp": 4.0,
            "i_marg": 1.0,
            "sigma_cond": 0.5,
            "sigma_marg": 1.0,
            "absorption_fraction": 0.75,
        },
        {
            "target": "ALPHA_CEN",
            "candidate": "optics.plate_scale_as_per_pix",
            "candidate_base_key": "optics.plate_scale_as_per_pix",
            "candidate_index": None,
            "frame_count": 5,
            "noise_mode": "shot_noise_only",
            "case_status": "ok",
            "variance_model": "provided_cube",
            "variance_source": "provided_cube",
            "cube_mean": 10.1,
            "variance_mean": 12.1,
            "data_as_variance_mean": 9.1,
            "render_variance_mean": 12.1,
            "f_pp": 9.0,
            "i_marg": 4.0,
            "sigma_cond": 1.0 / 3.0,
            "sigma_marg": 0.5,
            "absorption_fraction": 5.0 / 9.0,
        },
    ]
    case_summaries = [
        {"summary_path": str(tmp_path / "cases" / "case_a" / "summary.json")},
        {"summary_path": str(tmp_path / "cases" / "case_b" / "summary.json")},
    ]

    summary = module.write_candidate_fisher_artifacts(
        study_root=tmp_path,
        candidate_key="optics.plate_scale_as_per_pix",
        rows=rows,
        noise_audit_rows=noise_audit_rows,
        case_summaries=case_summaries,
        truth_value=0.006,
        target_name="ALPHA_CEN",
        frame_counts=(1, 5),
        noise_modes=("noiseless", "shot_noise_only"),
        dry_run=False,
    )

    csv_path = tmp_path / "optics_plate_scale_as_per_pix_fisher_summary.csv"
    json_path = tmp_path / "optics_plate_scale_as_per_pix_fisher_summary.json"
    noise_audit_csv = tmp_path / "optics_plate_scale_as_per_pix_fisher_noise_audit.csv"
    noise_audit_json = tmp_path / "optics_plate_scale_as_per_pix_fisher_noise_audit.json"
    loaded_json = _read_json(json_path)
    loaded_noise_json = _read_json(noise_audit_json)
    csv_rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8", newline="")))
    audit_csv_rows = list(
        csv.DictReader(noise_audit_csv.open("r", encoding="utf-8", newline=""))
    )

    assert summary["artifacts"]["aggregate_csv"] == str(csv_path.resolve())
    assert loaded_json["target"] == "ALPHA_CEN"
    assert loaded_json["candidate"] == "optics.plate_scale_as_per_pix"
    assert loaded_json["candidate_base_key"] == "optics.plate_scale_as_per_pix"
    assert loaded_json["candidate_index"] is None
    assert len(csv_rows) == 2
    assert len(audit_csv_rows) == 2
    assert "sigma_marg" in csv_rows[0]
    assert "target" in csv_rows[0]
    assert "comparisons" in loaded_noise_json
    assert (tmp_path / "optics_plate_scale_as_per_pix_fisher_sigma_marg_vs_frame_count.png").exists()
    assert (tmp_path / "optics_plate_scale_as_per_pix_fisher_absorption_fraction_vs_frame_count.png").exists()
    assert (tmp_path / "optics_plate_scale_as_per_pix_fisher_sigma_cond_vs_frame_count.png").exists()
    assert (tmp_path / "optics_plate_scale_as_per_pix_fisher_variance_mean_vs_frame_count.png").exists()
    assert loaded_json["artifacts"]["progress_log"] == str((tmp_path / "progress.log").resolve())


def test_write_candidate_fisher_artifacts_uses_candidate_specific_filenames(tmp_path: Path):
    module = _load_script_module()
    candidate_key = "optics.primary.zernike_coeffs_nm[3]"
    rows = [
        {
            "target": "ALPHA_CEN",
            "candidate": candidate_key,
            "candidate_base_key": "optics.primary.zernike_coeffs_nm",
            "candidate_index": 3,
            "study_mode": "fisher_only",
            "frame_count": 1,
            "noise_mode": "noiseless",
            "truth_value": 12.5,
            "case_name": "alpha_cen_optics_primary_zernike_coeffs_nm_i3_n001_noiseless",
            "case_root": str(tmp_path / "cases" / "case_a"),
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
        }
    ]

    summary = module.write_candidate_fisher_artifacts(
        study_root=tmp_path,
        candidate_key=candidate_key,
        rows=rows,
        noise_audit_rows=rows,
        case_summaries=[],
        truth_value=12.5,
        target_name="ALPHA_CEN",
        frame_counts=(1,),
        noise_modes=("noiseless",),
        dry_run=False,
    )

    prefix = "optics_primary_zernike_coeffs_nm_i3_fisher"
    assert Path(summary["artifacts"]["aggregate_csv"]).name == f"{prefix}_summary.csv"
    assert Path(summary["artifacts"]["aggregate_json"]).name == f"{prefix}_summary.json"
    assert Path(summary["artifacts"]["sigma_marg_plot"]).name == (
        f"{prefix}_sigma_marg_vs_frame_count.png"
    )
    assert summary["candidate"] == candidate_key
    assert summary["candidate_base_key"] == "optics.primary.zernike_coeffs_nm"
    assert summary["candidate_index"] == 3


def test_run_candidate_fisher_screen_aggregates_stubbed_default_candidate_runs(
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
                "candidate_base_key": "optics.plate_scale_as_per_pix",
                "candidate_index": None,
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
                "noise_audit": {
                    "variance_model": "provided_cube",
                    "variance_source": "provided_cube",
                    "cube_stats": {"mean": 10.0},
                    "variance_stats": {"mean": 12.0},
                    "data_as_variance_stats": {"mean": 8.0},
                    "render_variance_stats": {"mean": 12.0},
                    "data_variance_floor_clipped_count": 0,
                    "variance_mean_over_cube_mean": 1.2,
                    "data_variance_mean_over_cube_mean": 0.8,
                    "render_variance_mean_over_cube_mean": 1.2,
                },
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
        "resolve_candidate_truth_and_target",
        lambda **_kwargs: (0.006, "ALPHA_CEN"),
    )

    summary = module.run_candidate_fisher_screen(
        study_root=tmp_path / "study",
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        frame_counts=(1, 5),
        noise_modes=("noiseless", "shot_noise_only"),
        dry_run=False,
    )

    aggregate_json = _read_json(
        tmp_path / "study" / "optics_plate_scale_as_per_pix_fisher_summary.json"
    )
    aggregate_csv_rows = list(
        csv.DictReader(
            (tmp_path / "study" / "optics_plate_scale_as_per_pix_fisher_summary.csv").open(
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
    assert aggregate_json["candidate_base_key"] == "optics.plate_scale_as_per_pix"
    assert aggregate_json["candidate_index"] is None
    assert {row["noise_mode"] for row in aggregate_csv_rows} == {
        "noiseless",
        "shot_noise_only",
    }
    assert all(row["target"] == "ALPHA_CEN" for row in aggregate_csv_rows)
    assert all(row["case_status"] == "ok" for row in aggregate_csv_rows)
    assert (tmp_path / "study" / "progress.log").exists()


def test_run_candidate_fisher_screen_supports_indexed_candidate_stubbed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_minimal_templates(tmp_path)
    candidate_key = "optics.primary.zernike_coeffs_nm[3]"

    class _FakeStudyModule:
        @staticmethod
        def run_obs_subblock_study(**kwargs) -> dict:
            case_root = Path(kwargs["case_root"]).resolve()
            frame_count = int(kwargs["n_frames"])
            noise_mode = "shot_noise_only" if kwargs["noise_mode"] == "enabled" else "noiseless"
            summary_path = case_root / "study" / "fisher_only" / "summary.json"
            fisher_summary_path = case_root / "study" / "fisher_only" / "fisher_summary.json"
            fisher_summary_path.parent.mkdir(parents=True, exist_ok=True)
            fisher_summary = {
                "mode": "fisher_only",
                "candidate_parameter": candidate_key,
                "candidate_base_key": "optics.primary.zernike_coeffs_nm",
                "candidate_index": 3,
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
                "f_pp": 10.0,
                "i_marg": 4.0,
                "sigma_cond": 1.0 / math.sqrt(10.0),
                "sigma_marg": 0.5,
                "absorption_fraction": 0.6,
                "f_pp_is_finite": True,
                "i_marg_is_finite": True,
                "valid_conditional_sigma": True,
                "valid_marginal_sigma": True,
                "marginalization_status": "ok",
                "nuisance_block_status": "ok",
                "noise_audit": {
                    "variance_model": "provided_cube",
                    "variance_source": "provided_cube",
                    "cube_stats": {"mean": 10.0},
                    "variance_stats": {"mean": 12.0},
                    "data_as_variance_stats": {"mean": 8.0},
                    "render_variance_stats": {"mean": 12.0},
                    "data_variance_floor_clipped_count": 0,
                    "variance_mean_over_cube_mean": 1.2,
                    "data_variance_mean_over_cube_mean": 0.8,
                    "render_variance_mean_over_cube_mean": 1.2,
                },
                "artifacts": {
                    "fisher_summary_json": str(fisher_summary_path.resolve()),
                    "fisher_blocks_npz": str(
                        (case_root / "study" / "fisher_only" / "fisher_blocks.npz").resolve()
                    ),
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
        "resolve_candidate_truth_and_target",
        lambda **_kwargs: (12.5, "ALPHA_CEN"),
    )

    summary = module.run_candidate_fisher_screen(
        study_root=tmp_path / "study",
        candidate_key=candidate_key,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        frame_counts=(1,),
        noise_modes=("noiseless",),
        dry_run=False,
    )

    aggregate_json = _read_json(
        tmp_path / "study" / "optics_primary_zernike_coeffs_nm_i3_fisher_summary.json"
    )
    aggregate_csv_rows = list(
        csv.DictReader(
            (
                tmp_path / "study" / "optics_primary_zernike_coeffs_nm_i3_fisher_summary.csv"
            ).open("r", encoding="utf-8", newline="")
        )
    )

    assert summary["candidate"] == candidate_key
    assert summary["candidate_base_key"] == "optics.primary.zernike_coeffs_nm"
    assert summary["candidate_index"] == 3
    assert aggregate_json["candidate"] == candidate_key
    assert aggregate_json["candidate_base_key"] == "optics.primary.zernike_coeffs_nm"
    assert aggregate_json["candidate_index"] == 3
    assert len(aggregate_csv_rows) == 1
    assert aggregate_csv_rows[0]["candidate"] == candidate_key
    assert aggregate_csv_rows[0]["candidate_base_key"] == "optics.primary.zernike_coeffs_nm"
    assert aggregate_csv_rows[0]["candidate_index"] == "3"


def test_run_candidate_fisher_screen_reuses_existing_case_outputs_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_minimal_templates(tmp_path)
    study_root = tmp_path / "study"
    case = module.build_candidate_fisher_case_specs(
        study_root=study_root,
        frame_counts=(1,),
        noise_modes=("noiseless",),
    )[0]
    _write_existing_case_outputs(
        case.case_root,
        candidate_key=case.candidate_key,
        candidate_base_key=case.candidate_base_key,
        candidate_index=case.candidate_index,
        truth_value=0.006,
        frame_count=case.frame_count,
        noise_mode=case.noise_mode,
        stale_case_root=tmp_path / "copied_from_elsewhere" / case.case_name,
    )

    class _FailIfCalledStudyModule:
        @staticmethod
        def run_obs_subblock_study(**_kwargs) -> dict:
            raise AssertionError("run_obs_subblock_study should not be called")

    monkeypatch.setattr(module, "_load_study_module", lambda: _FailIfCalledStudyModule)
    monkeypatch.setattr(
        module,
        "resolve_candidate_truth_and_target",
        lambda **_kwargs: (0.006, "ALPHA_CEN"),
    )

    summary = module.run_candidate_fisher_screen(
        study_root=study_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        frame_counts=(1,),
        noise_modes=("noiseless",),
        dry_run=False,
    )

    aggregate_csv_rows = list(
        csv.DictReader(
            (study_root / "optics_plate_scale_as_per_pix_fisher_summary.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            )
        )
    )
    row = aggregate_csv_rows[0]
    assert summary["successful_case_count"] == 1
    assert row["case_status"] == "ok"
    assert row["case_summary_path"] == str(
        (case.case_root / "study" / "fisher_only" / "summary.json").resolve()
    )
    assert row["fisher_summary_json"] == str(
        (case.case_root / "study" / "fisher_only" / "fisher_summary.json").resolve()
    )
    assert row["fisher_blocks_npz"] == str(
        (case.case_root / "study" / "fisher_only" / "fisher_blocks.npz").resolve()
    )
    assert "case.reuse" in (study_root / "progress.log").read_text(encoding="utf-8")


def test_run_candidate_fisher_screen_falls_back_to_existing_case_output_after_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script_module()
    trace_template, render_template, inference_template = _write_minimal_templates(tmp_path)
    study_root = tmp_path / "study"
    case = module.build_candidate_fisher_case_specs(
        study_root=study_root,
        frame_counts=(1,),
        noise_modes=("noiseless",),
    )[0]
    _write_existing_case_outputs(
        case.case_root,
        candidate_key=case.candidate_key,
        candidate_base_key=case.candidate_base_key,
        candidate_index=case.candidate_index,
        truth_value=0.006,
        frame_count=case.frame_count,
        noise_mode=case.noise_mode,
        stale_case_root=tmp_path / "copied_from_elsewhere" / case.case_name,
    )

    class _AlwaysFailStudyModule:
        @staticmethod
        def run_obs_subblock_study(**_kwargs) -> dict:
            raise RuntimeError("SVD did not converge")

    monkeypatch.setattr(module, "_load_study_module", lambda: _AlwaysFailStudyModule)
    monkeypatch.setattr(
        module,
        "resolve_candidate_truth_and_target",
        lambda **_kwargs: (0.006, "ALPHA_CEN"),
    )

    summary = module.run_candidate_fisher_screen(
        study_root=study_root,
        trace_template=trace_template,
        render_template=render_template,
        inference_template=inference_template,
        frame_counts=(1,),
        noise_modes=("noiseless",),
        reuse_existing_cases=False,
        dry_run=False,
    )

    aggregate_csv_rows = list(
        csv.DictReader(
            (study_root / "optics_plate_scale_as_per_pix_fisher_summary.csv").open(
                "r",
                encoding="utf-8",
                newline="",
            )
        )
    )
    assert summary["successful_case_count"] == 1
    assert aggregate_csv_rows[0]["case_status"] == "ok"
    log_text = (study_root / "progress.log").read_text(encoding="utf-8")
    assert "case.reuse_after_error" in log_text
    assert "SVD did not converge" in log_text
