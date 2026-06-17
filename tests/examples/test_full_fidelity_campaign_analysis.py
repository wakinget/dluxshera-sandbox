import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path("examples/scripts/analyze_full_fidelity_binary_iterative_campaign.py")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def make_run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    labels = [
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
        "optics.plate_scale_as_per_pix",
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[0]",
    ]
    plan = {
        "theta_layout": {"labels": labels, "size": len(labels)},
        "layout_metadata": {
            "system": {
                "system_preset": "TEST_PRESET",
                "source_target": "ALPHA_CEN",
                "source_kind": "binary_target",
                "optics_kind": "three_plane",
                "detector_layer_stack_after_global_overrides": [
                    {"name": "pixel_mtf"},
                    {"name": "smear"},
                ],
            }
        },
    }
    summary = {
        "expected_output_rows": 1,
        "missing_output_rows": 0,
        "completed_subblocks": 1,
        "failed_subblocks": 0,
        "incomplete_windows": 0,
        "first_failure": None,
        "existing_outputs_by_kind": {"summary": 1},
        "windows_per_draw": 2,
        "subblocks_per_window": 1,
        "update_gain": 0.5,
        "update_mode": "physical_full",
        "iterative_window_diagnostic_rows": 2,
    }
    split = {
        "components": {
            "spectral_model": {"enabled": True, "matched": False},
            "high_order_wfe": {
                "enabled": True,
                "truth_label": "high_order_truth",
                "inference_label": "knowledge_error",
                "matched": False,
            },
            "trajectory_smear": {"enabled": True, "mode": "subblock_constant_layer", "target_layer": "smear"},
            "detector_noise": {"enabled": True, "noise_mode": "inherit"},
        },
        "artifact_paths": {
            "spectral_spectral_comparison": str(root / "model_split/spectral/spectral_comparison.json"),
            "spectral_spectral_deck_manifest": str(root / "model_split/spectral/spectral_deck_manifest.json"),
            "spectral_spectral_moments": str(root / "model_split/spectral/spectral_moments.json"),
        },
    }
    noise = {"enabled": True, "legacy_noise_mode": "inherit", "variance_floor": 0.5, "use_render_variance_resolved": True}
    write_json(root / "campaign_plan.json", plan)
    write_json(root / "campaign_summary.json", summary)
    write_json(root / "model_split/model_split.json", split)
    write_json(root / "model_split/model_split_summary.json", split)
    write_json(root / "noise/noise_request_normalized.json", noise)
    write_json(root / "noise/noise_render_provenance.json", {"mode": "inherit"})
    write_json(root / "noise/noise_inference_provenance.json", {"mode": "inherit"})
    write_json(root / "model_split/spectral/spectral_comparison.json", {})
    write_json(root / "model_split/spectral/spectral_deck_manifest.json", {})
    write_json(root / "model_split/spectral/spectral_moments.json", {})
    write_json(root / "model_split/high_order_wfe/high_order_wfe_summary.json", {})

    win_rows = []
    for idx, before, after, next_err in [(0, 0.10, 0.08, 0.09), (1, 0.09, 0.06, 0.05)]:
        win_rows.append(
            {
                "case_name": "case_000",
                "window_index": idx,
                "n_subblocks": 1,
                "update_gain": 0.5,
                "update_mode": "physical_full",
                "reference_error_norm_before": before,
                "posterior_error_norm_after": after,
                "next_reference_error_norm": next_err,
                "residual_norm_over_bias_norm": next_err / 0.1,
                "update_cosine_with_ideal": 0.5,
                "vector_gain": 0.5,
                "applied_vector_gain": 0.25,
                "separation_reference_error_before_microas": -2.0,
                "separation_posterior_error_after_microas": -1.5,
                "separation_next_reference_error_microas": -1.0,
                "separation_update_sign_toward_truth": True,
                "separation_next_reference_improved": True,
                "posterior_sigma_separation_microas": 2.0,
                "source_scalar_reference_error_norm_before": 0.1,
                "source_scalar_posterior_error_norm_after": 0.08,
                "plate_scale_reference_error_norm_before": 0.1,
                "plate_scale_posterior_error_norm_after": 0.08,
                "m1_zernike_reference_error_norm_before": 0.1,
                "m1_zernike_posterior_error_norm_after": 0.08,
                "m2_zernike_reference_error_norm_before": 0.1,
                "m2_zernike_posterior_error_norm_after": 0.08,
            }
        )
    write_df(root / "analysis/iterative_window_diagnostics.csv", pd.DataFrame(win_rows))

    truth = {label: 0.0 for label in labels}
    truth["source.separation_as"] = 1.0
    current0 = {label: 0.1 for label in labels}
    post0 = {label: 0.08 for label in labels}
    next0 = {label: 0.05 for label in labels}
    current1 = next0
    post1 = {label: 0.03 for label in labels}
    next1 = {label: 0.02 for label in labels}
    for idx, cur, post, nxt in [(0, current0, post0, next0), (1, current1, post1, next1)]:
        wdir = root / f"cases/case_000/windows/window_{idx:03d}"
        wdir.mkdir(parents=True, exist_ok=True)
        write_json(
            wdir / "iterative_reference_update.json",
            {
                "case_name": "case_000",
                "window_index": idx,
                "update_gain": 0.5,
                "update_mode": "physical_full",
                "current_offsets": cur,
                "posterior_offsets": post,
                "next_offsets": nxt,
                "truth_by_label": truth,
                "posterior_table_path": str(wdir / "posterior_by_label.csv"),
            },
        )
        posterior = pd.DataFrame(
            {
                "case_name": [f"case_000/windows/window_{idx:03d}"] * len(labels),
                "theta_label": labels,
                "truth_value": [truth[label] for label in labels],
                "reference_value": [truth[label] + cur[label] for label in labels],
                "posterior_mean": [truth[label] + post[label] for label in labels],
                "posterior_sigma": [0.1] * len(labels),
                "label_group": ["source"] * len(labels),
                "unit": ["unit"] * len(labels),
            }
        )
        write_df(wdir / "posterior_by_label.csv", posterior)
        write_df(
            wdir / "science_summary.csv",
            pd.DataFrame(
            {
                "case_name": [f"case_000/windows/window_{idx:03d}"],
                "posterior_separation_error_microas": [-1.0],
                "posterior_separation_sigma_microas": [2.0],
            }
            ),
        )
        write_df(wdir / "iterative_window_diagnostics.csv", pd.DataFrame([win_rows[idx]]))
        write_json(wdir / "case_manifest.json", {})

    subblock_dir = root / "subblock_runs/case_000/window_000/subblock_000/study"
    write_json(subblock_dir / "subprocess_diagnostics.json", {"command": ["python", "x.py", "--n-frames", "3", "--phi-ref", "truth_when_available"], "return_code": 0})
    write_json(subblock_dir / "schur_summary/subblock_summary.json", {"information_accounting": {"n_frames_total": 3, "summary_information_scale": "summed_likelihood"}})
    (subblock_dir / "subprocess.stdout.log").write_text("")
    (subblock_dir / "subprocess.stderr.log").write_text("")
    write_df(
        root / "subblock_status_iterative.csv",
        pd.DataFrame(
        [
            {
                "case_name": "case_000",
                "summary_path": str(subblock_dir / "schur_summary/subblock_summary.json"),
                "status": "ok",
                "return_code": 0,
                "subprocess_diagnostics_path": str(subblock_dir / "subprocess_diagnostics.json"),
                "stdout_log": str(subblock_dir / "subprocess.stdout.log"),
                "stderr_log": str(subblock_dir / "subprocess.stderr.log"),
                "window_index": 0,
                "window_subblock_index": 0,
                "global_subblock_index": 0,
                "elapsed_seconds": 1.2,
            }
        ]
        ),
    )

    traj_dir = root / "trajectory/subblock_000000"
    traj_dir.mkdir(parents=True, exist_ok=True)
    write_df(traj_dir / "frame_truth.csv", pd.DataFrame({"frame": [0, 1], "source.x_position_as": [1.0, 2.0], "source.y_position_as": [0.5, 0.6]}))
    write_df(traj_dir / "starting_guess_prediction.csv", pd.DataFrame({"frame": [0, 1], "source.x_position_as": [1.1, 2.2], "source.y_position_as": [0.4, 0.7]}))
    write_df(
        root / "trajectory/smear_summary.csv",
        pd.DataFrame(
        [
            {
                "subblock_index": 0,
                "window_index": 0,
                "smear_length_pix": 0.1,
                "smear_theta_deg": 45.0,
                "render_match": True,
                "inference_match": True,
            }
        ]
        ),
    )
    write_df(root / "trajectory/trajectory_filter_summary.csv", pd.DataFrame({"column": ["source.x_position_as"], "input_rms": [1.0], "output_rms": [0.1], "removed_rms": [0.9]}))
    return root


def run_script(run_root: Path, outdir: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--run-root", str(run_root), "--outdir", str(outdir), *args],
        check=False,
        text=True,
        capture_output=True,
    )


def test_analysis_script_runs_and_writes_review_bundle(tmp_path):
    run_root = make_run_root(tmp_path)
    outdir = tmp_path / "review"
    result = run_script(run_root, outdir, "--no-plots", "--max-image-examples", "2")
    assert result.returncode == 0, result.stderr
    assert (outdir / "review_summary.md").exists()
    assert (outdir / "campaign_dashboard.csv").exists()
    assert (outdir / "mismatch_dashboard.csv").exists()
    assert (outdir / "representative_image_comparison_status.json").exists() is False


def test_missing_optional_image_artifacts_do_not_fail_with_plots(tmp_path):
    run_root = make_run_root(tmp_path)
    outdir = tmp_path / "review"
    result = run_script(run_root, outdir, "--max-image-examples", "2")
    assert result.returncode == 0, result.stderr
    status = json.loads((outdir / "representative_image_comparison_status.json").read_text())
    assert status["available"] is False


def test_strict_mode_fails_when_required_artifact_missing(tmp_path):
    run_root = make_run_root(tmp_path)
    (run_root / "campaign_summary.json").unlink()
    result = run_script(run_root, tmp_path / "review", "--strict", "--no-plots")
    assert result.returncode != 0
    assert "Missing required campaign artifacts" in result.stderr


def test_dashboard_progress_slow_state_mismatch_and_smear_outputs(tmp_path):
    run_root = make_run_root(tmp_path)
    outdir = tmp_path / "review"
    result = run_script(run_root, outdir, "--no-plots")
    assert result.returncode == 0, result.stderr

    dashboard = pd.read_csv(outdir / "campaign_dashboard.csv")
    assert {"source target / components", "iterative update settings", "subblock settings"}.issubset(set(dashboard["Component"]))

    win = pd.read_csv(outdir / "iterative_window_progress.csv")
    assert len(win) == 2

    evolution = pd.read_csv(outdir / "slow_state_evolution.csv")
    assert "initial_reference" in set(evolution["state"])
    assert any(evolution["state"].str.startswith("final_reference"))

    mismatch = pd.read_csv(outdir / "mismatch_dashboard.csv")
    assert {"spectral", "high_order_wfe", "detector_layer_stack", "trajectory_truth_model", "detector_noise_model"}.issubset(set(mismatch["component"]))

    smear = pd.read_csv(outdir / "smear_summary_review.csv")
    assert len(smear) == 1
    assert smear.loc[0, "smear_length_pix"] == pytest.approx(0.1)

    report = (outdir / "review_summary.md").read_text()
    assert "Full-fidelity binary iterative campaign review" in report
