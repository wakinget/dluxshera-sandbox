import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dluxshera.inference.observation_belief import ObservationBeliefState, ObservationLikelihoodState, SubblockSummary
from dluxshera.inference.observation_summary import load_subblock_summary


SCRIPT = Path("examples/scripts/analyze_full_fidelity_binary_iterative_campaign.py")


def load_analyzer_module():
    spec = importlib.util.spec_from_file_location("analyze_full_fidelity_binary_iterative_campaign", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _summary_arrays(theta_ref: np.ndarray, info: np.ndarray, target: np.ndarray) -> np.ndarray:
    return info @ theta_ref - info @ target


def make_cumulative_run(
    tmp_path: Path,
    *,
    n_windows: int = 3,
    n_subblocks: int = 2,
    missing_summary: bool = False,
    missing_prior: bool = False,
    label_mismatch: bool = False,
    scale_conflict: bool = False,
    duration_conflict: bool = False,
    missing_duration: bool = False,
    shuffle_status: bool = True,
    include_adaptive_labels: bool = False,
    prior_context_mismatch: bool = False,
    tiny_negative_information: bool = False,
    material_indefinite_information: bool = False,
) -> Path:
    root = tmp_path / "synthetic_run"
    case = "case_000"
    if include_adaptive_labels:
        labels = [
            "source.separation_as",
            "source.contrast",
            "source.log_flux_total",
            "optics.plate_scale_as_per_pix",
            "optics.primary.zernike_coeffs_nm[0]",
            "optics.secondary.zernike_coeffs_nm[0]",
        ]
        truth = np.zeros(len(labels))
        prior_mean = np.array([0.006, -0.3, 0.1, 0.01, 0.2, -0.1])
        prior_sigma = np.array([0.1, 1.5, 0.5, 0.02, 1.0, 1.0])
    else:
        labels = ["source.separation_as", "source.contrast"]
        truth = np.array([0.0, 0.0])
        prior_mean = np.array([0.006, -0.3])
        prior_sigma = np.array([0.1, 1.5])
    prior = ObservationBeliefState.from_diagonal_prior(theta_labels=labels, mean=prior_mean, sigma=prior_sigma)
    table_prior_mean = prior_mean + (0.001 if prior_context_mismatch else 0.0)
    plan = {
        "schema_version": "synthetic",
        "run_root": f"/cluster/results/{root.name}",
        "theta_layout": {"labels": labels, "size": len(labels), "label_groups": ["source", "source"]},
        "prior_truth_by_label": dict(zip(labels, truth.tolist())),
        "trace_source": ({"n_frames_per_subblock": 1} if missing_duration else {"subblock_duration_s": 1.0, "n_frames_per_subblock": 1}),
        "subblock_command_options": {"summary_information_scale": "summed_likelihood"},
        "summary_paths": {case: []},
        "expected_outputs": [],
    }
    summary = {
        "expected_output_rows": n_windows * n_subblocks,
        "missing_output_rows": 1 if missing_summary else 0,
        "completed_subblocks": n_windows * n_subblocks - (1 if missing_summary else 0),
        "failed_subblocks": 0,
        "incomplete_windows": 1 if missing_summary else 0,
        "first_failure": None,
        "existing_outputs_by_kind": {"summary": n_windows * n_subblocks},
        "windows_per_draw": n_windows,
        "subblocks_per_window": n_subblocks,
        "update_gain": 1.0,
        "update_mode": "physical_full",
        "iterative_window_diagnostic_rows": n_windows,
    }
    write_json(root / "campaign_plan.json", plan)
    write_json(root / "campaign_summary.json", summary)
    write_json(root / "model_split/model_split.json", {"components": {}, "artifact_paths": {}})
    write_json(root / "model_split/model_split_summary.json", {"components": {}, "artifact_paths": {}})
    write_json(root / "noise/noise_request_normalized.json", {"enabled": False})
    write_json(root / "noise/noise_render_provenance.json", {"mode": "none"})
    write_json(root / "noise/noise_inference_provenance.json", {"mode": "none"})
    write_df(root / "trajectory/smear_summary.csv", pd.DataFrame([{"subblock_index": 0, "window_index": 0}]))
    if not missing_prior:
        write_df(
            root / "prior_draws.csv",
            pd.DataFrame(
                {
                    "case_name": [case] * len(labels),
                    "theta_label": labels,
                    "truth_value": truth,
                    "prior_mean": table_prior_mean,
                    "reference_value": table_prior_mean,
                    "prior_sigma": prior_sigma,
                }
            ),
        )

    status_rows = []
    win_rows = []
    current = prior_mean.copy()
    for w in range(n_windows):
        window_summaries = []
        window_targets = []
        wdir = root / f"cases/{case}/windows/window_{w:03d}"
        for s in range(n_subblocks):
            if include_adaptive_labels:
                theta_ref = prior_mean + np.array([0.0001 * (w + s + 1), 0.02 * s, 0.01 * w, 0.00001 * s, 0.01 * w, -0.02 * s])
                diag = np.array([200.0, 0.8, 20.0, 500.0, 1000.0 + 20 * s, 900.0 + 30 * w])
                info = np.diag(diag)
                info[0, 3] = info[3, 0] = 5.0
                target = np.array([0.001 + 0.0002 * w, 0.08 - 0.01 * s, 0.02, 0.01001, 0.0, 0.0])
            else:
                theta_ref = np.array([0.001 * (w + 1) + 0.0001 * s, -0.2 + 0.05 * w + 0.02 * s])
                info = np.array([[2.0e8 + 2.0e7 * s, 1.5e3], [1.5e3, 2.5 + w]])
                target = np.array([0.001 + 0.0002 * w, 0.08 - 0.01 * s])
            if tiny_negative_information and w == 0 and s == 0:
                info = np.diag([3.0e12, -0.1])
            if material_indefinite_information and w == 0 and s == 0:
                info = np.diag([1.0, -1.0e-3])
            score = _summary_arrays(theta_ref, info, target)
            actual_labels = ["source.contrast", "source.separation_as"] if label_mismatch and w == 1 and s == 0 else labels
            if actual_labels != labels:
                theta_ref_to_write = theta_ref[::-1]
                info_to_write = info[::-1, ::-1]
                score_to_write = score[::-1]
            else:
                theta_ref_to_write = theta_ref
                info_to_write = info
                score_to_write = score
            schur = root / f"subblock_runs/{case}/window_{w:03d}/subblock_{s:03d}/study/schur_summary"
            schur.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                schur / "subblock_summary_matrices.npz",
                theta_ref=theta_ref_to_write,
                reduced_information=info_to_write,
                reduced_score=score_to_write,
            )
            scale = "optimizer" if scale_conflict and w == 1 and s == 0 else "summed_likelihood"
            accounting = {"summary_information_scale": scale}
            if duration_conflict and w == 1 and s == 0:
                accounting["total_exposure_time_s"] = 2.0
            write_json(
                schur / "subblock_summary.json",
                {
                    "subblock_id": f"subblock_{s:03d}_subblock_summary",
                    "summary_kind": "synthetic_schur",
                    "theta_labels": actual_labels,
                    "theta_ref": theta_ref_to_write.tolist(),
                    "matrix_artifact_path": "subblock_summary_matrices.npz",
                    "information_accounting": accounting,
                    "summary_diagnostics": {"information_accounting": accounting},
                },
            )
            recorded = f"/cluster/results/{root.name}/subblock_runs/{case}/window_{w:03d}/subblock_{s:03d}/study/schur_summary/subblock_summary.json"
            if missing_summary and w == n_windows - 1 and s == n_subblocks - 1:
                (schur / "subblock_summary.json").unlink()
            plan["summary_paths"][case].append(recorded)
            plan["expected_outputs"].append(
                {
                    "case_name": case,
                    "window_index": w,
                    "subblock_index": s,
                    "summary_path": recorded,
                    "window_case_name": f"{case}/windows/window_{w:03d}",
                }
            )
            status_rows.append(
                {
                    "case_name": case,
                    "window_case_name": f"{case}/windows/window_{w:03d}",
                    "window_index": w,
                    "window_subblock_index": s,
                    "global_subblock_index": w * n_subblocks + s,
                    "summary_path": recorded,
                    "status": "ok",
                    "return_code": 0,
                    "elapsed_seconds": 1.0,
                }
            )
            if not (missing_summary and w == n_windows - 1 and s == n_subblocks - 1):
                loaded_summary = load_subblock_summary(schur / "subblock_summary.json")
                if scale_conflict:
                    loaded_summary = SubblockSummary.from_reduced_form(
                        subblock_id=loaded_summary.subblock_id,
                        theta_labels=loaded_summary.theta_labels,
                        theta_ref=loaded_summary.theta_ref,
                        reduced_information=loaded_summary.reduced_information,
                        reduced_score=loaded_summary.reduced_score,
                    )
                window_summaries.append(loaded_summary)
                window_targets.append(target)
        if window_summaries:
            local = ObservationLikelihoodState.from_summaries(theta_labels=labels, summaries=window_summaries).combine_with_prior(prior).posterior
            posterior_mean = local.mean
            posterior_sigma = local.sigma()
        else:
            posterior_mean = current
            posterior_sigma = np.full(len(labels), np.nan)
        next_ref = posterior_mean.copy()
        post_offsets = dict(zip(labels, (posterior_mean - truth).tolist()))
        current_offsets = dict(zip(labels, (current - truth).tolist()))
        next_offsets = dict(zip(labels, (next_ref - truth).tolist()))
        write_json(
            wdir / "iterative_reference_update.json",
            {
                "case_name": case,
                "window_case_name": f"{case}/windows/window_{w:03d}",
                "window_index": w,
                "current_offsets": current_offsets,
                "posterior_offsets": post_offsets,
                "next_offsets": next_offsets,
                "truth_by_label": dict(zip(labels, truth.tolist())),
                "posterior_table_path": str(wdir / "posterior_by_label.csv"),
            },
        )
        write_df(
            wdir / "posterior_by_label.csv",
            pd.DataFrame(
                {
                    "case_name": [f"{case}/windows/window_{w:03d}"] * len(labels),
                    "theta_label": labels,
                    "truth_value": truth,
                    "reference_value": current,
                    "theta_reference_offset": current - truth,
                    "posterior_mean": posterior_mean,
                    "posterior_error": posterior_mean - truth,
                    "posterior_sigma": posterior_sigma,
                    "label_group": ["source" if str(label).startswith("source.") else "optics" for label in labels],
                    "unit": ["arcsec" if label == "source.separation_as" else "dimensionless" for label in labels],
                }
            ),
        )
        write_df(wdir / "science_summary.csv", pd.DataFrame([{"case_name": f"{case}/windows/window_{w:03d}", "posterior_separation_error_microas": posterior_mean[0] * 1e6, "posterior_separation_sigma_microas": posterior_sigma[0] * 1e6}]))
        write_df(wdir / "summary_paths.csv", pd.DataFrame({"case_name": [f"{case}/windows/window_{w:03d}"] * n_subblocks, "summary_path": plan["summary_paths"][case][w * n_subblocks : (w + 1) * n_subblocks]}))
        win_rows.append(
            {
                "case_name": case,
                "window_index": w,
                "n_subblocks": n_subblocks,
                "reference_error_norm_before": float(np.linalg.norm(current)),
                "posterior_error_norm_after": float(np.linalg.norm(posterior_mean)),
                "next_reference_error_norm": float(np.linalg.norm(next_ref)),
                "separation_reference_error_before_microas": current[0] * 1e6,
                "separation_posterior_error_after_microas": posterior_mean[0] * 1e6,
                "separation_next_reference_error_microas": next_ref[0] * 1e6,
                "posterior_sigma_separation_microas": posterior_sigma[0] * 1e6,
            }
        )
        current = next_ref
    write_json(root / "campaign_plan.json", plan)
    if shuffle_status:
        status_rows = list(reversed(status_rows))
    write_df(root / "subblock_status_iterative.csv", pd.DataFrame(status_rows))
    write_df(root / "analysis/iterative_window_diagnostics.csv", pd.DataFrame(win_rows))
    return root


def run_script(run_root: Path, outdir: Path, *args: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:."
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--run-root", str(run_root), "--outdir", str(outdir), *args],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def test_cumulative_auto_and_on_modes_success(tmp_path):
    root = make_cumulative_run(tmp_path)
    outdir = tmp_path / "review_auto"
    result = run_script(root, outdir, "--no-plots")
    assert result.returncode == 0, result.stderr
    summary = json.loads((outdir / "cumulative_information/cumulative_summary.json").read_text())
    assert summary["status"] == "ok"
    assert summary["accepted_summary_count"] == 6
    assert (outdir / "cumulative_information/cumulative_input_inventory.csv").exists()
    result_on = run_script(root, tmp_path / "review_on", "--cumulative-information", "on", "--no-plots")
    assert result_on.returncode == 0, result_on.stderr


def test_cumulative_off_mode_preserves_existing_outputs(tmp_path):
    root = make_cumulative_run(tmp_path)
    outdir = tmp_path / "review"
    result = run_script(root, outdir, "--cumulative-information", "off", "--no-plots")
    assert result.returncode == 0, result.stderr
    assert (outdir / "iterative_window_progress.csv").exists()
    assert json.loads((outdir / "cumulative_information/cumulative_summary.json").read_text())["status"] == "disabled"


def test_prefix_correctness_prior_once_and_changing_references(tmp_path):
    root = make_cumulative_run(tmp_path)
    outdir = tmp_path / "review"
    result = run_script(root, outdir, "--cumulative-information", "on", "--no-plots")
    assert result.returncode == 0, result.stderr
    analyzer = load_analyzer_module()
    plan = json.loads((root / "campaign_plan.json").read_text())
    labels = tuple(plan["theta_layout"]["labels"])
    prior, _, _ = analyzer.reconstruct_initial_observation_prior(root, plan, "case_000", labels)
    assert prior is not None
    inventory = read_csv(outdir / "cumulative_information/cumulative_input_inventory.csv")
    rows = inventory[inventory["accepted_for_cumulative"].astype(bool)].sort_values(["window_index", "subblock_index"])
    summaries = [load_subblock_summary(Path(p)) for p in rows["resolved_summary_path"]]
    prefixes = []
    for w in sorted(rows["window_index"].unique()):
        batch = [summaries[i] for i, rw in enumerate(rows.to_dict(orient="records")) if rw["window_index"] <= w]
        prefixes.append(ObservationLikelihoodState.from_summaries(theta_labels=labels, summaries=batch).combine_with_prior(prior).posterior)
    posterior = read_csv(outdir / "cumulative_information/cumulative_posterior_by_label.csv")
    for post in prefixes:
        n_sub = len(post.metadata)  # keeps loop local; values checked below by window order
    for w, direct in zip(sorted(rows["window_index"].unique()), prefixes):
        frame = posterior[posterior["window_index"] == w].sort_values("theta_label")
        expected = dict(zip(labels, direct.mean))
        for _, row in frame.iterrows():
            assert row["cumulative_posterior_mean"] == pytest.approx(expected[row["theta_label"]])
    final = read_csv(outdir / "cumulative_information/cumulative_final_summary.csv").iloc[0]
    prior_repeated_precision = prior.precision * 3.0
    likelihood = ObservationLikelihoodState.from_summaries(theta_labels=labels, summaries=summaries)
    wrong = ObservationBeliefState(theta_labels=labels, mean=prior.mean, precision=prior_repeated_precision)
    wrong_post = likelihood.combine_with_prior(wrong).posterior
    assert final["cumulative_final_sep_err_uas"] != pytest.approx(wrong_post.mean[0] * 1e6)


def test_coupled_matrix_differs_from_naive_scalar_averaging(tmp_path):
    root = make_cumulative_run(tmp_path)
    outdir = tmp_path / "review"
    assert run_script(root, outdir, "--cumulative-information", "on", "--no-plots").returncode == 0
    posterior = read_csv(outdir / "cumulative_information/cumulative_posterior_by_label.csv")
    sep = posterior[posterior["theta_label"] == "source.separation_as"].sort_values("window_index")
    local_naive = sep["window_local_posterior_mean"].mean()
    cumulative_final = sep.iloc[-1]["cumulative_posterior_mean"]
    assert cumulative_final != pytest.approx(local_naive)


def test_inventory_ordering_repeated_ids_and_relocated_absolute_paths(tmp_path):
    root = make_cumulative_run(tmp_path, shuffle_status=True)
    outdir = tmp_path / "review"
    assert run_script(root, outdir, "--cumulative-information", "on", "--no-plots").returncode == 0
    inventory = read_csv(outdir / "cumulative_information/cumulative_input_inventory.csv")
    assert inventory["subblock_index"].tolist() == [0, 1, 0, 1, 0, 1]
    assert inventory["stable_summary_id"].is_unique
    assert inventory["summary_id"].duplicated().any()
    assert set(inventory["path_resolution_method"]) == {"plan_run_root_relative"}


def test_missing_summary_auto_warns_on_fails(tmp_path):
    root = make_cumulative_run(tmp_path, missing_summary=True)
    auto = run_script(root, tmp_path / "auto", "--no-plots")
    assert auto.returncode == 0, auto.stderr
    summary = json.loads((tmp_path / "auto/cumulative_information/cumulative_summary.json").read_text())
    assert summary["status"] in {"incomplete_window", "missing_summary_artifact"}
    required = run_script(root, tmp_path / "on", "--cumulative-information", "on", "--no-plots")
    assert required.returncode != 0
    assert "Cumulative analysis" in required.stderr


def test_prior_unavailable_auto_and_on(tmp_path):
    root = make_cumulative_run(tmp_path, missing_prior=True)
    auto = run_script(root, tmp_path / "auto", "--no-plots")
    assert auto.returncode == 0, auto.stderr
    assert json.loads((tmp_path / "auto/cumulative_information/cumulative_summary.json").read_text())["status"] == "missing_prior"
    required = run_script(root, tmp_path / "on", "--cumulative-information", "on", "--no-plots")
    assert required.returncode != 0


@pytest.mark.parametrize("kwargs,status", [({"label_mismatch": True}, "label_mismatch"), ({"scale_conflict": True}, "information_scale_mismatch")])
def test_label_and_information_scale_rejections(tmp_path, kwargs, status):
    root = make_cumulative_run(tmp_path, **kwargs)
    result = run_script(root, tmp_path / "review", "--cumulative-information", "on", "--no-plots")
    assert result.returncode != 0
    assert status in result.stderr


def test_no_plots_behavior_and_json_serialization(tmp_path):
    root = make_cumulative_run(tmp_path)
    outdir = tmp_path / "review"
    assert run_script(root, outdir, "--cumulative-information", "on", "--no-plots").returncode == 0
    assert not list((outdir / "plots").glob("cumulative_*.png"))
    summary = json.loads((outdir / "cumulative_information/cumulative_summary.json").read_text())
    state = json.loads((outdir / "cumulative_information/cumulative_likelihood_state.json").read_text())
    assert summary["schema_version"] == "full_fidelity_cumulative_information_review.v1"
    assert state["schema_version"] == "full_fidelity_cumulative_information_review.v1"


def test_existing_output_regression_between_off_and_on(tmp_path):
    root = make_cumulative_run(tmp_path)
    off = tmp_path / "off"
    on = tmp_path / "on"
    assert run_script(root, off, "--cumulative-information", "off", "--no-plots").returncode == 0
    assert run_script(root, on, "--cumulative-information", "on", "--no-plots").returncode == 0
    for name in ["iterative_window_progress.csv", "posterior_by_label_combined.csv", "science_summary_combined.csv"]:
        pd.testing.assert_frame_equal(read_csv(off / name), read_csv(on / name))


def test_information_rate_auto_and_on_modes_success(tmp_path):
    root = make_cumulative_run(tmp_path, n_windows=3, n_subblocks=2)
    outdir = tmp_path / "review_auto"
    result = run_script(root, outdir, "--no-plots")
    assert result.returncode == 0, result.stderr
    summary = json.loads((outdir / "information_rate/information_rate_summary.json").read_text())
    assert summary["schema_version"] == "full_fidelity_information_rate_review.v1"
    assert summary["status"] == "ok"
    assert summary["summary_inventory_counts"]["accepted"] == 6
    assert summary["duration_provenance_counts"] == {"campaign_plan.trace_source.subblock_duration_s": 6}
    result_on = run_script(root, tmp_path / "review_on", "--information-rate", "on", "--no-plots")
    assert result_on.returncode == 0, result_on.stderr


def test_information_rate_off_and_cumulative_off_independence(tmp_path):
    root = make_cumulative_run(tmp_path)
    off = tmp_path / "off"
    assert run_script(root, off, "--information-rate", "off", "--no-plots").returncode == 0
    assert json.loads((off / "information_rate/information_rate_summary.json").read_text())["status"] == "disabled"
    independent = tmp_path / "independent"
    result = run_script(root, independent, "--cumulative-information", "off", "--information-rate", "on", "--no-plots")
    assert result.returncode == 0, result.stderr
    assert json.loads((independent / "information_rate/information_rate_summary.json").read_text())["status"] == "ok"
    assert json.loads((independent / "cumulative_information/cumulative_summary.json").read_text())["status"] == "disabled"


def test_information_rate_row_counts_repeated_ids_tail_windows_and_candidates(tmp_path):
    root = make_cumulative_run(tmp_path, n_windows=4, n_subblocks=3, shuffle_status=True)
    outdir = tmp_path / "review"
    result = run_script(
        root,
        outdir,
        "--information-rate",
        "on",
        "--information-rate-tail-windows",
        "2",
        "--adaptive-cadence-min-subblocks",
        "2",
        "--adaptive-cadence-max-subblocks",
        "3",
        "--no-plots",
    )
    assert result.returncode == 0, result.stderr
    inventory = read_csv(outdir / "information_rate/information_rate_input_inventory.csv")
    assert inventory["stable_summary_id"].is_unique
    assert inventory["summary_id"].duplicated().any()
    prefix = read_csv(outdir / "information_rate/information_prefix_by_mode.csv")
    # Per-window prefixes plus one chronological frozen-factor prefix for two modes.
    assert len(prefix[prefix["analysis_scope"] == "per_window_subblock_prefix"]) == 4 * 3 * 2
    assert len(prefix[prefix["analysis_scope"] == "frozen_factor_observation_prefix"]) == 12 * 2
    candidates = read_csv(outdir / "information_rate/adaptive_cadence_candidates.csv")
    assert len(candidates) == 4 * 3 * 4
    assert candidates["resolved_candidate_block_length"].between(2, 3).all()
    summary = json.loads((outdir / "information_rate/information_rate_summary.json").read_text())
    case_prov = next(iter(summary["canonical_spectrum_provenance"].values()))
    assert case_prov["selected_tail_windows"] == [2, 3]


def test_information_rate_duration_conflict_and_missing_duration_modes(tmp_path):
    conflict = make_cumulative_run(tmp_path / "conflict", duration_conflict=True)
    auto = run_script(conflict, tmp_path / "conflict_auto", "--information-rate", "auto", "--no-plots")
    assert auto.returncode == 0, auto.stderr
    summary = json.loads((tmp_path / "conflict_auto/information_rate/information_rate_summary.json").read_text())
    assert summary["status"] == "duration_conflict"
    required = run_script(conflict, tmp_path / "conflict_on", "--information-rate", "on", "--no-plots")
    assert required.returncode != 0
    assert "duration_conflict" in required.stderr

    missing = make_cumulative_run(tmp_path / "missing", missing_duration=True)
    auto_missing = run_script(missing, tmp_path / "missing_auto", "--information-rate", "auto", "--no-plots")
    assert auto_missing.returncode == 0, auto_missing.stderr
    assert json.loads((tmp_path / "missing_auto/information_rate/information_rate_summary.json").read_text())["status"] == "missing_duration"
    required_missing = run_script(missing, tmp_path / "missing_on", "--information-rate", "on", "--no-plots")
    assert required_missing.returncode != 0
    assert "missing_duration" in required_missing.stderr


@pytest.mark.parametrize("kwargs,status", [({"label_mismatch": True}, "label_mismatch"), ({"scale_conflict": True}, "information_scale_mismatch")])
def test_information_rate_label_and_scale_rejections(tmp_path, kwargs, status):
    root = make_cumulative_run(tmp_path, **kwargs)
    result = run_script(root, tmp_path / "review", "--information-rate", "on", "--no-plots")
    assert result.returncode != 0
    assert status in result.stderr


def test_information_rate_no_plots_and_json_serialization(tmp_path):
    root = make_cumulative_run(tmp_path)
    outdir = tmp_path / "review"
    result = run_script(root, outdir, "--information-rate", "on", "--no-plots")
    assert result.returncode == 0, result.stderr
    assert not list((outdir / "information_rate").glob("*.png"))
    summary = json.loads((outdir / "information_rate/information_rate_summary.json").read_text())
    assert summary["schema_version"] == "full_fidelity_information_rate_review.v1"


def test_adaptive_cadence_analysis_modes_outputs_and_no_plots(tmp_path):
    root = make_cumulative_run(tmp_path, n_windows=3, n_subblocks=4, include_adaptive_labels=True)
    fixed = tmp_path / "fixed"
    result_fixed = run_script(root, fixed, "--information-rate", "on", "--adaptive-cadence-analysis", "fixed_prior", "--no-plots")
    assert result_fixed.returncode == 0, result_fixed.stderr
    assert (fixed / "information_rate/adaptive_cadence_candidates.csv").exists()
    assert not (fixed / "information_rate/adaptive_cadence_sequential_updates.csv").exists()

    sequential = tmp_path / "sequential"
    result_seq = run_script(root, sequential, "--information-rate", "on", "--adaptive-cadence-analysis", "sequential", "--adaptive-cadence-min-subblocks", "1", "--adaptive-cadence-max-subblocks", "4", "--no-plots")
    assert result_seq.returncode == 0, result_seq.stderr
    assert not (sequential / "information_rate/adaptive_cadence_candidates.csv").exists()
    updates = read_csv(sequential / "information_rate/adaptive_cadence_sequential_updates.csv")
    assert not updates.empty
    assert not list((sequential / "information_rate").glob("adaptive_sequential_*.png"))

    both = tmp_path / "both"
    result_both = run_script(root, both, "--information-rate", "on", "--adaptive-cadence-analysis", "both", "--adaptive-cadence-min-subblocks", "1", "--adaptive-cadence-max-subblocks", "4", "--no-plots")
    assert result_both.returncode == 0, result_both.stderr
    assert not read_csv(both / "information_rate/adaptive_cadence_candidates.csv").empty
    assert not read_csv(both / "information_rate/adaptive_cadence_sequential_summary.csv").empty


def test_named_mode_sets_thresholds_invariance_and_window_boundaries(tmp_path):
    root = make_cumulative_run(tmp_path, n_windows=3, n_subblocks=5, include_adaptive_labels=True)
    outdir = tmp_path / "review"
    result = run_script(
        root,
        outdir,
        "--information-rate",
        "on",
        "--information-gain-thresholds",
        "0.1,0.25",
        "--adaptive-cadence-gain-thresholds",
        "1,3",
        "--adaptive-cadence-min-subblocks",
        "1",
        "--adaptive-cadence-max-subblocks",
        "5",
        "--no-plots",
    )
    assert result.returncode == 0, result.stderr
    resolution = read_csv(outdir / "information_rate/adaptive_mode_set_resolution.csv")
    assert {"astrometric_core", "source_core", "high_information_calibration", "all_trackable"}.issubset(set(resolution["mode_set_name"]))
    astro = resolution[resolution["mode_set_name"] == "astrometric_core"]
    assert set(astro["requested_physical_label_or_group"]) == {"source.separation_as", "optics.plate_scale_as_per_pix"}
    summary = read_csv(outdir / "information_rate/adaptive_cadence_sequential_summary.csv")
    carry = summary[summary["sequence_scope"] == "observation_carry_window_bounded"]
    assert not carry.empty
    assert set(carry["final_information_invariance_status"]) == {"pass"}
    assert set(carry["gain_threshold"]) == {1.0, 3.0}
    fixed_candidates = read_csv(outdir / "information_rate/adaptive_cadence_candidates.csv")
    assert set(fixed_candidates["gain_threshold"]) == {0.1, 0.25, 1.0}
    updates = read_csv(outdir / "information_rate/adaptive_cadence_sequential_updates.csv")
    carry_updates = updates[updates["sequence_scope"] == "observation_carry_window_bounded"]
    assert (carry_updates["historical_window_index"] == carry_updates["historical_window_index"]).all()
    assert (carry_updates["global_buffer_start_subblock"] <= carry_updates["global_buffer_end_subblock"]).all()


def test_invalid_adaptive_cli_and_explicit_missing_label_failures(tmp_path):
    root = make_cumulative_run(tmp_path)
    bad_threshold = run_script(root, tmp_path / "bad_threshold", "--adaptive-cadence-gain-thresholds", "1,nan", "--no-plots")
    assert bad_threshold.returncode != 0
    bad_mode = run_script(root, tmp_path / "bad_mode", "--adaptive-cadence-mode-sets", "unknown", "--no-plots")
    assert bad_mode.returncode != 0
    bad_minmax = run_script(root, tmp_path / "bad_minmax", "--adaptive-cadence-min-subblocks", "5", "--adaptive-cadence-max-subblocks", "3", "--no-plots")
    assert bad_minmax.returncode != 0
    missing = run_script(root, tmp_path / "missing", "--information-rate", "on", "--adaptive-cadence-mode-sets", "astrometric_core", "--no-plots")
    assert missing.returncode != 0
    assert "requires missing label" in missing.stderr


def test_duplicate_prior_warning_is_deduplicated(tmp_path):
    root = make_cumulative_run(tmp_path, prior_context_mismatch=True)
    outdir = tmp_path / "review"
    result = run_script(root, outdir, "--cumulative-information", "on", "--information-rate", "on", "--no-plots")
    assert result.returncode == 0, result.stderr
    warnings = json.loads((outdir / "review_warnings.json").read_text())
    statuses = [row.get("status") for row in warnings["warnings"]]
    assert statuses.count("initial_prior_mean_context_differs") == 1
    summary = json.loads((outdir / "information_rate/information_rate_summary.json").read_text())
    assert summary["warning_deduplication"]["duplicates_removed"] >= 1


def test_information_rate_tiny_negative_matrices_are_projected_downstream(tmp_path):
    root = make_cumulative_run(tmp_path, n_windows=3, n_subblocks=3, tiny_negative_information=True)
    outdir = tmp_path / "review"
    result = run_script(root, outdir, "--cumulative-information", "off", "--information-rate", "on", "--adaptive-cadence-analysis", "both", "--adaptive-cadence-min-subblocks", "1", "--adaptive-cadence-max-subblocks", "3", "--no-plots")
    assert result.returncode == 0, result.stderr
    inventory = read_csv(outdir / "information_rate/information_rate_input_inventory.csv")
    projected = inventory[inventory["psd_projection_applied"].astype(bool)]
    assert len(projected) == 1
    row = projected.iloc[0]
    assert row["minimum_eigenvalue"] < 0.0
    assert row["raw_minimum_eigenvalue"] < 0.0
    assert row["clipping_status"] == "clipped_tiny_negative"
    assert row["projection_status"] == "clipped_tiny_negative"
    assert row["psd_projection_clipped_eigenvalue_count"] == 1
    assert row["projected_minimum_eigenvalue"] >= -row["psd_tolerance"]
    prefix = read_csv(outdir / "information_rate/information_prefix_by_mode.csv")
    assert not prefix.empty
    seq = read_csv(outdir / "information_rate/adaptive_cadence_sequential_summary.csv")
    if not seq.empty:
        assert set(seq["final_information_invariance_status"]) == {"pass"}
    summary = json.loads((outdir / "information_rate/information_rate_summary.json").read_text())
    psd = summary["psd_projection_summary"]
    assert psd["matrices_requiring_projection"] == 1
    assert psd["total_clipped_eigenvalues"] == 1
    assert psd["maximum_relative_projection_correction"] > 0.0
    warnings = json.loads((outdir / "review_warnings.json").read_text())
    statuses = [item.get("status") for item in warnings["warnings"]]
    assert statuses.count("tiny_negative_information_projected_to_psd") == 1
    assert not list((outdir / "information_rate").glob("*.png"))
    assert (outdir / "information_rate/adaptive_cadence_candidates.csv").exists()


def test_information_rate_materially_indefinite_information_still_fails(tmp_path):
    root = make_cumulative_run(tmp_path, material_indefinite_information=True)
    result = run_script(root, tmp_path / "review", "--cumulative-information", "off", "--information-rate", "on", "--no-plots")
    assert result.returncode != 0
    assert "materially_indefinite_information" in result.stderr
