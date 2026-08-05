import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path("examples/scripts/aggregate_full_fidelity_information_rate_family.py")
COMMIT = "421fd09550b9083cbe071051d0b574620e2a31aa"


def load_module():
    spec = importlib.util.spec_from_file_location("aggregate_full_fidelity_information_rate_family", SCRIPT)
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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def run_script(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(**os_environ(), PYTHONPATH="src:.")
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=Path.cwd(), env=env, text=True, capture_output=True, check=False)


def os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


def _run_name(amplitude_token: str, draw: int) -> str:
    return f"ff_howfe_production_center_cond_m2_hoke_{amplitude_token}nm_xp0p0_yp0p0_w10x30_draw_{draw:03d}"


def make_review_root(
    base: Path,
    *,
    amplitude_token: str = "0p5",
    draw: int = 0,
    separation_mode: int = 1,
    projected_count: int = 0,
    clipped: int = 0,
    commit: str = COMMIT,
    seq_rows: int = 40,
    inconsistent_sigma: bool = False,
    include_quasi: bool = True,
    schedule_variant: bool = False,
) -> Path:
    run_name = _run_name(amplitude_token, draw)
    root = base / run_name
    idir = root / "information_rate"
    idir.mkdir(parents=True, exist_ok=True)
    (root / ".information_rate_complete").write_text(json.dumps({"git_commit": commit}), encoding="utf-8")
    write_json(
        root / "review_warnings.json",
        {"warnings": [{"status": "initial_prior_mean_context_differs"}] + ([{"status": "tiny_negative_information_projected_to_psd"}] if projected_count else [])},
    )
    write_df(
        root / "separation_error_summary.csv",
        pd.DataFrame([{"final_sep_err_uas": 10.0 + draw, "final_posterior_sigma_sep_uas": 2.5}]),
    )
    write_df(root / "slow_parameter_error_summary.csv", pd.DataFrame([{"parameter_label": "source.separation_as", "final_err": 1.0}]))
    settings = {
        "tail_windows": 3,
        "thresholds": [0.1, 1.0, 3.0, 10.0, 30.0],
        "adaptive_cadence_min_subblocks": 1,
        "adaptive_cadence_max_subblocks": 5,
        "adaptive_cadence_mode_sets": ["astrometric_core", "source_core", "high_information_calibration", "all_trackable"],
        "adaptive_cadence_gain_thresholds": [0.1, 1.0, 3.0, 10.0, 30.0],
        "adaptive_cadence_high_information_wfe_count": 2,
        "quasi_degeneracy_rtol": 0.01,
    }
    write_json(
        idir / "information_rate_summary.json",
        {
            "schema_version": "full_fidelity_information_rate_review.v1",
            "status": "ok",
            "settings": settings,
            "summary_inventory_counts": {"discovered": 300, "accepted": 300},
            "adaptive_sequential_settings": {
                "mode_sets": settings["adaptive_cadence_mode_sets"],
                "gain_thresholds": settings["adaptive_cadence_gain_thresholds"],
                "high_information_wfe_count": 2,
            },
            "quasi_degeneracy_settings": {"quasi_degeneracy_rtol": 0.01},
        },
    )
    accepted = [True] * 300
    projected = [i < projected_count for i in range(300)]
    write_df(
        idir / "information_rate_input_inventory.csv",
        pd.DataFrame(
            {
                "accepted_status": accepted,
                "psd_projection_applied": projected,
                "psd_projection_clipped_eigenvalue_count": [1 if p else 0 for p in projected],
                "raw_minimum_eigenvalue": [-1e-12 if p else 1.0 for p in projected],
                "projection_relative_frobenius_delta": [2e-14 if p else 0.0 for p in projected],
                "projection_max_abs_delta": [1e-12 if p else 0.0 for p in projected],
                "projection_status": ["clipped_tiny_negative" if p else "not_needed" for p in projected],
                "clipping_status": ["clipped_tiny_negative" if p else "not_clipped" for p in projected],
            }
        ),
    )
    mode_ids = [0, 1, 2, 3, 4]
    rates = pd.DataFrame(
        {
            "canonical_mode_id": mode_ids,
            "canonical_eigenvalue_rate": [0.02, 1.2, 0.9, 0.4, 0.25],
            "information_replacement_timescale_s": [50, 0.83, 1.1, 2.5, 4.0],
            "gain_at_1s": [0.02, 1.2, 0.9, 0.4, 0.25],
            "gain_at_5s": [0.1, 6.0, 4.5, 2.0, 1.25],
            "gain_at_10s": [0.2, 12.0, 9.0, 4.0, 2.5],
            "gain_at_30s": [0.6, 36.0, 27.0, 12.0, 7.5],
            "gain_at_300s": [6.0, 360.0, 270.0, 120.0, 75.0],
            "gain_at_1800s_projected": [36.0, 2160.0, 1620.0, 720.0, 450.0],
            "first_window_rate": [0.018, 1.1, 0.8, 0.38, 0.2],
            "final_window_rate": [0.022, 1.25, 0.95, 0.42, 0.3],
            "median_window_rate": [0.02, 1.2, 0.9, 0.4, 0.25],
            "late_tail_rate": [0.02, 1.2, 0.9, 0.4, 0.25],
            "late_tail_median_window_rate": [0.02, 1.2, 0.9, 0.4, 0.25],
            "mean_window_rate": [0.02, 1.2, 0.9, 0.4, 0.25],
            "std_window_rate": [0.001, 0.1, 0.08, 0.02, 0.01],
            "window_rate_coefficient_of_variation": [0.05, 0.08, 0.09, 0.05, 0.04],
            "minimum_window_overlap": [0.95, 0.97, 0.96, 0.94, 0.93],
            "median_window_overlap": [0.98, 0.99, 0.98, 0.97, 0.96],
            "quasi_degenerate": [False, False, False, True, True],
            "dominant_physical_group": ["source", "source", "plate_scale", "m2_zernike", "source"],
            "dominant_labels": ["source.contrast", "source.separation_as", "optics.plate_scale_as_per_pix", "optics.secondary.zernike_coeffs_nm[0]", "source.log_flux_total"],
            "participation_ratio": [1.1, 1.0, 1.0, 1.5, 1.0],
        }
    )
    if separation_mode == 2:
        rates.loc[rates["canonical_mode_id"].isin([1, 2]), "dominant_labels"] = ["optics.plate_scale_as_per_pix", "source.separation_as"]
    write_df(idir / "information_rate_by_mode.csv", rates)
    win_rows = []
    for window in range(4):
        for mode in mode_ids:
            win_rows.append(
                {
                    "window_index": window,
                    "canonical_mode_id": mode,
                    "information_rate": float(rates.loc[rates["canonical_mode_id"] == mode, "canonical_eigenvalue_rate"].iloc[0]) * (0.9 + 0.05 * window),
                    "overlap_with_canonical_mode": 0.95 + 0.01 * min(window, 3),
                }
            )
    write_df(idir / "information_rate_by_window_mode.csv", pd.DataFrame(win_rows))
    loading_rows = []
    for mode in mode_ids:
        for label in ["source.separation_as", "optics.plate_scale_as_per_pix", "source.log_flux_total", "source.contrast"]:
            loading_rows.append({"canonical_mode_id": mode, "theta_label": label, "squared_composition_fraction": 1.0 if (mode == separation_mode and label == "source.separation_as") else 0.1})
    write_df(idir / "information_mode_loadings.csv", pd.DataFrame(loading_rows))
    plate_mode = 1 if separation_mode == 2 else 2
    resolution = [
        {
            "mode_set_name": "astrometric_core",
            "requested_physical_label_or_group": "source.separation_as",
            "canonical_mode_id": separation_mode,
            "canonical_rate": 1.2,
            "squared_loading_used_for_assignment": 0.95,
            "assignment_rank": 1,
            "next_best_mode": plate_mode,
            "next_best_loading": 0.05,
            "assignment_status": "ok_unique",
            "threshold_dependency": "none",
            "selected_mode_ids": f"{separation_mode};{plate_mode}",
        },
        {
            "mode_set_name": "astrometric_core",
            "requested_physical_label_or_group": "optics.plate_scale_as_per_pix",
            "canonical_mode_id": plate_mode,
            "canonical_rate": 0.9,
            "squared_loading_used_for_assignment": 0.9,
            "assignment_rank": 1,
            "next_best_mode": separation_mode,
            "next_best_loading": 0.06,
            "assignment_status": "ok_unique",
            "threshold_dependency": "none",
            "selected_mode_ids": f"{separation_mode};{plate_mode}",
        },
        {
            "mode_set_name": "source_core",
            "requested_physical_label_or_group": "source.contrast",
            "canonical_mode_id": 0,
            "canonical_rate": 0.02,
            "squared_loading_used_for_assignment": 0.8,
            "assignment_rank": 1,
            "next_best_mode": 4,
            "next_best_loading": 0.2,
            "assignment_status": "ok_unique",
            "threshold_dependency": "none",
            "selected_mode_ids": f"{separation_mode};{plate_mode};0;4",
        },
        {
            "mode_set_name": "source_core",
            "requested_physical_label_or_group": "source.log_flux_total",
            "canonical_mode_id": 4,
            "canonical_rate": 0.25,
            "squared_loading_used_for_assignment": 0.85,
            "assignment_rank": 1,
            "next_best_mode": 0,
            "next_best_loading": 0.1,
            "assignment_status": "ok_unique",
            "threshold_dependency": "none",
            "selected_mode_ids": f"{separation_mode};{plate_mode};0;4",
        },
        {
            "mode_set_name": "high_information_calibration",
            "requested_physical_label_or_group": "wfe_dominated_top_2",
            "canonical_mode_id": "3;4",
            "canonical_rate": "0.4;0.25",
            "squared_loading_used_for_assignment": np.nan,
            "assignment_rank": np.nan,
            "next_best_mode": "",
            "next_best_loading": np.nan,
            "assignment_status": "ok",
            "threshold_dependency": "none",
            "selected_mode_ids": f"{separation_mode};{plate_mode};3;4",
        },
        {
            "mode_set_name": "all_trackable",
            "requested_physical_label_or_group": "initial_trackability",
            "canonical_mode_id": f"{separation_mode};{plate_mode}",
            "canonical_rate": "1.2;0.9",
            "squared_loading_used_for_assignment": np.nan,
            "assignment_rank": np.nan,
            "next_best_mode": "",
            "next_best_loading": np.nan,
            "assignment_status": "ok",
            "threshold_dependency": "gain_threshold",
            "selected_mode_ids": f"{separation_mode};{plate_mode}",
        },
        {
            "mode_set_name": "all_trackable",
            "requested_physical_label_or_group": "initial_trackability",
            "canonical_mode_id": f"{separation_mode};{plate_mode};0",
            "canonical_rate": "1.2;0.9;0.02",
            "squared_loading_used_for_assignment": np.nan,
            "assignment_rank": np.nan,
            "next_best_mode": "",
            "next_best_loading": np.nan,
            "assignment_status": "ok",
            "threshold_dependency": "gain_threshold",
            "selected_mode_ids": f"{separation_mode};{plate_mode};0",
        },
    ]
    write_df(idir / "adaptive_mode_set_resolution.csv", pd.DataFrame(resolution))
    seq_summary_rows = []
    thresholds = [0.1, 1.0, 3.0, 10.0, 30.0]
    policies = {
        "astrometric_core": f"{separation_mode};{plate_mode}",
        "source_core": f"{separation_mode};{plate_mode};0;4",
        "high_information_calibration": f"{separation_mode};{plate_mode};3;4",
        "all_trackable": f"{separation_mode};{plate_mode}",
    }
    for scope in ["observation_carry_window_bounded", "window_restart"]:
        for policy, selected in policies.items():
            for threshold in thresholds:
                seq_summary_rows.append(
                    {
                        "case_name": "case_000",
                        "sequence_scope": scope,
                        "policy_mode_set_name": policy,
                        "gain_threshold": threshold,
                        "sigma_ratio_target": 1 / np.sqrt(1 + threshold),
                        "selected_mode_ids": selected if not (policy == "all_trackable" and threshold > 3) else selected + ";0",
                        "update_count": 2,
                        "natural_trigger_count": 2 if threshold == 3.0 else 1,
                        "maximum_latency_count": 0 if threshold == 3.0 else 1,
                        "historical_boundary_flush_count": 1,
                        "end_of_scope_flush_count": 1,
                        "first_block_length": 2,
                        "median_block_length": 2.5,
                        "final_block_length": 3,
                        "minimum_block_length": 2,
                        "maximum_block_length": 3,
                        "maximum_latency_fraction": 0.0 if threshold == 3.0 else 0.5,
                        "total_included_summaries": 300 if scope == "observation_carry_window_bounded" else 30,
                        "total_duration_s": 300.0 if scope == "observation_carry_window_bounded" else 30.0,
                        "final_precision_trace": 100.0,
                        "final_covariance_trace": 0.1,
                        "final_separation_sigma": 1e-6 + (1e-4 if inconsistent_sigma and policy == "source_core" else 0.0),
                        "final_plate_scale_sigma": 2e-6,
                        "information_only_status": "covariance_only_frozen_factor",
                        "final_information_invariance_status": "pass",
                    }
                )
    assert len(seq_summary_rows) == seq_rows
    write_df(idir / "adaptive_cadence_sequential_summary.csv", pd.DataFrame(seq_summary_rows))
    update_rows = []
    gain_rows = []
    for scope in ["observation_carry_window_bounded", "window_restart"]:
        for policy, selected in policies.items():
            for threshold in thresholds:
                for upd in range(2):
                    length = 2 + upd
                    if schedule_variant and policy == "high_information_calibration" and threshold == 3.0 and upd == 1:
                        length = 4
                    update_rows.append(
                        {
                            "case_name": "case_000",
                            "sequence_scope": scope,
                            "historical_window_index": upd if scope == "window_restart" else 0,
                            "policy_mode_set_name": policy,
                            "gain_threshold": threshold,
                            "sigma_ratio_target": 1 / np.sqrt(1 + threshold),
                            "update_index": upd,
                            "global_buffer_start_subblock": upd * 2,
                            "global_buffer_end_subblock": upd * 2 + length - 1,
                            "window_local_start_subblock": 0,
                            "window_local_end_subblock": length - 1,
                            "block_length": length,
                            "block_duration_s": float(length),
                            "cumulative_elapsed_time_s": float(sum(range(2, 2 + upd + 1))),
                            "selected_mode_ids": selected if not (policy == "all_trackable" and threshold > 3) else selected + ";0",
                            "controlling_mode_id": 0 if policy == "source_core" else separation_mode,
                            "minimum_selected_mode_gain": threshold + 0.5,
                            "maximum_selected_mode_gain": threshold + 1.0,
                            "closure_reason": "information_threshold" if threshold == 3.0 else "maximum_latency",
                            "triggered_naturally": threshold == 3.0,
                            "maximum_latency_reached": threshold != 3.0,
                            "historical_window_boundary_flush": False,
                            "end_of_scope_flush": upd == 1,
                            "information_only_status": "covariance_only_frozen_factor",
                        }
                    )
                    for mode in parse_ids(selected):
                        gain_rows.append(
                            {
                                "case_name": "case_000",
                                "sequence_scope": scope,
                                "historical_window_index": upd if scope == "window_restart" else 0,
                                "policy_mode_set_name": policy,
                                "gain_threshold": threshold,
                                "sigma_ratio_target": 1 / np.sqrt(1 + threshold),
                                "update_index": upd,
                                "canonical_mode_id": mode,
                                "mode_physical_interpretation": "source.contrast" if mode == 0 else "source.separation_as",
                                "current_relative_gain": threshold + mode + 0.1,
                                "controlling_mode": mode == (0 if policy == "source_core" else separation_mode),
                            }
                        )
    write_df(idir / "adaptive_cadence_sequential_updates.csv", pd.DataFrame(update_rows))
    write_df(idir / "adaptive_cadence_sequential_mode_gains.csv", pd.DataFrame(gain_rows))
    cand_rows = []
    prefix_rows = []
    for window in range(4):
        for threshold in [0.1, 0.25, 1.0]:
            for required in [1, 2, 4, 8]:
                cand_rows.append(
                    {
                        "case_name": "case_000",
                        "window_index": window,
                        "gain_threshold": threshold,
                        "required_top_mode_count": required,
                        "minimum_subblocks": 1,
                        "maximum_subblocks": 5,
                        "natural_crossing_prefix": 2,
                        "resolved_candidate_block_length": 2,
                        "trigger_reason": "information_threshold",
                        "controlling_modes": str(separation_mode),
                        "maximum_latency_reached": False,
                    }
                )
            prefix_rows.append({"case_name": "case_000", "window_index": window, "prefix_index": 1, "gain_threshold": threshold, "top_1_min_gain": 1.0})
    write_df(idir / "adaptive_cadence_candidates.csv", pd.DataFrame(cand_rows))
    write_df(idir / "adaptive_cadence_prefix_diagnostics.csv", pd.DataFrame(prefix_rows))
    physical_rows = []
    for scope, elapsed, sep_sigma in [
        ("late_tail_projection_30s", 30.0, 2e-6),
        ("late_tail_projection_300s", 300.0, 8e-7),
        ("late_tail_projection_1800s", 1800.0, 3e-7),
        ("frozen_factor_observation_prefix", 300.0, 1e-6),
    ]:
        for label, sigma in [("source.separation_as", sep_sigma), ("optics.plate_scale_as_per_pix", 2 * sep_sigma)]:
            physical_rows.append({"case_name": "case_000", "analysis_scope": scope, "theta_label": label, "posterior_marginal_sigma": sigma, "elapsed_time_s": elapsed})
    write_df(idir / "information_by_physical_label.csv", pd.DataFrame(physical_rows))
    overlap_rows = []
    for window in range(4):
        overlap_rows.append(
            {
                "case_name": "case_000",
                "comparison_scope": "window_rate_quasi_subspace",
                "window_index": window,
                "prefix_index": 30,
                "degeneracy_group": 0,
                "minimum_subspace_singular_value": 0.9 + 0.02 * window,
                "maximum_principal_angle_deg": 10 - window,
                "assignment_status": "quasi_subspace",
            }
        )
    write_df(idir / "mode_overlap.csv", pd.DataFrame(overlap_rows))
    write_df(idir / "degenerate_subspace_summary.csv", pd.DataFrame(columns=["case_name"]))
    quasi_df = (
        pd.DataFrame(
            [
                {
                    "case_name": "case_000",
                    "quasi_degeneracy_group": 0,
                    "member_mode_ids": "3;4",
                    "group_dimension": 2,
                    "eigenvalue_min": 0.25,
                    "eigenvalue_max": 0.4,
                    "adjacent_relative_gaps": "0.1",
                    "group_physical_composition": json.dumps({"source": 0.2, "m2_zernike": 0.8}),
                    "minimum_subspace_singular_value": 0.9,
                    "median_subspace_singular_value": 0.94,
                    "maximum_principal_angle_deg": 10.0,
                    "individual_mode_interpretation_note": "quasi-degenerate: prefer subspace stability",
                }
            ]
        )
        if include_quasi
        else pd.DataFrame(columns=["case_name", "quasi_degeneracy_group", "group_physical_composition"])
    )
    write_df(idir / "quasi_degenerate_subspace_summary.csv", quasi_df)
    return root


def parse_ids(text: str) -> list[int]:
    return [int(token) for token in str(text).split(";") if token]


def make_family(tmp_path: Path, *, inconsistent_sigma: bool = False, schedule_variant: bool = False) -> tuple[Path, Path, Path]:
    per_root = tmp_path / "per_root_psd_421fd09"
    r1 = make_review_root(per_root, amplitude_token="0p5", draw=0, separation_mode=1, projected_count=3, clipped=3, inconsistent_sigma=inconsistent_sigma, schedule_variant=schedule_variant)
    r2 = make_review_root(per_root, amplitude_token="1p0", draw=1, separation_mode=2, projected_count=10, clipped=10)
    root_list = tmp_path / "roots.txt"
    root_list.write_text(f"/cluster/{r1.name}\n/cluster/{r2.name}\n", encoding="utf-8")
    out = tmp_path / "out"
    return per_root, root_list, out


@pytest.mark.parametrize(
    "token,expected",
    [("0p01", 0.01), ("0p05", 0.05), ("0p1", 0.1), ("0p5", 0.5), ("1p0", 1.0)],
)
def test_metadata_parser_all_amplitudes(token, expected):
    module = load_module()
    meta = module.parse_run_metadata(_run_name(token, 3))
    assert meta["m2_ke_nm"] == pytest.approx(expected)
    assert meta["draw_index"] == 3
    assert meta["field_x"] == pytest.approx(0.0)
    assert meta["field_y"] == pytest.approx(0.0)
    assert meta["window_count"] == 10
    assert meta["subblocks_per_window"] == 30


def test_strict_integration_outputs_root_local_joins_and_no_plots(tmp_path):
    per_root, root_list, out = make_family(tmp_path)
    result = run_script(
        tmp_path,
        "--per-root-dir",
        str(per_root),
        "--root-list",
        str(root_list),
        "--outdir",
        str(out),
        "--expected-commit",
        COMMIT,
        "--expected-root-count",
        "2",
        "--expected-draws-per-amplitude",
        "1",
        "--expected-amplitudes",
        "0.5,1.0",
        "--strict",
        "true",
        "--no-plots",
    )
    assert result.returncode == 0, result.stderr
    inventory = read_csv(out / "family_input_inventory.csv")
    assert len(inventory) == 2
    assert set(inventory["inclusion_status"]) == {"included"}
    assignments = read_csv(out / "family_physical_mode_assignments_by_root.csv")
    sep = assignments[assignments["physical_concept"] == "source separation"].sort_values("m2_ke_nm")
    assert pd.to_numeric(sep["canonical_mode_id"]).tolist() == [1, 2]
    rates = read_csv(out / "family_physical_information_rates_by_root.csv")
    sep_rates = rates[rates["physical_concept"] == "source separation"]
    assert set(sep_rates["canonical_mode_id"]) == {1, 2}
    seq_amp = read_csv(out / "family_sequential_policy_by_amplitude.csv")
    assert {"observation_carry_window_bounded", "window_restart"} == set(seq_amp["sequence_scope"])
    all_trackable = seq_amp[seq_amp["policy_mode_set_name"] == "all_trackable"]
    assert all_trackable["selected_mode_ids"].nunique() >= 2
    assert not (out / "plots").exists()
    summary = json.loads((out / "family_information_rate_summary.json").read_text())
    assert summary["input_validation"]["roots_included"] == 2
    assert "future-controller guidance" in summary
    report = (out / "family_information_rate_summary.md").read_text()
    assert "## 10. Formal uncertainty versus actual estimator error" in report


def test_plotting_generates_expected_files(tmp_path):
    per_root, root_list, out = make_family(tmp_path)
    result = run_script(tmp_path, "--per-root-dir", str(per_root), "--root-list", str(root_list), "--outdir", str(out), "--expected-commit", COMMIT, "--strict", "true")
    assert result.returncode == 0, result.stderr
    for name in load_module().PLOT_NAMES:
        assert (out / "plots" / name).exists()


def test_non_strict_missing_file_preserves_excluded_root(tmp_path):
    per_root, root_list, out = make_family(tmp_path)
    (next(per_root.iterdir()) / "information_rate/adaptive_cadence_candidates.csv").unlink()
    result = run_script(tmp_path, "--per-root-dir", str(per_root), "--root-list", str(root_list), "--outdir", str(out), "--expected-commit", COMMIT, "--strict", "false", "--no-plots")
    assert result.returncode == 0, result.stderr
    inv = read_csv(out / "family_input_inventory.csv")
    assert "excluded" in set(inv["inclusion_status"])
    assert inv["exclusion_reason"].str.contains("missing:information_rate/adaptive_cadence_candidates.csv").any()


def test_strict_missing_file_and_expected_count_fail(tmp_path):
    per_root, root_list, out = make_family(tmp_path)
    (next(per_root.iterdir()) / "information_rate/adaptive_cadence_candidates.csv").unlink()
    result = run_script(tmp_path, "--per-root-dir", str(per_root), "--root-list", str(root_list), "--outdir", str(out), "--expected-root-count", "3", "--expected-commit", COMMIT, "--strict", "true", "--no-plots")
    assert result.returncode != 0
    assert "Strict family validation failed" in result.stderr


def test_duplicate_detection_direct(tmp_path):
    module = load_module()
    root = make_review_root(tmp_path / "a")
    inv, _, _ = module.build_input_inventory([root, root], {}, expected_commit=COMMIT, strict=False)
    assert inv["exclusion_reason"].str.contains("duplicate_review_root_name").any()


def test_gain3_event_extraction_schedule_equivalence_and_controlling_modes(tmp_path):
    per_root, root_list, out = make_family(tmp_path, schedule_variant=True)
    result = run_script(tmp_path, "--per-root-dir", str(per_root), "--root-list", str(root_list), "--outdir", str(out), "--expected-commit", COMMIT, "--strict", "true", "--no-plots")
    assert result.returncode == 0, result.stderr
    gain3 = read_csv(out / "family_gain3_acquisition_by_root.csv")
    assert (gain3["second_natural_block_length"].notna()).any()
    assert gain3["at_least_two_natural_triggers"].astype(bool).all()
    sched = read_csv(out / "family_policy_schedule_equivalence.csv")
    root_sched = sched[sched["run_name"] != "__amplitude_summary__"]
    assert {False, True}.issuperset(set(root_sched["exact_schedule_match"].astype(bool)))
    assert (~root_sched["exact_schedule_match"].astype(bool)).any()
    ctrl = read_csv(out / "family_controlling_modes_by_amplitude.csv")
    source_core = ctrl[ctrl["policy_mode_set_name"] == "source_core"]
    assert source_core["physical_interpretation"].astype(str).str.contains("contrast").any()


def test_policy_independent_covariance_rejects_material_inconsistency(tmp_path):
    per_root, root_list, out = make_family(tmp_path, inconsistent_sigma=True)
    result = run_script(tmp_path, "--per-root-dir", str(per_root), "--root-list", str(root_list), "--outdir", str(out), "--expected-commit", COMMIT, "--strict", "true", "--no-plots")
    assert result.returncode != 0
    assert "Policy-dependent final covariance" in result.stderr


def test_quasi_psd_accuracy_and_correlation_outputs(tmp_path):
    per_root, root_list, out = make_family(tmp_path)
    result = run_script(tmp_path, "--per-root-dir", str(per_root), "--root-list", str(root_list), "--outdir", str(out), "--expected-commit", COMMIT, "--strict", "true", "--no-plots")
    assert result.returncode == 0, result.stderr
    quasi = read_csv(out / "family_quasi_degenerate_subspaces_by_root.csv")
    assert not quasi.empty
    assert quasi["dominant_physical_group"].iloc[0] == "m2_zernike"
    stability = read_csv(out / "family_quasi_subspace_window_stability.csv")
    assert not stability.empty
    psd = read_csv(out / "family_psd_projection_by_amplitude.csv")
    assert int(psd["projected_matrix_count"].sum()) == 13
    assert int(psd["clipped_eigenvalue_count"].sum()) == 13
    accuracy = read_csv(out / "family_accuracy_and_information_by_root.csv")
    assert accuracy["signed_final_separation_error_uas"].tolist() == [10.0, 11.0]
    assert accuracy["absolute_final_separation_error_uas"].tolist() == [10.0, 11.0]
    corr = read_csv(out / "family_accuracy_information_correlations.csv")
    assert {"pearson_correlation", "rank_correlation"}.issubset(corr.columns)


def test_empty_optional_quasi_table_is_allowed(tmp_path):
    per_root = tmp_path / "per"
    root = make_review_root(per_root, include_quasi=False)
    out = tmp_path / "out"
    result = run_script(tmp_path, "--review-root", str(root), "--outdir", str(out), "--expected-commit", COMMIT, "--expected-root-count", "1", "--strict", "true", "--no-plots")
    assert result.returncode == 0, result.stderr
    assert (out / "family_quasi_degenerate_subspaces_by_root.csv").exists()


def test_determinism_ignoring_timestamps(tmp_path):
    per_root, root_list, out1 = make_family(tmp_path / "one")
    out2 = tmp_path / "two" / "out"
    args = ["--per-root-dir", str(per_root), "--root-list", str(root_list), "--expected-commit", COMMIT, "--strict", "true", "--no-plots"]
    r1 = run_script(tmp_path, *args, "--outdir", str(out1))
    r2 = run_script(tmp_path, *args, "--outdir", str(out2))
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    for csv_name in ["family_input_inventory.csv", "family_physical_mode_assignments_by_root.csv", "family_gain3_acquisition_by_root.csv"]:
        assert (out1 / csv_name).read_text() == (out2 / csv_name).read_text()
    j1 = json.loads((out1 / "family_information_rate_summary.json").read_text())
    j2 = json.loads((out2 / "family_information_rate_summary.json").read_text())
    for key in ["generated_at", "per_root_directory", "root_list_path", "output_file_inventory"]:
        j1.pop(key, None)
        j2.pop(key, None)
    assert j1 == j2
