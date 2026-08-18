from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_single_star_both_wfe_campaigns.py"
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "run_single_star_both_wfe_campaigns",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_import_and_name_formatting() -> None:
    module = load_module()
    assert module._condition_name("logflux", 0.1) == "logflux_0p1pct"
    assert module._condition_name("platescale", 0.3) == "platescale_0p3ppm"
    assert module._condition_name("both_wfe", 0.075) == "both_wfe_0p075nm"
    assert module._condition_name("both_wfe", 1.0) == "both_wfe_1p00nm"


def test_wfe_pairing_modes_are_deterministic_and_distinct() -> None:
    module = load_module()
    kwargs = {
        "amplitude_nm": 0.3,
        "condition": "both_wfe_0p30nm",
        "draw_index": 7,
    }
    m1_ind, m2_ind = module._wfe_vectors(**kwargs, wfe_pairing="independent")
    m1_ind_again, m2_ind_again = module._wfe_vectors(**kwargs, wfe_pairing="independent")
    m1_matched, m2_matched = module._wfe_vectors(**kwargs, wfe_pairing="matched")
    m1_diff, m2_diff = module._wfe_vectors(**kwargs, wfe_pairing="differential")

    assert np.allclose(m1_ind, m1_ind_again)
    assert np.allclose(m2_ind, m2_ind_again)
    assert not np.allclose(m1_ind, m2_ind)
    assert np.allclose(m1_matched, m2_matched)
    assert np.allclose(m1_diff, -m2_diff)


def test_x64_scalar_consistency_passes() -> None:
    module = load_module()
    context = module._system_context(
        system_preset=module.DEFAULT_SYSTEM_PRESET,
        exposure_time_s=0.05,
        n_lambda=3,
    )
    parser = module._build_parser()
    args = parser.parse_args(["--campaign", module.CAMPAIGN_A])
    args.run_name = "scalar"
    config = module._runner_config(
        args=args,
        cases=[
            module.CaseSpec(
                row_index=0,
                case_name="zero_bias_control",
                condition_name="zero_bias_control",
                condition_kind="control",
                draw_index=0,
                amplitude_value=0.0,
                offsets={},
                baseline_offsets={},
                wfe_pairing="independent",
            )
        ],
        run_name="scalar_child",
        n_subblocks=1,
    )
    rows = module._scalar_consistency_rows(context, child_config=config, args=args)
    assert context.x64_enabled is True
    assert all(row["passed"] for row in rows)
    assert all("parent_truth_value" in row and "child_truth_value" in row for row in rows)
    assert all(row["parent_x64_enabled"] is True for row in rows)
    assert all(row["child_x64_enabled"] is True for row in rows)
    assert {row["theta_label"] for row in rows} == {
        "source.log_flux_total",
        "optics.plate_scale_as_per_pix",
    }


def test_plan_only_campaign_a_writes_validation_artifacts(tmp_path: Path) -> None:
    module = load_module()
    result = module.main(
        [
            "--campaign",
            module.CAMPAIGN_A,
            "--results-root",
            str(tmp_path),
            "--run-name",
            "plan_a",
            "--plan-only",
            "--n-draws",
            "2",
            "--n-subblocks",
            "1",
            "--n-frames",
            "3",
            "--num-shards",
            "4",
            "--array-throttle",
            "2",
            "--quiet",
        ]
    )
    run_root = Path(result["run_root"])
    for name in [
        "campaign_plan_validation.json",
        "campaign_case_plan.csv",
        "campaign_shard_plan.csv",
        "expected_outputs.csv",
        "scalar_consistency_check.csv",
        "resolved_config.json",
    ]:
        assert (run_root / name).exists()
    assert (run_root / "sbatch" / f"{module.CAMPAIGN_A}.sbatch").exists()
    assert (run_root / "sbatch" / f"{module.CAMPAIGN_A}_aggregate.sbatch").exists()
    assert result["wfe_pairing"] == "independent"
    assert result["expected_total_subblock_solves"] == (1 + 17 * 2) * 1
    assert result["zero_bias_scalar_consistency_passed"] is True
    expected = read_csv(run_root / "expected_outputs.csv")
    case_plan = read_csv(run_root / "campaign_case_plan.csv")
    assert len(expected) == len(case_plan)
    for expected_row, plan_row in zip(expected, case_plan):
        assert expected_row["child_results_root"] == plan_row["child_results_root"]
        assert expected_row["child_run_name"] == plan_row["child_run_name"]
        assert expected_row["case_root"] == plan_row["case_root"]
        assert expected_row["posterior_by_parameter_csv"].startswith(expected_row["case_root"])
        assert expected_row["campaign_summary_json"] == str(Path(expected_row["child_run_root"]) / "campaign_summary.json")
        assert "/analysis/campaign_summary.json" not in expected_row["campaign_summary_json"]
        assert f"_shard_{int(expected_row['shard_index']):04d}" in expected_row["child_run_name"]


def test_plan_only_campaign_b_writes_window_configs(tmp_path: Path) -> None:
    module = load_module()
    result = module.main(
        [
            "--campaign",
            module.CAMPAIGN_B,
            "--results-root",
            str(tmp_path),
            "--run-name",
            "plan_b",
            "--plan-only",
            "--n-draws",
            "2",
            "--n-subblocks",
            "1",
            "--n-frames",
            "3",
            "--windows-per-draw",
            "2",
            "--num-shards",
            "4",
            "--quiet",
        ]
    )
    run_root = Path(result["run_root"])
    assert result["n_windows"] == 2
    assert result["campaign_b_production_executable"] is True
    assert result["expected_total_subblock_solves"] == (1 + 5 * 2) * 2 * 1
    for window in range(2):
        assert (run_root / f"window_{window:02d}" / "config.json").exists()
    expected = read_csv(run_root / "expected_outputs.csv")
    assert any(row["window_index"] == "1" for row in expected)
    assert all("_window_" in row["child_run_name"] for row in expected if row["window_index"])
    by_case: dict[str, set[str]] = {}
    for row in expected:
        by_case.setdefault(row["case_name"], set()).add(row["shard_index"])
    assert all(len(shards) == 1 for shards in by_case.values())


def test_sbatch_generation_uses_real_array_and_gattaca_env(tmp_path: Path) -> None:
    module = load_module()
    result = module.main(
        [
            "--campaign",
            module.CAMPAIGN_A,
            "--results-root",
            str(tmp_path),
            "--run-name",
            "plan_sbatch",
            "--plan-only",
            "--n-draws",
            "1",
            "--n-subblocks",
            "1",
            "--n-frames",
            "3",
            "--num-shards",
            "5",
            "--array-throttle",
            "3",
            "--slurm-mem",
            "192G",
            "--slurm-time",
            "2-00:00:00",
            "--quiet",
        ]
    )
    script = (Path(result["run_root"]) / "sbatch" / f"{module.CAMPAIGN_A}.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-4%3" in script
    assert "#SBATCH --mem=192G" in script
    assert "#SBATCH --time=2-00:00:00" in script
    assert "#SBATCH --partition=compute" in script
    assert "#SBATCH --account=shera_hpc" in script
    assert "source /cm/shared/apps/miniforge/etc/profile.d/conda.sh" in script
    assert "conda activate dluxshera-py311" in script
    assert "cd ~/dluxshera-sandbox" in script
    assert "PYTHONPATH=src python examples/scripts/run_single_star_both_wfe_campaigns.py" in script
    assert 'export OMP_NUM_THREADS=1' in script
    assert 'export JAX_COMPILATION_CACHE_DIR=/scratch/shera_hpc/$USER/jax_cache' in script
    assert '--shard-index "$SLURM_ARRAY_TASK_ID"' in script
    assert "--num-shards 5" in script
    assert "--n-draws 1" in script
    assert "--n-subblocks 1" in script
    assert "--n-frames 3" in script
    assert "--array-throttle 3" in script
    assert "--slurm-mem 192G" in script
    assert "--slurm-time 2-00:00:00" in script


def test_shard_plan_assigns_every_row_once_and_deterministically(tmp_path: Path) -> None:
    module = load_module()
    args = [
        "--campaign",
        module.CAMPAIGN_A,
        "--results-root",
        str(tmp_path),
        "--run-name",
        "plan_shards",
        "--plan-only",
        "--n-draws",
        "2",
        "--num-shards",
        "7",
        "--quiet",
    ]
    first = module.main(args)
    rows_first = read_csv(Path(first["run_root"]) / "campaign_shard_plan.csv")
    second = module.main(args)
    rows_second = read_csv(Path(second["run_root"]) / "campaign_shard_plan.csv")

    assert rows_first == rows_second
    row_indices = [int(row["row_index"]) for row in rows_first]
    assert sorted(row_indices) == sorted(set(row_indices))
    assert all(int(row["shard_index"]) == int(row["row_index"]) % 7 for row in rows_first)


def test_synthetic_output_test_is_marked_non_scientific(tmp_path: Path) -> None:
    module = load_module()
    result = module.main(
        [
            "--campaign",
            module.CAMPAIGN_A,
            "--results-root",
            str(tmp_path),
            "--run-name",
            "synthetic",
            "--synthetic-output-test",
            "--n-draws",
            "1",
            "--quiet",
        ]
    )
    payload = json.loads(
        (Path(result["run_root"]) / "analysis" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    assert payload["synthetic_output_test"] is True
    assert payload["not_scientific"] is True


def test_campaign_b_update_gain_math() -> None:
    module = load_module()
    label = "optics.primary.zernike_coeffs_nm[0]"
    gain_one = module._next_offsets_from_posterior(
        current_offsets={label: 1.0},
        posterior_rows_by_label={label: {"theta_label": label, "posterior_mean": "0.25"}},
        truth_by_label={label: 0.0},
        update_gain=1.0,
    )
    gain_half = module._next_offsets_from_posterior(
        current_offsets={label: 1.0},
        posterior_rows_by_label={label: {"theta_label": label, "posterior_mean": "0.25"}},
        truth_by_label={label: 0.0},
        update_gain=0.5,
    )
    assert gain_one[label] == pytest.approx(0.25)
    assert gain_half[label] == pytest.approx(0.625)


def test_campaign_b_execute_shard_updates_window_offsets_with_fake_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()

    def fake_run_child_config(config_path: Path, results_root: Path, run_name: str, args: Any) -> None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for case in config["experiment"]["case_generation"]["cases"]:
            case_root = results_root / run_name / "cases" / case["case_name"]
            case_root.mkdir(parents=True, exist_ok=True)
            rows = ["theta_label,truth_value,reference_value,posterior_mean,posterior_sigma\n"]
            offsets = case["theta_reference_offsets"]
            labels = [label for label in offsets if "zernike_coeffs_nm" in label]
            if not labels:
                labels = ["optics.primary.zernike_coeffs_nm[0]"]
            for label in labels:
                reference = float(offsets.get(label, 0.0))
                rows.append(f"{label},0.0,{reference},0.0,1.0\n")
            (case_root / "posterior_by_parameter.csv").write_text("".join(rows), encoding="utf-8")
            (case_root / "posterior_history.csv").write_text("theta_label\n", encoding="utf-8")
            (case_root / "case_summary.json").write_text("{}", encoding="utf-8")
        (results_root / run_name).mkdir(parents=True, exist_ok=True)
        (results_root / run_name / "campaign_summary.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "_run_child_config", fake_run_child_config)
    module.main(
        [
            "--campaign",
            module.CAMPAIGN_B,
            "--results-root",
            str(tmp_path),
            "--plan-only",
            "--run-name",
            "iter_fake",
            "--n-draws",
            "1",
            "--n-subblocks",
            "1",
            "--windows-per-draw",
            "2",
            "--num-shards",
            "1",
            "--update-gain",
            "0.5",
            "--quiet",
        ]
    )
    result = module.main(
        [
            "--campaign",
            module.CAMPAIGN_B,
            "--results-root",
            str(tmp_path),
            "--run-name",
            "iter_fake",
            "--execute-shard",
            "--shard-index",
            "0",
            "--num-shards",
            "1",
            "--n-draws",
            "1",
            "--n-subblocks",
            "1",
            "--windows-per-draw",
            "2",
            "--update-gain",
            "0.5",
            "--quiet",
        ]
    )
    assert result["windows"] == 2
    run_root = tmp_path / "iter_fake"
    shard_diag = run_root / "analysis" / "shard_diagnostics" / "iterative_window_diagnostics_shard_0000.csv"
    global_diag = run_root / "analysis" / "iterative_window_diagnostics.csv"
    assert result["iterative_diagnostics_path"] == str(shard_diag)
    assert result["iterative_diagnostics_count"] > 0
    assert shard_diag.exists()
    assert not global_diag.exists()
    window0 = json.loads((run_root / "shards" / "window_00_shard_0000.json").read_text(encoding="utf-8"))
    window1 = json.loads((run_root / "shards" / "window_01_shard_0000.json").read_text(encoding="utf-8"))
    case0 = next(case for case in window0["experiment"]["case_generation"]["cases"] if case["case_name"].startswith("both_wfe"))
    case1 = next(case for case in window1["experiment"]["case_generation"]["cases"] if case["case_name"] == case0["case_name"])
    label = next(label for label in case0["theta_reference_offsets"] if "zernike_coeffs_nm" in label)
    assert case1["theta_reference_offsets"][label] == pytest.approx(0.5 * case0["theta_reference_offsets"][label])
    aggregate = module.main(
        [
            "--campaign",
            module.CAMPAIGN_B,
            "--results-root",
            str(tmp_path),
            "--run-name",
            "iter_fake",
            "--aggregate-only",
            "--n-draws",
            "1",
            "--windows-per-draw",
            "2",
            "--update-gain",
            "0.5",
            "--quiet",
        ]
    )
    assert aggregate["iterative_window_diagnostic_rows"] > 0
    assert aggregate["shard_iterative_diagnostics_file_count"] == 1
    assert global_diag.exists()
    inventory = read_csv(run_root / "analysis" / "output_inventory.csv")
    assert inventory
    assert inventory[0]["shard_iterative_diagnostics_available"] == "True"


def test_aggregate_only_writes_inventory_missing_and_posterior_concat(tmp_path: Path) -> None:
    module = load_module()
    result = module.main(
        [
            "--campaign",
            module.CAMPAIGN_A,
            "--results-root",
            str(tmp_path),
            "--run-name",
            "aggregate",
            "--plan-only",
            "--n-draws",
            "1",
            "--n-subblocks",
            "1",
            "--quiet",
        ]
    )
    run_root = Path(result["run_root"])
    expected = read_csv(run_root / "expected_outputs.csv")
    first = expected[0]
    posterior_path = Path(first["posterior_by_parameter_csv"])
    posterior_path.parent.mkdir(parents=True, exist_ok=True)
    posterior_path.write_text(
        "theta_label,truth_value,reference_value,posterior_mean,posterior_sigma\n"
        "optics.primary.zernike_coeffs_nm[0],0.0,0.3,0.1,0.05\n"
        "optics.secondary.zernike_coeffs_nm[0],0.0,-0.2,-0.1,0.05\n",
        encoding="utf-8",
    )
    summary = module.main(
        [
            "--campaign",
            module.CAMPAIGN_A,
            "--results-root",
            str(tmp_path),
            "--run-name",
            "aggregate",
            "--aggregate-only",
            "--quiet",
        ]
    )
    assert summary["existing_posterior_tables"] == 1
    assert summary["missing_posterior_tables"] == len(expected) - 1
    assert (run_root / "analysis" / "output_inventory.csv").exists()
    assert (run_root / "analysis" / "missing_outputs.csv").exists()
    assert (run_root / "analysis" / "posterior_by_parameter_all_cases.csv").exists()
    assert (run_root / "analysis" / "wfe_vector_diagnostics.csv").exists()
