"""Fixture-based tests for the S12 CPU qualification gate.

These need no cluster and no V4 data: the gate consumes JSON reports, so the
tests build reports.  The field names in the fixtures below mirror the real
output of ``diagnose_ml_dataloader.py``; if that output changes shape, these
tests are the thing that should fail.

The distinction under test throughout is three-way, not two-way: a correct
implementation measured over too short a window must come out as REMEASURE, not
as PASS and not as FAIL.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HPC = Path(__file__).resolve().parents[2] / "work/experiments/ml/hpc"
if str(HPC) not in sys.path:
    sys.path.insert(0, str(HPC))

qualify_s12_pipeline = pytest.importorskip("qualify_s12_pipeline")

EXPERIMENTS = qualify_s12_pipeline.EXPERIMENTS
PASS = qualify_s12_pipeline.PASS
REMEASURE = qualify_s12_pipeline.REMEASURE
FAIL = qualify_s12_pipeline.FAIL
qualify = qualify_s12_pipeline.qualify

COMMIT = "a" * 40


def _identity_report(experiment, *, status="ok", implementation="reference",
                     clean=False, loader="train", commit=COMMIT):
    return {
        "host": {"hostname": "cn029", "source_commit": commit},
        "identity": {
            "status": status,
            "study_id": "S12",
            "experiment_id": experiment,
            "loader": loader,
            "noise_implementation": implementation,
            "effective_noise_implementation": implementation,
            "clean_baseline": clean,
            "prepared_dataset": {
                "artifact_id": "PREP-V4-v1",
                "prepared_dataset_hash": "deadbeef",
                "sample_shape": [160, 160],
            },
        },
    }


def _throughput_report(experiment, condition, *, implementation, pairs_per_second=120.0,
                       valid=True, status="ok", sustained_pairs=2560,
                       recommended=None, commit=COMMIT, hours=37.9):
    report = _identity_report(
        experiment,
        implementation=implementation or "reference",
        clean=(condition == "clean"),
        commit=commit,
    )
    report["loader_smoke"] = {
        "status": status,
        "result": {
            "sustained_pairs_per_second": pairs_per_second,
            "sustained_measurement_valid": valid,
            "sustained_seconds": 25.0 if valid else 1.8,
            "sustained_batches": 80 if valid else 392,
            "sustained_pairs": sustained_pairs,
            "first_batch_seconds": 3.1,
            "num_workers": 4,
            "multiprocessing_method": "forkserver",
            "recommended_batches_for_valid_measurement": recommended,
        },
    }
    report["capacity_projection"] = {
        "status": "ok" if valid else "not_defensible",
        "projected_campaign_hours": hours,
    }
    return report


def _validation_report(experiment, *, exact=True, records=1024, complete=True,
                       implementation="reference", partitions=None, status="ok"):
    report = _identity_report(
        experiment, implementation=implementation, loader="validation"
    )
    if partitions is None:
        partitions = [
            {"batch_size": 32, "batches": 32, "compared_records": 1024,
             "final_batch_records": 32, "final_batch_is_partial": False, "exact": True,
             "missing_indices": [], "duplicated_indices": [], "mismatched_field_count": 0},
            {"batch_size": 17, "batches": 61, "compared_records": 1024,
             "final_batch_records": 7, "final_batch_is_partial": True, "exact": True,
             "missing_indices": [], "duplicated_indices": [], "mismatched_field_count": 0},
        ]
    report["equivalence"] = {
        "status": status,
        "result": {
            "exact": exact,
            "records": records,
            "dataset_records_available": 1024,
            "complete_manifest_compared": complete,
            "loader": "validation",
            "partitions": partitions,
        },
    }
    return report


def _write_complete_set(root: Path, **overrides) -> Path:
    """Write one fully passing report set, with optional per-file overrides."""
    root.mkdir(parents=True, exist_ok=True)
    for experiment in EXPERIMENTS:
        files = {
            f"{experiment}-identity.json": _identity_report(experiment),
            f"{experiment}-clean.json": _throughput_report(
                experiment, "clean", implementation=None, pairs_per_second=900.0, hours=5.1),
            f"{experiment}-reference.json": _throughput_report(
                experiment, "reference", implementation="reference",
                pairs_per_second=120.0, hours=37.9),
            f"{experiment}-batched.json": _throughput_report(
                experiment, "batched", implementation="batched",
                pairs_per_second=440.0, hours=10.3),
            f"{experiment}-numpy.json": _throughput_report(
                experiment, "numpy", implementation="numpy",
                pairs_per_second=820.0, hours=5.5),
            f"{experiment}-validation-exact.json": _validation_report(experiment),
        }
        for name, payload in files.items():
            if name in overrides:
                payload = overrides[name]
            if payload is None:          # sentinel: omit the file entirely
                continue
            path = root / name
            if isinstance(payload, str):  # raw text, for malformed-JSON cases
                path.write_text(payload, encoding="utf-8")
            else:
                path.write_text(json.dumps(payload), encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# The passing case
# ---------------------------------------------------------------------------
def test_complete_successful_qualification(tmp_path):
    result = qualify(_write_complete_set(tmp_path / "r"), expected_commit=COMMIT)
    assert result["status"] == PASS
    assert result["scientific_failures"] == []
    assert result["remeasurement_required"] == []
    for experiment in EXPERIMENTS:
        entry = result["per_experiment"][experiment]
        assert entry["identity"]["ok"] is True
        assert entry["frozen_validation"]["ok"] is True
        assert entry["frozen_validation"]["records"] == 1024
        assert entry["frozen_validation"]["partitions_checked"] == 2
        assert all(outcome["ok"] for outcome in entry["throughput"].values())
        comparison = entry["comparison"]
        assert comparison["batched_speedup_over_reference"] == pytest.approx(440.0 / 120.0)
        assert comparison["numpy_speedup_over_reference"] == pytest.approx(820.0 / 120.0)
        assert comparison["numpy_speedup_over_batched"] == pytest.approx(820.0 / 440.0)
        assert comparison["best_noisy_fraction_of_clean"] == pytest.approx(820.0 / 900.0)
    assert "submit ONLY S12-E01-R001" in result["recommended_action"]
    # The summary renders and the caveat is always present.
    assert "CPU data-loading capacity only" in result["caveat"]
    assert "PASS" in qualify_s12_pipeline.format_summary(result)


def test_exit_statuses_are_distinct(tmp_path):
    """A shell must be able to branch on PASS vs REMEASURE vs FAIL."""
    assert qualify_s12_pipeline.EXIT_STATUS == {PASS: 0, FAIL: 1, REMEASURE: 2}
    root = _write_complete_set(tmp_path / "r")
    assert qualify_s12_pipeline.main(["--report-dir", str(root)]) == 0


def test_main_writes_machine_readable_output(tmp_path):
    root = _write_complete_set(tmp_path / "r")
    out = tmp_path / "nested" / "qualification.json"
    assert qualify_s12_pipeline.main(
        ["--report-dir", str(root), "--json-out", str(out)]
    ) == 0
    payload = json.loads(out.read_text())
    assert payload["status"] == PASS
    assert payload["per_experiment"]["S12-E01"]["throughput"]["numpy"]["ok"] is True


# ---------------------------------------------------------------------------
# Missing and malformed reports
# ---------------------------------------------------------------------------
def test_missing_report_fails(tmp_path):
    root = _write_complete_set(tmp_path / "r", **{"S12-E02-numpy.json": None})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("missing report S12-E02-numpy.json" in m for m in result["scientific_failures"])
    assert "Do not submit" in result["recommended_action"]


def test_corrupted_json_fails_distinctly_from_missing(tmp_path):
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-batched.json": '{"identity": {'})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("not valid JSON" in m for m in result["scientific_failures"])


def test_non_object_report_fails(tmp_path):
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-clean.json": "[1, 2, 3]"})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("expected a JSON object" in m for m in result["scientific_failures"])


def test_missing_loader_smoke_block_fails(tmp_path):
    broken = _identity_report("S12-E01")
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-reference.json": broken})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("no loader_smoke block" in m for m in result["scientific_failures"])


# ---------------------------------------------------------------------------
# Identity and implementation mismatches
# ---------------------------------------------------------------------------
def test_failed_artifact_identity_fails(tmp_path):
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E02-identity.json": _identity_report("S12-E02", status="failed")},
    )
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("identity.status='failed'" in m for m in result["scientific_failures"])


def test_wrong_experiment_in_report_fails(tmp_path):
    """A report for the other experiment must not satisfy this one."""
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E01-numpy.json": _throughput_report(
            "S12-E02", "numpy", implementation="numpy")},
    )
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("is for experiment 'S12-E02', expected 'S12-E01'" in m
               for m in result["scientific_failures"])


def test_wrong_implementation_in_report_fails(tmp_path):
    """The numpy slot must actually have run numpy."""
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E01-numpy.json": _throughput_report(
            "S12-E01", "numpy", implementation="batched")},
    )
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("ran implementation 'batched', expected 'numpy'" in m
               for m in result["scientific_failures"])


def test_effective_implementation_is_what_gets_checked(tmp_path):
    """A degraded strategy must not qualify as the requested one.

    `numpy` falls back to `reference` for configurations it cannot cover, so
    qualifying on the *requested* value would certify a run that never used
    NumPy at all.
    """
    degraded = _throughput_report("S12-E01", "numpy", implementation="numpy")
    degraded["identity"]["noise_implementation"] = "numpy"
    degraded["identity"]["effective_noise_implementation"] = "reference"
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-numpy.json": degraded})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("ran implementation 'reference', expected 'numpy'" in m
               for m in result["scientific_failures"])


def test_clean_baseline_flag_must_match_the_condition(tmp_path):
    not_clean = _throughput_report("S12-E01", "clean", implementation=None)
    not_clean["identity"]["clean_baseline"] = False
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-clean.json": not_clean})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("does not have clean_baseline=true" in m for m in result["scientific_failures"])

    noisy_but_clean = _throughput_report("S12-E01", "batched", implementation="batched")
    noisy_but_clean["identity"]["clean_baseline"] = True
    root2 = _write_complete_set(tmp_path / "r2", **{"S12-E01-batched.json": noisy_but_clean})
    result2 = qualify(root2)
    assert result2["status"] == FAIL
    assert any("unexpectedly has clean_baseline=true" in m
               for m in result2["scientific_failures"])


def test_stale_source_revision_fails_when_provenance_is_available(tmp_path):
    """Stale reports in a reused directory must not qualify a new revision."""
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E01-batched.json": _throughput_report(
            "S12-E01", "batched", implementation="batched", commit="b" * 40)},
    )
    result = qualify(root, expected_commit=COMMIT)
    assert result["status"] == FAIL
    assert any("stale report from a different revision" in m
               for m in result["scientific_failures"])
    # Without an expected commit the same set passes: the check is opt-in.
    assert qualify(root)["status"] == PASS


def test_absent_provenance_is_not_treated_as_a_mismatch(tmp_path):
    """Reports predating the provenance field must not fail retroactively."""
    no_commit = _throughput_report(
        "S12-E01", "numpy", implementation="numpy", commit=None)
    del no_commit["host"]["source_commit"]
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-numpy.json": no_commit})
    assert qualify(root, expected_commit=COMMIT)["status"] == PASS


# ---------------------------------------------------------------------------
# Measurement validity: the REMEASURE path
# ---------------------------------------------------------------------------
def test_invalid_sustained_measurement_requests_remeasurement_not_failure(tmp_path):
    """A fast NumPy loader measured too briefly is REMEASURE, never FAIL."""
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E01-numpy.json": _throughput_report(
            "S12-E01", "numpy", implementation="numpy",
            pairs_per_second=6200.0, valid=False, recommended=5370)},
    )
    result = qualify(root)
    assert result["status"] == REMEASURE
    assert result["scientific_failures"] == []
    assert any("rerun with --batches 5370" in m for m in result["remeasurement_required"])
    assert "No scientific check failed" in result["recommended_action"]
    assert qualify_s12_pipeline.main(["--report-dir", str(root)]) == 2


def test_remeasurement_without_a_recommendation_still_reports_clearly(tmp_path):
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E02-numpy.json": _throughput_report(
            "S12-E02", "numpy", implementation="numpy", valid=False, recommended=None)},
    )
    result = qualify(root)
    assert result["status"] == REMEASURE
    assert any("rerun with a larger --batches" in m for m in result["remeasurement_required"])


def test_scientific_failure_dominates_remeasurement(tmp_path):
    """A failed check plus a short measurement is FAIL, not REMEASURE."""
    root = _write_complete_set(
        tmp_path / "r",
        **{
            "S12-E01-numpy.json": _throughput_report(
                "S12-E01", "numpy", implementation="numpy", valid=False, recommended=5000),
            "S12-E02-identity.json": _identity_report("S12-E02", status="failed"),
        },
    )
    result = qualify(root)
    assert result["status"] == FAIL
    assert result["remeasurement_required"]  # still reported, just not decisive


def test_loader_failure_is_a_failure_not_a_remeasurement(tmp_path):
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E01-reference.json": _throughput_report(
            "S12-E01", "reference", implementation="reference", status="timeout")},
    )
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("loader_smoke.status='timeout'" in m for m in result["scientific_failures"])


# ---------------------------------------------------------------------------
# Throughput values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.0, -5.0, float("inf"), float("nan"), None, "fast"])
def test_nonfinite_or_nonpositive_throughput_fails(tmp_path, bad):
    report = _throughput_report("S12-E01", "batched", implementation="batched")
    report["loader_smoke"]["result"]["sustained_pairs_per_second"] = bad
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-batched.json": report})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("is not finite and positive" in m for m in result["scientific_failures"])


def test_zero_sampled_pairs_fails(tmp_path):
    """A rate computed over no actual pairs measures nothing."""
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E01-batched.json": _throughput_report(
            "S12-E01", "batched", implementation="batched", sustained_pairs=0)},
    )
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("no pairs were actually sampled" in m for m in result["scientific_failures"])


def test_projection_is_recomputed_when_the_report_lacks_one(tmp_path):
    report = _throughput_report("S12-E01", "numpy", implementation="numpy",
                                pairs_per_second=100.0)
    del report["capacity_projection"]
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-numpy.json": report})
    result = qualify(root)
    assert result["status"] == PASS
    hours = result["per_experiment"]["S12-E01"]["throughput"]["numpy"][
        "projected_loading_hours_600_epochs"]
    # 16384 pairs/epoch / 100 pairs/s * 600 epochs / 3600 s/h
    assert hours == pytest.approx(16384 / 100.0 * 600 / 3600.0)


# ---------------------------------------------------------------------------
# Frozen validation
# ---------------------------------------------------------------------------
def test_truncated_frozen_validation_comparison_fails(tmp_path):
    """Fewer than 1,024 records is not frozen-manifest equivalence."""
    partial = _validation_report("S12-E01", records=512, complete=False)
    partial["equivalence"]["result"]["partitions"][0]["compared_records"] = 512
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-validation-exact.json": partial})
    result = qualify(root)
    assert result["status"] == FAIL
    failures = " ".join(result["scientific_failures"])
    assert "compared 512 records, expected 1024" in failures
    assert "complete_manifest_compared is false" in failures


def test_single_partition_frozen_validation_fails(tmp_path):
    """One divisible width never exercises a partial final batch."""
    one = _validation_report("S12-E02")
    one["equivalence"]["result"]["partitions"] = one["equivalence"]["result"]["partitions"][:1]
    root = _write_complete_set(tmp_path / "r", **{"S12-E02-validation-exact.json": one})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("expected at least 2 batch-size partitions" in m
               for m in result["scientific_failures"])


def test_missing_duplicated_or_mismatched_records_fail(tmp_path):
    broken = _validation_report("S12-E01")
    broken["equivalence"]["result"]["partitions"][1].update(
        missing_indices=[7], duplicated_indices=[3], mismatched_field_count=2, exact=False)
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-validation-exact.json": broken})
    result = qualify(root)
    assert result["status"] == FAIL
    failures = " ".join(result["scientific_failures"])
    assert "missing_indices" in failures
    assert "duplicated_indices" in failures
    assert "mismatched field(s)" in failures


def test_non_exact_frozen_validation_fails(tmp_path):
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E02-validation-exact.json": _validation_report("S12-E02", exact=False)},
    )
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("not exact" in m for m in result["scientific_failures"])


@pytest.mark.parametrize("implementation", ["reference", "batched"])
def test_jax_backed_validation_backends_are_accepted(tmp_path, implementation):
    root = _write_complete_set(
        tmp_path / "r",
        **{f"{experiment}-validation-exact.json": _validation_report(
            experiment, implementation=implementation)
           for experiment in EXPERIMENTS},
    )
    result = qualify(root)
    assert result["status"] == PASS
    assert result["per_experiment"]["S12-E01"]["frozen_validation"][
        "validation_noise_implementation"] == implementation


def test_numpy_backed_frozen_validation_fails(tmp_path):
    """NumPy here would mean the frozen validation realization had changed."""
    root = _write_complete_set(
        tmp_path / "r",
        **{"S12-E01-validation-exact.json": _validation_report(
            "S12-E01", implementation="numpy")},
    )
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("frozen validation ran implementation 'numpy'" in m
               for m in result["scientific_failures"])


def test_equivalence_run_against_the_wrong_loader_fails(tmp_path):
    wrong = _validation_report("S12-E01")
    wrong["identity"]["loader"] = "train"
    root = _write_complete_set(tmp_path / "r", **{"S12-E01-validation-exact.json": wrong})
    result = qualify(root)
    assert result["status"] == FAIL
    assert any("expected 'validation'" in m for m in result["scientific_failures"])


# ---------------------------------------------------------------------------
# Operational feasibility, kept separate from scientific correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rate,band", [
    (120.0, "encouraging"),
    (70.0, "encouraging"),
    (55.0, "marginal"),
    (40.0, "marginal"),
    (21.6, "inadequate_for_single_72h_allocation_without_further_evidence"),
    (None, "unknown"),
    (float("nan"), "unknown"),
])
def test_feasibility_bands(rate, band):
    assert qualify_s12_pipeline.feasibility_band(rate) == band


def test_scientifically_valid_but_operationally_marginal_still_passes(tmp_path):
    """A slow but correct implementation must not fail scientific qualification.

    It must, however, stop the report from claiming the 72-hour campaign fits.
    """
    root = _write_complete_set(
        tmp_path / "r",
        **{f"{experiment}-{condition}.json": _throughput_report(
            experiment, condition, implementation=condition, pairs_per_second=rate, hours=hours)
           for experiment in EXPERIMENTS
           for condition, rate, hours in (
               ("reference", 21.6, 126.2), ("batched", 48.0, 56.8), ("numpy", 52.0, 52.5))},
    )
    result = qualify(root)
    assert result["status"] == PASS
    assert result["scientific_failures"] == []
    action = result["recommended_action"]
    assert "passed scientifically" in action
    assert "Do not claim" in action
    bands = {
        condition: info["band"]
        for condition, info in result["per_experiment"]["S12-E01"]["feasibility"].items()
    }
    assert bands["reference"] == "inadequate_for_single_72h_allocation_without_further_evidence"
    assert bands["batched"] == "marginal"
    assert bands["numpy"] == "marginal"


def test_recommended_strategy_uses_the_worst_experiment(tmp_path):
    """The campaign must fit E02 too, so ranking uses the minimum rate."""
    overrides = {}
    for experiment, numpy_rate in (("S12-E01", 900.0), ("S12-E02", 45.0)):
        overrides[f"{experiment}-numpy.json"] = _throughput_report(
            experiment, "numpy", implementation="numpy", pairs_per_second=numpy_rate)
        # batched is uniformly 200; reference stays at the 120 default, so the
        # three strategies have distinct minima and the ranking is unambiguous.
        overrides[f"{experiment}-batched.json"] = _throughput_report(
            experiment, "batched", implementation="batched", pairs_per_second=200.0)
    result = qualify(_write_complete_set(tmp_path / "r", **overrides))
    assert result["status"] == PASS
    # numpy is fastest on E01 (900) but worst on E02 (45), so ranking by the
    # per-strategy minimum picks batched (200 on both), not numpy.
    minima = {"numpy": 45.0, "batched": 200.0, "reference": 120.0}
    assert max(minima, key=minima.get) == "batched"
    assert "'batched'" in result["recommended_action"]
    assert "200.00 pairs/s" in result["recommended_action"]


def test_no_numpy_mode_qualifies_the_jax_strategies_alone(tmp_path):
    """Qualifying without NumPy must not require a NumPy report."""
    root = _write_complete_set(tmp_path / "r", **{
        f"{experiment}-numpy.json": None for experiment in EXPERIMENTS
    })
    assert qualify(root)["status"] == FAIL
    result = qualify(root, require_numpy=False)
    assert result["status"] == PASS
    assert "numpy" not in result["per_experiment"]["S12-E01"]["throughput"]
    assert qualify_s12_pipeline.main(["--report-dir", str(root), "--no-numpy"]) == 0


def test_single_experiment_selection(tmp_path):
    root = _write_complete_set(tmp_path / "r", **{
        f"S12-E02-{name}.json": None
        for name in ("identity", "clean", "reference", "batched", "numpy",
                     "validation-exact")
    })
    result = qualify(root, experiments=("S12-E01",))
    assert result["status"] == PASS
    assert result["experiments"] == ["S12-E01"]


# ---------------------------------------------------------------------------
# Launch-preview isolation: the submission helper must expose the output paths
# the runbook asserts on.  The sbatch command carries only scheduler options,
# so without these fields a preview cannot show where a 72-hour run would write
# or which sampler it would use.
# ---------------------------------------------------------------------------
def _submit_module():
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "work/experiments/ml/hpc/submit_study_run.py"
    spec = importlib.util.spec_from_file_location("submit_for_qualification_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preview(tmp_path, capsys, **extra):
    module = _submit_module()
    repo = Path(__file__).resolve().parents[2]
    argv = [
        "--site", "gattaca2",
        "--study", str(repo / "work/experiments/ml/s12/study.yaml"),
        "--repo-root", str(repo),
        "--conda-prefix", str(tmp_path / "env"),
        "--prepared-root", str(tmp_path / "prepared"),
        "--split-registry", str(tmp_path / "split.json"),
        "--scaler", str(tmp_path / "scaler.json"),
        "--validation-manifest", str(tmp_path / "validation"),
        "--test-manifest", str(tmp_path / "test"),
        "--artifact-lock", str(tmp_path / "lock.json"),
        "--noisy-eval-artifact", str(tmp_path / "noisy.json"),
        "--run-dir", str(tmp_path / "runs"),
        "--log-root", str(tmp_path / "logs"),
        "--persist-dir", str(tmp_path / "persist"),
        "--launch-packet", str(tmp_path / "launch"),
        "--source-commit", COMMIT,
        "--plan-preview",
    ]
    for key, value in extra.items():
        argv[:0] = [f"--{key.replace('_', '-')}", value]
    assert module.main(argv) == 0
    return json.loads(capsys.readouterr().out)


def test_preview_exposes_isolated_output_paths_and_strategy(tmp_path, capsys):
    """--run-dir, --persist-dir, --log-root and the strategy must be visible."""
    payload = _preview(tmp_path, capsys, noise_implementation="numpy")
    assert payload["submission_performed"] is False
    assert payload["run_count"] == 6
    assert {c["run_id"] for c in payload["commands"]} == {
        f"S12-E{e:02d}-R{r:03d}" for e in (1, 2) for r in (1, 2, 3)
    }
    for entry in payload["commands"]:
        submitted = entry["submitted_environment"]
        logs = entry["log_paths"]
        assert entry["missing_required_environment"] == []
        # Each output family under its own root, nested by run identity.
        assert submitted["ML_RUN_DIR"].endswith(
            f"runs/S12/{entry['experiment_id']}/{entry['run_id']}")
        assert submitted["ML_PERSIST_DIR"].endswith(
            f"persist/S12/{entry['experiment_id']}/{entry['run_id']}")
        assert str(tmp_path / "logs") in logs["output"]
        assert str(tmp_path / "logs") in logs["error"]
        assert logs["output"].endswith("%x-%j.out")
        assert logs["error"].endswith("%x-%j.err")
        # Run identity and the explicitly selected strategy.
        assert submitted["ML_SOURCE_COMMIT"] == COMMIT
        assert submitted["DLUXSHERA_NOISE_IMPLEMENTATION"] == "numpy"
        # No overwrite: that would delete the only record of the failed attempts.
        assert "ML_OVERWRITE" not in submitted
    assert (tmp_path / "launch" / "launch_preview.json").exists()


def test_preview_output_paths_stay_outside_the_source_checkout(tmp_path, capsys):
    """A 72-hour run must not write into the versioned source worktree."""
    repo = str(Path(__file__).resolve().parents[2])
    payload = _preview(tmp_path, capsys, noise_implementation="batched")
    for entry in payload["commands"]:
        for value in (
            entry["submitted_environment"]["ML_RUN_DIR"],
            entry["submitted_environment"]["ML_PERSIST_DIR"],
            entry["log_paths"]["output"],
            entry["log_paths"]["error"],
        ):
            assert repo not in value, value


def test_preview_reports_the_default_strategy_when_unset(tmp_path, capsys):
    """Reference remains the default in the submission path."""
    payload = _preview(tmp_path, capsys)
    strategies = {
        c["submitted_environment"]["DLUXSHERA_NOISE_IMPLEMENTATION"]
        for c in payload["commands"]
    }
    assert strategies == {"reference"}
