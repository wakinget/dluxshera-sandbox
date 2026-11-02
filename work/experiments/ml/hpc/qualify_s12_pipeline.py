#!/usr/bin/env python3
"""Decide whether S12 CPU qualification passed, from the reports already written.

This reruns nothing.  It reads the JSON reports produced by
``diagnose_ml_dataloader.py`` during the Gattaca2 CPU qualification and reduces
them to one decision, because a zero exit status from the diagnostic does not
mean the measurement it produced is usable.  The specific trap this exists to
close: a diagnostic can complete successfully while reporting
``sustained_measurement_valid: false``, which happens routinely for the NumPy
strategy because it drains the requested batches before the 20-second sustained
window elapses.  That is a *fast* loader, not a failed one, and it must not be
reported as either a pass or a failure.

Three outcomes, with distinct exit statuses so a shell can branch on them:

``PASS`` (0)
    Every required scientific check passed and every required throughput
    measurement is valid.  Operational feasibility is reported separately; a
    low-but-valid throughput does not fail a scientifically correct
    implementation.
``REMEASURE`` (2)
    No scientific check failed, but at least one throughput measurement spans
    too little time to support a capacity estimate.  The report names the
    larger ``--batches`` value to use.  Rerun those conditions; nothing needs
    reinvestigating.
``FAIL`` (1)
    A required validation failed, a report is missing or malformed, or a
    scientifically significant mismatch was found.  Do not proceed to
    submission.

Expected report filenames under ``--report-dir``, matching the runbook in
``docs/dev/s12_pipeline_performance.md``::

    <EXPERIMENT>-identity.json              artifact identity
    <EXPERIMENT>-clean.json                 clean (no-noise) baseline
    <EXPERIMENT>-reference.json             reference JAX
    <EXPERIMENT>-batched.json               batched JAX
    <EXPERIMENT>-numpy.json                 deterministic per-record NumPy
    <EXPERIMENT>-validation-exact.json      frozen-validation equivalence

Operational throughput bands are engineering guidance, not scientific
acceptance limits, and a CPU loading projection never certifies a 72-hour
campaign on its own.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: Experiments the qualification covers.
EXPERIMENTS = ("S12-E01", "S12-E02")

#: Throughput conditions required per experiment, mapped to report suffix.
#: ``clean`` carries no implementation identity: it disables augmentation, so
#: the strategy recorded in its report is not meaningful and is not checked.
THROUGHPUT_CONDITIONS = (
    ("clean", None),
    ("reference", "reference"),
    ("batched", "batched"),
    ("numpy", "numpy"),
)

#: Strategies permitted for frozen validation noise.  The frozen
#: noisy-validation recipe is a JAX realization; ``numpy`` appearing here would
#: mean the frozen validation samples had changed.
JAX_BACKED_VALIDATION = ("reference", "batched")

#: Ordered validation records in the frozen S12 manifest.
EXPECTED_VALIDATION_RECORDS = 1024

#: Operational throughput bands, pairs/s. Guidance for scheduling only.
ENCOURAGING_PAIRS_PER_SECOND = 70.0
MARGINAL_PAIRS_PER_SECOND = 40.0

#: Epochs and ordered pairs per epoch in the S12 prescription, used only to
#: recompute a projection when a report lacks one.
S12_EPOCHS = 600
S12_PAIRS_PER_EPOCH = 16384

PASS, REMEASURE, FAIL = "PASS", "REMEASURE", "FAIL"
EXIT_STATUS = {PASS: 0, FAIL: 1, REMEASURE: 2}


class ReportProblem(Exception):
    """A report is missing, unreadable, or not valid JSON."""


def load_report(path: Path) -> dict:
    """Load one diagnostic report, distinguishing missing from malformed.

    Both are failures, but they call for different operator action -- rerun the
    step versus investigate a truncated write -- so the messages differ.
    """
    if not path.exists():
        raise ReportProblem(f"missing report {path.name}; the step that writes it did not run")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReportProblem(
            f"{path.name} is not valid JSON ({exc}); a truncated report usually "
            "means the diagnostic was killed mid-write"
        ) from exc
    if not isinstance(payload, dict):
        raise ReportProblem(f"{path.name} is {type(payload).__name__}, expected a JSON object")
    return payload


def _finite_positive(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def check_identity(report: dict, *, experiment: str, expected_commit: str | None) -> dict:
    """Verify experiment-aware artifact validation succeeded for ``experiment``."""
    failures: list[str] = []
    identity = report.get("identity")
    if not isinstance(identity, dict):
        return {"ok": False, "failures": ["report has no identity block"]}
    status = identity.get("status")
    if status != "ok":
        detail = identity.get("error") or identity.get("reason") or ""
        failures.append(f"identity.status={status!r} (expected 'ok') {detail}".strip())
    actual = identity.get("experiment_id")
    if actual != experiment:
        failures.append(f"report is for experiment {actual!r}, expected {experiment!r}")
    prepared = identity.get("prepared_dataset") or {}
    failures.extend(_provenance_failures(report, expected_commit=expected_commit))
    return {
        "ok": not failures,
        "failures": failures,
        "prepared_dataset_artifact_id": prepared.get("artifact_id"),
        "prepared_dataset_hash": prepared.get("prepared_dataset_hash"),
        "sample_shape": prepared.get("sample_shape"),
    }


def _provenance_failures(report: dict, *, expected_commit: str | None) -> list[str]:
    """Reject a report produced from a different source revision.

    Stale reports in a reused directory are the realistic way a qualification
    passes on evidence that does not describe the reviewed code.  Provenance is
    only checked when the report carries it; absence is reported separately
    rather than treated as a mismatch.
    """
    if not expected_commit:
        return []
    recorded = (report.get("host") or {}).get("source_commit") or report.get("source_commit")
    if recorded in (None, ""):
        return []
    if str(recorded) != str(expected_commit):
        return [
            f"report records source commit {recorded!r}, expected {expected_commit!r}; "
            "this is a stale report from a different revision"
        ]
    return []


def check_throughput(
    report: dict,
    *,
    experiment: str,
    condition: str,
    expected_implementation: str | None,
    expected_commit: str | None,
) -> dict:
    """Validate one loader-throughput report.

    Separates three distinct states that a single boolean would conflate: a
    broken loader (FAIL), a loader that ran correctly but was measured over too
    short a window (REMEASURE), and a valid measurement whose rate may still be
    operationally inadequate (PASS, reported with a band).
    """
    failures: list[str] = []
    remeasure: list[str] = []

    identity = report.get("identity")
    if isinstance(identity, dict):
        actual_experiment = identity.get("experiment_id")
        if actual_experiment != experiment:
            failures.append(
                f"report is for experiment {actual_experiment!r}, expected {experiment!r}"
            )
        if expected_implementation is not None:
            # Compare the *effective* strategy: a requested strategy that
            # cannot cover a configuration degrades to reference inside the
            # sampler, and qualifying that as the requested one would be wrong.
            effective = identity.get("effective_noise_implementation") or identity.get(
                "noise_implementation"
            )
            if effective != expected_implementation:
                failures.append(
                    f"report ran implementation {effective!r}, expected "
                    f"{expected_implementation!r}"
                )
        if condition == "clean" and not identity.get("clean_baseline"):
            failures.append("clean-baseline report does not have clean_baseline=true")
        if condition != "clean" and identity.get("clean_baseline"):
            failures.append(f"{condition} report unexpectedly has clean_baseline=true")
    else:
        failures.append("report has no identity block")
    failures.extend(_provenance_failures(report, expected_commit=expected_commit))

    smoke = report.get("loader_smoke")
    if not isinstance(smoke, dict):
        return {
            "ok": False,
            "failures": failures + ["report has no loader_smoke block"],
            "remeasure": remeasure,
        }
    if smoke.get("status") != "ok":
        failures.append(
            f"loader_smoke.status={smoke.get('status')!r} (expected 'ok'): "
            f"{smoke.get('error') or ''}".strip()
        )
    result = smoke.get("result")
    if not isinstance(result, dict):
        return {
            "ok": False,
            "failures": failures + ["loader_smoke has no result block"],
            "remeasure": remeasure,
        }

    pairs_per_second = result.get("sustained_pairs_per_second")
    if not _finite_positive(pairs_per_second):
        failures.append(
            f"sustained_pairs_per_second={pairs_per_second!r} is not finite and positive"
        )
        pairs_per_second = None
    sustained_pairs = result.get("sustained_pairs") or 0
    if int(sustained_pairs) <= 0:
        failures.append(
            f"sustained_pairs={sustained_pairs!r}; no pairs were actually sampled, so "
            "there is nothing to measure"
        )

    recommended = result.get("recommended_batches_for_valid_measurement")
    if not result.get("sustained_measurement_valid"):
        hint = (
            f"rerun with --batches {int(recommended)}"
            if recommended
            else "rerun with a larger --batches"
        )
        remeasure.append(
            f"sustained measurement spans "
            f"{float(result.get('sustained_seconds') or 0.0):.1f}s over "
            f"{int(result.get('sustained_batches') or 0)} batch(es), below the validity "
            f"gate; {hint}. A short span means the loader was fast, not broken."
        )

    projection = report.get("capacity_projection")
    projected_hours = None
    if isinstance(projection, dict):
        projected_hours = projection.get("projected_campaign_hours")
    if projected_hours is None and pairs_per_second:
        projected_hours = S12_PAIRS_PER_EPOCH / float(pairs_per_second) * S12_EPOCHS / 3600.0

    return {
        "ok": not failures,
        "failures": failures,
        "remeasure": remeasure,
        "condition": condition,
        "implementation": expected_implementation,
        "sustained_pairs_per_second": pairs_per_second,
        "sustained_measurement_valid": bool(result.get("sustained_measurement_valid")),
        "sustained_seconds": result.get("sustained_seconds"),
        "sustained_batches": result.get("sustained_batches"),
        "sustained_pairs": int(sustained_pairs),
        "first_batch_seconds": result.get("first_batch_seconds"),
        "num_workers": result.get("num_workers"),
        "multiprocessing_method": result.get("multiprocessing_method"),
        "recommended_batches_for_valid_measurement": recommended,
        "projected_loading_hours_600_epochs": projected_hours,
    }


def check_frozen_validation(report: dict, *, experiment: str) -> dict:
    """Verify complete bitwise frozen-validation equivalence.

    Requires full-manifest coverage, both partitions, and a JAX-backed
    validation strategy.  A partial comparison that reports ``exact`` is not
    equivalence over the frozen manifest, so coverage is checked before the
    verdict.
    """
    failures: list[str] = []
    equivalence = report.get("equivalence")
    if not isinstance(equivalence, dict):
        return {"ok": False, "failures": ["report has no equivalence block"]}
    if equivalence.get("status") != "ok":
        failures.append(
            f"equivalence.status={equivalence.get('status')!r} (expected 'ok'): "
            f"{equivalence.get('error') or ''}".strip()
        )
    result = equivalence.get("result")
    if not isinstance(result, dict):
        return {"ok": False, "failures": failures + ["equivalence has no result block"]}

    if not result.get("exact"):
        failures.append("equivalence result is not exact")
    records = int(result.get("records") or 0)
    if records != EXPECTED_VALIDATION_RECORDS:
        failures.append(
            f"compared {records} records, expected {EXPECTED_VALIDATION_RECORDS}"
        )
    if not result.get("complete_manifest_compared"):
        failures.append(
            "complete_manifest_compared is false; a partial sample is not "
            "frozen-manifest equivalence"
        )

    partitions = result.get("partitions")
    if not isinstance(partitions, list) or len(partitions) < 2:
        failures.append(
            f"expected at least 2 batch-size partitions, got "
            f"{0 if not isinstance(partitions, list) else len(partitions)}; one "
            "indivisible width is required to exercise a partial final batch"
        )
        partitions = partitions if isinstance(partitions, list) else []
    for part in partitions:
        if not isinstance(part, dict):
            failures.append(f"partition entry is {type(part).__name__}, expected an object")
            continue
        label = f"partition batch_size={part.get('batch_size')}"
        if not part.get("exact"):
            failures.append(f"{label} is not exact")
        if int(part.get("compared_records") or 0) != EXPECTED_VALIDATION_RECORDS:
            failures.append(
                f"{label} compared {part.get('compared_records')} records, expected "
                f"{EXPECTED_VALIDATION_RECORDS}"
            )
        for key in ("missing_indices", "duplicated_indices"):
            if part.get(key):
                failures.append(f"{label} has {key}: {part.get(key)}")
        if int(part.get("mismatched_field_count") or 0):
            failures.append(
                f"{label} has {part.get('mismatched_field_count')} mismatched field(s)"
            )

    identity = report.get("identity")
    validation_implementation = None
    if isinstance(identity, dict):
        if identity.get("experiment_id") != experiment:
            failures.append(
                f"report is for experiment {identity.get('experiment_id')!r}, "
                f"expected {experiment!r}"
            )
        validation_implementation = identity.get(
            "effective_noise_implementation"
        ) or identity.get("noise_implementation")
        if validation_implementation not in JAX_BACKED_VALIDATION:
            failures.append(
                f"frozen validation ran implementation {validation_implementation!r}; "
                f"it must be one of {JAX_BACKED_VALIDATION} so the frozen "
                "noisy-validation realization is unchanged"
            )
        if identity.get("loader") not in (None, "validation"):
            failures.append(
                f"equivalence was run against loader {identity.get('loader')!r}, "
                "expected 'validation'"
            )
    else:
        failures.append("report has no identity block")

    return {
        "ok": not failures,
        "failures": failures,
        "records": records,
        "partitions_checked": len(partitions),
        "validation_noise_implementation": validation_implementation,
    }


def feasibility_band(pairs_per_second: float | None) -> str:
    """Classify throughput for scheduling. Not a scientific acceptance limit."""
    if not _finite_positive(pairs_per_second):
        return "unknown"
    rate = float(pairs_per_second)
    if rate >= ENCOURAGING_PAIRS_PER_SECOND:
        return "encouraging"
    if rate >= MARGINAL_PAIRS_PER_SECOND:
        return "marginal"
    return "inadequate_for_single_72h_allocation_without_further_evidence"


def _speedup(numerator: float | None, denominator: float | None) -> float | None:
    if not (_finite_positive(numerator) and _finite_positive(denominator)):
        return None
    return float(numerator) / float(denominator)


def qualify(
    report_dir: Path,
    *,
    experiments: tuple[str, ...] = EXPERIMENTS,
    expected_commit: str | None = None,
    require_numpy: bool = True,
) -> dict:
    """Reduce the qualification reports in ``report_dir`` to one decision."""
    conditions = [
        (condition, implementation)
        for condition, implementation in THROUGHPUT_CONDITIONS
        if require_numpy or condition != "numpy"
    ]
    per_experiment: dict[str, dict] = {}
    failures: list[str] = []
    remeasure: list[str] = []

    for experiment in experiments:
        entry: dict = {"throughput": {}}

        try:
            entry["identity"] = check_identity(
                load_report(report_dir / f"{experiment}-identity.json"),
                experiment=experiment,
                expected_commit=expected_commit,
            )
        except ReportProblem as exc:
            entry["identity"] = {"ok": False, "failures": [str(exc)]}
        failures.extend(f"{experiment} identity: {m}" for m in entry["identity"]["failures"])

        for condition, implementation in conditions:
            try:
                outcome = check_throughput(
                    load_report(report_dir / f"{experiment}-{condition}.json"),
                    experiment=experiment,
                    condition=condition,
                    expected_implementation=implementation,
                    expected_commit=expected_commit,
                )
            except ReportProblem as exc:
                outcome = {"ok": False, "failures": [str(exc)], "remeasure": [],
                           "condition": condition, "implementation": implementation,
                           "sustained_pairs_per_second": None,
                           "sustained_measurement_valid": False}
            entry["throughput"][condition] = outcome
            failures.extend(f"{experiment} {condition}: {m}" for m in outcome["failures"])
            remeasure.extend(f"{experiment} {condition}: {m}" for m in outcome.get("remeasure", []))

        try:
            entry["frozen_validation"] = check_frozen_validation(
                load_report(report_dir / f"{experiment}-validation-exact.json"),
                experiment=experiment,
            )
        except ReportProblem as exc:
            entry["frozen_validation"] = {"ok": False, "failures": [str(exc)]}
        failures.extend(
            f"{experiment} frozen validation: {m}"
            for m in entry["frozen_validation"]["failures"]
        )

        rates = {
            condition: entry["throughput"][condition].get("sustained_pairs_per_second")
            for condition, _ in conditions
        }
        entry["comparison"] = {
            "pairs_per_second": rates,
            "batched_speedup_over_reference": _speedup(rates.get("batched"), rates.get("reference")),
            "numpy_speedup_over_reference": _speedup(rates.get("numpy"), rates.get("reference")),
            "numpy_speedup_over_batched": _speedup(rates.get("numpy"), rates.get("batched")),
            "clean_baseline_pairs_per_second": rates.get("clean"),
            # How close the best noisy strategy gets to no-noise loading. Near
            # 1.0 means noise generation is no longer the bottleneck and
            # further sampler work would not help.
            "best_noisy_fraction_of_clean": _speedup(
                max((r for c, r in rates.items() if c != "clean" and _finite_positive(r)),
                    default=None),
                rates.get("clean"),
            ),
        }
        entry["feasibility"] = {
            condition: {
                "pairs_per_second": rates.get(condition),
                "band": feasibility_band(rates.get(condition)),
                "projected_loading_hours_600_epochs":
                    entry["throughput"][condition].get("projected_loading_hours_600_epochs"),
                "measurement_valid":
                    entry["throughput"][condition].get("sustained_measurement_valid"),
            }
            for condition, implementation in conditions
            if implementation is not None
        }
        per_experiment[experiment] = entry

    # Scientific failure dominates: a REMEASURE verdict would imply the only
    # problem is measurement duration, which is not true if a check failed.
    status = FAIL if failures else (REMEASURE if remeasure else PASS)
    return {
        "status": status,
        "report_dir": str(report_dir),
        "experiments": list(experiments),
        "expected_source_commit": expected_commit,
        "scientific_failures": failures,
        "remeasurement_required": remeasure,
        "per_experiment": per_experiment,
        "recommended_action": _recommended_action(status, per_experiment, remeasure),
        "caveat": (
            "CPU data-loading capacity only. GPU step time, validation, "
            "checkpointing, and repeated per-epoch worker startup are "
            "additional; a loading projection alone does not certify a 72-hour "
            "campaign."
        ),
    }


def _recommended_action(status: str, per_experiment: dict, remeasure: list[str]) -> str:
    if status == FAIL:
        return (
            "Do not submit. Resolve the listed scientific failures and rerun the "
            "affected qualification steps."
        )
    if status == REMEASURE:
        return (
            f"Rerun {len(remeasure)} condition(s) with the larger --batches value named "
            "in remeasurement_required, then rerun this gate. No scientific check failed."
        )
    valid = {}
    for experiment, entry in per_experiment.items():
        for condition, info in entry.get("feasibility", {}).items():
            if info.get("measurement_valid") and _finite_positive(info.get("pairs_per_second")):
                valid.setdefault(condition, []).append(float(info["pairs_per_second"]))
    if not valid:
        return "Qualification passed, but no valid throughput measurement is available."
    # Rank by the worst experiment for each strategy: the campaign must fit E02
    # as well as E01, so the minimum is the number that matters.
    ranked = sorted(valid.items(), key=lambda item: min(item[1]), reverse=True)
    best, rates = ranked[0]
    band = feasibility_band(min(rates))
    if band == "encouraging":
        return (
            f"Qualification passed. Fastest strategy clearing both experiments is "
            f"{best!r} at {min(rates):.2f} pairs/s (worst experiment). Preview the "
            "six-run matrix, then submit ONLY S12-E01-R001 and measure a real epoch."
        )
    return (
        f"Qualification passed scientifically, but the best strategy {best!r} reaches "
        f"only {min(rates):.2f} pairs/s on its worst experiment ({band}). Do not claim "
        "the 72-hour campaign is feasible from this alone; submit one E01 run and "
        "measure a real epoch, or plan checkpoint/resume across allocations."
    )


def format_summary(payload: dict) -> str:
    """Render the decision for a terminal."""
    lines: list[str] = []
    lines.append(f"S12 CPU qualification: {payload['status']}")
    lines.append(f"reports: {payload['report_dir']}")
    if payload.get("expected_source_commit"):
        lines.append(f"expected source commit: {payload['expected_source_commit']}")
    lines.append("")

    for experiment, entry in payload["per_experiment"].items():
        identity = entry.get("identity", {})
        lines.append(
            f"{experiment}  artifacts={'ok' if identity.get('ok') else 'FAILED'}"
            f"  frozen_validation="
            f"{'ok' if entry.get('frozen_validation', {}).get('ok') else 'FAILED'}"
            f" ({entry.get('frozen_validation', {}).get('records', 0)} records,"
            f" {entry.get('frozen_validation', {}).get('validation_noise_implementation')})"
        )
        lines.append(
            f"  {'condition':12} {'pairs/s':>12} {'valid':>6} {'hours/600ep':>12} "
            f"{'band':>12}  rerun"
        )
        for condition, outcome in entry["throughput"].items():
            rate = outcome.get("sustained_pairs_per_second")
            hours = outcome.get("projected_loading_hours_600_epochs")
            band = (
                feasibility_band(rate)
                if outcome.get("implementation") is not None
                else "-"
            )
            rate_text = f"{float(rate):12.2f}" if _finite_positive(rate) else f"{'n/a':>12}"
            hours_text = f"{float(hours):12.1f}" if _finite_positive(hours) else f"{'n/a':>12}"
            rerun = outcome.get("recommended_batches_for_valid_measurement")
            lines.append(
                f"  {condition:12} {rate_text} "
                f"{str(bool(outcome.get('sustained_measurement_valid'))):>6} "
                f"{hours_text} {band[:12]:>12}  {rerun or '-'}"
            )
        comparison = entry.get("comparison", {})
        ratios = (
            ("batched/reference", comparison.get("batched_speedup_over_reference")),
            ("numpy/reference", comparison.get("numpy_speedup_over_reference")),
            ("numpy/batched", comparison.get("numpy_speedup_over_batched")),
            ("best/clean", comparison.get("best_noisy_fraction_of_clean")),
        )
        lines.append("  ratios: " + "  ".join(
            f"{label}={value:.2f}x" if _finite_positive(value) else f"{label}=n/a"
            for label, value in ratios
        ))
        lines.append("")

    if payload["scientific_failures"]:
        lines.append(f"SCIENTIFIC FAILURES ({len(payload['scientific_failures'])}):")
        lines.extend(f"  - {item}" for item in payload["scientific_failures"])
        lines.append("")
    if payload["remeasurement_required"]:
        lines.append(f"REMEASUREMENT REQUIRED ({len(payload['remeasurement_required'])}):")
        lines.extend(f"  - {item}" for item in payload["remeasurement_required"])
        lines.append("")
    lines.append(f"action: {payload['recommended_action']}")
    lines.append(f"caveat: {payload['caveat']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Directory holding the qualification JSON reports.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        dest="experiments",
        help="Experiment to qualify. Repeatable. Defaults to S12-E01 and S12-E02.",
    )
    parser.add_argument(
        "--expected-source-commit",
        default=None,
        help="Reviewed full commit SHA. When a report records its source "
        "revision, a mismatch is a failure rather than a silent pass on stale "
        "evidence.",
    )
    parser.add_argument(
        "--no-numpy",
        action="store_true",
        default=False,
        help="Do not require a NumPy throughput report. Use only when "
        "qualifying the JAX strategies alone.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the machine-readable summary to this path.",
    )
    args = parser.parse_args(argv)

    payload = qualify(
        args.report_dir,
        experiments=tuple(args.experiments) or EXPERIMENTS,
        expected_commit=args.expected_source_commit,
        require_numpy=not args.no_numpy,
    )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")
    print(format_summary(payload))
    return EXIT_STATUS[payload["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
