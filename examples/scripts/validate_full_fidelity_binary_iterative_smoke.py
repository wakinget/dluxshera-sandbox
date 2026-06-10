"""Validate the full-fidelity binary iterative smoke wrapper.

This is a staged smoke validator around the existing thin wrapper. It does not
expand the physics scope; it checks translation, dry-run artifacts, tiny
execution artifacts when requested, and aggregate-only reuse of stored plans.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "dluxshera-matplotlib"))
WRAPPER_PATH = SCRIPTS_ROOT / "run_full_fidelity_binary_iterative_campaign.py"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)
DEFAULT_RESULTS_ROOT = REPO_ROOT / "Results" / "full_fidelity_validation"
REQUIRED_DRY_RUN_ARTIFACTS = (
    "campaign_plan.json",
    "resolved_config.json",
    "model_split_summary.json",
    "truth_realization.json",
    "truth_realization_by_label.csv",
    "subblock_plan.csv",
    "iterative_plan.csv",
    "template_hashes.csv",
    "expected_outputs.csv",
    "bias_cases.csv",
    "prior_draws.csv",
)
REQUIRED_TEMPLATES = (
    "trace_template.json",
    "render_template.json",
    "inference_template.json",
)
TRANSLATED_REQUIRED_EXPERIMENT_KEYS = (
    "system",
    "spectral_model",
    "high_order_wfe",
    "subblocks",
    "iterative",
    "observation_theta",
    "full_fidelity_smoke_contract",
)
SUMMARY_MATRIX_DEFAULT = "subblock_summary_matrices.npz"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_wrapper_module() -> Any:
    scripts_dir = str(SCRIPTS_ROOT)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_full_fidelity_binary_iterative_campaign",
        WRAPPER_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import wrapper from {WRAPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _git_commit_or_unknown() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _new_stage(name: str) -> dict[str, Any]:
    return {"status": "pending", "checks": [], "failures": []}


def _check(stage: dict[str, Any], name: str, ok: bool, detail: Any = None) -> bool:
    row = {"name": name, "status": "pass" if ok else "fail"}
    if detail is not None:
        row["detail"] = detail
    stage.setdefault("checks", []).append(row)
    if not ok:
        stage.setdefault("failures", []).append(row)
    return ok


def _finish_stage(stage: dict[str, Any]) -> dict[str, Any]:
    stage["status"] = "failed" if stage.get("failures") else "passed"
    return stage


def _run_wrapper(
    *,
    config_path: Path,
    results_root: Path,
    run_name: str,
    dry_run: bool = False,
    aggregate_only: bool = False,
    max_workers: int = 1,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(WRAPPER_PATH),
        "--config",
        str(config_path),
        "--results-root",
        str(results_root),
        "--run-name",
        run_name,
        "--max-workers",
        str(max_workers),
        "--quiet",
        "--no-resource-time",
    ]
    if dry_run:
        command.append("--dry-run")
    if aggregate_only:
        command.append("--aggregate-only")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _tail(path: Path, n_lines: int = 30) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def _classify_failure(text: str, *, default: str = "unknown_failure") -> str:
    lower = text.lower()
    patterns = (
        ("missing_trace_source", ("airbus", "trace source", "frame_truth", "trajectory")),
        ("missing_template", ("template", "no such file")),
        ("render_failure", ("render", "cube", "image")),
        ("inference_failure", ("optimizer", "inference", "starting guess")),
        ("schur_summary_failure", ("schur", "hessian", "matrix_artifact")),
        ("iterative_update_failure", ("iterative", "posterior", "reference update")),
        ("aggregate_failure", ("aggregate", "stored plan validation", "missing output")),
        ("config_parse_failure", ("yaml", "config", "parse")),
    )
    for label, needles in patterns:
        if any(needle in lower for needle in needles):
            return label
    return default


def _template_system_hash(path: Path) -> str:
    from dluxshera.utils.campaign_model_split import hash_campaign_model_config

    payload = _read_json(path)
    system = payload.get("system", {}) if isinstance(payload, Mapping) else {}
    return hash_campaign_model_config(system)


def run_static_validation(*, config_path: Path) -> dict[str, Any]:
    stage = _new_stage("static")
    try:
        module = _load_wrapper_module()
        _check(stage, "wrapper_imports_cleanly", True)
    except Exception as exc:
        _check(stage, "wrapper_imports_cleanly", False, repr(exc))
        return _finish_stage(stage)

    completed = subprocess.run(
        [sys.executable, str(WRAPPER_PATH), "--help"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _check(
        stage,
        "wrapper_help_works",
        completed.returncode == 0 and "--config" in completed.stdout,
        {"returncode": completed.returncode, "stderr": completed.stderr[-500:]},
    )

    try:
        raw = module.load_config_file(config_path)
        experiment = raw.get("experiment", raw)
        _check(stage, "config_file_loads", True)
        _check(
            stage,
            "config_kind_is_full_fidelity_smoke",
            str(experiment.get("kind")) == "full_fidelity_binary_iterative_smoke",
            experiment.get("kind"),
        )
        translated = module._full_fidelity_to_observation_bias(raw, run_name="static_validation")
        translated_exp = translated["experiment"]
        _check(
            stage,
            "translation_kind_is_observation_bias",
            translated_exp.get("kind") == "observation_bias_campaign",
            translated_exp.get("kind"),
        )
        _check(
            stage,
            "source_campaign_kind_preserved",
            translated_exp.get("source_campaign_kind") == "full_fidelity_binary_iterative_smoke",
            translated_exp.get("source_campaign_kind"),
        )
        _check(
            stage,
            "translated_source_kind_present",
            bool(translated_exp.get("system", {}).get("source", {}).get("kind")),
            translated_exp.get("system", {}).get("source", {}),
        )
        _check(
            stage,
            "translated_source_target_present",
            bool(translated_exp.get("system", {}).get("source", {}).get("target")),
            translated_exp.get("system", {}).get("source", {}),
        )
        for key in TRANSLATED_REQUIRED_EXPERIMENT_KEYS:
            _check(stage, f"translated_contains_{key}", key in translated_exp)
        bad = {"experiment": {"kind": "unsupported"}}
        try:
            module._full_fidelity_to_observation_bias(bad, run_name=None)
        except ValueError:
            _check(stage, "translation_rejects_unsupported_kind", True)
        else:
            _check(stage, "translation_rejects_unsupported_kind", False)
    except Exception as exc:
        _check(stage, "static_translation_checks_complete", False, repr(exc))
    return _finish_stage(stage)


def validate_dry_run_artifacts(
    *,
    run_root: Path,
    config_path: Path,
    expected_run_name: str,
) -> dict[str, Any]:
    stage = _new_stage("dry_run")
    stage["run_root"] = str(run_root)
    stage["required_artifacts"] = {}
    for relative in REQUIRED_DRY_RUN_ARTIFACTS:
        exists = (run_root / relative).is_file()
        stage["required_artifacts"][relative] = exists
        _check(stage, f"artifact_exists:{relative}", exists)
    templates_dir = run_root / "templates"
    _check(stage, "templates_directory_exists", templates_dir.is_dir(), str(templates_dir))
    for filename in REQUIRED_TEMPLATES:
        _check(stage, f"template_exists:{filename}", (templates_dir / filename).is_file())

    if stage["failures"]:
        return _finish_stage(stage)

    config = _read_json(run_root / "resolved_config.json")
    experiment = config.get("experiment", {})
    source = experiment.get("system", {}).get("source", {})
    _check(stage, "resolved_config_is_observation_bias", experiment.get("kind") == "observation_bias_campaign")
    _check(stage, "resolved_config_preserves_source_campaign_kind", experiment.get("source_campaign_kind") == "full_fidelity_binary_iterative_smoke")
    _check(stage, "resolved_source_kind_present", bool(source.get("kind")))
    _check(stage, "resolved_source_target_present", bool(source.get("target")))

    split = _read_json(run_root / "model_split_summary.json")
    truth_hash = str(split.get("truth_config_hash", ""))
    inference_hash = str(split.get("inference_config_hash", ""))
    components = split.get("components", {})
    stage["model_split_checks"] = {
        "truth_config_hash": truth_hash,
        "inference_config_hash": inference_hash,
        "components": components,
    }
    _check(stage, "model_split_truth_hash_present", bool(truth_hash))
    _check(stage, "model_split_inference_hash_present", bool(inference_hash))
    any_mismatch = any(
        isinstance(value, Mapping) and value.get("enabled") and value.get("matched") is False
        for value in components.values()
    )
    _check(
        stage,
        "model_split_hash_relation_auditable",
        truth_hash != inference_hash or not any_mismatch,
        {"truth_hash": truth_hash, "inference_hash": inference_hash, "any_mismatch": any_mismatch},
    )
    spectral = components.get("spectral_model", {}) if isinstance(components, Mapping) else {}
    high_order = components.get("high_order_wfe", {}) if isinstance(components, Mapping) else {}
    _check(stage, "spectral_provenance_exists_when_enabled", not spectral.get("enabled") or bool(spectral.get("artifact_root")))
    _check(stage, "high_order_wfe_provenance_exists_when_enabled", not high_order.get("enabled") or bool(high_order.get("artifact_root")))

    template_rows = _read_csv(run_root / "template_hashes.csv")
    _check(stage, "template_hashes_present", bool(template_rows))
    if template_rows:
        row = template_rows[0]
        for key in ("trace_template_hash", "render_template_hash", "inference_template_hash"):
            _check(stage, f"{key}_present", bool(row.get(key)))
        _check(stage, "render_template_tied_to_truth_system", row.get("truth_system_hash") == _template_system_hash(run_root / "templates" / "render_template.json"))
        _check(stage, "inference_template_tied_to_inference_system", row.get("inference_system_hash") == _template_system_hash(run_root / "templates" / "inference_template.json"))

    subblock_rows = _read_csv(run_root / "subblock_plan.csv")
    iterative_rows = _read_csv(run_root / "iterative_plan.csv")
    expected_rows = _read_csv(run_root / "expected_outputs.csv")
    stage["subblock_plan_checks"] = {"rows": len(subblock_rows)}
    stage["iterative_plan_checks"] = {"rows": len(iterative_rows)}
    n_cases = int(experiment.get("prior_draws", {}).get("n_cases", experiment.get("n_cases", 1)))
    iterative_cfg = experiment.get("iterative", {})
    subblocks_per_window = int(iterative_cfg.get("subblocks_per_window", 1))
    windows_per_draw = int(iterative_cfg.get("windows_per_draw", experiment.get("subblocks", {}).get("n_subblocks", 0)))
    expected_subblocks = n_cases * windows_per_draw * subblocks_per_window
    _check(stage, "expected_number_of_subblocks", len(subblock_rows) == expected_subblocks, {"expected": expected_subblocks, "actual": len(subblock_rows)})
    _check(stage, "expected_number_of_iterative_rows", len(iterative_rows) == expected_subblocks, {"expected": expected_subblocks, "actual": len(iterative_rows)})
    _check(stage, "expected_number_of_output_rows", len(expected_rows) == expected_subblocks, {"expected": expected_subblocks, "actual": len(expected_rows)})
    _check(stage, "summary_paths_unique", len({row.get("summary_path", "") for row in subblock_rows}) == len(subblock_rows))

    for index, row in enumerate(subblock_rows):
        prefix = f"subblock_row_{index}"
        command = row.get("command", "")
        for path_key in ("trace_template_path", "render_template_path", "inference_template_path", "frame_truth_path", "starting_guess_prediction_path"):
            value = row.get(path_key, "")
            _check(stage, f"{prefix}_{path_key}_exists", bool(value) and Path(value).exists(), value)
        if str(row.get("smear_enabled", "")).lower() == "true":
            for path_key in ("smear_truth_csv", "smear_model_csv", "smear_provenance_json"):
                value = row.get(path_key, "")
                _check(stage, f"{prefix}_{path_key}_exists", bool(value) and Path(value).exists(), value)
        for flag in (
            "--external-frame-truth-csv",
            "--starting-guess-csv",
            "--starting-guess-mode starting_guess_csv",
            "--summary-information-scale summed_likelihood",
        ):
            _check(stage, f"{prefix}_command_contains:{flag}", flag in command)

    _check(stage, "iterative_enabled", str(iterative_cfg.get("enabled")).lower() == "true" or iterative_cfg.get("enabled") is True)
    _check(stage, "windows_per_draw_matches_config", all(int(row.get("window_index", -1)) < windows_per_draw for row in iterative_rows))
    _check(stage, "subblocks_per_window_matches_config", all(int(row.get("window_subblock_index", -1)) < subblocks_per_window for row in iterative_rows))
    _check(stage, "update_mode_matches_config", all(row.get("update_mode") == str(iterative_cfg.get("update_mode", "physical_full")) for row in iterative_rows))
    expected_output_paths = {
        row.get(key, "")
        for row in expected_rows
        for key in ("summary_path", "case_posterior_path", "window_summary_path", "iterative_reference_update_path", "window_diagnostic_path", "realized_command_path")
        if row.get(key)
    }
    for index, row in enumerate(iterative_rows):
        for key in ("summary_path", "case_posterior_path", "window_summary_path", "iterative_reference_update_path", "window_diagnostic_path", "realized_command_path"):
            value = row.get(key, "")
            _check(stage, f"iterative_row_{index}_{key}_expected", bool(value) and value in expected_output_paths, value)

    return _finish_stage(stage)


def run_dry_run_validation(*, config_path: Path, results_root: Path, run_name: str) -> dict[str, Any]:
    completed = _run_wrapper(
        config_path=config_path,
        results_root=results_root,
        run_name=run_name,
        dry_run=True,
    )
    run_root = results_root.resolve() / run_name
    stage = validate_dry_run_artifacts(
        run_root=run_root,
        config_path=config_path,
        expected_run_name=run_name,
    )
    stage["command_returncode"] = completed.returncode
    stage["stdout_tail"] = completed.stdout[-2000:]
    stage["stderr_tail"] = completed.stderr[-4000:]
    _check(stage, "dry_run_command_succeeded", completed.returncode == 0, {"stderr_tail": completed.stderr[-2000:]})
    return _finish_stage(stage)


def _summary_matrix_path(summary_path: Path) -> Path:
    if not summary_path.exists():
        return summary_path.with_name(SUMMARY_MATRIX_DEFAULT)
    try:
        payload = _read_json(summary_path)
    except Exception:
        return summary_path.with_name(SUMMARY_MATRIX_DEFAULT)
    raw = payload.get("matrix_artifact_path") if isinstance(payload, Mapping) else None
    if raw:
        candidate = Path(str(raw))
        return candidate if candidate.is_absolute() else summary_path.parent / candidate
    return summary_path.with_name(SUMMARY_MATRIX_DEFAULT)


def validate_tiny_execution_artifacts(*, run_root: Path, run_returncode: int, stderr: str) -> dict[str, Any]:
    stage = _new_stage("tiny_exec")
    stage["run_root"] = str(run_root)
    stage["executed"] = True
    stage["command_returncode"] = run_returncode
    stage["stderr_tail"] = stderr[-4000:]

    status_rows = _read_csv(run_root / "subblock_status_iterative.csv")
    if not status_rows:
        status_rows = _read_csv(run_root / "subblock_status.csv")
    counts: dict[str, int] = {}
    for row in status_rows:
        counts[row.get("status", "unknown")] = counts.get(row.get("status", "unknown"), 0) + 1
    failed = [row for row in status_rows if row.get("status") not in ("ok", "success")]
    stage["child_status_counts"] = counts
    stage["failed_children"] = failed
    _check(stage, "subprocess_status_rows_exist", bool(status_rows))
    _check(stage, "no_silent_subprocess_failure", run_returncode == 0 or bool(failed), {"returncode": run_returncode})

    expected_rows = _read_csv(run_root / "expected_outputs.csv")
    missing_outputs: list[dict[str, str]] = []
    for row in expected_rows:
        summary_path = Path(row.get("summary_path", ""))
        if not summary_path.exists():
            missing_outputs.append({"kind": "summary", "path": str(summary_path)})
            continue
        matrix_path = _summary_matrix_path(summary_path)
        if not matrix_path.exists():
            missing_outputs.append({"kind": "matrix_npz", "path": str(matrix_path)})
        for key in ("case_posterior_path", "window_summary_path", "iterative_reference_update_path", "window_diagnostic_path", "realized_command_path"):
            value = row.get(key, "")
            if value and not Path(value).exists():
                missing_outputs.append({"kind": key, "path": value})
        case_root = summary_path.parent.parent
        for log_name in ("subprocess.stdout.log", "subprocess.stderr.log"):
            if not (case_root / log_name).exists():
                missing_outputs.append({"kind": log_name, "path": str(case_root / log_name)})
    stage["missing_outputs"] = missing_outputs
    _check(stage, "expected_summary_and_iterative_outputs_exist", not missing_outputs, missing_outputs[:20])

    failed_children: list[dict[str, Any]] = []
    for row in failed:
        stderr_log = Path(row.get("stderr_log", ""))
        tail = _tail(stderr_log)
        failed_children.append(
            {
                "summary_path": row.get("summary_path"),
                "return_code": row.get("return_code"),
                "failure_class": row.get("failure_class") or _classify_failure(tail),
                "stderr_tail": tail,
            }
        )
    if run_returncode != 0 and not failed_children:
        failed_children.append(
            {
                "summary_path": "",
                "return_code": run_returncode,
                "failure_class": _classify_failure(stderr),
                "stderr_tail": stderr[-4000:],
            }
        )
    stage["failed_children"] = failed_children
    if run_returncode != 0:
        _check(stage, "tiny_execution_completed", False, failed_children[:3])
    else:
        _check(stage, "tiny_execution_completed", True)
    return _finish_stage(stage)


def run_tiny_exec_validation(*, config_path: Path, results_root: Path, run_name: str, max_workers: int) -> dict[str, Any]:
    completed = _run_wrapper(
        config_path=config_path,
        results_root=results_root,
        run_name=run_name,
        max_workers=max_workers,
    )
    return validate_tiny_execution_artifacts(
        run_root=results_root.resolve() / run_name,
        run_returncode=completed.returncode,
        stderr=completed.stderr,
    )


def run_aggregate_only_validation(*, config_path: Path, results_root: Path, run_name: str) -> dict[str, Any]:
    run_root = results_root.resolve() / run_name
    stage = _new_stage("aggregate_only")
    before_hashes = {}
    for path in (run_root / "templates").glob("*.json"):
        before_hashes[str(path)] = _sha256(path)
    for path in (run_root / "model_split").glob("**/*"):
        if path.is_file():
            before_hashes[str(path)] = _sha256(path)

    completed = _run_wrapper(
        config_path=config_path,
        results_root=results_root,
        run_name=run_name,
        aggregate_only=True,
    )
    validation_json = run_root / "analysis" / "aggregate_only_plan_validation.json"
    stage["validation_json"] = str(validation_json)
    stage["command_returncode"] = completed.returncode
    stage["stderr_tail"] = completed.stderr[-4000:]
    _check(stage, "aggregate_only_command_succeeded", completed.returncode == 0, completed.stderr[-2000:])
    _check(stage, "aggregate_only_validation_json_exists", validation_json.exists(), str(validation_json))
    if validation_json.exists():
        payload = _read_json(validation_json)
        stage["mismatches"] = payload.get("mismatches", [])
        _check(stage, "aggregate_only_plan_validation_ok", payload.get("status") == "ok", payload)
    else:
        stage["mismatches"] = []
    after_hashes = {}
    for path in (run_root / "templates").glob("*.json"):
        after_hashes[str(path)] = _sha256(path)
    for path in (run_root / "model_split").glob("**/*"):
        if path.is_file():
            after_hashes[str(path)] = _sha256(path)
    _check(stage, "aggregate_only_did_not_rewrite_templates_or_model_split", before_hashes == after_hashes)
    _check(stage, "aggregate_status_exists", (run_root / "analysis" / "aggregate_status.json").exists())
    return _finish_stage(stage)


def _recommendation(stages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    for name, stage in stages.items():
        if stage.get("status") == "failed":
            blockers.append(f"{name} failed")
    tiny = stages.get("tiny_exec", {})
    aggregate = stages.get("aggregate_only", {})
    ready_for_next_smoke = not blockers
    ready_for_hpc = (
        stages.get("static", {}).get("status") == "passed"
        and stages.get("dry_run", {}).get("status") == "passed"
        and tiny.get("status") == "passed"
        and aggregate.get("status") == "passed"
    )
    if tiny and tiny.get("status") != "passed":
        warnings.append("Tiny execution did not pass; inspect classified child failures before HPC preflight.")
    return {
        "ready_for_next_smoke": bool(ready_for_next_smoke),
        "ready_for_hpc": bool(ready_for_hpc),
        "blockers": blockers,
        "warnings": warnings,
    }


def _write_markdown_report(path: Path, report: Mapping[str, Any]) -> None:
    stages = report.get("stages", {})
    rec = report.get("recommendation", {})
    lines = [
        "# Full-Fidelity Binary Iterative Smoke Validation",
        "",
        f"- Run name: `{report.get('run_name')}`",
        f"- Config: `{report.get('config_path')}`",
        f"- Results root: `{report.get('results_root')}`",
        f"- Git commit: `{report.get('git_commit_or_unknown')}`",
        "",
        "## Stage Status",
    ]
    for name, stage in stages.items():
        lines.append(f"- {name}: **{stage.get('status')}**")
    lines.extend(
        [
            "",
            "## Questions",
            f"- Did the wrapper parse and translate the config? {'yes' if stages.get('static', {}).get('status') == 'passed' else 'no'}",
            f"- Did the Data/Inference split remain auditable? {'yes' if stages.get('dry_run', {}).get('status') == 'passed' else 'no'}",
            f"- Did dry-run write complete plans? {'yes' if stages.get('dry_run', {}).get('status') == 'passed' else 'no'}",
            f"- Did a tiny execution run? {'yes' if stages.get('tiny_exec', {}).get('executed') else 'not requested'}",
            f"- Did aggregate-only reuse stored artifacts? {'yes' if stages.get('aggregate_only', {}).get('status') == 'passed' else 'not proven'}",
            "",
            "## Recommendation",
            f"- Ready for next tiny smoke: `{rec.get('ready_for_next_smoke')}`",
            f"- Ready for HPC preflight: `{rec.get('ready_for_hpc')}`",
        ]
    )
    blockers = rec.get("blockers", [])
    if blockers:
        lines.append("- Blockers: " + "; ".join(str(item) for item in blockers))
    warnings = rec.get("warnings", [])
    if warnings:
        lines.append("- Warnings: " + "; ".join(str(item) for item in warnings))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stage_sequence(stage: str) -> list[str]:
    if stage == "all":
        return ["static", "dry-run", "tiny-exec", "aggregate-only"]
    return [stage]


def run_validation(
    *,
    config_path: Path,
    results_root: Path,
    run_name: str,
    stage: str,
    max_workers: int,
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    report_run_name = run_name
    for item in _stage_sequence(stage):
        if item == "static":
            stages["static"] = run_static_validation(config_path=config_path)
        elif item == "dry-run":
            dry_run_name = f"{run_name}_dryrun" if stage == "all" else run_name
            stages["dry_run"] = run_dry_run_validation(
                config_path=config_path,
                results_root=results_root,
                run_name=dry_run_name,
            )
        elif item == "tiny-exec":
            stages["tiny_exec"] = run_tiny_exec_validation(
                config_path=config_path,
                results_root=results_root,
                run_name=run_name,
                max_workers=max_workers,
            )
        elif item == "aggregate-only":
            if stages.get("tiny_exec", {}).get("status") == "failed":
                skipped = _new_stage("aggregate_only")
                skipped["status"] = "skipped"
                skipped["failures"] = [{"name": "tiny_exec_failed", "status": "fail"}]
                stages["aggregate_only"] = skipped
            else:
                stages["aggregate_only"] = run_aggregate_only_validation(
                    config_path=config_path,
                    results_root=results_root,
                    run_name=run_name,
                )
        else:
            raise ValueError(f"Unsupported stage: {item}")

    report = {
        "schema_version": "full_fidelity_binary_iterative_smoke_validation.v1",
        "created_at": _now_iso(),
        "run_name": report_run_name,
        "results_root": str(results_root.resolve()),
        "config_path": str(config_path.resolve()),
        "git_commit_or_unknown": _git_commit_or_unknown(),
        "stages": stages,
        "recommendation": _recommendation(stages),
    }
    report_root = results_root.resolve() / report_run_name
    _write_json(report_root / "validation_report.json", report)
    _write_markdown_report(report_root / "validation_report.md", report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the full-fidelity binary iterative smoke wrapper.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-name", default="full_fidelity_binary_iterative_validation_v0")
    parser.add_argument(
        "--stage",
        choices=("static", "dry-run", "tiny-exec", "aggregate-only", "all"),
        default="dry-run",
    )
    parser.add_argument("--max-workers", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = run_validation(
        config_path=args.config,
        results_root=args.results_root,
        run_name=str(args.run_name),
        stage=str(args.stage),
        max_workers=int(args.max_workers),
    )
    failed = [
        name
        for name, stage in report.get("stages", {}).items()
        if stage.get("status") not in ("passed", "skipped")
    ]
    print(json.dumps({"status": "failed" if failed else "passed", "failed_stages": failed}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
