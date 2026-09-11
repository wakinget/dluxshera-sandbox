#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from dluxshera.datasets.schema import json_ready
from dluxshera.ml import expand_study_run_plan, load_study_prescription
from dluxshera.ml.hpc import (
    parse_sbatch_job_id,
    prepare_sbatch_submission,
    slurm_profile,
)


def _default_script(site: str) -> Path:
    key = site.lower().replace("-", "_")
    if key in {"ls6", "lonestar6"}:
        key = "tacc_ls6"
    return Path("work") / "experiments" / "ml" / "hpc" / "sites" / key / "train_ml.sbatch"


def _path_arg(path: Path | None) -> str | None:
    return None if path is None else str(path.expanduser().resolve())


def _resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _repo_path_arg(path: Path | None, *, repo_root: Path) -> str | None:
    return None if path is None else str(_resolve_repo_path(path, repo_root=repo_root))


def _parse_key_path(values: list[str], *, option: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in str(value):
            raise ValueError(f"{option} expects KEY=PATH, got {value!r}.")
        key, path = str(value).split("=", 1)
        if not key.strip() or not path.strip():
            raise ValueError(f"{option} expects non-empty KEY=PATH, got {value!r}.")
        out[key.strip()] = Path(path.strip())
    return out


def _audit_manifest_env(values: list[str]) -> str | None:
    mapping = {
        key: _path_arg(path)
        for key, path in _parse_key_path(values, option="--audit-manifest").items()
    }
    return None if not mapping else json.dumps(mapping, sort_keys=True)


def _repo_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit or print a site-aware ML study job.")
    parser.add_argument("--site", required=True, choices=("gattaca2", "tacc_ls6", "ls6", "lonestar6"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--plan-preview", action="store_true", default=False)
    parser.add_argument("--submit-plan", action="store_true", default=False)
    parser.add_argument("--launch-packet", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--conda-sh", type=Path, default=None)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--conda-prefix", type=Path, default=None)
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--split-registry", type=Path, default=None)
    parser.add_argument("--scaler", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--audit-manifest", action="append", default=[])
    parser.add_argument("--artifact-lock", type=Path, default=None)
    parser.add_argument("--eigenbasis-artifact", type=Path, default=None)
    parser.add_argument("--noisy-eval-artifact", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--persist-dir", type=Path, default=None)
    parser.add_argument("--persist-artifact-root", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--source-archive-id", default=None)
    parser.add_argument("--script", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--error", type=Path, default=None)
    parser.add_argument(
        "--extra-sbatch-arg",
        action="append",
        default=[],
        help="Additional scheduler option, for example a Gattaca2 GPU partition/GRES flag.",
    )
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args(argv)

    profile = slurm_profile(args.site)
    if args.plan_preview and args.submit_plan:
        raise ValueError("--plan-preview and --submit-plan are mutually exclusive.")
    if args.plan_preview or args.submit_plan:
        study = load_study_prescription(args.study)
        experiment_ids = args.experiment or ([args.experiment_id] if args.experiment_id else None)
        run_ids = args.run or ([args.run_id] if args.run_id else None)
        rows = expand_study_run_plan(
            study,
            experiment_ids=experiment_ids,
            run_ids=run_ids,
            output_root=args.run_dir,
        )
        repo_root = Path.cwd().resolve() if args.repo_root is None else args.repo_root.expanduser().resolve()
        script = _resolve_repo_path(
            args.script or _default_script(profile.name),
            repo_root=repo_root,
        )
        audit_env = _audit_manifest_env(args.audit_manifest)
        commands = []
        submissions = []
        for index, row in enumerate(rows):
            run_dir = Path(row.output_root) if row.output_root else None
            explicit_env = {
                "ML_SITE": profile.name,
                "ML_REPO_ROOT": str(repo_root),
                "ML_CONDA_SH": _path_arg(args.conda_sh),
                "ML_CONDA_ENV": args.conda_env,
                "ML_CONDA_PREFIX": _path_arg(args.conda_prefix),
                "ML_STUDY_PATH": _repo_path_arg(args.study, repo_root=repo_root),
                "ML_EXPERIMENT_ID": row.experiment_id,
                "ML_RUN_ID": row.run_id,
                "ML_PREPARED_ROOT": _path_arg(args.prepared_root),
                "ML_SPLIT_REGISTRY": _path_arg(args.split_registry),
                "ML_SCALER": _path_arg(args.scaler),
                "ML_VALIDATION_MANIFEST": _path_arg(args.validation_manifest),
                "ML_TEST_MANIFEST": _path_arg(args.test_manifest),
                "ML_AUDIT_MANIFESTS": audit_env,
                "ML_ARTIFACT_LOCK": _path_arg(args.artifact_lock),
                "ML_EIGENBASIS_ARTIFACT": _path_arg(args.eigenbasis_artifact),
                "ML_NOISY_EVAL_ARTIFACT": _path_arg(args.noisy_eval_artifact),
                "ML_RUN_DIR": _path_arg(run_dir),
                "ML_PERSIST_DIR": None if args.persist_dir is None else str((args.persist_dir / row.study_id / row.experiment_id / row.run_id).expanduser().resolve()),
                "ML_PERSIST_ARTIFACT_ROOT": _path_arg(args.persist_artifact_root),
                "ML_DEVICE": args.device,
                "ML_RESUME_CHECKPOINT": _path_arg(args.resume_checkpoint),
                "ML_OVERWRITE": "1" if args.overwrite else None,
                "ML_SOURCE_COMMIT": args.source_commit,
                "DLUXSHERA_SOURCE_COMMIT": args.source_commit,
                "ML_SOURCE_ARCHIVE_ID": args.source_archive_id,
                "DLUXSHERA_SOURCE_ARCHIVE_ID": args.source_archive_id,
            }
            submission = prepare_sbatch_submission(
                profile,
                script=script,
                job_name=row.run_id,
                submitted_env=explicit_env,
                log_root=args.log_root or repo_root / "work" / "experiments" / "ml" / "hpc" / "logs",
                extra_args=tuple(args.extra_sbatch_arg),
            )
            commands.append(
                {
                    "matrix_index": index,
                    "study_id": row.study_id,
                    "experiment_id": row.experiment_id,
                    "run_id": row.run_id,
                    "command": submission.command,
                    "missing_required_environment": list(submission.missing_required_environment),
                    "resolved_artifacts": {
                        "artifact_profile": row.artifact_profile,
                        "split_registry_artifact_id": row.split_registry_artifact_id,
                        "scaler_artifact_id": row.scaler_artifact_id,
                        "validation_artifact": row.validation_artifact,
                        "test_artifact": row.test_artifact,
                        "artifact_lock_id": row.artifact_lock_id,
                        "audit_artifacts": list(row.audit_artifacts),
                        "auxiliary_artifacts": dict(row.auxiliary_artifacts),
                        "shared_reference_runs": list(row.shared_reference_runs),
                    },
                }
            )
            if args.submit_plan:
                if submission.missing_required_environment:
                    missing = ", ".join(submission.missing_required_environment)
                    raise ValueError(
                        f"Missing required batch environment values for {row.run_id}: {missing}."
                    )
                for directory in submission.log_directories:
                    directory.mkdir(parents=True, exist_ok=True)
                started_at = dt.datetime.now(dt.timezone.utc).isoformat()
                entry = {
                    "study_id": row.study_id,
                    "experiment_id": row.experiment_id,
                    "run_id": row.run_id,
                    "sbatch_command": submission.command,
                    "timestamp": started_at,
                    "repository_sha": args.source_commit or _repo_sha(repo_root),
                    "resolved_artifacts": commands[-1]["resolved_artifacts"],
                    "submission_success": False,
                    "submission_failure": None,
                    "slurm_job_id": None,
                }
                try:
                    result = subprocess.run(
                        submission.command,
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=submission.environment,
                    )
                    entry["stdout"] = result.stdout
                    entry["stderr"] = result.stderr
                    entry["slurm_job_id"] = parse_sbatch_job_id(result.stdout)
                    entry["submission_success"] = True
                except Exception as exc:
                    entry["submission_failure"] = str(exc)
                submissions.append(entry)
        payload = {
            "site": profile.name,
            "study_id": study["study_id"],
            "run_count": len(rows),
            "commands": commands,
            "submission_performed": bool(args.submit_plan),
        }
        if args.submit_plan:
            payload["submissions"] = submissions
            payload["submitted_count"] = sum(1 for item in submissions if item["submission_success"])
            payload["failed_count"] = sum(1 for item in submissions if not item["submission_success"])
        launch_packet = args.launch_packet
        if args.submit_plan and launch_packet is None:
            launch_packet = repo_root / "work" / "experiments" / "ml" / "hpc" / "launches" / (
                f"{study['study_id']}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            )
        if launch_packet is not None:
            launch_packet.mkdir(parents=True, exist_ok=True)
            name = "launch_manifest.json" if args.submit_plan else "launch_preview.json"
            (launch_packet / name).write_text(
                json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            payload["launch_packet"] = str(launch_packet)
        print(json.dumps(json_ready(payload), indent=2, sort_keys=True))
        return 0
    if args.experiment_id is None:
        raise ValueError("--experiment-id is required unless --plan-preview is used.")
    run_id = args.run_id or f"{args.experiment_id}-R001"
    repo_root = Path.cwd().resolve() if args.repo_root is None else args.repo_root.expanduser().resolve()
    script = _resolve_repo_path(
        args.script or _default_script(profile.name),
        repo_root=repo_root,
    )
    audit_env = _audit_manifest_env(args.audit_manifest)
    log_root = (
        repo_root / "work" / "experiments" / "ml" / "hpc" / "logs"
        if args.log_root is None
        else args.log_root
    )
    explicit_env = {
        "ML_SITE": profile.name,
        "ML_REPO_ROOT": str(repo_root),
        "ML_CONDA_SH": _path_arg(args.conda_sh),
        "ML_CONDA_ENV": args.conda_env,
        "ML_CONDA_PREFIX": _path_arg(args.conda_prefix),
        "ML_STUDY_PATH": _repo_path_arg(args.study, repo_root=repo_root),
        "ML_EXPERIMENT_ID": args.experiment_id,
        "ML_RUN_ID": run_id,
        "ML_PREPARED_ROOT": _path_arg(args.prepared_root),
        "ML_SPLIT_REGISTRY": _path_arg(args.split_registry),
        "ML_SCALER": _path_arg(args.scaler),
        "ML_VALIDATION_MANIFEST": _path_arg(args.validation_manifest),
        "ML_TEST_MANIFEST": _path_arg(args.test_manifest),
        "ML_AUDIT_MANIFESTS": audit_env,
        "ML_ARTIFACT_LOCK": _path_arg(args.artifact_lock),
        "ML_EIGENBASIS_ARTIFACT": _path_arg(args.eigenbasis_artifact),
        "ML_NOISY_EVAL_ARTIFACT": _path_arg(args.noisy_eval_artifact),
        "ML_RUN_DIR": _path_arg(args.run_dir),
        "ML_PERSIST_DIR": _path_arg(args.persist_dir),
        "ML_PERSIST_ARTIFACT_ROOT": _path_arg(args.persist_artifact_root),
        "ML_DEVICE": args.device,
        "ML_RESUME_CHECKPOINT": _path_arg(args.resume_checkpoint),
        "ML_OVERWRITE": "1" if args.overwrite else None,
        "ML_SOURCE_COMMIT": args.source_commit,
        "DLUXSHERA_SOURCE_COMMIT": args.source_commit,
        "ML_SOURCE_ARCHIVE_ID": args.source_archive_id,
        "DLUXSHERA_SOURCE_ARCHIVE_ID": args.source_archive_id,
    }
    submission = prepare_sbatch_submission(
        profile,
        script=script,
        job_name=run_id,
        submitted_env=explicit_env,
        log_root=log_root,
        output=args.output,
        error=args.error,
        extra_args=tuple(args.extra_sbatch_arg),
    )
    exported_env = {
        key: submission.environment[key]
        for key in sorted(explicit_env)
        if key in submission.environment
    }
    summary = {
        "site": profile.name,
        "study_path": exported_env["ML_STUDY_PATH"],
        "experiment_id": exported_env["ML_EXPERIMENT_ID"],
        "run_id": exported_env["ML_RUN_ID"],
        "command": submission.command,
        "environment": exported_env,
        "output": str(submission.output),
        "error": str(submission.error),
        "log_directories": [str(path) for path in submission.log_directories],
        "missing_required_environment": list(submission.missing_required_environment),
    }
    if args.dry_run:
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        return 0
    if submission.missing_required_environment:
        missing = ", ".join(submission.missing_required_environment)
        raise ValueError(f"Missing required batch environment values: {missing}.")
    for directory in submission.log_directories:
        directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        submission.command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=submission.environment,
    )
    summary["stdout"] = result.stdout
    summary["stderr"] = result.stderr
    summary["job_id"] = parse_sbatch_job_id(result.stdout)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
