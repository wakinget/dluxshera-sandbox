#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dluxshera.datasets.schema import json_ready
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit or print a site-aware ML study job.")
    parser.add_argument("--site", required=True, choices=("gattaca2", "tacc_ls6", "ls6", "lonestar6"))
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--conda-sh", type=Path, default=None)
    parser.add_argument("--conda-env", default=None)
    parser.add_argument("--conda-prefix", type=Path, default=None)
    parser.add_argument("--prepared-root", type=Path, default=None)
    parser.add_argument("--split-registry", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
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
    run_id = args.run_id or f"{args.experiment_id}-R001"
    repo_root = Path.cwd().resolve() if args.repo_root is None else args.repo_root.expanduser().resolve()
    script = _resolve_repo_path(
        args.script or _default_script(profile.name),
        repo_root=repo_root,
    )
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
        "ML_VALIDATION_MANIFEST": _path_arg(args.validation_manifest),
        "ML_TEST_MANIFEST": _path_arg(args.test_manifest),
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
