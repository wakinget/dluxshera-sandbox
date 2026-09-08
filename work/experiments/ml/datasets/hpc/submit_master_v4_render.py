#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from dluxshera.datasets.master_v4 import (
    FrozenMasterV4,
    build_slurm_array_plan,
)
from dluxshera.datasets.schema import json_ready
from dluxshera.ml.hpc import SlurmProfile, build_sbatch_command, parse_sbatch_job_id

DEFAULT_PLAN_ROOT = Path(
    "/scratch-jpl/shera_hpc/dmckeith/dLuxShera-ML/render_v4/stateplans/master_v4"
)
DEFAULT_OUTPUT_ROOT = Path("/projects/shera_hpc/data/ml_training/shera_ml_master_v4")
DEFAULT_SCRATCH_ROOT = Path(
    "/scratch-jpl/shera_hpc/dmckeith/dLuxShera-ML/render_v4"
)
DEFAULT_REPO_ROOT = Path("/home/dmckeith/dluxshera-sandbox")
DEFAULT_SCRIPT = Path("work/experiments/ml/datasets/hpc/render_master_v4.sbatch")


def _repo_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _resolve_repo_path(path: Path, *, repo_root: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or submit the frozen SHERA ML Master V4 render array."
    )
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT)
    parser.add_argument("--renders-per-task", type=int, required=True)
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="64G")
    parser.add_argument("--time", default="08:00:00")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--account", default="shera_hpc")
    parser.add_argument("--partition", default="compute")
    parser.add_argument("--conda-sh", type=Path, default=Path("/cm/shared/apps/miniforge/etc/profile.d/conda.sh"))
    parser.add_argument("--conda-env", default="dluxshera-py311")
    parser.add_argument("--expected-repo-sha", default=None)
    parser.add_argument("--renderer-repository-sha", default=None)
    parser.add_argument("--job-name", default="shera-v4-render")
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--write-manifest", action="store_true", default=False)
    parser.add_argument("--allow-nonproduction-contract", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--submit", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.submit and args.dry_run:
        raise ValueError("--submit and --dry-run are mutually exclusive.")
    if not args.submit and not args.dry_run:
        raise ValueError("Use --dry-run to inspect or --submit to invoke sbatch.")

    plan_root = args.plan_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    scratch_root = args.scratch_root.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    script = _resolve_repo_path(args.script, repo_root=repo_root)

    if not args.allow_nonproduction_contract:
        from dluxshera.datasets.master_v4 import PRODUCTION_IDENTITIES

        expected = PRODUCTION_IDENTITIES
    else:
        expected = None
    plan = FrozenMasterV4(plan_root, expected=expected, validate_plan_hashes=True)

    array = build_slurm_array_plan(
        render_count=plan.render_count,
        renders_per_task=args.renders_per_task,
        concurrency=args.concurrency,
        nuisance_count=plan.nuisance_count,
    )
    profile = SlurmProfile(
        name="gattaca2",
        account=args.account,
        partition=args.partition,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        time=args.time,
    )
    logs = scratch_root / "logs"
    output = logs / "%x-%A_%a.out"
    error = logs / "%x-%A_%a.err"
    command = build_sbatch_command(
        profile,
        script=script,
        job_name=args.job_name,
        array=array.array_expression,
        output=output,
        error=error,
        export="ALL",
    )
    source_sha = args.renderer_repository_sha or _repo_sha(repo_root)
    expected_sha = args.expected_repo_sha or source_sha
    environment = {
        "V4_REPO_ROOT": str(repo_root),
        "V4_PLAN_ROOT": str(plan_root),
        "V4_OUTPUT_ROOT": str(output_root),
        "V4_SCRATCH_ROOT": str(scratch_root),
        "V4_RENDERS_PER_TASK": str(args.renders_per_task),
        "V4_RENDER_COUNT": str(plan.render_count),
        "V4_CONDA_SH": str(args.conda_sh),
        "V4_CONDA_ENV": args.conda_env,
        "EXPECTED_REPO_SHA": "" if expected_sha is None else expected_sha,
        "V4_RENDERER_REPOSITORY_SHA": "" if source_sha is None else source_sha,
    }
    manifest_path = args.manifest_path
    if manifest_path is None:
        stem_sha = "unknown" if source_sha is None else source_sha[:12]
        manifest_path = scratch_root / "manifests" / f"render_master_v4_campaign_{stem_sha}.json"
    campaign_manifest = {
        "schema_version": "shera_v4_render_campaign_manifest/1",
        "cluster_site": "gattaca2-jpl",
        "frozen_dataset_identity": {
            "dataset_version": plan.manifest["dataset_version"],
            "master_scientific_content_hash": plan.manifest[
                "master_scientific_content_hash"
            ],
            "science_vector_space_id": plan.manifest["science_vector_space_id"],
            "nuisance_vector_space_id": plan.manifest["nuisance_vector_space_id"],
            "nuisance_bank_hash": plan.manifest["selected_nuisance_bank_hash"],
            "render_system_contract_hash": plan.manifest[
                "render_system_contract_hash"
            ],
            "render_contract_hash": plan.manifest["render_contract_hash"],
        },
        "renderer_repository_sha": source_sha,
        "expected_repo_sha": expected_sha,
        "dlux_expected_provenance": {
            "import": "dLux",
            "version": "0.14.0",
            "source_commit": "d0ab6df843503535c05225b15197233144be3cf1",
        },
        "plan_root": str(plan_root),
        "output_root": str(output_root),
        "scratch_root": str(scratch_root),
        "render_count": plan.render_count,
        "renders_per_task": array.renders_per_task,
        "array_task_count": array.task_count,
        "final_partial_task_size": array.final_task_size,
        "array_expression": array.array_expression,
        "concurrency": args.concurrency,
        "cpus_per_task": args.cpus_per_task,
        "mem": args.mem,
        "time": args.time,
        "account": args.account,
        "partition": args.partition,
        "sbatch_command": command,
        "environment": environment,
        "manifest_path": str(manifest_path),
        "alignment_warning": array.alignment_warning,
    }
    summary = {
        "total_renders": plan.render_count,
        "renders_per_task": array.renders_per_task,
        "number_of_array_tasks": array.task_count,
        "final_partial_task_size": array.final_task_size,
        "array_expression": array.array_expression,
        "requested_concurrency": args.concurrency,
        "exact_sbatch_command": command,
        "environment": environment,
        "campaign_manifest": campaign_manifest,
    }
    if array.alignment_warning:
        summary["warning"] = array.alignment_warning

    if args.write_manifest or args.submit:
        _write_json(manifest_path, campaign_manifest)

    if args.dry_run:
        print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
        return 0

    logs.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, **environment},
    )
    summary["stdout"] = result.stdout
    summary["stderr"] = result.stderr
    summary["job_id"] = parse_sbatch_job_id(result.stdout)
    _write_json(manifest_path.with_suffix(".submission.json"), summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
