"""Preflight and read-only status reporting for generated campaign shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest(path: Path) -> list[dict[str, str]]:
    rows = _rows(path)
    if not rows:
        raise ValueError(f"Manifest is empty or missing: {path}")
    return rows


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _expected_n_theta(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    theta = experiment["observation_theta"]
    source = theta.get("source", {})
    optics = theta.get("optics", {})
    count = sum(bool(value) for value in source.values())
    count += int(bool(optics.get("plate_scale_as_per_pix", False)))
    for key in ("primary_zernikes", "secondary_zernikes"):
        request = optics.get(key, {})
        if isinstance(request, Mapping) and bool(request.get("enabled", False)):
            count += len(request.get("indices", []))
    return count


def _csv_count(path: Path) -> int:
    return len(_rows(path))


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in text_rows))
        for index in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * width for width in widths))
    for row in text_rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def preflight(manifest_path: Path, results_root: Path) -> int:
    manifest_rows = _manifest(manifest_path)
    total_expected = sum(int(row["expected_subblocks"]) for row in manifest_rows)
    table: list[list[Any]] = []
    failures = 0
    for index, row in enumerate(manifest_rows):
        shard_name = row["shard_name"]
        config_path = _repo_path(row["config_path"])
        command = [
            sys.executable,
            str(RUNNER),
            "--config",
            str(config_path),
            "--results-root",
            str(results_root),
            "--run-name",
            shard_name,
            "--dry-run",
            "--max-workers",
            "1",
            "--resource-time",
            "auto",
            "--quiet",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={**dict(os.environ), "PYTHONPATH": "src"},
            check=False,
            capture_output=True,
            text=True,
        )
        run_root = results_root / shard_name
        required = (
            "campaign_plan.json",
            "subblock_plan.csv",
            "iterative_plan.csv",
            "expected_outputs.csv",
        )
        missing = [name for name in required if not (run_root / name).exists()]
        expected_subblocks = int(row["expected_subblocks"])
        expected_windows = int(row["expected_windows"])
        actual_subblocks = _csv_count(run_root / "subblock_plan.csv")
        actual_expected_outputs = _csv_count(run_root / "expected_outputs.csv")
        actual_windows = len(
            {
                (item.get("case_name", ""), item.get("window_index", ""))
                for item in _rows(run_root / "iterative_plan.csv")
            }
        )
        theta_ok = False
        n_theta = ""
        if (run_root / "campaign_plan.json").exists():
            plan = json.loads(
                (run_root / "campaign_plan.json").read_text(encoding="utf-8")
            )
            labels = plan.get("theta_layout", {}).get("labels", [])
            n_theta = int(plan.get("dimension_estimate", {}).get("n_theta", len(labels)))
            source_expected_n_theta = int(
                row.get("expected_n_theta") or _expected_n_theta(config_path)
            )
            theta_ok = bool(labels) and n_theta == source_expected_n_theta
        first_not_parent = not (
            index == 0
            and len(manifest_rows) > 1
            and actual_subblocks >= total_expected
        )
        passed = (
            completed.returncode == 0
            and not missing
            and actual_subblocks == expected_subblocks
            and actual_expected_outputs == expected_subblocks
            and actual_windows == expected_windows
            and theta_ok
            and first_not_parent
        )
        if not passed:
            failures += 1
        detail = "ok"
        if completed.returncode != 0:
            detail = (completed.stderr.strip().splitlines() or ["runner failed"])[-1]
        elif missing:
            detail = "missing:" + ",".join(missing)
        elif not first_not_parent:
            detail = "first shard contains parent-size plan"
        elif actual_subblocks != expected_subblocks:
            detail = f"subblocks {actual_subblocks}!={expected_subblocks}"
        elif actual_expected_outputs != expected_subblocks:
            detail = (
                f"expected outputs {actual_expected_outputs}!="
                f"{expected_subblocks}"
            )
        elif actual_windows != expected_windows:
            detail = f"windows {actual_windows}!={expected_windows}"
        elif not theta_ok:
            detail = f"theta layout mismatch (n={n_theta})"
        table.append(
            [
                "PASS" if passed else "FAIL",
                shard_name,
                f"{actual_subblocks}/{expected_subblocks}",
                f"{actual_windows}/{expected_windows}",
                n_theta,
                detail,
            ]
        )
    _print_table(
        ("result", "shard", "subblocks", "windows", "n_theta", "detail"),
        table,
    )
    return 1 if failures else 0


def _latest_files(run_root: Path, limit: int = 3) -> str:
    if not run_root.exists():
        return ""
    files = [path for path in run_root.rglob("*") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return ";".join(str(path.relative_to(run_root)) for path in files[:limit])


def status(manifest_path: Path, results_root: Path) -> int:
    manifest_rows = _manifest(manifest_path)
    table: list[list[Any]] = []
    for row in manifest_rows:
        shard_name = row["shard_name"]
        run_root = results_root / "observation_bias_campaign" / shard_name
        expected = int(row["expected_subblocks"])
        status_rows = _rows(run_root / "subblock_status_iterative.csv")
        statuses = Counter(item.get("status", "") for item in status_rows)
        elapsed = []
        for item in status_rows:
            try:
                elapsed.append(float(item["elapsed_seconds"]))
            except (KeyError, TypeError, ValueError):
                pass
        median_elapsed = f"{statistics.median(elapsed):.1f}" if elapsed else ""
        completed_summaries = sum(
            1
            for _ in run_root.glob(
                "subblock_runs/**/schur_summary/subblock_summary.json"
            )
        )
        expected_outputs = _csv_count(run_root / "expected_outputs.csv")
        subblock_plan = _csv_count(run_root / "subblock_plan.csv")
        complete = (
            run_root.exists()
            and (run_root / "campaign_plan.json").exists()
            and (run_root / "campaign_summary.json").exists()
            and len(status_rows) == expected
            and statuses.get("ok", 0) == expected
            and completed_summaries == expected
        )
        table.append(
            [
                shard_name,
                "yes" if run_root.exists() else "no",
                "yes" if (run_root / "campaign_plan.json").exists() else "no",
                "yes" if (run_root / "campaign_summary.json").exists() else "no",
                expected_outputs,
                subblock_plan,
                len(status_rows),
                statuses.get("ok", 0),
                statuses.get("error", 0) + statuses.get("failed", 0),
                completed_summaries,
                median_elapsed,
                "yes" if complete else "no",
            ]
        )
        print(f"\n{shard_name} latest: {_latest_files(run_root)}")
    print()
    _print_table(
        (
            "shard",
            "root",
            "plan",
            "summary",
            "expected",
            "planned",
            "status",
            "ok",
            "error",
            "schur",
            "median_s",
            "complete",
        ),
        table,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--results-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "preflight":
        raise SystemExit(preflight(args.manifest, args.results_root))
    raise SystemExit(status(args.manifest, args.results_root))


if __name__ == "__main__":
    main()
