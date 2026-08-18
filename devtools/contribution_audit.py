#!/usr/bin/env python3
"""Generate a lightweight Git contribution audit report for dLuxShera."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/dev/notes/contribution_audit_2026.md"
JSON_PATH = ROOT / "docs/dev/notes/contribution_audit_2026_metrics.json"
CSV_PATH = ROOT / "docs/dev/notes/contribution_audit_2026_numstat.csv"
SINCE_DATE = "2026-01-01"
PREFERRED_BRANCH = "observation-level-update"

INCLUDE_PATHS = [
    "src/**",
    "tests/**",
    "examples/**",
    "docs/**",
    "devtools/**",
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "requirements*.txt",
    "environment*.yml",
    "*.md",
    "*.yaml",
    "*.yml",
    "*.toml",
    "*.cfg",
    "*.ini",
]

EXCLUDE_PATHS = [
    "Results/**",
    "**/.ipynb_checkpoints/**",
    "devtools/context_snapshot_*/**",
    "**/*.fits",
    "**/*.npz",
    "**/*.npy",
    "**/*.png",
    "**/*.jpg",
    "**/*.jpeg",
    "**/*.pdf",
    "**/*.gif",
    "**/*.mp4",
    "**/*.zip",
    "**/*.tar",
    "**/*.gz",
    ".venv/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    "**/__pycache__/**",
    "**/.DS_Store",
    "**/*.ipynb",
]

AREA_ORDER = [
    "src/",
    "tests/",
    "examples/",
    "docs/",
    "devtools/",
    "top-level config / packaging / README",
    "other tracked text files",
]

TOP_LEVEL_CONFIG_NAMES = {
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    ".pre-commit-config.yaml",
    "mypy.ini",
}
TOP_LEVEL_PREFIXES = ("requirements", "environment")
TOP_LEVEL_SUFFIXES = (".md", ".yaml", ".yml", ".toml", ".cfg", ".ini")


@dataclass
class AreaMetrics:
    files: set[str] = field(default_factory=set)
    added: int = 0
    deleted: int = 0

    @property
    def net(self) -> int:
        return self.added - self.deleted

    @property
    def churn(self) -> int:
        return self.added + self.deleted

    def to_json(self) -> dict[str, int]:
        return {
            "files_changed": len(self.files),
            "lines_added": self.added,
            "lines_deleted": self.deleted,
            "net_lines": self.net,
            "churn": self.churn,
        }


@dataclass
class ScopeMetrics:
    name: str
    range_label: str
    command: list[str]
    commit_count: int
    rows: list[dict[str, object]]
    notes: list[str]

    @property
    def files(self) -> set[str]:
        return {str(row["path"]) for row in self.rows}

    @property
    def added(self) -> int:
        return sum(int(row["added"]) for row in self.rows)

    @property
    def deleted(self) -> int:
        return sum(int(row["deleted"]) for row in self.rows)

    @property
    def net(self) -> int:
        return self.added - self.deleted

    @property
    def churn(self) -> int:
        return self.added + self.deleted

    @property
    def by_area(self) -> dict[str, AreaMetrics]:
        areas = {name: AreaMetrics() for name in AREA_ORDER}
        for row in self.rows:
            path = str(row["path"])
            area = area_for_path(path)
            areas[area].files.add(path)
            areas[area].added += int(row["added"])
            areas[area].deleted += int(row["deleted"])
        return areas

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "range": self.range_label,
            "commits_counted": self.commit_count,
            "files_changed": len(self.files),
            "lines_added": self.added,
            "lines_deleted": self.deleted,
            "net_lines": self.net,
            "churn": self.churn,
            "breakdown_by_area": {k: v.to_json() for k, v in self.by_area.items()},
            "notes": self.notes,
            "command": self.command,
        }


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def run_git_maybe(args: list[str]) -> str | None:
    try:
        return run_git(args)
    except subprocess.CalledProcessError:
        return None


def pathspec() -> list[str]:
    includes = [f":(glob){item}" for item in INCLUDE_PATHS]
    excludes = [f":(exclude,glob){item}" for item in EXCLUDE_PATHS]
    return includes + excludes


def parse_numstat(output: str, scope: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added, deleted, raw_path = parts[0], parts[1], parts[2]
        if not (added.isdigit() and deleted.isdigit()):
            continue
        path = normalize_numstat_path(raw_path)
        rows.append(
            {
                "scope": scope,
                "path": path,
                "added": int(added),
                "deleted": int(deleted),
                "area": area_for_path(path),
            }
        )
    return rows


def normalize_numstat_path(path: str) -> str:
    # Rename rows can appear as "old => new" or "{old => new}/file". Use the new side.
    if " => " in path:
        if path.startswith("{") and "}" in path:
            prefix, suffix = path.split("}", 1)
            new_prefix = prefix.split(" => ", 1)[1]
            path = f"{new_prefix}{suffix}"
        else:
            path = path.split(" => ", 1)[1]
    return path.strip()


def area_for_path(path: str) -> str:
    for prefix in ("src/", "tests/", "examples/", "docs/", "devtools/"):
        if path.startswith(prefix):
            return prefix
    if "/" not in path:
        if (
            path in TOP_LEVEL_CONFIG_NAMES
            or path.startswith(TOP_LEVEL_PREFIXES)
            or path.endswith(TOP_LEVEL_SUFFIXES)
        ):
            return "top-level config / packaging / README"
    return "other tracked text files"


def current_branch() -> str:
    return run_git(["rev-parse", "--abbrev-ref", "HEAD"])


def working_tree_status() -> str:
    status = run_git(["status", "--porcelain"])
    return status or "clean"


def select_branch_target() -> tuple[str, str, list[str]]:
    notes: list[str] = []
    if run_git_maybe(["rev-parse", "--verify", PREFERRED_BRANCH]) is not None:
        return PREFERRED_BRANCH, "preferred local branch exists", notes
    if run_git_maybe(["rev-parse", "--verify", f"origin/{PREFERRED_BRANCH}"]) is not None:
        notes.append(f"Used origin/{PREFERRED_BRANCH} because no local branch existed.")
        return f"origin/{PREFERRED_BRANCH}", "preferred origin branch exists", notes
    notes.append(f"Fell back to HEAD because {PREFERRED_BRANCH} was not found locally or on origin.")
    return "HEAD", "fallback to HEAD", notes


def build_branch_scope() -> tuple[ScopeMetrics, dict[str, str]]:
    target, target_reason, notes = select_branch_target()
    base = run_git(["merge-base", "origin/main", target])
    base_short = run_git(["rev-parse", "--short", base])
    target_short = run_git(["rev-parse", "--short", target])
    range_expr = f"{base}..{target}"
    command = ["diff", "--numstat", range_expr, "--", *pathspec()]
    rows = parse_numstat(run_git(command), "branch")
    commit_count = int(run_git(["rev-list", "--count", range_expr, "--", *pathspec()]) or "0")
    meta = {
        "target": target,
        "target_reason": target_reason,
        "base": base,
        "base_short": base_short,
        "target_short": target_short,
        "range_expr": range_expr,
    }
    scope = ScopeMetrics(
        name="Scope A: observation-level branch contribution",
        range_label=f"{base_short}..{target_short} ({target} vs merge-base with origin/main)",
        command=["git", *command],
        commit_count=commit_count,
        rows=rows,
        notes=notes,
    )
    return scope, meta


def build_time_scope() -> ScopeMetrics:
    log_range = f"--since={SINCE_DATE}"
    command = ["log", log_range, "--no-merges", "--numstat", "--pretty=tformat:", "HEAD", "--", *pathspec()]
    rows = parse_numstat(run_git(command), "since_2026_01_01")
    commit_count = int(run_git(["rev-list", "--count", log_range, "--no-merges", "HEAD", "--", *pathspec()]) or "0")
    return ScopeMetrics(
        name="Scope B: contribution since 2026-01-01",
        range_label=f"{SINCE_DATE} through HEAD, no author filter, merge commits excluded",
        command=["git", *command],
        commit_count=commit_count,
        rows=rows,
        notes=["No author filter was used; merge commits were excluded for the time-window scope."],
    )


def notebook_count(range_args: list[str]) -> int:
    out = run_git([*range_args, "--", ":(glob)**/*.ipynb"])
    return len([line for line in out.splitlines() if line.strip()])


def fmt_int(value: int) -> str:
    return f"{value:,}"


def metrics_table(scope: ScopeMetrics) -> str:
    rows = [
        ("Commits counted", fmt_int(scope.commit_count)),
        ("Files changed", fmt_int(len(scope.files))),
        ("Lines added", fmt_int(scope.added)),
        ("Lines deleted", fmt_int(scope.deleted)),
        ("Net lines", fmt_int(scope.net)),
        ("Churn (added + deleted)", fmt_int(scope.churn)),
    ]
    return "\n".join(["| Metric | Value |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in rows])


def area_table(scope: ScopeMetrics) -> str:
    lines = [
        "| Area | Files | Added | Deleted | Net | Churn |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for area, metrics in scope.by_area.items():
        lines.append(
            f"| {area} | {fmt_int(len(metrics.files))} | {fmt_int(metrics.added)} | "
            f"{fmt_int(metrics.deleted)} | {fmt_int(metrics.net)} | {fmt_int(metrics.churn)} |"
        )
    return "\n".join(lines)


def top_files(scope: ScopeMetrics, limit: int = 12) -> list[str]:
    totals: dict[str, AreaMetrics] = {}
    for row in scope.rows:
        path = str(row["path"])
        totals.setdefault(path, AreaMetrics()).files.add(path)
        totals[path].added += int(row["added"])
        totals[path].deleted += int(row["deleted"])
    ranked = sorted(totals.items(), key=lambda item: item[1].churn, reverse=True)[:limit]
    return [f"{path} ({fmt_int(metrics.churn)} churn)" for path, metrics in ranked]


def build_report(branch_scope: ScopeMetrics, time_scope: ScopeMetrics, branch_meta: dict[str, str]) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    head = run_git(["rev-parse", "HEAD"])
    head_short = run_git(["rev-parse", "--short", "HEAD"])
    branch = current_branch()
    dirty = working_tree_status()
    dirty_text = "clean" if dirty == "clean" else "dirty; committed-history audit excludes uncommitted/untracked files"
    branch_notebooks = notebook_count(["diff", "--name-only", branch_meta["range_expr"]])
    time_notebooks = notebook_count(["log", f"--since={SINCE_DATE}", "--no-merges", "--name-only", "--pretty=tformat:", "HEAD"])

    one_sentence = (
        f"Since January 2026, dLuxShera accumulated approximately {fmt_int(time_scope.net)} net tracked text/source lines "
        f"and {fmt_int(time_scope.churn)} total changed lines across {fmt_int(len(time_scope.files))} files, excluding generated artifacts, "
        "spanning observation-level estimation, Schur/subblock summaries, calibration campaigns, trajectory support, HPC utilities, tests, docs, and reproducibility tooling."
    )

    short_summary = (
        f"The current {PREFERRED_BRANCH} branch represents approximately {fmt_int(branch_scope.net)} net tracked text/source lines "
        f"and {fmt_int(branch_scope.churn)} total changed lines across {fmt_int(len(branch_scope.files))} files relative to its merge-base with origin/main. "
        f"Across committed history since {SINCE_DATE}, the comparable audit shows approximately {fmt_int(time_scope.net)} net lines and "
        f"{fmt_int(time_scope.churn)} total changed lines across {fmt_int(len(time_scope.files))} files, with generated artifacts, binary outputs, bulky results, and notebooks excluded from LOC totals. "
        "These numbers are best read as engineering contribution scale, not as a standalone productivity metric."
    )

    return f"""# dLuxShera contribution audit, 2026

Generated: {generated}

## Supervisor-ready summary

**One-sentence version:** {one_sentence}

**Slightly longer version:** {short_summary}

## Headline metrics

| Scope | Commits | Files | Added | Deleted | Net | Churn |
|---|---:|---:|---:|---:|---:|---:|
| Branch: `{branch_scope.range_label}` | {fmt_int(branch_scope.commit_count)} | {fmt_int(len(branch_scope.files))} | {fmt_int(branch_scope.added)} | {fmt_int(branch_scope.deleted)} | {fmt_int(branch_scope.net)} | {fmt_int(branch_scope.churn)} |
| Since `{SINCE_DATE}` through `HEAD` | {fmt_int(time_scope.commit_count)} | {fmt_int(len(time_scope.files))} | {fmt_int(time_scope.added)} | {fmt_int(time_scope.deleted)} | {fmt_int(time_scope.net)} | {fmt_int(time_scope.churn)} |

## Recommended wording for award nomination

- **Conservative:** Since January 2026, the tracked dLuxShera repository history shows approximately {fmt_int(time_scope.churn)} changed text/source lines across {fmt_int(len(time_scope.files))} files, excluding generated artifacts and notebooks, with work concentrated in observation-level estimation, campaign infrastructure, tests, documentation, and reproducibility tooling.
- **LOC-forward:** The committed repository history since January 2026 contains approximately {fmt_int(time_scope.added)} added and {fmt_int(time_scope.deleted)} deleted tracked text/source lines, or {fmt_int(time_scope.net)} net lines and {fmt_int(time_scope.churn)} total changed lines across {fmt_int(len(time_scope.files))} files, excluding generated artifacts.
- **Capability-forward:** The contribution advanced dLuxShera from core modeling scripts toward a more complete observation-level inference and campaign platform, adding belief-state update mechanics, image-backed Schur/subblock summaries, single-star calibration and Monte Carlo campaign wrappers, trajectory-driven workflows, HPC readiness utilities, and supporting tests/docs.
- **Award-nomination style:** Over roughly the first half of 2026, this work delivered a substantial, reproducible engineering expansion of dLuxShera: approximately {fmt_int(time_scope.churn)} tracked changed lines across {fmt_int(len(time_scope.files))} source, test, example, documentation, and tooling files, while adding major capabilities for observation-level estimation, calibration campaigns, trajectory-driven studies, and HPC-ready experimentation.

## Method

This is a lightweight, reproducible Git audit intended to estimate contribution scale, not productivity or impact by itself.

- **Branch scope:** measured `{branch_meta['target']}` against its merge-base with `origin/main`: `{branch_meta['base_short']}..{branch_meta['target_short']}`.
- **Time-window scope:** measured committed tracked repository changes since `{SINCE_DATE}` through `HEAD`, with no author filter and merge commits excluded.
- **Included paths:** `src/**`, `tests/**`, `examples/**`, `docs/**`, `devtools/**`, plus top-level Markdown/config/packaging files such as `README.md`, `AGENTS.md`, `pyproject.toml`, `requirements*.txt`, `environment*.yml`, `*.md`, `*.yaml`, `*.yml`, `*.toml`, `*.cfg`, and `*.ini`.
- **Excluded paths:** generated results, context snapshots, binary/image/science data products, archives, virtualenv/cache directories, `__pycache__`, `.DS_Store`, and notebooks.
- **Notebook treatment:** notebooks were excluded from LOC totals because notebook JSON can distort line metrics. Matching notebook files observed: {branch_notebooks} in branch scope and {time_notebooks} in the time-window scope.
- **Added lines:** lines introduced in tracked text/source files by Git numstat.
- **Deleted lines:** lines removed from tracked text/source files by Git numstat.
- **Net lines:** added minus deleted lines.
- **Churn:** added plus deleted lines; useful for estimating engineering change volume.

LOC and churn are rough proxies. They should be interpreted alongside files touched, tests, docs, reproducibility work, and delivered capabilities.

## Scope A: observation-level branch contribution

{metrics_table(branch_scope)}

Notes: {' '.join(branch_scope.notes) if branch_scope.notes else 'Preferred local branch `observation-level-update` was used.'}

## Scope B: contribution since 2026-01-01

{metrics_table(time_scope)}

Notes: {' '.join(time_scope.notes)}

## Breakdown by repository area

### Branch scope

{area_table(branch_scope)}

### Since 2026-01-01

{area_table(time_scope)}

## Capability summary

Based on changed files, commit messages, and existing docs/scripts, the audited changes represent work in these areas:

- Observation-level estimation and belief-state update architecture, including demonstration workflows and design notes.
- Image-backed subblock Schur summaries, structured curvature/FIM paths, preconditioning, diagnostics, and observation-summary simulation.
- Single-star calibration campaign infrastructure, including orchestration, scheduling, early stopping, resource-time handling, and campaign wrappers.
- Observation-bias and Monte Carlo campaign support, including truth-realization draws, stochastic summary modes, and reusable campaign utilities.
- Trajectory-driven subblock and campaign support using trace generation, rendering pipelines, and integrated trace sources.
- HPC readiness work, including preflight/readiness notes, Slurm/batch assets, runtime profiling, memory benchmarking, subprocess diagnostics, and campaign utilities.
- Tests, docs, examples, packaging/environment updates, and developer workflow improvements supporting reproducibility and maintainability.

Top churn files in the branch scope:

{chr(10).join(f'- {item}' for item in top_files(branch_scope))}

Top churn files since {SINCE_DATE}:

{chr(10).join(f'- {item}' for item in top_files(time_scope))}

## Caveats

- This audit reports tracked repository changes, excluding generated artifacts and bulky binary/science outputs.
- It does not separate authors; per the stated assumption, repository contributions are treated as the user's work, including Codex-assisted work.
- The working tree status at generation time was `{dirty_text}`.
- The time-window scope uses committed history only and excludes merge commits to avoid double-counting merged branch diffs.
- Line metrics are sensitive to refactors, file moves, formatting, and generated text committed to tracked paths.

## Raw command provenance

- Current branch: `{branch}`
- `HEAD`: `{head}` (`{head_short}`)
- Working tree status: `{dirty_text}`
- Branch target selection: `{branch_meta['target']}` ({branch_meta['target_reason']})
- Branch merge-base: `{branch_meta['base']}` (`{branch_meta['base_short']}`)
- Branch numstat command: `{' '.join(branch_scope.command)}`
- Time-window numstat command: `{' '.join(time_scope.command)}`
- Raw metrics JSON: `docs/dev/notes/contribution_audit_2026_metrics.json`
- Raw numstat CSV: `docs/dev/notes/contribution_audit_2026_numstat.csv`
"""


def write_sidecars(scopes: Iterable[ScopeMetrics], branch_meta: dict[str, str]) -> None:
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "head": run_git(["rev-parse", "HEAD"]),
        "current_branch": current_branch(),
        "working_tree_status": working_tree_status(),
        "branch_metadata": branch_meta,
        "include_paths": INCLUDE_PATHS,
        "exclude_paths": EXCLUDE_PATHS,
        "scopes": {scope.name: scope.to_json() for scope in scopes},
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scope", "area", "path", "added", "deleted"])
        writer.writeheader()
        for scope in scopes:
            for row in scope.rows:
                writer.writerow(row)


def main() -> None:
    branch_scope, branch_meta = build_branch_scope()
    time_scope = build_time_scope()
    report = build_report(branch_scope, time_scope, branch_meta)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    write_sidecars([branch_scope, time_scope], branch_meta)
    print(f"Wrote {REPORT_PATH.relative_to(ROOT)}")
    print(f"Wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"Wrote {CSV_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
