"""Prepare independent condition or draw shards for iterative campaigns.

The source YAML is copied semantically. Generated configs only change the run
name and the prior-draw selection metadata needed to isolate a condition or
draw while preserving the parent campaign's RNG stream.
"""

from __future__ import annotations

import argparse
import copy
import csv
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WRAPPER = (
    "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_iterative_campaign_hpc.sbatch"
)
PRIMARY_SIGMA_KEY = "optics.primary.zernike_coeffs_nm[*]"
SECONDARY_SIGMA_KEY = "optics.secondary.zernike_coeffs_nm[*]"
MANIFEST_FIELDS = (
    "shard_name",
    "shard_mode",
    "source_config_path",
    "config_path",
    "expected_run_root",
    "condition_label",
    "m1_sigma_nm",
    "m2_sigma_nm",
    "draw_start",
    "draw_stop",
    "draw_index",
    "expected_subblocks",
    "expected_windows",
    "expected_subblocks_per_window",
    "expected_n_theta",
    "recommended_time",
    "recommended_cpus_per_task",
    "recommended_mem",
    "recommended_max_workers",
    "sbatch_command",
)


@dataclass(frozen=True)
class Resources:
    time: str
    cpus_per_task: int
    mem: str
    max_workers: int


@dataclass(frozen=True)
class Shard:
    name: str
    mode: str
    condition_label: str
    condition_index: int
    draw_start: int
    draw_stop: int
    draw_index: int | None
    config: dict[str, Any]
    expected_subblocks: int
    expected_windows: int
    expected_subblocks_per_window: int
    m1_sigma_nm: float | None
    m2_sigma_nm: float | None


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(value)


def _safe_name(value: str, *, name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise ValueError(f"{name} does not contain any filesystem-safe characters.")
    return cleaned


def _sigma_nm(condition: Mapping[str, Any], key: str) -> float | None:
    sigmas = condition.get("sigmas", {})
    if not isinstance(sigmas, Mapping):
        return None
    entry = sigmas.get(key)
    if not isinstance(entry, Mapping) or entry.get("sigma") is None:
        return None
    return float(entry["sigma"])


def _expected_n_theta(config: Mapping[str, Any]) -> int:
    experiment = _mapping(config.get("experiment"), name="experiment")
    theta = _mapping(
        experiment.get("observation_theta"),
        name="experiment.observation_theta",
    )
    source = _mapping(theta.get("source", {}), name="observation_theta.source")
    optics = _mapping(theta.get("optics", {}), name="observation_theta.optics")
    count = sum(bool(value) for value in source.values())
    count += int(bool(optics.get("plate_scale_as_per_pix", False)))
    for key in ("primary_zernikes", "secondary_zernikes"):
        request = optics.get(key, {})
        if isinstance(request, Mapping) and bool(request.get("enabled", False)):
            count += len(request.get("indices", []))
    return count


def load_campaign_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Campaign YAML must contain a mapping.")
    config = copy.deepcopy(dict(payload))
    experiment = _mapping(config.get("experiment"), name="experiment")
    if str(experiment.get("kind", "")) != "full_fidelity_binary_iterative":
        raise ValueError(
            "Sharding currently requires experiment.kind="
            "'full_fidelity_binary_iterative'."
        )
    iterative = _mapping(experiment.get("iterative"), name="experiment.iterative")
    if not bool(iterative.get("enabled", False)):
        raise ValueError("Sharding requires experiment.iterative.enabled=true.")
    prior_draws = _mapping(experiment.get("prior_draws"), name="experiment.prior_draws")
    if not bool(prior_draws.get("enabled", False)):
        raise ValueError("Sharding requires experiment.prior_draws.enabled=true.")
    conditions = prior_draws.get("conditions")
    if not isinstance(conditions, Sequence) or isinstance(conditions, (str, bytes)):
        raise ValueError("Sharding requires experiment.prior_draws.conditions.")
    if not conditions:
        raise ValueError("At least one prior-draw condition is required.")
    return config


def build_shards(
    config: Mapping[str, Any],
    *,
    run_name_prefix: str,
    mode: str,
) -> list[Shard]:
    if mode not in {"condition", "draw"}:
        raise ValueError("mode must be 'condition' or 'draw'.")
    source = copy.deepcopy(dict(config))
    experiment = _mapping(source["experiment"], name="experiment")
    iterative = _mapping(experiment["iterative"], name="experiment.iterative")
    prior_draws = _mapping(experiment["prior_draws"], name="experiment.prior_draws")
    conditions = list(prior_draws["conditions"])
    draws_per_condition = int(
        prior_draws.get(
            "n_cases",
            prior_draws.get(
                "draws_per_condition",
                prior_draws.get("prior_draws_per_condition", 0),
            ),
        )
    )
    if draws_per_condition <= 0:
        raise ValueError("prior_draws.n_cases must be positive.")
    windows = int(iterative.get("windows_per_draw", 1))
    subblocks_per_window = int(iterative.get("subblocks_per_window", 1))
    if windows <= 0 or subblocks_per_window <= 0:
        raise ValueError("Iterative window and subblock counts must be positive.")

    prefix = _safe_name(run_name_prefix, name="run-name-prefix")
    shards: list[Shard] = []
    seen_names: set[str] = set()
    for condition_index, condition_raw in enumerate(conditions):
        condition = _mapping(
            condition_raw,
            name=f"prior_draws.conditions[{condition_index}]",
        )
        condition_label = str(condition.get("condition_name", "")).strip()
        if not condition_label:
            raise ValueError(
                f"prior_draws.conditions[{condition_index}] requires condition_name."
            )
        safe_condition = _safe_name(condition_label, name="condition_name")
        draw_indices = range(draws_per_condition) if mode == "draw" else (None,)
        for selected_draw in draw_indices:
            shard_config = copy.deepcopy(source)
            shard_experiment = shard_config["experiment"]
            shard_prior = shard_experiment["prior_draws"]
            shard_prior["conditions"] = [copy.deepcopy(condition)]
            shard_prior["condition_index_start"] = condition_index
            if selected_draw is None:
                draw_start = 0
                draw_stop = draws_per_condition
                shard_prior["n_cases"] = draws_per_condition
                shard_prior["draw_index_start"] = 0
                shard_prior["global_draw_index_start"] = (
                    condition_index * draws_per_condition
                )
                shard_prior["rng_skip_draws"] = (
                    condition_index * draws_per_condition
                )
                shard_name = f"{prefix}_cond_{safe_condition}"
                draw_index = None
            else:
                draw_start = int(selected_draw)
                draw_stop = draw_start + 1
                shard_prior["n_cases"] = 1
                shard_prior["draw_index_start"] = draw_start
                shard_prior["global_draw_index_start"] = (
                    condition_index * draws_per_condition + draw_start
                )
                shard_prior["rng_skip_draws"] = (
                    condition_index * draws_per_condition + draw_start
                )
                shard_name = (
                    f"{prefix}_cond_{safe_condition}_draw_{draw_start:03d}"
                )
                draw_index = draw_start
            shard_name = _safe_name(shard_name, name="shard_name")
            if shard_name in seen_names:
                raise ValueError(f"Generated duplicate shard name: {shard_name}.")
            seen_names.add(shard_name)
            shard_experiment["run_name"] = shard_name
            selected_draws = draw_stop - draw_start
            shards.append(
                Shard(
                    name=shard_name,
                    mode=mode,
                    condition_label=condition_label,
                    condition_index=condition_index,
                    draw_start=draw_start,
                    draw_stop=draw_stop,
                    draw_index=draw_index,
                    config=shard_config,
                    expected_subblocks=(
                        selected_draws * windows * subblocks_per_window
                    ),
                    expected_windows=selected_draws * windows,
                    expected_subblocks_per_window=subblocks_per_window,
                    m1_sigma_nm=_sigma_nm(condition, PRIMARY_SIGMA_KEY),
                    m2_sigma_nm=_sigma_nm(condition, SECONDARY_SIGMA_KEY),
                )
            )
    return shards


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _expected_run_root(results_root: Path | None, shard_name: str) -> str:
    if results_root is None:
        return f"$RESULTS_ROOT/observation_bias_campaign/{shard_name}"
    return str(
        (results_root.resolve() / "observation_bias_campaign" / shard_name)
    )


def _short_job_name(shard: Shard) -> str:
    condition = _safe_name(shard.condition_label, name="condition_label")
    if shard.draw_index is None:
        return f"ff-{condition}"[:80]
    return f"ff-{condition}-d{shard.draw_index:03d}"[:80]


def _sbatch_command(
    *,
    shard: Shard,
    config_path: Path,
    results_root: Path | None,
    resources: Resources,
) -> str:
    result_export = (
        f"RESULTS_ROOT={shlex.quote(str(results_root.resolve()))}"
        if results_root is not None
        else 'RESULTS_ROOT="$RESULTS_ROOT"'
    )
    exports = ",".join(
        (
            "ALL",
            result_export,
            f"CONFIG={shlex.quote(_repo_relative(config_path))}",
            f"RUN_NAME={shlex.quote(shard.name)}",
            f"MAX_WORKERS={resources.max_workers}",
            "FAIL_FAST=1",
            "ANALYZE_AFTER_RUN=1",
            "USE_RESOURCE_TIME=1",
        )
    )
    return " ".join(
        (
            "sbatch",
            f"--time={shlex.quote(resources.time)}",
            f"--cpus-per-task={resources.cpus_per_task}",
            f"--mem={shlex.quote(resources.mem)}",
            f"--job-name={shlex.quote(_short_job_name(shard))}",
            '--output="$RESULTS_ROOT/slurm_logs/%x-%j.out"',
            '--error="$RESULTS_ROOT/slurm_logs/%x-%j.err"',
            f"--export={exports}",
            DEFAULT_WRAPPER,
        )
    )


def _script_header() -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run from the repository root on the cluster.",
        "# source \"$(conda info --base)/etc/profile.d/conda.sh\"",
        "# conda activate dluxshera-py311",
        "",
    ]


def _submit_script(
    *,
    rows: Sequence[Mapping[str, Any]],
    results_root: Path | None,
) -> str:
    lines = _script_header()
    default_root = (
        str(results_root.resolve())
        if results_root is not None
        else "/scratch/shera_hpc/$USER/dluxshera"
    )
    lines.extend(
        [
            f'RESULTS_ROOT="${{RESULTS_ROOT:-{default_root}}}"',
            'mkdir -p "$RESULTS_ROOT/slurm_logs"',
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"# {row['shard_name']}",
                f"CONFIG={shlex.quote(str(row['config_path']))}",
                f"RUN_NAME={shlex.quote(str(row['shard_name']))}",
                "MAX_WORKERS="
                + shlex.quote(str(row["recommended_max_workers"])),
                "export CONFIG RUN_NAME MAX_WORKERS RESULTS_ROOT",
                str(row["sbatch_command"]),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _preflight_script(*, manifest_path: Path, mode: str) -> str:
    lines = _script_header()
    lines.extend(
        [
            'PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-'
            + _repo_relative(manifest_path.parent / "preflight_results")
            + '}"',
            'mkdir -p "$PREFLIGHT_ROOT"',
            "PYTHONPATH=src python "
            "examples/scripts/check_full_fidelity_campaign_shards.py preflight "
            f"--manifest {shlex.quote(_repo_relative(manifest_path))} "
            '--results-root "$PREFLIGHT_ROOT"',
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _status_script(
    *,
    manifest_path: Path,
    results_root: Path | None,
) -> str:
    lines = _script_header()
    default_root = (
        str(results_root.resolve())
        if results_root is not None
        else "/scratch/shera_hpc/$USER/dluxshera"
    )
    lines.extend(
        [
            f'RESULTS_ROOT="${{RESULTS_ROOT:-{default_root}}}"',
            "PYTHONPATH=src python "
            "examples/scripts/check_full_fidelity_campaign_shards.py status "
            f"--manifest {shlex.quote(_repo_relative(manifest_path))} "
            '--results-root "$RESULTS_ROOT"',
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _readme(
    *,
    source_config: Path,
    mode: str,
    resources: Resources,
    shard_count: int,
) -> str:
    return f"""# Full-fidelity iterative campaign shards

Source config: `{_repo_relative(source_config)}`

This directory contains {shard_count} `{mode}` shards. The original 300-subblock
campaign exceeded a 24-hour Slurm allocation: 50 successful rows had a median
subblock runtime of about 4278 seconds. Iterative windows within a draw are
sequential because each posterior/reference update feeds the next window.
Conditions and prior draws are independent, while subblocks within one window
are independent.

Condition sharding is the recommended production mode because it reduces the
current 2x2 campaign to four jobs without fragmenting it into 20 small jobs.
Draw mode is available for tighter scheduling or reruns. `MAX_WORKERS=5` matches
the current five subblocks per window; larger values do not add useful
parallelism unless the runner gains another safe parallel axis.

## Workflow

From the repository root:

```bash
./preflight_{mode}_shards.sh
./submit_{mode}_shards.sh
./summarize_shard_status.sh
```

Recommended resources are `{resources.time}`, {resources.cpus_per_task} CPUs,
`{resources.mem}`, and `MAX_WORKERS={resources.max_workers}`. Preflight uses
`--dry-run --max-workers 1 --resource-time auto`, verifies required plan
artifacts, checks expected counts and theta layout, and rejects a first shard
that accidentally contains the full parent campaign.

Each shard keeps its own run root and runs the existing analyzer independently.
`shard_manifest.csv` is the source of truth connecting shards to the parent
campaign and should be referenced from the Campaign Tracker with submitted job
IDs. A future multi-run aggregator may concatenate compatible per-shard
analysis tables; no cross-shard science aggregation is performed here.

## GPU benchmark

Do not switch production submissions to GPU by default. Benchmark one
condition, one draw, one window, and five subblocks separately with
`MAX_WORKERS=1` or `2`, then compare CPU and GPU subblock runtimes.
"""


def prepare_shards(
    *,
    config_path: Path,
    outdir: Path,
    run_name_prefix: str,
    mode: str,
    results_root: Path | None,
    resources: Resources,
    dry_run: bool,
    overwrite: bool,
) -> list[dict[str, Any]]:
    config = load_campaign_config(config_path)
    expected_n_theta = _expected_n_theta(config)
    shards = build_shards(
        config,
        run_name_prefix=run_name_prefix,
        mode=mode,
    )
    configs_dir = outdir / "configs"
    manifest_path = outdir / "shard_manifest.csv"
    rows: list[dict[str, Any]] = []
    for shard in shards:
        config_out = configs_dir / f"{shard.name}.yaml"
        command = _sbatch_command(
            shard=shard,
            config_path=config_out,
            results_root=results_root,
            resources=resources,
        )
        rows.append(
            {
                "shard_name": shard.name,
                "shard_mode": shard.mode,
                "source_config_path": _repo_relative(config_path),
                "config_path": _repo_relative(config_out),
                "expected_run_root": _expected_run_root(
                    results_root,
                    shard.name,
                ),
                "condition_label": shard.condition_label,
                "m1_sigma_nm": (
                    "" if shard.m1_sigma_nm is None else shard.m1_sigma_nm
                ),
                "m2_sigma_nm": (
                    "" if shard.m2_sigma_nm is None else shard.m2_sigma_nm
                ),
                "draw_start": shard.draw_start,
                "draw_stop": shard.draw_stop,
                "draw_index": (
                    "" if shard.draw_index is None else shard.draw_index
                ),
                "expected_subblocks": shard.expected_subblocks,
                "expected_windows": shard.expected_windows,
                "expected_subblocks_per_window": (
                    shard.expected_subblocks_per_window
                ),
                "expected_n_theta": expected_n_theta,
                "recommended_time": resources.time,
                "recommended_cpus_per_task": resources.cpus_per_task,
                "recommended_mem": resources.mem,
                "recommended_max_workers": resources.max_workers,
                "sbatch_command": command,
            }
        )

    if dry_run:
        for row in rows:
            print(
                f"{row['shard_name']}: {row['expected_subblocks']} subblocks, "
                f"{row['expected_windows']} windows -> {row['config_path']}"
            )
        return rows

    targets = [
        manifest_path,
        outdir / f"submit_{mode}_shards.sh",
        outdir / f"preflight_{mode}_shards.sh",
        outdir / "summarize_shard_status.sh",
        outdir / "README.md",
        *(configs_dir / f"{shard.name}.yaml" for shard in shards),
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite generated files: "
            + ", ".join(str(path) for path in existing[:5])
        )
    configs_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        path = configs_dir / f"{shard.name}.yaml"
        path.write_text(
            yaml.safe_dump(
                shard.config,
                sort_keys=False,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    scripts = {
        outdir / f"submit_{mode}_shards.sh": _submit_script(
            rows=rows,
            results_root=results_root,
        ),
        outdir / f"preflight_{mode}_shards.sh": _preflight_script(
            manifest_path=manifest_path,
            mode=mode,
        ),
        outdir / "summarize_shard_status.sh": _status_script(
            manifest_path=manifest_path,
            results_root=results_root,
        ),
    }
    for path, text in scripts.items():
        path.write_text(text, encoding="utf-8")
        path.chmod(0o755)
    (outdir / "README.md").write_text(
        _readme(
            source_config=config_path,
            mode=mode,
            resources=resources,
            shard_count=len(shards),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(shards)} {mode} shards to {outdir}")
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate full-fidelity iterative campaign shard configs."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--run-name-prefix", required=True)
    parser.add_argument("--mode", choices=("condition", "draw"), default="condition")
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--time")
    parser.add_argument("--cpus-per-task", type=int, default=10)
    parser.add_argument("--mem", default="128G")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cpus_per_task <= 0 or args.max_workers <= 0:
        raise SystemExit("--cpus-per-task and --max-workers must be positive.")
    default_time = "36:00:00" if args.mode == "condition" else "12:00:00"
    prepare_shards(
        config_path=args.config,
        outdir=args.outdir,
        run_name_prefix=args.run_name_prefix,
        mode=args.mode,
        results_root=args.results_root,
        resources=Resources(
            time=args.time or default_time,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            max_workers=args.max_workers,
        ),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
