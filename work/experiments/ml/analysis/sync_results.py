from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

__all__ = [
    "DEFAULT_COMPACT_ARTIFACTS",
    "SyncConfig",
    "build_rsync_command",
    "effective_local_root",
    "write_sync_manifest",
]

DEFAULT_COMPACT_ARTIFACTS = (
    "run_manifest.json",
    "run_config_resolved.json",
    "history.csv",
    "metrics.json",
    "evaluation_predictions.npz",
    "source_snapshot.txt",
)

CHECKPOINT_ARTIFACTS = ("checkpoint_best.pt", "checkpoint_last.pt")
LOG_ARTIFACTS = ("*.out", "*.err", "slurm-*.out", "slurm-*.err", "*.log")


@dataclass(frozen=True)
class SyncConfig:
    """Resolved rsync configuration for compact ML result imports."""

    site: str
    host: str
    user: str | None
    remote_root: str
    local_root: Path
    include_checkpoints: bool = False
    include_logs: bool = False
    study: str | None = None


def build_rsync_command(config: SyncConfig, *, dry_run: bool = False) -> list[str]:
    """Build the conservative compact-artifact rsync command."""

    source = _remote_source(config)
    destination = str(effective_local_root(config))
    command = ["rsync", "-av", "--prune-empty-dirs"]
    if dry_run:
        command.append("--dry-run")
    for pattern in _include_patterns(config):
        command.extend(["--include", pattern])
    command.extend(["--exclude", "*", source, destination])
    return command


def write_sync_manifest(
    path: Path,
    config: SyncConfig,
    *,
    artifact_policy: Sequence[str] | None = None,
) -> dict[str, object]:
    """Write local sync provenance after a successful non-dry-run import."""

    payload: dict[str, object] = {
        "schema_version": "dluxshera_ml_sync_manifest/1",
        "site": config.site,
        "host": config.host,
        "remote_root": config.remote_root,
        "local_root": str(effective_local_root(config)),
        "synced_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifact_policy": list(artifact_policy or DEFAULT_COMPACT_ARTIFACTS),
        "include_checkpoints": bool(config.include_checkpoints),
        "include_logs": bool(config.include_logs),
    }
    if config.study is not None:
        payload["study"] = config.study
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize compact dLuxShera ML analysis artifacts from a remote result tree."
    )
    parser.add_argument("--site", default="tacc_ls6", choices=["tacc_ls6"])
    parser.add_argument("--user", default=None, help="Remote username; defaults to current local user.")
    parser.add_argument("--host", default=None, help="Remote SSH host override.")
    parser.add_argument("--remote-root", default=None, help="Remote ML result root override.")
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("Results/hpc_imports/ml/ls6"),
        help="Local compact mirror root.",
    )
    parser.add_argument("--study", default=None, help="Optional study subdirectory such as S01 or S05.")
    parser.add_argument("--include-checkpoints", action="store_true")
    parser.add_argument("--include-logs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    user = args.user or getpass.getuser()
    host = args.host or "ls6.tacc.utexas.edu"
    remote_root = args.remote_root or f"/work/11689/{user}/ls6/dLuxShera-Results/ml/"
    config = SyncConfig(
        site=args.site,
        host=host,
        user=user,
        remote_root=remote_root,
        local_root=args.local_root,
        include_checkpoints=args.include_checkpoints,
        include_logs=args.include_logs,
        study=args.study,
    )
    command = build_rsync_command(config, dry_run=args.dry_run)
    print(" ".join(command))
    if args.dry_run:
        return 0
    destination = effective_local_root(config)
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, check=True)
    write_sync_manifest(destination / "sync_manifest.json", config)
    return 0


def effective_local_root(config: SyncConfig) -> Path:
    """Return the actual local rsync destination for this sync."""

    if config.study:
        return config.local_root / config.study.strip("/")
    return config.local_root


def _include_patterns(config: SyncConfig) -> list[str]:
    patterns = ["*/", *DEFAULT_COMPACT_ARTIFACTS]
    if config.include_checkpoints:
        patterns.extend(CHECKPOINT_ARTIFACTS)
    if config.include_logs:
        patterns.extend(LOG_ARTIFACTS)
    return patterns


def _remote_source(config: SyncConfig) -> str:
    root = config.remote_root.rstrip("/")
    if config.study:
        root = f"{root}/{config.study.strip('/')}/"
    else:
        root = f"{root}/"
    return f"{config.user}@{config.host}:{root}" if config.user else f"{config.host}:{root}"


if __name__ == "__main__":
    raise SystemExit(main())
