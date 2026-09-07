from __future__ import annotations

import json
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[2] / "work" / "experiments" / "ml" / "analysis"
sys.path.insert(0, str(ANALYSIS_DIR))

from sync_results import (  # noqa: E402
    DEFAULT_COMPACT_ARTIFACTS,
    SyncConfig,
    build_rsync_command,
    effective_local_root,
    write_sync_manifest,
)


def _config(tmp_path: Path, **kwargs) -> SyncConfig:
    values = {
        "site": "tacc_ls6",
        "host": "ls6.tacc.utexas.edu",
        "user": "user",
        "remote_root": "/work/11689/user/ls6/dLuxShera-Results/ml/",
        "local_root": tmp_path / "Results" / "hpc_imports" / "ml" / "ls6",
    }
    values.update(kwargs)
    return SyncConfig(**values)


def test_rsync_command_uses_compact_allowlist_without_delete_or_checkpoints(tmp_path: Path) -> None:
    command = build_rsync_command(_config(tmp_path))

    assert command[:3] == ["rsync", "-av", "--prune-empty-dirs"]
    assert "--delete" not in command
    for artifact in DEFAULT_COMPACT_ARTIFACTS:
        assert artifact in command
    assert "checkpoint_best.pt" not in command
    assert "checkpoint_last.pt" not in command
    assert command[-2].startswith("user@ls6.tacc.utexas.edu:/work/11689/user/ls6/")
    assert command[-1].endswith("/ls6")


def test_rsync_command_supports_checkpoints_logs_study_and_dry_run(tmp_path: Path) -> None:
    config = _config(tmp_path, include_checkpoints=True, include_logs=True, study="S05")
    command = build_rsync_command(
        config,
        dry_run=True,
    )

    assert "--dry-run" in command
    assert "checkpoint_best.pt" in command
    assert "checkpoint_last.pt" in command
    assert "*.out" in command
    assert command[-2].endswith("/S05/")
    assert command[-1].endswith("/ls6/S05")
    assert effective_local_root(config) == config.local_root / "S05"


def test_sync_manifest_serialization_omits_credentials_and_records_policy(tmp_path: Path) -> None:
    config = _config(tmp_path, include_checkpoints=True)
    manifest_path = config.local_root / "sync_manifest.json"
    payload = write_sync_manifest(manifest_path, config)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded == payload
    assert loaded["schema_version"] == "dluxshera_ml_sync_manifest/1"
    assert loaded["site"] == "tacc_ls6"
    assert loaded["host"] == "ls6.tacc.utexas.edu"
    assert loaded["remote_root"] == "/work/11689/user/ls6/dLuxShera-Results/ml/"
    assert loaded["include_checkpoints"] is True
    assert loaded["include_logs"] is False
    assert "user@" not in json.dumps(loaded)
