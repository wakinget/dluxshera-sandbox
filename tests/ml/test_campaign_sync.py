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


def test_staged_import_is_idempotent_and_refuses_all_changes(tmp_path):
    import pytest
    from sync_results import import_staged_results
    config = _config(tmp_path)
    staged = tmp_path / 'staged'
    staged.mkdir()
    (staged / 'metrics.json').write_text('original')
    import_staged_results(staged, config)
    write_sync_manifest(config.local_root / 'sync_manifest.json', config)
    import_staged_results(staged, config)
    (staged / 'run_manifest.json').write_text('{}')
    import_staged_results(staged, config)
    assert (config.local_root / 'run_manifest.json').read_text() == '{}'
    manifest = json.loads((config.local_root / 'sync_manifest.json').read_text())
    import hashlib
    assert manifest['files_sha256']['metrics.json'] == hashlib.sha256(b'original').hexdigest()
    (staged / 'metrics.json').write_text('changed')
    (staged / 'history.csv').write_text('new')
    with pytest.raises(ValueError, match='collision'):
        import_staged_results(staged, config)
    assert (config.local_root / 'metrics.json').read_text() == 'original'
    assert not (config.local_root / 'history.csv').exists()


def test_staged_import_rejects_source_changes(tmp_path):
    import pytest
    from sync_results import import_staged_results
    config = _config(tmp_path)
    write_sync_manifest(config.local_root / 'sync_manifest.json', config)
    staged = tmp_path / 'staged'
    staged.mkdir()
    with pytest.raises(ValueError, match='Source identity'):
        import_staged_results(staged, _config(tmp_path, remote_root='/scratch/new'))


def test_study_cannot_escape_namespace(tmp_path):
    import pytest
    for study in ['..', '../S06', '/S06', '']:
        with pytest.raises(ValueError):
            _config(tmp_path, study=study)


def test_cli_stages_remote_study_and_records_inventory(tmp_path, monkeypatch):
    import sync_results
    commands = []
    def transfer(command, check):
        commands.append(command)
        (Path(command[-1]) / 'run_manifest.json').write_text('{"run_id":"S06-E01-R001"}')
    monkeypatch.setattr(sync_results.subprocess, 'run', transfer)
    local = tmp_path / 'imports'
    assert sync_results.main(['--user', 'user', '--study', 'S06', '--local-root', str(local)]) == 0
    assert commands[0][-2].endswith('/ml/S06/')
    assert (local / 'S06/run_manifest.json').exists()
    assert 'run_manifest.json' in json.loads((local / 'S06/sync_manifest.json').read_text())['files_sha256']
    commands.clear()
    sync_results.main(['--dry-run', '--local-root', str(local)])
    assert not commands
