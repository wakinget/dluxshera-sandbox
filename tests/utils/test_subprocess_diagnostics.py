from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from dluxshera.utils.subprocess_diagnostics import classify_return_code, run_subprocess_with_diagnostics


def test_run_subprocess_with_diagnostics_success(tmp_path: Path) -> None:
    out = tmp_path / "out.log"
    err = tmp_path / "err.log"
    diag = tmp_path / "diag.json"
    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "import sys; print('ok'); print('warn', file=sys.stderr)"],
        cwd=tmp_path,
        env=None,
        stdout_log=out,
        stderr_log=err,
        diagnostics_json=diag,
        memory_diagnostics=True,
        resource_time=False,
    )
    assert payload.return_code == 0
    raw = json.loads(diag.read_text(encoding="utf-8"))
    assert raw["command"][0] == sys.executable
    assert Path(raw["stdout_log"]).exists()
    assert Path(raw["memory_sampler"]["samples_jsonl"]).exists()


def test_run_subprocess_with_diagnostics_nonzero(tmp_path: Path) -> None:
    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "import sys; sys.exit(3)"],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        memory_diagnostics=False,
        resource_time=False,
    )
    assert payload.return_code == 3
    assert payload.failure_class == "nonzero_exit"


def test_classify_return_code_probable_sigkill() -> None:
    klass, hint = classify_return_code(-9)
    assert klass == "probable_sigkill"
    assert hint is not None


def test_resource_time_gnu_requested_but_unavailable_does_not_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_which = shutil.which

    def fake_which(name: str) -> str | None:
        if name == "/usr/bin/time":
            return None
        return original_which(name)

    monkeypatch.setattr(shutil, "which", fake_which)
    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time="gnu",
    )
    assert payload.return_code == 0
    raw = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert raw["resource_time"]["resource_time_mode_requested"] == "gnu"
    assert raw["resource_time"]["resource_time_mode_effective"] == "disabled"


def test_resource_time_disabled_mode_records_effective_disabled(tmp_path: Path) -> None:
    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time="disabled",
    )
    assert payload.return_code == 0
    raw = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert raw["resource_time"]["resource_time_mode_effective"] == "disabled"
