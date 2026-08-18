from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dluxshera.utils import subprocess_diagnostics as sd
from dluxshera.utils.subprocess_diagnostics import (
    ResourceTimeUnavailableError,
    classify_return_code,
    detect_resource_time_command,
    run_subprocess_with_diagnostics,
)


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


def test_detect_resource_time_command_accepts_gnu_time_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sd, "_time_candidates", lambda: ["/usr/bin/time"])

    def fake_run(command, **kwargs):
        out_path = Path(command[3])
        out_path.write_text("Maximum resident set size (kbytes): 10\n", encoding="utf-8")
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(sd.subprocess, "run", fake_run)
    assert detect_resource_time_command() == ["/usr/bin/time", "-v"]


def test_detect_resource_time_command_returns_none_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sd, "_time_candidates", lambda: [])
    assert detect_resource_time_command() is None


def test_detect_resource_time_command_rejects_incompatible_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sd, "_time_candidates", lambda: ["/usr/bin/time"])
    monkeypatch.setattr(
        sd.subprocess,
        "run",
        lambda *args, **kwargs: type("Completed", (), {"returncode": 1})(),
    )
    assert detect_resource_time_command() is None


def test_resource_time_auto_unavailable_falls_back_without_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sd, "detect_resource_time_command", lambda: None)
    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time="auto",
    )
    assert payload.return_code == 0
    raw = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert raw["resource_time"]["resource_time_mode_requested"] == "auto"
    assert raw["resource_time"]["resource_time_mode_effective"] == "disabled"
    assert raw["resource_time"]["enabled"] is False
    assert raw["resource_time"]["reason"] == "external_time_unavailable_or_incompatible"


def test_resource_time_enabled_unavailable_fails_clearly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sd, "detect_resource_time_command", lambda: None)
    with pytest.raises(ResourceTimeUnavailableError, match="External resource timing was required"):
        run_subprocess_with_diagnostics(
            command=[sys.executable, "-c", "print('ok')"],
            cwd=tmp_path,
            env=None,
            stdout_log=tmp_path / "o.log",
            stderr_log=tmp_path / "e.log",
            diagnostics_json=tmp_path / "d.json",
            resource_time="enabled",
        )


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
