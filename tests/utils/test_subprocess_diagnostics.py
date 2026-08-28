from __future__ import annotations

import json
import fcntl
import os
import sys
import time
from pathlib import Path

import pytest

from dluxshera.utils import subprocess_diagnostics as sd
from dluxshera.utils.subprocess_diagnostics import (
    ResourceTimeUnavailableError,
    classify_return_code,
    detect_resource_time_command,
    run_subprocess_with_diagnostics,
    subprocess_succeeded,
    subprocess_timed_out,
)


def _wait_for_lock_release(lock_path: Path, *, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if lock_path.exists():
            with lock_path.open("a+", encoding="utf-8") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)
                    return True
        time.sleep(0.02)
    return False


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


def test_subprocess_timeout_terminates_child_and_writes_diagnostics(tmp_path: Path) -> None:
    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "import time; time.sleep(5)"],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time=False,
        subprocess_timeout_s=0.2,
        timeout_terminate_grace_s=0.2,
    )

    assert payload.failure_class == "timeout"
    assert payload.return_code != 0
    raw = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert raw["timeout"]["requested"] is True
    assert raw["timeout"]["timed_out"] is True
    assert raw["timeout"]["timeout_s"] == 0.2
    assert raw["failure_class"] == "timeout"


def test_subprocess_timeout_preserves_expected_summary_marker(tmp_path: Path) -> None:
    summary_path = tmp_path / "schur_summary" / "subblock_summary.json"
    script = (
        "from pathlib import Path; import json, time; "
        f"p=Path({str(summary_path)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        "p.write_text(json.dumps({'ok': True}), encoding='utf-8'); "
        "time.sleep(5)"
    )

    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", script],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time=False,
        subprocess_timeout_s=0.2,
        timeout_terminate_grace_s=0.2,
        expected_summary_path=summary_path,
    )

    assert payload.failure_class == "timeout"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {"ok": True}
    raw = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    expected = raw["timeout"]["expected_summary"]
    assert expected["exists"] is True
    assert expected["json_readable"] is True


def test_subprocess_timeout_kills_descendant_process_group(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("process-group descendant termination is POSIX-specific")
    pid_path = tmp_path / "grandchild.pid"
    script = (
        "import subprocess, sys, time; "
        "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"open({str(pid_path)!r}, 'w', encoding='utf-8').write(str(p.pid)); "
        "time.sleep(30)"
    )

    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", script],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time=False,
        subprocess_timeout_s=0.5,
        timeout_terminate_grace_s=0.2,
    )

    assert payload.failure_class == "timeout"
    assert pid_path.exists()
    grandchild_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.time() + 2.0
    while time.time() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild process still exists after timeout: {grandchild_pid}")


def test_subprocess_timeout_child_sigterm_handler_exits_zero_is_failure(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("SIGTERM handler timeout behavior is POSIX-specific")
    script = """
import signal
import time

def exit_zero(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, exit_zero)
while True:
    time.sleep(1)
"""

    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", script],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time=False,
        subprocess_timeout_s=0.2,
        timeout_terminate_grace_s=0.5,
    )

    assert payload.return_code == 0
    assert payload.timed_out is True
    assert subprocess_timed_out(payload) is True
    assert payload.failure_class == "timeout"
    assert payload.succeeded is False
    assert subprocess_succeeded(payload) is False


def test_subprocess_timeout_wrapper_child_topology_terminates_with_sigterm(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("process-group wrapper termination is POSIX-specific")
    child_pid_path = tmp_path / "wrapped_child.pid"
    child_lock_path = tmp_path / "wrapped_child.lock"
    script = """
import subprocess
import sys
import os
import time

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import fcntl, os, sys, time; handle=open(sys.argv[1], 'w', encoding='utf-8'); fcntl.flock(handle, fcntl.LOCK_EX); open(sys.argv[2], 'w', encoding='utf-8').write(str(os.getpid())); time.sleep(60)",
    sys.argv[2],
    sys.argv[1],
])
deadline = time.monotonic() + 5.0
while not os.path.exists(sys.argv[1]) and time.monotonic() < deadline:
    time.sleep(0.01)
raise SystemExit(child.wait())
"""

    payload = run_subprocess_with_diagnostics(
        command=[
            sys.executable,
            "-c",
            script,
            str(child_pid_path),
            str(child_lock_path),
        ],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time=False,
        subprocess_timeout_s=0.2,
        timeout_terminate_grace_s=0.5,
    )

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    assert payload.failure_class == "timeout"
    assert payload.timed_out is True
    assert payload.timeout["termination"]["force_kill_used"] is False
    assert payload.timeout["termination"]["group_gone_after_sigterm"] is True
    assert _wait_for_lock_release(child_lock_path)


def test_timeout_writes_diagnostics_when_resource_time_output_is_malformed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("timeout resource-time wrapper behavior is POSIX-specific")
    fake_time = tmp_path / "fake_time.py"
    fake_time.write_text(
        """
import os
import signal
import subprocess
import sys
import time

output_path = sys.argv[3]
child = subprocess.Popen(sys.argv[4:])

def handle_sigterm(signum, frame):
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("Maximum resident set size (kbytes): not-an-int\\n")
    try:
        child.terminate()
    except Exception:
        pass
    raise SystemExit(0)

signal.signal(signal.SIGTERM, handle_sigterm)
raise SystemExit(child.wait())
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sd,
        "detect_resource_time_command",
        lambda: [sys.executable, str(fake_time), "-v"],
    )

    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "import time; time.sleep(60)"],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time="auto",
        subprocess_timeout_s=0.2,
        timeout_terminate_grace_s=0.5,
    )

    raw = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert payload.failure_class == "timeout"
    assert payload.timed_out is True
    assert raw["failure_class"] == "timeout"
    assert raw["timeout"]["timed_out"] is True
    assert raw["resource_time"]["resource_time_parse_status"] == "error"
    assert "Unable to parse external resource-time output" in raw["resource_time"][
        "resource_time_warning"
    ]


def test_timeout_escalates_when_group_leader_exits_but_descendant_ignores_sigterm(
    tmp_path: Path,
) -> None:
    if os.name != "posix":
        pytest.skip("process-group SIGKILL escalation is POSIX-specific")
    child_pid_path = tmp_path / "stubborn_child.pid"
    child_lock_path = tmp_path / "stubborn_child.lock"
    script = """
import signal
import subprocess
import sys
import time
import os

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import fcntl, os, signal, sys, time; handle=open(sys.argv[1], 'w', encoding='utf-8'); fcntl.flock(handle, fcntl.LOCK_EX); open(sys.argv[2], 'w', encoding='utf-8').write(str(os.getpid())); signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
    sys.argv[2],
    sys.argv[1],
])
deadline = time.monotonic() + 5.0
while not os.path.exists(sys.argv[1]) and time.monotonic() < deadline:
    time.sleep(0.01)

def exit_zero(signum, frame):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, exit_zero)
while True:
    time.sleep(1)
"""

    payload = run_subprocess_with_diagnostics(
        command=[
            sys.executable,
            "-c",
            script,
            str(child_pid_path),
            str(child_lock_path),
        ],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time=False,
        subprocess_timeout_s=0.3,
        timeout_terminate_grace_s=0.2,
    )

    child_pid = int(child_pid_path.read_text(encoding="utf-8"))
    actions = payload.timeout["termination"]["actions"]
    assert payload.failure_class == "timeout"
    assert payload.timed_out is True
    assert payload.succeeded is False
    assert payload.timeout["termination"]["force_kill_used"] is True
    assert payload.timeout["termination"]["group_gone_after_sigterm"] is False
    assert any(action["action"] == "sigkill_process_group" for action in actions)
    assert _wait_for_lock_release(child_lock_path)


def test_subprocess_timeout_none_preserves_normal_wait_behavior(tmp_path: Path) -> None:
    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "import time; time.sleep(0.05); print('ok')"],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        resource_time=False,
        subprocess_timeout_s=None,
    )

    assert payload.return_code == 0
    raw = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    assert raw["timeout"]["requested"] is False
    assert raw["timeout"]["timed_out"] is False


def test_subprocess_timeout_stops_memory_sampler_and_finalizes_logs(tmp_path: Path) -> None:
    payload = run_subprocess_with_diagnostics(
        command=[sys.executable, "-c", "import sys, time; print('tail', file=sys.stderr); time.sleep(5)"],
        cwd=tmp_path,
        env=None,
        stdout_log=tmp_path / "o.log",
        stderr_log=tmp_path / "e.log",
        diagnostics_json=tmp_path / "d.json",
        memory_diagnostics=True,
        resource_time=False,
        sample_interval_s=0.05,
        subprocess_timeout_s=0.2,
        timeout_terminate_grace_s=0.2,
    )

    assert payload.failure_class == "timeout"
    raw = json.loads((tmp_path / "d.json").read_text(encoding="utf-8"))
    samples = Path(raw["memory_sampler"]["samples_jsonl"])
    assert samples.exists()
    assert raw["stderr_tail"]
