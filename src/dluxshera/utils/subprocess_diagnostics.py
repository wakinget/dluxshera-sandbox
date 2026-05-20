"""Helpers for subprocess execution with optional resource diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "subprocess_resource_diagnostics.v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_rss_mb(pid: int) -> float | None:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / 1024.0
    except OSError:
        return None
    return None


def classify_return_code(return_code: int) -> tuple[str | None, str | None]:
    """Classify subprocess return codes for diagnostics-only evidence."""
    if return_code == -signal.SIGKILL:
        return "probable_sigkill", "Process ended with SIGKILL (-9); possible memory pressure."
    if return_code < 0:
        return "signal_termination", f"Process ended by signal {-return_code}."
    if return_code > 0:
        return "nonzero_exit", "Process exited with nonzero status."
    return None, None


@dataclass(frozen=True)
class SubprocessDiagnostics:
    schema_version: str
    created_at: str
    command: list[str]
    cwd: str
    stdout_log: str
    stderr_log: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    return_code: int
    failure_class: str | None
    failure_hint: str | None
    resource_time: dict[str, Any]
    memory_sampler: dict[str, Any]


def run_subprocess_with_diagnostics(
    *,
    command: Sequence[str],
    cwd: Path,
    env: dict[str, str] | None,
    stdout_log: Path,
    stderr_log: Path,
    diagnostics_json: Path,
    memory_diagnostics: bool = False,
    resource_time: bool = True,
    sample_interval_s: float = 0.25,
) -> SubprocessDiagnostics:
    """Run subprocess and write stable diagnostics JSON.

    Parameters
    ----------
    command
        Child command tokens.
    cwd
        Working directory for subprocess.
    """
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    time_available = shutil.which("/usr/bin/time") is not None
    time_stderr_path = stderr_log.with_name(f"{stderr_log.stem}.time.stderr.log")

    final_command = list(command)
    if resource_time and time_available:
        final_command = ["/usr/bin/time", "-v", "-o", str(time_stderr_path), *final_command]

    started_at = _now_iso()
    started_perf = time.perf_counter()
    stop_sampling = threading.Event()
    samples: list[dict[str, Any]] = []

    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(
            final_command,
            cwd=str(cwd),
            env=env,
            text=True,
            stdout=out,
            stderr=err,
        )

        def _sampler() -> None:
            while not stop_sampling.is_set():
                parent_rss = _read_rss_mb(os.getpid())
                child_rss = _read_rss_mb(process.pid)
                total = None
                if parent_rss is not None or child_rss is not None:
                    total = float((parent_rss or 0.0) + (child_rss or 0.0))
                samples.append({"ts": _now_iso(), "parent_rss_mb": parent_rss, "child_rss_mb": child_rss, "total_rss_mb": total})
                time.sleep(sample_interval_s)

        sampler_thread: threading.Thread | None = None
        if memory_diagnostics:
            sampler_thread = threading.Thread(target=_sampler, daemon=True)
            sampler_thread.start()

        return_code = process.wait()
        stop_sampling.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=1.0)

    finished_at = _now_iso()
    elapsed_seconds = time.perf_counter() - started_perf
    failure_class, failure_hint = classify_return_code(return_code)

    samples_jsonl_path: str | None = None
    peak_parent = peak_child = peak_total = None
    if memory_diagnostics:
        samples_path = diagnostics_json.with_suffix(".samples.jsonl")
        with samples_path.open("w", encoding="utf-8") as handle:
            for row in samples:
                handle.write(json.dumps(row) + "\n")
        samples_jsonl_path = str(samples_path)
        parent_values = [r["parent_rss_mb"] for r in samples if r["parent_rss_mb"] is not None]
        child_values = [r["child_rss_mb"] for r in samples if r["child_rss_mb"] is not None]
        total_values = [r["total_rss_mb"] for r in samples if r["total_rss_mb"] is not None]
        peak_parent = max(parent_values) if parent_values else None
        peak_child = max(child_values) if child_values else None
        peak_total = max(total_values) if total_values else None

    resource_block: dict[str, Any] = {"enabled": bool(resource_time), "available": bool(time_available), "maximum_resident_set_kb": None, "elapsed_wall_clock": None, "raw_stderr_path": str(time_stderr_path) if resource_time and time_available else None}
    if resource_time and time_available and time_stderr_path.exists():
        text = time_stderr_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if "Maximum resident set size" in line:
                resource_block["maximum_resident_set_kb"] = int(line.split(":", 1)[1].strip())
            if "Elapsed (wall clock) time" in line:
                resource_block["elapsed_wall_clock"] = line.split(":", 1)[1].strip()

    payload = SubprocessDiagnostics(
        schema_version=SCHEMA_VERSION,
        created_at=_now_iso(),
        command=list(command),
        cwd=str(cwd),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=float(elapsed_seconds),
        return_code=int(return_code),
        failure_class=failure_class,
        failure_hint=failure_hint,
        resource_time=resource_block,
        memory_sampler={"enabled": bool(memory_diagnostics), "peak_parent_rss_mb": peak_parent, "peak_child_rss_mb": peak_child, "peak_total_rss_mb": peak_total, "samples_jsonl": samples_jsonl_path},
    )
    diagnostics_json.write_text(json.dumps(asdict(payload), indent=2), encoding="utf-8")
    return payload
