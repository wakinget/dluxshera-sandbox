"""Helpers for subprocess execution with optional resource diagnostics."""
from __future__ import annotations
import json, os, shutil, signal, subprocess, sys, tempfile, threading, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, Literal

SCHEMA_VERSION = "subprocess_resource_diagnostics.v1"
ResourceTimeMode = Literal["auto", "enabled", "disabled", "gnu", "portable"]


class ResourceTimeUnavailableError(RuntimeError):
    """Raised when strict external resource timing is requested but unavailable."""

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_rss_mb(pid: int) -> float | None:
    p = Path(f"/proc/{pid}/status")
    if not p.exists():
        return None
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return float(parts[1]) / 1024.0 if len(parts) >= 2 else None
    except OSError:
        return None
    return None

def _children_of(pid:int)->list[int]:
    ch=Path(f"/proc/{pid}/task/{pid}/children")
    if not ch.exists():
        return []
    try:
        txt=ch.read_text(encoding='utf-8').strip()
        return [int(x) for x in txt.split() if x.strip()]
    except Exception:
        return []

def _descendant_tree_pids(root_pid:int)->set[int]:
    out:set[int]=set()
    stack=[root_pid]
    while stack:
        cur=stack.pop()
        for c in _children_of(cur):
            if c not in out:
                out.add(c); stack.append(c)
    return out

def classify_return_code(return_code: int) -> tuple[str | None, str | None]:
    if return_code == -signal.SIGKILL:
        return "probable_sigkill", "Process ended with SIGKILL (-9); possible memory pressure."
    if return_code < 0:
        return "signal_termination", f"Process ended by signal {-return_code}."
    if return_code > 0:
        return "nonzero_exit", "Process exited with nonzero status."
    return None, None

@dataclass(frozen=True)
class SubprocessDiagnostics:
    schema_version: str; created_at: str; command: list[str]; cwd: str; stdout_log: str; stderr_log: str
    started_at: str; finished_at: str; elapsed_seconds: float; return_code: int; failure_class: str | None; failure_hint: str | None
    resource_time: dict[str, Any]; memory_sampler: dict[str, Any]
    last_stderr_line: str | None; stderr_tail: list[str]

def stderr_tail(path: Path, *, max_lines: int = 10) -> list[str]:
    """Return the last non-empty child stderr lines from one subprocess log."""

    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return []
    return lines[-max_lines:]

def _resolve_resource_time_mode(resource_time: bool | str | None) -> tuple[bool, ResourceTimeMode]:
    if resource_time is None:
        return True, "auto"
    if isinstance(resource_time, bool):
        return resource_time, ("enabled" if resource_time else "disabled")
    mode = str(resource_time).strip().lower()
    if mode not in {"auto", "enabled", "gnu", "portable", "disabled"}:
        raise ValueError(
            "resource_time must be bool or one of: auto, enabled, gnu, portable, disabled."
        )
    return mode != "disabled", mode  # type: ignore[return-value]


def _time_candidates() -> list[str]:
    candidates: list[str] = []
    if Path("/usr/bin/time").exists():
        candidates.append("/usr/bin/time")
    path_time = shutil.which("time")
    if path_time is not None and path_time not in candidates:
        candidates.append(path_time)
    return candidates


def detect_resource_time_command() -> list[str] | None:
    """
    Return a GNU-compatible external time prefix, or None.

    The returned command intentionally uses only implementations that pass a
    probe for the flags used by this helper. BSD/macOS ``time`` usually exists
    but rejects ``-v``; that must be treated as unavailable.
    """

    for candidate in _time_candidates():
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                out_path = Path(tmpdir) / "time.stderr.log"
                probe = subprocess.run(
                    [
                        candidate,
                        "-v",
                        "-o",
                        str(out_path),
                        sys.executable,
                        "-c",
                        "pass",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if probe.returncode == 0 and out_path.exists():
                    return [candidate, "-v"]
        except Exception:
            continue
    return None


def _build_resource_time_prefix(
    *,
    requested_enabled: bool,
    requested_mode: ResourceTimeMode,
    time_stderr_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "mode": requested_mode,
        "enabled": False,
        "reason": "disabled" if not requested_enabled else None,
        "command": None,
        "resource_time_requested": requested_enabled,
        "resource_time_mode_requested": requested_mode,
        "resource_time_mode_effective": "disabled",
        "resource_time_available": False,
        "resource_time_command": None,
        "resource_time_parse_status": "disabled",
        "resource_time_warning": None,
    }
    if not requested_enabled:
        return [], metadata

    detected = detect_resource_time_command()
    if detected is not None:
        cmd = [*detected, "-o", str(time_stderr_path)]
        metadata.update(
            {
                "enabled": True,
                "reason": None,
                "command": cmd,
                "resource_time_mode_effective": "gnu",
                "resource_time_available": True,
                "resource_time_command": cmd,
                "resource_time_parse_status": "pending",
            }
        )
        return cmd, metadata

    reason = "external_time_unavailable_or_incompatible"
    if requested_mode in {"enabled", "gnu"}:
        raise ResourceTimeUnavailableError(
            "External resource timing was required, but no GNU-compatible "
            "`time -v -o` command is available. Re-run with --no-resource-time "
            "or omit --resource-time to use auto fallback."
        )

    metadata.update(
        {
            "reason": reason,
            "resource_time_warning": (
                "No GNU-compatible external `time -v -o` binary found; only parent elapsed timing recorded."
            ),
        }
    )
    return [], metadata


def require_resource_time_available(resource_time: bool | str | None) -> None:
    """Fail before campaign fan-out when strict external timing was requested."""

    requested_enabled, requested_mode = _resolve_resource_time_mode(resource_time)
    if requested_enabled and requested_mode in {"enabled", "gnu"}:
        if detect_resource_time_command() is None:
            raise ResourceTimeUnavailableError(
                "External resource timing was required, but no GNU-compatible "
                "`time -v -o` command is available. Re-run with --no-resource-time "
                "or omit --resource-time to use auto fallback."
            )


def run_subprocess_with_diagnostics(*,command: Sequence[str],cwd: Path,env: dict[str, str] | None,stdout_log: Path,stderr_log: Path,diagnostics_json: Path,memory_diagnostics: bool = False,resource_time: bool | str | None = None,sample_interval_s: float = 0.25) -> SubprocessDiagnostics:
    stdout_log.parent.mkdir(parents=True, exist_ok=True); stderr_log.parent.mkdir(parents=True, exist_ok=True); diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    requested_enabled, requested_mode = _resolve_resource_time_mode(resource_time)
    time_stderr_path = stderr_log.with_name(f"{stderr_log.stem}.time.stderr.log")
    prefix, rt_meta = _build_resource_time_prefix(
        requested_enabled=requested_enabled,
        requested_mode=requested_mode,
        time_stderr_path=time_stderr_path,
    )
    final_command = [*prefix, *list(command)]
    started_at = _now_iso(); started_perf = time.perf_counter(); stop_sampling = threading.Event(); samples=[]; descendants_seen=set(); notes=[]
    with stdout_log.open("w", encoding="utf-8") as out, stderr_log.open("w", encoding="utf-8") as err:
        process = subprocess.Popen(final_command,cwd=str(cwd),env=env,text=True,stdout=out,stderr=err)
        def _sampler()->None:
            while not stop_sampling.is_set():
                direct=_read_rss_mb(process.pid)
                tree=None
                try:
                    pids=_descendant_tree_pids(process.pid)
                    descendants_seen.update(pids)
                    vals=[_read_rss_mb(pid) for pid in pids|{process.pid}]
                    vals=[v for v in vals if v is not None]
                    tree=float(sum(vals)) if vals else None
                except Exception as exc:
                    notes.append(f"descendant_tree_unavailable:{exc}")
                samples.append({"ts":_now_iso(),"direct_child_rss_mb":direct,"descendant_tree_rss_mb":tree})
                time.sleep(sample_interval_s)
        t=None
        if memory_diagnostics:
            t=threading.Thread(target=_sampler,daemon=True); t.start()
        return_code = process.wait(); stop_sampling.set();
        if t is not None: t.join(timeout=1.0)
    finished_at=_now_iso(); elapsed_seconds=time.perf_counter()-started_perf
    failure_class,failure_hint=classify_return_code(return_code)
    samples_jsonl_path=None; peak_direct=peak_tree=None
    if memory_diagnostics:
        sp=diagnostics_json.with_suffix('.samples.jsonl')
        with sp.open('w',encoding='utf-8') as h:
            for r in samples: h.write(json.dumps(r)+"\n")
        samples_jsonl_path=str(sp)
        direct=[r['direct_child_rss_mb'] for r in samples if r.get('direct_child_rss_mb') is not None]
        tree=[r['descendant_tree_rss_mb'] for r in samples if r.get('descendant_tree_rss_mb') is not None]
        peak_direct=max(direct) if direct else None; peak_tree=max(tree) if tree else None
    resource_block={
        "enabled":bool(rt_meta["enabled"]),
        "mode_requested":requested_mode,
        "mode_effective":rt_meta["resource_time_mode_effective"],
        "available":bool(rt_meta["resource_time_available"]),
        "maximum_resident_set_kb":None,
        "maximum_resident_set_mb":None,
        "elapsed_wall_clock":None,
        "raw_stderr_path":str(time_stderr_path) if rt_meta["resource_time_available"] else None,
        "resource_time_requested": rt_meta["resource_time_requested"],
        "resource_time_mode_requested": rt_meta["resource_time_mode_requested"],
        "resource_time_mode_effective": rt_meta["resource_time_mode_effective"],
        "resource_time_available": rt_meta["resource_time_available"],
        "resource_time_command": rt_meta["resource_time_command"],
        "resource_time_parse_status": rt_meta["resource_time_parse_status"],
        "resource_time_warning": rt_meta["resource_time_warning"],
        "mode": rt_meta["mode"],
        "reason": rt_meta["reason"],
        "command": rt_meta["command"],
    }
    if rt_meta["resource_time_available"] and time_stderr_path.exists():
        txt=time_stderr_path.read_text(encoding='utf-8')
        if rt_meta["resource_time_mode_effective"] == "gnu":
            for line in txt.splitlines():
                if "Maximum resident set size" in line:
                    kb=int(line.split(':',1)[1].strip()); resource_block['maximum_resident_set_kb']=kb; resource_block['maximum_resident_set_mb']=float(kb)/1024.0
                if "Elapsed (wall clock) time" in line:
                    resource_block['elapsed_wall_clock']=line.split(':',1)[1].strip()
            resource_block["resource_time_parse_status"] = "ok"
        else:
            resource_block["resource_time_parse_status"] = "not_supported_for_mode"
    elif rt_meta["resource_time_available"]:
        resource_block["resource_time_parse_status"] = "missing_output"
    peak_source="resource_time" if resource_block.get('maximum_resident_set_mb') is not None else "memory_sampler"
    child_stderr_tail = stderr_tail(stderr_log)
    payload=SubprocessDiagnostics(schema_version=SCHEMA_VERSION,created_at=_now_iso(),command=list(command),cwd=str(cwd),stdout_log=str(stdout_log),stderr_log=str(stderr_log),started_at=started_at,finished_at=finished_at,elapsed_seconds=float(elapsed_seconds),return_code=int(return_code),failure_class=failure_class,failure_hint=failure_hint,resource_time=resource_block,memory_sampler={"enabled":bool(memory_diagnostics),"peak_direct_child_rss_mb":peak_direct,"peak_descendant_tree_rss_mb":peak_tree,"descendant_pids_seen":sorted(descendants_seen),"sampling_notes":notes,"samples_jsonl":samples_jsonl_path,"peak_total_rss_mb":peak_tree},last_stderr_line=(child_stderr_tail[-1] if child_stderr_tail else None),stderr_tail=child_stderr_tail)
    d=asdict(payload); d['peak_rss_source']=peak_source
    diagnostics_json.write_text(json.dumps(d, indent=2), encoding='utf-8')
    return payload
