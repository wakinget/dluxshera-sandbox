"""Helpers for subprocess execution with optional resource diagnostics."""
from __future__ import annotations
import json, os, shutil, signal, subprocess, threading, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA_VERSION = "subprocess_resource_diagnostics.v1"

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

def run_subprocess_with_diagnostics(*,command: Sequence[str],cwd: Path,env: dict[str, str] | None,stdout_log: Path,stderr_log: Path,diagnostics_json: Path,memory_diagnostics: bool = False,resource_time: bool = True,sample_interval_s: float = 0.25) -> SubprocessDiagnostics:
    stdout_log.parent.mkdir(parents=True, exist_ok=True); stderr_log.parent.mkdir(parents=True, exist_ok=True); diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
    time_available = shutil.which("/usr/bin/time") is not None
    time_stderr_path = stderr_log.with_name(f"{stderr_log.stem}.time.stderr.log")
    final_command = ["/usr/bin/time", "-v", "-o", str(time_stderr_path), *list(command)] if resource_time and time_available else list(command)
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
    resource_block={"enabled":bool(resource_time),"available":bool(time_available),"maximum_resident_set_kb":None,"maximum_resident_set_mb":None,"elapsed_wall_clock":None,"raw_stderr_path":str(time_stderr_path) if resource_time and time_available else None}
    if resource_time and time_available and time_stderr_path.exists():
        txt=time_stderr_path.read_text(encoding='utf-8')
        for line in txt.splitlines():
            if "Maximum resident set size" in line:
                kb=int(line.split(':',1)[1].strip()); resource_block['maximum_resident_set_kb']=kb; resource_block['maximum_resident_set_mb']=float(kb)/1024.0
            if "Elapsed (wall clock) time" in line:
                resource_block['elapsed_wall_clock']=line.split(':',1)[1].strip()
    peak_source="resource_time" if resource_block.get('maximum_resident_set_mb') is not None else "memory_sampler"
    payload=SubprocessDiagnostics(schema_version=SCHEMA_VERSION,created_at=_now_iso(),command=list(command),cwd=str(cwd),stdout_log=str(stdout_log),stderr_log=str(stderr_log),started_at=started_at,finished_at=finished_at,elapsed_seconds=float(elapsed_seconds),return_code=int(return_code),failure_class=failure_class,failure_hint=failure_hint,resource_time=resource_block,memory_sampler={"enabled":bool(memory_diagnostics),"peak_direct_child_rss_mb":peak_direct,"peak_descendant_tree_rss_mb":peak_tree,"descendant_pids_seen":sorted(descendants_seen),"sampling_notes":notes,"samples_jsonl":samples_jsonl_path,"peak_total_rss_mb":peak_tree})
    d=asdict(payload); d['peak_rss_source']=peak_source
    diagnostics_json.write_text(json.dumps(d, indent=2), encoding='utf-8')
    return payload
