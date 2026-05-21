from __future__ import annotations

import contextlib
import json
import os
import platform
import resource
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

CACHEABILITY_VALUES = (
    "not_cacheable",
    "cacheable_with_same_inputs",
    "cacheable_with_same_structure",
    "amortizable_jax_compile",
    "artifact_reusable",
    "diagnostic_overhead",
    "unknown",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return str(value)


def _maxrss_to_mb(raw: float) -> float:
    return raw / 1024.0 if platform.system().lower() != "darwin" else raw / (1024.0 * 1024.0)


def _memory_snapshot() -> dict[str, float | None]:
    rss_mb = None
    peak_rss_mb = None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_rss_mb = _maxrss_to_mb(float(usage.ru_maxrss))
    except Exception:
        pass
    if tracemalloc.is_tracing():
        current, peak = tracemalloc.get_traced_memory()
        return {
            "rss_mb": rss_mb,
            "peak_rss_mb": peak_rss_mb,
            "tracemalloc_current_mb": float(current) / (1024 * 1024),
            "tracemalloc_peak_mb": float(peak) / (1024 * 1024),
        }
    return {"rss_mb": rss_mb, "peak_rss_mb": peak_rss_mb, "tracemalloc_current_mb": None, "tracemalloc_peak_mb": None}


@dataclass
class RuntimeProfileEvent:
    stage: str
    status: str
    started_at_unix: float
    finished_at_unix: float
    duration_s: float
    cacheability: str = "unknown"
    category: str = "unknown"
    details: dict[str, Any] = field(default_factory=dict)
    rss_mb: float | None = None
    peak_rss_mb: float | None = None
    tracemalloc_current_mb: float | None = None
    tracemalloc_peak_mb: float | None = None
    exception_type: str | None = None
    exception_message: str | None = None
    schema_version: str = "runtime_profile_event.v1"


class RuntimeProfiler:
    def __init__(self, *, run_context: Mapping[str, Any] | None = None):
        self.run_context = dict(run_context or {})
        self.events: list[RuntimeProfileEvent] = []
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    @contextlib.contextmanager
    def profile_stage(self, stage: str, *, cacheability: str = "unknown", category: str = "unknown", details: Mapping[str, Any] | None = None) -> Iterator[None]:
        started = time.time()
        tic = time.perf_counter()
        event = RuntimeProfileEvent(stage=stage, status="completed", started_at_unix=started, finished_at_unix=started, duration_s=0.0, cacheability=cacheability, category=category, details=dict(details or {}))
        try:
            yield
        except Exception as exc:
            event.status = "failed"
            event.exception_type = type(exc).__name__
            event.exception_message = str(exc)
            raise
        finally:
            event.finished_at_unix = time.time()
            event.duration_s = time.perf_counter() - tic
            mem = _memory_snapshot()
            event.rss_mb = mem["rss_mb"]
            event.peak_rss_mb = mem["peak_rss_mb"]
            event.tracemalloc_current_mb = mem["tracemalloc_current_mb"]
            event.tracemalloc_peak_mb = mem["tracemalloc_peak_mb"]
            self.events.append(event)

    def summary_payload(self, *, outputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
        stage_totals: dict[str, float] = {}
        for event in self.events:
            stage_totals[event.stage] = stage_totals.get(event.stage, 0.0) + float(event.duration_s)
        env = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "process_id": os.getpid(),
        }
        total = sum(event.duration_s for event in self.events)
        return {
            "schema_version": "runtime_profile_summary.v1",
            "created_at": _now_iso(),
            "run_context": _safe(self.run_context),
            "environment": env,
            "outputs": _safe(dict(outputs or {})),
            "totals": {
                "total_profiled_duration_s": total,
                "completed_stage_count": sum(1 for e in self.events if e.status == "completed"),
                "failed_stage_count": sum(1 for e in self.events if e.status == "failed"),
            },
            "stage_totals": stage_totals,
            "events": [_safe(asdict(e)) for e in self.events],
        }


def write_profile_summary_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_profile_timeline_jsonl(path: Path, events: list[RuntimeProfileEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(_safe(asdict(event))) + "\n")


def block_until_ready_if_jax(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if isinstance(value, list):
        return [block_until_ready_if_jax(v) for v in value]
    if isinstance(value, tuple):
        return tuple(block_until_ready_if_jax(v) for v in value)
    if isinstance(value, dict):
        return {k: block_until_ready_if_jax(v) for k, v in value.items()}
    return value
