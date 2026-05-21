from __future__ import annotations

import json

from dluxshera.utils.runtime_profile import (
    CACHEABILITY_VALUES,
    RuntimeProfiler,
    block_until_ready_if_jax,
    write_profile_summary_json,
    write_profile_timeline_jsonl,
)


def test_runtime_profile_writes_summary_and_timeline(tmp_path):
    profiler = RuntimeProfiler(run_context={"mode": "test"})
    with profiler.profile_stage("stage_a", cacheability="not_cacheable"):
        _ = 1 + 1

    summary_path = tmp_path / "summary.json"
    timeline_path = tmp_path / "timeline.jsonl"
    write_profile_summary_json(summary_path, profiler.summary_payload(outputs={"summary_json": str(summary_path)}))
    write_profile_timeline_jsonl(timeline_path, profiler.events)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == "runtime_profile_summary.v1"
    assert summary["totals"]["completed_stage_count"] == 1

    lines = timeline_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["schema_version"] == "runtime_profile_event.v1"


def test_runtime_profile_records_failure():
    profiler = RuntimeProfiler()
    try:
        with profiler.profile_stage("broken"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert profiler.events[-1].status == "failed"
    assert profiler.events[-1].exception_type == "RuntimeError"


def test_block_until_ready_non_jax_values():
    payload = {"a": [1, 2], "b": (3, {"c": 4})}
    assert block_until_ready_if_jax(payload) == payload


def test_cacheability_values_are_stable():
    assert "amortizable_jax_compile" in CACHEABILITY_VALUES
