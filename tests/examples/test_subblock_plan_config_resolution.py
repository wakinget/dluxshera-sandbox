from __future__ import annotations

import pytest

from dluxshera.utils.full_fidelity_review import resolve_subblock_plan_settings


def _config(
    *,
    n_subblocks: int = 2,
    trace_window_n_subblocks: int | None = None,
    windows_per_draw: int = 2,
    subblocks_per_window: int = 1,
    partial_policy: str | None = None,
) -> dict:
    window = {"start_s": 60.0}
    if trace_window_n_subblocks is not None:
        window["n_subblocks"] = trace_window_n_subblocks
    iterative = {
        "enabled": True,
        "windows_per_draw": windows_per_draw,
        "subblocks_per_window": subblocks_per_window,
    }
    if partial_policy is not None:
        iterative["partial_window_policy"] = partial_policy
    return {
        "experiment": {
            "subblocks": {
                "n_subblocks": n_subblocks,
                "trace_source": {"mode": "trajectory", "window": window},
            },
            "iterative": iterative,
        }
    }


def test_consistent_subblocks_and_trace_window_passes_with_warning() -> None:
    resolved, warnings = resolve_subblock_plan_settings(
        _config(n_subblocks=2, trace_window_n_subblocks=2)
    )

    assert resolved["resolved_n_subblocks"] == 2
    assert resolved["consistency_status"] == "consistent"
    assert any("redundant but agrees" in warning for warning in warnings)


def test_inconsistent_trace_window_warns_by_default_and_fails_strict() -> None:
    cfg = _config(n_subblocks=2, trace_window_n_subblocks=3)

    resolved, warnings = resolve_subblock_plan_settings(cfg)

    assert resolved["resolved_n_subblocks"] == 2
    assert resolved["consistency_status"] == "inconsistent"
    assert any("disagrees with resolved total subblocks=2" in warning for warning in warnings)
    with pytest.raises(ValueError, match="trace_source.window.n_subblocks=3 disagrees"):
        resolve_subblock_plan_settings(cfg, strict=True)


def test_iterative_grouping_product_matches_canonical_count() -> None:
    resolved, warnings = resolve_subblock_plan_settings(
        _config(n_subblocks=2, windows_per_draw=2, subblocks_per_window=1)
    )

    assert resolved["expected_iterative_subblocks"] == 2
    assert resolved["consistency_status"] == "consistent"
    assert any("matching subblocks.n_subblocks=2" in warning for warning in warnings)


def test_iterative_grouping_mismatch_warns_and_fails_strict_without_policy() -> None:
    cfg = _config(n_subblocks=4, windows_per_draw=2, subblocks_per_window=1)

    resolved, warnings = resolve_subblock_plan_settings(cfg)

    assert resolved["expected_iterative_subblocks"] == 2
    assert resolved["consistency_status"] == "inconsistent"
    assert any("Remove subblocks.n_subblocks" in warning for warning in warnings)
    with pytest.raises(ValueError, match="Remove subblocks.n_subblocks"):
        resolve_subblock_plan_settings(cfg, strict=True)


def test_iterative_grouping_mismatch_allows_explicit_partial_policy() -> None:
    cfg = _config(
        n_subblocks=4,
        windows_per_draw=2,
        subblocks_per_window=1,
        partial_policy="allow_unused_subblocks",
    )
    with pytest.raises(ValueError, match="Remove subblocks.n_subblocks"):
        resolve_subblock_plan_settings(cfg, strict=True)


def test_iterative_enabled_can_derive_total_when_n_subblocks_omitted() -> None:
    cfg = _config(n_subblocks=2, windows_per_draw=3, subblocks_per_window=5)
    del cfg["experiment"]["subblocks"]["n_subblocks"]

    resolved, warnings = resolve_subblock_plan_settings(cfg, strict=True)

    assert resolved["resolved_total_subblocks"] == 15
    assert resolved["subblock_count_source"] == "experiment.iterative.windows_per_draw*subblocks_per_window"
    assert any("deriving total subblocks" in warning for warning in warnings)


def test_trace_window_n_subblocks_is_optional() -> None:
    resolved, warnings = resolve_subblock_plan_settings(_config(trace_window_n_subblocks=None))

    assert resolved["resolved_n_subblocks"] == 2
    assert resolved["trace_source_window_n_subblocks"] is None
    assert resolved["canonical_source"] == "experiment.iterative.windows_per_draw*subblocks_per_window"
    assert not any("trace_source.window.n_subblocks" in warning for warning in warnings)
