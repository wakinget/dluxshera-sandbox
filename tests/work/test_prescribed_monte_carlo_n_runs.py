from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_prescribed_module():
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / "work" / "experiments" / "prescribed_monte_carlo.py",
        repo_root / "examples" / "recipes" / "prescribed_monte_carlo.py",
    ]
    for module_path in candidate_paths:
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("prescribed_monte_carlo", module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError("Unable to load prescribed_monte_carlo module.")


def test_apply_experiment_n_runs_pads_defaults():
    module = _load_prescribed_module()
    plan_rows = [
        {"_plan_label": "a", "seed": 1},
        {"_plan_label": "b", "seed": 2, "enabled": True},
    ]
    resolved, meta = module._apply_experiment_n_runs(plan_rows, 5)

    assert len(resolved) == 5
    assert resolved[0]["seed"] == 1
    assert resolved[1]["seed"] == 2
    assert resolved[2] == {"_plan_label": None}
    assert resolved[3] == {"_plan_label": None}
    assert resolved[4] == {"_plan_label": None}
    assert meta["padded_runs"] == 3
    assert meta["truncated_runs"] == 0


def test_apply_experiment_n_runs_truncates_plan():
    module = _load_prescribed_module()
    plan_rows = [
        {"_plan_label": "a", "seed": 1},
        {"_plan_label": "b", "seed": 2},
        {"_plan_label": "c", "seed": 3},
    ]
    resolved, meta = module._apply_experiment_n_runs(plan_rows, 2)

    assert len(resolved) == 2
    assert [row["seed"] for row in resolved] == [1, 2]
    assert meta["padded_runs"] == 0
    assert meta["truncated_runs"] == 1


def test_apply_experiment_n_runs_none_uses_plan():
    module = _load_prescribed_module()
    plan_rows = [
        {"_plan_label": "a", "seed": 1},
        {"_plan_label": "b", "seed": 2},
    ]
    resolved, meta = module._apply_experiment_n_runs(plan_rows, None)

    assert resolved == plan_rows
    assert meta["resolved_runs"] == 2
    assert meta["n_runs"] is None


def test_apply_experiment_n_runs_rejects_non_positive():
    module = _load_prescribed_module()
    plan_rows = [{"seed": 1}]

    with pytest.raises(ValueError, match="positive"):
        module._apply_experiment_n_runs(plan_rows, 0)


def test_print_preview_includes_enabled_column(capsys):
    module = _load_prescribed_module()
    run_specs = [
        {"run_id": "run_0001", "enabled": True, "seed": 1},
        {"enabled": False, "seed": 2},
    ]

    module._print_preview(run_specs)

    output = capsys.readouterr().out
    assert "enabled" in output.splitlines()[0]
    assert "True" in output
    assert "False" in output
