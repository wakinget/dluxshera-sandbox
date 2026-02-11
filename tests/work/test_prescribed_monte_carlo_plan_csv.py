from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_load_plan_csv_standard_preserves_note_text(tmp_path):
    module = _load_prescribed_module()
    plan_path = tmp_path / "plan.csv"
    plan_path.write_text(
        "seed,note,comment\n"
        "1,123,true\n"
        "2,true,  hello world  \n"
        "3, ,none\n",
        encoding="utf-8",
    )

    rows = module._load_plan_csv(plan_path)

    assert rows[0]["seed"] == 1
    assert rows[0]["note"] == "123"
    assert rows[0]["comment"] == "true"

    assert rows[1]["seed"] == 2
    assert rows[1]["note"] == "true"
    assert rows[1]["comment"] == "hello world"

    assert rows[2]["seed"] == 3
    assert rows[2]["note"] is None
    assert rows[2]["comment"] is None


def test_load_plan_csv_transposed_preserves_note_text(tmp_path):
    module = _load_prescribed_module()
    plan_path = tmp_path / "plan_transposed.csv"
    plan_path.write_text(
        "key,run_a,run_b\n"
        "seed,1,2\n"
        "note,123,true\n"
        "comments, ,none\n",
        encoding="utf-8",
    )

    rows = module._load_plan_csv(plan_path)

    assert rows[0]["_plan_label"] == "run_a"
    assert rows[0]["seed"] == 1
    assert rows[0]["note"] == "123"
    assert rows[0]["comments"] is None

    assert rows[1]["_plan_label"] == "run_b"
    assert rows[1]["seed"] == 2
    assert rows[1]["note"] == "true"
    assert rows[1]["comments"] is None
