from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OBS_SCRIPT = ROOT / "examples" / "scripts" / "run_observation_bias_campaign.py"
WRAPPER_SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


def _load_script(path: Path, name: str):
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregate_only_replay_uses_stored_subblock_smear_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    obs = _load_script(OBS_SCRIPT, "run_observation_bias_aggregate_smear_test")
    wrapper = _load_script(WRAPPER_SCRIPT, "run_full_fidelity_translate_aggregate_smear_test")
    translated = wrapper._full_fidelity_to_observation_bias(
        wrapper.load_config_file(CONFIG),
        run_name="aggregate_only_stored_smear",
    )
    config_path = tmp_path / "translated.json"
    config_path.write_text(json.dumps(translated, indent=2), encoding="utf-8")

    dry_run = obs.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="aggregate_only_stored_smear",
        dry_run=True,
        aggregate_only=False,
        quiet=True,
        resource_time="disabled",
    )
    run_root = Path(dry_run["run_root"])
    plan = json.loads((run_root / "campaign_plan.json").read_text(encoding="utf-8"))
    first_row = next(iter(plan["subblock_plan"].values()))[0]
    assert first_row["smear_representative_kernel_json"]

    def fail_template_write(*args, **kwargs):  # pragma: no cover - failure path assertion
        raise AssertionError("aggregate-only replay must not rewrite subblock smear templates")

    monkeypatch.setattr(obs, "_write_subblock_smear_templates", fail_template_write)
    monkeypatch.setattr(
        obs,
        "aggregate_iterative_outputs",
        lambda plan: {"status": "ok", "run_root": str(plan.run_root), "expected": len(plan.expected_output_rows)},
    )

    replay = obs.run_observation_bias_campaign(
        config_path=config_path,
        results_root=tmp_path,
        run_name="aggregate_only_stored_smear",
        dry_run=False,
        aggregate_only=True,
        quiet=True,
        resource_time="disabled",
    )

    assert replay["status"] == "ok"
    validation = json.loads(
        (run_root / "analysis" / "aggregate_only_plan_validation.json").read_text(
            encoding="utf-8"
        )
    )
    assert validation["status"] == "ok"
    assert validation["replay_mode"] == "stored_plan_subblock_constant_smear"
