from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
INSPECT = ROOT / "examples" / "scripts" / "inspect_full_fidelity_smear.py"
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


@pytest.fixture()
def smoke_dryrun(tmp_path: Path) -> Path:
    wrapper = _load_script(WRAPPER, "run_full_fidelity_binary_iterative_inspect_test")
    status = wrapper.run_full_fidelity_binary_iterative_campaign(
        config_path=CONFIG,
        results_root=tmp_path,
        run_name="smear_inspect_dryrun",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )
    return Path(status["run_root"])


def test_inspect_full_fidelity_smear_strict_passes(smoke_dryrun: Path) -> None:
    inspect = _load_script(INSPECT, "inspect_full_fidelity_smear_pass_test")

    rows = inspect.inspect_run(smoke_dryrun, strict=True)

    assert len(rows) == 4
    assert all(row["render_match"] is True for row in rows)
    assert all(row["inference_match"] is True for row in rows)


def test_inspect_full_fidelity_smear_fails_on_placeholder_leak(smoke_dryrun: Path) -> None:
    inspect = _load_script(INSPECT, "inspect_full_fidelity_smear_fail_test")
    plan = json.loads((smoke_dryrun / "campaign_plan.json").read_text(encoding="utf-8"))
    first_row = next(iter(plan["subblock_plan"].values()))[0]
    render_path = Path(first_row["render_template_path"])
    payload = json.loads(render_path.read_text(encoding="utf-8"))

    def patch_smear(value):
        if isinstance(value, dict):
            if value.get("name") == "smear" and isinstance(value.get("kernel"), dict):
                value["kernel"]["length"] = 1.0e-12
                return True
            return any(patch_smear(child) for child in value.values())
        if isinstance(value, list):
            return any(patch_smear(child) for child in value)
        return False

    assert patch_smear(payload)
    render_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="placeholder_leak|render_mismatch"):
        inspect.inspect_run(smoke_dryrun, strict=True)
