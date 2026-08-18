from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)
STALE_NOTE = "Trajectory high-pass filtering and smear are reserved for a later task."


def _load_wrapper():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_full_fidelity_binary_iterative_wfe_provenance_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_fidelity_dryrun_artifacts_do_not_emit_stale_wfe_follow_on_note(
    tmp_path: Path,
) -> None:
    module = _load_wrapper()
    status = module.run_full_fidelity_binary_iterative_campaign(
        config_path=CONFIG,
        results_root=tmp_path,
        run_name="wfe_provenance_dryrun",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )

    run_root = Path(status["run_root"])
    summary_paths = sorted(run_root.glob("**/high_order_wfe_summary.json"))
    assert summary_paths
    for path in run_root.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".md", ".txt", ".csv", ".sh"}:
            assert STALE_NOTE not in path.read_text(encoding="utf-8")
