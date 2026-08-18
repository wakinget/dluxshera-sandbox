from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "audit_full_fidelity_config.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


def _module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("audit_full_fidelity_config_redundant_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_reports_semantic_redundant_fields(tmp_path: Path) -> None:
    module = _module()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    iterative = cfg["experiment"]["iterative"]
    cfg["experiment"]["subblocks"]["n_subblocks"] = (
        int(iterative["windows_per_draw"]) * int(iterative["subblocks_per_window"])
    )
    path = tmp_path / "redundant.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    audit = module.build_audit(path, tmp_path, strict=False)

    overlap_keys = {(row["field_a"], row["field_b"]) for row in audit["semantic_overlaps"]}
    assert (
        "experiment.subblocks.n_subblocks",
        "experiment.iterative.windows_per_draw*subblocks_per_window",
    ) in overlap_keys
    assert all("canonical_field" in row for row in audit["semantic_overlaps"])
