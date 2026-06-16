from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
AUDIT = ROOT / "examples" / "scripts" / "audit_full_fidelity_config.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


def _load(path: Path, name: str):
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_writes_noise_provenance(tmp_path: Path) -> None:
    module = _load(WRAPPER, "run_full_fidelity_binary_iterative_noise_prov_test")
    status = module.run_full_fidelity_binary_iterative_campaign(
        config_path=CONFIG,
        results_root=tmp_path,
        run_name="noise_provenance_dryrun",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )

    run_root = Path(status["run_root"])
    normalized = json.loads((run_root / "noise" / "noise_request_normalized.json").read_text())
    render = json.loads((run_root / "noise" / "noise_render_provenance.json").read_text())
    inference = json.loads((run_root / "noise" / "noise_inference_provenance.json").read_text())

    assert normalized["shot_noise"] is True
    assert normalized["photon_noise"] is True
    assert normalized["read_noise"] is True
    assert render["render_terms_forwarded"] is True
    assert render["render_template_noise_block"]["read_noise"] is True
    assert inference["inference_variance_floor"] == 0.5
    assert inference["variance_floor_source"] == "experiment.subblocks.noise.variance_floor"


def test_strict_audit_reports_active_structured_noise(tmp_path: Path) -> None:
    module = _load(AUDIT, "audit_full_fidelity_config_noise_prov_test")

    audit = module.build_audit(CONFIG, tmp_path, strict=True)

    noise = audit["noise_policy_summary"]
    assert noise["structured_noise_supported"] is True
    assert noise["render_terms_forwarded"] is True
    assert noise["inference_variance_floor_forwarded"] is True
    assert noise["read_noise_source"] in {"detector_spec", "config_override"} or noise["read_noise_source"].startswith("detector")
    assert not any("only recorded" in warning for warning in audit["warnings"])
