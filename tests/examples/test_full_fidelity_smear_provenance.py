from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dluxshera.utils.smear_audit import load_named_smear_kernel


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)


def _load_wrapper():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_full_fidelity_binary_iterative_smear_provenance_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def smoke_dryrun(tmp_path: Path) -> Path:
    module = _load_wrapper()
    status = module.run_full_fidelity_binary_iterative_campaign(
        config_path=CONFIG,
        results_root=tmp_path,
        run_name="smear_provenance_dryrun",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )
    return Path(status["run_root"])


def test_smear_provenance_exposes_representative_kernel(smoke_dryrun: Path) -> None:
    paths = sorted(smoke_dryrun.glob("trajectory/subblock_*/smear_provenance.json"))
    assert len(paths) == 4
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        kernel = data["representative_kernel"]
        assert kernel["source"] == "subblock_linear_fit_one_frame_exposure"
        assert data["truth_kernel"]["length"] == pytest.approx(kernel["length"])
        assert data["model_kernel"]["length"] == pytest.approx(kernel["length"])
        assert data["matched_model"] is True
        assert data["render_template_path"]
        assert data["inference_template_path"]
        render_kernel = load_named_smear_kernel(Path(data["render_template_path"]))
        inference_kernel = load_named_smear_kernel(Path(data["inference_template_path"]))
        assert render_kernel["length"] == pytest.approx(kernel["length"])
        assert render_kernel["theta_deg"] == pytest.approx(kernel["theta_deg"])
        assert inference_kernel["length"] == pytest.approx(kernel["length"])
        assert inference_kernel["theta_deg"] == pytest.approx(kernel["theta_deg"])


def test_smear_summary_contains_one_row_per_subblock(smoke_dryrun: Path) -> None:
    summary = smoke_dryrun / "trajectory" / "smear_summary.csv"
    rows = list(csv.DictReader(summary.open("r", encoding="utf-8", newline="")))

    assert len(rows) == 4
    assert {row["template_status"] for row in rows} == {"ok"}
    assert {row["render_match"] for row in rows} == {"True"}
    assert {row["inference_match"] for row in rows} == {"True"}


def test_model_split_labels_global_seed_kernel_as_per_subblock_placeholder(
    smoke_dryrun: Path,
) -> None:
    plan = json.loads((smoke_dryrun / "campaign_plan.json").read_text(encoding="utf-8"))
    trajectory_smear = plan["model_split"]["provenance"]["trajectory_smear"]
    truth_policy = trajectory_smear["truth_policy"]
    inference_policy = trajectory_smear["inference_policy"]

    assert "representative_kernel" not in truth_policy
    assert "representative_kernel" not in inference_policy
    assert truth_policy["representative_kernel_scope"] == "per_subblock"
    assert inference_policy["representative_kernel_scope"] == "per_subblock"
    assert truth_policy["global_template_seed_kernel"]["length"] == pytest.approx(1.0e-12)
