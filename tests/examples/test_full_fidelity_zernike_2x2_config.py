from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
CONFIG = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_zernike_2x2_self_correction_hpc_v1.yaml"
)
SMOKE = (
    ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)

EXPECTED = {
    "m1_0p3nm_m2_0p3nm": (0.3, 0.3),
    "m1_1p0nm_m2_0p3nm": (1.0, 0.3),
    "m1_0p3nm_m2_1p0nm": (0.3, 1.0),
    "m1_1p0nm_m2_1p0nm": (1.0, 1.0),
}


def _load_wrapper():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "run_full_fidelity_zernike_2x2_config_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_zernike_2x2_config_translates_and_preserves_non_zernike_decks() -> None:
    module = _load_wrapper()
    cfg = _yaml(CONFIG)
    smoke = _yaml(SMOKE)
    translated = module._full_fidelity_to_observation_bias(cfg, run_name="unit_zernike")
    experiment = translated["experiment"]

    assert experiment["subblocks"]["phi_ref"] == "truth_when_available"
    for key in ("spectral_model", "high_order_wfe", "detector_overrides"):
        assert cfg["experiment"][key] == smoke["experiment"][key]
    for key in ("noise", "trace_source", "trajectory_processing", "trace_jitter"):
        assert cfg["experiment"]["subblocks"][key] == smoke["experiment"]["subblocks"][key]
    for key in (
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
        "optics.plate_scale_as_per_pix",
    ):
        assert cfg["experiment"]["prior_draws"]["sigmas"][key] == smoke["experiment"]["prior_draws"]["sigmas"][key]


def test_zernike_2x2_condition_matrix_and_size(tmp_path: Path) -> None:
    module = _load_wrapper()
    cfg = _yaml(CONFIG)
    conditions = cfg["experiment"]["prior_draws"]["conditions"]

    assert len(conditions) == 4
    for condition in conditions:
        name = condition["condition_name"]
        assert name in EXPECTED
        primary, secondary = EXPECTED[name]
        assert condition["sigmas"]["optics.primary.zernike_coeffs_nm[*]"]["sigma"] == primary
        assert condition["sigmas"]["optics.secondary.zernike_coeffs_nm[*]"]["sigma"] == secondary

    status = module.run_full_fidelity_binary_iterative_campaign(
        config_path=CONFIG,
        results_root=tmp_path,
        run_name="zernike_2x2_dryrun",
        dry_run=True,
        aggregate_only=False,
        resume=False,
        max_workers=1,
        fail_fast=True,
        quiet=True,
        resource_time="disabled",
    )
    run_root = Path(status["run_root"])
    plan = json.loads((run_root / "campaign_plan.json").read_text(encoding="utf-8"))
    labels = plan["theta_layout"]["labels"]

    assert plan["smear_audit"]["n_subblocks"] == 15
    assert sum(len(commands) for commands in plan["subblock_commands"].values()) == 300
    assert len(plan["bias_cases"]) == 20
    assert len([label for label in labels if label.startswith("optics.primary.zernike_coeffs_nm[")]) == 8
    assert len([label for label in labels if label.startswith("optics.secondary.zernike_coeffs_nm[")]) == 8
    assert len(labels) == 20

    rows = list(csv.DictReader((run_root / "prior_draws.csv").open("r", encoding="utf-8", newline="")))
    assert {row["condition_name"] for row in rows} == set(EXPECTED)
    for name, (primary, secondary) in EXPECTED.items():
        condition_rows = [row for row in rows if row["condition_name"] == name]
        assert len({row["case_name"] for row in condition_rows}) == 5
        primary_rows = [
            row for row in condition_rows if row["theta_label"].startswith("optics.primary.zernike_coeffs_nm[")
        ]
        secondary_rows = [
            row for row in condition_rows if row["theta_label"].startswith("optics.secondary.zernike_coeffs_nm[")
        ]
        assert {float(row["prior_sigma"]) for row in primary_rows} == {primary}
        assert {float(row["prior_sigma"]) for row in secondary_rows} == {secondary}
