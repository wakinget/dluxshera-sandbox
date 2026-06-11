from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "examples" / "scripts" / "audit_full_fidelity_config.py"
CONFIG_PATH = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_smoke.yaml"
)
REVIEW_CONFIG_PATH = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_review.yaml"
)
SKELETON_PATH = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_algorithm_campaign_v1.yaml"
)


def load_module() -> Any:
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("audit_full_fidelity_config", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _active_leaf_paths(payload: dict[str, Any]) -> set[str]:
    paths: set[str] = set()

    def walk(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{prefix}.{key}" if prefix else str(key))
            return
        paths.add(prefix)

    walk(payload, "")
    return paths


def test_audit_script_writes_review_artifacts(tmp_path: Path) -> None:
    module = load_module()

    audit = module.build_audit(CONFIG_PATH, tmp_path, run_name="audit_test")

    assert (tmp_path / "config_audit.md").is_file()
    assert (tmp_path / "config_audit.json").is_file()
    assert (tmp_path / "translated_observation_bias_config.json").is_file()
    assert (tmp_path / "field_reference.csv").is_file()
    assert (tmp_path / "field_reference.json").is_file()
    assert (tmp_path / "resolved_component_summary.json").is_file()
    assert audit["config_kind"] == "full_fidelity_binary_iterative_smoke"
    assert audit["executable_today"] is True
    assert "contract_findings" in audit


def test_audit_smoke_uses_explicit_small_spectral_grid_without_fast(tmp_path: Path) -> None:
    module = load_module()

    audit = module.build_audit(CONFIG_PATH, tmp_path, run_name="audit_test")
    summary = audit["resolved_component_summary"]["spectral_model"]

    assert "experiment.spectral_model.fast" not in audit["smoke_only_cost_reducers_or_labels"]
    assert not any("spectral_model.fast" in warning for warning in audit["warnings"])
    assert summary["configured_truth_n_lambda"] == 7
    assert summary["effective_truth_n_lambda"] == 7
    assert summary["effective_inference_n_lambda"] == 3


def test_audit_marks_synthetic_spectral_fast_and_effective_clamp(tmp_path: Path) -> None:
    module = load_module()
    raw = module.load_config_file(CONFIG_PATH)
    raw["experiment"]["spectral_model"]["fast"] = True
    raw["experiment"]["spectral_model"]["truth"]["n_lambda"] = 51
    synthetic = tmp_path / "synthetic_fast.yaml"
    import yaml

    synthetic.write_text(yaml.safe_dump(raw), encoding="utf-8")

    audit = module.build_audit(synthetic, tmp_path / "audit", run_name="audit_test")
    summary = audit["resolved_component_summary"]["spectral_model"]

    assert "experiment.spectral_model.fast" in audit["smoke_only_cost_reducers_or_labels"]
    assert summary["configured_truth_n_lambda"] == 51
    assert summary["effective_truth_n_lambda"] == 7


def test_audit_lists_all_active_smoke_leaf_fields(tmp_path: Path) -> None:
    module = load_module()
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = _active_leaf_paths(raw)

    audit = module.build_audit(CONFIG_PATH, tmp_path, run_name="audit_test")
    actual = {row["field_path"] for row in audit["field_rows"]}

    assert expected <= actual


def test_audit_identifies_truth_reference_mismatch_fields(tmp_path: Path) -> None:
    module = load_module()

    audit = module.build_audit(CONFIG_PATH, tmp_path, run_name="audit_test")

    assert "experiment.spectral_model.truth/inference.n_lambda" in audit["truth_reference_mismatch_fields"]
    assert "experiment.high_order_wfe.truth vs inference.knowledge_error" in audit["truth_reference_mismatch_fields"]


def test_audit_identifies_future_skeleton_as_non_executable(tmp_path: Path) -> None:
    module = load_module()

    audit = module.build_audit(SKELETON_PATH, tmp_path, run_name="audit_test")

    assert audit["config_kind"] == "full_fidelity_algorithm_campaign"
    assert audit["future_schema_skeleton"] is True
    assert audit["executable_today"] is False
    assert "non-executable" in str(audit["translation_error"])


def test_audit_review_config_strict_passes(tmp_path: Path) -> None:
    module = load_module()

    audit = module.build_audit(REVIEW_CONFIG_PATH, tmp_path, run_name="review_audit", strict=True)

    assert audit["config_kind"] == "full_fidelity_binary_iterative_review"
    assert audit["config_tier"] == "review"
    assert audit["executable_today"] is True
    assert audit["contract_findings"] == []


def test_audit_strict_rejects_unsupported_enum(tmp_path: Path) -> None:
    module = load_module()
    raw = yaml.safe_load(REVIEW_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["experiment"]["spectral_model"]["inference"]["out_of_band_response"] = "bogus"
    bad = tmp_path / "bad_review.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")

    try:
        module.build_audit(bad, tmp_path / "audit", run_name="bad", strict=True)
    except ValueError as exc:
        assert "unsupported_enum_value" in str(exc)
    else:
        raise AssertionError("strict audit accepted unsupported enum")
