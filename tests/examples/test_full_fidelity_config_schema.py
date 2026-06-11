from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dluxshera.config.io import load_config_file
from dluxshera.utils.full_fidelity_config_schema import (
    iter_string_fields,
    registry_entry_for_path,
    validate_config_contract,
)

ROOT = Path(__file__).resolve().parents[2] / "examples/recipes/full_fidelity_algorithm_campaign_template"
REVIEW = ROOT / "full_fidelity_binary_iterative_review.yaml"
SMOKE = ROOT / "full_fidelity_binary_iterative_smoke.yaml"


@pytest.mark.parametrize("path,tier", [(REVIEW, "review"), (SMOKE, "smoke")])
def test_every_string_field_has_registry_entry(path: Path, tier: str) -> None:
    cfg = load_config_file(path)
    missing = [(p, v) for p, v in iter_string_fields(cfg) if registry_entry_for_path(p)[1] is None]
    assert missing == []
    assert validate_config_contract(cfg, config_tier=tier, strict=True)["has_errors"] is False


def test_unsupported_enum_value_produces_audit_error(tmp_path: Path) -> None:
    cfg = load_config_file(REVIEW)
    cfg["experiment"]["spectral_model"]["inference"]["out_of_band_response"] = "bogus"
    result = validate_config_contract(cfg, config_tier="review", strict=True)
    assert result["has_errors"] is True
    assert any(f["code"] == "unsupported_enum_value" for f in result["findings"])


def test_future_value_in_executable_config_is_reported() -> None:
    cfg = load_config_file(REVIEW)
    cfg["experiment"]["spectral_model"]["inference"]["out_of_band_response"] = "edge_hold"
    result = validate_config_contract(cfg, config_tier="review", strict=True)
    assert result["has_errors"] is True
    assert any(f["code"] == "future_value_used" for f in result["findings"])


def test_smoke_only_fast_in_review_is_error() -> None:
    cfg = load_config_file(REVIEW)
    cfg["experiment"]["spectral_model"]["fast"] = True
    result = validate_config_contract(cfg, config_tier="review", strict=True)
    assert result["has_errors"] is True
    assert any(f["code"] in {"smoke_only_field_in_review", "fast_in_review_config"} for f in result["findings"])
