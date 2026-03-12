from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


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


def test_extract_prior_override_sigma():
    module = _load_prescribed_module()
    row_clean, overrides = module._extract_prior_overrides(
        {"prior.source.x_position_as.sigma": 0.2, "seed": 1}
    )

    assert row_clean == {"seed": 1}
    assert overrides == {"source.x_position_as": {"sigma": 0.2}}


def test_extract_prior_override_std_normalizes_to_sigma():
    module = _load_prescribed_module()
    _, overrides = module._extract_prior_overrides(
        {"prior.source.x_position_as.std": 0.3}
    )

    assert overrides == {"source.x_position_as": {"sigma": 0.3}}


def test_extract_prior_override_vector_sigma():
    module = _load_prescribed_module()
    _, overrides = module._extract_prior_overrides(
        {"prior.optics.primary.zernike_coeffs_nm.sigma": [1, 2, 3]}
    )

    assert overrides == {"optics.primary.zernike_coeffs_nm": {"sigma": [1, 2, 3]}}


def test_extract_prior_override_null_is_ignored_and_warned(capsys):
    module = _load_prescribed_module()
    _, overrides = module._extract_prior_overrides(
        {"prior.source.x_position_as.sigma": None}
    )

    assert overrides == {}
    captured = capsys.readouterr()
    assert "null" in captured.out.lower()


def test_apply_prior_override_unknown_infer_key_warns_and_skips(capsys):
    module = _load_prescribed_module()
    base_prior_info = {"source.x_position_as": {"dist": "Normal", "sigma": 1.0}}

    merged, applied = module._apply_prior_overrides(
        base_prior_info,
        {"unknown.key": {"sigma": 2.0}},
        infer_keys=("source.x_position_as",),
        base_store=None,
    )

    assert merged == base_prior_info
    assert applied == []
    captured = capsys.readouterr()
    assert "unknown infer key" in captured.out.lower()


def test_strip_private_keys_recursively_removes_leading_underscore_keys():
    module = _load_prescribed_module()

    payload = {
        "_comment": "top-level",
        "experiment": {
            "run_id_prefix": "mc",
            "_disabled": True,
        },
        "overrides": {
            "config": {
                "bandwidth_m": 1.1e-7,
                "_bandwidth_m": 9.9e-7,
            },
            "store": {
                "binary": {
                    "x_position_as": 0.01,
                    "_x_position_as": 99.0,
                },
            },
        },
        "nested_list": [
            {"_comment": "ignored", "active": 1},
            {"k": [{"_x": 1, "y": 2}]},
        ],
    }

    stripped = module._strip_private_keys(payload)

    assert "_comment" not in stripped
    assert stripped["experiment"] == {"run_id_prefix": "mc"}
    assert stripped["overrides"]["config"] == {"bandwidth_m": 1.1e-7}
    assert stripped["overrides"]["store"] == {"binary": {"x_position_as": 0.01}}
    assert stripped["nested_list"] == [{"active": 1}, {"k": [{"y": 2}]}]


def test_load_prescription_strips_private_keys_before_overrides_validation(tmp_path):
    module = _load_prescribed_module()

    prescription_path = tmp_path / "prescription.yaml"
    prescription_path.write_text(
        """system:
  preset: SHERA_TESTBED_3P
experiment:
  _comment: drop me
  notes: keep me
  prescribed_mc:
    defaults:
      seed: 11
      truth:
        source:
          x_position_as: 0.123
          _x_position_as: 0.999
""",
        encoding="utf-8",
    )

    prescription = module._load_prescription(prescription_path)

    assert "_comment" not in prescription["experiment"]
    truth = prescription["experiment"]["prescribed_mc"]["defaults"]["truth"]
    assert truth["source"]["x_position_as"] == 0.123
    assert "_x_position_as" not in truth["source"]


def test_resolve_loss_kind_prefers_run_spec_then_defaults():
    module = _load_prescribed_module()

    defaults = {"optimizer": {"loss": "map"}}
    assert module._resolve_loss_kind({}, defaults) == "map"

    run_spec = {"optimizer": {"loss": "nll"}}
    assert module._resolve_loss_kind(run_spec, defaults) == "nll"

    with pytest.raises(ValueError):
        module._resolve_loss_kind({"optimizer": {"loss": "invalid"}}, defaults)
