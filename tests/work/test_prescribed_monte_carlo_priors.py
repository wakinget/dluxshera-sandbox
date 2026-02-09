from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prescribed_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "work" / "experiments" / "prescribed_monte_carlo.py"
    spec = importlib.util.spec_from_file_location("prescribed_monte_carlo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load prescribed_monte_carlo module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_prior_override_sigma():
    module = _load_prescribed_module()
    row_clean, overrides = module._extract_prior_overrides(
        {"prior.binary.x_position_as.sigma": 0.2, "seed": 1}
    )

    assert row_clean == {"seed": 1}
    assert overrides == {"binary.x_position_as": {"sigma": 0.2}}


def test_extract_prior_override_std_normalizes_to_sigma():
    module = _load_prescribed_module()
    _, overrides = module._extract_prior_overrides(
        {"prior.binary.x_position_as.std": 0.3}
    )

    assert overrides == {"binary.x_position_as": {"sigma": 0.3}}


def test_extract_prior_override_vector_sigma():
    module = _load_prescribed_module()
    _, overrides = module._extract_prior_overrides(
        {"prior.primary.zernike_coeffs_nm.sigma": [1, 2, 3]}
    )

    assert overrides == {"primary.zernike_coeffs_nm": {"sigma": [1, 2, 3]}}


def test_extract_prior_override_null_is_ignored_and_warned(capsys):
    module = _load_prescribed_module()
    _, overrides = module._extract_prior_overrides(
        {"prior.binary.x_position_as.sigma": None}
    )

    assert overrides == {}
    captured = capsys.readouterr()
    assert "null" in captured.out.lower()


def test_apply_prior_override_unknown_infer_key_warns_and_skips(capsys):
    module = _load_prescribed_module()
    base_prior_info = {"binary.x_position_as": {"dist": "Normal", "sigma": 1.0}}

    merged, applied = module._apply_prior_overrides(
        base_prior_info,
        {"unknown.key": {"sigma": 2.0}},
        infer_keys=("binary.x_position_as",),
        base_store=None,
    )

    assert merged == base_prior_info
    assert applied == []
    captured = capsys.readouterr()
    assert "unknown infer key" in captured.out.lower()
