from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "generate_target_grating_portraits.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("target_grating_portraits_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config_payload(*, psf_npix: int = 32, pupil_npix: int = 64) -> dict:
    return {
        "system": {
            "preset": "SHERA_FLIGHT_3P",
            "optics": {"psf_npix": psf_npix, "pupil_npix": pupil_npix},
            "detector": {"layers": []},
        }
    }


def test_parse_targets_all_and_subset() -> None:
    module = _load_module()
    all_targets = module._parse_targets("all")
    assert "ALPHA_CEN" in all_targets
    assert module._parse_targets("ALPHA_CEN,61_CYG") == ["ALPHA_CEN", "61_CYG"]


def test_parse_targets_rejects_unknown() -> None:
    module = _load_module()
    with pytest.raises(ValueError):
        module._parse_targets("ALPHA_CEN,NOT_REAL")


def test_grating_phase_flip_changes_pattern() -> None:
    module = _load_module()
    mask = np.zeros((8, 8), dtype=float)
    mask[:, 4:] = 1.0

    flipped = module._build_phase_flipped_grating_opd(
        binary_mask=mask,
        amplitude_opd_m=1e-9,
        frequency=4.0,
        angle_deg=45.0,
        phase_flip=True,
    )
    plain = module._build_phase_flipped_grating_opd(
        binary_mask=mask,
        amplitude_opd_m=1e-9,
        frequency=4.0,
        angle_deg=45.0,
        phase_flip=False,
    )
    assert flipped.shape == mask.shape
    assert np.isfinite(flipped).all()
    assert np.max(np.abs(flipped - plain)) > 1.0e-12


def test_cli_dry_run_writes_plan(tmp_path: Path) -> None:
    module = _load_module()
    config_path = _write_config(tmp_path / "cfg.json", _config_payload())
    results_dir = tmp_path / "results"

    module.main(
        [
            "--config",
            str(config_path),
            "--targets",
            "ALPHA_CEN",
            "--results-dir",
            str(results_dir),
            "--run-name",
            "dry_run_case",
            "--psf-npix",
            "32",
            "--pupil-npix",
            "512",
            "--exposure-time-s",
            "0.05",
            "--n-lambda",
            "11",
            "--dry-run",
        ]
    )

    plan_path = results_dir / "dry_run_case" / "dry_run_plan.json"
    assert plan_path.is_file()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["schema"] == "target_grating_portraits.v1"
    assert plan["targets"] == ["ALPHA_CEN"]
    assert plan["grating"]["phase_amplitude_rad"] == pytest.approx(np.pi / 16.0)
    assert plan["optics_overrides"]["psf_npix"] == 32
    assert plan["optics_overrides"]["pupil_npix"] == 512
    assert plan["source_overrides"]["exposure_time_s"] == pytest.approx(0.05)
    assert plan["source_overrides"]["n_lambda"] == 11


def test_cli_dry_run_with_alpha_cen_a_single_star_writes_plan(tmp_path: Path) -> None:
    module = _load_module()
    config_path = _write_config(tmp_path / "cfg.json", _config_payload())
    results_dir = tmp_path / "results"

    module.main(
        [
            "--config",
            str(config_path),
            "--targets",
            "ALPHA_CEN",
            "--include-alpha-cen-a-single-star",
            "--results-dir",
            str(results_dir),
            "--run-name",
            "single_star_dry_run_case",
            "--psf-npix",
            "32",
            "--pupil-npix",
            "512",
            "--dry-run",
        ]
    )

    plan_path = results_dir / "single_star_dry_run_case" / "dry_run_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["targets"] == ["ALPHA_CEN"]
    assert plan["single_star_placeholders_requested"] == [module.ALPHA_CEN_A_SINGLE_KEY]
    note = plan["single_star_placeholder_note"]
    assert "Alpha Cen" in note
    assert "component A" in note


def test_single_star_panel_title_omits_binary_fields() -> None:
    module = _load_module()

    title = module._format_panel_title(
        display_name=module.ALPHA_CEN_A_SINGLE_DISPLAY_NAME,
        entry_key=module.ALPHA_CEN_A_SINGLE_KEY,
        summary={
            "source_kind": "single_star",
            "x_position_as": 0.0,
            "y_position_as": 0.0,
            "log_flux_total": 6.0,
        },
    )

    assert "single_star" in title
    assert "sep=" not in title
    assert "contrast=" not in title
