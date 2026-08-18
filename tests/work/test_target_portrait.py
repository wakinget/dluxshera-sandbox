from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from astropy.io import fits

from dluxshera.components.sources import TARGET_SPECS


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "work" / "scratch" / "target_portrait.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("target_portrait_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_config(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config_payload(*, psf_npix: int = 32, detector_layers: list | None = None) -> dict:
    return {
        "system": {
            "preset": "SHERA_FLIGHT_3P",
            "optics": {"psf_npix": psf_npix},
            "detector": {"layers": [] if detector_layers is None else detector_layers},
        }
    }


def test_unknown_target_parse_error_is_clear(capsys) -> None:
    module = _load_module()

    with pytest.raises(SystemExit) as exc:
        module._parse_args(["--target", "NOT_A_TARGET"])

    assert exc.value.code == 2
    stderr = capsys.readouterr().err
    assert "Unknown target" in stderr
    assert "ALPHA_CEN" in stderr


def test_config_argument_is_accepted(tmp_path: Path) -> None:
    module = _load_module()
    config_path = _write_config(tmp_path / "portrait.json", _config_payload())

    args = module._parse_args(["--target", "ALPHA_CEN", "--config", str(config_path)])

    assert args.config == config_path


def test_single_star_alpha_cen_a_flag_is_accepted() -> None:
    module = _load_module()

    args = module._parse_args(["--target", "ALPHA_CEN", "--single-star-alpha-cen-a"])

    assert args.single_star_alpha_cen_a is True


def test_psf_npix_argument_is_no_longer_accepted(capsys) -> None:
    module = _load_module()

    with pytest.raises(SystemExit) as exc:
        module._parse_args(["--target", "ALPHA_CEN", "--psf-npix", "32"])

    assert exc.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("stretch", ["linear", "sqrt", "log"])
def test_stretch_option_accepts_supported_modes(stretch: str) -> None:
    module = _load_module()

    args = module._parse_args(["--target", "ALPHA_CEN", "--stretch", stretch])

    assert args.stretch == stretch


def test_stretch_option_rejects_invalid_choice(capsys) -> None:
    module = _load_module()

    with pytest.raises(SystemExit) as exc:
        module._parse_args(["--target", "ALPHA_CEN", "--stretch", "asinh"])

    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_prepare_target_system_cfg_preserves_target_authority() -> None:
    module = _load_module()
    base_system_cfg = {
        "source": {
            "kind": "alpha_cen",
            "target": "ALPHA_CEN",
            "contrast": 99.0,
            "log_flux_total": 12.0,
            "position_angle_deg": 271.0,
            "separation_as": 42.0,
            "vmag_a": 1.0,
            "vmag_b": 2.0,
            "wavelength_m": 5.5e-7,
            "bandwidth_m": 4.1e-8,
            "n_lambda": 3,
        },
        "optics": {"psf_npix": 256},
        "detector": {"layers": []},
    }

    prepared = module._prepare_target_system_cfg(
        base_system_cfg,
        target_key="61_CYG",
    )

    assert prepared["source"]["kind"] == "binary_target"
    assert prepared["source"]["target"] == "61_CYG"
    for key in module.TARGET_AUTHORITY_OVERRIDE_KEYS:
        assert key not in prepared["source"]
    assert prepared["optics"]["psf_npix"] == 256
    assert base_system_cfg["source"]["contrast"] == 99.0


def test_source_summary_tolerates_single_star_without_binary_keys() -> None:
    module = _load_module()
    store = module.ParameterStore.from_dict(
        {
            "source.log_flux_total": 6.0,
            "source.x_position_as": 0.0,
            "source.y_position_as": 0.0,
            "source.position_angle_deg": 0.0,
            "optics.plate_scale_as_per_pix": 0.123,
            "optics.psf_npix": 8,
        }
    )

    summary = module._source_summary(store, source_kind="single_star")

    assert summary["source_kind"] == "single_star"
    assert summary["separation_as"] is None
    assert summary["contrast"] is None
    assert summary["x_position_as"] == pytest.approx(0.0)


def test_config_driven_system_resolution_can_change_psf_npix(tmp_path: Path) -> None:
    module = _load_module()
    config_path = _write_config(tmp_path / "portrait.json", _config_payload(psf_npix=32))

    system_cfg = module._resolve_system_cfg(
        config_path=config_path,
        system_preset="SHERA_FLIGHT_3P",
    )

    assert system_cfg["preset"] == "SHERA_FLIGHT_3P"
    assert system_cfg["optics"]["psf_npix"] == 32


def test_detector_layers_are_config_driven(tmp_path: Path) -> None:
    module = _load_module()
    config_path = _write_config(
        tmp_path / "portrait_layers.json",
        _config_payload(psf_npix=32, detector_layers=[]),
    )

    system_cfg = module._resolve_system_cfg(
        config_path=config_path,
        system_preset="SHERA_FLIGHT_3P",
    )
    system_cfg = module._prepare_target_system_cfg(
        system_cfg,
        target_key="ALPHA_CEN",
    )

    assert system_cfg["detector"]["layers"] == []


def test_normalized_flux_mode_sets_store_log_flux_total(tmp_path: Path) -> None:
    module = _load_module()
    config_path = _write_config(tmp_path / "portrait.json", _config_payload(psf_npix=32))
    base_system_cfg = module._resolve_system_cfg(
        config_path=config_path,
        system_preset="SHERA_FLIGHT_3P",
    )
    system_cfg = module._prepare_target_system_cfg(
        base_system_cfg,
        target_key="ALPHA_CEN",
    )

    _, store = module._build_forward_store(system_cfg, normalize_total_flux=True)

    assert float(np.asarray(store.get("source.log_flux_total"))) == pytest.approx(0.0)


def test_artifact_writing_smoke(tmp_path: Path) -> None:
    module = _load_module()
    config_path = _write_config(tmp_path / "portrait.json", _config_payload(psf_npix=8))
    resolved_system_cfg = module._resolve_system_cfg(
        config_path=config_path,
        system_preset="SHERA_FLIGHT_3P",
    )
    system_cfg = module._prepare_target_system_cfg(
        resolved_system_cfg,
        target_key="ALPHA_CEN",
    )
    _, store = module._build_forward_store(system_cfg, normalize_total_flux=False)
    image = np.linspace(1.0, 64.0, 64, dtype=float).reshape(8, 8)

    paths = module._write_artifacts(
        outdir=tmp_path / "portrait",
        target_key="ALPHA_CEN",
        spec=TARGET_SPECS["ALPHA_CEN"],
        image=image,
        store=store,
        config_path=config_path,
        system_preset="SHERA_FLIGHT_3P",
        resolved_system_preset=str(resolved_system_cfg.get("preset")),
        normalize_total_flux=False,
        stretch="sqrt",
        vmin=1.0,
        vmax=64.0,
        timestamp="20260410-010203",
        created_at="2026-04-10T01:02:03",
    )

    assert paths["psf_fits"].is_file()
    assert paths["clean_png"].is_file()
    assert paths["annotated_png"].is_file()
    assert paths["manifest_json"].is_file()

    with fits.open(paths["psf_fits"]) as hdul:
        assert hdul[0].data.shape == image.shape

    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["target_key"] == "ALPHA_CEN"
    assert manifest["stretch"] == "sqrt"
    assert manifest["artifacts"]["psf_fits"] == paths["psf_fits"].name
    assert manifest["config_path"] == str(config_path)
    assert manifest["requested_system_preset"] == "SHERA_FLIGHT_3P"
    assert manifest["resolved_system_preset"] == "SHERA_FLIGHT_3P"
    assert manifest["resolved"]["psf_npix"] == 8


def test_write_fits_normalizes_non_ascii_target_name(tmp_path: Path) -> None:
    module = _load_module()
    output_path = tmp_path / "xi_boo.fits"
    summary = {
        "separation_as": 4.690,
        "position_angle_deg": 286.461,
        "contrast": 9.4,
        "log_flux_total": 8.2,
        "plate_scale_as_per_pix": 0.123,
        "psf_npix": 8,
    }

    module._write_fits(
        output_path=output_path,
        image=np.ones((8, 8)),
        target_key="XI_BOO",
        spec=TARGET_SPECS["XI_BOO"],
        summary=summary,
    )

    with fits.open(output_path) as hdul:
        assert hdul[0].header["NAME"] == "Xi Bootis"
