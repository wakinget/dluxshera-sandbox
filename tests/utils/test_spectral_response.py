from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dluxshera.utils.spectral_response import (
    DEFAULT_DETECTOR_QE_PATH,
    DEFAULT_FILTER_RESPONSE_PATH,
    DETECTOR_QE_PROXY_ASSUMPTION,
    build_effective_spectrum,
    build_truth_inference_spectral_deck,
    interpolate_response_curve,
    load_response_curve_csv,
    resolve_response_curve_path,
    write_spectral_deck_artifacts,
)

TEMPLATE_PATH = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_algorithm_campaign_v1.yaml"
)


def _real_detector_qe() -> dict[str, object]:
    return {
        "label": "LTN4323_QE_proxy_for_HWK4123",
        "path": DEFAULT_DETECTOR_QE_PATH,
        "wavelength_column": "Wavelength (nm)",
        "wavelength_unit": "nm",
        "response_column": "QE",
        "response_unit": "dimensionless",
        "response_scale": 1.0,
        "detector_model_proxy_for": "HWK4123",
        "assumption": DETECTOR_QE_PROXY_ASSUMPTION,
    }


def _real_filter_response() -> dict[str, object]:
    return {
        "label": "SHERA_Notch_Filter_V2",
        "path": DEFAULT_FILTER_RESPONSE_PATH,
        "wavelength_column": "Wavelength (nm)",
        "wavelength_unit": "nm",
        "response_column": "T (%)",
        "response_unit": "percent_transmission",
        "response_scale": 0.01,
    }


def _flat_response(label: str = "flat") -> dict[str, object]:
    return {"label": label, "response": 1.0}


def test_effective_spectrum_normalizes_flat_sed_and_response() -> None:
    spectrum = build_effective_spectrum(
        label="flat",
        wavelength_min_nm=500.0,
        wavelength_max_nm=600.0,
        n_lambda=5,
        sed=1.0,
        detector_qe=_flat_response("qe"),
        filter_response=_flat_response("filter"),
    )

    assert spectrum.wavelengths_m.shape == (5,)
    assert np.isclose(np.sum(spectrum.weights), 1.0)
    assert spectrum.flux_factor > 0.0
    assert np.all(spectrum.raw_response > 0.0)
    assert spectrum.diagnostics["weights_sum"] == pytest.approx(1.0)


def test_spectral_moments_center_symmetric_response_near_550_nm() -> None:
    wavelengths_m = np.linspace(500.0, 600.0, 21) * 1e-9
    center_m = 550e-9

    def gaussian(_wavelengths_m: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * ((_wavelengths_m - center_m) / 12e-9) ** 2)

    spectrum = build_effective_spectrum(
        label="symmetric",
        wavelengths_m=wavelengths_m,
        sed=1.0,
        response_components=[{"label": "gaussian", "callable": gaussian}],
    )

    assert spectrum.diagnostics["lambda_eff_nm"] == pytest.approx(550.0, abs=0.25)
    assert 8.0 < spectrum.diagnostics["bandwidth_rms_nm"] < 15.0
    assert spectrum.diagnostics["peak_wavelength_nm"] == pytest.approx(550.0)


def test_truth_out_of_inference_band_fraction_is_positive() -> None:
    deck = build_truth_inference_spectral_deck(
        sed=1.0,
        truth_config={"n_lambda": 31, "wavelength_min_nm": 480.0, "wavelength_max_nm": 720.0},
        inference_config={"n_lambda": 7, "wavelength_min_nm": 540.0, "wavelength_max_nm": 660.0},
        detector_qe=_flat_response("qe"),
        filter_response=_flat_response("filter"),
    )

    assert deck.comparison["truth_out_of_inference_band_fraction"] > 0.0
    assert deck.truth.diagnostics["out_of_band_fraction"] > 0.0


def test_narrow_band_inference_recomputes_and_normalizes_weights() -> None:
    deck = build_truth_inference_spectral_deck(
        sed=lambda wavelengths_m: wavelengths_m * 1e9,
        truth_config={"n_lambda": 30, "wavelength_min_nm": 500.0, "wavelength_max_nm": 700.0},
        inference_config={"n_lambda": 7, "wavelength_min_nm": 530.0, "wavelength_max_nm": 670.0},
        detector_qe=_flat_response("qe"),
    )

    assert deck.truth.diagnostics["n_lambda"] > deck.inference.diagnostics["n_lambda"]
    assert np.isclose(np.sum(deck.inference.weights), 1.0)
    assert deck.inference.wavelengths_m[0] >= deck.truth.wavelengths_m[0]
    assert deck.inference.wavelengths_m[-1] <= deck.truth.wavelengths_m[-1]
    assert deck.provenance["assumptions"]["inference_spectrum_recomputed_not_sliced"] is True


def test_csv_response_loader_and_interpolation_unit_conversion(tmp_path: Path) -> None:
    path = tmp_path / "response.csv"
    path.write_text("wavelength,response\n500,0.0\n550,0.5\n600,1.0\n")

    wavelengths_m, response = load_response_curve_csv(path, wavelength_unit="nm")
    assert wavelengths_m[0] == pytest.approx(500e-9)
    assert response[-1] == pytest.approx(1.0)

    target_m = np.array([450.0, 525.0, 575.0, 650.0]) * 1e-9
    interp = interpolate_response_curve(target_m, wavelengths_m, response)
    assert interp[0] == pytest.approx(0.0)
    assert interp[-1] == pytest.approx(0.0)
    assert interp[1] == pytest.approx(0.25)
    assert interp[2] == pytest.approx(0.75)


def test_negative_response_values_raise_by_default(tmp_path: Path) -> None:
    path = tmp_path / "negative.csv"
    path.write_text("wavelength,response\n500,0.2\n550,-0.1\n600,0.3\n")

    with pytest.raises(ValueError, match="negative response"):
        load_response_curve_csv(path, wavelength_unit="nm")

    wavelengths_m = np.linspace(500.0, 600.0, 3) * 1e-9
    with pytest.raises(ValueError, match="negative response"):
        build_effective_spectrum(
            label="bad",
            wavelengths_m=wavelengths_m,
            sed=1.0,
            response_components=[{"label": "bad_curve", "response": np.array([1.0, -0.1, 1.0])}],
        )


def test_artifact_writer_emits_expected_files_and_columns(tmp_path: Path) -> None:
    deck = build_truth_inference_spectral_deck(
        sed=1.0,
        truth_config={"n_lambda": 9, "wavelength_min_nm": 500.0, "wavelength_max_nm": 700.0},
        inference_config={"n_lambda": 5, "wavelength_min_nm": 540.0, "wavelength_max_nm": 660.0},
        detector_qe=_flat_response("qe"),
    )
    paths = write_spectral_deck_artifacts(deck, tmp_path / "spectral")

    expected = {
        "truth_weights",
        "inference_weights",
        "spectral_moments",
        "spectral_comparison",
        "spectral_deck_manifest",
    }
    assert set(paths) == expected
    for path in paths.values():
        assert path.is_file()

    manifest = json.loads(paths["spectral_deck_manifest"].read_text())
    assert manifest["schema_version"] == "spectral_throughput_deck.v1"
    assert "detected post-response" in manifest["note"]

    with paths["truth_weights"].open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert {"wavelength_m", "weight", "raw_response"}.issubset(reader.fieldnames)
        rows = list(reader)
    assert len(rows) == deck.truth.diagnostics["n_lambda"]


def test_full_fidelity_template_spectral_model_is_consumable() -> None:
    payload = yaml.safe_load(TEMPLATE_PATH.read_text())
    spectral = payload["experiment"]["spectral_model"]

    deck = build_truth_inference_spectral_deck(
        sed=1.0,
        truth_config=spectral["truth"],
        inference_config=spectral["inference"],
        detector_qe=_flat_response("qe"),
        filter_response=_flat_response("filter"),
        provenance={"source_config": "full_fidelity_algorithm_campaign_v1.yaml::experiment.spectral_model"},
    )

    assert deck.truth.diagnostics["n_lambda"] == spectral["truth"]["n_lambda"]
    assert deck.inference.diagnostics["n_lambda"] == spectral["inference"]["n_lambda"]
    assert np.isclose(deck.truth.weights.sum(), 1.0)
    assert np.isclose(deck.inference.weights.sum(), 1.0)
    assert deck.schema_version == "spectral_throughput_deck.v1"


def test_real_response_curve_paths_resolve_to_packaged_data() -> None:
    filter_path = resolve_response_curve_path(DEFAULT_FILTER_RESPONSE_PATH)
    qe_path = resolve_response_curve_path(DEFAULT_DETECTOR_QE_PATH)

    assert str(filter_path).endswith("src/dluxshera/data/filter_response/SHERA Notch Filter V2.csv")
    assert str(qe_path).endswith("src/dluxshera/data/detector_qe/LTN4323_QE.csv")
    assert filter_path.is_file()
    assert qe_path.is_file()


def test_real_filter_and_detector_qe_csvs_load_and_interpolate() -> None:
    filter_wavelengths, filter_response = load_response_curve_csv(
        DEFAULT_FILTER_RESPONSE_PATH,
        wavelength_column="Wavelength (nm)",
        response_column="T (%)",
        wavelength_unit="nm",
        response_scale=0.01,
    )
    qe_wavelengths, qe_response = load_response_curve_csv(
        DEFAULT_DETECTOR_QE_PATH,
        wavelength_column="Wavelength (nm)",
        response_column="QE",
        wavelength_unit="nm",
    )

    assert filter_wavelengths.size > 10
    assert qe_wavelengths.size > 10
    assert np.all(np.isfinite(filter_response))
    assert np.all(np.isfinite(qe_response))
    assert np.min(filter_response) >= 0.0
    assert np.max(filter_response) <= 1.0
    assert np.min(qe_response) >= 0.0
    assert np.max(qe_response) <= 1.0

    target = np.linspace(500.0, 700.0, 9) * 1e-9
    interp_filter = interpolate_response_curve(target, filter_wavelengths, filter_response)
    interp_qe = interpolate_response_curve(target, qe_wavelengths, qe_response)
    assert np.all(np.isfinite(interp_filter))
    assert np.all(np.isfinite(interp_qe))
    assert np.all(interp_filter >= 0.0)
    assert np.all(interp_qe >= 0.0)


def test_real_response_curves_build_normalized_spectral_deck() -> None:
    deck = build_truth_inference_spectral_deck(
        sed=lambda wavelengths_m: wavelengths_m * 1e9,
        truth_config={"n_lambda": 30, "wavelength_min_nm": 500.0, "wavelength_max_nm": 700.0},
        inference_config={"n_lambda": 7, "wavelength_min_nm": 540.0, "wavelength_max_nm": 660.0},
        detector_qe=_real_detector_qe(),
        filter_response=_real_filter_response(),
    )

    assert deck.truth.flux_factor > 0.0
    assert deck.inference.flux_factor > 0.0
    assert np.isclose(deck.truth.weights.sum(), 1.0)
    assert np.isclose(deck.inference.weights.sum(), 1.0)
    assert np.isfinite(deck.truth.diagnostics["lambda_eff_nm"])
    assert np.isfinite(deck.inference.diagnostics["lambda_eff_nm"])
    truth_components = deck.truth.provenance["response_components"]
    assert any(component.get("label") == "LTN4323_QE_proxy_for_HWK4123" for component in truth_components)
    assert any(component.get("response_column") == "T (%)" for component in truth_components)
    assert any(component.get("detector_model_proxy_for") == "HWK4123" for component in truth_components)
