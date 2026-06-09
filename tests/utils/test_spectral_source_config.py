from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from dluxshera.utils.spectral_response import (
    build_effective_spectrum,
    build_truth_inference_spectral_deck,
)
from dluxshera.utils.spectral_source_config import (
    apply_effective_spectrum_to_source_config,
    build_spectral_truth_inference_system_configs,
    spectrum_to_source_spectral_config,
)

TEMPLATE_PATH = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_algorithm_campaign_v1.yaml"
)


def _flat_response(label: str = "flat") -> dict[str, object]:
    return {"label": label, "response": 1.0}


def _spectrum(label: str = "truth", n_lambda: int = 5):
    return build_effective_spectrum(
        label=label,
        wavelength_min_nm=500.0,
        wavelength_max_nm=600.0,
        n_lambda=n_lambda,
        sed=lambda wavelengths_m: wavelengths_m * 1e9,
        detector_qe=_flat_response("qe"),
    )


def _base_system(source_kind: str = "binary") -> dict[str, object]:
    source: dict[str, object] = {
        "kind": source_kind,
        "wavelength_m": 550e-9,
        "bandwidth_m": 110e-9,
        "n_lambda": 3,
        "x_position_as": 0.0,
        "y_position_as": 0.0,
        "log_flux_total": 6.5,
    }
    if source_kind != "single_star":
        source.update(
            {
                "separation_as": 10.0,
                "position_angle_deg": 90.0,
                "contrast": 3.0,
            }
        )
    return {
        "source": source,
        "optics": {"kind": "two_plane"},
        "detector": {"model": "none", "layers": []},
    }


def test_spectrum_to_binary_source_config_uses_existing_component_weights() -> None:
    spectrum = _spectrum(n_lambda=7)
    patch = spectrum_to_source_spectral_config(spectrum, source_kind="binary")

    assert patch["n_lambda"] == 7
    assert len(patch["wavelengths_m"]) == 7
    assert "component_weights" in patch
    component_weights = np.asarray(patch["component_weights"])
    assert component_weights.shape == (2, 7)
    np.testing.assert_allclose(component_weights.sum(axis=1), np.ones(2))
    assert patch["spectral_deck_provenance"]["flux_factor_usage"] == "diagnostic_provenance_only"


def test_spectrum_to_single_star_source_config_uses_weights_vector() -> None:
    spectrum = _spectrum(n_lambda=4)
    patch = spectrum_to_source_spectral_config(spectrum, source_kind="single_star")

    assert "weights" in patch
    assert "component_weights" not in patch
    assert np.sum(patch["weights"]) == pytest.approx(1.0)
    assert patch["spectral_deck_provenance"]["component_labels"] == ["star"]


def test_apply_effective_spectrum_to_source_config_does_not_mutate_and_preserves_flux() -> None:
    source_cfg = _base_system("binary")["source"]
    original = deepcopy(source_cfg)
    spectrum = _spectrum(n_lambda=6)

    patched, provenance = apply_effective_spectrum_to_source_config(source_cfg, spectrum)

    assert source_cfg == original
    assert patched is not source_cfg
    assert patched["log_flux_total"] == original["log_flux_total"]
    assert patched["contrast"] == original["contrast"]
    assert patched["n_lambda"] == 6
    assert np.asarray(patched["component_weights"]).sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert provenance["preserved_flux_parameters"] == ["contrast", "log_flux_total"]


def test_spectral_deck_builds_distinct_truth_and_inference_system_configs() -> None:
    deck = build_truth_inference_spectral_deck(
        sed=lambda wavelengths_m: wavelengths_m * 1e9,
        truth_config={"n_lambda": 11, "wavelength_min_nm": 490.0, "wavelength_max_nm": 710.0},
        inference_config={"n_lambda": 5, "wavelength_min_nm": 540.0, "wavelength_max_nm": 660.0},
        detector_qe=_flat_response("qe"),
    )
    base_system = _base_system("binary")
    original = deepcopy(base_system)

    truth_cfg, inference_cfg, provenance = build_spectral_truth_inference_system_configs(
        base_system_cfg=base_system,
        deck=deck,
    )

    assert base_system == original
    assert truth_cfg["source"]["n_lambda"] == 11
    assert inference_cfg["source"]["n_lambda"] == 5
    assert truth_cfg["source"]["wavelengths_m"] != inference_cfg["source"]["wavelengths_m"]
    assert truth_cfg["source"]["log_flux_total"] == original["source"]["log_flux_total"]
    assert inference_cfg["source"]["contrast"] == original["source"]["contrast"]
    assert provenance["truth"]["spectrum"]["n_lambda"] == 11
    assert provenance["inference"]["spectrum"]["n_lambda"] == 5
    assert provenance["truth"]["spectrum"]["spectrum_label"] == "truth"
    assert provenance["inference"]["spectrum"]["spectrum_label"] == "inference"
    assert provenance["truth"]["spectrum"]["flux_factor"] == pytest.approx(deck.truth.flux_factor)
    assert provenance["inference"]["spectrum"]["lambda_eff_nm"] == pytest.approx(
        deck.inference.diagnostics["lambda_eff_nm"]
    )


def test_outer_system_wrapper_is_preserved() -> None:
    deck = build_truth_inference_spectral_deck(
        sed=1.0,
        truth_config={"n_lambda": 9, "wavelength_min_nm": 500.0, "wavelength_max_nm": 700.0},
        inference_config={"n_lambda": 3, "wavelength_min_nm": 540.0, "wavelength_max_nm": 660.0},
        detector_qe=_flat_response("qe"),
    )
    wrapped = {"system": _base_system("single_star"), "experiment": {"kind": "noop"}}

    truth_cfg, inference_cfg, _provenance = build_spectral_truth_inference_system_configs(
        base_system_cfg=wrapped,
        deck=deck,
    )

    assert set(truth_cfg) == {"system", "experiment"}
    assert truth_cfg["experiment"] == wrapped["experiment"]
    assert "weights" in truth_cfg["system"]["source"]
    assert inference_cfg["system"]["source"]["n_lambda"] == 3


def test_full_fidelity_template_spectral_model_can_patch_synthetic_system_config() -> None:
    payload = yaml.safe_load(TEMPLATE_PATH.read_text())
    spectral = payload["experiment"]["spectral_model"]
    deck = build_truth_inference_spectral_deck(
        sed=1.0,
        truth_config=spectral["truth"],
        inference_config=spectral["inference"],
        detector_qe=_flat_response("qe"),
        filter_response=_flat_response("filter"),
    )

    truth_cfg, inference_cfg, provenance = build_spectral_truth_inference_system_configs(
        base_system_cfg=_base_system("binary_target"),
        deck=deck,
    )

    assert truth_cfg["source"]["n_lambda"] == spectral["truth"]["n_lambda"]
    assert inference_cfg["source"]["n_lambda"] == spectral["inference"]["n_lambda"]
    assert np.asarray(truth_cfg["source"]["component_weights"]).sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert provenance["active_inference_parameters_added"] == []
    assert provenance["deck_comparison"]["truth_out_of_inference_band_fraction"] >= 0.0


def test_unsupported_source_kind_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported source kind"):
        spectrum_to_source_spectral_config(_spectrum(), source_kind="planet")
