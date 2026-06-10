from __future__ import annotations

from pathlib import Path

import yaml


TEMPLATE_PATH = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_algorithm_campaign_v1.yaml"
)
SMOKE_TEMPLATE_PATH = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_binary_iterative_smoke.yaml"
)


def test_full_fidelity_campaign_template_loads_and_keeps_design_contract() -> None:
    payload = yaml.safe_load(TEMPLATE_PATH.read_text())
    experiment = payload["experiment"]

    assert set(payload) == {"experiment"}
    assert experiment["kind"] == "full_fidelity_algorithm_campaign"
    assert (
        experiment["schema_version"]
        == "full_fidelity_algorithm_campaign.v1"
    )

    for key in (
        "target",
        "observation",
        "trajectory",
        "smear",
        "spectral_model",
        "optics",
        "detector",
        "noise",
        "active_state",
        "iterative_update",
        "knockdowns",
        "outputs",
    ):
        assert key in experiment

    assert (
        experiment["iterative_update"]["summary_information_scale"]
        == "summed_likelihood"
    )

    detector = experiment["detector"]
    assert detector["pixel_offsets"]["knowledge_error"]["rms_pixel"] == 0.001
    assert (
        detector["flat_field"]["knowledge_error"]["rms_fractional_response"]
        == 0.001
    )

    optics = experiment["optics"]
    assert optics["primary"]["wfe"]["truth"]["rms_opd_nm"] == 20.0
    assert optics["secondary"]["wfe"]["truth"]["rms_opd_nm"] == 20.0

    spectral = experiment["spectral_model"]
    assert spectral["truth"]["n_lambda"] > spectral["inference"]["n_lambda"]
    assert spectral["source_seds"]["mode"] == "target"
    assert spectral["source_seds"]["single_star_default"]["sed_file"] == "alfCenA_SED.dat"
    assert spectral["source_seds"]["binary_target"]["mode"] == "from_source_target"
    assert "source.target" in spectral["source_seds"]["binary_target"]["note"]
    assert spectral["source_seds"]["generic_binary"]["fallback_policy"] == "require_explicit_or_smoke_alpha_cen"
    assert "shared" not in spectral["source_seds"]["binary_target"]["note"].lower()

    truth_components = spectral["truth"]["components"]
    detector_qe = truth_components["detector_qe"]
    m2_filter = truth_components["m2_filter_response"]
    assert detector_qe["path"] == "data/detector_qe/LTN4323_QE.csv"
    assert detector_qe["response_column"] == "QE"
    assert detector_qe["wavelength_column"] == "Wavelength (nm)"
    assert detector_qe["detector_model_proxy_for"] == "HWK4123"
    assert "near-term proxy for HWK4123" in detector_qe["assumption"]
    assert m2_filter["path"] == "data/filter_response/SHERA Notch Filter V2.csv"
    assert m2_filter["response_column"] == "R (%)"
    assert m2_filter["response_unit"] == "percent_reflection"
    assert m2_filter["response_scale"] == 0.01
    assert spectral["inference"]["components"]["detector_qe"]["mode"] == "same_as_truth"
    assert spectral["inference"]["components"]["m2_filter_response"]["mode"] == "same_as_truth"


def test_full_fidelity_binary_iterative_smoke_template_is_tiny_executable_smoke() -> None:
    payload = yaml.safe_load(SMOKE_TEMPLATE_PATH.read_text())
    experiment = payload["experiment"]

    assert experiment["kind"] == "full_fidelity_binary_iterative_smoke"
    assert experiment["source_kind"] == "binary_target"
    assert experiment["target"] == "ALPHA_CEN"
    assert experiment["n_cases"] == 1
    assert experiment["subblocks"]["n_frames"] == 3
    assert experiment["subblocks"]["trace_source"]["mode"] == "trajectory"
    assert experiment["subblocks"]["trajectory_processing"]["smear"]["render"]["mode"] == "metadata_only"
    assert experiment["iterative"]["enabled"] is True
    assert experiment["iterative"]["windows_per_draw"] == 2
    assert experiment["iterative"]["subblocks_per_window"] == 1
    assert experiment["iterative"]["update_safety"]["posterior_sigma_inflation"] == 10.0
    assert experiment["spectral_model"]["truth"]["n_lambda"] > experiment["spectral_model"]["inference"]["n_lambda"]
    assert experiment["high_order_wfe"]["truth"]["npix"] == 16
