from __future__ import annotations

from pathlib import Path

import yaml


TEMPLATE_PATH = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_algorithm_campaign_v1.yaml"
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
