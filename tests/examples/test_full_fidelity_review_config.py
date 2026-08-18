from __future__ import annotations

from pathlib import Path

import numpy as np

from dluxshera.config.io import load_config_file
from dluxshera.utils import full_fidelity_review as review

ROOT = Path(__file__).resolve().parents[2] / "examples/recipes/full_fidelity_algorithm_campaign_template"
REVIEW = ROOT / "full_fidelity_binary_iterative_review.yaml"


def _ctx(tmp_path: Path):
    cfg = load_config_file(REVIEW)
    return cfg, review.build_model_split_from_smoke(cfg, tmp_path / "review", run_label="review_config_test", write_artifacts=True)


def test_review_config_has_no_spectral_fast_and_builds(tmp_path: Path) -> None:
    cfg, ctx = _ctx(tmp_path)
    assert "fast" not in cfg["experiment"]["spectral_model"]
    assert ctx["truth_system_cfg"]["source"]["n_lambda"] == 3
    assert ctx["inference_system_cfg"]["source"]["n_lambda"] == 3


def test_review_config_documents_and_validates_spectral_modes(tmp_path: Path) -> None:
    cfg, ctx = _ctx(tmp_path)
    exp = cfg["experiment"]
    assert exp["spectral_model"]["photometry_mode"] == "preserve_detected_flux_parameters"
    assert exp["spectral_model"]["inference"]["out_of_band_response"] == "zero"
    assert exp["spectral_model"]["inference"]["components"]["detector_qe"]["mode"] == "same_as_truth"
    summary = review.summarize_spectral_deck(ctx["model_split"])
    assert summary["warnings"] == []


def test_review_high_order_wfe_error_low_order_projection_removed(tmp_path: Path) -> None:
    _, ctx = _ctx(tmp_path)
    summary = review.summarize_wfe_artifacts(ctx["model_split"])
    for mirror in ("primary", "secondary"):
        coeffs = summary["mirrors"][mirror]["zernike_coefficients_nm"]["error"]
        assert max(abs(v) for v in coeffs.values()) < 1.0e-6


def test_review_structured_noise_translates_to_legacy_with_metadata(tmp_path: Path) -> None:
    cfg, ctx = _ctx(tmp_path)
    sub = ctx["translated_config"]["experiment"]["subblocks"]
    assert cfg["experiment"]["subblocks"]["noise"]["shot_noise"] is True
    assert sub["noise"] == "inherit"
    assert sub["noise_model"]["legacy_runner_flag"] == "inherit"
    assert sub["noise_model"]["separate_term_control"] is True
    assert sub["noise_model"]["normalized"]["enabled"] is True
    assert sub["noise_model"]["normalized"]["shot_noise"] is True
    assert sub["noise_model"]["normalized"]["read_noise"] is True
    assert sub["noise_model"]["normalized"]["dark_current"] is False
    assert sub["noise_model"]["render_template_terms"]["photon_noise"] is True
    assert sub["noise_model"]["render_template_terms"]["read_noise"] is True
    assert sub["noise_model"]["render_template_terms"]["dark_current"] is False


def test_review_early_stopping_and_schedule_fields_survive_translation(tmp_path: Path) -> None:
    _, ctx = _ctx(tmp_path)
    sub = ctx["translated_config"]["experiment"]["subblocks"]
    assert sub["reference_schedule_kind"] == "linear_warmup"
    assert sub["reference_schedule_warmup_steps"] == 10
    assert sub["reference_early_stopping_enabled"] is True
    assert sub["reference_early_stopping_min_iter"] == 20
    assert sub["reference_early_stopping_patience"] == 10


def test_review_trajectory_csv_alias_and_trace_jitter_policy(tmp_path: Path) -> None:
    _, ctx = _ctx(tmp_path)
    traj = review.load_trajectory_for_review(ctx["translated_config"])
    assert traj["available"] is True
    jitter = review.compare_trace_jitter_enabled_disabled(ctx["translated_config"])
    assert jitter["status"] == "downstream_template_override"
    assert jitter["is_additive_to_materialized_trajectory_csv"] is False


def test_review_observation_theta_zernike_mapping_and_prior_kinds(tmp_path: Path) -> None:
    cfg, ctx = _ctx(tmp_path)
    obs = cfg["experiment"]["observation_theta"]["optics"]
    assert obs["primary_zernikes"]["indices"] == "from_system"
    wfe = review.summarize_wfe_artifacts(ctx["model_split"])
    assert wfe["mirrors"]["primary"]["noll_index_mapping"]["Z4"] == 0
    kinds = {v["kind"] for v in cfg["experiment"]["prior_draws"]["sigmas"].values()}
    assert kinds == {"absolute", "fractional"}
