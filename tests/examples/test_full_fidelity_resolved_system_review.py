from __future__ import annotations

from pathlib import Path

import numpy as np

from dluxshera.utils import full_fidelity_review as review


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml"


def _context(tmp_path: Path):
    cfg = review.load_smoke_config(CONFIG_PATH)
    ctx = review.build_model_split_from_smoke(cfg, tmp_path / "review", run_label="pytest_review", write_artifacts=True)
    return cfg, ctx


def test_smoke_config_loads() -> None:
    cfg = review.load_smoke_config(CONFIG_PATH)
    assert cfg["experiment"]["kind"] == "full_fidelity_binary_iterative_smoke"
    assert cfg["experiment"]["target"] == "ALPHA_CEN"


def test_translation_produces_observation_bias_config() -> None:
    cfg = review.load_smoke_config(CONFIG_PATH)
    translated = review.translate_smoke_to_observation_bias(cfg, run_name="unit_review")
    exp = translated["experiment"]
    assert exp["kind"] == "observation_bias_campaign"
    assert exp["source_campaign_kind"] == "full_fidelity_binary_iterative_smoke"
    assert exp["system"]["source"]["target"] == "ALPHA_CEN"


def test_model_split_builds_and_source_wavelength_fields_exist(tmp_path: Path) -> None:
    _, ctx = _context(tmp_path)
    split = ctx["model_split"]
    assert split.truth_config_hash
    assert split.inference_config_hash
    for system in (ctx["truth_system_cfg"], ctx["inference_system_cfg"]):
        source = system["source"]
        assert source["wavelength_m"] > 0
        assert source["bandwidth_m"] >= 0
        assert source["n_lambda"] > 0
        assert source["wavelengths_m"]
        assert source["component_weights"]


def test_spectral_weights_sum_to_one_per_component(tmp_path: Path) -> None:
    _, ctx = _context(tmp_path)
    for system in (ctx["truth_system_cfg"], ctx["inference_system_cfg"]):
        arr = review.extract_spectral_arrays(system)
        weights = arr["component_weights"]
        assert weights.ndim == 2
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(weights.shape[0]), atol=1e-12)


def test_flux_parameters_preserved_when_requested(tmp_path: Path) -> None:
    cfg, ctx = _context(tmp_path)
    result = review.preserve_flux_review(
        ctx["base_system_cfg"],
        ctx["truth_system_cfg"],
        ctx["inference_system_cfg"],
        ctx["translated_config"]["experiment"].get("spectral_model"),
    )
    assert result["preserve_flux_parameters"] is True
    assert not any("changed despite" in warning for warning in result["warnings"])
    assert cfg["experiment"]["spectral_model"]["preserve_flux_parameters"] is True


def test_wfe_review_summary_reports_requested_and_measured_rms(tmp_path: Path) -> None:
    _, ctx = _context(tmp_path)
    summary = review.summarize_wfe_artifacts(ctx["model_split"])
    assert summary["enabled"] is True
    for mirror in ("primary", "secondary"):
        item = summary["mirrors"][mirror]
        assert item["requested_truth_rms_nm"] == 0.3
        assert item["requested_knowledge_error_rms_nm"] == 0.1
        assert np.isfinite(item["truth_stats"]["rms_nm"])
        assert np.isfinite(item["knowledge_error_stats"]["rms_nm"])
        assert item["truth_opd_nm"].shape == (16, 16)


def test_detector_layer_summary_handles_absent_calibration_maps(tmp_path: Path) -> None:
    _, ctx = _context(tmp_path)
    summary = review.summarize_detector_config(ctx["truth_system_cfg"])
    assert summary["model"] == "HWK4123"
    assert summary["layers"]
    maps = review.load_detector_calibration_maps(ctx["truth_system_cfg"])
    assert isinstance(maps, dict)


def test_trajectory_review_loads_configured_window(tmp_path: Path) -> None:
    _, ctx = _context(tmp_path)
    trajectory = review.load_trajectory_for_review(ctx["translated_config"])
    assert trajectory["available"] is True
    assert trajectory["summary"]["selected_start_s"] == 60.0
    assert trajectory["summary"]["n_subblocks"] == 2
    assert trajectory["summary"]["n_frames"] == 6


def test_high_pass_diagnostic_returns_finite_arrays(tmp_path: Path) -> None:
    _, ctx = _context(tmp_path)
    trajectory = review.load_trajectory_for_review(ctx["translated_config"])
    hp = review.make_high_pass_trajectory_diagnostic(trajectory, timescale_s=15.0)
    assert hp["available"] is True
    for series in hp["series"].values():
        assert np.all(np.isfinite(series["high_pass"]))
        assert np.isfinite(series["rms_high_pass"])


def test_trace_jitter_comparison_reports_clear_status(tmp_path: Path) -> None:
    _, ctx = _context(tmp_path)
    result = review.compare_trace_jitter_enabled_disabled(ctx["translated_config"])
    assert result["status"] in {"downstream_template_override", "legacy_iid_trace_mode"}
    assert result["conclusion"]
    assert "rms_difference" in result


def test_noise_demo_reproducible_shot_read_combined() -> None:
    first = review.noise_demo(seed=777, shape=(8, 8), read_noise=0.5)
    second = review.noise_demo(seed=777, shape=(8, 8), read_noise=0.5)
    for key in ("shot", "read", "combined", "combined_variance"):
        np.testing.assert_allclose(first[key], second[key])
    assert first["diagnostics"]["read_residual_var"] > 0
    assert first["diagnostics"]["combined_residual_var"] > 0


def test_notebook_noise_review_records_full_render_and_display_crop() -> None:
    notebook = (REPO_ROOT / "examples/notebooks/full_fidelity_resolved_system_review.ipynb").read_text(encoding="utf-8")
    assert "NOISE_REVIEW_MIN_PSF_NPIX = 160" in notebook
    assert "NOISE_REVIEW_DEFAULT_PSF_NPIX = 256" in notebook
    assert "NOISE_REVIEW_DISPLAY_CROP_NPIX = 160" in notebook
    assert "render_noise_review_images" in notebook
    assert "render_tiny_review_images" not in notebook
    assert "Noise render shape:" in notebook
    assert "Display crop shape:" in notebook
    assert "rendered_psf_npix" in notebook
    assert "displayed_crop_npix" in notebook
