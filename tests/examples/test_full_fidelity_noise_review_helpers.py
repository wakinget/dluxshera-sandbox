from __future__ import annotations

import numpy as np

from dluxshera.utils import full_fidelity_review as review


def test_noise_review_helper_uses_resolved_system_render(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_render(system_cfg, *, psf_npix=None):
        calls.append({"system_cfg": dict(system_cfg), "psf_npix": psf_npix})
        return np.ones((256, 256), dtype=float) * 20.0

    monkeypatch.setattr(review, "_render_truth_system_image", fake_render)
    result = review.render_noise_review_images(
        {
            "experiment": {
                "subblocks": {
                    "exposure_time_s": 2.0,
                    "noise": {
                        "enabled": True,
                        "shot_noise": True,
                        "read_noise": True,
                        "read_noise_electrons": 2.0,
                        "dark_current": False,
                    }
                }
            }
        },
        {"optics": {"psf_npix": 256}, "source": {"exposure_time_s": 1.0}, "detector": {"model": "HWK4123"}},
        seed=5,
        display_crop_npix=160,
    )

    assert calls
    assert calls[0]["psf_npix"] == 256
    assert result["available"] is True
    assert result["source"] == "resolved_truth_system_binder"
    assert result["noiseless"].shape == (256, 256)
    assert result["display"]["noiseless"].shape == (160, 160)
    assert result["render_shape"] == (256, 256)
    assert result["display_shape"] == (160, 160)
    assert result["render_noise"]["read_noise_electrons"] == 2.0
    assert result["render_noise"]["exposure_time_s"] == 2.0
    assert result["render_noise"]["exposure_time_s_source"] == "translated_config.experiment.subblocks.exposure_time_s"
    assert result["variance_diagnostics"]["mean_expected_variance"] > 20.0


def test_resolve_review_psf_npix_uses_config_value() -> None:
    value, prov = review.resolve_review_psf_npix({"optics": {"psf_npix": 256}})
    assert value == 256
    assert prov["source_field_path"] == "truth_system.optics.psf_npix"
    assert prov["minimum_enforced"] is False


def test_resolve_review_psf_npix_enforces_minimum() -> None:
    value, prov = review.resolve_review_psf_npix({"optics": {"psf_npix": 16}}, minimum=160)
    assert value == 160
    assert prov["requested_value"] == 16
    assert prov["minimum_enforced"] is True
    assert prov["warnings"]


def test_resolve_review_psf_npix_fallback_is_256() -> None:
    value, prov = review.resolve_review_psf_npix({}, minimum=160, default=256)
    assert value == 256
    assert prov["source_field_path"] == "default"


def test_exposure_time_resolver_prefers_subblock_and_reports_candidates() -> None:
    value, prov = review.resolve_noise_review_exposure_time_s(
        {"experiment": {"subblocks": {"exposure_time_s": 0.05}, "system": {"source": {"exposure_time_s": 1.0}}}},
        {"source": {"exposure_time_s": 1.0}},
    )
    assert value == 0.05
    assert prov["source_field_path"] == "translated_config.experiment.subblocks.exposure_time_s"
    assert len(prov["all_candidate_values"]) >= 3
    assert prov["warnings"]


def test_dark_current_variance_scales_with_resolved_exposure(monkeypatch) -> None:
    def fake_render(system_cfg, *, psf_npix=None):
        return np.ones((256, 256), dtype=float) * 10.0

    monkeypatch.setattr(review, "_render_truth_system_image", fake_render)
    result = review.render_noise_review_images(
        {
            "experiment": {
                "subblocks": {
                    "exposure_time_s": 2.0,
                    "noise": {
                        "enabled": True,
                        "shot_noise": False,
                        "read_noise": False,
                        "dark_current": True,
                        "dark_current_e_per_s": 3.0,
                    },
                }
            }
        },
        {"optics": {"psf_npix": 256}, "source": {"exposure_time_s": 1.0}, "detector": {"model": "HWK4123"}},
        seed=1,
        display_crop_npix=None,
    )
    assert result["variance_diagnostics"]["mean_expected_variance"] == 6.0
    assert result["variance_diagnostics"]["expected_dark_electrons_per_pix"] == 6.0


def test_shot_variance_scales_with_model_counts(monkeypatch) -> None:
    def fake_render(system_cfg, *, psf_npix=None):
        return np.ones((256, 256), dtype=float) * 12.0

    monkeypatch.setattr(review, "_render_truth_system_image", fake_render)
    result = review.render_noise_review_images(
        {"experiment": {"subblocks": {"exposure_time_s": 1.0, "noise": {"enabled": True, "shot_noise": True}}}},
        {"optics": {"psf_npix": 256}, "source": {"exposure_time_s": 1.0}, "detector": {"model": "HWK4123"}},
        seed=1,
        display_crop_npix=None,
    )
    assert result["variance_diagnostics"]["mean_expected_variance"] == 12.0
    assert result["variance_diagnostics"]["expected_photon_variance_peak"] == 12.0


def test_synthetic_noise_demo_is_not_primary_notebook_evidence() -> None:
    notebook = (review.repo_root() / "examples/notebooks/full_fidelity_resolved_system_review.ipynb").read_text(encoding="utf-8")
    assert "render_noise_review_images" in notebook
    assert "render_tiny_review_images" not in notebook
    assert "noise_demo(" not in notebook
