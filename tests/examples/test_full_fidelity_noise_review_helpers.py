from __future__ import annotations

import numpy as np

from dluxshera.utils import full_fidelity_review as review


def test_noise_review_helper_uses_resolved_system_render(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_render(system_cfg):
        calls.append(dict(system_cfg))
        return np.ones((8, 8), dtype=float) * 20.0

    monkeypatch.setattr(review, "_render_truth_system_image", fake_render)
    result = review.render_tiny_review_images(
        {
            "experiment": {
                "subblocks": {
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
        {"source": {"exposure_time_s": 1.0}, "detector": {"model": "HWK4123"}},
        seed=5,
        crop_npix=None,
    )

    assert calls
    assert result["available"] is True
    assert result["source"] == "resolved_truth_system_binder"
    assert result["noiseless"].shape == (8, 8)
    assert result["render_noise"]["read_noise_electrons"] == 2.0
    assert result["variance_diagnostics"]["mean_expected_variance"] > 20.0


def test_synthetic_noise_demo_is_not_primary_notebook_evidence() -> None:
    notebook = (review.repo_root() / "examples/notebooks/full_fidelity_resolved_system_review.ipynb").read_text(encoding="utf-8")
    assert "render_tiny_review_images" in notebook
    assert "noise_demo(" not in notebook
