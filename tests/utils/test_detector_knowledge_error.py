from __future__ import annotations

import numpy as np
import pytest

from dluxshera.builders.detector import build_detector
from dluxshera.utils.detector_knowledge_error import (
    apply_campaign_detector_calibration_knowledge_error,
    normalize_detector_calibration_knowledge_error,
)


def _system(tmp_path):
    dx = np.zeros((4, 4), dtype=float)
    dy = np.zeros((4, 4), dtype=float)
    prf = np.ones((4, 4), dtype=float)
    dx_path = tmp_path / "dx.npy"
    dy_path = tmp_path / "dy.npy"
    prf_path = tmp_path / "prf.npy"
    np.save(dx_path, dx)
    np.save(dy_path, dy)
    np.save(prf_path, prf)
    return {
        "optics": {"psf_npix": 4, "oversample": 1},
        "detector": {
            "layers": [
                {"name": "pixel_offsets", "kind": "ApplyPixelOffsets", "dx_path": str(dx_path), "dy_path": str(dy_path)},
                {"name": "pixel_response", "kind": "ApplyPixelResponse", "prf_path": str(prf_path)},
            ]
        },
    }


def _request(seed: int = 123) -> dict:
    return {
        "enabled": True,
        "apply_to": "inference",
        "realization_policy": "fixed_per_experiment",
        "seed": seed,
        "pixel_offsets": {"enabled": True, "sigma_pix": 0.001, "distribution": "normal"},
        "pixel_response": {"enabled": True, "sigma_fractional": 0.001, "distribution": "normal"},
    }


def test_disabled_config_is_noop(tmp_path) -> None:
    system = _system(tmp_path)
    truth, inference, provenance, paths = apply_campaign_detector_calibration_knowledge_error(
        truth_system_cfg=system,
        inference_system_cfg=system,
        request={"enabled": False},
        seed_context={"base_seed": 7},
        write_artifacts=False,
    )
    assert truth == system
    assert inference == system
    assert provenance["enabled"] is False
    assert paths == {}


def test_normalization_preserves_explicit_seed_and_rejects_bad_values() -> None:
    normalized = normalize_detector_calibration_knowledge_error(_request(seed=999))
    assert normalized["seed"] == 999
    assert normalized["realization_policy"] == "fixed_per_experiment"
    assert normalized["pixel_offsets"]["sigma_pix"] == 0.001
    with pytest.raises(ValueError, match="non-negative"):
        normalize_detector_calibration_knowledge_error({"enabled": True, "pixel_offsets": {"enabled": True, "sigma_pix": -1}})
    with pytest.raises(ValueError, match="realization_policy"):
        normalize_detector_calibration_knowledge_error({"enabled": True, "realization_policy": "per_frame"})


def test_campaign_level_per_run_policy_is_rejected(tmp_path) -> None:
    system = _system(tmp_path)
    request = _request(seed=123)
    request["realization_policy"] = "per_run"
    with pytest.raises(ValueError, match="per_run"):
        apply_campaign_detector_calibration_knowledge_error(
            truth_system_cfg=system,
            inference_system_cfg=system,
            request=request,
            seed_context={"base_seed": 42},
            write_artifacts=False,
        )


def test_model_split_patch_applies_only_to_inference_and_writes_provenance(tmp_path) -> None:
    system = _system(tmp_path)
    truth, inference, provenance, paths = apply_campaign_detector_calibration_knowledge_error(
        truth_system_cfg=system,
        inference_system_cfg=system,
        request=_request(seed=555),
        seed_context={"base_seed": 42},
        artifact_root=tmp_path / "model_split",
        write_artifacts=True,
    )
    assert "knowledge_error" not in truth["detector"]["layers"][0]
    assert inference["detector"]["layers"][0]["knowledge_error"]["seed"] == 555
    assert inference["detector"]["layers"][1]["knowledge_error"]["scale"] == 0.001
    assert inference["detector"]["layers"][1]["knowledge_error"]["clip_min"] == 0.0
    assert provenance["truth_patched"] is False
    assert provenance["inference_patched"] is True
    assert "detector_knowledge_error_provenance_json" in paths
    assert (tmp_path / "model_split/detector_knowledge_error/detector_knowledge_error_provenance.json").exists()


def test_perturbed_detector_maps_are_repeatable_and_seed_sensitive(tmp_path) -> None:
    system = _system(tmp_path)
    _, inference_a, _, _ = apply_campaign_detector_calibration_knowledge_error(
        truth_system_cfg=system,
        inference_system_cfg=system,
        request=_request(seed=1),
        seed_context={"base_seed": 42},
        write_artifacts=False,
    )
    _, inference_b, _, _ = apply_campaign_detector_calibration_knowledge_error(
        truth_system_cfg=system,
        inference_system_cfg=system,
        request=_request(seed=1),
        seed_context={"base_seed": 42},
        write_artifacts=False,
    )
    _, inference_c, _, _ = apply_campaign_detector_calibration_knowledge_error(
        truth_system_cfg=system,
        inference_system_cfg=system,
        request=_request(seed=2),
        seed_context={"base_seed": 42},
        write_artifacts=False,
    )
    det_a, _ = build_detector({"system": inference_a})
    det_b, _ = build_detector({"system": inference_b})
    det_c, _ = build_detector({"system": inference_c})
    assert det_a.layers["pixel_offsets"].dx_map.shape == (4, 4)
    assert det_a.layers["pixel_response"].pixel_response.shape == (4, 4)
    np.testing.assert_allclose(det_a.layers["pixel_offsets"].dx_map, det_b.layers["pixel_offsets"].dx_map)
    np.testing.assert_allclose(det_a.layers["pixel_response"].pixel_response, det_b.layers["pixel_response"].pixel_response)
    assert not np.allclose(det_a.layers["pixel_offsets"].dx_map, 0.0)
    assert not np.allclose(det_a.layers["pixel_offsets"].dx_map, det_c.layers["pixel_offsets"].dx_map)
    assert np.isfinite(np.asarray(det_a.layers["pixel_response"].pixel_response)).all()
    assert np.asarray(det_a.layers["pixel_response"].pixel_response).min() > 0.0
