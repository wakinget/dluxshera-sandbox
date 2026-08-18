from __future__ import annotations

from pathlib import Path

import numpy as np

from dluxshera.utils import full_fidelity_review as review


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml"


def _summary(tmp_path: Path) -> dict:
    cfg = review.load_smoke_config(CONFIG_PATH)
    ctx = review.build_model_split_from_smoke(
        cfg,
        tmp_path / "review",
        run_label="wfe_review_helper_test",
        write_artifacts=True,
    )
    return review.summarize_wfe_artifacts(ctx["model_split"])


def test_wfe_review_summary_exposes_decomposition_and_sum_check(tmp_path: Path) -> None:
    summary = _summary(tmp_path)

    assert summary["enabled"] is True
    for mirror in ("primary", "secondary"):
        item = summary["mirrors"][mirror]
        mask = item["mask"]
        raw = item["raw_ptt_removed_truth_opd_nm"]
        low = item["low_order_truth_reconstruction_nm"]
        truth_residual = item["truth_high_order_residual_opd_nm"]
        error_residual = item["knowledge_error_high_order_residual_opd_nm"]
        inference = item["inference_high_order_opd_nm"]

        np.testing.assert_allclose(raw[mask], low[mask] + truth_residual[mask], atol=1.0e-10)
        np.testing.assert_allclose(inference[mask], truth_residual[mask] + error_residual[mask], atol=1.0e-10)
        assert np.max(np.abs(item["inference_sum_residual_nm"][mask])) < 1.0e-10
        assert item["rms_nm"]["inference_sum_residual"] < 1.0e-10


def test_wfe_review_summary_separates_stored_coefficients_from_residual_projection(tmp_path: Path) -> None:
    summary = _summary(tmp_path)

    for mirror in ("primary", "secondary"):
        item = summary["mirrors"][mirror]
        stored = item["stored_low_order_coefficients_nm"]
        projection = item["residual_low_order_projection_nm"]
        labels = [f"Z{i}" for i in range(4, 12)]

        assert list(stored["truth"]) == labels
        assert item["noll_index_mapping"]["Z4"] == 0
        assert item["coefficient_array_index_mapping"][0]["label"] == "Z4"
        assert max(abs(projection["truth_high_order_residual"][key]) for key in labels) < 1.0e-8
        assert max(abs(projection["knowledge_error_residual"][key]) for key in labels) < 1.0e-8
        assert max(abs(projection["inference_high_order"][key]) for key in labels) < 1.0e-8
        assert all(np.isfinite(stored["truth"][key]) for key in labels)
        assert all(np.isfinite(stored["inference"][key]) for key in labels)
        assert all(np.isfinite(stored["error"][key]) for key in labels)
        assert max(abs(stored["truth"][key]) for key in labels) > 1.0e-6


def test_masked_plotting_helpers_set_bad_pixels_to_grey() -> None:
    arr = np.arange(9, dtype=float).reshape(3, 3)
    mask = np.array(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ],
        dtype=bool,
    )

    masked = review.masked_for_imshow(arr, mask)
    assert np.isnan(masked[0, 0])
    assert masked[1, 1] == arr[1, 1]
    assert review.symmetric_nan_limits(masked, percentile=100.0) == (-7.0, 7.0)

    cmap = review.cmap_with_bad("RdBu_r", bad="0.5")
    bad_rgba = cmap(np.ma.masked_invalid([np.nan]))[0]
    np.testing.assert_allclose(bad_rgba[:3], np.array([0.5, 0.5, 0.5]), atol=0.01)
