from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dluxshera.inference.observation_belief import SubblockSummary
from dluxshera.inference.observation_forecast import (
    PriorContext,
    build_default_prior_sigma,
    require_identical_summary_theta_labels,
    resolve_prior_context_for_summaries,
    summarize_summary_theta_ref_compatibility,
)


def _summary(
    *,
    subblock_id: str = "summary_a",
    theta_labels: tuple[str, ...] = ("source.separation_as", "source.contrast"),
    theta_ref: tuple[float, ...] = (1.25, 3.5),
) -> SubblockSummary:
    return SubblockSummary.from_reduced_form(
        subblock_id=subblock_id,
        theta_labels=theta_labels,
        theta_ref=np.asarray(theta_ref, dtype=float),
        reduced_information=np.eye(len(theta_labels)),
        reduced_score=np.zeros(len(theta_labels)),
        summary_kind="image_backed_schur",
    )


def _write_summary_payload(tmp_path: Path, *, stem: str = "summary") -> Path:
    path = tmp_path / f"{stem}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "image_backed_subblock_summary.v1",
                "theta_labels": ["source.separation_as", "source.contrast"],
                "theta_ref": [1.25, 3.5],
                "reduced_information": [[1.0, 0.0], [0.0, 1.0]],
                "reduced_score": [0.0, 0.0],
                "prior_context": {
                    "recommended_prior_mean_source": "summary_theta_ref",
                    "case": stem,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def test_require_identical_summary_theta_labels_accepts_matching_labels():
    summaries = [_summary(subblock_id="a"), _summary(subblock_id="b")]

    assert require_identical_summary_theta_labels(summaries) == (
        "source.separation_as",
        "source.contrast",
    )


def test_require_identical_summary_theta_labels_rejects_mismatched_labels():
    first = _summary(subblock_id="a")
    second = _summary(
        subblock_id="b",
        theta_labels=("source.separation_as", "source.log_flux_total"),
        theta_ref=(1.25, 7.0),
    )

    with pytest.raises(ValueError, match="identical theta_labels"):
        require_identical_summary_theta_labels([first, second])


def test_summarize_summary_theta_ref_compatibility_reports_identical_values():
    summaries = [_summary(subblock_id="a"), _summary(subblock_id="b")]

    compatibility = summarize_summary_theta_ref_compatibility(summaries)

    assert compatibility["all_equal_within_tolerance"] is True
    assert compatibility["warnings"] == []
    assert compatibility["first_summary_theta_ref"]["source.separation_as"] == (
        pytest.approx(1.25)
    )


def test_summarize_summary_theta_ref_compatibility_reports_spread_and_warning():
    summaries = [
        _summary(subblock_id="a", theta_ref=(1.25, 3.5)),
        _summary(subblock_id="b", theta_ref=(1.30, 3.0)),
    ]

    compatibility = summarize_summary_theta_ref_compatibility(summaries)

    assert compatibility["all_equal_within_tolerance"] is False
    assert compatibility["max_abs_spread_by_label"]["source.separation_as"] == (
        pytest.approx(0.05)
    )
    assert compatibility["max_abs_spread_by_label"]["source.contrast"] == (
        pytest.approx(0.5)
    )
    assert compatibility["warnings"]


def test_resolve_prior_context_for_summaries_defaults_to_summary_theta_ref(
    tmp_path: Path,
):
    summary = _summary()
    summary_path = _write_summary_payload(tmp_path)

    context = resolve_prior_context_for_summaries(
        [summary],
        summary_paths=[summary_path],
        prior_source="auto",
    )

    assert isinstance(context, PriorContext)
    assert context.theta_labels == ("source.separation_as", "source.contrast")
    assert context.prior_mean_source == "summary_theta_ref"
    np.testing.assert_allclose(context.prior_mean, np.array([1.25, 3.5]))
    assert context.provenance["summary_paths"] == [str(summary_path)]
    assert "summary_theta_ref_compatibility" in context.provenance
    assert context.provenance["recommended_prior_context"]["case"] == "summary"
    assert context.warnings == ()


def test_resolve_prior_context_records_theta_ref_spread_warning(tmp_path: Path):
    summaries = [
        _summary(subblock_id="a", theta_ref=(1.25, 3.5)),
        _summary(subblock_id="b", theta_ref=(1.30, 3.0)),
    ]
    paths = [
        _write_summary_payload(tmp_path, stem="summary_a"),
        _write_summary_payload(tmp_path, stem="summary_b"),
    ]

    context = resolve_prior_context_for_summaries(
        summaries,
        summary_paths=paths,
        prior_source="auto",
    )

    compatibility = context.provenance["summary_theta_ref_compatibility"]
    assert context.prior_mean_source == "summary_theta_ref"
    assert compatibility["all_equal_within_tolerance"] is False
    assert context.warnings


def test_build_default_prior_sigma_returns_positive_finite_values():
    labels = (
        "source.separation_as",
        "source.log_flux_total",
        "source.contrast",
        "optics.plate_scale_as_per_pix",
        "optics.primary.zernike_coeffs_nm[3]",
        "custom.parameter",
    )

    sigma = build_default_prior_sigma(labels)

    assert sigma.shape == (len(labels),)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0.0)
    assert sigma[0] == pytest.approx(0.1)
    assert sigma[3] == pytest.approx(0.001)
    assert sigma[4] == pytest.approx(3.0)
    assert sigma[5] == pytest.approx(1.0)
