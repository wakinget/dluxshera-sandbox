from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from dluxshera.inference.observation_belief import (
    ObservationBeliefState,
    SubblockSummary,
    update_observation_belief,
)
from dluxshera.inference.observation_forecast import PriorContext


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_observation_summary_simulator.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_observation_summary_simulator_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_summary_json(
    tmp_path: Path,
    *,
    stem: str = "summary",
    theta_labels: tuple[str, ...] = ("source.separation_as", "source.contrast"),
    theta_ref: tuple[float, ...] = (1.25, 3.5),
    reduced_information: np.ndarray | None = None,
    reduced_score: np.ndarray | None = None,
    summary_information_scale: str | None = "summed_likelihood",
) -> Path:
    if reduced_information is None:
        reduced_information = np.array([[4.0, 0.1], [0.1, 2.0]], dtype=float)
    if reduced_score is None:
        reduced_score = np.array([-0.25, 0.1], dtype=float)
    payload = {
        "schema_version": "image_backed_subblock_summary.v1",
        "generator": "unit_test",
        "subblock_id": stem,
        "summary_kind": "image_backed_schur",
        "theta_labels": list(theta_labels),
        "theta_ref": list(theta_ref),
        "reduced_information": np.asarray(reduced_information, dtype=float).tolist(),
        "reduced_score": np.asarray(reduced_score, dtype=float).tolist(),
        "summary_diagnostics": {
            "summary_kind": "image_backed_schur",
            "score_norm": float(np.linalg.norm(reduced_score)),
        },
        "prior_context": {
            "recommended_prior_mean_source": "summary_theta_ref",
        },
    }
    if summary_information_scale is not None:
        payload["information_accounting"] = {
            "summary_information_scale": summary_information_scale,
            "summary_frame_reduce": "sum",
            "summary_subblock_reduce": (
                "mean" if summary_information_scale == "optimizer" else "sum"
            ),
        }
    path = tmp_path / f"{stem}_subblock_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_parse_n_subblocks_grid_accepts_comma_separated_values():
    module = _load_script_module()

    assert module.parse_n_subblocks_grid("1,3,10, 30") == (1, 3, 10, 30)


def test_parse_n_subblocks_grid_rejects_duplicate_values():
    module = _load_script_module()

    with pytest.raises(ValueError, match="duplicate"):
        module.parse_n_subblocks_grid("1,3,3")


@pytest.mark.parametrize(
    "raw, match",
    [
        ("", "at least one"),
        ("1,nope", "non-integer"),
        ("0,1", "positive"),
        ("-1,1", "positive"),
    ],
)
def test_parse_n_subblocks_grid_rejects_invalid_inputs(raw: str, match: str):
    module = _load_script_module()

    with pytest.raises(ValueError, match=match):
        module.parse_n_subblocks_grid(raw)


def test_parse_simulator_mode_accepts_supported_modes():
    module = _load_script_module()

    assert module.parse_simulator_mode("replicate") == "replicate"
    assert (
        module.parse_simulator_mode("fixed_information_score_noise")
        == "fixed_information_score_noise"
    )


def test_parse_simulator_mode_rejects_unsupported_mode():
    module = _load_script_module()

    with pytest.raises(ValueError, match="Unsupported"):
        module.parse_simulator_mode("bootstrap_real_summaries")


def test_replicate_mode_repeats_one_input_summary():
    module = _load_script_module()
    summary = SubblockSummary.from_reduced_form(
        subblock_id="source_a",
        theta_labels=("source.separation_as",),
        theta_ref=np.array([1.0]),
        reduced_information=np.array([[2.0]]),
        reduced_score=np.array([0.0]),
    )

    batch = module.replicate_summaries([summary], n_subblocks=4)

    assert batch.source_indices == (0, 0, 0, 0)
    assert [item.subblock_id for item in batch.summaries] == [
        "source_a",
        "source_a",
        "source_a",
        "source_a",
    ]
    assert batch.provenance["counts_by_source_summary_index"][0]["count"] == 4


def test_replicate_mode_tiles_multiple_input_summaries():
    module = _load_script_module()
    summaries = [
        SubblockSummary.from_reduced_form(
            subblock_id=f"source_{index}",
            theta_labels=("source.separation_as",),
            theta_ref=np.array([1.0]),
            reduced_information=np.array([[2.0 + index]]),
            reduced_score=np.array([0.0]),
        )
        for index in range(2)
    ]

    batch = module.replicate_summaries(summaries, n_subblocks=5)

    assert batch.source_indices == (0, 1, 0, 1, 0)
    assert [item.subblock_id for item in batch.summaries] == [
        "source_0",
        "source_1",
        "source_0",
        "source_1",
        "source_0",
    ]


def test_truth_vector_construction_supports_theta_ref_prior_explicit_and_offset(
    tmp_path: Path,
):
    module = _load_script_module()
    labels = ("source.separation_as", "source.contrast")
    summary = SubblockSummary.from_reduced_form(
        subblock_id="source_a",
        theta_labels=labels,
        theta_ref=np.array([1.25, 3.5]),
        reduced_information=np.eye(2),
        reduced_score=np.zeros(2),
    )
    prior_context = PriorContext(
        theta_labels=labels,
        prior_mean=np.array([1.0, 4.0]),
        prior_mean_source="unit_test",
        provenance={},
    )

    theta_true, provenance = module.construct_truth_vector(
        theta_labels=labels,
        summaries=[summary],
        prior_context=prior_context,
        truth_mode="theta-ref",
    )
    np.testing.assert_allclose(theta_true, [1.25, 3.5])
    assert provenance["truth_mode"] == "theta-ref"

    theta_true, _ = module.construct_truth_vector(
        theta_labels=labels,
        summaries=[summary],
        prior_context=prior_context,
        truth_mode="prior-mean",
    )
    np.testing.assert_allclose(theta_true, [1.0, 4.0])

    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps({"source.separation_as": 1.1, "source.contrast": 3.2}),
        encoding="utf-8",
    )
    theta_true, _ = module.construct_truth_vector(
        theta_labels=labels,
        summaries=[summary],
        prior_context=prior_context,
        truth_mode="explicit",
        truth_json_path=truth_path,
    )
    np.testing.assert_allclose(theta_true, [1.1, 3.2])

    theta_true, _ = module.construct_truth_vector(
        theta_labels=labels,
        summaries=[summary],
        prior_context=prior_context,
        truth_mode="offset",
        truth_offset="source.separation_as=0.05,source.contrast=-0.25",
    )
    np.testing.assert_allclose(theta_true, [1.30, 3.25])


def test_explicit_truth_json_requires_every_label(tmp_path: Path):
    module = _load_script_module()
    labels = ("source.separation_as", "source.contrast")
    summary = SubblockSummary.from_reduced_form(
        subblock_id="source_a",
        theta_labels=labels,
        theta_ref=np.array([1.25, 3.5]),
        reduced_information=np.eye(2),
        reduced_score=np.zeros(2),
    )
    prior_context = PriorContext(
        theta_labels=labels,
        prior_mean=np.array([1.0, 4.0]),
        prior_mean_source="unit_test",
        provenance={},
    )
    truth_path = tmp_path / "truth_missing.json"
    truth_path.write_text(
        json.dumps({"source.separation_as": 1.1}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing"):
        module.construct_truth_vector(
            theta_labels=labels,
            summaries=[summary],
            prior_context=prior_context,
            truth_mode="explicit",
            truth_json_path=truth_path,
        )


def test_identical_theta_labels_validation_accepts_matching_labels():
    module = _load_script_module()
    summaries = [
        SubblockSummary.from_reduced_form(
            subblock_id=f"summary_{index}",
            theta_labels=("source.separation_as", "source.contrast"),
            theta_ref=np.array([1.0, 3.0]),
            reduced_information=np.eye(2),
            reduced_score=np.zeros(2),
        )
        for index in range(2)
    ]

    assert module.require_identical_theta_labels(summaries) == (
        "source.separation_as",
        "source.contrast",
    )


def test_identical_theta_labels_validation_rejects_mismatched_labels():
    module = _load_script_module()
    first = SubblockSummary.from_reduced_form(
        subblock_id="summary_a",
        theta_labels=("source.separation_as",),
        theta_ref=np.array([1.0]),
        reduced_information=np.eye(1),
        reduced_score=np.zeros(1),
    )
    second = SubblockSummary.from_reduced_form(
        subblock_id="summary_b",
        theta_labels=("source.contrast",),
        theta_ref=np.array([3.0]),
        reduced_information=np.eye(1),
        reduced_score=np.zeros(1),
    )

    with pytest.raises(ValueError, match="identical theta_labels"):
        module.require_identical_theta_labels([first, second])


def test_score_noise_sampling_reproducibility_and_alpha_zero():
    module = _load_script_module()
    information = np.array([[2.0, 0.2], [0.2, 1.0]], dtype=float)

    noise, diagnostics = module.sample_score_noise_from_information(
        information,
        np.random.default_rng(123),
        alpha=0.0,
        eig_floor_abs=0.0,
        eig_floor_rel=1.0e-12,
    )
    np.testing.assert_allclose(noise, np.zeros(2))
    assert diagnostics["sampling_method"] == "eigen_psd_floor"

    first, _ = module.sample_score_noise_from_information(
        information,
        np.random.default_rng(123),
        alpha=1.0,
        eig_floor_abs=0.0,
        eig_floor_rel=1.0e-12,
    )
    second, _ = module.sample_score_noise_from_information(
        information,
        np.random.default_rng(123),
        alpha=1.0,
        eig_floor_abs=0.0,
        eig_floor_rel=1.0e-12,
    )
    third, _ = module.sample_score_noise_from_information(
        information,
        np.random.default_rng(124),
        alpha=1.0,
        eig_floor_abs=0.0,
        eig_floor_rel=1.0e-12,
    )
    np.testing.assert_allclose(first, second)
    assert first.shape == (2,)
    assert not np.allclose(first, third)


def test_score_noise_sampling_floors_small_negative_eigenvalues_and_rejects_large():
    module = _load_script_module()
    small_negative = np.diag([1.0, -1.0e-14])
    noise, diagnostics = module.sample_score_noise_from_information(
        small_negative,
        np.random.default_rng(123),
        alpha=1.0,
        eig_floor_abs=0.0,
        eig_floor_rel=1.0e-12,
    )
    assert noise.shape == (2,)
    assert diagnostics["n_eigenvalues_below_floor"] >= 1

    with pytest.raises(ValueError, match="negative eigenvalues"):
        module.sample_score_noise_from_information(
            np.diag([1.0, -1.0e-3]),
            np.random.default_rng(123),
            alpha=1.0,
            eig_floor_abs=0.0,
            eig_floor_rel=1.0e-12,
        )


def test_score_noise_synthesis_uses_expected_score_and_nested_prefixes():
    module = _load_script_module()
    summary = SubblockSummary.from_reduced_form(
        subblock_id="source_a",
        theta_labels=("source.separation_as", "source.contrast"),
        theta_ref=np.array([1.0, 2.0]),
        reduced_information=np.array([[2.0, 0.5], [0.5, 1.0]], dtype=float),
        reduced_score=np.zeros(2),
    )
    theta_true = np.array([0.75, 2.25], dtype=float)

    batch = module.synthesize_score_noise_summaries(
        [summary],
        n_subblocks=10,
        theta_true=theta_true,
        rng=np.random.default_rng(123),
        trial_id=0,
        score_noise_alpha=0.0,
        eig_floor_abs=0.0,
        eig_floor_rel=1.0e-12,
    )

    expected_score = summary.reduced_information @ (summary.theta_ref - theta_true)
    np.testing.assert_allclose(batch.summaries[0].reduced_score, expected_score)
    assert [item.subblock_id for item in batch.summaries[:3]] == [
        item.subblock_id for item in batch.summaries
    ][:3]
    assert batch.provenance["nested_prefix_policy"].startswith("synthesize_max_grid")

    weak_prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=summary.theta_labels,
        mean=np.array([0.0, 0.0]),
        sigma=np.array([1.0e6, 1.0e6]),
    )
    update = update_observation_belief(weak_prior, batch.summaries[:10])
    np.testing.assert_allclose(update.posterior.mean, theta_true, rtol=1.0e-6)


def test_score_noise_synthesis_tiles_multiple_templates_in_input_order():
    module = _load_script_module()
    summaries = [
        SubblockSummary.from_reduced_form(
            subblock_id=f"source_{index}",
            theta_labels=("source.separation_as",),
            theta_ref=np.array([1.0 + index]),
            reduced_information=np.array([[2.0 + index]]),
            reduced_score=np.zeros(1),
        )
        for index in range(2)
    ]

    batch = module.synthesize_score_noise_summaries(
        summaries,
        n_subblocks=5,
        theta_true=np.array([1.0]),
        rng=np.random.default_rng(123),
        trial_id=2,
        score_noise_alpha=0.0,
        eig_floor_abs=0.0,
        eig_floor_rel=1.0e-12,
    )

    assert batch.source_indices == (0, 1, 0, 1, 0)
    assert [row["source_template_index"] for row in batch.diagnostics] == [
        0,
        1,
        0,
        1,
        0,
    ]


def test_forecast_row_reports_separation_sigma_in_arcsec_and_microarcseconds():
    module = _load_script_module()
    prior = ObservationBeliefState.from_diagonal_prior(
        theta_labels=("source.separation_as",),
        mean=np.array([1.0]),
        sigma=np.array([0.1]),
    )
    summary = SubblockSummary.from_reduced_form(
        subblock_id="summary_a",
        theta_labels=("source.separation_as",),
        theta_ref=np.array([1.0]),
        reduced_information=np.array([[99.0]]),
        reduced_score=np.array([0.0]),
    )
    update = update_observation_belief(prior, [summary])

    row = module.build_forecast_row(
        n_subblocks=1,
        mode="replicate",
        n_input_summaries=1,
        theta_labels=prior.theta_labels,
        prior_sigma=np.array([0.1]),
        update_result=update,
    )

    expected_sigma = update.posterior.sigma()[0]
    assert row["separation_label_found"] is True
    assert row["separation_posterior_sigma_as"] == pytest.approx(expected_sigma)
    assert row["separation_posterior_sigma_uas"] == pytest.approx(
        expected_sigma * 1.0e6
    )
    assert row["separation_posterior_sigma_over_prior_sigma"] == pytest.approx(
        expected_sigma / 0.1
    )


def test_summary_simulator_dry_run_returns_payload_without_writing_outputs(
    tmp_path: Path,
):
    module = _load_script_module()
    summary_path = _write_summary_json(tmp_path)

    result = module.main(
        [
            "--summary-json",
            str(summary_path),
            "--n-subblocks",
            "1,3",
            "--results-root",
            str(tmp_path / "results"),
            "--run-name",
            "dry_run_case",
            "--dry-run",
        ]
    )

    assert result["dry_run"] is True
    assert result["artifacts"] == {}
    assert result["n_subblocks_grid"] == [1, 3]
    assert result["theta_labels"] == ["source.separation_as", "source.contrast"]
    assert result["prior_mean_source"] == "summary_theta_ref"
    assert not (tmp_path / "results" / "dry_run_case").exists()
    assert "manifest_json" in result["planned_artifacts"]
    assert result["manifest"]["prior_sigma_policy"]["base"] == (
        "dluxshera.inference.observation_forecast.build_default_prior_sigma"
    )
    assert result["summary_scale_validation"]["accepted_summary_information_scale"] == (
        "summed_likelihood"
    )


def test_summary_simulator_rejects_optimizer_scale_summary_by_default(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_summary_json(
        tmp_path,
        stem="optimizer_summary",
        summary_information_scale="optimizer",
    )

    with pytest.raises(ValueError, match="optimizer"):
        module.run_observation_summary_simulator(
            summary_paths=[summary_path],
            n_subblocks_grid=(1,),
            results_root=tmp_path / "results",
            run_name="rejected_optimizer_summary",
            dry_run=True,
        )

    result = module.run_observation_summary_simulator(
        summary_paths=[summary_path],
        n_subblocks_grid=(1,),
        results_root=tmp_path / "results",
        run_name="allowed_optimizer_summary",
        allow_optimizer_scale_summaries=True,
        dry_run=True,
    )

    validation = result["summary_scale_validation"]
    assert validation["summary_scale_policy"] == "allow_optimizer"
    assert validation["accepted_summary_information_scale"] == "optimizer"
    assert validation["override_used"] is True


def test_summary_simulator_raises_early_when_separation_label_is_missing(
    tmp_path: Path,
):
    module = _load_script_module()
    summary_path = _write_summary_json(
        tmp_path,
        stem="contrast_only",
        theta_labels=("source.contrast",),
        theta_ref=(3.5,),
        reduced_information=np.array([[2.0]], dtype=float),
        reduced_score=np.array([0.1], dtype=float),
    )

    with pytest.raises(ValueError, match="requires source.separation_as"):
        module.run_observation_summary_simulator(
            summary_paths=[summary_path],
            n_subblocks_grid=(1,),
            results_root=tmp_path / "results",
            run_name="missing_separation_case",
        )

    assert not (tmp_path / "results" / "missing_separation_case").exists()


def test_summary_simulator_main_writes_required_artifacts(tmp_path: Path):
    module = _load_script_module()
    first_path = _write_summary_json(tmp_path, stem="summary_a")
    second_path = _write_summary_json(
        tmp_path,
        stem="summary_b",
        reduced_information=np.array([[3.0, 0.0], [0.0, 1.5]], dtype=float),
        reduced_score=np.array([-0.1, 0.05], dtype=float),
    )

    result = module.main(
        [
            "--summary-json",
            str(first_path),
            str(second_path),
            "--n-subblocks",
            "1,3",
            "--results-root",
            str(tmp_path / "results"),
            "--run-name",
            "artifact_case",
        ]
    )

    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    for key in (
        "manifest_json",
        "forecast_results_csv",
        "posterior_table_by_n_subblocks_csv",
        "cumulative_sigma_history_csv",
        "information_diagnostics_csv",
        "separation_sigma_vs_n_subblocks_png",
        "prior_normalized_sigma_vs_n_subblocks_png",
    ):
        assert key in artifacts
        assert artifacts[key].exists()
        assert artifacts[key].stat().st_size > 0

    manifest = json.loads(artifacts["manifest_json"].read_text(encoding="utf-8"))
    forecast_rows = _read_csv_rows(artifacts["forecast_results_csv"])
    posterior_rows = _read_csv_rows(
        artifacts["posterior_table_by_n_subblocks_csv"]
    )
    information_rows = _read_csv_rows(artifacts["information_diagnostics_csv"])

    assert manifest["mode"] == "replicate"
    assert manifest["prior_sigma_policy"]["base"] == (
        "dluxshera.inference.observation_forecast.build_default_prior_sigma"
    )
    assert manifest["input_summary_paths"] == [
        str(first_path.resolve()),
        str(second_path.resolve()),
    ]
    assert manifest["replicate_mode"]["provenance_by_n_subblocks"]["3"][
        "source_index_sequence"
    ] == [0, 1, 0]
    assert [row["n_subblocks"] for row in forecast_rows] == ["1", "3"]
    assert "separation_posterior_sigma_uas" in forecast_rows[0]
    assert len(posterior_rows) == 4
    assert posterior_rows[0]["units"] == "arcsec"
    assert len(information_rows) == 2
    assert "posterior_precision_rank_estimate" in information_rows[0]


def test_score_noise_mode_writes_stochastic_artifacts(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_summary_json(tmp_path, stem="summary_score_noise")

    result = module.main(
        [
            "--summary-json",
            str(summary_path),
            "--mode",
            "fixed_information_score_noise",
            "--n-subblocks",
            "1,3",
            "--n-trials",
            "3",
            "--seed",
            "123",
            "--score-noise-alpha",
            "1.0",
            "--truth-mode",
            "theta-ref",
            "--results-root",
            str(tmp_path / "results"),
            "--run-name",
            "score_noise_case",
        ]
    )

    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    for key in (
        "manifest_json",
        "trial_forecast_results_csv",
        "forecast_results_csv",
        "trial_posterior_table_csv",
        "stochastic_synthesis_diagnostics_csv",
        "information_diagnostics_csv",
        "separation_error_vs_n_subblocks_png",
    ):
        assert key in artifacts
        assert artifacts[key].exists()
        assert artifacts[key].stat().st_size > 0

    manifest = json.loads(artifacts["manifest_json"].read_text(encoding="utf-8"))
    trial_rows = _read_csv_rows(artifacts["trial_forecast_results_csv"])
    aggregate_rows = _read_csv_rows(artifacts["forecast_results_csv"])
    posterior_rows = _read_csv_rows(artifacts["trial_posterior_table_csv"])

    assert manifest["mode"] == "fixed_information_score_noise"
    assert manifest["stochastic_mode"]["n_trials"] == 3
    assert manifest["stochastic_mode"]["seed"] == 123
    assert manifest["stochastic_mode"]["score_noise_alpha"] == 1.0
    assert manifest["stochastic_mode"]["truth"]["truth_mode"] == "theta-ref"
    assert "source.separation_as" in manifest["stochastic_mode"]["truth"]["truth"]
    assert len(trial_rows) == 6
    assert len(aggregate_rows) == 2
    assert len(posterior_rows) == 12
    assert "separation_truth_as" in trial_rows[0]
    assert "separation_error_uas" in trial_rows[0]
    assert "separation_rms_error_uas" in aggregate_rows[0]
