from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from dluxshera.inference.observation_belief import SubblockSummary
from dluxshera.inference.observation_summary import (
    ImageBackedSubblockSummaryArtifact,
    build_combined_local_parameter_layout,
    partition_local_curvature,
    schur_reduce_local_quadratic,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "run_observation_belief_update_demo.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "run_observation_belief_update_demo_script",
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_prior_override_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "prior_override_config.json"
    _write_json(
        config_path,
        {
            "system": {
                "source": {
                    "separation_as": 11.25,
                    "contrast": 4.5,
                    "exposure_time_s": 1.0,
                },
                "optics": {
                    "throughput": 1.0e-6,
                    "primary_noll_indices": [4, 5],
                    "secondary_noll_indices": [4, 5],
                },
            }
        },
    )
    return config_path


def _write_real_summary_artifact(tmp_path: Path) -> Path:
    layout = build_combined_local_parameter_layout(
        (
            "theta.source.separation_as",
            "theta.source.log_flux_total",
            "theta.source.contrast",
            "theta.optics.plate_scale_as_per_pix",
        ),
        (
            "phi.frame[0].source.x_position_as",
            "phi.frame[0].source.y_position_as",
        ),
    )
    gradient = np.array([-2.0, 0.15, -0.05, 0.25, 0.1, -0.2], dtype=float)
    curvature = np.array(
        [
            [5.0, 0.2, 0.1, 0.1, 0.7, 0.0],
            [0.2, 4.5, 0.3, -0.1, -0.2, 0.4],
            [0.1, 0.3, 2.5, 0.05, 0.1, -0.1],
            [0.1, -0.1, 0.05, 4.0, 0.0, 0.2],
            [0.7, -0.2, 0.1, 0.0, 3.0, 0.1],
            [0.0, 0.4, -0.1, 0.2, 0.1, 2.2],
        ],
        dtype=float,
    )
    blocks = partition_local_curvature(
        layout=layout,
        combined_gradient=gradient,
        combined_curvature=curvature,
    )
    reduced = schur_reduce_local_quadratic(blocks=blocks)
    summary = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=(
            "source.separation_as",
            "source.log_flux_total",
            "source.contrast",
            "optics.plate_scale_as_per_pix",
        ),
        theta_ref=np.array([11.25, 7.01, 3.4, 0.01]),
        reduced_information=reduced.reduced_information,
        reduced_score=reduced.reduced_score,
        summary_kind="image_backed_schur",
    )
    artifact = ImageBackedSubblockSummaryArtifact(
        summary=summary,
        layout=layout,
        theta_ref=np.array([11.25, 7.01, 3.4, 0.01]),
        phi_ref=np.array([0.1, -0.2]),
        reduced=reduced,
        metadata={
            "generator": "unit_test",
            "information_accounting": {
                "summary_information_scale": "summed_likelihood",
                "summary_frame_reduce": "sum",
                "summary_subblock_reduce": "sum",
            },
            "prior_context": {
                "recommended_prior_mean_source": "summary_theta_ref",
                "theta_ref_by_label": {
                    "source.separation_as": 11.25,
                    "source.log_flux_total": 7.01,
                    "source.contrast": 3.4,
                    "optics.plate_scale_as_per_pix": 0.01,
                },
                "effective_store_values": {
                    "source.exposure_time_s": 0.05,
                },
            },
        },
        combined_gradient=gradient,
        combined_curvature=curvature,
    )
    summary_path = tmp_path / "real_subblock_summary.json"
    matrix_path = tmp_path / "real_subblock_summary_matrices.npz"
    artifact.write(summary_json_path=summary_path, matrix_npz_path=matrix_path)
    return summary_path


def _write_real_summary_artifact_with_theta_ref(
    tmp_path: Path,
    *,
    theta_ref: np.ndarray,
    stem: str,
) -> Path:
    summary_path = _write_real_summary_artifact(tmp_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    matrix_path = summary_path.parent / payload["matrix_artifact_path"]
    with np.load(matrix_path) as arrays:
        matrix_payload = {key: arrays[key] for key in arrays.files}
    matrix_payload["theta_ref"] = np.asarray(theta_ref, dtype=float)
    np.savez_compressed(matrix_path, **matrix_payload)
    payload["theta_ref"] = np.asarray(theta_ref, dtype=float).tolist()
    payload["subblock_id"] = stem
    payload["prior_context"]["theta_ref_by_label"] = {
        label: float(theta_ref[index]) for index, label in enumerate(payload["theta_labels"])
    }
    updated_summary_path = tmp_path / f"{stem}.json"
    updated_matrix_path = tmp_path / f"{stem}_matrices.npz"
    updated_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    updated_matrix_path.write_bytes(matrix_path.read_bytes())
    payload["matrix_artifact_path"] = updated_matrix_path.name
    updated_summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return updated_summary_path


def test_build_prior_mean_from_store_resolves_scalars_and_derived_values(tmp_path: Path):
    module = _load_script_module()
    config_path = _write_prior_override_config(tmp_path)

    store, _, provenance = module.build_prior_store_from_system(
        config_path=config_path,
        system_preset="SHERA_TESTBED_3P",
    )
    labels = (
        "source.separation_as",
        "source.contrast",
        "source.log_flux_total",
        "optics.plate_scale_as_per_pix",
    )

    prior_mean = module.build_prior_mean_from_store(labels, store=store)

    np.testing.assert_allclose(
        prior_mean,
        np.array(
            [
                float(store.get("source.separation_as")),
                float(store.get("source.contrast")),
                float(store.get("source.log_flux_total")),
                float(store.get("optics.plate_scale_as_per_pix")),
            ]
        ),
    )
    assert provenance["prior_mean_source"] == "resolved_system_store"
    assert provenance["system_preset"] == "SHERA_TESTBED_3P"
    assert provenance["system_config_path"] == str(config_path.resolve())


def test_build_prior_mean_from_store_resolves_indexed_zernike_labels(tmp_path: Path):
    module = _load_script_module()
    config_path = _write_prior_override_config(tmp_path)

    store, _, _ = module.build_prior_store_from_system(
        config_path=config_path,
        system_preset="SHERA_TESTBED_3P",
    )
    labels = (
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.secondary.zernike_coeffs_nm[1]",
    )

    prior_mean = module.build_prior_mean_from_store(labels, store=store)

    np.testing.assert_allclose(
        prior_mean,
        np.array(
            [
                float(store.get("optics.primary.zernike_coeffs_nm")[0]),
                float(store.get("optics.secondary.zernike_coeffs_nm")[1]),
            ]
        ),
    )


def test_build_synthetic_reduced_information_is_symmetric_psd():
    module = _load_script_module()
    layout = module.ObservationThetaLayout.from_config(
        module.build_demo_theta_layout_config(
            enable_zernikes=True,
            zernike_indices=(0, 1),
            include_plate_scale=True,
        )
    )

    information = module.build_synthetic_reduced_information(
        layout=layout,
        rng=np.random.default_rng(5),
    )

    np.testing.assert_allclose(information, information.T)
    assert np.min(np.linalg.eigvalsh(information)) >= -1.0e-10


def test_build_cumulative_update_rows_include_prior_normalized_diagnostics():
    module = _load_script_module()
    prior = module.ObservationBeliefState.from_diagonal_prior(
        theta_labels=("source.separation_as",),
        mean=np.array([0.0]),
        sigma=np.array([2.0]),
    )
    summary = module.SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=prior.theta_labels,
        theta_ref=np.array([0.0]),
        reduced_information=np.array([[3.0]]),
        reduced_score=np.array([-3.0]),
    )
    update = module.update_observation_belief(prior, [summary])

    rows = module.build_cumulative_update_rows(
        labels=prior.theta_labels,
        cumulative_steps=update.cumulative_steps,
        truth=np.array([1.0]),
        prior_sigma=np.array([2.0]),
    )

    assert len(rows) == 1
    row = rows[0]
    posterior_sigma = float(update.cumulative_steps[0].sigma()[0])
    posterior_error = float(update.cumulative_steps[0].mean[0] - 1.0)
    assert row["posterior_sigma__source_separation_as"] == pytest.approx(
        posterior_sigma
    )
    assert row["posterior_sigma_over_prior_sigma__source_separation_as"] == (
        pytest.approx(posterior_sigma / 2.0)
    )
    assert row["abs_posterior_error_over_prior_sigma__source_separation_as"] == (
        pytest.approx(abs(posterior_error) / 2.0)
    )
    assert row["posterior_variance_over_prior_variance__source_separation_as"] == (
        pytest.approx((posterior_sigma / 2.0) ** 2)
    )


def test_prior_normalized_sigma_shrinks_when_positive_information_is_added():
    module = _load_script_module()
    prior = module.ObservationBeliefState.from_diagonal_prior(
        theta_labels=("source.separation_as",),
        mean=np.array([0.0]),
        sigma=np.array([2.0]),
    )
    summary = module.SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=prior.theta_labels,
        theta_ref=np.array([0.0]),
        reduced_information=np.array([[4.0]]),
        reduced_score=np.array([0.0]),
    )
    update = module.update_observation_belief(prior, [summary])

    rows = module.build_cumulative_update_rows(
        labels=prior.theta_labels,
        cumulative_steps=update.cumulative_steps,
        truth=np.array([0.0]),
        prior_sigma=np.array([2.0]),
    )

    assert rows[0]["posterior_sigma_over_prior_sigma__source_separation_as"] < 1.0


def test_build_prior_whitened_eigenmode_rows_reports_gain_and_physical_coefficients():
    module = _load_script_module()
    labels = (
        "source.separation_as",
        "source.log_flux_total",
    )
    gain = np.diag([1.0, 9.0])
    basis = module.build_observation_eigenbasis(gain, labels)

    rows = module.build_prior_whitened_eigenmode_rows(
        basis=basis,
        prior_sigma=np.array([2.0, 0.5]),
    )

    assert rows[0]["gain_eigenvalue"] == pytest.approx(9.0)
    assert rows[0]["posterior_whitened_eigenvalue"] == pytest.approx(10.0)
    assert rows[0]["top_label_1"] == "source.log_flux_total"
    assert rows[0]["top_norm_coeff_1"] == pytest.approx(1.0)
    assert rows[0]["top_physical_coeff_1"] == pytest.approx(0.5)


def test_observation_belief_demo_dry_run_plans_without_writing(tmp_path: Path):
    module = _load_script_module()
    config_path = _write_prior_override_config(tmp_path)

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "dry_run_case",
            "--config",
            str(config_path),
            "--system-preset",
            "SHERA_TESTBED_3P",
            "--n-subblocks",
            "3",
            "--seed",
            "7",
            "--zernike-indices",
            "0,1",
            "--dry-run",
        ]
    )

    assert result["dry_run"] is True
    assert result["artifacts"] == {}
    assert not (tmp_path / "dry_run_case").exists()
    assert result["summary"]["n_subblocks"] == 3
    assert result["summary"]["prior_mean_provenance"]["prior_mean_source"] == (
        "resolved_system_store"
    )
    assert result["summary"]["prior_mean_provenance"]["system_preset"] == "SHERA_TESTBED_3P"
    assert "prior_normalized_reporting" in result["summary"]["diagnostics"]
    assert "prior_whitened_information_gain" in result["summary"]


def test_observation_belief_demo_writes_required_artifacts(tmp_path: Path):
    module = _load_script_module()
    config_path = _write_prior_override_config(tmp_path)

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "artifact_case",
            "--config",
            str(config_path),
            "--system-preset",
            "SHERA_TESTBED_3P",
            "--n-subblocks",
            "4",
            "--seed",
            "11",
            "--zernike-indices",
            "0,1",
        ]
    )

    assert result["dry_run"] is False
    run_dir = tmp_path / "artifact_case"
    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    for key in (
        "observation_update_summary_json",
        "posterior_table_csv",
        "eigenmode_table_csv",
        "prior_whitened_eigenmode_table_csv",
        "cumulative_update_table_csv",
        "posterior_sigma_over_prior_sigma_vs_n_subblocks_png",
        "posterior_error_over_prior_sigma_vs_n_subblocks_png",
        "prior_whitened_information_gain_spectrum_png",
    ):
        assert key in artifacts
        assert artifacts[key].exists()
        assert artifacts[key].stat().st_size > 0

    summary = json.loads(
        artifacts["observation_update_summary_json"].read_text(encoding="utf-8")
    )
    posterior_rows = _read_csv_rows(artifacts["posterior_table_csv"])
    eigen_rows = _read_csv_rows(artifacts["eigenmode_table_csv"])
    prior_whitened_rows = _read_csv_rows(artifacts["prior_whitened_eigenmode_table_csv"])
    cumulative_rows = _read_csv_rows(artifacts["cumulative_update_table_csv"])
    store, _, _ = module.build_prior_store_from_system(
        config_path=config_path,
        system_preset="SHERA_TESTBED_3P",
    )

    assert summary["update"]["n_summaries"] == 4
    assert len(posterior_rows) == len(summary["theta_layout"]["labels"])
    assert len(eigen_rows) == len(summary["theta_layout"]["labels"])
    assert len(prior_whitened_rows) == len(summary["theta_layout"]["labels"])
    assert len(cumulative_rows) == 4
    assert summary["eigenbasis"]["weak_mode_count"] >= 1
    assert "prior_whitened_information_gain" in summary
    assert "prior_normalized_reporting" in summary["diagnostics"]
    assert summary["prior"]["mean"]["source.log_flux_total"] == pytest.approx(
        float(store.get("source.log_flux_total"))
    )
    assert summary["prior"]["mean"]["optics.plate_scale_as_per_pix"] == pytest.approx(
        float(store.get("optics.plate_scale_as_per_pix"))
    )
    assert summary["prior"]["mean"]["source.log_flux_total"] != pytest.approx(12.0)
    assert summary["prior"]["mean"]["optics.plate_scale_as_per_pix"] != pytest.approx(
        0.03
    )
    assert "posterior_sigma_over_prior_sigma" in posterior_rows[0]
    assert "posterior_error_over_prior_sigma" in posterior_rows[0]
    assert (
        "posterior_sigma_over_prior_sigma__source_separation_as"
        in cumulative_rows[0]
    )
    assert (
        "abs_posterior_error_over_prior_sigma__source_separation_as"
        in cumulative_rows[0]
    )
    assert "gain_eigenvalue" in prior_whitened_rows[0]
    assert "top_norm_coeff_1" in prior_whitened_rows[0]
    assert "top_physical_coeff_1" in prior_whitened_rows[0]
    assert "raw_eigenvalue" in eigen_rows[0]
    assert "floored_eigenvalue" in eigen_rows[0]
    assert (run_dir / "synthetic_subblock_summaries").is_dir()


def test_observation_belief_demo_loads_real_summary_artifact(tmp_path: Path):
    module = _load_script_module()
    config_path = _write_prior_override_config(tmp_path)
    summary_path = _write_real_summary_artifact(tmp_path)

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "real_summary_case",
            "--config",
            str(config_path),
            "--system-preset",
            "SHERA_TESTBED_3P",
            "--summary-path",
            str(summary_path),
        ]
    )

    artifacts = {name: Path(path) for name, path in result["artifacts"].items()}
    summary = json.loads(
        artifacts["observation_update_summary_json"].read_text(encoding="utf-8")
    )
    posterior_rows = _read_csv_rows(artifacts["posterior_table_csv"])

    assert result["dry_run"] is False
    assert summary["summary_input_mode"] == "external_summary_artifacts"
    assert summary["update"]["n_summaries"] == 1
    assert summary["truth"]["kind"] == "not_available"
    assert summary["summary_paths"] == [str(summary_path.resolve())]
    assert summary["prior_mean_source"] == "explicit_prior_config"
    assert summary["summary_scale_validation"]["summary_scale_policy"] == "require_summed"
    assert summary["summary_scale_validation"]["accepted_summary_information_scale"] == (
        "summed_likelihood"
    )
    assert "posterior_error" not in posterior_rows[0]
    assert artifacts["posterior_sigma_vs_n_subblocks_png"].exists()
    assert artifacts["posterior_sigma_over_prior_sigma_vs_n_subblocks_png"].exists()
    assert "synthetic_subblock_summaries_dir" not in artifacts


def test_real_summary_mode_uses_theta_ref_as_prior_mean_by_default(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_real_summary_artifact(tmp_path)

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "real_summary_default_prior_case",
            "--summary-path",
            str(summary_path),
        ]
    )

    summary = json.loads(
        Path(result["artifacts"]["observation_update_summary_json"]).read_text(encoding="utf-8")
    )
    posterior_rows = _read_csv_rows(Path(result["artifacts"]["posterior_table_csv"]))
    prior_mean_by_label = summary["prior"]["mean"]

    assert summary["prior_mean_source"] == "summary_theta_ref"
    assert summary["prior_mean_provenance"]["summary_theta_ref_compatibility"][
        "all_equal_within_tolerance"
    ] is True
    assert prior_mean_by_label["source.log_flux_total"] == pytest.approx(7.01)
    assert prior_mean_by_label["source.log_flux_total"] != pytest.approx(11.57, rel=1.0e-2)
    assert posterior_rows[0]["prior_mean_source"] == "summary_theta_ref"


def test_explicit_prior_config_overrides_summary_theta_ref(tmp_path: Path):
    module = _load_script_module()
    config_path = _write_prior_override_config(tmp_path)
    summary_path = _write_real_summary_artifact(tmp_path)

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "real_summary_explicit_prior_case",
            "--config",
            str(config_path),
            "--system-preset",
            "SHERA_TESTBED_3P",
            "--summary-path",
            str(summary_path),
        ]
    )

    summary = json.loads(
        Path(result["artifacts"]["observation_update_summary_json"]).read_text(encoding="utf-8")
    )
    store, _, _ = module.build_prior_store_from_system(
        config_path=config_path,
        system_preset="SHERA_TESTBED_3P",
    )

    assert summary["prior_mean_source"] == "explicit_prior_config"
    assert summary["prior"]["mean"]["source.log_flux_total"] == pytest.approx(
        float(store.get("source.log_flux_total"))
    )
    assert summary["prior"]["mean"]["source.log_flux_total"] != pytest.approx(7.01)


def test_multiple_real_summaries_with_mixed_theta_ref_record_warning(tmp_path: Path):
    module = _load_script_module()
    summary_path_a = _write_real_summary_artifact_with_theta_ref(
        tmp_path,
        theta_ref=np.array([11.25, 7.01, 3.4, 0.01]),
        stem="summary_a",
    )
    summary_path_b = _write_real_summary_artifact_with_theta_ref(
        tmp_path,
        theta_ref=np.array([11.25, 7.25, 3.6, 0.01]),
        stem="summary_b",
    )

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "real_summary_mixed_theta_ref_case",
            "--summary-path",
            str(summary_path_a),
            "--summary-path",
            str(summary_path_b),
        ]
    )

    summary = json.loads(
        Path(result["artifacts"]["observation_update_summary_json"]).read_text(encoding="utf-8")
    )
    compatibility = summary["prior_mean_provenance"]["summary_theta_ref_compatibility"]

    assert summary["prior_mean_source"] == "summary_theta_ref"
    assert compatibility["all_equal_within_tolerance"] is False
    assert compatibility["max_abs_spread_by_label"]["source.log_flux_total"] == pytest.approx(
        0.24
    )
    assert summary["prior_warnings"]


def test_real_summary_mode_dry_run_reports_summary_theta_ref_prior_provenance(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_real_summary_artifact(tmp_path)

    result = module.main(
        [
            "--results-dir",
            str(tmp_path),
            "--run-name",
            "real_summary_dry_run_case",
            "--summary-path",
            str(summary_path),
            "--dry-run",
        ]
    )

    assert result["dry_run"] is True
    assert result["summary"]["prior_mean_source"] == "summary_theta_ref"
    assert result["summary"]["prior"]["mean"]["source.log_flux_total"] == pytest.approx(7.01)


def test_real_summary_mode_rejects_unclassified_artifact_by_default(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_real_summary_artifact(tmp_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload.pop("information_accounting")
    payload["metadata"].pop("information_accounting")
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="information-accounting"):
        module.main(
            [
                "--results-dir",
                str(tmp_path),
                "--run-name",
                "missing_scale_case",
                "--summary-path",
                str(summary_path),
                "--dry-run",
            ]
        )
