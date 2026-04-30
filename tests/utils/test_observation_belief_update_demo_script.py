from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


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
