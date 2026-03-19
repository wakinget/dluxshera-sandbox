from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "scripts"
    / "aggregate_detector_ke_sweep.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "aggregate_detector_ke_sweep_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _write_row_results(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_extract_detector_knowledge_error_metadata_from_inference_layers():
    module = _load_script_module()
    prescription = {
        "experiment": {
            "inference_system": {
                "detector": {
                    "layers": [
                        {
                            "name": "pixel_offsets",
                            "dx_path": "dx.fits",
                            "dy_path": "dy.fits",
                            "knowledge_error": {
                                "model": "gaussian",
                                "scale": 1e-3,
                                "realization_policy": "per_run",
                            },
                        },
                        {
                            "name": "pixel_response",
                            "prf_path": "prf.fits",
                            "knowledge_error": {
                                "model": "gaussian",
                                "scale": 2e-3,
                                "realization_policy": "fixed_per_experiment",
                            },
                        },
                    ]
                }
            }
        }
    }

    meta = module.extract_detector_knowledge_error_metadata(prescription)

    assert meta["inference_system_present"] is True
    assert meta["pixel_offsets_configured_scale"] == 1e-3
    assert meta["pixel_offsets_configured_model"] == "gaussian"
    assert meta["pixel_offsets_configured_realization_policy"] == "per_run"
    assert meta["pixel_offsets_dx_path"] == "dx.fits"
    assert meta["pixel_offsets_dy_path"] == "dy.fits"
    assert meta["pixel_response_configured_scale"] == 2e-3
    assert meta["pixel_response_configured_model"] == "gaussian"
    assert meta["pixel_response_configured_realization_policy"] == "fixed_per_experiment"
    assert meta["pixel_response_prf_path"] == "prf.fits"


def test_extract_detector_knowledge_error_metadata_defaults_when_inference_missing():
    module = _load_script_module()
    prescription = {"experiment": {}}

    meta = module.extract_detector_knowledge_error_metadata(prescription)

    assert meta["inference_system_present"] is False
    assert meta["pixel_offsets_configured_scale"] == 0.0
    assert meta["pixel_offsets_configured_model"] is None
    assert meta["pixel_offsets_configured_realization_policy"] is None
    assert meta["pixel_response_configured_scale"] == 0.0
    assert meta["pixel_response_configured_model"] is None
    assert meta["pixel_response_configured_realization_policy"] is None


def test_compute_component_error_prefers_final_delta_then_falls_back():
    module = _load_script_module()

    row_with_delta = {
        "final_delta.source.separation_as": "0.25",
        "final.source.separation_as": "9.0",
        "truth.source.separation_as": "1.0",
    }
    row_without_delta = {
        "final.source.separation_as": "1.2",
        "truth.source.separation_as": "1.0",
    }

    assert module.compute_component_error(row_with_delta, "source.separation_as") == 0.25
    fallback = module.compute_component_error(row_without_delta, "source.separation_as")
    assert fallback is not None
    assert abs(fallback - 0.2) < 1e-12


def test_extract_realized_detector_ke_metadata_from_meta_payload():
    module = _load_script_module()
    meta_payload = {
        "prescribed": {
            "detector_ke_realization_mode": "per_run",
            "inference_cfg_hash": "cfg_hash_123",
            "inference_forward_spec_hash": "spec_hash_456",
        },
        "detector_knowledge_error": {
            "inference": {
                "layers": {
                    "pixel_offsets": {
                        "name": "pixel_offsets",
                        "model": "gaussian",
                        "scale": "1e-2",
                        "realization_policy": "per_run",
                        "seed": 101,
                        "seed_source": "run_seed",
                    },
                    "pixel_response": {
                        "name": "pixel_response",
                        "model": "gaussian",
                        "scale": 5e-3,
                        "realization_policy": "fixed_per_experiment",
                        "seed": 202,
                        "seed_source": "experiment_seed",
                    },
                }
            }
        },
    }

    realized = module.extract_realized_detector_knowledge_error_metadata(meta_payload)
    assert realized["run_meta_present"] is True
    assert realized["detector_ke_realization_mode"] == "per_run"
    assert realized["inference_cfg_hash"] == "cfg_hash_123"
    assert realized["inference_forward_spec_hash"] == "spec_hash_456"
    assert realized["pixel_offsets_realized_model"] == "gaussian"
    assert realized["pixel_offsets_realized_scale"] == 1e-2
    assert realized["pixel_offsets_realized_realization_policy"] == "per_run"
    assert realized["pixel_offsets_realized_seed"] == 101
    assert realized["pixel_offsets_realized_seed_source"] == "run_seed"
    assert realized["pixel_response_realized_model"] == "gaussian"
    assert realized["pixel_response_realized_scale"] == 5e-3
    assert realized["pixel_response_realized_realization_policy"] == "fixed_per_experiment"
    assert realized["pixel_response_realized_seed"] == 202
    assert realized["pixel_response_realized_seed_source"] == "experiment_seed"


def test_extract_realized_detector_ke_metadata_handles_missing_detector_section():
    module = _load_script_module()
    meta_payload = {
        "run_id": "mc_0001",
        "prescribed": {"detector_ke_realization_mode": "fixed_per_experiment"},
    }

    realized = module.extract_realized_detector_knowledge_error_metadata(meta_payload)
    assert realized["run_meta_present"] is True
    assert realized["detector_ke_realization_mode"] == "fixed_per_experiment"
    assert realized["pixel_offsets_realized_model"] is None
    assert realized["pixel_offsets_realized_seed"] is None


def test_aggregate_detector_ke_sweep_skips_partial_or_bad_experiments(tmp_path):
    module = _load_script_module()
    root = tmp_path / "detector_ke_sweep"
    root.mkdir(parents=True, exist_ok=True)

    # Valid experiment.
    ke0 = root / "ke_0"
    ke0.mkdir()
    _write_yaml(
        ke0 / "prescription.yaml",
        """
        experiment:
          inference_system:
            detector:
              layers:
                - name: pixel_offsets
                  dx_path: dx0.fits
                  dy_path: dy0.fits
                  knowledge_error:
                    model: gaussian
                    scale: 0.0
                    realization_policy: fixed_per_experiment
        """,
    )
    _write_row_results(
        ke0 / "results.csv",
        header=[
            "run_id",
            "status",
            "seed",
            "final_delta.source.separation_as",
            "final.source.separation_as",
            "truth.source.separation_as",
            "final_delta.optics.primary.zernike_coeffs_nm[0]",
            "final_delta.optics.primary.zernike_coeffs_nm[1]",
        ],
        rows=[
            ["run_0001", "ok", "1", "0.01", "", "", "1.0", "-1.0"],
            ["run_0002", "ok", "2", "", "1.10", "1.00", "2.0", "2.0"],
        ],
    )
    _write_json(
        ke0 / "runs" / "run_0001" / "meta.json",
        {
            "run_id": "run_0001",
            "prescribed": {
                "detector_ke_realization_mode": "fixed_per_experiment",
                "inference_cfg_hash": "cfg-ke0",
                "inference_forward_spec_hash": "spec-ke0",
            },
            "detector_knowledge_error": {
                "inference": {
                    "layers": {
                        "pixel_offsets": {
                            "name": "pixel_offsets",
                            "model": "gaussian",
                            "scale": 0.0,
                            "realization_policy": "fixed_per_experiment",
                            "seed": 1234,
                            "seed_source": "experiment_seed",
                        }
                    }
                }
            },
        },
    )

    # Partial experiment (missing results.csv) should be skipped by default.
    ke_partial = root / "ke_1e-3"
    ke_partial.mkdir()
    _write_yaml(
        ke_partial / "prescription.yaml",
        """
        experiment:
          inference_system:
            detector:
              layers:
                - name: pixel_offsets
                  knowledge_error:
                    model: gaussian
                    scale: 0.001
        """,
    )

    # Malformed orientation experiment (column-oriented results.csv) should be skipped.
    ke_bad = root / "ke_1e-2"
    ke_bad.mkdir()
    _write_yaml(
        ke_bad / "prescription.yaml",
        """
        experiment:
          inference_system:
            detector:
              layers:
                - name: pixel_offsets
                  knowledge_error:
                    model: gaussian
                    scale: 0.01
        """,
    )
    _write_row_results(
        ke_bad / "results.csv",
        header=["key", "run_0001"],
        rows=[["status", "ok"]],
    )

    run_rows, summary_rows, components, result_columns, stats = module.aggregate_detector_ke_sweep(
        root=root,
        pattern="ke_*",
        strict=False,
        verbose=False,
    )

    assert stats.discovered == 3
    assert stats.loaded == 1
    assert stats.skipped == 2

    assert len(run_rows) == 2
    assert len(summary_rows) == 1
    assert "source.separation_as" in components
    assert "optics.primary.zernike_coeffs_nm[0]" in components
    assert "run_id" in result_columns

    sep_by_run = {row["run_id"]: row["sep_error_as"] for row in run_rows}
    assert abs(sep_by_run["run_0001"] - 0.01) < 1e-12
    assert abs(sep_by_run["run_0002"] - 0.1) < 1e-12

    rows_by_run = {row["run_id"]: row for row in run_rows}
    assert rows_by_run["run_0001"]["pixel_offsets_configured_scale"] == 0.0
    assert rows_by_run["run_0001"]["pixel_offsets_configured_model"] == "gaussian"
    assert (
        rows_by_run["run_0001"]["pixel_offsets_configured_realization_policy"]
        == "fixed_per_experiment"
    )
    assert rows_by_run["run_0001"]["run_meta_present"] is True
    assert rows_by_run["run_0001"]["pixel_offsets_realized_seed"] == 1234
    assert (
        rows_by_run["run_0001"]["pixel_offsets_realized_realization_policy"]
        == "fixed_per_experiment"
    )
    assert rows_by_run["run_0002"]["run_meta_present"] is False
    assert rows_by_run["run_0002"]["pixel_offsets_realized_seed"] is None


def test_build_sweep_summary_rows_handles_vector_components():
    module = _load_script_module()
    run_rows = [
        {
            "status": "ok",
            "sweep_label": "ke_a",
            "experiment_dir": "ke_a",
            "pixel_offsets_configured_scale": 0.001,
            "pixel_offsets_configured_model": "gaussian",
            "pixel_offsets_configured_realization_policy": "per_run",
            "pixel_response_configured_scale": 0.0,
            "pixel_response_configured_model": None,
            "pixel_response_configured_realization_policy": None,
            "final_delta.optics.primary.zernike_coeffs_nm[0]": "1.0",
            "final_delta.optics.primary.zernike_coeffs_nm[1]": "2.0",
        },
        {
            "status": "ok",
            "sweep_label": "ke_a",
            "experiment_dir": "ke_a",
            "pixel_offsets_configured_scale": 0.001,
            "pixel_offsets_configured_model": "gaussian",
            "pixel_offsets_configured_realization_policy": "per_run",
            "pixel_response_configured_scale": 0.0,
            "pixel_response_configured_model": None,
            "pixel_response_configured_realization_policy": None,
            "final_delta.optics.primary.zernike_coeffs_nm[0]": "-1.0",
            "final_delta.optics.primary.zernike_coeffs_nm[1]": "2.0",
        },
    ]
    components = [
        "optics.primary.zernike_coeffs_nm[0]",
        "optics.primary.zernike_coeffs_nm[1]",
    ]

    summary_rows = module.build_sweep_summary_rows(run_rows, components)
    assert len(summary_rows) == 1
    row = summary_rows[0]

    assert row["n_total"] == 2
    assert row["n_success"] == 2
    assert row["optics.primary.zernike_coeffs_nm[0]_mean_bias"] == 0.0
    assert row["optics.primary.zernike_coeffs_nm[0]_rmse"] == 1.0
    assert row["optics.primary.zernike_coeffs_nm[1]_std_bias"] == 0.0


def test_aggregate_detector_ke_sweep_strict_raises_on_partial_inputs(tmp_path):
    module = _load_script_module()
    root = tmp_path / "detector_ke_sweep"
    root.mkdir(parents=True, exist_ok=True)

    ke_partial = root / "ke_1e-3"
    ke_partial.mkdir()
    _write_yaml(
        ke_partial / "prescription.yaml",
        """
        experiment:
          inference_system:
            detector:
              layers:
                - name: pixel_offsets
                  knowledge_error:
                    model: gaussian
                    scale: 0.001
        """,
    )

    with pytest.raises(ValueError, match="missing results.csv"):
        module.aggregate_detector_ke_sweep(
            root=root,
            pattern="ke_*",
            strict=True,
            verbose=False,
        )
