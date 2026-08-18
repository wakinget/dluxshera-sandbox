from __future__ import annotations

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
    / "inspect_subblock_summary.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "inspect_subblock_summary_script",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary_artifact(tmp_path: Path) -> Path:
    layout = build_combined_local_parameter_layout(
        ("theta.source.separation_as", "theta.optics.plate_scale_as_per_pix"),
        (
            "phi.frame[0].source.x_position_as",
            "phi.frame[0].source.y_position_as",
        ),
    )
    gradient = np.array([-1.0, 0.25, 0.1, -0.2], dtype=float)
    curvature = np.array(
        [
            [4.0, 0.2, 0.5, 0.0],
            [0.2, 3.0, -0.1, 0.3],
            [0.5, -0.1, 2.5, 0.2],
            [0.0, 0.3, 0.2, 1.8],
        ],
        dtype=float,
    )
    blocks = partition_local_curvature(
        layout=layout,
        combined_gradient=gradient,
        combined_curvature=curvature,
    )
    reduced = schur_reduce_local_quadratic(blocks=blocks, damping=1.0e-8)
    summary = SubblockSummary.from_reduced_form(
        subblock_id="subblock_000000",
        theta_labels=(
            "source.separation_as",
            "optics.plate_scale_as_per_pix",
        ),
        theta_ref=np.array([1.5, 0.01]),
        reduced_information=reduced.reduced_information,
        reduced_score=reduced.reduced_score,
        summary_kind="image_backed_schur",
        damping_used=1.0e-8,
        diagnostics={"damping_value": 1.0e-8},
    )
    artifact = ImageBackedSubblockSummaryArtifact(
        summary=summary,
        layout=layout,
        theta_ref=np.array([1.5, 0.01]),
        phi_ref=np.array([0.1, -0.2]),
        reduced=reduced,
        metadata={
            "generator": "unit_test",
            "case_root": str(tmp_path / "case"),
            "cube_path": str(tmp_path / "cube.fits"),
            "config_path": str(tmp_path / "config.json"),
            "objective": {
                "objective_kind_used": "data_only",
                "inference_objective": {
                    "noise_model": {"variance_model": "provided_cube"}
                },
            },
        },
        combined_gradient=gradient,
        combined_curvature=curvature,
    )
    summary_path = tmp_path / "subblock_summary.json"
    matrix_path = tmp_path / "subblock_summary_matrices.npz"
    artifact.write(summary_json_path=summary_path, matrix_npz_path=matrix_path)
    return summary_path


def test_inspect_subblock_summary_reports_theta_and_phi_dimensions(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_summary_artifact(tmp_path)

    report = module.main([str(summary_path)])

    assert report["dimensions"]["n_theta"] == 2
    assert report["dimensions"]["n_phi"] == 2
    assert report["provenance"]["objective_kind"] == "data_only"
    assert report["provenance"]["variance_model"] == "provided_cube"


def test_inspect_subblock_summary_missing_matrix_sidecar_fails_clearly(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_summary_artifact(tmp_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    matrix_path = summary_path.parent / payload["matrix_artifact_path"]
    matrix_path.unlink()

    with pytest.raises(FileNotFoundError, match="matrix sidecar"):
        module.main([str(summary_path)])


def test_inspect_subblock_summary_shape_mismatch_fails_clearly(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_summary_artifact(tmp_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    matrix_path = summary_path.parent / payload["matrix_artifact_path"]
    with np.load(matrix_path) as arrays:
        matrix_payload = {key: arrays[key] for key in arrays.files}
    matrix_payload["phi_ref"] = np.array([0.1], dtype=float)
    np.savez_compressed(matrix_path, **matrix_payload)

    with pytest.raises(ValueError, match="phi_ref shape does not match phi_labels length"):
        module.main([str(summary_path)])


def test_inspect_subblock_summary_writes_optional_report_json(tmp_path: Path):
    module = _load_script_module()
    summary_path = _write_summary_artifact(tmp_path)
    report_json = tmp_path / "inspection_report.json"

    report = module.main([str(summary_path), "--report-json", str(report_json)])

    assert report_json.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["subblock_id"] == report["subblock_id"]
