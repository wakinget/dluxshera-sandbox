from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "work/experiments/export_training_dataset_eigenmodes.py"


def _load_exporter_module():
    spec = importlib.util.spec_from_file_location("export_training_dataset_eigenmodes", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_variant_writer_exports_complete_mode_component_schema(tmp_path):
    exporter = _load_exporter_module()
    fim = np.array([[2.0, 0.0], [0.0, 0.5]])
    eig = exporter._eigendecompose(fim)
    rows = [
        {
            "parameter_index": 0,
            "parameter_label": "alpha",
            "parameter_group": "source",
            "parameter_unit": "arcsec",
            "base_key": "source.alpha",
            "component_index": "",
            "noll_index": "",
            "truth_value": 1.0,
            "included_by": "unit_test",
            "description": "alpha",
        },
        {
            "parameter_index": 1,
            "parameter_label": "beta",
            "parameter_group": "optics",
            "parameter_unit": "nm",
            "base_key": "optics.beta",
            "component_index": "",
            "noll_index": "",
            "truth_value": 2.0,
            "included_by": "unit_test",
            "description": "beta",
        },
    ]

    stats = exporter._write_variant_outputs(
        variant_dir=tmp_path,
        variant="sweep_fixed_nuisance",
        purpose="unit test",
        basis_keys=["source.alpha", "optics.beta"],
        rows=rows,
        fim=fim,
        eig=eig,
        manifest_extra={"fim_construction_source": "unit test"},
    )

    expected = {
        "manifest.json",
        "parameter_labels.csv",
        "fim_matrix.csv",
        "eigenvalues.csv",
        "eigenvectors_long.csv",
        "eigenvectors_wide.csv",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert stats["n_parameters"] == 2
    assert stats["n_modes"] == 2

    with (tmp_path / "eigenvectors_long.csv").open(newline="", encoding="utf-8") as handle:
        long_rows = list(csv.DictReader(handle))
    assert len(long_rows) == 4
    assert {row["parameter_label"] for row in long_rows} == {"alpha", "beta"}
    assert {row["whitened"] for row in long_rows} == {"false"}
    assert {row["truncated"] for row in long_rows} == {"false"}
