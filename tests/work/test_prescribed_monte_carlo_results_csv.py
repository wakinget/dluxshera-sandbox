from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_prescribed_module():
    repo_root = Path(__file__).resolve().parents[2]
    candidate_paths = [
        repo_root / "work" / "experiments" / "prescribed_monte_carlo.py",
        repo_root / "examples" / "recipes" / "prescribed_monte_carlo.py",
    ]
    for module_path in candidate_paths:
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("prescribed_monte_carlo", module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise RuntimeError("Unable to load prescribed_monte_carlo module.")


def _sample_run_entries() -> list[dict]:
    return [
        {
            "run_id": "run_0001",
            "plan_label": "baseline",
            "summary": {
                "status": "ok",
                "loss_init": 10.0,
                "loss_final": 2.0,
                "param_summary": {
                    "binary.x_position_as": {
                        "truth": 0.2,
                        "init": 0.9,
                        "final": 0.3,
                        "init_delta": 0.7,
                        "final_delta": 0.1,
                    },
                    "primary.zernike_coeffs_nm": {
                        "truth": [10.0, 20.0],
                        "init": [8.0, 19.0],
                        "final": [9.0, 21.0],
                        "init_delta": [-2.0, -1.0],
                        "final_delta": [-1.0, 1.0],
                    },
                },
            },
            "meta": {"prescribed": {"seed": 101}, "optimizer": {}},
        },
        {
            "run_id": "run_0002",
            "plan_label": "partial",
            "summary": {
                "status": "ok",
                "loss_init": 6.0,
                "loss_final": 3.0,
                "param_summary": {
                    "binary.x_position_as": {
                        "init": 1.4,
                        "final": 1.1,
                    }
                },
            },
            "meta": {"prescribed": {"seed": 202}, "optimizer": {}},
        },
    ]


def test_write_results_csv_column_orientation(tmp_path):
    module = _load_prescribed_module()
    out_path = tmp_path / "results.csv"
    run_entries = _sample_run_entries()
    infer_keys = ("binary.x_position_as", "primary.zernike_coeffs_nm")

    module._write_results_csv(
        out_path,
        run_entries,
        infer_keys,
        results_orientation="col",
    )

    with out_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        rows = list(reader)

    assert header == ["key", "run_0001", "run_0002"]
    rows_by_key = {row["key"]: row for row in rows}

    assert "loss_init" in rows_by_key
    assert "final.binary.x_position_as" in rows_by_key
    assert "final.primary.zernike_coeffs_nm[0]" in rows_by_key
    assert "final.primary.zernike_coeffs_nm[1]" in rows_by_key

    assert rows_by_key["loss_init"]["run_0001"] == "10.0"
    assert rows_by_key["loss_init"]["run_0002"] == "6.0"

    assert rows_by_key["final.binary.x_position_as"]["run_0001"] == "0.3"
    assert rows_by_key["final.binary.x_position_as"]["run_0002"] == "1.1"

    assert rows_by_key["final.primary.zernike_coeffs_nm[0]"]["run_0001"] == "9.0"
    assert rows_by_key["final.primary.zernike_coeffs_nm[0]"]["run_0002"] == ""

    assert rows_by_key["truth.binary.x_position_as"]["run_0001"] == "0.2"
    assert rows_by_key["truth.binary.x_position_as"]["run_0002"] == ""


def test_write_results_csv_row_orientation(tmp_path):
    module = _load_prescribed_module()
    out_path = tmp_path / "results.csv"
    run_entries = _sample_run_entries()
    infer_keys = ("binary.x_position_as", "primary.zernike_coeffs_nm")

    module._write_results_csv(
        out_path,
        run_entries,
        infer_keys,
        results_orientation="row",
    )

    with out_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        rows = list(reader)

    assert header is not None
    assert header[0] == "run_id"
    assert [row["run_id"] for row in rows] == ["run_0001", "run_0002"]

    rows_by_run = {row["run_id"]: row for row in rows}
    assert "loss_init" in header
    assert "final.binary.x_position_as" in header
    assert "final.primary.zernike_coeffs_nm[0]" in header

    assert rows_by_run["run_0001"]["loss_init"] == "10.0"
    assert rows_by_run["run_0002"]["loss_init"] == "6.0"

    assert rows_by_run["run_0001"]["final.binary.x_position_as"] == "0.3"
    assert rows_by_run["run_0002"]["final.binary.x_position_as"] == "1.1"

    assert rows_by_run["run_0001"]["final.primary.zernike_coeffs_nm[1]"] == "21.0"
    assert rows_by_run["run_0002"]["final.primary.zernike_coeffs_nm[1]"] == ""

    assert rows_by_run["run_0001"]["truth.binary.x_position_as"] == "0.2"
    assert rows_by_run["run_0002"]["truth.binary.x_position_as"] == ""
