from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from astropy.io import fits


def _load_review_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "examples" / "scripts" / "review_m2_hoke_bridge_wfe_maps.py"
    spec = importlib.util.spec_from_file_location("review_m2_hoke_bridge_wfe_maps", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load review_m2_hoke_bridge_wfe_maps module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fits(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(data=np.asarray(array)).writeto(path, overwrite=True)


def _write_low_order_csv(path: Path, *, kind: str) -> None:
    fieldnames = [
        "mirror",
        "zernike_label",
        "truth_coeff_nm",
        "knowledge_coeff_nm",
        "error_nm",
    ]
    rows = []
    for noll in range(4, 12):
        rows.append(
            {
                "mirror": "secondary",
                "zernike_label": f"Z{noll}",
                "truth_coeff_nm": "0.0",
                "knowledge_coeff_nm": "0.0",
                "error_nm": "0.0",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if kind == "truth":
                writer.writerow({**row, "knowledge_coeff_nm": "", "error_nm": ""})
            elif kind == "knowledge":
                writer.writerow(row)
            else:
                writer.writerow({**row, "truth_coeff_nm": "", "knowledge_coeff_nm": ""})


def test_bridge_review_summary_verifies_precomputed_split(tmp_path: Path) -> None:
    module = _load_review_module()
    bundle = tmp_path / "bundle"
    config_root = bundle / "model_split" / "high_order_wfe" / "config_maps"
    maps_root = bundle / "model_split" / "high_order_wfe" / "maps"
    config_root.mkdir(parents=True)
    maps_root.mkdir(parents=True)

    primary_truth = np.zeros((16, 16), dtype=float)
    secondary_truth = np.arange(256, dtype=float).reshape(16, 16) / 100.0
    secondary_error = np.ones((16, 16), dtype=float) * 0.5
    secondary_knowledge = secondary_truth + secondary_error
    mask = np.ones((16, 16), dtype=np.uint8)

    np.save(config_root / "primary_high_order_truth_opd_nm.npy", primary_truth)
    np.save(config_root / "secondary_high_order_truth_opd_nm.npy", secondary_truth)
    np.save(config_root / "secondary_high_order_error_opd_nm.npy", secondary_error)

    _write_fits(maps_root / "primary_mask.fits", mask)
    _write_fits(maps_root / "secondary_mask.fits", mask)
    _write_fits(maps_root / "primary_high_order_truth_opd_nm.fits", primary_truth)
    _write_fits(maps_root / "primary_high_order_error_opd_nm.fits", primary_truth)
    _write_fits(maps_root / "primary_high_order_knowledge_opd_nm.fits", primary_truth)
    _write_fits(maps_root / "secondary_high_order_truth_opd_nm.fits", secondary_truth)
    _write_fits(maps_root / "secondary_high_order_error_opd_nm.fits", secondary_error)
    _write_fits(
        maps_root / "secondary_high_order_knowledge_opd_nm.fits",
        secondary_knowledge,
    )

    _write_low_order_csv(maps_root / "low_order_zernike_truth.csv", kind="truth")
    _write_low_order_csv(maps_root / "low_order_zernike_knowledge.csv", kind="knowledge")
    _write_low_order_csv(maps_root / "low_order_zernike_errors.csv", kind="errors")

    summary_path = bundle / "model_split" / "high_order_wfe" / "high_order_wfe_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "truth_seed": 1,
                "primary": {"truth_seed": 2},
                "secondary": {
                    "truth_seed": 3,
                    "knowledge_seed": 4,
                    "truth_full_rms_nm": 20.0,
                    "truth_high_order_rms_nm": 1.0,
                    "requested_knowledge_error_rms_nm": 0.5,
                },
            }
        ),
        encoding="utf-8",
    )

    review = module.build_review_summary(bundle, repo_root=tmp_path)

    assert review["array_checks"]["primary_npy_truth_equals_fits_truth"] is True
    assert review["array_checks"]["secondary_npy_truth_equals_fits_truth"] is True
    assert review["array_checks"]["secondary_npy_error_equals_fits_error"] is True
    assert review["array_checks"]["primary_error_is_zero"] is True
    assert review["array_checks"]["secondary_knowledge_equals_truth_plus_error"] is True
    assert review["maps"]["secondary_high_order_error"]["pupil_rms_nm"] == 0.5
