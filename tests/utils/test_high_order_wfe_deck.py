from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from dluxshera.utils.high_order_wfe import (
    DEFAULT_LOW_ORDER_NOLL_INDICES,
    PTT_NOLL_INDICES,
    build_high_order_wfe_deck,
    build_mirror_wfe_deck,
    fit_zernike_coefficients_nm,
    generate_power_law_opd_map,
    make_pupil_mask,
    reconstruct_zernike_opd_nm,
    write_high_order_wfe_deck_artifacts,
)

TEMPLATE_PATH = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_algorithm_campaign_v1.yaml"
)


def masked_rms(arr: np.ndarray, mask: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr[mask]))))


def test_power_law_map_determinism_and_seed_variation() -> None:
    mask = make_pupil_mask((32, 32))
    a = generate_power_law_opd_map((32, 32), alpha=2.5, seed=10, rms_opd_nm=20.0, mask=mask)
    b = generate_power_law_opd_map((32, 32), alpha=2.5, seed=10, rms_opd_nm=20.0, mask=mask)
    c = generate_power_law_opd_map((32, 32), alpha=2.5, seed=11, rms_opd_nm=20.0, mask=mask)

    assert np.allclose(a, b)
    assert not np.allclose(a, c)
    assert np.all(np.isfinite(a))
    assert abs(masked_rms(a, mask) - 20.0) < 1e-10


def test_mirror_deck_normalizes_and_removes_ptt() -> None:
    deck = build_mirror_wfe_deck("primary", shape=(48, 48), seed=5)
    ptt = fit_zernike_coefficients_nm(deck.full_truth.opd_nm, PTT_NOLL_INDICES, mask=deck.full_truth.mask)

    assert abs(deck.full_truth.rms_nm - 20.0) < 1e-10
    assert max(abs(v) for v in ptt.values()) < 1e-10
    assert deck.full_truth.provenance["mask_policy"] == "circular_fallback"


def test_low_order_labels_and_active_mapping_are_explicit() -> None:
    deck = build_mirror_wfe_deck("secondary", shape=(40, 40), seed=7)
    expected = {f"Z{i}" for i in DEFAULT_LOW_ORDER_NOLL_INDICES}

    assert set(deck.low_order_truth_coeffs_nm) == expected
    mapping = deck.diagnostics["low_order_mapping"]
    assert mapping["Z4"]["noll_index"] == 4
    assert mapping["Z4"]["active_index"] == 0
    assert mapping["Z11"]["state_label"] == "optics.secondary.zernike_coeffs_nm[7]"


def test_residual_reconstructs_full_truth() -> None:
    deck = build_mirror_wfe_deck("primary", shape=(48, 48), seed=8)
    low = reconstruct_zernike_opd_nm(
        deck.low_order_truth_coeffs_nm,
        deck.full_truth.shape,
        mask=deck.full_truth.mask,
    )
    reconstructed = low + deck.high_order_truth.opd_nm

    assert np.max(np.abs((reconstructed - deck.full_truth.opd_nm)[deck.full_truth.mask])) < 1e-10


def test_knowledge_coefficients_and_high_order_error_are_deterministic() -> None:
    a = build_mirror_wfe_deck("primary", shape=(48, 48), seed=9)
    b = build_mirror_wfe_deck("primary", shape=(48, 48), seed=9)
    c = build_mirror_wfe_deck("primary", shape=(48, 48), seed=10)

    assert a.low_order_knowledge_error_nm == b.low_order_knowledge_error_nm
    assert a.low_order_knowledge_error_nm != c.low_order_knowledge_error_nm
    for key, truth in a.low_order_truth_coeffs_nm.items():
        assert np.isclose(
            a.low_order_knowledge_coeffs_nm[key] - truth,
            a.low_order_knowledge_error_nm[key],
        )

    assert abs(a.high_order_knowledge_error.rms_nm - 0.3) < 1e-10
    assert np.allclose(
        a.high_order_knowledge.opd_nm,
        a.high_order_truth.opd_nm + a.high_order_knowledge_error.opd_nm,
    )


def test_artifact_writer_outputs_manifest_csvs_and_fits(tmp_path: Path) -> None:
    deck = build_high_order_wfe_deck(shape=(32, 32), seed=12)
    written = write_high_order_wfe_deck_artifacts(deck, tmp_path / "optics")

    expected = {
        "high_order_wfe_deck_manifest.json",
        "low_order_zernike_truth.csv",
        "low_order_zernike_knowledge.csv",
        "low_order_zernike_errors.csv",
        "primary_full_truth_opd_nm.fits",
        "primary_high_order_truth_opd_nm.fits",
        "primary_high_order_knowledge_opd_nm.fits",
        "primary_high_order_error_opd_nm.fits",
        "primary_mask.fits",
        "secondary_full_truth_opd_nm.fits",
        "secondary_high_order_truth_opd_nm.fits",
        "secondary_high_order_knowledge_opd_nm.fits",
        "secondary_high_order_error_opd_nm.fits",
        "secondary_mask.fits",
    }
    assert expected <= set(written)

    manifest = json.loads((tmp_path / "optics" / "high_order_wfe_deck_manifest.json").read_text())
    assert manifest["schema_version"] == "high_order_wfe_deck.v1"
    assert manifest["opd_unit"] == "nm"
    assert manifest["primary"]["provenance"]["mask_policy"] == "circular_fallback"
    assert manifest["primary"]["provenance"]["high_order_error_seed"] is not None

    with (tmp_path / "optics" / "low_order_zernike_knowledge.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {"mirror", "noll_index", "active_index", "truth_coeff_nm", "knowledge_coeff_nm", "error_nm", "sigma_nm", "seed"} <= set(rows[0])
    assert len(rows) == 16

    data = fits.getdata(tmp_path / "optics" / "primary_full_truth_opd_nm.fits")
    header = fits.getheader(tmp_path / "optics" / "primary_full_truth_opd_nm.fits")
    assert data.shape == (32, 32)
    assert np.all(np.isfinite(data))
    assert header["BUNIT"] == "nm"
    assert header["OPDUNIT"] == "nm"


def test_full_fidelity_template_builds_default_deck() -> None:
    payload = yaml.safe_load(TEMPLATE_PATH.read_text())
    optics = payload["experiment"]["optics"]
    deck = build_high_order_wfe_deck(
        shape=(32, 32),
        seed=payload["experiment"]["seed"],
        primary_config=optics["primary"]["wfe"],
        secondary_config=optics["secondary"]["wfe"],
    )

    assert deck.primary.full_truth.rms_nm == pytest.approx(20.0)
    assert deck.secondary.full_truth.rms_nm == pytest.approx(20.0)
    assert deck.primary.provenance["high_order_error_rms_nm"] == pytest.approx(0.3)
    assert deck.secondary.provenance["low_order_sigma_nm_per_coeff"] == pytest.approx(2.0)
