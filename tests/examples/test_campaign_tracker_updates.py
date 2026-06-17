from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRACKER = ROOT / "Results" / "campaign notes" / "Campaign Tracker.md"


def test_campaign_tracker_contains_new_full_fidelity_hpc_entries() -> None:
    text = TRACKER.read_text(encoding="utf-8")
    for name in (
        "full_fidelity_registration_solve_smoke_hpc_v1",
        "full_fidelity_zernike_2x2_self_correction_hpc_v1",
    ):
        assert name in text
        assert "Config/preflight pending" in text
    assert "local X/Y/PA registration solve" in text
    assert "must" not in text.lower() or "truth_when_available" in text
    assert "12 subblocks" in text
    assert "physical_full" in text
    assert "small high-order WFE knowledge-error" in text
    assert "2x2 M1/M2 Zernike sigma matrix" in text
    assert "300 subblocks" in text
    assert "eigenbasis-update campaign should rerun the same matrix" in text
