"""Smoke test for the two-plane astrometry demo script."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from Examples.scripts.run_twoplane_astrometry_demo import main


def test_twoplane_astrometry_demo_runs(tmp_path):
    result = main(fast=True, save_plots=True, add_noise=False, save_plots_dir=tmp_path)
    assert result.truth_psf is not None
    assert result.noisy_psf is not None
    assert any(tmp_path.glob("*.png"))
