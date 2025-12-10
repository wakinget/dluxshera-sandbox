"""Smoke test for the canonical astrometry demo script."""

import sys
from pathlib import Path

# Ensure repository root and src/ are importable for the demo script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from Examples.scripts.run_canonical_astrometry_demo import main


def test_canonical_astrometry_demo_runs():
    # Fast mode should keep runtime low and avoid file outputs.
    result = main(fast=True, save_plots=False, add_noise=False)
    assert result.truth_psf is not None
    assert result.variant_psf is not None
