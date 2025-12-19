"""Smoke test for the canonical astrometry demo script."""

from dluxshera.demos.canonical_astrometry import main


def test_canonical_astrometry_demo_runs(tmp_path):
    # Fast mode should keep runtime low and can write plots to a temp dir.
    result = main(fast=True, save_plots=False, add_noise=False, save_plots_dir=tmp_path)
    assert result.truth_psf is not None
    assert result.variant_psf is not None
    # Ensure at least one plot is produced when a directory is provided.
    assert (tmp_path / "psf_truth.png").exists()
