"""Smoke test for the two-plane astrometry demo script."""

import sys
from importlib import util
from pathlib import Path


def _load_twoplane_main():
    module_path = Path(__file__).resolve().parents[2] / "examples" / "scripts" / "run_twoplane_astrometry_demo.py"
    spec = util.spec_from_file_location("twoplane_demo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load two-plane demo script")
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


main = _load_twoplane_main()


def test_twoplane_astrometry_demo_runs(tmp_path):
    result = main(fast=True, save_plots=True, add_noise=False, save_plots_dir=tmp_path)
    assert result.truth_psf is not None
    assert result.noisy_psf is not None
    assert any(tmp_path.glob("*.png"))
