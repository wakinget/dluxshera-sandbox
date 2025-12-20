"""Run the canonical three-plane astrometry demo.

The heavy lifting lives in ``dluxshera.demos.canonical_astrometry`` so notebooks
and tests can import it directly. This wrapper keeps the CLI minimal and
runnable from the examples folder:

```
python examples/scripts/run_canonical_astrometry_demo.py [--fast] [--save-plots] [--add-noise] [--save-plots-dir PATH]
```
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dluxshera.demos.canonical_astrometry import main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical astrometry demo")
    parser.add_argument("--fast", action="store_true", help="Use fewer optimisation steps")
    parser.add_argument("--save-plots", action="store_true", help="Save demo plots")
    parser.add_argument("--save-plots-dir", type=Path, default=None, help="Directory to save plots (implies --save-plots)")
    parser.add_argument("--add-noise", action="store_true", help="Add Gaussian noise to the synthetic PSF")
    parser.add_argument("--run-dir", type=Path, default=None, help="Optional run directory for artifacts")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Base directory for run artifacts (run_id subdir will be created)")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id for artifact writing")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save best/final checkpoints when writing artifacts")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        fast=args.fast,
        save_plots=args.save_plots or args.save_plots_dir is not None,
        add_noise=args.add_noise,
        save_plots_dir=args.save_plots_dir,
        run_dir=args.run_dir,
        runs_dir=args.runs_dir,
        run_id=args.run_id,
        save_checkpoints=args.save_checkpoints,
    )
