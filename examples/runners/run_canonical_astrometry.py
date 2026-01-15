"""
Run the canonical three-plane astrometry retrieval recipe.

This file is a thin entrypoint that executes the canonical astrometry *recipe*
script (the read-first, top-to-bottom workflow). Use this runner when you want
a simple “just run it” command (e.g., from a terminal, VS Code task, or CI),
without burying the workflow behind demo-specific helpers.

How to run
- From the repository root (recommended):
    python examples/runners/run_canonical_astrometry.py

- In VS Code:
    Open this file and use “Run Python File” (or the ▶ button).

Configuration / options
- This runner is intentionally minimal and typically does not expose many CLI
  arguments. To change model/config/inference options (including enabling or
  disabling eigenmodes), edit the options block near the top of the canonical
  recipe script that this runner calls.

Outputs
- The recipe will create a run directory and save artifacts/plots using the
  repository’s built-in run-artifacts and plotting utilities. Check the console
  output for the exact run directory path.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def load_recipe():
    recipe_path = Path(__file__).resolve().parents[1] / "recipes" / "canonical_astrometry.py"
    spec = importlib.util.spec_from_file_location("canonical_astrometry_recipe", recipe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recipe at {recipe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the canonical astrometry recipe")
    parser.add_argument("--fast", action="store_true", help="Use fewer optimisation steps")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recipe = load_recipe()
    recipe.main(fast=args.fast)
