"""Thin runner for the canonical astrometry recipe."""
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
