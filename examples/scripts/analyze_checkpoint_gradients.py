from __future__ import annotations

import argparse
from pathlib import Path

from dluxshera.inference.diagnostics import compute_checkpoint_gradients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute gradients at a saved checkpoint.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing checkpoint files.")
    parser.add_argument(
        "--checkpoint",
        choices=["best", "final"],
        default="best",
        help="Which checkpoint to load (default: best).",
    )
    parser.add_argument(
        "--builder",
        required=True,
        help="Loss builder (callable or 'module:func' string). The builder should return a loss(theta) callable.",
    )
    parser.add_argument(
        "--compute-curvature",
        action="store_true",
        help="Also compute diagonal curvature and learning-rate vectors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = compute_checkpoint_gradients(
        args.run_dir,
        builder=args.builder,
        checkpoint=args.checkpoint,
        compute_curvature=bool(args.compute_curvature),
    )
    artifact = summary.get("artifact")
    print(f"Saved gradient diagnostics to {artifact}")


if __name__ == "__main__":
    main()
