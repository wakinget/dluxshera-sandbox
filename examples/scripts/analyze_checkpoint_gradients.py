from __future__ import annotations

"""Analyze gradients at a saved inference checkpoint.

Use this script to compute gradients (and optional curvature diagnostics) at a
previously saved checkpoint from a dluxshera run. This is a lightweight, focused
diagnostic: it loads ``checkpoint_best.npz`` or ``checkpoint_final.npz``, builds
the same loss function used during inference, and evaluates gradients at that
parameter vector to help inspect convergence pathologies or biased minima.

Recommended usage
-----------------
1. Run a standard optimization and ensure checkpoints are saved.
2. Supply a loss builder that returns ``loss(theta) -> scalar``.
3. Inspect the generated artifacts in ``<run_dir>/diag/``.

Examples
--------
Compute gradients at the best checkpoint:

```
python examples/scripts/analyze_checkpoint_gradients.py \
    /path/to/run_dir \
    --builder dluxshera.examples.builders:build_loss
```

Compute gradients plus curvature proxy at the final checkpoint:

```
python examples/scripts/analyze_checkpoint_gradients.py \
    /path/to/run_dir \
    --checkpoint final \
    --builder dluxshera.examples.builders:build_loss \
    --compute-curvature
```
"""

import argparse
from pathlib import Path

from dluxshera.inference.diagnostics import compute_checkpoint_gradients


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint gradient diagnostics."""
    parser = argparse.ArgumentParser(
        description="Compute gradients at a saved checkpoint and write diagnostics.",
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing checkpoint files and metadata.",
    )
    parser.add_argument(
        "--checkpoint",
        choices=["best", "final"],
        default="best",
        help="Which checkpoint to load (default: best).",
    )
    parser.add_argument(
        "--builder",
        required=True,
        help=(
            "Loss builder (callable or 'module:func' string). The builder should "
            "return a loss(theta) callable."
        ),
    )
    parser.add_argument(
        "--compute-metric",
        action="store_true",
        help="Also compute diagonal metric and learning-rate scale vectors.",
    )
    return parser.parse_args()


def main() -> None:
    """Run checkpoint diagnostics and print the saved artifact path."""
    args = parse_args()
    summary = compute_checkpoint_gradients(
        args.run_dir,
        builder=args.builder,
        checkpoint=args.checkpoint,
        compute_metric=bool(args.compute_metric),
    )
    artifact = summary.get("artifact")
    print(f"Saved gradient diagnostics to {artifact}")


if __name__ == "__main__":
    main()
