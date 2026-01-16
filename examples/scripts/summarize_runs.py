"""Summarize run artifact directories into a sweep CSV.

This script is a lightweight CLI wrapper around
``dluxshera.inference.sweeps.write_sweep_csv``. It scans a run directory (or
its immediate children) for per-run artifact folders containing both
``meta.json`` and ``summary.json``, flattens key metadata, and writes the
resulting table to a CSV file for quick inspection or downstream analysis.

Use this when you have one or more completed runs on disk and want a compact
tabular summary. It is intentionally shallow: only the provided directory and
its direct subdirectories are considered run candidates.

Examples
--------
Summarize runs stored under a common root directory and write the default CSV:

```
python examples/scripts/summarize_runs.py --runs-dir path/to/runs
```

Write to a custom CSV path:

```
python examples/scripts/summarize_runs.py \
    --runs-dir path/to/runs \
    --out path/to/runs/sweep_summary.csv
```

Add extra metadata columns (repeatable for multiple keys):

```
python examples/scripts/summarize_runs.py \
    --runs-dir path/to/runs \
    --extra-meta optimizer.learning_rate \
    --extra-meta theta.theta_space
```
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dluxshera.inference.sweeps import write_sweep_csv


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the sweep CSV writer.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``runs_dir``, optional ``out``, and any repeated
        ``extra_meta`` keys.
    """
    parser = argparse.ArgumentParser(description="Summarize runs into a sweep CSV.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="Directory containing run subdirectories.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (defaults to <runs-dir>/sweep_summary.csv).",
    )
    parser.add_argument(
        "--extra-meta",
        action="append",
        default=[],
        help="Additional dotted meta keys to include as columns (can be repeated).",
    )
    return parser.parse_args()


def main() -> None:
    """Write a sweep CSV from run artifacts and report the row count."""
    args = parse_args()
    out = args.out or (args.runs_dir / "sweep_summary.csv")

    count = write_sweep_csv(args.runs_dir, out, include_meta_fields=args.extra_meta)
    print(f"Wrote {count} runs to {out}")


if __name__ == "__main__":
    main()
