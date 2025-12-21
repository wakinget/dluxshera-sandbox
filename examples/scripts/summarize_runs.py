from __future__ import annotations

import argparse
from pathlib import Path

from dluxshera.inference.sweeps import write_sweep_csv


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    out = args.out or (args.runs_dir / "sweep_summary.csv")

    count = write_sweep_csv(args.runs_dir, out, include_meta_fields=args.extra_meta)
    print(f"Wrote {count} runs to {out}")


if __name__ == "__main__":
    main()
