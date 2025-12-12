# devtools/print_tree.py
from __future__ import annotations
from pathlib import Path
import argparse

# Import shared helpers
from devtools.introspection import (
    print_tree,
    save_project_index_json,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print project tree and optionally export a JSON code index."
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--json-out", type=str, default=None)

    args = parser.parse_args()
    repo_root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]

    # Print tree (original behavior)
    print_tree(repo_root, max_depth=args.max_depth)

    # Optional JSON index
    if args.json_out:
        save_project_index_json(repo_root, Path(args.json_out), max_depth=args.max_depth)
        print(f"\n[print_tree] Wrote JSON index to {args.json_out}")
