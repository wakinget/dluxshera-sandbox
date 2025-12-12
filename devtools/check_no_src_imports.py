"""Fail if ``src.*`` imports exist in tracked Python files.

This guardrail ensures modules import the installed ``dluxshera`` package
instead of relying on the repository layout being on ``PYTHONPATH``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PATTERN = r"^\s*(from|import)\s+src\."
SEARCH_DIRS = ("src", "tests", "devtools", "Examples")


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    existing_paths = [str(repo_root / path) for path in SEARCH_DIRS if (repo_root / path).exists()]

    if not existing_paths:
        print("No search targets found.")
        return 0

    cmd = [
        "rg",
        "--hidden",
        "--iglob",
        "*.py",
        "-n",
        PATTERN,
        *existing_paths,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode not in (0, 1):
        print("ripgrep failed:")
        print(result.stderr.strip())
        return result.returncode

    if result.stdout.strip():
        print("Found disallowed src.* imports:\n")
        print(result.stdout.strip())
        return 1

    print("No src.* imports detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
