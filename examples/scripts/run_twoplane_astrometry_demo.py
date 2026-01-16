"""Deprecated two-plane astrometry demo entrypoint.

This script previously hosted a lightweight two-plane demo. The canonical
workflow now lives in:
- examples/recipes/twoplane_astrometry.py (read-first recipe)
- examples/runners/run_twoplane_astrometry.py (execute-first runner)
"""
from __future__ import annotations

import sys


def main() -> None:
    message = (
        "This demo has moved. Use:\n"
        "  - examples/recipes/twoplane_astrometry.py (read-first)\n"
        "  - examples/runners/run_twoplane_astrometry.py (execute-first)"
    )
    print(message)


if __name__ == "__main__":
    main()
    sys.exit(0)
