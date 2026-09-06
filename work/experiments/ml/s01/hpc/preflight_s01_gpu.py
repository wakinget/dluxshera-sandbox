#!/usr/bin/env python3
from __future__ import annotations

import runpy
import sys
from pathlib import Path


if __name__ == "__main__":
    if "--experiment-id" not in sys.argv:
        sys.argv[1:1] = ["--experiment-id", "S01-E01"]
    target = Path(__file__).resolve().parents[2] / "hpc" / "preflight_ml_gpu.py"
    raise SystemExit(runpy.run_path(str(target), run_name="__main__"))
