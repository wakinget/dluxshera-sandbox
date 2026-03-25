from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


def _write_trace_csv(path: Path, n_frame: int) -> None:
    fieldnames = [
        "frame_index",
        "time_s",
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n_frame):
            writer.writerow(
                {
                    "frame_index": i,
                    "time_s": 0.05 * i,
                    "source.x_position_as": 0.001 * i,
                    "source.y_position_as": -0.001 * i,
                    "source.position_angle_deg": 25.0 + i,
                }
            )


def test_visualize_obs_subblock_script_smoke(tmp_path: Path):
    n_frame = 6
    cube = np.random.default_rng(123).random((n_frame, 12, 12))

    cube_path = tmp_path / "obs_subblock_20260101-000000_cube.fits"
    fits.PrimaryHDU(cube).writeto(cube_path)

    trace_path = tmp_path / "obs_subblock_20260101-000000_frame_truth.csv"
    _write_trace_csv(trace_path, n_frame=n_frame)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "generator": "examples/recipes/observation_subblock.py",
                "system": {"preset": "SHERA_TESTBED_3P"},
                "artifacts": {
                    "cube_fits": cube_path.name,
                    "frame_truth_csv": trace_path.name,
                },
            }
        ),
        encoding="utf-8",
    )

    outdir = tmp_path / "quicklook"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    subprocess.run(
        [
            sys.executable,
            "examples/scripts/visualize_obs_subblock.py",
            "--cube",
            str(cube_path),
            "--manifest",
            str(manifest_path),
            "--outdir",
            str(outdir),
            "--stride",
            "1",
            "--stretch",
            "linear",
        ],
        check=True,
        env=env,
    )

    for name in ["preview.gif", "summary.png", "trace_summary.png"]:
        output_path = outdir / name
        assert output_path.exists(), f"Missing output {output_path}"
        assert output_path.stat().st_size > 0
