from __future__ import annotations

from pathlib import Path

from dluxshera.utils.obs_subblock_io import (
    build_obs_subblock_artifact_paths,
    build_obs_subblock_manifest,
    write_obs_subblock_manifest,
)


def test_obs_subblock_manifest_contains_required_fields_and_relative_artifacts(tmp_path):
    outdir = tmp_path / "run_001"
    outdir.mkdir(parents=True, exist_ok=True)
    trace_path = tmp_path / "trace.csv"

    artifacts = build_obs_subblock_artifact_paths(
        outdir=outdir,
        file_prefix="obs_subblock",
        timestamp="20260324-120000",
    )
    manifest = build_obs_subblock_manifest(
        schema_version="obs_subblock_manifest.v1",
        created_at="2026-03-24T12:00:00.000",
        generator="examples/recipes/observation_subblock.py",
        frame_count=3,
        varying_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
        requested_varying_keys=("source.x_position_as",),
        applied_varying_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
        trace_format="csv",
        trace_path=trace_path,
        trace_extra_columns=("note",),
        artifacts=artifacts,
        outdir=outdir,
        time_start_s=0.0,
        time_stop_s=0.1,
        inputs={"config_path": "/tmp/prescription.yaml"},
        system_info={"preset": "SHERA_TESTBED_3P"},
        shared_truth={"source": {"exposure_time_s": 0.05}},
        seed=42,
        noise={"enabled": False},
        runtime_info={"jax_enable_x64": True},
        render_info={
            "cube_dtype": "float64",
            "cube_dtype_source": "in_memory_before_fits_write",
        },
    )

    manifest_path = outdir / "manifest.json"
    write_obs_subblock_manifest(output_path=manifest_path, manifest=manifest)

    assert manifest["schema_version"] == "obs_subblock_manifest.v1"
    assert manifest["frame_count"] == 3
    assert manifest["varying_keys"] == [
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
    ]
    assert manifest["applied_varying_keys"] == manifest["varying_keys"]
    assert manifest["requested_varying_keys"] == ["source.x_position_as"]
    assert manifest["trace"]["format"] == "csv"
    assert manifest["trace"]["path"] == "../trace.csv"
    assert manifest["trace"]["extra_columns"] == ["note"]
    assert manifest["artifacts"]["manifest_json"] == "manifest.json"
    assert manifest["artifacts"]["cube_fits"].endswith("_cube.fits")
    assert not manifest["artifacts"]["cube_fits"].startswith("/")
    assert manifest["artifacts"]["frame_truth_csv"].endswith("_frame_truth.csv")
    assert manifest["inputs"]["config_path"] == "/tmp/prescription.yaml"
    assert manifest["shared_truth"]["source"]["exposure_time_s"] == 0.05
    assert manifest["seed"] == 42
    assert manifest["noise"]["enabled"] is False
    assert manifest["runtime"]["jax_enable_x64"] is True
    assert manifest["render"]["cube_dtype"] == "float64"
    assert manifest["render"]["cube_dtype_source"] == "in_memory_before_fits_write"
    assert manifest_path.exists()
