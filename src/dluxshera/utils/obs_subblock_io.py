"""Artifact and manifest helpers for observation sub-block workflows."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy.io import fits


def timestamp_tag(*, now: dt.datetime | None = None) -> str:
    """Return a sortable timestamp tag ``YYYYMMDD-HHMMSS``."""

    current = now or dt.datetime.now()
    return current.strftime("%Y%m%d-%H%M%S")


def now_iso_local_ms(*, now: dt.datetime | None = None) -> str:
    """Return local timestamp with millisecond precision."""

    current = now or dt.datetime.now()
    return current.isoformat(timespec="milliseconds")


def build_obs_subblock_artifact_paths(
    *,
    outdir: Path,
    file_prefix: str,
    timestamp: str,
) -> dict[str, Path]:
    """Return canonical artifact paths for one rendered sub-block."""

    cube_path = outdir / f"{file_prefix}_{timestamp}_cube.fits"
    truth_path = outdir / f"{file_prefix}_{timestamp}_frame_truth.csv"
    manifest_path = outdir / "manifest.json"
    return {
        "cube_fits": cube_path,
        "frame_truth_csv": truth_path,
        "manifest_json": manifest_path,
    }


def write_obs_subblock_cube_fits(
    *,
    output_path: Path,
    cube: np.ndarray,
    header_cards: Mapping[str, Any] | None = None,
) -> None:
    """Write a rendered observation sub-block cube to FITS."""

    header = fits.Header()
    if header_cards:
        for key, value in header_cards.items():
            if value is None:
                continue
            header.set(str(key).upper(), value)
    fits.PrimaryHDU(data=np.asarray(cube), header=header).writeto(
        output_path, overwrite=True
    )


def write_obs_subblock_truth_csv(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | list[str],
) -> None:
    """Write per-frame truth rows to CSV."""

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def to_jsonable_obs_subblock_payload(payload: Any) -> Any:
    """Return a JSON-compatible snapshot of a manifest payload fragment."""

    return json.loads(json.dumps(payload, default=str))


def find_obs_subblock_sidecar_manifest(artifact_path: Path) -> Path | None:
    """Return ``artifact_path.parent / manifest.json`` when it exists."""

    candidate = artifact_path.resolve().parent / "manifest.json"
    if not candidate.exists():
        return None
    return candidate


def _relative_path(path: Path, *, outdir: Path) -> str:
    """Return a POSIX path relative to ``outdir`` when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(outdir.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, outdir.resolve())).as_posix()


def build_obs_subblock_manifest(
    *,
    schema_version: str,
    created_at: str,
    generator: str,
    frame_count: int,
    varying_keys: tuple[str, ...] | list[str],
    requested_varying_keys: tuple[str, ...] | list[str] | None = None,
    applied_varying_keys: tuple[str, ...] | list[str] | None = None,
    trace_format: str,
    trace_path: Path,
    trace_extra_columns: tuple[str, ...] | list[str],
    artifacts: Mapping[str, Path],
    outdir: Path,
    time_start_s: float | None,
    time_stop_s: float | None,
    inputs: Mapping[str, Any] | None = None,
    system_info: Mapping[str, Any] | None = None,
    shared_truth: Mapping[str, Any] | None = None,
    seed: int | None = None,
    noise: Mapping[str, Any] | None = None,
    runtime_info: Mapping[str, Any] | None = None,
    render_info: Mapping[str, Any] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build a manifest payload for an observation sub-block run."""

    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "created_at": created_at,
        "generator": generator,
        "frame_count": int(frame_count),
        "varying_keys": list(varying_keys),
        "applied_varying_keys": (
            list(applied_varying_keys)
            if applied_varying_keys is not None
            else list(varying_keys)
        ),
        "trace": {
            "format": trace_format,
            "path": _relative_path(trace_path, outdir=outdir),
            "extra_columns": list(trace_extra_columns),
        },
        "time_start_s": None if time_start_s is None else float(time_start_s),
        "time_stop_s": None if time_stop_s is None else float(time_stop_s),
        "artifacts": {
            name: _relative_path(Path(path), outdir=outdir)
            for name, path in artifacts.items()
        },
    }
    if inputs is not None:
        manifest["inputs"] = to_jsonable_obs_subblock_payload(dict(inputs))
    if system_info is not None:
        manifest["system"] = to_jsonable_obs_subblock_payload(dict(system_info))
    if shared_truth is not None:
        manifest["shared_truth"] = to_jsonable_obs_subblock_payload(dict(shared_truth))
    if seed is not None:
        manifest["seed"] = int(seed)
    if noise is not None:
        manifest["noise"] = to_jsonable_obs_subblock_payload(dict(noise))
    if runtime_info is not None:
        manifest["runtime"] = to_jsonable_obs_subblock_payload(dict(runtime_info))
    if render_info is not None:
        manifest["render"] = to_jsonable_obs_subblock_payload(dict(render_info))
    if notes is not None:
        manifest["notes"] = str(notes)
    if requested_varying_keys is not None:
        manifest["requested_varying_keys"] = list(requested_varying_keys)
    return manifest


def write_obs_subblock_manifest(*, output_path: Path, manifest: Mapping[str, Any]) -> None:
    """Write an observation sub-block manifest JSON file."""

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, default=str)


__all__ = [
    "build_obs_subblock_artifact_paths",
    "build_obs_subblock_manifest",
    "find_obs_subblock_sidecar_manifest",
    "now_iso_local_ms",
    "to_jsonable_obs_subblock_payload",
    "timestamp_tag",
    "write_obs_subblock_cube_fits",
    "write_obs_subblock_manifest",
    "write_obs_subblock_truth_csv",
]
