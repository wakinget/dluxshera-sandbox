"""Render an observation sub-block cube from a canonical explicit trace.

This recipe is the middle stage of the observation sub-block workflow:

1. a trace-builder produces a canonical per-frame CSV trace
2. this renderer applies those frame updates to one shared resolved system
3. downstream quick-look or inference consumes the rendered cube plus manifest

The saved manifest is intended to make a render run reviewable on its own. In
addition to artifact paths and trace metadata, it records the resolved system
config snapshot, shared truth overrides, noise settings, and the source config
path used for the render.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np
from tqdm import tqdm

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.noise import apply_observation_noise
from dluxshera.utils.obs_subblock_keys import (
    OBS_SUBBLOCK_V1_DEFAULT_VARYING_KEYS,
    apply_obs_subblock_overrides_preserving_derived,
    canonical_obs_subblock_varying_keys,
    get_obs_subblock_store_value,
    parse_obs_subblock_varying_keys,
    partition_obs_subblock_overrides_by_kind,
    split_obs_subblock_frame_overrides,
    validate_supported_obs_subblock_key_addresses,
)
from dluxshera.utils.obs_subblock_io import (
    build_obs_subblock_artifact_paths,
    build_obs_subblock_manifest,
    find_obs_subblock_sidecar_manifest,
    now_iso_local_ms,
    timestamp_tag,
    to_jsonable_obs_subblock_payload,
    write_obs_subblock_cube_fits,
    write_obs_subblock_manifest,
    write_obs_subblock_truth_csv,
)
from dluxshera.utils.obs_subblock_trace import (
    load_obs_subblock_trace_csv,
)


DEFAULT_PRESCRIPTION_PATH = Path(
    "examples/recipes/observation_subblock_template/subblock_generation_prescription.yaml"
)
DEFAULT_OUTDIR_ROOT = Path("Results/obs_subblocks")
MANIFEST_SCHEMA_VERSION = "obs_subblock_manifest.v1"
GENERATOR_ID = "examples/recipes/observation_subblock.py"


def _required_dict(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict.")
    return value


def _required_int(parent: dict[str, Any], key: str, *, path: str) -> int:
    value = parent.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{path}.{key} must be an integer.")
    return value


def _required_str(parent: dict[str, Any], key: str, *, path: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return value


def _resolve_relative_path(
    value: str,
    *,
    config_path: Path | None,
    field_name: str,
) -> Path:
    """Resolve a config path relative to the prescription file when relative."""

    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    if config_path is not None:
        return (config_path.parent / raw_path).resolve()
    raise ValueError(
        f"{field_name} is relative but config_path is not available to resolve it."
    )


def _flatten_truth_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested truth overrides into dotted store keys."""

    flattened: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                joined = f"{prefix}.{key}" if prefix else str(key)
                _walk(joined, child)
        else:
            flattened[prefix] = value

    _walk("", payload)
    return flattened


def _stable_hash_payload(payload: Any) -> str:
    """Return a stable SHA256 hash for a JSON-serializable payload."""

    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_experiment_cfg(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the experiment block for observation sub-blocks."""

    kind = _required_str(experiment_cfg, "kind", path="experiment")
    if kind != "subblock_generation":
        raise ValueError(
            "experiment.kind must be 'subblock_generation' for this recipe."
        )

    seed = _required_int(experiment_cfg, "seed", path="experiment")

    truth = experiment_cfg.get("truth", {})
    if not isinstance(truth, dict):
        raise ValueError("experiment.truth must be a mapping/dict.")

    subblock_cfg = _required_dict(
        experiment_cfg, "subblock", path="experiment"
    )
    varying_keys_value = subblock_cfg.get("varying_keys")
    requested_varying_keys: tuple[str, ...] | None = None
    if varying_keys_value is None:
        varying_keys_input = list(OBS_SUBBLOCK_V1_DEFAULT_VARYING_KEYS)
    else:
        if not isinstance(varying_keys_value, list):
            raise ValueError(
                "experiment.observation_subblock.varying_keys must be a list[str] when provided."
            )
        varying_keys_input = list(varying_keys_value)
        requested_varying_keys = tuple(varying_keys_value)
    varying_addresses = parse_obs_subblock_varying_keys(varying_keys_input)
    validate_supported_obs_subblock_key_addresses(varying_addresses)
    applied_varying_keys = canonical_obs_subblock_varying_keys(varying_addresses)

    trace_cfg = _required_dict(subblock_cfg, "trace", path="experiment.observation_subblock")
    trace_format = trace_cfg.get("format", "csv")
    if trace_format != "csv":
        raise ValueError(
            "experiment.observation_subblock.trace.format currently supports only 'csv'."
        )
    trace_path = _required_str(trace_cfg, "path", path="experiment.observation_subblock.trace")

    validate_cfg = subblock_cfg.get("validate", {})
    if not isinstance(validate_cfg, dict):
        raise ValueError("experiment.observation_subblock.validate must be a mapping/dict.")
    require_contiguous = validate_cfg.get("require_contiguous_frame_index", True)
    require_monotonic = validate_cfg.get("require_monotonic_time", True)
    if not isinstance(require_contiguous, bool):
        raise ValueError(
            "experiment.observation_subblock.validate.require_contiguous_frame_index "
            "must be a bool."
        )
    if not isinstance(require_monotonic, bool):
        raise ValueError(
            "experiment.observation_subblock.validate.require_monotonic_time must be a bool."
        )

    noise = experiment_cfg.get("noise", {})
    if not isinstance(noise, dict):
        raise ValueError("experiment.noise must be a mapping/dict.")
    noise_enabled = noise.get("enabled", False)
    if not isinstance(noise_enabled, bool):
        raise ValueError("experiment.noise.enabled must be a bool when provided.")
    noise_cfg = {
        "enabled": noise_enabled,
        "photon_noise": bool(noise.get("photon_noise", True)),
        "read_noise": bool(noise.get("read_noise", False)),
        "dark_current": bool(noise.get("dark_current", False)),
    }

    outputs = experiment_cfg.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("experiment.outputs must be a mapping/dict.")
    file_prefix = outputs.get("file_prefix", "obs_subblock")
    if not isinstance(file_prefix, str) or not file_prefix.strip():
        raise ValueError("experiment.outputs.file_prefix must be a non-empty string.")
    frame_truth_format = outputs.get("frame_truth_format", "csv")
    if frame_truth_format != "csv":
        raise ValueError(
            "experiment.outputs.frame_truth_format currently supports only 'csv'."
        )
    outdir_value = outputs.get("outdir")
    if outdir_value is not None and (
        not isinstance(outdir_value, str) or not outdir_value.strip()
    ):
        raise ValueError("experiment.outputs.outdir must be a non-empty string when set.")

    notes_value = experiment_cfg.get("notes")
    if notes_value is not None and not isinstance(notes_value, str):
        raise ValueError("experiment.notes must be a string when provided.")

    return {
        "kind": kind,
        "seed": seed,
        "truth": truth,
        "observation_subblock": {
            "requested_varying_keys": requested_varying_keys,
            "applied_varying_keys": applied_varying_keys,
            "trace": {"format": "csv", "path": trace_path},
            "validate": {
                "require_contiguous_frame_index": require_contiguous,
                "require_monotonic_time": require_monotonic,
            },
        },
        "noise": noise_cfg,
        "outputs": {
            "outdir": outdir_value,
            "file_prefix": file_prefix.strip(),
            "frame_truth_format": "csv",
        },
        "notes": notes_value,
    }


def generate_obs_subblock(
    *,
    config_path: Path | None = None,
    system_preset: str | None = None,
    results_dir: Path | None = None,
    run_name: str | None = None,
    dry_run: bool = False,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Render one observation sub-block from an explicit per-frame trace CSV.

    Parameters
    ----------
    config_path : Path | None, optional
        YAML/JSON config path with canonical top-level ``system`` and
        ``experiment`` blocks. Defaults to the bundled template.
    system_preset : str | None, optional
        Optional system preset override merged before ``config_path`` content.
    results_dir : Path | None, optional
        Optional output root override. When omitted, uses
        ``experiment.outputs.outdir`` if provided, else
        ``Results/observation_subblock``.
    run_name : str | None, optional
        Optional run directory label under the output root. Defaults to
        timestamp tag.
    dry_run : bool, optional
        Validate config/trace and report expected outputs without rendering.
    show_progress : bool | None, optional
        Controls tqdm progress output during per-frame rendering. ``None``
        auto-detects from terminal interactivity.

    Returns
    -------
    dict[str, Any]
        Summary including output directory and artifact paths.
    """

    cfg_path = Path(config_path) if config_path is not None else DEFAULT_PRESCRIPTION_PATH
    progress_enabled = bool(sys.stderr.isatty()) if show_progress is None else bool(show_progress)

    print(f"Loading observation sub-block config from: {cfg_path}")
    user_cfg = load_user_config(
        config_path=cfg_path,
        system_preset=system_preset,
        experiment_preset=None,
    )
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    experiment_cfg = resolved_cfg.get("experiment")

    if system_cfg is None:
        raise ValueError("Observation sub-block recipe requires a top-level 'system' block.")
    if experiment_cfg is None:
        raise ValueError(
            "Observation sub-block recipe requires a top-level 'experiment' block."
        )

    experiment = _validate_experiment_cfg(experiment_cfg)

    requested_varying_keys = experiment["observation_subblock"]["requested_varying_keys"]
    applied_varying_keys = tuple(experiment["observation_subblock"]["applied_varying_keys"])
    varying_addresses = parse_obs_subblock_varying_keys(list(applied_varying_keys))
    trace_cfg = experiment["observation_subblock"]["trace"]
    trace_path = _resolve_relative_path(
        trace_cfg["path"],
        config_path=cfg_path,
        field_name="experiment.observation_subblock.trace.path",
    )
    print(f"Loading trace CSV from: {trace_path}")
    if requested_varying_keys is not None and tuple(requested_varying_keys) != tuple(
        applied_varying_keys
    ):
        requested_text = ", ".join(str(key) for key in requested_varying_keys)
        applied_text = ", ".join(applied_varying_keys)
        print(
            "Note: varying_keys were normalized to canonical renderer keys. "
            f"requested=[{requested_text}] applied=[{applied_text}]"
        )

    configured_outdir = experiment["outputs"]["outdir"]
    if results_dir is not None:
        outdir_root = Path(results_dir)
    elif configured_outdir is not None:
        outdir_root = _resolve_relative_path(
            configured_outdir,
            config_path=cfg_path,
            field_name="experiment.outputs.outdir",
        )
    else:
        outdir_root = DEFAULT_OUTDIR_ROOT

    stamp = timestamp_tag()
    run_label = run_name or stamp
    outdir = outdir_root / run_label
    artifacts = build_obs_subblock_artifact_paths(
        outdir=outdir,
        file_prefix=experiment["outputs"]["file_prefix"],
        timestamp=stamp,
    )
    print(f"Preparing output directory: {outdir}")

    print("Building forward specification and base truth state...")
    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec)
    validate_supported_obs_subblock_key_addresses(
        varying_addresses,
        forward_spec=forward_spec,
        reference_store=base_store,
    )

    truth_overrides = _flatten_truth_overrides(experiment["truth"])
    (
        primitive_truth_overrides,
        derived_truth_overrides,
        unknown_truth_keys,
    ) = partition_obs_subblock_overrides_by_kind(
        truth_overrides,
        forward_spec=forward_spec,
    )
    if unknown_truth_keys:
        raise ValueError(
            "experiment.truth contains keys not present or unsupported in "
            "forward_spec: "
            + ", ".join(sorted(unknown_truth_keys.keys()))
        )
    base_store = apply_obs_subblock_overrides_preserving_derived(
        base_store,
        forward_spec=forward_spec,
        primitive_overrides=primitive_truth_overrides,
        derived_overrides=derived_truth_overrides,
    )

    validate_cfg = experiment["observation_subblock"]["validate"]
    trace = load_obs_subblock_trace_csv(
        trace_path,
        required_varying_keys=applied_varying_keys,
        require_contiguous_frame_index=validate_cfg["require_contiguous_frame_index"],
        require_monotonic_time=validate_cfg["require_monotonic_time"],
    )

    if dry_run:
        print("Dry run: validated configuration and trace.")
        print(f"  frames: {trace.frame_count}")
        print(f"  output_dir: {outdir}")
        print(f"  expected_cube: {artifacts['cube_fits']}")
        print(f"  expected_truth_csv: {artifacts['frame_truth_csv']}")
        print(f"  expected_manifest: {artifacts['manifest_json']}")
        return {
            "dry_run": True,
            "frame_count": trace.frame_count,
            "output_dir": str(outdir),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
        }

    outdir.mkdir(parents=True, exist_ok=True)

    print("Constructing binder...")
    binder = SheraBinder(system_cfg, forward_spec, base_store)

    noise_cfg = experiment["noise"]
    rng_key = jr.PRNGKey(int(experiment["seed"]))

    print(
        "Rendering frames..."
        + (" (first frame may include JIT compilation)" if trace.frame_count > 0 else "")
    )
    frame_images: list[np.ndarray] = []
    resolved_truth_rows: list[dict[str, Any]] = []
    frame_iter = trace.rows
    if progress_enabled:
        frame_iter = tqdm(
            trace.rows,
            total=trace.frame_count,
            desc="obs_subblock frames",
            unit="frame",
            leave=False,
        )
    for trace_row in frame_iter:
        primitive_overrides, derived_overrides = split_obs_subblock_frame_overrides(
            base_store=base_store,
            forward_spec=forward_spec,
            addresses=varying_addresses,
            values_by_key=trace_row,
        )
        frame_store = apply_obs_subblock_overrides_preserving_derived(
            base_store,
            forward_spec=forward_spec,
            primitive_overrides=primitive_overrides,
            derived_overrides=derived_overrides,
        )
        frame_delta = binder.strip_structural(frame_store)

        frame_image = binder.model(frame_delta)
        if noise_cfg["enabled"]:
            rng_key, noise_key = jr.split(rng_key)
            frame_image, _ = apply_observation_noise(
                frame_image,
                noise_cfg=noise_cfg,
                rng_key=noise_key,
                detector_spec=getattr(binder.detector, "spec", None),
                exposure_time_s=frame_store.get("source.exposure_time_s", default=None),
            )

        frame_images.append(np.asarray(frame_image))

        resolved_row = dict(trace_row)
        for address in varying_addresses:
            resolved_row[address.canonical] = get_obs_subblock_store_value(
                frame_store,
                address=address,
            )
        resolved_truth_rows.append(resolved_row)

    print("Writing FITS cube, truth CSV, and manifest...")
    cube = np.stack(frame_images, axis=0)
    write_obs_subblock_cube_fits(
        output_path=artifacts["cube_fits"],
        cube=cube,
        header_cards={
            "SCHEMA": MANIFEST_SCHEMA_VERSION,
            "NFRAME": cube.shape[0],
        },
    )

    truth_columns = tuple((*trace.required_columns, *trace.extra_columns))
    write_obs_subblock_truth_csv(
        output_path=artifacts["frame_truth_csv"],
        rows=resolved_truth_rows,
        fieldnames=truth_columns,
    )

    trace_manifest_path = find_obs_subblock_sidecar_manifest(trace.source_path)
    manifest_inputs = {
        "config_path": str(cfg_path.resolve()),
    }
    if system_preset is not None:
        manifest_inputs["system_preset_override"] = str(system_preset)
    if trace_manifest_path is not None:
        manifest_inputs["trace_manifest_json"] = str(trace_manifest_path)

    system_info = {
        "preset": system_cfg.get("preset"),
        "config_hash": _stable_hash_payload(system_cfg),
        "resolved_config": to_jsonable_obs_subblock_payload(system_cfg),
    }
    manifest = build_obs_subblock_manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at=now_iso_local_ms(),
        generator=GENERATOR_ID,
        frame_count=trace.frame_count,
        varying_keys=applied_varying_keys,
        requested_varying_keys=requested_varying_keys,
        applied_varying_keys=applied_varying_keys,
        trace_format="csv",
        trace_path=trace.source_path,
        trace_extra_columns=trace.extra_columns,
        artifacts=artifacts,
        outdir=outdir,
        time_start_s=trace.time_start_s,
        time_stop_s=trace.time_stop_s,
        inputs=manifest_inputs,
        system_info=system_info,
        shared_truth=experiment["truth"],
        seed=int(experiment["seed"]),
        noise=noise_cfg,
        notes=experiment.get("notes"),
    )
    write_obs_subblock_manifest(output_path=artifacts["manifest_json"], manifest=manifest)

    print(f"Rendered observation sub-block with {trace.frame_count} frames.")
    print(f"Wrote artifacts under: {outdir}")

    return {
        "dry_run": False,
        "frame_count": trace.frame_count,
        "output_dir": str(outdir),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render an observation sub-block from trace CSV.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PRESCRIPTION_PATH,
        help="Path to observation sub-block prescription YAML/JSON.",
    )
    parser.add_argument(
        "--system-preset",
        type=str,
        default=None,
        help="Optional system preset override merged before config file content.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional output root override.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory label under the output root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate config/trace and print expected outputs without rendering.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable tqdm frame progress output.",
    )
    return parser


def main() -> None:
    """Run the observation sub-block renderer CLI."""

    args = _build_parser().parse_args()
    generate_obs_subblock(
        config_path=args.config,
        system_preset=args.system_preset,
        results_dir=args.results_dir,
        run_name=args.run_name,
        dry_run=bool(args.dry_run),
        show_progress=False if bool(args.no_progress) else None,
    )


if __name__ == "__main__":
    main()
