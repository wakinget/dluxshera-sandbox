"""Observation sub-block renderer (Phase 2 minimal implementation).

This recipe renders a short, explicit-trace image stack from one resolved
system configuration. It is intentionally narrow in v1:
- explicit CSV trace input (no motion-model helpers),
- frame-varying ``source.x/y`` and ``source.position_angle_deg`` only,
- one central-field image cube output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.random as jr
import numpy as np

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.noise import apply_observation_noise
from dluxshera.utils.obs_subblock_io import (
    build_obs_subblock_artifact_paths,
    build_obs_subblock_manifest,
    now_iso_local_ms,
    timestamp_tag,
    write_obs_subblock_cube_fits,
    write_obs_subblock_manifest,
    write_obs_subblock_truth_csv,
)
from dluxshera.utils.obs_subblock_trace import (
    APPLIED_V1_VARYING_KEYS,
    REQUIRED_TRACE_COLUMNS,
    load_obs_subblock_trace_csv,
)


DEFAULT_PRESCRIPTION_PATH = Path(
    "examples/recipes/observation_subblock_template/prescription.yaml"
)
DEFAULT_OUTDIR_ROOT = Path("Results/observation_subblock")
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
    if kind != "observation_subblock":
        raise ValueError(
            "experiment.kind must be 'observation_subblock' for this recipe."
        )

    seed = _required_int(experiment_cfg, "seed", path="experiment")

    truth = experiment_cfg.get("truth", {})
    if not isinstance(truth, dict):
        raise ValueError("experiment.truth must be a mapping/dict.")

    subblock_cfg = _required_dict(
        experiment_cfg, "observation_subblock", path="experiment"
    )
    varying_keys_value = subblock_cfg.get("varying_keys")
    requested_varying_keys: tuple[str, ...] | None = None
    if varying_keys_value is not None:
        if not isinstance(varying_keys_value, list) or not all(
            isinstance(item, str) for item in varying_keys_value
        ):
            raise ValueError(
                "experiment.observation_subblock.varying_keys must be a list[str] when provided."
            )
        requested_varying_keys = tuple(varying_keys_value)

    trace_cfg = _required_dict(subblock_cfg, "trace", path="experiment.observation_subblock")
    trace_format = trace_cfg.get("format", "csv")
    if trace_format != "csv":
        raise ValueError(
            "experiment.observation_subblock.trace.format must be 'csv' in v1."
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
            "experiment.outputs.frame_truth_format must be 'csv' in v1."
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
            "applied_varying_keys": APPLIED_V1_VARYING_KEYS,
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

    Returns
    -------
    dict[str, Any]
        Summary including output directory and artifact paths.
    """

    cfg_path = Path(config_path) if config_path is not None else DEFAULT_PRESCRIPTION_PATH
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

    trace_cfg = experiment["observation_subblock"]["trace"]
    trace_path = _resolve_relative_path(
        trace_cfg["path"],
        config_path=cfg_path,
        field_name="experiment.observation_subblock.trace.path",
    )
    validate_cfg = experiment["observation_subblock"]["validate"]
    trace = load_obs_subblock_trace_csv(
        trace_path,
        require_contiguous_frame_index=validate_cfg["require_contiguous_frame_index"],
        require_monotonic_time=validate_cfg["require_monotonic_time"],
    )
    requested_varying_keys = experiment["observation_subblock"]["requested_varying_keys"]
    applied_varying_keys = experiment["observation_subblock"]["applied_varying_keys"]
    if requested_varying_keys is not None and tuple(requested_varying_keys) != tuple(
        applied_varying_keys
    ):
        print(
            "Note: requested varying_keys differs from applied v1 renderer keys; "
            "rendering still applies source.x_position_as/source.y_position_as/"
            "source.position_angle_deg only."
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

    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec)

    truth_overrides = _flatten_truth_overrides(experiment["truth"])
    unknown_truth_keys = sorted(key for key in truth_overrides if key not in forward_spec)
    if unknown_truth_keys:
        raise ValueError(
            "experiment.truth contains keys not present in forward_spec: "
            + ", ".join(unknown_truth_keys)
        )
    if truth_overrides:
        base_store = base_store.replace(truth_overrides)
    base_store = base_store.refresh_derived(forward_spec)

    binder = SheraBinder(system_cfg, forward_spec, base_store)

    noise_cfg = experiment["noise"]
    rng_key = jr.PRNGKey(int(experiment["seed"]))

    frame_images: list[np.ndarray] = []
    resolved_truth_rows: list[dict[str, Any]] = []
    for trace_row in trace.rows:
        frame_overrides = {key: trace_row[key] for key in APPLIED_V1_VARYING_KEYS}
        frame_store = base_store.replace(frame_overrides).refresh_derived(forward_spec)
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
        resolved_row["source.x_position_as"] = float(
            np.asarray(frame_store.get("source.x_position_as"))
        )
        resolved_row["source.y_position_as"] = float(
            np.asarray(frame_store.get("source.y_position_as"))
        )
        resolved_row["source.position_angle_deg"] = float(
            np.asarray(frame_store.get("source.position_angle_deg"))
        )
        resolved_truth_rows.append(resolved_row)

    cube = np.stack(frame_images, axis=0)
    write_obs_subblock_cube_fits(
        output_path=artifacts["cube_fits"],
        cube=cube,
        header_cards={
            "SCHEMA": MANIFEST_SCHEMA_VERSION,
            "NFRAME": cube.shape[0],
        },
    )

    truth_columns = tuple(
        column
        for column in (*REQUIRED_TRACE_COLUMNS, *trace.extra_columns)
    )
    write_obs_subblock_truth_csv(
        output_path=artifacts["frame_truth_csv"],
        rows=resolved_truth_rows,
        fieldnames=truth_columns,
    )

    system_info = {
        "preset": (user_cfg.get("system") or {}).get("preset"),
        "config_hash": _stable_hash_payload(system_cfg),
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
        system_info=system_info,
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
    )


if __name__ == "__main__":
    main()
