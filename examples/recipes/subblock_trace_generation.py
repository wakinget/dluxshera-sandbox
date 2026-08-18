"""Generate canonical explicit traces for observation sub-block rendering.

This recipe defines the first stage of the observation sub-block workflow. It
produces a per-frame CSV trace plus a manifest that captures the normalized
trace `plan`, anchors, source config path, and optional shared system snapshot
used to anchor any omitted bases.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dluxshera.config.io import load_user_config
from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.obs_subblock_io import (
    now_iso_local_ms,
    timestamp_tag,
    to_jsonable_obs_subblock_payload,
    write_obs_subblock_manifest,
    write_obs_subblock_truth_csv,
)
from dluxshera.utils.obs_subblock_keys import (
    apply_obs_subblock_overrides_preserving_derived,
    canonical_obs_subblock_varying_keys,
    collect_obs_subblock_anchor_values,
    parse_obs_subblock_varying_keys,
    partition_obs_subblock_overrides_by_kind,
    validate_supported_obs_subblock_key_addresses,
)
from dluxshera.utils.obs_subblock_trace_builders import (
    SUPPORTED_TRACE_EFFECT_KINDS,
    build_obs_subblock_trace_plan,
    generate_obs_subblock_trace_rows,
    resolve_obs_subblock_trace_anchors,
)


DEFAULT_PRESCRIPTION_PATH = Path(
    "examples/recipes/observation_subblock_trace_template/subblock_trace_prescription.yaml"
)
DEFAULT_OUTDIR_ROOT = Path("Results/observation_subblock_trace")
MANIFEST_SCHEMA_VERSION = "obs_subblock_trace_manifest.v1"
GENERATOR_ID = "examples/recipes/subblock_trace_generation.py"


def _required_dict(parent: dict[str, Any], key: str, *, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{path}.{key} must be a mapping/dict.")
    return value


def _required_str(parent: dict[str, Any], key: str, *, path: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return value


def _flatten_truth_overrides(payload: dict[str, Any]) -> dict[str, Any]:
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


def _resolve_relative_path(
    value: str,
    *,
    config_path: Path | None,
    field_name: str,
) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute():
        return raw_path
    if config_path is not None:
        return (config_path.parent / raw_path).resolve()
    raise ValueError(
        f"{field_name} is relative but config_path is not available to resolve it."
    )


def _relative_path(path: Path, *, outdir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(outdir.resolve()).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, outdir.resolve())).as_posix()


def _validate_experiment_cfg(experiment_cfg: dict[str, Any]) -> dict[str, Any]:
    kind = _required_str(experiment_cfg, "kind", path="experiment")
    if kind != "subblock_trace_generation":
        raise ValueError(
            "experiment.kind must be 'subblock_trace_generation' for this recipe."
        )

    seed_value = experiment_cfg.get("seed")
    if seed_value is not None and not isinstance(seed_value, int):
        raise ValueError("experiment.seed must be an integer when provided.")

    trace_cfg = _required_dict(experiment_cfg, "trace", path="experiment")
    trace_build_plan = build_obs_subblock_trace_plan(trace_cfg, seed=seed_value)
    truth = experiment_cfg.get("truth", {})
    if not isinstance(truth, dict):
        raise ValueError("experiment.truth must be a mapping/dict.")

    outputs = experiment_cfg.get("outputs", {})
    if not isinstance(outputs, dict):
        raise ValueError("experiment.outputs must be a mapping/dict.")
    file_prefix = outputs.get("file_prefix", "obs_subblock_trace")
    if not isinstance(file_prefix, str) or not file_prefix.strip():
        raise ValueError("experiment.outputs.file_prefix must be a non-empty string.")
    outdir_value = outputs.get("outdir")
    if outdir_value is not None and (
        not isinstance(outdir_value, str) or not outdir_value.strip()
    ):
        raise ValueError("experiment.outputs.outdir must be a non-empty string when set.")
    write_manifest = outputs.get("write_manifest", True)
    if not isinstance(write_manifest, bool):
        raise ValueError("experiment.outputs.write_manifest must be a bool.")

    notes_value = experiment_cfg.get("notes")
    if notes_value is not None and not isinstance(notes_value, str):
        raise ValueError("experiment.notes must be a string when provided.")

    return {
        "kind": kind,
        "trace": trace_build_plan,
        "truth": truth,
        "outputs": {
            "outdir": outdir_value,
            "file_prefix": file_prefix.strip(),
            "write_manifest": write_manifest,
        },
        "notes": notes_value,
    }


def _build_trace_artifact_paths(
    *,
    outdir: Path,
    file_prefix: str,
    timestamp: str,
) -> dict[str, Path]:
    trace_path = outdir / f"{file_prefix}_{timestamp}_frame_truth.csv"
    manifest_path = outdir / "manifest.json"
    return {
        "trace_csv": trace_path,
        "manifest_json": manifest_path,
    }


def _stable_hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def generate_subblock_trace_generation(
    *,
    config_path: Path | None = None,
    results_dir: Path | None = None,
    run_name: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate a canonical explicit trace CSV for one observation sub-block."""

    cfg_path = Path(config_path) if config_path is not None else DEFAULT_PRESCRIPTION_PATH
    user_cfg = load_user_config(
        config_path=cfg_path,
        system_preset=None,
        experiment_preset=None,
    )
    resolved_cfg = resolve_config(user_cfg)
    system_cfg = resolved_cfg.get("system")
    experiment_cfg = resolved_cfg.get("experiment")
    if experiment_cfg is None:
        raise ValueError(
            "Subblock trace-generation recipe requires a top-level 'experiment' block."
        )

    experiment = _validate_experiment_cfg(experiment_cfg)
    trace = experiment["trace"]
    varying_addresses = parse_obs_subblock_varying_keys(list(trace.varying_keys))
    varying_keys = canonical_obs_subblock_varying_keys(varying_addresses)
    if varying_keys != tuple(trace.varying_keys):
        raise ValueError("Internal varying-key canonicalization mismatch in trace plan.")

    nominal_anchors: dict[str, float] | None = None
    system_info: dict[str, Any] | None = None
    if system_cfg is not None:
        forward_spec = compose_forward_spec(system_cfg)
        base_store = ParameterStore.from_spec_defaults(forward_spec)

        truth_overrides = _flatten_truth_overrides(experiment["truth"])
        primitive_overrides, derived_overrides, unknown_truth_keys = (
            partition_obs_subblock_overrides_by_kind(
                truth_overrides,
                forward_spec=forward_spec,
            )
        )
        if unknown_truth_keys:
            raise ValueError(
                "experiment.truth contains keys not present or unsupported in "
                "forward_spec: " + ", ".join(sorted(unknown_truth_keys.keys()))
            )
        base_store = apply_obs_subblock_overrides_preserving_derived(
            base_store,
            forward_spec=forward_spec,
            primitive_overrides=primitive_overrides,
            derived_overrides=derived_overrides,
        )

        validate_supported_obs_subblock_key_addresses(
            varying_addresses,
            forward_spec=forward_spec,
            reference_store=base_store,
        )
        nominal_anchors = collect_obs_subblock_anchor_values(
            base_store,
            addresses=varying_addresses,
        )
        system_info = {
            "preset": system_cfg.get("preset"),
            "config_hash": _stable_hash_payload(system_cfg),
            "resolved_config": to_jsonable_obs_subblock_payload(system_cfg),
        }
    else:
        if experiment["truth"]:
            raise ValueError(
                "experiment.truth requires a top-level system block in "
                "subblock_trace_generation recipe."
            )
        validate_supported_obs_subblock_key_addresses(varying_addresses)

    anchors = resolve_obs_subblock_trace_anchors(
        trace,
        nominal_anchors=nominal_anchors,
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
    artifacts = _build_trace_artifact_paths(
        outdir=outdir,
        file_prefix=experiment["outputs"]["file_prefix"],
        timestamp=stamp,
    )

    if dry_run:
        print("Dry run: validated trace-generation configuration.")
        print(f"  frames: {trace.n_frames}")
        print(f"  dt_s: {trace.dt_s}")
        print(f"  output_dir: {outdir}")
        print(f"  expected_trace_csv: {artifacts['trace_csv']}")
        if experiment["outputs"]["write_manifest"]:
            print(f"  expected_manifest: {artifacts['manifest_json']}")
        return {
            "dry_run": True,
            "frame_count": trace.n_frames,
            "output_dir": str(outdir),
            "artifacts": {name: str(path) for name, path in artifacts.items()},
        }

    outdir.mkdir(parents=True, exist_ok=True)

    rows = generate_obs_subblock_trace_rows(trace, anchors=anchors)
    write_obs_subblock_truth_csv(
        output_path=artifacts["trace_csv"],
        rows=rows,
        fieldnames=("frame_index", "time_s", *trace.varying_keys),
    )

    if experiment["outputs"]["write_manifest"]:
        plan_manifest = {
            key: {
                "base": key_plan.base,
                "effects": [dict(effect) for effect in key_plan.effects],
            }
            for key, key_plan in trace.key_plans.items()
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "created_at": now_iso_local_ms(),
            "generator": GENERATOR_ID,
            "inputs": {
                "config_path": str(cfg_path.resolve()),
            },
            "frame_count": trace.n_frames,
            "dt_s": trace.dt_s,
            "time_start_s": float(rows[0]["time_s"]),
            "time_stop_s": float(rows[-1]["time_s"]),
            "supported_effect_kinds": list(SUPPORTED_TRACE_EFFECT_KINDS),
            "varying_keys": list(trace.varying_keys),
            "applied_varying_keys": list(trace.varying_keys),
            "anchors": {key: float(value) for key, value in anchors.items()},
            "trace_spec": {
                "n_frames": trace.n_frames,
                "dt_s": trace.dt_s,
                "seed": trace.seed,
                "varying_keys": list(trace.varying_keys),
                "plan": plan_manifest,
            },
            "trace": {
                "format": "csv",
                "path": _relative_path(artifacts["trace_csv"], outdir=outdir),
                "required_columns": ["frame_index", "time_s", *trace.varying_keys],
            },
            "artifacts": {
                name: _relative_path(path, outdir=outdir)
                for name, path in artifacts.items()
            },
        }
        if system_info is not None:
            manifest["system"] = system_info
        manifest["shared_truth"] = to_jsonable_obs_subblock_payload(experiment["truth"])
        if experiment["notes"] is not None:
            manifest["notes"] = experiment["notes"]
        write_obs_subblock_manifest(
            output_path=artifacts["manifest_json"],
            manifest=manifest,
        )

    print(f"Generated observation sub-block trace with {trace.n_frames} frames.")
    print(f"Wrote artifacts under: {outdir}")

    return {
        "dry_run": False,
        "frame_count": trace.n_frames,
        "output_dir": str(outdir),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an explicit trace CSV for observation sub-block rendering."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PRESCRIPTION_PATH,
        help="Path to subblock trace-generation prescription YAML/JSON.",
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
        help="Validate config and report expected outputs without writing artifacts.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    generate_subblock_trace_generation(
        config_path=args.config,
        results_dir=args.results_dir,
        run_name=args.run_name,
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
