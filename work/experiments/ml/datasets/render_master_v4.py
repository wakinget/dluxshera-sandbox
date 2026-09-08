#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from dluxshera.datasets.master_v4 import (
    DEFAULT_SHARD_SIZE,
    FrozenMasterV4,
    ResolvedRenderState,
    render_output_paths,
)
from dluxshera.datasets.rendering import (
    apply_absolute_vector_to_store,
    render_image,
    write_fits,
)
from dluxshera.datasets.schema import json_ready
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec

SCHEMA_VERSION = "shera_v4_raw_render_sidecar/1"
SUMMARY_SCHEMA_VERSION = "shera_v4_render_task_summary/1"


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    _json_dump(tmp, payload)
    os.replace(tmp, path)


def _atomic_fits(path: Path, *, image: np.ndarray, header_data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        write_fits(output_path=tmp, image=image, header_data=header_data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _repo_sha(repo_root: Path | None = None) -> str | None:
    root = Path.cwd() if repo_root is None else Path(repo_root)
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _dlux_provenance() -> dict[str, Any]:
    try:
        import dLux
    except Exception as exc:  # noqa: BLE001
        return {"import": "dLux", "available": False, "error": str(exc)}
    module_path = Path(getattr(dLux, "__file__", "") or "")
    commit = None
    if module_path:
        for parent in [module_path.parent, *module_path.parents]:
            if (parent / ".git").exists():
                commit = _repo_sha(parent)
                break
    return {
        "import": "dLux",
        "available": True,
        "version": getattr(dLux, "__version__", None),
        "module_file": str(module_path) if module_path else None,
        "source_commit": commit,
    }


def _build_renderer_system(plan: FrozenMasterV4) -> tuple[Any, ParameterStore, SheraBinder]:
    system_cfg = plan.render_system_contract["resolved_system"]
    forward_spec = compose_forward_spec(system_cfg)
    base_store = ParameterStore.from_spec_defaults(forward_spec).refresh_derived(
        forward_spec
    )
    return forward_spec, base_store, SheraBinder(system_cfg, forward_spec, base_store)


def _expected_sidecar(
    *,
    state: ResolvedRenderState,
    plan: FrozenMasterV4,
    output_root: Path,
    fits_path: Path,
    image: np.ndarray | None = None,
    repo_sha: str | None = None,
    dlux_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    science_extras = {
        key: value
        for key, value in state.science_row.items()
        if key
        not in {
            "ordered_physical_science_vector",
            "ordered_physical_delta_vector",
            "ordered_fisher_scaled_delta",
        }
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": plan.manifest["dataset_version"],
        "render_index": state.render_index,
        "science_global_index": state.science_global_index,
        "nuisance_bank_index": state.nuisance_bank_index,
        "dataset_family": state.dataset_family,
        "split_role": state.split_role,
        "science_plan_row_index": state.science_plan_row_index,
        "science_state_id": state.science_state_id,
        "nuisance_state_id": state.nuisance_state_id,
        "render_state_id": state.render_state_id,
        "science_vector_space_id": plan.manifest["science_vector_space_id"],
        "nuisance_vector_space_id": plan.manifest["nuisance_vector_space_id"],
        "master_scientific_content_hash": plan.manifest[
            "master_scientific_content_hash"
        ],
        "nuisance_bank_hash": plan.manifest["selected_nuisance_bank_hash"],
        "render_system_contract_hash": plan.manifest["render_system_contract_hash"],
        "render_contract_hash": plan.manifest["render_contract_hash"],
        "ordered_science_labels": list(state.science_labels),
        "ordered_physical_science_vector": list(state.physical_science_vector),
        "ordered_fisher_science_vector": list(state.fisher_science_vector),
        "ordered_fisher_scaled_science_vector": list(state.fisher_science_vector),
        "ordered_nuisance_labels": list(state.nuisance_labels),
        "ordered_physical_nuisance_vector": list(state.physical_nuisance_vector),
        "image_shape": None if image is None else list(image.shape),
        "dtype": None if image is None else str(image.dtype),
        "relative_fits_path": str(fits_path.relative_to(output_root)),
        "renderer_repository_sha": repo_sha,
        "dlux_version": (dlux_provenance or {}).get("version"),
        "dlux_source_commit": (dlux_provenance or {}).get("source_commit"),
        "dlux_provenance": dict(dlux_provenance or {}),
        "science_row_metadata": science_extras,
        "nuisance_row_metadata": dict(state.nuisance_row),
    }


def _sidecar_identity_matches(path: Path, expected: Mapping[str, Any]) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    keys = (
        "schema_version",
        "dataset_version",
        "render_index",
        "science_global_index",
        "nuisance_bank_index",
        "dataset_family",
        "split_role",
        "science_state_id",
        "nuisance_state_id",
        "render_state_id",
        "science_vector_space_id",
        "nuisance_vector_space_id",
        "master_scientific_content_hash",
        "nuisance_bank_hash",
        "render_system_contract_hash",
        "render_contract_hash",
    )
    return all(payload.get(key) == expected.get(key) for key in keys)


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return payload if isinstance(payload, dict) else None


def _verify_fits(path: Path, *, expected_shape: tuple[int, ...] | None = None) -> tuple[bool, str | None]:
    try:
        with fits.open(path, memmap=False) as handle:
            data = np.asarray(handle[0].data)
    except Exception as exc:  # noqa: BLE001
        return False, f"FITS open failed: {exc}"
    if expected_shape is not None and tuple(data.shape) != tuple(expected_shape):
        return False, f"FITS shape {tuple(data.shape)} != expected {expected_shape}"
    if not np.all(np.isfinite(data)):
        return False, "FITS image contains non-finite values"
    return True, None


def _existing_status(
    *,
    fits_path: Path,
    json_path: Path,
    expected: Mapping[str, Any],
    verify_image: bool,
) -> tuple[str, str | None]:
    fits_exists = fits_path.exists()
    json_exists = json_path.exists()
    if not fits_exists and not json_exists:
        return "missing", None
    if fits_exists != json_exists:
        return "invalid", "partial output: exactly one of FITS/JSON exists"
    if not _sidecar_identity_matches(json_path, expected):
        return "invalid", "sidecar identity does not match frozen expected state"
    if verify_image:
        sidecar = _read_sidecar(json_path) or {}
        expected_shape = sidecar.get("image_shape", expected.get("image_shape"))
        if expected_shape is not None:
            expected_shape = tuple(int(value) for value in expected_shape)
        ok, reason = _verify_fits(fits_path, expected_shape=expected_shape)
        if not ok:
            return "invalid", reason
    return "valid", None


def _apply_state(
    *,
    state: ResolvedRenderState,
    plan: FrozenMasterV4,
    base_store: ParameterStore,
    forward_spec: Any,
) -> ParameterStore:
    return apply_absolute_vector_to_store(
        base_store=base_store,
        labels=state.science_labels,
        physical_values=state.physical_science_vector,
        forward_spec=forward_spec,
        component_indices=plan.component_indices,
        nuisance_labels=state.nuisance_labels,
        nuisance_physical_deltas=state.physical_nuisance_vector,
    )


def _parse_indices(value: str) -> list[int]:
    indices = []
    for raw in value.split(","):
        item = raw.strip()
        if item:
            indices.append(int(item))
    if not indices:
        raise ValueError("--render-indices did not contain any indices.")
    return indices


def _requested_indices(args: argparse.Namespace) -> list[int]:
    modes = [
        args.render_index is not None,
        args.render_indices is not None,
        args.start_index is not None or args.stop_index is not None,
        args.print_smoke_indices,
    ]
    if sum(bool(mode) for mode in modes) != 1:
        raise ValueError(
            "Select exactly one of --render-index, --render-indices, "
            "--start-index/--stop-index, or --print-smoke-indices."
        )
    if args.render_index is not None:
        return [int(args.render_index)]
    if args.render_indices is not None:
        return _parse_indices(args.render_indices)
    if args.print_smoke_indices:
        return []
    if args.start_index is None or args.stop_index is None:
        raise ValueError("--start-index and --stop-index must be provided together.")
    if args.stop_index <= args.start_index:
        raise ValueError("--stop-index must be greater than --start-index.")
    return list(range(int(args.start_index), int(args.stop_index)))


def _recommended_smoke_indices(plan: FrozenMasterV4) -> list[int]:
    indices = [0, plan.render_count - 1]
    nuisance_count = plan.nuisance_count
    for entry in plan.render_contract["families"]:
        first = int(entry["first_render_index"])
        last = int(entry["last_render_index"])
        indices.extend([first, min(first + 1, last), min(first + nuisance_count - 1, last), last])
    for entry in plan.render_contract["families"]:
        if entry["family"] != "joint_full_v4" or entry["split_role"] != "train":
            continue
        path = plan.plan_root / str(entry["plan_path"])
        seen: dict[str, int] = {}
        for idx, row in enumerate(_smoke_rows(path)):
            stratum = str(row.get("scale_stratum"))
            if stratum not in seen:
                seen[stratum] = idx
            if len(seen) >= 4:
                break
        science_start = int(entry["science_start_index"])
        for offset, row_index in enumerate(seen.values()):
            indices.append((science_start + row_index) * nuisance_count + (offset % nuisance_count))
    radial_targets = ("0-100", "500-1000", "1000-1500", "1500-2000")
    radial_seen: dict[str, int] = {}
    for entry in plan.render_contract["families"]:
        if entry["family"] != "radial_capture_v4":
            continue
        path = plan.plan_root / str(entry["plan_path"])
        science_start = int(entry["science_start_index"])
        for row_index, row in enumerate(_smoke_rows(path)):
            radial_bin = str(row.get("radial_bin"))
            if radial_bin in radial_targets and radial_bin not in radial_seen:
                nuisance_index = (len(radial_seen) + 1) % nuisance_count
                radial_seen[radial_bin] = (science_start + row_index) * nuisance_count + nuisance_index
            if len(radial_seen) == len(radial_targets):
                break
    indices.extend(radial_seen[target] for target in radial_targets if target in radial_seen)
    return list(dict.fromkeys(indices))[:20]


def _smoke_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

def run(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    if not args.allow_nonproduction_contract:
        from dluxshera.datasets.master_v4 import PRODUCTION_IDENTITIES

        expected = PRODUCTION_IDENTITIES
    else:
        expected = None
    plan = FrozenMasterV4(
        args.plan_root,
        expected=expected,
        validate_plan_hashes=not args.skip_plan_hash_validation,
    )
    if args.print_smoke_indices:
        indices = _recommended_smoke_indices(plan)
        print(",".join(str(index) for index in indices))
        return {"recommended_smoke_indices": indices, "render_count": plan.render_count}

    indices = _requested_indices(args)
    states = plan.resolve_indices(indices)
    output_root = args.output_root.expanduser().resolve()
    repo_sha = args.renderer_repository_sha or _repo_sha(Path.cwd())
    dlux_provenance = _dlux_provenance()

    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "requested_render_indices": indices,
        "requested_render_range": None
        if args.start_index is None
        else [args.start_index, args.stop_index],
        "attempted": len(states),
        "rendered": 0,
        "skipped_valid": 0,
        "invalid_existing": 0,
        "failed": 0,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "renderer_repository_sha": repo_sha,
        "dlux_provenance": dlux_provenance,
        "dry_run": bool(args.dry_run),
        "verify_only": bool(args.verify_only),
    }
    setup_start = time.perf_counter()
    forward_spec = base_store = binder = None
    if not args.dry_run and not args.verify_only:
        forward_spec, base_store, binder = _build_renderer_system(plan)
    summary["setup_time_s"] = time.perf_counter() - setup_start

    render_times: list[float] = []
    first_render_time = None
    for idx, state in enumerate(states, start=1):
        fits_path, json_path = render_output_paths(
            output_root=output_root,
            dataset_family=state.dataset_family,
            split_role=state.split_role,
            render_index=state.render_index,
            shard_size=args.shard_size,
        )
        expected_sidecar = _expected_sidecar(
            state=state,
            plan=plan,
            output_root=output_root,
            fits_path=fits_path,
            repo_sha=repo_sha,
            dlux_provenance=dlux_provenance,
        )
        status, reason = _existing_status(
            fits_path=fits_path,
            json_path=json_path,
            expected=expected_sidecar,
            verify_image=args.verify_only,
        )
        if status == "valid":
            summary["skipped_valid"] += 1
            continue
        if status == "invalid":
            summary["invalid_existing"] += 1
            if not args.overwrite_invalid:
                summary["failed"] += 1
                raise FileExistsError(
                    f"Invalid existing output for render_index={state.render_index}: {reason}"
                )
            fits_path.unlink(missing_ok=True)
            json_path.unlink(missing_ok=True)
        if args.verify_only:
            summary["failed"] += 1
            raise FileNotFoundError(f"Missing output for render_index={state.render_index}.")
        if args.dry_run:
            print(
                json.dumps(
                    json_ready(
                        {
                            "render_index": state.render_index,
                            "fits_path": str(fits_path),
                            "json_path": str(json_path),
                            "render_state_id": state.render_state_id,
                        }
                    ),
                    sort_keys=True,
                )
            )
            continue

        render_start = time.perf_counter()
        applied_store = _apply_state(
            state=state,
            plan=plan,
            base_store=base_store,
            forward_spec=forward_spec,
        )
        result = render_image(binder=binder, applied_store=applied_store)
        elapsed = time.perf_counter() - render_start
        if first_render_time is None:
            first_render_time = elapsed
        else:
            render_times.append(elapsed)
        sidecar = _expected_sidecar(
            state=state,
            plan=plan,
            output_root=output_root,
            fits_path=fits_path,
            image=result.image,
            repo_sha=repo_sha,
            dlux_provenance=dlux_provenance,
        )
        _atomic_fits(
            fits_path,
            image=result.image,
            header_data={
                "DATASET": (plan.manifest["dataset_version"], "SHERA dataset version"),
                "RENDER": (state.render_index, "V4 render index"),
                "SCIENCE": (state.science_global_index, "V4 science global index"),
                "NUISANCE": (state.nuisance_bank_index, "V4 nuisance bank index"),
                "FAMILY": (state.dataset_family, "V4 family"),
                "SPLIT": (state.split_role, "V4 split"),
                "NOISE": (False, "Observation noise added"),
            },
        )
        _atomic_json(json_path, sidecar)
        summary["rendered"] += 1
        if idx % max(1, int(args.progress_interval)) == 0:
            elapsed_total = time.perf_counter() - t0
            print(
                f"[render_master_v4] progress {idx}/{len(states)} "
                f"rendered={summary['rendered']} skipped={summary['skipped_valid']} "
                f"elapsed_s={elapsed_total:.2f}"
            )

    summary["first_render_time_s"] = first_render_time
    summary["steady_render_count"] = len(render_times)
    summary["steady_render_mean_s"] = None if not render_times else float(np.mean(render_times))
    summary["elapsed_walltime_s"] = time.perf_counter() - t0
    if args.summary_path is not None:
        _json_dump(args.summary_path, summary)
    print(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render frozen SHERA ML Master V4 states.")
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--render-index", type=int, default=None)
    parser.add_argument("--render-indices", default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--stop-index", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--verify-only", action="store_true", default=False)
    parser.add_argument("--overwrite-invalid", action="store_true", default=False)
    parser.add_argument("--allow-nonproduction-contract", action="store_true", default=False)
    parser.add_argument("--skip-plan-hash-validation", action="store_true", default=False)
    parser.add_argument("--print-smoke-indices", action="store_true", default=False)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--renderer-repository-sha", default=None)
    parser.add_argument("--progress-interval", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
