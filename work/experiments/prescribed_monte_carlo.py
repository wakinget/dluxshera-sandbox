"""Prescribed Monte Carlo experiment scaffold (Step 2: execution loop).

Purpose: incubate the prescription/plan workflow in work/experiments.
Local helpers are defined here for now (TODO: migrate to shared util).

Step 2 execution behavior
-------------------------
- Resolve experiment-wide config/store overrides once (config_id + overrides.config/store).
- For each enabled run: resolve truth/init stores, generate synthetic data (+ optional noise),
  run optimization, and write run-level artifacts under runs/<run_id>/...
- init.mode == "prior": sample around truth using priors, then apply explicit init overrides.
- init.mode == "explicit": apply explicit init overrides only; missing values remain at the
  baseline store and derived values are refreshed after overrides.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dluxshera.inference.optimization import (
    EigenThetaMap,
    fim_theta,
    generate_fim_labels,
    make_binder_nll_fn,
    map_labels_to_keys,
    run_shera_gd,
)
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.run_artifacts import (
    _now_iso_local_ms,
    build_param_summary,
    patch_summary,
)
from dluxshera.params.packing import (
    build_eigen_index_map,
    build_index_map,
    pack_params,
    unpack_params as store_unpack_params,
)
from dluxshera.params.spec import build_inference_spec_basic, make_inference_subspec
from dluxshera.params.store import ParameterStore, strip_structural
from dluxshera.systems.three_plane import (
    SHERA_FLIGHT_CONFIG,
    SHERA_TESTBED_CONFIG,
    SheraThreePlaneConfig,
    SheraThreePlaneBinder,
    build_forward_spec_from_config,
)

# TODO: migrate to shared util
def _timestamp_tag() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# TODO: migrate to shared util
def _load_prescription(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# TODO: migrate to shared util
def _parse_cell(value: str | None) -> Any:
    if value is None:
        return None
    raw = value.strip()
    if raw == "" or raw.lower() in {"null", "none"}:
        return None
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(parsed, list):
            return [float(item) if isinstance(item, (int, float)) else item for item in parsed]
        return parsed
    if raw.lstrip("-").isdigit():
        try:
            return int(raw)
        except ValueError:
            pass
    try:
        return float(raw)
    except ValueError:
        return raw


# TODO: migrate to shared util
def _load_plan_csv(path: Path) -> list[dict[str, Any]]:
    """Load plan rows from CSV in wide (runs-as-rows) or transposed (keys-as-rows) format."""
    with path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#") and line.strip()]

    if not lines:
        return []

    header = next(csv.reader([lines[0]]))
    if header and header[0] == "key":
        reader = csv.reader(lines)
        header = next(reader)
        run_columns = header[1:]
        rows: list[dict[str, Any]] = []
        for run_col in run_columns:
            label = run_col.strip() if run_col is not None else ""
            rows.append({"_plan_label": label or None})
        for row in reader:
            if not row:
                continue
            key = row[0].strip() if len(row) > 0 and row[0] is not None else ""
            if not key:
                continue
            for idx, _ in enumerate(run_columns):
                value = row[idx + 1] if len(row) > idx + 1 else ""
                if value is None or value.strip() == "":
                    continue
                rows[idx][key] = _parse_cell(value)
        return rows

    reader = csv.DictReader(lines)
    rows: list[dict[str, Any]] = []
    for row in reader:
        parsed: dict[str, Any] = {}
        for key, value in row.items():
            if value is None or value.strip() == "":
                continue
            parsed[key] = _parse_cell(value)
        rows.append(parsed)
    return rows


# TODO: migrate to shared util
def _set_nested(target: dict[str, Any], keys: list[str], value: Any) -> None:
    current = target
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


# TODO: migrate to shared util
def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


# TODO: migrate to shared util
def _unflatten_row(row: dict[str, Any]) -> dict[str, Any]:
    structured: dict[str, Any] = {}
    for key, value in row.items():
        if "." in key:
            _set_nested(structured, key.split("."), value)
        else:
            structured[key] = value
    return structured


# TODO: migrate to shared util
def _resolve_run_spec(presc: dict[str, Any], row: dict[str, Any], index: int) -> dict[str, Any]:
    return _resolve_run_spec_with_id(presc, row, index=index, run_id_index=index)


def _resolve_run_spec_with_id(
    presc: dict[str, Any],
    row: dict[str, Any],
    *,
    index: int,
    run_id_index: int | None,
) -> dict[str, Any]:
    defaults = copy.deepcopy(presc.get("defaults", {}))
    resolved = copy.deepcopy(defaults)

    experiment = presc.get("experiment", {})
    run_id_prefix = experiment.get("run_id_prefix", "run")

    row = dict(row)
    row.pop("_plan_label", None)
    structured_row = _unflatten_row(row)
    run_id = structured_row.pop("run_id", None)
    if run_id:
        resolved["run_id"] = run_id
    elif run_id_index is not None:
        resolved["run_id"] = f"{run_id_prefix}_{run_id_index:04d}"

    seed_override = structured_row.pop("seed", None)

    _deep_update(resolved, structured_row)

    resolved["model"] = copy.deepcopy(presc.get("model", {}))

    resolved.setdefault("init", {})
    resolved.setdefault("noise", {})
    resolved.setdefault("eigen", {})
    resolved.setdefault("truth", {})
    resolved.setdefault("optimizer", {})
    resolved.setdefault("model", {})

    base_seed = presc.get("defaults", {}).get("seed")
    if base_seed is None:
        raise ValueError("Prescription defaults must include a base seed.")
    resolved_seed = base_seed if seed_override is None else seed_override
    resolved["seed"] = int(resolved_seed)

    return resolved


# TODO: migrate to shared util
def _get_nested(payload: dict[str, Any], keys: list[str]) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


# TODO: migrate to shared util
def _print_preview(run_specs: list[dict[str, Any]], limit: int | None = None) -> None:
    headers = [
        "run_id",
        "seed",
        "init.mode",
        "eigen.use_eigen",
        "eigen.whiten_basis",
        "eigen.truncate_k",
        "eigen.truncate_by_eigval",
        "truth.x",
        "truth.y",
        "init.x",
        "init.y",
    ]

    def cell(spec: dict[str, Any], key: str) -> str:
        if key == "run_id":
            value = spec.get("run_id")
        elif key == "seed":
            value = spec.get("seed")
        elif key == "init.mode":
            value = _get_nested(spec, ["init", "mode"])
        elif key == "eigen.use_eigen":
            value = _get_nested(spec, ["eigen", "use_eigen"])
        elif key == "eigen.whiten_basis":
            value = _get_nested(spec, ["eigen", "whiten_basis"])
        elif key == "eigen.truncate_k":
            value = _get_nested(spec, ["eigen", "truncate_k"])
        elif key == "eigen.truncate_by_eigval":
            value = _get_nested(spec, ["eigen", "truncate_by_eigval"])
        elif key == "truth.x":
            value = _get_nested(spec, ["truth", "binary", "x_position_as"])
        elif key == "truth.y":
            value = _get_nested(spec, ["truth", "binary", "y_position_as"])
        elif key == "init.x":
            value = _get_nested(spec, ["init", "binary", "x_position_as"])
        elif key == "init.y":
            value = _get_nested(spec, ["init", "binary", "y_position_as"])
        else:
            value = None
        return "" if value is None else str(value)

    preview = run_specs if limit is None else run_specs[:limit]
    rows = [[cell(spec, key) for key in headers] for spec in preview]
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    header_line = " | ".join(header.ljust(width) for header, width in zip(headers, widths))
    divider = "-+-".join("-" * width for width in widths)

    print(header_line)
    print(divider)
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


# TODO: migrate to shared util
def _resolve_outdir(outdir: str | None, run_name: str | None) -> Path:
    if outdir and run_name:
        return Path(outdir) / run_name
    if outdir:
        return Path(outdir) / f"prescribed_mc_{_timestamp_tag()}"
    if run_name:
        return Path("Results") / run_name
    return Path("Results") / f"prescribed_mc_{_timestamp_tag()}"


def _row_enabled(row: dict[str, Any]) -> bool:
    value = row.get("enabled")
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "none", "null"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"true", "1", "yes"}:
            return True
    return bool(value)


def _flatten_store_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, entry in value.items():
                joined = f"{prefix}.{key}" if prefix else key
                _walk(joined, entry)
        else:
            flattened[prefix] = value

    _walk("", payload)
    return flattened


def _resolve_config_id(config_id: str | None) -> SheraThreePlaneConfig:
    if not config_id:
        raise ValueError("Prescription must include model.config_id.")
    mapping = {
        "SHERA_TESTBED_CONFIG": SHERA_TESTBED_CONFIG,
        "SHERA_FLIGHT_CONFIG": SHERA_FLIGHT_CONFIG,
        "shera_testbed": SHERA_TESTBED_CONFIG,
        "shera_flight": SHERA_FLIGHT_CONFIG,
    }
    if config_id in mapping:
        return mapping[config_id]
    raise ValueError(f"Unknown config_id '{config_id}'.")


def _apply_config_overrides(
    cfg: SheraThreePlaneConfig,
    overrides: dict[str, Any],
) -> SheraThreePlaneConfig:
    if not overrides:
        return cfg
    field_names = {field.name for field in dataclasses.fields(cfg)}
    unknown = [key for key in overrides if key not in field_names]
    if unknown:
        raise ValueError(
            "Unknown config override(s): " + ", ".join(sorted(unknown))
        )
    normalized = dict(overrides)
    for key in ("primary_noll_indices", "secondary_noll_indices"):
        if key in normalized and isinstance(normalized[key], list):
            normalized[key] = tuple(normalized[key])
    return cfg.replace(**normalized)


def _repo_relative_path(path: str | Path | None, *, repo_root: Path) -> str | None:
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return Path(os.path.relpath(resolved, repo_root)).as_posix()


def _config_payload(cfg: SheraThreePlaneConfig, *, repo_root: Path) -> dict[str, Any]:
    payload = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else dict(cfg)
    if isinstance(payload, dict) and "diffractive_pupil_path" in payload:
        payload = {
            **payload,
            "diffractive_pupil_path": _repo_relative_path(
                payload.get("diffractive_pupil_path"), repo_root=repo_root
            ),
        }
    return payload


def _maybe_warn_missing_artifacts(run_dir: Path) -> None:
    required = ["meta.json", "summary.json", "trace.npz"]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        print(
            f"WARNING: run artifacts missing in {run_dir}: {', '.join(missing)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prescribed Monte Carlo scaffold")
    parser.add_argument(
        "--prescription",
        type=Path,
        default=Path("work/experiments/prescription_template.json"),
        help="Path to prescription JSON",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("work/experiments/plan_template.csv"),
        help="Path to plan CSV",
    )
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--num-preview", type=int, default=None)

    args = parser.parse_args()

    prescription = _load_prescription(args.prescription)
    plan_rows = _load_plan_csv(args.plan)

    for row in plan_rows:
        forbidden = [
            key
            for key in row
            if not key.startswith("_")
            if key == "model"
            or key.startswith("model.")
            or key == "overrides"
            or key.startswith("overrides.")
        ]
        if forbidden:
            raise ValueError(
                "Plan rows cannot override model settings; remove: "
                + ", ".join(sorted(forbidden))
            )

    model_config_id = _get_nested(prescription, ["model", "config_id"])
    config_overrides = _get_nested(prescription, ["overrides", "config"]) or {}
    store_overrides = _get_nested(prescription, ["overrides", "store"]) or {}
    config_override_keys = (
        ", ".join(sorted(config_overrides.keys())) if config_overrides else "none"
    )
    store_override_keys = (
        ", ".join(sorted(store_overrides.keys())) if store_overrides else "none"
    )

    plan_labels = [row.get("_plan_label") for row in plan_rows]
    run_specs: list[dict[str, Any]] = []
    run_id_index = 0
    for index, row in enumerate(plan_rows):
        enabled = _row_enabled(row)
        if enabled:
            run_id_index += 1
            resolved = _resolve_run_spec_with_id(
                prescription,
                row,
                index=index + 1,
                run_id_index=run_id_index,
            )
        else:
            resolved = _resolve_run_spec_with_id(
                prescription,
                row,
                index=index + 1,
                run_id_index=None,
            )
        run_specs.append(resolved)

    outdir = _resolve_outdir(args.outdir, args.run_name)
    print(f"Resolved outdir: {outdir}")
    print(f"Model config_id: {model_config_id}")
    print(f"Config overrides: {config_override_keys}")
    print(f"Store overrides: {store_override_keys}")
    if any(label is not None for label in plan_labels):
        print("Plan column labels -> run_id mapping:")
        for label, spec in zip(plan_labels, run_specs):
            label_display = label or "(auto)"
            print(f"  {label_display} -> {spec.get('run_id')}")
    print(f"Resolved {len(run_specs)} run(s). Preview:")
    _print_preview(run_specs, args.num_preview)

    if args.dry_run:
        print("Dry run enabled; exiting before optimization.")
        return

    repo_root = Path(__file__).resolve().parents[2]
    jax.config.update("jax_enable_x64", True)

    cfg = _resolve_config_id(model_config_id)
    cfg = _apply_config_overrides(cfg, config_overrides)

    forward_spec = build_forward_spec_from_config(cfg)
    inference_spec = build_inference_spec_basic(cfg)

    infer_keys = tuple(prescription.get("infer_keys", []))
    if not infer_keys:
        raise ValueError("Prescription must include non-empty infer_keys.")
    inference_subspec = make_inference_subspec(
        base_spec=inference_spec,
        infer_keys=infer_keys,
        cfg=cfg,
    )

    base_store = ParameterStore.from_spec_defaults(forward_spec)
    if store_overrides:
        base_store = base_store.replace(_flatten_store_overrides(store_overrides))
    base_store = base_store.refresh_derived(forward_spec)

    prior_info = prescription.get("priors", {})
    prior_spec = PriorSpec.from_info(base_store, prior_info)

    outdir.mkdir(parents=True, exist_ok=True)
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_id_prefix = _get_nested(prescription, ["experiment", "run_id_prefix"]) or "run"
    run_counter = 0

    for index, (row, run_spec) in enumerate(zip(plan_rows, run_specs)):
        if not _row_enabled(row):
            continue

        run_counter += 1
        run_id = run_spec.get("run_id") or f"{run_id_prefix}_{run_counter:04d}"
        run_spec["run_id"] = run_id

        print(f"\n--- Run {run_counter} ({run_id}) ---")

        seed_value = run_spec.get("seed")
        if seed_value is None:
            raise ValueError(f"Run {run_id} resolved to a null seed.")
        seed = int(seed_value)
        rng_key = jr.PRNGKey(seed)
        rng_key, init_key = jr.split(rng_key)
        rng_key, noise_key = jr.split(rng_key)

        truth_overrides = _flatten_store_overrides(run_spec.get("truth", {}))
        truth_store = base_store.replace(truth_overrides)
        truth_store = truth_store.refresh_derived(forward_spec)

        binder = SheraThreePlaneBinder(cfg, forward_spec, truth_store)

        data = binder.model()
        add_noise_value = _get_nested(run_spec, ["noise", "add_noise"])
        if add_noise_value is None:
            add_noise_value = _get_nested(prescription, ["defaults", "noise", "add_noise"])
        add_noise = bool(add_noise_value)
        if add_noise:
            if np.min(data) > 100:
                data = np.sqrt(data) * jr.normal(noise_key, data.shape) + data
            else:
                data = jr.poisson(noise_key, data)
        data_var = data

        nll_loss_fn, _ = make_binder_nll_fn(
            binder=binder,
            infer_keys=infer_keys,
            data=data,
            var=data_var,
            noise_model="gaussian",
            reduce="sum",
            theta0_store=truth_store,
        )
        fim_labels = generate_fim_labels(infer_keys, cfg=cfg, store=truth_store)
        loss_fn = nll_loss_fn

        theta_true = pack_params(inference_subspec, truth_store)
        loss_true = float(loss_fn(theta_true))

        fim_point = theta_true
        F = fim_theta(nll_loss_fn, fim_point)
        fim_diag = jnp.diag(F)

        eigen_cfg = run_spec.get("eigen", {})
        use_eigen_value = eigen_cfg.get("use_eigen")
        if use_eigen_value is None:
            use_eigen_value = _get_nested(prescription, ["defaults", "eigen", "use_eigen"])
        use_eigen = bool(use_eigen_value)
        whiten_basis_value = eigen_cfg.get("whiten_basis")
        if whiten_basis_value is None:
            whiten_basis_value = _get_nested(
                prescription, ["defaults", "eigen", "whiten_basis"]
            )
        whiten_basis = bool(whiten_basis_value)
        truncate_k = eigen_cfg.get("truncate_k")
        truncate_by_eigval = eigen_cfg.get("truncate_by_eigval")

        if use_eigen:
            theta_space = "eigen"
            precond_meta_base = {
                "method": "eigen",
                "whiten_basis": whiten_basis,
                "truncate_k": truncate_k,
                "truncate_by_eigval": truncate_by_eigval,
            }
        else:
            eigen_map = None
            theta_space = "primitive"
            precond_meta_base = {"method": "fim_diag"}

        init_cfg = run_spec.get("init", {})
        init_mode = init_cfg.get("mode") or _get_nested(
            prescription, ["defaults", "init", "mode"]
        )
        init_overrides = {
            key: value for key, value in init_cfg.items() if key != "mode"
        }
        init_overrides_flat = _flatten_store_overrides(init_overrides)

        if init_mode == "prior":
            init_store = prior_spec.sample_near(
                center_store=truth_store,
                rng_key=init_key,
                keys=infer_keys,
            )
            if init_overrides_flat:
                init_store = init_store.replace(init_overrides_flat)
        elif init_mode == "explicit":
            init_store = truth_store
            if init_overrides_flat:
                init_store = init_store.replace(init_overrides_flat)
        else:
            raise ValueError(f"Unknown init.mode '{init_mode}'")
        init_store = init_store.refresh_derived(forward_spec)

        _, theta0 = make_binder_nll_fn(
            binder=binder,
            infer_keys=infer_keys,
            data=data,
            var=data_var,
            noise_model="gaussian",
            reduce="sum",
            theta0_store=init_store,
        )

        if use_eigen:
            if truncate_k is not None and truncate_by_eigval is not None:
                print(
                    "truncate_k is set; ignoring truncate_by_eigval="
                    f"{truncate_by_eigval}."
                )

            theta_ref = theta0
            eigen_map_full = EigenThetaMap.from_fim(
                F,
                theta_ref,
                whiten=whiten_basis,
            )
            eigvals_full = (
                np.asarray(eigen_map_full.eigvals)
                if eigen_map_full.eigvals is not None
                else None
            )

            if truncate_k is not None:
                k = int(truncate_k)
            elif truncate_by_eigval is not None and eigvals_full is not None:
                k = int(np.sum(eigvals_full >= truncate_by_eigval))
            else:
                k = None

            if k is not None:
                if k <= 0:
                    print("truncate_by_eigval removed all modes; keeping top-1.")
                    k = 1
                eigen_map = EigenThetaMap.from_fim(
                    F,
                    theta_ref,
                    truncate=k,
                    whiten=whiten_basis,
                )
            else:
                eigen_map = eigen_map_full

            eigvals_kept = (
                np.asarray(eigen_map.eigvals)
                if eigen_map.eigvals is not None
                else np.array([])
            )

            z0 = eigen_map.z_from_theta(theta0)
            if whiten_basis:
                lr_vec = np.ones_like(z0)
                curvature_vec = np.ones_like(z0)
            else:
                lr_vec = 1.0 / (eigvals_kept + 1e-12)
                curvature_vec = eigvals_kept

            index_map = build_eigen_index_map(eigen_map)
            loss_opt = lambda z: loss_fn(eigen_map.theta_from_z(z))
            theta0_opt = z0
            metric_payload = {
                "theta_ref": np.asarray(theta0_opt),
                "metric_diag": np.asarray(curvature_vec),
                "lr_scale": np.asarray(lr_vec),
            }
            precond_meta = {
                **precond_meta_base,
                "lr_vec": np.asarray(lr_vec),
            }
        else:
            index_map = build_index_map(inference_subspec, init_store, theta=theta0)
            lr_vec = 1.0 / (np.asarray(fim_diag) + 1e-12)
            curvature_vec = fim_diag
            loss_opt = loss_fn
            theta0_opt = theta0
            metric_payload = {
                "theta_ref": np.asarray(theta0_opt),
                "metric_diag": np.asarray(curvature_vec),
                "lr_scale": np.asarray(lr_vec),
            }
            precond_meta = {
                **precond_meta_base,
                "lr_vec": np.asarray(lr_vec),
            }

        labels_by_key = map_labels_to_keys(
            infer_keys,
            fim_labels,
            store=init_store if use_eigen else None,
            index_map=None if use_eigen else index_map,
        )

        optimizer_cfg = run_spec.get("optimizer", {})
        n_iter_value = optimizer_cfg.get("n_iter")
        if n_iter_value is None:
            raise ValueError(f"Run {run_id} resolved to a null optimizer.n_iter.")
        base_lr_value = optimizer_cfg.get("base_lr")
        if base_lr_value is None:
            raise ValueError(f"Run {run_id} resolved to a null optimizer.base_lr.")
        n_iter = int(n_iter_value)
        base_lr = float(base_lr_value)

        config_payload = _config_payload(cfg, repo_root=repo_root)
        theta_final_opt, trace, artifacts = run_shera_gd(
            loss_fn=loss_opt,
            theta0=theta0_opt,
            index_map=index_map,
            learning_rate=base_lr,
            lr_vec=lr_vec,
            num_steps=n_iter,
            runs_dir=runs_dir,
            run_id=run_id,
            return_artifacts=True,
            theta_space=theta_space,
            metric=metric_payload,
            extra_meta={
                "optimizer": {"preconditioning": precond_meta},
                "theta": {"labels_by_key": labels_by_key},
                "model": {
                    "config_id": model_config_id,
                    "config": config_payload,
                },
                "prescribed": {
                    "index": run_counter - 1,
                    "seed": seed,
                    "run_id": run_id,
                    "init_mode": init_mode,
                    "add_noise": add_noise,
                    "use_eigen": use_eigen,
                },
            },
        )

        if use_eigen and eigen_map is not None:
            theta_final = eigen_map.theta_from_z(theta_final_opt)
        else:
            theta_final = theta_final_opt

        final_store = store_unpack_params(inference_subspec, theta_final, init_store)
        _ = binder.model(
            strip_structural(final_store, structural_keys=binder.structural_store_keys())
        )

        loss_init = float(loss_fn(theta0))
        loss_final = float(loss_fn(theta_final))
        improvement_ratio = loss_init / loss_final if loss_final != 0 else float("nan")

        if artifacts is not None:
            run_dir = Path(artifacts["run_dir"]) if artifacts.get("run_dir") else None
            if run_dir is not None:
                truth_dict = {key: truth_store.get(key) for key in infer_keys}
                init_dict = {key: init_store.get(key) for key in infer_keys}
                final_dict = {key: final_store.get(key) for key in infer_keys}
                param_summary = build_param_summary(
                    init_dict, final_dict, truth=truth_dict
                )
                patch_summary(
                    run_dir,
                    {
                        "param_summary": param_summary,
                        "loss_true": loss_true,
                        "improvement_ratio": improvement_ratio,
                        "run_note": run_spec.get("note"),
                        "run_seed": seed,
                        "run_created_at": _now_iso_local_ms(),
                    },
                )
                _maybe_warn_missing_artifacts(run_dir)

        print(
            "Run summary: loss(true)={:.6g}, loss(init)={:.6g}, loss(final)={:.6g}".format(
                loss_true,
                loss_init,
                loss_final,
            )
        )

    print(f"\nExecution complete. Wrote runs to: {runs_dir}")

if __name__ == "__main__":
    main()
