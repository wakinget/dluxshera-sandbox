"""Prescribed Monte Carlo experiment scaffold (Step 1: parse + preview only).

Purpose: incubate the prescription/plan workflow in work/experiments.
Local helpers are defined here for now (TODO: migrate to shared util).
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
from pathlib import Path
from typing import Any


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
    with path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#") and line.strip()]
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
    defaults = copy.deepcopy(presc.get("defaults", {}))
    resolved = copy.deepcopy(defaults)

    experiment = presc.get("experiment", {})
    run_id_prefix = experiment.get("run_id_prefix", "run")

    structured_row = _unflatten_row(row)
    run_id = structured_row.pop("run_id", None)
    if not run_id:
        run_id = f"{run_id_prefix}_{index:04d}"
    resolved["run_id"] = run_id

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
    # TODO (Step 2): split resolved["seed"] via jax.random.split for init/noise streams.
    # TODO (Step 2): enforce strict explicit init (init.mode == \"explicit\" and/or init.strict).
    # TODO (Step 2): enforce model overrides experiment-wide only.

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
        forbidden = [key for key in row if key == "model" or key.startswith("model.")]
        if forbidden:
            raise ValueError(
                "Plan rows cannot override model settings; remove: "
                + ", ".join(sorted(forbidden))
            )

    model_config_id = _get_nested(prescription, ["model", "config_id"])
    model_overrides = _get_nested(prescription, ["model", "overrides"]) or {}
    override_keys = (
        ", ".join(sorted(model_overrides.keys())) if model_overrides else "none"
    )

    run_specs = [
        _resolve_run_spec(prescription, row, index + 1)
        for index, row in enumerate(plan_rows)
    ]

    outdir = _resolve_outdir(args.outdir, args.run_name)
    print(f"Resolved outdir: {outdir}")
    print(f"Model config_id: {model_config_id}")
    print(f"Model overrides: {override_keys}")
    print(f"Resolved {len(run_specs)} run(s). Preview:")
    _print_preview(run_specs, args.num_preview)

    if args.dry_run:
        print("Dry run enabled; exiting before optimization.")
        return

    print("TODO: optimization execution will be implemented in Step 2.")


if __name__ == "__main__":
    main()
