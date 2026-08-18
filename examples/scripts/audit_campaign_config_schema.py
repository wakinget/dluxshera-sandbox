"""Cheap schema audit for full-fidelity iterative campaign configs.

This utility intentionally stays on the lightweight side of the campaign stack:
it parses YAML and optional shard manifests, resolves duplicated cadence fields,
and reports contradictions. It does not translate configs, build campaign plans,
render models, write smear templates, or run subblock preflights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SOURCE_KIND = "full_fidelity_binary_iterative"
TRANSLATED_KIND = "observation_bias_campaign"
DOUBLED_OBSERVATION_BIAS = "observation_bias_campaign/observation_bias_campaign"


def _mapping(value: Any, *, name: str, errors: list[str]) -> dict[str, Any]:
    if value is None:
        errors.append(f"{name} is required.")
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be a mapping.")
        return {}
    return dict(value)


def _optional_mapping(value: Any, *, name: str, errors: list[str]) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be a mapping when provided.")
        return {}
    return dict(value)


def _positive_int(
    value: Any,
    *,
    name: str,
    errors: list[str],
    default: int | None = None,
) -> int | None:
    if value is None:
        if default is not None:
            return default
        errors.append(f"{name} is required.")
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a positive integer.")
        return None
    if out <= 0:
        errors.append(f"{name} must be a positive integer.")
        return None
    return out


def _has_doubled_observation_bias(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return DOUBLED_OBSERVATION_BIAS in normalized


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Campaign YAML must contain a mapping.")
    return dict(payload)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Shard manifest is empty: {path}")
    return rows


def audit_config(config_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    config = _read_yaml(config_path)
    experiment = _mapping(config.get("experiment", config), name="experiment", errors=errors)
    kind = str(experiment.get("kind", ""))
    run_name = str(experiment.get("run_name", ""))
    source_kind_ok = kind == SOURCE_KIND
    if not source_kind_ok:
        if kind == TRANSLATED_KIND:
            warnings.append(
                "Config is already translated observation_bias_campaign; "
                "prepare_full_fidelity_campaign_shards.py expects the source "
                f"{SOURCE_KIND} schema."
            )
        else:
            errors.append(
                "experiment.kind must be "
                f"{SOURCE_KIND!r} for source full-fidelity shard configs."
            )

    iterative = _mapping(
        experiment.get("iterative"),
        name="experiment.iterative",
        errors=errors,
    )
    iterative_forecast = _optional_mapping(
        experiment.get("iterative_forecast"),
        name="experiment.iterative_forecast",
        errors=errors,
    )
    subblocks = _optional_mapping(
        experiment.get("subblocks"),
        name="experiment.subblocks",
        errors=errors,
    )
    forecast = _optional_mapping(
        experiment.get("forecast"),
        name="experiment.forecast",
        errors=errors,
    )

    windows = _positive_int(
        iterative.get("windows_per_draw"),
        name="experiment.iterative.windows_per_draw",
        errors=errors,
    )
    subblocks_per_window = _positive_int(
        iterative.get("subblocks_per_window"),
        name="experiment.iterative.subblocks_per_window",
        errors=errors,
    )
    total_subblocks = (
        None
        if windows is None or subblocks_per_window is None
        else windows * subblocks_per_window
    )

    forecast_actual = _positive_int(
        iterative_forecast.get("actual_windows"),
        name="experiment.iterative_forecast.actual_windows",
        errors=errors,
        default=windows,
    )
    forecast_projected = _positive_int(
        iterative_forecast.get("projected_windows"),
        name="experiment.iterative_forecast.projected_windows",
        errors=errors,
        default=forecast_actual,
    )
    forecast_subblocks_per_window = _positive_int(
        iterative_forecast.get("subblocks_per_window"),
        name="experiment.iterative_forecast.subblocks_per_window",
        errors=errors,
        default=subblocks_per_window,
    )

    if (
        "actual_windows" in iterative_forecast
        and forecast_actual is not None
        and windows is not None
        and forecast_actual != windows
    ):
        errors.append(
            "experiment.iterative_forecast.actual_windows conflicts with "
            "experiment.iterative.windows_per_draw: "
            f"{forecast_actual} != {windows}."
        )
    if (
        "subblocks_per_window" in iterative_forecast
        and forecast_subblocks_per_window is not None
        and subblocks_per_window is not None
        and forecast_subblocks_per_window != subblocks_per_window
    ):
        errors.append(
            "experiment.iterative_forecast.subblocks_per_window conflicts with "
            "experiment.iterative.subblocks_per_window: "
            f"{forecast_subblocks_per_window} != {subblocks_per_window}."
        )
    if (
        forecast_projected is not None
        and forecast_actual is not None
        and forecast_projected < forecast_actual
    ):
        errors.append(
            "experiment.iterative_forecast.projected_windows must be >= "
            "experiment.iterative_forecast.actual_windows."
        )

    raw_n_subblocks = subblocks.get("n_subblocks")
    if raw_n_subblocks is not None and total_subblocks is not None:
        n_subblocks = _positive_int(
            raw_n_subblocks,
            name="experiment.subblocks.n_subblocks",
            errors=errors,
        )
        if n_subblocks is not None and n_subblocks != total_subblocks:
            errors.append(
                "experiment.subblocks.n_subblocks conflicts with actual cadence: "
                f"{n_subblocks} != {windows}*{subblocks_per_window}={total_subblocks}."
            )

    trace_source = subblocks.get("trace_source", {})
    if isinstance(trace_source, Mapping):
        window = trace_source.get("window", {})
        if isinstance(window, Mapping) and window.get("n_subblocks") is not None:
            trace_n = _positive_int(
                window.get("n_subblocks"),
                name="experiment.subblocks.trace_source.window.n_subblocks",
                errors=errors,
            )
            if trace_n is not None and total_subblocks is not None and trace_n != total_subblocks:
                errors.append(
                    "experiment.subblocks.trace_source.window.n_subblocks conflicts "
                    f"with actual cadence: {trace_n} != {total_subblocks}."
                )
        elif window and not isinstance(window, Mapping):
            errors.append("experiment.subblocks.trace_source.window must be a mapping.")
    elif trace_source:
        errors.append("experiment.subblocks.trace_source must be a mapping.")

    if bool(forecast.get("enabled", False)):
        warnings.append(
            "experiment.forecast is the legacy non-iterative forecast path; "
            "iterative full-fidelity templates should usually keep it disabled "
            "and use experiment.iterative_forecast."
        )
    if not iterative_forecast:
        warnings.append(
            "experiment.iterative_forecast is absent; projected observation "
            "forecast artifacts will not use the full-fidelity projection layer."
        )

    cadence_consistent = not any("cadence" in item or "conflicts" in item for item in errors)
    forecast_consistent = not any("iterative_forecast" in item for item in errors)
    return {
        "config": str(config_path),
        "experiment_kind": kind,
        "run_name": run_name,
        "source_config_kind_ok": source_kind_ok,
        "windows_per_draw": windows,
        "subblocks_per_window": subblocks_per_window,
        "total_realized_subblocks": total_subblocks,
        "iterative_forecast": {
            "enabled": bool(iterative_forecast.get("enabled", False)),
            "actual_windows": forecast_actual,
            "projected_windows": forecast_projected,
            "subblocks_per_window": forecast_subblocks_per_window,
        },
        "forecast": {
            "enabled": bool(forecast.get("enabled", False)),
            "path": "legacy_non_iterative",
        },
        "cadence_consistent": cadence_consistent,
        "forecast_consistent": forecast_consistent,
        "results_root_policy": "documented_only_parent_results_root_expected_by_shard_helper",
        "warnings": warnings,
        "errors": errors,
    }


def audit_manifest(manifest_path: Path, config_audit: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    rows = _read_manifest(manifest_path)
    required = {
        "shard_name",
        "shard_mode",
        "source_config_path",
        "config_path",
        "expected_run_root",
        "condition_label",
        "draw_start",
        "draw_stop",
        "draw_index",
        "expected_subblocks",
        "expected_windows",
        "expected_subblocks_per_window",
        "expected_n_theta",
        "recommended_time",
        "recommended_cpus_per_task",
        "recommended_mem",
        "recommended_max_workers",
        "sbatch_command",
    }
    present = set(rows[0].keys())
    missing = sorted(required - present)
    if missing:
        errors.append("Shard manifest is missing required columns: " + ", ".join(missing))

    windows = config_audit.get("windows_per_draw")
    subblocks_per_window = config_audit.get("subblocks_per_window")
    total = config_audit.get("total_realized_subblocks")
    for index, row in enumerate(rows, start=2):
        label = row.get("shard_name") or f"row {index}"
        run_root = row.get("expected_run_root", "")
        if _has_doubled_observation_bias(run_root):
            errors.append(
                f"{label}: expected_run_root contains doubled "
                f"{DOUBLED_OBSERVATION_BIAS!r}."
            )
        command = row.get("sbatch_command", "")
        if _has_doubled_observation_bias(command):
            errors.append(
                f"{label}: sbatch_command contains doubled "
                f"{DOUBLED_OBSERVATION_BIAS!r}."
            )
        if command.startswith("sbatch ") and " -M " not in command and " --cluster=" not in command:
            warnings.append(
                f"{label}: sbatch command uses plain sbatch; add sbatch -M edge "
                "manually if Edge submission is required."
            )
        try:
            draw_start = int(row.get("draw_start", "0") or 0)
            draw_stop = int(row.get("draw_stop", "0") or 0)
        except ValueError:
            errors.append(f"{label}: draw_start and draw_stop must be integers.")
            draw_start = 0
            draw_stop = 0
        selected_draws = draw_stop - draw_start
        if selected_draws <= 0:
            errors.append(f"{label}: draw_stop must be greater than draw_start.")
        if total is not None and windows is not None and subblocks_per_window is not None:
            expected_subblocks = selected_draws * int(total)
            expected_windows = selected_draws * int(windows)
            _check_int_field(
                row,
                "expected_subblocks",
                expected_subblocks,
                label=label,
                errors=errors,
            )
            _check_int_field(
                row,
                "expected_windows",
                expected_windows,
                label=label,
                errors=errors,
            )
            _check_int_field(
                row,
                "expected_subblocks_per_window",
                int(subblocks_per_window),
                label=label,
                errors=errors,
            )
        _check_command_contains(
            command,
            f"--time={row.get('recommended_time', '')}",
            label=label,
            errors=errors,
        )
        _check_command_contains(
            command,
            f"--cpus-per-task={row.get('recommended_cpus_per_task', '')}",
            label=label,
            errors=errors,
        )
        _check_command_contains(
            command,
            f"--mem={row.get('recommended_mem', '')}",
            label=label,
            errors=errors,
        )
        _check_command_contains(
            command,
            f"MAX_WORKERS={row.get('recommended_max_workers', '')}",
            label=label,
            errors=errors,
        )
    return {
        "path": str(manifest_path),
        "row_count": len(rows),
        "warnings": warnings,
        "errors": errors,
    }


def _check_int_field(
    row: Mapping[str, str],
    field: str,
    expected: int,
    *,
    label: str,
    errors: list[str],
) -> None:
    try:
        actual = int(row.get(field, ""))
    except ValueError:
        errors.append(f"{label}: {field} must be an integer.")
        return
    if actual != expected:
        errors.append(f"{label}: {field}={actual} but expected {expected}.")


def _check_command_contains(
    command: str,
    needle: str,
    *,
    label: str,
    errors: list[str],
) -> None:
    if needle.endswith("="):
        return
    if needle not in command:
        errors.append(f"{label}: sbatch_command is missing {needle!r}.")


def build_audit(config_path: Path, shard_manifest: Path | None = None) -> dict[str, Any]:
    audit = audit_config(config_path)
    if shard_manifest is not None:
        manifest_audit = audit_manifest(shard_manifest, audit)
        audit["shard_manifest"] = manifest_audit
        audit["warnings"].extend(manifest_audit["warnings"])
        audit["errors"].extend(manifest_audit["errors"])
    return audit


def _print_report(audit: Mapping[str, Any]) -> None:
    forecast = audit["iterative_forecast"]
    print(f"Config: {audit['config']}")
    print(f"experiment.kind: {audit['experiment_kind']}")
    print(f"run_name: {audit['run_name']}")
    print(
        "actual cadence: "
        f"{audit['windows_per_draw']} windows x "
        f"{audit['subblocks_per_window']} subblocks/window = "
        f"{audit['total_realized_subblocks']} subblocks"
    )
    print(
        "forecast cadence: "
        f"actual_windows={forecast['actual_windows']}, "
        f"projected_windows={forecast['projected_windows']}, "
        f"subblocks_per_window={forecast['subblocks_per_window']}"
    )
    print(f"source_config_kind_ok: {_yes_no(bool(audit['source_config_kind_ok']))}")
    print(f"cadence_consistent: {_yes_no(bool(audit['cadence_consistent']))}")
    print(f"forecast_consistent: {_yes_no(bool(audit['forecast_consistent']))}")
    print(f"results_root_policy: {audit['results_root_policy']}")
    if "shard_manifest" in audit:
        manifest = audit["shard_manifest"]
        print(f"shard_manifest: {manifest['path']}")
        print(f"shard_manifest_rows: {manifest['row_count']}")
    print("warnings:")
    if audit["warnings"]:
        for warning in audit["warnings"]:
            print(f"  - {warning}")
    else:
        print("  - none")
    print("errors:")
    if audit["errors"]:
        for error in audit["errors"]:
            print(f"  - {error}")
    else:
        print("  - none")


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cheaply audit full-fidelity campaign schema fields."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--as-json", action="store_true")
    parser.add_argument("--check-shard-manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        audit = build_audit(args.config, args.check_shard_manifest)
    except Exception as exc:  # pragma: no cover - CLI guardrail
        audit = {
            "config": str(args.config),
            "warnings": [],
            "errors": [str(exc)],
        }
    if args.as_json:
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        _print_report(audit)
    raise SystemExit(1 if audit.get("errors") else 0)


if __name__ == "__main__":
    main()
