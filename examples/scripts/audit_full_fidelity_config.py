"""Audit executable full-fidelity review/smoke configs without running inference.

The audit translates the narrow full-fidelity schema into the existing
observation-bias campaign schema, classifies active fields, and writes review
artifacts that make wrapper-consumed, forwarded, smoke-only, no-op, and deferred
settings explicit.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from dluxshera.config.io import load_config_file
from dluxshera.utils.full_fidelity_config_schema import (
    CONFIG_FIELD_REGISTRY,
    iter_string_fields,
    registry_entry_for_path,
    registry_rows,
    validate_config_contract,
)
REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = REPO_ROOT / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
DEFAULT_CONFIG = (
    REPO_ROOT
    / "examples"
    / "recipes"
    / "full_fidelity_algorithm_campaign_template"
    / "full_fidelity_binary_iterative_review.yaml"
)
DEFAULT_OUTDIR = REPO_ROOT / "Results" / "full_fidelity_config_audit" / "review_v0"

WRAPPER_CONSUMED_TOP_LEVEL = {
    "kind",
    "schema_version",
    "run_name",
    "seed",
    "source_kind",
    "target",
    "n_cases",
    "system_preset",
}
FORWARDED_TOP_LEVEL = {
    "spectral_model",
    "high_order_wfe",
    "subblocks",
    "iterative",
    "seeding",
    "observation_theta",
    "prior_draws",
    "truth_realization",
    "eigenbasis",
    "forecast",
}
SMOKE_ONLY_TOP_LEVEL = {"n_draws"}
MODEL_SPLIT_PREFIXES = (
    "experiment.spectral_model",
    "experiment.high_order_wfe",
    "experiment.subblocks.trajectory_processing.smear",
    "experiment.subblocks.noise",
)
FUTURE_ONLY_BLOCKS = (
    "detector",
    "observation",
    "trajectory",
    "smear",
    "optics",
    "noise",
    "active_state",
    "iterative_update",
    "knockdowns",
    "outputs",
)
DEFERRED_FIELDS = (
    "experiment.detector.pixel_offsets",
    "experiment.detector.flat_field",
    "experiment.detector.nonlinearity",
    "experiment.detector.dark_current",
    "experiment.observation.full_observation_duration_s",
    "experiment.smear.truth.mode",
    "experiment.active_state.nuisance_not_solved",
    "experiment.iterative_update.update_modes",
    "experiment.knockdowns",
    "experiment.outputs.write_plots",
)

FIELD_REFERENCE: list[dict[str, Any]] = [
    {"field_path": "experiment.kind", "required": True, "consumed_by": "wrapper", "runtime_effect": "selects translator", "fidelity_effect": "none", "provenance_effect": "records source schema", "safe_to_omit": False, "notes": "Must be full_fidelity_binary_iterative_smoke for executable smoke configs."},
    {"field_path": "experiment.schema_version", "required": True, "consumed_by": "wrapper/provenance", "runtime_effect": "none", "fidelity_effect": "none", "provenance_effect": "schema label", "safe_to_omit": False, "notes": "Used by reviewers to distinguish the smoke schema from the future skeleton."},
    {"field_path": "experiment.run_name", "required": False, "consumed_by": "wrapper", "runtime_effect": "output path only", "fidelity_effect": "none", "provenance_effect": "run identity", "safe_to_omit": True, "notes": "CLI --run-name overrides this value."},
    {"field_path": "experiment.seed", "required": False, "consumed_by": "wrapper/observation-bias", "runtime_effect": "deterministic seeding", "fidelity_effect": "changes random realization", "provenance_effect": "base seed", "safe_to_omit": True, "notes": "Default is 42 in the wrapper."},
    {"field_path": "experiment.source_kind", "required": False, "consumed_by": "wrapper", "runtime_effect": "source resolver choice", "fidelity_effect": "selects binary target source semantics", "provenance_effect": "source label", "safe_to_omit": True, "notes": "Defaults to binary_target."},
    {"field_path": "experiment.target", "required": False, "consumed_by": "wrapper/model-split", "runtime_effect": "SED/source lookup", "fidelity_effect": "target-aware spectral deck", "provenance_effect": "science target", "safe_to_omit": True, "notes": "Defaults to ALPHA_CEN."},
    {"field_path": "experiment.n_cases", "required": False, "consumed_by": "wrapper", "runtime_effect": "case count when prior_draws omitted", "fidelity_effect": "none", "provenance_effect": "smoke scale", "safe_to_omit": True, "notes": "Keep 1 for smoke; use 2-4 for a less tiny validation."},
    {"field_path": "experiment.n_draws", "required": False, "consumed_by": "smoke label only", "runtime_effect": "none today", "fidelity_effect": "none", "provenance_effect": "documents intended tiny draw count", "safe_to_omit": True, "notes": "prior_draws.n_cases controls generated prior cases."},
    {"field_path": "experiment.system_preset", "required": False, "consumed_by": "wrapper", "runtime_effect": "system resolver", "fidelity_effect": "selects SHERA preset", "provenance_effect": "system label", "safe_to_omit": True, "notes": "Defaults to SHERA_FLIGHT_3P."},
    {"field_path": "experiment.spectral_model.enabled", "required": False, "consumed_by": "model-split helper", "runtime_effect": "enables spectral deck", "fidelity_effect": "truth/reference spectral mismatch possible", "provenance_effect": "component summary", "safe_to_omit": True, "notes": "If false/omitted, spectral model split is disabled."},
    {"field_path": "experiment.spectral_model.fast", "required": False, "consumed_by": "model-split helper", "runtime_effect": "clamps truth<=7 and inference<=5 wavelengths", "fidelity_effect": "reduces spectral sampling fidelity", "provenance_effect": "smoke shortcut", "safe_to_omit": True, "notes": "Not a substitute for explicit n_lambda/range/response settings."},
    {"field_path": "experiment.spectral_model.preserve_flux_parameters", "required": False, "consumed_by": "model-split helper", "runtime_effect": "none significant", "fidelity_effect": "keeps scalar flux parameters separate from spectral weights", "provenance_effect": "spectral config", "safe_to_omit": True, "notes": "Default is true in model-split composition."},
    {"field_path": "experiment.spectral_model.source_seds.mode", "required": False, "consumed_by": "model-split helper", "runtime_effect": "SED lookup", "fidelity_effect": "target-aware SED selection", "provenance_effect": "spectral source mode", "safe_to_omit": True, "notes": "target is the current smoke path."},
    {"field_path": "experiment.spectral_model.truth.n_lambda", "required": False, "consumed_by": "model-split helper", "runtime_effect": "truth wavelength grid cost", "fidelity_effect": "truth spectral sampling", "provenance_effect": "truth deck", "safe_to_omit": True, "notes": "With fast=true the effective value is clamped to <=7."},
    {"field_path": "experiment.spectral_model.truth.components.detector_qe.enabled", "required": False, "consumed_by": "model-split helper", "runtime_effect": "response-table work if enabled", "fidelity_effect": "detector QE realism", "provenance_effect": "truth component", "safe_to_omit": True, "notes": "Disabled in smoke to keep dependencies/cost small."},
    {"field_path": "experiment.spectral_model.truth.components.m2_filter_response.enabled", "required": False, "consumed_by": "model-split helper", "runtime_effect": "response-table work if enabled", "fidelity_effect": "filter response realism", "provenance_effect": "truth component", "safe_to_omit": True, "notes": "Disabled in smoke to keep dependencies/cost small."},
    {"field_path": "experiment.spectral_model.inference.n_lambda", "required": False, "consumed_by": "model-split helper", "runtime_effect": "inference/reference spectral grid cost", "fidelity_effect": "reference spectral sampling", "provenance_effect": "inference deck", "safe_to_omit": True, "notes": "Use 5-7 for a less tiny validation; fast=true clamps to <=5."},
    {"field_path": "experiment.high_order_wfe.truth.npix", "required": False, "consumed_by": "model-split helper", "runtime_effect": "WFE map generation and optics array size", "fidelity_effect": "high-order spatial sampling", "provenance_effect": "truth WFE deck", "safe_to_omit": True, "notes": "16 is smoke scale; 32 or 64 is a next-step validation value."},
    {"field_path": "experiment.high_order_wfe.truth.amplitude_nm_rms", "required": False, "consumed_by": "model-split helper", "runtime_effect": "none significant", "fidelity_effect": "truth high-order WFE amplitude", "provenance_effect": "truth WFE deck", "safe_to_omit": True, "notes": "Controls physical truth/reference mismatch when knowledge_error is nonzero."},
    {"field_path": "experiment.high_order_wfe.artifacts.write_maps", "required": False, "consumed_by": "model-split helper", "runtime_effect": "artifact I/O", "fidelity_effect": "none", "provenance_effect": "debug artifact availability", "safe_to_omit": True, "notes": "False keeps smoke artifacts small."},
    {"field_path": "experiment.subblocks.n_frames", "required": False, "consumed_by": "observation-bias campaign", "runtime_effect": "linear-ish render/inference cost", "fidelity_effect": "temporal sampling", "provenance_effect": "plan scale", "safe_to_omit": True, "notes": "3 is smoke scale; 5-10 is a less tiny validation."},
    {"field_path": "experiment.subblocks.reference_n_iter", "required": False, "consumed_by": "observation-bias campaign", "runtime_effect": "optimizer iterations", "fidelity_effect": "reference-solve convergence", "provenance_effect": "optimizer config", "safe_to_omit": True, "notes": "3 is deliberately tiny."},
    {"field_path": "experiment.subblocks.trajectory_processing.smear.render.mode", "required": False, "consumed_by": "model-split metadata", "runtime_effect": "metadata_only avoids dynamic smear rendering", "fidelity_effect": "dynamic smear not active", "provenance_effect": "smear sidecar mode", "safe_to_omit": True, "notes": "Only none/metadata_only are wired in the smoke wrapper."},
    {"field_path": "experiment.iterative.update_gain", "required": False, "consumed_by": "observation-bias iterative update", "runtime_effect": "none significant", "fidelity_effect": "update damping/stability", "provenance_effect": "iterative plan", "safe_to_omit": True, "notes": "0.25 is conservative for smoke."},
    {"field_path": "experiment.iterative.update_safety.posterior_sigma_inflation", "required": False, "consumed_by": "observation-bias iterative update", "runtime_effect": "none significant", "fidelity_effect": "conservative posterior uncertainty", "provenance_effect": "update safety", "safe_to_omit": True, "notes": "10.0 is a smoke safety guard."},
    {"field_path": "experiment.observation_theta.optics.primary_zernikes.indices", "required": False, "consumed_by": "observation theta layout", "runtime_effect": "state dimension", "fidelity_effect": "active low-order optical parameters", "provenance_effect": "theta layout", "safe_to_omit": True, "notes": "[0] is tiny; from_system or more indices increases state dimension."},
    {"field_path": "experiment.prior_draws.sigmas.*", "required": False, "consumed_by": "observation-bias campaign", "runtime_effect": "prior draw generation", "fidelity_effect": "initial offset size", "provenance_effect": "prior draw table", "safe_to_omit": True, "notes": "Initialization only; does not add physical realism terms."},
]


def _load_wrapper() -> Any:
    scripts_dir = str(WRAPPER_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("run_full_fidelity_binary_iterative_campaign", WRAPPER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {WRAPPER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(child, child_prefix))
        return rows
    if isinstance(value, list):
        return [(prefix, value)]
    return [(prefix, value)]


def _get_path(root: Mapping[str, Any], path: str) -> Any:
    current: Any = root
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _json_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def _classify_field(path: str) -> dict[str, Any]:
    parts = path.split(".")
    top = parts[1] if len(parts) > 1 and parts[0] == "experiment" else parts[0]
    consumed_by = "unknown/currently unused"
    category = "unknown"
    if top in WRAPPER_CONSUMED_TOP_LEVEL:
        consumed_by = "full-fidelity smoke wrapper"
        category = "wrapper_consumed"
    elif top in FORWARDED_TOP_LEVEL:
        consumed_by = "forwarded unchanged into observation-bias campaign"
        category = "forwarded"
    elif top in SMOKE_ONLY_TOP_LEVEL:
        consumed_by = "smoke-only label/shortcut"
        category = "smoke_only"
    elif top in FUTURE_ONLY_BLOCKS:
        consumed_by = "not consumed by current smoke wrapper"
        category = "future_only"
    if path.startswith(MODEL_SPLIT_PREFIXES):
        consumed_by = "model-split helper via observation-bias campaign"
        category = "model_split_recognized"
    if path == "experiment.spectral_model.fast":
        consumed_by = "model-split helper"
        category = "smoke_cost_reducer"
    return {"field_path": path, "category": category, "consumed_by": consumed_by}


def _field_reference_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = registry_rows()
    for out in rows:
        out["example_value"] = _get_path(config, str(out["field_path"]))
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(value) for key, value in row.items()})


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    columns = [
        "field_path",
        "valid_values",
        "default",
        "implemented_status",
        "consumed_by",
        "runtime_effect",
        "fidelity_effect",
        "provenance_effect",
        "safe_to_omit",
        "notes",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = [_json_value(row.get(key)).replace("|", "\\|") for key in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _effective_spectral_summary(experiment: Mapping[str, Any]) -> dict[str, Any]:
    spectral = experiment.get("spectral_model", {}) if isinstance(experiment.get("spectral_model"), Mapping) else {}
    truth = spectral.get("truth", {}) if isinstance(spectral.get("truth"), Mapping) else {}
    inference = spectral.get("inference", {}) if isinstance(spectral.get("inference"), Mapping) else {}
    configured_truth = int(truth.get("n_lambda", 7)) if "n_lambda" in truth else None
    configured_inference = int(inference.get("n_lambda", 5)) if "n_lambda" in inference else None
    fast = bool(spectral.get("fast", False))
    return {
        "enabled": bool(spectral.get("enabled", False)),
        "fast": fast,
        "configured_truth_n_lambda": configured_truth,
        "configured_inference_n_lambda": configured_inference,
        "effective_truth_n_lambda": min(configured_truth, 7) if fast and configured_truth is not None else configured_truth,
        "effective_inference_n_lambda": min(configured_inference, 5) if fast and configured_inference is not None else configured_inference,
        "fast_semantics": "When true, campaign_model_split clamps truth n_lambda to <=7 and inference n_lambda to <=5.",
    }


def _component_summary(experiment: Mapping[str, Any]) -> dict[str, Any]:
    spectral = _effective_spectral_summary(experiment)
    high_order = experiment.get("high_order_wfe", {}) if isinstance(experiment.get("high_order_wfe"), Mapping) else {}
    ho_truth = high_order.get("truth", {}) if isinstance(high_order.get("truth"), Mapping) else {}
    ho_inf = high_order.get("inference", {}) if isinstance(high_order.get("inference"), Mapping) else {}
    subblocks = experiment.get("subblocks", {}) if isinstance(experiment.get("subblocks"), Mapping) else {}
    smear = {}
    trajectory_processing = subblocks.get("trajectory_processing") if isinstance(subblocks, Mapping) else None
    if isinstance(trajectory_processing, Mapping) and isinstance(trajectory_processing.get("smear"), Mapping):
        smear = dict(trajectory_processing["smear"])
    return {
        "spectral_model": spectral,
        "high_order_wfe": {
            "enabled": bool(high_order.get("enabled", False)),
            "truth_npix": ho_truth.get("npix"),
            "truth_amplitude_nm_rms": ho_truth.get("amplitude_nm_rms"),
            "inference_mode": ho_inf.get("mode"),
            "knowledge_error_amplitude_nm_rms": (ho_inf.get("knowledge_error", {}) or {}).get("amplitude_nm_rms") if isinstance(ho_inf.get("knowledge_error", {}), Mapping) else None,
            "truth_reference_mismatch": bool((ho_inf.get("knowledge_error", {}) or {}).get("enabled", False)) if isinstance(ho_inf.get("knowledge_error", {}), Mapping) else False,
        },
        "subblocks": {
            "n_subblocks": subblocks.get("n_subblocks"),
            "n_frames": subblocks.get("n_frames"),
            "noise": subblocks.get("noise"),
            "reference_n_iter": subblocks.get("reference_n_iter"),
            "smear": smear,
        },
    }


def _config_tier(experiment: Mapping[str, Any], config_path: Path) -> str:
    kind = str(experiment.get("kind", ""))
    if "review" in kind or "review" in config_path.name:
        return "review"
    if "smoke" in kind or "smoke" in config_path.name:
        return "smoke"
    return "translated"


def build_audit(config_path: Path, outdir: Path, *, run_name: str | None = None, strict: bool = False) -> dict[str, Any]:
    wrapper = _load_wrapper()
    raw = load_config_file(config_path)
    experiment = raw.get("experiment", raw)
    if not isinstance(experiment, Mapping):
        raise ValueError("Config must contain a mapping-valued experiment block.")
    experiment = dict(experiment)
    warnings_out = wrapper.validate_full_fidelity_smoke_config(raw, emit_warnings=False)
    kind = str(experiment.get("kind", ""))
    tier = _config_tier(experiment, config_path)
    contract = validate_config_contract({"experiment": experiment}, config_tier=tier, strict=strict)
    translated: dict[str, Any] | None = None
    translation_error: str | None = None
    try:
        translated = wrapper._full_fidelity_to_observation_bias(raw, run_name=run_name)
    except Exception as exc:  # keep audit useful for skeleton/misuse configs
        translation_error = str(exc)

    field_rows = []
    for path, value in _flatten({"experiment": experiment}):
        row = _classify_field(path)
        row["value"] = value
        field_rows.append(row)

    unknown = [row for row in field_rows if row["category"] == "unknown"]
    future_only = [row for row in field_rows if row["category"] == "future_only"]
    truth_reference_mismatch = []
    matched_truth_reference = []
    spectral_summary = _effective_spectral_summary(experiment)
    if spectral_summary["enabled"]:
        if spectral_summary["effective_truth_n_lambda"] != spectral_summary["effective_inference_n_lambda"]:
            truth_reference_mismatch.append("experiment.spectral_model.truth/inference.n_lambda")
        else:
            matched_truth_reference.append("experiment.spectral_model.truth/inference.n_lambda")
    high_order = _component_summary(experiment)["high_order_wfe"]
    if high_order["enabled"] and high_order["truth_reference_mismatch"]:
        truth_reference_mismatch.append("experiment.high_order_wfe.truth vs inference.knowledge_error")
    elif high_order["enabled"]:
        matched_truth_reference.append("experiment.high_order_wfe")

    reference_rows = _field_reference_rows({"experiment": experiment})
    accepted_but_noop_used = []
    for field_path, value in iter_string_fields({"experiment": experiment}):
        _, entry = registry_entry_for_path(field_path)
        if entry and entry.get("implemented_status") == "accepted_but_noop":
            accepted_but_noop_used.append({"field_path": field_path, "value": value})

    audit = {
        "schema_version": "full_fidelity_config_audit.v1",
        "config_path": str(config_path),
        "config_kind": kind,
        "config_tier": tier,
        "strict": bool(strict),
        "executable_today": kind in {"full_fidelity_binary_iterative_smoke", "full_fidelity_binary_iterative_review", "observation_bias_campaign"} and translation_error is None,
        "future_schema_skeleton": kind == "full_fidelity_algorithm_campaign",
        "translation_error": translation_error,
        "warnings": warnings_out,
        "contract_findings": contract["findings"],
        "contract_has_errors": contract["has_errors"],
        "undocumented_string_fields": [
            f["field_path"] for f in contract["findings"] if f["code"] == "undocumented_string_field"
        ],
        "unsupported_enum_values": [
            f for f in contract["findings"] if f["code"] == "unsupported_enum_value"
        ],
        "future_only_or_deferred_used": [
            f for f in contract["findings"] if f["code"] in {"future_value_used", "future_field_used"}
        ],
        "accepted_but_noop_fields_used": accepted_but_noop_used,
        "smoke_only_in_review": [
            f for f in contract["findings"] if f["code"] in {"smoke_only_field_in_review", "smoke_only_value_in_review", "fast_in_review_config"}
        ],
        "consumed_by_wrapper": [row["field_path"] for row in field_rows if row["category"] == "wrapper_consumed"],
        "forwarded_to_observation_bias": [row["field_path"] for row in field_rows if row["category"] == "forwarded"],
        "recognized_by_model_split_helper": [row["field_path"] for row in field_rows if row["category"] == "model_split_recognized"],
        "smoke_only_cost_reducers_or_labels": [row["field_path"] for row in field_rows if row["category"] in {"smoke_only", "smoke_cost_reducer"}],
        "unknown_or_currently_unused": [row["field_path"] for row in unknown],
        "future_only_present": [row["field_path"] for row in future_only],
        "truth_reference_mismatch_fields": truth_reference_mismatch,
        "matched_truth_reference_fields": matched_truth_reference,
        "deferred_fields_absent": [path for path in DEFERRED_FIELDS if _get_path({"experiment": experiment}, path) is None],
        "field_rows": field_rows,
        "field_reference": reference_rows,
        "resolved_component_summary": _component_summary(experiment),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    _write_json(outdir / "config_audit.json", audit)
    _write_json(outdir / "translated_observation_bias_config.json", translated or {"error": translation_error})
    _write_json(outdir / "field_reference.json", reference_rows)
    _write_json(outdir / "resolved_component_summary.json", audit["resolved_component_summary"])
    _write_csv(outdir / "field_reference.csv", reference_rows)

    md = _render_audit_markdown(audit)
    (outdir / "config_audit.md").write_text(md, encoding="utf-8")
    if strict and audit["contract_has_errors"]:
        raise ValueError(
            "Strict full-fidelity config audit failed: "
            + "; ".join(f"{f['field_path']}[{f['code']}]" for f in audit["contract_findings"] if f["severity"] == "error")
        )
    return audit


def _render_audit_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Full-Fidelity Config Audit",
        "",
        f"- Config: `{audit['config_path']}`",
        f"- Kind: `{audit['config_kind']}`",
        f"- Executable today: `{audit['executable_today']}`",
        f"- Future schema skeleton: `{audit['future_schema_skeleton']}`",
    ]
    if audit.get("translation_error"):
        lines.append(f"- Translation error: `{audit['translation_error']}`")
    lines.extend(["", "## Warnings"])
    warnings_out = list(audit.get("warnings", []))
    lines.extend([f"- {item}" for item in warnings_out] or ["- None"])
    lines.extend(["", "## Contract Findings"])
    findings = list(audit.get("contract_findings", []))
    lines.extend([f"- `{f['severity']}` `{f['field_path']}` `{f['code']}`: {f['message']}" for f in findings] or ["- None"])
    for title, key in (
        ("Consumed By Wrapper", "consumed_by_wrapper"),
        ("Forwarded To Observation Bias", "forwarded_to_observation_bias"),
        ("Recognized By Model-Split Helper", "recognized_by_model_split_helper"),
        ("Smoke-Only Cost Reducers Or Labels", "smoke_only_cost_reducers_or_labels"),
        ("Unknown Or Currently Unused", "unknown_or_currently_unused"),
        ("Accepted But No-Op Fields Used", "accepted_but_noop_fields_used"),
        ("Truth/Reference Mismatch", "truth_reference_mismatch_fields"),
        ("Matched Truth/Reference", "matched_truth_reference_fields"),
        ("Deferred Fields Absent", "deferred_fields_absent"),
    ):
        lines.extend(["", f"## {title}"])
        values = list(audit.get(key, []))
        lines.extend([f"- `{value}`" for value in values] or ["- None"])
    lines.extend(["", "## Field Reference", "", _markdown_table(list(audit.get("field_reference", [])))])
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a full-fidelity review/smoke config without rendering images or running inference.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config to audit.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Output directory for audit artifacts.")
    parser.add_argument("--run-name", default="full_fidelity_config_audit", help="Run name used only in the translated config artifact.")
    parser.add_argument("--strict", action="store_true", help="Fail on undocumented strings and unsupported/future enum values.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    audit = build_audit(args.config, args.outdir, run_name=args.run_name, strict=bool(args.strict))
    print(json.dumps({"outdir": str(args.outdir), "config_kind": audit["config_kind"], "warnings": audit["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
