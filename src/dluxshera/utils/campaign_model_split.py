"""Shared Data/Inference model split contract for campaign wrappers."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .campaign_high_order_wfe import apply_high_order_wfe_campaign_config
from .campaigns import json_ready
from .detector_layer_overrides import (
    detector_blur_warnings,
    detector_layer_stack,
    patch_smear_layer_for_policy,
    validate_no_accidental_default_smear,
)
from .spectral_response import (
    build_target_aware_spectral_deck,
    build_truth_inference_spectral_deck,
    write_spectral_deck_artifacts,
)
from .spectral_source_config import build_spectral_truth_inference_system_configs

SCHEMA_VERSION = "campaign_model_split.v1"


@dataclass(frozen=True)
class CampaignModelSplit:
    """Immutable truth/render vs inference/reference campaign config split."""

    truth_system_cfg: dict[str, Any]
    inference_system_cfg: dict[str, Any]
    provenance: dict[str, Any]
    artifact_paths: dict[str, str]
    truth_config_hash: str
    inference_config_hash: str
    enabled_components: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": SCHEMA_VERSION,
                "truth_config_hash": self.truth_config_hash,
                "inference_config_hash": self.inference_config_hash,
                "components": self.enabled_components,
                "artifact_paths": self.artifact_paths,
                "provenance": self.provenance,
            }
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(json_ready(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_campaign_model_config(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for a campaign model config."""

    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _cfg(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    return copy.deepcopy(dict(raw or {}))


def _extract_source(system_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    system = system_cfg.get("system") if isinstance(system_cfg.get("system"), Mapping) else system_cfg
    source = system.get("source") if isinstance(system, Mapping) else None
    if not isinstance(source, Mapping):
        raise ValueError("Campaign model split spectral composition requires system.source.")
    return source


def _response_component(raw: Mapping[str, Any] | None, *, default_label: str, default_response: float = 1.0) -> dict[str, Any] | None:
    cfg = _cfg(raw)
    if cfg and not bool(cfg.get("enabled", True)):
        return None
    if not cfg:
        return {"label": default_label, "response": float(default_response)}
    if "response" in cfg:
        return {"label": str(cfg.get("label", default_label)), "response": cfg.get("response")}
    out = {
        "label": str(cfg.get("label", default_label)),
        "path": cfg.get("path"),
        "wavelength_column": cfg.get("wavelength_column"),
        "wavelength_unit": cfg.get("wavelength_unit", "nm"),
        "response_column": cfg.get("response_column"),
        "response_unit": cfg.get("response_unit", "dimensionless"),
        "response_scale": float(cfg.get("response_scale", 1.0)),
    }
    return {k: v for k, v in out.items() if v is not None}


def _component_block(model_cfg: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    role_cfg = model_cfg.get(role)
    return role_cfg if isinstance(role_cfg, Mapping) else {}


def _build_spectral_deck(
    *,
    base_system_cfg: Mapping[str, Any],
    spectral_model_cfg: Mapping[str, Any],
    source_kind: str | None,
    target: str | None,
    seed_context: Mapping[str, Any],
) -> Any:
    truth_cfg = _cfg(spectral_model_cfg.get("truth") if isinstance(spectral_model_cfg.get("truth"), Mapping) else None)
    inference_cfg = _cfg(spectral_model_cfg.get("inference") if isinstance(spectral_model_cfg.get("inference"), Mapping) else None)
    if bool(spectral_model_cfg.get("fast", False)):
        truth_cfg["n_lambda"] = min(int(truth_cfg.get("n_lambda", 7)), 7)
        inference_cfg["n_lambda"] = min(int(inference_cfg.get("n_lambda", 5)), 5)

    truth_components = _cfg(truth_cfg.get("components") if isinstance(truth_cfg.get("components"), Mapping) else None)
    inference_components = _cfg(inference_cfg.get("components") if isinstance(inference_cfg.get("components"), Mapping) else None)
    detector_qe = _response_component(truth_components.get("detector_qe"), default_label="synthetic_flat_qe")
    filter_response = _response_component(
        truth_components.get("m2_filter_response") or truth_components.get("filter_response"),
        default_label="synthetic_flat_filter",
    )
    inference_detector_qe = None
    inf_det = inference_components.get("detector_qe")
    if isinstance(inf_det, Mapping) and str(inf_det.get("mode", "")).lower() != "same_as_truth":
        inference_detector_qe = _response_component(inf_det, default_label="inference_detector_qe")
    inference_filter_response = None
    inf_filter = inference_components.get("m2_filter_response") or inference_components.get("filter_response")
    if isinstance(inf_filter, Mapping) and str(inf_filter.get("mode", "")).lower() != "same_as_truth":
        inference_filter_response = _response_component(inf_filter, default_label="inference_filter_response")

    source_seds = _cfg(spectral_model_cfg.get("source_seds") if isinstance(spectral_model_cfg.get("source_seds"), Mapping) else None)
    sed_mode = str(source_seds.get("mode", spectral_model_cfg.get("sed_mode", "target")))
    explicit_paths = _cfg(source_seds.get("explicit_paths") if isinstance(source_seds.get("explicit_paths"), Mapping) else None)
    source = _extract_source(base_system_cfg)
    kind = str(source_kind or source.get("kind", "binary_target")).lower()
    if kind in {"binary", "binary_target", "alpha_cen"}:
        generic_fallback = str(
            source_seds.get(
                "generic_binary_fallback",
                "alpha_cen" if kind == "binary" and sed_mode in {"target", "real"} else "require_explicit",
            )
        )
        return build_target_aware_spectral_deck(
            source_cfg=source,
            truth_config=truth_cfg,
            inference_config=inference_cfg,
            detector_qe=detector_qe,
            filter_response=filter_response,
            inference_detector_qe=inference_detector_qe,
            inference_filter_response=inference_filter_response,
            sed_mode="target" if sed_mode == "real" else sed_mode,
            target=target or source.get("target"),
            sed_path=source_seds.get("sed_path"),
            sed_a_path=explicit_paths.get("primary") or explicit_paths.get("sed_a_path"),
            sed_b_path=explicit_paths.get("secondary") or explicit_paths.get("sed_b_path"),
            generic_binary_fallback=generic_fallback,
            provenance={"seed_context": json_ready(seed_context), "campaign_model_split": SCHEMA_VERSION},
        )
    return build_truth_inference_spectral_deck(
        truth_config=truth_cfg,
        inference_config=inference_cfg,
        detector_qe=detector_qe,
        filter_response=filter_response,
        inference_detector_qe=inference_detector_qe,
        inference_filter_response=inference_filter_response,
        provenance={"seed_context": json_ready(seed_context), "campaign_model_split": SCHEMA_VERSION},
    )


def _component_summary(*, enabled: bool, truth_label: str | None = None, inference_label: str | None = None, matched: bool = True, artifact_root: Path | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "enabled": bool(enabled),
        "truth_label": truth_label,
        "inference_label": inference_label,
        "matched": bool(matched),
        "artifact_root": None if artifact_root is None else str(artifact_root),
    }
    payload.update(dict(extra or {}))
    return payload


def build_campaign_model_split(
    *,
    base_system_cfg: Mapping[str, Any],
    spectral_model_cfg: Mapping[str, Any] | None = None,
    high_order_wfe_cfg: Mapping[str, Any] | None = None,
    scalar_reference_offsets: Mapping[str, Any] | None = None,
    detector_noise_metadata: Mapping[str, Any] | None = None,
    run_root: Path | None = None,
    artifact_root: Path | None = None,
    seed_context: Mapping[str, Any] | None = None,
    source_kind: str | None = None,
    target: str | None = None,
    write_artifacts: bool = True,
    trajectory_smear_metadata: Mapping[str, Any] | None = None,
) -> CampaignModelSplit:
    """Compose truth/render and inference/reference system configs.

    The invariant is explicit: returned truth config is for trace/render data;
    returned inference config is for recovered-reference/inference products.
    """

    seed_context = dict(seed_context or {})
    artifact_root = Path(artifact_root or (Path(run_root) / "model_split" if run_root is not None else "model_split"))
    truth_system = copy.deepcopy(dict(base_system_cfg))
    inference_system = copy.deepcopy(dict(base_system_cfg))
    artifact_paths: dict[str, str] = {}
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "seed_context": json_ready(seed_context),
        "source_kind": source_kind,
        "target": target,
        "composition_order": [
            "resolve_base_system_preset",
            "global_detector_layer_overrides",
            "spectral_model",
            "high_order_wfe",
            "scalar_reference_offsets",
            "trajectory_smear_policy",
            "detector_noise",
        ],
        "invariant": "trace/render use truth_system_cfg; inference/reference use inference_system_cfg",
        "detector_layers_from_base": detector_layer_stack(base_system_cfg),
    }

    spectral_cfg = _cfg(spectral_model_cfg)
    spectral_enabled = bool(spectral_cfg.get("enabled", False))
    if spectral_enabled:
        deck = _build_spectral_deck(
            base_system_cfg=truth_system,
            spectral_model_cfg=spectral_cfg,
            source_kind=source_kind,
            target=target,
            seed_context=seed_context,
        )
        spectral_root = artifact_root / "spectral"
        if write_artifacts:
            spectral_artifacts = write_spectral_deck_artifacts(deck, spectral_root)
            artifact_paths.update({f"spectral_{k}": str(v) for k, v in spectral_artifacts.items()})
        truth_system, inference_system, spectral_prov = build_spectral_truth_inference_system_configs(
            base_system_cfg=truth_system,
            deck=deck,
            preserve_flux_parameters=bool(spectral_cfg.get("preserve_flux_parameters", True)),
        )
        provenance["spectral_model"] = spectral_prov
        truth_label = spectral_prov.get("truth", {}).get("spectrum", {}).get("spectrum_label")
        inference_label = spectral_prov.get("inference", {}).get("spectrum", {}).get("spectrum_label")
        matched = hash_campaign_model_config(truth_system) == hash_campaign_model_config(inference_system)
        spectral_summary = _component_summary(
            enabled=True,
            truth_label=truth_label,
            inference_label=inference_label,
            matched=matched,
            artifact_root=spectral_root,
        )
    else:
        provenance["spectral_model"] = {"enabled": False, "disabled_reason": "spectral_model.enabled is false or absent"}
        spectral_summary = _component_summary(enabled=False)

    hcfg = _cfg(high_order_wfe_cfg)
    high_order_enabled = bool(hcfg.get("enabled", False))
    high_order_root = artifact_root / "high_order_wfe"
    high_order = apply_high_order_wfe_campaign_config(
        system_cfg=truth_system,
        high_order_wfe_cfg=hcfg,
        seed_context=seed_context,
        artifact_root=high_order_root,
        write_artifacts=write_artifacts,
    )
    truth_system = high_order.truth_system_cfg
    if high_order_enabled:
        # Preserve earlier inference-only patches, e.g. spectral reference
        # weights, while taking the reference high-order WFE block generated
        # from the truth deck.
        reference_optics = high_order.inference_system_cfg.get("optics", {})
        inference_optics = copy.deepcopy(dict(inference_system.get("optics", {}) or {}))
        if isinstance(reference_optics, Mapping) and "high_order_wfe" in reference_optics:
            inference_optics["high_order_wfe"] = copy.deepcopy(reference_optics["high_order_wfe"])
            inference_system["optics"] = inference_optics
    artifact_paths.update({f"high_order_wfe_{k}": str(v) for k, v in high_order.artifact_paths.items()})
    provenance["high_order_wfe"] = high_order.to_dict()
    high_order_summary = _component_summary(
        enabled=high_order_enabled,
        truth_label="high_order_truth" if high_order_enabled else None,
        inference_label="knowledge_error" if high_order_enabled else None,
        matched=not bool(high_order.provenance.get("knowledge_enabled", False)),
        artifact_root=high_order_root if high_order_enabled else None,
        extra={"warnings": list(high_order.warnings)},
    )

    scalar_offsets = _cfg(scalar_reference_offsets)
    scalar_summary = {"enabled": bool(scalar_offsets), "n_offsets": len(scalar_offsets)}
    provenance["scalar_reference_offsets"] = scalar_summary

    smear = _cfg(trajectory_smear_metadata)
    render_cfg = smear.get("render", {}) if isinstance(smear.get("render"), Mapping) else {}
    smear_enabled = bool(smear.get("enabled", False))
    smear_render_mode = str(render_cfg.get("mode", "metadata_only" if smear_enabled else "disabled"))
    if smear_render_mode == "subblock_constant_layer":
        truth_system, truth_smear_prov = patch_smear_layer_for_policy(
            truth_system,
            smear,
            context="campaign_model_split.truth",
            strict=True,
        )
        model_policy = str(render_cfg.get("model_layer_policy", "from_inference_smear"))
        if model_policy in {"same_as_truth", "from_inference_smear"}:
            inference_system, inference_smear_prov = patch_smear_layer_for_policy(
                inference_system,
                smear,
                context="campaign_model_split.inference",
                strict=True,
            )
        else:
            inference_system, inference_smear_prov = patch_smear_layer_for_policy(
                inference_system,
                {"enabled": False, "render": {"mode": "disabled", "target_layer": render_cfg.get("target_layer", "smear")}},
                context="campaign_model_split.inference",
                strict=True,
            )
    else:
        truth_system, truth_smear_prov = patch_smear_layer_for_policy(
            truth_system,
            smear,
            context="campaign_model_split.truth",
            strict=True,
        )
        inference_system, inference_smear_prov = patch_smear_layer_for_policy(
            inference_system,
            smear,
            context="campaign_model_split.inference",
            strict=True,
        )
    smear_warnings = []
    smear_warnings.extend(
        validate_no_accidental_default_smear(
            truth_system,
            system_preset=str(truth_system.get("preset", "")),
            smear_cfg=smear,
            strict=True,
        )
    )
    smear_warnings.extend(detector_blur_warnings(truth_system, smear_cfg=smear))
    smear_summary = {
        "enabled": smear_enabled,
        "mode": smear_render_mode,
        "target_layer": render_cfg.get("target_layer", render_cfg.get("layer_name", "smear")),
        "warnings": smear_warnings,
    }
    provenance["trajectory_smear"] = {
        "config": smear,
        "truth_policy": truth_smear_prov,
        "inference_policy": inference_smear_prov,
        "warnings": smear_warnings,
        "note": (
            "metadata_only writes smear sidecars/provenance only; rendered detector smear is removed."
            if smear_render_mode == "metadata_only"
            else ""
        ),
    }
    provenance["detector_layers_after_smear_policy"] = {
        "truth": detector_layer_stack(truth_system),
        "inference": detector_layer_stack(inference_system),
    }

    detector_noise = _cfg(detector_noise_metadata)
    detector_noise_summary = {
        "enabled": bool(detector_noise.get("enabled", False)),
        "noise_mode": str(detector_noise.get("noise_mode", detector_noise.get("mode", "disabled"))),
    }
    provenance["detector_noise"] = detector_noise

    truth_hash = hash_campaign_model_config(truth_system)
    inference_hash = hash_campaign_model_config(inference_system)
    enabled_components = {
        "spectral_model": spectral_summary,
        "high_order_wfe": high_order_summary,
        "scalar_reference_offsets": scalar_summary,
        "trajectory_smear": smear_summary,
        "detector_noise": detector_noise_summary,
    }
    split = CampaignModelSplit(
        truth_system_cfg=truth_system,
        inference_system_cfg=inference_system,
        provenance=provenance,
        artifact_paths=artifact_paths,
        truth_config_hash=truth_hash,
        inference_config_hash=inference_hash,
        enabled_components=json_ready(enabled_components),
    )
    if write_artifacts:
        summary_path = artifact_root / "model_split_summary.json"
        json_path = artifact_root / "model_split.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summarize_campaign_model_split(split), indent=2), encoding="utf-8")
        json_path.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
        artifact_paths["model_split_summary_json"] = str(summary_path)
        artifact_paths["model_split_json"] = str(json_path)
        split = CampaignModelSplit(
            truth_system_cfg=split.truth_system_cfg,
            inference_system_cfg=split.inference_system_cfg,
            provenance=split.provenance,
            artifact_paths=artifact_paths,
            truth_config_hash=split.truth_config_hash,
            inference_config_hash=split.inference_config_hash,
            enabled_components=split.enabled_components,
        )
    return split


def _inject_model_split(payload: Mapping[str, Any], split: CampaignModelSplit, *, role: str) -> dict[str, Any]:
    out = copy.deepcopy(dict(payload))
    experiment = out.setdefault("experiment", {})
    if not isinstance(experiment, dict):
        raise ValueError("template experiment block must be a mapping")
    experiment["model_split"] = {**split.to_dict(), "template_role": role}
    return out


def write_campaign_model_split_templates(
    *,
    template_root: Path,
    trace_payload: Mapping[str, Any],
    render_payload: Mapping[str, Any],
    inference_payload: Mapping[str, Any],
    split: CampaignModelSplit,
) -> dict[str, Path]:
    """Write trace/render/inference templates with split provenance injected."""

    template_root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "trace": _inject_model_split(trace_payload, split, role="truth_trace"),
        "render": _inject_model_split(render_payload, split, role="truth_render"),
        "inference": _inject_model_split(inference_payload, split, role="inference_reference"),
    }
    payloads["trace"]["system"] = copy.deepcopy(split.truth_system_cfg)
    payloads["render"]["system"] = copy.deepcopy(split.truth_system_cfg)
    payloads["inference"]["system"] = copy.deepcopy(split.inference_system_cfg)
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = template_root / f"{name}_template.json"
        path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
        paths[name] = path
    return paths


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def template_hash_row(template_paths: Mapping[str, Path], split: CampaignModelSplit) -> dict[str, Any]:
    return {
        "trace_template_path": str(Path(template_paths["trace"]).resolve()),
        "render_template_path": str(Path(template_paths["render"]).resolve()),
        "inference_template_path": str(Path(template_paths["inference"]).resolve()),
        "trace_template_hash": file_sha256(Path(template_paths["trace"])),
        "render_template_hash": file_sha256(Path(template_paths["render"])),
        "inference_template_hash": file_sha256(Path(template_paths["inference"])),
        "truth_system_hash": split.truth_config_hash,
        "inference_system_hash": split.inference_config_hash,
        "model_split_json": split.artifact_paths.get("model_split_json", ""),
        "model_split_summary_json": split.artifact_paths.get("model_split_summary_json", ""),
    }


def summarize_campaign_model_split(split: CampaignModelSplit) -> dict[str, Any]:
    return json_ready(
        {
            "schema_version": f"{SCHEMA_VERSION}.summary",
            "truth_config_hash": split.truth_config_hash,
            "inference_config_hash": split.inference_config_hash,
            "matched_truth_inference": split.truth_config_hash == split.inference_config_hash,
            "components": split.enabled_components,
            "artifact_paths": split.artifact_paths,
        }
    )


def validate_campaign_model_split_artifacts(plan_payload: Mapping[str, Any]) -> None:
    """Validate stored model-split artifacts referenced by a campaign plan."""

    missing: list[str] = []
    split = plan_payload.get("model_split")
    if isinstance(split, Mapping):
        for key in ("model_split_json", "model_split_summary_json"):
            value = split.get("artifact_paths", {}).get(key) if isinstance(split.get("artifact_paths"), Mapping) else None
            if value and not Path(str(value)).exists():
                missing.append(str(value))
    rows = plan_payload.get("template_hashes")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for key in ("trace_template_path", "render_template_path", "inference_template_path", "model_split_json"):
                value = row.get(key)
                if value and not Path(str(value)).exists():
                    missing.append(str(value))
    if missing:
        raise FileNotFoundError(
            "Aggregate-only requires stored model-split artifacts from the original run root; missing: "
            + ", ".join(sorted(set(missing)))
        )


__all__ = [
    "SCHEMA_VERSION",
    "CampaignModelSplit",
    "build_campaign_model_split",
    "write_campaign_model_split_templates",
    "summarize_campaign_model_split",
    "hash_campaign_model_config",
    "file_sha256",
    "template_hash_row",
    "validate_campaign_model_split_artifacts",
]
