from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .campaigns import json_ready
from .high_order_wfe import (
    HighOrderWfeDeck,
    MirrorWfeDeck,
    build_high_order_wfe_deck,
    write_high_order_wfe_deck_artifacts,
)

SCHEMA_VERSION = "campaign_high_order_wfe.v1"
DEFAULT_MIRRORS = ("primary", "secondary")


@dataclass(frozen=True)
class HighOrderWfeCampaignResult:
    """Hold campaign-ready high-order WFE configs and provenance."""

    truth_system_cfg: dict[str, Any]
    inference_system_cfg: dict[str, Any]
    provenance: dict[str, Any]
    artifact_paths: dict[str, str]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return json_ready(
            {
                "schema_version": SCHEMA_VERSION,
                "provenance": self.provenance,
                "artifact_paths": self.artifact_paths,
                "warnings": list(self.warnings),
            }
        )


def _cfg(mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(mapping or {})


def _stable_seed(seed_context: Mapping[str, Any], explicit_seed: Any = None) -> int:
    if explicit_seed is not None:
        return int(explicit_seed)
    payload = json.dumps(json_ready(seed_context), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _array_ref(array: np.ndarray, *, path: Path | None, write_array: bool = True) -> dict[str, Any]:
    if path is None:
        return {"array_nm": np.asarray(array, dtype=float).tolist()}
    if write_array:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(array, dtype=float))
    return {"array_path": str(path)}


def _map_cfg_from_mirror(
    mirror: MirrorWfeDeck,
    *,
    truth: bool,
    knowledge_enabled: bool,
    config_map_root: Path | None = None,
    write_config_maps: bool = True,
) -> dict[str, Any]:
    # The optics builder currently consumes this legacy realization shape. Use
    # deterministic precomputed arrays to avoid regenerating maps in child runs.
    base = {
        "enabled": True,
        "map": {
            "kind": "precomputed_array_nm",
            "rms_nm": float(mirror.high_order_truth.rms_nm),
            "seed": mirror.provenance.get("truth_seed"),
            **_array_ref(
                mirror.high_order_truth.opd_nm,
                path=None
                if config_map_root is None
                else config_map_root / f"{mirror.mirror}_high_order_truth_opd_nm.npy",
                write_array=write_config_maps,
            ),
        },
        "knowledge_error": {"enabled": False},
    }
    if not truth and knowledge_enabled:
        base["knowledge_error"] = {
            "enabled": True,
            "kind": "precomputed_array_nm",
            "rms_nm": float(mirror.high_order_knowledge_error.rms_nm),
            "seed": mirror.provenance.get("high_order_error_seed"),
            **_array_ref(
                mirror.high_order_knowledge_error.opd_nm,
                path=None
                if config_map_root is None
                else config_map_root / f"{mirror.mirror}_high_order_error_opd_nm.npy",
                write_array=write_config_maps,
            ),
        }
    return base


def _deck_cfg_to_optics_block(
    deck: HighOrderWfeDeck,
    *,
    mirrors: tuple[str, ...],
    truth: bool,
    knowledge_enabled: bool,
    config_map_root: Path | None = None,
    write_config_maps: bool = True,
) -> dict[str, Any]:
    block: dict[str, Any] = {"enabled": bool(mirrors), "schema_version": SCHEMA_VERSION}
    if "primary" in mirrors:
        block["primary"] = _map_cfg_from_mirror(
            deck.primary,
            truth=truth,
            knowledge_enabled=knowledge_enabled,
            config_map_root=config_map_root,
            write_config_maps=write_config_maps,
        )
    if "secondary" in mirrors:
        block["secondary"] = _map_cfg_from_mirror(
            deck.secondary,
            truth=truth,
            knowledge_enabled=knowledge_enabled,
            config_map_root=config_map_root,
            write_config_maps=write_config_maps,
        )
    return block


def _mirror_summary(mirror: MirrorWfeDeck, *, active: bool) -> dict[str, Any]:
    return {
        "active": bool(active),
        "truth_seed": mirror.provenance.get("truth_seed"),
        "knowledge_seed": mirror.provenance.get("high_order_error_seed"),
        "truth_full_rms_nm": mirror.full_truth.rms_nm,
        "truth_high_order_rms_nm": mirror.high_order_truth.rms_nm,
        "knowledge_error_rms_nm": mirror.high_order_knowledge_error.rms_nm,
        "truth_inference_difference_rms_nm": mirror.high_order_knowledge_error.rms_nm,
        "low_order_removed": list(mirror.diagnostics.get("low_order_mapping", {})),
    }


def _write_summary_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    return str(path)


def apply_high_order_wfe_campaign_config(
    *,
    system_cfg: Mapping[str, Any],
    high_order_wfe_cfg: Mapping[str, Any] | None,
    seed_context: Mapping[str, Any],
    artifact_root: Path | None = None,
    write_artifacts: bool = True,
) -> HighOrderWfeCampaignResult:
    """Apply campaign-level high-order WFE config to system configs.

    The returned truth/reference configs differ only by the high-order WFE block.
    High-order map pixels are inserted as static OPD maps and are not added to
    the observation theta layout.
    """

    cfg = _cfg(high_order_wfe_cfg)
    truth_system = copy.deepcopy(dict(system_cfg))
    inference_system = copy.deepcopy(dict(system_cfg))
    if not bool(cfg.get("enabled", False)):
        provenance = {
            "schema_version": SCHEMA_VERSION,
            "enabled": False,
            "disabled_reason": "experiment.high_order_wfe.enabled is false or absent",
        }
        return HighOrderWfeCampaignResult(
            truth_system_cfg=truth_system,
            inference_system_cfg=inference_system,
            provenance=provenance,
            artifact_paths={},
        )

    truth_cfg = _cfg(cfg.get("truth"))
    inference_cfg = _cfg(cfg.get("inference"))
    artifact_cfg = _cfg(cfg.get("artifacts"))
    validation_cfg = _cfg(cfg.get("validation"))

    mirrors = tuple(str(m).strip().lower() for m in truth_cfg.get("mirrors", DEFAULT_MIRRORS))
    bad = sorted(set(mirrors) - set(DEFAULT_MIRRORS))
    if bad:
        raise ValueError(f"Unsupported high_order_wfe.truth.mirrors entries: {bad}")
    pairing = str(truth_cfg.get("pairing", "independent"))
    if pairing != "independent":
        raise ValueError("Only high_order_wfe.truth.pairing='independent' is supported in v1 campaign wiring.")

    base_seed = _stable_seed(seed_context, cfg.get("seed"))
    truth_seed = _stable_seed({**dict(seed_context), "role": "truth"}, truth_cfg.get("seed"))
    if truth_cfg.get("seed") is None:
        truth_seed = base_seed
    knowledge_cfg = _cfg(inference_cfg.get("knowledge_error"))
    knowledge_enabled = bool(inference_cfg.get("enabled", True)) and bool(knowledge_cfg.get("enabled", True))

    shape_npix = int(truth_cfg.get("npix", cfg.get("npix", 64)))
    mask_policy = str(truth_cfg.get("mask_mode", truth_cfg.get("mask_policy", "circular_fallback")))
    amp = float(truth_cfg.get("amplitude_nm_rms", truth_cfg.get("rms_opd_nm", 1.0)))
    error_amp = float(knowledge_cfg.get("amplitude_nm_rms", knowledge_cfg.get("rms_nm", 0.3))) if knowledge_enabled else 0.0
    alpha = float(truth_cfg.get("power_law_alpha", truth_cfg.get("alpha", 2.5)))
    error_alpha = float(knowledge_cfg.get("power_law_alpha", knowledge_cfg.get("alpha", alpha)))

    mirror_wfe_cfg = {
        "truth": {
            "rms_opd_nm": amp,
            "power_law_alpha": alpha,
            "fit_low_order_zernikes": [4, 5, 6, 7, 8, 9, 10, 11]
            if truth_cfg.get("remove_low_order_zernikes", True)
            else [],
        },
        "knowledge": {
            "low_order_sigma_nm_per_coeff": 0.0,
            "high_order_error_rms_nm": error_amp,
            "high_order_error_power_law_alpha": error_alpha,
        },
    }
    deck = build_high_order_wfe_deck(
        shape=(shape_npix, shape_npix),
        seed=truth_seed,
        primary_config=mirror_wfe_cfg,
        secondary_config=mirror_wfe_cfg,
        mask_policy=mask_policy,
    )

    artifact_paths: dict[str, str] = {}
    config_map_root = Path(artifact_root) / "config_maps" if artifact_root is not None else None

    truth_optics = dict(truth_system.get("optics", {}) or {})
    inference_optics = dict(inference_system.get("optics", {}) or {})
    truth_optics["high_order_wfe"] = _deck_cfg_to_optics_block(
        deck,
        mirrors=mirrors,
        truth=True,
        knowledge_enabled=False,
        config_map_root=config_map_root,
        write_config_maps=write_artifacts,
    )
    inference_optics["high_order_wfe"] = _deck_cfg_to_optics_block(
        deck,
        mirrors=mirrors,
        truth=False,
        knowledge_enabled=knowledge_enabled,
        config_map_root=config_map_root,
        write_config_maps=write_artifacts,
    )
    truth_system["optics"] = truth_optics
    inference_system["optics"] = inference_optics

    if config_map_root is not None:
        for path in sorted(config_map_root.glob("*.npy")):
            artifact_paths[f"config_map_{path.stem}"] = str(path)
    summary_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "mirrors": list(mirrors),
        "pairing": pairing,
        "seed_context": json_ready(seed_context),
        "base_seed": int(base_seed),
        "truth_seed": int(truth_seed),
        "knowledge_enabled": knowledge_enabled,
        "truth_amplitude_nm_rms": amp,
        "knowledge_error_amplitude_nm_rms": error_amp,
        "mask_policy": mask_policy,
        "npix": shape_npix,
        "remove_low_order_zernikes": bool(truth_cfg.get("remove_low_order_zernikes", True)),
        "primary": _mirror_summary(deck.primary, active="primary" in mirrors),
        "secondary": _mirror_summary(deck.secondary, active="secondary" in mirrors),
        "deck_comparison": deck.comparison,
        "trajectory_follow_on": {
            "high_pass_filter_enabled": False,
            "intra_frame_smear_enabled": False,
            "note": "Trajectory high-pass filtering and smear are reserved for a later task.",
        },
    }
    if artifact_root is not None:
        root = Path(artifact_root)
        if write_artifacts and bool(artifact_cfg.get("write_maps", True)):
            artifact_paths.update(write_high_order_wfe_deck_artifacts(deck, root / "maps"))
        if write_artifacts and bool(artifact_cfg.get("write_summary_json", True)):
            artifact_paths["high_order_wfe_summary_json"] = _write_summary_json(
                root / "high_order_wfe_summary.json",
                summary_payload,
            )
    summary_payload["artifact_paths"] = dict(artifact_paths)

    warnings: list[str] = []
    if bool(validation_cfg.get("require_nonzero_difference_when_enabled", False)) and knowledge_enabled and error_amp <= 0.0:
        raise ValueError("High-order WFE validation requires nonzero truth/inference difference, but knowledge error amplitude is zero.")
    if not knowledge_enabled:
        warnings.append("High-order WFE inference knowledge error is disabled; truth/reference maps match.")

    return HighOrderWfeCampaignResult(
        truth_system_cfg=truth_system,
        inference_system_cfg=inference_system,
        provenance=summary_payload,
        artifact_paths=artifact_paths,
        warnings=tuple(warnings),
    )


__all__ = [
    "SCHEMA_VERSION",
    "HighOrderWfeCampaignResult",
    "apply_high_order_wfe_campaign_config",
]
