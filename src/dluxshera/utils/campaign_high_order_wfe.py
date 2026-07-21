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
    fit_zernike_coefficients_nm,
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


def _array_hash(array: np.ndarray, *, mask: np.ndarray | None = None) -> str:
    arr = np.asarray(array, dtype=np.float64)
    valid = np.ones(arr.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != arr.shape:
        raise ValueError(f"Map/hash mask shape mismatch: {valid.shape} vs {arr.shape}.")
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("utf-8"))
    digest.update(np.ascontiguousarray(arr[valid]).tobytes())
    return digest.hexdigest()


def _normalised_map_hash(array: np.ndarray, *, mask: np.ndarray) -> str | None:
    arr = np.asarray(array, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    rms = float(np.sqrt(np.mean(np.square(arr[valid])))) if np.any(valid) else 0.0
    if rms == 0.0:
        return None
    return _array_hash(np.round(arr / rms, decimals=10), mask=valid)


def _mirror_seed(seed: int, mirror: str) -> int:
    return _stable_seed({"configured_seed": int(seed), "mirror": mirror, "role": "high_order_error"})


def _validate_nonnegative(value: float, *, name: str) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0.")


def _mirror_knowledge_configs(
    *,
    mirrors: tuple[str, ...],
    knowledge_cfg: Mapping[str, Any],
    inference_enabled: bool,
    truth_alpha: float,
    low_order_noll: list[int],
    truth_remove_low_order: bool,
    seed_context: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    raw_mirrors = knowledge_cfg.get("mirrors")
    mirror_overrides = _cfg(raw_mirrors if isinstance(raw_mirrors, Mapping) else None)
    if raw_mirrors is not None and not isinstance(raw_mirrors, Mapping):
        raise ValueError("high_order_wfe.inference.knowledge_error.mirrors must be a mapping.")
    bad = sorted(set(str(key).strip().lower() for key in mirror_overrides) - set(DEFAULT_MIRRORS))
    if bad:
        raise ValueError(
            "Unsupported high_order_wfe.inference.knowledge_error.mirrors entries: "
            + ", ".join(bad)
        )
    shared_enabled = bool(inference_enabled) and bool(knowledge_cfg.get("enabled", True))
    shared_amp = float(knowledge_cfg.get("amplitude_nm_rms", knowledge_cfg.get("rms_nm", 0.3)))
    _validate_nonnegative(
        shared_amp,
        name="high_order_wfe.inference.knowledge_error.amplitude_nm_rms",
    )
    raw_seed = knowledge_cfg.get("seed")
    fixed_seed = (
        _stable_seed(
            {**dict(seed_context), "role": "high_order_knowledge_error"},
            raw_seed,
        )
        if raw_seed is not None
        else None
    )
    raw_alpha = knowledge_cfg.get("power_law_alpha", knowledge_cfg.get("alpha", truth_alpha))
    error_alpha = truth_alpha if str(raw_alpha).strip().lower() == "same_as_truth" else float(raw_alpha)
    remove_error_low_order = bool(
        knowledge_cfg.get("remove_low_order_zernikes", truth_remove_low_order)
    )
    out: dict[str, dict[str, Any]] = {}
    for mirror in DEFAULT_MIRRORS:
        override = _cfg(mirror_overrides.get(mirror))
        enabled = bool(inference_enabled) and bool(override.get("enabled", shared_enabled))
        amp = float(
            override.get(
                "amplitude_nm_rms",
                override.get("rms_nm", shared_amp),
            )
        )
        _validate_nonnegative(
            amp,
            name=f"high_order_wfe.inference.knowledge_error.mirrors.{mirror}.amplitude_nm_rms",
        )
        mirror_seed = override.get("seed")
        if mirror_seed is None and fixed_seed is not None:
            mirror_seed = _mirror_seed(fixed_seed, mirror)
        if mirror not in mirrors:
            enabled = False
            amp = 0.0
        out[mirror] = {
            "enabled": bool(enabled),
            "requested_amplitude_nm_rms": amp if enabled else 0.0,
            "seed": None if mirror_seed is None else int(mirror_seed),
            "power_law_alpha": error_alpha,
            "remove_low_order_zernikes": remove_error_low_order,
            "remove_noll_indices": low_order_noll if remove_error_low_order else [],
        }
    return out


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
    knowledge_enabled_by_mirror: Mapping[str, bool],
    config_map_root: Path | None = None,
    write_config_maps: bool = True,
) -> dict[str, Any]:
    block: dict[str, Any] = {"enabled": bool(mirrors), "schema_version": SCHEMA_VERSION}
    if "primary" in mirrors:
        block["primary"] = _map_cfg_from_mirror(
            deck.primary,
            truth=truth,
            knowledge_enabled=bool(knowledge_enabled_by_mirror.get("primary", False)),
            config_map_root=config_map_root,
            write_config_maps=write_config_maps,
        )
    if "secondary" in mirrors:
        block["secondary"] = _map_cfg_from_mirror(
            deck.secondary,
            truth=truth,
            knowledge_enabled=bool(knowledge_enabled_by_mirror.get("secondary", False)),
            config_map_root=config_map_root,
            write_config_maps=write_config_maps,
        )
    return block


def _mirror_summary(
    mirror: MirrorWfeDeck,
    *,
    active: bool,
    knowledge_enabled: bool,
    requested_error_rms_nm: float,
) -> dict[str, Any]:
    return {
        "active": bool(active),
        "truth_matched": not bool(knowledge_enabled),
        "knowledge_error_enabled": bool(knowledge_enabled),
        "requested_knowledge_error_rms_nm": float(requested_error_rms_nm),
        "truth_seed": mirror.provenance.get("truth_seed"),
        "knowledge_seed": mirror.provenance.get("high_order_error_seed"),
        "truth_full_rms_nm": mirror.full_truth.rms_nm,
        "truth_high_order_rms_nm": mirror.high_order_truth.rms_nm,
        "knowledge_error_rms_nm": mirror.high_order_knowledge_error.rms_nm,
        "measured_knowledge_error_rms_nm": mirror.high_order_knowledge_error.rms_nm,
        "truth_inference_difference_rms_nm": (
            mirror.high_order_knowledge_error.rms_nm if knowledge_enabled else 0.0
        ),
        "truth_map_hash": _array_hash(mirror.high_order_truth.opd_nm, mask=mirror.high_order_truth.mask),
        "knowledge_error_map_hash": _array_hash(
            mirror.high_order_knowledge_error.opd_nm,
            mask=mirror.high_order_knowledge_error.mask,
        ),
        "normalised_knowledge_error_map_hash": _normalised_map_hash(
            mirror.high_order_knowledge_error.opd_nm,
            mask=mirror.high_order_knowledge_error.mask,
        ),
        "low_order_removed": list(mirror.diagnostics.get("low_order_mapping", {})),
        "low_order_truth_coefficients_nm": dict(mirror.low_order_truth_coeffs_nm),
        "low_order_inference_coefficients_nm": dict(mirror.low_order_knowledge_coeffs_nm),
        "low_order_error_coefficients_nm": dict(mirror.low_order_knowledge_error_nm),
        "low_order_mapping": copy.deepcopy(dict(mirror.diagnostics.get("low_order_mapping", {}))),
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
    knowledge_pairing = str(knowledge_cfg.get("pairing", "independent"))
    if knowledge_pairing != "independent":
        raise ValueError(
            "Only high_order_wfe.inference.knowledge_error.pairing='independent' "
            "is supported in v1 campaign wiring."
        )

    shape_npix = int(truth_cfg.get("npix", cfg.get("npix", 64)))
    mask_policy = str(truth_cfg.get("mask_mode", truth_cfg.get("mask_policy", "circular_fallback")))
    amp = float(truth_cfg.get("amplitude_nm_rms", truth_cfg.get("rms_opd_nm", 1.0)))
    _validate_nonnegative(amp, name="high_order_wfe.truth.amplitude_nm_rms")
    alpha = float(truth_cfg.get("power_law_alpha", truth_cfg.get("alpha", 2.5)))
    truth_remove_low_order = bool(truth_cfg.get("remove_low_order_zernikes", True))
    low_order_noll = [
        int(i)
        for i in truth_cfg.get(
            "remove_zernike_modes",
            [4, 5, 6, 7, 8, 9, 10, 11]
            if truth_remove_low_order
            else [],
        )
    ]
    mirror_knowledge = _mirror_knowledge_configs(
        mirrors=mirrors,
        knowledge_cfg=knowledge_cfg,
        inference_enabled=bool(inference_cfg.get("enabled", True)),
        truth_alpha=alpha,
        low_order_noll=low_order_noll,
        truth_remove_low_order=truth_remove_low_order,
        seed_context=seed_context,
    )
    knowledge_enabled_by_mirror = {
        mirror: bool(spec["enabled"]) and float(spec["requested_amplitude_nm_rms"]) >= 0.0
        for mirror, spec in mirror_knowledge.items()
    }

    def mirror_wfe_cfg(mirror: str) -> dict[str, Any]:
        spec = mirror_knowledge[mirror]
        return {
            "truth": {
                "rms_opd_nm": amp,
                "power_law_alpha": alpha,
                "fit_low_order_zernikes": low_order_noll,
            },
            "knowledge": {
                "low_order_sigma_nm_per_coeff": 0.0,
                "high_order_error_rms_nm": spec["requested_amplitude_nm_rms"],
                "high_order_error_power_law_alpha": spec["power_law_alpha"],
                "high_order_error_remove_noll_indices": spec["remove_noll_indices"],
                "high_order_error_seed": spec["seed"],
            },
        }

    deck = build_high_order_wfe_deck(
        shape=(shape_npix, shape_npix),
        seed=truth_seed,
        primary_config=mirror_wfe_cfg("primary"),
        secondary_config=mirror_wfe_cfg("secondary"),
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
        knowledge_enabled_by_mirror={},
        config_map_root=config_map_root,
        write_config_maps=write_artifacts,
    )
    inference_optics["high_order_wfe"] = _deck_cfg_to_optics_block(
        deck,
        mirrors=mirrors,
        truth=False,
        knowledge_enabled_by_mirror=knowledge_enabled_by_mirror,
        config_map_root=config_map_root,
        write_config_maps=write_artifacts,
    )
    truth_system["optics"] = truth_optics
    inference_system["optics"] = inference_optics

    if config_map_root is not None:
        for path in sorted(config_map_root.glob("*.npy")):
            artifact_paths[f"config_map_{path.stem}"] = str(path)
    knowledge_enabled = any(bool(spec["enabled"]) for spec in mirror_knowledge.values())
    nonzero_knowledge_enabled = any(
        bool(spec["enabled"]) and float(spec["requested_amplitude_nm_rms"]) > 0.0
        for spec in mirror_knowledge.values()
    )
    shared_error_amp = knowledge_cfg.get("amplitude_nm_rms", knowledge_cfg.get("rms_nm", None))
    summary_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "enabled": True,
        "mirrors": list(mirrors),
        "pairing": pairing,
        "knowledge_error_pairing": knowledge_pairing,
        "seed_context": json_ready(seed_context),
        "base_seed": int(base_seed),
        "truth_seed": int(truth_seed),
        "knowledge_enabled": knowledge_enabled,
        "knowledge_error_seed": knowledge_cfg.get("seed"),
        "truth_amplitude_nm_rms": amp,
        "knowledge_error_amplitude_nm_rms": shared_error_amp,
        "knowledge_error_by_mirror": copy.deepcopy(mirror_knowledge),
        "mask_policy": mask_policy,
        "npix": shape_npix,
        "remove_low_order_zernikes": truth_remove_low_order,
        "remove_zernike_modes": low_order_noll,
        "primary": _mirror_summary(
            deck.primary,
            active="primary" in mirrors,
            knowledge_enabled=bool(mirror_knowledge["primary"]["enabled"]),
            requested_error_rms_nm=float(
                mirror_knowledge["primary"]["requested_amplitude_nm_rms"]
            ),
        ),
        "secondary": _mirror_summary(
            deck.secondary,
            active="secondary" in mirrors,
            knowledge_enabled=bool(mirror_knowledge["secondary"]["enabled"]),
            requested_error_rms_nm=float(
                mirror_knowledge["secondary"]["requested_amplitude_nm_rms"]
            ),
        ),
        "deck_comparison": deck.comparison,
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
    if (
        bool(validation_cfg.get("require_nonzero_difference_when_enabled", False))
        and knowledge_enabled
        and not nonzero_knowledge_enabled
    ):
        raise ValueError("High-order WFE validation requires nonzero truth/inference difference, but knowledge error amplitude is zero.")
    if not knowledge_enabled:
        warnings.append("High-order WFE inference knowledge error is disabled; truth/reference maps match.")
    max_abs_low_order = validation_cfg.get("max_abs_low_order_projection_nm")
    if max_abs_low_order is not None and knowledge_enabled and low_order_noll:
        limit = float(max_abs_low_order)
        low_order_projection: dict[str, dict[str, float]] = {}
        for mirror_deck in (deck.primary, deck.secondary):
            coeffs = fit_zernike_coefficients_nm(
                mirror_deck.high_order_knowledge_error.opd_nm,
                low_order_noll,
                mask=mirror_deck.high_order_knowledge_error.mask,
            )
            max_abs = max((abs(float(v)) for v in coeffs.values()), default=0.0)
            low_order_projection[mirror_deck.mirror] = {
                **{key: float(value) for key, value in coeffs.items()},
                "max_abs_projection_nm": float(max_abs),
            }
            if max_abs > limit:
                message = (
                    f"{mirror_deck.mirror} high-order knowledge-error low-order projection "
                    f"{max_abs:.6g} nm exceeds validation.max_abs_low_order_projection_nm={limit:.6g}."
                )
                if bool(validation_cfg.get("fail_on_low_order_projection", True)):
                    raise ValueError(message)
                warnings.append(message)
        summary_payload["knowledge_error_low_order_projection_nm"] = low_order_projection

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
