"""Detector calibration-map knowledge-error helpers."""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .campaigns import json_ready, write_json
from .noise import apply_knowledge_error, make_subseed

SCHEMA_VERSION = "detector_calibration_knowledge_error.v1"
SUPPORTED_POLICIES = ("fixed_per_experiment", "per_run")
SUPPORTED_APPLY_TO = ("inference", "truth", "render", "both")


def _find_repo_root(start: Path) -> Path:
    start = start.resolve()
    for path in (start, *start.parents):
        if (path / ".git").exists() or (path / "pyproject.toml").exists():
            return path
    return start


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def _resolve_repo_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    out = Path(path).expanduser()
    if out.is_absolute():
        return out
    return (_REPO_ROOT / out).resolve()


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits

        data = fits.getdata(path)
        if data is None:
            raise ValueError(f"FITS file {path} does not contain image data.")
        return np.asarray(data, dtype=float)
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=float)
    if suffix == ".npz":
        with np.load(path) as z:
            for key in ("data", "arr_0", "dx", "dy", "prf", "pixel_response"):
                if key in z.files:
                    return np.asarray(z[key], dtype=float)
            return np.asarray(z[z.files[0]], dtype=float)
    raise ValueError(f"Unsupported detector calibration map type: {path}")


def _array_stats(arr: Any) -> dict[str, Any]:
    value = np.asarray(arr, dtype=float)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(np.nanmin(value)),
        "max": float(np.nanmax(value)),
        "mean": float(np.nanmean(value)),
        "std": float(np.nanstd(value)),
        "rms": float(np.sqrt(np.nanmean(value * value))),
        "finite": bool(np.isfinite(value).all()),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
    }


def normalize_realization_policy(value: Any) -> str:
    if value is None:
        return "fixed_per_experiment"
    policy = str(value).strip().lower()
    if policy not in SUPPORTED_POLICIES:
        raise ValueError(
            "detector calibration knowledge-error realization_policy must be one of "
            + ", ".join(SUPPORTED_POLICIES)
            + "."
        )
    return policy


def _layer_metadata_key(layer_name: str | None, idx: int, *, used: set[str]) -> str:
    base = (layer_name or f"layer_{idx}").strip() if layer_name is not None else f"layer_{idx}"
    if not base:
        base = f"layer_{idx}"
    key = base
    if key in used:
        key = f"{base}_{idx}"
    used.add(key)
    return key


def seed_detector_layer_knowledge_errors(
    system_cfg: Mapping[str, Any],
    *,
    experiment_seed: int,
    token_prefix: str,
    run_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach deterministic seeds to existing detector layer ``knowledge_error`` blocks."""

    cfg = copy.deepcopy(dict(system_cfg))
    detector_cfg = cfg.get("detector", {}) if isinstance(cfg, dict) else {}
    layers = detector_cfg.get("layers", []) if isinstance(detector_cfg, dict) else []
    seeded_layers: list[Any] = []
    metadata_layers: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    has_per_run = False

    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            seeded_layers.append(layer)
            continue
        layer_copy = dict(layer)
        knowledge_error = layer_copy.get("knowledge_error")
        if isinstance(knowledge_error, dict):
            seeded_ke = dict(knowledge_error)
            policy = normalize_realization_policy(seeded_ke.get("realization_policy"))
            seeded_ke["realization_policy"] = policy
            has_per_run = has_per_run or policy == "per_run"
            explicit_seed = seeded_ke.get("seed") is not None
            if not explicit_seed:
                seed_base = run_seed if policy == "per_run" and run_seed is not None else experiment_seed
                seeded_ke["seed"] = make_subseed(seed_base, f"{token_prefix}.{layer_copy.get('name') or 'layer'}.{idx}")
            layer_copy["knowledge_error"] = seeded_ke
            key = _layer_metadata_key(
                str(layer_copy.get("name")).strip() if layer_copy.get("name") is not None else None,
                idx,
                used=used,
            )
            metadata_layers[key] = {
                "name": layer_copy.get("name"),
                "index": idx,
                "model": seeded_ke.get("model"),
                "scale": seeded_ke.get("scale"),
                "realization_policy": policy,
                "seed": seeded_ke.get("seed"),
                "seed_source": "explicit"
                if explicit_seed
                else ("run_seed" if policy == "per_run" and run_seed is not None else "experiment_seed"),
            }
        seeded_layers.append(layer_copy)

    if isinstance(detector_cfg, dict):
        detector_cfg = dict(detector_cfg)
        detector_cfg["layers"] = seeded_layers
        cfg["detector"] = detector_cfg
    return cfg, {
        "schema_version": f"{SCHEMA_VERSION}.layer_seed_metadata",
        "token_prefix": token_prefix,
        "experiment_seed": int(experiment_seed),
        "run_seed": None if run_seed is None else int(run_seed),
        "has_per_run_realization": has_per_run,
        "layers": metadata_layers,
    }


def detector_ke_has_per_run_realization(system_cfg: Mapping[str, Any]) -> bool:
    detector_cfg = system_cfg.get("detector", {}) if isinstance(system_cfg, Mapping) else {}
    layers = detector_cfg.get("layers", []) if isinstance(detector_cfg, Mapping) else []
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        knowledge_error = layer.get("knowledge_error")
        if not isinstance(knowledge_error, Mapping):
            continue
        if normalize_realization_policy(knowledge_error.get("realization_policy")) == "per_run":
            return True
    return False


def _normalize_component(
    raw: Any,
    *,
    component: str,
    sigma_key: str,
    default_sigma: float,
) -> dict[str, Any]:
    cfg = dict(raw) if isinstance(raw, Mapping) else {}
    enabled = bool(cfg.get("enabled", False))
    sigma = float(cfg.get(sigma_key, cfg.get("sigma", cfg.get("scale", default_sigma))))
    if sigma < 0.0 or not math.isfinite(sigma):
        raise ValueError(f"detector_calibration_knowledge_error.{component}.{sigma_key} must be non-negative and finite.")
    distribution = str(cfg.get("distribution", cfg.get("model", "normal"))).strip().lower()
    if distribution not in {"normal", "gaussian"}:
        raise ValueError(f"detector_calibration_knowledge_error.{component}.distribution currently supports only 'normal'.")
    out = {
        "enabled": enabled,
        sigma_key: sigma,
        "distribution": "normal",
        "model": "gaussian",
        "scale": sigma,
    }
    if "clip_min" in cfg:
        out["clip_min"] = None if cfg.get("clip_min") is None else float(cfg["clip_min"])
    if "clip_max" in cfg:
        out["clip_max"] = None if cfg.get("clip_max") is None else float(cfg["clip_max"])
    return out


def normalize_detector_calibration_knowledge_error(raw: Any) -> dict[str, Any]:
    """Normalize campaign-level detector calibration-map knowledge-error config."""

    cfg = dict(raw) if isinstance(raw, Mapping) else {}
    enabled = bool(cfg.get("enabled", False))
    apply_to = str(cfg.get("apply_to", "inference")).strip().lower()
    if apply_to not in SUPPORTED_APPLY_TO:
        raise ValueError(
            "detector_calibration_knowledge_error.apply_to must be one of "
            + ", ".join(SUPPORTED_APPLY_TO)
            + "."
        )
    policy = normalize_realization_policy(cfg.get("realization_policy"))
    seed = cfg.get("seed")
    if seed is not None:
        seed = int(seed)
    pixel_response = _normalize_component(
        cfg.get("pixel_response", cfg.get("flat_field")),
        component="pixel_response",
        sigma_key="sigma_fractional",
        default_sigma=0.001,
    )
    pixel_response.setdefault("clip_min", 0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "enabled": enabled,
        "apply_to": apply_to,
        "realization_policy": policy,
        "seed": seed,
        "pixel_offsets": _normalize_component(
            cfg.get("pixel_offsets"),
            component="pixel_offsets",
            sigma_key="sigma_pix",
            default_sigma=0.001,
        ),
        "pixel_response": pixel_response,
    }


def _layer_ke(
    component_cfg: Mapping[str, Any],
    *,
    realization_policy: str,
    seed: int | None,
) -> dict[str, Any]:
    out = {
        "model": "gaussian",
        "scale": float(component_cfg["scale"]),
        "realization_policy": realization_policy,
    }
    if "clip_min" in component_cfg:
        out["clip_min"] = component_cfg.get("clip_min")
    if "clip_max" in component_cfg:
        out["clip_max"] = component_cfg.get("clip_max")
    if seed is not None:
        out["seed"] = int(seed)
    return out


def _patch_layers(
    system_cfg: Mapping[str, Any],
    normalized: Mapping[str, Any],
    *,
    side: str,
    seed_base: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = copy.deepcopy(dict(system_cfg))
    detector = dict(cfg.get("detector", {}) or {})
    layers = list(detector.get("layers", []) or [])
    out_layers: list[Any] = []
    patched: dict[str, Any] = {}
    for idx, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            out_layers.append(layer)
            continue
        layer_copy = dict(layer)
        name = str(layer_copy.get("name", ""))
        if name == "pixel_offsets" and normalized["pixel_offsets"]["enabled"]:
            seed = normalized.get("seed")
            if seed is None:
                seed = make_subseed(seed_base, f"detector_calibration_knowledge_error.{side}.pixel_offsets")
            layer_copy["knowledge_error"] = _layer_ke(
                normalized["pixel_offsets"],
                realization_policy=normalized["realization_policy"],
                seed=int(seed),
            )
            patched[name] = {
                "layer_name": name,
                "index": idx,
                "fields": ["dx_map", "dy_map"],
                "sigma_pix": float(normalized["pixel_offsets"]["sigma_pix"]),
                "seed": int(seed),
                "seed_source": "explicit" if normalized.get("seed") is not None else "experiment_seed",
            }
        if name == "pixel_response" and normalized["pixel_response"]["enabled"]:
            seed = normalized.get("seed")
            if seed is None:
                seed = make_subseed(seed_base, f"detector_calibration_knowledge_error.{side}.pixel_response")
            layer_copy["knowledge_error"] = _layer_ke(
                normalized["pixel_response"],
                realization_policy=normalized["realization_policy"],
                seed=int(seed),
            )
            patched[name] = {
                "layer_name": name,
                "index": idx,
                "fields": ["pixel_response"],
                "sigma_fractional": float(normalized["pixel_response"]["sigma_fractional"]),
                "seed": int(seed),
                "seed_source": "explicit" if normalized.get("seed") is not None else "experiment_seed",
            }
        out_layers.append(layer_copy)
    detector["layers"] = out_layers
    cfg["detector"] = detector
    return cfg, patched


def _summarize_realized_maps(system_cfg: Mapping[str, Any], patched_layers: Mapping[str, Any]) -> dict[str, Any]:
    detector = system_cfg.get("detector", {}) if isinstance(system_cfg, Mapping) else {}
    layers = detector.get("layers", []) if isinstance(detector, Mapping) else []
    summaries: dict[str, Any] = {}
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        name = str(layer.get("name", ""))
        if name not in patched_layers:
            continue
        ke = layer.get("knowledge_error")
        if not isinstance(ke, Mapping):
            continue
        fields: dict[str, str | None]
        if name == "pixel_offsets":
            fields = {"dx_map": layer.get("dx_path"), "dy_map": layer.get("dy_path")}
        elif name == "pixel_response":
            fields = {"pixel_response": layer.get("prf_path")}
        else:
            fields = {}
        field_summaries: dict[str, Any] = {}
        for field, raw_path in fields.items():
            path = _resolve_repo_path(raw_path)
            if path is None or not path.exists():
                field_summaries[field] = {"available": False, "path": None if path is None else str(path)}
                continue
            nominal = _load_array(path)
            token = f"{name}.dx" if field == "dx_map" else f"{name}.dy" if field == "dy_map" else f"{name}.prf"
            realized, used_seed = apply_knowledge_error(nominal, knowledge_cfg=ke, base_seed=None, token=token)
            realized_arr = np.asarray(realized, dtype=float)
            delta = realized_arr - nominal
            field_summaries[field] = {
                "available": True,
                "path": str(path),
                "used_subseed": used_seed,
                "nominal": _array_stats(nominal),
                "delta": _array_stats(delta),
                "realized": _array_stats(realized_arr),
            }
        summaries[name] = field_summaries
    return summaries


def apply_campaign_detector_calibration_knowledge_error(
    *,
    truth_system_cfg: Mapping[str, Any],
    inference_system_cfg: Mapping[str, Any],
    request: Any,
    seed_context: Mapping[str, Any] | None = None,
    artifact_root: Path | None = None,
    write_artifacts: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Patch truth/inference detector configs according to a campaign-level request."""

    normalized = normalize_detector_calibration_knowledge_error(request)
    truth = copy.deepcopy(dict(truth_system_cfg))
    inference = copy.deepcopy(dict(inference_system_cfg))
    artifact_paths: dict[str, str] = {}
    seed_context = dict(seed_context or {})
    seed_base = int(normalized["seed"] if normalized.get("seed") is not None else seed_context.get("base_seed", 0))
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request": normalized,
        "seed_context": json_ready(seed_context),
        "enabled": bool(normalized["enabled"]),
        "apply_to": normalized["apply_to"],
        "realization_scope": "global_model_split_template_fixed_across_cases_and_subblocks",
        "truth_patched": False,
        "inference_patched": False,
        "patched_layers": {"truth": {}, "inference": {}},
        "realized_map_summaries": {"truth": {}, "inference": {}},
    }
    if not normalized["enabled"]:
        provenance["disabled_reason"] = "detector_calibration_knowledge_error.enabled is false or absent"
        return truth, inference, provenance, artifact_paths
    if normalized["realization_policy"] == "per_run":
        raise ValueError(
            "detector_calibration_knowledge_error.realization_policy='per_run' is not "
            "supported by the full-fidelity campaign-level model split because trace/render/"
            "inference templates are global artifacts. Use fixed_per_experiment for static "
            "calibration mismatch, or layer-level detector knowledge_error in prescribed-MC "
            "workflows for per-run realizations."
        )

    apply_to = normalized["apply_to"]
    if apply_to in {"truth", "render", "both"}:
        truth, patched = _patch_layers(truth, normalized, side="truth", seed_base=seed_base)
        provenance["truth_patched"] = bool(patched)
        provenance["patched_layers"]["truth"] = patched
        provenance["realized_map_summaries"]["truth"] = _summarize_realized_maps(truth, patched)
    if apply_to in {"inference", "both"}:
        inference, patched = _patch_layers(inference, normalized, side="inference", seed_base=seed_base)
        provenance["inference_patched"] = bool(patched)
        provenance["patched_layers"]["inference"] = patched
        provenance["realized_map_summaries"]["inference"] = _summarize_realized_maps(inference, patched)
    if normalized["pixel_offsets"]["enabled"] and not (
        provenance["patched_layers"]["truth"].get("pixel_offsets") or provenance["patched_layers"]["inference"].get("pixel_offsets")
    ):
        raise ValueError("detector calibration KE requested pixel_offsets but no detector layer named 'pixel_offsets' was found.")
    if normalized["pixel_response"]["enabled"] and not (
        provenance["patched_layers"]["truth"].get("pixel_response") or provenance["patched_layers"]["inference"].get("pixel_response")
    ):
        raise ValueError("detector calibration KE requested pixel_response but no detector layer named 'pixel_response' was found.")
    if artifact_root is not None and write_artifacts:
        path = Path(artifact_root) / "detector_knowledge_error" / "detector_knowledge_error_provenance.json"
        write_json(path, provenance)
        artifact_paths["detector_knowledge_error_provenance_json"] = str(path)
    return truth, inference, provenance, artifact_paths


__all__ = [
    "SCHEMA_VERSION",
    "apply_campaign_detector_calibration_knowledge_error",
    "detector_ke_has_per_run_realization",
    "normalize_detector_calibration_knowledge_error",
    "normalize_realization_policy",
    "seed_detector_layer_knowledge_errors",
]
