"""Review helpers for the full-fidelity resolved-system notebook.

These functions are diagnostic utilities. They intentionally avoid launching a
campaign or optimization run; they resolve the smoke config, build the same
truth/reference model split used by the wrapper, and expose compact arrays,
tables, and summaries for notebooks/tests.
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "dluxshera-matplotlib"))

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from dluxshera.config.io import load_config_file
from dluxshera.config.resolver import resolve_config
from dluxshera.components.detectors import GSENSE2020BSI_SPEC, HWK4123_SPEC, DetectorSpec
from dluxshera.params.store import ParameterStore
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.campaign_model_split import (
    CampaignModelSplit,
    build_campaign_model_split,
    summarize_campaign_model_split,
)
from dluxshera.utils.detector_layer_overrides import (
    apply_detector_layer_overrides,
    detector_blur_warnings,
    detector_layer_stack,
)
from dluxshera.utils.full_fidelity_defaults import DEFAULT_FULL_FIDELITY_SYSTEM_PRESET
from dluxshera.utils.high_order_wfe import (
    fit_zernike_coefficients_nm,
    make_pupil_mask,
    remove_zernike_modes,
    reconstruct_zernike_opd_nm,
)
from dluxshera.utils.noise import (
    apply_observation_noise,
    detector_spec_for_model,
    expected_noise_variance,
    normalize_noise_request,
    resolve_detector_noise_spec,
)
from dluxshera.utils.obs_subblock_trajectory import (
    DEFAULT_OUTPUT_KEYS,
    prepare_airbus_subblocks,
)
from dluxshera.utils.spectral_response import (
    DEFAULT_DETECTOR_QE_PATH,
    DEFAULT_FILTER_RESPONSE_PATH,
    load_response_curve_csv,
)

DEFAULT_SMOKE_CONFIG = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_smoke.yaml"
)
DEFAULT_REVIEW_CONFIG = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/full_fidelity_binary_iterative_review.yaml"
)
DEFAULT_OUTPUT_ROOT = Path("Results/full_fidelity_resolved_system_review")
LOW_ORDER_NOLL_INDICES = (4, 5, 6, 7, 8, 9, 10, 11)


def repo_root(start: str | Path | None = None) -> Path:
    """Find the repository root from ``start`` or the current working directory."""

    p = Path(start or Path.cwd()).resolve()
    if p.is_file():
        p = p.parent
    for candidate in (p, *p.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return p


def resolve_repo_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (repo_root(root) / p).resolve()


def load_smoke_config(path: str | Path = DEFAULT_SMOKE_CONFIG) -> dict[str, Any]:
    """Load the smoke YAML as a plain dictionary."""

    return dict(load_config_file(resolve_repo_path(path)))


def _load_smoke_wrapper_module() -> Any:
    """Load the smoke wrapper by file path.

    TODO: replace this private importer with a public translator once the wrapper
    exposes one from package code.
    """

    root = repo_root()
    script = root / "examples" / "scripts" / "run_full_fidelity_binary_iterative_campaign.py"
    scripts_dir = str(script.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("_dluxshera_full_fidelity_smoke_wrapper", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import smoke wrapper from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def translate_smoke_to_observation_bias(config: Mapping[str, Any], *, run_name: str | None = None) -> dict[str, Any]:
    """Translate smoke config with the same private wrapper path used by the campaign."""

    module = _load_smoke_wrapper_module()
    return module._full_fidelity_to_observation_bias(config, run_name=run_name)


def _experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    exp = config.get("experiment", config)
    if not isinstance(exp, Mapping):
        raise ValueError("Config must contain a mapping-valued experiment block.")
    return dict(exp)


def _subblock_exposure_time_s(experiment_cfg: Mapping[str, Any]) -> float | None:
    sub = experiment_cfg.get("subblocks", {})
    if not isinstance(sub, Mapping) or sub.get("exposure_time_s") is None:
        return None
    exposure = float(sub["exposure_time_s"])
    if exposure <= 0.0 or not math.isfinite(exposure):
        raise ValueError("subblocks.exposure_time_s must be positive and finite.")
    return exposure


def _resolve_base_system_config(translated_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], ParameterStore, dict[str, Any]]:
    exp = _experiment(translated_cfg)
    system_seed = exp.get("system") if isinstance(exp.get("system"), Mapping) else None
    if system_seed is None:
        system_seed = {"preset": exp.get("system_preset", DEFAULT_FULL_FIDELITY_SYSTEM_PRESET)}
    user_cfg = {"system": copy.deepcopy(dict(system_seed))}
    exposure = _subblock_exposure_time_s(exp)
    if exposure is not None:
        source = dict(user_cfg["system"].get("source", {}) or {})
        source["exposure_time_s"] = exposure
        user_cfg["system"]["source"] = source
    resolved = resolve_config(user_cfg)
    system_cfg = dict(resolved["system"])
    detector_stack_from_preset = detector_layer_stack(system_cfg)
    detector_override_provenance = None
    detector_overrides = exp.get("detector_overrides")
    if isinstance(detector_overrides, Mapping):
        system_cfg, detector_override_provenance = apply_detector_layer_overrides(
            system_cfg,
            detector_overrides,
            context="full_fidelity_review.global",
        )
    spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    provenance = {
        "system_preset": system_cfg.get("preset"),
        "source_kind": (system_cfg.get("source") or {}).get("kind"),
        "source_target": (system_cfg.get("source") or {}).get("target"),
        "optics_kind": (system_cfg.get("optics") or {}).get("kind"),
        "detector_model": (system_cfg.get("detector") or {}).get("model"),
        "detector_layer_stack_from_preset": detector_stack_from_preset,
        "detector_layer_stack_after_global_overrides": detector_layer_stack(system_cfg),
        "detector_layer_overrides": detector_override_provenance,
    }
    return system_cfg, store, provenance


def build_model_split_from_smoke(
    config: Mapping[str, Any],
    outdir: str | Path,
    *,
    run_label: str | None = None,
    write_artifacts: bool = True,
) -> dict[str, Any]:
    """Build the base/truth/inference system split without launching a campaign."""

    translated = translate_smoke_to_observation_bias(config, run_name=run_label)
    exp = _experiment(translated)
    run_root = Path(outdir).resolve()
    base_system_cfg, store, system_prov = _resolve_base_system_config(translated)
    sub = dict(exp.get("subblocks", {}) or {})
    source_cfg = base_system_cfg.get("source", {}) if isinstance(base_system_cfg.get("source"), Mapping) else {}
    smear_cfg = sub.get("trajectory_processing", {}).get("smear", {}) if isinstance(sub.get("trajectory_processing"), Mapping) else {}
    split = build_campaign_model_split(
        base_system_cfg=base_system_cfg,
        spectral_model_cfg=exp.get("spectral_model"),
        high_order_wfe_cfg=exp.get("high_order_wfe"),
        detector_noise_metadata={
            "enabled": str(sub.get("noise", "disabled")) != "disabled",
            "noise_mode": str(sub.get("noise", "disabled")),
        },
        run_root=run_root,
        artifact_root=run_root / "model_split",
        seed_context={
            "wrapper": "observation_bias_campaign",
            "run_name": str(run_label or exp.get("run_name", "full_fidelity_review")),
            "base_seed": int(exp.get("seed", 42)),
        },
        source_kind=str(source_cfg.get("kind", "binary")),
        target=source_cfg.get("target"),
        write_artifacts=write_artifacts,
        trajectory_smear_metadata=smear_cfg if isinstance(smear_cfg, Mapping) else None,
    )
    return {
        "translated_config": translated,
        "experiment": exp,
        "base_system_cfg": base_system_cfg,
        "base_store": store,
        "system_provenance": system_prov,
        "model_split": split,
        "truth_system_cfg": split.truth_system_cfg,
        "inference_system_cfg": split.inference_system_cfg,
    }


def source_block(system_cfg: Mapping[str, Any]) -> dict[str, Any]:
    source = system_cfg.get("source", {}) if isinstance(system_cfg, Mapping) else {}
    return dict(source or {})


def summarize_source_config(system_cfg: Mapping[str, Any]) -> dict[str, Any]:
    src = source_block(system_cfg)
    wavelengths = np.asarray(src.get("wavelengths_m", []), dtype=float).reshape(-1)
    weights = np.asarray(src.get("weights", []), dtype=float).reshape(-1)
    component_weights = np.asarray(src.get("component_weights", []), dtype=float)
    return {
        "kind": src.get("kind"),
        "target": src.get("target"),
        "wavelength_m": src.get("wavelength_m"),
        "bandwidth_m": src.get("bandwidth_m"),
        "n_lambda": src.get("n_lambda"),
        "has_wavelengths_m": bool(wavelengths.size),
        "has_weights": bool(weights.size),
        "has_component_weights": bool(component_weights.size),
        "wavelengths_nm": (wavelengths * 1e9).tolist(),
        "weights_sum": float(np.sum(weights)) if weights.size else None,
        "component_weight_sums": np.sum(component_weights, axis=1).tolist() if component_weights.ndim == 2 else [],
        "log_flux_total": src.get("log_flux_total"),
        "contrast": src.get("contrast"),
        "spectral_deck_provenance": src.get("spectral_deck_provenance", {}),
    }


def extract_spectral_arrays(system_cfg: Mapping[str, Any]) -> dict[str, Any]:
    src = source_block(system_cfg)
    wavelengths = np.asarray(src.get("wavelengths_m", []), dtype=float).reshape(-1)
    n = int(src.get("n_lambda", wavelengths.size or 0) or 0)
    if wavelengths.size == 0 and n > 0:
        center = float(src.get("wavelength_m"))
        bw = float(src.get("bandwidth_m", 0.0))
        wavelengths = np.linspace(center - 0.5 * bw, center + 0.5 * bw, n)
    component_weights = np.asarray(src.get("component_weights", []), dtype=float)
    if component_weights.ndim != 2 and wavelengths.size:
        weights = np.asarray(src.get("weights", []), dtype=float).reshape(-1)
        if weights.size != wavelengths.size:
            weights = np.full(wavelengths.size, 1.0 / wavelengths.size)
        component_weights = np.vstack([weights, weights])
    labels = list(src.get("component_labels", ["primary", "secondary"]))
    if len(labels) < component_weights.shape[0] if component_weights.ndim == 2 else False:
        labels = ["primary", "secondary"][: component_weights.shape[0]]
    return {
        "wavelengths_m": wavelengths,
        "wavelengths_nm": wavelengths * 1e9,
        "component_labels": labels,
        "component_weights": component_weights,
        "source": src,
    }


def spectral_review_tables(base_cfg: Mapping[str, Any], truth_cfg: Mapping[str, Any], inference_cfg: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for role, cfg in (("base", base_cfg), ("truth", truth_cfg), ("inference", inference_cfg)):
        arr = extract_spectral_arrays(cfg)
        weights = arr["component_weights"]
        role_rows: list[dict[str, Any]] = []
        if weights.ndim == 2:
            for comp_index, label in enumerate(arr["component_labels"][: weights.shape[0]]):
                for i, wavelength_nm in enumerate(arr["wavelengths_nm"]):
                    role_rows.append({
                        "role": role,
                        "component": label,
                        "index": i,
                        "wavelength_nm": float(wavelength_nm),
                        "weight": float(weights[comp_index, i]),
                    })
        rows[role] = role_rows
    return rows


def summarize_spectral_deck(model_split: CampaignModelSplit) -> dict[str, Any]:
    truth = summarize_source_config(model_split.truth_system_cfg)
    inference = summarize_source_config(model_split.inference_system_cfg)
    warnings: list[str] = []
    for role, summary in (("truth", truth), ("inference", inference)):
        for idx, total in enumerate(summary["component_weight_sums"]):
            if not np.isclose(float(total), 1.0, rtol=0.0, atol=1e-10):
                warnings.append(f"{role} component_weights row {idx} sums to {total}, not 1.")
    return {
        "truth": truth,
        "inference": inference,
        "provenance": model_split.provenance.get("spectral_model", {}),
        "comparison": model_split.provenance.get("spectral_model", {}).get("combined_comparison", {}),
        "warnings": warnings,
    }


def preserve_flux_review(base_cfg: Mapping[str, Any], truth_cfg: Mapping[str, Any], inference_cfg: Mapping[str, Any], spectral_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    preserve_requested = bool((spectral_cfg or {}).get("preserve_flux_parameters", False))
    base = source_block(base_cfg)
    truth = source_block(truth_cfg)
    inference = source_block(inference_cfg)
    warnings: list[str] = []
    for key in ("log_flux_total", "contrast"):
        if preserve_requested and key in base:
            if truth.get(key) != base.get(key):
                warnings.append(f"truth source.{key} changed despite preserve_flux_parameters=true")
            if inference.get(key) != base.get(key):
                warnings.append(f"inference source.{key} changed despite preserve_flux_parameters=true")
    if "preserve_flux_parameters" in (spectral_cfg or {}) and "spectral_model" not in str(truth.get("spectral_deck_provenance", {})):
        warnings.append("preserve_flux_parameters is configured; consumption should be verified against spectral provenance.")
    return {
        "preserve_flux_parameters": preserve_requested,
        "base": {k: base.get(k) for k in ("log_flux_total", "contrast")},
        "truth": {k: truth.get(k) for k in ("log_flux_total", "contrast")},
        "inference": {k: inference.get(k) for k in ("log_flux_total", "contrast")},
        "warnings": warnings,
    }


def _load_opd_from_cfg(item: Mapping[str, Any]) -> np.ndarray | None:
    if not isinstance(item, Mapping):
        return None
    if "array_nm" in item:
        return np.asarray(item["array_nm"], dtype=float)
    if "array_path" in item:
        return np.asarray(np.load(resolve_repo_path(str(item["array_path"]))), dtype=float)
    return None


def _mirror_wfe_maps(system_cfg: Mapping[str, Any], mirror: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    optics = system_cfg.get("optics", {}) if isinstance(system_cfg.get("optics"), Mapping) else {}
    block = optics.get("high_order_wfe", {}) if isinstance(optics.get("high_order_wfe"), Mapping) else {}
    mirror_cfg = block.get(mirror, {}) if isinstance(block.get(mirror), Mapping) else {}
    map_cfg = mirror_cfg.get("map", {}) if isinstance(mirror_cfg.get("map"), Mapping) else {}
    err_cfg = mirror_cfg.get("knowledge_error", {}) if isinstance(mirror_cfg.get("knowledge_error"), Mapping) else {}
    return _load_opd_from_cfg(map_cfg), _load_opd_from_cfg(err_cfg)


def _masked_stats(arr: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    vals = np.asarray(arr, dtype=float)[mask]
    return {
        "mean_nm": float(np.mean(vals)),
        "rms_nm": float(np.sqrt(np.mean(vals * vals))),
        "min_nm": float(np.min(vals)),
        "p01_nm": float(np.percentile(vals, 1)),
        "p50_nm": float(np.percentile(vals, 50)),
        "p99_nm": float(np.percentile(vals, 99)),
        "max_nm": float(np.max(vals)),
    }


def masked_for_imshow(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a float image with pixels outside ``mask`` set to NaN."""

    data = np.asarray(arr, dtype=float).copy()
    valid = np.asarray(mask, dtype=bool)
    if data.shape != valid.shape:
        raise ValueError(f"Map/mask shape mismatch: {data.shape} vs {valid.shape}.")
    data[~valid] = np.nan
    return data


def cmap_with_bad(base: str = "RdBu_r", bad: str = "0.5") -> Any:
    """Return a copied Matplotlib colormap with NaN/bad pixels set to grey."""

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(base).copy()
    cmap.set_bad(bad)
    return cmap


def symmetric_nan_limits(arr: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    """Return symmetric color limits from finite values in ``arr``."""

    vals = np.asarray(arr, dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.nanpercentile(np.abs(finite), float(percentile)))
    if not np.isfinite(limit) or limit == 0.0:
        limit = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    if not np.isfinite(limit) or limit == 0.0:
        limit = 1.0
    return -limit, limit


def _wfe_artifact_manifest(model_split: CampaignModelSplit) -> dict[str, Any] | None:
    for key, path in model_split.artifact_paths.items():
        if key.endswith("high_order_wfe_deck_manifest.json"):
            p = resolve_repo_path(path)
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
    return None


def _manifest_map(manifest: Mapping[str, Any] | None, mirror: str, name: str) -> np.ndarray | None:
    if not manifest:
        return None
    path = (((manifest.get("artifacts") or {}).get(f"{mirror}_{name}_opd_nm.fits")))
    if not path:
        return None
    try:
        from astropy.io import fits

        return np.asarray(fits.getdata(resolve_repo_path(path)), dtype=float)
    except Exception:
        return None


def _manifest_coefficients(manifest: Mapping[str, Any] | None, mirror: str) -> dict[str, dict[str, float]] | None:
    if not manifest or not isinstance(manifest.get(mirror), Mapping):
        return None
    item = manifest[mirror]
    truth = item.get("low_order_truth_coeffs_nm")
    inference = item.get("low_order_knowledge_coeffs_nm")
    error = item.get("low_order_knowledge_error_nm")
    if not all(isinstance(x, Mapping) for x in (truth, inference, error)):
        return None
    return {
        "truth": {str(k): float(v) for k, v in truth.items()},
        "inference": {str(k): float(v) for k, v in inference.items()},
        "error": {str(k): float(v) for k, v in error.items()},
    }


def _provenance_coefficients(prov: Mapping[str, Any], mirror: str) -> dict[str, dict[str, float]] | None:
    item = prov.get(mirror)
    if not isinstance(item, Mapping):
        return None
    truth = item.get("low_order_truth_coefficients_nm")
    inference = item.get("low_order_inference_coefficients_nm")
    error = item.get("low_order_error_coefficients_nm")
    if not all(isinstance(x, Mapping) for x in (truth, inference, error)):
        return None
    return {
        "truth": {str(k): float(v) for k, v in truth.items()},
        "inference": {str(k): float(v) for k, v in inference.items()},
        "error": {str(k): float(v) for k, v in error.items()},
    }


def _max_abs_coeff(coeffs: Mapping[str, float]) -> float:
    return max((abs(float(v)) for v in coeffs.values()), default=0.0)


def summarize_wfe_artifacts(model_split: CampaignModelSplit, *, noll_indices: Sequence[int] = LOW_ORDER_NOLL_INDICES) -> dict[str, Any]:
    """Summarize truth/reference high-order OPD maps inserted in system configs."""

    out: dict[str, Any] = {"mirrors": {}, "warnings": []}
    prov = model_split.provenance.get("high_order_wfe", {}).get("provenance", model_split.provenance.get("high_order_wfe", {}))
    if not prov.get("enabled", False):
        out["enabled"] = False
        return out
    out["enabled"] = True
    requested_truth = float(prov.get("truth_amplitude_nm_rms", np.nan))
    requested_error = float(prov.get("knowledge_error_amplitude_nm_rms", np.nan))
    manifest = _wfe_artifact_manifest(model_split)
    projection_limit = float(
        prov.get("validation", {}).get("max_abs_low_order_projection_nm", 1.0e-8)
        if isinstance(prov.get("validation"), Mapping)
        else 1.0e-8
    )
    sum_limit = 1.0e-10
    for mirror in ("primary", "secondary"):
        truth, _ = _mirror_wfe_maps(model_split.truth_system_cfg, mirror)
        ref_truth, err = _mirror_wfe_maps(model_split.inference_system_cfg, mirror)
        if truth is None or ref_truth is None:
            out["warnings"].append(f"{mirror} high-order WFE map is absent.")
            continue
        error = np.zeros_like(truth) if err is None else np.asarray(err, dtype=float)
        inference = ref_truth + error
        mask = make_pupil_mask(truth.shape, mode=str(prov.get("mask_policy", "circular_fallback")))
        raw_truth = _manifest_map(manifest, mirror, "full_truth")
        stored_coeff = _manifest_coefficients(manifest, mirror) or _provenance_coefficients(prov, mirror)
        warnings: list[str] = []
        if raw_truth is None:
            warnings.append("Raw PTT-removed truth OPD artifact is unavailable; reconstructing from stored coefficients and high-order residual.")
        if stored_coeff is None:
            warnings.append("Stored low-order coefficient artifacts are unavailable; falling back to refit from reconstructed raw truth.")
            low_truth_fallback = fit_zernike_coefficients_nm(raw_truth if raw_truth is not None else truth, mask, list(noll_indices))
            stored_coeff = {
                "truth": low_truth_fallback,
                "inference": {key: np.nan for key in low_truth_fallback},
                "error": {key: np.nan for key in low_truth_fallback},
            }
        labels = [f"Z{int(i)}" for i in noll_indices]
        low_truth = {label: float(stored_coeff["truth"].get(label, 0.0)) for label in labels}
        low_recon = reconstruct_zernike_opd_nm(low_truth, truth.shape, mask=mask)
        if raw_truth is None:
            raw_truth = low_recon + truth
        raw_truth = np.asarray(raw_truth, dtype=float)
        truth_residual = np.asarray(truth, dtype=float)
        knowledge_error_residual = np.asarray(inference - truth_residual, dtype=float)
        inference_sum_residual = inference - truth_residual - knowledge_error_residual
        residual_projection = {
            "truth_high_order_residual": fit_zernike_coefficients_nm(truth_residual, mask, list(noll_indices)),
            "knowledge_error_residual": fit_zernike_coefficients_nm(knowledge_error_residual, mask, list(noll_indices)),
            "inference_high_order": fit_zernike_coefficients_nm(inference, mask, list(noll_indices)),
        }
        coeff_truth = residual_projection["truth_high_order_residual"]
        coeff_inference = residual_projection["inference_high_order"]
        coeff_error = residual_projection["knowledge_error_residual"]
        residual_truth, _ = remove_zernike_modes(truth_residual, list(noll_indices), mask=mask)
        residual_error, _ = remove_zernike_modes(knowledge_error_residual, list(noll_indices), mask=mask)
        measured_truth = _masked_stats(truth_residual, mask)
        measured_error = _masked_stats(knowledge_error_residual, mask)
        max_sum = float(np.max(np.abs(inference_sum_residual[mask])))
        max_projection = max(_max_abs_coeff(coeffs) for coeffs in residual_projection.values())
        if max_sum > sum_limit:
            warnings.append(f"inference sum residual max {max_sum:.6g} nm exceeds {sum_limit:.6g} nm.")
        if max_projection > projection_limit:
            warnings.append(f"residual low-order projection max {max_projection:.6g} nm exceeds {projection_limit:.6g} nm.")
        if np.nanmax(np.abs(raw_truth[mask])) < 1.0e-6 and requested_truth > 1.0e-3:
            warnings.append("Raw truth OPD magnitude is unexpectedly tiny for nm units; check map units.")
        if truth_residual.shape[0] != int(prov.get("npix", truth_residual.shape[0])):
            warnings.append("High-order WFE map shape differs from provenance npix.")
        rms = {
            "raw_ptt_removed_truth": _masked_stats(raw_truth, mask)["rms_nm"],
            "low_order_reconstruction": _masked_stats(low_recon, mask)["rms_nm"],
            "truth_high_order_residual": measured_truth["rms_nm"],
            "knowledge_error_residual": measured_error["rms_nm"],
            "inference_high_order": _masked_stats(inference, mask)["rms_nm"],
            "inference_sum_residual": _masked_stats(inference_sum_residual, mask)["rms_nm"],
        }
        out["mirrors"][mirror] = {
            "raw_ptt_removed_truth_opd_nm": raw_truth,
            "low_order_truth_reconstruction_nm": low_recon,
            "truth_high_order_residual_opd_nm": truth_residual,
            "knowledge_error_high_order_residual_opd_nm": knowledge_error_residual,
            "inference_high_order_opd_nm": inference,
            "inference_sum_residual_nm": inference_sum_residual,
            "stored_low_order_coefficients_nm": stored_coeff,
            "residual_low_order_projection_nm": residual_projection,
            "rms_nm": rms,
            "truth_opd_nm": truth_residual,
            "inference_opd_nm": inference,
            "knowledge_error_opd_nm": knowledge_error_residual,
            "mask": mask,
            "requested_truth_rms_nm": requested_truth,
            "requested_knowledge_error_rms_nm": requested_error,
            "truth_stats": measured_truth,
            "inference_stats": _masked_stats(inference, mask),
            "knowledge_error_stats": measured_error,
            "zernike_coefficients_nm": {
                "truth": coeff_truth,
                "inference": coeff_inference,
                "error": coeff_error,
            },
            "low_order_residual_rms_nm": {
                "truth": _masked_stats(residual_truth, mask)["rms_nm"],
                "error": _masked_stats(residual_error, mask)["rms_nm"],
            },
            "noll_index_mapping": {f"Z{int(i)}": idx for idx, i in enumerate(noll_indices)},
            "coefficient_array_index_mapping": {
                idx: {"label": f"Z{int(i)}", "noll_index": int(i)}
                for idx, i in enumerate(noll_indices)
            },
            "warnings": warnings,
        }
        out["warnings"].extend(f"{mirror}: {warning}" for warning in warnings)
        if np.isfinite(requested_error) and not np.isclose(measured_error["rms_nm"], requested_error, rtol=0.05, atol=1e-6):
            out["warnings"].append(f"{mirror} knowledge-error RMS {measured_error['rms_nm']:.4g} nm differs from requested {requested_error:.4g} nm.")
    out["provenance"] = prov
    return out


def summarize_optics_config(system_cfg: Mapping[str, Any]) -> dict[str, Any]:
    optics = dict(system_cfg.get("optics", {}) or {})
    high = optics.get("high_order_wfe", {}) if isinstance(optics.get("high_order_wfe"), Mapping) else {}
    return {
        "preset": system_cfg.get("preset"),
        "kind": optics.get("kind"),
        "psf_npix": optics.get("psf_npix"),
        "oversample": optics.get("oversample"),
        "pupil_npix": optics.get("pupil_npix"),
        "plate_scale_as_per_pix": optics.get("plate_scale_as_per_pix"),
        "primary_noll_indices": optics.get("primary_noll_indices"),
        "secondary_noll_indices": optics.get("secondary_noll_indices"),
        "high_order_wfe_enabled": bool(high.get("enabled", False)),
        "diffractive_pupil_path": optics.get("dp_path"),
        "special_keys": sorted(k for k in optics if any(token in k.lower() for token in ("mask", "filter", "pupil", "phase", "dp"))),
    }


def optics_diff_table(base_cfg: Mapping[str, Any], truth_cfg: Mapping[str, Any], inference_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = [
        "optics.kind", "optics.psf_npix", "optics.oversample", "optics.pupil_npix",
        "optics.plate_scale_as_per_pix", "optics.primary_noll_indices", "optics.secondary_noll_indices",
        "optics.high_order_wfe.enabled", "optics.dp_path",
    ]
    return _diff_paths(base_cfg, truth_cfg, inference_cfg, paths)


def _get_path(mapping: Mapping[str, Any], path: str) -> Any:
    cur: Any = mapping
    for key in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _set_path(mapping: dict[str, Any], path: str, value: Any) -> None:
    cur: dict[str, Any] = mapping
    parts = path.split(".")
    for key in parts[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _finite_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out > 0.0 and math.isfinite(out):
        return out
    return None


def _center_crop_array(arr: np.ndarray, crop_npix: int | None) -> np.ndarray:
    image = np.asarray(arr)
    if crop_npix is None:
        return image
    crop = int(crop_npix)
    if crop <= 0 or min(image.shape[-2:]) <= crop:
        return image
    ny, nx = image.shape[-2:]
    cy, cx = ny // 2, nx // 2
    half = crop // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(ny, y0 + crop)
    x1 = min(nx, x0 + crop)
    y0 = max(0, y1 - crop)
    x0 = max(0, x1 - crop)
    return image[..., y0:y1, x0:x1]


def resolve_review_psf_npix(
    truth_system_cfg: Mapping[str, Any],
    review_cfg: Mapping[str, Any] | None = None,
    *,
    minimum: int = 160,
    default: int = 256,
) -> tuple[int, dict[str, Any]]:
    """Resolve the noise-review render size from review overrides or system config."""

    candidates: list[dict[str, Any]] = []

    def add_candidate(path: str, value: Any) -> None:
        parsed = None
        if value is not None:
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                parsed = None
        candidates.append({"path": path, "value": value, "resolved": parsed})

    if isinstance(review_cfg, Mapping):
        for path in (
            "noise_review.psf_npix",
            "noise_review.rendered_psf_npix",
            "render_noise.psf_npix",
            "psf_npix",
        ):
            value = _get_path(review_cfg, path)
            if value is not None:
                add_candidate(f"review.{path}", value)

    for path in (
        "optics.psf_npix",
        "system.optics.psf_npix",
        "source.psf_npix",
        "system.source.psf_npix",
    ):
        value = _get_path(truth_system_cfg, path)
        if value is not None:
            add_candidate(f"truth_system.{path}", value)

    add_candidate("default", default)
    selected = next((item for item in candidates if item["resolved"] is not None), candidates[-1])
    requested = int(selected["resolved"] if selected["resolved"] is not None else default)
    warnings_out: list[str] = []
    final = requested
    minimum_enforced = False
    if requested < int(minimum):
        final = int(minimum)
        minimum_enforced = True
        warnings_out.append(
            f"Resolved noise-review psf_npix={requested} from {selected['path']} is below minimum {minimum}; "
            f"using {final} for the main render."
        )
    return final, {
        "requested_value": requested,
        "minimum": int(minimum),
        "default": int(default),
        "minimum_enforced": minimum_enforced,
        "final_value": int(final),
        "source_field_path": selected["path"],
        "all_candidate_values": candidates,
        "warnings": warnings_out,
    }


def resolve_noise_review_exposure_time_s(
    translated_config: Mapping[str, Any],
    truth_system_cfg: Mapping[str, Any],
    *,
    default: float | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve exposure time for review rendering and variance diagnostics.

    Priority order is campaign/subblock exposure, translated system source
    exposure, resolved truth-system source exposure, then an explicit fallback.
    """

    candidates: list[dict[str, Any]] = []

    def add_candidate(path: str, value: Any) -> None:
        candidates.append({"path": path, "value": value, "resolved": _finite_positive_float(value)})

    for path in (
        "experiment.subblocks.exposure_time_s",
        "subblocks.exposure_time_s",
        "experiment.system.source.exposure_time_s",
        "system.source.exposure_time_s",
        "experiment.source.exposure_time_s",
        "source.exposure_time_s",
    ):
        value = _get_path(translated_config, path)
        if value is not None:
            add_candidate(f"translated_config.{path}", value)

    for path in ("source.exposure_time_s", "system.source.exposure_time_s"):
        value = _get_path(truth_system_cfg, path)
        if value is not None:
            add_candidate(f"truth_system.{path}", value)

    if default is not None:
        add_candidate("fallback.default", default)

    selected = next((item for item in candidates if item["resolved"] is not None), None)
    warnings_out: list[str] = []
    if selected is None:
        raise ValueError("Could not resolve a positive exposure_time_s for the noise review.")

    selected_value = float(selected["resolved"])
    valid = [item for item in candidates if item["resolved"] is not None]
    disagree = [
        item for item in valid
        if not np.isclose(float(item["resolved"]), selected_value, rtol=1e-9, atol=0.0)
    ]
    if disagree:
        warnings_out.append(
            "Noise-review exposure-time candidates disagree; using "
            f"{selected_value:g} s from {selected['path']}."
        )
    if selected["path"] == "fallback.default":
        warnings_out.append(
            f"Noise-review exposure_time_s fell back to explicit default {selected_value:g} s; no config provenance was found."
        )

    return selected_value, {
        "exposure_time_s": selected_value,
        "source_field_path": selected["path"],
        "warning_if_default_used": selected["path"] == "fallback.default",
        "all_candidate_values": candidates,
        "warnings": warnings_out,
    }


def _diff_paths(base: Mapping[str, Any], truth: Mapping[str, Any], inference: Mapping[str, Any], paths: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        bv = _get_path(base, path)
        tv = _get_path(truth, path)
        iv = _get_path(inference, path)
        rows.append({
            "config_path": path,
            "base_value": bv,
            "truth_value": tv,
            "inference_value": iv,
            "status": "matched" if tv == iv else "mismatch",
        })
    return rows


def summarize_detector_config(system_cfg: Mapping[str, Any]) -> dict[str, Any]:
    detector = dict(system_cfg.get("detector", {}) or {})
    spec = detector_spec_for_model(detector.get("model"))
    layers = []
    calibration_paths: list[dict[str, Any]] = []
    for idx, layer in enumerate(detector.get("layers", []) or []):
        item = dict(layer)
        layer_summary = {
            "index": idx,
            "name": item.get("name", f"layer_{idx}"),
            "kind": item.get("kind"),
            "key_parameters": {k: v for k, v in item.items() if k not in {"name", "kind"}},
        }
        layers.append(layer_summary)
        for key in ("dx_path", "dy_path", "prf_path", "flat_path", "dark_path", "read_noise_path"):
            if item.get(key):
                calibration_paths.append({"layer": layer_summary["name"], "kind": layer_summary["kind"], "field": key, "path": item[key]})
    return {
        "model": detector.get("model"),
        "pixel_pitch_m": spec.pixel_pitch_m,
        "array_size": spec.array_size,
        "read_noise": spec.read_noise,
        "dark_current": spec.dark_current,
        "layers": layers,
        "calibration_paths": calibration_paths,
        "has_calibration_maps": bool(calibration_paths),
    }


def load_detector_calibration_maps(system_cfg: Mapping[str, Any]) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    detector = dict(system_cfg.get("detector", {}) or {})
    for layer in detector.get("layers", []) or []:
        if not isinstance(layer, Mapping):
            continue
        name = str(layer.get("name", layer.get("kind", "layer")))
        for key in ("dx_path", "dy_path", "prf_path", "flat_path", "dark_path", "read_noise_path"):
            path = layer.get(key)
            if not path:
                continue
            p = resolve_repo_path(str(path))
            try:
                if p.suffix.lower() == ".npy":
                    arr = np.load(p)
                elif p.suffix.lower() in {".fits", ".fit", ".fts"}:
                    from astropy.io import fits
                    arr = fits.getdata(p)
                else:
                    continue
                maps[f"{name}.{key}"] = np.asarray(arr, dtype=float)
            except Exception:
                continue
    return maps


def summarize_noise_config(config: Mapping[str, Any], system_cfg: Mapping[str, Any]) -> dict[str, Any]:
    exp = _experiment(config)
    sub = dict(exp.get("subblocks", {}) or {})
    original = sub.get("noise", "disabled")
    structured = sub.get("noise_model", {}).get("requested", {}) if isinstance(sub.get("noise_model"), Mapping) else original
    normalized = normalize_noise_request(structured)
    detector_noise = resolve_detector_noise_spec(system_cfg, normalized)
    warnings_out = list(detector_noise.get("warnings", []))
    separate_term_control = (
        sub.get("noise_model", {}).get("separate_term_control")
        if isinstance(sub.get("noise_model"), Mapping)
        else True
    )
    if separate_term_control is False and normalized.get("enabled") and any(
        bool(normalized.get(key)) for key in ("read_noise", "dark_current")
    ):
        warnings_out.append(
            "Structured shot/read/dark-current request is translated through a coarse legacy noise flag; "
            "term-specific runner controls are not fully available in the campaign wrapper."
        )
    use_render_variance = sub.get("use_render_variance", "auto")
    variance_model = "provided_cube" if use_render_variance is True else "data"
    if str(use_render_variance).lower() == "true":
        variance_model = "provided_cube"
    return {
        "noise_request_original": original,
        "noise_request_normalized": normalized,
        "render_noise": {
            "enabled": bool(normalized["enabled"]),
            "shot_noise": bool(detector_noise["shot_noise_enabled"]),
            "read_noise": bool(detector_noise["read_noise_enabled"]),
            "dark_current": bool(detector_noise["dark_current_enabled"]),
            "read_noise_electrons": detector_noise["read_noise_electrons"],
            "read_noise_source": detector_noise["read_noise_source"],
            "dark_current_e_per_s": detector_noise["dark_current_e_per_s"],
            "dark_current_source": detector_noise["dark_current_source"],
            "exposure_time_s": detector_noise["exposure_time_s"],
            "write_variance": bool(normalized.get("write_variance", True)),
            "separate_term_control": separate_term_control,
        },
        "inference_noise_model": {
            "variance_model": variance_model,
            "variance_floor": normalized.get("variance_floor", sub.get("variance_floor")),
            "use_render_variance": use_render_variance,
        },
        "shot_noise_signal_dependent": bool(detector_noise["shot_noise_enabled"]),
        "read_noise_signal_independent": bool(detector_noise["read_noise_enabled"]),
        "dark_current_expected_variance_included": bool(detector_noise["dark_current_enabled"]),
        "warnings": warnings_out,
        "noise_mode": str(original) if not isinstance(original, Mapping) else ("enabled" if normalized["enabled"] else "disabled"),
        "structured_request": structured,
        "shot_noise_enabled": bool(detector_noise["shot_noise_enabled"]),
        "read_noise_enabled": bool(detector_noise["read_noise_enabled"]),
        "read_noise": detector_noise["read_noise_electrons"],
        "read_noise_unit": "electrons RMS per pixel",
        "read_noise_source": detector_noise["read_noise_source"],
        "dark_current_enabled": bool(detector_noise["dark_current_enabled"]),
        "dark_current": detector_noise["dark_current_e_per_s"],
        "dark_current_unit": "electrons / s / pixel",
        "dark_current_source": detector_noise["dark_current_source"],
        "exposure_time_s": detector_noise["exposure_time_s"],
        "variance_floor": normalized.get("variance_floor", sub.get("variance_floor")),
        "use_render_variance": use_render_variance,
        "separate_term_control": separate_term_control,
    }


def noise_demo(seed: int = 123, shape: tuple[int, int] = (16, 16), read_noise: float = 0.5) -> dict[str, Any]:
    yy, xx = np.mgrid[-1:1:complex(shape[0]), -1:1:complex(shape[1])]
    image = 1000.0 * np.exp(-0.5 * (xx * xx + yy * yy) / 0.12**2) + 20.0
    spec = DetectorSpec(model_name="review_demo", read_noise=float(read_noise), dark_current=0.0)
    key = jr.PRNGKey(int(seed))
    shot, shot_var = apply_observation_noise(jnp.asarray(image), noise_cfg={"enabled": True, "photon_noise": True}, rng_key=key)
    read, read_var = apply_observation_noise(jnp.asarray(image), noise_cfg={"enabled": True, "photon_noise": False, "read_noise": True}, rng_key=key, detector_spec=spec)
    combined, combined_var = apply_observation_noise(jnp.asarray(image), noise_cfg={"enabled": True, "photon_noise": True, "read_noise": True}, rng_key=key, detector_spec=spec)
    out = {
        "noiseless": image,
        "shot": np.asarray(shot),
        "read": np.asarray(read),
        "combined": np.asarray(combined),
        "shot_variance": np.asarray(shot_var),
        "read_variance": np.asarray(read_var),
        "combined_variance": np.asarray(combined_var),
        "seed": int(seed),
        "read_noise": float(read_noise),
    }
    out["diagnostics"] = {
        "shot_residual_var": float(np.var(out["shot"] - image)),
        "read_residual_var": float(np.var(out["read"] - image)),
        "combined_residual_var": float(np.var(out["combined"] - image)),
        "mean_model_variance": float(np.mean(out["combined_variance"])),
    }
    return out


def _trajectory_cfg(config: Mapping[str, Any]) -> dict[str, Any]:
    exp = _experiment(config)
    sub = dict(exp.get("subblocks", {}) or {})
    return dict(sub.get("trace_source", {}) or {})


def resolve_subblock_plan_settings(
    config: Mapping[str, Any],
    *,
    strict: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Resolve overlapping subblock/window/iterative settings for review.

    When iterative mode is enabled, ``windows_per_draw * subblocks_per_window``
    is the canonical number of subblocks generated per prior draw.
    ``experiment.subblocks.n_subblocks`` and ``trace_source.window.n_subblocks``
    are optional redundant fields and must agree when present.
    """

    exp = _experiment(config)
    sub = dict(exp.get("subblocks", {}) or {})
    trace = dict(sub.get("trace_source", {}) or {})
    window = dict(trace.get("window", {}) or {})
    iterative = dict(exp.get("iterative", {}) or {})
    warnings_out: list[str] = []
    errors: list[str] = []

    iterative_enabled = bool(iterative.get("enabled", False))
    raw_n_subblocks = sub.get("n_subblocks")
    n_subblocks = None if raw_n_subblocks is None else int(raw_n_subblocks)
    if n_subblocks is not None and n_subblocks < 1:
        errors.append(f"subblocks.n_subblocks={n_subblocks} must be >= 1.")

    trace_window_n = window.get("n_subblocks")
    trace_window_n = None if trace_window_n is None else int(trace_window_n)

    windows_per_draw = iterative.get("windows_per_draw")
    subblocks_per_window = iterative.get("subblocks_per_window")
    expected_iterative_subblocks = None
    subblock_count_source = "experiment.subblocks.n_subblocks"
    if iterative_enabled:
        windows_per_draw = int(windows_per_draw or 1)
        if subblocks_per_window is None:
            if n_subblocks is None:
                errors.append(
                    "iterative.enabled=true requires iterative.subblocks_per_window "
                    "or subblocks.n_subblocks."
                )
                subblocks_per_window = 1
            else:
                subblocks_per_window = int(n_subblocks)
                warnings_out.append(
                    "iterative.subblocks_per_window is omitted; using subblocks.n_subblocks "
                    f"as the fallback value ({subblocks_per_window})."
                )
        else:
            subblocks_per_window = int(subblocks_per_window)
        expected_iterative_subblocks = windows_per_draw * subblocks_per_window
        if windows_per_draw < 1 or subblocks_per_window < 1:
            errors.append("iterative.windows_per_draw and iterative.subblocks_per_window must be >= 1.")
        elif n_subblocks is not None and expected_iterative_subblocks != n_subblocks:
            errors.append(
                "iterative.windows_per_draw * iterative.subblocks_per_window = "
                f"{expected_iterative_subblocks}, but subblocks.n_subblocks = {n_subblocks}. "
                "Remove subblocks.n_subblocks or make it match the iterative grouping."
            )
        else:
            if n_subblocks is None:
                warnings_out.append(
                    "subblocks.n_subblocks is omitted; deriving total subblocks from "
                    f"iterative grouping ({expected_iterative_subblocks})."
                )
            else:
                warnings_out.append(
                    f"iterative.windows_per_draw * iterative.subblocks_per_window = "
                    f"{expected_iterative_subblocks}, matching subblocks.n_subblocks={n_subblocks}."
                )
        resolved_n_subblocks = expected_iterative_subblocks
        subblock_count_source = "experiment.iterative.windows_per_draw*subblocks_per_window"
    else:
        windows_per_draw = None if windows_per_draw is None else int(windows_per_draw)
        subblocks_per_window = None if subblocks_per_window is None else int(subblocks_per_window)
        resolved_n_subblocks = int(n_subblocks or 1)
        if n_subblocks is None:
            warnings_out.append("subblocks.n_subblocks is omitted with iterative disabled; defaulting to 1.")

    if trace_window_n is not None:
        if trace_window_n != resolved_n_subblocks:
            errors.append(
                f"trace_source.window.n_subblocks={trace_window_n} disagrees with "
                f"resolved total subblocks={resolved_n_subblocks}. Remove "
                "trace_source.window.n_subblocks or make it match."
            )
        else:
            warnings_out.append(
                f"trace_source.window.n_subblocks={trace_window_n} is redundant but agrees "
                f"with resolved total subblocks={resolved_n_subblocks}."
            )

    consistency_status = "consistent" if not errors else "inconsistent"
    resolved = {
        "subblocks_n_subblocks": n_subblocks,
        "trace_source_window_n_subblocks": trace_window_n,
        "iterative_windows_per_draw": windows_per_draw,
        "iterative_subblocks_per_window": subblocks_per_window,
        "expected_iterative_subblocks": expected_iterative_subblocks,
        "resolved_n_subblocks": resolved_n_subblocks,
        "resolved_total_subblocks": resolved_n_subblocks,
        "resolved_windows_per_draw": windows_per_draw,
        "resolved_subblocks_per_window": subblocks_per_window,
        "subblock_count_source": subblock_count_source,
        "canonical_source": subblock_count_source,
        "consistency_status": consistency_status,
        "warnings": warnings_out,
        "errors": errors,
        "policy": (
            "When iterative is enabled, iterative windows_per_draw * subblocks_per_window "
            "is canonical; subblocks.n_subblocks and trace_source.window.n_subblocks are "
            "optional redundant fields and must agree when present."
        ),
    }
    if strict and errors:
        raise ValueError("Strict subblock plan validation failed: " + " ".join(errors))
    if errors:
        warnings_out = warnings_out + errors
    return resolved, warnings_out


def _filter_component_labels(kind: str) -> dict[str, str]:
    normalized = str(kind or "none").strip().lower()
    if normalized == "high_pass":
        return {
            "filtered": "high-pass filtered residual",
            "removed": "low-frequency component removed",
            "removed_definition": "removed = raw - filtered = low-frequency trend removed by filter",
        }
    if normalized == "low_pass":
        return {
            "filtered": "low-pass filtered trend",
            "removed": "high-frequency residual removed",
            "removed_definition": "removed = raw - filtered = high-frequency residual removed by filter",
        }
    if normalized == "band_pass":
        return {
            "filtered": "band-pass filtered component",
            "removed": "out-of-band component removed",
            "removed_definition": "removed = raw - filtered = out-of-band component removed by filter",
        }
    return {
        "filtered": "filtered trajectory",
        "removed": "component removed by filter",
        "removed_definition": "removed = raw - filtered",
    }


def trajectory_timing_summary_table(trajectory_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return per-subblock frame timing and fit residual diagnostics."""

    rows: list[dict[str, Any]] = []
    summary = trajectory_review.get("summary", {})
    subblock_duration_s = None
    if isinstance(summary, Mapping) and summary.get("subblock_duration_s") is not None:
        subblock_duration_s = float(summary["subblock_duration_s"])
    for block in trajectory_review.get("blocks", []) or []:
        times = np.asarray(block.frame_times_s, dtype=float)
        diffs = np.diff(times)
        subblock_start = float(times[0])
        subblock_end = (
            subblock_start + subblock_duration_s
            if subblock_duration_s is not None
            else float(times[-1])
        )
        row = {
            "subblock_index": int(block.subblock_index),
            "subblock_start_s": subblock_start,
            "subblock_end_s": subblock_end,
            "n_frames": int(times.size),
            "first_frame_time_s": float(times[0]),
            "last_frame_time_s": float(times[-1]),
            "frame_dt_s": float(np.median(diffs)) if diffs.size else None,
            "frame_span_s": float(times[-1] - times[0]) if times.size else 0.0,
            "fit_model": "linear per subblock",
            "x_rms_residual_as": None,
            "y_rms_residual_as": None,
            "pa_rms_residual_deg": None,
        }
        key_map = {
            "source.x_position_as": "x_rms_residual_as",
            "source.y_position_as": "y_rms_residual_as",
            "source.position_angle_deg": "pa_rms_residual_deg",
        }
        for key, out_key in key_map.items():
            if key in block.diagnostics:
                row[out_key] = float(block.diagnostics[key]["rms_residual"])
        rows.append(row)
    return rows


def trajectory_filter_provenance_table(trajectory_review: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return per-key raw/filtered/removed RMS and filter metadata."""

    if not trajectory_review.get("available"):
        return []
    traj = trajectory_review["trajectory"]
    provenance = dict(traj.filter_provenance or {})
    raw_values = traj.unfiltered_values or traj.values
    filtered_values = traj.values
    keys = list(filtered_values)
    frame_times = np.asarray(trajectory_review.get("frame_times_s", []), dtype=float)
    selected_start = float(frame_times[0]) if frame_times.size else None
    selected_end = float(frame_times[-1]) if frame_times.size else None
    t = np.asarray(traj.time_s, dtype=float)
    selected_mask = np.ones_like(t, dtype=bool)
    if selected_start is not None and selected_end is not None:
        selected_mask = (t >= selected_start) & (t <= selected_end)
    rows = []
    for key in keys:
        raw = np.asarray(raw_values[key], dtype=float)
        filtered = np.asarray(filtered_values[key], dtype=float)
        removed = raw - filtered
        rows.append(
            {
                "key": key,
                "filter_enabled": bool(provenance.get("enabled", False)),
                "filter_kind": provenance.get("kind", "none"),
                "method": provenance.get("method"),
                "order": provenance.get("order"),
                "cutoff_period_s": provenance.get("cutoff_period_s"),
                "cutoff_hz": provenance.get("cutoff_hz"),
                "low_cutoff_period_s": provenance.get("low_cutoff_period_s"),
                "low_cutoff_hz": provenance.get("low_cutoff_hz"),
                "high_cutoff_period_s": provenance.get("high_cutoff_period_s"),
                "high_cutoff_hz": provenance.get("high_cutoff_hz"),
                "raw_rms": float(np.sqrt(np.mean(np.square(raw)))),
                "filtered_rms": float(np.sqrt(np.mean(np.square(filtered)))),
                "removed_rms": float(np.sqrt(np.mean(np.square(removed)))),
                "selected_window_raw_rms": float(np.sqrt(np.mean(np.square(raw[selected_mask])))),
                "selected_window_filtered_rms": float(np.sqrt(np.mean(np.square(filtered[selected_mask])))),
                "removed_component_label": _filter_component_labels(str(provenance.get("kind", "none")))["removed"],
                "removed_component_definition": _filter_component_labels(str(provenance.get("kind", "none")))["removed_definition"],
            }
        )
    return rows


def plot_trajectory_review_components(
    trajectory_review: Mapping[str, Any],
    *,
    keys: Sequence[str] | None = None,
    max_legend_subblocks: int = 8,
) -> list[Any]:
    """Plot raw, filtered, removed, and selected-frame fit panels per key."""

    if not trajectory_review.get("available"):
        return []
    import matplotlib.pyplot as plt

    traj = trajectory_review["trajectory"]
    provenance = dict(traj.filter_provenance or {})
    labels = _filter_component_labels(str(provenance.get("kind", "none")))
    raw_values = traj.unfiltered_values or traj.values
    filtered_values = traj.values
    t = np.asarray(traj.time_s, dtype=float)
    summary = trajectory_review.get("summary", {})
    selected_frame_times = np.asarray(trajectory_review.get("frame_times_s", []), dtype=float)
    selected_start = summary.get("selected_window_start_s") if isinstance(summary, Mapping) else None
    selected_end = summary.get("selected_window_end_s") if isinstance(summary, Mapping) else None
    selected_start = float(selected_start) if selected_start is not None else (float(selected_frame_times[0]) if selected_frame_times.size else None)
    selected_end = float(selected_end) if selected_end is not None else (float(selected_frame_times[-1]) if selected_frame_times.size else None)
    subblock_duration_s = summary.get("subblock_duration_s") if isinstance(summary, Mapping) else None
    subblock_duration_s = float(subblock_duration_s) if subblock_duration_s is not None else None
    blocks = list(trajectory_review.get("blocks", []) or [])
    plot_keys = list(keys or filtered_values.keys())
    figures = []

    for key in plot_keys:
        fig, axes = plt.subplots(1, 4, figsize=(20, 3.8), constrained_layout=True)
        raw = np.asarray(raw_values[key], dtype=float)
        filtered = np.asarray(filtered_values[key], dtype=float)
        removed = raw - filtered
        panel_data = (
            (raw, f"raw trajectory: {key}"),
            (filtered, f"{labels['filtered']}: {key}"),
            (removed, f"{labels['removed']}: {key}"),
        )
        for ax, (series, title) in zip(axes[:3], panel_data):
            ax.plot(t, series, color="tab:blue", linewidth=1.2)
            if selected_start is not None and selected_end is not None:
                ax.axvspan(selected_start, selected_end, color="tab:orange", alpha=0.14, label="selected window")
            for block in blocks:
                block_times = np.asarray(block.frame_times_s, dtype=float)
                block_end = block_times[0] + subblock_duration_s if subblock_duration_s is not None else block_times[-1]
                ax.axvspan(block_times[0], block_end, color="tab:green", alpha=0.10)
                ax.axvline(block_times[0], color="tab:green", alpha=0.35, linewidth=0.8)
                if len(blocks) <= max_legend_subblocks:
                    ax.text(
                        block_times[0],
                        0.98,
                        str(int(block.subblock_index)),
                        transform=ax.get_xaxis_transform(),
                        ha="left",
                        va="top",
                        fontsize=8,
                        color="tab:green",
                    )
            ax.set_title(title)
            ax.set_xlabel("time [s]")
            ax.grid(alpha=0.25)
        axes[2].set_title(f"{labels['removed']}: {key}\n{labels['removed_definition']}")

        ax = axes[3]
        cmap = plt.get_cmap("tab10")
        for idx, block in enumerate(blocks):
            color = cmap(idx % 10)
            times = np.asarray(block.frame_times_s, dtype=float)
            block_end = times[0] + subblock_duration_s if subblock_duration_s is not None else times[-1]
            ax.axvspan(times[0], block_end, color=color, alpha=0.07)
            label_prefix = f"subblock {int(block.subblock_index)}"
            use_label = len(blocks) <= max_legend_subblocks
            ax.plot(
                times,
                np.asarray(block.truth[key], dtype=float),
                linestyle="None",
                marker="o",
                color=color,
                label=f"{label_prefix} truth" if use_label else None,
            )
            ax.plot(
                times,
                np.asarray(block.prediction[key], dtype=float),
                linestyle="--",
                color=color,
                label=f"{label_prefix} linear fit" if use_label else None,
            )
        ax.set_title(f"selected subblock frame samples and linear fits: {key}")
        ax.set_xlabel("time [s]")
        ax.grid(alpha=0.25)
        if len(blocks) <= max_legend_subblocks:
            ax.legend(fontsize=8)
        for ax in axes:
            ax.set_ylabel(key)
        figures.append(fig)
    return figures


def load_trajectory_for_review(config: Mapping[str, Any], *, strict: bool = False) -> dict[str, Any]:
    exp = _experiment(config)
    sub = dict(exp.get("subblocks", {}) or {})
    trace = _trajectory_cfg(config)
    if str(trace.get("mode", "iid_jitter")) != "trajectory":
        return {"mode": str(trace.get("mode", "iid_jitter")), "available": False, "reason": "trace_source.mode is not trajectory"}
    source = dict(trace.get("source", {}) or {})
    window = dict(trace.get("window", {}) or {})
    sampling = dict(trace.get("sampling", {}) or {})
    processing = dict(trace.get("processing", {}) or {})
    legacy_processing = dict(sub.get("trajectory_processing", {}) or {})
    filter_cfg = dict(
        processing.get("filter")
        or legacy_processing.get("filter")
        or legacy_processing.get("high_pass_filter")
        or {}
    )
    path = resolve_repo_path(source.get("path", "src/dluxshera/data/airbus_data/Thirty_Min_Observation_Window.csv"))
    subblock_plan, subblock_warnings = resolve_subblock_plan_settings(config, strict=strict)
    n_subblocks = int(subblock_plan["resolved_n_subblocks"])
    start_s = float(window.get("start_s", 0.0))
    frame_dt_s = float(sampling.get("frame_dt_s", sub.get("exposure_time_s", 0.05)))
    subblock_duration_s = float(
        sampling.get(
            "subblock_duration_s",
            exp.get("forecast", {}).get("subblock_duration_s", 1.0)
            if isinstance(exp.get("forecast"), Mapping)
            else 1.0,
        )
    )
    n_frames_per_subblock = int(sampling.get("n_frames_per_subblock", sub.get("n_frames", 3)))
    trajectory, frame_times, blocks = prepare_airbus_subblocks(
        path=path,
        start_s=start_s,
        duration_s=window.get("duration_s"),
        n_subblocks=n_subblocks,
        sample_dt_s=float(source.get("sample_dt_s", 0.1)),
        frame_dt_s=frame_dt_s,
        subblock_duration_s=subblock_duration_s,
        n_frames_per_subblock=n_frames_per_subblock,
        output_keys=trace.get("output_keys", DEFAULT_OUTPUT_KEYS),
        fit_keys=dict(trace.get("starting_guess", {}) or {}).get("fit_keys"),
        interpolation=str(sampling.get("interpolation", "linear")),
        filter_config=filter_cfg,
    )
    return {
        "available": True,
        "mode": "trajectory",
        "path": str(path),
        "trajectory": trajectory,
        "frame_times_s": frame_times,
        "blocks": blocks,
        "summary": {
            "raw_span_s": list(trajectory.raw.span),
            "raw_sample_count": trajectory.raw.sample_count,
            "selected_start_s": float(frame_times[0]),
            "selected_end_s": float(frame_times[-1]),
            "selected_window_start_s": start_s,
            "selected_window_end_s": start_s + n_subblocks * subblock_duration_s,
            "frame_dt_s": frame_dt_s,
            "subblock_duration_s": subblock_duration_s,
            "n_frames_per_subblock": n_frames_per_subblock,
            "n_subblocks": len(blocks),
            "n_frames": int(frame_times.size),
            "output_keys": list(trajectory.values),
            "filter": dict(trajectory.filter_provenance or {}),
        },
        "subblock_plan": subblock_plan,
        "warnings": subblock_warnings,
    }


def make_high_pass_trajectory_diagnostic(trajectory_review: Mapping[str, Any], *, timescale_s: float = 15.0) -> dict[str, Any]:
    if not trajectory_review.get("available"):
        return {"available": False, "reason": trajectory_review.get("reason", "trajectory unavailable")}
    traj = trajectory_review["trajectory"]
    t = np.asarray(traj.time_s, dtype=float)
    dt = float(np.median(np.diff(t)))
    window = max(3, int(round(float(timescale_s) / dt)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    out = {"available": True, "timescale_s": float(timescale_s), "window_samples": int(window), "series": {}}
    for key, values in traj.values.items():
        arr = np.asarray(values, dtype=float)
        padded = np.pad(arr, (window // 2, window // 2), mode="edge")
        low = np.convolve(padded, kernel, mode="valid")
        high = arr - low
        out["series"][key] = {"raw": arr, "low_pass": low, "high_pass": high, "rms_high_pass": float(np.sqrt(np.mean(high * high)))}
    out["note"] = "Moving-average diagnostic only; configured production filtering is reported separately from trajectory.filter_provenance."
    return out


def compare_trace_jitter_enabled_disabled(config: Mapping[str, Any]) -> dict[str, Any]:
    """Report trace-jitter behavior in trajectory mode from implementation wiring."""

    exp = _experiment(config)
    sub = dict(exp.get("subblocks", {}) or {})
    trace = dict(sub.get("trace_source", {}) or {})
    jitter = dict(sub.get("trace_jitter", {}) or {})
    if str(trace.get("mode", "iid_jitter")) == "trajectory":
        return {
            "status": "downstream_template_override",
            "is_additive_to_materialized_trajectory_csv": False,
            "is_ignored": False,
            "jitter_config": jitter,
            "rms_difference": {"source.x_position_as": 0.0, "source.y_position_as": 0.0, "source.position_angle_deg": 0.0},
            "conclusion": (
                "Trajectory frame_truth.csv is generated without trace_jitter. The observation-bias wrapper "
                "forwards trace_jitter as run_obs_subblock_study CLI overrides, where it only changes an "
                "existing iid_jitter/random_walk effect in the trace template. In trajectory mode with external "
                "frame truth, this is downstream behavior and not visible in the materialized trajectory CSV."
            ),
        }
    return {
        "status": "legacy_iid_trace_mode",
        "is_additive_to_materialized_trajectory_csv": None,
        "is_ignored": False,
        "jitter_config": jitter,
        "rms_difference": {},
        "conclusion": "Non-trajectory trace mode preserves legacy trace-template jitter behavior.",
    }


def _render_truth_system_image(system_cfg: Mapping[str, Any], *, psf_npix: int | None = None) -> np.ndarray:
    render_cfg = copy.deepcopy(dict(system_cfg))
    if psf_npix is not None:
        _set_path(render_cfg, "optics.psf_npix", int(psf_npix))
    spec = compose_forward_spec(render_cfg)
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    binder = SheraBinder(render_cfg, spec, store)
    image = np.asarray(binder.model(), dtype=float)
    if image.ndim > 2:
        image = np.asarray(image).reshape((-1, *image.shape[-2:])).sum(axis=0)
    return image


def render_noise_review_images(
    config: Mapping[str, Any],
    truth_system_cfg: Mapping[str, Any],
    *,
    seed: int = 123,
    review_cfg: Mapping[str, Any] | None = None,
    min_psf_npix: int = 160,
    default_psf_npix: int = 256,
    display_crop_npix: int | None = None,
    exposure_time_s: float | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Render the resolved truth system at full review size and audit noise terms."""

    exp = _experiment(config)
    sub = dict(exp.get("subblocks", {}) or {})
    noise_request = sub.get("noise_model", {}).get("requested", sub.get("noise", "disabled")) if isinstance(sub.get("noise_model"), Mapping) else sub.get("noise", "disabled")
    normalized = normalize_noise_request(noise_request)
    detector_noise = resolve_detector_noise_spec(truth_system_cfg, normalized, strict=strict)
    warnings_out = list(detector_noise.get("warnings", []))
    rendered_psf_npix, psf_provenance = resolve_review_psf_npix(
        truth_system_cfg,
        review_cfg,
        minimum=min_psf_npix,
        default=default_psf_npix,
    )
    resolved_exposure, exposure_provenance = resolve_noise_review_exposure_time_s(
        config,
        truth_system_cfg,
        default=exposure_time_s,
    )
    warnings_out.extend(psf_provenance.get("warnings", []))
    warnings_out.extend(exposure_provenance.get("warnings", []))
    detector_exposure = _finite_positive_float(detector_noise.get("exposure_time_s"))
    if detector_exposure is not None and not np.isclose(detector_exposure, resolved_exposure, rtol=1e-9, atol=0.0):
        warnings_out.append(
            f"Detector-noise exposure_time_s={detector_exposure:g} differs from resolved render exposure_time_s={resolved_exposure:g}."
        )
    detector_noise = dict(detector_noise)
    detector_noise["exposure_time_s"] = resolved_exposure

    noiseless = _render_truth_system_image(truth_system_cfg, psf_npix=rendered_psf_npix)

    detector_spec = DetectorSpec(
        model_name=str(detector_noise.get("detector_model")),
        read_noise=detector_noise.get("read_noise_electrons"),
        dark_current=detector_noise.get("dark_current_e_per_s"),
    )
    key = jr.PRNGKey(int(seed))
    noisy, render_variance = apply_observation_noise(
        jnp.asarray(noiseless),
        noise_cfg=normalized,
        rng_key=key,
        detector_spec=detector_spec,
        exposure_time_s=resolved_exposure,
    )
    expected_variance = expected_noise_variance(
        jnp.asarray(noiseless),
        noise_cfg=normalized,
        detector_noise=detector_noise,
        exposure_time_s=resolved_exposure,
    )
    noisy_np = np.asarray(noisy)
    render_variance_np = np.asarray(render_variance)
    expected_variance_np = np.asarray(expected_variance)
    residual = noisy_np - noiseless
    denom = np.sqrt(np.maximum(expected_variance_np, 1.0e-12))
    normalized_residual = residual / denom
    if bool(normalized.get("write_variance", True)) and not np.any(np.isfinite(render_variance_np)):
        warnings_out.append("write_variance=true but no finite variance map was produced by the render helper.")

    diagnostics = {
        "mean_expected_variance": float(np.mean(expected_variance_np)),
        "mean_render_variance": float(np.mean(render_variance_np)),
        "residual_variance": float(np.var(residual)),
        "normalized_residual_std": float(np.std(normalized_residual)),
        "render_expected_variance_max_abs_diff": float(np.max(np.abs(render_variance_np - expected_variance_np))),
        "model_image_sum": float(np.sum(noiseless)),
        "model_image_peak": float(np.max(noiseless)),
        "expected_photon_variance_peak": float(np.max(np.maximum(noiseless, 0.0))) if normalized.get("shot_noise") else 0.0,
        "read_noise_electrons_rms": detector_noise["read_noise_electrons"],
        "dark_current_e_per_s_per_pix": detector_noise["dark_current_e_per_s"],
        "expected_dark_electrons_per_pix": (
            float(detector_noise["dark_current_e_per_s"]) * float(resolved_exposure)
            if detector_noise.get("dark_current_e_per_s") is not None
            else None
        ),
        "render_exposure_time_s": float(resolved_exposure),
        "expected_variance_exposure_time_s": float(resolved_exposure),
    }
    if diagnostics["model_image_peak"] <= 0.0 and normalized.get("shot_noise"):
        warnings_out.append("Shot noise is enabled but the rendered model image has non-positive peak counts.")

    display = {
        "noiseless": _center_crop_array(noiseless, display_crop_npix),
        "configured_noisy": _center_crop_array(noisy_np, display_crop_npix),
        "noise_residual": _center_crop_array(residual, display_crop_npix),
        "expected_variance": _center_crop_array(expected_variance_np, display_crop_npix),
        "render_variance": _center_crop_array(render_variance_np, display_crop_npix),
        "normalized_residual": _center_crop_array(normalized_residual, display_crop_npix),
    }
    crop_shape = tuple(int(v) for v in display["noiseless"].shape[-2:])
    return {
        "available": True,
        "source": "resolved_truth_system_binder",
        "seed": int(seed),
        "noise_request_original": noise_request,
        "noise_request_normalized": normalized,
        "render_noise": {
            "enabled": bool(normalized["enabled"]),
            "shot_noise": bool(detector_noise["shot_noise_enabled"]),
            "read_noise": bool(detector_noise["read_noise_enabled"]),
            "dark_current": bool(detector_noise["dark_current_enabled"]),
            "read_noise_electrons": detector_noise["read_noise_electrons"],
            "read_noise_source": detector_noise["read_noise_source"],
            "dark_current_e_per_s": detector_noise["dark_current_e_per_s"],
            "dark_current_source": detector_noise["dark_current_source"],
            "exposure_time_s": resolved_exposure,
            "exposure_time_s_source": exposure_provenance["source_field_path"],
            "write_variance": bool(normalized.get("write_variance", True)),
            "rendered_psf_npix": int(rendered_psf_npix),
            "displayed_crop_npix": None if display_crop_npix is None else int(display_crop_npix),
            "render_shape": tuple(int(v) for v in noiseless.shape[-2:]),
            "display_shape": crop_shape,
        },
        "psf_npix_provenance": psf_provenance,
        "exposure_time_s_provenance": exposure_provenance,
        "variance_diagnostics": diagnostics,
        "warnings": warnings_out,
        "noiseless": noiseless,
        "configured_noisy": noisy_np,
        "noise_residual": residual,
        "expected_variance": expected_variance_np,
        "render_variance": render_variance_np,
        "normalized_residual": normalized_residual,
        "display": display,
        "render_shape": tuple(int(v) for v in noiseless.shape[-2:]),
        "display_shape": crop_shape,
    }


def render_tiny_review_images(
    config: Mapping[str, Any],
    truth_system_cfg: Mapping[str, Any],
    *,
    seed: int = 123,
    crop_npix: int | None = 64,
    strict: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper for the old diagnostic crop helper."""

    result = render_noise_review_images(
        config,
        truth_system_cfg,
        seed=seed,
        display_crop_npix=crop_npix,
        strict=strict,
    )
    result["warnings"] = list(result.get("warnings", [])) + [
        "render_tiny_review_images is a diagnostic compatibility wrapper; use render_noise_review_images for the main review."
    ]
    return result


def write_review_artifacts(
    outdir: str | Path,
    *,
    base_system_cfg: Mapping[str, Any],
    truth_system_cfg: Mapping[str, Any],
    inference_system_cfg: Mapping[str, Any],
    model_split: CampaignModelSplit,
    spectral_summary: Mapping[str, Any] | None = None,
    wfe_summary: Mapping[str, Any] | None = None,
    detector_summary: Mapping[str, Any] | None = None,
    noise_summary: Mapping[str, Any] | None = None,
    trajectory_summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def write_payload(name: str, payload: Any) -> None:
        path = root / name
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml

                path.write_text(yaml.safe_dump(_jsonable(payload), sort_keys=False), encoding="utf-8")
            except Exception:
                path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        else:
            path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        written[name] = str(path)

    write_payload("resolved_base_system.yaml", base_system_cfg)
    write_payload("resolved_truth_system.yaml", truth_system_cfg)
    write_payload("resolved_inference_system.yaml", inference_system_cfg)
    write_payload("model_split_summary.json", summarize_campaign_model_split(model_split))
    if spectral_summary is not None:
        write_payload("spectral_review_summary.json", spectral_summary)
    if wfe_summary is not None:
        write_payload("wfe_review_summary.json", _strip_arrays(wfe_summary))
    if detector_summary is not None:
        write_payload("detector_layer_summary.json", detector_summary)
    if noise_summary is not None:
        write_payload("noise_review_summary.json", noise_summary)
        if isinstance(noise_summary, Mapping):
            write_payload("noise_request_normalized.json", noise_summary.get("noise_request_normalized", {}))
            write_payload("noise_render_provenance.json", noise_summary.get("render_noise", {}))
            write_payload("noise_inference_provenance.json", noise_summary.get("inference_noise_model", {}))
            if noise_summary.get("variance_diagnostics") is not None:
                write_payload("noise_variance_summary.json", noise_summary.get("variance_diagnostics", {}))
    if trajectory_summary is not None:
        write_payload("trajectory_review_summary.json", _strip_arrays(trajectory_summary))
    notes = root / "config_review_notes.md"
    notes.write_text("# Config review notes\n\n- Reviewer decision:\n- Follow-up changes:\n", encoding="utf-8")
    written["config_review_notes.md"] = str(notes)
    return written


def write_spectral_review_csv(path: str | Path, tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    rows = [dict(row) for role_rows in tables.values() for row in role_rows]
    fieldnames = ["role", "component", "index", "wavelength_nm", "weight"]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return str(p)


def response_curve_review(spectral_cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    spectral_cfg = spectral_cfg or {}
    truth = spectral_cfg.get("truth", {}) if isinstance(spectral_cfg.get("truth"), Mapping) else {}
    comps = truth.get("components", {}) if isinstance(truth.get("components"), Mapping) else {}
    out: dict[str, Any] = {}
    for label, default_path in (("detector_qe", DEFAULT_DETECTOR_QE_PATH), ("m2_filter_response", DEFAULT_FILTER_RESPONSE_PATH)):
        cfg = comps.get(label, {}) if isinstance(comps.get(label), Mapping) else {}
        enabled = bool(cfg.get("enabled", True)) if cfg else False
        path = cfg.get("path", default_path)
        default_wavelength_column = "Wavelength (nm)"
        default_response_column = "QE" if label == "detector_qe" else "T (%)"
        default_response_scale = 1.0 if label == "detector_qe" else 0.01
        try:
            wave, resp = load_response_curve_csv(
                path,
                wavelength_column=str(cfg.get("wavelength_column", default_wavelength_column)),
                response_column=str(cfg.get("response_column", default_response_column)),
                wavelength_unit=str(cfg.get("wavelength_unit", "nm")),
                response_scale=float(cfg.get("response_scale", default_response_scale)),
            )
            available = True
            error = None
        except Exception as exc:
            wave = np.asarray([])
            resp = np.asarray([])
            available = False
            error = str(exc)
        out[label] = {"enabled": enabled, "available": available, "path": str(path), "wavelengths_nm": wave * 1e9, "response": resp, "error": error}
    return out


def summary_dashboard(
    *,
    config: Mapping[str, Any],
    base_cfg: Mapping[str, Any],
    truth_cfg: Mapping[str, Any],
    inference_cfg: Mapping[str, Any],
    model_split: CampaignModelSplit,
    trajectory_review: Mapping[str, Any],
    trace_jitter_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    exp = _experiment(config)
    spectral = summarize_spectral_deck(model_split)
    detector_truth = summarize_detector_config(truth_cfg)
    detector_inf = summarize_detector_config(inference_cfg)
    noise = summarize_noise_config(config, truth_cfg)
    return [
        {"Component": "source target / component SEDs", "Status": "review", "Truth setting": source_block(truth_cfg).get("target"), "Inference setting": source_block(inference_cfg).get("target"), "Difference / mismatch": spectral.get("comparison", {}), "Reviewer action": "inspect SED/weights"},
        {"Component": "spectral grid", "Status": "mismatch" if source_block(truth_cfg).get("n_lambda") != source_block(inference_cfg).get("n_lambda") else "matched", "Truth setting": source_block(truth_cfg).get("n_lambda"), "Inference setting": source_block(inference_cfg).get("n_lambda"), "Difference / mismatch": "fast clamp/reference band", "Reviewer action": "confirm acceptable"},
        {"Component": "QE", "Status": "configured", "Truth setting": exp.get("spectral_model", {}).get("truth", {}).get("components", {}).get("detector_qe", {}), "Inference setting": exp.get("spectral_model", {}).get("inference", {}).get("components", {}).get("detector_qe", {}), "Difference / mismatch": "see response review", "Reviewer action": "inspect"},
        {"Component": "M2 filter", "Status": "configured", "Truth setting": exp.get("spectral_model", {}).get("truth", {}).get("components", {}).get("m2_filter_response", {}), "Inference setting": exp.get("spectral_model", {}).get("inference", {}).get("components", {}).get("m2_filter_response", {}), "Difference / mismatch": "see response review", "Reviewer action": "inspect"},
        {"Component": "flux parameters", "Status": "preserved", "Truth setting": {k: source_block(truth_cfg).get(k) for k in ("log_flux_total", "contrast")}, "Inference setting": {k: source_block(inference_cfg).get(k) for k in ("log_flux_total", "contrast")}, "Difference / mismatch": "scalar band-integrated parameters", "Reviewer action": "verify"},
        {"Component": "high-order WFE maps", "Status": "enabled" if model_split.enabled_components.get("high_order_wfe", {}).get("enabled") else "disabled", "Truth setting": "truth maps", "Inference setting": "truth + knowledge error", "Difference / mismatch": model_split.enabled_components.get("high_order_wfe", {}), "Reviewer action": "inspect RMS/maps"},
        {"Component": "low-order Zernike coefficients", "Status": "review", "Truth setting": summarize_optics_config(truth_cfg).get("primary_noll_indices"), "Inference setting": summarize_optics_config(inference_cfg).get("primary_noll_indices"), "Difference / mismatch": "active index mapping starts at array index 0", "Reviewer action": "verify mapping"},
        {"Component": "optics preset", "Status": "review", "Truth setting": summarize_optics_config(truth_cfg), "Inference setting": summarize_optics_config(inference_cfg), "Difference / mismatch": "see diff table", "Reviewer action": "decide if new preset needed"},
        {"Component": "detector layers", "Status": "matched" if detector_truth["layers"] == detector_inf["layers"] else "mismatch", "Truth setting": detector_truth["layers"], "Inference setting": detector_inf["layers"], "Difference / mismatch": "see detector maps", "Reviewer action": "inspect"},
        {"Component": "calibration maps", "Status": "present" if detector_truth["has_calibration_maps"] else "absent", "Truth setting": detector_truth["calibration_paths"], "Inference setting": detector_inf["calibration_paths"], "Difference / mismatch": "matched config paths", "Reviewer action": "inspect maps if present"},
        {"Component": "noise", "Status": noise["noise_mode"], "Truth setting": noise, "Inference setting": "variance model in subblock", "Difference / mismatch": "demo only when disabled", "Reviewer action": "decide noise mode"},
        {"Component": "trajectory", "Status": "available" if trajectory_review.get("available") else "unavailable", "Truth setting": trajectory_review.get("summary", {}), "Inference setting": "per-subblock linear fit", "Difference / mismatch": "residuals by frame", "Reviewer action": "inspect segment"},
        {"Component": "high-pass filter", "Status": "diagnostic_only", "Truth setting": "not production-enabled", "Inference setting": "not production-enabled", "Difference / mismatch": "15 s notebook diagnostic", "Reviewer action": "decide follow-up"},
        {"Component": "trace jitter", "Status": trace_jitter_review.get("status"), "Truth setting": trace_jitter_review.get("jitter_config"), "Inference setting": "downstream trace template override", "Difference / mismatch": trace_jitter_review.get("conclusion"), "Reviewer action": "document/rename if needed"},
        {"Component": "smear", "Status": str(model_split.enabled_components.get("trajectory_smear", {}).get("mode")), "Truth setting": exp.get("subblocks", {}).get("trajectory_processing", {}).get("smear", {}), "Inference setting": "metadata sidecars", "Difference / mismatch": "no render layer for metadata_only", "Reviewer action": "inspect sidecars"},
        {"Component": "observation theta", "Status": "configured", "Truth setting": exp.get("observation_theta"), "Inference setting": exp.get("observation_theta"), "Difference / mismatch": "shared layout", "Reviewer action": "review active state"},
        {"Component": "prior draws", "Status": "configured", "Truth setting": exp.get("prior_draws"), "Inference setting": exp.get("prior_draws"), "Difference / mismatch": "initialization only", "Reviewer action": "inspect sigmas"},
        {"Component": "iterative update settings", "Status": "configured", "Truth setting": exp.get("iterative"), "Inference setting": exp.get("iterative"), "Difference / mismatch": "not executed in notebook", "Reviewer action": "inspect before campaign"},
    ]


def _strip_arrays(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "min": float(np.min(value)) if value.size else None, "max": float(np.max(value)) if value.size else None}
    if isinstance(value, Mapping):
        return {str(k): _strip_arrays(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_arrays(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


__all__ = [
    "DEFAULT_OUTPUT_ROOT", "DEFAULT_REVIEW_CONFIG", "DEFAULT_SMOKE_CONFIG", "LOW_ORDER_NOLL_INDICES",
    "build_model_split_from_smoke", "compare_trace_jitter_enabled_disabled",
    "cmap_with_bad", "extract_spectral_arrays", "load_detector_calibration_maps", "load_smoke_config",
    "load_trajectory_for_review", "make_high_pass_trajectory_diagnostic", "noise_demo",
    "masked_for_imshow", "optics_diff_table", "plot_trajectory_review_components", "preserve_flux_review",
    "render_noise_review_images", "render_tiny_review_images", "repo_root", "resolve_noise_review_exposure_time_s",
    "resolve_repo_path", "resolve_review_psf_npix", "resolve_subblock_plan_settings", "response_curve_review",
    "spectral_review_tables", "trajectory_filter_provenance_table", "trajectory_timing_summary_table",
    "summarize_detector_config", "summarize_noise_config", "summarize_optics_config",
    "summarize_source_config", "summarize_spectral_deck", "summarize_wfe_artifacts", "symmetric_nan_limits",
    "summary_dashboard", "translate_smoke_to_observation_bias", "write_review_artifacts",
    "write_spectral_review_csv",
]
