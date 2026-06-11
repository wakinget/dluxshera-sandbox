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
from dluxshera.systems.base import compose_forward_spec
from dluxshera.utils.campaign_model_split import (
    CampaignModelSplit,
    build_campaign_model_split,
    summarize_campaign_model_split,
)
from dluxshera.utils.high_order_wfe import (
    fit_zernike_coefficients_nm,
    make_pupil_mask,
    remove_zernike_modes,
)
from dluxshera.utils.noise import apply_observation_noise
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
        system_seed = {"preset": exp.get("system_preset", "SHERA_FLIGHT_3P")}
    user_cfg = {"system": copy.deepcopy(dict(system_seed))}
    exposure = _subblock_exposure_time_s(exp)
    if exposure is not None:
        source = dict(user_cfg["system"].get("source", {}) or {})
        source["exposure_time_s"] = exposure
        user_cfg["system"]["source"] = source
    resolved = resolve_config(user_cfg)
    system_cfg = dict(resolved["system"])
    spec = compose_forward_spec(system_cfg)
    store = ParameterStore.from_spec_defaults(spec).refresh_derived(spec)
    provenance = {
        "system_preset": system_cfg.get("preset"),
        "source_kind": (system_cfg.get("source") or {}).get("kind"),
        "source_target": (system_cfg.get("source") or {}).get("target"),
        "optics_kind": (system_cfg.get("optics") or {}).get("kind"),
        "detector_model": (system_cfg.get("detector") or {}).get("model"),
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
    for mirror in ("primary", "secondary"):
        truth, _ = _mirror_wfe_maps(model_split.truth_system_cfg, mirror)
        ref_truth, err = _mirror_wfe_maps(model_split.inference_system_cfg, mirror)
        if truth is None or ref_truth is None:
            out["warnings"].append(f"{mirror} high-order WFE map is absent.")
            continue
        error = np.zeros_like(truth) if err is None else np.asarray(err, dtype=float)
        inference = ref_truth + error
        mask = make_pupil_mask(truth.shape, mode=str(prov.get("mask_policy", "circular_fallback")))
        coeff_truth = fit_zernike_coefficients_nm(truth, list(noll_indices), mask=mask)
        coeff_inference = fit_zernike_coefficients_nm(inference, list(noll_indices), mask=mask)
        coeff_error = fit_zernike_coefficients_nm(inference - truth, list(noll_indices), mask=mask)
        residual_truth, _ = remove_zernike_modes(truth, list(noll_indices), mask=mask)
        residual_error, _ = remove_zernike_modes(inference - truth, list(noll_indices), mask=mask)
        measured_truth = _masked_stats(truth, mask)
        measured_error = _masked_stats(inference - truth, mask)
        out["mirrors"][mirror] = {
            "truth_opd_nm": truth,
            "inference_opd_nm": inference,
            "knowledge_error_opd_nm": inference - truth,
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
        }
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


def detector_spec_for_model(model: str | None) -> DetectorSpec:
    return {"GSENSE2020BSI": GSENSE2020BSI_SPEC, "HWK4123": HWK4123_SPEC}.get(str(model), GSENSE2020BSI_SPEC)


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
    mode = str(sub.get("noise", "disabled"))
    structured = sub.get("noise_model", {}).get("requested", {}) if isinstance(sub.get("noise_model"), Mapping) else {}
    if not structured and isinstance(sub.get("noise"), Mapping):
        structured = dict(sub["noise"])
    detector = summarize_detector_config(system_cfg)
    return {
        "noise_mode": mode,
        "structured_request": structured,
        "shot_noise_enabled": bool(structured.get("shot_noise", mode not in {"disabled", "none", "off"})),
        "read_noise_enabled": bool(structured.get("read_noise", mode in {"read", "read_noise", "shot_read", "photon_read"})),
        "read_noise": detector["read_noise"],
        "read_noise_unit": "electrons RMS per pixel",
        "dark_current_enabled": bool(structured.get("dark_current", False)),
        "dark_current": detector["dark_current"],
        "dark_current_unit": "electrons / s / pixel",
        "variance_floor": sub.get("variance_floor"),
        "use_render_variance": sub.get("use_render_variance", "inherited/default"),
        "separate_term_control": (
            sub.get("noise_model", {}).get("separate_term_control")
            if isinstance(sub.get("noise_model"), Mapping)
            else None
        ),
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


def load_trajectory_for_review(config: Mapping[str, Any]) -> dict[str, Any]:
    exp = _experiment(config)
    sub = dict(exp.get("subblocks", {}) or {})
    trace = _trajectory_cfg(config)
    if str(trace.get("mode", "iid_jitter")) != "trajectory":
        return {"mode": str(trace.get("mode", "iid_jitter")), "available": False, "reason": "trace_source.mode is not trajectory"}
    source = dict(trace.get("source", {}) or {})
    window = dict(trace.get("window", {}) or {})
    sampling = dict(trace.get("sampling", {}) or {})
    path = resolve_repo_path(source.get("path", "src/dluxshera/data/airbus_data/Thirty_Min_Observation_Window.csv"))
    n_subblocks = int(window.get("n_subblocks", sub.get("n_subblocks", 1)))
    trajectory, frame_times, blocks = prepare_airbus_subblocks(
        path=path,
        start_s=float(window.get("start_s", 0.0)),
        duration_s=window.get("duration_s"),
        n_subblocks=n_subblocks,
        sample_dt_s=float(source.get("sample_dt_s", 0.1)),
        frame_dt_s=float(sampling.get("frame_dt_s", sub.get("exposure_time_s", 0.05))),
        subblock_duration_s=float(sampling.get("subblock_duration_s", exp.get("forecast", {}).get("subblock_duration_s", 1.0) if isinstance(exp.get("forecast"), Mapping) else 1.0)),
        n_frames_per_subblock=int(sampling.get("n_frames_per_subblock", sub.get("n_frames", 3))),
        output_keys=trace.get("output_keys", DEFAULT_OUTPUT_KEYS),
        fit_keys=dict(trace.get("starting_guess", {}) or {}).get("fit_keys"),
        interpolation=str(sampling.get("interpolation", "linear")),
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
            "n_subblocks": len(blocks),
            "n_frames": int(frame_times.size),
            "output_keys": list(trajectory.values),
        },
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
    out["note"] = "Notebook diagnostic only; campaign trace source currently rejects high_pass_filter.enabled=true."
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


def render_tiny_review_images(*_: Any, **__: Any) -> dict[str, Any]:
    """Optional rendering hook placeholder for the notebook.

    Runtime rendering can be expensive and depends on interactive environment
    choices, so the notebook can call this and show a clear unsupported status
    unless Dylan chooses to add an explicit renderer path.
    """

    return {"available": False, "reason": "Tiny rendering is intentionally optional; no production campaign is launched by this helper."}


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
    "extract_spectral_arrays", "load_detector_calibration_maps", "load_smoke_config",
    "load_trajectory_for_review", "make_high_pass_trajectory_diagnostic", "noise_demo",
    "optics_diff_table", "preserve_flux_review", "render_tiny_review_images",
    "repo_root", "resolve_repo_path", "response_curve_review", "spectral_review_tables",
    "summarize_detector_config", "summarize_noise_config", "summarize_optics_config",
    "summarize_source_config", "summarize_spectral_deck", "summarize_wfe_artifacts",
    "summary_dashboard", "translate_smoke_to_observation_bias", "write_review_artifacts",
    "write_spectral_review_csv",
]
