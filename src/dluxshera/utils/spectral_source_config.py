"""Apply spectral-throughput deck outputs to source/system configs.

This module is the narrow bridge between source-level effective spectra and the
existing dLuxShera system configuration schema. It does not render images or add
spectral shape to the active inference state.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from .spectral_response import EffectiveSpectrum, SourceSpectralDeck, SpectralDeck

__all__ = [
    "spectrum_to_source_spectral_config",
    "apply_effective_spectrum_to_source_config",
    "apply_spectral_deck_to_system_configs",
    "apply_source_spectral_deck_to_system_configs",
    "build_spectral_truth_inference_system_configs",
]

BINARY_SOURCE_KINDS = {"binary", "binary_target", "alpha_cen"}
SINGLE_SOURCE_KINDS = {"single_star"}


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _validate_spectrum(spectrum: EffectiveSpectrum) -> tuple[np.ndarray, np.ndarray]:
    wavelengths = np.asarray(spectrum.wavelengths_m, dtype=float).reshape(-1)
    weights = np.asarray(spectrum.weights, dtype=float).reshape(-1)
    if wavelengths.size == 0:
        raise ValueError("EffectiveSpectrum wavelengths_m must contain at least one sample.")
    if wavelengths.shape != weights.shape:
        raise ValueError("EffectiveSpectrum wavelengths_m and weights must have identical shapes.")
    if np.any(wavelengths <= 0.0):
        raise ValueError("EffectiveSpectrum wavelengths_m must be strictly positive.")
    if wavelengths.size > 1 and np.any(np.diff(wavelengths) <= 0.0):
        raise ValueError("EffectiveSpectrum wavelengths_m must be strictly increasing.")
    if np.any(weights < 0.0):
        raise ValueError("EffectiveSpectrum weights must be non-negative.")
    total = float(np.sum(weights))
    if not total > 0.0:
        raise ValueError("EffectiveSpectrum weights must have a positive sum.")
    return wavelengths, weights / total


def _source_kind(source_cfg: Mapping[str, Any]) -> str:
    return str(source_cfg.get("kind", "binary_target")).lower()


def _source_component_labels(source_cfg: Mapping[str, Any], *, source_kind: str) -> tuple[str, ...]:
    labels = source_cfg.get("component_labels")
    if labels is not None:
        parsed = tuple(str(label) for label in labels)
        if parsed:
            return parsed
    if source_kind in SINGLE_SOURCE_KINDS:
        return ("star",)
    return ("primary", "secondary")


def _wavelength_summary(wavelengths_m: np.ndarray) -> tuple[float, float, int]:
    wavelength_m = float(0.5 * (wavelengths_m[0] + wavelengths_m[-1]))
    bandwidth_m = float(wavelengths_m[-1] - wavelengths_m[0]) if wavelengths_m.size > 1 else 0.0
    return wavelength_m, bandwidth_m, int(wavelengths_m.size)


def _spectrum_provenance(spectrum: EffectiveSpectrum, *, source_kind: str, component_labels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "spectrum_label": spectrum.label,
        "source_kind": source_kind,
        "component_labels": list(component_labels),
        "n_lambda": int(spectrum.diagnostics.get("n_lambda", len(spectrum.weights))),
        "flux_factor": float(spectrum.flux_factor),
        "lambda_eff_nm": float(spectrum.diagnostics.get("lambda_eff_nm", np.nan)),
        "bandwidth_rms_nm": float(spectrum.diagnostics.get("bandwidth_rms_nm", np.nan)),
        "weights_sum": float(np.sum(spectrum.weights)),
        "weight_normalization": "sample_sum",
        "wavelength_unit": "m",
        "flux_factor_usage": "diagnostic_provenance_only",
        "log_flux_total_and_contrast": "preserved_detected_post_response_band_integrated",
        "spectral_shape_active_inference_parameter": False,
        "builder_note": (
            "generic binary and single_star builders consume explicit wavelengths_m; "
            "binary_target/alpha_cen builders consume linear bandpass endpoints plus component_weights"
        ),
    }


def spectrum_to_source_spectral_config(
    spectrum: EffectiveSpectrum,
    *,
    source_kind: str = "binary_target",
    component_labels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return source-config fields representing an effective spectrum.

    The returned mapping uses the existing source config block and adds explicit
    ``wavelengths_m`` plus normalized weights. Generic ``single_star`` sources
    receive one ``weights`` vector. Binary-like sources receive duplicated
    ``component_weights`` rows only for this single-spectrum compatibility path.
    Target-aware binary decks should use ``SourceSpectralDeck`` so primary and
    secondary rows can differ.

    Parameters
    ----------
    spectrum:
        Effective detected spectrum with wavelengths in meters and normalized
        weights.
    source_kind:
        Existing source kind, for example ``"single_star"``, ``"binary"``,
        ``"binary_target"``, or ``"alpha_cen"``.
    component_labels:
        Optional component labels for provenance. Defaults to ``("star",)`` for
        single-star and ``("primary", "secondary")`` for binary-like sources.

    Returns
    -------
    dict[str, Any]
        JSON-friendly source-config patch fields.
    """

    wavelengths, weights = _validate_spectrum(spectrum)
    kind = str(source_kind).lower()
    labels = component_labels or _source_component_labels({}, source_kind=kind)
    wavelength_m, bandwidth_m, n_lambda = _wavelength_summary(wavelengths)

    patch: dict[str, Any] = {
        "wavelength_m": wavelength_m,
        "bandwidth_m": bandwidth_m,
        "n_lambda": n_lambda,
        "wavelengths_m": wavelengths.tolist(),
        "spectral_deck_label": spectrum.label,
        "spectral_deck_provenance": _spectrum_provenance(
            spectrum,
            source_kind=kind,
            component_labels=labels,
        ),
    }
    if kind in SINGLE_SOURCE_KINDS:
        patch["weights"] = weights.tolist()
    elif kind in BINARY_SOURCE_KINDS:
        patch["component_weights"] = np.vstack([weights, weights]).tolist()
    else:
        raise ValueError(
            f"Unsupported source kind {source_kind!r}; expected one of "
            f"{sorted(BINARY_SOURCE_KINDS | SINGLE_SOURCE_KINDS)}."
        )
    return patch


def component_spectra_to_source_spectral_config(
    spectra: Mapping[str, EffectiveSpectrum],
    *,
    source_kind: str,
) -> dict[str, Any]:
    """Return source-config spectral fields from component spectra."""

    kind = str(source_kind).lower()
    if kind in SINGLE_SOURCE_KINDS:
        spectrum = spectra.get("star") or next(iter(spectra.values()))
        return spectrum_to_source_spectral_config(spectrum, source_kind=kind)
    if kind not in BINARY_SOURCE_KINDS:
        raise ValueError(f"Unsupported source kind {source_kind!r}.")
    if "primary" not in spectra or "secondary" not in spectra:
        raise ValueError("Binary component spectra must contain primary and secondary.")
    p_wavelengths, p_weights = _validate_spectrum(spectra["primary"])
    s_wavelengths, s_weights = _validate_spectrum(spectra["secondary"])
    if not np.allclose(p_wavelengths, s_wavelengths, rtol=0.0, atol=0.0):
        raise ValueError("Binary component spectra must share one wavelength grid.")
    wavelength_m, bandwidth_m, n_lambda = _wavelength_summary(p_wavelengths)
    component_weights = np.vstack([p_weights, s_weights])
    return {
        "wavelength_m": wavelength_m,
        "bandwidth_m": bandwidth_m,
        "n_lambda": n_lambda,
        "wavelengths_m": p_wavelengths.tolist(),
        "component_weights": component_weights.tolist(),
        "spectral_deck_label": "component_spectral_deck",
        "spectral_deck_provenance": {
            "source_kind": kind,
            "component_labels": ["primary", "secondary"],
            "n_lambda": n_lambda,
            "component_spectra": {
                label: _spectrum_provenance(spec, source_kind=kind, component_labels=(label,))
                for label, spec in spectra.items()
            },
            "component_weights_differ": bool(not np.allclose(p_weights, s_weights)),
            "weight_normalization": "row_sample_sum",
            "flux_factor_usage": "diagnostic_provenance_only",
            "spectral_shape_active_inference_parameter": False,
        },
    }


def apply_effective_spectrum_to_source_config(
    source_cfg: Mapping[str, Any],
    spectrum: EffectiveSpectrum,
    *,
    preserve_flux_parameters: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a source config patched with one effective spectrum.

    Inputs are deep-copied and never mutated. By default existing
    ``log_flux_total`` and ``contrast`` fields are preserved, so the spectral
    deck changes chromatic shape only and leaves detected band-integrated flux
    semantics intact.
    """

    patched = deepcopy(dict(source_cfg))
    kind = _source_kind(patched)
    labels = _source_component_labels(patched, source_kind=kind)
    before_flux = {
        key: deepcopy(patched[key])
        for key in ("log_flux_total", "contrast")
        if key in patched
    }
    spectral_patch = spectrum_to_source_spectral_config(
        spectrum,
        source_kind=kind,
        component_labels=labels,
    )
    patched.update(spectral_patch)
    if preserve_flux_parameters:
        patched.update(before_flux)

    provenance = {
        "applied_to": "source",
        "source_kind": kind,
        "spectrum": spectral_patch["spectral_deck_provenance"],
        "preserve_flux_parameters": bool(preserve_flux_parameters),
        "preserved_flux_parameters": sorted(before_flux),
    }
    return patched, _as_jsonable(provenance)


def _extract_system_block(system_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    cfg = deepcopy(dict(system_cfg))
    if "system" in cfg:
        system = cfg.get("system")
        if not isinstance(system, Mapping):
            raise ValueError("Top-level system_cfg['system'] must be a mapping/dict.")
        return deepcopy(dict(system)), True
    return cfg, False


def _repack_system_block(original: Mapping[str, Any], patched_system: dict[str, Any], *, had_outer_system: bool) -> dict[str, Any]:
    if not had_outer_system:
        return patched_system
    repacked = deepcopy(dict(original))
    repacked["system"] = patched_system
    return repacked


def _apply_to_system_config(
    system_cfg: Mapping[str, Any],
    spectrum: EffectiveSpectrum,
    *,
    preserve_flux_parameters: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system, had_outer = _extract_system_block(system_cfg)
    source = system.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("system config must contain a mapping at system.source.")
    patched_source, provenance = apply_effective_spectrum_to_source_config(
        source,
        spectrum,
        preserve_flux_parameters=preserve_flux_parameters,
    )
    system["source"] = patched_source
    return _repack_system_block(system_cfg, system, had_outer_system=had_outer), provenance


def _apply_source_deck_to_system_config(
    system_cfg: Mapping[str, Any],
    spectra: Mapping[str, EffectiveSpectrum],
    *,
    deck: SourceSpectralDeck,
    role: str,
    preserve_flux_parameters: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    system, had_outer = _extract_system_block(system_cfg)
    source = system.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("system config must contain a mapping at system.source.")
    patched_source = deepcopy(dict(source))
    before_flux = {
        key: deepcopy(patched_source[key])
        for key in ("log_flux_total", "contrast")
        if key in patched_source
    }
    patch = component_spectra_to_source_spectral_config(
        spectra,
        source_kind=str(patched_source.get("kind", deck.source_kind)),
    )
    patched_source.update(patch)
    if preserve_flux_parameters:
        patched_source.update(before_flux)
    system["source"] = patched_source
    provenance = {
        "applied_to": "source",
        "role": role,
        "source_kind": str(patched_source.get("kind", deck.source_kind)).lower(),
        "target": deck.target,
        "spectrum": patch["spectral_deck_provenance"],
        "sed_resolution": deck.provenance.get("sed_resolution"),
        "preserve_flux_parameters": bool(preserve_flux_parameters),
        "preserved_flux_parameters": sorted(before_flux),
    }
    return _repack_system_block(system_cfg, system, had_outer_system=had_outer), _as_jsonable(provenance)


def apply_spectral_deck_to_system_configs(
    base_system_cfg: Mapping[str, Any],
    deck: SpectralDeck | SourceSpectralDeck,
    *,
    truth_label: str = "truth",
    inference_label: str = "inference",
    preserve_flux_parameters: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return truth/inference system configs patched from one spectral deck.

    The truth config receives ``deck.truth`` and the inference config receives
    ``deck.inference``. Both are derived from the same base config through deep
    copies, so callers can safely reuse the input mapping.
    """

    truth_cfg, truth_prov = _apply_to_system_config(
        base_system_cfg,
        deck.truth,
        preserve_flux_parameters=preserve_flux_parameters,
    )
    inference_cfg, inference_prov = _apply_to_system_config(
        base_system_cfg,
        deck.inference,
        preserve_flux_parameters=preserve_flux_parameters,
    )
    provenance = {
        "schema_version": "spectral_source_config.v1",
        "truth_label": truth_label,
        "inference_label": inference_label,
        "truth": truth_prov,
        "inference": inference_prov,
        "deck_schema_version": deck.schema_version,
        "deck_comparison": _as_jsonable(deck.comparison),
        "deck_flux_factor_ratio_inference_over_truth": deck.comparison.get(
            "flux_factor_ratio_inference_over_truth"
        ),
        "active_inference_parameters_added": [],
        "preserve_flux_parameters": bool(preserve_flux_parameters),
    }
    return truth_cfg, inference_cfg, _as_jsonable(provenance)


def apply_source_spectral_deck_to_system_configs(
    base_system_cfg: Mapping[str, Any],
    deck: SourceSpectralDeck,
    *,
    preserve_flux_parameters: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return truth/inference configs patched with component spectra."""

    truth_cfg, truth_prov = _apply_source_deck_to_system_config(
        base_system_cfg,
        deck.truth_by_component,
        deck=deck,
        role="truth",
        preserve_flux_parameters=preserve_flux_parameters,
    )
    inference_cfg, inference_prov = _apply_source_deck_to_system_config(
        base_system_cfg,
        deck.inference_by_component,
        deck=deck,
        role="inference",
        preserve_flux_parameters=preserve_flux_parameters,
    )
    provenance = {
        "schema_version": "spectral_source_config.v1",
        "truth": truth_prov,
        "inference": inference_prov,
        "deck_schema_version": deck.schema_version,
        "comparison_by_component": _as_jsonable(deck.comparison_by_component),
        "combined_comparison": _as_jsonable(deck.combined_comparison),
        "active_inference_parameters_added": [],
        "preserve_flux_parameters": bool(preserve_flux_parameters),
    }
    return truth_cfg, inference_cfg, _as_jsonable(provenance)


def build_spectral_truth_inference_system_configs(
    *,
    base_system_cfg: Mapping[str, Any],
    deck: SpectralDeck,
    truth_label: str = "truth",
    inference_label: str = "inference",
    preserve_flux_parameters: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build truth and inference system configs from a base config and deck.

    This is an explicit alias for future campaign/deck generators. It keeps the
    call site readable when the surrounding code is already constructing truth
    and knowledge/inference decks.
    """

    if isinstance(deck, SourceSpectralDeck):
        return apply_source_spectral_deck_to_system_configs(
            base_system_cfg,
            deck,
            preserve_flux_parameters=preserve_flux_parameters,
        )

    return apply_spectral_deck_to_system_configs(
        base_system_cfg,
        deck,
        truth_label=truth_label,
        inference_label=inference_label,
        preserve_flux_parameters=preserve_flux_parameters,
    )
