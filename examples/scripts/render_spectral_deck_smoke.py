#!/usr/bin/env python3
"""Render a tiny truth/model PSF pair from a spectral throughput deck.

This script is intentionally render-only. It builds a synthetic spectral deck,
patches one base system config into truth and inference source configs, renders
one image from each, and writes lightweight artifacts for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dluxshera_matplotlib")

import numpy as np
import yaml

from dluxshera.config.resolver import resolve_config
from dluxshera.params.store import ParameterStore
from dluxshera.params.transform_registry import DEFAULT_SYSTEM_ID
from dluxshera.systems import SheraBinder
from dluxshera.systems.base import compose_forward_spec
from dluxshera.systems.two_plane import SHERA_TWOPLANE_SYSTEM_ID
from dluxshera.utils.spectral_response import (
    DEFAULT_DETECTOR_QE_PATH,
    DEFAULT_FILTER_RESPONSE_PATH,
    DETECTOR_QE_PROXY_ASSUMPTION,
    build_truth_inference_spectral_deck,
    resolve_response_curve_path,
    write_spectral_deck_artifacts,
)
from dluxshera.utils.spectral_source_config import (
    build_spectral_truth_inference_system_configs,
)

TEMPLATE_PATH = Path(
    "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_algorithm_campaign_v1.yaml"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _read_spectral_model(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    return dict(payload["experiment"]["spectral_model"])


def _enabled_component(component: Any) -> dict[str, Any] | None:
    if isinstance(component, dict):
        if component.get("enabled", True) is False:
            return None
        if component.get("mode") == "same_as_truth":
            return None
        return dict(component)
    if component is True:
        return {}
    return None


def _component_from_template(
    spectral_model: dict[str, Any],
    name: str,
    *,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    truth_components = spectral_model.get("truth", {}).get("components", {})
    if not isinstance(truth_components, dict):
        return dict(fallback)
    component = _enabled_component(truth_components.get(name))
    if component is None:
        return dict(fallback)
    resolved = dict(fallback)
    resolved.update(component)
    return resolved


def _real_response_components(
    spectral_model: dict[str, Any],
    *,
    filter_response: Path | None,
    detector_qe: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    detector = _component_from_template(
        spectral_model,
        "detector_qe",
        fallback={
            "enabled": True,
            "path": DEFAULT_DETECTOR_QE_PATH,
            "label": "LTN4323_QE_proxy_for_HWK4123",
            "wavelength_column": "Wavelength (nm)",
            "wavelength_unit": "nm",
            "response_column": "QE",
            "response_unit": "dimensionless",
            "response_scale": 1.0,
            "detector_model_proxy_for": "HWK4123",
            "assumption": DETECTOR_QE_PROXY_ASSUMPTION,
        },
    )
    filt = _component_from_template(
        spectral_model,
        "m2_filter_response",
        fallback={
            "enabled": True,
            "path": DEFAULT_FILTER_RESPONSE_PATH,
            "label": "SHERA_Notch_Filter_V2",
            "wavelength_column": "Wavelength (nm)",
            "wavelength_unit": "nm",
            "response_column": "T (%)",
            "response_unit": "percent_transmission",
            "response_scale": 0.01,
        },
    )
    if detector_qe is not None:
        detector["path"] = str(detector_qe)
    if filter_response is not None:
        filt["path"] = str(filter_response)

    selected = {
        "response_mode": "real",
        "detector_qe": {
            "path": detector["path"],
            "resolved_path": str(resolve_response_curve_path(detector["path"])),
            "label": detector.get("label"),
            "response_column": detector.get("response_column"),
            "wavelength_unit": detector.get("wavelength_unit"),
            "response_scale": detector.get("response_scale", 1.0),
            "assumption": detector.get("assumption"),
            "detector_model_proxy_for": detector.get("detector_model_proxy_for"),
        },
        "filter_response": {
            "path": filt["path"],
            "resolved_path": str(resolve_response_curve_path(filt["path"])),
            "label": filt.get("label"),
            "response_column": filt.get("response_column"),
            "wavelength_unit": filt.get("wavelength_unit"),
            "response_scale": filt.get("response_scale", 1.0),
        },
    }
    return detector, filt, selected


def _fast_system_config(*, source_kind: str) -> dict[str, Any]:
    resolved = resolve_config({"system": {"preset": "SHERA_FLIGHT_3P_SIMPLE"}})
    system = resolved["system"]
    system["source"] = {
        "kind": source_kind,
        "wavelength_m": 550e-9,
        "bandwidth_m": 110e-9,
        "n_lambda": 3,
        "x_position_as": 0.0,
        "y_position_as": 0.0,
        "position_angle_deg": 60.0,
        "log_flux_total": 6.0,
    }
    if source_kind in {"binary", "binary_target", "alpha_cen"}:
        system["source"].update(
            {
                "target": "ALPHA_CEN" if source_kind in {"binary_target", "alpha_cen"} else None,
                "separation_as": 8.0,
                "contrast": 3.0,
            }
        )
    if source_kind == "alpha_cen":
        system["source"]["target"] = "ALPHA_CEN"

    system["optics"].update(
        {
            "pupil_npix": 64,
            "psf_npix": 64,
            "oversample": 1,
            "primary_noll_indices": [],
            "secondary_noll_indices": [],
            "dp_path": None,
            "diffractive_pupil_path": None,
        }
    )
    system["detector"]["layers"] = [
        {"name": "downsample", "kind": "Downsample", "kernel_size": 1}
    ]
    return system


def _render(system_cfg: dict[str, Any]) -> np.ndarray:
    spec = compose_forward_spec({"system": system_cfg})
    store = ParameterStore.from_spec_defaults(spec)
    system_id = DEFAULT_SYSTEM_ID if spec.system_id == SHERA_TWOPLANE_SYSTEM_ID else None
    store = store.refresh_derived(spec, system_id=system_id)
    binder = SheraBinder(system_cfg, spec, store)
    return np.asarray(binder.model(binder.strip_structural(store)))


def run(
    *,
    outdir: Path,
    source_kind: str,
    fast: bool,
    response_mode: str,
    filter_response: Path | None,
    detector_qe: Path | None,
) -> dict[str, Any]:
    spectral_model = _read_spectral_model(TEMPLATE_PATH)
    truth_cfg = dict(spectral_model["truth"])
    inference_cfg = dict(spectral_model["inference"])
    if fast:
        truth_cfg["n_lambda"] = min(int(truth_cfg.get("n_lambda", 30)), 11)
        inference_cfg["n_lambda"] = min(int(inference_cfg.get("n_lambda", 7)), 5)

    if response_mode == "synthetic-flat":
        detector_component = {"label": "synthetic_flat_qe", "response": 1.0}
        filter_component = {"label": "synthetic_flat_filter", "response": 1.0}
        selected_responses = {
            "response_mode": "synthetic-flat",
            "detector_qe": {"label": "synthetic_flat_qe", "path": None},
            "filter_response": {"label": "synthetic_flat_filter", "path": None},
        }
    else:
        detector_component, filter_component, selected_responses = _real_response_components(
            spectral_model,
            filter_response=filter_response,
            detector_qe=detector_qe,
        )

    deck = build_truth_inference_spectral_deck(
        sed=lambda wavelengths_m: wavelengths_m * 1e9,
        truth_config=truth_cfg,
        inference_config=inference_cfg,
        detector_qe=detector_component,
        filter_response=filter_component,
        provenance={
            "script": "examples/scripts/render_spectral_deck_smoke.py",
            "selected_responses": selected_responses,
        },
    )

    outdir.mkdir(parents=True, exist_ok=True)
    spectral_paths = write_spectral_deck_artifacts(deck, outdir / "spectral")
    base_system = _fast_system_config(source_kind=source_kind)
    truth_system, inference_system, spectral_provenance = build_spectral_truth_inference_system_configs(
        base_system_cfg=base_system,
        deck=deck,
    )

    truth_image = _render(truth_system)
    inference_image = _render(inference_system)
    residual = truth_image - inference_image

    np.save(outdir / "truth_psf.npy", truth_image)
    np.save(outdir / "inference_psf.npy", inference_image)
    np.save(outdir / "residual_psf.npy", residual)
    (outdir / "truth_system_config.json").write_text(
        json.dumps(_json_ready(truth_system), indent=2, sort_keys=True) + "\n"
    )
    (outdir / "inference_system_config.json").write_text(
        json.dumps(_json_ready(inference_system), indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "schema_version": "spectral_deck_render_smoke.v1",
        "source_kind": source_kind,
        "fast": bool(fast),
        "truth_shape": list(truth_image.shape),
        "inference_shape": list(inference_image.shape),
        "truth_sum": float(np.sum(truth_image)),
        "inference_sum": float(np.sum(inference_image)),
        "residual_l2": float(np.sqrt(np.sum(residual**2))),
        "images_identical": bool(np.allclose(truth_image, inference_image)),
        "selected_responses": selected_responses,
        "truth_lambda_eff_nm": deck.truth.diagnostics.get("lambda_eff_nm"),
        "inference_lambda_eff_nm": deck.inference.diagnostics.get("lambda_eff_nm"),
        "truth_out_of_inference_band_fraction": deck.comparison.get(
            "truth_out_of_inference_band_fraction"
        ),
        "flux_factor_ratio_inference_over_truth": deck.comparison.get(
            "flux_factor_ratio_inference_over_truth"
        ),
        "spectral_provenance": spectral_provenance,
        "spectral_artifacts": {key: str(path) for key, path in spectral_paths.items()},
    }
    (outdir / "render_summary.json").write_text(
        json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path("Results/spectral_deck_smoke"))
    parser.add_argument(
        "--source-kind",
        choices=("single_star", "binary", "binary_target", "alpha_cen"),
        default="binary",
    )
    parser.add_argument("--fast", action="store_true", help="Use smaller wavelength counts for quick smoke renders.")
    parser.add_argument(
        "--response-mode",
        choices=("real", "synthetic-flat"),
        default="real",
        help="Use real configured response CSVs or synthetic flat responses.",
    )
    parser.add_argument(
        "--filter-response",
        type=Path,
        default=None,
        help=f"Override filter response CSV path. Default comes from template or {DEFAULT_FILTER_RESPONSE_PATH}.",
    )
    parser.add_argument(
        "--detector-qe",
        type=Path,
        default=None,
        help=f"Override detector QE CSV path. Default comes from template or {DEFAULT_DETECTOR_QE_PATH}.",
    )
    args = parser.parse_args()
    summary = run(
        outdir=args.outdir,
        source_kind=args.source_kind,
        fast=args.fast,
        response_mode=args.response_mode,
        filter_response=args.filter_response,
        detector_qe=args.detector_qe,
    )
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
