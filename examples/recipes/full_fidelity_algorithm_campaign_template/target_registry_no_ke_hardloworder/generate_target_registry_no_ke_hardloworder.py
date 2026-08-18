"""Emit target-registry No-KE hard-low-order full-fidelity source configs."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[4]
CAMPAIGN_NAME = (
    "full_fidelity_target_registry_no_ke_hardloworder_"
    "n10_w10x30_projected_30min_v1"
)
BASELINE = (
    ROOT
    / "examples/recipes/full_fidelity_algorithm_campaign_template/"
    "full_fidelity_info_damped_detector_ke_projected_30min_v1.yaml"
)
OUTDIR = Path(__file__).resolve().parent
TARGETS = (
    "ALPHA_CEN",
    "61_CYG",
    "70_OPH",
    "36_OPH",
    "XI_BOO",
    "P_ERI",
    "HR_2667_2668",
)


def target_seed(target: str) -> int:
    digest = hashlib.sha256(f"{CAMPAIGN_NAME}:{target}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def hard_sigmas() -> dict[str, dict[str, Any]]:
    return {
        "source.separation_as": {
            "kind": "absolute",
            "sigma": 1.0e-4,
            "unit": "arcsec",
        },
        "source.log_flux_total": {
            "kind": "absolute",
            "sigma": 1.0e-4,
            "unit": "log_flux",
        },
        "source.contrast": {
            "kind": "fractional",
            "sigma": 1.0e-4,
        },
        "optics.plate_scale_as_per_pix": {
            "kind": "fractional",
            "sigma": 1.0e-4,
        },
        "optics.primary.zernike_coeffs_nm[*]": {
            "kind": "absolute",
            "sigma": 1.0,
            "unit": "nm",
        },
        "optics.secondary.zernike_coeffs_nm[*]": {
            "kind": "absolute",
            "sigma": 1.0,
            "unit": "nm",
        },
    }


def build_config(base: dict[str, Any], target: str) -> dict[str, Any]:
    config = copy.deepcopy(base)
    experiment = config["experiment"]
    run_name = f"ff_targetreg_no_ke_hardlo_{target}_n10_w10x30_v1"

    experiment.update(
        {
            "kind": "full_fidelity_binary_iterative",
            "source_kind": "binary_target",
            "target": target,
            "system_preset": "SHERA_FLIGHT_3P_CONV",
            "run_name": run_name,
            "n_cases": 10,
        }
    )

    experiment["detector_calibration_knowledge_error"] = {
        "enabled": False,
        "apply_to": "inference",
        "realization_policy": "fixed_per_experiment",
        "pixel_offsets": {
            "enabled": True,
            "sigma_pix": 0.001,
            "distribution": "normal",
        },
        "pixel_response": {
            "enabled": True,
            "sigma_fractional": 0.001,
            "distribution": "normal",
        },
    }

    high_order_wfe = experiment["high_order_wfe"]
    high_order_wfe["enabled"] = True
    high_order_wfe["truth"] = {
        "enabled": True,
        "mirrors": ["primary", "secondary"],
        "mode": "synthetic",
        "npix": 256,
        "amplitude_nm_rms": 20.0,
        "power_law_alpha": 2.5,
        "seed": 20260610,
        "pairing": "independent",
        "remove_low_order_zernikes": True,
        "remove_zernike_modes": [4, 5, 6, 7, 8, 9, 10, 11],
    }
    high_order_wfe["inference"] = {
        "enabled": True,
        "mode": "knowledge_error",
        "use_truth_common_map": True,
        "knowledge_error": {
            "enabled": False,
            "amplitude_nm_rms": 0.0,
            "power_law_alpha": "same_as_truth",
            "remove_low_order_zernikes": True,
            "realization_policy": "additive_correlated",
        },
    }
    high_order_wfe["artifacts"] = {
        "write_maps": True,
        "write_png_quicklooks": False,
        "write_summary_json": True,
    }
    high_order_wfe["validation"] = {
        "require_nonzero_difference_when_enabled": False,
        "max_abs_low_order_projection_nm": 1.0e-4,
        "fail_on_low_order_projection": True,
    }

    subblocks = experiment["subblocks"]
    subblocks.pop("n_subblocks", None)
    trace_window = subblocks.get("trace_source", {}).get("window", {})
    if isinstance(trace_window, dict):
        trace_window.pop("n_subblocks", None)

    experiment["iterative"]["windows_per_draw"] = 10
    experiment["iterative"]["subblocks_per_window"] = 30
    forecast = experiment["iterative_forecast"]
    forecast.pop("actual_windows", None)
    forecast.pop("subblocks_per_window", None)
    forecast["projected_windows"] = 60
    forecast["observation_duration_s"] = 1800.0

    sigmas = hard_sigmas()
    experiment["prior_draws"] = {
        "enabled": True,
        "n_cases": 10,
        "draws_per_condition": 10,
        "center": "truth",
        "distribution": "normal",
        "draw_seed": target_seed(target),
        "case_name_template": f"{target}_hardloworder_draw_{{draw_index:03d}}",
        "sigmas": copy.deepcopy(sigmas),
        "conditions": [
            {
                "condition_name": "m1_1p0nm_m2_1p0nm",
                "sigmas": {
                    "optics.primary.zernike_coeffs_nm[*]": copy.deepcopy(
                        sigmas["optics.primary.zernike_coeffs_nm[*]"]
                    ),
                    "optics.secondary.zernike_coeffs_nm[*]": copy.deepcopy(
                        sigmas["optics.secondary.zernike_coeffs_nm[*]"]
                    ),
                },
            }
        ],
    }

    return config


def main() -> None:
    base = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or "experiment" not in base:
        raise SystemExit(f"Invalid baseline config: {BASELINE}")
    for target in TARGETS:
        config = build_config(base, target)
        path = OUTDIR / f"ff_targetreg_no_ke_hardlo_{target}_n10_w10x30_v1.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
