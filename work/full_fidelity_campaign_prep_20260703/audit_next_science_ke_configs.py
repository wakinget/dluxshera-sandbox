from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


PRIMARY_KEY = "optics.primary.zernike_coeffs_nm[*]"
SECONDARY_KEY = "optics.secondary.zernike_coeffs_nm[*]"


EXPECTED = {
    "hoke_0p1": {
        "kind": "hoke",
        "amplitude_nm": 0.1,
        "m1_sigma_nm": 0.01,
        "m2_sigma_nm": 0.01,
    },
    "hoke_1p0": {
        "kind": "hoke",
        "amplitude_nm": 1.0,
        "m1_sigma_nm": 0.01,
        "m2_sigma_nm": 0.01,
    },
    "pixelposke_1em4": {
        "kind": "pixelposke",
        "sigma_pix": 1.0e-4,
        "m1_sigma_nm": 0.3,
        "m2_sigma_nm": 0.3,
    },
    "pixelposke_5em4": {
        "kind": "pixelposke",
        "sigma_pix": 5.0e-4,
        "m1_sigma_nm": 0.3,
        "m2_sigma_nm": 0.3,
    },
    "pixelposke_1em3": {
        "kind": "pixelposke",
        "sigma_pix": 1.0e-3,
        "m1_sigma_nm": 0.3,
        "m2_sigma_nm": 0.3,
    },
}


def as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def load_experiment(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML must contain a mapping: {path}")
    experiment = payload.get("experiment")
    if not isinstance(experiment, Mapping):
        raise ValueError(f"YAML missing experiment mapping: {path}")
    return dict(experiment)


def expected_for_run(run_name: str) -> dict[str, Any] | None:
    for token, expected in EXPECTED.items():
        if token in run_name:
            return dict(expected)
    return None


def first_condition_sigmas(experiment: Mapping[str, Any]) -> tuple[float | None, float | None]:
    prior = as_mapping(experiment.get("prior_draws"))
    conditions = prior.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        return None, None
    condition = as_mapping(conditions[0])
    sigmas = as_mapping(condition.get("sigmas"))
    primary = as_mapping(sigmas.get(PRIMARY_KEY)).get("sigma")
    secondary = as_mapping(sigmas.get(SECONDARY_KEY)).get("sigma")
    return (
        None if primary is None else float(primary),
        None if secondary is None else float(secondary),
    )


def close(a: float | None, b: float, *, rtol: float = 1e-9) -> bool:
    return a is not None and abs(float(a) - float(b)) <= max(rtol * abs(float(b)), 1e-12)


def audit(path: Path) -> tuple[bool, str]:
    experiment = load_experiment(path)
    run_name = str(experiment.get("run_name", ""))
    expected = expected_for_run(run_name)
    if expected is None:
        return True, f"SKIP {path}: run_name={run_name}"

    iterative = as_mapping(experiment.get("iterative"))
    windows = int(iterative.get("windows_per_draw", 0) or 0)
    subblocks = int(iterative.get("subblocks_per_window", 0) or 0)
    hoke = experiment.get("high_order_wfe")
    detector = experiment.get("detector_calibration_knowledge_error")
    m1_sigma, m2_sigma = first_condition_sigmas(experiment)

    print(f"\npath: {path}")
    print(f"run_name: {run_name}")
    print(f"windows_per_draw: {windows}")
    print(f"subblocks_per_window: {subblocks}")
    print(f"high_order_wfe: {hoke}")
    print(f"detector_calibration_knowledge_error: {detector}")
    print(f"primary_low_order_sigma_nm: {m1_sigma}")
    print(f"secondary_low_order_sigma_nm: {m2_sigma}")

    failures: list[str] = []
    if windows != 10:
        failures.append(f"windows_per_draw {windows} != 10")
    if subblocks != 30:
        failures.append(f"subblocks_per_window {subblocks} != 30")
    if not close(m1_sigma, float(expected["m1_sigma_nm"])):
        failures.append(f"primary sigma {m1_sigma} != {expected['m1_sigma_nm']}")
    if not close(m2_sigma, float(expected["m2_sigma_nm"])):
        failures.append(f"secondary sigma {m2_sigma} != {expected['m2_sigma_nm']}")

    if expected["kind"] == "hoke":
        hoke_map = as_mapping(hoke)
        detector_map = as_mapping(detector)
        ke = as_mapping(as_mapping(hoke_map.get("inference")).get("knowledge_error"))
        amplitude = ke.get("amplitude_nm_rms")
        if not hoke_map or not bool(hoke_map.get("enabled", False)):
            failures.append("hoke config has no enabled high_order_wfe block")
        if not bool(ke.get("enabled", False)):
            failures.append("hoke config has no enabled high_order_wfe.inference.knowledge_error block")
        if not close(None if amplitude is None else float(amplitude), float(expected["amplitude_nm"])):
            failures.append(f"HO-WFE KE amplitude {amplitude} != {expected['amplitude_nm']}")
        if bool(detector_map.get("enabled", False)):
            failures.append("hoke detector_calibration_knowledge_error must be disabled")
    else:
        hoke_map = as_mapping(hoke)
        detector_map = as_mapping(detector)
        pix = as_mapping(detector_map.get("pixel_offsets"))
        resp = as_mapping(detector_map.get("pixel_response"))
        sigma_pix = pix.get("sigma_pix")
        if not detector_map or not bool(detector_map.get("enabled", False)):
            failures.append("pixelposke config has no enabled detector_calibration_knowledge_error block")
        if detector_map.get("apply_to") != "inference":
            failures.append("detector_calibration_knowledge_error.apply_to must be inference")
        if detector_map.get("realization_policy") != "fixed_per_experiment":
            failures.append("detector realization_policy must be fixed_per_experiment")
        if not bool(pix.get("enabled", False)):
            failures.append("pixel_offsets.enabled must be true")
        if not close(None if sigma_pix is None else float(sigma_pix), float(expected["sigma_pix"])):
            failures.append(f"pixel_offsets.sigma_pix {sigma_pix} != {expected['sigma_pix']}")
        if pix.get("distribution") != "normal":
            failures.append("pixel_offsets.distribution must be normal")
        if bool(resp.get("enabled", False)):
            failures.append("pixel_response.enabled must be false")
        if float(resp.get("sigma_fractional", 0.0) or 0.0) != 0.0:
            failures.append("pixel_response.sigma_fractional must be 0.0")
        if bool(hoke_map.get("enabled", False)):
            failures.append("pixelposke high_order_wfe must be disabled")

    if failures:
        return False, "FAIL " + str(path) + ": " + "; ".join(failures)
    return True, f"PASS {path}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit next science-sweep YAMLs for required HO-WFE and detector "
            "knowledge-error blocks."
        )
    )
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()
    failures = 0
    for path in args.configs:
        ok, message = audit(path)
        print(message)
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
