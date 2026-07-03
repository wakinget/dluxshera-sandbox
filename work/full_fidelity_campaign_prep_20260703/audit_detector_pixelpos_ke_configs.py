from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


def mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def audit(path: Path) -> tuple[bool, str]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment = mapping(mapping(payload).get("experiment", payload))
    det = mapping(experiment.get("detector_calibration_knowledge_error"))
    pix = mapping(det.get("pixel_offsets"))
    resp = mapping(det.get("pixel_response"))
    enabled = bool(det.get("enabled", False))
    apply_to = str(det.get("apply_to", ""))
    policy = str(det.get("realization_policy", ""))
    pix_enabled = bool(pix.get("enabled", False))
    pix_sigma = pix.get("sigma_pix")
    pix_dist = str(pix.get("distribution", ""))
    resp_enabled = bool(resp.get("enabled", False))
    resp_sigma = float(resp.get("sigma_fractional", 0.0) or 0.0)
    ok = (
        enabled
        and apply_to == "inference"
        and policy == "fixed_per_experiment"
        and pix_enabled
        and pix_sigma is not None
        and float(pix_sigma) > 0.0
        and pix_dist == "normal"
        and (not resp_enabled)
        and resp_sigma == 0.0
    )
    detail = (
        f"enabled={enabled} apply_to={apply_to} policy={policy} "
        f"pixel_offsets.enabled={pix_enabled} sigma_pix={pix_sigma} "
        f"distribution={pix_dist} pixel_response.enabled={resp_enabled} "
        f"sigma_fractional={resp_sigma}"
    )
    return ok, detail


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify detector pixel-position KE is enabled and pixel response KE is disabled."
    )
    parser.add_argument("configs", nargs="+", type=Path)
    args = parser.parse_args()
    failures = 0
    for path in args.configs:
        ok, detail = audit(path)
        print(f"{'PASS' if ok else 'FAIL'} {path}: {detail}")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
