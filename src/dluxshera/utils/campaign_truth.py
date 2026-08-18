"""Shared campaign truth-realization helpers."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from dluxshera.utils.obs_subblock_keys import (
    parse_obs_subblock_key_address,
    set_obs_subblock_mapping_value,
)


ZERNIKE_THETA_LABEL_PATTERN = re.compile(
    r"^optics\.(primary|secondary)\.zernike_coeffs_nm\[(\d+)\]$"
)


@dataclass(frozen=True)
class TruthRealizationResult:
    truth_overrides_by_label: dict[str, float]
    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def parse_zernike_theta_label(label: str) -> tuple[str, int] | None:
    match = ZERNIKE_THETA_LABEL_PATTERN.fullmatch(label)
    if match is None:
        return None
    return str(match.group(1)), int(match.group(2))


def zernike_labels_by_mirror(labels: Sequence[str]) -> dict[str, list[tuple[int, str]]]:
    grouped: dict[str, list[tuple[int, str]]] = {"primary": [], "secondary": []}
    for label in labels:
        parsed = parse_zernike_theta_label(str(label))
        if parsed is None:
            continue
        mirror, index = parsed
        grouped[mirror].append((index, str(label)))
    for mirror in grouped:
        grouped[mirror].sort(key=lambda item: item[0])
    return grouped


def realize_campaign_truth(
    *,
    experiment_cfg: Mapping[str, Any],
    labels: Sequence[str],
    base_truth_by_label: Mapping[str, float],
) -> TruthRealizationResult:
    """Realize optional campaign-level Zernike truth offsets."""

    raw_cfg = experiment_cfg.get("truth_realization", {}) or {}
    if not isinstance(raw_cfg, Mapping) or not bool(raw_cfg.get("enabled", False)):
        return TruthRealizationResult({}, [], {"enabled": False, "status": "disabled"})
    mode = str(raw_cfg.get("mode", ""))
    if mode != "zernike_per_coefficient_sigma":
        raise ValueError(f"Unsupported truth_realization.mode: {mode!r}")
    seed = int(raw_cfg.get("seed", 0))
    combine_with_system_truth = bool(raw_cfg.get("combine_with_system_truth", False))
    zernike_cfg = raw_cfg.get("zernikes", {}) or {}
    if not isinstance(zernike_cfg, Mapping):
        raise ValueError("truth_realization.zernikes must be a mapping.")
    grouped = zernike_labels_by_mirror(labels)
    rng = np.random.default_rng(seed)
    overrides: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for mirror in ("primary", "secondary"):
        mirror_cfg = zernike_cfg.get(mirror, {}) or {}
        if not isinstance(mirror_cfg, Mapping):
            raise ValueError(f"truth_realization.zernikes.{mirror} must be a mapping.")
        if not bool(mirror_cfg.get("enabled", False)):
            continue
        sigma_nm = float(mirror_cfg.get("sigma_nm", mirror_cfg.get("rms_nm", 0.0)))
        mean_nm = float(mirror_cfg.get("mean_nm", 0.0))
        if sigma_nm < 0.0 or (not math.isfinite(sigma_nm)) or (not math.isfinite(mean_nm)):
            raise ValueError(f"truth_realization.zernikes.{mirror} sigma/mean must be finite, sigma>=0.")
        indices_cfg = mirror_cfg.get("indices", "from_observation_theta")
        available = dict(grouped[mirror])
        if indices_cfg == "from_observation_theta":
            selected_indices = sorted(available)
            selected_by = "from_observation_theta"
        elif isinstance(indices_cfg, Sequence) and not isinstance(indices_cfg, (str, bytes)):
            selected_indices = [int(value) for value in indices_cfg]
            selected_by = "explicit_indices"
        else:
            raise ValueError(f"Unsupported truth_realization.zernikes.{mirror}.indices: {indices_cfg!r}")
        if not selected_indices:
            raise ValueError(f"truth_realization.zernikes.{mirror} selected no indices.")
        missing = [index for index in selected_indices if index not in available]
        if missing:
            raise ValueError(f"truth_realization.zernikes.{mirror} requested missing observation-theta indices: {missing}")
        for index in selected_indices:
            label = available[index]
            draw_z = float(rng.normal())
            draw_value_nm = float(draw_z * sigma_nm)
            base_truth = float(base_truth_by_label[label])
            truth_value = (base_truth if combine_with_system_truth else 0.0) + mean_nm + draw_value_nm
            truth_offset = float(truth_value - base_truth)
            overrides[label] = float(truth_value)
            rows.append(
                {
                    "theta_label": label,
                    "label": label,
                    "mirror": mirror,
                    "zernike_index": int(index),
                    "enabled": True,
                    "truth_seed": int(seed),
                    "mode": mode,
                    "distribution": "normal",
                    "mean_nm": mean_nm,
                    "sigma_nm": sigma_nm,
                    "draw_z": draw_z,
                    "draw_value_nm": draw_value_nm,
                    "truth_offset": truth_offset,
                    "truth_offset_nm": truth_offset,
                    "base_truth_value_nm": base_truth,
                    "nominal_truth_value": base_truth,
                    "truth_value_nm": float(truth_value),
                    "realized_truth_value": float(truth_value),
                    "unit": "nm",
                    "group": f"optics.{mirror}_zernikes",
                    "combine_with_system_truth": combine_with_system_truth,
                    "selected_by": selected_by,
                    "status": "applied",
                }
            )
    return TruthRealizationResult(
        overrides,
        rows,
        {
            "enabled": True,
            "mode": mode,
            "seed": int(seed),
            "combine_with_system_truth": combine_with_system_truth,
            "n_overrides": len(overrides),
        },
    )


def apply_truth_overrides_to_system_config(
    system_cfg: Mapping[str, Any],
    overrides_by_label: Mapping[str, float],
) -> dict[str, Any]:
    """Return a system config copy with realized Zernike truth values applied."""

    import copy

    out = copy.deepcopy(dict(system_cfg))
    reference_vectors: dict[str, list[float]] = {}
    for label in overrides_by_label:
        parsed = parse_zernike_theta_label(str(label))
        if parsed is None:
            continue
        _mirror, index = parsed
        address = parse_obs_subblock_key_address(str(label))
        vector = reference_vectors.setdefault(address.base_key, [])
        if index >= len(vector):
            vector.extend([0.0] * (index + 1 - len(vector)))
    for label, value in overrides_by_label.items():
        parsed = parse_zernike_theta_label(str(label))
        if parsed is None:
            continue
        _mirror, index = parsed
        address = parse_obs_subblock_key_address(str(label))
        set_obs_subblock_mapping_value(
            out,
            address=address,
            value=float(value),
            reference_vector=reference_vectors[address.base_key],
        )
    return out
