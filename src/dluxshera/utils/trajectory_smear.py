"""Trajectory-derived within-exposure smear sidecar helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .detector_layer_overrides import patch_smear_layer_for_policy
from .obs_subblock_trajectory import CanonicalTrajectory, SubblockTrajectory, write_rows_csv

SMEAR_TRUTH_FILENAME = "frame_smear_truth.csv"
SMEAR_MODEL_FILENAME = "frame_smear_model.csv"
SMEAR_PROVENANCE_FILENAME = "smear_provenance.json"
LINEAR_EXTRAPOLATE_EDGE_POLICIES = {
    "linear_extrapolate",
    "nearest_segment_extrapolate",
    "symmetric_linear_extrapolate",
}
SMEAR_FIELDNAMES: tuple[str, ...] = (
    "frame_index",
    "time_s",
    "exposure_start_s",
    "exposure_mid_s",
    "exposure_end_s",
    "smear_dx_as",
    "smear_dy_as",
    "smear_length_as",
    "smear_length_detector_pix",
    "smear_theta_deg",
    "smear_sigma_perp_detector_pix",
    "smear_kernel_size",
    "smear_enabled",
    "smear_source",
    "smear_policy",
    "notes",
)


@dataclass(frozen=True)
class SmearConfig:
    """Validated trajectory-smear configuration.

    Parameters
    ----------
    enabled
        Whether smear sidecars should be derived and written.
    exposure_time_s
        Exposure duration in seconds.
    exposure_interval
        Exposure alignment convention: ``centered``, ``start_aligned``, or
        ``end_aligned``. ``smear_theta_deg`` is measured counter-clockwise from
        detector +X toward detector +Y using the trajectory X/Y convention.
    edge_policy
        Out-of-domain exposure handling policy: ``error``, ``clamp``,
        ``drop``, or symmetric endpoint linear extrapolation.
    max_extrapolation_s
        Maximum leading/trailing extrapolation duration for linear endpoint
        extrapolation, or ``auto`` for the larger of one exposure duration and
        one nearest trajectory sample interval.
    plate_scale_as_per_pix
        Detector plate scale used to convert arcseconds to detector pixels.
    """

    enabled: bool = False
    exposure_time_s: float = 0.05
    exposure_interval: str = "centered"
    edge_policy: str = "error"
    max_extrapolation_s: float | str | None = "auto"
    plate_scale_as_per_pix: float | None = None
    truth_sigma_perp_detector_pix: float = 0.25
    truth_kernel_size: int = 9
    inference_mode: str = "matched"
    inference_length_scale: float = 1.0
    inference_theta_offset_deg: float = 0.0
    inference_min_length_detector_pix: float = 0.0
    inference_constant_length_detector_pix: float | None = None
    inference_constant_theta_deg: float | None = None
    render_mode: str = "metadata_only"
    render_representative: str = "median"
    render_layer_name: str = "trajectory_smear"
    render_apply_to: str = "truth"
    render_model_layer_policy: str = "from_inference_smear"
    source: str = "resolved_trajectory"
    angle_convention: str = "image_x_to_y_deg"
    raw_config: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if self.exposure_time_s < 0.0 or not math.isfinite(self.exposure_time_s):
            raise ValueError("trajectory_processing.smear.exposure.time_s must be finite and non-negative.")
        if self.exposure_interval not in {"centered", "start_aligned", "end_aligned"}:
            raise ValueError("trajectory_processing.smear.exposure.interval is unsupported.")
        if self.edge_policy not in {"error", "clamp", "drop", *LINEAR_EXTRAPOLATE_EDGE_POLICIES}:
            raise ValueError(
                "trajectory_processing.smear.exposure.edge_policy must be error, clamp, drop, "
                "symmetric_linear_extrapolate, linear_extrapolate, or nearest_segment_extrapolate."
            )
        if isinstance(self.max_extrapolation_s, str):
            if self.max_extrapolation_s != "auto":
                raise ValueError("trajectory_processing.smear.exposure.max_extrapolation_s must be auto or non-negative.")
        elif self.max_extrapolation_s is not None:
            max_extrap = float(self.max_extrapolation_s)
            if max_extrap < 0.0 or not math.isfinite(max_extrap):
                raise ValueError("trajectory_processing.smear.exposure.max_extrapolation_s must be auto or non-negative.")
        if self.angle_convention != "image_x_to_y_deg":
            raise ValueError("Only angle_convention='image_x_to_y_deg' is supported.")
        if self.enabled and (self.plate_scale_as_per_pix is None or self.plate_scale_as_per_pix <= 0.0):
            raise ValueError("Smear detector-pixel conversion requires a positive plate_scale_as_per_pix.")
        if self.truth_sigma_perp_detector_pix <= 0.0:
            raise ValueError("smear.truth.sigma_perp_detector_pix must be positive.")
        if self.truth_kernel_size <= 0 or self.truth_kernel_size % 2 == 0:
            raise ValueError("smear.truth.kernel_size must be a positive odd integer.")
        if self.inference_mode not in {
            "matched",
            "matched_subblock_constant",
            "scaled",
            "angle_offset",
            "disabled",
            "constant",
        }:
            raise ValueError("Unsupported smear.inference.mode.")
        if self.render_mode == "per_frame":
            raise ValueError("smear.render.mode='per_frame' is future/deferred and not implemented.")
        if self.render_mode not in {"disabled", "none", "metadata_only", "subblock_constant_layer"}:
            raise ValueError("smear.render.mode must be disabled, metadata_only, or subblock_constant_layer.")
        if self.render_representative not in {"median", "mean", "rms", "max"}:
            raise ValueError("Unsupported smear.render.representative.")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def parse_smear_config(
    cfg: Mapping[str, Any] | None,
    *,
    exposure_time_s: float,
    plate_scale_as_per_pix: float | None,
) -> SmearConfig:
    """Return a validated smear configuration from a trajectory-processing block."""

    root = _as_mapping(cfg)
    smear = _as_mapping(root.get("smear"))
    exposure = _as_mapping(smear.get("exposure"))
    truth = _as_mapping(smear.get("truth"))
    inference = _as_mapping(smear.get("inference"))
    conversion = _as_mapping(smear.get("coordinate_conversion"))
    render = _as_mapping(smear.get("render"))
    raw_time = exposure.get("time_s", "from_subblocks")
    resolved_exposure = float(exposure_time_s if raw_time in (None, "from_subblocks") else raw_time)
    parsed = SmearConfig(
        enabled=bool(smear.get("enabled", False)),
        exposure_time_s=resolved_exposure,
        exposure_interval=str(exposure.get("interval", "centered")),
        edge_policy=str(exposure.get("edge_policy", smear.get("edge_policy", "error"))),
        max_extrapolation_s=exposure.get("max_extrapolation_s", smear.get("max_extrapolation_s", "auto")),
        plate_scale_as_per_pix=plate_scale_as_per_pix,
        truth_sigma_perp_detector_pix=float(truth.get("sigma_perp_detector_pix", 0.25)),
        truth_kernel_size=int(truth.get("kernel_size", 9)),
        inference_mode=str(inference.get("mode", "matched_subblock_constant")),
        inference_length_scale=float(inference.get("length_scale", 1.0)),
        inference_theta_offset_deg=float(inference.get("theta_offset_deg", 0.0)),
        inference_min_length_detector_pix=float(inference.get("min_length_detector_pix", 0.0)),
        inference_constant_length_detector_pix=(
            None if inference.get("constant_length_detector_pix") is None else float(inference["constant_length_detector_pix"])
        ),
        inference_constant_theta_deg=(
            None if inference.get("constant_theta_deg") is None else float(inference["constant_theta_deg"])
        ),
        render_mode=str(render.get("mode", "metadata_only" if bool(smear.get("enabled", False)) else "disabled")),
        render_representative=str(render.get("representative", "median")),
        render_layer_name=str(render.get("target_layer", render.get("layer_name", "smear"))),
        render_apply_to=str(render.get("apply_to", "truth")),
        render_model_layer_policy=str(render.get("model_layer_policy", "from_inference_smear")),
        source=str(truth.get("source", "resolved_trajectory")),
        angle_convention=str(conversion.get("angle_convention", "image_x_to_y_deg")),
        raw_config=root,
    )
    parsed.validate()
    return parsed


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exposure_bounds(time_s: float, cfg: SmearConfig) -> tuple[float, float]:
    dt = float(cfg.exposure_time_s)
    if cfg.exposure_interval == "centered":
        return float(time_s - 0.5 * dt), float(time_s + 0.5 * dt)
    if cfg.exposure_interval == "start_aligned":
        return float(time_s), float(time_s + dt)
    return float(time_s - dt), float(time_s)


@dataclass(frozen=True)
class _EdgeDecision:
    start_eval_s: float
    end_eval_s: float
    leading_extrapolation_s: float = 0.0
    trailing_extrapolation_s: float = 0.0
    note: str = ""


def _nearest_sample_interval_s(times: np.ndarray, *, leading: bool) -> float:
    if times.size < 2:
        raise ValueError("Endpoint linear extrapolation requires at least two trajectory samples.")
    if leading:
        return float(times[1] - times[0])
    return float(times[-1] - times[-2])


def _resolved_max_extrapolation_s(times: np.ndarray, cfg: SmearConfig, *, leading: bool) -> float:
    if cfg.max_extrapolation_s in (None, "auto"):
        return max(float(cfg.exposure_time_s), _nearest_sample_interval_s(times, leading=leading))
    return float(cfg.max_extrapolation_s)


def _apply_edge_policy(
    start: float,
    end: float,
    domain: tuple[float, float],
    cfg: SmearConfig,
    *,
    trajectory_time_s: np.ndarray | None = None,
) -> _EdgeDecision | None:
    lo, hi = domain
    if start >= lo - 1.0e-12 and end <= hi + 1.0e-12:
        return _EdgeDecision(max(start, lo), min(end, hi))
    if cfg.edge_policy == "drop":
        return None
    if cfg.edge_policy == "clamp":
        return _EdgeDecision(max(start, lo), min(end, hi), note="edge_policy_clamp")
    if cfg.edge_policy in LINEAR_EXTRAPOLATE_EDGE_POLICIES:
        if trajectory_time_s is None:
            raise ValueError("Endpoint linear extrapolation requires trajectory sample times.")
        times = np.asarray(trajectory_time_s, dtype=float)
        leading = max(0.0, lo - start)
        trailing = max(0.0, end - hi)
        if end < lo - 1.0e-12 or start > hi + 1.0e-12:
            raise ValueError(
                "Exposure interval does not overlap trajectory domain and cannot be endpoint-extrapolated: "
                f"exposure=[{start}, {end}], domain=[{lo}, {hi}]."
            )
        max_leading = _resolved_max_extrapolation_s(times, cfg, leading=True)
        max_trailing = _resolved_max_extrapolation_s(times, cfg, leading=False)
        if leading > max_leading + 1.0e-12 or trailing > max_trailing + 1.0e-12:
            raise ValueError(
                "Exposure interval requires endpoint extrapolation beyond the allowed margin: "
                f"exposure=[{start}, {end}], domain=[{lo}, {hi}], "
                f"leading_extrapolation_s={leading}, trailing_extrapolation_s={trailing}, "
                f"max_leading_extrapolation_s={max_leading}, max_trailing_extrapolation_s={max_trailing}."
            )
        note_parts: list[str] = []
        if leading > 0.0:
            note_parts.append(f"leading={leading:.12g}s")
        if trailing > 0.0:
            note_parts.append(f"trailing={trailing:.12g}s")
        note = (
            f"endpoint_linear_extrapolated({', '.join(note_parts)})"
            if note_parts
            else ""
        )
        return _EdgeDecision(start, end, leading_extrapolation_s=leading, trailing_extrapolation_s=trailing, note=note)
    raise ValueError(
        "Exposure interval falls outside trajectory domain: "
        f"exposure=[{start}, {end}], domain=[{lo}, {hi}]."
    )


def _interp_with_endpoint_linear_extrapolation(
    time_s: float,
    trajectory_time_s: np.ndarray,
    values: np.ndarray,
) -> float:
    t = float(time_s)
    times = np.asarray(trajectory_time_s, dtype=float)
    series = np.asarray(values, dtype=float)
    if t < float(times[0]):
        dt = float(times[1] - times[0])
        return float(series[0] + (series[1] - series[0]) / dt * (t - times[0]))
    if t > float(times[-1]):
        dt = float(times[-1] - times[-2])
        return float(series[-1] + (series[-1] - series[-2]) / dt * (t - times[-1]))
    return float(np.interp(t, times, series))


def derive_truth_smear_rows(
    trajectory: CanonicalTrajectory,
    block: SubblockTrajectory,
    cfg: SmearConfig,
) -> list[dict[str, Any]]:
    """Derive per-frame truth smear rows from the resolved trajectory.

    The X/Y displacement is end-minus-start in arcseconds. The detector line
    angle is counter-clockwise from +X toward +Y in degrees.
    """

    if "source.x_position_as" not in trajectory.values or "source.y_position_as" not in trajectory.values:
        raise ValueError("Trajectory smear requires source.x_position_as and source.y_position_as.")
    if np.any(np.diff(np.asarray(trajectory.time_s, dtype=float)) <= 0.0):
        raise ValueError("Trajectory times must be strictly increasing for smear derivation.")
    cfg.validate()
    rows: list[dict[str, Any]] = []
    trajectory_time_s = np.asarray(trajectory.time_s, dtype=float)
    domain = (float(trajectory_time_s[0]), float(trajectory_time_s[-1]))
    x = np.asarray(trajectory.values["source.x_position_as"], dtype=float)
    y = np.asarray(trajectory.values["source.y_position_as"], dtype=float)
    assert cfg.plate_scale_as_per_pix is not None
    for frame_index, mid in enumerate(np.asarray(block.frame_times_s, dtype=float)):
        start, end = _exposure_bounds(float(mid), cfg)
        edge = _apply_edge_policy(start, end, domain, cfg, trajectory_time_s=trajectory_time_s)
        if edge is None:
            rows.append(_disabled_row(frame_index, float(mid), start, end, cfg, "edge_policy_drop"))
            continue
        start_eval = edge.start_eval_s
        end_eval = edge.end_eval_s
        x_start = _interp_with_endpoint_linear_extrapolation(start_eval, trajectory_time_s, x)
        x_end = _interp_with_endpoint_linear_extrapolation(end_eval, trajectory_time_s, x)
        y_start = _interp_with_endpoint_linear_extrapolation(start_eval, trajectory_time_s, y)
        y_end = _interp_with_endpoint_linear_extrapolation(end_eval, trajectory_time_s, y)
        dx = x_end - x_start
        dy = y_end - y_start
        length_as = float(math.hypot(dx, dy))
        length_pix = length_as / float(cfg.plate_scale_as_per_pix)
        theta = 0.0 if length_as == 0.0 else float(math.degrees(math.atan2(dy, dx)))
        rows.append(
            {
                "frame_index": int(frame_index),
                "time_s": float(mid),
                "exposure_start_s": float(start_eval),
                "exposure_mid_s": float(mid),
                "exposure_end_s": float(end_eval),
                "smear_dx_as": float(dx),
                "smear_dy_as": float(dy),
                "smear_length_as": length_as,
                "smear_length_detector_pix": float(length_pix),
                "smear_theta_deg": theta,
                "smear_sigma_perp_detector_pix": float(cfg.truth_sigma_perp_detector_pix),
                "smear_kernel_size": int(cfg.truth_kernel_size),
                "smear_enabled": True,
                "smear_source": cfg.source,
                "smear_policy": "truth",
                "notes": edge.note,
                "_edge_policy": cfg.edge_policy,
                "_edge_leading_extrapolation_s": float(edge.leading_extrapolation_s),
                "_edge_trailing_extrapolation_s": float(edge.trailing_extrapolation_s),
                "_trajectory_domain_start_s": domain[0],
                "_trajectory_domain_end_s": domain[1],
            }
        )
    return rows


def _disabled_row(frame_index: int, mid: float, start: float, end: float, cfg: SmearConfig, note: str) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "time_s": float(mid),
        "exposure_start_s": float(start),
        "exposure_mid_s": float(mid),
        "exposure_end_s": float(end),
        "smear_dx_as": 0.0,
        "smear_dy_as": 0.0,
        "smear_length_as": 0.0,
        "smear_length_detector_pix": 0.0,
        "smear_theta_deg": 0.0,
        "smear_sigma_perp_detector_pix": float(cfg.truth_sigma_perp_detector_pix),
        "smear_kernel_size": int(cfg.truth_kernel_size),
        "smear_enabled": False,
        "smear_source": cfg.source,
        "smear_policy": "disabled",
        "notes": note,
    }


def derive_model_smear_rows(truth_rows: Sequence[Mapping[str, Any]], cfg: SmearConfig) -> list[dict[str, Any]]:
    """Derive model/inference smear rows from truth rows and mismatch policy."""

    rows: list[dict[str, Any]] = []
    for row in truth_rows:
        model = dict(row)
        if cfg.inference_mode == "disabled":
            model.update(
                {
                    "smear_dx_as": 0.0,
                    "smear_dy_as": 0.0,
                    "smear_length_as": 0.0,
                    "smear_length_detector_pix": 0.0,
                    "smear_enabled": False,
                    "smear_policy": "inference_disabled",
                    "notes": "model smear disabled/unmodeled",
                }
            )
        elif cfg.inference_mode in {"matched", "matched_subblock_constant"}:
            model["smear_policy"] = "inference_matched"
        else:
            length_pix = float(model["smear_length_detector_pix"])
            theta = float(model["smear_theta_deg"])
            if cfg.inference_mode == "scaled":
                length_pix *= float(cfg.inference_length_scale)
            elif cfg.inference_mode == "angle_offset":
                theta += float(cfg.inference_theta_offset_deg)
            elif cfg.inference_mode == "constant":
                if cfg.inference_constant_length_detector_pix is None or cfg.inference_constant_theta_deg is None:
                    raise ValueError("smear.inference.constant requires constant length and theta.")
                length_pix = float(cfg.inference_constant_length_detector_pix)
                theta = float(cfg.inference_constant_theta_deg)
            length_pix = max(length_pix, float(cfg.inference_min_length_detector_pix))
            length_as = length_pix * float(cfg.plate_scale_as_per_pix or 1.0)
            radians = math.radians(theta)
            model.update(
                {
                    "smear_dx_as": float(length_as * math.cos(radians)),
                    "smear_dy_as": float(length_as * math.sin(radians)),
                    "smear_length_as": float(length_as),
                    "smear_length_detector_pix": float(length_pix),
                    "smear_theta_deg": float(theta),
                    "smear_policy": f"inference_{cfg.inference_mode}",
                    "notes": "model smear mismatch applied",
                }
            )
        rows.append(model)
    return rows


def summarize_smear_rows(rows: Sequence[Mapping[str, Any]], *, prefix: str) -> dict[str, Any]:
    lengths = np.asarray([float(row.get("smear_length_detector_pix", 0.0)) for row in rows], dtype=float)
    enabled = np.asarray([str(row.get("smear_enabled", "")).lower() == "true" or row.get("smear_enabled") is True for row in rows])
    nonzero = lengths[lengths > 0.0]
    return {
        f"{prefix}_length_pix_median": float(np.median(nonzero)) if nonzero.size else 0.0,
        f"{prefix}_length_pix_max": float(np.max(lengths)) if lengths.size else 0.0,
        f"{prefix}_enabled_frame_count": int(np.count_nonzero(enabled)),
    }


def summarize_endpoint_extrapolation(
    rows: Sequence[Mapping[str, Any]],
    *,
    trajectory: CanonicalTrajectory,
    cfg: SmearConfig,
) -> dict[str, Any]:
    times = np.asarray(trajectory.time_s, dtype=float)
    domain = (float(times[0]), float(times[-1]))
    leading_values = [float(row.get("_edge_leading_extrapolation_s", 0.0) or 0.0) for row in rows]
    trailing_values = [float(row.get("_edge_trailing_extrapolation_s", 0.0) or 0.0) for row in rows]
    exposure_rows = []
    for row, leading, trailing in zip(rows, leading_values, trailing_values):
        if leading <= 0.0 and trailing <= 0.0:
            continue
        exposure_rows.append(
            {
                "frame_index": int(row.get("frame_index", -1)),
                "exposure_start_s": float(row.get("exposure_start_s", 0.0)),
                "exposure_mid_s": float(row.get("exposure_mid_s", row.get("time_s", 0.0))),
                "exposure_end_s": float(row.get("exposure_end_s", 0.0)),
                "leading_extrapolation_s": leading,
                "trailing_extrapolation_s": trailing,
            }
        )
    leading_max = _resolved_max_extrapolation_s(times, cfg, leading=True) if cfg.edge_policy in LINEAR_EXTRAPOLATE_EDGE_POLICIES else None
    trailing_max = _resolved_max_extrapolation_s(times, cfg, leading=False) if cfg.edge_policy in LINEAR_EXTRAPOLATE_EDGE_POLICIES else None
    return {
        "edge_policy": cfg.edge_policy,
        "max_extrapolation_s": cfg.max_extrapolation_s,
        "resolved_max_leading_extrapolation_s": leading_max,
        "resolved_max_trailing_extrapolation_s": trailing_max,
        "used": bool(exposure_rows),
        "leading_used": any(value > 0.0 for value in leading_values),
        "trailing_used": any(value > 0.0 for value in trailing_values),
        "leading_extrapolation_s_max": max(leading_values) if leading_values else 0.0,
        "trailing_extrapolation_s_max": max(trailing_values) if trailing_values else 0.0,
        "trajectory_domain_start_s": domain[0],
        "trajectory_domain_end_s": domain[1],
        "exposures": exposure_rows,
        "note": (
            "Endpoint trajectory values were linearly extrapolated from the nearest trajectory segment."
            if exposure_rows
            else ""
        ),
    }


def representative_line_kernel(rows: Sequence[Mapping[str, Any]], *, representative: str = "median") -> dict[str, Any]:
    """Return a deterministic representative line-kernel config for a subblock."""

    if not rows:
        raise ValueError("Cannot derive representative smear kernel from empty rows.")
    lengths_all = np.asarray([float(row.get("smear_length_detector_pix", 0.0)) for row in rows], dtype=float)
    lengths = lengths_all[lengths_all > 0.0]
    if lengths.size == 0:
        length = 1.0e-12
    elif representative == "mean":
        length = float(np.mean(lengths))
    elif representative == "rms":
        length = float(np.sqrt(np.mean(np.square(lengths))))
    elif representative == "max":
        length = float(np.max(lengths))
    else:
        length = float(np.median(lengths))
    angles = np.asarray([float(row.get("smear_theta_deg", 0.0)) for row in rows if float(row.get("smear_length_detector_pix", 0.0)) > 0.0], dtype=float)
    if angles.size:
        theta = float(math.degrees(math.atan2(np.mean(np.sin(np.deg2rad(angles))), np.mean(np.cos(np.deg2rad(angles))))))
    else:
        theta = 0.0
    first = rows[0]
    return {
        "kind": "line",
        "length": float(length),
        "sigma_perp": float(first.get("smear_sigma_perp_detector_pix", 0.25)),
        "theta_deg": theta,
        "kernel_size": int(first.get("smear_kernel_size", 9)),
        "units": "detector_pix",
        "representative": representative,
    }


def subblock_constant_line_kernel_from_fit(block: SubblockTrajectory, cfg: SmearConfig) -> dict[str, Any]:
    """Return a line-kernel from the subblock linear X/Y fit and one exposure.

    This is the full-fidelity rendered-smear policy: fit over the subblock, but
    scale the detector line length to a single rendered frame exposure.
    """

    if cfg.plate_scale_as_per_pix is None or cfg.plate_scale_as_per_pix <= 0.0:
        raise ValueError("subblock_constant_layer requires a positive plate_scale_as_per_pix.")
    try:
        slope_x = float(block.fit_coefficients["source.x_position_as"][1])
        slope_y = float(block.fit_coefficients["source.y_position_as"][1])
    except KeyError as exc:
        raise ValueError(
            "subblock_constant_layer requires fitted source.x_position_as and source.y_position_as."
        ) from exc
    dx_as = slope_x * float(cfg.exposure_time_s)
    dy_as = slope_y * float(cfg.exposure_time_s)
    dx_pix = dx_as / float(cfg.plate_scale_as_per_pix)
    dy_pix = dy_as / float(cfg.plate_scale_as_per_pix)
    length_pix = float(math.hypot(dx_pix, dy_pix))
    theta = 0.0 if length_pix == 0.0 else float(math.degrees(math.atan2(dy_pix, dx_pix)))
    return {
        "kind": "line",
        "length": length_pix,
        "sigma_perp": float(cfg.truth_sigma_perp_detector_pix),
        "theta_deg": theta,
        "kernel_size": int(cfg.truth_kernel_size),
        "units": "detector_pix",
        "source": "subblock_linear_fit_one_frame_exposure",
        "slope_x_as_per_s": slope_x,
        "slope_y_as_per_s": slope_y,
        "dx_frame_as": float(dx_as),
        "dy_frame_as": float(dy_as),
        "dx_frame_pix": float(dx_pix),
        "dy_frame_pix": float(dy_pix),
        "exposure_time_s": float(cfg.exposure_time_s),
        "plate_scale_as_per_pix": float(cfg.plate_scale_as_per_pix),
    }


def write_smear_sidecars(
    *,
    outdir: Path,
    trajectory: CanonicalTrajectory,
    block: SubblockTrajectory,
    cfg: SmearConfig,
    processing_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write truth/model smear sidecars and provenance for one subblock."""

    truth_rows = derive_truth_smear_rows(trajectory, block, cfg)
    model_rows = derive_model_smear_rows(truth_rows, cfg)
    truth_path = (outdir / SMEAR_TRUTH_FILENAME).resolve()
    model_path = (outdir / SMEAR_MODEL_FILENAME).resolve()
    provenance_path = (outdir / SMEAR_PROVENANCE_FILENAME).resolve()
    write_rows_csv(truth_path, truth_rows, SMEAR_FIELDNAMES)
    write_rows_csv(model_path, model_rows, SMEAR_FIELDNAMES)
    representative = (
        subblock_constant_line_kernel_from_fit(block, cfg)
        if cfg.render_mode == "subblock_constant_layer"
        else representative_line_kernel(truth_rows, representative=cfg.render_representative)
    )
    model_representative = (
        dict(representative)
        if cfg.inference_mode in {"matched", "matched_subblock_constant"}
        else representative_line_kernel(model_rows, representative=cfg.render_representative)
    )
    matched_model = (
        float(representative.get("length", 0.0)) == float(model_representative.get("length", 0.0))
        and float(representative.get("theta_deg", 0.0)) == float(model_representative.get("theta_deg", 0.0))
    )
    provenance = {
        "schema_version": "trajectory_smear_provenance.v1",
        "subblock_index": int(block.subblock_index),
        "window_index": (processing_context or {}).get("window_index"),
        "render_mode": cfg.render_mode,
        "inference_mode": cfg.inference_mode,
        "target_layer": cfg.render_layer_name,
        "source": representative.get("source", cfg.source),
        "exposure_time_s": float(cfg.exposure_time_s),
        "plate_scale_as_per_pix": cfg.plate_scale_as_per_pix,
        "representative_kernel": representative,
        "truth_kernel": representative,
        "model_kernel": model_representative,
        "matched_model": bool(matched_model),
        "pa_smear_mode": "ignored_for_line_kernel",
        "input_frame_truth_csv": str((processing_context or {}).get("input_frame_truth_csv", "")),
        "frame_smear_truth_csv": str(truth_path),
        "frame_smear_model_csv": str(model_path),
        "render_template_path": str((processing_context or {}).get("render_template_path", "")),
        "inference_template_path": str((processing_context or {}).get("inference_template_path", "")),
        "warnings": [],
        "source_trajectory_path": str(trajectory.raw.source_path),
        "source_trajectory_sha256": _file_sha256(trajectory.raw.source_path),
        "source_kind": trajectory.raw.source_kind,
        "processing_mode": "resolved_trajectory_sidecar",
        "exposure": {
            "time_s": float(cfg.exposure_time_s),
            "interval": cfg.exposure_interval,
            "edge_policy": cfg.edge_policy,
            "max_extrapolation_s": cfg.max_extrapolation_s,
        },
        "interpolation_policy": "linear_with_endpoint_extrapolation"
        if cfg.edge_policy in LINEAR_EXTRAPOLATE_EDGE_POLICIES
        else "linear",
        "endpoint_extrapolation": summarize_endpoint_extrapolation(
            truth_rows,
            trajectory=trajectory,
            cfg=cfg,
        ),
        "plate_scale": {
            "source": "system_config_or_geometry",
            "key": "optics.plate_scale_as_per_pix",
            "as_per_pix": cfg.plate_scale_as_per_pix,
        },
        "mismatch_policy": {"mode": cfg.inference_mode},
        "render": {
            "mode": cfg.render_mode,
            "representative": cfg.render_representative,
            "layer_name": cfg.render_layer_name,
            "representative_kernel": representative,
            "pa_smear_mode": "ignored_for_line_kernel",
            "per_frame_dynamic_kernels_deferred": True,
        },
        "truth_summary": summarize_smear_rows(truth_rows, prefix="smear_truth"),
        "model_summary": summarize_smear_rows(model_rows, prefix="smear_model"),
        "config": dict(cfg.raw_config or {}),
        "processing_context": dict(processing_context or {}),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "smear_truth_csv": truth_path,
        "smear_model_csv": model_path,
        "smear_provenance_json": provenance_path,
        "truth_rows": truth_rows,
        "model_rows": model_rows,
        "representative_kernel": representative,
        "provenance": provenance,
        **summarize_smear_rows(truth_rows, prefix="smear_truth"),
        **summarize_smear_rows(model_rows, prefix="smear_model"),
    }


def inject_subblock_smear_layer(render_cfg: dict[str, Any], *, cfg: SmearConfig, representative_kernel: Mapping[str, Any]) -> dict[str, Any]:
    """Patch the configured smear layer, injecting only when explicitly allowed."""

    smear_raw = cfg.raw_config.get("smear", {}) if isinstance(cfg.raw_config, Mapping) else {}
    render_raw = smear_raw.get("render", {}) if isinstance(smear_raw, Mapping) else {}
    allow_injection = bool(render_raw.get("allow_layer_injection", False)) if isinstance(render_raw, Mapping) else False
    require_existing = bool(render_raw.get("require_existing_layer", True)) if isinstance(render_raw, Mapping) else True
    system = render_cfg.get("system") if isinstance(render_cfg.get("system"), Mapping) else render_cfg
    if isinstance(system, Mapping):
        try:
            patched_system, _ = patch_smear_layer_for_policy(
                system,
                smear_raw if isinstance(smear_raw, Mapping) else {"enabled": cfg.enabled, "render": {"mode": cfg.render_mode}},
                representative_kernel=representative_kernel,
                context="trajectory_subblock.render",
                strict=True,
            )
            if "system" in render_cfg:
                render_cfg["system"] = patched_system
            else:
                render_cfg.update(patched_system)
            return render_cfg
        except ValueError:
            if require_existing or not allow_injection:
                raise
    system = render_cfg.setdefault("system", {})
    detector = system.setdefault("detector", {})
    layers = detector.setdefault("layers", [])
    if not isinstance(layers, list):
        raise ValueError("render config system.detector.layers must be a list to inject smear layer.")
    if not allow_injection:
        raise ValueError("smear.render.allow_layer_injection must be true to insert a new smear layer.")
    layer = {
        "name": cfg.render_layer_name,
        "kind": "ApplyConvolution",
        "kernel": {
            "kind": "line",
            "length": float(representative_kernel["length"]),
            "sigma_perp": float(representative_kernel["sigma_perp"]),
            "theta_deg": float(representative_kernel["theta_deg"]),
            "kernel_size": int(representative_kernel["kernel_size"]),
            "units": "detector_pix",
        },
    }
    layers[:] = [item for item in layers if not (isinstance(item, Mapping) and item.get("name") == cfg.render_layer_name)]
    layers.append(layer)
    return render_cfg


__all__ = [
    "SMEAR_MODEL_FILENAME",
    "SMEAR_PROVENANCE_FILENAME",
    "SMEAR_TRUTH_FILENAME",
    "SmearConfig",
    "derive_model_smear_rows",
    "derive_truth_smear_rows",
    "inject_subblock_smear_layer",
    "parse_smear_config",
    "representative_line_kernel",
    "summarize_endpoint_extrapolation",
    "subblock_constant_line_kernel_from_fit",
    "summarize_smear_rows",
    "write_smear_sidecars",
]
