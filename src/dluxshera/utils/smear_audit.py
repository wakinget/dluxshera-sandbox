"""Audit helpers for full-fidelity subblock smear templates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_LENGTH_TOL_PIX = 1.0e-12
DEFAULT_THETA_TOL_DEG = 1.0e-9
DEFAULT_PLACEHOLDER_THRESHOLD_PIX = 1.0e-10


def _as_path(value: Any, *, run_root: Path | None = None) -> Path:
    path = Path(str(value))
    if not path.is_absolute() and run_root is not None:
        path = run_root / path
    return path


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def find_named_detector_layer(payload: Mapping[str, Any], *, layer_name: str = "smear") -> Mapping[str, Any] | None:
    """Return the first recursively nested detector layer with ``name == layer_name``."""

    for item in _walk_mappings(payload):
        if item.get("name") == layer_name and isinstance(item.get("kernel"), Mapping):
            return item
    return None


def load_named_smear_kernel(path: Path, *, layer_name: str = "smear") -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layer = find_named_detector_layer(payload, layer_name=layer_name)
    if layer is None:
        raise ValueError(f"Template {path} does not contain named detector layer {layer_name!r}.")
    kernel = layer.get("kernel")
    if not isinstance(kernel, Mapping):
        raise ValueError(f"Template {path} layer {layer_name!r} does not contain a kernel mapping.")
    return dict(kernel)


def _angle_delta_deg(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def kernel_matches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    length_tol_pix: float = DEFAULT_LENGTH_TOL_PIX,
    theta_tol_deg: float = DEFAULT_THETA_TOL_DEG,
) -> bool:
    return (
        abs(float(expected.get("length", 0.0)) - float(actual.get("length", 0.0))) <= length_tol_pix
        and _angle_delta_deg(float(expected.get("theta_deg", 0.0)), float(actual.get("theta_deg", 0.0))) <= theta_tol_deg
    )


def _status_for_row(
    *,
    expected: Mapping[str, Any],
    render_kernel: Mapping[str, Any] | None,
    inference_kernel: Mapping[str, Any] | None,
    render_match: bool,
    inference_match: bool,
    length_tol_pix: float,
    theta_tol_deg: float,
    placeholder_threshold_pix: float,
) -> tuple[str, list[str], bool, str]:
    messages: list[str] = []
    expected_length = float(expected.get("length", 0.0))
    near_zero = expected_length <= placeholder_threshold_pix
    near_zero_reason = "trajectory_fit_length_below_threshold" if near_zero else ""
    if render_kernel is None:
        messages.append("missing_render_smear_layer")
    elif not render_match:
        messages.append(
            "render_mismatch"
            f": length_delta={abs(expected_length - float(render_kernel.get('length', 0.0))):.6e}"
            f", theta_delta={_angle_delta_deg(float(expected.get('theta_deg', 0.0)), float(render_kernel.get('theta_deg', 0.0))):.6e}"
        )
    if inference_kernel is None:
        messages.append("missing_inference_smear_layer")
    elif not inference_match:
        messages.append(
            "inference_mismatch"
            f": length_delta={abs(expected_length - float(inference_kernel.get('length', 0.0))):.6e}"
            f", theta_delta={_angle_delta_deg(float(expected.get('theta_deg', 0.0)), float(inference_kernel.get('theta_deg', 0.0))):.6e}"
        )
    if expected_length > placeholder_threshold_pix:
        for label, kernel in (("render", render_kernel), ("inference", inference_kernel)):
            if kernel is not None and float(kernel.get("length", 0.0)) <= placeholder_threshold_pix:
                messages.append(f"{label}_placeholder_leak")
    status = "ok" if not messages else ";".join(messages)
    return status, messages, near_zero, near_zero_reason


def _unique_smear_rows(rows: Sequence[Mapping[str, Any]], *, strict: bool) -> list[Mapping[str, Any]]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not row.get("smear_representative_kernel_json"):
            continue
        key = str(row.get("global_subblock_index", row.get("subblock_index", row.get("smear_provenance_json", ""))))
        fingerprint_keys = (
            "smear_representative_kernel_json",
            "smear_provenance_json",
            "render_template_path",
            "inference_template_path",
            "smear_render_mode",
            "smear_model_policy",
        )
        existing_fingerprint = {name: unique[key].get(name) for name in fingerprint_keys} if key in unique else {}
        fingerprint = {name: row.get(name) for name in fingerprint_keys}
        if key in unique and existing_fingerprint != fingerprint:
            message = f"Duplicate smear row for subblock key {key!r} differs across plan entries."
            if strict:
                raise ValueError(message)
        unique.setdefault(key, row)
    return [unique[key] for key in sorted(unique, key=lambda item: int(item) if str(item).isdigit() else str(item))]


def build_smear_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    run_root: Path | None = None,
    strict: bool = False,
    length_tol_pix: float = DEFAULT_LENGTH_TOL_PIX,
    theta_tol_deg: float = DEFAULT_THETA_TOL_DEG,
    placeholder_threshold_pix: float = DEFAULT_PLACEHOLDER_THRESHOLD_PIX,
) -> list[dict[str, Any]]:
    """Build one audit row per unique subblock with a representative smear kernel."""

    unique_rows = _unique_smear_rows(rows, strict=strict)
    if strict and not unique_rows:
        raise ValueError("No subblock rows with smear_representative_kernel_json were found.")
    summary: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in unique_rows:
        expected = json.loads(str(row["smear_representative_kernel_json"]))
        target_layer = str(row.get("smear_layer_name") or row.get("target_layer") or "smear")
        render_path = _as_path(row.get("render_template_path", ""), run_root=run_root)
        inference_path = _as_path(row.get("inference_template_path", ""), run_root=run_root)
        render_kernel: dict[str, Any] | None = None
        inference_kernel: dict[str, Any] | None = None
        row_failures: list[str] = []
        if not render_path.exists():
            row_failures.append(f"missing render template {render_path}")
        else:
            try:
                render_kernel = load_named_smear_kernel(render_path, layer_name=target_layer)
            except ValueError as exc:
                row_failures.append(str(exc))
        if not inference_path.exists():
            row_failures.append(f"missing inference template {inference_path}")
        else:
            try:
                inference_kernel = load_named_smear_kernel(inference_path, layer_name=target_layer)
            except ValueError as exc:
                row_failures.append(str(exc))
        render_match = render_kernel is not None and kernel_matches(
            expected,
            render_kernel,
            length_tol_pix=length_tol_pix,
            theta_tol_deg=theta_tol_deg,
        )
        inference_match = inference_kernel is not None and kernel_matches(
            expected,
            inference_kernel,
            length_tol_pix=length_tol_pix,
            theta_tol_deg=theta_tol_deg,
        )
        status, messages, near_zero, near_zero_reason = _status_for_row(
            expected=expected,
            render_kernel=render_kernel,
            inference_kernel=inference_kernel,
            render_match=render_match,
            inference_match=inference_match,
            length_tol_pix=length_tol_pix,
            theta_tol_deg=theta_tol_deg,
            placeholder_threshold_pix=placeholder_threshold_pix,
        )
        row_failures.extend(messages)
        if strict and row_failures:
            failures.append(f"subblock {row.get('subblock_index')}: " + "; ".join(row_failures))
        summary.append(
            {
                "subblock_index": int(row.get("subblock_index", 0)),
                "window_index": row.get("window_index", ""),
                "render_mode": row.get("smear_render_mode", ""),
                "inference_mode": row.get("smear_model_policy", ""),
                "source": expected.get("source", ""),
                "exposure_time_s": expected.get("exposure_time_s", ""),
                "plate_scale_as_per_pix": expected.get("plate_scale_as_per_pix", ""),
                "slope_x_as_per_s": expected.get("slope_x_as_per_s", ""),
                "slope_y_as_per_s": expected.get("slope_y_as_per_s", ""),
                "dx_frame_as": expected.get("dx_frame_as", ""),
                "dy_frame_as": expected.get("dy_frame_as", ""),
                "dx_frame_pix": expected.get("dx_frame_pix", ""),
                "dy_frame_pix": expected.get("dy_frame_pix", ""),
                "smear_length_pix": expected.get("length", ""),
                "smear_theta_deg": expected.get("theta_deg", ""),
                "render_template_smear_length_pix": "" if render_kernel is None else render_kernel.get("length", ""),
                "render_template_smear_theta_deg": "" if render_kernel is None else render_kernel.get("theta_deg", ""),
                "inference_template_smear_length_pix": "" if inference_kernel is None else inference_kernel.get("length", ""),
                "inference_template_smear_theta_deg": "" if inference_kernel is None else inference_kernel.get("theta_deg", ""),
                "render_match": bool(render_match),
                "inference_match": bool(inference_match),
                "near_zero_smear": bool(near_zero),
                "near_zero_reason": near_zero_reason,
                "template_status": status,
                "smear_provenance_json": row.get("smear_provenance_json", ""),
                "render_template_path": str(render_path),
                "inference_template_path": str(inference_path),
            }
        )
    if strict and failures:
        raise ValueError("Smear template audit failed: " + " | ".join(failures))
    return summary


def plan_smear_rows(plan_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    subblock_plan = plan_payload.get("subblock_plan", {})
    rows: list[dict[str, Any]] = []
    if isinstance(subblock_plan, Mapping):
        for case_rows in subblock_plan.values():
            if isinstance(case_rows, list):
                rows.extend(dict(row) for row in case_rows if isinstance(row, Mapping))
    elif isinstance(subblock_plan, list):
        rows.extend(dict(row) for row in subblock_plan if isinstance(row, Mapping))
    return rows


__all__ = [
    "DEFAULT_LENGTH_TOL_PIX",
    "DEFAULT_PLACEHOLDER_THRESHOLD_PIX",
    "DEFAULT_THETA_TOL_DEG",
    "build_smear_summary_rows",
    "find_named_detector_layer",
    "kernel_matches",
    "load_named_smear_kernel",
    "plan_smear_rows",
]
