"""Named detector-layer override helpers for resolved system configs."""

from __future__ import annotations

import copy
import warnings
from typing import Any, Mapping


def _layers(system_config: Mapping[str, Any]) -> list[Any]:
    detector = system_config.get("detector") if isinstance(system_config, Mapping) else None
    layers = detector.get("layers") if isinstance(detector, Mapping) else None
    if layers is None:
        return []
    if not isinstance(layers, list):
        raise ValueError("system.detector.layers must be a list for detector-layer overrides.")
    return layers


def detector_layer_stack(system_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compact ordered detector-layer summaries from a resolved system config."""

    out: list[dict[str, Any]] = []
    for index, layer in enumerate(_layers(system_config)):
        if not isinstance(layer, Mapping):
            out.append({"index": index, "name": None, "kind": None, "valid": False})
            continue
        summary = {
            "index": index,
            "name": layer.get("name"),
            "kind": layer.get("kind"),
            "valid": True,
        }
        kernel = layer.get("kernel")
        if isinstance(kernel, Mapping):
            summary["kernel"] = copy.deepcopy(dict(kernel))
        for key in ("dx_path", "dy_path", "prf_path", "kernel_size", "sigma"):
            if key in layer:
                summary[key] = layer[key]
        out.append(summary)
    return out


def get_detector_layer(system_config: Mapping[str, Any], name: str) -> dict[str, Any] | None:
    for layer in _layers(system_config):
        if isinstance(layer, dict) and layer.get("name") == name:
            return layer
    return None


def _merge_nested(dst: dict[str, Any], patch: Mapping[str, Any]) -> None:
    for key, value in patch.items():
        if key in {"action", "allow_missing"}:
            continue
        if isinstance(value, Mapping) and isinstance(dst.get(key), dict):
            _merge_nested(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


def apply_detector_layer_overrides(
    system_config: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
    *,
    context: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply named detector-layer updates/removals to a resolved system config.

    Supported actions are ``keep``, ``update``, ``remove``, and ``disable``.
    ``disable`` currently removes the named layer, because the detector factory
    does not require stable layer names.
    """

    patched = copy.deepcopy(dict(system_config))
    override_layers = {}
    if isinstance(overrides, Mapping):
        raw_layers = overrides.get("layers", {})
        if isinstance(raw_layers, Mapping):
            override_layers = dict(raw_layers)
    provenance: dict[str, Any] = {
        "context": context,
        "before": detector_layer_stack(patched),
        "applied": [],
        "after": None,
    }
    if not override_layers:
        provenance["after"] = detector_layer_stack(patched)
        return patched, provenance

    layers = _layers(patched)
    for name, spec in override_layers.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"detector_overrides.layers.{name} must be a mapping.")
        action = str(spec.get("action", "update")).lower()
        allow_missing = bool(spec.get("allow_missing", False))
        matches = [idx for idx, layer in enumerate(layers) if isinstance(layer, Mapping) and layer.get("name") == name]
        if not matches:
            if allow_missing:
                provenance["applied"].append({"layer": name, "action": action, "status": "missing_allowed"})
                continue
            where = f" in {context}" if context else ""
            raise ValueError(f"Detector layer override{where} references missing layer {name!r}.")
        if len(matches) > 1:
            raise ValueError(f"Detector layer override references non-unique layer name {name!r}.")
        index = matches[0]
        if action == "keep":
            provenance["applied"].append({"layer": name, "action": action, "status": "kept", "index": index})
            continue
        if action in {"remove", "disable"}:
            removed = layers.pop(index)
            provenance["applied"].append(
                {"layer": name, "action": action, "status": "removed", "index": index, "removed_kind": removed.get("kind")}
            )
            continue
        if action == "update":
            layer = layers[index]
            assert isinstance(layer, dict)
            before = copy.deepcopy(layer)
            _merge_nested(layer, spec)
            provenance["applied"].append(
                {"layer": name, "action": action, "status": "updated", "index": index, "before": before, "after": copy.deepcopy(layer)}
            )
            continue
        raise ValueError(f"Unsupported detector layer action {action!r} for layer {name!r}.")

    provenance["after"] = detector_layer_stack(patched)
    return patched, provenance


def _smear_mode(smear_cfg: Mapping[str, Any] | None) -> tuple[bool, str, Mapping[str, Any]]:
    smear = smear_cfg if isinstance(smear_cfg, Mapping) else {}
    render = smear.get("render", {}) if isinstance(smear.get("render"), Mapping) else {}
    enabled = bool(smear.get("enabled", False))
    mode = str(render.get("mode", "metadata_only" if enabled else "disabled"))
    return enabled, mode, render


def _active_smear_layer(system_config: Mapping[str, Any], layer_name: str) -> dict[str, Any] | None:
    layer = get_detector_layer(system_config, layer_name)
    if layer is None:
        return None
    kernel = layer.get("kernel") if isinstance(layer.get("kernel"), Mapping) else {}
    kind = str(kernel.get("kind", layer.get("kind", ""))).lower()
    length = float(kernel.get("length", 0.0) or 0.0)
    return layer if kind == "line" and abs(length) > 0.0 else None


def patch_smear_layer_for_policy(
    system_config: Mapping[str, Any],
    smear_cfg: Mapping[str, Any] | None,
    *,
    representative_kernel: Mapping[str, Any] | None = None,
    context: str = "",
    strict: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Patch/remove the named smear layer according to trajectory smear policy."""

    enabled, mode, render = _smear_mode(smear_cfg)
    target_layer = str(render.get("target_layer", render.get("layer_name", "smear")))
    provenance: dict[str, Any] = {
        "context": context,
        "enabled": enabled,
        "mode": mode,
        "target_layer": target_layer,
        "before": detector_layer_stack(system_config),
        "applied": None,
        "after": None,
        "warnings": [],
    }
    if (not enabled) or mode in {"disabled", "none", "metadata_only"}:
        patched, applied = apply_detector_layer_overrides(
            system_config,
            {"layers": {target_layer: {"action": "remove", "allow_missing": True}}},
            context=f"{context}.smear_disabled" if context else "smear_disabled",
        )
        provenance["applied"] = applied
        provenance["after"] = detector_layer_stack(patched)
        if _active_smear_layer(patched, target_layer) is not None:
            message = (
                f"Smear render mode {mode!r} must not leave active nonzero detector layer "
                f"{target_layer!r} in {context or 'system config'}."
            )
            provenance["warnings"].append(message)
            if strict:
                raise ValueError(message)
            warnings.warn(message, UserWarning, stacklevel=2)
        return patched, provenance

    if mode != "subblock_constant_layer":
        raise ValueError(f"Unsupported smear.render.mode {mode!r}.")

    require_existing = bool(render.get("require_existing_layer", True))
    allow_injection = bool(render.get("allow_layer_injection", False))
    layer = get_detector_layer(system_config, target_layer)
    if layer is None and require_existing and not allow_injection:
        raise ValueError(f"smear.render.mode=subblock_constant_layer requires existing detector layer {target_layer!r}.")
    if representative_kernel is None:
        defaults = render.get("defaults", {}) if isinstance(render.get("defaults"), Mapping) else {}
        representative_kernel = {
            "kind": "line",
            "length": float(defaults.get("length", 1.0e-12)),
            "theta_deg": float(defaults.get("theta_deg", 0.0)),
            "sigma_perp": float(defaults.get("sigma_perp", 0.1)),
            "kernel_size": int(defaults.get("kernel_size", 11)),
            "units": str(defaults.get("units", "detector_pix")),
        }
    defaults = render.get("defaults", {}) if isinstance(render.get("defaults"), Mapping) else {}
    kernel_patch = {
        "kind": "line",
        "length": float(representative_kernel["length"]),
        "theta_deg": float(representative_kernel["theta_deg"]),
        "sigma_perp": float(representative_kernel.get("sigma_perp", defaults.get("sigma_perp", 0.1))),
        "kernel_size": int(representative_kernel.get("kernel_size", defaults.get("kernel_size", 11))),
        "units": str(representative_kernel.get("units", defaults.get("units", "detector_pix"))),
    }
    patched, applied = apply_detector_layer_overrides(
        system_config,
        {"layers": {target_layer: {"action": "update", "kind": "ApplyConvolution", "kernel": kernel_patch}}},
        context=f"{context}.smear_patch" if context else "smear_patch",
    )
    provenance["applied"] = applied
    provenance["after"] = detector_layer_stack(patched)
    provenance["representative_kernel"] = kernel_patch
    return patched, provenance


def validate_no_accidental_default_smear(
    system_config: Mapping[str, Any],
    *,
    system_preset: str | None,
    smear_cfg: Mapping[str, Any] | None,
    strict: bool = True,
) -> list[str]:
    """Warn/fail if SHERA_FLIGHT_3P_CONV keeps active smear while render is off."""

    enabled, mode, render = _smear_mode(smear_cfg)
    target_layer = str(render.get("target_layer", render.get("layer_name", "smear")))
    if system_preset != "SHERA_FLIGHT_3P_CONV" or (enabled and mode not in {"disabled", "metadata_only", "none"}):
        return []
    if _active_smear_layer(system_config, target_layer) is None:
        return []
    message = (
        "SHERA_FLIGHT_3P_CONV contains a default nonzero smear layer, but "
        f"smear render mode is {mode!r}; remove or disable detector layer {target_layer!r}."
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, UserWarning, stacklevel=2)
    return [message]


def detector_blur_warnings(system_config: Mapping[str, Any], *, smear_cfg: Mapping[str, Any] | None = None) -> list[str]:
    warnings_out: list[str] = []
    layer = get_detector_layer(system_config, "jitter")
    if isinstance(layer, Mapping):
        kernel = layer.get("kernel") if isinstance(layer.get("kernel"), Mapping) else layer
        sigma_x = float(kernel.get("sigma_x", kernel.get("sigma", 0.0)) or 0.0)
        sigma_y = float(kernel.get("sigma_y", kernel.get("sigma", 0.0)) or 0.0)
        enabled, mode, _ = _smear_mode(smear_cfg)
        if (enabled or mode == "subblock_constant_layer") and (sigma_x > 0.05 or sigma_y > 0.05):
            warnings_out.append(
                "Detector jitter layer has sigma_x or sigma_y > 0.05 detector pixels while trajectory smear/frame truth is enabled; this may double-count pointing blur."
            )
    return warnings_out


__all__ = [
    "apply_detector_layer_overrides",
    "detector_blur_warnings",
    "detector_layer_stack",
    "get_detector_layer",
    "patch_smear_layer_for_policy",
    "validate_no_accidental_default_smear",
]
