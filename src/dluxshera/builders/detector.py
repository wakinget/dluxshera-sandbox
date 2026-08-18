"""Detector builder responsibilities (detector assembly and runtime wiring).

This module translates declarative detector blocks into runtime detector
objects. It owns:
  - config shape normalization for declarative detector blocks
  - calibration/path resolution for detector maps
  - detector-layer construction from declarative layers
  - detector contract construction (ParamSpec)
  - lightweight runtime patching for detector-layer bindings
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
import jax.numpy as jnp
import numpy as np

from ..components.detectors import (
    GSENSE2020BSI_SPEC,
    HWK4123_SPEC,
    DetectorSpec,
    SheraDetector,
)
from ..params.spec import ParamField, ParamSpec
from ..layers.detector_layers import ApplyConvolution, ApplyPixelOffsets
from dLux.layers.detector_layers import ApplyJitter, ApplyPixelResponse, Downsample
from ..utils.noise import apply_knowledge_error


# Runtime bindings for detectors are currently empty; supported detector runtime
# updates are handled explicitly in ``apply_runtime_bindings``.
DETECTOR_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()
SUPPORTED_DETECTOR_LAYER_KINDS: tuple[str, ...] = (
    "Downsample",
    "ApplyPixelOffsets",
    "ApplyPixelResponse",
    "ApplyJitter",
    "ApplyConvolution",
)


# ---------------------------------------------------------------------------
# Config + path utilities
# ---------------------------------------------------------------------------
def _cfg_get(root, path: str, default=None):
    """Read a dotted config path from a mapping- or attribute-based config object."""
    cur = root
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, Mapping):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur

# ---------------------------------------------------------------------------
# Map conditioning (shared by declarative layers)
# ---------------------------------------------------------------------------

def _center_crop(arr: jnp.ndarray, target_size: int) -> jnp.ndarray:
    """Center-crop a square detector map to ``target_size`` pixels."""
    start = (arr.shape[0] - target_size) // 2
    end = start + target_size
    return arr[start:end, start:end]


def _center_pad_reflect(arr: jnp.ndarray, target_size: int) -> jnp.ndarray:
    """Center-pad a square detector map to ``target_size`` pixels with reflect mode."""
    pad_total = target_size - arr.shape[0]
    before = pad_total // 2
    after = pad_total - before
    return jnp.pad(arr, ((before, after), (before, after)), mode="reflect")


def _condition_detector_map(
    arr,
    *,
    map_name: str,
    target_shape: tuple[int, int],
    warn_on_map_conditioning: bool = False,
) -> jnp.ndarray:
    """Condition detector-resolution maps to the requested PSF shape.

    Policy:
      - if larger than requested: center-crop
      - if smaller than requested: center-pad with reflect
      - if equal: no-op
    """
    conditioned = jnp.asarray(arr, dtype=float)
    if conditioned.ndim != 2:
        raise ValueError(f"{map_name} must be a 2D array; got ndim={conditioned.ndim}.")
    if conditioned.shape[0] != conditioned.shape[1]:
        raise ValueError(f"{map_name} must be square; got shape={conditioned.shape}.")

    if conditioned.shape == target_shape:
        return conditioned

    target_size = target_shape[0]
    src_shape = conditioned.shape

    if conditioned.shape[0] > target_size:
        conditioned = _center_crop(conditioned, target_size)
        policy = "center-crop"
    else:
        conditioned = _center_pad_reflect(conditioned, target_size)
        policy = "center-pad+reflect"

    if warn_on_map_conditioning:
        warnings.warn(
            (
                f"Conditioned {map_name} at detector build time: "
                f"provided shape {src_shape} -> requested shape {target_shape} "
                f"using policy {policy}."
            ),
            stacklevel=2,
        )
    return conditioned


def _resolve_detector_spec(cfg) -> DetectorSpec:
    """Resolve detector metadata from config, defaulting to the testbed model."""
    detector_model = _cfg_get(cfg, "system.detector.model", default=None)
    if detector_model is None and isinstance(cfg, Mapping):
        detector_model = cfg.get("model", None)
    if detector_model is None and isinstance(cfg, Mapping):
        detector_cfg = cfg.get("detector", None)
        if isinstance(detector_cfg, Mapping):
            detector_model = detector_cfg.get("model", None)
    if detector_model is None:
        detector_model = getattr(cfg, "detector_model", None)
    if detector_model is None:
        return GSENSE2020BSI_SPEC

    model_to_spec = {
        GSENSE2020BSI_SPEC.model_name: GSENSE2020BSI_SPEC,
        HWK4123_SPEC.model_name: HWK4123_SPEC,
    }
    try:
        return model_to_spec[detector_model]
    except KeyError as exc:
        known = ", ".join(sorted(model_to_spec))
        raise ValueError(
            f"Unknown detector_model={detector_model!r}. Expected one of: {known}."
        ) from exc


def _find_repo_root(start: Path) -> Path:
    """Walk parents until we find a repo marker. Fallback to start if none found."""
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "setup.cfg").exists():
            return p
    return start


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def _resolve_repo_path(path: str | Path | None) -> Path | None:
    """Resolve a path that may be repo-root-relative.

    Kept local to the detector builder because calibration maps are detector-
    specific assets rather than general config I/O.
    """
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _load_array(path: Path) -> jnp.ndarray:
    """Load a detector calibration array from FITS, .npy, or .npz."""

    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits

        data = fits.getdata(path)
        if data is None:
            raise ValueError(f"FITS file {path} does not contain image data.")
        if data.ndim != 2:
            raise ValueError(f"FITS calibration map must be 2D; got shape {data.shape} from {path}.")
        return jnp.asarray(np.asarray(data), dtype=float)

    if suffix == ".npy":
        arr = np.load(path)
        return jnp.asarray(arr, dtype=float)

    if suffix == ".npz":
        with np.load(path) as z:
            # Prefer common keys if present; otherwise take the first array.
            for key in ("data", "arr_0", "dx", "dy", "prf", "pixel_response"):
                if key in z.files:
                    return jnp.asarray(z[key], dtype=float)
            return jnp.asarray(z[z.files[0]], dtype=float)

    raise ValueError(
        f"Unsupported calibration file type: {path} (expected .fits/.fit/.fts, .npy, or .npz)"
    )



# ---------------------------------------------------------------------------
# Layer construction helpers
# ---------------------------------------------------------------------------
def _layer_contract_prefix(layer_name: str) -> str:
    """Return the detector contract prefix for a named detector layer."""
    return f"detector.layers.{layer_name}"


def _downsample_kernel_size(layer_cfg: Mapping, *, context: str) -> int:
    """Return the configured downsample factor for a detector layer."""
    kernel_size = layer_cfg.get("kernel_size", layer_cfg.get("factor", None))
    if kernel_size is None:
        raise ValueError(f"{context} requires `kernel_size` (or alias `factor`).")
    return int(kernel_size)


def _warn_on_map_conditioning(layer_cfg: Mapping, *, context: str) -> bool:
    """Return whether detector map conditioning should emit warnings."""
    warn = layer_cfg.get("warn_on_map_conditioning", False)
    if not isinstance(warn, bool):
        raise ValueError(f"{context}.warn_on_map_conditioning must be a bool.")
    return warn


def _detector_to_psf_scale_by_layer_name(
    layers_cfg: list[Mapping],
    *,
    optics_oversample: float = 1.0,
) -> dict[str, float]:
    """Return current-grid pixels per detector pixel at each layer.

    The detector stack begins on the optics oversampled PSF grid. Each
    ``Downsample`` layer reduces the current-grid sampling for all subsequent
    layers by its kernel size.
    """
    scale_by_name: dict[str, float] = {}
    current_scale = float(optics_oversample)
    for idx, layer_cfg in enumerate(layers_cfg):
        layer_name = layer_cfg["name"]
        scale_by_name[layer_name] = current_scale
        if layer_cfg["kind"] == "Downsample":
            current_scale /= _downsample_kernel_size(
                layer_cfg,
                context=f"system.detector.layers[{idx}]",
            )
    return scale_by_name


def _parse_apply_convolution_kernel_cfg(
    layer_cfg: Mapping,
    *,
    context: str,
) -> dict[str, object]:
    """Parse and validate the nested ``ApplyConvolution.kernel`` mapping."""
    kernel_cfg = layer_cfg.get("kernel", None)
    if not isinstance(kernel_cfg, Mapping):
        raise ValueError(f"{context}.kernel must be a mapping/dict.")

    kernel_kind = kernel_cfg.get("kind")
    if not isinstance(kernel_kind, str) or not kernel_kind.strip():
        raise ValueError(f"Missing required config key: {context}.kernel.kind")
    if kernel_kind not in {"gaussian", "box", "line"}:
        raise ValueError(
            f"{context}.kernel.kind={kernel_kind!r} is not supported. "
            "ApplyConvolution supports kernel.kind values ['gaussian', 'box', 'line']."
        )

    def _positive_float(key: str) -> float:
        if key not in kernel_cfg:
            raise ValueError(f"Missing required config key: {context}.kernel.{key}")
        try:
            value = float(kernel_cfg[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.kernel.{key} must be a positive number.") from exc
        if value <= 0.0:
            raise ValueError(f"{context}.kernel.{key} must be a positive number.")
        return value

    def _any_float(key: str) -> float:
        if key not in kernel_cfg:
            raise ValueError(f"Missing required config key: {context}.kernel.{key}")
        try:
            return float(kernel_cfg[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}.kernel.{key} must be numeric.") from exc

    if "kernel_size" not in kernel_cfg:
        raise ValueError(f"Missing required config key: {context}.kernel.kernel_size")
    raw_kernel_size = kernel_cfg["kernel_size"]
    try:
        kernel_size = int(raw_kernel_size)
        kernel_size_float = float(raw_kernel_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}.kernel.kernel_size must be a positive odd integer."
        ) from exc
    if kernel_size_float != float(kernel_size) or kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError(f"{context}.kernel.kernel_size must be a positive odd integer.")

    units = kernel_cfg.get("units")
    if not isinstance(units, str) or not units.strip():
        raise ValueError(f"Missing required config key: {context}.kernel.units")
    if units not in {"detector_pix", "psf_pix"}:
        raise ValueError(
            f"{context}.kernel.units must be one of ['detector_pix', 'psf_pix']."
        )

    parsed: dict[str, object] = {
        "kernel_kind": kernel_kind,
        "kernel_size": kernel_size,
        "units": units,
    }
    if kernel_kind == "gaussian":
        parsed["sigma_x"] = _positive_float("sigma_x")
        parsed["sigma_y"] = _positive_float("sigma_y")
        parsed["theta_deg"] = _any_float("theta_deg")
    elif kernel_kind == "box":
        parsed["width_x"] = _positive_float("width_x")
        parsed["width_y"] = _positive_float("width_y")
        parsed["theta_deg"] = 0.0
    else:
        parsed["length"] = _positive_float("length")
        parsed["sigma_perp"] = _positive_float("sigma_perp")
        parsed["theta_deg"] = _any_float("theta_deg")
    return parsed


def build_detector_layer(
    layer_cfg: Mapping,
    *,
    target_shape: tuple[int, int],
    base_seed: int | None = None,
    detector_to_psf_scale: float = 1.0,
) -> tuple[str, object]:
    """Build a detector layer from declarative layer config."""
    name = layer_cfg["name"]
    kind = layer_cfg["kind"]

    if kind == "Downsample":
        kernel_size = _downsample_kernel_size(
            layer_cfg,
            context=f"detector layer {name!r}",
        )
        return (name, Downsample(kernel_size))

    if kind == "ApplyPixelOffsets":
        dx_path = _resolve_repo_path(layer_cfg.get("dx_path", None))
        dy_path = _resolve_repo_path(layer_cfg.get("dy_path", None))
        interp_method = layer_cfg.get("interp_method", "cubic")
        warn_on_map_conditioning = _warn_on_map_conditioning(
            layer_cfg,
            context=f"detector layer {name!r}",
        )

        if dx_path is not None:
            dx_map_raw = _load_array(dx_path)
        else:
            dx_map_raw = jnp.zeros(target_shape, dtype=float)
            if dy_path is not None:
                warnings.warn(
                    "pixel_offsets layer: dx_path missing; defaulting dx_map to zeros.",
                    UserWarning,
                    stacklevel=2,
                )

        if dy_path is not None:
            dy_map_raw = _load_array(dy_path)
        else:
            dy_map_raw = jnp.zeros(target_shape, dtype=float)
            if dx_path is not None:
                warnings.warn(
                    "pixel_offsets layer: dy_path missing; defaulting dy_map to zeros.",
                    UserWarning,
                    stacklevel=2,
                )

        dx_map = _condition_detector_map(
            dx_map_raw,
            map_name="dx_map",
            target_shape=target_shape,
            warn_on_map_conditioning=warn_on_map_conditioning,
        )
        dy_map = _condition_detector_map(
            dy_map_raw,
            map_name="dy_map",
            target_shape=target_shape,
            warn_on_map_conditioning=warn_on_map_conditioning,
        )

        knowledge_error = layer_cfg.get("knowledge_error", None)
        if knowledge_error:
            dx_map, _ = apply_knowledge_error(
                dx_map,
                knowledge_cfg=knowledge_error,
                base_seed=base_seed,
                token=f"{name}.dx",
            )
            dy_map, _ = apply_knowledge_error(
                dy_map,
                knowledge_cfg=knowledge_error,
                base_seed=base_seed,
                token=f"{name}.dy",
            )
        return (
            name,
            ApplyPixelOffsets(
                dx_map=dx_map,
                dy_map=dy_map,
                interp_method=interp_method,
                detector_to_psf_scale=detector_to_psf_scale,
            ),
        )

    if kind == "ApplyPixelResponse":
        prf_path = _resolve_repo_path(layer_cfg.get("prf_path", None))
        warn_on_map_conditioning = _warn_on_map_conditioning(
            layer_cfg,
            context=f"detector layer {name!r}",
        )
        if prf_path is None:
            pixel_response_raw = jnp.ones(target_shape, dtype=float)
        else:
            pixel_response_raw = _load_array(prf_path)

        pixel_response = _condition_detector_map(
            pixel_response_raw,
            map_name="pixel_response",
            target_shape=target_shape,
            warn_on_map_conditioning=warn_on_map_conditioning,
        )
        knowledge_error = layer_cfg.get("knowledge_error", None)
        if knowledge_error:
            pixel_response, _ = apply_knowledge_error(
                pixel_response,
                knowledge_cfg=knowledge_error,
                base_seed=base_seed,
                token=f"{name}.prf",
            )
        return (name, ApplyPixelResponse(pixel_response))

    if kind == "ApplyJitter":
        sigma = float(layer_cfg.get("sigma", 1e-12))
        kernel_size = int(layer_cfg.get("kernel_size", 3))
        return (name, ApplyJitter(sigma=sigma, kernel_size=kernel_size))

    if kind == "ApplyConvolution":
        kernel_cfg = _parse_apply_convolution_kernel_cfg(
            layer_cfg,
            context=f"detector layer {name!r}",
        )
        return (
            name,
            ApplyConvolution(
                kernel_kind=kernel_cfg["kernel_kind"],
                sigma_x=kernel_cfg.get("sigma_x", 1.0),
                sigma_y=kernel_cfg.get("sigma_y", 1.0),
                width_x=kernel_cfg.get("width_x", 1.0),
                width_y=kernel_cfg.get("width_y", 1.0),
                theta_deg=kernel_cfg.get("theta_deg", 0.0),
                length=kernel_cfg.get("length", 1.0),
                sigma_perp=kernel_cfg.get("sigma_perp", 0.25),
                kernel_size=kernel_cfg["kernel_size"],
                units=kernel_cfg["units"],
                detector_to_psf_scale=detector_to_psf_scale,
            ),
        )

    supported = ", ".join(SUPPORTED_DETECTOR_LAYER_KINDS)
    raise ValueError(
        f"Unknown detector layer kind {kind!r}. Supported kinds: {supported}."
    )


def _normalize_detector_cfg(detector_cfg) -> Mapping:
    """Return the detector block mapping from supported wrapper shapes.

    Accepts:
      - a detector mapping
      - a mapping with top-level ``system`` containing a detector mapping
      - config objects exposing ``detector_layers`` / ``detector_model`` attributes

    Raises
    ------
    ValueError
        If no detector block can be found.
    """

    if isinstance(detector_cfg, Mapping):
        if "detector" in detector_cfg:
            detector_block = detector_cfg["detector"]
            if not isinstance(detector_block, Mapping):
                raise ValueError("system.detector must be a mapping/dict.")
            return detector_block
        if "system" in detector_cfg:
            system_block = detector_cfg["system"]
            if not isinstance(system_block, Mapping):
                raise ValueError("system must be a mapping/dict.")
            detector_block = system_block.get("detector", None)
            if detector_block is None:
                raise ValueError("system.detector is required for detector construction.")
            if not isinstance(detector_block, Mapping):
                raise ValueError("system.detector must be a mapping/dict.")
            return detector_block
        return detector_cfg

    candidate = _cfg_get(detector_cfg, "system.detector", default=None)
    if candidate is not None:
        if not isinstance(candidate, Mapping):
            raise ValueError("system.detector must be a mapping/dict.")
        return candidate

    layers = getattr(detector_cfg, "detector_layers", None)
    model = getattr(detector_cfg, "detector_model", None)
    if layers is None:
        raise ValueError("Detector config must provide system.detector.layers in declarative form.")
    return {"model": model, "layers": layers}


def _validate_layers_cfg(layers_cfg: object) -> list[Mapping]:
    """Validate and return the declarative detector layer list."""
    if layers_cfg is None:
        raise ValueError("system.detector.layers is required for detector construction.")
    if not isinstance(layers_cfg, list):
        raise ValueError("system.detector.layers must be a list of layer dictionaries.")

    validated: list[Mapping] = []
    seen_names: set[str] = set()
    for idx, layer in enumerate(layers_cfg):
        if not isinstance(layer, Mapping):
            raise ValueError(f"system.detector.layers[{idx}] must be a mapping/dict.")
        name = layer.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Missing required config key: system.detector.layers[{idx}].name")
        kind = layer.get("kind")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError(f"Missing required config key: system.detector.layers[{idx}].kind")
        if name in seen_names:
            raise ValueError(
                f"Duplicate detector layer name {name!r} at system.detector.layers[{idx}]. "
                "Detector layer names must be unique."
            )
        seen_names.add(name)
        validated.append(layer)
    return validated


def build_detector_contract(detector_cfg) -> ParamSpec:
    """Build the detector ParamSpec contract from a detector config mapping.

    The contract incorporates detector metadata from the selected model and
    layer-specific fields based on the declared ``detector.layers`` config.
    """

    cfg = _normalize_detector_cfg(detector_cfg)
    spec = _resolve_detector_spec(cfg)

    fields: list[ParamField] = [
        ParamField(
            key="detector.pixel_pitch_m",
            group="detector",
            kind="primitive",
            dtype=float,
            shape=(),
            default=spec.pixel_pitch_m,
            structural=False,
        ),
        ParamField(
            key="detector.read_noise",
            group="detector",
            kind="primitive",
            dtype=float,
            shape=(),
            default=spec.read_noise,
            structural=False,
        ),
        ParamField(
            key="detector.dark_current",
            group="detector",
            kind="primitive",
            dtype=float,
            shape=(),
            default=spec.dark_current,
            structural=False,
        ),
        ParamField(
            key="detector.full_well",
            group="detector",
            kind="primitive",
            dtype=float,
            shape=(),
            default=spec.full_well,
            structural=False,
        ),
        ParamField(
            key="detector.qe",
            group="detector",
            kind="primitive",
            dtype=float,
            shape=(),
            default=spec.qe,
            structural=False,
        ),
        ParamField(
            key="detector.adc_bits",
            group="detector",
            kind="primitive",
            dtype=int,
            shape=(),
            default=spec.adc_bits,
            structural=False,
        ),
    ]

    layers_cfg = _validate_layers_cfg(cfg.get("layers", None) if isinstance(cfg, Mapping) else None)

    def _path_default(raw):
        resolved = _resolve_repo_path(raw)
        return str(resolved) if resolved is not None else None

    for idx, layer_cfg in enumerate(layers_cfg):
        layer_name = layer_cfg["name"]
        layer_kind = layer_cfg["kind"]
        prefix = _layer_contract_prefix(layer_name)

        if layer_kind == "ApplyJitter":
            sigma_val = layer_cfg.get("sigma", 1e-12)
            if sigma_val is None:
                sigma_val = 1e-12
            kernel_val = layer_cfg.get("kernel_size", 3)
            if kernel_val is None:
                kernel_val = 3
            sigma = float(sigma_val)
            kernel_size = int(kernel_val)
            fields.extend(
                [
                    ParamField(
                        key=f"{prefix}.sigma",
                        group="detector",
                        kind="primitive",
                        dtype=float,
                        shape=(),
                        default=sigma,
                        bounds=(0.0, None),
                        structural=False,
                        doc="Detector jitter sigma [pixels], runtime-overridable from the store.",
                    ),
                    ParamField(
                        key=f"{prefix}.kernel_size",
                        group="detector",
                        kind="primitive",
                        dtype=int,
                        shape=(),
                        default=kernel_size,
                        structural=True,
                    ),
                ]
            )
            continue

        if layer_kind == "ApplyPixelOffsets":
            interp_method = layer_cfg.get("interp_method", "cubic") or "cubic"
            dx_path = _path_default(layer_cfg.get("dx_path", None))
            dy_path = _path_default(layer_cfg.get("dy_path", None))
            fields.extend(
                [
                    ParamField(
                        key=f"{prefix}.interp_method",
                        group="detector",
                        kind="primitive",
                        dtype=str,
                        shape=(),
                        default=interp_method,
                        structural=False,
                    ),
                    ParamField(
                        key=f"{prefix}.dx_path",
                        group="detector",
                        kind="primitive",
                        dtype=str,
                        shape=(),
                        default=dx_path,
                        structural=True,
                    ),
                    ParamField(
                        key=f"{prefix}.dy_path",
                        group="detector",
                        kind="primitive",
                        dtype=str,
                        shape=(),
                        default=dy_path,
                        structural=True,
                    ),
                ]
            )
            continue

        if layer_kind == "ApplyPixelResponse":
            prf_path = _path_default(layer_cfg.get("prf_path", None))
            fields.append(
                ParamField(
                    key=f"{prefix}.prf_path",
                    group="detector",
                    kind="primitive",
                    dtype=str,
                    shape=(),
                    default=prf_path,
                    structural=True,
                )
            )
            continue

        if layer_kind == "Downsample":
            kernel_size = _downsample_kernel_size(
                layer_cfg,
                context=f"system.detector.layers[{idx}]",
            )
            fields.append(
                ParamField(
                    key=f"{prefix}.kernel_size",
                    group="detector",
                    kind="primitive",
                    dtype=int,
                    shape=(),
                    default=int(kernel_size),
                    structural=False,
                )
            )
            continue

        if layer_kind == "ApplyConvolution":
            kernel_cfg = _parse_apply_convolution_kernel_cfg(
                layer_cfg,
                context=f"system.detector.layers[{idx}]",
            )
            fields.append(
                ParamField(
                    key=f"{prefix}.kernel_kind",
                    group="detector",
                    kind="primitive",
                    dtype=str,
                    shape=(),
                    default=kernel_cfg["kernel_kind"],
                    structural=True,
                )
            )
            if kernel_cfg["kernel_kind"] == "gaussian":
                fields.extend(
                    [
                        ParamField(
                            key=f"{prefix}.sigma_x",
                            group="detector",
                            kind="primitive",
                            dtype=float,
                            shape=(),
                            default=kernel_cfg["sigma_x"],
                            bounds=(0.0, None),
                            structural=False,
                            doc="Gaussian convolution sigma along x [runtime-overridable].",
                        ),
                        ParamField(
                            key=f"{prefix}.sigma_y",
                            group="detector",
                            kind="primitive",
                            dtype=float,
                            shape=(),
                            default=kernel_cfg["sigma_y"],
                            bounds=(0.0, None),
                            structural=False,
                            doc="Gaussian convolution sigma along y [runtime-overridable].",
                        ),
                        ParamField(
                            key=f"{prefix}.theta_deg",
                            group="detector",
                            kind="primitive",
                            dtype=float,
                            shape=(),
                            default=kernel_cfg["theta_deg"],
                            structural=False,
                            doc="Gaussian convolution rotation angle [degrees, runtime-overridable].",
                        ),
                    ]
                )
            elif kernel_cfg["kernel_kind"] == "box":
                fields.extend(
                    [
                        ParamField(
                            key=f"{prefix}.width_x",
                            group="detector",
                            kind="primitive",
                            dtype=float,
                            shape=(),
                            default=kernel_cfg["width_x"],
                            bounds=(0.0, None),
                            structural=False,
                            doc="Box convolution width along x [runtime-overridable].",
                        ),
                        ParamField(
                            key=f"{prefix}.width_y",
                            group="detector",
                            kind="primitive",
                            dtype=float,
                            shape=(),
                            default=kernel_cfg["width_y"],
                            bounds=(0.0, None),
                            structural=False,
                            doc="Box convolution width along y [runtime-overridable].",
                        ),
                    ]
                )
            else:
                fields.extend(
                    [
                        ParamField(
                            key=f"{prefix}.length",
                            group="detector",
                            kind="primitive",
                            dtype=float,
                            shape=(),
                            default=kernel_cfg["length"],
                            bounds=(0.0, None),
                            structural=False,
                            doc="Line-smear total length [runtime-overridable].",
                        ),
                        ParamField(
                            key=f"{prefix}.sigma_perp",
                            group="detector",
                            kind="primitive",
                            dtype=float,
                            shape=(),
                            default=kernel_cfg["sigma_perp"],
                            bounds=(0.0, None),
                            structural=False,
                            doc="Line-smear cross-track Gaussian sigma [runtime-overridable].",
                        ),
                        ParamField(
                            key=f"{prefix}.theta_deg",
                            group="detector",
                            kind="primitive",
                            dtype=float,
                            shape=(),
                            default=kernel_cfg["theta_deg"],
                            structural=False,
                            doc="Line-smear orientation angle [degrees, runtime-overridable].",
                        ),
                    ]
                )
            fields.extend(
                [
                    ParamField(
                        key=f"{prefix}.kernel_size",
                        group="detector",
                        kind="primitive",
                        dtype=int,
                        shape=(),
                        default=kernel_cfg["kernel_size"],
                        structural=True,
                    ),
                    ParamField(
                        key=f"{prefix}.units",
                        group="detector",
                        kind="primitive",
                        dtype=str,
                        shape=(),
                        default=kernel_cfg["units"],
                        structural=True,
                    ),
                ]
            )
            continue

        supported = ", ".join(SUPPORTED_DETECTOR_LAYER_KINDS)
        raise ValueError(
            f"Unknown detector layer kind {layer_kind!r}. Supported kinds: {supported}."
        )

    return ParamSpec(fields)


# ---------------------------------------------------------------------------
# Public builder entry points
# ---------------------------------------------------------------------------
def build_detector(cfg, *, base_seed: int | None = None) -> tuple[SheraDetector, ParamSpec]:
    """Construct the baseline detector for a Shera system.

    Requires a declarative ``system.detector.layers`` config. No legacy flat
    detector fields are supported.
    """
    detector_cfg_block = _normalize_detector_cfg(cfg)
    if not isinstance(detector_cfg_block, Mapping):
        raise ValueError("Detector config must resolve to a mapping/dict.")

    detector_layers_cfg = _validate_layers_cfg(detector_cfg_block.get("layers", None))

    psf_npix = None
    if isinstance(cfg, Mapping):
        optics_block = cfg.get("optics", None)
        if isinstance(optics_block, Mapping):
            psf_npix = optics_block.get("psf_npix", None)
    if psf_npix is None:
        psf_npix = _cfg_get(cfg, "system.optics.psf_npix", default=None)
    if psf_npix is None:
        psf_npix = getattr(cfg, "psf_npix", None)
    if psf_npix is None:
        raise ValueError("system.optics.psf_npix is required to build the detector.")

    optics_oversample = None
    if isinstance(cfg, Mapping):
        optics_block = cfg.get("optics", None)
        if isinstance(optics_block, Mapping):
            optics_oversample = optics_block.get("oversample", None)
    if optics_oversample is None:
        optics_oversample = _cfg_get(cfg, "system.optics.oversample", default=None)
    if optics_oversample is None:
        optics_oversample = getattr(cfg, "oversample", None)
    if optics_oversample is None:
        optics_oversample = 1.0
        for idx, layer_cfg in enumerate(detector_layers_cfg):
            if layer_cfg["kind"] == "Downsample":
                optics_oversample *= _downsample_kernel_size(
                    layer_cfg,
                    context=f"system.detector.layers[{idx}]",
                )
    optics_oversample = float(optics_oversample)
    if optics_oversample <= 0.0:
        raise ValueError("system.optics.oversample must be positive when building the detector.")

    target_shape = (int(psf_npix), int(psf_npix))
    detector_to_psf_scale = _detector_to_psf_scale_by_layer_name(
        detector_layers_cfg,
        optics_oversample=optics_oversample,
    )
    layers = [
        build_detector_layer(
            layer_cfg,
            target_shape=target_shape,
            base_seed=base_seed,
            detector_to_psf_scale=detector_to_psf_scale[layer_cfg["name"]],
        )
        for layer_cfg in detector_layers_cfg
    ]

    spec = _resolve_detector_spec(detector_cfg_block)

    detector = SheraDetector(layers=layers, spec=spec)
    detector_contract = build_detector_contract(cfg)
    return detector, detector_contract


def apply_runtime_bindings(
    detector: SheraDetector,
    store,
    bindings: tuple[tuple[str, str], ...] = DETECTOR_RUNTIME_BINDINGS,
) -> SheraDetector:
    """Apply runtime ParameterStore overrides onto a cached detector.

    Scope is intentionally narrow: selected detector-layer primitives such as
    jitter sigma and convolution Gaussian parameters can be overridden
    directly, and any explicit detector bindings (currently none) would be
    applied via ``detector.set``. Structural detector changes are still
    handled via binder-level rebuilds.
    """

    if store is None:
        return detector

    rebuilt_layers = []
    detector_updated = False
    for layer_name, layer in detector.layers.items():
        runtime_sigma = store.get(f"detector.layers.{layer_name}.sigma", default=None)
        if runtime_sigma is not None and isinstance(layer, ApplyJitter):
            detector_updated = True
            rebuilt_layers.append(
                (
                    layer_name,
                    ApplyJitter(
                        sigma=float(runtime_sigma),
                        kernel_size=int(layer.kernel_size),
                    ),
                )
            )
        elif isinstance(layer, ApplyConvolution):
            runtime_sigma_x = store.get(f"detector.layers.{layer_name}.sigma_x", default=None)
            runtime_sigma_y = store.get(f"detector.layers.{layer_name}.sigma_y", default=None)
            runtime_theta_deg = store.get(f"detector.layers.{layer_name}.theta_deg", default=None)
            runtime_width_x = store.get(f"detector.layers.{layer_name}.width_x", default=None)
            runtime_width_y = store.get(f"detector.layers.{layer_name}.width_y", default=None)
            runtime_length = store.get(f"detector.layers.{layer_name}.length", default=None)
            runtime_sigma_perp = store.get(
                f"detector.layers.{layer_name}.sigma_perp", default=None
            )
            if any(
                v is not None
                for v in (
                    runtime_sigma_x,
                    runtime_sigma_y,
                    runtime_theta_deg,
                    runtime_width_x,
                    runtime_width_y,
                    runtime_length,
                    runtime_sigma_perp,
                )
            ):
                detector_updated = True
                rebuilt_layers.append(
                    (
                        layer_name,
                        ApplyConvolution(
                            kernel_kind=layer.kernel_kind,
                            sigma_x=(
                                float(runtime_sigma_x)
                                if runtime_sigma_x is not None
                                else float(layer.sigma_x)
                            ),
                            sigma_y=(
                                float(runtime_sigma_y)
                                if runtime_sigma_y is not None
                                else float(layer.sigma_y)
                            ),
                            theta_deg=(
                                float(runtime_theta_deg)
                                if runtime_theta_deg is not None
                                else float(layer.theta_deg)
                            ),
                            width_x=(
                                float(runtime_width_x)
                                if runtime_width_x is not None
                                else float(layer.width_x)
                            ),
                            width_y=(
                                float(runtime_width_y)
                                if runtime_width_y is not None
                                else float(layer.width_y)
                            ),
                            length=(
                                float(runtime_length)
                                if runtime_length is not None
                                else float(layer.length)
                            ),
                            sigma_perp=(
                                float(runtime_sigma_perp)
                                if runtime_sigma_perp is not None
                                else float(layer.sigma_perp)
                            ),
                            kernel_size=int(layer.kernel_size),
                            units=str(layer.units),
                            detector_to_psf_scale=float(layer.detector_to_psf_scale),
                        ),
                    )
                )
            else:
                rebuilt_layers.append((layer_name, layer))
        else:
            rebuilt_layers.append((layer_name, layer))

    if detector_updated:
        detector = SheraDetector(layers=rebuilt_layers, spec=detector.spec)

    for store_key, set_path in bindings:
        val = store.get(store_key, default=None)
        if val is None:
            continue
        detector = detector.set(set_path, val)
    return detector


__all__ = [
    "DETECTOR_RUNTIME_BINDINGS",
    "apply_runtime_bindings",
    "build_detector",
    "build_detector_contract",
    "build_detector_layer",
]
