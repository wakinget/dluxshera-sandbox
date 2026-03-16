"""Detector builder responsibilities (detector assembly and runtime wiring).

This module translates declarative detector blocks into runtime detector
objects. It owns:
  - config shape normalization for declarative detector blocks
  - calibration/path resolution for detector maps
  - detector-layer construction from declarative layers
  - detector contract construction (ParamSpec)
  - lightweight runtime patching for detector jitters/bindings
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
from ..layers.detector_layers import ApplyPixelOffsets
from dLux.layers.detector_layers import ApplyJitter, ApplyPixelResponse, Downsample
from ..utils.noise import apply_knowledge_error


# Runtime bindings for detectors are currently empty; detector runtime updates are
# limited to jitter (handled explicitly in apply_runtime_bindings).
DETECTOR_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()
SUPPORTED_DETECTOR_LAYERS: tuple[str, ...] = (
    "downsample",
    "pixel_offsets",
    "pixel_response",
    "jitter",
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
    """Load a detector calibration array from .npy or .npz."""
    if path.suffix == ".npy":
        arr = np.load(path)
        return jnp.asarray(arr, dtype=float)

    if path.suffix == ".npz":
        with np.load(path) as z:
            # Prefer common keys if present; otherwise take the first array.
            for key in ("data", "arr_0", "dx", "dy", "prf", "pixel_response"):
                if key in z.files:
                    return jnp.asarray(z[key], dtype=float)
            return jnp.asarray(z[z.files[0]], dtype=float)

    raise ValueError(f"Unsupported calibration file type: {path} (expected .npy or .npz)")



# ---------------------------------------------------------------------------
# Layer construction helpers
# ---------------------------------------------------------------------------
def build_detector_layer(
    name: str,
    layer_cfg: Mapping,
    *,
    target_shape: tuple[int, int],
    base_seed: int | None = None,
) -> tuple[str, object]:
    """Build a detector layer from declarative layer config."""
    if name == "downsample":
        kernel_size = layer_cfg.get("kernel_size", layer_cfg.get("factor", None))
        if kernel_size is None:
            raise ValueError("downsample layer requires `kernel_size` (or alias `factor`).")
        return ("downsample", Downsample(int(kernel_size)))

    if name == "pixel_offsets":
        dx_path = _resolve_repo_path(layer_cfg.get("dx_path", None))
        dy_path = _resolve_repo_path(layer_cfg.get("dy_path", None))
        interp_method = layer_cfg.get("interp_method", "cubic2")

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

        dx_map = _condition_detector_map(dx_map_raw, map_name="dx_map", target_shape=target_shape)
        dy_map = _condition_detector_map(dy_map_raw, map_name="dy_map", target_shape=target_shape)

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
            "pixel_offsets",
            ApplyPixelOffsets(dx_map=dx_map, dy_map=dy_map, interp_method=interp_method),
        )

    if name == "pixel_response":
        prf_path = _resolve_repo_path(layer_cfg.get("prf_path", None))
        if prf_path is None:
            pixel_response_raw = jnp.ones(target_shape, dtype=float)
        else:
            pixel_response_raw = _load_array(prf_path)

        pixel_response = _condition_detector_map(
            pixel_response_raw, map_name="pixel_response", target_shape=target_shape
        )
        knowledge_error = layer_cfg.get("knowledge_error", None)
        if knowledge_error:
            pixel_response, _ = apply_knowledge_error(
                pixel_response,
                knowledge_cfg=knowledge_error,
                base_seed=base_seed,
                token=f"{name}.prf",
            )
        return ("pixel_response", ApplyPixelResponse(pixel_response))

    if name == "jitter":
        sigma = float(layer_cfg.get("sigma", 1e-12))
        kernel_size = int(layer_cfg.get("kernel_size", 3))
        return ("jitter", ApplyJitter(sigma=sigma, kernel_size=kernel_size))

    supported = ", ".join(SUPPORTED_DETECTOR_LAYERS)
    raise ValueError(f"Unknown detector layer name {name!r}. Supported layers: {supported}.")


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
    for idx, layer in enumerate(layers_cfg):
        if not isinstance(layer, Mapping):
            raise ValueError(f"system.detector.layers[{idx}] must be a mapping/dict.")
        if "name" not in layer:
            raise ValueError(f"Missing required config key: system.detector.layers[{idx}].name")
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
    layer_map: dict[str, Mapping] = {layer["name"]: layer for layer in layers_cfg}

    def _path_default(raw):
        resolved = _resolve_repo_path(raw)
        return str(resolved) if resolved is not None else None

    jitter_cfg = layer_map.get("jitter")
    if jitter_cfg is not None:
        sigma_val = jitter_cfg.get("sigma", 1e-12)
        if sigma_val is None:
            sigma_val = 1e-12
        kernel_val = jitter_cfg.get("kernel_size", 3)
        if kernel_val is None:
            kernel_val = 3
        sigma = float(sigma_val)
        kernel_size = int(kernel_val)
        fields.extend(
            [
                ParamField(
                    key="detector.jitter.sigma",
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
                    key="detector.jitter.kernel_size",
                    group="detector",
                    kind="primitive",
                    dtype=int,
                    shape=(),
                    default=kernel_size,
                    structural=True,
                ),
            ]
        )

    pixel_offsets_cfg = layer_map.get("pixel_offsets")
    if pixel_offsets_cfg is not None:
        interp_method = pixel_offsets_cfg.get("interp_method", "cubic2") or "cubic2"
        dx_path = _path_default(pixel_offsets_cfg.get("dx_path", None))
        dy_path = _path_default(pixel_offsets_cfg.get("dy_path", None))
        fields.extend(
            [
                ParamField(
                    key="detector.pixel_offsets.interp_method",
                    group="detector",
                    kind="primitive",
                    dtype=str,
                    shape=(),
                    default=interp_method,
                    structural=False,
                ),
                ParamField(
                    key="detector.pixel_offsets.dx_path",
                    group="detector",
                    kind="primitive",
                    dtype=str,
                    shape=(),
                    default=dx_path,
                    structural=True,
                ),
                ParamField(
                    key="detector.pixel_offsets.dy_path",
                    group="detector",
                    kind="primitive",
                    dtype=str,
                    shape=(),
                    default=dy_path,
                    structural=True,
                ),
            ]
        )

    pixel_response_cfg = layer_map.get("pixel_response")
    if pixel_response_cfg is not None:
        prf_path = _path_default(pixel_response_cfg.get("prf_path", None))
        fields.append(
            ParamField(
                key="detector.pixel_response.prf_path",
                group="detector",
                kind="primitive",
                dtype=str,
                shape=(),
                default=prf_path,
                structural=True,
            )
        )

    downsample_cfg = layer_map.get("downsample")
    if downsample_cfg is not None:
        kernel_size = downsample_cfg.get("kernel_size", downsample_cfg.get("factor", 1))
        if kernel_size is None:
            kernel_size = 1
        fields.append(
            ParamField(
                key="detector.downsample.kernel_size",
                group="detector",
                kind="primitive",
                dtype=int,
                shape=(),
                default=int(kernel_size),
                structural=False,
            )
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

    target_shape = (int(psf_npix), int(psf_npix))
    layers = [
        build_detector_layer(
            layer_cfg["name"],
            layer_cfg,
            target_shape=target_shape,
            base_seed=base_seed,
        )
        for layer_cfg in detector_layers_cfg
    ]

    spec = _resolve_detector_spec(detector_cfg_block)

    detector = SheraDetector(layers=layers, spec=spec)
    detector_contract = build_detector_contract(detector_cfg_block)
    return detector, detector_contract


def apply_runtime_bindings(
    detector: SheraDetector,
    store,
    bindings: tuple[tuple[str, str], ...] = DETECTOR_RUNTIME_BINDINGS,
) -> SheraDetector:
    """Apply runtime ParameterStore overrides onto a cached detector.

    Scope is intentionally narrow: jitter sigma can be overridden directly,
    and any explicit detector bindings (currently none) would be applied via
    ``detector.set``. Structural detector changes are still handled via
    binder-level rebuilds.
    """

    if store is None:
        return detector

    runtime_sigma = store.get("detector.jitter.sigma", default=None)
    if runtime_sigma is not None and "jitter" in detector.layers:
        jitter_layer = detector.layers["jitter"]
        rebuilt_layers = []
        for layer_name, layer in detector.layers.items():
            if layer_name == "jitter":
                rebuilt_layers.append(
                    (
                        "jitter",
                        ApplyJitter(
                            sigma=float(runtime_sigma),
                            kernel_size=int(jitter_layer.kernel_size),
                        ),
                    )
                )
            else:
                rebuilt_layers.append((layer_name, layer))
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
