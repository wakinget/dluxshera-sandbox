"""Detector builder responsibilities (detector assembly and runtime wiring)."""

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


DETECTOR_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()
SUPPORTED_DETECTOR_LAYERS: tuple[str, ...] = (
    "downsample",
    "pixel_offsets",
    "pixel_response",
    "jitter",
)


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
    """Resolve a path that may be repo-root-relative."""
    if path is None:
        return None
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (_REPO_ROOT / p).resolve()


def _load_array(path: Path) -> jnp.ndarray:
    """Load a calibration array from .npy or .npz."""
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


def _build_legacy_detector_layers(cfg, *, target_shape: tuple[int, int]):
    """Build detector layers from flat legacy detector config fields."""
    ppu_dx_path = _resolve_repo_path(getattr(cfg, "ppu_dx_path", None))
    ppu_dy_path = _resolve_repo_path(getattr(cfg, "ppu_dy_path", None))
    interp_method = getattr(cfg, "ppu_interp_method", "cubic2")

    dx_map_raw = _load_array(ppu_dx_path) if ppu_dx_path is not None else None
    dy_map_raw = _load_array(ppu_dy_path) if ppu_dy_path is not None else None

    if dx_map_raw is None:
        dx_map_raw = jnp.zeros(target_shape, dtype=float)
    if dy_map_raw is None:
        dy_map_raw = jnp.zeros(target_shape, dtype=float)

    dx_map = _condition_detector_map(dx_map_raw, map_name="dx_map", target_shape=target_shape)
    dy_map = _condition_detector_map(dy_map_raw, map_name="dy_map", target_shape=target_shape)

    prf_path = _resolve_repo_path(getattr(cfg, "prf_path", None))
    if prf_path is None:
        pixel_response_raw = jnp.ones(target_shape, dtype=float)
    else:
        pixel_response_raw = _load_array(prf_path)

    pixel_response = _condition_detector_map(
        pixel_response_raw, map_name="pixel_response", target_shape=target_shape
    )

    jitter_sigma = float(getattr(cfg, "jitter_sigma", 1e-12))
    jitter_kernel = int(getattr(cfg, "jitter_kernel_size", 3))

    return [
        ("downsample", Downsample(cfg.oversample)),
        (
            "pixel_offsets",
            ApplyPixelOffsets(dx_map=dx_map, dy_map=dy_map, interp_method=interp_method),
        ),
        ("pixel_response", ApplyPixelResponse(pixel_response)),
        ("jitter", ApplyJitter(sigma=jitter_sigma, kernel_size=jitter_kernel)),
    ]


def build_detector_layer(
    name: str,
    layer_cfg: Mapping,
    *,
    target_shape: tuple[int, int],
) -> tuple[str, object]:
    """Build a detector layer from declarative layer config."""
    if name == "downsample":
        factor = layer_cfg.get("factor", layer_cfg.get("oversample", None))
        if factor is None:
            raise ValueError("downsample layer requires `factor` (or alias `oversample`).")
        return ("downsample", Downsample(int(factor)))

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
        return ("pixel_response", ApplyPixelResponse(pixel_response))

    if name == "jitter":
        sigma = float(layer_cfg.get("sigma", 1e-12))
        kernel_size = int(layer_cfg.get("kernel_size", 3))
        return ("jitter", ApplyJitter(sigma=sigma, kernel_size=kernel_size))

    supported = ", ".join(SUPPORTED_DETECTOR_LAYERS)
    raise ValueError(f"Unknown detector layer name {name!r}. Supported layers: {supported}.")

def _build_detector_contract(detector: SheraDetector) -> ParamSpec:
    """Build a minimal detector ParamSpec contract from an assembled detector."""

    jitter_layer = detector.layers.get("jitter", None)
    if jitter_layer is None:
        return ParamSpec()

    return ParamSpec(
        [
            ParamField(
                key="detector.jitter.sigma",
                group="detector",
                kind="primitive",
                dtype=float,
                shape=(),
                default=float(jitter_layer.sigma),
                bounds=(0.0, None),
                doc="Detector jitter sigma [pixels], runtime-overridable from the store.",
            )
        ]
    )


def build_detector(cfg) -> tuple[SheraDetector, ParamSpec]:
    """Construct the baseline detector for a Shera system."""
    detector_layers_cfg = _cfg_get(cfg, "system.detector.layers", default=None)
    spec = _resolve_detector_spec(cfg)

    if detector_layers_cfg:
        psf_npix = _cfg_get(cfg, "system.optics.psf_npix", default=None)
        if psf_npix is None:
            psf_npix = getattr(cfg, "psf_npix", None)
        if psf_npix is None:
            raise ValueError(
                "Cannot build detector from system.detector.layers: missing system.optics.psf_npix."
            )

        target_shape = (int(psf_npix), int(psf_npix))
        layers = []
        for layer_cfg in detector_layers_cfg:
            name = layer_cfg.get("name", None)
            if name is None:
                raise ValueError("Each detector layer entry must define a `name` field.")
            layers.append(build_detector_layer(name, layer_cfg, target_shape=target_shape))
    else:
        warnings.warn(
            "system.detector.layers not provided; building detector from legacy flat detector config.",
            UserWarning,
            stacklevel=2,
        )
        psf_npix = int(cfg.psf_npix)
        target_shape = (psf_npix, psf_npix)
        layers = _build_legacy_detector_layers(cfg, target_shape=target_shape)

    detector = SheraDetector(layers=layers, spec=spec)
    return detector, _build_detector_contract(detector)


def apply_runtime_bindings(
    detector: SheraDetector,
    store,
    bindings: tuple[tuple[str, str], ...] = DETECTOR_RUNTIME_BINDINGS,
) -> SheraDetector:
    """Apply runtime ParameterStore overrides onto a cached detector."""

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
    "build_detector_layer",
]
