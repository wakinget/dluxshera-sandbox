"""Detector builder responsibilities (detector assembly and runtime wiring)."""

from __future__ import annotations

import warnings
from pathlib import Path
import jax.numpy as jnp
import numpy as np

from ..components.detectors import (
    GSENSE2020BSI_SPEC,
    HWK4123_SPEC,
    DetectorSpec,
    SheraDetector,
)
from ..layers.detector_layers import ApplyPixelOffsets
from dLux.layers.detector_layers import ApplyJitter, ApplyPixelResponse, Downsample


DETECTOR_RUNTIME_BINDINGS: tuple[tuple[str, str], ...] = ()


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

def build_detector(cfg) -> SheraDetector:
    """Construct the baseline detector for a Shera system."""
    psf_npix = int(cfg.psf_npix)
    target_shape = (psf_npix, psf_npix)

    # --- Pixel offsets (PPU) ---
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

    # --- Pixel response (PRF) ---
    prf_path = _resolve_repo_path(getattr(cfg, "prf_path", None))
    if prf_path is None:
        pixel_response_raw = jnp.ones(target_shape, dtype=float)
    else:
        pixel_response_raw = _load_array(prf_path)

    pixel_response = _condition_detector_map(
        pixel_response_raw, map_name="pixel_response", target_shape=target_shape
    )

    # --- Jitter ---
    jitter_sigma = float(getattr(cfg, "jitter_sigma", 1e-12))
    jitter_kernel = int(getattr(cfg, "jitter_kernel_size", 3))

    spec = _resolve_detector_spec(cfg)

    layers = [
        ("downsample", Downsample(cfg.oversample)),
        # ("pixel_offsets", ApplyPixelOffsets(dx_map=dx_map, dy_map=dy_map, interp_method=interp_method)),
        # ("pixel_response", ApplyPixelResponse(pixel_response)),
        # ("jitter", ApplyJitter(sigma=jitter_sigma, kernel_size=jitter_kernel)),
    ]
    return SheraDetector(layers=layers, spec=spec)


def apply_runtime_bindings(
    detector: SheraDetector,
    store,
    bindings: tuple[tuple[str, str], ...] = DETECTOR_RUNTIME_BINDINGS,
) -> SheraDetector:
    """Apply runtime ParameterStore overrides onto a cached detector."""

    if store is None:
        return detector

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
]
