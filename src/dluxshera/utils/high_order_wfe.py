from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import dLux.utils as dlu


@dataclass(frozen=True)
class HighOrderWFERealization:
    truth_opd_nm: np.ndarray
    inference_opd_nm: np.ndarray
    truth_metadata: dict[str, Any]
    knowledge_metadata: dict[str, Any]


def white_noise_2d(shape: tuple[int, int], seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


def one_over_f_noise_2d(shape: tuple[int, int], alpha: float = 2.0, seed: int | None = None) -> np.ndarray:
    ny, nx = shape
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((ny, nx))
    fy = np.fft.fftfreq(ny)
    fx = np.fft.fftfreq(nx)
    kx, ky = np.meshgrid(fx, fy)
    fr = np.sqrt(kx**2 + ky**2)
    env = np.where(fr > 0, fr ** (-alpha / 2.0), 0.0)
    filt = np.fft.ifft2(np.fft.fft2(raw) * env).real
    std = float(np.std(filt))
    return filt if std == 0.0 else filt / std


def make_pupil_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(-1.0, 1.0, shape[0])
    x = np.linspace(-1.0, 1.0, shape[1])
    return np.meshgrid(x, y)


def _valid_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    return np.asarray(mask) > 0


def normalize_opd_rms(opd_nm: np.ndarray, target_rms_nm: float, mask: np.ndarray | None = None) -> np.ndarray:
    valid = _valid_mask(mask, opd_nm.shape)
    rms = float(np.sqrt(np.mean(np.square(opd_nm[valid]))))
    if target_rms_nm == 0 or rms == 0:
        return np.zeros_like(opd_nm) if target_rms_nm == 0 else opd_nm
    return opd_nm * (target_rms_nm / rms)


def fit_zernike_modes(opd_nm: np.ndarray, noll_indices: list[int], mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if not noll_indices:
        return np.zeros((0,), dtype=float), np.zeros_like(opd_nm)
    coords = dlu.pixel_coords(opd_nm.shape[0], 2.0)
    basis = np.asarray([np.asarray(dlu.zernike(i, coords, 2.0)) for i in noll_indices])
    valid = _valid_mask(mask, opd_nm.shape)
    A = basis[:, valid].T
    b = np.asarray(opd_nm)[valid]
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    fitted = np.tensordot(coeffs, basis, axes=(0, 0))
    return coeffs, fitted


def remove_zernike_modes(opd_nm: np.ndarray, noll_indices: list[int] | None, mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if not noll_indices:
        return opd_nm, {"removed_noll_indices": [], "coefficients_nm": []}
    coeffs, fitted = fit_zernike_modes(opd_nm, list(noll_indices), mask=mask)
    return opd_nm - fitted, {"removed_noll_indices": list(noll_indices), "coefficients_nm": coeffs.tolist()}


def remove_plane(opd_nm: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    return remove_zernike_modes(opd_nm, [1, 2, 3], mask=mask)


def generate_high_order_wfe_map(shape: tuple[int, int], *, kind: str = "one_over_f", alpha: float = 2.0, rms_nm: float = 0.0, seed: int | None = None, remove_zernike_noll: list[int] | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if rms_nm == 0.0:
        arr = np.zeros(shape)
    elif kind == "white":
        arr = white_noise_2d(shape, seed=seed)
    else:
        arr = one_over_f_noise_2d(shape, alpha=alpha, seed=seed)
    arr, zmeta = remove_zernike_modes(arr, remove_zernike_noll, mask=mask)
    arr = normalize_opd_rms(arr, rms_nm, mask=mask)
    valid = _valid_mask(mask, shape)
    measured = float(np.sqrt(np.mean(np.square(arr[valid])))) if np.any(valid) else 0.0
    return arr, {"shape": shape, "units": "nm", "kind": kind, "alpha": alpha, "seed": seed, "target_rms_nm": float(rms_nm), "measured_rms_nm": measured, **zmeta}


def realize_high_order_wfe_pair(shape: tuple[int, int], truth_cfg: Mapping[str, Any] | None, knowledge_cfg: Mapping[str, Any] | None, mask: np.ndarray | None = None) -> HighOrderWFERealization:
    truth_cfg = truth_cfg or {}
    truth, truth_meta = generate_high_order_wfe_map(shape, kind=str(truth_cfg.get("kind", truth_cfg.get("spectrum", "one_over_f"))), alpha=float(truth_cfg.get("alpha", 2.0)), rms_nm=float(truth_cfg.get("rms_nm", 0.0)), seed=truth_cfg.get("seed"), remove_zernike_noll=truth_cfg.get("remove_zernike_noll"), mask=mask)
    knowledge_cfg = knowledge_cfg or {}
    if not knowledge_cfg.get("enabled", False):
        resid = np.zeros(shape)
        kmeta = {"enabled": False, "target_rms_nm": 0.0, "measured_rms_nm": 0.0}
    else:
        resid, kmeta = generate_high_order_wfe_map(shape, kind=str(knowledge_cfg.get("kind", "white")), alpha=float(knowledge_cfg.get("alpha", 2.0)), rms_nm=float(knowledge_cfg.get("rms_nm", 0.0)), seed=knowledge_cfg.get("seed"), remove_zernike_noll=knowledge_cfg.get("remove_zernike_noll"), mask=mask)
        kmeta["enabled"] = True
    return HighOrderWFERealization(truth_opd_nm=truth, inference_opd_nm=truth + resid, truth_metadata=truth_meta, knowledge_metadata=kmeta)


__all__ = [
    "HighOrderWFERealization",
    "one_over_f_noise_2d",
    "white_noise_2d",
    "make_pupil_coordinates",
    "fit_zernike_modes",
    "remove_zernike_modes",
    "remove_plane",
    "normalize_opd_rms",
    "generate_high_order_wfe_map",
    "realize_high_order_wfe_pair",
]
