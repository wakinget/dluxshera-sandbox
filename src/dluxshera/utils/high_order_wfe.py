from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import dLux.utils as dlu
from astropy.io import fits

SCHEMA_VERSION = "high_order_wfe_deck.v1"
DEFAULT_LOW_ORDER_NOLL_INDICES = (4, 5, 6, 7, 8, 9, 10, 11)
PTT_NOLL_INDICES = (1, 2, 3)


@dataclass(frozen=True)
class WfeMap:
    """Represent one pupil-supported OPD map in nanometres.

    Use this as the durable deck-facing container for truth, knowledge, and
    knowledge-error maps. Diagnostics are computed over the boolean pupil mask;
    values outside the mask are retained in ``opd_nm`` but do not define RMS or
    mean summaries.
    """

    label: str
    opd_nm: np.ndarray
    mask: np.ndarray
    rms_nm: float
    mean_nm: float
    shape: tuple[int, int]
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class MirrorWfeDeck:
    """Represent one mirror's low-order and high-order WFE deck entries.

    The active observation state is the low-order Z4-Z11 coefficient vector.
    The high-order maps are nuisance realism: retained in artifacts and future
    optical configs, but not treated as active inference parameters by this deck.
    """

    mirror: str
    full_truth: WfeMap
    low_order_truth_coeffs_nm: dict[str, float]
    low_order_knowledge_coeffs_nm: dict[str, float]
    low_order_knowledge_error_nm: dict[str, float]
    high_order_truth: WfeMap
    high_order_knowledge: WfeMap
    high_order_knowledge_error: WfeMap
    diagnostics: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class HighOrderWfeDeck:
    """Represent the reusable high-order WFE deck for primary and secondary.

    The deck stores independent M1/M2 truth maps, low-order coefficient truth,
    coefficient knowledge errors, and additive high-order map knowledge errors.
    It is intended to be written to artifacts before later config/render wiring.
    """

    primary: MirrorWfeDeck
    secondary: MirrorWfeDeck
    schema_version: str
    provenance: dict[str, Any]
    comparison: dict[str, Any]


@dataclass(frozen=True)
class HighOrderWFERealization:
    truth_opd_nm: np.ndarray
    inference_opd_nm: np.ndarray
    truth_metadata: dict[str, Any]
    knowledge_metadata: dict[str, Any]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _as_shape(shape: tuple[int, int] | list[int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape, got {shape!r}.")
    return int(shape[0]), int(shape[1])


def _masked_mean_rms(opd_nm: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    valid = np.asarray(mask, dtype=bool)
    if not np.any(valid):
        raise ValueError("Pupil mask must contain at least one valid pixel.")
    vals = np.asarray(opd_nm, dtype=float)[valid]
    mean = float(np.mean(vals))
    rms = float(np.sqrt(np.mean(np.square(vals))))
    return mean, rms


def _wfe_map(label: str, opd_nm: np.ndarray, mask: np.ndarray, *, diagnostics: Mapping[str, Any] | None = None, provenance: Mapping[str, Any] | None = None) -> WfeMap:
    arr = np.asarray(opd_nm, dtype=float)
    valid = np.asarray(mask, dtype=bool)
    if arr.shape != valid.shape:
        raise ValueError(f"Map/mask shape mismatch: {arr.shape} vs {valid.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} contains NaN or inf values.")
    mean, rms = _masked_mean_rms(arr, valid)
    return WfeMap(
        label=str(label),
        opd_nm=arr,
        mask=valid,
        rms_nm=rms,
        mean_nm=mean,
        shape=arr.shape,
        diagnostics=dict(diagnostics or {}),
        provenance=dict(provenance or {}),
    )


def white_noise_2d(shape: tuple[int, int], seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(_as_shape(shape))


def one_over_f_noise_2d(shape: tuple[int, int], alpha: float = 2.0, seed: int | None = None) -> np.ndarray:
    return generate_power_law_opd_map(shape, alpha=alpha, seed=seed, rms_opd_nm=1.0, mask=None)


def make_pupil_coordinates(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    y = np.linspace(-1.0, 1.0, int(shape[0]))
    x = np.linspace(-1.0, 1.0, int(shape[1]))
    return np.meshgrid(x, y)


def make_pupil_mask(shape: tuple[int, int], *, mode: str = "circular_fallback", radius: float = 1.0) -> np.ndarray:
    """Return a centered boolean pupil mask.

    Parameters
    ----------
    shape : tuple[int, int]
        Map shape in pixels.
    mode : str
        ``"circular_fallback"`` is the v1 lightweight default. ``"full"`` is
        available for compatibility tests and marks every pixel valid.
    radius : float
        Radius in normalized pupil coordinates for circular masks.

    Returns
    -------
    numpy.ndarray
        Boolean support mask with the requested shape.
    """

    shape = _as_shape(shape)
    if mode == "full":
        return np.ones(shape, dtype=bool)
    if mode != "circular_fallback":
        raise ValueError(f"Unsupported pupil mask mode {mode!r}.")
    x, y = make_pupil_coordinates(shape)
    return (x * x + y * y) <= float(radius) ** 2


def _valid_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    valid = np.asarray(mask, dtype=bool)
    if valid.shape != shape:
        raise ValueError(f"Mask shape {valid.shape} does not match map shape {shape}.")
    return valid


def normalize_opd_rms(opd_nm: np.ndarray, target_rms_nm: float, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(opd_nm, dtype=float)
    valid = _valid_mask(mask, arr.shape)
    rms = float(np.sqrt(np.mean(np.square(arr[valid]))))
    target = float(target_rms_nm)
    if target == 0.0:
        return np.zeros_like(arr)
    if rms == 0.0:
        raise ValueError("Cannot normalize a zero-RMS OPD map to nonzero RMS.")
    return arr * (target / rms)


def generate_power_law_opd_map(shape: tuple[int, int], *, alpha: float = 2.5, seed: int | None = None, rms_opd_nm: float = 20.0, mask: np.ndarray | None = None) -> np.ndarray:
    """Generate a deterministic correlated power-law OPD map.

    Fourier-domain complex white noise is scaled by radial frequency as
    ``f**(-alpha / 2)`` so the approximate 2D power spectrum follows
    ``1/f**alpha``. The DC term is suppressed, the inverse transform is real,
    and the result is normalized over ``mask`` to ``rms_opd_nm``.
    """

    shape = _as_shape(shape)
    rng = np.random.default_rng(seed)
    coeff = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    fy = np.fft.fftfreq(shape[0])
    fx = np.fft.fftfreq(shape[1])
    kx, ky = np.meshgrid(fx, fy)
    fr = np.sqrt(kx * kx + ky * ky)
    scale = np.zeros(shape, dtype=float)
    nz = fr > 0.0
    scale[nz] = fr[nz] ** (-float(alpha) / 2.0)
    arr = np.fft.ifft2(coeff * scale).real
    arr = arr - float(np.mean(arr))
    arr = normalize_opd_rms(arr, float(rms_opd_nm), mask=mask)
    if not np.all(np.isfinite(arr)):
        raise ValueError("Generated OPD map contains NaN or inf values.")
    return arr


def _zernike_basis(shape: tuple[int, int], noll_indices: list[int] | tuple[int, ...]) -> np.ndarray:
    if shape[0] != shape[1]:
        raise ValueError("Zernike fitting currently requires square OPD maps.")
    coords = dlu.pixel_coords(shape[0], 2.0)
    return np.asarray([np.asarray(dlu.zernike(int(i), coords, 2.0)) for i in noll_indices], dtype=float)


def fit_zernike_coefficients_nm(opd_nm: np.ndarray, noll_indices: list[int] | tuple[int, ...], mask: np.ndarray | None = None) -> dict[str, float]:
    """Fit Noll-indexed Zernike OPD coefficients over a pupil mask.

    Coefficients are returned in nanometres with stable labels ``Z<Noll>``.
    The fit is an ordinary least-squares projection on the masked pixels and is
    used for both PTT removal and low-order Z4-Z11 truth extraction.
    """

    arr = np.asarray(opd_nm, dtype=float)
    indices = [int(i) for i in noll_indices]
    if not indices:
        return {}
    basis = _zernike_basis(arr.shape, indices)
    valid = _valid_mask(mask, arr.shape)
    coeffs, *_ = np.linalg.lstsq(basis[:, valid].T, arr[valid], rcond=None)
    return {f"Z{i}": float(c) for i, c in zip(indices, coeffs)}


def fit_zernike_modes(opd_nm: np.ndarray, noll_indices: list[int], mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    coeff_dict = fit_zernike_coefficients_nm(opd_nm, noll_indices, mask=mask)
    coeffs = np.asarray([coeff_dict[f"Z{int(i)}"] for i in noll_indices], dtype=float)
    fitted = reconstruct_zernike_opd_nm(coeff_dict, np.asarray(opd_nm).shape, mask=mask)
    return coeffs, fitted


def reconstruct_zernike_opd_nm(coefficients_nm: Mapping[str, float], shape: tuple[int, int], mask: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct an OPD map from labelled Noll Zernike coefficients.

    Parameters
    ----------
    coefficients_nm : Mapping[str, float]
        Mapping such as ``{"Z4": 1.2}`` in nanometres.
    shape : tuple[int, int]
        Output map shape.
    mask : numpy.ndarray, optional
        Accepted for API symmetry; reconstruction is produced on the full grid.
    """

    del mask
    items = sorted(((int(k[1:]), float(v)) for k, v in coefficients_nm.items()), key=lambda x: x[0])
    if not items:
        return np.zeros(_as_shape(shape), dtype=float)
    indices = [i for i, _ in items]
    coeffs = np.asarray([v for _, v in items], dtype=float)
    basis = _zernike_basis(_as_shape(shape), indices)
    return np.tensordot(coeffs, basis, axes=(0, 0))


def remove_zernike_modes(opd_nm: np.ndarray, noll_indices: list[int] | None, mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if not noll_indices:
        return np.asarray(opd_nm, dtype=float), {"removed_noll_indices": [], "coefficients_nm": []}
    coeffs = fit_zernike_coefficients_nm(opd_nm, list(noll_indices), mask=mask)
    fitted = reconstruct_zernike_opd_nm(coeffs, np.asarray(opd_nm).shape, mask=mask)
    return np.asarray(opd_nm, dtype=float) - fitted, {"removed_noll_indices": list(noll_indices), "coefficients_nm": coeffs}


def remove_plane(opd_nm: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    return remove_zernike_modes(opd_nm, list(PTT_NOLL_INDICES), mask=mask)


def generate_high_order_wfe_map(shape: tuple[int, int], *, kind: str = "one_over_f", alpha: float = 2.0, rms_nm: float = 0.0, seed: int | None = None, remove_zernike_noll: list[int] | None = None, mask: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    if rms_nm == 0.0:
        arr = np.zeros(_as_shape(shape))
    elif kind == "white":
        arr = white_noise_2d(shape, seed=seed)
        arr = normalize_opd_rms(arr, rms_nm, mask=mask)
    else:
        arr = generate_power_law_opd_map(shape, alpha=alpha, seed=seed, rms_opd_nm=rms_nm, mask=mask)
    arr, zmeta = remove_zernike_modes(arr, remove_zernike_noll, mask=mask)
    arr = normalize_opd_rms(arr, rms_nm, mask=mask)
    valid = _valid_mask(mask, _as_shape(shape))
    measured = float(np.sqrt(np.mean(np.square(arr[valid])))) if np.any(valid) else 0.0
    return arr, {"shape": tuple(arr.shape), "units": "nm", "kind": kind, "alpha": float(alpha), "seed": seed, "target_rms_nm": float(rms_nm), "measured_rms_nm": measured, **zmeta}


def _seed(base_seed: int, mirror_offset: int, local_offset: int) -> int:
    return int(base_seed) + int(mirror_offset) * 1000 + int(local_offset)


def _mirror_offset(mirror: str) -> int:
    return {"primary": 1, "secondary": 2}.get(str(mirror), 9)


def build_mirror_wfe_deck(mirror: str, *, shape: tuple[int, int] = (128, 128), seed: int = 42, mask: np.ndarray | None = None, mask_policy: str = "circular_fallback", truth_rms_opd_nm: float = 20.0, truth_power_law_alpha: float = 2.5, low_order_noll_indices: tuple[int, ...] = DEFAULT_LOW_ORDER_NOLL_INDICES, low_order_sigma_nm_per_coeff: float = 2.0, high_order_error_rms_nm: float = 0.3, high_order_error_power_law_alpha: float | None = None) -> MirrorWfeDeck:
    """Build one deterministic mirror WFE truth/knowledge decomposition."""

    shape = _as_shape(shape)
    valid = make_pupil_mask(shape, mode=mask_policy) if mask is None else _valid_mask(mask, shape)
    mask_source = mask_policy if mask is None else "explicit"
    mo = _mirror_offset(mirror)
    truth_seed = _seed(seed, mo, 11)
    low_order_seed = _seed(seed, mo, 23)
    error_seed = _seed(seed, mo, 37)
    alpha_err = truth_power_law_alpha if high_order_error_power_law_alpha is None else float(high_order_error_power_law_alpha)

    raw = generate_power_law_opd_map(shape, alpha=truth_power_law_alpha, seed=truth_seed, rms_opd_nm=1.0, mask=valid)
    ptt_removed, ptt_meta = remove_zernike_modes(raw, list(PTT_NOLL_INDICES), mask=valid)
    full = normalize_opd_rms(ptt_removed, truth_rms_opd_nm, mask=valid)
    full_truth = _wfe_map(
        f"{mirror}_full_truth",
        full,
        valid,
        diagnostics={"requested_rms_nm": float(truth_rms_opd_nm), "ptt_removed_coefficients_nm_pre_norm": ptt_meta["coefficients_nm"]},
        provenance={"seed": truth_seed, "power_law_alpha": float(truth_power_law_alpha), "mask_policy": mask_source, "removed_noll_indices": list(PTT_NOLL_INDICES), "opd_unit": "nm"},
    )

    low_truth = fit_zernike_coefficients_nm(full, low_order_noll_indices, mask=valid)
    low_recon = reconstruct_zernike_opd_nm(low_truth, shape, mask=valid)
    residual = full - low_recon
    high_truth = _wfe_map(
        f"{mirror}_high_order_truth",
        residual,
        valid,
        diagnostics={"low_order_removed_noll_indices": list(low_order_noll_indices)},
        provenance={"derived_from": full_truth.label, "opd_unit": "nm"},
    )

    rng = np.random.default_rng(low_order_seed)
    low_errors = {f"Z{i}": float(rng.normal(0.0, low_order_sigma_nm_per_coeff)) for i in low_order_noll_indices}
    low_knowledge = {k: float(low_truth[k] + low_errors[k]) for k in low_truth}

    if float(high_order_error_rms_nm) == 0.0:
        error = np.zeros(shape, dtype=float)
    else:
        error = generate_power_law_opd_map(shape, alpha=alpha_err, seed=error_seed, rms_opd_nm=high_order_error_rms_nm, mask=valid)
        error, _ = remove_zernike_modes(error, list(PTT_NOLL_INDICES) + list(low_order_noll_indices), mask=valid)
        error = normalize_opd_rms(error, high_order_error_rms_nm, mask=valid)
    high_error = _wfe_map(
        f"{mirror}_high_order_error",
        error,
        valid,
        diagnostics={"requested_rms_nm": float(high_order_error_rms_nm)},
        provenance={"seed": error_seed, "power_law_alpha": alpha_err, "opd_unit": "nm"},
    )
    high_knowledge = _wfe_map(
        f"{mirror}_high_order_knowledge",
        high_truth.opd_nm + high_error.opd_nm,
        valid,
        diagnostics={"truth_rms_nm": high_truth.rms_nm, "error_rms_nm": high_error.rms_nm},
        provenance={"mode": "truth_plus_additive_correlated_error", "opd_unit": "nm"},
    )
    corr = float(np.corrcoef(high_truth.opd_nm[valid].ravel(), high_error.opd_nm[valid].ravel())[0, 1]) if high_truth.rms_nm > 0 and high_error.rms_nm > 0 else 0.0
    mapping = {f"Z{i}": {"noll_index": int(i), "active_index": n, "state_label": f"optics.{mirror}.zernike_coeffs_nm[{n}]"} for n, i in enumerate(low_order_noll_indices)}
    return MirrorWfeDeck(
        mirror=str(mirror),
        full_truth=full_truth,
        low_order_truth_coeffs_nm=low_truth,
        low_order_knowledge_coeffs_nm=low_knowledge,
        low_order_knowledge_error_nm=low_errors,
        high_order_truth=high_truth,
        high_order_knowledge=high_knowledge,
        high_order_knowledge_error=high_error,
        diagnostics={"low_order_mapping": mapping, "high_order_error_truth_correlation": corr, "residual_reconstruction_max_abs_nm": float(np.max(np.abs((low_recon + residual - full)[valid])))},
        provenance={"mirror": str(mirror), "base_seed": int(seed), "truth_seed": truth_seed, "low_order_knowledge_seed": low_order_seed, "high_order_error_seed": error_seed, "low_order_sigma_nm_per_coeff": float(low_order_sigma_nm_per_coeff), "high_order_error_rms_nm": float(high_order_error_rms_nm), "mask_policy": mask_source},
    )


def _cfg_get(mapping: Mapping[str, Any] | None, path: tuple[str, ...], default: Any = None) -> Any:
    cur: Any = mapping or {}
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_high_order_wfe_deck(*, shape: tuple[int, int] = (128, 128), seed: int = 42, primary_config: Mapping[str, Any] | None = None, secondary_config: Mapping[str, Any] | None = None, mask: np.ndarray | None = None, mask_policy: str = "circular_fallback", schema_version: str = SCHEMA_VERSION) -> HighOrderWfeDeck:
    """Build deterministic primary and secondary WFE decks.

    ``primary_config`` and ``secondary_config`` may be direct ``wfe`` mappings
    from the full-fidelity template. Missing fields use the v1 defaults:
    20 nm RMS OPD truth, alpha 2.5, 2 nm low-order coefficient knowledge error,
    and 0.3 nm RMS high-order additive map error.
    """

    def kwargs(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
        return {
            "truth_rms_opd_nm": float(_cfg_get(cfg, ("truth", "rms_opd_nm"), 20.0)),
            "truth_power_law_alpha": float(_cfg_get(cfg, ("truth", "power_law_alpha"), 2.5)),
            "low_order_noll_indices": tuple(int(i) for i in _cfg_get(cfg, ("truth", "fit_low_order_zernikes"), DEFAULT_LOW_ORDER_NOLL_INDICES)),
            "low_order_sigma_nm_per_coeff": float(_cfg_get(cfg, ("knowledge", "low_order_sigma_nm_per_coeff"), 2.0)),
            "high_order_error_rms_nm": float(_cfg_get(cfg, ("knowledge", "high_order_error_rms_nm"), 0.3)),
            "high_order_error_power_law_alpha": _cfg_get(cfg, ("knowledge", "high_order_error_power_law_alpha"), None),
        }

    primary = build_mirror_wfe_deck("primary", shape=shape, seed=seed, mask=mask, mask_policy=mask_policy, **kwargs(primary_config))
    secondary = build_mirror_wfe_deck("secondary", shape=shape, seed=seed, mask=mask, mask_policy=mask_policy, **kwargs(secondary_config))
    comparison = {
        "primary_full_truth_rms_nm": primary.full_truth.rms_nm,
        "secondary_full_truth_rms_nm": secondary.full_truth.rms_nm,
        "primary_high_order_truth_rms_nm": primary.high_order_truth.rms_nm,
        "secondary_high_order_truth_rms_nm": secondary.high_order_truth.rms_nm,
        "primary_secondary_full_truth_correlation": float(np.corrcoef(primary.full_truth.opd_nm[primary.full_truth.mask].ravel(), secondary.full_truth.opd_nm[secondary.full_truth.mask].ravel())[0, 1]),
    }
    return HighOrderWfeDeck(primary=primary, secondary=secondary, schema_version=schema_version, provenance={"base_seed": int(seed), "shape": tuple(shape), "mask_policy": mask_policy, "generated_timestamp_utc": datetime.now(timezone.utc).isoformat()}, comparison=comparison)


def _map_manifest(w: WfeMap) -> dict[str, Any]:
    return {"label": w.label, "shape": list(w.shape), "rms_nm": w.rms_nm, "mean_nm": w.mean_nm, "diagnostics": _jsonable(w.diagnostics), "provenance": _jsonable(w.provenance), "opd_unit": "nm"}


def _write_fits(path: Path, data: np.ndarray, *, unit: str, header_items: Mapping[str, Any]) -> None:
    h = fits.Header()
    h["BUNIT"] = unit
    for key, value in header_items.items():
        card = str(key).upper()[:8]
        if isinstance(value, (str, int, float, bool)) and value is not None:
            h[card] = value
    fits.PrimaryHDU(data=np.asarray(data), header=h).writeto(path, overwrite=True)


def _write_coeff_csvs(root: Path, deck: HighOrderWfeDeck) -> None:
    truth_rows = []
    knowledge_rows = []
    error_rows = []
    for mirror_deck in (deck.primary, deck.secondary):
        mapping = mirror_deck.diagnostics["low_order_mapping"]
        for label, meta in mapping.items():
            base = {"mirror": mirror_deck.mirror, "zernike_label": label, "noll_index": meta["noll_index"], "active_index": meta["active_index"], "state_label": meta["state_label"], "sigma_nm": mirror_deck.provenance["low_order_sigma_nm_per_coeff"], "seed": mirror_deck.provenance["low_order_knowledge_seed"]}
            truth_rows.append({**base, "truth_coeff_nm": mirror_deck.low_order_truth_coeffs_nm[label]})
            knowledge_rows.append({**base, "truth_coeff_nm": mirror_deck.low_order_truth_coeffs_nm[label], "knowledge_coeff_nm": mirror_deck.low_order_knowledge_coeffs_nm[label], "error_nm": mirror_deck.low_order_knowledge_error_nm[label]})
            error_rows.append({**base, "error_nm": mirror_deck.low_order_knowledge_error_nm[label]})
    for filename, rows in (("low_order_zernike_truth.csv", truth_rows), ("low_order_zernike_knowledge.csv", knowledge_rows), ("low_order_zernike_errors.csv", error_rows)):
        with (root / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def write_high_order_wfe_deck_artifacts(deck: HighOrderWfeDeck, outdir: str | Path) -> dict[str, str]:
    """Write FITS maps, CSV coefficient tables, and a JSON deck manifest."""

    root = Path(outdir)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    _write_coeff_csvs(root, deck)
    for name in ("low_order_zernike_truth.csv", "low_order_zernike_knowledge.csv", "low_order_zernike_errors.csv"):
        written[name] = str(root / name)

    for mirror_deck in (deck.primary, deck.secondary):
        prefix = mirror_deck.mirror
        maps = {
            f"{prefix}_full_truth_opd_nm.fits": mirror_deck.full_truth,
            f"{prefix}_high_order_truth_opd_nm.fits": mirror_deck.high_order_truth,
            f"{prefix}_high_order_knowledge_opd_nm.fits": mirror_deck.high_order_knowledge,
            f"{prefix}_high_order_error_opd_nm.fits": mirror_deck.high_order_knowledge_error,
        }
        for filename, wmap in maps.items():
            _write_fits(root / filename, wmap.opd_nm, unit="nm", header_items={"SCHEMA": deck.schema_version, "MIRROR": mirror_deck.mirror, "RMSNM": wmap.rms_nm, "OPDUNIT": "nm"})
            written[filename] = str(root / filename)
        mask_name = f"{prefix}_mask.fits"
        _write_fits(root / mask_name, mirror_deck.full_truth.mask.astype(np.uint8), unit="boolean", header_items={"SCHEMA": deck.schema_version, "MIRROR": mirror_deck.mirror})
        written[mask_name] = str(root / mask_name)

    manifest = {
        "schema_version": deck.schema_version,
        "generated_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "opd_unit": "nm",
        "coefficient_unit": "nm",
        "primary": {"maps": {"full_truth": _map_manifest(deck.primary.full_truth), "high_order_truth": _map_manifest(deck.primary.high_order_truth), "high_order_knowledge": _map_manifest(deck.primary.high_order_knowledge), "high_order_error": _map_manifest(deck.primary.high_order_knowledge_error)}, "low_order_truth_coeffs_nm": deck.primary.low_order_truth_coeffs_nm, "low_order_knowledge_coeffs_nm": deck.primary.low_order_knowledge_coeffs_nm, "low_order_knowledge_error_nm": deck.primary.low_order_knowledge_error_nm, "diagnostics": _jsonable(deck.primary.diagnostics), "provenance": _jsonable(deck.primary.provenance)},
        "secondary": {"maps": {"full_truth": _map_manifest(deck.secondary.full_truth), "high_order_truth": _map_manifest(deck.secondary.high_order_truth), "high_order_knowledge": _map_manifest(deck.secondary.high_order_knowledge), "high_order_error": _map_manifest(deck.secondary.high_order_knowledge_error)}, "low_order_truth_coeffs_nm": deck.secondary.low_order_truth_coeffs_nm, "low_order_knowledge_coeffs_nm": deck.secondary.low_order_knowledge_coeffs_nm, "low_order_knowledge_error_nm": deck.secondary.low_order_knowledge_error_nm, "diagnostics": _jsonable(deck.secondary.diagnostics), "provenance": _jsonable(deck.secondary.provenance)},
        "comparison": _jsonable(deck.comparison),
        "provenance": _jsonable(deck.provenance),
        "artifacts": written,
    }
    manifest_path = root / "high_order_wfe_deck_manifest.json"
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2, sort_keys=True))
    written["high_order_wfe_deck_manifest.json"] = str(manifest_path)
    return written


def realize_high_order_wfe_pair(shape: tuple[int, int], truth_cfg: Mapping[str, Any] | None, knowledge_cfg: Mapping[str, Any] | None, mask: np.ndarray | None = None) -> HighOrderWFERealization:
    truth_cfg = truth_cfg or {}
    truth, truth_meta = generate_high_order_wfe_map(shape, kind=str(truth_cfg.get("kind", truth_cfg.get("spectrum", "one_over_f"))), alpha=float(truth_cfg.get("alpha", truth_cfg.get("power_law_alpha", 2.0))), rms_nm=float(truth_cfg.get("rms_nm", truth_cfg.get("rms_opd_nm", 0.0))), seed=truth_cfg.get("seed"), remove_zernike_noll=truth_cfg.get("remove_zernike_noll"), mask=mask)
    knowledge_cfg = knowledge_cfg or {}
    if not knowledge_cfg.get("enabled", False):
        resid = np.zeros(_as_shape(shape))
        kmeta = {"enabled": False, "target_rms_nm": 0.0, "measured_rms_nm": 0.0}
    else:
        resid, kmeta = generate_high_order_wfe_map(shape, kind=str(knowledge_cfg.get("kind", "white")), alpha=float(knowledge_cfg.get("alpha", 2.0)), rms_nm=float(knowledge_cfg.get("rms_nm", 0.0)), seed=knowledge_cfg.get("seed"), remove_zernike_noll=knowledge_cfg.get("remove_zernike_noll"), mask=mask)
        kmeta["enabled"] = True
    return HighOrderWFERealization(truth_opd_nm=truth, inference_opd_nm=truth + resid, truth_metadata=truth_meta, knowledge_metadata=kmeta)


__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_LOW_ORDER_NOLL_INDICES",
    "PTT_NOLL_INDICES",
    "WfeMap",
    "MirrorWfeDeck",
    "HighOrderWfeDeck",
    "HighOrderWFERealization",
    "white_noise_2d",
    "one_over_f_noise_2d",
    "make_pupil_coordinates",
    "make_pupil_mask",
    "normalize_opd_rms",
    "generate_power_law_opd_map",
    "fit_zernike_coefficients_nm",
    "fit_zernike_modes",
    "reconstruct_zernike_opd_nm",
    "remove_zernike_modes",
    "remove_plane",
    "generate_high_order_wfe_map",
    "build_mirror_wfe_deck",
    "build_high_order_wfe_deck",
    "write_high_order_wfe_deck_artifacts",
    "realize_high_order_wfe_pair",
]
