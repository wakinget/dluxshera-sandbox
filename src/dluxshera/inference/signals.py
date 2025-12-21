from __future__ import annotations

"""
Signal-builder utilities for decoding optimizer traces into plot-ready time series.

The public entry point is :func:`build_signals`, which computes the “intro”
signal set used by the initial diagnostics panels. The builder expects a trace
with at least ``theta[T, D]`` and ``loss[T]``, run metadata, a decoder that maps
per-step ``theta`` vectors to decoded parameter values, and an optional truth
mapping. When truth is absent, truth-dependent signals are filled with NaNs to
keep the output shape stable.
"""

from typing import Callable, Mapping, MutableMapping, Optional

import numpy as np

from dluxshera.params.store import ParameterStore

DecodedMapping = Mapping[str, object]


def _coerce_mapping(decoded: object) -> DecodedMapping:
    if isinstance(decoded, ParameterStore):
        return decoded
    if isinstance(decoded, Mapping):
        return decoded
    raise TypeError(
        "decoder must return a Mapping or ParameterStore; "
        f"got {type(decoded).__name__}"
    )


def _lookup(decoded: DecodedMapping, key: str) -> np.ndarray:
    if isinstance(decoded, ParameterStore):
        return np.asarray(decoded.get(key))
    if key in decoded:
        return np.asarray(decoded[key])
    raise KeyError(f"Decoded mapping is missing required key {key!r}")


def _broadcast_truth(truth: Optional[Mapping[str, object]], key: str, shape) -> Optional[np.ndarray]:
    if truth is None:
        return None
    if key not in truth:
        return None
    value = np.asarray(truth[key])
    try:
        return np.broadcast_to(value, shape)
    except ValueError:
        return value


def _residual(est: np.ndarray, truth: Optional[np.ndarray]) -> np.ndarray:
    if truth is None:
        return np.full_like(est, np.nan, dtype=float)
    return est - truth


def _ppm_error(est: np.ndarray, truth: Optional[np.ndarray]) -> np.ndarray:
    if truth is None:
        return np.full_like(est, np.nan, dtype=float)
    return 1e6 * (est - truth) / truth


def build_signals(
    trace: Mapping[str, object],
    meta: Mapping[str, object],
    *,
    decoder: Callable[[np.ndarray], DecodedMapping] | DecodedMapping,
    truth: Optional[Mapping[str, object]] = None,
    signal_set: str = "intro",
) -> MutableMapping[str, np.ndarray]:
    """
    Build a dictionary of diagnostic signals from an optimizer trace.

    Parameters
    ----------
    trace:
        Mapping containing at least ``theta`` and ``loss`` arrays.
    meta:
        Run metadata (currently unused but accepted for future extensibility).
    decoder:
        Callable mapping a per-step ``theta`` vector to decoded parameter
        values, or a static mapping/ParameterStore used for all steps.
    truth:
        Optional mapping of truth values for residual computation. When
        ``None`` or when a specific truth key is absent, truth-dependent
        signals are filled with NaNs to preserve shapes.
    signal_set:
        Signal recipe name. Only ``"intro"`` is supported currently.

    Returns
    -------
    signals:
        Mapping from signal names to numpy arrays.
    """

    if signal_set != "intro":
        raise ValueError(f"Unsupported signal_set {signal_set!r} (expected 'intro').")

    theta = np.asarray(trace["theta"])
    T = theta.shape[0]

    decoded_steps: list[DecodedMapping] = []
    if callable(decoder):
        for t in range(T):
            decoded_steps.append(_coerce_mapping(decoder(theta[t])))
    else:
        decoded_steps = [_coerce_mapping(decoder) for _ in range(T)]

    def stack_decoded(key: str) -> np.ndarray:
        values = [_lookup(decoded, key) for decoded in decoded_steps]
        return np.stack(values, axis=0)

    signals: MutableMapping[str, np.ndarray] = {}

    x_est = stack_decoded("binary.x_position_as")
    y_est = stack_decoded("binary.y_position_as")
    sep_est = stack_decoded("binary.separation_as")
    ps_est = stack_decoded("system.plate_scale_as_per_pix")
    raw_flux_est = stack_decoded("binary.raw_fluxes")
    zern_est = stack_decoded("primary.zernike_coeffs_nm")

    x_true = _broadcast_truth(truth, "binary.x_position_as", x_est.shape)
    y_true = _broadcast_truth(truth, "binary.y_position_as", y_est.shape)
    sep_true = _broadcast_truth(truth, "binary.separation_as", sep_est.shape)
    ps_true = _broadcast_truth(truth, "system.plate_scale_as_per_pix", ps_est.shape)
    raw_flux_true = _broadcast_truth(truth, "binary.raw_fluxes", raw_flux_est.shape)
    zern_true = _broadcast_truth(truth, "primary.zernike_coeffs_nm", zern_est.shape)

    signals["binary.x_error_uas"] = 1e6 * _residual(x_est, x_true).reshape((T,))
    signals["binary.y_error_uas"] = 1e6 * _residual(y_est, y_true).reshape((T,))
    signals["binary.separation_error_uas"] = 1e6 * _residual(sep_est, sep_true).reshape((T,))

    ps_residual = _residual(ps_est, ps_true)
    if ps_true is None:
        signals["system.plate_scale_error_ppm"] = np.full((T,), np.nan, dtype=float)
    else:
        signals["system.plate_scale_error_ppm"] = 1e6 * (ps_residual / ps_true).reshape((T,))

    signals["binary.raw_flux_error_ppm"] = _ppm_error(raw_flux_est, raw_flux_true)

    zern_error = _residual(zern_est, zern_true)
    signals["primary.zernike_error_nm"] = zern_error
    zern_mask = np.isfinite(zern_error)
    sum_sq = np.nansum(np.square(zern_error), axis=-1)
    counts = np.sum(zern_mask, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        zern_rms = np.sqrt(sum_sq / counts)
    zern_rms = np.where(counts > 0, zern_rms, np.nan)
    signals["primary.zernike_rms_nm"] = zern_rms

    return signals


__all__ = ["build_signals"]
