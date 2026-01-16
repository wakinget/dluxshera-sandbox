"""Printing helpers for optimization summaries."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

Labeler = Callable[[str, int], Sequence[str] | str | None]


def _as_np(value: Any) -> np.ndarray:
    return np.asarray(value)


def _fmt_scalar(value: Any, *, prec: int = 8) -> str:
    try:
        return f"{float(value):.{prec}g}"
    except Exception:
        return str(value)


def _is_scalar(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    return arr.ndim == 0 or arr.size == 1


def _resolve_labels(
    labels: Mapping[str, Sequence[str] | str] | Labeler | None,
    key: str,
    n: int,
) -> list[str]:
    if labels is None:
        return [str(i) for i in range(n)]

    if callable(labels):
        resolved = labels(key, n)
    elif isinstance(labels, Mapping):
        resolved = labels.get(key)
    else:
        resolved = None

    if resolved is None:
        return [str(i) for i in range(n)]

    if isinstance(resolved, str):
        if n == 1:
            return [resolved]
        print(
            f"    [WARN] label mismatch for {key}: expected {n}, got scalar label",
        )
        return [str(i) for i in range(n)]

    label_list = list(resolved)
    if len(label_list) != n:
        print(
            f"    [WARN] label mismatch for {key}: expected {n}, got {len(label_list)}",
        )
    return label_list


def _print_vector(
    key: str,
    true_val: np.ndarray,
    init_val: np.ndarray,
    final_val: np.ndarray,
    *,
    labels: Mapping[str, Sequence[str] | str] | Labeler | None = None,
    prec: int = 8,
) -> None:
    t = np.ravel(_as_np(true_val))
    i = np.ravel(_as_np(init_val))
    f = np.ravel(_as_np(final_val))

    if t.size != i.size or t.size != f.size:
        print(f"    [WARN] size mismatch: true={t.size}, init={i.size}, final={f.size}")
        n = min(t.size, i.size, f.size)
        t, i, f = t[:n], i[:n], f[:n]

    n = t.size
    label_list = _resolve_labels(labels, key, n)
    if len(label_list) != n:
        n = min(n, len(label_list))
        t, i, f = t[:n], i[:n], f[:n]
        label_list = label_list[:n]

    for idx, lab, tv, iv, fv in zip(range(n), label_list, t, i, f):
        dt_i = float(iv - tv)
        dt_f = float(fv - tv)
        print(
            f"    [{idx:>3}] {lab:>4} : "
            f"true={_fmt_scalar(tv, prec=prec)}  "
            f"init={_fmt_scalar(iv, prec=prec)}  (Δ={_fmt_scalar(dt_i, prec=prec)})  "
            f"final={_fmt_scalar(fv, prec=prec)} (Δ={_fmt_scalar(dt_f, prec=prec)})"
        )


def print_optimization_summary(
    true_vals: Mapping[str, Any],
    init_vals: Mapping[str, Any],
    final_vals: Mapping[str, Any],
    *,
    header: str | None = None,
    labels: Mapping[str, Sequence[str] | str] | Labeler | None = None,
    scalar_prec: int = 8,
) -> None:
    if header is not None:
        print(header)
        print("")

    for key in true_vals:
        true_val = true_vals.get(key)
        init_val = init_vals.get(key)
        final_val = final_vals.get(key)

        t = _as_np(true_val)
        i = _as_np(init_val)
        f = _as_np(final_val)

        print(f"- {key}")
        if _is_scalar(t) and _is_scalar(i) and _is_scalar(f):
            tv = t.reshape(()) if t.size == 1 else t
            iv = i.reshape(()) if i.size == 1 else i
            fv = f.reshape(()) if f.size == 1 else f
            print(f"    true : {_fmt_scalar(tv, prec=scalar_prec)}")
            print(
                f"    init : {_fmt_scalar(iv, prec=scalar_prec)}  "
                f"(Δ={_fmt_scalar(float(iv - tv), prec=scalar_prec)})"
            )
            print(
                f"    final: {_fmt_scalar(fv, prec=scalar_prec)}  "
                f"(Δ={_fmt_scalar(float(fv - tv), prec=scalar_prec)})"
            )
        else:
            print(f"    shape true/init/final: {t.shape} / {i.shape} / {f.shape}")
            _print_vector(
                key,
                t,
                i,
                f,
                labels=labels,
                prec=scalar_prec,
            )

        print("")
