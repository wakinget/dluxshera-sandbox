"""Diagnostic helpers for post-run gradient analysis."""
from __future__ import annotations

import importlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .preconditioning import PreconditioningConfig, compute_precond_vectors
from .run_artifacts import load_checkpoint, load_meta, load_summary


def _resolve_builder(builder: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(builder):
        return builder
    if isinstance(builder, str):
        if ":" not in builder:
            raise ValueError("Builder string must be of the form 'module:func'.")
        module_path, func_name = builder.split(":", maxsplit=1)
        module = importlib.import_module(module_path)
        func = getattr(module, func_name)
        if not callable(func):
            raise TypeError(f"Resolved builder {builder!r} is not callable.")
        return func
    raise TypeError("builder must be a callable or 'module:func' string.")


def _call_builder(builder: Callable[..., Any], meta: Mapping[str, Any], run_dir: Path) -> Callable[[np.ndarray], Any]:
    sig = inspect.signature(builder)
    kwargs = {}
    if "meta" in sig.parameters:
        kwargs["meta"] = meta
    if "run_dir" in sig.parameters:
        kwargs["run_dir"] = run_dir
    return builder(**kwargs) if kwargs else builder()


def _load_theta_from_checkpoint(checkpoint_path: Path, checkpoint: str) -> np.ndarray:
    payload = load_checkpoint(checkpoint_path.parent, which=checkpoint)
    if checkpoint == "best":
        for key in ("theta_best", "theta"):
            if key in payload:
                return np.asarray(payload[key])
    if checkpoint == "final":
        for key in ("theta_final", "theta"):
            if key in payload:
                return np.asarray(payload[key])
    # fallback: take the first array
    if payload:
        return np.asarray(next(iter(payload.values())))
    raise ValueError(f"No theta found in checkpoint_{checkpoint}.npz")


def _compute_block_norms(grad: np.ndarray, meta: Mapping[str, Any]) -> dict[str, float]:
    theta_meta = meta.get("theta") if isinstance(meta, Mapping) else None
    if not isinstance(theta_meta, Mapping):
        return {}
    index_map = theta_meta.get("index_map")
    if not isinstance(index_map, Mapping):
        return {}
    entries: Sequence[Mapping[str, Any]] = index_map.get("entries", [])
    block_norms: dict[str, float] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        start = int(entry.get("start", 0))
        stop = int(entry.get("stop", start))
        block = entry.get("block") or entry.get("name") or f"slice_{start}_{stop}"
        if start < 0 or stop > grad.size or start >= stop:
            continue
        g_slice = grad[start:stop]
        block_norms[str(block)] = float(np.linalg.norm(g_slice))
    return block_norms


def compute_checkpoint_gradients(
    run_dir: Path,
    *,
    builder: str | Callable[..., Any],
    checkpoint: str = "best",
    compute_curvature: bool = False,
) -> dict[str, Any]:
    """Compute and save gradients at a stored checkpoint.

    Parameters
    ----------
    run_dir : Path
        Run directory containing ``checkpoint_<which>.npz`` and metadata files.
    builder : str | Callable[..., Any]
        Builder that returns a loss function of the form ``loss(theta) -> scalar``.
        If a string, it must be ``\"module:func\"``. The builder may accept
        ``meta`` and/or ``run_dir`` keyword arguments.
    checkpoint : str
        Which checkpoint to load (\"best\" or \"final\"). Defaults to \"best\".
    compute_curvature : bool
        When True, also compute a diagonal curvature proxy and learning-rate
        vector using the Phase D preconditioning helper.

    Returns
    -------
    dict
        Summary dictionary containing norms and file paths.
    """

    run_dir = Path(run_dir)
    meta = load_meta(run_dir)
    summary = load_summary(run_dir) if (run_dir / "summary.json").exists() else {}

    theta = _load_theta_from_checkpoint(run_dir / f"checkpoint_{checkpoint}.npz", checkpoint)

    builder_fn = _resolve_builder(builder)
    loss_fn = _call_builder(builder_fn, meta, run_dir)
    if not callable(loss_fn):
        raise TypeError("Builder must return a callable loss function.")

    loss_fn_wrapped = lambda th: jnp.asarray(loss_fn(jnp.asarray(th)))
    grad_fn = jax.grad(loss_fn_wrapped)
    grad = np.asarray(grad_fn(theta))

    grad_norm = float(np.linalg.norm(grad))
    max_abs_grad = float(np.max(np.abs(grad))) if grad.size else 0.0
    block_norms = _compute_block_norms(grad, meta)

    diag_dir = run_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    npz_name = f"grad_at_{checkpoint}.npz"
    npz_path = diag_dir / npz_name

    diag_payload: dict[str, Any] = {
        "grad": grad,
        "grad_norm": grad_norm,
        "max_abs_grad": max_abs_grad,
    }

    curv_meta = None
    if compute_curvature:
        optimizer_meta = meta.get("optimizer") if isinstance(meta, Mapping) else {}
        base_lr = None
        if isinstance(optimizer_meta, Mapping):
            base_lr = optimizer_meta.get("base_lr")
        cfg = PreconditioningConfig(base_lr=base_lr)
        precond_outputs = compute_precond_vectors(
            loss_fn=loss_fn_wrapped,
            theta0=jnp.asarray(theta),
            method_cfg=cfg,
            index_map=meta.get("theta", {}).get("index_map") if isinstance(meta, Mapping) else None,
        )
        diag_payload["curv_diag"] = precond_outputs["curv_diag"]
        diag_payload["lr_vec"] = precond_outputs["lr_vec"]
        curv_meta = precond_outputs["config"]

    np.savez_compressed(npz_path, **diag_payload)

    summary_payload: dict[str, Any] = {
        "run_id": summary.get("run_id") or meta.get("run_id") or run_dir.name,
        "checkpoint": checkpoint,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "theta_dim": int(theta.size),
        "grad_norm": grad_norm,
        "max_abs_grad": max_abs_grad,
        "block_grad_norms": block_norms,
        "artifact": str(npz_path),
        "builder": builder if isinstance(builder, str) else getattr(builder, "__name__", "callable"),
    }
    if curv_meta is not None:
        summary_payload["curvature"] = curv_meta

    json_path = diag_dir / "grad_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    return summary_payload
