from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable, Mapping, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np


PrecondMethod = Union[
    str,
    "PreconditioningConfig",
]


@dataclass
class PreconditioningConfig:
    method: str = "ema_grad2"
    eps: float = 1e-8
    lr_clip: Optional[Tuple[float, float]] = None
    curv_floor: Optional[float] = None
    rng_seed: Optional[int] = None
    refresh_every: Optional[int] = None
    base_lr: Optional[float] = None

    def to_meta(self) -> Mapping[str, object]:
        payload = asdict(self)
        if self.lr_clip is not None:
            payload["lr_clip"] = list(self.lr_clip)
        return payload


def _coerce_config(cfg: Union[PreconditioningConfig, Mapping[str, object]]) -> PreconditioningConfig:
    if isinstance(cfg, PreconditioningConfig):
        return cfg
    if isinstance(cfg, Mapping):
        return PreconditioningConfig(**cfg)
    raise TypeError("method_cfg must be a PreconditioningConfig or mapping.")


def _validate_alignment(vec: jnp.ndarray, theta: jnp.ndarray, index_map: Optional[Mapping[str, object]]) -> None:
    dim = int(theta.size)
    if vec.size != dim:
        raise ValueError(
            f"Preconditioning vector length {vec.size} does not match theta dimension {dim}."
        )

    if index_map is None:
        return

    entries = index_map.get("entries") if isinstance(index_map, Mapping) else None
    if not entries:
        return

    last_stop = entries[-1].get("stop") if isinstance(entries[-1], Mapping) else None
    if last_stop is None:
        return
    if int(last_stop) != dim:
        raise ValueError(
            "IndexMap length mismatch: last stop "
            f"{last_stop} does not match theta dimension {dim}."
        )


def compute_precond_vectors(
    *,
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray],
    theta0: jnp.ndarray,
    method_cfg: Union[PreconditioningConfig, Mapping[str, object]],
    index_map: Optional[Mapping[str, object]] = None,
    rng: Optional[jax.random.KeyArray] = None,
) -> dict[str, np.ndarray]:
    """Compute lr_vec (and related vectors) in θ-space.

    Parameters
    ----------
    loss_fn :
        Callable accepting a 1D JAX array and returning a scalar loss.
    theta0 :
        Reference θ vector at which to estimate curvature/preconditioning.
    method_cfg :
        Configuration describing the preconditioning method. Accepts a
        :class:`PreconditioningConfig` or a mapping that can initialise one.
    index_map :
        Optional IndexMap used to validate alignment. If provided, the final
        entry's ``stop`` value must match ``theta0.size``.
    rng :
        Optional PRNG key for stochastic methods. Not used for the current
        deterministic implementation.
    """

    del rng  # placeholder for stochastic variants

    cfg = _coerce_config(method_cfg)
    theta0 = jnp.asarray(theta0)

    if cfg.method != "ema_grad2":
        raise ValueError(f"Unsupported preconditioning method: {cfg.method!r}")

    base_lr = cfg.base_lr if cfg.base_lr is not None else 1.0

    grad_fn = jax.grad(lambda th: jnp.asarray(loss_fn(th)))
    g0 = grad_fn(theta0)
    curv_diag = jnp.square(g0)

    if cfg.curv_floor is not None:
        curv_diag = jnp.maximum(curv_diag, cfg.curv_floor)

    precond = jnp.reciprocal(jnp.sqrt(curv_diag + cfg.eps))
    lr_vec = precond * base_lr

    if cfg.lr_clip is not None:
        lr_min, lr_max = cfg.lr_clip
        lr_vec = jnp.clip(lr_vec, lr_min, lr_max)

    _validate_alignment(lr_vec, theta0, index_map)

    for name, arr in {
        "curv_diag": curv_diag,
        "precond": precond,
        "lr_vec": lr_vec,
    }.items():
        if not jnp.all(jnp.isfinite(arr)):
            raise ValueError(f"Non-finite values encountered in {name}.")

    return {
        "curv_diag": np.asarray(curv_diag),
        "precond": np.asarray(precond),
        "lr_vec": np.asarray(lr_vec),
        "config": replace(cfg, base_lr=float(base_lr)).to_meta(),
    }

