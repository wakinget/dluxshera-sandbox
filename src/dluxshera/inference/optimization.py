# src/dluxshera/inference/optimization.py
from __future__ import annotations

import jax
import jax.numpy as np
import jax.scipy.stats as jstats
import numpy as onp
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import optax
from typing import Optional, Literal, Callable, Sequence, Tuple, Dict, Any, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .losses import gaussian_image_nll
from .run_artifacts import _now_iso_local_ms, save_run
from .preconditioning import PreconditioningConfig, compute_precond_vectors

from ..systems.three_plane import SheraThreePlaneConfig
from ..systems.two_plane import SheraTwoPlaneConfig
from ..config import resolve_config
from ..params.spec import ParamSpec, ParamKey
from ..params.store import ParameterStore, subset_store
from ..params.packing import (
    build_index_map,
    pack_params as store_pack_params,
    unpack_params as store_unpack_params,
)

if TYPE_CHECKING:
    from ..systems import SheraBinder


def _build_artifacts_mapping(
    *,
    checkpoints: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None,
    metric: Optional[Mapping[str, np.ndarray]] = None,
    signals: Optional[Mapping[str, np.ndarray]] = None,
) -> Optional[dict[str, dict[str, object]]]:
    artifacts: dict[str, dict[str, object]] = {}

    if signals is not None:
        artifacts["signals"] = {"kind": "npz", "content": signals}

    if metric is not None:
        artifacts["metric"] = {"kind": "npz", "content": metric}

    if checkpoints is not None:
        for name, payload in checkpoints.items():
            artifact_name = f"checkpoint_{name}"
            artifacts[artifact_name] = {
                "kind": "npz",
                "content": payload,
                "filename": f"{artifact_name}.npz",
            }

    return artifacts or None

############################
# Exports
############################

__all__ = [

    # new likelihood / loss utilities
    "gaussian_image_nll",
    "gaussian_loglikelihood_image",
    "poisson_loglikelihood_image",
    "gaussian_loss",
    "poisson_loss",
    "make_loss_fn",
    "make_image_nll_fn",
    "make_binder_nll_fn",
    "diagnose_first_step",

    # simple θ–space optimizer
    "run_simple_gd",
    "run_gd_with_artifacts",
    "run_shera_gd",
    "run_image_gd",

    # θ-space eigen reparameterisation
    "EigenThetaMap",

    # legacy likelihood / step fns (model-params space)
    "loglikelihood", "loss_fn",

    # reparameterisation utils
    "generate_fim_labels",
    "map_labels_to_keys",
    "build_fim_diagonal_preconditioner",
]




# NOTE:
# - New Binder + θ-space loss/FIM/optim live here.
#   Once the new path stabilises, we will split FIM/eigen utilities into
#   a dedicated inference.fim / inference.eigen module.


############################
# New P0 likelihood kernels
############################

NoiseModel = Literal["gaussian", "poisson"]

def gaussian_loglikelihood_image(
    model_image: np.ndarray,
    data_image: np.ndarray,
    var_image: np.ndarray,
    *,
    reduce: Optional[Literal["sum", "mean"]] = "sum",
) -> np.ndarray:
    """
    Per-pixel Gaussian log-likelihood for an image.

    Parameters
    ----------
    model_image :
        Model prediction (expected value) per pixel.
    data_image :
        Observed data per pixel.
    var_image :
        Per-pixel variance (σ^2). Must be positive; no check is enforced here.
    reduce :
        If "sum", returns scalar sum over pixels.
        If "mean", returns mean over pixels.
        If None, returns the per-pixel log-likelihood array.

    Returns
    -------
    np.ndarray
        Log-likelihood (scalar if reduced, otherwise same shape as inputs).
    """
    # Convert variance → standard deviation
    sigma = np.sqrt(var_image)

    # jstats.norm.logpdf expects (x, loc, scale)
    logp = jstats.norm.logpdf(data_image, loc=model_image, scale=sigma)

    if reduce == "sum":
        return np.nansum(logp)
    elif reduce == "mean":
        return np.nanmean(logp)
    else:
        return logp


def poisson_loglikelihood_image(
    model_image: np.ndarray,
    data_image: np.ndarray,
    *,
    reduce: Optional[Literal["sum", "mean"]] = "sum",
) -> np.ndarray:
    """
    Per-pixel Poisson log-likelihood for an image.

    Parameters
    ----------
    model_image :
        Expected counts λ per pixel (must be non-negative).
    data_image :
        Observed integer counts per pixel.
    reduce :
        If "sum", returns scalar sum over pixels.
        If "mean", returns mean over pixels.
        If None, returns the per-pixel log-likelihood array.

    Notes
    -----
    This uses jax.scipy.stats.poisson.logpmf, which implements:
        logP(d | λ) = d * log(λ) - λ - log(d!)

    The -log(d!) term does not depend on the model, but is included in
    the result; it's harmless for optimization.
    """
    logp = jstats.poisson.logpmf(data_image, model_image)

    if reduce == "sum":
        return np.nansum(logp)
    elif reduce == "mean":
        return np.nanmean(logp)
    else:
        return logp



def gaussian_loss(
    model,
    data: np.ndarray,
    var: np.ndarray,
    *,
    reduce: Literal["sum", "mean"] = "sum",
) -> np.ndarray:
    """
    Negative Gaussian log-likelihood for a Shera model object.

    Parameters
    ----------
    model :
        Object with a `.model()` method returning the predicted image.
    data :
        Observed image.
    var :
        Per-pixel variance for the Gaussian noise model.
    reduce :
        "sum" or "mean" reduction over pixels.

    Returns
    -------
    np.ndarray
        Scalar loss (negative log-likelihood with the chosen reduction).
    """
    model_image = model.model()
    return gaussian_image_nll(model_image, data, var, reduce=reduce)


def poisson_loss(
    model,
    data: np.ndarray,
    var: Optional[np.ndarray] = None,
    *,
    reduce: Literal["sum", "mean"] = "sum",
) -> np.ndarray:
    """
    Negative Poisson log-likelihood for a Shera model object.

    Parameters
    ----------
    model :
        Object with a `.model()` method returning the predicted expected
        counts (λ) per pixel.
    data :
        Observed integer counts per pixel.
    var :
        Unused; kept for API compatibility with Gaussian loss.
    reduce :
        "sum" or "mean" reduction over pixels.

    Returns
    -------
    np.ndarray
        Scalar loss (negative log-likelihood with the chosen reduction).
    """
    del var  # unused
    model_image = model.model()
    loglike = poisson_loglikelihood_image(model_image, data, reduce=reduce)
    return -loglike



def make_loss_fn(
    noise_model: NoiseModel = "gaussian",
    *,
    reduce: Literal["sum", "mean"] = "sum",
):
    """
    Factory returning a loss function with signature (model, data, var) -> loss.

    This is designed to plug directly into your existing step functions
    that expect a `loss_fn(model, data, var)` callable.
    """
    if noise_model == "gaussian":
        def _loss(model, data, var):
            return gaussian_loss(model, data, var, reduce=reduce)
        return _loss

    elif noise_model == "poisson":
        def _loss(model, data, var):
            # `var` is ignored inside poisson_loss, but we keep it in
            # the signature so existing step functions still work.
            return poisson_loss(model, data, var, reduce=reduce)
        return _loss

    else:
        raise ValueError(f"Unknown noise_model {noise_model!r} (expected 'gaussian' or 'poisson').")


# -------------------------------------------------------------------------
# Bridge from (cfg, spec, store, infer_keys, data, var) → loss(theta)
# -------------------------------------------------------------------------

# NLL -> Negative Log-Likelihood
def make_image_nll_fn(
    cfg: SheraThreePlaneConfig,
    forward_spec: ParamSpec,
    base_store: ParameterStore,
    infer_keys: Sequence[ParamKey],
    data: np.ndarray,
    var: np.ndarray,
    *,
    noise_model: NoiseModel = "gaussian",
    reduce: Literal["sum", "mean"] = "sum",
    build_model_fn: Optional[
        Callable[[SheraThreePlaneConfig, ParamSpec, ParameterStore], Any]
    ] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Build a loss(theta) function for image-based Shera inference.

    Parameters
    ----------
    cfg
        Structural SheraThreePlaneConfig (testbed vs flight, etc.).
    forward_spec
        Full forward ParamSpec catalog. Inference-space layout is defined by
        ``forward_spec.subset(infer_keys)``.
    base_store
        ParameterStore providing a baseline set of parameter values.
        This is the state we *overlay* when unpacking theta.
    infer_keys
        Sequence of ParamKeys to include in theta, e.g.
        ["binary.separation_as", "binary.x_position_as", ...].
    data
        Observed image (PSF cutout etc.), as a JAX array.
    var
        Per-pixel variance map, same shape as `data`. For Poisson
        noise it is ignored but kept for API compatibility.
    noise_model
        "gaussian" or "poisson": selects which NLL kernel to use.
    reduce
        "sum" or "mean" reduction over pixels inside the NLL kernels.
    build_model_fn
        Callable (cfg, forward_spec, store) -> model. If None, a
        default Shera three-plane builder is imported lazily.

    Returns
    -------
    loss_fn
        Callable taking a flat theta vector and returning scalar NLL.
        Signature: loss_fn(theta) -> scalar.
    theta0
        Initial packed parameter vector constructed from
        (forward_spec.subset(infer_keys), base_store).
    """
    # Inference layout is derived directly from the forward spec.
    sub_spec = forward_spec.subset(infer_keys)

    # Pack base_store → theta0 (this defines ordering of infer_keys)
    theta0 = store_pack_params(sub_spec, base_store)

    # Choose a per-model loss kernel (model, data, var) -> loss
    if noise_model == "gaussian":
        def _model_loss(m, d, v):
            return gaussian_loss(m, d, v, reduce=reduce)
    elif noise_model == "poisson":
        def _model_loss(m, d, v):
            # `var` is ignored inside poisson_loss, but we keep it in
            # the signature so existing step functions still work.
            return poisson_loss(m, d, v, reduce=reduce)
    else:
        raise ValueError(
            f"Unknown noise_model {noise_model!r} "
            f"(expected 'gaussian' or 'poisson')."
        )

    # Lazily resolve the model-building function to avoid circular imports
    if build_model_fn is None:
        from ..legacy.builders import build_legacy_shera_threeplane_model
        build_model_fn = build_legacy_shera_threeplane_model

    data = np.asarray(data)
    var = np.asarray(var)

    def loss_fn(theta: np.ndarray) -> np.ndarray:
        # Unpack theta back into a ParameterStore, overlaying on base_store
        # NOTE: unpack_params(spec_subset, theta, base_store)
        store_theta = store_unpack_params(sub_spec, theta, base_store)

        # Build model from cfg + (forward_spec, store_theta)
        model = build_model_fn(cfg, forward_spec, store_theta)

        # Evaluate loss
        return _model_loss(model, data, var)

    return loss_fn, theta0


def make_binder_nll_fn(
    *,
    binder: object,
    infer_keys: Sequence[ParamKey],
    data: np.ndarray,
    var: np.ndarray,
    noise_model: NoiseModel = "gaussian",
    reduce: Literal["sum", "mean"] = "sum",
    theta0_store: ParameterStore | None = None,
    return_predict_fn: bool = False,
):
    """
    Build a Binder-first θ-space image NLL closure.

    Parameters
    ----------
    binder :
        Pre-built Shera binder providing ``forward_spec`` and
        ``base_forward_store``. All θ unpacking overlays the binder's base
        store to guarantee semantic alignment with the data-generating
        forward path.
    infer_keys :
        Sequence of keys to include in θ (ordering defines packing order).
    data, var :
        Observed image and per-pixel variance (Gaussian) or placeholder array
        (Poisson). Both are converted to JAX arrays.
    noise_model, reduce :
        Noise model selector and reduction for the NLL.
    theta0_store :
        Optional store used *only* to initialise ``theta0``. The binder's base
        store is still used as the unpack overlay. If ``None``, uses
        ``binder.base_forward_store``.
    return_predict_fn :
        If True, also return ``predict_fn(theta) -> image`` that mirrors the
        loss path.
    """
    forward_spec = binder.forward_spec
    base_forward_store = binder.base_forward_store

    sub_spec = forward_spec.subset(infer_keys)

    data = np.asarray(data)
    var = np.asarray(var)

    theta0_store = base_forward_store if theta0_store is None else theta0_store
    theta0 = store_pack_params(sub_spec, theta0_store, dtype=data.dtype)

    if noise_model == "gaussian":
        def image_nll(model_image):
            return gaussian_image_nll(model_image, data, var, reduce=reduce)
    elif noise_model == "poisson":
        def image_nll(model_image):
            return -poisson_loglikelihood_image(model_image, data, reduce=reduce)
    else:
        raise ValueError(
            f"Unknown noise_model {noise_model!r} "
            f"(expected 'gaussian' or 'poisson')."
        )

    def theta_to_store_delta(theta: np.ndarray) -> ParameterStore:
        full_store = store_unpack_params(sub_spec, theta, base_forward_store)
        store_delta = subset_store(full_store, infer_keys)

        # Keep binder.model on the non-structural fast path when the binder
        # exposes the full convenience API, but allow lightweight binder test
        # doubles that only advertise structural keys.
        strip_structural = getattr(binder, "strip_structural", None)
        if callable(strip_structural):
            return strip_structural(store_delta)

        structural_store_keys = getattr(binder, "structural_store_keys", None)
        if callable(structural_store_keys):
            structural = structural_store_keys()
            if structural:
                filtered = {
                    key: value
                    for key, value in store_delta.items()
                    if key not in structural
                }
                return ParameterStore.from_dict(filtered)

        return store_delta

    def loss_fn(theta: np.ndarray) -> np.ndarray:
        store_delta = theta_to_store_delta(theta)
        model_image = binder.model(store_delta)
        return image_nll(model_image)

    if return_predict_fn:
        def predict_fn(theta: np.ndarray) -> np.ndarray:
            store_delta = theta_to_store_delta(theta)
            return binder.model(store_delta)

        return loss_fn, theta0, predict_fn

    return loss_fn, theta0


def make_binder_image_nll_fn(
    cfg,
    forward_spec: ParamSpec,
    base_forward_store: ParameterStore,
    infer_keys: Sequence[ParamKey],
    data: np.ndarray,
    var: np.ndarray,
    *,
    binder: Optional[object] = None,
    noise_model: NoiseModel = "gaussian",
    reduce: Literal["sum", "mean"] = "sum",
    return_predict_fn: bool = False,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray] | Tuple[
    Callable[[np.ndarray], np.ndarray], np.ndarray, Callable[[np.ndarray], np.ndarray]
]:
    """
    Canonical θ-space image NLL using :class:`SheraBinder`.

    The returned loss is intentionally explicit:

    ``theta`` → ``ParameterStore`` delta → ``binder.model(store_delta)`` →
    image → Gaussian/Poisson NLL.

    Parameters
    ----------
    cfg, forward_spec, base_forward_store
        Inputs required to construct a binder if one is not supplied.
    infer_keys
        Ordering of parameters packed into ``theta``.
    data, var
        Observed image and per-pixel variance.
    binder
        Optional pre-built :class:`SheraBinder`. If omitted a binder
        is constructed using ``cfg``, ``forward_spec``, and ``base_forward_store``.
    noise_model, reduce
        Noise model selector and reduction for the NLL.
    return_predict_fn
        If ``True``, also return a callable ``predict_fn(theta) -> image`` that
        uses the exact binder/model path underlying the loss. This is helpful
        for debugging stationary-point issues (e.g. verifying that
        ``pred(theta_true)`` matches the stored ``data`` when gradients are
        unexpectedly non-zero). Set to ``False`` for the standard
        loss-only tuple.
    """
    from ..systems import SheraBinder

    if binder is not None:
        mismatches = []
        if cfg is not None and getattr(binder, "cfg", cfg) is not cfg:
            mismatches.append("cfg")
        if forward_spec is not None and getattr(binder, "forward_spec", forward_spec) is not forward_spec:
            mismatches.append("forward_spec")
        if base_forward_store is not None and getattr(binder, "base_forward_store", base_forward_store) is not base_forward_store:
            mismatches.append("base_forward_store")
        if mismatches:
            raise ValueError(
                "binder provided alongside cfg/forward_spec/base_forward_store that do not "
                "match the binder. Use make_binder_nll_fn for binder-first construction."
            )

        return make_binder_nll_fn(
            binder=binder,
            infer_keys=infer_keys,
            data=data,
            var=var,
            noise_model=noise_model,
            reduce=reduce,
            theta0_store=base_forward_store,
            return_predict_fn=return_predict_fn,
        )

    if isinstance(cfg, Mapping):
        resolved_cfg = resolve_config(cfg)
        if "system" not in resolved_cfg:
            raise ValueError(
                "cfg must contain a top-level 'system' block when provided as a mapping."
            )
        cfg = {"system": resolved_cfg["system"]}

    if isinstance(cfg, Mapping):
        binder_obj = SheraBinder(
            cfg,
            forward_spec,
            base_forward_store,
        )
    elif isinstance(cfg, SheraThreePlaneConfig):
        binder_obj = SheraBinder(
            cfg,
            forward_spec,
            base_forward_store,
        )
    elif isinstance(cfg, SheraTwoPlaneConfig):
        binder_obj = SheraBinder(
            cfg,
            forward_spec,
            base_forward_store,
        )
    else:
        raise TypeError(
            "cfg must be either a resolved nested config mapping (system/experiment schema) "
            "or a SheraThreePlaneConfig/SheraTwoPlaneConfig dataclass. "
            "Legacy flat config schemas are not supported."
        )

    return make_binder_nll_fn(
        binder=binder_obj,
        infer_keys=infer_keys,
        data=data,
        var=var,
        noise_model=noise_model,
        reduce=reduce,
        theta0_store=base_forward_store,
        return_predict_fn=return_predict_fn,
    )


def loss_canonical(
    theta,
    cfg,
    forward_spec,
    infer_keys,
    base_forward_store,
    data,
    var,
    *,
    noise_model: str = "gaussian",
    reduce: str = "sum",
):
    """
    Canonical θ-space negative log-likelihood for Shera image inference.

    Parameters
    ----------
    theta : jax.Array
        Flat parameter vector in the ordering defined by ``infer_keys``.
    cfg :
        SheraThreePlaneConfig describing the structural optical configuration.
    forward_spec : ParamSpec
        Forward-model ParamSpec describing *all* parameters in the model, both
        inferred and fixed. This is what the SheraBinder validates
        against.
    infer_keys : tuple[str, ...]
        Keys of the parameters that live in θ-space (and their ordering).
    base_forward_store : ParameterStore
        Baseline ParameterStore containing fixed parameters and nominal values
        for the inferred ones.
    data : jax.Array
        Observed image data.
    var : jax.Array
        Per-pixel variance image (Gaussian case).

    Returns
    -------
    loss : jax.Array
        Scalar negative log-likelihood.
    """
    # Delegate to the existing Binder-based helper, which already handles:
    #   - Binder construction
    #   - pack/unpack between θ and ParameterStore
    #   - Gaussian / Poisson image NLL
    loss_fn, _theta0 = make_binder_image_nll_fn(
        cfg,
        forward_spec,   # FULL spec here
        base_forward_store,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        reduce=reduce,
    )

    return loss_fn(theta)


def _resolve_run_dir(
    run_dir: Optional[str | Path],
    runs_dir: Optional[str | Path],
    run_id: Optional[str],
) -> tuple[Path, str]:
    if run_dir is not None:
        resolved = Path(run_dir)
        resolved_run_id = run_id or resolved.name
        return resolved, resolved_run_id

    if runs_dir is None:
        raise ValueError("Either run_dir or runs_dir must be provided when enabling artifacts.")

    resolved_run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    resolved = Path(runs_dir) / resolved_run_id
    return resolved, resolved_run_id


def _gd_loop(
    loss_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    *,
    learning_rate: float = 1e-2,
    num_steps: int = 100,
    optimizer: Optional[optax.GradientTransformation] = None,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run a pure in-memory gradient-descent loop in θ-space.

    This is the lowest-level optimizer helper in the refactor-era stack. It
    operates only on a packed parameter vector ``theta`` and a scalar loss
    function, producing a minimal trace without touching disk. It is intended
    for unit tests, toy quadratics, and algorithmic experiments where artifact
    bookkeeping is not needed.

    The loop computes gradients with JAX autodiff and applies optax updates.
    It is agnostic to Shera-specific metadata, index maps, or artifact schemas.

    Parameters
    ----------
    loss_fn :
        Callable with signature ``loss_fn(theta) -> scalar``. The function must
        be JAX-differentiable with respect to ``theta``.
    theta0 :
        Initial parameter vector of shape ``(D,)`` (or any array-like that
        broadcasts to that shape).
    learning_rate :
        Step size used when ``optimizer`` is ``None``.
    num_steps :
        Number of gradient updates to perform.
    optimizer :
        Optional optax ``GradientTransformation``. If ``None``, a plain
        ``optax.sgd(learning_rate)`` update is used.
    show_progress :
        If ``True``, wraps the loop in a ``tqdm`` progress bar.

    Returns
    -------
    theta_final :
        Final parameter vector after ``num_steps`` updates, shape ``(D,)``.
    trace :
        Dictionary of minimal trace arrays:

        - ``"theta"``: stacked θ history, shape ``(num_steps + 1, D)`` (includes
          the initial ``theta0`` and the final iterate).
        - ``"loss"``: loss values at each recorded θ, shape
          ``(num_steps + 1,)``.
        - ``"grad_norm"``: gradient L2 norms per update, shape
          ``(num_steps,)`` (present when updates are computed).
        - ``"step_norm"``: update L2 norms per update, shape
          ``(num_steps,)`` (present when updates are computed).

    See Also
    --------
    run_simple_gd : Minimal wrapper around this loop with optional artifacts.
    run_gd_with_artifacts : Canonical artifact-producing wrapper.
    run_shera_gd : Shera-specific front end built on ``run_gd_with_artifacts``.
    """
    theta = np.asarray(theta0)

    if optimizer is None:
        optimizer = optax.sgd(learning_rate=learning_rate)
    opt_state = optimizer.init(theta)

    losses = []
    theta_history = [theta]
    grad_norms = []
    step_norms = []

    iterator = range(num_steps)
    if show_progress:
        iterator = tqdm(iterator)

    for _ in iterator:
        loss, g = jax.value_and_grad(loss_fn)(theta)
        losses.append(loss)

        updates, opt_state = optimizer.update(g, opt_state, params=theta)
        theta = optax.apply_updates(theta, updates)

        grad_norms.append(np.linalg.norm(g))
        step_norms.append(np.linalg.norm(updates))
        theta_history.append(theta)

    losses.append(loss_fn(theta))

    trace = {
        "loss": np.stack(losses),
        "theta": np.stack(theta_history),
    }
    if grad_norms:
        trace["grad_norm"] = np.stack(grad_norms)
    if step_norms:
        trace["step_norm"] = np.stack(step_norms)

    return theta, trace


def run_simple_gd(
    loss_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    *,
    learning_rate: float = 1e-2,
    num_steps: int = 100,
    optimizer: Optional[optax.GradientTransformation] = None,
    run_dir: Optional[str | Path] = None,
    runs_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    save_checkpoints: bool = False,
    artifact_meta: Optional[Mapping[str, Any]] = None,
    artifact_summary: Optional[Mapping[str, Any]] = None,
    artifact_theta_space: Optional[str] = None,
    artifact_metric: Optional[Mapping[str, np.ndarray]] = None,
    return_artifacts: bool = False,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]] | Tuple[np.ndarray, Dict[str, np.ndarray], Optional[dict]]:
    """
    Run a minimal θ-space gradient-descent loop with optional run artifacts.

    This is a lightweight front end that delegates the numerical optimization
    to :func:`_gd_loop` and optionally writes the standard trace/meta/summary
    artifacts described in ``docs/architecture/optimization_artifacts_and_plotting.md``.
    It is model-agnostic: the only inputs are a packed θ vector and a scalar
    loss function.

    Parameters
    ----------
    loss_fn :
        Callable taking a 1D JAX array ``theta`` and returning a scalar loss.
        For Shera use cases, this is typically the closure returned by
        :func:`make_image_nll_fn`.
    theta0 :
        Initial packed parameter vector of shape ``(D,)``.
    learning_rate :
        Step size for the default Adam optimizer (ignored if ``optimizer`` is
        provided).
    num_steps :
        Number of gradient steps to take.
    optimizer :
        Optional optax optimizer. If ``None``, ``optax.adam(learning_rate)``
        is used.
    run_dir / runs_dir / run_id :
        Optional run directory configuration. If provided, run artifacts
        (trace/meta/summary, and optional checkpoints) are saved via
        :func:`dluxshera.inference.run_artifacts.save_run`. Default is disabled.
    save_checkpoints :
        Whether to emit best/final checkpoints.
    artifact_meta / artifact_summary :
        Optional mappings merged into the auto-generated meta/summary payloads
        when artifacts are enabled.
    artifact_theta_space :
        Optional label for the θ space recorded in meta (e.g., "primitive" or
        "eigen"). Defaults to "primitive".
    artifact_metric :
        Optional mapping containing metric/preconditioning outputs (e.g.,
        ``{"theta_ref": ..., "metric_diag": ..., "lr_scale": ...}``) to be
        saved in ``metric.npz`` when artifacts are enabled.
    return_artifacts :
        When ``True``, return a third element containing an in-memory artifact
        payload (or ``None`` if artifacts are explicitly disabled). When
        ``run_dir``/``runs_dir`` are not provided, this payload is assembled
        without writing to disk.

    Returns
    -------
    theta_final :
        Final parameter vector after `num_steps` updates.
    history :
        Dict of simple diagnostics:

        - ``"loss"``: array of shape ``(num_steps,)`` with per-step losses.
        - ``"theta"``: array of shape ``(num_steps, D)`` with per-step θ values.
    artifacts :
        Only returned when ``return_artifacts`` is ``True``. Contains the same
        payload structure as :func:`run_gd_with_artifacts` (trace/meta/summary
        plus optional checkpoints/preconditioning data), or ``None`` when
        artifacts are disabled.

    See Also
    --------
    _gd_loop : Low-level, I/O-free gradient-descent loop.
    run_gd_with_artifacts : Canonical artifact-producing wrapper.
    run_shera_gd : Shera-specific front end with optional per-parameter LRs.
    dluxshera.inference.run_artifacts.save_run : Artifact writer.
    docs/architecture/optimization_artifacts_and_plotting.md : Artifact schema.
    """
    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    artifacts_requested = (
        run_dir is not None
        or runs_dir is not None
        or save_checkpoints
        or artifact_meta is not None
        or artifact_summary is not None
        or artifact_metric is not None
        or return_artifacts
    )

    if not artifacts_requested:
        theta, full_trace = _gd_loop(
            loss_fn,
            theta0,
            learning_rate=learning_rate,
            num_steps=num_steps,
            optimizer=optimizer,
        )

        history = {
            "loss": full_trace["loss"][:-1],
            "theta": full_trace["theta"][1:],
        }

        if return_artifacts:
            return theta, history, None

        return theta, history

    theta, history, artifact_payload = run_gd_with_artifacts(
        loss_fn=loss_fn,
        theta0=theta0,
        learning_rate=learning_rate,
        num_steps=num_steps,
        optimizer=optimizer,
        index_map=None,
        run_dir=run_dir,
        runs_dir=runs_dir,
        run_id=run_id,
        save_checkpoints=save_checkpoints,
        theta_space=artifact_theta_space or "primitive",
        metric=artifact_metric,
        extra_meta=artifact_meta,
        extra_summary=artifact_summary,
        return_artifacts=True,
        show_progress=True,
    )

    if return_artifacts:
        return theta, history, artifact_payload

    return theta, history


def run_gd_with_artifacts(
    loss_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    *,
    learning_rate: float = 1e-2,
    num_steps: int = 100,
    optimizer: Optional[optax.GradientTransformation] = None,
    index_map: Optional[Mapping[str, Any]] = None,
    run_dir: Optional[str | Path] = None,
    runs_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    save_checkpoints: bool = False,
    theta_space: str = "primitive",
    metric: Optional[Mapping[str, np.ndarray]] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
    extra_summary: Optional[Mapping[str, Any]] = None,
    return_artifacts: bool = True,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]] | Tuple[np.ndarray, Dict[str, np.ndarray], Optional[dict]]:
    """
    Run θ-space gradient descent and assemble canonical optimization artifacts.

    This is the canonical wrapper around :func:`_gd_loop`. It builds the
    standardized trace/meta/summary structures described in
    ``docs/architecture/optimization_artifacts_and_plotting.md`` and optionally
    persists them via :func:`dluxshera.inference.run_artifacts.save_run`. It
    remains model-agnostic: all metadata needed to interpret θ (such as an
    IndexMap) is passed explicitly.

    Parameters
    ----------
    loss_fn :
        Callable with signature ``loss_fn(theta) -> scalar``. Must be
        JAX-differentiable with respect to ``theta``.
    theta0 :
        Initial packed parameter vector of shape ``(D,)``.
    learning_rate :
        Base learning rate used when ``optimizer`` is ``None``.
    num_steps :
        Number of gradient updates to perform.
    optimizer :
        Optional optax optimizer. If ``None``, ``optax.sgd(learning_rate)`` is
        used.
    index_map :
        Optional IndexMap describing how θ indices map to parameter names and
        blocks. When provided it is stored under ``meta["theta"]["index_map"]``;
        see :func:`dluxshera.params.packing.build_index_map`.
    run_dir / runs_dir / run_id :
        Optional run directory configuration controlling how artifacts are
        written to disk.

        If ``run_dir`` is provided, it is treated as the exact directory for
        this run and all artifacts (trace, meta, summary, checkpoints, etc.)
        are written directly into that directory.

        If ``runs_dir`` is provided, it is treated as a parent "runs root".
        A per-run subdirectory is created under ``runs_dir``, using
        ``run_id`` if supplied or an auto-generated identifier otherwise,
        and artifacts are written into that subdirectory.

        If neither ``run_dir`` nor ``runs_dir`` is provided, no files are
        written to disk; artifacts are still assembled in memory and returned
        via the function’s return value when ``return_artifacts`` is ``True``.

        In typical usage, pass a concrete ``run_dir`` for a single ad-hoc run,
        or pass ``runs_dir`` (plus an optional ``run_id``) when running many
        experiments into a common root directory.
    save_checkpoints :
        Whether to save best/final checkpoints as ``checkpoint_*.npz``.
    theta_space :
        Label describing the θ parameterization (e.g., ``"primitive"``,
        ``"eigen"``) stored in metadata.
    metric :
        Optional mapping of metric/preconditioning outputs (e.g.,
        ``{"theta_ref": ..., "metric_diag": ..., "lr_scale": ...}``) saved as
        ``metric.npz``.
    extra_meta :
        Extra metadata to merge into ``meta.json``. Keys ``"theta"`` and
        ``"optimizer"`` are merged into their respective sub-dicts.
    extra_summary :
        Extra fields merged into ``summary.json``.
    return_artifacts :
        If ``True``, return a third element with the assembled artifact payload.
    show_progress :
        If ``True``, show a ``tqdm`` progress bar during optimization.

    Returns
    -------
    theta_final :
        Final parameter vector, shape ``(D,)``.
    history :
        Dict containing per-step ``"loss"`` and ``"theta"`` arrays with shapes
        ``(num_steps,)`` and ``(num_steps, D)`` respectively.
    artifacts :
        Only returned when ``return_artifacts`` is ``True``. The payload is a
        dictionary with keys ``run_dir``, ``run_id``, ``trace``, ``meta``,
        ``summary``, ``checkpoints``, and ``metric``. This mirrors the files
        written by the artifact system.

    See Also
    --------
    _gd_loop : Low-level, I/O-free gradient-descent loop.
    run_simple_gd : Minimal wrapper with optional artifacts.
    run_shera_gd : Shera-specific front end with per-parameter LR support.
    dluxshera.inference.run_artifacts.save_run : Artifact writer.
    docs/architecture/optimization_artifacts_and_plotting.md : Artifact schema.
    docs/architecture/inference_and_loss.md : Inference/loss pipeline context.
    """
    theta, full_trace = _gd_loop(
        loss_fn,
        theta0,
        learning_rate=learning_rate,
        num_steps=num_steps,
        optimizer=optimizer,
        show_progress=show_progress,
    )

    history = {
        "loss": full_trace["loss"][:-1],
        "theta": full_trace["theta"][1:],
    }

    trace: Dict[str, np.ndarray] = {
        "loss": history["loss"],
        "theta": history["theta"],
    }
    if "grad_norm" in full_trace:
        trace["grad_norm"] = full_trace["grad_norm"]
    if "step_norm" in full_trace:
        trace["step_norm"] = full_trace["step_norm"]
    trace["base_lr"] = np.full((history["loss"].shape[0],), learning_rate)

    artifacts_enabled = run_dir is not None or runs_dir is not None
    resolved_run_dir = None
    resolved_run_id = run_id or "in-memory"
    if artifacts_enabled:
        resolved_run_dir, resolved_run_id = _resolve_run_dir(run_dir, runs_dir, run_id)
    created_at = _now_iso_local_ms()

    base_meta: Dict[str, Any] = {
        "run_id": resolved_run_id,
        "created_at": created_at,
        "theta": {
            "dim": int(theta.size),
            "theta_space": theta_space,
        },
        "optimizer": {
            "kind": "optax",
            "name": "sgd" if optimizer is None else type(optimizer).__name__,
            "learning_rate": learning_rate,
            "num_steps": num_steps,
        },
    }
    if index_map is not None:
        base_meta["theta"]["index_map"] = index_map

    if extra_meta:
        for key, value in extra_meta.items():
            if key == "theta" and isinstance(value, Mapping):
                base_meta.setdefault("theta", {}).update(value)
            elif key == "optimizer" and isinstance(value, Mapping):
                base_meta.setdefault("optimizer", {}).update(value)
            else:
                base_meta[key] = value

    loss_array = onp.asarray(trace["loss"])
    loss_init = float(loss_array[0]) if loss_array.size else None
    loss_final = float(loss_array[-1]) if loss_array.size else None
    best_idx = int(onp.nanargmin(loss_array)) if loss_array.size else None
    loss_best = float(loss_array[best_idx]) if best_idx is not None else None

    base_summary: Dict[str, Any] = {
        "status": "ok",
        "run_id": resolved_run_id,
        "created_at": created_at,
        "num_steps_completed": int(loss_array.size),
        "loss_init": loss_init,
        "loss_final": loss_final,
    }

    if extra_summary:
        base_summary.update(extra_summary)

    checkpoints = None
    if save_checkpoints and best_idx is not None:
        checkpoints = {
            "best": {
                "theta_best": trace["theta"][best_idx],
                "best_step": best_idx,
                "best_loss": loss_best,
            },
            "final": {
                "theta_final": trace["theta"][-1],
                "final_step": int(loss_array.size - 1),
                "final_loss": loss_final,
            },
        }

    artifact_metric = None
    if metric is not None:
        artifact_metric = dict(metric)

    artifact_payload = None
    if return_artifacts or artifacts_enabled:
        artifact_payload = {
            "run_dir": resolved_run_dir,
            "run_id": resolved_run_id,
            "trace": trace,
            "meta": base_meta,
            "summary": base_summary,
            "checkpoints": checkpoints,
            "metric": artifact_metric,
        }

    if artifacts_enabled:
        save_run(
            resolved_run_dir,
            trace=trace,
            meta=base_meta,
            summary=base_summary,
            artifacts=_build_artifacts_mapping(
                checkpoints=checkpoints,
                metric=artifact_metric,
            ),
        )

    if return_artifacts:
        return theta, history, artifact_payload

    return theta, history


def run_shera_gd(
    *,
    loss_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    index_map: Optional[Mapping[str, Any]] = None,
    learning_rate: float = 0.5,
    lr_vec: Optional[np.ndarray] = None,
    num_steps: int = 100,
    optimizer_kind: str = "sgd",
    optimizer_kwargs: Optional[Mapping[str, Any]] = None,
    run_dir: Optional[str | Path] = None,
    runs_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    save_checkpoints: bool = False,
    theta_space: str = "primitive",
    metric: Optional[Mapping[str, np.ndarray]] = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
    extra_summary: Optional[Mapping[str, Any]] = None,
    return_artifacts: bool = True,
    show_progress: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]] | Tuple[np.ndarray, Dict[str, np.ndarray], Optional[dict]]:
    """
    Shera-specific front end for θ-space gradient descent.

    This helper wraps :func:`run_gd_with_artifacts` and adds Shera conventions
    such as optional per-parameter learning-rate vectors and index-map metadata.
    It is the preferred entry point for Shera θ-space optimization when
    artifact logging or Shera-specific metadata is desired.

    Parameters
    ----------
    loss_fn :
        Callable with signature ``loss_fn(theta) -> scalar``.
    theta0 :
        Initial packed parameter vector of shape ``(D,)``.
    index_map :
        Optional IndexMap describing θ layout, typically created by
        :func:`dluxshera.params.packing.build_index_map` from a Shera
        inference spec.
    learning_rate :
        Base learning rate used when ``lr_vec`` is not supplied. When
        ``lr_vec`` is provided, this scalar additionally scales the per-parameter
        vector before it is passed to the optimizer.
    lr_vec :
        Optional per-parameter learning-rate vector of shape ``(D,)``. When
        provided, an element-wise ``optax.sgd(learning_rate=lr_vec)`` optimizer
        is used, yielding updates ``theta_{t+1} = theta_t - (learning_rate *
        lr_vec) ⊙ grad``. The ``learning_rate`` scalar is still recorded in
        metadata, and artifacts should store the raw ``lr_vec`` while the base
        scalar captures the global scaling.
    num_steps :
        Number of gradient updates to perform.
    optimizer_kind :
        Name of optax optimizer to use ("sgd" or "adam"). Defaults to SGD.
    optimizer_kwargs :
        Optional mapping forwarded to the optax optimizer constructor (e.g.,
        {"b1": 0.9, "b2": 0.999, "eps": 1e-8} for Adam). No strict validation
        is performed.
    run_dir / runs_dir / run_id :
        Optional run directory configuration for artifact output.
    save_checkpoints :
        Whether to save best/final checkpoints.
    theta_space :
        Label describing the θ parameterization (e.g., ``"primitive"``,
        ``"eigen"``) stored in metadata.
    metric :
        Optional mapping of metric/preconditioning outputs saved in
        ``metric.npz``.
    extra_meta / extra_summary :
        Optional metadata merged into the saved ``meta.json`` / ``summary.json``
        (useful for Shera-specific run identifiers or config IDs).
    return_artifacts :
        If ``True``, return the assembled artifact payload in-memory.
    show_progress :
        If ``True``, show a ``tqdm`` progress bar during optimization.

    Returns
    -------
    theta_final :
        Final parameter vector, shape ``(D,)``.
    history :
        Per-step diagnostics containing ``"loss"`` and ``"theta"`` arrays.
    artifacts :
        Only returned when ``return_artifacts`` is ``True``. See
        :func:`run_gd_with_artifacts` for payload structure.

    See Also
    --------
    run_gd_with_artifacts : Canonical artifact-producing GD wrapper.
    run_simple_gd : Minimal wrapper without Shera-specific metadata.
    _gd_loop : Pure in-memory GD loop.
    docs/architecture/optimization_artifacts_and_plotting.md : Artifact schema.
    docs/architecture/inference_and_loss.md : Shera inference context.
    """
    def _scale_by_vector(vec: np.ndarray) -> optax.GradientTransformation:
        """Elementwise scaling of updates by vector ``vec``."""
        vec = np.asarray(vec)
        def init_fn(_):
            return None
        def update_fn(updates, state, params=None):
            return jax.tree_util.tree_map(lambda g: g * vec, updates), state
        return optax.GradientTransformation(init_fn, update_fn)

    opt_kwargs = dict(optimizer_kwargs or {})
    optimizer: optax.GradientTransformation | None = None

    if optimizer_kind == "sgd":
        if lr_vec is not None:
            scaled_lr_vec = learning_rate * np.asarray(lr_vec)
            optimizer = optax.sgd(learning_rate=scaled_lr_vec, **opt_kwargs)
        else:
            optimizer = optax.sgd(learning_rate=learning_rate, **opt_kwargs)
    elif optimizer_kind == "adam":
        txs = [optax.scale_by_adam(**opt_kwargs)]
        if lr_vec is not None:
            txs.append(_scale_by_vector(lr_vec))
        txs.append(optax.scale(-learning_rate))
        optimizer = optax.chain(*txs)
    else:
        raise ValueError(
            f"Unsupported optimizer_kind={optimizer_kind!r}; expected 'sgd' or 'adam'."
        )

    return run_gd_with_artifacts(
        loss_fn,
        theta0,
        learning_rate=learning_rate,
        num_steps=num_steps,
        optimizer=optimizer,
        index_map=index_map,
        run_dir=run_dir,
        runs_dir=runs_dir,
        run_id=run_id,
        save_checkpoints=save_checkpoints,
        theta_space=theta_space,
        metric=metric,
        extra_meta=extra_meta,
        extra_summary=extra_summary,
        return_artifacts=return_artifacts,
        show_progress=show_progress,
    )


def diagnose_first_step(
    *,
    loss_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    learning_rate: float = 1e-2,
    lr_vec: Optional[np.ndarray] = None,
    optimizer_kind: str = "sgd",
    index_map: Optional[Mapping[str, Any]] = None,
    verbose: bool = False,
    top_k: int = 10,
) -> dict[str, object]:
    """
    Minimal first-step diagnostic for θ-space optimizers.

    Evaluates loss/grad at ``theta0`` and simulates the first update using the
    same SGD/Adam wrappers as ``run_shera_gd``. Returns a payload with finiteness
    flags and the proposed ``theta1``. When ``verbose`` is ``True`` and any
    non-finite values are detected, offending indices (and IndexMap labels when
    provided) are printed to stderr/stdout.
    """
    theta0 = np.asarray(theta0)
    loss0 = np.asarray(loss_fn(theta0))
    grad_fn = jax.grad(lambda th: np.asarray(loss_fn(th)))
    grad0 = np.asarray(grad_fn(theta0))

    def _print_nonfinite(arr, label):
        mask = ~np.isfinite(arr)
        if not mask.any():
            return
        idx = np.where(mask)[0]
        print(f"[diagnose_first_step] non-finite {label} at indices {idx}")
        if index_map and isinstance(index_map, Mapping):
            entries = index_map.get("entries", [])
            for i in idx:
                for entry in entries:
                    if entry.get("start") <= i < entry.get("stop"):
                        name = entry.get("name", "unknown")
                        print(f"  index {i} -> {name}")
                        break

    opt_kwargs = {}
    optimizer: optax.GradientTransformation

    if optimizer_kind == "adam":
        def _scale_by_vector(vec: np.ndarray) -> optax.GradientTransformation:
            vec = np.asarray(vec)
            def init_fn(_):
                return None
            def update_fn(updates, state, params=None):
                return jax.tree_util.tree_map(lambda g: g * vec, updates), state
            return optax.GradientTransformation(init_fn, update_fn)

        txs = [optax.scale_by_adam(**opt_kwargs)]
        if lr_vec is not None:
            txs.append(_scale_by_vector(lr_vec))
        txs.append(optax.scale(-learning_rate))
        optimizer = optax.chain(*txs)
    else:
        if lr_vec is not None:
            optimizer = optax.sgd(learning_rate=learning_rate * np.asarray(lr_vec), **opt_kwargs)
        else:
            optimizer = optax.sgd(learning_rate=learning_rate, **opt_kwargs)

    opt_state = optimizer.init(theta0)
    updates, _ = optimizer.update(grad0, opt_state, params=theta0)
    theta1 = optax.apply_updates(theta0, updates)
    loss1 = np.asarray(loss_fn(theta1))
    delta = theta1 - theta0

    if verbose:
        if not np.isfinite(loss0):
            print("[diagnose_first_step] non-finite loss0")
        if not np.isfinite(loss1):
            print("[diagnose_first_step] non-finite loss1")
        _print_nonfinite(grad0, "grad0")
        _print_nonfinite(theta1, "theta1")

    # Top-k magnitude helpers
    def _top_entries(arr):
        flat = np.abs(arr).ravel()
        if flat.size == 0:
            return []
        k = min(top_k, flat.size)
        idx_sorted = np.argsort(flat)[::-1][:k]
        return [(int(i), float(arr.ravel()[i])) for i in idx_sorted]

    return {
        "loss0": float(loss0),
        "loss0_finite": bool(np.isfinite(loss0)),
        "grad0": grad0,
        "grad0_finite": bool(np.all(np.isfinite(grad0))),
        "theta0_min": float(np.min(theta0)),
        "theta0_max": float(np.max(theta0)),
        "theta1": theta1,
        "theta1_finite": bool(np.all(np.isfinite(theta1))),
        "loss1": float(loss1),
        "loss1_finite": bool(np.isfinite(loss1)),
        "grad0_min": float(np.min(grad0)),
        "grad0_max": float(np.max(grad0)),
        "delta_min": float(np.min(delta)),
        "delta_max": float(np.max(delta)),
        "lr_vec_min": float(np.min(lr_vec)) if lr_vec is not None else None,
        "lr_vec_max": float(np.max(lr_vec)) if lr_vec is not None else None,
        "top_grad": _top_entries(grad0),
        "top_delta": _top_entries(delta),
    }


def run_image_gd(
    cfg,
    forward_spec,
    store_init,
    infer_keys,
    data,
    var,
    *,
    noise_model: NoiseModel = "gaussian",
    learning_rate: float = 1e-2,
    lr_vec: Optional[np.ndarray] = None,
    num_steps: int = 50,
    run_dir: Optional[str | Path] = None,
    runs_dir: Optional[str | Path] = None,
    run_id: Optional[str] = None,
    save_checkpoints: bool = False,
    save_signals: bool = False,
    save_plots: bool = False,
    signals_truth: Optional[Mapping[str, Any]] = None,
    signals_decoder=None,
    enable_precond: bool = False,
    precond_cfg: Optional[Mapping[str, Any]] = None,
):
    """
    Run gradient descent in θ-space for image-based NLL using the Shera model.

    Now uses SheraBinder + `make_binder_image_nll_fn` under the hood.

    Parameters
    ----------
    save_signals / save_plots :
        When artifacts are enabled, compute intro Signals and optionally write
        ``signals.npz`` plus plots under ``<run_dir>/plots``.
    signals_truth :
        Optional mapping of truth values forwarded to :func:`build_signals` for
        residual computation.
    signals_decoder :
        Optional decoder override passed to :func:`build_signals`. Defaults to
        unpacking θ on top of ``store_init`` and refreshing derived parameters
        with the provided ``forward_spec``.
    enable_precond / precond_cfg :
        When enabled, compute a per-parameter learning-rate vector and related
        diagonal curvature summaries at the start of the run and save them to
        ``metric.npz`` via the artifact writer.
    lr_vec :
        Optional per-parameter learning-rate vector for θ-space updates. When
        provided, the optimizer uses element-wise SGD with this vector.
    """
    # Build canonical Binder-based loss and θ0
    loss_theta, theta0 = make_binder_image_nll_fn(
        cfg,
        forward_spec,
        store_init,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        reduce="sum",
    )

    artifacts_enabled = run_dir is not None or runs_dir is not None
    sub_spec = forward_spec.subset(infer_keys)
    artifact_meta = None
    precond_enabled = enable_precond or precond_cfg is not None

    if precond_enabled and not artifacts_enabled:
        raise ValueError("enable_precond=True requires run_dir or runs_dir for artifact saving.")

    index_map = None
    if artifacts_enabled:
        index_map = build_index_map(sub_spec, store_init, theta=theta0)
        artifact_meta = {
            "theta": {
                "theta_space": "primitive",
                "index_map": index_map,
            },
            "objective": {
                "name": "binder_image_nll",
                "noise_model": noise_model,
            },
            "spec": {
                "infer_keys": tuple(infer_keys),
            },
        }

    artifact_metric = None
    precond_meta = None
    if precond_enabled:
        cfg_dict = dict(precond_cfg) if precond_cfg is not None else {}
        cfg_dict.setdefault("base_lr", learning_rate)
        method_cfg = PreconditioningConfig(**cfg_dict)
        precond_outputs = compute_precond_vectors(
            loss_fn=loss_theta,
            theta0=theta0,
            method_cfg=method_cfg,
            index_map=index_map,
        )

        base_lr = method_cfg.base_lr if method_cfg.base_lr is not None else learning_rate
        lr_scale = (
            precond_outputs["lr_vec"] / base_lr
            if base_lr not in (None, 0.0)
            else precond_outputs["lr_vec"]
        )
        artifact_metric = {
            "theta_ref": np.asarray(theta0),
            "metric_diag": precond_outputs["curv_diag"],
            "lr_scale": lr_scale,
            "precond": precond_outputs["precond"],
        }
        precond_meta = precond_outputs["config"]

        if artifact_meta is None:
            artifact_meta = {}
        artifact_meta.setdefault("optimizer", {})
        artifact_meta["optimizer"]["preconditioning"] = precond_meta

    # Simple θ-space GD
    signals_enabled = artifacts_enabled and (save_signals or save_plots)
    needs_artifact_payload = signals_enabled or precond_enabled

    optimizer = None
    if lr_vec is not None:
        lr_vec = np.asarray(lr_vec)
        optimizer = optax.sgd(learning_rate=lr_vec)

    gd_result = run_simple_gd(
        loss_theta,
        theta0,
        learning_rate=learning_rate,
        optimizer=optimizer,
        num_steps=num_steps,
        run_dir=run_dir,
        runs_dir=runs_dir,
        run_id=run_id,
        save_checkpoints=save_checkpoints,
        artifact_meta=artifact_meta,
        artifact_theta_space="primitive",
        artifact_metric=artifact_metric,
        return_artifacts=needs_artifact_payload,
    )
    theta_final, history = gd_result[:2]
    artifact_payload = gd_result[2] if needs_artifact_payload else None

    if signals_enabled and artifact_payload is not None:
        from .plotting import plot_signals_panels
        from .signals import build_signals

        sub_spec = forward_spec.subset(infer_keys)

        if signals_decoder is not None:
            decoder_fn = signals_decoder
        else:
            def decoder_fn(theta_vec):
                store_theta = store_unpack_params(sub_spec, theta_vec, store_init)
                return store_theta.refresh_derived(forward_spec)

        signals = build_signals(
            artifact_payload["trace"],
            artifact_payload["meta"],
            decoder=decoder_fn,
            truth=signals_truth,
            signal_set="intro",
        )

        plots: list[Path] = []
        if save_plots:
            plots = plot_signals_panels(
                signals,
                artifact_payload["run_dir"],
                panel_set="intro",
                title_prefix=artifact_payload.get("run_id"),
            )

        summary = artifact_payload["summary"]
        if save_signals or save_plots:
            save_run(
                artifact_payload["run_dir"],
                trace=artifact_payload["trace"],
                meta=artifact_payload["meta"],
                summary=summary,
                artifacts=_build_artifacts_mapping(
                    checkpoints=artifact_payload["checkpoints"],
                    metric=artifact_payload.get("metric"),
                    signals=signals if save_signals else None,
                ),
            )

    # Map final θ back into a ParameterStore
    store_final = store_unpack_params(sub_spec, theta_final, store_init)

    return theta_final, store_final, history




# -------------------------------------------------------------------------
# θ-space Fisher Information utilities (Binder-based)
# -------------------------------------------------------------------------

def generate_fim_labels(
    infer_keys: Sequence[ParamKey],
    *,
    cfg: SheraThreePlaneConfig | SheraTwoPlaneConfig | Mapping[str, Any] | None,
    store: ParameterStore | None = None,
) -> list[str]:
    """
    Generate human-readable θ-labels for refactor-era FIM utilities.

    Parameters
    ----------
    infer_keys :
        Ordered parameter keys that define the packed θ vector.
    cfg :
        Shera config used to resolve Zernike Noll indices for
        ``optics.primary.zernike_coeffs_nm`` and ``optics.secondary.zernike_coeffs_nm``.
    store :
        Optional ParameterStore providing concrete values. When present, this
        is used to infer vector lengths. If absent (or missing a key), the
        helper attempts to fall back to any available ParamSpec shape derived
        from ``cfg``.

    Notes
    -----
    - Scalar parameters are labeled with their key as-is.
    - Vector parameters are expanded:
        * ``optics.primary.zernike_coeffs_nm`` → ``M1 Z{n}`` using
          ``cfg.primary_noll_indices``.
        * ``optics.secondary.zernike_coeffs_nm`` → ``M2 Z{n}`` using
          ``cfg.secondary_noll_indices``.
        * Other vectors use ``"{key}[{i}]"`` based on the inferred length.
    """
    spec = None
    translations = {
        "optics.plate_scale_as_per_pix": "Plate Scale",
        "source.contrast": "Contrast",
        "source.log_flux_total": "Log Flux",
        "source.x_position_as": "Binary X",
        "source.y_position_as": "Binary Y",
        "source.separation_as": "Binary Separation",
        "source.position_angle_deg": "Position Angle",
    }

    def _vector_length(key: ParamKey) -> int | None:
        if store is not None and key in store:
            value = store.get(key)
            if value is not None:
                arr = np.asarray(value)
                if arr.ndim > 0:
                    return int(arr.size)
        if cfg is None or isinstance(cfg, Mapping):
            return None
        nonlocal spec
        if spec is None:
            from ..systems.three_plane import build_forward_spec_from_config
            from ..systems.two_plane import build_forward_spec_from_config as build_twoplane_forward_spec_from_config

            if isinstance(cfg, SheraThreePlaneConfig):
                spec = build_forward_spec_from_config(cfg)
            elif isinstance(cfg, SheraTwoPlaneConfig):
                spec = build_twoplane_forward_spec_from_config(cfg)
        if spec is not None and key in spec:
            field = spec.get(key)
            if field.shape:
                size = 1
                for dim in field.shape:
                    size *= int(dim)
                return size
        return None

    labels: list[str] = []
    for key in infer_keys:
        length = _vector_length(key)
        if key in translations and (length is None or length == 1):
            labels.append(translations[key])
            continue
        if length is None:
            labels.append(key)
            continue

        if key == "optics.primary.zernike_coeffs_nm":
            if isinstance(cfg, Mapping):
                optics_block = cfg.get("optics", cfg)
                nolls = optics_block.get("primary_noll_indices", ())
            else:
                nolls = getattr(cfg, "primary_noll_indices", ()) if cfg is not None else ()
            if nolls:
                labels.extend([f"M1 Z{n}" for n in nolls])
                continue
        if key == "optics.secondary.zernike_coeffs_nm":
            if isinstance(cfg, Mapping):
                optics_block = cfg.get("optics", cfg)
                nolls = optics_block.get("secondary_noll_indices", ())
            else:
                nolls = getattr(cfg, "secondary_noll_indices", ()) if cfg is not None else ()
            if nolls:
                labels.extend([f"M2 Z{n}" for n in nolls])
                continue

        labels.extend([f"{key}[{i}]" for i in range(length)])

    return labels


def map_labels_to_keys(
    infer_keys: Sequence[ParamKey],
    flat_labels: Sequence[str],
    *,
    store: ParameterStore | None = None,
    index_map: Mapping[str, object] | None = None,
) -> dict[str, str | list[str]]:
    """
    Map flat labels back onto their parameter keys for printing utilities.

    Parameters
    ----------
    infer_keys :
        Ordered parameter keys that define the packed θ vector.
    flat_labels :
        Flat list of labels aligned with the packed θ vector.
    store :
        ParameterStore used to infer per-key sizes when ``index_map`` is not
        supplied or lacks a key.
    index_map :
        Optional index map with ``entries`` containing ``name``, ``start``, and
        ``stop`` values to slice ``flat_labels``.

    Returns
    -------
    labels_by_key :
        Mapping suitable for ``print_optimization_summary(..., labels=...)``.
    """
    if store is None and index_map is None:
        raise ValueError("map_labels_to_keys requires either store or index_map.")

    entry_lookup: dict[str, Mapping[str, object]] = {}
    if index_map is not None:
        entries = index_map.get("entries", [])
        if isinstance(entries, Sequence):
            entry_lookup = {
                entry.get("name"): entry
                for entry in entries
                if isinstance(entry, Mapping) and "name" in entry
            }

    labels_by_key: dict[str, str | list[str]] = {}
    label_index = 0

    for key in infer_keys:
        size: int
        key_labels: list[str]
        entry = entry_lookup.get(key) if entry_lookup else None

        if entry is not None and "start" in entry and "stop" in entry:
            start = int(entry["start"])
            stop = int(entry["stop"])
            size = max(stop - start, 0)
            key_labels = list(flat_labels[start:stop])
        else:
            if store is not None and key in store:
                value = store.get(key)
                if value is None:
                    size = 1
                else:
                    arr = onp.asarray(value)
                    size = 1 if arr.ndim == 0 or arr.size == 1 else int(arr.size)
            else:
                size = 1

            key_labels = (
                list(flat_labels[label_index:label_index + size])
                if label_index < len(flat_labels)
                else []
            )
            label_index += size

        if size == 1:
            labels_by_key[key] = key_labels[0] if key_labels else key
        else:
            labels_by_key[key] = key_labels

    return labels_by_key


def fim_theta(
    loss_fn: Callable[[np.ndarray], np.ndarray],
    theta_ref: np.ndarray,
) -> np.ndarray:
    """
    Compute a Fisher Information Matrix (FIM) in θ-space for a given
    scalar loss or NLL function.

    This is a thin wrapper around `jax.hessian(loss_fn)(theta_ref)`,
    but keeps the intent clear: the Hessian of the *negative*
    log-likelihood (or loss) at a reference point.

    Parameters
    ----------
    loss_fn :
        Callable taking a 1D θ vector and returning a scalar loss or
        negative log-likelihood. In Shera usage, this is usually the
        closure returned by `make_binder_image_nll_fn(...)`.
    theta_ref :
        1D JAX array representing the reference parameter vector at
        which to evaluate the FIM (e.g. truth or current MAP).

    Returns
    -------
    F : jnp.ndarray
        (N, N) Fisher matrix in θ coordinates, where N = theta_ref.size.
    """
    theta_ref = np.asarray(theta_ref)

    # We don't need any special tricks here; θ is already flat.
    return jax.hessian(loss_fn)(theta_ref)


def build_fim_diagonal_preconditioner(
    fim: np.ndarray,
    *,
    curvature_floor: float = 1e-8,
    eps: float = 1e-12,
    lr_clip: Optional[tuple[float, float]] = None,
) -> dict[str, onp.ndarray | dict[str, object]]:
    """Build the canonical primitive-theta diagonal FIM preconditioner.

    This implements the convention used by the canonical astrometry recipe:

    ``curvature_vec = max(diag(FIM), curvature_floor)``
    ``lr_vec = 1 / (curvature_vec + eps)``

    ``lr_vec`` is a scale vector only. ``run_shera_gd`` applies the global
    ``learning_rate`` separately, yielding SGD updates
    ``theta <- theta - learning_rate * lr_vec * grad``.
    """

    fim_arr = onp.asarray(fim, dtype=float)
    if fim_arr.ndim != 2 or fim_arr.shape[0] != fim_arr.shape[1]:
        raise ValueError("FIM must be a square matrix.")
    if not onp.all(onp.isfinite(fim_arr)):
        raise ValueError("FIM contains non-finite values.")

    curvature_floor = float(curvature_floor)
    eps = float(eps)
    if curvature_floor < 0.0:
        raise ValueError("curvature_floor must be non-negative.")
    if eps < 0.0:
        raise ValueError("eps must be non-negative.")

    fim_sym = 0.5 * (fim_arr + fim_arr.T)
    fim_diag = onp.diag(fim_sym)
    curvature_floored_count = int(onp.count_nonzero(fim_diag < curvature_floor))
    curvature_vec = onp.maximum(fim_diag, curvature_floor)
    lr_vec_unclipped = onp.reciprocal(curvature_vec + eps)
    lr_vec = onp.array(lr_vec_unclipped, copy=True)
    lr_clip_applied_count = 0
    if lr_clip is not None:
        lr_min, lr_max = lr_clip
        lr_min = float(lr_min)
        lr_max = float(lr_max)
        if lr_min <= 0.0:
            raise ValueError("lr_clip lower bound must be positive.")
        if lr_max < lr_min:
            raise ValueError("lr_clip upper bound must be >= lower bound.")
        lr_clip_applied_count = int(
            onp.count_nonzero((lr_vec < lr_min) | (lr_vec > lr_max))
        )
        lr_vec = onp.clip(lr_vec, lr_min, lr_max)

    for name, arr in {
        "fim_diag": fim_diag,
        "curvature_vec": curvature_vec,
        "lr_vec_unclipped": lr_vec_unclipped,
        "lr_vec": lr_vec,
    }.items():
        if not onp.all(onp.isfinite(arr)):
            raise ValueError(f"Non-finite values encountered in {name}.")
    if onp.any(lr_vec <= 0.0):
        raise ValueError("Preconditioning vector must be strictly positive.")

    return {
        "fim": fim_sym,
        "fim_diag": fim_diag,
        "curvature_vec": curvature_vec,
        "lr_vec_unclipped": lr_vec_unclipped,
        "lr_vec": lr_vec,
        "config": {
            "method": "fim_diag",
            "curvature_floor": curvature_floor,
            "curvature_floored_count": curvature_floored_count,
            "eps": eps,
            "lr_clip": None if lr_clip is None else [float(lr_clip[0]), float(lr_clip[1])],
            "lr_clip_applied_count": lr_clip_applied_count,
        },
    }


def fim_theta_shera(
    cfg,
    forward_spec: ParamSpec,
    base_forward_store: ParameterStore,
    infer_keys: Sequence[ParamKey],
    data: np.ndarray,
    var: np.ndarray,
    *,
    noise_model: NoiseModel = "gaussian",
    reduce: Literal["sum", "mean"] = "sum",
    return_labels: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convenience helper: build a Binder-based θ-space NLL for Shera and
    compute its Fisher Information Matrix at the corresponding θ₀.

    Parameters
    ----------
    cfg :
        SheraTwoPlaneConfig or SheraThreePlaneConfig (e.g. SHERA_TESTBED_CONFIG).
    forward_spec :
        Forward ParamSpec describing model keys.
    base_forward_store :
        ParameterStore providing the baseline parameter values (truth
        or current best-fit).
    infer_keys :
        Sequence of ParamKeys included in θ (order matters).
    data :
        Observed image.
    var :
        Per-pixel variance image (ignored for Poisson, but kept for API).
    noise_model :
        "gaussian" or "poisson".
    reduce :
        Reduction inside the NLL (passed through to the image kernels).

    return_labels :
        If True, also return a list of human-readable labels aligned with the
        packed θ vector.

    Returns
    -------
    F : jnp.ndarray
        (N, N) Fisher matrix in θ-space.
    theta0 : jnp.ndarray
        The θ vector at which the FIM was evaluated (the same θ₀
        returned by `make_binder_image_nll_fn`).
    labels : list[str]
        Optional labels aligned with θ (returned when ``return_labels=True``).
    """
    # Reuse the canonical Binder-based NLL closure
    loss_fn, theta0 = make_binder_image_nll_fn(
        cfg,
        forward_spec,
        base_forward_store,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        reduce=reduce,
    )

    F = fim_theta(loss_fn, theta0)
    if return_labels:
        labels = generate_fim_labels(
            infer_keys,
            cfg=cfg,
            store=base_forward_store,
        )
        return F, theta0, labels
    return F, theta0


@dataclass
class EigenThetaMap:
    """Linear reparameterisation between canonical θ-coordinates and an eigen basis.

    This class is intentionally *pure θ-space* and JAX-friendly. It only stores
    the linear algebra needed to move between coordinates and is agnostic to
    Binder/ParameterStore details. Shapes follow the convention that
    eigenvectors are **columns** of ``eigvecs``.

    Conventions
    ----------
    Let ``F`` be a symmetric positive semi-definite curvature matrix (e.g.
    Fisher / Hessian) at ``theta_ref`` with eigendecomposition ``F = V Λ Vᵀ``,
    eigenvalues ``λ_j`` and eigenvectors ``v_j`` (columns of ``V``). For a
    perturbation ``δθ = θ - theta_ref``:

    - If ``whiten=False`` (plain eigen coordinates)::

          z_j = v_jᵀ δθ
          θ   = theta_ref + Σ_j z_j v_j

    - If ``whiten=True`` (scaled so quadratic loss ≈ ½‖z‖²)::

          z_j = sqrt(λ_j) * v_jᵀ δθ
          θ   = theta_ref + Σ_j (z_j / sqrt(λ_j)) v_j

    Optional truncation keeps only the leading ``k`` eigenmodes. For whitened
    coordinates this makes the local quadratic form close to identity in the
    retained subspace. Methods are light-weight and safe to call from inside
    jitted regions.
    """

    theta_ref: np.ndarray          # shape (N,)
    eigvecs:   np.ndarray          # shape (N, k), columns = basis vectors
    eigvals:   Optional[np.ndarray] = None  # shape (k,), optional metadata
    whiten:    bool = False

    # -----------------------------
    # Constructors
    # -----------------------------
    @classmethod
    def from_fim(
        cls,
        F: np.ndarray,
        theta_ref: np.ndarray,
        *,
        truncate: Optional[int] = None,
        whiten: bool = False,
    ) -> "EigenThetaMap":
        """
        Build an EigenThetaMap from a (symmetric, PSD) curvature matrix ``F``.

        Parameters
        ----------
        F :
            (N, N) Fisher / Hessian / curvature matrix at theta_ref.
        theta_ref :
            Reference θ vector (N,).
        truncate :
            If not None, keep only the top-k eigenmodes (k <= N), ordered by
            descending eigenvalue.
        whiten :
            If True, coordinates z are scaled by sqrt(λ_j) so that the
            Hessian in z-space is approximately the identity in the retained
            subspace.

        Returns
        -------
        EigenThetaMap
        """
        F_np = np.asarray(F)
        # eigh → eigenvalues ascending; reorder to descending magnitude for
        # convenience when truncating to the most informative modes.
        evals, evecs = np.linalg.eigh(F_np)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        if truncate is not None:
            k = int(truncate)
            evals = evals[:k]
            evecs = evecs[:, :k]

        return cls(
            theta_ref=np.asarray(theta_ref),
            eigvecs=np.asarray(evecs),
            eigvals=np.asarray(evals),
            whiten=whiten,
        )

    # -----------------------------
    # Maps
    # -----------------------------
    def z_from_theta(self, theta: np.ndarray) -> np.ndarray:
        """Map θ → z in eigen coordinates (optionally whitened)."""
        theta = np.asarray(theta)
        delta = theta - self.theta_ref  # (N,)

        # Project onto eigenvectors
        coords = self.eigvecs.T @ delta  # (k,)

        if self.whiten:
            if self.eigvals is None:
                raise ValueError("whiten=True but eigvals is None.")
            scales = np.sqrt(self.eigvals + 1e-12)  # (k,)
            coords = coords * scales

        return coords

    def theta_from_z(self, z: np.ndarray) -> np.ndarray:
        """Map eigen coordinates z → θ (undoing whitening if requested)."""
        z = np.asarray(z)

        if self.whiten:
            if self.eigvals is None:
                raise ValueError("whiten=True but eigvals is None.")
            scales = np.sqrt(self.eigvals + 1e-12)
            z = z / scales

        delta = self.eigvecs @ z  # (N,)

        return self.theta_ref + delta

    # Aliases for backward compatibility / readability
    def to_eigen(self, theta: np.ndarray) -> np.ndarray:
        return self.z_from_theta(theta)

    def from_eigen(self, z: np.ndarray) -> np.ndarray:
        return self.theta_from_z(z)

    def to_theta(self, z: np.ndarray) -> np.ndarray:
        return self.theta_from_z(z)

    # Convenience properties
    @property
    def dim_theta(self) -> int:
        return int(self.theta_ref.size)

    @property
    def dim_eigen(self) -> int:
        return int(self.eigvecs.shape[1])


############################
# Loss and Update Functions
############################

def loglikelihood(model, data, var):
    """Normal log-likelihood."""
    return jax.scipy.stats.norm.logpdf(model.model(), loc=data, scale=np.sqrt(var))

def loss_fn(model, data, var):
    """Negative log-likelihood (loss function)."""
    return -np.nansum(loglikelihood(model, data, var))
