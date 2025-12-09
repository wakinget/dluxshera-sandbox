# src/dluxshera/inference/optimization.py
from __future__ import annotations

import jax
import jax.numpy as np
import jax.scipy.stats as jstats
import numpy as onp
from jax import config, grad, linearize, jit, lax
import dLux as dl
import dLux.layers as dll
import dLux.utils as dlu
import optax
import equinox as eqx
import zodiax as zdx
import numpyro.distributions as dist
import math
import json
from typing import Optional, Literal, Callable, Sequence, Tuple, Dict, Any
from dataclasses import dataclass

from ..optics.config import SheraThreePlaneConfig
from ..params.spec import ParamSpec, ParamKey
from ..params.store import ParameterStore
from ..params.packing import (
    pack_params as store_pack_params,
    unpack_params as store_unpack_params,
)

############################
# Exports
############################

__all__ = [
    # fisher / curvature
    "hessian", "FIM", "_perturb",

    # optax helpers
    "scheduler", "sgd", "get_optimiser", "get_lr_model",
    "assign_lr_vector", "get_lr_from_curvature",

    # containers
    "BaseModeller", "ModelParams",
    "SheraTwoPlaneParams", "SheraThreePlaneParams", "EigenParams",

    # new likelihood / loss utilities
    "gaussian_loglikelihood_image",
    "poisson_loglikelihood_image",
    "gaussian_loss",
    "poisson_loss",
    "make_loss_fn",
    "make_image_nll_fn",

    # simple θ–space optimizer
    "run_simple_gd",
    "run_image_gd",

    # θ-space eigen reparameterisation
    "EigenThetaMap",

    # legacy likelihood / step fns (model-params space)
    "loglikelihood", "loss_fn", "step_fn",

    # reparameterisation utils
    "generate_fim_labels", "pack_params", "unpack_params", "build_basis",

    # priors
    "construct_priors_from_dict",
]




# NOTE:
# - Legacy ModelParams/EigenParams + FIM/step_fn live here for now.
# - New Binder + θ-space loss/FIM/optim also live here.
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
    loglike = gaussian_loglikelihood_image(model_image, data, var, reduce=reduce)
    return -loglike


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
    inference_spec: ParamSpec,
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
    inference_spec
        ParamSpec describing the inference-space keys
        (typically `build_inference_spec_basic()`).
    base_store
        ParameterStore providing a baseline set of parameter values.
        This is the state we *overlay* when unpacking theta.
    infer_keys
        Sequence of ParamKeys to include in theta, e.g.
        ["binary.separation_as", "binary.x_position", ...].
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
        Callable (cfg, inference_spec, store) -> model. If None, a
        default Shera three-plane builder is imported lazily.

    Returns
    -------
    loss_fn
        Callable taking a flat theta vector and returning scalar NLL.
        Signature: loss_fn(theta) -> scalar.
    theta0
        Initial packed parameter vector constructed from
        (inference_spec, base_store, infer_keys).
    """
    # Subset spec to just the keys we want to infer
    sub_spec = inference_spec.subset(infer_keys)

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
        from ..core.builder import build_shera_threeplane_model
        build_model_fn = build_shera_threeplane_model

    data = np.asarray(data)
    var = np.asarray(var)

    def loss_fn(theta: np.ndarray) -> np.ndarray:
        # Unpack theta back into a ParameterStore, overlaying on base_store
        # NOTE: unpack_params(spec_subset, theta, base_store)
        store_theta = store_unpack_params(sub_spec, theta, base_store)

        # Build model from cfg + (inference_spec, store_theta)
        model = build_model_fn(cfg, inference_spec, store_theta)

        # Evaluate loss
        return _model_loss(model, data, var)

    return loss_fn, theta0


def make_binder_image_nll_fn(
    cfg: SheraThreePlaneConfig,
    inference_spec: ParamSpec,
    base_store: ParameterStore,
    infer_keys: Sequence[ParamKey],
    data: np.ndarray,
    var: np.ndarray,
    *,
    noise_model: NoiseModel = "gaussian",
    reduce: Literal["sum", "mean"] = "sum",
    use_system_graph: bool = False,
) -> Tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """
    Canonical θ-space image NLL using SheraThreePlaneBinder.

    This is the Binder-based sibling of `make_image_nll_fn`, but instead of
    rebuilding SheraThreePlane_Model on each call, it:

      - builds static optics + detector once via SheraThreePlaneBinder
      - only rebuilds the source from a ParameterStore inside the loss
      - uses image-level Gaussian/Poisson NLL kernels directly

    The `use_system_graph` flag optionally routes forward execution through a
    minimal SystemGraph wrapper instead of directly calling the Binder. This
    is a no-op for outputs but exercises the new graph layer.
    """
    from ..core.binder import SheraThreePlaneBinder
    from ..graph.system_graph import build_threeplane_system_graph

    # 1) Build a static Binder and/or SystemGraph
    binder = SheraThreePlaneBinder(cfg, inference_spec, base_store)
    graph = None
    if use_system_graph:
        graph = build_threeplane_system_graph(cfg, inference_spec, base_store)

    # 2) Subset spec + build initial θ from base_store
    sub_spec = inference_spec.subset(infer_keys)
    theta0 = store_pack_params(sub_spec, base_store, dtype=data.dtype)

    # 3) Choose an image-level NLL kernel
    data = np.asarray(data)
    var = np.asarray(var)

    if noise_model == "gaussian":
        def image_nll(model_image, data_image, var_image):
            return -gaussian_loglikelihood_image(
                model_image, data_image, var_image, reduce=reduce
            )
    elif noise_model == "poisson":
        def image_nll(model_image, data_image, var_image):
            # var_image unused, kept for symmetry
            return -poisson_loglikelihood_image(
                model_image, data_image, reduce=reduce
            )
    else:
        raise ValueError(
            f"Unknown noise_model {noise_model!r} "
            f"(expected 'gaussian' or 'poisson')."
        )

    # 4) Define θ → loss(θ) using binder.forward(...)
    def loss_fn(theta: np.ndarray) -> np.ndarray:
        # θ → ParameterStore overlayed on base_store
        store_theta = store_unpack_params(sub_spec, theta, base_store)
        # Binder merges store_theta into its base_store internally; optionally
        # run through the SystemGraph to exercise the new execution path.
        if graph is not None:
            model_image = graph.forward(store_theta)
        else:
            model_image = binder.forward(store_theta)
        return image_nll(model_image, data, var)

    return loss_fn, theta0


def loss_canonical(
    theta,
    cfg,
    inference_spec,
    infer_keys,
    base_store,
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
    inference_spec : ParamSpec
        Full ParamSpec describing *all* parameters in the model, both inferred
        and fixed. This is what the SheraThreePlaneBinder validates against.
    infer_keys : tuple[str, ...]
        Keys of the parameters that live in θ-space (and their ordering).
    base_store : ParameterStore
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
        inference_spec,   # FULL spec here
        base_store,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        reduce=reduce,
    )

    return loss_fn(theta)


def run_simple_gd(
    loss_fn: Callable[[np.ndarray], np.ndarray],
    theta0: np.ndarray,
    *,
    learning_rate: float = 1e-2,
    num_steps: int = 100,
    optimizer: Optional[optax.GradientTransformation] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Minimal gradient-descent loop over a packed parameter vector θ.

    Parameters
    ----------
    loss_fn :
        Callable taking a 1D JAX array `theta` and returning a scalar loss.
        In our Shera use-case this will typically be the closure produced by
        `make_image_nll_fn(...)`.
    theta0 :
        Initial 1D parameter vector.
    learning_rate :
        Step size for the default Adam optimizer (ignored if `optimizer`
        is provided).
    num_steps :
        Number of gradient steps to take.
    optimizer :
        Optional optax optimizer. If None, `optax.adam(learning_rate)` is used.

    Returns
    -------
    theta_final :
        Final parameter vector after `num_steps` updates.
    history :
        Dict of simple diagnostics:
          - "loss": 1D array of loss values at each step.
    """
    theta = np.asarray(theta0)

    if optimizer is None:
        optimizer = optax.adam(learning_rate)

    opt_state = optimizer.init(theta)

    @jax.jit
    def _step(theta, opt_state):
        loss, g = jax.value_and_grad(loss_fn)(theta)
        updates, opt_state = optimizer.update(g, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    losses = []

    for _ in range(num_steps):
        theta, opt_state, loss = _step(theta, opt_state)
        losses.append(loss)

    history = {"loss": np.stack(losses)}
    return theta, history


def run_image_gd(
    cfg,
    inference_spec,
    store_init,
    infer_keys,
    data,
    var,
    *,
    noise_model: NoiseModel = "gaussian",
    learning_rate: float = 1e-2,
    num_steps: int = 50,
):
    """
    Run gradient descent in θ-space for image-based NLL using the Shera model.

    Now uses SheraThreePlaneBinder + `make_binder_image_nll_fn` under the hood.
    """
    # Build canonical Binder-based loss and θ0
    loss_theta, theta0 = make_binder_image_nll_fn(
        cfg,
        inference_spec,
        store_init,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        reduce="sum",
    )

    # Simple θ-space GD
    theta_final, history = run_simple_gd(
        loss_theta,
        theta0,
        learning_rate=learning_rate,
        num_steps=num_steps,
    )

    # Map final θ back into a ParameterStore
    sub_spec = inference_spec.subset(infer_keys)
    store_final = store_unpack_params(sub_spec, theta_final, store_init)

    return theta_final, store_final, history




# -------------------------------------------------------------------------
# θ-space Fisher Information utilities (Binder-based)
# -------------------------------------------------------------------------

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


def fim_theta_shera(
    cfg: SheraThreePlaneConfig,
    inference_spec: ParamSpec,
    base_store: ParameterStore,
    infer_keys: Sequence[ParamKey],
    data: np.ndarray,
    var: np.ndarray,
    *,
    noise_model: NoiseModel = "gaussian",
    reduce: Literal["sum", "mean"] = "sum",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience helper: build a Binder-based θ-space NLL for Shera and
    compute its Fisher Information Matrix at the corresponding θ₀.

    Parameters
    ----------
    cfg :
        SheraThreePlaneConfig (e.g. SHERA_TESTBED_CONFIG).
    inference_spec :
        ParamSpec describing inference-level keys.
    base_store :
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

    Returns
    -------
    F : jnp.ndarray
        (N, N) Fisher matrix in θ-space.
    theta0 : jnp.ndarray
        The θ vector at which the FIM was evaluated (the same θ₀
        returned by `make_binder_image_nll_fn`).
    """
    # Reuse the canonical Binder-based NLL closure
    loss_fn, theta0 = make_binder_image_nll_fn(
        cfg,
        inference_spec,
        base_store,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        reduce=reduce,
    )

    F = fim_theta(loss_fn, theta0)
    return F, theta0


@dataclass
class EigenThetaMap:
    """
    Linear reparameterisation between canonical θ-coordinates and an
    eigenbasis of a local curvature matrix (e.g. Fisher / Hessian).

    This is intentionally *pure θ-space*: it doesn't know anything about
    ParamSpec, ParameterStore, or ModelParams. It just stores:

      - theta_ref : reference point in θ-space (typically the fiducial / MAP)
      - eigvecs   : columns are eigenvectors in θ-space (shape (N, k))
      - eigvals   : corresponding eigenvalues (shape (k,))
      - whiten    : whether coordinates are whitened by sqrt(λ_j)

    Conventions
    ----------
    Let F ≈ Hessian of the loss at theta_ref and F = V Λ Vᵀ with
    eigenvalues λ_j and eigenvectors v_j (columns of V). For a perturbation
    δθ = θ - theta_ref:

    If `whiten=False` (unwhitened coordinates):

        z_j = v_jᵀ δθ
        θ   = theta_ref + Σ_j z_j v_j

    If `whiten=True` (locally “σ-units” along each mode):

        z_j = sqrt(λ_j) * v_jᵀ δθ
        θ   = theta_ref + Σ_j (z_j / sqrt(λ_j)) v_j

    In the whitened case, a local quadratic loss ½ δθᵀ F δθ becomes
    ½ ||z||² in the retained subspace (up to truncation tolerances).
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
        Build an EigenThetaMap from a (symmetric, PSD) curvature matrix F.

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
        F_np = onp.asarray(F)
        # eigh → eigenvalues ascending; we want descending by magnitude
        evals, evecs = onp.linalg.eigh(F_np)
        idx = evals.argsort()[::-1]
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
    def to_eigen(self, theta: np.ndarray) -> np.ndarray:
        """
        Map θ → z in eigen coordinates.

        If whiten=False:
            z_j = v_j^T (θ - θ_ref)

        If whiten=True:
            z_j = sqrt(λ_j) * v_j^T (θ - θ_ref)

        Parameters
        ----------
        theta :
            Canonical parameter vector (N,).

        Returns
        -------
        z :
            Eigen coordinates (k,).
        """
        theta = np.asarray(theta)
        delta = theta - self.theta_ref           # (N,)

        # Project onto eigenvectors
        coords = self.eigvecs.T @ delta  # (k,)

        if self.whiten:
            if self.eigvals is None:
                raise ValueError("whiten=True but eigvals is None.")
            scales = np.sqrt(self.eigvals + 1e-12)  # (k,)
            coords = coords * scales

        return coords   # (k, N) @ (N,) -> (k,)

    def from_eigen(self, z: np.ndarray) -> np.ndarray:
        """
        Map eigen coordinates z → θ.

        If whiten=False:
            θ = θ_ref + sum_j z_j v_j

        If whiten=True:
            θ = θ_ref + sum_j (z_j / sqrt(λ_j)) v_j

        Parameters
        ----------
        z :
            Eigen coordinates (k,).

        Returns
        -------
        theta :
            Canonical θ vector (N,).
        """
        z = np.asarray(z)

        if self.whiten:
            if self.eigvals is None:
                raise ValueError("whiten=True but eigvals is None.")
            scales = np.sqrt(self.eigvals + 1e-12)
            z = z / scales

        delta = self.eigvecs @ z  # (N,)

        return self.theta_ref + delta  # (N, k) @ (k,) -> (N,)

    # Convenience properties
    @property
    def dim_theta(self) -> int:
        return int(self.theta_ref.size)

    @property
    def dim_eigen(self) -> int:
        return int(self.eigvecs.shape[1])


############################
# Legacy Fisher Matrix Utilities
############################

def hessian(f, x):
    """Compute the Hessian using JAX linearization and HVP trick."""
    _, hvp = linearize(grad(f), x)
    hvp = jit(hvp)
    basis = np.eye(x.size).reshape(-1, *x.shape)
    return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)

def FIM(pytree, parameters, loglike_fn, *loglike_args, **loglike_kwargs):
    """Compute the Fisher Information Matrix (FIM) over the given parameters."""
    pytree = zdx.tree.set_array(pytree, parameters)

    if len(parameters) == 1:
        parameters = [parameters]

    leaves = [pytree.get(p) for p in parameters]
    shapes = [leaf.shape for leaf in leaves]
    lengths = [leaf.size for leaf in leaves]
    N = np.array(lengths).sum()
    X = np.zeros(N)

    def loglike_fn_vec(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)

    return jax.hessian(loglike_fn_vec)(X)

def _perturb(X, pytree, parameters, shapes, lengths):
    n, xs = 0, []
    if isinstance(parameters, str):
        parameters = [parameters]
    indexes = range(len(parameters))

    for i, param, shape, length in zip(indexes, parameters, shapes, lengths):
        if length == 1:
            xs.append(X[i + n])
        else:
            xs.append(lax.dynamic_slice(X, (i + n,), (length,)).reshape(shape))
            n += length - 1

    return pytree.add(parameters, xs)


############################
# Optimizer Scheduling
############################

def scheduler(lr, start, *args):
    """Piecewise constant learning rate schedule."""
    shed_dict = {start: 1e100}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e100, shed_dict)

base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)
sgd = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))



def get_optimiser(pytree, optimisers, parameters=None):
    """
    Build an optimizer and return (model_params, optim, state).

    Parameters
    ----------
    pytree : ModelParams or EigenParams (or subclass like SheraThreePlaneParams)
        The parameter container to optimise.
    optimisers : dict
        Mapping from external parameter names (e.g. 'm1_aperture.coefficients')
        to optax.GradientTransformation.
    parameters : list[str], optional
        Subset of parameter names (external) to include. If None, uses all keys.

    Returns
    -------
    model_params : ModelParams/EigenParams
        Same type as input `pytree`, restricted to selected parameters.
    optim : optax.GradientTransformation
        Optax multi_transform optimizer.
    state : optax.OptState
        Initial optimizer state.
    """
    if parameters is not None:
        optimisers = {p: optimisers[p] for p in parameters}
    else:
        parameters = list(optimisers.keys())

    # external->internal mapping
    ext2int = {}
    if hasattr(pytree, "get_param_path_map"):
        path_map = pytree.get_param_path_map()  # internal -> external
        ext2int = {v: k for k, v in path_map.items()}

    # Build filtered param dict using internal keys for lookup
    filtered = {}
    for p_ext in parameters:
        p_int = ext2int.get(p_ext, p_ext)
        filtered[p_ext] = pytree.get(p_int)

    # Replace params in same type as input
    model_params = pytree.set("params", filtered)

    # Multi-transform expects labels keyed by external names
    # --- Build a label tree that has ONLY strings/None ---
    label_tree = jax.tree_util.tree_map(lambda _: None, model_params)          # None for every leaf (incl. p_ref, B, …)
    label_tree = label_tree.set("params", {p: p for p in parameters}) # strings for the optimised leaves

    optim = optax.multi_transform(optimisers, label_tree)
    state = optim.init(model_params)
    return model_params, optim, state



def get_lr_model(pytree, parameters, loglike_fn, *loglike_args, **loglike_kwargs):
    """Returns model-specific learning rates estimated from Fisher Information Matrix."""
    fmat = FIM(pytree, parameters, loglike_fn, *loglike_args, **loglike_kwargs)
    lr_vec = 1 / np.diag(fmat)

    idx = 0
    lr_model = {}
    for param in parameters:
        leaf = np.array(pytree.get(param))
        size, shape = leaf.size, leaf.shape
        lr_model[param] = lr_vec[idx : idx + size].reshape(shape)
        idx += size

    return ModelParams(lr_model)

def assign_lr_vector(lr_vec, target, order=None):
    """
    Pack a flat learning-rate vector into a ModelParams container.

    Parameters
    ----------
    lr_vec : array-like
        Flat vector of learning rates. Its length must equal the total number
        of scalar elements across all selected leaves.
    target : ModelParams or EigenParams (or subclass, e.g. SheraThreePlaneParams)
        The optimization container whose .params defines the leaf keys/shapes.
    order : list[str], optional
        External parameter keys to consume from lr_vec. If None, uses
        list(target.params.keys()).

    Returns
    -------
    lr_model : ModelParams
        A ModelParams with external keys and shapes matching the target.
    """
    lr_vec = np.asarray(lr_vec)

    # internal -> external, then invert to external -> internal
    ext2int = {}
    if hasattr(target, "get_param_path_map"):
        path_map = target.get_param_path_map()      # internal -> external
        ext2int = {v: k for k, v in path_map.items()}  # external -> internal

    keys_ext = order if order is not None else list(target.params.keys())

    sizes, shapes = [], []
    for k_ext in keys_ext:
        # Prefer the stored external leaf (avoids computed getters)
        if k_ext in getattr(target, "params", {}):
            leaf = np.asarray(target.params[k_ext])
        else:
            k_int = ext2int.get(k_ext, k_ext)       # fallback to internal
            leaf = np.asarray(target.get(k_int))
        sizes.append(int(leaf.size))
        shapes.append(leaf.shape)

    total = sum(sizes)
    if lr_vec.size != total:
        raise ValueError(
            f"assign_lr_vector: size mismatch. lr_vec has {lr_vec.size} elems "
            f"but selected leaves sum to {total}."
        )

    out, i = {}, 0
    for k_ext, n, shape in zip(keys_ext, sizes, shapes):
        out[k_ext] = lr_vec[i:i+n].reshape(shape)
        i += n

    # Keep the same node type as `target` (needed for tree alignment in step_fn)
    return target.set("params", out)


def get_lr_from_curvature(curv_vec, target, order=None, eps=1e-12):
    """
    Convenience wrapper: compute lr_vec = 1/(curvature+eps) then assign.

    Parameters
    ----------
    curv_vec : array-like
        Curvature per degree of freedom (e.g., diag(FIM), or eigenvalues if
        `target` is EigenParams with a single 'eigen_coefficients' leaf).
    target : ModelParams or EigenParams
        The optimization container whose structure we want to mirror.
    order : list[str], optional
        Deterministic order of keys to consume from lr_vec (see assign_lr_vector).
    eps : float
        Regularizer to avoid division by zero.

    Returns
    -------
    lr_model : ModelParams
        Structured learning-rate carrier aligned with `target`.
    """
    lr_vec = 1.0 / (np.asarray(curv_vec) + eps)
    return assign_lr_vector(lr_vec, target, order=order)



#############################
# Parameter Container Classes
#############################

class BaseModeller(zdx.Base):  # zdx.Base inherits eqx.Module
    """Base class to manage model parameters stored in a dictionary."""
    params: dict = eqx.field()  # differentiable leaves live here

    # ---------- NEW: normalization helpers ----------
    @staticmethod
    def _promote_float_like(x):
        """Promote Python/NumPy float-like values to jax arrays.
        Leave ints/bools/None unchanged. Preserve existing jax/np arrays."""
        if x is None:
            return None
        # Keep ints and bools structural/static (not differentiable)
        if isinstance(x, (bool, onp.bool_)):
            return x
        if isinstance(x, (int, onp.integer)):
            return x

        # jax/numPy arrays: preserve but ensure float arrays are jax floats
        dtype = np.float64 if config.x64_enabled else np.float32
        if isinstance(x, (np.ndarray, jax.Array)):
            # If it's already jax, keep as-is
            return x
        if isinstance(x, onp.ndarray):
            # Convert float arrays to jax float; others (e.g. int arrays) to jax arrays too if you prefer.
            if onp.issubdtype(x.dtype, onp.floating):
                return np.asarray(x, dtype=dtype)
            else:
                return np.asarray(x)

        # Python floats (or numpy floating scalars): promote to jax float arrays
        if isinstance(x, (float, onp.floating)):
            return np.array(x, dtype=dtype)

        # Lists/tuples: try to convert to a float jax array if all numeric floats
        if isinstance(x, (list, tuple)):
            try:
                arr = onp.asarray(x)
                if onp.issubdtype(arr.dtype, onp.floating):
                    return np.asarray(arr, dtype=dtype)
                # If list is ints or mixed: leave as-is (they’re usually structural, not differentiable)
                return x
            except Exception:
                return x

        # Anything else (dicts, objects, callables): leave alone
        return x

    @classmethod
    def _normalize_params_dict(cls, d: dict) -> dict:
        # Shallow normalization is enough (your params are flat)
        return {k: cls._promote_float_like(v) for k, v in d.items()}
    # ---------- /NEW ----------

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(f"Attribute {key} not found in {self.__class__.__name__}")

    def __getitem__(self, key):
        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]
        return values

    def get(self, key):
        if key in self.params:
            return self.params[key]
        raise ValueError(f"key: {key} not found in object: {type(self).__name__}")

class ModelParams(BaseModeller):
    """Encapsulates a subset of model parameters with math operations for optimization."""

    # # ---------- NEW: frozen Module-safe __init__ ----------
    # def __init__(self, params: dict):
    #     normalized = self._normalize_params_dict(params)
    #     # eqx.Modules are frozen; set via object.__setattr__
    #     object.__setattr__(self, "params", normalized)
    # # ---------- /NEW ----------

    @property
    def keys(self):
        return list(self.params.keys())

    @property
    def values(self):
        return list(self.params.values())

    def replace(self, values):
        """
        Replace all parameters with the provided values.

        Parameters
        ----------
        values : dict
            A dictionary containing updated parameter values. Any parameters
            not included in this dictionary will be replaced with None.

        Returns
        ----------
        ModelParams
            A new ModelParams object with the updated parameter values.

        Notes
        ----------
        - This method expects a fully defined parameter dictionary.
        - Missing keys will be replaced with None, potentially leading
          to unexpected behavior.
        """
        normalized = self._normalize_params_dict({param: values.get(param) for param in self.keys})
        return self.set("params", normalized)

    def from_model(self, values):
        normalized = self._normalize_params_dict({param: values.get(param) for param in self.keys})
        return self.set("params", normalized)

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def __mul__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def inject(self, other):
        normalized = [self._promote_float_like(v) for v in self.values]
        return other.set(self.keys, normalized)

    def to_json(self, filepath: str):
        serializable = {
            k: (np.array(v).tolist() if v is not None else None)
            for k, v in self.params.items()
        }
        with open(filepath, "w") as f:
            json.dump(serializable, f)

    @classmethod
    def from_json(cls, filepath: str) -> "ModelParams":
        with open(filepath, "r") as f:
            raw = json.load(f)

        # Don’t collapse to Python scalars; normalize to jax where needed.
        params = {}
        for k, v in raw.items():
            params[k] = v  # leave raw shape; normalize below
        return cls(cls._normalize_params_dict(params))


class SheraThreePlaneParams(ModelParams):
    """Parameter container for the Shera Three-Plane Optical System."""

    def __init__(self, params=None, point_design=None):
        """
        Initialize the parameter set for the Shera Three-Plane Optical System.

        Parameters
        ----------
        params : dict, optional
            A dictionary of parameter overrides. These values will replace the
            default parameters, including those set by the point design.

        point_design : str, optional
            Specifies which telescope point design to use. Valid options are:
                - "shera_testbed" (default)
                - "shera_flight"

            If not specified, the default "shera_testbed" parameters will be used.
        """
        # Define the two point designs
        point_designs = {
            "shera_testbed": {
                "p1_diameter": 0.09,
                "p2_diameter": 0.025,
                "m1_focal_length": 0.35796,
                "m2_focal_length": -0.041935,
                "plane_separation": 0.320,
                "pixel_size": 6.5e-6,
                "bandwidth": 110.,  # nm
                "log_flux": np.asarray(6.78),
            },
            "shera_flight": {
                "p1_diameter": 0.22,
                "p2_diameter": 0.025,
                "m1_focal_length": 0.604353,
                "m2_focal_length": -0.0545,
                "plane_separation": 0.55413,
                "pixel_size": 4.6e-6,
                "bandwidth": 41.,  # nm
                "log_flux": np.asarray(7.13),
            }
        }


        if point_design is None:
            point_design = "shera_testbed" # Default point design
        defaults = dict(point_designs[point_design])  # copy

        # Add other default parameters
        defaults.update({
            # Sampling and resolution settings
            "rng_seed": 0,
            "pupil_npix": 256,
            "psf_npix": 256,

            # Source parameters
            "x_position": np.asarray(0.),
            "y_position": np.asarray(0.),
            "separation": np.asarray(10.),
            "position_angle": np.asarray(90.),
            "contrast": np.asarray(0.3),
            "wavelength": 550.,  # nm
            "n_wavelengths": 3,

            # M1 Aberrations
            "m1_zernike_noll": np.arange(4, 12),
            "m1_zernike_amp": np.zeros(8),
            "m1_calibrated_power_law": 2.5,
            "m1_calibrated_amplitude": 0,
            "m1_uncalibrated_power_law": 2.5,
            "m1_uncalibrated_amplitude": 0,

            # M2 Aberrations
            "m2_zernike_noll": np.arange(4, 12),
            "m2_zernike_amp": np.zeros(8),
            "m2_calibrated_power_law": 2.5,
            "m2_calibrated_amplitude": 0,
            "m2_uncalibrated_power_law": 2.5,
            "m2_uncalibrated_amplitude": 0
        })

        # Update user-provided values
        if params is not None:
            defaults = {**defaults, **params}

        # Normalize everything to JAX arrays
        defaults = self._normalize_params_dict(defaults)

        # Initialize ModelParams with defaults
        super().__init__(defaults)

    def validate(self):
        """Validate the internal consistency of the parameter sets."""
        # Check that Zernike indexes and amplitudes match in length
        for prefix in ["m1_", "m2_"]:
            noll = self.params[f"{prefix}zernike_noll"]
            amp = self.params[f"{prefix}zernike_amp"]
            if len(noll) != len(amp):
                raise ValueError(
                    f"{prefix}zernike_noll and {prefix}zernike_amp must have the same length."
                )
        print("Validation successful.")

    def to_dict(self):
        """Flatten the parameter hierarchy for easy export."""
        return self.params

    def replace(self, values):
        """
        Replace parameters with the provided values.

        Parameters
        ----------
        values : dict
            A dictionary containing updated parameter values. Only the
            specified keys are updated, all other parameters are preserved.

        Returns
        -------
        SheraThreePlaneParams
            A new SheraThreePlaneParams object with the updated parameter values.

        Notes
        -----
        - This method preserves existing parameters that are not explicitly updated.
        - Does not support nested dictionary updates.
        """
        return self.set("params", {**self.params, **values})

    @staticmethod
    def get_param_path_map():
        '''Returns the parameter path that maps params from this class to the parameters of the model'''
        return {
            "m1_zernike_amp": "m1_aperture.coefficients",
            "m2_zernike_amp": "m2_aperture.coefficients"
        }

    def get(self, key):
        # This custom get method allows us to calculate what
        # psf_pixel_scale would be given the other parameters.
        # Falls back to the default get() behavior for other parameters
        if key == "psf_pixel_scale":
            if "psf_pixel_scale" in self.params:
                return self.params["psf_pixel_scale"] # Return a stored value if present
            return self.compute_psf_pixel_scale() # Otherwise compute
        return super().get(key)

    def update_from_model(self, model: "SheraThreePlane_Model") -> "SheraThreePlaneParams":
        """
        Return a new SheraThreePlaneParams object with updated values pulled from a model.

        For each parameter in this object, we attempt to update it using the model:
        - If a model-facing path is defined (via get_param_path_map), we use that.
        - Otherwise, we try to extract the parameter directly using its own name.

        Parameters
        ----------
        model : SheraThreePlane_Model
            The model object containing the most recent optimized parameter values.

        Returns
        -------
        SheraThreePlaneParams
            A new parameter object with updated values from the model.
        """
        updated = {}
        param_map = self.get_param_path_map()  # Maps self keys -> model keys

        for key in self.params:
            model_key = param_map.get(key, key)
            try:
                updated_value = model.get(model_key)
            except Exception as e:
                raise ValueError(f"Could not retrieve '{model_key}' from model: {e}")
            updated[key] = updated_value

        return self.replace(updated)

    def compute_EFL(self):
        """
        Return Effective Focal Length (EFL) in m, derived from the model's optics.
        """
        # Get focal lengths and pixel size
        f1 = self.get("m1_focal_length")
        f2 = self.get("m2_focal_length")
        sep = self.get("plane_separation")

        # Effective focal length for the two-mirror relay:
        # EFL = (1/f1 + 1/f2 - sep/(f1*f2))^-1
        EFL = 1.0 / (1.0 / f1 + 1.0 / f2 - sep / (f1 * f2)) # meters
        return EFL

    def compute_psf_pixel_scale(self):
        """
        Computes the PSF pixel scale in arcseconds/pixel based on mirror geometry and pixel size.

        Returns
        -------
        float
            The computed psf_pixel_scale in arcseconds/pixel.
        """
        # Get the EFL + Pixel size
        EFL = self.compute_EFL()
        pixel_size = self.get("pixel_size")
        return np.asarray(dlu.rad2arcsec(pixel_size / EFL)) # as/pixel

class SheraTwoPlaneParams(ModelParams):
    """Parameter container for the Shera Two-Plane Optical System."""

    def __init__(self, params=None, point_design=None):
        """
        Initialize the parameter set for the Shera Two-Plane Optical System.

        Parameters
        ----------
        params : dict, optional
            A dictionary of parameter overrides. These values will replace the
            default parameters, including those set by the point design.

        point_design : str, optional
            Specifies which telescope point design to use. Valid options are:
                - "shera_testbed" (default)
                - "shera_flight"

            If not specified, the default "shera_testbed" parameters will be used.
        """
        # Define the two point designs
        point_designs = {
            "shera_testbed": {
                "p1_diameter": 0.09,
                "p2_diameter": 0.025,
                "psf_pixel_scale": 0.355,
                "bandwidth": 110.,  # nm
                "log_flux": 6.78,
            },
            "shera_flight": {
                "p1_diameter": 0.22,
                "p2_diameter": 0.025,
                "psf_pixel_scale": 0.123,
                "bandwidth": 41.,  # nm
                "log_flux": 7.13,
            }
        }


        if point_design is None:
            point_design = "shera_testbed" # Default point design
        defaults = point_designs.get(point_design)

        # Add other default parameters
        defaults.update({
            # Sampling and resolution settings
            "rng_seed": 0,
            "pupil_npix": 256,
            "psf_npix": 256,

            # Source parameters
            "x_position": 0.,
            "y_position": 0.,
            "separation": 10.,
            "position_angle": 90.,
            "contrast": 0.3,
            "wavelength": 550.,  # nm
            "n_wavelengths": 3,

            # Aberrations
            "zernike_noll": np.arange(4, 11),
            "zernike_amp": np.zeros(7),
            "calibrated_power_law": 2.5,
            "calibrated_amplitude": 0,
            "uncalibrated_power_law": 2.5,
            "uncalibrated_amplitude": 0,
        })

        # Update user-provided values
        if params is not None:
            defaults = {**defaults, **params}

        # Initialize ModelParams with defaults
        super().__init__(defaults)

    def validate(self):
        """Validate the internal consistency of the parameter sets."""
        # Check that Zernike indexes and amplitudes match in length
        for prefix in ["m1_", "m2_"]:
            noll = self.params[f"{prefix}zernike_noll"]
            amp = self.params[f"{prefix}zernike_amp"]
            if len(noll) != len(amp):
                raise ValueError(
                    f"{prefix}zernike_noll and {prefix}zernike_amp must have the same length."
                )
        print("Validation successful.")

    def to_dict(self):
        """Flatten the parameter hierarchy for easy export."""
        return self.params

    def replace(self, values):
        """
        Replace parameters with the provided values.

        Parameters
        ----------
        values : dict
            A dictionary containing updated parameter values. Only the
            specified keys are updated, all other parameters are preserved.

        Returns
        -------
        SheraThreePlaneParams
            A new SheraThreePlaneParams object with the updated parameter values.

        Notes
        -----
        - This method preserves existing parameters that are not explicitly updated.
        - Does not support nested dictionary updates.
        """
        return self.set("params", {**self.params, **values})

    @staticmethod
    def get_param_path_map():
        '''Returns the parameter path that maps params from this class to the parameters of the model'''
        return {
            "zernike_amp": "coefficients",
        }

class EigenParams(ModelParams):
    """
    Wrapper around eigenmode coefficients, with same structure as ModelParams:
    - All differentiable leaves live in `params`, here {"eigen_coefficients": jax.Array}.
    - Metadata is static (p_ref, B, pure_keys, template).
    """
    # Differentiable leaves:
    # params: dict = eqx.field()  # inherited from ModelParams

    # <-- make arrays NON-static
    p_ref: onp.ndarray = eqx.field(static=False)
    B:     onp.ndarray = eqx.field(static=False)
    # <-- keep small metadata static (hashable/comparable)
    pure_keys: list = eqx.field(static=True)       # list[str]
    shape_map: dict = eqx.field(static=True)       # {name: tuple(shape)}

    def to_pure(self):
        """Project eigen coefficients back into pure parameter vector."""
        c = self.get("eigen_coefficients")  # a jax.Array
        # p_ref, B are NumPy; that's OK because they are static. Convert as needed:
        return np.asarray(self.p_ref) + np.asarray(self.B) @ c

    def inject(self, model):
        """Project to pure parameters and inject into the model."""
        p = self.to_pure()
        d, i = {}, 0
        for name in self.pure_keys:
            shape = self.shape_map[name]
            size = 1 if not shape else int(math.prod(shape))
            d[name] = p[i:i+size].reshape(shape)
            i += size
        return ModelParams(d).inject(model)




def construct_priors_from_dict(param_info):
    """
    Constructs NumPyro-compatible priors from a simplified parameter info dictionary.

    Parameters
    ----------
    param_info : dict
        Dictionary of parameter metadata in the form:
        {
            "param_name": {
                "mean": float or array,
                "sigma": float,
                "dist": "Normal" | "Uniform" | "LogNormal"
            },
            ...
        }

    Returns
    -------
    dict
        Dictionary of {param: numpyro distribution}.
    """
    param_priors = {}

    for param, info in param_info.items():
        mu = info["mean"]
        sigma = info["sigma"]
        dist_type = info["dist"]

        if dist_type == "Normal":
            param_priors[param] = dist.Normal(loc=mu, scale=sigma)
        elif dist_type == "Uniform":
            param_priors[param] = dist.Uniform(low=mu - sigma, high=mu + sigma)
        elif dist_type == "LogNormal":
            param_priors[param] = dist.LogNormal(loc=np.log(mu), scale=sigma)
        else:
            raise ValueError(f"Unsupported distribution type '{dist_type}' for parameter '{param}'")

    return param_priors




############################
# Loss and Update Functions
############################

def loglikelihood(model, data, var):
    """Normal log-likelihood."""
    return jax.scipy.stats.norm.logpdf(model.model(), loc=data, scale=np.sqrt(var))

def loss_fn(model, data, var):
    """Negative log-likelihood (loss function)."""
    return -np.nansum(loglikelihood(model, data, var))

# --- shared helper ---
def _loss_with_params(params, model, data, var, loss_fn):
    return loss_fn(params.inject(model), data, var)

def loss_with_injected(model_params, model, data, var, loss_fn):
    return loss_fn(model_params.inject(model), data, var)

# ============ GENERAL - Should work for Pure Modes + Eigenmodes ============
@eqx.filter_jit
def step_fn_general(model_params, data, var, model, lr_model, optim, state, loss_fn):
    # grads w.r.t. the leaves listed by model_params.keys
    loss, raw_grads = zdx.filter_value_and_grad(model_params.keys)(
        lambda p, m, d, v: loss_with_injected(p, m, d, v, loss_fn)
    )(model_params, model, data, var)

    # elementwise LR scaling (same tree structure as model_params)
    scaled_grads = jax.tree_util.tree_map(lambda g, s: g * s, raw_grads, lr_model)

    # optax update on exactly those leaves
    updates, state = optim.update(scaled_grads, state, model_params)

    # apply updates to the params container
    model_params = zdx.apply_updates(model_params, updates)

    # build next-step model by injecting the (updated) params
    model = model_params.inject(model)

    # return enough stuff for logging
    return loss, raw_grads, scaled_grads, updates, model, model_params, state



# ============ PURE SPACE ============
@eqx.filter_jit
def step_fn(model_params, data, var, model, lr_model, optim, state, loss_fn):
    # grads wrt external keys in ModelParams
    loss, raw_grads = zdx.filter_value_and_grad(model_params.keys)(
        lambda p, m, d, v: _loss_with_params(p, m, d, v, loss_fn)
    )(model_params, model, data, var)

    # scale grads elementwise by lr_model (same ModelParams structure)
    scaled_grads = jax.tree_util.tree_map(lambda g, s: g * s, raw_grads, lr_model)

    updates, state = optim.update(scaled_grads, state, model_params)
    model_params = zdx.apply_updates(model_params, updates)
    model = model_params.inject(model)
    return loss, raw_grads, scaled_grads, updates, model, model_params, state


# ============ EIGEN SPACE ============
@eqx.filter_jit
def step_fn_eigen(eparams, data, var, model, lr_model, optim, state, loss_fn):
    # Pull current coefficients (1D array)
    c = eparams.params["eigen_coefficients"]

    # Define loss as a function of *only* the coefficients
    def loss_from_c(c_flat, model, data, var):
        e_tmp = eqx.tree_at(lambda t: t.params["eigen_coefficients"], eparams, c_flat)
        return loss_fn(e_tmp.inject(model), data, var)

    loss, g_c = jax.value_and_grad(loss_from_c)(c, model, data, var)
    g_c = g_c * lr_model.params["eigen_coefficients"]  # elementwise scaling
    # g_c = g_c * 0  # zero out gradients

    # Update c only (so optax state is tiny)
    updates, state = optim.update(g_c, state, c)
    c_new = optax.apply_updates(c, updates)
    eparams = eqx.tree_at(lambda t: t.params["eigen_coefficients"], eparams, c_new)

    # Build next-step model
    model = eparams.inject(model)
    return loss, g_c, model, eparams, state



def sweep_param(model, param, sweep_info, loss_fn, *loss_args, **loss_kwargs):
    """
    Perform a 1D parameter sweep for any scalar or vector parameter in the model.

    Parameters
    ----------
    model : object
        The optical model that supports `.get(param)` and `.set(param, value)` methods.
    param : str
        Name of the parameter to sweep.
    sweep_info : dict
        Dictionary of the form {param: (span, steps)} specifying the sweep range and resolution.
    loss_fn : callable
        Function to evaluate model loss, must accept (model, *args, **kwargs).
    *loss_args : tuple
        Positional arguments passed to the loss function.
    **loss_kwargs : dict
        Keyword arguments passed to the loss function.

    Returns
    -------
    results : list of dict
        Each result entry contains:
        - 'parameter' : str, name of the parameter
        - 'index' : int or None, for vector parameters
        - 'value' : float, value of the parameter at that sweep point
        - 'loss' : float, scalar loss value
    """
    from ..utils.utils import get_sweep_values

    results = []
    span, steps = sweep_info[param]
    value = model.get(param)

    # Check if vector-valued parameter (ndim > 0)
    if np.ndim(value) > 0:
        for i in range(len(value)):
            center = float(value[i])
            sweep_values = get_sweep_values(center, span, steps)

            for val in sweep_values:
                new_value = value.at[i].set(val)
                model_ = model.set(param, new_value)
                loss = float(loss_fn(model_, *loss_args, **loss_kwargs))

                results.append({
                    "parameter": param,
                    "index": i,
                    "value": float(val),
                    "loss": loss,
                })

    # Scalar parameter
    else:
        center = float(value)
        sweep_values = get_sweep_values(center, span, steps)

        for val in sweep_values:
            model_ = model.set(param, val)
            loss = float(loss_fn(model_, *loss_args, **loss_kwargs))

            results.append({
                "parameter": param,
                "index": None,
                "value": float(val),
                "loss": loss,
            })

    return results


############################
# Reparameterization Utilities
############################

def generate_fim_labels(params, model_params):
    """
    Generate human-readable labels for FIM plotting.

    Parameters
    ----------
    params : list[str]
        Parameter keys used in the optimizer/FIM.
    model_params : ModelParams
        Parameter container holding the actual arrays (e.g. zernike_amp).

    Returns
    -------
    labels : list[str]
        One label per flattened parameter entry.
    """
    labels = []
    for param in params:
        if param == "m1_aperture.coefficients":
            labels.extend([f"M1 Z{n}" for n in model_params.m1_zernike_noll])
        elif param == "m2_aperture.coefficients":
            labels.extend([f"M2 Z{n}" for n in model_params.m2_zernike_noll])
        else:
            labels.append(param)
    return labels



def pack_params(values_pytree, params, model_template, from_model=False):
    """
    Flatten values into a vector + labels.
    Can handle either ModelParams/SheraThreePlaneParams or a SheraThreePlane_Model.
    """
    from src.dluxshera.core.modeling import SheraThreePlane_Model # Importing locally avoids a circular import error
    labels = []
    flat_values = []

    path_map = model_template.get_param_path_map()
    inv_path_map = {v: k for k, v in path_map.items()}

    for param in params:
        if isinstance(values_pytree, SheraThreePlane_Model):
            # External path lookup directly from model
            value = values_pytree.get(param)
            actual_key = inv_path_map.get(param, param)
        else:
            # Internal storage key lookup from Params container
            actual_key = inv_path_map.get(param, param)
            value = values_pytree.get(actual_key)

        if np.ndim(value) == 0:
            flat_values.append(value)
            labels.append(param)
        else:
            if actual_key == "m1_zernike_amp":
                nolls = model_template.m1_zernike_noll
                labels.extend([f"M1 Z{n}" for n in nolls])
            elif actual_key == "m2_zernike_amp":
                nolls = model_template.m2_zernike_noll
                labels.extend([f"M2 Z{n}" for n in nolls])
            else:
                labels.extend([f"{param}[{i}]" for i in range(value.size)])
            flat_values.extend(np.ravel(value))

    return np.array(flat_values), labels




def unpack_params(flat_values, params, model_template, pytree_cls=ModelParams):
    """
    Reconstruct ModelParams from a flat vector, keeping external param names.

    Parameters
    ----------
    flat_values : array-like
        Flattened parameter values (same order as pack_params).
    params : list[str]
        External optimizer parameter keys (e.g. 'm1_aperture.coefficients').
    model_template : ModelParams
        Template with shapes (e.g. SheraThreePlaneParams).
    pytree_cls : class
        Class to use for constructing the output (default=ModelParams).

    Returns
    -------
    model_params : ModelParams
        Structured parameters with external names as keys.
    """
    path_map = model_template.get_param_path_map()   # internal → external
    # Invert for convenience
    inv_path_map = {v: k for k, v in path_map.items()}

    idx = 0
    param_dict = {}

    for param in params:
        # Map external → internal only for shape lookup
        actual_key = inv_path_map.get(param, param)
        leaf = np.array(model_template.get(actual_key))
        size, shape = leaf.size, leaf.shape

        slice_vals = np.array(flat_values[idx: idx + size]).reshape(shape)
        idx += size

        # Store back under external name so it aligns with history
        param_dict[param] = slice_vals

    return pytree_cls(param_dict)



def build_basis(eigvecs, eigvals, truncate=None, whiten=False):
    """
    Construct a basis matrix B from eigenvectors and eigenvalues.

    Parameters
    ----------
    eigvecs : (N, N) array
        Eigenvectors from FIM decomposition (columns).
    eigvals : (N,) array
        Eigenvalues from FIM decomposition, sorted descending.
    truncate : int or None
        If provided, number of top eigenmodes to keep (k <= N).
    whiten : bool
        If True, scale each eigenvector by 1/sqrt(lambda).

    Returns
    -------
    B : (N, k) array
        Basis matrix mapping eigen coefficients to parameter space.
    """
    N = eigvecs.shape[0]
    k = truncate if truncate is not None else N
    V = eigvecs[:, :k]
    if whiten:
        scales = 1.0 / np.sqrt(eigvals[:k] + 1e-12)
        V = V @ np.diag(scales)
    return V

