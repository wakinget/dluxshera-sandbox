from __future__ import annotations

from typing import Literal, Optional

import jax
import jax.numpy as np
from jax import grad, jit, lax, linearize
import numpyro.distributions as dist
import optax
import zodiax as zdx

from .params import ModelParams

__all__ = [
    "hessian",
    "FIM",
    "_perturb",
    "scheduler",
    "sgd",
    "get_optimiser",
    "get_lr_model",
    "assign_lr_vector",
    "get_lr_from_curvature",
    "generate_fim_labels_legacy",
    "build_basis",
    "construct_priors_from_dict",
]


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
    label_tree = jax.tree_util.tree_map(lambda _: None, model_params)  # None for every leaf (incl. p_ref, B, …)
    label_tree = label_tree.set("params", {p: p for p in parameters})  # strings for the optimised leaves

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
        lr_model[param] = lr_vec[idx: idx + size].reshape(shape)
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
        path_map = target.get_param_path_map()  # internal -> external
        ext2int = {v: k for k, v in path_map.items()}  # external -> internal

    keys_ext = order if order is not None else list(target.params.keys())

    sizes, shapes = [], []
    for k_ext in keys_ext:
        # Prefer the stored external leaf (avoids computed getters)
        if k_ext in getattr(target, "params", {}):
            leaf = np.asarray(target.params[k_ext])
        else:
            k_int = ext2int.get(k_ext, k_ext)  # fallback to internal
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
        out[k_ext] = lr_vec[i:i + n].reshape(shape)
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


def generate_fim_labels_legacy(params, model_params):
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


############################
# Reparameterization Utilities
############################

def build_basis(eigvecs, eigvals, truncate: Optional[int] = None, whiten: bool = False):
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


############################
# Priors
############################

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
