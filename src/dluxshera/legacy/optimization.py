from __future__ import annotations

from typing import Literal, Optional

import jax
import jax.numpy as np
from jax import grad, jit, lax, linearize
import numpyro.distributions as dist
import optax
import equinox as eqx
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
    "loss_with_injected",
    "step_fn",
    "step_fn_eigen",
    "sweep_param",
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


############################
# Legacy Loss and Update Functions
############################

def _loss_with_params(params, model, data, var, loss_fn):
    return loss_fn(params.inject(model), data, var)


def loss_with_injected(model_params, model, data, var, loss_fn):
    return loss_fn(model_params.inject(model), data, var)


@eqx.filter_jit
def step_fn(model_params, data, var, model, lr_model, optim, state, loss_fn):
    # NOTE: model_params.params may contain keys with dots (e.g. "m1_aperture.coefficients").
    # zodiax treats dots as path navigation, so we take grads w.r.t. the params dict directly.
    def _loss_from_params(params_dict, m, d, v):
        mp = model_params.set("params", params_dict)
        return loss_with_injected(mp, m, d, v, loss_fn)

    loss, raw_grads_dict = jax.value_and_grad(_loss_from_params)(
        model_params.params, model, data, var
    )

    # Lift dict grads back into a pytree matching model_params (non-param leaves -> None)
    none_tree = jax.tree_util.tree_map(lambda _: None, model_params)
    raw_grads = none_tree.set("params", raw_grads_dict)

    # Elementwise LR scaling on the params dict only
    scaled_grads_dict = jax.tree_util.tree_map(
        lambda g, s: g * s, raw_grads_dict, lr_model.params
    )
    scaled_grads = none_tree.set("params", scaled_grads_dict)

    # optax update on exactly those leaves
    updates, state = optim.update(scaled_grads, state, model_params)

    # apply updates to the params container
    model_params = zdx.apply_updates(model_params, updates)

    # build next-step model by injecting the (updated) params
    model = model_params.inject(model)

    # return enough stuff for logging
    return loss, raw_grads, scaled_grads, updates, model, model_params, state


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
