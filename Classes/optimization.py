# Classes/optimization.py

import jax
import jax.numpy as np
import numpy as onp
from jax import grad, linearize, jit, lax, tree, config as jax_config
import dLux.utils as dlu
import optax
import equinox as eqx
import zodiax as zdx
import numpyro.distributions as dist
import json
from Classes.utils import get_sweep_values, set_array

############################
# Exports
############################

__all__ = [
    "hessian", "FIM", "_perturb",
    "scheduler", "sgd", "get_optimiser", "get_lr_model",
    "BaseModeller", "ModelParams", "SheraThreePlaneParams",
    "loglikelihood", "loss_fn", "step_fn"
]




############################
# Fisher Matrix Utilities
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


# def get_optimiser(pytree, optimisers):
#     """
#     Build an optimizer and return (params, optim, state).
#
#     Preserves the type of the input pytree (ModelParams vs EigenParams).
#     """
#     parameters = list(optimisers.keys())
#     if isinstance(pytree, EigenParams):
#         # preserve eigen-specific attributes
#         model_params = EigenParams(
#             eigen_coefficients=pytree.get("eigen_coefficients"),
#             p_ref=pytree.p_ref,
#             B=pytree.B,
#             params=pytree.params,
#             template=pytree.template,
#         )
#     else:
#         # default: regular ModelParams
#         model_params = ModelParams({p: pytree.get(p) for p in parameters})
#
#     # Build the optimizer (Optax)
#     optim = optax.multi_transform(
#         {p: optimisers[p] for p in parameters}, param_labels=model_params
#     )
#     state = optim.init(model_params)
#     return model_params, optim, state

def get_optimiser(pytree, optimisers):
    """
    Build an optimizer and return (params, optim, state).

    - For ModelParams: builds a multi_transform optimizer with per-parameter
      transforms (so you can assign different learning rates/optimizers).
    - For EigenParams: builds a single optimiser on the eigen_coefficients leaf.
      Gradient preconditioning (e.g., 1/λ_i scaling) should be handled outside
      of Optax before updates.

    Returns
    -------
    model_params : ModelParams or EigenParams
        Copy of the input pytree, filtered to the parameters to optimise.
    optim : optax.GradientTransformation
        Optax optimizer or multi_transform wrapper.
    state : optax.OptState
        Optimizer state.
    """
    if isinstance(pytree, EigenParams):
        # --- Eigenmode case ---
        model_params = EigenParams(
            eigen_coefficients=pytree.get("eigen_coefficients"),
            p_ref=onp.array(pytree.p_ref),
            B=onp.array(pytree.B),
            params=pytree.params,
            template=pytree.template,
        )
        # Only one differentiable leaf → use a single optimiser
        opt = list(optimisers.values())[0]  # should just be one
        optim = opt
        state = optim.init(model_params)

    else:
        # --- Pure parameter case ---
        parameters = list(optimisers.keys())
        model_params = ModelParams({p: pytree.get(p) for p in parameters})

        # Build multi_transform with per-parameter optimisers
        param_labels = ModelParams({p: p for p in parameters})
        optim = optax.multi_transform(optimisers, param_labels)
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

def assign_lr_vector(lr_vec, pytree, params, model_template=None):
    """
    Assign a precomputed learning-rate vector to a ModelParams structure.

    Parameters
    ----------
    lr_vec : array-like
        Learning rate vector of length equal to flattened parameters.
    pytree : ModelParams
        Parameter container with the actual storage keys (e.g. initial_model_params).
    params : list[str]
        Optimizer parameter keys (may include external names like 'm1_aperture.coefficients').
    model_template : ModelParams, optional
        Template used to provide path mappings and shapes. Defaults to `pytree`.

    Returns
    -------
    lr_model : ModelParams
        Learning rates structured to match params.
    """
    if model_template is None:
        model_template = pytree

    # Forward + inverse mappings
    path_map = model_template.get_param_path_map()
    inv_path_map = {v: k for k, v in path_map.items()}

    idx = 0
    lr_dict = {}

    for param in params:
        actual_key = inv_path_map.get(param, param)
        leaf = np.array(pytree.get(actual_key))
        size, shape = leaf.size, leaf.shape
        lr_dict[param] = lr_vec[idx: idx + size].reshape(shape)
        idx += size

    return ModelParams(lr_dict)


def get_lr_from_curvature(curv_vec, pytree=None, params=None, model_template=None,
                          key=None, eps=1e-12):
    """
    Compute learning rates as 1/curvature and assign into ModelParams.

    Parameters
    ----------
    curv_vec : array-like
        Curvature vector (e.g. diag(FIM) or eigenvalues).
    pytree : ModelParams, optional
        Parameter container with actual storage keys. Required if `key` is None.
    params : list[str], optional
        Optimizer parameter keys (external names). Required if `key` is None.
    model_template : ModelParams, optional
        Used for path mapping and shapes.
    key : str, optional
        If provided, assign the entire lr_vec under this single key
        (e.g. "eigen.coefficients").
    eps : float
        Regularization to avoid divide-by-zero.

    Returns
    -------
    lr_model : ModelParams
        Learning rates structured to match params or stored under a single key.
    """
    lr_vec = 1.0 / (np.asarray(curv_vec) + eps)

    if key is not None:
        # Eigenmode case: all learning rates live under one key
        return ModelParams({key: lr_vec})

    # Pure parameter case: delegate to assign_lr_vector
    if pytree is None or params is None:
        raise ValueError("Must provide `pytree` and `params` unless using `key`.")
    return assign_lr_vector(lr_vec, pytree, params, model_template)


#############################
# Parameter Container Classes
#############################

class BaseModeller(zdx.Base):
    """Base class to manage model parameters stored in a dictionary."""
    params: dict

    def __init__(self, params):
        self.params = params

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
        return self.set("params", dict([(param, values.get(param)) for param in self.keys]))

    def from_model(self, values):
        return self.set("params", dict([(param, values.get(param)) for param in self.keys]))

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
        return other.set(self.keys, self.values)

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

        params = {}
        for k, v in raw.items():
            if v is None:
                params[k] = None
            else:
                arr = np.array(v)
                if isinstance(v, (int, float)):
                    arr = v
                elif arr.shape == (1,):
                    arr = float(arr[0]) if isinstance(arr[0], float) else int(arr[0])
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr.reshape(-1)
                params[k] = arr

        return cls(params)


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
                "log_flux": 6.78,
            },
            "shera_flight": {
                "p1_diameter": 0.22,
                "p2_diameter": 0.025,
                "m1_focal_length": 0.604353,
                "m2_focal_length": -0.0545,
                "plane_separation": 0.55413,
                "pixel_size": 4.6e-6,
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

            # M1 Aberrations
            "m1_zernike_noll": np.arange(4, 11),
            "m1_zernike_amp": np.zeros(7),
            "m1_calibrated_power_law": 2.5,
            "m1_calibrated_amplitude": 0,
            "m1_uncalibrated_power_law": 2.5,
            "m1_uncalibrated_amplitude": 0,

            # M2 Aberrations
            "m2_zernike_noll": np.arange(4, 11),
            "m2_zernike_amp": np.zeros(7),
            "m2_calibrated_power_law": 2.5,
            "m2_calibrated_amplitude": 0,
            "m2_uncalibrated_power_law": 2.5,
            "m2_uncalibrated_amplitude": 0
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
            "m1_zernike_amp": "m1_aperture.coefficients",
            "m2_zernike_amp": "m2_aperture.coefficients"
        }

    def get(self, key):
        # This custom get method allows us to calculate what
        # psf_pixel_scale would be given the other parameters.
        # Falls back to the default get() behavior for other parameters
        if key == "psf_pixel_scale":
            return self.compute_psf_pixel_scale()
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

    def compute_psf_pixel_scale(self):
        """
        Computes the PSF pixel scale in arcseconds/pixel based on mirror geometry and pixel size.

        Returns
        -------
        float
            The computed psf_pixel_scale in arcseconds/pixel.
        """
        # Get focal lengths and pixel size
        f1 = self.get("m1_focal_length")
        f2 = self.get("m2_focal_length")
        sep = self.get("plane_separation")
        pixel_size = self.get("pixel_size")  # e.g., 6.5e-6 meters
        EFL = (f1**-1 + f2**-1 - sep * f1**-1 * f2**-1)**-1
        return dlu.rad2arcsec(pixel_size / EFL)

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
    Wrapper around eigenmode coefficients. Behaves similar to ModelParams,
    but inject() maps back into pure parameter space before updating the model.
    """
    eigen_coefficients: jax.Array  # coefficients in eigenbasis
    p_ref: np.ndarray = eqx.field(static=True)  # reference vector of pure params
    B: np.ndarray = eqx.field(static=True)  # eigenbasis matrix
    params: list      = eqx.field(static=True)  # list of param names
    template: object  = eqx.field(static=True)  # model template (e.g. initial_model_params)

    def keys(self):
        return ["eigen_coefficients"]

    def get(self, key):
        if key == "eigen_coefficients":
            return self.eigen_coefficients
        raise ValueError(f"key: {key} not found in object: {type(self).__name__}")

    def set(self, key, value):
        if key == "eigen_coefficients":
            return eqx.tree_at(lambda e: e.eigen_coefficients, self, value)
        raise ValueError(f"key: {key} not found in object: {type(self).__name__}")

    def inject(self, model):
        # project to pure parameters and inject into the model
        p = self.to_pure()
        pure_params = unpack_params(p, self.params, self.template, pytree_cls=ModelParams)
        return pure_params.inject(model)

    def to_pure(self):
        """Project eigen coefficients back into pure parameter vector."""
        return self.p_ref + self.B @ self.eigen_coefficients




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

# # Original Function - Updated 20250902
# @eqx.filter_jit
# def step_fn(model_params, data, var, model, lr_model, optim, state):
#     """Performs one optimization step and updates model parameters."""
#     loss, grads = zdx.filter_value_and_grad(model_params.keys)(loss_fn)(model, data, var)
#     grads = tree.map(lambda x, y: x * y, lr_model.replace(grads), lr_model)
#     updates, state = optim.update(grads, state, model_params)
#     model_params = zdx.apply_updates(model_params, updates)
#     model = model_params.inject(model)
#     return loss, model, model_params, state

@eqx.filter_jit
def step_fn(model_params, data, var, model, lr_model, optim, state):
    """
    Perform one optimization step and update model parameters.

    This function supports two parameterizations:
    1. Pure parameter space (ModelParams): updates directly on physical parameters.
    2. Eigenmode space (EigenParams): updates on eigen_coefficients, then projects
       back into pure space before injecting into the model.

    Parameters
    ----------
    model_params : ModelParams or EigenParams
        Current parameter container for optimization.
    data : ndarray
        Observed image data.
    var : ndarray
        Noise variance associated with the data.
    model : SheraThreePlane_Model
        Current optical model to be updated.
    lr_model : ModelParams
        Learning-rate structure matching parameters (or eigen_coefficients).
    optim : optax.GradientTransformation
        Optimizer object (e.g., SGD, Adam).
    state : optax.OptState
        Current optimizer state.

    Returns
    -------
    loss : float
        Current loss value for this iteration.
    model : SheraThreePlane_Model
        Updated optical model with new parameters injected.
    model_params : ModelParams or EigenParams
        Updated parameter container (depending on optimization mode).
    state : optax.OptState
        Updated optimizer state.
    """
    if isinstance(model_params, EigenParams):
        # --- Eigenmode branch ---
        # Project eigen coefficients → pure parameter vector
        p = model_params.to_pure()
        pure_params = unpack_params(p, model_params.params, model_params.template)

        # Inject into model so gradients are taken wrt physical parameters
        model = set_array(model, dict(zip(pure_params.keys, pure_params.values)))

        # Compute gradients wrt pure parameters
        loss, grads_pure = zdx.filter_value_and_grad(pure_params.keys)(loss_fn)(model, data, var)

        # Pack pure grads into vector, map into eigenmode space via Bᵀ
        grads_vec, _ = pack_params(grads_pure, pure_params.keys, model_params.template)
        grads = {"eigen_coefficients": model_params.B.T @ grads_vec}

        # Apply learning-rate scaling
        grads = tree.map(lambda x, y: x * y, lr_model.replace(grads), lr_model)

        # Update optimizer state and apply updates
        updates, state = optim.update(grads, state, model_params)
        model_params = zdx.apply_updates(model_params, updates)

        # Re-inject updated pure parameters into the model
        p_updated = model_params.to_pure()
        pure_params_updated = unpack_params(p_updated, model_params.params, model_params.template)
        model = set_array(model, dict(zip(pure_params_updated.keys, pure_params_updated.values)))

    else:
        # --- Pure parameter branch ---
        loss, grads = zdx.filter_value_and_grad(model_params.keys)(loss_fn)(model, data, var)

        # Apply learning-rate scaling
        grads = tree.map(lambda x, y: x * y, lr_model.replace(grads), lr_model)

        # Update optimizer state and apply updates
        updates, state = optim.update(grads, state, model_params)
        model_params = zdx.apply_updates(model_params, updates)

        # Re-inject updated parameters into the model
        model = model_params.inject(model)

    return loss, model, model_params, state


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
    from Classes.modeling import SheraThreePlane_Model # Importing locally avoids a circular import error
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

