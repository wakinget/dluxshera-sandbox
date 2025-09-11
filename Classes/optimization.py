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
    # fisher / curvature
    "hessian", "FIM", "_perturb",

    # optax helpers
    "scheduler", "sgd", "get_optimiser", "get_lr_model",
    "assign_lr_vector", "get_lr_from_curvature",

    # containers
    "BaseModeller", "ModelParams",
    "SheraTwoPlaneParams", "SheraThreePlaneParams", "EigenParams",

    # likelihood / step
    "loglikelihood", "loss_fn", "step_fn",

    # reparameterisation utils
    "generate_fim_labels", "pack_params", "unpack_params", "build_basis",

    # priors
    "construct_priors_from_dict",
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
    param_labels = model_params.set("params", {p: p for p in parameters})
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
    Wrapper around eigenmode coefficients, with same structure as ModelParams:
    - All differentiable leaves live in `params`, here {"eigen_coefficients": jax.Array}.
    - Metadata is static (p_ref, B, pure_keys, template).
    """
    # Differentiable leaves:
    # params: dict = eqx.field()  # inherited from ModelParams

    # Static metadata
    p_ref: onp.ndarray = eqx.field(static=True)   # store as NumPy to avoid static JAX warnings
    B: onp.ndarray     = eqx.field(static=True)
    pure_keys: list    = eqx.field(static=True)   # names of the underlying pure params (external)
    template: object   = eqx.field(static=True)   # e.g. initial_model_params

    def to_pure(self):
        """Project eigen coefficients back into pure parameter vector."""
        c = self.get("eigen_coefficients")  # a jax.Array
        # p_ref, B are NumPy; that's OK because they are static. Convert as needed:
        return np.asarray(self.p_ref) + np.asarray(self.B) @ c

    def inject(self, model):
        """Project to pure parameters and inject into the model."""
        p = self.to_pure()
        pure_params = unpack_params(p, self.pure_keys, self.template, pytree_cls=ModelParams)
        return pure_params.inject(model)




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
    One optimization step, regardless of parameterization.

    Works for:
      - ModelParams (pure parameter space)
      - EigenParams (eigenmode space; projection happens inside .inject())

    Requirements:
      - model_params.inject(model) must return a model with those params applied.
      - lr_model must share the *same pytree structure* as model_params
        (e.g., EigenParams for eigen mode, ModelParams for pure mode).

    Returns
    -------
    loss, model, model_params, state
    """

    # Define loss as a function of "params" (pure or eigen) by delegating to .inject()
    def _loss_with_params(params, model, data, var):
        model_in = params.inject(model)         # projection+injection if EigenParams
        return loss_fn(model_in, data, var)

    # Compute loss and grads wrt "model_params" leaves
    # loss, grads = zdx.filter_value_and_grad(model_params.keys)(_loss_with_params)(model_params, model, data, var)
    loss, grads = eqx.filter_value_and_grad(_loss_with_params)(model_params, model, data, var)

    # Sanity: lr_model must align with grads/params structurally
    # (If you ever see a tree/type mismatch in optax.update/apply_updates,
    #  it usually means lr_model wasn't constructed with the same structure.)
    grads = jax.tree_util.tree_map(lambda g, s: g * s, grads, lr_model)

    # Optax step
    updates, state = optim.update(grads, state, model_params)
    model_params = zdx.apply_updates(model_params, updates)

    # Inject the *updated* params into the model for the next iteration
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

