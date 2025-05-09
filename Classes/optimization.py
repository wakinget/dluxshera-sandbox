# Classes/optimization.py

import jax
import jax.numpy as np
from jax import grad, linearize, jit, lax, tree, config as jax_config
import optax
import equinox as eqx
import zodiax as zdx
# from zodiax import tree


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


def get_optimiser(pytree, optimisers, parameters=None):
    """Builds multi-transform optimizer and initial optimizer state."""
    if parameters is not None:
        optimisers = dict([(p, optimisers[p]) for p in parameters])
    else:
        parameters = list(optimisers.keys())

    model_params = ModelParams(dict([(p, pytree.get(p)) for p in parameters]))
    param_spec = ModelParams(dict([(param, param) for param in parameters]))
    optim = optax.multi_transform(optimisers, param_spec)
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
                "m1_diameter": 0.09,
                "m2_diameter": 0.025,
                "m1_focal": 0.35796,
                "m2_focal": -0.041935,
                "plane_separation": 0.320,
                "pixel_size": 6.5e-6
            },
            "shera_flight": {
                "m1_diameter": 0.22,
                "m2_diameter": 0.025,
                "m1_focal": 0.604353,
                "m2_focal": -0.0545,
                "plane_separation": 0.55413,
                "pixel_size": 4.6e-6
            }
        }

        # Set the default parameters (shera_testbed)
        defaults = point_designs.get(point_design, point_designs["shera_testbed"])

        # Add other default parameters
        defaults.update({
            # Sampling and resolution settings
            "pupil_npix": 256,
            "psf_npix": 256,

            # Source parameters
            "x_position": 0,
            "y_position": 0,
            "separation": 10,
            "position_angle": 90,
            "contrast": 0.3,
            "log_flux": 6.832,
            "wavelength": 550,
            "bandwidth": 110,
            "n_wavelengths": 5,


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

        # Initialize with defaults
        super().__init__(defaults)

        # Update with user-provided values
        if params is not None:
            self.replace(params)

        # Initialize frozen set
        self._fixed = set()

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

    def freeze(self, *patterns):
        """Freeze parameters that match the given patterns."""
        for pattern in patterns:
            for key in self.params.keys():
                if pattern in key:
                    self._fixed.add(key)

    def unfreeze(self, *patterns):
        """Unfreeze parameters that match the given patterns."""
        for pattern in patterns:
            for key in list(self._fixed):
                if pattern in key:
                    self._fixed.discard(key)

    def is_frozen(self, key):
        """Check if a parameter is currently frozen."""
        return key in self._fixed

    def get_frozen_params(self):
        """Return a list of all currently frozen parameters."""
        return list(self._fixed)

    def to_dict(self):
        """Flatten the parameter hierarchy for easy export."""
        return self.params


############################
# Loss and Update Functions
############################

def loglikelihood(model, data, var):
    """Normal log-likelihood."""
    return jax.scipy.stats.norm.logpdf(model.model(), loc=data, scale=np.sqrt(var))

def loss_fn(model, data, var):
    """Negative log-likelihood (loss function)."""
    return -np.nansum(loglikelihood(model, data, var))


@eqx.filter_jit
def step_fn(model_params, data, var, model, lr_model, optim, state):
    """Performs one optimization step and updates model parameters."""
    loss, grads = zdx.filter_value_and_grad(model_params.keys)(loss_fn)(model, data, var)
    grads = tree.map(lambda x, y: x * y, lr_model.replace(grads), lr_model)
    updates, state = optim.update(grads, state, model_params)
    model_params = zdx.apply_updates(model_params, updates)
    model = model_params.inject(model)
    return loss, model, model_params, state
