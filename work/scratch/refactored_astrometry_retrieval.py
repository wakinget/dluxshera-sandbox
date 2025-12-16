

# The goal of this script is to set up the astrometry retrieval algorithm using the refactored dLuxShera codebase

# High Level Steps:
# Create a Config
# Build two Parameter Specs
    # One for the forward model
    # Another for the parameter inference
# Build a forward Parameter Store -> forward_store
    # Store won't contain derived values by default
    # Use a helper function to update the store with derived values
        # forward_store = refresh_derived(store, ...)
# Build an inference Parameter Store -> truth_store
# Build a SheraThreePlaneBinder from config, forward_spec, and forward_store

# Define Inference Keys + PriorSpec
# Sample from PriorSpec to seed initial model

# Imports
import jax
from pathlib import Path
import time, datetime, os
import jax.numpy as jnp
import numpy as np
import numpy.random._generator as rng
import jax.random as jr

from dluxshera.optics.config import SheraThreePlaneConfig, SHERA_TESTBED_CONFIG
from dluxshera.params.packing import unpack_params as store_unpack_params
from dluxshera.params.spec import (
    ParamKey,
    ParamSpec,
    build_forward_model_spec_from_config,
    build_inference_spec_basic,
)
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transforms import get_resolver
from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.optimization import make_binder_image_nll_fn, run_simple_gd
from dluxshera.plot.plotting import plot_parameter_history, plot_parameter_history_grid, plot_psf_comparison, plot_psf_single


# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

inferno = mpl.colormaps["inferno"]
seismic = mpl.colormaps["seismic"]
coolwarm = mpl.colormaps["coolwarm"]

inferno.set_bad("k", 0.5)
seismic.set_bad("k", 0.5)
coolwarm.set_bad("k", 0.5)

plt.rcParams['image.cmap'] = 'inferno'
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = 'lower'
plt.rcParams['figure.dpi'] = 120


# Directories
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DP_PATH = _PACKAGE_ROOT / "data" / "diffractive_pupil.npy"
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEFAULT_RESULTS_DIR = Path("Results/refactored_astrometry_retrieval_"+timestamp)


# Initial Settings
jax.config.update("jax_enable_x64", True)
rng_seed = 42
add_noise = False


##########################
# Start Building the model
##########################

rng_key = jr.PRNGKey(rng_seed)

# Start with a pre-defined config
cfg = SHERA_TESTBED_CONFIG

# Use the config to set Zernike Noll indices
cfg.primary_noll_indices = tuple(range(4, 12))
cfg.secondary_noll_indices = tuple(range(4, 12))

# Create Parameter Specs from the config
forward_spec = build_forward_model_spec_from_config(cfg)
inference_spec = build_inference_spec_basic()

# Create forward Parameter Store from the specs
forward_truth_store = ParameterStore.from_spec_defaults(forward_spec)

# Update any desired parameters - This defines the Truth value for the Data
forward_truth_store = forward_truth_store.replace(
    {
        "binary.separation_as": 10.0,
        "binary.position_angle_deg": 90.0,
        "binary.x_position_as": 0.0,
        "binary.y_position_as": 0.0,
        "imaging.exposure_time_s": 1.0,
    }
)

# Compute derived parameters
sys_id = "shera_threeplane"
forward_truth_store = refresh_derived(forward_truth_store, forward_spec, get_resolver(sys_id), system_id=sys_id)

# Create an inference Parameter Store
inference_store = ParameterStore.from_spec_defaults(inference_spec)
# Update the inference_store with values from forward_truth_store
# This should align inference_store and forward_truth_store
inference_store = inference_store.replace(
    {
        "binary.separation_as": forward_truth_store.get("binary.separation_as"),
        "binary.position_angle_deg": forward_truth_store.get("binary.position_angle_deg"),
        "binary.x_position_as": forward_truth_store.get("binary.x_position_as"),
        "binary.y_position_as": forward_truth_store.get("binary.y_position_as"),
        "binary.log_flux_total": forward_truth_store.get("binary.log_flux_total"),
        "binary.contrast": forward_truth_store.get("binary.contrast"),
        "system.plate_scale_as_per_pix": forward_truth_store.get("system.plate_scale_as_per_pix"),
        "primary.zernike_coeffs": forward_truth_store.get("primary.zernike_coeffs"),
        "secondary.zernike_coeffs": forward_truth_store.get("secondary.zernike_coeffs"),
    }
)
# This specific way of aligning the inference store with the forward_truth_store could be improved using a method
# Ex. inference_store = inference_store.update_from_store(forward_truth_store) or similar

# Create the Binder
binder = SheraThreePlaneBinder(cfg, forward_spec, forward_truth_store)
# The binder is the object that acts like the dLux Telescope.
# It holds the source, optics + detector, and exposes the .model() method

# Generate the true Data PSF
data = binder.model()

# Optionally add noise to the data
if add_noise:
    rng_key, split_key = jr.split(rng_key)
    if np.min(data) > 100:
        data = np.sqrt(data) * jr.normal(split_key, data.shape) + data # Gaussian Approximation
    else:
        data = jr.poisson(split_key, data) # Add Poisson shot noise

# Assume image variance is given by shot noise
data_var = data


######################
# Set up the inference
######################

# Choose inference keys
infer_keys = (
    "binary.separation_as",
    "binary.position_angle_deg",
    "binary.x_position_as",
    "binary.y_position_as",
    "binary.log_flux_total",
    "binary.contrast",
    "system.plate_scale_as_per_pix",
    "primary.zernike_coeffs",
    # "secondary.zernike_coeffs", # Remove secondary Zernike's for stability
)
inference_subspec = inference_spec.subset(infer_keys)

# Set up prior knowledge
priors = {
    "binary.separation_as": 1e-3,
    "binary.position_angle_deg": 1e-1,
    "binary.x_position_as": 1e-3,
    "binary.y_position_as": 1e-3,
    "binary.log_flux_total": 1e-3,
    "binary.contrast": 1e-3,
    "system.plate_scale_as_per_pix": 1e-3,
    "primary.zernike_coeffs": np.full_like(inference_store.get("primary.zernike_coeffs"), 1.0),
    "secondary.zernike_coeffs": np.full_like(inference_store.get("primary.zernike_coeffs"), 1.0),
}
prior_spec = PriorSpec.from_sigmas(inference_store, priors)


# Draw an initial point for the model from the priors
rng_key, split_key = jr.split(rng_key)
init_store = prior_spec.sample_near(inference_store, rng_key=split_key, keys=infer_keys)
init_update = {key: init_store.get(key) for key in infer_keys}
init_psf = binder.model(init_store)


# Build the Loss function
nll_loss_fn, theta0 = make_binder_image_nll_fn(
    cfg=binder.cfg,
    forward_spec=forward_spec,
    base_forward_store=init_store,
    infer_keys=infer_keys,
    data=data,
    var=data_var,
    noise_model="gaussian",
    reduce="sum",
)
# nll_loss_fn(theta) gives the negative log-likelihood loss for a given input theta vector

def map_loss_fn(theta: np.ndarray) -> np.ndarray:
    store_theta = store_unpack_params(inference_subspec, theta, init_store)
    nll_loss = nll_loss_fn(theta)
    prior_gaussian_loss = prior_spec.quadratic_penalty(store_theta, center_store=forward_truth_store, keys=infer_keys)
    return nll_loss + prior_gaussian_loss
# the map_loss_fn(theta) gives the maximum a posteriori (MAP) loss value for an input theta vector
# This incorporates a quadratic penalty from prior knowledge on top of the NLL loss

# Choose which loss function to use
loss_fn = nll_loss_fn
loss0 = loss_fn(theta0)

# Now run the gradient descent optimization
n_iter = 100
theta_final, history = run_simple_gd(
    loss_fn=loss_fn, # nll_loss_fn, or map_loss_fn
    theta0=theta0,
    learning_rate=0.5,
    num_steps=n_iter,
)

# Collect GD outputs
final_store = store_unpack_params(inference_subspec, theta_final, init_store)
final_psf = binder.model(final_store)


##################
# Plot the Outputs
##################

# Make a plot of our Starting Point
plot_psf_comparison(
    data=data,
    model=init_psf,
    var=data_var,
    model_label="Initial Model",
    save_path=DEFAULT_RESULTS_DIR / "initial_psf_comparison.png",
)

# Make a plot of our Ending Point
plot_psf_comparison(
    data=data,
    model=final_psf,
    var=data_var,
    model_label="Final Model",
    save_path=DEFAULT_RESULTS_DIR / "final_psf_comparison.png",
)

# Plot the loss history
losses = np.asarray(history["loss"])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes = axes.flatten()
# Left: Full loss history
plot_parameter_history(
    names="Loss",
    histories=losses,
    true_vals=loss0,
    ax=axes[0],
    title="Optimization Loss History",
    show=False,
    close=False,
)
# Right: Zoom into last 10 iterations
axes[1].plot(np.arange(n_iter - 10, n_iter) + 1, losses[-10:])
axes[1].set_title(f"Last 10 Iterations, Final= {losses[-1]:.3f}")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss")
axes[1].axhline(loss0, linestyle="--", color="k", alpha=0.6, label="True Loss")
final_delta = np.abs(losses[-1] - loss0)
if final_delta != 0:
    axes[1].set_ylim(loss0-3*final_delta, loss0+3*final_delta)
fig.tight_layout()
fig.savefig(DEFAULT_RESULTS_DIR / "loss_history.png", dpi=300)
plt.close()

# Plot parameter histories
# plot_parameter_history_grid()
