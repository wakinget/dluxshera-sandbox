

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
    make_inference_subspec,
)
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transforms import get_resolver
from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.optimization import make_binder_image_nll_fn, run_simple_gd, fim_theta, fim_theta_shera
from dluxshera.plot.plotting import plot_parameter_history, plot_parameter_history_grid, plot_psf_comparison, plot_psf_single
from dluxshera.params.packing import pack_params, unpack_params

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
print("Starting Simulation...")
print("Creating Config, Spec, Store, and Binder...")

rng_key = jr.PRNGKey(rng_seed)

# Start with a pre-defined config
cfg = SHERA_TESTBED_CONFIG
# Use the config to set Zernike Noll indices
cfg = cfg.replace(primary_noll_indices=tuple(range(4, 12)),
                  secondary_noll_indices=tuple(range(4, 12)))

# Create Parameter Specs from the config
forward_spec = build_forward_model_spec_from_config(cfg)

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
forward_truth_store = forward_truth_store.refresh_derived(forward_spec)

# Create the Binder
binder = SheraThreePlaneBinder(cfg, forward_spec, forward_truth_store)
# The binder is the object that acts like the dLux Telescope.
# It holds the source, optics + detector, and exposes the .model() method

###############
# Generate data
###############
print("Generating synthetic data...")

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
print("Configuring Inference...")

# Choose inference keys
infer_keys = (
    "binary.separation_as",
    "binary.position_angle_deg",
    "binary.x_position_as",
    "binary.y_position_as",
    "binary.log_flux_total",
    "binary.contrast",
    "system.plate_scale_as_per_pix",
    "primary.zernike_coeffs_nm",
    # "secondary.zernike_coeffs_nm", # Remove secondary Zernike's for stability
)
inference_subspec = make_inference_subspec(base_spec=forward_spec, infer_keys=infer_keys, cfg=cfg)

# Set up prior knowledge
priors = {
    "binary.separation_as": 1e-6,
    "binary.position_angle_deg": 1e-3,
    "binary.x_position_as": 1e-6,
    "binary.y_position_as": 1e-6,
    "binary.log_flux_total": 1e-6,
    "binary.contrast": 1e-6,
    "system.plate_scale_as_per_pix": 1e-6,
    "primary.zernike_coeffs_nm": np.full_like(
        forward_truth_store.get("primary.zernike_coeffs_nm"), 1e-2
    ),
    "secondary.zernike_coeffs_nm": np.full_like(
        forward_truth_store.get("secondary.zernike_coeffs_nm"), 1e-2
    ),
}
prior_spec = PriorSpec.from_sigmas(forward_truth_store, priors)

print("Drawing starting point from priors...")
# Draw an initial point for the model from the priors
rng_key, split_key = jr.split(rng_key)
init_store = prior_spec.sample_near(forward_truth_store, rng_key=split_key, keys=infer_keys)
init_psf = binder.model(init_store)

print("Building the loss function...")
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

# Calculate True + Initial Loss values
theta_true = pack_params(inference_subspec, forward_truth_store)
loss_true = loss_fn(theta_true)
loss0 = loss_fn(theta0)

print("Running gradient descent optimization...")
# Now run the gradient descent optimization
n_iter = 200
theta_final, history = run_simple_gd(
    loss_fn=loss_fn, # nll_loss_fn, or map_loss_fn
    theta0=theta0,
    learning_rate=1e-2,
    num_steps=n_iter,
)

# Collect GD outputs
final_store = store_unpack_params(inference_subspec, theta_final, init_store)
final_psf = binder.model(final_store)

##################
# Print a Summary
##################
def _as_np(x):
    return np.asarray(x)

def _fmt_scalar(x, *, prec=8):
    try:
        return f"{float(x):.{prec}g}"
    except Exception:
        return str(x)

def _is_scalar(arr: np.ndarray) -> bool:
    arr = np.asarray(arr)
    return arr.ndim == 0 or arr.size == 1

def _iter_labels_for_key(key: str, n: int):
    # Special-case Zernike coeff labels if we can
    if key == "primary.zernike_coeffs_nm":
        nolls = getattr(cfg, "primary_noll_indices", None)
        if nolls is not None and len(nolls) == n:
            return [f"Z{int(z)}" for z in nolls]
    if key == "secondary.zernike_coeffs_nm":
        nolls = getattr(cfg, "secondary_noll_indices", None)
        if nolls is not None and len(nolls) == n:
            return [f"Z{int(z)}" for z in nolls]

    # Generic fallback: index labels
    return [str(i) for i in range(n)]

def _print_vector(key: str, true_val, init_val, final_val, *, prec=8):
    t = np.ravel(_as_np(true_val))
    i = np.ravel(_as_np(init_val))
    f = np.ravel(_as_np(final_val))

    if t.size != i.size or t.size != f.size:
        print(f"    [WARN] size mismatch: true={t.size}, init={i.size}, final={f.size}")
        n = min(t.size, i.size, f.size)
        t, i, f = t[:n], i[:n], f[:n]

    labels = _iter_labels_for_key(key, t.size)

    # Print one line per element
    for idx, lab, tv, iv, fv in zip(range(t.size), labels, t, i, f):
        dt_i = float(iv - tv)
        dt_f = float(fv - tv)
        print(
            f"    [{idx:>3}] {lab:>4} : "
            f"true={_fmt_scalar(tv, prec=prec)}  "
            f"init={_fmt_scalar(iv, prec=prec)}  (Δ={_fmt_scalar(dt_i, prec=prec)})  "
            f"final={_fmt_scalar(fv, prec=prec)} (Δ={_fmt_scalar(dt_f, prec=prec)})"
        )

print("\n==============================")
print("Gradient Descent Summary")
print("==============================")
print(f"n_iter = {n_iter}")
print(f"loss(true theta) = {_fmt_scalar(loss_true)}")
print(f"loss(init theta0) = {_fmt_scalar(loss0)}")
print(f"loss(final theta)       = {_fmt_scalar(loss_fn(theta_final))}")
print("")

for k in infer_keys:
    true_val = forward_truth_store.get(k)
    init_val = init_store.get(k)
    final_val = final_store.get(k)

    t = _as_np(true_val)
    i = _as_np(init_val)
    f = _as_np(final_val)

    print(f"- {k}")
    if _is_scalar(t) and _is_scalar(i) and _is_scalar(f):
        # Scalar print
        tv = t.reshape(()) if t.size == 1 else t
        iv = i.reshape(()) if i.size == 1 else i
        fv = f.reshape(()) if f.size == 1 else f
        print(f"    true : {_fmt_scalar(tv)}")
        print(f"    init : {_fmt_scalar(iv)}  (Δ={_fmt_scalar(float(iv - tv))})")
        print(f"    final: {_fmt_scalar(fv)}  (Δ={_fmt_scalar(float(fv - tv))})")
    else:
        # Full vector print (flattened)
        print(f"    shape true/init/final: {t.shape} / {i.shape} / {f.shape}")
        _print_vector(k, t, i, f)

    print("")


##################
# Plot the Outputs
##################
print("Plotting outputs...")
psf_extent_as = binder.cfg.psf_npix * binder.base_forward_store.get("system.plate_scale_as_per_pix") / 2 * np.array([-1, 1, -1, 1])

# Make a plot of our Starting Point
plot_psf_comparison(
    data=data,
    model=init_psf,
    var=data_var,
    extent=psf_extent_as,
    model_label="Initial Model",
    save_path=DEFAULT_RESULTS_DIR / "initial_psf_comparison.png",
)

# Make a plot of our Ending Point
plot_psf_comparison(
    data=data,
    model=final_psf,
    var=data_var,
    extent=psf_extent_as,
    model_label="Final Model",
    save_path=DEFAULT_RESULTS_DIR / "final_psf_comparison.png",
)

# Plot the loss history
losses = np.asarray(history["loss"])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes = axes.flatten()
# Left: Full loss history
plot_parameter_history(
    names=("Loss",),
    histories=(losses,),
    true_vals=(float(loss0),),
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
