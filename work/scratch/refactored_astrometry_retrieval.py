

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
    make_inference_subspec, build_inference_spec_basic,
)
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.params.transforms import get_resolver
from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.optimization import (
    generate_fim_labels_refactor,
    map_labels_to_keys,
    make_binder_nll_fn,
    run_shera_gd,
    fim_theta,
)
from dluxshera.inference.run_artifacts import build_index_map
from dluxshera.inference.signals import build_signals
from dluxshera.plot.plotting import (
    plot_fim,
    plot_parameter_history,
    plot_psf_comparison,
    plot_signals_panels,
    plot_signals_grid,
)
from dluxshera.plot.printing import print_optimization_summary
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

# Start simulation timer
t0_script = time.time()

rng_key = jr.PRNGKey(rng_seed)

# Start with a pre-defined config
cfg = SHERA_TESTBED_CONFIG
# Use the config to set Zernike Noll indices
# The Noll indices are a 'structural' parameter of the model,
# this setting influences the size of the basis, and
# generally shouldn't be changed after creation.
cfg = cfg.replace(primary_noll_indices=tuple(range(4, 12)),
                  secondary_noll_indices=tuple(range(4, 12)))

# Create Parameter Specs from the config
forward_spec = build_forward_model_spec_from_config(cfg)
inference_spec = build_inference_spec_basic(cfg)

# Create forward Parameter Store from the specs
forward_truth_store = ParameterStore.from_spec_defaults(forward_spec)

# Update any desired parameters - This defines the Truth value for the Data
forward_truth_store = forward_truth_store.replace(
    {
        "binary.separation_as": 10.0,
        "binary.position_angle_deg": 90.0,
        "binary.x_position_as": 0.0,
        "binary.y_position_as": 0.0,
        "imaging.exposure_time_s": 1800.0,
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
    if np.min(data) > 100: # Use Gaussian Approximation
        data = np.sqrt(data) * jr.normal(split_key, data.shape) + data
    else: # Add Poisson shot noise
        data = jr.poisson(split_key, data)

# Assume image variance is given by shot noise
data_var = data


######################
# Set up the inference
######################
print("Configuring Inference...")
# Choose inference keys
# Only these keys will be optimized
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
inference_subspec = make_inference_subspec(base_spec=inference_spec, infer_keys=infer_keys, cfg=cfg)

# Set up prior knowledge
prior_info = {
    "binary.separation_as": {"sigma": 1e-6, "dist": "Normal"},
    "binary.position_angle_deg": {"sigma": 1e-3, "dist": "Uniform"},
    "binary.x_position_as": {"sigma": 1e-6, "dist": "Normal"},
    "binary.y_position_as": {"sigma": 1e-6, "dist": "Normal"},
    "binary.log_flux_total": {"sigma": 1e-6, "dist": "LogNormal"},
    "binary.contrast": {"sigma": 1e-6, "dist": "LogNormal"},
    "system.plate_scale_as_per_pix": {"sigma": 1e-6, "dist": "LogNormal"},
    "primary.zernike_coeffs_nm": {
        "sigma": np.full_like(forward_truth_store.get("primary.zernike_coeffs_nm"), 1e-2),
        "dist": "Normal",
    },
    "secondary.zernike_coeffs_nm": {
        "sigma": np.full_like(forward_truth_store.get("secondary.zernike_coeffs_nm"), 1e-2),
        "dist": "Normal",
    },
}
prior_spec = PriorSpec.from_info(forward_truth_store, prior_info)

print("Drawing starting point from priors...")
# Draw an initial point for the model from the priors
rng_key, split_key = jr.split(rng_key)
init_store = prior_spec.sample_near(forward_truth_store, rng_key=split_key, keys=infer_keys)
init_psf = binder.model(init_store)

print("Building the loss function...")
# Build the Loss function
nll_loss_fn, theta0 = make_binder_nll_fn(
    binder=binder,
    infer_keys=infer_keys,
    data=data,
    var=data_var,
    noise_model="gaussian",
    reduce="sum",
    theta0_store=init_store,
)
# nll_loss_fn(theta) gives the negative log-likelihood loss for a given input theta vector
index_map = build_index_map(inference_subspec, init_store, theta=theta0)
fim_labels = generate_fim_labels_refactor(
    infer_keys,
    cfg=cfg,
    store=init_store,
)

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

# Calculate the gradients
grads_true = jax.grad(loss_fn)(theta_true)
grads0 = jax.grad(loss_fn)(theta0)

print("Computing Fisher Information Matrix (FIM) for preconditioning...")
F = fim_theta(loss_fn, theta_true)
plot_fim(F, fim_labels, save_path=DEFAULT_RESULTS_DIR / "fim.png",
         vmin=4, vmax=14, show=False)

fim_diag = jnp.diag(F)
lr_vec = 1.0 / (np.asarray(fim_diag) + 1e-12)

print("FIM diag: min={:.3e}, max={:.3e}".format(float(jnp.min(fim_diag)), float(jnp.max(fim_diag))))
print("LR vec : min={:.3e}, max={:.3e}".format(float(jnp.min(lr_vec)), float(jnp.max(lr_vec))))

print("\nRefactored FIM diagonal and learning rates (via index_map):")
for entry in index_map["entries"]:
    name  = entry["name"]
    start = entry["start"]
    stop  = entry["stop"]
    shape = entry.get("shape", ())

    n = stop - start

    if n == 1:
        print(
            f"  {name:40s} : "
            f"curv={fim_diag[start]:.3e}  lr={lr_vec[start]:.3e}"
        )
    else:
        print(f"  {name:40s} : shape={shape}")
        for i, (c, l) in enumerate(zip(fim_diag[start:stop], lr_vec[start:stop])):
            print(
                f"    {name}[{i:02d}] : "
                f"curv={c:.3e}  lr={l:.3e}"
            )


print("Running FIM-preconditioned gradient descent...")
# Now run the gradient descent optimization
n_iter = 100
theta_final, trace = run_shera_gd(
    loss_fn=loss_fn,
    theta0=theta0,
    index_map=index_map,
    learning_rate=0.5,
    lr_vec=lr_vec,
    num_steps=n_iter,
    runs_dir=DEFAULT_RESULTS_DIR,
    return_artifacts=False,
    theta_space="primitive",
    curvature=fim_diag,
    precond={"lr_vec": lr_vec},
)
# trace carries the theta + loss trace used by build_signals
# _artifacts is meant for on-disk logging.

final_store = store_unpack_params(inference_subspec, theta_final, init_store)

# Collect GD outputs
final_psf = binder.model(final_store)

##################
# Print a Summary
##################
labels_by_key = map_labels_to_keys(
    infer_keys,
    fim_labels,
    index_map=index_map,
)

print("\n==============================")
print("FIM-preconditioned Gradient Descent Summary")
print("==============================")
print(f"n_iter = {n_iter}")
print(f"loss(true theta) = {loss_true:.8g}")
print(f"loss(init theta0) = {loss0:.8g}")
print(f"loss(final theta)       = {float(loss_fn(theta_final)):.8g}")
print("")

summary_true = {k: forward_truth_store.get(k) for k in infer_keys}
summary_init = {k: init_store.get(k) for k in infer_keys}
summary_final = {k: final_store.get(k) for k in infer_keys}
print_optimization_summary(
    summary_true,
    summary_init,
    summary_final,
    labels=labels_by_key,
)


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
losses = np.asarray(trace["loss"])
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes = axes.flatten()
# Left: Full loss history
plot_parameter_history(
    names=("Loss",),
    histories=(losses,),
    true_vals=(float(loss_true),),
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
axes[1].axhline(loss_true, linestyle="--", color="k", alpha=0.6, label="True Loss")
final_delta = np.abs(losses[-1] - loss_true)
if final_delta != 0:
    axes[1].set_ylim(loss_true-3*final_delta, loss_true+3*final_delta)
fig.tight_layout()
fig.savefig(DEFAULT_RESULTS_DIR / "loss_history.png", dpi=300)
plt.close()

# Build the signals and plot parameter residuals
# build_signals computes 'signals' from the trace.
# Signals represent residual errors, sometimes scaled to specific units like uas or ppm
signals = build_signals(
    trace,
    meta={},
    decoder=lambda theta: store_unpack_params(
        inference_subspec,
        theta,
        init_store,
    ).refresh_derived(inference_spec),
    truth=forward_truth_store,
    signal_set="intro",
)
# plot_signals_panels(
#     signals,
#     DEFAULT_RESULTS_DIR,
#     title_prefix="Refactored astrometry retrieval",
#     include_zernike_rms=False,
# )
plot_signals_grid(
    signals,
    DEFAULT_RESULTS_DIR,
    include_zernike_rms=False,
    figsize=(15, 10),
    show=False,
)

t1_script = time.time()
print("Script finished in %.3f sec" % (t1_script-t0_script))
