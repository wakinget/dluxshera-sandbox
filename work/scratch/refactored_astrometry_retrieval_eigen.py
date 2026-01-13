# This script initializes a 3-plane model, generates data, and uses the eigenmode reparameterization to run a single
# astrometry retrieval optimization loop to recover the specified parameter eigenmodes
# This script is set up to mirror refactored_astrometry_retrieval.py

# High Level Steps:
# Create/choose a Config
#   Pre-defined configs are available, but are customizable
# Build two Parameter Specs
#   One for the forward model -> forward_spec
#   Another for the parameter inference -> inference_spec
#   The Spec describes what parameters exist, and how the model uses them
#   The Spec does not actually store values for parameters
#   The forward_spec holds all parameters necessary to simulate data
#       Certain parameters in the forward_spec are derived from others
#           Ex. system.plate_scale derived from focal lengths + mirror separation
# Build a forward Parameter Store -> forward_truth_store
#   The store holds parameter values
#   The store uses a helper function to compute and populate derived values like the system plate scale
#       Transforms registered to each derived parameter are used to compute from primitive parameters
# Build a SheraThreePlaneBinder from config, forward_spec, and forward_truth_store
#   The binder is what 'binds' the parameters to the optics, source, and detector objects from dLux
# Use the binder to generate synthetic Data
#   Optionally add noise to the data
# Define Inference Keys + Priors
    # infer_keys defines which parameters to solve for
    # prior_info defines how well we know each parameter
# Sample from Priors to seed an initial starting point for the model
# Define the loss function
#   We normally use the Negative Log-Likelihood (NLL) between data and model
#   A MAP loss function that additionally incorporates a penalty based on priors is also available
# Compute the Fisher information matrix (FIM)
#   We use the inverse curvature of each parameter, given by the (inverse) diagonal of the FIM to define
#       per-parameter learning rates
# Run the gradient descent optimization
# Collect, Print, Plot and Save the results

# Imports
import jax
from pathlib import Path
import time, datetime, os
import hashlib
import json
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
from dluxshera.params.store import ParameterStore, refresh_derived, strip_structural
from dluxshera.params.transforms import get_resolver
from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.optimization import (
    generate_fim_labels_refactor,
    map_labels_to_keys,
    make_binder_nll_fn,
    run_shera_gd,
    fim_theta,
    EigenThetaMap,
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
DEFAULT_RESULTS_DIR = Path("Results/refactored_astrometry_retrieval_eigen_"+timestamp)


# Initial Settings
jax.config.update("jax_enable_x64", True)
rng_seed = 42
add_noise = False

# Eigenmode Settings
use_eigen = True          # Enables re-parameterization
whiten_basis = True       # If True, scales each eigenvector by 1/sqrt(lambda)
truncate_k = None         # int or None; keep top-k eigenmodes when set
truncate_by_eigval = None # float or None; only used when truncate_k is None.
                          # Drop eigenmodes with eigenvalue < this threshold.

# Inference Settings
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
    "secondary.zernike_coeffs_nm", # Optionally comment this one out
)

##########################
# Start Building the model
##########################
print("Starting Simulation...")
print("Creating Config, Spec, Store, and Binder...")
print("Eigenmode configuration:")
print(f"  use_eigen={use_eigen}")
print(f"  whiten_basis={whiten_basis}")
print(f"  truncate_k={truncate_k}")
print(f"  truncate_by_eigval={truncate_by_eigval}")

# Start simulation timer
t0_script = time.time()

rng_key = jr.PRNGKey(rng_seed)

# Start with a pre-defined config
cfg = SHERA_TESTBED_CONFIG
# config objects hold 'structural' parameters of the model
# ex: pupil/psf sampling, mirror focal lengths + sizes, etc.
# Use the config to set Zernike Noll indices
# The Noll indices are 'structural' to the model, if they change,
# then we need to rebuild the internal 'basis'.
# We generally set these structural parameters up prior to creating
# the model, and generally refrain from changing these parameters later
cfg = cfg.replace(primary_noll_indices=tuple(range(4, 12)),
                  secondary_noll_indices=tuple(range(4, 12)))

# Create Parameter Specs from the config
# parameter specs describe the available parameters and how they are used.
# The forward model spec describes all the parameters required to produce a forward PSF
# Most parameters are considered 'primitive' like the binary X/Y position or the pixel pitch
# Some parameters are 'derived' and are computed using a specific registered transform
# Ex: system.focal_length_m derived from system.m1_focal_length_m, system.m2_focal_length_m,
#       and system.m1_m2_separation_m
# Ex: system.plate_scale_as_per_pix derived from system.focal_length_m and system.pixel_pitch_m
forward_spec = build_forward_model_spec_from_config(cfg)
# The inference spec describes the set of parameters that we are allowed to solve for
# In the inference spec, all parameters are considered 'primitive' for the purposes of the optimization
# Ex: system.plate_scale_as_per_pix is 'derived' in the forward_spec, but 'primitive' in the inference_spec
# The difference between forward model and inference specs may seem confusing at first,
# but we are just being explicit about what parameters exist and how we use them.
# This provides flexibility for different model types (2-, 3-, or, 4-plane, etc.) that may be parameterized differently
inference_spec = build_inference_spec_basic(cfg)
# `build_inference_spec_basic()` is meant to return all inference parameters that we might want to use,
# but we might not want to use all of them.

# Create forward-model Parameter Store from the spec
forward_truth_store = ParameterStore.from_spec_defaults(forward_spec)
# The spec *describes* the parameters, while the store *holds* the parameters
# The store represents a set of parameters with specific values
# ParameterStore.from_spec_defaults(spec) populates a ParameterStore
#   using default values from the provided spec. By default, this method does not populate any derived

# If desired, replace any default parameters with custom values
# The forward_truth_store defines the 'truth' values for the Data
forward_truth_store = forward_truth_store.replace(
    {
        "binary.separation_as": 10.0,
        "binary.position_angle_deg": 90.0,
        "binary.x_position_as": 0.0,
        "binary.y_position_as": 0.0,
        "imaging.exposure_time_s": 1800.0,
        # "system.plate_scale_as_per_pix": 0.355, # This is normally derived, but user may override here
    }
)

# Compute derived parameters
forward_truth_store = forward_truth_store.refresh_derived(forward_spec)
# If desired, could I update the 'system.plate_scale_as_per_pix' here?

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
# Here we choose the subset of parameters that are listed in infer_keys
inference_subspec = make_inference_subspec(base_spec=inference_spec, infer_keys=infer_keys, cfg=cfg)

# Set up prior knowledge
# Choose the type of distribution and the standard deviation for each parameter
prior_info = {
    "binary.separation_as":          {"sigma": 1e-6, "dist": "Normal"},
    "binary.position_angle_deg":     {"sigma": 1e-3, "dist": "Uniform"},
    "binary.x_position_as":          {"sigma": 1e-6, "dist": "Normal"},
    "binary.y_position_as":          {"sigma": 1e-6, "dist": "Normal"},
    "binary.log_flux_total":         {"sigma": 1e-6, "dist": "LogNormal"},
    "binary.contrast":               {"sigma": 1e-6, "dist": "LogNormal"},
    "system.plate_scale_as_per_pix": {"sigma": 1e-6, "dist": "LogNormal"},
    "primary.zernike_coeffs_nm":     {
        "sigma": np.full_like(forward_truth_store.get("primary.zernike_coeffs_nm"), 1e-2),
        "dist": "Normal",
    },
    "secondary.zernike_coeffs_nm":   {
        "sigma": np.full_like(forward_truth_store.get("secondary.zernike_coeffs_nm"), 1e-2),
        "dist": "Normal",
    },
}
# PriorSpec.from_info() pulls the mean of the distribution from the provided store,
# and uses the sigma and distribution type from prior_info
prior_spec = PriorSpec.from_info(forward_truth_store, prior_info)

print("Drawing starting point from priors...")
# Draw an initial point for the model from the priors
rng_key, split_key = jr.split(rng_key)
init_store = prior_spec.sample(rng_key=split_key, keys=infer_keys)
# We use prior_spec.sample to draw a random sample from the priors using the stored prior_info
init_psf = binder.model(strip_structural(init_store, structural_keys=binder.structural_keys()))
# To compute the initial (perturbed) PSF, we can provide a store to the binder.model() method.
# By default, the .model() method will throw an error if the user provides any structural parameter updates.
# We can use the strip_structural helper to remove structural keys listed by the binder

# We can allow structural updates by passing the 'allow_rebuild' input argument to .model()
# In this case, the binder will rebuild the entire model using the provided store, and then evaluate the PSF
# This is slower due to the rebuild, and so we prefer to be strict about passing only non-structural parameters
# init_psf = binder.model(init_store, allow_rebuild=True) # Should be identical to earlier call

print("Building the loss function...")
# NLL = Negative Log-Likelihood
# make_binder_nll_fn builds and returns the loss function: nll_loss_fn(theta)
# nll_loss_fn(theta) gives the negative log-likelihood loss for a given input theta vector
# Internally, the input theta is merged into the binder, a model image is produced and compared to the data image,
# and the negative log-likelihood is returned.
# User may choose a "gaussian" or "poisson" log-likelihood, and also to return the "sum" or "mean" over the pixels
nll_loss_fn, theta0 = make_binder_nll_fn(
    binder=binder,
    infer_keys=infer_keys,
    data=data,
    var=data_var,
    noise_model="gaussian", # "gaussian" or "poisson"
    reduce="sum", # "sum" or "mean"
    theta0_store=init_store,
)
# Generate human-readable labels for parameter names
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
# the map_loss_fn(theta) gives the Maximum A Posteriori (MAP) loss value for an input theta vector
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

def build_eigen_index_map(dim: int) -> dict:
    entries = []
    for i in range(dim):
        entries.append(
            {
                "name": f"eigen.mode[{i:02d}]",
                "start": i,
                "stop": i + 1,
                "shape": [],
                "block": "eigen",
            }
        )

    payload = [(entry["name"], entry["shape"]) for entry in entries]
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    layout_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    return {"entries": entries, "layout_hash": layout_hash}

print("Computing Fisher Information Matrix (FIM) for preconditioning...")
theta_ref = theta_true  # Use truth for synthetic demo; swap to theta0 for realism later.
F = fim_theta(nll_loss_fn, theta_ref)
plot_fim(F, fim_labels, save_path=DEFAULT_RESULTS_DIR / "fim.png",
         vmin=4, vmax=14, show=False)

fim_diag = jnp.diag(F)

if use_eigen:
    if truncate_k is not None and truncate_by_eigval is not None:
        print(
            "truncate_k is set; ignoring truncate_by_eigval="
            f"{truncate_by_eigval}."
        )

    eigen_map_full = EigenThetaMap.from_fim(F, theta_ref, whiten=whiten_basis)
    eigvals_full = np.asarray(eigen_map_full.eigvals) if eigen_map_full.eigvals is not None else None

    if truncate_k is not None:
        k = int(truncate_k)
    elif truncate_by_eigval is not None and eigvals_full is not None:
        k = int(np.sum(eigvals_full >= truncate_by_eigval))
    else:
        k = None

    if k is not None:
        if k <= 0:
            print("truncate_by_eigval removed all modes; keeping top-1.")
            k = 1
        eigen_map = EigenThetaMap.from_fim(F, theta_ref, truncate=k, whiten=whiten_basis)
    else:
        eigen_map = eigen_map_full

    eigvals_kept = np.asarray(eigen_map.eigvals) if eigen_map.eigvals is not None else np.array([])
    k_kept = eigen_map.dim_eigen
    if eigvals_kept.size > 0:
        min_eval = float(np.min(eigvals_kept))
        max_eval = float(np.max(eigvals_kept))
    else:
        min_eval = float("nan")
        max_eval = float("nan")

    print("\nEigenThetaMap summary:")
    print(f"  N total dims: {eigen_map.dim_theta}")
    print(f"  k kept dims : {k_kept}")
    print(f"  eigenvalues : min={min_eval:.3e}, max={max_eval:.3e}")
    print(f"  whiten_basis: {whiten_basis}")

    z0 = eigen_map.z_from_theta(theta0)
    if whiten_basis:
        lr_vec = np.ones_like(z0)
        curvature_vec = np.ones_like(z0)
    else:
        lr_vec = 1.0 / (eigvals_kept + 1e-12)
        curvature_vec = eigvals_kept

    index_map = build_eigen_index_map(k_kept)
    loss_opt = lambda z: loss_fn(eigen_map.theta_from_z(z))
    theta0_opt = z0
    theta_space = "eigen"
    precond_meta = {
        "lr_vec": lr_vec,
        "whiten_basis": whiten_basis,
        "truncate_k": truncate_k,
        "truncate_by_eigval": truncate_by_eigval,
    }
else:
    index_map = build_index_map(inference_subspec, init_store, theta=theta0)
    lr_vec = 1.0 / (np.asarray(fim_diag) + 1e-12)
    curvature_vec = fim_diag
    loss_opt = loss_fn
    theta0_opt = theta0
    theta_space = "primitive"
    precond_meta = {"lr_vec": lr_vec}

print("FIM diag: min={:.3e}, max={:.3e}".format(float(jnp.min(fim_diag)), float(jnp.max(fim_diag))))
print("LR vec : min={:.3e}, max={:.3e}".format(float(jnp.min(lr_vec)), float(jnp.max(lr_vec))))

print("\nRefactored curvature and learning rates (via index_map):")
for entry in index_map["entries"]:
    name  = entry["name"]
    start = entry["start"]
    stop  = entry["stop"]
    shape = entry.get("shape", ())

    n = stop - start

    if n == 1:
        print(
            f"  {name:40s} : "
            f"curv={curvature_vec[start]:.3e}  lr={lr_vec[start]:.3e}"
        )
    else:
        print(f"  {name:40s} : shape={shape}")
        for i, (c, l) in enumerate(zip(curvature_vec[start:stop], lr_vec[start:stop])):
            print(
                f"    {name}[{i:02d}] : "
                f"curv={c:.3e}  lr={l:.3e}"
            )


print("Running preconditioned gradient descent...")
# Now run the gradient descent optimization
n_iter = 100
base_lr = 0.5
theta_final_opt, trace = run_shera_gd(
    loss_fn=loss_opt,
    theta0=theta0_opt,
    index_map=index_map,
    learning_rate=base_lr,
    lr_vec=lr_vec,
    num_steps=n_iter,
    runs_dir=DEFAULT_RESULTS_DIR,
    return_artifacts=False,
    theta_space=theta_space,
    curvature=curvature_vec,
    precond=precond_meta,
)
# trace carries the theta + loss trace used by build_signals
# _artifacts is meant for on-disk logging.

if use_eigen:
    # Map the final eigenmode coefficients back to pure parameters
    theta_final = eigen_map.theta_from_z(theta_final_opt)
else:
    theta_final = theta_final_opt

final_store = store_unpack_params(inference_subspec, theta_final, init_store)

# Collect GD outputs
final_psf = binder.model(strip_structural(final_store, structural_keys=binder.structural_keys()))

##################
# Print a Summary
##################
labels_by_key = map_labels_to_keys(
    infer_keys,
    fim_labels,
    store=init_store if use_eigen else None,
    index_map=None if use_eigen else index_map,
)

print("\n==============================")
if use_eigen:
    print("Eigenmode Gradient Descent Summary")
else:
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
if use_eigen:
    decoder = lambda z: store_unpack_params(
        inference_subspec,
        eigen_map.theta_from_z(z),
        init_store,
    ).refresh_derived(inference_spec)
else:
    decoder = lambda theta: store_unpack_params(
        inference_subspec,
        theta,
        init_store,
    ).refresh_derived(inference_spec)

signals = build_signals(
    trace,
    meta={},
    decoder=decoder,
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
