"""
Compare legacy vs refactor Shera model PSF + FIM.

Runs the legacy 3-plane model from work/legacy/AR-Basic_3Plane.py and the
refactor-era model from work/scratch/refactored_astrometry_retrieval.py,
then compares PSFs and FIMs.
"""

# Core imports
import os
import time
import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

from dluxshera.inference.optimization import (
    FIM,
    generate_fim_labels,
    loss_fn,
    construct_priors_from_dict,
    ModelParams,
    SheraThreePlaneParams,
)
from dluxshera.core.modeling import SheraThreePlane_Model
from dluxshera.utils.utils import (
    calculate_log_flux,
    nanrms,
    save_prior_info,
    load_prior_info,
)
from dluxshera.plot.plotting import (
    plot_psf_single,
)

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.spec import (
    build_forward_model_spec_from_config,
    make_inference_subspec,
    build_inference_spec_basic,
)
from dluxshera.params.store import ParameterStore
from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.inference.prior import PriorSpec
from dluxshera.inference.optimization import (
    generate_fim_labels_refactor,
    make_binder_nll_fn,
    fim_theta,
)
from dluxshera.inference.run_artifacts import build_index_map
from dluxshera.params.packing import pack_params

inferno = mpl.colormaps["inferno"]

inferno.set_bad("k", 0.5)

plt.rcParams["image.cmap"] = "inferno"
plt.rcParams["font.family"] = "serif"
plt.rcParams["image.origin"] = "lower"
plt.rcParams["figure.dpi"] = 120


jax.config.update("jax_enable_x64", True)


################################
## Main Simulation Parameters ##
################################

# Start simulation timer
t0_script = time.time()

# Set up file paths
script_path = Path(__file__).resolve()
script_dir = script_path.parent
save_path = script_dir / "Results"
save_path.mkdir(parents=True, exist_ok=True)
script_name = script_path.stem
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Starting Simulation: {script_name} - {timestamp}")

# Plotting/Saving Settings
save_plots = True
N_saved_plots = 5  # Limit the number of plots that are saved, the first N plots will be saved
present_plots = False
print2console = True
save_FIM = False

save_results = True
results_savename = f"{script_name}_{timestamp}.xlsx"
overwrite_results = False

# Provide filenames here if you want to load in previously saved settings
data_param_filename = None
model_param_filename = None
prior_info_filename = None
save_params = True


# Observation Settings
N_observations = 1  # Number of repeated observations
exposure_time = 1800  # sec, total exposure time of the observation
frame_rate = 20  # Hz, observation frame rate
exposure_per_frame = 1 / frame_rate  # seconds
N_frames = frame_rate * exposure_time  # frames

# Image Noise Settings
add_shot_noise = False
sigma_read = 0  # e-/frame rms read noise

# Set up initial parameters
point_design = "shera_testbed"
default_params = SheraThreePlaneParams(point_design=point_design)  # Gets default parameters
default_params = default_params.set("rng_seed", 0)  # Specify a seed here
log_flux = calculate_log_flux(default_params.p1_diameter, default_params.bandwidth / 1000, exposure_time)
default_params = default_params.set("log_flux", log_flux)

# First define the initial parameters for the data
data_initial_params = ModelParams(
    {
        "pupil_npix": 256,
        "psf_npix": 256,
        "wavelength": 550.0,
        "n_wavelengths": 3,
        # Astrometry Settings
        "x_position": 0.0,
        "y_position": 0.0,
        "separation": 10,
        "position_angle": 90.0,
        "contrast": 3.0,
        # "log_flux": 6.78,
        "pixel_size": 6.5e-6,
        # Zernike Settings
        "m1_zernike_noll": jnp.arange(4, 12),
        "m1_zernike_amp": jnp.zeros(8),
        "m2_zernike_noll": jnp.arange(4, 12),
        "m2_zernike_amp": jnp.zeros(8),
        # Calibrated 1/f WFE Settings
        "m1_calibrated_power_law": 2.5,
        "m1_calibrated_amplitude": 0,
        "m2_calibrated_power_law": 2.5,
        "m2_calibrated_amplitude": 0,
        # Uncalibrated 1/f WFE Settings
        "m1_uncalibrated_power_law": 2.5,
        "m1_uncalibrated_amplitude": 0,
        "m2_uncalibrated_power_law": 2.5,
        "m2_uncalibrated_amplitude": 0,
    }
)

# Then define the initial parameters for the model
model_initial_params = ModelParams(
    {
        "pupil_npix": 256,
        "psf_npix": 256,
        "wavelength": 550.0,
        "n_wavelengths": 3,
        # Astrometry Settings
        "x_position": 0.0,
        "y_position": 0.0,
        "separation": 10.0,
        "position_angle": 90.0,
        "contrast": 3.0,
        # "log_flux": 6.78,
        "pixel_size": 6.5e-6,
        # Zernike Settings
        "m1_zernike_noll": jnp.arange(4, 12),
        "m1_zernike_amp": jnp.zeros(8),
        "m2_zernike_noll": jnp.arange(4, 12),
        "m2_zernike_amp": jnp.zeros(8),
        # Calibrated 1/f WFE Settings
        "m1_calibrated_power_law": 2.5,
        "m1_calibrated_amplitude": 0,
        "m2_calibrated_power_law": 2.5,
        "m2_calibrated_amplitude": 0,
    }
)

if prior_info_filename is not None:
    # Load prior_info from json file
    prior_info = load_prior_info(os.path.join(save_path, prior_info_filename))
else:
    # Set up priors, specifies distribution type, and sigma
    # The optimized model will be initially perturbed according to these priors
    prior_info = {
        "x_position": (1e-6, "Normal"),  # as
        "y_position": (1e-6, "Normal"),  # as
        "separation": (1e-6, "Normal"),  # as
        "position_angle": (1e-3, "Uniform"),  # deg
        "log_flux": (1e-6, "LogNormal"),  # log10(flux)
        "contrast": (1e-6, "LogNormal"),  # ratio (unitless)
        "psf_pixel_scale": (1e-6, "LogNormal"),  # as/pix
        "m1_aperture.coefficients": (1e-2, "Normal"),  # nm
        "m2_aperture.coefficients": (1e-2, "Normal"),  # nm
    }

if save_params:
    # Save parameters to a file, so we can load them later
    # Save data_params
    save_name = f"{script_name}_DataParams_{timestamp}.json"
    data_initial_params.to_json(os.path.join(save_path, save_name))
    # Save initial_model_params
    save_name = f"{script_name}_ModelParams_{timestamp}.json"
    model_initial_params.to_json(os.path.join(save_path, save_name))
    # Save prior_info
    save_name = f"{script_name}_PriorInfo_{timestamp}.json"
    save_prior_info(prior_info, os.path.join(save_path, save_name))
    # prior_info_loaded = load_prior_info("priors.json")

# Optimization Settings
n_iter = 100
lr = 0.5


# Legacy aliasing to mirror original script expectations
onp = np
np = jnp

# Define the parameters to solve for
optimisers = {
    "separation": None,
    "position_angle": None,
    "x_position": None,
    "y_position": None,
    "log_flux": None,
    "contrast": None,
    "psf_pixel_scale": None,
    "m1_aperture.coefficients": None,
    # "m2_aperture.coefficients": None,
}
params = list(optimisers.keys())


######################
## Simulation Start ##
######################

# Start the simulation(s)
t0_simulation = time.time()
rng_key = jr.PRNGKey(default_params.rng_seed)
path_map = default_params.get_param_path_map()
inv_path_map = {v: k for k, v in path_map.items()}
row_counter = 1
obs_digits = len(str(N_observations))

# Create the Data model
data_params = data_initial_params.inject(default_params)
data_model = SheraThreePlane_Model(data_params)

# Create the model
initial_model_params = model_initial_params.inject(default_params)
model = SheraThreePlane_Model(initial_model_params)

# Model the Data PSF
data_psf = data_model.model()
if save_params:
    data_saved_params = data_model.extract_params()
    save_name = f"{script_name}_DataParams_{timestamp}.json"
    data_saved_params.to_json(os.path.join(save_path, save_name))


# Calculate priors centered on current values
prior_info = {
    k: {"mean": model.get(k if k not in path_map else path_map[k]), "sigma": v[0], "dist": v[1]}
    for k, v in prior_info.items()
}
priors = construct_priors_from_dict(prior_info)

# Examine the Model
m1_mask = model.m1_aperture.transmission
m1_nanmask = np.where(m1_mask, m1_mask, np.nan)
m2_mask = model.m2_aperture.transmission
m2_nanmask = np.where(m2_mask, m2_mask, np.nan)
model_psf = model.model()
pupil_extent_mm = model.diameter * 1e3 / 2 * np.array([-1, 1, -1, 1])
m2_extent_mm = model.p2_diameter * 1e3 / 2 * np.array([-1, 1, -1, 1])
psf_extent_as = model.psf_npixels * model.psf_pixel_scale / 2 * np.array([-1, 1, -1, 1])

# Make a plot of the Data PSF
if save_plots:
    plot_name = "DataPSF"
    save_name = f"{script_name}_{plot_name}_{timestamp}.png"
    plot_psf_single(
        psf=data_psf,
        extent=psf_extent_as,
        title="Data PSF",
        cmap=inferno,
        normalise=False,
        stretch="sqrt",
        cbar_label="Photons",
        save_path=os.path.join(save_path, save_name),
        show=present_plots,
    )

# === Calculate the Fisher Information Matrix ===
print("\nCalculating Fisher Information Matrix...")
fim = FIM(
    model,  # your model pytree
    params,  # list of parameters you're solving for
    loss_fn,  # your log likelihood (negative)
    model_psf,
    model_psf,  # arguments: model output and noise variance
)
print("FIM shape:", fim.shape)
# === Plot the Fisher Information Matrix ===
fim_labels = generate_fim_labels(params, initial_model_params)

legacy_psf = onp.asarray(model_psf)
legacy_data_psf = onp.asarray(data_psf)
legacy_fim = onp.asarray(fim)
legacy_labels = fim_labels


# Restore numpy alias for refactor block
np = onp


##########################
# Start Building the model
##########################
print("Starting Refactor Simulation...")
print("Creating Config, Spec, Store, and Binder...")

# Start simulation timer
t0_script_refactor = time.time()

rng_seed = 42
add_noise = False

rng_key = jr.PRNGKey(rng_seed)

# Start with a pre-defined config
cfg = SHERA_TESTBED_CONFIG
# Use the config to set Zernike Noll indices
# The Noll indices are a 'structural' parameter of the model,
# this setting influences the size of the basis, and
# generally shouldn't be changed after creation.
cfg = cfg.replace(primary_noll_indices=tuple(range(4, 12)), secondary_noll_indices=tuple(range(4, 12)))

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
    if np.min(data) > 100:  # Use Gaussian Approximation
        data = np.sqrt(data) * jr.normal(split_key, data.shape) + data
    else:  # Add Poisson shot noise
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

# Choose which loss function to use
loss_fn_refactor = nll_loss_fn

# Calculate True + Initial Loss values
theta_true = pack_params(inference_subspec, forward_truth_store)

print("Computing Fisher Information Matrix (FIM) for preconditioning...")
F = fim_theta(loss_fn_refactor, theta_true)

refactor_psf = np.asarray(data)
refactor_fim = np.asarray(F)
refactor_labels = fim_labels
refactor_theta_true = np.asarray(theta_true)
refactor_index_map = index_map


######################
# Comparison & output
######################

comparison_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
comparison_dir = Path("Results") / f"compare_legacy_refactor_{comparison_timestamp}"
comparison_dir.mkdir(parents=True, exist_ok=True)

print("\n==== PSF comparison ====")
print(f"Legacy PSF shape: {legacy_psf.shape}, dtype={legacy_psf.dtype}")
print(f"Refactor PSF shape: {refactor_psf.shape}, dtype={refactor_psf.dtype}")
print(f"Legacy PSF min/max/sum/peak: {legacy_psf.min():.6g} / {legacy_psf.max():.6g} / {legacy_psf.sum():.6g} / {legacy_psf.max():.6g}")
print(f"Refactor PSF min/max/sum/peak: {refactor_psf.min():.6g} / {refactor_psf.max():.6g} / {refactor_psf.sum():.6g} / {refactor_psf.max():.6g}")

abs_diff = legacy_psf - refactor_psf
ref_norm = np.linalg.norm(refactor_psf)
rel_l2 = np.linalg.norm(abs_diff) / ref_norm if ref_norm != 0 else np.nan
max_abs = np.max(np.abs(abs_diff))
max_rel = np.max(np.abs(abs_diff) / np.maximum(np.abs(refactor_psf), 1e-12))

print(f"PSF rel L2 diff: {rel_l2:.6g}")
print(f"PSF max abs diff: {max_abs:.6g}")
print(f"PSF max rel diff: {max_rel:.6g}")

np.savez(
    comparison_dir / "psf_comparison.npz",
    psf_legacy=legacy_psf,
    psf_refactor=refactor_psf,
    psf_diff=abs_diff,
)

# Save a simple 3-panel PNG
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes = axes.flatten()
common_vmin = min(legacy_psf.min(), refactor_psf.min())
common_vmax = max(legacy_psf.max(), refactor_psf.max())

im0 = axes[0].imshow(legacy_psf, vmin=common_vmin, vmax=common_vmax)
axes[0].set_title("Legacy PSF")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(refactor_psf, vmin=common_vmin, vmax=common_vmax)
axes[1].set_title("Refactor PSF")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(abs_diff, cmap="seismic")
axes[2].set_title("Legacy - Refactor")
fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.tight_layout()
fig.savefig(comparison_dir / "psf_comparison.png", dpi=200)
plt.close(fig)


print("\n==== FIM comparison ====")
np.savez(
    comparison_dir / "fim_comparison.npz",
    fim_legacy=legacy_fim,
    labels_legacy=np.array(legacy_labels, dtype=object),
    fim_refactor=refactor_fim,
    labels_refactor=np.array(refactor_labels, dtype=object),
)

legacy_label_to_index = {label: idx for idx, label in enumerate(legacy_labels)}
refactor_label_to_index = {label: idx for idx, label in enumerate(refactor_labels)}

mapped_pairs = [
    ("separation", "binary.separation_as"),
    ("position_angle", "binary.position_angle_deg"),
    ("x_position", "binary.x_position_as"),
    ("y_position", "binary.y_position_as"),
    ("log_flux", "binary.log_flux_total"),
    ("contrast", "binary.contrast"),
    ("psf_pixel_scale", "system.plate_scale_as_per_pix"),
]

matched_rows = []
ratios = []

for legacy_key, refactor_key in mapped_pairs:
    if legacy_key in legacy_label_to_index and refactor_key in refactor_label_to_index:
        li = legacy_label_to_index[legacy_key]
        ri = refactor_label_to_index[refactor_key]
        legacy_val = legacy_fim[li, li]
        refactor_val = refactor_fim[ri, ri]
        ratio = refactor_val / legacy_val if legacy_val != 0 else np.nan
        matched_rows.append((legacy_key, legacy_val, refactor_val, ratio))
        ratios.append(ratio)
    else:
        matched_rows.append((f"{legacy_key} -> {refactor_key}", np.nan, np.nan, np.nan))

# Map Zernike coefficients by index
legacy_zernike_labels = [label for label in legacy_labels if "m1_aperture.coefficients[" in label]
refactor_zernike_labels = [label for label in refactor_labels if "primary.zernike_coeffs_nm[" in label]

legacy_zernike_labels_sorted = sorted(
    legacy_zernike_labels,
    key=lambda x: int(x.split("[")[-1].split("]")[0]),
)
refactor_zernike_labels_sorted = sorted(
    refactor_zernike_labels,
    key=lambda x: int(x.split("[")[-1].split("]")[0]),
)

for legacy_label, refactor_label in zip(legacy_zernike_labels_sorted, refactor_zernike_labels_sorted):
    li = legacy_label_to_index[legacy_label]
    ri = refactor_label_to_index[refactor_label]
    legacy_val = legacy_fim[li, li]
    refactor_val = refactor_fim[ri, ri]
    ratio = refactor_val / legacy_val if legacy_val != 0 else np.nan
    matched_rows.append((legacy_label, legacy_val, refactor_val, ratio))
    ratios.append(ratio)

print("name | legacy_diag | refactor_diag | ratio(ref/legacy)")
for name, legacy_val, refactor_val, ratio in matched_rows:
    print(f"{name} | {legacy_val:.6g} | {refactor_val:.6g} | {ratio:.6g}")

ratios = np.array([r for r in ratios if np.isfinite(r)])
if ratios.size > 0:
    print(
        "FIM diag ratio stats: mean={:.6g}, median={:.6g}, p10={:.6g}, p90={:.6g}".format(
            np.mean(ratios),
            np.median(ratios),
            np.percentile(ratios, 10),
            np.percentile(ratios, 90),
        )
    )
else:
    print("FIM diag ratio stats: no finite ratios available")

print("\n==== Summary ====")
print(f"PSF rel L2 diff: {rel_l2:.6g}")
print(f"PSF max abs diff: {max_abs:.6g}")
print("FIM diag ratio stats: see above")
print(f"Results written to: {comparison_dir}")

