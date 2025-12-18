# In this script, we will perform an eigenvalue analysis on one of our optical models.
# We are looking to explore how the model behaves, look for degeneracies, understand
# which directions are most sensitive, etc.

# Core jax
import jax
import jax.numpy as np
import jax.random as jr

# Optimisation
import zodiax as zdx
import optax

# Optics
# import dLux as dl
# import dLux.layers as dll
# import dLux.utils as dlu
from src.dluxshera.inference.optimization import loss_fn, construct_priors_from_dict
from src.dluxshera.inference.optimization import ModelParams, SheraThreePlaneParams
from src.dluxshera.inference.optimization import FIM, generate_fim_labels, pack_params, build_basis
from src.dluxshera.core.modeling import SheraThreePlane_Model
from src.dluxshera.utils.utils import calculate_log_flux, set_array, nanrms, save_prior_info, load_prior_info

# Plotting/visualisation
import numpy as onp  # original numpy
import matplotlib as mpl
import matplotlib.pyplot as plt
import time, datetime, os
import scipy.io

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


jax.config.update("jax_enable_x64", True)


################################
## Main Simulation Parameters ##
################################

# Start simulation timer
t0_script = time.time()
save_path = os.path.join(os.getcwd(), "..", "Results")
script_name = os.path.splitext((os.path.basename(__file__)))[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Starting Simulation: {script_name} - {timestamp}")



# Plotting/Saving Settings
save_plots = True # True / False
N_saved_obs = 5
present_plots = False # True / False
print2console = True
# plot_FIM = True
save_FIM = False

save_results = True
results_savename = f"{script_name}_{timestamp}.xlsx"
overwrite_results = False

# Provide filenames here if you want to load in previously saved settings
data_param_filename = None
model_param_filename = None
prior_info_filename = None
# prior_info_filename = "AR-3P_PriorInfo_20250718_134125.json"
save_params = False


# Eigenmode Options
use_eigen = True # Enables re-parameterization
truncate_k = None     # Integer or None, optionally truncates to top k eigenmodes
whiten_basis = False # Bool, optionally scales each eigenvector by 1/sqrt(lambda)


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
point_design = 'shera_testbed'
default_params = SheraThreePlaneParams(point_design=point_design)  # Gets default parameters
default_params = default_params.set("rng_seed", 0)  # Specify a seed here
log_flux = calculate_log_flux(default_params.p1_diameter, default_params.bandwidth / 1000, exposure_time)
default_params = default_params.set('log_flux', log_flux)

# First define the initial parameters for the data
data_initial_params = ModelParams({
    "pupil_npix": 256,
    "psf_npix": 256,
    "wavelength": 550.,
    "n_wavelengths": 3,
    # Astrometry Settings
    "x_position":       np.asarray(0.0),
    "y_position":       np.asarray(0.0),
    "separation":       np.asarray(10.),
    "position_angle":   np.asarray(90.0),
    "contrast":         np.asarray(0.3),
    # "log_flux": 6.78,
    "pixel_size": 6.5e-6,

    # Zernike Settings
    "m1_zernike_noll": np.arange(4, 12),
    "m1_zernike_amp": np.zeros(8),
    "m2_zernike_noll": np.arange(4, 12),
    "m2_zernike_amp": np.zeros(8),

    # Calibrated 1/f WFE Settings
    "m1_calibrated_power_law": 2.5,
    "m1_calibrated_amplitude": 0,
    "m2_calibrated_power_law": 2.5,
    "m2_calibrated_amplitude": 0,

    # Uncalibrated 1/f WFE Settings
    "m1_uncalibrated_power_law": 2.5,
    "m1_uncalibrated_amplitude": 0,
    "m2_uncalibrated_power_law": 2.5,
    "m2_uncalibrated_amplitude": 0
})

# Then define the initial parameters for the model
model_initial_params = ModelParams({
    "pupil_npix": 256,
    "psf_npix": 256,
    "wavelength": 550.,
    "n_wavelengths": 3,
    # Astrometry Settings
    "x_position":       np.asarray(0.0),
    "y_position":       np.asarray(0.0),
    "separation":       np.asarray(10.),
    "position_angle":   np.asarray(90.0),
    "contrast":         np.asarray(0.3),
    # "log_flux": 6.78,
    "pixel_size": 6.5e-6,

    # Zernike Settings
    "m1_zernike_noll": np.arange(4, 12),
    "m1_zernike_amp": np.zeros(8),
    "m2_zernike_noll": np.arange(4, 12),
    "m2_zernike_amp": np.zeros(8),

    # Calibrated 1/f WFE Settings
    "m1_calibrated_power_law": 2.5,
    "m1_calibrated_amplitude": 0,
    "m2_calibrated_power_law": 2.5,
    "m2_calibrated_amplitude": 0,
})



if prior_info_filename is not None:
    # Load prior_info from json file
    prior_info = load_prior_info(os.path.join("../Results", prior_info_filename))
else:
    # Set up priors, specifies distribution type, and sigma
    # The optimized model will be initially perturbed according to these priors
    prior_info = {
        'x_position': (1e-2, "Normal"),               # as
        'y_position': (1e-2, "Normal"),               # as
        'separation': (1e-3, "Normal"),               # as
        'position_angle': (1e-3, "Uniform"),          # deg
        'log_flux': (1e-3, "LogNormal"),              # log10(flux)
        'contrast': (1e-3, "LogNormal"),              # ratio (unitless)
        'psf_pixel_scale': (1e-3, "LogNormal"),       # as/pix
        'm1_aperture.coefficients': (1, "Normal"), # nm
        'm2_aperture.coefficients': (1, "Normal")  # nm
    }

if save_params:
    # Save parameters to a file, so we can load them later
    # Save data_params
    save_name = f"{script_name}_DataParams_{timestamp}.json"
    data_initial_params.to_json(os.path.join("../Results", save_name))
    # Save initial_model_params
    save_name = f"{script_name}_ModelParams_{timestamp}.json"
    model_initial_params.to_json(os.path.join("../Results", save_name))
    # Save prior_info
    save_name = f"{script_name}_PriorInfo_{timestamp}.json"
    save_prior_info(prior_info, os.path.join("../Results", save_name))
    # prior_info_loaded = load_prior_info("priors.json")

# Optimization Settings
n_iter = 100
# Define the parameters to solve for
lr = 0.5
# opt = optax.sgd(lr,0)
opt = optax.sgd(lr)
optimiser_label = "optax.sgd"
optimisers = {
    "separation": opt,
    "position_angle": opt,
    "x_position": opt,
    "y_position": opt,
    "log_flux": opt,
    "contrast": opt,
    "psf_pixel_scale": opt,
    "m1_aperture.coefficients": opt,
    "m2_aperture.coefficients": opt,
}
params = list(optimisers.keys())


######################
## Simulation Start ##
######################

# Start the simulation
t0_simulation = time.time()
rng_key = jr.PRNGKey(default_params.rng_seed)
path_map = default_params.get_param_path_map()
inv_path_map = {v: k for k, v in path_map.items()}
row_counter = 1
obs_digits = len(str(N_observations))

# Create the Data model
data_params = data_initial_params.inject(default_params)
data_model = SheraThreePlane_Model(data_params)
# data_model = SheraThreePlane_Model(default_params)
data_model = set_array(data_model, None)

# Create the model
initial_model_params = model_initial_params.inject(default_params)
model = SheraThreePlane_Model(initial_model_params)
# model = SheraThreePlane_Model(default_params)
model = set_array(model, None)

# Model the Data PSF
data_psf = data_model.model()
# model_psf = model.model()

# fig = plt.figure(figsize=(5, 5))
# ax = plt.axes()
# plt.title("Data PSF")
# plt.xlabel("X (as)")
# psf_extent_as = model.psf_npixels * model.psf_pixel_scale / 2 * np.array([-1, 1, -1, 1])
# im = ax.imshow(data_psf, extent=psf_extent_as)
# cbar = fig.colorbar(im, cax=merge_cbar(ax))
# cbar.set_label("Photons")
# plt.tight_layout()
# plt.show(block=True)

# Calculate priors centered on current values
prior_info = {
    k: {
        "mean": model.get(k if k not in path_map else path_map[k]),
        "sigma": v[0],
        "dist": v[1]
    }
    for k, v in prior_info.items()
}
priors = construct_priors_from_dict(prior_info)
# priors are used to perturb the model

# Examine the Model
m1_mask = model.m1_aperture.transmission
m1_nanmask = np.where(m1_mask, m1_mask, np.nan)
m2_mask = model.m2_aperture.transmission
m2_nanmask = np.where(m2_mask, m2_mask, np.nan)
model_psf = model.model()
pupil_extent_mm = model.diameter * 1e3 / 2 * np.array([-1, 1, -1, 1])
m2_extent_mm = model.p2_diameter * 1e3 / 2 * np.array([-1, 1, -1, 1])
psf_extent_as = model.psf_npixels * model.psf_pixel_scale / 2 * np.array([-1, 1, -1, 1])



# === Calculate the Fisher Information Matrix ===
print("\nCalculating Fisher Information Matrix...")
fim = FIM(
    model,  # your model pytree
    params,  # list of parameters you're solving for
    loss_fn,  # your log likelihood (negative)
    model_psf, model_psf  # arguments: model output and noise variance
)
print("FIM shape:", fim.shape)
# === Plot the Fisher Information Matrix ===
fim_labels = generate_fim_labels(params, initial_model_params)
fim_log = np.log10(np.abs(fim) + 1e-20)
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(fim_log, cmap="viridis", vmin=4, vmax=14)
ax.set_title("Log10 Fisher Information Matrix")
ax.set_xlabel("Parameters")
ax.set_ylabel("Parameters")
ax.set_xticks(np.arange(len(fim_labels)))
ax.set_yticks(np.arange(len(fim_labels)))
ax.set_xticklabels(fim_labels, rotation=45, ha="right")
ax.set_yticklabels(fim_labels)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Log Information")
plt.tight_layout()
if save_plots:
    plot_name = "FIM"
    save_name = f"{script_name}_{plot_name}_{timestamp}.png"
    plt.savefig(os.path.join(save_path, save_name))
if present_plots:
    plt.show()
else:
    plt.close()

if save_FIM:
    # Save the FIM as a .mat file
    save_name = f"{script_name}_FIM_Data_{timestamp}.mat"
    scipy.io.savemat(os.path.join("../Results", save_name), {
        'FIM': onp.asarray(fim, dtype=onp.float64),  # NumPy 2D array
        'param_names': onp.array(fim_labels, dtype=object)  # List of parameter names (str)
    })



# === Eigenvalue Decomposition ===
eigvals, eigvecs = np.linalg.eigh(fim)

# Sort eigenvalues (and vectors) from largest to smallest
sorted_indices = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorted_indices]
eigvecs = eigvecs[:, sorted_indices]


plt.figure(figsize=(6,4))
plt.semilogy(eigvals, marker="o")
plt.title("FIM Eigenvalue Spectrum")
plt.xlabel("Eigenmode Index")
plt.ylabel("Eigenvalue (log scale)")
plt.grid(True)
plt.tight_layout()
if save_plots:
    plot_name = "EigenvalueSpectrum"
    save_name = f"{script_name}_{plot_name}_{timestamp}.png"
    plt.savefig(os.path.join(save_path, save_name))
if present_plots:
    plt.show()
else:
    plt.close()



# # === Plot All Eigenmodes ===
# max_eigval = eigvals[0]
# for i in range(len(eigvals)):
#     mode = eigvecs[:, i]
#     eigval = eigvals[i]
#     rel_strength = eigval / max_eigval if max_eigval != 0 else 0
#
#     plt.figure(figsize=(8, 4))
#     plt.bar(fim_labels, mode)
#     plt.title(f"Eigenmode {i} | λ = {eigval:.2e} | Rel = {rel_strength:.2e}")
#     plt.xticks(rotation=45, ha="right")
#     plt.ylabel("Eigenvector Weight")
#     plt.tight_layout()
#
#     if save_plots:
#         plot_name = f"Eigenmode_{i:03d}"
#         save_name = f"{script_name}_{plot_name}_{timestamp}.png"
#         plt.savefig(os.path.join(save_path, save_name))
#     if present_plots:
#         plt.show()
#     else:
#         plt.close()



# Build reference flat vector and labels
p_ref, fim_labels = pack_params(initial_model_params, params, initial_model_params)
# Build the eigenmode basis
B = build_basis(eigvecs, eigvals, truncate=truncate_k, whiten=whiten_basis)



###############################################################


# Record true values
true_vals = {param: data_model.get(param) for param in params}
true_vals["raw_fluxes"] = data_model.raw_fluxes
true_vals["m1_zernike_opd"] = data_model.m1_aperture.eval_basis()
true_vals["m1_zernike_opd_rms_nm"] = 1e9 * nanrms(true_vals["m1_zernike_opd"][m1_mask.astype(bool)])
true_vals["m2_zernike_opd"] = data_model.m2_aperture.eval_basis()
true_vals["m2_zernike_opd_rms_nm"] = 1e9 * nanrms(true_vals["m2_zernike_opd"][m2_mask.astype(bool)])
true_vals["m1_total_opd"] = true_vals["m1_zernike_opd"] + data_model.m1_wfe.opd
true_vals["m1_total_opd_rms_nm"] = 1e9 * nanrms(true_vals["m1_total_opd"][m1_mask.astype(bool)])
true_vals["m2_total_opd"] = true_vals["m2_zernike_opd"] + data_model.m2_wfe.opd
true_vals["m2_total_opd_rms_nm"] = 1e9 * nanrms(true_vals["m2_total_opd"][m2_mask.astype(bool)])


# Take the value and gradient transformation of the loss function
val_grad_fn = zdx.filter_value_and_grad(params)(loss_fn)


# Start the Observation Loop
obs_keys = jr.split(rng_key, N_observations)  # One key per observation
for obs_i in range(N_observations):
    t0_obs = time.time()  # Start observation timer
    obs_key = obs_keys[obs_i]

    # Add Shot Noise to the PSF
    if add_shot_noise:
        obs_key, subkey = jr.split(obs_key)
        if exposure_time >= 7200: # If exposure is > 2 hours
            photons = np.sqrt(data_psf) * jr.normal(subkey, data_psf.shape) + data_psf # Gaussian Approximation
        else:
            photons = jr.poisson(subkey, data_psf)  # Add photon noise
    else:
        photons = data_psf

    # Add Read Noise
    obs_key, subkey = jr.split(obs_key)
    read_noise = sigma_read * np.sqrt(N_frames) * jr.normal(subkey, data_psf.shape)

    # Combine to get fake data
    data = photons + read_noise

    # Calculate the image variance (equivalent to shot noise + read noise)
    var = np.maximum(data + (sigma_read * np.sqrt(N_frames)) ** 2, (sigma_read * np.sqrt(N_frames)) ** 2)

    # fig = plt.figure(figsize=(5, 5))
    # ax = plt.axes()
    # plt.title("Data")
    # plt.xlabel("X (as)")
    # im = ax.imshow(data, extent=psf_extent_as)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("Photons")
    # plt.tight_layout()
    # plt.show(block=True)

    # data_model is already initialized w/ the true values
    # Get the true loss and the gradients
    true_loss, true_grads = val_grad_fn(data_model, data, var)

    # if print2console: # Print Summary of Inputs
    #     print("\nAstrometry Retrieval Initial Inputs:")
    #     print("Starting RNG Seed: %d" % default_params.rng_seed)
    #     print("Source X, Y Position: %.3f as, %.3f as" % (true_vals["x_position"], true_vals["y_position"]))
    #     print("Source Angle: %.3f deg" % true_vals["position_angle"])
    #     print("Source Separation: %.3f as" % true_vals["separation"])
    #     print("Source Log Flux: %.3f" % true_vals["log_flux"])
    #     print("Source Contrast: %.3f A:B" % true_vals["contrast"])
    #     print("Detector Platescale: %.3f as/pix" % true_vals["psf_pixel_scale"])
    #     print("Data Includes %.3f nm rms of calibrated 1/f^%.2f noise" % (initial_model_params.m1_calibrated_amplitude*1e9, initial_model_params.m1_calibrated_power_law))
    #     print("Data Includes %.3f nm rms of uncalibrated 1/f^%.2f noise" % (data_params.m1_uncalibrated_amplitude*1e9, data_params.m1_uncalibrated_power_law))
    #     print("Data Includes %d Zernikes on M1: %s @%.2f nm rms" %
    #           (data_params.m1_zernike_noll.size, ", ".join(f"Z{z}" for z in data_params.m1_zernike_noll), nanrms(data_params.m1_zernike_amp)) )
    #     print("Data Includes %d Zernikes on M2: %s @%.2f nm rms" %
    #           (data_params.m2_zernike_noll.size, ", ".join(f"Z{z}" for z in data_params.m2_zernike_noll), nanrms(data_params.m2_zernike_amp)) )
    #     # print("Data Includes %.3f as of jitter" % jitter_amplitude)
    #     print("Data Includes Shot Noise: %s" % add_shot_noise)
    #     print("Data Includes Read Noise @ %.2f e- per frame" % sigma_read)
    #     # print("Model Fits for %d Zernikes on M1: Z%d - Z%d" % (initial_model_params.m1_zernike_noll.size, np.min(initial_model_params.m1_zernike_noll), np.max(initial_model_params.m1_zernike_noll)))
    #     # print("Model Fits for %d Zernikes on M2: Z%d - Z%d" % (initial_model_params.m2_zernike_noll.size, np.min(initial_model_params.m2_zernike_noll), np.max(initial_model_params.m2_zernike_noll)))
    #     print("Model Fits for %d Zernikes on M1: %s" %
    #           (initial_model_params.m1_zernike_noll.size, ", ".join(f"Z{z}" for z in initial_model_params.m1_zernike_noll)) )
    #     if "m2_aperture.coefficients" in params:
    #         print("Model Fits for %d Zernikes on M2: %s" %
    #               (initial_model_params.m2_zernike_noll.size, ", ".join(f"Z{z}" for z in initial_model_params.m2_zernike_noll)) )
    #     else:
    #         print("Model does NOT Fit for M2 Zernikes")
    #     print("True Loss Value: %.5g" % true_loss)



    # # Generate a Plot of the Input System
    # fig = plt.figure(figsize=(15, 5))
    # ax = plt.subplot(2, 4, 1) # M1 Calibrated WFE
    # # data_model.m1_calibration.opd
    # plt.title("M1 Calibrated 1/f WFE: %.3f nm rms" % (initial_model_params.m1_calibrated_amplitude * 1e9))
    # plt.xlabel("X (mm)")
    # plt.ylabel("Y (mm)")
    # im = ax.imshow(1e9 * data_model.m1_calibration.opd * m1_nanmask, inferno, extent=pupil_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm")
    # ax = plt.subplot(2, 4, 2) # M1 Uncalibrated WFE
    # # data_model.m1_wfe.opd
    # plt.title("M1 Uncalibrated 1/f WFE: %.3f nm rms" % (data_params.m1_uncalibrated_amplitude * 1e9))
    # plt.xlabel("X (mm)")
    # im = ax.imshow(1e9 * data_model.m1_wfe.opd * m1_nanmask, inferno, extent=pupil_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm")
    # ax = plt.subplot(2, 4, 3) # M1 Zernike Basis
    # plt.title("M1 Zernike OPD: %.3f nm rms" % true_vals["m1_zernike_opd_rms_nm"])
    # plt.xlabel("X (mm)")
    # im = ax.imshow(1e9* true_vals["m1_zernike_opd"] * m1_nanmask, inferno, extent=pupil_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm")
    # ax = plt.subplot(2, 4, 4) # DP Mask
    # # data_model.dp.opd
    # plt.title("Diffractive Pupil OPD")
    # plt.xlabel("X (mm)")
    # im = ax.imshow(1e9 * data_model.dp.opd * m1_nanmask, inferno, extent=pupil_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm")
    # ax = plt.subplot(2, 4, 5) # M2 Calibrated WFE
    # # data_model.m2_calibration.opd
    # plt.title("M2 Calibrated 1/f WFE: %.3f nm rms" % (initial_model_params.m2_calibrated_amplitude * 1e9))
    # plt.xlabel("X (mm)")
    # plt.ylabel("Y (mm)")
    # im = ax.imshow(1e9 * data_model.m2_calibration.opd * m2_nanmask, inferno, extent=m2_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm")
    # ax = plt.subplot(2, 4, 6) # M2 Uncalibrated WFE
    # # data_model.m2_wfe.opd
    # plt.title("M2 Uncalibrated 1/f WFE: %.3f nm rms" % (data_params.m2_uncalibrated_amplitude * 1e9))
    # plt.xlabel("X (mm)")
    # im = ax.imshow(1e9 * data_model.m2_wfe.opd * m2_nanmask, inferno, extent=m2_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm")
    # ax = plt.subplot(2, 4, 7) # M2 Zernike Basis
    # plt.title("M2 Zernike OPD: %.3f nm rms" % true_vals["m2_zernike_opd_rms_nm"])
    # plt.xlabel("X (mm)")
    # im = ax.imshow(1e9* true_vals["m2_zernike_opd"] * m2_nanmask, inferno, extent=m2_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm")
    # ax = plt.subplot(2, 4, 8) # Data
    # plt.title("Data")
    # plt.xlabel("X (as)")
    # im = ax.imshow(data, extent=psf_extent_as)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("Photons")
    # plt.tight_layout()
    # if save_plots and obs_i < N_saved_obs:
    #     obs_digits = len(str(N_observations))
    #     plot_name = "DataInput"
    #     save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
    #     plt.savefig(os.path.join(save_path, save_name))
    # if present_plots:
    #     plt.show()
    # else:
    #     plt.close()

    # Draw perturbations from priors
    if print2console:
        print("\nDrawing perturbations from priors:")
    obs_key, *subkeys = jr.split(obs_key, len(priors) + 1)
    perturbations = {}
    initial_vals = {}
    for i, param in enumerate(params):
        sample = priors[param].sample(subkeys[i])

        # Temporary Override
        # if param == "m1_aperture.coefficients": sample = true_vals[param] + np.array(
        #     [0.702, 0.799, -1.74, -0.495, 1.23, -0.9, 0.524])
        # elif param == "separation": pass
        # elif param != "m2_aperture.coefficients": sample = true_vals[param]
        # else: sample = true_vals[param]
        # sample = true_vals[param]

        nominal = true_vals[param]
        if param == "m1_aperture.coefficients" and nominal.shape != sample.shape:
            delta = sample - nominal[:sample.shape[0]]
        else:
            delta = sample - nominal

        # nominal = true_vals[param]
        # delta = sample - nominal
        # delta = sample - nominal[:sample.shape[0]]
        perturbations[param] = delta
        initial_vals[param] = sample

        # Print the perturbations to the console
        if print2console:
            if np.ndim(sample) == 0:
                print(f"{param}: sample={sample:.3g}, nominal={nominal:.3g}, delta={delta:.3g}")
            else:
                sample_str = ", ".join(f"{v:.3g}" for v in np.ravel(sample))
                delta_str = ", ".join(f"{v:.3g}" for v in np.ravel(delta))
                print(f"{param}: sample=[{sample_str}], delta=[{delta_str}]")

    # Apply the perturbations to the model
    obs_model = set_array(model.add(params, list(perturbations.values())), params)
    # This is meant to pull the original values from model, and create a new obs_model to use for this observation,
    # essentially resetting the model after each observation

    # Record loss and grads using base model
    base_loss, base_grads = val_grad_fn(model, data, var)

    # Record initial values
    initial_loss, initial_grads = val_grad_fn(obs_model, data, var)

    # 1) Flatten gradients in *pure* parameter order, aligned with FIM
    g_true_pure, labels = pack_params(true_grads, params, initial_model_params)
    g_base_pure, _ = pack_params(base_grads, params, initial_model_params)
    g_init_pure, _ = pack_params(initial_grads, params, initial_model_params)

    # 3) Project to eigenmode space via chain rule (p = p_ref + B c ⇒ ∂L/∂c = B^T ∂L/∂p)
    g_true_eig = B.T @ g_true_pure
    g_base_eig = B.T @ g_base_pure
    g_init_eig = B.T @ g_init_pure


    # =========================
    # GRADIENT DIAGNOSTICS + PLOTS
    # =========================

    # --- small utilities (inline to keep things simple)
    def vnorm(v):
        return float(np.sqrt(np.sum(v ** 2) + 1e-30))

    def cos(a, b):
        return float(np.dot(a, b) / np.sqrt((np.sum(a * a) + 1e-30) * (np.sum(b * b) + 1e-30)))


    # --- basic norms & alignment
    print("\n[Grad norms | pure]")
    print(" ||g_true|| = %.3e   ||g_base|| = %.3e   ||g_init|| = %.3e" %
          (vnorm(g_true_pure), vnorm(g_base_pure), vnorm(g_init_pure)))

    print("[Cosine alignment | pure]")
    print("  cos(true,base)=%.3f  cos(true,init)=%.3f  cos(base,init)=%.3f" % (
        cos(g_true_pure, g_base_pure), cos(g_true_pure, g_init_pure), cos(g_base_pure, g_init_pure)))

    print("\n[Grad norms | eigen]")
    print(" ||g_true|| = %.3e   ||g_base|| = %.3e   ||g_init|| = %.3e" %
          (vnorm(g_true_eig), vnorm(g_base_eig), vnorm(g_init_eig)))

    print("[Cosine alignment | eigen]")
    print("  cos(true,base)=%.3f  cos(true,init)=%.3f  cos(base,init)=%.3f" % (
        cos(g_true_eig, g_base_eig), cos(g_true_eig, g_init_eig), cos(g_base_eig, g_init_eig)))

    # --- "are we pulling back toward truth or away?" at the initial (perturbed) point
    p_true_vec, _ = pack_params(data_model.extract_params(), params, initial_model_params)
    p_init_vec, _ = pack_params(obs_model.extract_params(), params, initial_model_params)
    delta_pi = p_init_vec - p_true_vec  # points from truth -> current
    proj = float(np.dot(g_init_pure, delta_pi))
    print("\n[Bias test @ initial point]")
    print("  cos(g_init, delta) = %.3f  (proj = %+ .3e)" % (cos(g_init_pure, delta_pi), proj))
    # Interpretation: gradient points uphill. If cos(g, delta) > 0, then -g (descent) pulls back toward truth.
    # If cos(g, delta) < 0, descent pushes further away (a bias red flag).

    # --- rank by absolute gradient (pure)
    abs_g = onp.abs(onp.asarray(g_init_pure))
    order = onp.argsort(-abs_g)
    topk = min(12, len(order))
    print("\n[Top |∂L/∂p| components @ initial]")
    for i in order[:topk]:
        print(f"  {i:2d}  {labels[i]:35s}  ∂L/∂p = {g_init_pure[i]:+.3e}")

    # --- scale-aware view: dimensionless gradient per-σ using FIM diagonal
    diag = onp.asarray(onp.diag(fim))
    sigma = 1.0 / onp.sqrt(onp.clip(diag, 1e-30, None))  # CRLB std per parameter (approx)
    g_init_per_sigma = onp.asarray(g_init_pure) * sigma  # "pull per 1σ step"
    order_ps = onp.argsort(-onp.abs(g_init_per_sigma))
    print("\n[Top |∂L/∂(p in 1σ)| components @ initial]")
    for i in order_ps[:topk]:
        print(f"  {i:2d}  {labels[i]:35s}  per-σ pull = {g_init_per_sigma[i]:+.3e}")

    # --- optional: natural-gradient direction (precondition by F^-1)
    #     (tells you what a well-conditioned step would do; good for diagnosing conditioning vs mismatch)
    damp = 1e-12 * onp.max(diag)
    g_nat = onp.linalg.solve(onp.asarray(fim) + damp * onp.eye(fim.shape[0]), onp.asarray(g_init_pure))
    print("\n[Natural-gradient check]")
    print("  ||F^{-1} g_init|| = %.3e   cos(g_init, F^{-1} g_init) = %.3f"
          % (onp.linalg.norm(g_nat), cos(g_init_pure, g_nat)))

    # ----------------
    # PLOTS
    # ----------------

    # 1) Pure-basis components (signed log helps dynamic range)
    fig, ax = plt.subplots(figsize=(10, 4))
    signed_log = onp.sign(g_init_pure) * onp.log10(onp.abs(g_init_pure) + 1e-20)
    ax.bar(onp.arange(len(labels)), signed_log)
    ax.set_title("Signed log10 |∂L/∂p|  (pure basis, initial)")
    ax.set_ylabel("sign(g) · log10|g|")
    ax.set_xticks(onp.arange(len(labels)));
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_path, f"{script_name}_Grad_Pure_SignedLog_{timestamp}.png"))
    plt.close()

    # 2) Eigen-basis components (line plot)
    m = onp.arange(B.shape[1])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m, onp.asarray(g_init_eig), marker="o", ms=3)
    ax.set_title("∂L/∂c  (eigenmode basis, initial)")
    ax.set_xlabel("Eigenmode index");
    ax.set_ylabel("∂L/∂c")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_path, f"{script_name}_Grad_Eigen_{timestamp}.png"))
    plt.close()


    # 3) Cumulative gradient energy across eigenmodes
    def cum_energy(v):
        v = onp.asarray(v);
        num = onp.cumsum(v * v);
        den = max(onp.sum(v * v), 1e-30)
        return num / den


    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(m, cum_energy(g_true_eig), label="True")
    ax.plot(m, cum_energy(g_base_eig), label="Base")
    ax.plot(m, cum_energy(g_init_eig), label="Initial")
    ax.set_title("Cumulative gradient energy (eigenmodes)")
    ax.set_xlabel("Eigenmode index");
    ax.set_ylabel("Fraction of ||g||²");
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_path, f"{script_name}_GradEnergyCum_{timestamp}.png"))
    plt.close()

    # 4) Predicted 1D loss slices (local quadratic) along: (a) -grad, (b) top-3 eigenmodes
    #    Using ΔL(α) ≈ α g·d + 0.5 α² dᵀ F d, with F≈FIM. This avoids rebuilding models.
    alphas = onp.linspace(-3.0, 3.0, 121)  # in "σ-units" per-direction (we'll scale to that)
    F = onp.asarray(fim)


    def slice_curve(g, F, d):
        # scale α to "σ units" along d, where σ_d² ≈ dᵀ F^{-1} d
        w = onp.linalg.solve(F + 1e-12 * onp.eye(F.shape[0]), d)
        sigma_d = onp.sqrt(max(onp.dot(d, w), 1e-30))
        a = alphas * sigma_d
        g_dot = float(onp.dot(g, d))
        curv = float(onp.dot(d, F @ d))
        dL = a * g_dot + 0.5 * (a * a) * curv
        return a / sigma_d, dL  # x-axis in σ units


    # (a) along -grad direction at initial
    d_g = onp.asarray(g_init_pure).copy()
    xg, yg = slice_curve(onp.asarray(g_init_pure), F, d_g)

    # (b) along top-3 eigenmodes (by |∂L/∂c| at initial)
    idx_modes = onp.argsort(-onp.abs(onp.asarray(g_init_eig)))[:3]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xg, yg, label="along -grad (init)")
    for j in idx_modes:
        d = onp.asarray(eigvecs[:, j])  # unwhitened basis
        xj, yj = slice_curve(onp.asarray(g_init_pure), F, d)
        ax.plot(xj, yj, label=f"mode {int(j)} (λ≈{eigvals[j]:.2e})")
    ax.set_title("Predicted ΔL slices (quadratic approx)")
    ax.set_xlabel("α (σ-units along direction)")
    ax.set_ylabel("ΔL (approx)")
    ax.legend()
    ax.axvline(0, color="k", lw=0.8, alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_path, f"{script_name}_LossSlices_{timestamp}.png"))
    plt.close()

    # --- per-parameter contributions to returning toward truth
    delta_pi = onp.asarray(p_init_vec - p_true_vec)
    g_init_np = onp.asarray(g_init_pure)

    contrib = g_init_np * delta_pi  # >0 means -g pulls back toward truth in that parameter
    order_c = onp.argsort(-onp.abs(contrib))
    topk = min(12, len(order_c))

    print("\n[Per-parameter g·delta contributions (help vs. hurt)]")
    tot = float(onp.sum(contrib))
    print(f"  total g·delta = {tot:+.3e}  (positive ⇒ descent reduces distance)")

    for i in order_c[:topk]:
        sign = "help" if contrib[i] > 0 else "hurt"
        print(f"  {i:2d}  {labels[i]:35s}  g·δ = {contrib[i]:+.3e}   [{sign}]")

    # simple bar plot (signed)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(onp.arange(len(labels)), contrib)
    ax.set_title("Per-parameter contribution g_i · δ_i  (positive ⇒ -g helps)")
    ax.set_ylabel("g·δ")
    ax.set_xticks(onp.arange(len(labels)));
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="k", lw=0.8, alpha=0.6)
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(save_path, f"{script_name}_Grad_Contrib_{timestamp}.png"))
    plt.close()

    t1_obs = time.time()
    print("Observation %d/%d finished in %.3f sec" % (obs_i + 1, N_observations, t1_obs - t0_obs))

t1_script = time.time()
print("Script finished in %.3f sec" % (t1_script-t0_script))
