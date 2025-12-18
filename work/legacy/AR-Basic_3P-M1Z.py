# In this script, we will perform phase retrieval with ∂Lux: recovering Zernike coefficients for an
# aberrated JNEXT telescope by gradient descent.

# Core jax
import jax
import jax.numpy as np
import jax.random as jr
from jax import tree

# Optimisation
import zodiax as zdx
import optax

# Optics
from src.dluxshera.inference.optimization import FIM, get_optimiser, get_lr_model, loss_fn, step_fn, construct_priors_from_dict
from src.dluxshera.inference.optimization import ModelParams, SheraThreePlaneParams
from src.dluxshera.core.modeling import SheraThreePlane_Model
from src.dluxshera.utils.utils import calculate_log_flux, set_array, nanrms
from src.dluxshera.plot.plotting import merge_cbar, plot_psf_comparison, plot_parameter_history

# Plotting/visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import time, datetime, os
import pandas as pd

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
N_saved_plots = 5 # Limit the number of plots that are saved, the first N plots will be saved
present_plots = False # True / False
save_results = True
print2console = True

# Observation Settings
N_observations = 25 # Number of repeated observations
exposure_time = 1800 # sec, total exposure time of the observation
frame_rate = 20 # Hz, observation frame rate
exposure_per_frame = 1/frame_rate  # seconds
N_frames = frame_rate*exposure_time  # frames

# Image Noise Settings
add_shot_noise = False
sigma_read = 0 # e-/frame rms read noise

# Set up initial parameters
point_design = 'shera_testbed'
default_params = SheraThreePlaneParams(point_design=point_design) # Gets default parameters
default_params = default_params.set("rng_seed", 2) # Specify a seed here
log_flux = calculate_log_flux(default_params.p1_diameter, default_params.bandwidth/1000, exposure_time)
default_params = default_params.set('log_flux', log_flux)

# First define the initial parameters for the data
data_initial_params = ModelParams({
    "pupil_npix": 256,
    "psf_npix": 256,
    "wavelength": 550.,
    "n_wavelengths": 3,
    # Astrometry Settings
    "x_position": 0.0,
    "y_position": 0.0,
    "separation": 10,
    "position_angle": 90.0,
    "contrast": 0.3,
    # "log_flux": 6.78,
    "pixel_size": 6.5e-6,

    # Zernike Settings
    "m1_zernike_noll": np.arange(4, 11),
    "m1_zernike_amp": np.zeros(7),
    "m2_zernike_noll": np.arange(4, 11),
    "m2_zernike_amp": np.zeros(7),

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
    "x_position": 0.0,
    "y_position": 0.0,
    "separation": 10,
    "position_angle": 90.0,
    "contrast": 0.3,
    # "log_flux": 6.78,
    "pixel_size": 6.5e-6,

    # Zernike Settings
    "m1_zernike_noll": np.arange(4, 11),
    "m1_zernike_amp": np.zeros(7),
    "m2_zernike_noll": None,
    "m2_zernike_amp": None,

    # Calibrated 1/f WFE Settings
    "m1_calibrated_power_law": 2.5,
    "m1_calibrated_amplitude": 0,
    "m2_calibrated_power_law": 2.5,
    "m2_calibrated_amplitude": 0,
})


# Save parameters to a file, so we can load them later
# Save data_params
save_name = f"{script_name}_DataParams_{timestamp}.json"
data_initial_params.to_json(os.path.join("../Results", save_name))
# Save initial_model_params
save_name = f"{script_name}_ModelParams_{timestamp}.json"
model_initial_params.to_json(os.path.join("../Results", save_name))

# Set up priors, specifies distribution sigma, and type
# The optimized model will be initially perturbed according to these priors
prior_info = {
    'x_position':               (0.1, "Normal"),
    'y_position':               (0.1, "Normal"),
    'separation':               (1e-6, "Normal"),
    'position_angle':           (1e-6, "Uniform"),
    'log_flux':                 (1e-6, "LogNormal"),
    'contrast':                 (1e-6, "LogNormal"),
    'psf_pixel_scale':          (1e-6, "LogNormal"),
    'm1_aperture.coefficients': (1, "Normal"),
    # 'm2_aperture.coefficients': (1e-6, "Normal")
}
# prior_info = {
#     'x_position': (30*1.5e-6, "Normal"),
#     'y_position': (30*1.5e-6, "Normal"),
#     'separation': (30*3e-6, "Normal"),
#     'position_angle': (30*1.8e-5, "Uniform"),
#     'log_flux': (30*1e-6, "LogNormal"),
#     'contrast': (30*2e-6, "LogNormal"),
#     'psf_pixel_scale': (30*1e-7, "LogNormal"),
#     'coefficients': (30*5e-4, "Normal"),
# }

# Optimization Settings
n_iter = 100
# Define the parameters to solve for
lr = 0.5
# opt = optax.sgd(lr,0)
opt = optax.sgd(lr, momentum=0.3)
optimiser_label = "optax.sgd-m0p3"
optimisers = {
    "separation": opt,
    "position_angle": opt,
    "x_position": opt,
    "y_position": opt,
    "log_flux": opt,
    "contrast": opt,
    "psf_pixel_scale": opt,
    "m1_aperture.coefficients": opt,
    # "m2_aperture.coefficients": opt,
}
params = list(optimisers.keys())


######################
## Simulation Start ##
######################

# Start the simulation(s)
t0_simulation = time.time() # Start simulation timer
rng_key = jr.PRNGKey(default_params.rng_seed)

# Create the Data model
data_params = data_initial_params.inject(default_params)
data_model = SheraThreePlane_Model(data_params)
# data_model = SheraThreePlane_Model(default_params)

# Create the model
initial_model_params = model_initial_params.inject(default_params)
model = SheraThreePlane_Model(initial_model_params)
# model = SheraThreePlane_Model(default_params)


# Model the Data PSF
data_psf = data_model.model()

# # Plot the Data PSF
# fig = plt.figure(figsize=(5, 5))
# ax = plt.axes()
# plt.title("Data PSF")
# plt.xlabel("X (as)")
# psf_extent_as = model.psf_npixels * model.psf_pixel_scale / 2 * np.array([-1, 1, -1, 1])
# im = ax.imshow(data_psf, extent=psf_extent_as)
# cbar = fig.colorbar(im, cax=merge_cbar(ax))
# cbar.set_label("Photons")
# plt.tight_layout()
# plt.show(block=False)


# Calculate priors centered on current values
path_map = default_params.get_param_path_map()
inv_path_map = {v: k for k, v in path_map.items()}
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
fim_labels = [] # Define proper axis labels for the FIM plot
for param in params:
    if param == "m1_aperture.coefficients":
        fim_labels.extend([f"M1 Z{n}" for n in initial_model_params.m1_zernike_noll])
    # elif param == "m2_aperture.coefficients":
    #     fim_labels.extend([f"M2 Z{n}" for n in initial_model_params.m2_zernike_noll])
    else:
        fim_labels.append(param)
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
    obs_digits = len(str(N_observations))
    plot_name = "FIM"
    save_name = f"{script_name}_{plot_name}_{timestamp}.png"
    plt.savefig(os.path.join(save_path, save_name))
if present_plots:
    plt.show()
else:
    plt.close()





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

# Get the learning rate model (variance estimate)
lr_model = get_lr_model(model, params, loss_fn, model_psf, model_psf)
param_std = tree.map(lambda x: np.sqrt(x), lr_model)
# print("lr_model:")
# for name, value in lr_model.params.items():
#     if np.ndim(value) == 0:  # Scalar case
#         std = np.sqrt(value)
#         print(f"{name}: var={value:.3e}, std={std:.3e}")
#     else:  # Array case
#         value_flat = np.ravel(value)
#         std_flat = np.sqrt(value_flat)
#         var_str = ", ".join(f"{v:.3e}" for v in value_flat)
#         std_str = ", ".join(f"{s:.3e}" for s in std_flat)
#         print(f"{name}:\n  var=[{var_str}]\n  std=[{std_str}]")

if print2console:
    print("Cramer Rao Lower Bound from FIM:")
    print("Source X, Y Position STD: %.3f uas, %.3f uas" % (
    param_std.get("x_position") * 1e6, param_std.get("y_position") * 1e6))
    print("Separation STD: %.3f uas" % (param_std.get("separation") * 1e6))
    print("Source Angle STD: %.3f as" % (param_std.get("position_angle") * 60 ** 2))
    print("Log Flux STD: %.3f" % param_std.get("log_flux"))
    print("Contrast STD: %.3f" % param_std.get("contrast"))
    print("Platescale STD: %.3f" % param_std.get("psf_pixel_scale"))
    m1_coeff_str = ", ".join(f"{coeff:.3f}" for coeff in param_std.get("m1_aperture.coefficients"))
    print("M1 Zernike Coefficients: [" + m1_coeff_str + "] nm")
    # print("M1 Zernike Coeff RMS: %.3f nm" % (nanrms(param_std.get("m1_aperture.coefficients"))))
    # m2_coeff_str = ", ".join(f"{coeff:.3f}" for coeff in param_std.get("m2_aperture.coefficients"))
    # print("M2 Zernike Coefficients: [" + m2_coeff_str + "] nm")
    # print("M2 Zernike Coeff RMS: %.3f nm" % (nanrms(param_std.get("m2_aperture.coefficients"))))


# Start the Observation Loop
obs_keys = jr.split(rng_key, N_observations)  # One key per observation
for obs_i in np.arange(N_observations):
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


    if print2console: # Print Summary of Inputs
        print("\nAstrometry Retrieval Initial Inputs:")
        print("Starting RNG Seed: %d" % default_params.rng_seed)
        print("Source X, Y Position: %.3f as, %.3f as" % (true_vals["x_position"], true_vals["y_position"]))
        print("Source Angle: %.3f deg" % true_vals["position_angle"])
        print("Source Separation: %.3f as" % true_vals["separation"])
        print("Source Log Flux: %.3f" % true_vals["log_flux"])
        print("Source Contrast: %.3f A:B" % true_vals["contrast"])
        print("Detector Platescale: %.3f as/pix" % true_vals["psf_pixel_scale"])
        print("Data Includes %.3f nm rms of calibrated 1/f^%.2f noise" % (initial_model_params.m1_calibrated_amplitude*1e9, initial_model_params.m1_calibrated_power_law))
        print("Data Includes %.3f nm rms of uncalibrated 1/f^%.2f noise" % (data_params.m1_uncalibrated_amplitude*1e9, data_params.m1_uncalibrated_power_law))
        # print("Data Includes %d Zernikes on M1: Z%d - Z%d @%.2f nm rms" % (data_params.m1_zernike_noll.size, np.min(data_params.m1_zernike_noll), np.max(data_params.m1_zernike_noll), nanrms(data_params.m1_zernike_amp)))
        # print("Data Includes %d Zernikes on M2: Z%d - Z%d @%.2f nm rms" % (data_params.m2_zernike_noll.size, np.min(data_params.m2_zernike_noll), np.max(data_params.m2_zernike_noll), nanrms(data_params.m2_zernike_amp)))
        print("Data Includes %d Zernikes on M1: %s @%.2f nm rms" %
              (data_params.m1_zernike_noll.size, ", ".join(f"Z{z}" for z in data_params.m1_zernike_noll), nanrms(data_params.m1_zernike_amp)) )
        print("Data Includes %d Zernikes on M2: %s @%.2f nm rms" %
              (data_params.m2_zernike_noll.size, ", ".join(f"Z{z}" for z in data_params.m2_zernike_noll), nanrms(data_params.m2_zernike_amp)) )
        # print("Data Includes %.3f as of jitter" % jitter_amplitude)
        print("Data Includes Shot Noise: %s" % add_shot_noise)
        print("Data Includes Read Noise @ %.2f e- per frame" % sigma_read)
        # print("Model Fits for %d Zernikes on M1: Z%d - Z%d" % (initial_model_params.m1_zernike_noll.size, np.min(initial_model_params.m1_zernike_noll), np.max(initial_model_params.m1_zernike_noll)))
        # print("Model Fits for %d Zernikes on M2: Z%d - Z%d" % (initial_model_params.m2_zernike_noll.size, np.min(initial_model_params.m2_zernike_noll), np.max(initial_model_params.m2_zernike_noll)))
        print("Model Fits for %d Zernikes on M1: %s" %
              (initial_model_params.m1_zernike_noll.size, ", ".join(f"Z{z}" for z in initial_model_params.m1_zernike_noll)) )
        # print("Model Fits for %d Zernikes on M2: %s" %
        #       (initial_model_params.m2_zernike_noll.size, ", ".join(f"Z{z}" for z in initial_model_params.m2_zernike_noll)) )
        print("True Loss Value: %.5g" % true_loss)



    # Generate a Plot of the Input System
    fig = plt.figure(figsize=(15, 5))
    ax = plt.subplot(2, 4, 1) # M1 Calibrated WFE
    # data_model.m1_calibration.opd
    plt.title("M1 Calibrated 1/f WFE: %.3f nm rms" % (initial_model_params.m1_calibrated_amplitude * 1e9))
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    im = ax.imshow(1e9 * data_model.m1_calibration.opd * m1_nanmask, inferno, extent=pupil_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm")
    ax = plt.subplot(2, 4, 2) # M1 Uncalibrated WFE
    # data_model.m1_wfe.opd
    plt.title("M1 Uncalibrated 1/f WFE: %.3f nm rms" % (data_params.m1_uncalibrated_amplitude * 1e9))
    plt.xlabel("X (mm)")
    im = ax.imshow(1e9 * data_model.m1_wfe.opd * m1_nanmask, inferno, extent=pupil_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm")
    ax = plt.subplot(2, 4, 3) # M1 Zernike Basis
    plt.title("M1 Zernike OPD: %.3f nm rms" % true_vals["m1_zernike_opd_rms_nm"])
    plt.xlabel("X (mm)")
    im = ax.imshow(1e9* true_vals["m1_zernike_opd"] * m1_nanmask, inferno, extent=pupil_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm")
    ax = plt.subplot(2, 4, 4) # DP Mask
    # data_model.dp.opd
    plt.title("Diffractive Pupil OPD")
    plt.xlabel("X (mm)")
    im = ax.imshow(1e9 * data_model.dp.opd * m1_nanmask, inferno, extent=pupil_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm")
    ax = plt.subplot(2, 4, 5) # M2 Calibrated WFE
    # data_model.m2_calibration.opd
    plt.title("M2 Calibrated 1/f WFE: %.3f nm rms" % (initial_model_params.m2_calibrated_amplitude * 1e9))
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    im = ax.imshow(1e9 * data_model.m2_calibration.opd * m2_nanmask, inferno, extent=m2_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm")
    ax = plt.subplot(2, 4, 6) # M2 Uncalibrated WFE
    # data_model.m2_wfe.opd
    plt.title("M2 Uncalibrated 1/f WFE: %.3f nm rms" % (data_params.m2_uncalibrated_amplitude * 1e9))
    plt.xlabel("X (mm)")
    im = ax.imshow(1e9 * data_model.m2_wfe.opd * m2_nanmask, inferno, extent=m2_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm")
    ax = plt.subplot(2, 4, 7) # M2 Zernike Basis
    plt.title("M2 Zernike OPD: %.3f nm rms" % true_vals["m2_zernike_opd_rms_nm"])
    plt.xlabel("X (mm)")
    im = ax.imshow(1e9* true_vals["m2_zernike_opd"] * m2_nanmask, inferno, extent=m2_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm")
    ax = plt.subplot(2, 4, 8) # Data
    plt.title("Data")
    plt.xlabel("X (as)")
    im = ax.imshow(data, extent=psf_extent_as)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("Photons")
    plt.tight_layout()
    if save_plots and obs_i < N_saved_plots:
        obs_digits = len(str(N_observations))
        plot_name = "DataInput"
        save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
        plt.savefig(os.path.join(save_path, save_name))
    if present_plots:
        plt.show()
    else:
        plt.close()






    # # Use the LR model to initialise an incorrect model
    # leaves = [np.array(lr_model.get(param)) for param in params]
    # keys = jr.split(obs_key, len(leaves))
    # perturbations = [30 * np.sqrt(leaf) * jr.normal(keys[i], leaf.shape) for i, leaf in enumerate(leaves)]
    # model = set_array(model.add(params, perturbations), params)
    # print("model perturbations:")
    # for i, param in enumerate(params):
    #     pert = perturbations[i]
    #     if np.ndim(pert) == 0:  # Scalar
    #         print(f"{param}: {pert:.3e}")
    #     else:  # Array
    #         pert_flat = np.ravel(pert)
    #         pert_str = ", ".join(f"{v:.3e}" for v in pert_flat)
    #         print(f"{param}: [{pert_str}]")


    # Draw perturbations from priors
    if print2console:
        print("\nDrawing perturbations from priors:")
    key, *subkeys = jr.split(obs_key, len(priors) + 1)
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
        delta = sample - nominal
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

    # Record initial values
    initial_loss, initial_grads = val_grad_fn(obs_model, data, var)
    initial_vals["raw_fluxes"] = obs_model.raw_fluxes
    initial_vals["m1_zernike_opd"] = obs_model.m1_aperture.eval_basis()
    initial_vals["m1_zernike_opd_rms_nm"] = 1e9* nanrms(initial_vals["m1_zernike_opd"][m1_mask.astype(bool)])
    # initial_vals["m2_zernike_opd"] = model.m2_aperture.eval_basis()
    # initial_vals["m2_zernike_opd_rms_nm"] = 1e9* nanrms(initial_vals["m2_zernike_opd"][m2_mask.astype(bool)])


    # Initialise our solver
    model_params, optim, state = get_optimiser(obs_model, optimisers)


    # Now we can Optimize
    t0_optim = time.time()
    # history = dict([(param, []) for param in params])
    history = {param: [initial_vals[param]] for param in params}
    losses, models_out = [initial_loss], [obs_model]
    for i in tqdm(range(n_iter)):
        loss, obs_model, model_params, state = step_fn(
            model_params, data, var, obs_model, lr_model, optim, state
        )

        losses.append(loss)
        models_out.append(obs_model)
        for param, value in model_params.params.items():
            history[param].append(value)


    #######################
    ## Post-Optimization ##
    #######################

    # Final Optimized Model Params
    # opt_params = initial_model_params.update_from_model(models_out[-1])
    opt_params = models_out[-1].extract_params()

    # # Save what we need to perform a parameter sweep around the final optimized parameters
    # # Save the Data + var arrays
    # save_name = f"{script_name}_Data+Var_{timestamp}_Obs{obs_i + 1:0{obs_digits}d}.pkl"
    # with open(os.path.join("../Results", save_name), "wb") as f:
    #     pickle.dump({"data": data, "var": var}, f)
    # # Save data_params
    # save_name = f"{script_name}_DataParams_{timestamp}_Obs{obs_i + 1:0{obs_digits}d}.json"
    # data_params.to_json(os.path.join("../Results", save_name))
    # # Save initial_model_params
    # save_name = f"{script_name}_InitialModelParams_{timestamp}_Obs{obs_i + 1:0{obs_digits}d}.json"
    # initial_model_params.to_json(os.path.join("../Results", save_name))
    # # Save opt_params
    # save_name = f"{script_name}_OptimizedModelParams_{timestamp}_Obs{obs_i + 1:0{obs_digits}d}.json"
    # opt_params.to_json(os.path.join("../Results", save_name))

    # params = SheraThreePlaneParams()
    # params.to_json("test.json")
    #
    # loaded = SheraThreePlaneParams.from_json("test.json")
    # print(type(loaded))


    # Record additional optimization histories
    history["raw_fluxes"] = [m.raw_fluxes for m in models_out]
    history["m1_zernike_opd"] = [m.m1_aperture.eval_basis() for m in models_out]
    history["m1_zernike_opd_rms_nm"] = [ 1e9 * nanrms(opd[m1_mask.astype(bool)]) for opd in history["m1_zernike_opd"] ]
    # history["m2_zernike_opd"] = [m.m2_aperture.eval_basis() for m in models_out]
    # history["m2_zernike_opd_rms_nm"] = [ 1e9 * nanrms(opd[m2_mask.astype(bool)]) for opd in history["m2_zernike_opd"] ]

    # history contains lists for each iteration, convert them all to arrays for easier operations
    history = {k: np.array(v) for k, v in history.items()}


    # If the data and model used different noll indices, we need to align their array sizes
    # The reason is that we want to compute residuals = history - true_vals, but their array sizes must align
    # First the primary mirror Zernike terms
    m1_noll = np.union1d(data_params.m1_zernike_noll, model_initial_params.m1_zernike_noll)  # Uniquely combines the model and data noll indices
    # m2_noll = np.union1d(data_params.m2_zernike_noll, model_initial_params.m2_zernike_noll)  # Used later to compare true coeffs to recovered coeffs
    m1_coeff_dict = dict(zip(data_params.m1_zernike_noll.tolist(), data_params.m1_zernike_amp))
    m2_coeff_dict = dict(zip(data_params.m2_zernike_noll.tolist(), data_params.m2_zernike_amp))
    if not np.array_equal(model_initial_params.m1_zernike_noll, data_params.m1_zernike_noll): # Checks if the noll indices are different
        # Expand the true_vals, fill in any missing coefficients with 0
        # m1_coeff_dict is defined earlier, it maps data noll indices to true coefficients
        true_vals["m1_aperture.coefficients"] = np.array([m1_coeff_dict.get(n, 0.0) for n in m1_noll])
        # Expand the recovered values in history
        # history["m1_aperture.coefficients"] is an array of size (n_iter, len(model_initial_params.m1_zernike_noll))
        aligned = np.full((n_iter, m1_noll.size), np.nan) # An expanded 2D array filled with nans
        model_indices = np.searchsorted(m1_noll, model_initial_params.m1_zernike_noll) # locates the model nolls within the expanded array
        # Fill in columns present in history array, remaining columns are left as nan
        aligned[:, model_indices] = history["m1_aperture.coefficients"]
        # Now replace the original array with the expanded array
        history["m1_aperture.coefficients"] = aligned
    # # Now for the secondary mirror Zernike terms
    # if not np.array_equal(model_initial_params.m2_zernike_noll, data_params.m2_zernike_noll):  # Checks if the noll indices are different
    #     # Expand the true_vals, fill in any missing coefficients with 0
    #     # m2_coeff_dict is defined earlier, it maps data noll indices to true coefficients
    #     true_vals["m2_aperture.coefficients"] = np.array([m2_coeff_dict.get(n, 0.0) for n in m2_noll])
    #     # Expand the recovered values in history
    #     # history["m2_aperture.coefficients"] is an array of size (n_iter, len(model_initial_params.m2_zernike_noll))
    #     aligned = np.full((n_iter, m2_noll.size), np.nan)  # An expanded 2D array filled with nans
    #     model_indices = np.searchsorted(m2_noll, model_initial_params.m2_zernike_noll) # locates the model nolls within the expanded array
    #     # Fill in columns present in history array, remaining columns are left as nan
    #     aligned[:, model_indices] = history["m2_aperture.coefficients"]
    #     # Now replace the original array with the expanded array
    #     history["m2_aperture.coefficients"] = aligned



    ### Compute residuals for each parameter
    residuals = {}
    for param in history.keys():
        residuals[param] = history[param] - true_vals[param]

    # # Prepend the initial starting point to the list of residuals
    # for param in initial_vals.keys():
    #     init_resid = initial_vals[param] - true_vals[param]
    #     # Add the initial residual to the list
    #     if np.ndim(init_resid) == 0:
    #         residuals[param] = np.insert(residuals[param], 0, init_resid)
    #     else:
    #         residuals[param] = np.insert(residuals[param], 0, init_resid, axis=0)


    # Convert certain residuals into fractional errors
    residuals["raw_flux_error_ppm"] = 1e6* residuals["raw_fluxes"] / true_vals["raw_fluxes"]
    residuals["platescale_error_ppm"] = 1e6* residuals["psf_pixel_scale"] / true_vals["psf_pixel_scale"]

    # Compute Total OPD residuals
    # Total OPD is considered to include the Zernike WFE + Uncalibrated 1/f WFE
    # Calibrated WFE is not considered, since it is built into the model
    # The model attempts to recover the Total OPD during the optimization (the zernike's + uncalibrated WFE)
    # If no uncalibrated WFE is present, then the model simply attempts to recover the zernike coefficients
    # If uncalibrated WFE is present, then the recovered coefficients include some signal coming from the 1/f WFE
    # I could attempt to fit the Total OPD surface to a set of zernike coefficients, which might make for a better 'Ground Truth'
    residuals["m1_total_opd"] = history["m1_zernike_opd"] - true_vals["m1_total_opd"]
    residuals["m1_total_opd_rms_nm"] = [ 1e9 * nanrms(opd[m1_mask.astype(bool)]) for opd in residuals["m1_total_opd"] ]
    # residuals["m2_total_opd"] = history["m2_zernike_opd"] - true_vals["m2_total_opd"]
    # residuals["m2_total_opd_rms_nm"] = [ 1e9 * nanrms(opd[m2_mask.astype(bool)]) for opd in residuals["m2_total_opd"] ]



    # Compare Data to Recovered Model - Shows the final result of the optimization
    if save_plots and obs_i < N_saved_plots:
        save_name = f"{script_name}_Recovered_PSF_Comparison_{timestamp}_Obs{obs_i + 1:0{obs_digits}d}.png"
        plot_psf_comparison(
            data=data,
            model=models_out[-1],
            var=var,
            extent=psf_extent_as,
            model_label="Recovered PSF",
            display=present_plots,
            save=True,
            save_name=save_name,
        )


    # Plot loss history
    if save_plots and obs_i < N_saved_plots:
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes = axes.flatten()

        # Left: Full loss history
        save_name = f"{script_name}_Loss_History_{timestamp}_Obs{obs_i + 1:0{obs_digits}d}.png"
        plot_parameter_history(
            names="Loss",
            histories=losses,
            true_vals=true_loss,
            ax=axes[0],
            title="Optimization Loss History",
            display=False,
            save=False
        )
        # Right: Zoom into last 10 iterations
        axes[1].plot(np.arange(n_iter - 10, n_iter) + 1, losses[-10:])
        axes[1].set_title(f"Last 10 Iterations, Final= {losses[-1]:.3f}")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Loss")
        axes[1].axhline(true_loss, linestyle="--", color="k", alpha=0.6, label="True Loss")
        final_delta = np.abs(losses[-1] - true_loss)
        if final_delta != 0:
            axes[1].set_ylim(true_loss-3*final_delta, true_loss+3*final_delta)

        fig.tight_layout()
        fig.savefig(f"../Results/{save_name}", dpi=300)
        if present_plots:
            plt.show()
        else:
            plt.close()


    # Plot parameter histories
    # Plot the optimization results
    # Show the Losses, Flux, and Position Errors
    fig = plt.figure(figsize=(15, 10))
    plt.subplot(4, 2, 1)
    plt.title(f"Binary Separation Error, Final= {residuals['separation'][-1]*1e6:.3f} uas")
    plt.xlabel("Iteration")
    plt.ylabel("Separation Error (uas)")
    plt.plot(np.arange(n_iter + 1), 1e6 * residuals["separation"])
    plt.axhline(0, linestyle="--", color="k", alpha=0.6)

    plt.subplot(4, 2, 2)
    plt.title(f"Binary XY Position Error, Final= {residuals['x_position'][-1]*1e6:.3f}, {residuals['y_position'][-1]*1e6:.3f} uas")
    plt.xlabel("Iteration")
    plt.ylabel("Position Error (uas)")
    plt.plot(np.arange(n_iter + 1), 1e6 * residuals["x_position"], label="X")
    plt.plot(np.arange(n_iter + 1), 1e6 * residuals["y_position"], label="Y")
    plt.axhline(0, linestyle="--", color="k", alpha=0.6)
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.title(f"Binary Flux Error, Final= ({residuals['raw_flux_error_ppm'][-1, 0]:.3f}, {residuals['raw_flux_error_ppm'][-1, 1]:.3f}) ppm")
    plt.xlabel("Iteration")
    plt.ylabel("Flux Error (ppm)")
    plt.plot(np.arange(n_iter + 1), residuals['raw_flux_error_ppm'][:,0], label="Star A")
    plt.plot(np.arange(n_iter + 1), residuals['raw_flux_error_ppm'][:,1], label="Star B")
    plt.axhline(0, linestyle="--", color="k", alpha=0.6)
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.title(f"Binary Angle Error, Final= {residuals['position_angle'][-1]*60**2:.3f} as")
    plt.xlabel("Iteration")
    plt.ylabel("Angle Error (as)")
    plt.plot(np.arange(n_iter + 1), 60**2 * residuals['position_angle'])
    plt.axhline(0, linestyle="--", color="k", alpha=0.6)

    plt.subplot(4, 2, 5)
    plt.title(f"Platescale Error, Final= {residuals['platescale_error_ppm'][-1]:.3f} ppm")
    plt.xlabel("Iteration")
    plt.ylabel("Platescale Error (ppm)")
    plt.plot(np.arange(n_iter + 1), residuals['platescale_error_ppm'])
    plt.axhline(0, linestyle="--", color="k", alpha=0.6)

    plt.subplot(4, 2, 7)
    plt.title(f"M1 Zernike Coefficients, Final= {nanrms(residuals['m1_aperture.coefficients'][-1,:]):.3f} nm rms")
    plt.xlabel("Iteration")
    plt.ylabel("Z Coefficient Error (nm)")
    [plt.plot(np.arange(n_iter + 1), residuals['m1_aperture.coefficients'][:, i], label=f"Z{n}") for i, n in enumerate(m1_noll)]
    plt.axhline(0, linestyle="--", color="k", alpha=0.6)
    plt.legend(loc="upper right")

    # plt.subplot(4, 2, 8)
    # plt.title(f"M2 Zernike Coefficients, Final= {nanrms(residuals['m2_aperture.coefficients'][-1,:]):.3f} nm rms")
    # plt.xlabel("Iteration")
    # plt.ylabel("Z Coefficient Error (nm)")
    # [plt.plot(np.arange(n_iter + 1), residuals['m2_aperture.coefficients'][:, i], label=f"Z{n}") for i, n in enumerate(m2_noll)]
    # plt.axhline(0, linestyle="--", color="k", alpha=0.6)
    # plt.legend(loc="upper right")

    fig.suptitle("Parameter Optimization", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.0)

    if save_plots and obs_i < N_saved_plots:
        save_name = f"{script_name}_Recovered_Parameters_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
        plt.savefig(os.path.join(save_path, save_name))
    if present_plots:
        plt.show()
    else:
        plt.close()



    # Plot the recovered M1 Zernike Coefficients + OPD
    # Define noll index x-ticks for plotting
    nticks_max = 12 # Never show more than this many
    tstep = max(1, (m1_noll.size-1)//nticks_max + 1)
    m1_noll_ticks = m1_noll[::tstep]

    vmin = 1e9* np.min(np.array([true_vals["m1_total_opd"], history["m1_zernike_opd"][-1]]))
    vmax = 1e9* np.max(np.array([true_vals["m1_total_opd"], history["m1_zernike_opd"][-1]]))

    fig = plt.figure(figsize=(20, 10))
    plt.suptitle("M1 OPD Recovery", fontsize=16)
    gs = gridspec.GridSpec(2, 3) # Creates a grid of subplots that I can address

    ax = fig.add_subplot(gs[0, 0]) # OPD Residual RMS Error
    plt.title(f"OPD RMS Residual, Final= {residuals['m1_total_opd_rms_nm'][-1]:.3f} nm rms")
    plt.xlabel("Iteration")
    plt.ylabel("OPD RMS Error (nm)")
    plt.plot(np.arange(n_iter + 1), residuals["m1_total_opd_rms_nm"])
    plt.axhline(0, linestyle="--", color="k", alpha=0.6)

    ax = fig.add_subplot(gs[0, 1:3]) # Z Coefficient Residual Bar Plot
    plt.title("Recovered Coefficients")
    plt.xlabel("Noll Index")
    plt.ylabel("Coefficient Amplitude (nm)")
    plt.scatter(m1_noll, true_vals["m1_aperture.coefficients"], label="True", zorder=2)
    plt.scatter(model_initial_params.m1_zernike_noll, initial_vals["m1_aperture.coefficients"], label="Initial", marker='+', zorder=3)
    plt.scatter(m1_noll, history["m1_aperture.coefficients"][-1,:], label="Recovered", marker='x', zorder=4)
    plt.bar(m1_noll, residuals["m1_aperture.coefficients"][-1,:], label='Residual', zorder=1)
    plt.axhline(0, linestyle="--", color="k", alpha=0.6)
    plt.xticks(m1_noll_ticks)
    plt.legend(loc="upper left")


    ax = fig.add_subplot(gs[1, 0]) # True Total OPD
    plt.title(f"True Total OPD: %.3fnm rms" % true_vals["m1_total_opd_rms_nm"])
    plt.xlabel("X (mm)")
    im = plt.imshow(1e9* true_vals["m1_total_opd"] * m1_nanmask, inferno, vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm", labelpad=0)

    ax = fig.add_subplot(gs[1, 1]) # Recovered OPD
    plt.title(f"Found OPD: %.3fnm rms" % history["m1_zernike_opd_rms_nm"][-1])
    plt.xlabel("X (mm)")
    im = plt.imshow(1e9* history["m1_zernike_opd"][-1,:,:] * m1_nanmask, inferno, vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm", labelpad=0)

    ax = fig.add_subplot(gs[1, 2]) # Residual OPD
    plt.title(f"OPD Residual: %.3fnm rms" % residuals["m1_total_opd_rms_nm"][-1])
    plt.xlabel("X (mm)")
    im = plt.imshow(1e9* residuals["m1_total_opd"][-1,:,:] * m1_nanmask, inferno, extent=pupil_extent_mm)
    cbar = fig.colorbar(im, cax=merge_cbar(ax))
    cbar.set_label("nm", labelpad=0)

    if save_plots and obs_i < N_saved_plots:
        plot_name = "M1-OPD-Recovery"
        save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
        plt.savefig(os.path.join(save_path, save_name))
    if present_plots:
        plt.show()
    else:
        plt.close()




    # # Plot the recovered M2 Zernike Coefficients + OPD
    # # Define noll index x-ticks for plotting
    # nticks_max = 12  # Never show more than this many
    # tstep = max(1, (m2_noll.size - 1) // nticks_max + 1)
    # m2_noll_ticks = m2_noll[::tstep]
    #
    # vmin = 1e9 * np.min(np.array([true_vals["m2_total_opd"], history["m2_zernike_opd"][-1]]))
    # vmax = 1e9 * np.max(np.array([true_vals["m2_total_opd"], history["m2_zernike_opd"][-1]]))
    #
    # fig = plt.figure(figsize=(20, 10))
    # plt.suptitle("M2 OPD Recovery", fontsize=16)
    # gs = gridspec.GridSpec(2, 3)  # Creates a grid of subplots that I can address
    #
    # ax = fig.add_subplot(gs[0, 0])  # OPD Residual RMS Error
    # plt.title(f"OPD RMS Residual, Final= {residuals['m2_total_opd_rms_nm'][-1]:.3f} nm rms")
    # plt.xlabel("Iteration")
    # plt.ylabel("OPD RMS Error (nm)")
    # plt.plot(np.arange(n_iter + 1), residuals["m2_total_opd_rms_nm"])
    # plt.axhline(0, linestyle="--", color="k", alpha=0.6)
    #
    # ax = fig.add_subplot(gs[0, 1:3])  # Z Coefficient Residual Bar Plot
    # plt.title("Recovered Coefficients")
    # plt.xlabel("Noll Index")
    # plt.ylabel("Coefficient Amplitude (nm)")
    # plt.scatter(m2_noll, true_vals["m2_aperture.coefficients"], label="True", zorder=2)
    # plt.scatter(model_initial_params.m2_zernike_noll, initial_vals["m2_aperture.coefficients"], label="Initial", marker='+', zorder=3)
    # plt.scatter(m2_noll, history["m2_aperture.coefficients"][-1, :], label="Recovered", marker='x', zorder=4)
    # plt.bar(m2_noll, residuals["m2_aperture.coefficients"][-1, :], label='Residual', zorder=1)
    # plt.axhline(0, linestyle="--", color="k", alpha=0.6)
    # plt.xticks(m2_noll_ticks)
    # plt.legend(loc="upper left")
    #
    # ax = fig.add_subplot(gs[1, 0])  # True Total OPD
    # plt.title(f"True Total OPD: %.3fnm rms" % true_vals["m2_total_opd_rms_nm"])
    # plt.xlabel("X (mm)")
    # im = plt.imshow(1e9 * true_vals["m2_total_opd"] * m2_nanmask, inferno, vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm", labelpad=0)
    #
    # ax = fig.add_subplot(gs[1, 1])  # Recovered OPD
    # plt.title(f"Found OPD: %.3fnm rms" % history["m2_zernike_opd_rms_nm"][-1])
    # plt.xlabel("X (mm)")
    # im = plt.imshow(1e9 * history["m2_zernike_opd"][-1, :, :] * m2_nanmask, inferno, vmin=vmin, vmax=vmax,
    #                 extent=pupil_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm", labelpad=0)
    #
    # ax = fig.add_subplot(gs[1, 2])  # Residual OPD
    # plt.title(f"OPD Residual: %.3fnm rms" % residuals["m2_total_opd_rms_nm"][-1])
    # plt.xlabel("X (mm)")
    # im = plt.imshow(1e9 * residuals["m2_total_opd"][-1, :, :] * m2_nanmask, inferno, extent=pupil_extent_mm)
    # cbar = fig.colorbar(im, cax=merge_cbar(ax))
    # cbar.set_label("nm", labelpad=0)
    #
    # if save_plots and obs_i < N_saved_plots:
    #     plot_name = "M2-OPD-Recovery"
    #     save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i + 1:0{obs_digits}d}.png"
    #     plt.savefig(os.path.join(save_path, save_name))
    # if present_plots:
    #     plt.show()
    # else:
    #     plt.close()





    # # Now I want to create a plot that compares the recovered M1 and M2 zernike coefficients against each other
    # common_noll = np.intersect1d(model_initial_params.m1_zernike_noll, model_initial_params.m2_zernike_noll) # Finds common zernike terms
    # n_plots = len(common_noll)
    # rows, cols = choose_subplot_grid(n_plots)
    # fig, axes = plt.subplots(int(rows), int(cols), figsize=(4 * int(cols), 3 * int(rows)), sharex=True)
    # axes = onp.array(axes).flatten()  # flatten in case it's 2D
    # fig.suptitle("M1 vs M2 Zernike Coefficient Comparison", fontsize=14)
    # for ax, n in zip(axes, common_noll):
    #     i_m1 = np.where(model_initial_params.m1_zernike_noll == n)[0][0]
    #     i_m2 = np.where(model_initial_params.m2_zernike_noll == n)[0][0]
    #     history_m1 = history["m1_aperture.coefficients"][:, i_m1]
    #     history_m2 = history["m2_aperture.coefficients"][:, i_m2]
    #     true_m1 = true_vals["m1_aperture.coefficients"][i_m1]
    #     true_m2 = true_vals["m2_aperture.coefficients"][i_m2]
    #
    #     ax.set_title(f"Z{n}")
    #     ax.plot(np.arange(n_iter + 1), history_m1, label="M1")
    #     ax.plot(np.arange(n_iter + 1), history_m2, label="M2")
    #     ax.axhline(true_m1, linestyle="--", color="tab:blue", label="True M1", alpha=0.6)
    #     ax.axhline(true_m2, linestyle="--", color="tab:orange", label="True M2", alpha=0.6)
    #     ax.axhline(0, linestyle="--", color="k", alpha=0.3)
    #     ax.set_ylabel("Amplitude (nm)")
    #     ax.legend()
    # # Hide any unused axes
    # for ax in axes[n_plots:]:
    #     ax.axis("off")
    # axes[-1].set_xlabel("Iteration")
    # plt.tight_layout()
    #
    # if save_plots and obs_i < N_saved_plots:
    #     plot_name = "M1-M2-ZernikeCoefficient-Comparison"
    #     save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i + 1:0{obs_digits}d}.png"
    #     plt.savefig(os.path.join(save_path, save_name))
    # if present_plots:
    #     plt.show()
    # else:
    #     plt.close()



    t1_optim = time.time()
    if print2console:
        # Print Results of Optimization
        print("\nAstrometry Retrieval  Results:")
        print("%d iterations in %.3f sec" % (n_iter, t1_optim-t0_optim))
        print("Final Loss Value: %.5g" % losses[-1])
        print("Source X, Y Position Error: %.3f uas, %.3f uas" % (residuals["x_position"][-1]*1e6, residuals["y_position"][-1]*1e6))
        print("Separation Error: %.3f uas" % (residuals["separation"][-1]*1e6))
        print("Source Angle Error: %.3f as" % (residuals["position_angle"][-1]*60**2))
        print("Fractional Flux Error: %.3f ppm A, %.3f ppm B" % (residuals["raw_flux_error_ppm"][-1, 0], residuals["raw_flux_error_ppm"][-1, 1]))
        print("Recovered Log Flux: %.3f" % history["log_flux"][-1])
        print("Recovered Contrast: %.3f" % history["contrast"][-1])
        print("Fractional Platescale Error: %.3f ppm" % residuals["platescale_error_ppm"][-1])
        print("Residual M1 Zernike OPD Error: %.3f nm rms" % residuals["m1_zernike_opd_rms_nm"][-1])
        # print("Residual M2 Zernike OPD Error: %.3f nm rms" % residuals["m2_zernike_opd_rms_nm"][-1])



    # if save_results:

        # List of dicts to save:
        # model_params, data_params
        # true_vals
        # initial_vals
        # history
        # residuals

        # What would be left?
        # photon/read noise
        # "Exposure (s / frame)": exposure_per_frame,
        # "Coadded Frames": N_frames,
        # "Optimiser": optimiser_label,



    if save_results:
        # Construct dicts to save the results
        final_results = {
            "Starting RNG Seed": default_params.rng_seed,
            "N Observations": N_observations,
            "Optimizer Iterations": n_iter,
            # "Jitter Amplitude (as)": jitter_amplitude,
            "Input Source Position X (as)": true_vals["x_position"],
            "Input Source Position Y (as)": true_vals["y_position"],
            "Input Source Separation (as)": true_vals["separation"],
            "Input Source Angle (deg)": true_vals["position_angle"],
            "Input Source Log Flux": true_vals["log_flux"],
            "Input Source Contrast (A:B)": true_vals["contrast"],
            "Input Source A Raw Flux": true_vals["raw_fluxes"][0],
            "Input Source B Raw Flux": true_vals["raw_fluxes"][1],
            "Input Platescale (as/pixel)": true_vals["psf_pixel_scale"],

            "Input M1 Zernikes (Noll Index)": ", ".join(map(str, data_params.m1_zernike_noll)),
            "Input M1 Zernike Coefficient RMS Amplitude (nm)": nanrms(data_params.m1_zernike_amp),
            "Input M1 Zernike Coefficient Amplitudes (nm)": ", ".join(map(str, data_params.m1_zernike_amp)),
            "Input M1 Zernike OPD RMS Error (nm)": true_vals["m1_zernike_opd_rms_nm"],
            "Input M1 Calibrated 1/f Amplitude (nm rms)": model_initial_params.m1_calibrated_amplitude*1e9,
            "Input M1 Calibrated 1/f Power Law": model_initial_params.m1_calibrated_power_law,
            "Input M1 Uncalibrated 1/f Amplitude (nm rms)": data_params.m1_uncalibrated_amplitude*1e9,
            "Input M1 Uncalibrated 1/f Power Law": data_params.m1_uncalibrated_power_law,

            "Input M2 Zernikes (Noll Index)": ", ".join(map(str, data_params.m2_zernike_noll)),
            "Input M2 Zernike Coefficient RMS Amplitude (nm)": nanrms(data_params.m2_zernike_amp),
            "Input M2 Zernike Coefficient Amplitudes (nm)": ", ".join(map(str, data_params.m2_zernike_amp)),
            "Input M2 Zernike OPD RMS Error (nm)": true_vals["m2_zernike_opd_rms_nm"],
            "Input M2 Calibrated 1/f Amplitude (nm rms)": model_initial_params.m2_calibrated_amplitude * 1e9,
            "Input M2 Calibrated 1/f Power Law": model_initial_params.m2_calibrated_power_law,
            "Input M2 Uncalibrated 1/f Amplitude (nm rms)": data_params.m2_uncalibrated_amplitude * 1e9,
            "Input M2 Uncalibrated 1/f Power Law": data_params.m2_uncalibrated_power_law,

            "Photon Noise": add_shot_noise,
            "Read Noise (e- / frame)": sigma_read,
            "Exposure (s / frame)": exposure_per_frame,
            "Coadded Frames": N_frames,
            "Optimiser": optimiser_label,

            "Initial Source Position X (as)": initial_vals["x_position"],
            "Initial Source Position Y (as)": initial_vals["y_position"],
            "Initial Source Separation (as)": initial_vals["separation"],
            "Initial Source Angle (deg)": initial_vals["position_angle"],
            "Initial Source Log Flux": initial_vals["log_flux"],
            "Initial Source Contrast (A:B)": initial_vals["contrast"],
            "Initial Source A Raw Flux": initial_vals["raw_fluxes"][0],
            "Initial Source B Raw Flux": initial_vals["raw_fluxes"][1],
            "Initial Platescale (as/pixel)": initial_vals["psf_pixel_scale"],
            "Initial M1 Zernike Coefficient Amplitudes (nm)": ", ".join(
                    map(str, initial_vals["m1_aperture.coefficients"])),
            # "Initial M2 Zernike Coefficient Amplitudes (nm)": ", ".join(
            #         map(str, initial_vals["m2_aperture.coefficients"])),

            "Found Source Position X (as)": history["x_position"][-1],
            "Found Source Position Y (as)": history["y_position"][-1],
            "Found Source Separation (as)": history["separation"][-1],
            "Found Source Angle (deg)": history["position_angle"][-1],
            "Found Source Log Flux": history["log_flux"][-1],
            "Found Source Contrast (A:B)": history["contrast"][-1],
            "Found Source A Raw Flux": history["raw_fluxes"][-1, 0],
            "Found Source B Raw Flux": history["raw_fluxes"][-1, 1],
            "Found Platescale (as/pixel)": history["psf_pixel_scale"][-1],
            "Found M1 Zernikes (Noll Index)": ", ".join(map(str, model_initial_params.m1_zernike_noll)),
            "Found M1 Zernike Coefficient Amplitudes (nm)": ", ".join(
                    map(str, history["m1_aperture.coefficients"][-1,:])),
            # "Found M2 Zernikes (Noll Index)": ", ".join(map(str, model_initial_params.m1_zernike_noll)),
            # "Found M2 Zernike Coefficient Amplitudes (nm)": ", ".join(
            #         map(str, history["m2_aperture.coefficients"][-1, :])),

            "Residual Source Position X (uas)": residuals["x_position"][-1] * 1e6,
            "Residual Source Position Y (uas)": residuals["y_position"][-1] * 1e6,
            "Residual Source Separation (uas)": residuals["separation"][-1] * 1e6,
            "Residual Source Angle (as)": residuals["position_angle"][-1] * 60 ** 2,
            "Residual Source A Flux (ppm)": residuals["raw_flux_error_ppm"][-1, 0],
            "Residual Source B Flux (ppm)": residuals["raw_flux_error_ppm"][-1, 1],
            "Residual Platescale (ppm)": residuals["platescale_error_ppm"][-1],
            "Residual M1 Zernike OPD RMS Error (nm)": residuals["m1_zernike_opd_rms_nm"][-1],
            # "Residual M2 Zernike OPD RMS Error (nm)": residuals["m2_zernike_opd_rms_nm"][-1],
            "Residual M1 Zernike Coefficient Errors (nm)": ", ".join(
                    map(str, residuals["m1_aperture.coefficients"][-1, :])),
            # "Residual M2 Zernike Coefficient Errors (nm)": ", ".join(
            #     map(str, residuals["m2_aperture.coefficients"][-1, :])),

        }


        # Construct Data Frames
        final_results_df = pd.DataFrame([final_results])

        # Construct Save Name
        save_name = f"{script_name}_{timestamp}.xlsx"

        if obs_i == 0: # Create a new file for the first simulation
            final_results_df.to_excel(os.path.join(save_path, save_name), index=False, sheet_name="Results")
        else: # Append results for subsequent simulations
            with pd.ExcelWriter(os.path.join(save_path, save_name), engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                final_results_df.to_excel(writer, index=False, header=False, sheet_name="Results", startrow=int(obs_i)+1)

        print(f"Simulation {obs_i+1}/{N_observations} results saved to {save_name}")


    t1_simulation = time.time()
    print("Simulation %d finished in %.3f sec" % (obs_i+1, t1_simulation-t0_simulation))



t1_script = time.time()
print("Script finished in %.3f sec" % (t1_script-t0_script))
