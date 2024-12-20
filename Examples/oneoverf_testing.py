# In this script, we will perform phase retrieval with ∂Lux: recovering Zernike coefficients for an
# aberrated JNEXT telescope by gradient descent.

# Core jax
import jax
import jax.numpy as np
import jax.random as jr
from jax import jit, grad, linearize, lax, config, tree

# Optimisation
import equinox as eqx
import zodiax as zdx
import optax

# Optics
import dLux as dl
import dLux.layers as dll
import dLux.utils as dlu
import dLuxToliman as dlT
from Classes.optical_systems import JNEXTOpticalSystem
from Classes.oneoverf import oneoverf_noise_2D

# Plotting/visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import time, datetime, os

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

def merge_cbar(ax):
    return make_axes_locatable(ax).append_axes("right", size="5%", pad=0.0)

# We want to construct a basic optical system, generate a 1/f opd map, and add it to the system

rng_key = jr.PRNGKey(0)
keys = jr.split(rng_key, 4)

# Construct Optical System
noll_ind = np.arange(4, 15)
zCoeffs_true = 0.5 * jr.normal(keys[0], noll_ind.shape)
optics = dlT.TolimanOpticalSystem(
    wf_npixels=512,
    psf_npixels=128,
    oversample=3,
    noll_indices=noll_ind
)

# Normalise the aberration basis to be in units of nm
optics = optics.multiply('basis', 1e-9)
# Set the zernike coefficients
optics = optics.set("coefficients", zCoeffs_true)
zernike_opd = optics.aperture.eval_basis

# Generate some 1/f noise
oneoverf_opd = oneoverf_noise_2D(optics.wf_npixels, 2.5, key=keys[1]) * 1e-9 # 1nm rms
# Generate a new layer
wfe_layer = dll.AberratedLayer(opd=oneoverf_opd)
# Add the wfe to the optics
optics = optics.insert_layer(('wfe', wfe_layer), 2)


# Create Detector
# prf_true = 1 + 0*jr.normal(rng_key, [det_npix, det_npix]) # Pixel Response Function
# det_jitter = 0*1/optics.psf_pixel_scale # Amplitude of jitter in pixels
# det_layers = [
#     ('pixel_response', dll.ApplyPixelResponse(prf_true))
#     # ('jitter', dll.ApplyJitter(det_jitter))
# ]
# detector = dl.LayeredDetector(det_layers)
# detector = dl.LayeredDetector([dll.ApplyPixelResponse(prf_true)])
detector = dl.LayeredDetector(
    layers=[("downsample", dl.Downsample(optics.oversample))]
)


# Calculate the source flux
starA_default_flux = 1.267e11 # photons / second of exposure / square meter of aperture / micron of band
starB_default_flux = 4.557e10 # photons / second of exposure / square meter of aperture / micron of band
default_contrast = starA_default_flux / starB_default_flux
default_total_flux = starA_default_flux + starB_default_flux

exposure_per_frame = 0.1 # seconds
N_frames = 18000 # frames
total_exposure = exposure_per_frame * N_frames # seconds
aperture_area = np.pi*(optics.diameter/4)**2 # square meters of aperture (doesn't include obscurations)
bandpass = 0.11 # bandpass size in microns
total_integrated_flux = default_total_flux * total_exposure * aperture_area * bandpass # photons
total_integrated_log_flux = np.log10(total_integrated_flux)
# print(total_integrated_log_flux)
# default_log_flux = 6.832 # Corresponds to a single image w/ 0.1s exposure time

# Create a point source
source = dlT.AlphaCen(n_wavels=5, separation=10, position_angle=90, log_flux=total_integrated_log_flux)  # creating Alpha Centauri source


# Combine into an instrument
model = dl.Telescope(
    source=source,
    optics=optics,
    detector=detector,
)


# Examine the Model
pupil_opd = np.where(model.aperture.transmission, model.pupil.opd, np.nan)
wfe_opd = np.where(model.aperture.transmission, model.wfe.opd, np.nan)
model_psf = model.model()
pupil_extent_mm = optics.diameter*1e3/2*np.array([-1, 1, -1, 1])
psf_extent_as = optics.psf_npixels*optics.psf_pixel_scale/2*np.array([-1, 1, -1, 1])



# Now make some fake data
photons = jr.poisson(keys[2], model_psf) # Add photon noise

# Add read noise
sigma_read = 5 * np.sqrt(N_frames)
read_noise = sigma_read * jr.normal(keys[3], model_psf.shape)

# Combine to get some fake data
data = photons + read_noise
# data = psf # No Noise

# Get the uncertainties
var = np.maximum(data + sigma_read**2, sigma_read**2)


fig = plt.figure(figsize=(10, 10))
ax = plt.subplot(2, 2, 1)
plt.title("Zernike Aberrations")
im = ax.imshow(pupil_opd, inferno, extent=pupil_extent_mm)
fig.colorbar(im, cax=merge_cbar(ax))
ax = plt.subplot(2, 2, 2)
plt.title("1/f Noise")
im = ax.imshow(wfe_opd, inferno, extent=pupil_extent_mm)
fig.colorbar(im, cax=merge_cbar(ax))
ax = plt.subplot(2, 2, 3)
plt.title("Model PSF")
im = ax.imshow(model_psf, extent=psf_extent_as)
fig.colorbar(im, cax=merge_cbar(ax))
ax = plt.subplot(2, 2, 4)
plt.title("Data")
im = ax.imshow(data, extent=psf_extent_as)
fig.colorbar(im, cax=merge_cbar(ax))
plt.tight_layout()
plt.show(block=True)


# # Plot
# cmap = mpl.colormaps['inferno']
# cmap.set_bad('k',.5)
# plt.figure(figsize=(10, 4))
# plt.subplot(1, 2, 1)
# plt.imshow(support_mask * opd_true * 1e6, cmap=cmap, extent=aperture_extnt_mm)
# plt.title("Aberrations")
# plt.xlabel('X (mm)')
# plt.ylabel('Y (mm)')
# plt.colorbar(label='um')
#
# plt.subplot(1, 2, 2)
# plt.imshow(data, extent=psf_extnt)
# plt.title("Data")
# plt.xlabel('X (as)')
# plt.ylabel('Y (as)')
# plt.colorbar(label='Photons')
# plt.show(block=True)


# # Record true values
# posX_true = data_model.get('source.x_position')
# posY_true = data_model.get('source.y_position')
# sep_true = data_model.get('source.separation')
# posAng_true = data_model.get('source.position_angle')
# logFlux_true = data_model.get('source.log_flux')
# contrast_true = data_model.get('source.contrast')
# rawFlux_true = data_model.get('source.raw_fluxes')
# platescale_true = data_model.get('optics.psf_pixel_scale')
# zCoeffs_true = data_model.get('optics.coefficients')
# opd_true = data_model.optics.aperture.eval_basis()
#
# nZern_data = optics_ho.layers['aperture'].coefficients.size
# nZern_model = optics.layers['aperture'].coefficients.size
#
# print("Astrometry Retrieval Initial Inputs:")
# print("Source X, Y Position: %.3f as, %.3f as" % (posX_true, posY_true))
# print("Source Angle: %.3f deg" % posAng_true)
# print("Source Separation: %.3f as" % sep_true)
# print("Source Log Flux: %.3f" % logFlux_true)
# print("Source Contrast: %.3f A:B" % contrast_true)
# print("Detector Platescale: %.3f as/pix" % platescale_true)
# print("Data Includes %d Zernikes: Z%d - Z%d" % (nZern_data, 4, 3+nZern_data))
# print("Model Fits for %d Zernikes: Z%d - Z%d" % (nZern_model, 4, 3+nZern_model))
#
#
# # Define Loss and Update Functions
# # Define the log likelihood
# def loglikelihood(model, data, var):
#     return jax.scipy.stats.norm.logpdf(model.model(), loc=data, scale=np.sqrt(var))
#
# # Define the loss function
# def loss_fn(model, data, var):
#     return -np.nansum(loglikelihood(model, data, var))
#
# # Define the step function
# @eqx.filter_jit
# def step_fn(model_params, data, var, model, lr_model, optim, state):
#     print("Compiling update function")
#
#     # Get the loss and the gradients
#     loss, grads = val_grad_fn(model, data, var)
#
#     # Normalise the gradients
#     grads = tree.map(lambda x, y: x * y, lr_model.replace(grads), lr_model)
#
#     # Update the model parameters
#     updates, state = optim.update(grads, state, model_params)
#     model_params = zdx.apply_updates(model_params, updates)
#
#     # Re-inject into the model and return
#     model = model_params.inject(model)
#     return loss, model, model_params, state
#
# # Set up the Optimization
# # Define the parameters to solve for
# lr = 0.5
# optimisers = {
#     "separation": optax.sgd(lr, 0),
#     "position_angle": optax.sgd(lr, 0),
#     "x_position": optax.sgd(lr, 0),
#     "y_position": optax.sgd(lr, 0),
#     "log_flux": optax.sgd(lr, 0),
#     "contrast": optax.sgd(lr, 0),
#     "psf_pixel_scale": optax.sgd(lr, 0),
#     "coefficients": optax.sgd(lr, 0),
# }
# params = list(optimisers.keys())
#
# # Take the value and gradient transformation of the loss function
# val_grad_fn = zdx.filter_value_and_grad(params)(loss_fn)
#
# # Get the learning rate normalisation model (variance estimate)
# lr_model = get_lr_model(model, params, loss_fn, model.model(), var)
#
# # Use the LR model to initialise an incorrect model
# leaves = [np.array(lr_model.get(param)) for param in params]
# keys = jr.split(jr.PRNGKey(0), len(leaves))
# perturbations = [3 * np.sqrt(leaf) * jr.normal(keys[i], leaf.shape) for i, leaf in enumerate(leaves)]
# model = set_array(model.add(params, perturbations), params)
#
# # Initialise our solver
# model_params, optim, state = get_optimiser(model, optimisers)
#
#
# # Now we can Optimize
# t0_optim = time.time()
# n_iter = 50
# history = dict([(param, []) for param in params])
# losses, models_out = [], []
# for i in tqdm(range(n_iter)):
#     loss, model, model_params, state = step_fn(
#         model_params, data, var, model, lr_model, optim, state
#     )
#
#     losses.append(loss)
#     models_out.append(model)
#     for param, value in model_params.params.items():
#         history[param].append(value)
#
# # Get our uncertainties
# err_model = tree.map(
#     lambda x: np.sqrt(x),
#     get_lr_model(model, params, loss_fn, data, var)
# )
#
# # Plot the Losses
# plt.figure(figsize=(15, 5))
# plt.subplot(1, 2, 1)
# plt.title("Losses")
# plt.plot(losses)
# plt.subplot(1, 2, 2)
# plt.title("Final 10 Losses")
# plt.plot(losses[-10:])
# plt.tight_layout()
# if save_plots:
#     plot_name = "Losses"
#     save_name = f"{script_name}_{plot_name}_{timestamp}.png"
#     plt.savefig(os.path.join(save_path, save_name))
# if present_plots:
#     plt.show(block=True)
# else:
#     plt.close()
#
#
# # Get the residual and Z-score
# psf_found = models_out[-1].model()
# psf_resid = data - psf_found
# z_score = psf_resid / np.sqrt(var)
#
# # Get the colorbar Scale
# v_res = np.nanmax(np.abs(psf_resid))
# v_z = np.nanmax(np.abs(z_score))
#
# # Get the chi2
# dof = data.size - np.array([leaf.size for leaf in model_params.params.values()]).sum()
# chi2 = np.nansum(z_score**2) / dof
#
# # Plot Recovered PSF
# fig = plt.figure(figsize=(10, 10))
# plt.suptitle(r"$\chi^2_\nu$: {:.3f}".format(chi2))
# ax = plt.subplot(2, 2, 1)
# ax.set(title="Data")
# im = ax.imshow(data, inferno)
# fig.colorbar(im, cax=merge_cbar(ax))
# ax = plt.subplot(2, 2, 2)
# ax.set(title="Recovered PSF")
# im = ax.imshow(psf_found, inferno)
# fig.colorbar(im, cax=merge_cbar(ax))
# ax = plt.subplot(2, 2, 3)
# ax.set(title="Residuals")
# im = plt.imshow(psf_resid, seismic, vmin=-v_res, vmax=v_res)
# fig.colorbar(im, cax=merge_cbar(ax))
# ax = plt.subplot(2, 2, 4)
# ax.set(title="Z-Score")
# im = plt.imshow(z_score, seismic, vmin=-v_z, vmax=v_z)
# fig.colorbar(im, cax=merge_cbar(ax))
# fig.tight_layout()
# if save_plots:
#     plot_name = "RecoveredPSF"
#     save_name = f"{script_name}_{plot_name}_{timestamp}.png"
#     plt.savefig(os.path.join(save_path, save_name))
# if present_plots:
#     plt.show(block=True)
# else:
#     plt.close()
#
#
#
# # Format the other outputs
# posX_found = np.array([ind for ind in history['x_position']])
# posY_found = np.array([ind for ind in history['y_position']])
# sep_found = np.array([ind for ind in history['separation']])
# posAng_found = np.array([ind for ind in history['position_angle']])
# logFlux_found = np.array([ind for ind in history['log_flux']])
# contrast_found = np.array([ind for ind in history['contrast']])
# rawFlux_found = np.array([model.get('source.raw_fluxes') for model in models_out])
# platescale_found = np.array([ind for ind in history['psf_pixel_scale']])
# zCoeffs_found = np.array([ind for ind in history['coefficients']])
# opd_found = np.array([model.aperture.eval_basis() for model in models_out]) # shape: (n_iter, wf_npixels, wf_npixels)
# # prf_found = np.array([model.get(flatfield) for model in models_out])
#
#
# # Calculate Residuals = Found - True
# posX_resid = posX_found - posX_true
# posY_resid = posY_found - posY_true
# sep_resid = sep_found - sep_true
# posAng_resid = posAng_found - posAng_true
# logFlux_resid = logFlux_found - logFlux_true
# contrast_resid = contrast_found - contrast_true
# rawFlux_resid = rawFlux_found - rawFlux_true
# platescale_resid = platescale_found - platescale_true
#
# # prf_resid = prf_found - prf_true
# rawFlux_fracResid = rawFlux_resid[-1,:] / rawFlux_true
#
# opd_true_tiled = np.tile(opd_true, [n_iter, 1, 1])
# opd_resid = opd_found - opd_true_tiled
# opd_rmse_nm = 1e9*np.mean(opd_resid**2, axis=(-1,-2))**0.5
#
# if nZern_model != nZern_data:
#     if nZern_data > nZern_model:
#         # We need to extend the found zCoeffs to include the higher order terms that were present in the data but not in the model
#         zCoeffs_found_extnd = np.zeros((zCoeffs_found.shape[0], zCoeffs_true.shape[0]))
#         zCoeffs_found_extnd = zCoeffs_found_extnd.at[:, :nZern_model].set(zCoeffs_found) # Higher order modes set to zero
#         zCoeffs_resid = zCoeffs_found_extnd - zCoeffs_true
#     else: # Model contains more zernike terms than were present in the data
#         zCoeffs_true_extnd = np.zeros(zCoeffs_found.shape[1])
#         zCoeffs_true_extnd = zCoeffs_true_extnd.at[:nZern_data].set(zCoeffs_true)  # Higher order modes set to zero
#         zCoeffs_resid = zCoeffs_found - zCoeffs_true_extnd
# else:
#     zCoeffs_resid = zCoeffs_found - zCoeffs_true
#
#
# t1_optim = time.time()
#
# print("Astrometry Retrieval Results:")
# print("%d iterations in %.3f sec" % (n_iter, t1_optim-t0_optim))
# print("Source X, Y Position Error: %.3f uas, %.3f uas" % (posX_resid[-1]*1e6, posY_resid[-1]*1e6))
# print("Separation Error: %.3f uas" % (sep_resid[-1]*1e6))
# print("Source Angle Error: %.3f as" % (posAng_resid[-1]*60**2))
# print("Fractional Flux Error: %.3f ppm A, %.3f ppm B" % (rawFlux_fracResid[0]*1e6, rawFlux_fracResid[1]*1e6))
# print("Recovered Log Flux: %.3f" % logFlux_found[-1])
# print("Recovered Contrast: %.3f" % contrast_found[-1])
# print("Fractional Platescale Error: %.3f ppm" % (platescale_resid[-1]/platescale_true*1e6))
# print("Residual Zernike OPD Error: %.3f nm rms" % opd_rmse_nm[-1])
#
#
# # # Plot the found PSF
# # cmap = mpl.colormaps['inferno']
# # plt.figure(figsize=(15, 4))
# # plt.subplot(1, 3, 1)
# # plt.imshow(data, extent=psf_extent_as)
# # plt.title("Data")
# # plt.xlabel('X (as)')
# # plt.ylabel('Y (as)')
# # plt.colorbar(label='Photons')
# # plt.subplot(1, 3, 2)
# # plt.imshow(psf_found, extent=psf_extent_as)
# # plt.title("Recovered PSF")
# # plt.xlabel('X (as)')
# # plt.ylabel('Y (as)')
# # plt.colorbar(label='Photons')
# # plt.subplot(1, 3, 3)
# # plt.imshow(psf_found - data, extent=psf_extent_as)
# # plt.title("Difference Image")
# # plt.xlabel('X (as)')
# # plt.ylabel('Y (as)')
# # plt.colorbar(label='Photons')
# # plt.show(block=True)
#
#
#
# # Plot the optimization results
# # Show the Losses, Flux, and Position Errors
# plt.figure(figsize=(16, 13))
# plt.subplot(3, 2, (1, 2))
# plt.xlabel("Iterations")
# plt.ylabel("Log10( Loss )")
# plt.plot(np.log10(np.array(losses)[:n_iter]))
# # plt.title(f"Log10 Loss, Final= {np.log10(np.array(losses)[-1]):.3f}")
# plt.title(f"Log Loss, Final= %.3f" % np.log10(np.array(losses[-1])))
# plt.subplot(3, 2, 3)
# plt.title(f"Stellar Flux Error, Final= (%.3f, %.3f) ppm" % (rawFlux_fracResid[0]*1e6, rawFlux_fracResid[1]*1e6))
# plt.xlabel("Iterations")
# plt.ylabel("Flux Error (Photons)")
# plt.plot(rawFlux_resid[:n_iter, 0], label="Star A")
# plt.plot(rawFlux_resid[:n_iter, 1], label="Star B")
# plt.axhline(0, c='k', alpha=0.5)
# plt.legend(loc="upper right")
# plt.subplot(3, 2, 4)
# plt.title(f"Stellar Position Error, Final= (%.3f, %.3f) uas" % (posX_resid[-1]*1e6, posY_resid[-1]*1e6))
# plt.xlabel("Iterations")
# plt.ylabel("Positional Error (as)")
# plt.plot(posX_resid[:n_iter], label="X")
# plt.plot(posY_resid[:n_iter], label="Y")
# plt.axhline(0, c='k', alpha=0.5)
# plt.legend(loc="upper right")
# plt.subplot(3, 2, 5)
# plt.title("Source Angle Error, Final= %.3f as" % (posAng_resid[-1]*60**2))
# plt.xlabel("Iterations")
# plt.ylabel("Angle Error (deg)")
# plt.plot(posAng_resid[:n_iter])
# plt.axhline(0, c='k', alpha=0.5)
# plt.subplot(3, 2, 6)
# plt.title(f"Separation Error, Final= {sep_resid[-1]*1e6:.3f} uas")
# plt.xlabel("Iterations")
# plt.ylabel("Separation Error (uas)")
# plt.plot(sep_resid[:n_iter]*1e6)
# plt.axhline(0, c='k', alpha=0.5)
# plt.tight_layout()
# if save_plots:
#     plot_name = "RecoveredParams"
#     save_name = f"{script_name}_{plot_name}_{timestamp}.png"
#     plt.savefig(os.path.join(save_path, save_name))
# if present_plots:
#     plt.show(block=True)
# else:
#     plt.close()
#
# # Make a plot showing the retrieved phase
# vmin = np.min(np.array([opd_true_tiled, opd_found]))
# vmax = np.max(np.array([opd_true_tiled, opd_found]))
#
# # Coefficients
# index = np.arange(np.max(np.array([nZern_data, nZern_model])))+4
#
# plt.figure(figsize=(20, 10))
# plt.suptitle("Optical Aberrations")
# plt.subplot(2, 2, 1)
# plt.title("RMS OPD residual")
# plt.xlabel("Iterations")
# plt.ylabel("RMS OPD (nm)")
# plt.plot(opd_rmse_nm)
# plt.axhline(0, c='k', alpha=0.5)
# plt.subplot(2, 2, 2)
# plt.title("Recovered Zernike Coefficients")
# plt.xlabel("Noll Index")
# plt.ylabel("Coefficient Amplitude (nm)")
# plt.scatter(index[:nZern_data], zCoeffs_true, label="True Value", zorder=2)
# plt.scatter(index[:nZern_model].T, zCoeffs_found[-1,:], label="Recovered Value", marker='x', zorder=3)
# plt.bar(index.T, zCoeffs_resid[-1,:], label='Residual', zorder=1)
# plt.axhline(0, c='k', alpha=0.5)
# plt.xticks(index)
# plt.legend(loc="upper left")
# plt.subplot(2, 3, 4)
# plt.title("True OPD")
# plt.imshow(opd_true, vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
# plt.colorbar()
# plt.subplot(2, 3, 5)
# plt.title("Found OPD")
# plt.imshow(opd_found[-1, :, :], vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
# plt.colorbar()
# plt.subplot(2, 3, 6)
# plt.title(f"OPD Residual: {opd_rmse_nm[-1]:.2f} nm rms")
# plt.imshow(opd_resid[-1, :, :], extent=pupil_extent_mm)
# plt.colorbar()
# if save_plots:
#     plot_name = "RecoveredOPD"
#     save_name = f"{script_name}_{plot_name}_{timestamp}.png"
#     plt.savefig(os.path.join(save_path, save_name))
# if present_plots:
#     plt.show(block=True)
# else:
#     plt.close()
#
#
#
# t1_script = time.time()
# print("Script finished in %.3f sec" % (t1_script-t0_script))
