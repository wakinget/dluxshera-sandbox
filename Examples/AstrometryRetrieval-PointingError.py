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
from Classes.oneoverf import *

# Plotting/visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


def merge_cbar(ax):
    return make_axes_locatable(ax).append_axes("right", size="5%", pad=0.0)

def hessian(f, x):
    # Jit the sub-function here since it is called many times
    _, hvp = linearize(grad(f), x)
    hvp = jit(hvp)

    # Build and stack
    basis = np.eye(x.size).reshape(-1, *x.shape)
    return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)


def FIM(
    pytree,
    parameters,
    loglike_fn,
    *loglike_args,
    **loglike_kwargs,
):
    # Build X vec
    pytree = zdx.tree.set_array(pytree, parameters)

    if len(parameters) == 1:
        parameters = [parameters]

    leaves = [pytree.get(p) for p in parameters]
    shapes = [leaf.shape for leaf in leaves]
    lengths = [leaf.size for leaf in leaves]
    N = np.array(lengths).sum()
    X = np.zeros(N)

    # Build function to calculate FIM and calculate
    def loglike_fn_vec(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)

    # return hessian(loglike_fn_vec, X)
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


def set_array(pytree, parameters):
    dtype = np.float64 if config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = tree.map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)


def scheduler(lr, start, *args):
    shed_dict = {start: 1e100}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e100, shed_dict)


base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)
sgd = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))


def get_optimiser(pytree, optimisers, parameters=None):

    # Get the parameters and opt_dict
    if parameters is not None:
        optimisers = dict([(p, optimisers[p]) for p in parameters])
    else:
        parameters = list(optimisers.keys())

    model_params = ModelParams(dict([(p, pytree.get(p)) for p in parameters]))
    param_spec = ModelParams(dict([(param, param) for param in parameters]))
    optim = optax.multi_transform(optimisers, param_spec)

    # Build the optimised object - the 'model_params' object
    state = optim.init(model_params)
    return model_params, optim, state


def get_lr_model(
    pytree,
    parameters,
    loglike_fn,
    *loglike_args,
    **loglike_kwargs,
):

    fmat = FIM(model, parameters, loglike_fn, *loglike_args, **loglike_kwargs)
    lr_vec = 1 / np.diag(fmat)

    # lr_model = eqx.filter(model, zdx.boolean_filter(model, parameters))

    idx = 0
    lr_model = {}
    for param in parameters:
        leaf = np.array(model.get(param))
        size, shape = leaf.size, leaf.shape
        lr_model[param] = lr_vec[idx : idx + size].reshape(shape)
        # lr_model = lr_model.set(param, lr_vec[idx : idx + size].reshape(shape))
        idx += size

    return ModelParams(lr_model)


class BaseModeller(zdx.Base):
    params: dict

    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def __getitem__(self, key):

        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]

        return values


class ModelParams(BaseModeller):

    @property
    def keys(self):
        return list(self.params.keys())

    @property
    def values(self):
        return list(self.params.values())

    def __getattr__(self, key):
        if key in self.keys:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set(
            "params", dict([(param, getattr(values, param)) for param in self.keys])
        )

    def from_model(self, values):
        return self.set(
            "params", dict([(param, values.get(param)) for param in self.keys])
        )

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
        # Injects the values of this class into another class
        return other.set(self.keys, self.values)


# Define Loss and Update Functions
# Define the log likelihood
def loglikelihood(model, data, var):
    return jax.scipy.stats.norm.logpdf(model.model(), loc=data, scale=np.sqrt(var))

# Define the loss function
def loss_fn(model, data, var):
    return -np.nansum(loglikelihood(model, data, var))

# Define the step function
@eqx.filter_jit
def step_fn(model_params, data, var, model, lr_model, optim, state):
    print("Compiling update function")

    # Get the loss and the gradients
    loss, grads = val_grad_fn(model, data, var)

    # Normalise the gradients
    grads = tree.map(lambda x, y: x * y, lr_model.replace(grads), lr_model)

    # Update the model parameters
    updates, state = optim.update(grads, state, model_params)
    model_params = zdx.apply_updates(model_params, updates)

    # Re-inject into the model and return
    model = model_params.inject(model)
    return loss, model, model_params, state

def nanrms(arr, axis=None): # Returns the rms value of the input array, ignoring nans
    return np.nanmean(arr**2, axis=axis)**0.5





################################
## Main Simulation Parameters ##
################################


# Start simulation timer
t0_script = time.time()

save_path = os.path.join(os.getcwd(), "..", "Results")
script_name = os.path.splitext((os.path.basename(__file__)))[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Plotting/Saving Settings
save_plots = False # True / False
present_plots = False # True / False
save_results = False
print2console = True

# RNG Settings
starting_seed = 0
# seeds = np.arange(20)

# Pointing Error Settings
n_pointings = 1
pointing_error_rms_amplitude = 0 # as, in X and Y

# Source Settings
star_contrast = 0.3
initial_separation = 10
initial_angle = 90
initial_center = (0, 0)

# Zernike OPD Settings
model_noll_ind = np.arange(4,106) # List of Zernikes present in the model
data_noll_ind = np.arange(4,106) # List of Zernikes present in the data
zCoeff_amp_rms = 0 # Zernike coefficient rms amplitude for the data

# 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106


# 1/f Noise Settings
n_maps = 1
oneoverf_alpha = 2.5
oneoverf_amp_rms = 1e-9

# Image Noise Settings
add_shot_noise = True
sigma_read = 5 # e-/frame rms read noise

# Optimization Settings
n_iter = 100


# Start the simulation(s)
t0_simulation = time.time() # Start simulation timer
n_sims = n_pointings * n_maps
rng_key0 = jr.PRNGKey(starting_seed)


# Construct Optical Systems for the data and for the model
model_optics = dlT.TolimanOpticalSystem(
    wf_npixels=512,
    psf_npixels=128,
    oversample=3,
    noll_indices=model_noll_ind,
)

data_optics = dlT.TolimanOpticalSystem(
    wf_npixels=512,
    psf_npixels=128,
    oversample=3,
    noll_indices=data_noll_ind,
)

# Normalise the aberration basis to be in units of nm
model_optics = model_optics.multiply('basis', 1e-9)
data_optics = data_optics.multiply('basis', 1e-9)

# Define aperture mask
ap_mask = model_optics.aperture.transmission
ap_nanmask = np.where(ap_mask, ap_mask, np.nan)

# Add a WFE layer to the data model - this holds the 1/f map later
wfe_layer = dll.AberratedLayer(np.zeros((model_optics.wf_npixels, model_optics.wf_npixels)))
data_optics = data_optics.insert_layer(('wfe', wfe_layer), 2)

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
    layers=[("downsample", dl.Downsample(model_optics.oversample))]
)

# Create a point source
# Scale the source flux
starA_default_flux = 1.267e11  # photons / second of exposure / square meter of aperture / micron of band
starB_default_flux = 4.557e10  # photons / second of exposure / square meter of aperture / micron of band
default_contrast = starA_default_flux / starB_default_flux
default_total_flux = starA_default_flux + starB_default_flux

exposure_per_frame = 0.1  # seconds
N_frames = 18000  # frames
total_exposure = exposure_per_frame * N_frames  # seconds

aperture_area = np.pi * (model_optics.diameter / 4) ** 2  # square meters of aperture (doesn't include obscurations)
bandpass = 0.11  # bandpass size in microns

total_integrated_flux = default_total_flux * total_exposure * aperture_area * bandpass  # photons
total_integrated_log_flux = np.log10(total_integrated_flux)
# print(total_integrated_log_flux)
# default_log_flux = 6.832 # Corresponds to a single image w/ 0.1s exposure time

source = dlT.AlphaCen(n_wavels=5, x_position=initial_center[0], y_position=initial_center[1], separation=initial_separation,
                      position_angle=initial_angle, log_flux=total_integrated_log_flux, contrast=star_contrast)  # creating Alpha Centauri source

# Combine into an instrument
model = dl.Telescope(
    source=source,
    optics=model_optics,
    detector=detector,
)

# Create a model for the data that includes additional wfe
data_model = dl.Telescope(
    source=source,
    optics=data_optics,
    detector=detector,
)

# Examine the Model
model_psf = model.model()
pupil_extent_mm = model_optics.diameter * 1e3 / 2 * np.array([-1, 1, -1, 1])
psf_extent_as = model_optics.psf_npixels * model_optics.psf_pixel_scale / 2 * np.array([-1, 1, -1, 1])

# Set up the Optimization
# Define the parameters to solve for
lr = 0.5
optimisers = {
    "separation": optax.sgd(lr, 0),
    "position_angle": optax.sgd(lr, 0),
    "x_position": optax.sgd(lr, 0),
    "y_position": optax.sgd(lr, 0),
    "log_flux": optax.sgd(lr, 0),
    "contrast": optax.sgd(lr, 0),
    "psf_pixel_scale": optax.sgd(lr, 0),
    "coefficients": optax.sgd(lr, 0),
}
params = list(optimisers.keys())


keys = jr.split(rng_key0, 6*n_maps)
for map_i in np.arange(n_maps):
    # Generate new WFE map
    oneoverf_opd_true = oneoverf_noise_2D(model_optics.wf_npixels, oneoverf_alpha, key=keys[6*map_i])
    oneoverf_opd_true = remove_PTT(oneoverf_opd_true, model_optics.aperture.transmission.astype(bool))  # Remove PTT from aperture
    oneoverf_opd_true = oneoverf_opd_true * (oneoverf_amp_rms / nanrms(oneoverf_opd_true[ap_mask.astype(bool)]))  # Scale the 1/f noise map over the aperture
    data_model = data_model.set('optics.wfe.opd', oneoverf_opd_true)
    # wfe_layer = dll.AberratedLayer(oneoverf_opd_true)
    # data_optics = data_optics.insert_layer(('wfe', wfe_layer), 2)

    # pointing_keys = jr.split(map_keys[map_i], 5)
    for pointing_i in np.arange(n_pointings):

        # Now simulate a new pointing error
        pointing_err_xy_as = pointing_error_rms_amplitude* jr.normal(keys[6*map_i+1], (2,))
        data_model = data_model.set('source.x_position', initial_center[0] + pointing_err_xy_as[0])
        data_model = data_model.set('source.y_position', initial_center[1] + pointing_err_xy_as[1])

        # Set random zernike coefficients
        zCoeffs_true = zCoeff_amp_rms * jr.normal(keys[6*map_i+2], data_model.coefficients.shape)
        data_model = data_model.set("coefficients", zCoeffs_true)


        # Model the Data PSF
        data_psf = data_model.model()

        # Add Shot Noise to the Image
        if add_shot_noise:
            photons = jr.poisson(keys[6*map_i+3], data_psf) # Add photon noise
        else:
            photons = data_psf


        # Add Read Noise
        read_noise = sigma_read * np.sqrt(N_frames) * jr.normal(keys[6*map_i+4], data_psf.shape)

        # Combine to get some fake data
        data = photons + read_noise
        # data = data_psf # No Noise

        # Get the uncertainties
        var = np.maximum(data + (sigma_read * np.sqrt(N_frames))**2, (sigma_read * np.sqrt(N_frames))**2)



        # Record true values
        posX_true = data_model.get('source.x_position')
        posY_true = data_model.get('source.y_position')
        sep_true = data_model.get('source.separation')
        posAng_true = data_model.get('source.position_angle')
        logFlux_true = data_model.get('source.log_flux')
        contrast_true = data_model.get('source.contrast')
        rawFlux_true = data_model.get('source.raw_fluxes')
        platescale_true = data_model.get('optics.psf_pixel_scale')
        # zCoeffs_true = data_model.get('optics.coefficients')
        zernike_opd_true = data_model.optics.aperture.eval_basis()
        # oneoverf_opd_true

        zernike_opd_true_rms_nm = 1e9* nanrms(zernike_opd_true[ap_mask.astype(bool)])

        if print2console:
            # Print Summary of Inputs
            print("Astrometry Retrieval Initial Inputs:")
            print("Starting RNG Seed: %d" % starting_seed)
            print("Source X, Y Position: %.3f as, %.3f as" % (posX_true, posY_true))
            print("Source Angle: %.3f deg" % posAng_true)
            print("Source Separation: %.3f as" % sep_true)
            print("Source Log Flux: %.3f" % logFlux_true)
            print("Source Contrast: %.3f A:B" % contrast_true)
            print("Detector Platescale: %.3f as/pix" % platescale_true)
            print("Data Includes %.3f nm rms of 1/f^%.2f noise" % (oneoverf_amp_rms*1e9, oneoverf_alpha))
            print("Data Includes %d Zernikes: Z%d - Z%d @%.2f nm rms" % (data_noll_ind.size, np.min(data_noll_ind), np.max(data_noll_ind), zCoeff_amp_rms))
            print("Data Includes Shot Noise: %s" % add_shot_noise)
            print("Data Includes Read Noise @ %.2f e- per frame" % sigma_read)
            print("Model Fits for %d Zernikes: Z%d - Z%d" % (model_noll_ind.size, np.min(model_noll_ind), np.max(model_noll_ind)))



        fig = plt.figure(figsize=(12, 5))
        ax = plt.subplot(1, 3, 1)
        plt.title("Input Zernike Aberrations: %.3f nm rms" % zernike_opd_true_rms_nm)
        plt.xlabel("X (mm)")
        im = ax.imshow(1e9* zernike_opd_true * ap_nanmask, inferno, extent=pupil_extent_mm)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        cbar.set_label("nm")
        ax = plt.subplot(1, 3, 2)
        plt.title("Input 1/f^%.2f Noise: %.3f nm rms" % (oneoverf_alpha, oneoverf_amp_rms*1e9))
        plt.xlabel("X (mm)")
        im = ax.imshow(1e9* oneoverf_opd_true * ap_nanmask, inferno, extent=pupil_extent_mm)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        cbar.set_label("nm")
        ax = plt.subplot(1, 3, 3)
        plt.title("Data")
        plt.xlabel("X (as)")
        im = ax.imshow(data, extent=psf_extent_as)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        cbar.set_label("Photons")
        plt.tight_layout()
        if save_plots:
            map_digits = len(str(n_maps))
            pointing_digits = len(str(n_pointings))
            plot_name = "DataInput"
            save_name = f"{script_name}_{plot_name}_{timestamp}_Map{map_i+1:0{map_digits}d}_Pointing{pointing_i+1:0{pointing_digits}d}.png"
            plt.savefig(os.path.join(save_path, save_name))
        if present_plots:
            plt.show(block=True)
        else:
            plt.close()



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


        # Take the value and gradient transformation of the loss function
        val_grad_fn = zdx.filter_value_and_grad(params)(loss_fn)

        # Get the learning rate normalisation model (variance estimate)
        lr_model = get_lr_model(model, params, loss_fn, model.model(), var)

        # Use the LR model to initialise an incorrect model
        leaves = [np.array(lr_model.get(param)) for param in params]
        keys = jr.split(keys[6*map_i+5], len(leaves))
        perturbations = [3 * np.sqrt(leaf) * jr.normal(keys[i], leaf.shape) for i, leaf in enumerate(leaves)]
        model = set_array(model.add(params, perturbations), params)

        # Initialise our solver
        model_params, optim, state = get_optimiser(model, optimisers)


        # Now we can Optimize
        t0_optim = time.time()
        history = dict([(param, []) for param in params])
        losses, models_out = [], []
        for i in tqdm(range(n_iter)):
            loss, model, model_params, state = step_fn(
                model_params, data, var, model, lr_model, optim, state
            )

            losses.append(loss)
            models_out.append(model)
            for param, value in model_params.params.items():
                history[param].append(value)

        # Get our uncertainties
        err_model = tree.map(
            lambda x: np.sqrt(x),
            get_lr_model(model, params, loss_fn, data, var)
        )

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


        # Get the residual and Z-score
        psf_found = models_out[-1].model()
        psf_resid = data - psf_found
        z_score = psf_resid / np.sqrt(var)

        # Get the colorbar Scale
        v_res = np.nanmax(np.abs(psf_resid))
        v_z = np.nanmax(np.abs(z_score))

        # Get the chi2
        dof = data.size - np.array([leaf.size for leaf in model_params.params.values()]).sum()
        chi2 = np.nansum(z_score**2) / dof

        # Plot Recovered PSF
        fig = plt.figure(figsize=(10, 10))
        plt.suptitle(r"$\chi^2_\nu$: {:.3f}".format(chi2))
        ax = plt.subplot(2, 2, 1)
        ax.set(title="Data")
        ax.set(xlabel="X (as)")
        im = ax.imshow(data, inferno, extent=psf_extent_as)
        fig.colorbar(im, cax=merge_cbar(ax))
        ax = plt.subplot(2, 2, 2)
        ax.set(title="Recovered PSF")
        ax.set(xlabel="X (as)")
        im = ax.imshow(psf_found, inferno, extent=psf_extent_as)
        fig.colorbar(im, cax=merge_cbar(ax))
        ax = plt.subplot(2, 2, 3)
        ax.set(title="Residuals")
        ax.set(xlabel="X (as)")
        im = plt.imshow(psf_resid, seismic, extent=psf_extent_as, vmin=-v_res, vmax=v_res)
        fig.colorbar(im, cax=merge_cbar(ax))
        ax = plt.subplot(2, 2, 4)
        ax.set(title="Z-Score")
        ax.set(xlabel="X (as)")
        im = plt.imshow(z_score, seismic, extent=psf_extent_as, vmin=-v_z, vmax=v_z)
        fig.colorbar(im, cax=merge_cbar(ax))
        fig.tight_layout()
        if save_plots:
            plot_name = "RecoveredPSF"
            save_name = f"{script_name}_{plot_name}_{timestamp}_Map{map_i+1:0{map_digits}d}_Pointing{pointing_i+1:0{pointing_digits}d}.png"
            plt.savefig(os.path.join(save_path, save_name))
        if present_plots:
            plt.show(block=True)
        else:
            plt.close()




        # Format the other outputs
        posX_found = np.array([ind for ind in history['x_position']])
        posY_found = np.array([ind for ind in history['y_position']])
        sep_found = np.array([ind for ind in history['separation']])
        posAng_found = np.array([ind for ind in history['position_angle']])
        logFlux_found = np.array([ind for ind in history['log_flux']])
        contrast_found = np.array([ind for ind in history['contrast']])
        rawFlux_found = np.array([model.get('source.raw_fluxes') for model in models_out])
        platescale_found = np.array([ind for ind in history['psf_pixel_scale']])
        zCoeffs_found = np.array([ind for ind in history['coefficients']])
        zernike_opd_found = np.array([model.aperture.eval_basis() for model in models_out]) # shape: (n_iter, wf_npixels, wf_npixels)
        zernike_opd_found_rms_nm = 1e9* np.nanmean((zernike_opd_found * ap_nanmask[None, :, :])**2, axis=(-1,-2))**0.5 # Shape (n_iter,)
        # prf_found = np.array([model.get(flatfield) for model in models_out])




        # Calculate Residuals = Found - True
        posX_resid = posX_found - posX_true
        posY_resid = posY_found - posY_true
        sep_resid = sep_found - sep_true
        posAng_resid = posAng_found - posAng_true
        logFlux_resid = logFlux_found - logFlux_true
        contrast_resid = contrast_found - contrast_true
        rawFlux_resid = rawFlux_found - rawFlux_true
        platescale_resid = platescale_found - platescale_true

        rawFlux_fracResid = rawFlux_resid / rawFlux_true

        # Analyze the Zernike OPD
        zernike_opd_true_tiled = np.tile(zernike_opd_true, [n_iter, 1, 1])
        zernike_opd_resid = zernike_opd_found - zernike_opd_true_tiled # Shape: (n_iter, wf_npix, wf_npix)
        # zernike_opd_resid_rms_nm = 1e9* np.nanmean((zernike_opd_resid * ap_nanmask[None, :, :])**2, axis=(-1,-2))**0.5 # Shape (n_iter,)
        zernike_opd_resid_rms_nm = 1e9* nanrms(zernike_opd_resid * ap_nanmask[None, :, :], axis=(-1,-2))

        # Analyze the total OPD including zernikes + 1/f noise
        total_opd_true = zernike_opd_true + oneoverf_opd_true
        total_opd_true_rms_nm = 1e9* nanrms(total_opd_true[ap_mask.astype(bool)])
        total_opd_true_tiled = np.tile(zernike_opd_true + oneoverf_opd_true, [n_iter, 1, 1])
        total_opd_resid = zernike_opd_found - total_opd_true_tiled # Shape: (n_iter, wf_npix, wf_npix)
        # total_opd_resid_rms_nm = 1e9* np.nanmean((total_opd_resid * ap_nanmask[None, :, :])**2, axis=(-1,-2))**0.5 # Shape (n_iter,)
        total_opd_resid_rms_nm = 1e9* nanrms(total_opd_resid * ap_nanmask[None, :, :], axis=(-1,-2))

        nZern_data = data_noll_ind.size
        nZern_model = model_noll_ind.size

        if nZern_model != nZern_data:
            if nZern_data > nZern_model:
                # We need to extend the found zCoeffs to include the higher order terms that were present in the data but not in the model
                zCoeffs_found_extnd = np.zeros((zCoeffs_found.shape[0], zCoeffs_true.shape[0]))
                zCoeffs_found_extnd = zCoeffs_found_extnd.at[:, :nZern_model].set(zCoeffs_found) # Higher order modes set to zero
                zCoeffs_resid = zCoeffs_found_extnd - zCoeffs_true
            else: # Model contains more zernike terms than were present in the data
                zCoeffs_true_extnd = np.zeros(zCoeffs_found.shape[1])
                zCoeffs_true_extnd = zCoeffs_true_extnd.at[:nZern_data].set(zCoeffs_true)  # Higher order modes set to zero
                zCoeffs_resid = zCoeffs_found - zCoeffs_true_extnd
        else:
            zCoeffs_resid = zCoeffs_found - zCoeffs_true


        t1_optim = time.time()


        if print2console:
            # Print Results of Optimization
            print("Astrometry Retrieval  Results:")
            print("%d iterations in %.3f sec" % (n_iter, t1_optim-t0_optim))
            print("Source X, Y Position Error: %.3f uas, %.3f uas" % (posX_resid[-1]*1e6, posY_resid[-1]*1e6))
            print("Separation Error: %.3f uas" % (sep_resid[-1]*1e6))
            print("Source Angle Error: %.3f as" % (posAng_resid[-1]*60**2))
            print("Fractional Flux Error: %.3f ppm A, %.3f ppm B" % (rawFlux_fracResid[-1, 0]*1e6, rawFlux_fracResid[-1, 1]*1e6))
            print("Recovered Log Flux: %.3f" % logFlux_found[-1])
            print("Recovered Contrast: %.3f" % contrast_found[-1])
            print("Fractional Platescale Error: %.3f ppm" % (platescale_resid[-1]/platescale_true*1e6))
            print("Residual Zernike OPD Error: %.3f nm rms" % zernike_opd_resid_rms_nm[-1])


        # # Plot the found PSF
        # cmap = mpl.colormaps['inferno']
        # plt.figure(figsize=(15, 4))
        # plt.subplot(1, 3, 1)
        # plt.imshow(data, extent=psf_extent_as)
        # plt.title("Data")
        # plt.xlabel('X (as)')
        # plt.ylabel('Y (as)')
        # plt.colorbar(label='Photons')
        # plt.subplot(1, 3, 2)
        # plt.imshow(psf_found, extent=psf_extent_as)
        # plt.title("Recovered PSF")
        # plt.xlabel('X (as)')
        # plt.ylabel('Y (as)')
        # plt.colorbar(label='Photons')
        # plt.subplot(1, 3, 3)
        # plt.imshow(psf_found - data, extent=psf_extent_as)
        # plt.title("Difference Image")
        # plt.xlabel('X (as)')
        # plt.ylabel('Y (as)')
        # plt.colorbar(label='Photons')
        # plt.show(block=True)


        # Plot the optimization results
        # Show the Losses, Flux, and Position Errors
        plt.figure(figsize=(16, 13))
        plt.subplot(3, 2, 1)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.plot(np.arange(n_iter)+1, losses)
        plt.title(f"Losses, Final= %.3f" % losses[-1])
        plt.subplot(3, 2, 2)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.plot(np.arange(n_iter-10,n_iter)+1, losses[-10:])
        plt.title("Final 10 Losses")
        plt.subplot(3, 2, 3)
        plt.title(f"Stellar Flux Error, Final= (%.3f, %.3f) ppm" % (rawFlux_fracResid[-1, 0]*1e6, rawFlux_fracResid[-1, 1]*1e6))
        plt.xlabel("Iteration")
        plt.ylabel("Flux Error (ppm)")
        plt.plot(np.arange(n_iter)+1, rawFlux_fracResid[:n_iter, 0]*1e6, label="Star A")
        plt.plot(np.arange(n_iter)+1, rawFlux_fracResid[:n_iter, 1]*1e6, label="Star B")
        plt.axhline(0, c='k', alpha=0.5)
        plt.legend(loc="upper right")
        plt.subplot(3, 2, 4)
        plt.title(f"Stellar Position Error, Final= (%.3f, %.3f) uas" % (posX_resid[-1]*1e6, posY_resid[-1]*1e6))
        plt.xlabel("Iteration")
        plt.ylabel("Position Error (uas)")
        plt.plot(np.arange(n_iter)+1, posX_resid[:n_iter]*1e6, label="X")
        plt.plot(np.arange(n_iter)+1, posY_resid[:n_iter]*1e6, label="Y")
        plt.axhline(0, c='k', alpha=0.5)
        plt.legend(loc="upper right")
        plt.subplot(3, 2, 5)
        plt.title("Source Angle Error, Final= %.3f as" % (posAng_resid[-1]*60**2))
        plt.xlabel("Iteration")
        plt.ylabel("Angle Error (as)")
        plt.plot(np.arange(n_iter)+1, posAng_resid[:n_iter]*60**2)
        plt.axhline(0, c='k', alpha=0.5)
        plt.subplot(3, 2, 6)
        plt.title(f"Separation Error, Final= {sep_resid[-1]*1e6:.3f} uas")
        plt.xlabel("Iteration")
        plt.ylabel("Separation Error (uas)")
        plt.plot(np.arange(n_iter)+1, sep_resid[:n_iter]*1e6)
        plt.axhline(0, c='k', alpha=0.5)
        plt.tight_layout()
        if save_plots:
            plot_name = "RecoveredParams"
            save_name = f"{script_name}_{plot_name}_{timestamp}_Map{map_i+1:0{map_digits}d}_Pointing{pointing_i+1:0{pointing_digits}d}.png"
            plt.savefig(os.path.join(save_path, save_name))
        if present_plots:
            plt.show(block=True)
        else:
            plt.close()

        # Make a plot showing the retrieved phase
        vmin = 1e9* np.min(np.array([zernike_opd_true_tiled, zernike_opd_found]))
        vmax = 1e9* np.max(np.array([zernike_opd_true_tiled, zernike_opd_found]))

        # Zernike Coefficients
        noll_index = np.arange(4, np.max(np.concatenate((model_noll_ind, data_noll_ind)))+1)

        # Set up X Ticks for plotting
        nticks_max = 12
        ticks_step = max(1, (noll_index.size-1)//nticks_max + 1)
        noll_ticks = noll_index[::ticks_step]


        # plt.figure(figsize=(20, 10))
        # plt.suptitle("Zernike OPD Recovery")
        # plt.subplot(2, 2, 1)
        # plt.title("Residual OPD RMS Error")
        # plt.xlabel("Iteration")
        # plt.ylabel("OPD RMS Error (nm)")
        # plt.plot(np.arange(n_iter)+1, zernike_opd_resid_rms_nm)
        # plt.axhline(0, c='k', alpha=0.5)
        # plt.subplot(2, 2, 2)
        # plt.title("Recovered Zernike Coefficients")
        # plt.xlabel("Noll Index")
        # plt.ylabel("Coefficient Amplitude (nm)")
        # plt.scatter(noll_index[:nZern_data], zCoeffs_true, label="True Value", zorder=2)
        # plt.scatter(noll_index[:nZern_model].T, zCoeffs_found[-1,:], label="Recovered Value", marker='x', zorder=3)
        # plt.bar(noll_index.T, zCoeffs_resid[-1,:], label='Residual', zorder=1)
        # plt.axhline(0, c='k', alpha=0.5)
        # plt.xticks(noll_ticks)
        # plt.legend(loc="upper left")
        # ax = plt.subplot(2, 3, 4)
        # plt.title(f"True Zernike OPD: %.3fnm rms" % zernike_opd_true_rms_nm)
        # plt.xlabel("X (mm)")
        # im = plt.imshow(1e9* zernike_opd_true * ap_nanmask, inferno, vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
        # cbar = fig.colorbar(im, cax=merge_cbar(ax))
        # # cbar.set_label("nm", labelpad=0)
        # # cbar.ax.yaxis.set_label_position('right')
        # ax = plt.subplot(2, 3, 5)
        # plt.title(f"Found OPD: %.3fnm rms" % zernike_opd_found_rms_nm[-1])
        # plt.xlabel("X (mm)")
        # im = plt.imshow(1e9* zernike_opd_found[-1, :, :] * ap_nanmask, inferno, vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
        # cbar = fig.colorbar(im, cax=merge_cbar(ax))
        # # cbar.set_label("nm", labelpad=0)
        # ax = plt.subplot(2, 3, 6)
        # plt.title(f"OPD Residual: %.3fnm rms" % zernike_opd_resid_rms_nm[-1])
        # plt.xlabel("X (mm)")
        # im = plt.imshow(1e9* zernike_opd_resid[-1, :, :] * ap_nanmask, inferno, extent=pupil_extent_mm)
        # cbar = fig.colorbar(im, cax=merge_cbar(ax))
        # # cbar.set_label("nm", labelpad=0)
        # if save_plots:
        #     plot_name = "RecoveredZernikeOPD"
        #     save_name = f"{script_name}_{plot_name}_{timestamp}_Sim{sim_i+1:0{digits}d}.png"
        #     plt.savefig(os.path.join(save_path, save_name))
        # if present_plots:
        #     plt.show(block=True)
        # else:
        #     plt.close()



        # Make a plot showing the retrieved phase
        vmin = 1e9* np.min(np.array([total_opd_true_tiled, zernike_opd_found]))
        vmax = 1e9* np.max(np.array([total_opd_true_tiled, zernike_opd_found]))

        plt.figure(figsize=(20, 10))
        plt.suptitle("Total OPD Recovery")
        plt.subplot(2, 2, 1)
        plt.title("Residual OPD RMS Error")
        plt.xlabel("Iteration")
        plt.ylabel("OPD RMS Error (nm)")
        plt.plot(np.arange(n_iter)+1, total_opd_resid_rms_nm)
        plt.axhline(0, c='k', alpha=0.5)
        plt.subplot(2, 2, 2)
        plt.title("Recovered Zernike Coefficients")
        plt.xlabel("Noll Index")
        plt.ylabel("Coefficient Amplitude (nm)")
        plt.scatter(noll_index[:nZern_data], zCoeffs_true, label="True Value", zorder=2)
        plt.scatter(noll_index[:nZern_model].T, zCoeffs_found[-1,:], label="Recovered Value", marker='x', zorder=3)
        plt.bar(noll_index.T, zCoeffs_resid[-1,:], label='Residual', zorder=1)
        plt.axhline(0, c='k', alpha=0.5)
        plt.xticks(noll_ticks)
        plt.legend(loc="upper left")
        ax = plt.subplot(2, 3, 4)
        plt.title(f"True Total OPD: %.3fnm rms" % total_opd_true_rms_nm)
        plt.xlabel("X (mm)")
        im = plt.imshow(1e9* (zernike_opd_true+oneoverf_opd_true) * ap_nanmask, inferno, vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        # cbar.set_label("nm", labelpad=0)
        ax = plt.subplot(2, 3, 5)
        plt.title(f"Found OPD: %.3fnm rms" % zernike_opd_found_rms_nm[-1])
        plt.xlabel("X (mm)")
        im = plt.imshow(1e9* zernike_opd_found[-1, :, :] * ap_nanmask, inferno, vmin=vmin, vmax=vmax, extent=pupil_extent_mm)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        # cbar.set_label("nm", labelpad=0)
        ax = plt.subplot(2, 3, 6)
        plt.title(f"OPD Residual: %.3fnm rms" % total_opd_resid_rms_nm[-1])
        plt.xlabel("X (mm)")
        im = plt.imshow(1e9* total_opd_resid[-1, :, :] * ap_nanmask, inferno, extent=pupil_extent_mm)
        cbar = fig.colorbar(im, cax=merge_cbar(ax))
        # cbar.set_label("nm", labelpad=0)
        if save_plots:
            plot_name = "RecoveredTotalOPD"
            save_name = f"{script_name}_{plot_name}_{timestamp}_Map{map_i+1:0{map_digits}d}_Pointing{pointing_i+1:0{pointing_digits}d}.png"
            plt.savefig(os.path.join(save_path, save_name))
        if present_plots:
            plt.show(block=True)
        else:
            plt.close()




        if save_results:
            # Construct dicts to save the results
            final_results = {
                "Starting RNG Seed": starting_seed,
                "Maps": n_maps,
                "Pointings": n_pointings,
                "Optimizer Iterations": n_iter,
                "Pointing Error RMS Amplitude (as)": pointing_error_rms_amplitude,
                "Input Source Position X (as)": posX_true,
                "Input Source Position Y (as)": posY_true,
                "Input Source Separation (as)": sep_true,
                "Input Source Angle (deg)": posAng_true,
                "Input Source Log Flux": logFlux_true,
                "Input Source Contrast (A:B)": contrast_true,
                "Input Source A Raw Flux": rawFlux_true[0],
                "Input Source B Raw Flux": rawFlux_true[1],
                "Input Platescale (as/pixel)": platescale_true,
                "Input Zernikes (Noll Index)": ", ".join(map(str, data_noll_ind)),
                "Input Zernike Coefficient RMS Amplitude (nm)": zCoeff_amp_rms,
                "Input Zernike Coefficient Amplitudes (nm)": ", ".join(map(str, zCoeffs_true)),
                "Input Zernike RMS Error (nm)": zernike_opd_true_rms_nm,
                "Input 1/f Amplitude (nm rms)": oneoverf_amp_rms*1e9,
                "Input 1/f Power Law": oneoverf_alpha,
                "Photon Noise": add_shot_noise,
                "Read Noise (e- / frame)": sigma_read,
                "Exposure (s / frame)": exposure_per_frame,
                "Coadded Frames": N_frames,

                "Found Source Position X (as)": posX_found[-1],
                "Found Source Position Y (as)": posY_found[-1],
                "Found Source Separation (as)": sep_found[-1],
                "Found Source Angle (deg)": posAng_found[-1],
                "Found Source Log Flux": logFlux_found[-1],
                "Found Source Contrast (A:B)": contrast_found[-1],
                "Found Source A Raw Flux": rawFlux_found[-1, 0],
                "Found Source B Raw Flux": rawFlux_found[-1, 1],
                "Found Platescale (as/pixel)": platescale_found[-1],
                "Found Zernikes (Noll Index)": ", ".join(map(str, model_noll_ind)),
                "Found Zernike Coefficient Amplitudes (nm)": ", ".join(map(str, zCoeffs_found[-1,:])),

                "Residual Source Position X (uas)": posX_resid[-1] * 1e6,
                "Residual Source Position Y (uas)": posY_resid[-1] * 1e6,
                "Residual Source Separation (uas)": sep_resid[-1] * 1e6,
                "Residual Source Angle (as)": posAng_resid[-1] * 60 ** 2,
                "Residual Source A Flux (ppm)": rawFlux_fracResid[-1, 0] * 1e6,
                "Residual Source B Flux (ppm)": rawFlux_fracResid[-1, 1] * 1e6,
                "Residual Platescale (ppm)": platescale_resid[-1] / platescale_true * 1e6,
                "Residual Zernike RMS Error (nm)": zernike_opd_resid_rms_nm[-1],

            }


            # Construct Data Frames
            final_results_df = pd.DataFrame([final_results])

            # Construct Save Name
            save_name = f"{script_name}_{timestamp}.xlsx"

            if map_i == 0 and pointing_i == 0: # Create a new file for the first simulation
                final_results_df.to_excel(os.path.join(save_path, save_name), index=False, sheet_name="Results")
            else: # Append results for subsequent simulations
                with pd.ExcelWriter(os.path.join(save_path, save_name), engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
                    final_results_df.to_excel(writer, index=False, header=False, sheet_name="Results", startrow=int(map_i*n_pointings+pointing_i)+1)

            print(f"Simulation {map_i*n_pointings+pointing_i+1}/{n_maps*n_pointings} results saved to {save_name}")


    t1_simulation = time.time()
    print("Simulation %d finished in %.3f sec" % (map_i*n_pointings+pointing_i+1, t1_simulation-t0_simulation))



t1_script = time.time()
print("Script finished in %.3f sec" % (t1_script-t0_script))
