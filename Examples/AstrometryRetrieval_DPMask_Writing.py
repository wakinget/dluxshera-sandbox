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
from zodiax import filter_vmap
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
from astropy.io import fits

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
    # print(pytree)

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

    # print(model)
    # print(parameters)
    # print(loglike_fn)

    fmat = FIM(model, parameters, loglike_fn, *loglike_args, **loglike_kwargs)
    # print(fmat)
    lr_vec = 1 / np.diag(fmat)
    # print(lr_vec)

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

    def get(self, key):
        if key in self.params:
            return self.params[key]
        raise ValueError(f"key: {key} not found in object: {type(self).__name__}")

    # def get(self, key: str):
    #     parts = key.split(".")
    #     val = self
    #     for part in parts:
    #         if isinstance(val, dict):
    #             val = val[part]
    #         elif hasattr(val, part):
    #             val = getattr(val, part)
    #         else:
    #             raise KeyError(f"Invalid path segment: {part}")
    #     return val


class ModelParams(BaseModeller):

    @property
    def keys(self):
        return list(self.params.keys())

    @property
    def values(self):
        return list(self.params.values())

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set(
            "params", dict([(param, values.get(param)) for param in self.keys])
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


class SheraMultiPlaneSystem(dl.MultiPlaneOpticalSystem):
    """
    A MultiPlaneOpticalSystem of the SHERA telescope with a primary and secondary mirror
    """

    m1_focal_length: float
    m2_focal_length: float
    m1_m2_separation: float
    pixel_pitch: float

    def __init__(
        self,
        wf_npixels: int = 256,
        psf_npixels: int = 256,
        oversample: int = 4,
        pixel_pitch: float = 4.6,  # um per pixel
        mask: jax.Array = None,
        radial_orders: jax.Array = None,
        noll_indices: jax.Array = None,
        coefficients: jax.Array = None,
        m1_diameter: float = 0.220,
        m2_diameter: float = 0.025,
        m1_focal_length: float = 0.604353,
        m2_focal_length: float = -0.0545,
        m1_m2_separation: float = 0.554130,
        n_struts: int = 3,
        strut_width: float = 0.002,
        strut_rotation: float = -np.pi / 2,
    ):
        """
        A pre-built dLux optics layer of the Shera optical system, including a secondary mirror.

        Parameters
        ----------
        wf_npixels : int
            The pixel width the wavefront layer.
        psf_npixels : int
            The pixel width of the PSF.
        oversample : int
            The Nyquist oversampling factor of the PSF.
        psf_pixel_scale : float
            The pixel scale of the PSF in arcseconds per pixel.
        mask : Array
            The diffractive mask array to apply to the wavefront layer.
        radial_orders : Array = None
            The radial orders of the zernike polynomials to be used for the
            aberrations. Input of [0, 1] would give [Piston, Tilt X, Tilt Y],
            [1, 2] would be [Tilt X, Tilt Y, Defocus, Astig X, Astig Y], etc.
            The order must be increasing but does not have to be consecutive.
            If you want to specify specific zernikes across radial orders the
            noll_indices argument should be used instead.
        noll_indices : Array
            The zernike noll indices to be used for the aberrations. [1, 2, 3]
            would give [Piston, Tilt X, Tilt Y], [2, 3, 4] would be [Tilt X,
            Tilt Y, Defocus.
        coefficients : Array
            The coefficients of the Zernike polynomials.
        m1_diameter : float
            The diameter of the primary mirror in metres.
        m2_diameter : float
            The diameter of the secondary mirror in metres.
        m1_focal_length : float
            The focal length of the primary mirror in metres.
        m2_focal_length : float
            The focal length of the secondary mirror in metres.
        m1_m2_separation : float
            The separation between the primary and secondary in metres.
        n_struts : int
            The number of uniformly spaced struts holding the secondary mirror.
        strut_width : float
            The width of the struts in metres.
        strut_rotation : float
            The angular rotation of the struts in radians.
        """

        # Set attributes
        self.m1_focal_length = m1_focal_length
        self.m2_focal_length = m2_focal_length
        self.m1_m2_separation = m1_m2_separation
        self.pixel_pitch = pixel_pitch

        # Diameter
        diameter = [m1_diameter, m2_diameter]

        # Calculate the angular pixel scale
        psf_pixel_scale = dlu.rad2arcsec(self.pixel_pitch*1e-6 / self.focal_length)

        # Generate Pupil Aperture
        pupil_oversample = 5
        coords = dlu.pixel_coords(pupil_oversample * wf_npixels, diameter[0])
        outer = dlu.circle(coords, m1_diameter / 2)
        inner = dlu.circle(coords, m2_diameter / 2, invert=True)
        strut_angles = np.linspace(0,360, n_struts+1)
        strut_angles = strut_angles[:-1] + np.rad2deg(strut_rotation)
        spiders = dlu.spider(coords, strut_width, strut_angles)
        m1_transmission = dlu.combine([outer, inner, spiders], pupil_oversample)
        m2_transmission = dlu.downsample(outer, pupil_oversample)

        # Generate a zernike basis
        if radial_orders is not None:
            radial_orders = np.array(radial_orders)

            if (radial_orders < 0).any():
                raise ValueError("Radial orders must be >= 0")

            noll_indices = []
            for order in radial_orders:
                start = dlu.triangular_number(order)
                stop = dlu.triangular_number(order + 1)
                noll_indices.append(np.arange(start, stop) + 1)
            noll_indices = np.concatenate(noll_indices).astype(int)

        if noll_indices is None:
            m1_aperture = dll.TransmissiveLayer(m1_transmission, normalise=True)
            m2_aperture = dll.TransmissiveLayer(m2_transmission, normalise=True)
        else:
            # Generate Basis
            coords = dlu.pixel_coords(wf_npixels, diameter[0])
            basis = np.array(
                [dlu.zernike(i, coords, m1_diameter) for i in noll_indices]
            )

            if coefficients is None:
                coefficients = np.zeros(len(noll_indices))

            # Combine into BasisOptic class
            m1_aperture = dll.BasisOptic(basis, m1_transmission, coefficients, normalise=True)
            m2_aperture = dll.BasisOptic(basis, m2_transmission, coefficients, normalise=True)

        # # Generate Aperture
        # aperture = dLux.apertures.ApertureFactory(
        #     npixels=wf_npixels,
        #     radial_orders=radial_orders,
        #     noll_indices=noll_indices,
        #     coefficients=coefficients,
        #     secondary_ratio=m2_diameter / m1_diameter,
        #     nstruts=n_struts,
        #     strut_ratio=strut_width / m1_diameter,
        #     strut_rotation=strut_rotation,
        # )

        # Put this here for now, will be in dLux eventually
        def scale_array(array: jax.Array, size_out: int, order: int) -> jax.Array:
            xs = np.linspace(0, array.shape[0], size_out)
            xs, ys = np.meshgrid(xs, xs)
            return jax.scipy.ndimage.map_coordinates(array, np.array([ys, xs]), order=order)

        # Generate DP Mask
        if mask is None:
            path = os.path.join(os.path.dirname(__file__), "../venv/lib/python3.9/site-packages/dLuxToliman/diffractive_pupil.npy")
            # print("DP Path: %s" % path)
            # arr_in = np.load(path)
            # ratio = wf_npixels / arr_in.shape[-1]
            mask = scale_array(np.load(path), wf_npixels, order=1)

            # Enforce full binary
            mask = mask.at[np.where(mask <= 0.5)].set(0.0)
            mask = mask.at[np.where(mask > 0.5)].set(1.0)

            # Enforce full binary
            mask = dlu.phase2opd(mask * np.pi, 550e-9)

            # Turn into optic
            mask = dll.AberratedLayer(mask)


        # Define Layers for the Primary and for the Secondary
        m1_layers = [("m1_aperture", m1_aperture), ("dp_mask", mask)]
        m2_layers = [("m2_aperture", m2_aperture)]

        # Propagator Properties
        psf_npixels = int(psf_npixels)
        oversample = float(oversample)
        psf_pixel_scale = float(psf_pixel_scale)

        super().__init__(
            wf_npixels = wf_npixels,
            diameter = diameter,
            plane_layers = [m1_layers, m2_layers],
            plane_separations = self.m1_m2_separation,
            plane_magnifications = self.m1_magnification,
            psf_npixels = psf_npixels,
            oversample = int(oversample),
            psf_pixel_scale = psf_pixel_scale,
        )

    @property
    def focal_length(self) -> float:
        """
        Returns the effective focal length of the telescope
        -------
        focal_length : float, metres
            The focal length of the telescope in metres
        """

        # Reduce the two optic system
        m1_phi = 1 / self.m1_focal_length  # diopters, M1 power
        m2_phi = 1 / self.m2_focal_length  # diopters, M2 power
        phi_total = m1_phi + m2_phi - self.m1_m2_separation * m1_phi * m2_phi  # Total power

        return 1 / phi_total  # effective focal length

    @property
    def m1_magnification(self) -> float:
        """
        Returns the magnification of the primary mirror
        -------
        m1_magnification : float, unitless
            The magnification of the primary mirror
        """

        return 1 / (1 - self.m1_m2_separation / self.m1_focal_length)

    def _apply_aperture(self, wavelength, offset):
        """
        Overwrite so mask can be stored as array
        """
        wf = self._construct_wavefront(wavelength, offset)
        wf *= self.m1_aperture
        wf = wf.normalise()
        wf += self.dp_mask
        return wf


# Add scale_array here so I can use it locally
def scale_array(array: jax.Array, size_out: int, order: int) -> jax.Array:
    xs = np.linspace(0, array.shape[0], size_out)
    xs, ys = np.meshgrid(xs, xs)
    return jax.scipy.ndimage.map_coordinates(array, np.array([ys, xs]), order=order)

def sine_wave_2D(size, amplitude, frequency, angle=0, phase=None):
    """
    Generate a rotated 2D sine wave pattern.

    Parameters:
    -----------
    size : int
        The size of the output 2D array (size x size).
    amplitude : float
        The amplitude of the sine wave.
    frequency : float
        The frequency of the sine wave (cycles per unit length).
    angle : float (Optional)
        The angle (in degrees) to rotate the sine wave pattern counter-clockwise.
    phase : float array (Optional)
        The phase of the sine wave specified as a full (size x size) array.


    Returns:
    --------
    sine_wave : np.array
        A 2D array containing the sine wave pattern.
    """

    if phase is None:
        phase = np.zeros((size, size))

    # Normalized coordinate grid
    x = np.linspace(-0.5, 0.5, size)
    y = np.linspace(-0.5, 0.5, size)
    X, Y = np.meshgrid(x, y)

    # Convert angle to radians
    theta = np.deg2rad(angle)

    # Rotate coordinate system
    X_rot = X * np.cos(theta) + Y * np.sin(theta)
    Y_rot = -X * np.sin(theta) + Y * np.cos(theta)

    # Compute the sine wave along rotated axes
    sine_wave = amplitude * (np.sin(2 * np.pi * frequency * X_rot - phase) +
                             np.sin(2 * np.pi * frequency * Y_rot - phase))

    return sine_wave




################################
## Main Simulation Parameters ##
################################


# Start simulation timer
t0_script = time.time()

save_path = os.path.join(os.getcwd(), "..", "Results")
script_name = os.path.splitext((os.path.basename(__file__)))[0]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# run_optimization = False # Only runs the optimization if enabled

# Plotting/Saving Settings
save_plots = True # True / False
N_saved_plots = 3 # Limit the number of plots that are saved, the first N plots will be saved
present_plots = False # True / False
save_results = False
print2console = True

# RNG Settings
rng0 = 0
rng_key, subkey = jr.split(jr.PRNGKey(rng0))

# Pointing Error Settings
jitter_amplitude = 0 # as

# Source Settings
star_contrast = 2.78
initial_separation = 10
initial_angle = 90
initial_center = (0, 0)
central_wavelength = 550 # nm
bandwidth = 110 # nm
n_wavelengths = 5

## Primary Mirror OPD Settings
# Zernike WFE
m1_noll_ind = np.arange(4, 5)
rng_key, subkey = jr.split(rng_key)
m1_zCoeff_initial = 0 * jr.normal(subkey, m1_noll_ind.shape)
# m1_zCoeff_initial = np.zeros_like(m1_noll_ind)
# m1_zCoeff_initial = m1_zCoeff_initial.at[1].set(0)
m1_zCoeff_amp_rms = nanrms(m1_zCoeff_initial)
m1_zCoeff_delta_amp_rms = 0 # nm rms, describes changes to zernike coefficients from observation to observation
# 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106
# Calibrated 1/f WFE
m1_calibrated_wfe_amp = 0e-9
m1_calibrated_wfe_alpha = 2.5
# Uncalibrated 1/f WFE
m1_uncalibrated_wfe_amp = 0e-9
m1_uncalibrated_wfe_alpha = 2.5

## Secondary Mirror OPD Settings
# Zernike OPD
# m2_noll_ind = np.arange(4, 11) # Noll indices for Secondary
rng_key, subkey = jr.split(rng_key)
m2_zCoeff_initial = 0 * jr.normal(subkey, m1_noll_ind.shape)
# m2_zCoeff_initial = np.zeros_like(m1_noll_ind)
# m2_zCoeff_initial = m2_zCoeff_initial.at[1].set(0) # Add Tip/Tilt
m2_zCoeff_amp_rms = nanrms(m2_zCoeff_initial)
m2_zCoeff_delta_amp_rms = 0 # nm rms, describes changes to zernike coefficients
# 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106
# Calibrated 1/f WFE
m2_calibrated_wfe_amp = 0e-9
m2_calibrated_wfe_alpha = 2.5
# Uncalibrated 1/f WFE
m2_uncalibrated_wfe_amp = 0e-9
m2_uncalibrated_wfe_alpha = 2.5


# Detector and Imaging Settings
add_shot_noise = False
sigma_read = 0 # e-/frame rms read noise
detector_pixel_pitch = 6.5 # um, detector pixel size
psf_npix = 512

# Telescope Parameters
pupil_npix = 2000
# SHERA Testbed Prescription
m1_diam = 0.09
m2_diam = 0.025
m1_focal_length = 0.35796
m2_focal_length = -0.041935
m1_m2_separation = 0.320
# # SHERA Flight Prescription
# m1_diam = 0.22
# m2_diam = 0.025
# m1_focal_length = 0.604353
# m2_focal_length = -0.0545
# m1_m2_separation = 0.554130

# Observation Settings
N_observations = 1 # Number of repeated observations
total_exposure_time = 0.05 # sec, total exposure time of the observation
frame_rate = 20 # Hz, observation frame rate
exposure_per_frame = 1/frame_rate  # seconds
N_frames = frame_rate*total_exposure_time  # frames


# Optimization Settings
n_iter = 30

# Define the parameters to solve for
lr = 0.5
# opt = optax.sgd(lr,0)
opt = optax.sgd(lr, nesterov=True, momentum=0.5)
optimiser_label = "optax.sgd-nesterov-m0p5"
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


# Start the simulation(s)
t0_simulation = time.time() # Start simulation timer


# Construct Optical Systems for the data and for the model
model_optics = SheraMultiPlaneSystem(
    wf_npixels = pupil_npix,
    psf_npixels = psf_npix,
    oversample = 3,
    pixel_pitch = detector_pixel_pitch,
    noll_indices = m1_noll_ind,
    m1_diameter = m1_diam,
    m2_diameter = m2_diam,
    m1_focal_length = m1_focal_length,
    m2_focal_length = m2_focal_length,
    m1_m2_separation = m1_m2_separation,
)

data_optics = SheraMultiPlaneSystem(
    wf_npixels = pupil_npix,
    psf_npixels = psf_npix,
    oversample = 3,
    pixel_pitch = detector_pixel_pitch,
    noll_indices = m1_noll_ind,
    m1_diameter = m1_diam,
    m2_diameter = m2_diam,
    m1_focal_length = m1_focal_length,
    m2_focal_length = m2_focal_length,
    m1_m2_separation = m1_m2_separation,
)


# Normalise the zernike basis to be in units of nm
model_optics = model_optics.multiply('m1_aperture.basis', 1e-9)
model_optics = model_optics.multiply('m2_aperture.basis', 1e-9)
data_optics = data_optics.multiply('m1_aperture.basis', 1e-9)
data_optics = data_optics.multiply('m2_aperture.basis', 1e-9)

# Define aperture masks
m1_mask = model_optics.m1_aperture.transmission
m1_nanmask = np.where(m1_mask, m1_mask, np.nan)
m2_mask = model_optics.m2_aperture.transmission
m2_nanmask = np.where(m2_mask, m2_mask, np.nan)

# Initialize the Zernike WFE
model_optics = model_optics.set('m1_aperture.coefficients', m1_zCoeff_initial)
model_optics = model_optics.set('m2_aperture.coefficients', m2_zCoeff_initial)
data_optics = data_optics.set('m1_aperture.coefficients', m1_zCoeff_initial)
data_optics = data_optics.set('m2_aperture.coefficients', m2_zCoeff_initial)
m1_zernike_opd_initial = data_optics.m1_aperture.eval_basis()
m2_zernike_opd_initial = data_optics.m2_aperture.eval_basis()
m1_zernike_opd_initial_rms_nm = 1e9* nanrms(m1_zernike_opd_initial[m1_mask.astype(bool)])
m2_zernike_opd_initial_rms_nm = 1e9* nanrms(m2_zernike_opd_initial[m2_mask.astype(bool)])


# Initialize the Calibrated 1/f WFE
rng_key, subkey = jr.split(rng_key)
m1_cal_wfe = oneoverf_noise_2D(model_optics.wf_npixels, m1_calibrated_wfe_alpha, key=subkey)
m1_cal_wfe = remove_PTT(m1_cal_wfe, model_optics.m1_aperture.transmission.astype(bool))  # Remove PTT from aperture
m1_cal_wfe = m1_cal_wfe * (m1_calibrated_wfe_amp / nanrms(m1_cal_wfe[m1_mask.astype(bool)]))  # Scale the 1/f noise map over the aperture
rng_key, subkey = jr.split(rng_key)
m2_cal_wfe = oneoverf_noise_2D(model_optics.wf_npixels, m2_calibrated_wfe_alpha, key=subkey)
m2_cal_wfe = remove_PTT(m2_cal_wfe, model_optics.m2_aperture.transmission.astype(bool))  # Remove PTT from aperture
m2_cal_wfe = m2_cal_wfe * (m2_calibrated_wfe_amp / nanrms(m2_cal_wfe[m2_mask.astype(bool)]))  # Scale the 1/f noise map over the aperture
m1_cal_layer = dll.AberratedLayer(m1_cal_wfe)
m2_cal_layer = m1_cal_layer.set('opd', m2_cal_wfe)
model_optics = model_optics.insert_layer(('calibration', m1_cal_layer), 3, 0)
model_optics = model_optics.insert_layer(('calibration', m2_cal_layer), 3, 1)
data_optics = data_optics.insert_layer(('calibration', m1_cal_layer), 3, 0)
data_optics = data_optics.insert_layer(('calibration', m2_cal_layer), 3, 1)
# Calibration is built into both data and model


# Initialize the Uncalibrated 1/f WFE
rng_key, subkey = jr.split(rng_key)
m1_uncal_wfe = oneoverf_noise_2D(model_optics.wf_npixels, m1_uncalibrated_wfe_alpha, key=subkey)
m1_uncal_wfe = remove_PTT(m1_uncal_wfe, model_optics.m1_aperture.transmission.astype(bool))  # Remove PTT from aperture
m1_uncal_wfe = m1_uncal_wfe * (m1_uncalibrated_wfe_amp / nanrms(m1_uncal_wfe[m1_mask.astype(bool)]))  # Scale the 1/f noise map over the aperture
rng_key, subkey = jr.split(rng_key)
m2_uncal_wfe = oneoverf_noise_2D(model_optics.wf_npixels, m2_uncalibrated_wfe_alpha, key=subkey)
m2_uncal_wfe = remove_PTT(m2_uncal_wfe, model_optics.m2_aperture.transmission.astype(bool))  # Remove PTT from aperture
m2_uncal_wfe = m2_uncal_wfe * (m2_uncalibrated_wfe_amp / nanrms(m2_uncal_wfe[m2_mask.astype(bool)]))  # Scale the 1/f noise map over the aperture
m1_uncal_layer = dll.AberratedLayer(m1_uncal_wfe)
m2_uncal_layer = m1_uncal_layer.set('opd', m2_uncal_wfe)
# model_optics = model_optics.insert_layer(('wfe', m1_uncal_layer), 3, 0)
# model_optics = model_optics.insert_layer(('wfe', m2_uncal_layer), 3, 1)
data_optics = data_optics.insert_layer(('wfe', m1_uncal_layer), 4, 0)
data_optics = data_optics.insert_layer(('wfe', m2_uncal_layer), 4, 1)
# Uncalibrated WFE is built into the data only

# Initialize Detector
if jitter_amplitude == 0:
    detector = dl.LayeredDetector(
        layers=[("downsample", dl.Downsample(model_optics.oversample))
                ]
    )
else:
    detector = dl.LayeredDetector(
        layers=[("jitter", dl.ApplyJitter(sigma=jitter_amplitude/model_optics.psf_pixel_scale)),
                ("downsample", dl.Downsample(model_optics.oversample))
                ]
    )

# Create a point source
# Scale the source flux
starA_default_flux = 1.267e11  # photons / second of exposure / square meter of aperture / micron of band
starB_default_flux = 4.557e10  # photons / second of exposure / square meter of aperture / micron of band
default_contrast = starA_default_flux / starB_default_flux
default_total_flux = starA_default_flux + starB_default_flux

aperture_area = np.pi * (model_optics.diameter[0] / 2) ** 2  # square meters of aperture (doesn't include M2 obscuration)
bandpass = (central_wavelength-bandwidth/2, central_wavelength+bandwidth/2)

total_integrated_flux = default_total_flux * total_exposure_time * aperture_area * (bandwidth/1000)  # photons
total_integrated_log_flux = np.log10(total_integrated_flux)
# print(total_integrated_log_flux)
# default_log_flux = 6.832 # Corresponds to a single image w/ 0.1s exposure time

source = dlT.AlphaCen(n_wavels=n_wavelengths, x_position=initial_center[0], y_position=initial_center[1], separation=initial_separation,
                      position_angle=initial_angle, log_flux=total_integrated_log_flux, contrast=star_contrast, bandpass=bandpass)  # creating Alpha Centauri source


# Combine individual elements into an instrument
model = dl.Telescope(
    source = source,
    optics = model_optics,
    detector = detector,
)

data_model = dl.Telescope(
    source = source,
    optics = data_optics,
    detector = detector,
)

# Examine the Model
model_psf = model.model()
m1_extent_mm = model_optics.diameter[0] * 1e3 / 2 * np.array([-1, 1, -1, 1])
m2_extent_mm = model_optics.diameter[1] * 1e3 / 2 * np.array([-1, 1, -1, 1])
psf_extent_as = model_optics.psf_npixels * model_optics.psf_pixel_scale / 2 * np.array([-1, 1, -1, 1])


## Write Sinusoidal Grating onto DP array
# Load the DP Mask
path = os.path.join(os.path.dirname(__file__), "../venv/lib/python3.9/site-packages/dLuxToliman/diffractive_pupil.npy")
dp0 = np.load(path)
mask = scale_array(dp0, pupil_npix, order=1)

# Construct Sinusoidal Grating
amp = dlu.phase2opd(np.pi/16, central_wavelength*1e-9)
freq = 128 # Cycles per aperture
angle = 45 # Rotation angle of the Grating
phase = mask*np.pi
grating = sine_wave_2D(pupil_npix, amp, freq, angle, phase)

# Enforce full binary
# mask = mask.at[np.where(mask <= 0.5)].set(0.0)
# mask = mask.at[np.where(mask > 0.5)].set(1.0)

# Convert to OPD
dp_mask_opd = dlu.phase2opd(mask * np.pi/2, central_wavelength*1e-9) - central_wavelength*1e-9/4
# phase_flip = mask.at[np.where(mask <= 0.5)].set(-1.0)
circ_ap = model_optics.m2_aperture.transmission
# circ_ap = model_optics.m1_aperture.transmission

# Plot the DP Array
# dp_array = model_optics.dp_mask.opd
plt.figure()
plt.subplot(1,3,1)
plt.imshow(circ_ap, extent= m1_extent_mm)
plt.imshow(1e9*dp_mask_opd * circ_ap, extent= m1_extent_mm)
plt.title('DP Mask Surface')
plt.xlabel('X (mm)')
plt.colorbar(label='nm')

# Plot the Grating
plt.subplot(1,3,2)
# plt.imshow(grating)
# plt.title('Grating')
plt.imshow(1e9*grating * circ_ap, extent= m1_extent_mm)
plt.title('Phase Flipped Grating')
plt.xlabel('X (mm)')
plt.colorbar(label='nm')

# Plot the sum of the DP Array and the Grating
plt.subplot(1,3,3)
# plt.imshow(dp_array + grating)
plt.imshow(1e9*(grating + dp_mask_opd) * circ_ap, extent= m1_extent_mm)
plt.title('Total Surface')
plt.xlabel('X (mm)')
plt.colorbar(label='nm')
plt.show()

full_dp_mask_surface_nm = 1e9*(grating + dp_mask_opd) * circ_ap


# Create a PrimaryHDU object with the data
hdu = fits.PrimaryHDU(full_dp_mask_surface_nm)

# Add metadata to the header
short_path = os.path.normpath(path)
subpath = short_path.split('/venv', 1)[-1]
short_path = '/venv' + subpath
print(short_path)
hdu.header['WHO'] = 'Dylan McKeithen'
# hdu.header['WHAT'] = "This is a write of USydney's Diffractive Pupil, optimized for their optical system, scaled to the size of SHERA, with the sinusoidal diffraction grating written on top, describing the surface height of the primary mirror, for use by MDL."
hdu.header['WHAT'] = "USydney's DP, scaled to SHERA, with grating on top"
hdu.header['WHEN'] = timestamp
hdu.header['WHERE'] = '/dLuxShera/Examples/AstrometryRetrieval_DPMask_Writing.py'
hdu.header['WHY'] = "8-fold -> 3-fold symmetry"
hdu.header['Z_UNIT'] = 'NM'
hdu.header['XY_UNIT'] = 'MM'
hdu.header['DIAM'] = m1_diam*1e3
hdu.header['DX'] = m1_diam*1e3 / pupil_npix
hdu.header['DP_FILE'] = short_path
hdu.header['FREQ'] = freq
hdu.header['PV_DEPTH'] = amp*1e9*4
hdu.header['WAVEL'] = central_wavelength

# Write to a FITS file
save_name = f"SHERA_9cm_DP_Surface_{timestamp}.fits"
hdu.writeto(save_name, overwrite=True)
print("DP Written to File Successfully.")

# # Take the value and gradient transformation of the loss function
# val_grad_fn = zdx.filter_value_and_grad(params)(loss_fn)
#
# # Start the Observation Loop
# obs_keys = jr.split(rng_key, N_observations)  # One key per observation
# for obs_i in np.arange(N_observations):
#     obs_key = obs_keys[obs_i]
#
#     # Update Primary zernike coefficients, adds a delta to the initial coefficients
#     obs_key, subkey = jr.split(obs_key)
#     m1_zCoeffs_true_delta = m1_zCoeff_delta_amp_rms * jr.normal(subkey, data_model.m1_aperture.coefficients.shape)
#     data_model = data_model.add("m1_aperture.coefficients", m1_zCoeffs_true_delta)
#     # Update Secondary zernike coefficients, adds a delta to the initial coefficients
#     obs_key, subkey = jr.split(obs_key)
#     m2_zCoeffs_true_delta = m2_zCoeff_delta_amp_rms * jr.normal(subkey, data_model.m2_aperture.coefficients.shape)
#     data_model = data_model.add("m2_aperture.coefficients", m2_zCoeffs_true_delta)
#
#     # Update other parameters?
#
#     # Model the Data PSF
#     data_psf = data_model.model()
#
#     # Model the Beam at the Secondary
#     # Extract info from source, necessary because Telescope doesn't have a propagate_to_plane method, must operate on optics
#     weights = data_model.source.norm_weights
#     fluxes = data_model.source.raw_fluxes
#     positions = data_model.source.xy_positions
#     wavels = 1e-9 * data_model.source.wavelengths
#     input_weights = weights * fluxes[:, None]
#     # Define a function to iterate over two stars
#     prop_fn = lambda position, weight: data_model.optics.propagate_to_plane(
#         wavelengths = wavels,
#         offset = position,
#         weights = weight,
#         plane_index=1)
#     data_m2 = filter_vmap(prop_fn)(positions, input_weights) # Size (2, wf_npix, wf_npix), one for each star
#     # data_m2 = data_m2.sum(0) # Add both stars together
#
#     # fig = plt.figure(figsize=(5, 5))
#     # ax = plt.axes()
#     # plt.title("Data PSF")
#     # plt.xlabel("X (as)")
#     # im = ax.imshow(data_psf, extent=psf_extent_as)
#     # cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     # cbar.set_label("Photons")
#     # plt.tight_layout()
#     # plt.show(block=True)
#
#     # Add Shot Noise to the Image
#     if add_shot_noise:
#         obs_key, subkey = jr.split(obs_key)
#         if total_exposure_time >= 7200: # If exposure is > 2 hours
#             photons = np.sqrt(data_psf) * jr.normal(subkey, data_psf.shape) + data_psf # Gaussian Approximation
#         else:
#             photons = jr.poisson(subkey, data_psf)  # Add photon noise
#     else:
#         photons = data_psf
#
#     # fig = plt.figure(figsize=(5, 5))
#     # ax = plt.axes()
#     # plt.title("Photons")
#     # plt.xlabel("X (as)")
#     # im = ax.imshow(photons, extent=psf_extent_as)
#     # cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     # cbar.set_label("Photons")
#     # plt.tight_layout()
#     # plt.show(block=True)
#
#     # Add Read Noise
#     obs_key, subkey = jr.split(obs_key)
#     read_noise = sigma_read * np.sqrt(N_frames) * jr.normal(subkey, data_psf.shape)
#
#     # Combine to get some fake data
#     data = photons + read_noise
#     # data = data_psf # No Noise
#
#     # Get the uncertainties
#     if add_shot_noise:
#         var = np.maximum(data + (sigma_read * np.sqrt(N_frames))**2, (sigma_read * np.sqrt(N_frames))**2)
#     else:
#         # var = np.maximum((sigma_read * np.sqrt(N_frames))**2, 1e-20) * np.ones_like(data) # No shot noise, ensures a minimum value to prevent nans
#         var = np.maximum(data + (sigma_read * np.sqrt(N_frames)) ** 2, (sigma_read * np.sqrt(N_frames)) ** 2)  # Same as for shot noise
#
#     # Record true values
#     posX_true = data_model.get('source.x_position')
#     posY_true = data_model.get('source.y_position')
#     sep_true = data_model.get('source.separation')
#     posAng_true = data_model.get('source.position_angle')
#     logFlux_true = data_model.get('source.log_flux')
#     contrast_true = data_model.get('source.contrast')
#     rawFlux_true = data_model.get('source.raw_fluxes')
#     platescale_true = data_model.get('optics.psf_pixel_scale')
#     m1_zernike_coeffs_true = data_model.get('m1_aperture.coefficients') # Includes initial + delta
#     m1_zernike_opd_true = data_model.optics.m1_aperture.eval_basis()
#     m1_zernike_opd_true_rms_nm = 1e9* nanrms(m1_zernike_opd_true[m1_mask.astype(bool)])
#     m2_zernike_coeffs_true = data_model.get('m2_aperture.coefficients') # Includes initial + delta
#     m2_zernike_opd_true = data_model.optics.m2_aperture.eval_basis()
#     m2_zernike_opd_true_rms_nm = 1e9* nanrms(m2_zernike_opd_true[m2_mask.astype(bool)])
#     # true_vals = [np.array(data_model.get(param)) for param in params]
#
#     # Get the loss and the gradients
#     true_loss, true_grads = val_grad_fn(data_model, data, var)
#
#     if print2console:
#         # Print Summary of Inputs
#         print("Astrometry Retrieval Initial Inputs:")
#         print("Starting RNG Seed: %d" % rng0)
#         print("Source X, Y Position: %.3f as, %.3f as" % (posX_true, posY_true))
#         print("Source Angle: %.3f deg" % posAng_true)
#         print("Source Separation: %.3f as" % sep_true)
#         # print("Separation Change: %.3f uas" % (deltaSep_true*1e6))
#         print("Source Log Flux: %.3f" % logFlux_true)
#         print("Source Contrast: %.3f A:B" % contrast_true)
#         print("Detector Platescale: %.3f as/pix" % platescale_true)
#
#         print("Optics initialized with %d Zernikes: Z%d - Z%d" % (m1_noll_ind.size, np.min(m1_noll_ind), np.max(m1_noll_ind)))
#
#         print("M1 initialized with %.3f nm rms of Zernike WFE" % m1_zernike_opd_initial_rms_nm)
#         print("M1 Zernike coefficients change at %.3f nm rms" % m1_zCoeff_delta_amp_rms)
#         print("M1 finalized with %.3f nm rms of Zernike WFE" % m1_zernike_opd_true_rms_nm)
#
#         print("M2 initialized with %.3f nm rms of Zernike WFE" % m2_zernike_opd_initial_rms_nm)
#         print("M2 Zernike coefficients change at %.3f nm rms" % m2_zCoeff_delta_amp_rms)
#         print("M2 finalized with %.3f nm rms of Zernike WFE" % m2_zernike_opd_true_rms_nm)
#
#         print("M1 Includes %.3f nm rms of calibrated 1/f^%.2f noise" % (m1_calibrated_wfe_amp*1e9, m1_calibrated_wfe_alpha))
#         print("M1 Includes %.3f nm rms of uncalibrated 1/f^%.2f noise" % (m1_uncalibrated_wfe_amp*1e9, m1_uncalibrated_wfe_alpha))
#         print("M2 Includes %.3f nm rms of calibrated 1/f^%.2f noise" % (m2_calibrated_wfe_amp*1e9, m2_calibrated_wfe_alpha))
#         print("M2 Includes %.3f nm rms of uncalibrated 1/f^%.2f noise" % (m2_uncalibrated_wfe_amp*1e9, m2_uncalibrated_wfe_alpha))
#
#         print("Data Includes %.3f as of jitter" % jitter_amplitude)
#         print("Data Includes Shot Noise: %s" % add_shot_noise)
#         print("Data Includes Read Noise @ %.2f e- per frame" % sigma_read)
#
#         print("Model Fits for %d Zernikes: Z%d - Z%d" % (m1_noll_ind.size, np.min(m1_noll_ind), np.max(m1_noll_ind)))
#         print("True Loss Value: %.5g" % true_loss)
#
#
#
#     fig = plt.figure(figsize=(15, 7))
#     ax = plt.subplot(2, 4, 1)
#     plt.title("M1 Calibrated WFE: %.3f nm rms" % (m1_calibrated_wfe_amp * 1e9))
#     plt.xlabel("X (mm)")
#     im = ax.imshow(1e9 * m1_cal_wfe * m1_nanmask, inferno, extent=m1_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     cbar.set_label("nm")
#     ax = plt.subplot(2, 4, 2)
#     plt.title("M1 Uncalibrated WFE: %.3f nm rms" % (m1_uncalibrated_wfe_amp * 1e9))
#     plt.xlabel("X (mm)")
#     im = ax.imshow(1e9 * m1_uncal_wfe * m1_nanmask, inferno, extent=m1_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     cbar.set_label("nm")
#     ax = plt.subplot(2, 4, 3)
#     plt.title("M1 True Zernike Aberrations: %.3f nm rms" % m1_zernike_opd_true_rms_nm)
#     plt.xlabel("X (mm)")
#     im = ax.imshow(1e9* m1_zernike_opd_true * m1_nanmask, inferno, extent=m1_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     cbar.set_label("nm")
#     ax = plt.subplot(2, 4, 4)
#     plt.title("Propagated Beam at M2")
#     plt.xlabel("X (mm)")
#     im = ax.imshow(data_m2.sum(0), extent=m2_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     cbar.set_label("Photons")
#     ax = plt.subplot(2, 4, 5)
#     plt.title("M2 Calibrated WFE: %.3f nm rms" % (m2_calibrated_wfe_amp * 1e9))
#     plt.xlabel("X (mm)")
#     im = ax.imshow(1e9 * m2_cal_wfe * m2_nanmask, inferno, extent=m2_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     cbar.set_label("nm")
#     ax = plt.subplot(2, 4, 6)
#     plt.title("M2 Uncalibrated WFE: %.3f nm rms" % (m2_uncalibrated_wfe_amp * 1e9))
#     plt.xlabel("X (mm)")
#     im = ax.imshow(1e9 * m2_uncal_wfe * m2_nanmask, inferno, extent=m2_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     cbar.set_label("nm")
#     ax = plt.subplot(2, 4, 7)
#     plt.title("M2 True Zernike Aberrations: %.3f nm rms" % m2_zernike_opd_true_rms_nm)
#     plt.xlabel("X (mm)")
#     im = ax.imshow(1e9* m2_zernike_opd_true * m2_nanmask, inferno, extent=m2_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     cbar.set_label("nm")
#     ax = plt.subplot(2, 4, 8)
#     plt.title("Final Data PSF")
#     plt.xlabel("X (as)")
#     im = ax.imshow(data, extent=psf_extent_as)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     cbar.set_label("Photons")
#     plt.tight_layout()
#     if save_plots and obs_i < N_saved_plots:
#         obs_digits = len(str(N_observations))
#         plot_name = "DataInput"
#         save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
#         plt.savefig(os.path.join(save_path, save_name))
#     if present_plots:
#         plt.show()
#     else:
#         plt.close()
#
#
#
#     # # Plot
#     # cmap = mpl.colormaps['inferno']
#     # cmap.set_bad('k',.5)
#     # plt.figure(figsize=(10, 4))
#     # plt.subplot(1, 2, 1)
#     # plt.imshow(support_mask * opd_true * 1e6, cmap=cmap, extent=aperture_extnt_mm)
#     # plt.title("Aberrations")
#     # plt.xlabel('X (mm)')
#     # plt.ylabel('Y (mm)')
#     # plt.colorbar(label='um')
#     #
#     # plt.subplot(1, 2, 2)
#     # plt.imshow(data, extent=psf_extnt)
#     # plt.title("Data")
#     # plt.xlabel('X (as)')
#     # plt.ylabel('Y (as)')
#     # plt.colorbar(label='Photons')
#     # plt.show(block=True)
#
#
#     # Get the learning rate normalisation model (variance estimate)
#     lr_model = get_lr_model(model, params, loss_fn, model.model(), var)
#
#
#
#     # Use the LR model to initialise an incorrect model
#     # print("Top-level model fields:", dir(lr_model))
#     # print("Params to retrieve:", params)
#     leaves = [np.array(lr_model.get(param)) for param in params]
#     keys = jr.split(obs_key, len(leaves))
#     perturbations = [30 * np.sqrt(leaf) * jr.normal(keys[i], leaf.shape) for i, leaf in enumerate(leaves)]
#     model = set_array(model.add(params, perturbations), params)
#     # print(perturbations)
#     # print((model.separation-10)*1e6)
#
#     # Initialise our solver
#     model_params, optim, state = get_optimiser(model, optimisers)
#
#
#     # Now we can Optimize
#     t0_optim = time.time()
#     history = dict([(param, []) for param in params])
#     losses, models_out = [], []
#     for i in tqdm(range(n_iter)):
#         loss, model, model_params, state = step_fn(
#             model_params, data, var, model, lr_model, optim, state
#         )
#
#         losses.append(loss)
#         models_out.append(model)
#         for param, value in model_params.params.items():
#             history[param].append(value)
#
#     # Get our uncertainties
#     err_model = tree.map(
#         lambda x: np.sqrt(x),
#         get_lr_model(model, params, loss_fn, data, var)
#     )
#
#     # # Plot the Losses
#     # plt.figure(figsize=(15, 5))
#     # plt.subplot(1, 2, 1)
#     # plt.title("Losses")
#     # plt.plot(losses)
#     # plt.subplot(1, 2, 2)
#     # plt.title("Final 10 Losses")
#     # plt.plot(losses[-10:])
#     # plt.tight_layout()
#     # if save_plots:
#     #     plot_name = "Losses"
#     #     save_name = f"{script_name}_{plot_name}_{timestamp}.png"
#     #     plt.savefig(os.path.join(save_path, save_name))
#     # if present_plots:
#     #     plt.show(block=True)
#     # else:
#     #     plt.close()
#
#
#     # Get the residual and Z-score
#     psf_found = models_out[-1].model()
#     psf_resid = data - psf_found
#     z_score = psf_resid / np.sqrt(var)
#
#     # Get the colorbar Scale
#     v_res = np.nanmax(np.abs(psf_resid))
#     v_z = np.nanmax(np.abs(z_score))
#
#     # Get the chi2
#     dof = data.size - np.array([leaf.size for leaf in model_params.params.values()]).sum()
#     chi2 = np.nansum(z_score**2) / dof
#
#     # Plot Recovered PSF
#     fig = plt.figure(figsize=(10, 10))
#     plt.suptitle(r"$\chi^2_\nu$: {:.3f}".format(chi2))
#     ax = plt.subplot(2, 2, 1)
#     ax.set(title="Data")
#     ax.set(xlabel="X (as)")
#     im = ax.imshow(data, inferno, extent=psf_extent_as)
#     fig.colorbar(im, cax=merge_cbar(ax))
#     ax = plt.subplot(2, 2, 2)
#     ax.set(title="Recovered PSF")
#     ax.set(xlabel="X (as)")
#     im = ax.imshow(psf_found, inferno, extent=psf_extent_as)
#     fig.colorbar(im, cax=merge_cbar(ax))
#     ax = plt.subplot(2, 2, 3)
#     ax.set(title="Residuals")
#     ax.set(xlabel="X (as)")
#     im = plt.imshow(psf_resid, seismic, extent=psf_extent_as, vmin=-v_res, vmax=v_res)
#     fig.colorbar(im, cax=merge_cbar(ax))
#     ax = plt.subplot(2, 2, 4)
#     ax.set(title="Z-Score")
#     ax.set(xlabel="X (as)")
#     im = plt.imshow(z_score, seismic, extent=psf_extent_as, vmin=-v_z, vmax=v_z)
#     fig.colorbar(im, cax=merge_cbar(ax))
#     fig.tight_layout()
#     if save_plots and obs_i < N_saved_plots:
#         plot_name = "RecoveredPSF"
#         save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
#         plt.savefig(os.path.join(save_path, save_name))
#     if present_plots:
#         plt.show()
#     else:
#         plt.close()
#
#
#
#
#     # Format the other outputs
#     posX_found = np.array([ind for ind in history['x_position']])
#     posY_found = np.array([ind for ind in history['y_position']])
#     sep_found = np.array([ind for ind in history['separation']])
#     posAng_found = np.array([ind for ind in history['position_angle']])
#     logFlux_found = np.array([ind for ind in history['log_flux']])
#     contrast_found = np.array([ind for ind in history['contrast']])
#     rawFlux_found = np.array([model.get('source.raw_fluxes') for model in models_out])
#     platescale_found = np.array([ind for ind in history['psf_pixel_scale']])
#     m1_zCoeffs_found = np.array([ind for ind in history['m1_aperture.coefficients']])
#     m1_zernike_opd_found = np.array([model.m1_aperture.eval_basis() for model in models_out]) # shape: (n_iter, wf_npixels, wf_npixels)
#     m1_zernike_opd_found_rms_nm = 1e9* np.nanmean((m1_zernike_opd_found * m1_nanmask[None, :, :])**2, axis=(-1,-2))**0.5 # Shape (n_iter,)
#     m2_zCoeffs_found = np.array([ind for ind in history['m2_aperture.coefficients']])
#     m2_zernike_opd_found = np.array([model.m2_aperture.eval_basis() for model in models_out]) # shape: (n_iter, wf_npixels, wf_npixels)
#     m2_zernike_opd_found_rms_nm = 1e9* np.nanmean((m2_zernike_opd_found * m2_nanmask[None, :, :])**2, axis=(-1,-2))**0.5 # Shape (n_iter,)
#     # prf_found = np.array([model.get(flatfield) for model in models_out])
#
#     # Get the uncertainties from err_model
#     # This represents the standard deviation of each parameter estimated from the Fisher Information Matrix
#     posX_std = err_model.x_position
#     posY_std = err_model.y_position
#     sep_std = err_model.separation
#     posAng_std = err_model.position_angle
#     logFlux_std = err_model.log_flux
#     contrast_std = err_model.contrast
#     platescale_std = err_model.psf_pixel_scale
#     m1_zCoeffs_std = err_model.get('m1_aperture.coefficients')
#     m2_zCoeffs_std = err_model.get('m2_aperture.coefficients')
#
#
#     # Calculate Residuals = Found - True
#     posX_resid = posX_found - posX_true
#     posY_resid = posY_found - posY_true
#     sep_resid = sep_found - sep_true
#     posAng_resid = posAng_found - posAng_true
#     logFlux_resid = logFlux_found - logFlux_true
#     contrast_resid = contrast_found - contrast_true
#     rawFlux_resid = rawFlux_found - rawFlux_true
#     platescale_resid = platescale_found - platescale_true
#
#     rawFlux_fracResid = rawFlux_resid / rawFlux_true
#
#     # Analyze the Zernike OPD
#     m1_zernike_opd_true_tiled = np.tile(m1_zernike_opd_true, [n_iter, 1, 1])
#     m1_zernike_opd_resid = m1_zernike_opd_found - m1_zernike_opd_true_tiled # Shape: (n_iter, wf_npix, wf_npix)
#     m1_zernike_opd_resid_rms_nm = 1e9* nanrms(m1_zernike_opd_resid * m1_nanmask[None, :, :], axis=(-1,-2))
#     m2_zernike_opd_true_tiled = np.tile(m2_zernike_opd_true, [n_iter, 1, 1])
#     m2_zernike_opd_resid = m2_zernike_opd_found - m2_zernike_opd_true_tiled # Shape: (n_iter, wf_npix, wf_npix)
#     m2_zernike_opd_resid_rms_nm = 1e9* nanrms(m2_zernike_opd_resid * m2_nanmask[None, :, :], axis=(-1,-2))
#
#     # Analyze the total OPD including zernikes + uncalibrated 1/f wfe (Calibrated WFE not included)
#     m1_total_opd_true = m1_zernike_opd_true + m1_uncal_wfe
#     m1_total_opd_true_rms_nm = 1e9* nanrms(m1_total_opd_true[m1_mask.astype(bool)])
#     m1_total_opd_true_tiled = np.tile(m1_total_opd_true, [n_iter, 1, 1])
#     m1_total_opd_resid = m1_zernike_opd_found - m1_total_opd_true_tiled # Shape: (n_iter, wf_npix, wf_npix)
#     m1_total_opd_resid_rms_nm = 1e9* nanrms(m1_total_opd_resid * m1_nanmask[None, :, :], axis=(-1,-2))
#     m2_total_opd_true = m2_zernike_opd_true + m2_uncal_wfe
#     m2_total_opd_true_rms_nm = 1e9* nanrms(m2_total_opd_true[m2_mask.astype(bool)])
#     m2_total_opd_true_tiled = np.tile(m2_total_opd_true, [n_iter, 1, 1])
#     m2_total_opd_resid = m2_zernike_opd_found - m2_total_opd_true_tiled # Shape: (n_iter, wf_npix, wf_npix)
#     m2_total_opd_resid_rms_nm = 1e9* nanrms(m2_total_opd_resid * m2_nanmask[None, :, :], axis=(-1,-2))
#
#     nZern_data = data_model.m1_aperture.coefficients.size
#     nZern_model = model.m1_aperture.coefficients.size
#     if nZern_model != nZern_data:
#         if nZern_data > nZern_model:
#             # We need to extend the found zCoeffs to include the higher order terms that were present in the data but not in the model
#             m1_zCoeffs_found_extnd = np.zeros((m1_zCoeffs_found.shape[0], m1_zCoeff_initial.shape[0]))
#             m1_zCoeffs_found_extnd = m1_zCoeffs_found_extnd.at[:, :nZern_model].set(m1_zCoeffs_found) # Higher order modes set to zero
#             m1_zCoeffs_resid = m1_zCoeffs_found_extnd - m1_zCoeff_initial
#         else: # Model contains more zernike terms than were present in the data
#             m1_zCoeffs_true_extnd = np.zeros(m1_zCoeffs_found.shape[1])
#             m1_zCoeffs_true_extnd = m1_zCoeffs_true_extnd.at[:nZern_data].set(m1_zCoeff_initial)  # Higher order modes set to zero
#             zCoeffs_resid = m1_zCoeffs_found - m1_zCoeffs_true_extnd
#     else:
#         m1_zCoeffs_resid = m1_zCoeffs_found - m1_zCoeff_initial
#         m2_zCoeffs_resid = m2_zCoeffs_found - m2_zCoeff_initial
#
#
#     t1_optim = time.time()
#
#
#     if print2console:
#         # Print Results of Optimization
#         print("Astrometry Retrieval Results:")
#         print("%d iterations in %.3f sec" % (n_iter, t1_optim-t0_optim))
#         print("Final Loss Value: %.5g" % losses[-1])
#         print("Source X, Y Position Error: %.3f uas, %.3f uas" % (posX_resid[-1]*1e6, posY_resid[-1]*1e6))
#         print("Separation Error: %.3f uas" % (sep_resid[-1]*1e6))
#         print("Source Angle Error: %.3f as" % (posAng_resid[-1]*60**2))
#         print("Fractional Flux Error: %.3f ppm A, %.3f ppm B" % (rawFlux_fracResid[-1, 0]*1e6, rawFlux_fracResid[-1, 1]*1e6))
#         print("Recovered Log Flux: %.3f" % logFlux_found[-1])
#         print("Recovered Contrast: %.3f" % contrast_found[-1])
#         print("Fractional Platescale Error: %.3f ppm" % (platescale_resid[-1]/platescale_true*1e6))
#         print("Residual M1 Zernike OPD Error: %.3f nm rms" % m1_total_opd_resid_rms_nm[-1])
#         print("Residual M2 Zernike OPD Error: %.3f nm rms" % m2_total_opd_resid_rms_nm[-1])
#
#         print("Parameter Uncertainties:")
#         print("Source X, Y Position STD: %.3f uas, %.3f uas" % (posX_std*1e6, posY_std*1e6))
#         print("Separation STD: %.3f uas" % (sep_std*1e6))
#         print("Source Angle STD: %.3f as" % (posAng_std*60**2))
#         print("Log Flux STD: %.3f" % logFlux_std)
#         print("Contrast STD: %.3f" % contrast_std)
#         print("M1 Zernike Coeffs STD RMS: %.3f nm" % (nanrms(m1_zCoeffs_std)))
#         print("M2 Zernike Coeffs STD RMS: %.3f nm" % (nanrms(m2_zCoeffs_std)))
#
#
#
#
#     # # Plot the found PSF
#     # cmap = mpl.colormaps['inferno']
#     # plt.figure(figsize=(15, 4))
#     # plt.subplot(1, 3, 1)
#     # plt.imshow(data, extent=psf_extent_as)
#     # plt.title("Data")
#     # plt.xlabel('X (as)')
#     # plt.ylabel('Y (as)')
#     # plt.colorbar(label='Photons')
#     # plt.subplot(1, 3, 2)
#     # plt.imshow(psf_found, extent=psf_extent_as)
#     # plt.title("Recovered PSF")
#     # plt.xlabel('X (as)')
#     # plt.ylabel('Y (as)')
#     # plt.colorbar(label='Photons')
#     # plt.subplot(1, 3, 3)
#     # plt.imshow(psf_found - data, extent=psf_extent_as)
#     # plt.title("Difference Image")
#     # plt.xlabel('X (as)')
#     # plt.ylabel('Y (as)')
#     # plt.colorbar(label='Photons')
#     # plt.show(block=True)
#
#
#     # Plot the optimization results
#     # Show the Losses, Flux, and Position Errors
#     plt.figure(figsize=(16, 13))
#     plt.subplot(3, 2, 1)
#     plt.xlabel("Iteration")
#     plt.ylabel("Loss")
#     plt.plot(np.arange(n_iter)+1, losses)
#     plt.title(f"Losses, Final= %.4g, Ideal= %.4g" % (losses[-1], true_loss))
#     plt.subplot(3, 2, 2)
#     plt.xlabel("Iteration")
#     plt.ylabel("Loss")
#     plt.plot(np.arange(n_iter-10,n_iter)+1, losses[-10:])
#     plt.title("Final 10 Losses")
#     plt.subplot(3, 2, 3)
#     plt.title(f"Stellar Flux Error, Final= (%.3f, %.3f) ppm" % (rawFlux_fracResid[-1, 0]*1e6, rawFlux_fracResid[-1, 1]*1e6))
#     plt.xlabel("Iteration")
#     plt.ylabel("Flux Error (ppm)")
#     plt.plot(np.arange(n_iter)+1, rawFlux_fracResid[:n_iter, 0]*1e6, label="Star A")
#     plt.plot(np.arange(n_iter)+1, rawFlux_fracResid[:n_iter, 1]*1e6, label="Star B")
#     plt.axhline(0, c='k', alpha=0.5)
#     plt.legend(loc="upper right")
#     plt.subplot(3, 2, 4)
#     plt.title(f"Stellar Position Error, Final= (%.3f, %.3f) uas" % (posX_resid[-1]*1e6, posY_resid[-1]*1e6))
#     plt.xlabel("Iteration")
#     plt.ylabel("Position Error (uas)")
#     plt.plot(np.arange(n_iter)+1, posX_resid[:n_iter]*1e6, label="X")
#     plt.plot(np.arange(n_iter)+1, posY_resid[:n_iter]*1e6, label="Y")
#     plt.axhline(0, c='k', alpha=0.5)
#     plt.legend(loc="upper right")
#     plt.subplot(3, 2, 5)
#     plt.title("Source Angle Error, Final= %.3f as" % (posAng_resid[-1]*60**2))
#     plt.xlabel("Iteration")
#     plt.ylabel("Angle Error (as)")
#     plt.plot(np.arange(n_iter)+1, posAng_resid[:n_iter]*60**2)
#     plt.axhline(0, c='k', alpha=0.5)
#     plt.subplot(3, 2, 6)
#     plt.title(f"Separation Error, Final= {sep_resid[-1]*1e6:.3f} uas")
#     plt.xlabel("Iteration")
#     plt.ylabel("Separation Error (uas)")
#     plt.plot(np.arange(n_iter)+1, sep_resid[:n_iter]*1e6)
#     plt.axhline(0, c='k', alpha=0.5)
#     plt.tight_layout()
#     if save_plots and obs_i < N_saved_plots:
#         plot_name = "RecoveredParams"
#         save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
#         plt.savefig(os.path.join(save_path, save_name))
#     if present_plots:
#         plt.show()
#     else:
#         plt.close()
#
#
#     # Zernike Coefficients
#     # noll_index = np.arange(4, np.max(np.concatenate((model_noll_ind, data_noll_ind)))+1)
#     noll_index = m1_noll_ind
#
#     # Set up X Ticks for plotting
#     nticks_max = 12
#     ticks_step = max(1, (noll_index.size-1)//nticks_max + 1)
#     noll_ticks = noll_index[::ticks_step]
#
#
#     # Make a plot showing the retrieved phase
#     vmin = 1e9* np.min(np.array([m1_total_opd_true_tiled, m1_zernike_opd_found]))
#     vmax = 1e9* np.max(np.array([m1_total_opd_true_tiled, m1_zernike_opd_found]))
#
#     plt.figure(figsize=(20, 10))
#     plt.suptitle("M1 Total OPD Recovery")
#     plt.subplot(2, 2, 1)
#     plt.title("M1 Residual OPD Error")
#     plt.xlabel("Iteration")
#     plt.ylabel("M1 RMS Error (nm)")
#     plt.plot(np.arange(n_iter)+1, m1_total_opd_resid_rms_nm)
#     plt.axhline(0, c='k', alpha=0.5)
#     plt.subplot(2, 2, 2)
#     plt.title("Recovered Zernike Coefficients")
#     plt.xlabel("Noll Index")
#     plt.ylabel("Coefficient Amplitude (nm)")
#     plt.scatter(noll_index[:nZern_data], m1_zCoeff_initial, label="True Value", zorder=2)
#     plt.scatter(noll_index[:nZern_model].T, m1_zCoeffs_found[-1,:], label="Recovered Value", marker='x', zorder=3)
#     plt.bar(noll_index.T, m1_zCoeffs_resid[-1,:], label='Residual', zorder=1)
#     plt.axhline(0, c='k', alpha=0.5)
#     plt.xticks(noll_ticks)
#     plt.legend(loc="upper left")
#     ax = plt.subplot(2, 3, 4)
#     plt.title(f"M1 True Total OPD: %.3fnm rms" % m1_total_opd_true_rms_nm)
#     plt.xlabel("X (mm)")
#     im = plt.imshow(1e9* m1_total_opd_true * m1_nanmask, inferno, vmin=vmin, vmax=vmax, extent=m1_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     # cbar.set_label("nm", labelpad=0)
#     ax = plt.subplot(2, 3, 5)
#     plt.title(f"Found OPD: %.3fnm rms" % m1_zernike_opd_found_rms_nm[-1])
#     plt.xlabel("X (mm)")
#     im = plt.imshow(1e9* m1_zernike_opd_found[-1, :, :] * m1_nanmask, inferno, vmin=vmin, vmax=vmax, extent=m1_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     # cbar.set_label("nm", labelpad=0)
#     ax = plt.subplot(2, 3, 6)
#     plt.title(f"OPD Residual: %.3fnm rms" % m1_total_opd_resid_rms_nm[-1])
#     plt.xlabel("X (mm)")
#     im = plt.imshow(1e9* m1_total_opd_resid[-1, :, :] * m1_nanmask, inferno, extent=m1_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     # cbar.set_label("nm", labelpad=0)
#     if save_plots and obs_i < N_saved_plots:
#         plot_name = "M1RecoveredOPD"
#         save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
#         plt.savefig(os.path.join(save_path, save_name))
#     if present_plots:
#         plt.show()
#     else:
#         plt.close()
#
#
#     # Make a plot showing the retrieved phase
#     vmin = 1e9* np.min(np.array([m2_total_opd_true_tiled, m2_zernike_opd_found]))
#     vmax = 1e9* np.max(np.array([m2_total_opd_true_tiled, m2_zernike_opd_found]))
#
#     plt.figure(figsize=(20, 10))
#     plt.suptitle("M2 Total OPD Recovery")
#     plt.subplot(2, 2, 1)
#     plt.title("M2 Residual OPD Error")
#     plt.xlabel("Iteration")
#     plt.ylabel("M2 RMS Error (nm)")
#     plt.plot(np.arange(n_iter)+1, m2_total_opd_resid_rms_nm)
#     plt.axhline(0, c='k', alpha=0.5)
#     plt.subplot(2, 2, 2)
#     plt.title("Recovered Zernike Coefficients")
#     plt.xlabel("Noll Index")
#     plt.ylabel("Coefficient Amplitude (nm)")
#     plt.scatter(noll_index[:nZern_data], m2_zCoeff_initial, label="True Value", zorder=2)
#     plt.scatter(noll_index[:nZern_model].T, m2_zCoeffs_found[-1,:], label="Recovered Value", marker='x', zorder=3)
#     plt.bar(noll_index.T, m2_zCoeffs_resid[-1,:], label='Residual', zorder=1)
#     plt.axhline(0, c='k', alpha=0.5)
#     plt.xticks(noll_ticks)
#     plt.legend(loc="upper left")
#     ax = plt.subplot(2, 3, 4)
#     plt.title(f"M2 True Total OPD: %.3fnm rms" % m2_total_opd_true_rms_nm)
#     plt.xlabel("X (mm)")
#     im = plt.imshow(1e9* m2_total_opd_true * m2_nanmask, inferno, vmin=vmin, vmax=vmax, extent=m2_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     # cbar.set_label("nm", labelpad=0)
#     ax = plt.subplot(2, 3, 5)
#     plt.title(f"Found OPD: %.3fnm rms" % m2_zernike_opd_found_rms_nm[-1])
#     plt.xlabel("X (mm)")
#     im = plt.imshow(1e9* m2_zernike_opd_found[-1, :, :] * m2_nanmask, inferno, vmin=vmin, vmax=vmax, extent=m2_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     # cbar.set_label("nm", labelpad=0)
#     ax = plt.subplot(2, 3, 6)
#     plt.title(f"OPD Residual: %.3fnm rms" % m2_total_opd_resid_rms_nm[-1])
#     plt.xlabel("X (mm)")
#     im = plt.imshow(1e9* m2_total_opd_resid[-1, :, :] * m2_nanmask, inferno, extent=m2_extent_mm)
#     cbar = fig.colorbar(im, cax=merge_cbar(ax))
#     # cbar.set_label("nm", labelpad=0)
#     if save_plots and obs_i < N_saved_plots:
#         plot_name = "M2RecoveredOPD"
#         save_name = f"{script_name}_{plot_name}_{timestamp}_Obs{obs_i+1:0{obs_digits}d}.png"
#         plt.savefig(os.path.join(save_path, save_name))
#     if present_plots:
#         plt.show()
#     else:
#         plt.close()
#
#
#     if save_results:
#         # Construct dicts to save the results
#         final_results = {
#             "Starting RNG Seed": rng0,
#             "N Observations": N_observations,
#             "Optimizer Iterations": n_iter,
#             "Input Source Position X (as)": posX_true,
#             "Input Source Position Y (as)": posY_true,
#             "Input Source Separation (as)": sep_true,
#             "Input Source Angle (deg)": posAng_true,
#             "Input Source Log Flux": logFlux_true,
#             "Input Source Contrast (A:B)": contrast_true,
#             "Input Source A Raw Flux": rawFlux_true[0],
#             "Input Source B Raw Flux": rawFlux_true[1],
#             "Input Platescale (as/pixel)": platescale_true,
#             "Input M1 Zernikes (Noll Index)": ", ".join(map(str, m1_noll_ind)),
#             "Input M1 Zernike Coefficient RMS Amplitude (nm)": m1_zCoeff_amp_rms,
#             "Input M1 Zernike Coefficient Amplitudes (nm)": ", ".join(map(str, m1_zCoeff_initial)),
#             "Input M1 Zernike RMS Error (nm)": m1_zernike_opd_true_rms_nm,
#             "Input M1 Calibrated 1/f Amplitude (nm rms)": m1_calibrated_wfe_amp*1e9,
#             "Input M1 Calibrated 1/f Power Law": m1_calibrated_wfe_alpha,
#             "Input M1 Uncalibrated 1/f Amplitude (nm rms)": m1_uncalibrated_wfe_amp*1e9,
#             "Input M1 Uncalibrated 1/f Power Law": m1_uncalibrated_wfe_alpha,
#             "Input M2 Zernikes (Noll Index)": ", ".join(map(str, m1_noll_ind)),
#             "Input M2 Zernike Coefficient RMS Amplitude (nm)": m2_zCoeff_amp_rms,
#             "Input M2 Zernike Coefficient Amplitudes (nm)": ", ".join(map(str, m2_zCoeff_initial)),
#             "Input M2 Zernike RMS Error (nm)": m2_zernike_opd_true_rms_nm,
#             "Input M2 Calibrated 1/f Amplitude (nm rms)": m2_calibrated_wfe_amp*1e9,
#             "Input M2 Calibrated 1/f Power Law": m2_calibrated_wfe_alpha,
#             "Input M2 Uncalibrated 1/f Amplitude (nm rms)": m2_uncalibrated_wfe_amp*1e9,
#             "Input M2 Uncalibrated 1/f Power Law": m2_uncalibrated_wfe_alpha,
#             "Photon Noise": add_shot_noise,
#             "Read Noise (e- / frame)": sigma_read,
#             "Exposure (s / frame)": exposure_per_frame,
#             "Coadded Frames": N_frames,
#             "Jitter Amplitude (as)": jitter_amplitude,
#
#             "Optimiser": optimiser_label,
#
#             "Found Source Position X (as)": posX_found[-1],
#             "Found Source Position Y (as)": posY_found[-1],
#             "Found Source Separation (as)": sep_found[-1],
#             "Found Source Angle (deg)": posAng_found[-1],
#             "Found Source Log Flux": logFlux_found[-1],
#             "Found Source Contrast (A:B)": contrast_found[-1],
#             "Found Source A Raw Flux": rawFlux_found[-1, 0],
#             "Found Source B Raw Flux": rawFlux_found[-1, 1],
#             "Found Platescale (as/pixel)": platescale_found[-1],
#             "Found Zernikes (Noll Index)": ", ".join(map(str, m1_noll_ind)),
#             "Found M1 Zernike Coefficient Amplitudes (nm)": ", ".join(map(str, m1_zCoeffs_found[-1,:])),
#             "Found M2 Zernike Coefficient Amplitudes (nm)": ", ".join(map(str, m2_zCoeffs_found[-1, :])),
#
#             "Residual Source Position X (uas)": posX_resid[-1] * 1e6,
#             "Residual Source Position Y (uas)": posY_resid[-1] * 1e6,
#             "Residual Source Separation (uas)": sep_resid[-1] * 1e6,
#             "Residual Source Angle (as)": posAng_resid[-1] * 60 ** 2,
#             "Residual Source A Flux (ppm)": rawFlux_fracResid[-1, 0] * 1e6,
#             "Residual Source B Flux (ppm)": rawFlux_fracResid[-1, 1] * 1e6,
#             "Residual Platescale (ppm)": platescale_resid[-1] / platescale_true * 1e6,
#             "Residual M1 Zernike RMS Error (nm)": m1_zernike_opd_resid_rms_nm[-1],
#             "Residual M2 Zernike RMS Error (nm)": m2_zernike_opd_resid_rms_nm[-1],
#
#         }
#
#
#         # Construct Data Frames
#         final_results_df = pd.DataFrame([final_results])
#
#         # Construct Save Name
#         save_name = f"{script_name}_{timestamp}.xlsx"
#
#         if obs_i == 0: # Create a new file for the first simulation
#             final_results_df.to_excel(os.path.join(save_path, save_name), index=False, sheet_name="Results")
#         else: # Append results for subsequent simulations
#             with pd.ExcelWriter(os.path.join(save_path, save_name), engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
#                 final_results_df.to_excel(writer, index=False, header=False, sheet_name="Results", startrow=int(obs_i)+1)
#
#         print(f"Simulation {obs_i+1}/{N_observations} results saved to {save_name}")
#
#
#     t1_simulation = time.time()
#     print("Simulation %d finished in %.3f sec" % (obs_i+1, t1_simulation-t0_simulation))
#


t1_script = time.time()
print("Script finished in %.3f sec" % (t1_script-t0_script))
