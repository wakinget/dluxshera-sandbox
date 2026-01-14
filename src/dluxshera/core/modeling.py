# src/dluxshera/core/modeling.py

from __future__ import annotations

import dLuxToliman as dlT

from ..optics.config import SheraThreePlaneConfig
from ..optics.builder import build_shera_threeplane_optics
from ..params.spec import ParamSpec
from ..params.store import ParameterStore
from .universe import build_alpha_cen_source


import dLux as dl
import dLux.layers as dll
import jax.numpy as np
import jax.random as jr
from ..optics.optical_systems import SheraThreePlaneOptics, SheraTwoPlaneOptics
from ..legacy.params import SheraTwoPlaneParams
from ..utils.oneoverf import oneoverf_noise_2D, remove_PTT
from ..utils.utils import nanrms


class SheraTwoPlane_Model(dl.Telescope):
    """
    Builds a 2-Plane (pupil + focal) Optical Model using the given params.
    Outputs the telescope model.

    Parameters
    ----------
    params : SheraTwoPlaneParams (optional)
        The optical system parameters, including telescope design, sampling,
        source properties, and aberrations. Default values are used if unspecified.

    Returns
    -------
    model : dl.Telescope
        The full Telescope model object, with source, optics, and detector.
    """

    def __init__(self, params=None):
        if params is None:
            params = SheraTwoPlaneParams()

        # Initialize the optical system given input params
        optics = self._initialize_optics(params)

        # Initialize the source
        source = self._initialize_source(params)

        # Initialize the detector (no jitter for now)
        detector = dl.LayeredDetector(
            layers=[("downsample", dl.Downsample(optics.oversample))]
        )

        # Initialize the parent Telescope class
        super().__init__(source=source, optics=optics, detector=detector)

    def _initialize_optics(self, params):
        """
        Initialize the optical system.
        """

        optics = SheraTwoPlaneOptics(
            wf_npixels = params.get("pupil_npix"),
            psf_npixels = params.get("psf_npix"),
            oversample = 1,
            psf_pixel_scale = params.get("psf_pixel_scale"),
            noll_indices = params.get("zernike_noll"),
            m1_diameter = params.get("p1_diameter"),
            m2_diameter = params.get("p2_diameter"),
            n_struts = 4,
            strut_width = 0.002,
            strut_rotation = -np.pi / 4,
            dp_design_wavel = params.get("wavelength")*1e-9
        )

        # Normalize the Zernike basis to be in units of nm
        optics = optics.multiply('aperture.basis', 1e-9)

        # Set Zernike coefficients (units of nm)
        optics = optics.set('aperture.coefficients', params.get("zernike_amp"))

        # Initialize the Calibrated 1/f WFE
        rng_key, subkey = jr.split(jr.PRNGKey(params.get("rng_seed")))
        cal_wfe = oneoverf_noise_2D(optics.wf_npixels, params.get("calibrated_power_law"), key=subkey)
        cal_wfe = remove_PTT(cal_wfe, optics.aperture.transmission.astype(bool))  # Remove PTT from aperture
        cal_wfe = cal_wfe * (params.get("calibrated_amplitude") / nanrms(
            cal_wfe[optics.aperture.transmission.astype(bool)]))  # Scale the 1/f noise map over the aperture
        cal_layer = dll.AberratedLayer(cal_wfe)
        optics = optics.insert_layer(('calibration', cal_layer), 3)

        # Initialize the Uncalibrated 1/f WFE
        rng_key, subkey = jr.split(rng_key)
        uncal_wfe = oneoverf_noise_2D(optics.wf_npixels, params.get("uncalibrated_power_law"), key=subkey)
        uncal_wfe = remove_PTT(uncal_wfe, optics.aperture.transmission.astype(bool))  # Remove PTT from aperture
        uncal_wfe = uncal_wfe * (params.get("uncalibrated_amplitude") / nanrms(
            uncal_wfe[optics.aperture.transmission.astype(bool)]))  # Scale the 1/f noise map over the aperture
        uncal_layer = dll.AberratedLayer(uncal_wfe)
        optics = optics.insert_layer(('wfe', uncal_layer), 4)
        return optics

    def _initialize_source(self, params):
        """
        Initialize the source.
        """
        wavelength = params.get("wavelength")  # Central wavelength (nm)
        bandwidth = params.get("bandwidth")  # Bandwidth (nm)
        bandpass = (wavelength - bandwidth / 2, wavelength + bandwidth / 2)
        return dlT.AlphaCen(
            n_wavels = params.get("n_wavelengths"),
            x_position = params.get("x_position"),
            y_position = params.get("y_position"),
            separation = params.get("separation"),
            position_angle = params.get("position_angle"),
            log_flux = params.get("log_flux"),
            contrast = params.get("contrast"),
            bandpass = bandpass
        )

    @staticmethod
    def get_param_path_map():
        """
        Returns the parameter path that maps params from this class to the parameters of the model.
        """
        return {
            "x_position": "x_position",
            "y_position": "y_position",
            "separation": "separation",
            "position_angle": "position_angle",
            "contrast": "contrast",
            "log_flux": "log_flux",
            "zernike_amp": "aperture.coefficients",
            "pupil_npix": "wf_npix",
            "psf_npix": "psf_npixels"
        }

    @staticmethod
    def get_param_transform_map():
        """
        Returns a mapping of parameter names to custom transformation functions.
        These functions are used to convert model attributes into parameter values
        when extracting parameters from the model.
        """
        return {
            "bandwidth": lambda model: np.diff(np.array(model.bandpass)),
            "wavelength": lambda model: np.mean(model.wavelengths),
            "n_wavelengths": lambda model: model.wavelengths.size,
        }

    def extract_params(self):
        """
        Extract the current parameters from this SheraThreePlane_Model instance.

        Returns
        -------
        SheraThreePlaneParams
            A new SheraThreePlaneParams object populated with the current model parameters.
        """

        # Determine the point design from the model diameter
        if self.diameter == 0.09:
            pd = "shera_testbed"
        elif self.diameter == 0.22:
            pd = "shera_flight"

        # Initialize a new SheraTwoPlaneParams object
        extracted_params = SheraTwoPlaneParams(point_design=pd)

        # Retrieve the parameter path map and transformation map
        param_path_map = self.get_param_path_map()
        param_transform_map = self.get_param_transform_map()

        # Extract all parameters from the model
        for param_key in extracted_params.keys:
            try:
                # Use a custom transformation function if available
                if param_key in param_transform_map:
                    value = param_transform_map[param_key](self)
                else:
                    # Use the model path if available, otherwise fall back to the param_key
                    model_path = param_path_map.get(param_key, param_key)
                    value = self.get(model_path)

                # Set the extracted value
                extracted_params = extracted_params.set(param_key, value)
            except (AttributeError, KeyError, ValueError):
                # Skip parameters that are not present in the model
                continue

        return extracted_params


#------------------------------
#-  Parameter Refactor Code   -
#------------------------------

@dataclass(frozen=True)
class SheraThreePlaneComponents:
    """
    Lightweight container bundling together the core Shera three-plane
    objects for a single configuration + parameter state.

    This is intentionally minimal and does *not* embed any inference logic;
    it just packages:

      - the structural config (geometry, grids, bandpass, etc.),
      - the ParamSpec used to define the inference-level parameter schema,
      - the ParameterStore holding the current parameter values,
      - the three-plane optics object,
      - the AlphaCen source object.

    Higher-level code (e.g. inference routines, model wrappers) can consume
    this bundle and decide how to run forward models, attach likelihoods,
    etc., without having to know how to construct the pieces.
    """

    cfg: SheraThreePlaneConfig
    spec: ParamSpec
    store: ParameterStore

    optics: SheraThreePlaneOptics
    source: dlT.AlphaCen


def build_shera_threeplane_components(
    cfg: SheraThreePlaneConfig,
    spec: ParamSpec,
    store: ParameterStore,
) -> SheraThreePlaneComponents:
    """
    Construct the core Shera three-plane components (optics + source) from
    a config, spec, and ParameterStore.

    This is the P0 "end-to-end" builder that ties together:

      - structural configuration (SheraThreePlaneConfig),
      - parameter schema (ParamSpec),
      - parameter values (ParameterStore),

    and returns a single, immutable bundle that can be used by downstream
    modeling and inference code.

    Parameters
    ----------
    cfg :
        Structural configuration of the three-plane optics (geometry,
        grids, bandpass, Zernike basis structure, diffractive pupil, etc.).

    spec :
        ParamSpec that defines the valid keys and basic metadata for the
        inference-level parameters in the store.

    store :
        ParameterStore holding the current values for the inference-level
        parameters (binary separation/PA, centroid, flux, Zernike
        coefficients, etc.).

    Returns
    -------
    SheraThreePlaneComponents
        A dataclass bundling together `cfg`, `spec`, `store`, the
        `SheraThreePlaneOptics` optics object, and the `AlphaCen` source
        object.
    """
    # Ensure the store is consistent with the spec before using it.
    store = store.validate_against(spec)

    # Optics: three-plane system with Zernike coefficients injected from store.
    optics = build_shera_threeplane_optics(cfg, store=store, spec=spec)

    # Source: AlphaCen built from the same ParameterStore.
    source = build_alpha_cen_source(store, n_wavels=cfg.n_lambda)

    return SheraThreePlaneComponents(
        cfg=cfg,
        spec=spec,
        store=store,
        optics=optics,
        source=source,
    )
