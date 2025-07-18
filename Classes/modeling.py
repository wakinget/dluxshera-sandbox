import dLux as dl
import dLuxToliman as dlT
import dLux.layers as dll
import jax.numpy as np
import jax.random as jr
from Classes.optical_systems import SheraThreePlaneSystem, JNEXTOpticalSystem
from Classes.optimization import SheraTwoPlaneParams, SheraThreePlaneParams
from Classes.oneoverf import *
from Classes.utils import nanrms, set_array, calculate_log_flux
from Classes.plotting import merge_cbar

__all__ = [
    "SheraThreePlane_ForwardModel",
    "SheraThreePlane_Model"
]


def SheraThreePlane_ForwardModel(params, return_model=False):
    """
    Builds a SheraThreePlane Optical Model using the given params,
    Generates a PSF for the given parameters, optionally outputs the model itself

    Parameters
    ----------
    params : SheraThreePlaneParams
        The optical system parameters, including telescope design, sampling,
        source properties, and aberrations.

    return_model : bool, optional
        If True, return the full Telescope model object along with the PSF.

    Returns
    -------
    psf : np.ndarray
        The generated PSF based on the current system parameters.

    model : dl.Telescope, optional
        The full Telescope model object, returned if return_model=True.
    """
    # A note: M2 zernike nolls are currently assumed to be the same as M1 zernike nolls.
    # I will need to modify the SheraThreePlaneOpticalSystem to fix this.

    # Initialize the optical system given input params
    model_optics = SheraThreePlaneSystem(
        wf_npixels = params.get("pupil_npix"),
        psf_npixels = params.get("psf_npix"),
        oversample = 3,
        detector_pixel_pitch = params.get("pixel_size"),
        noll_indices = params.get("m1_zernike_noll"),
        p1_diameter = params.get("m1_diameter"),
        p2_diameter = params.get("m2_diameter"),
        m1_focal_length = params.get("m1_focal"),
        m2_focal_length = params.get("m2_focal"),
        plane_separation = params.get("plane_separation")
    )

    # Normalise the zernike basis to be in units of nm
    model_optics = model_optics.multiply('m1_aperture.basis', 1e-9)
    model_optics = model_optics.multiply('m2_aperture.basis', 1e-9)

    # Set Zernike coefficients (units of nm)
    model_optics = model_optics.set('m1_aperture.coefficients', params.get("m1_zernike_amp"))
    model_optics = model_optics.set('m2_aperture.coefficients', params.get("m2_zernike_amp"))

    # Initialize the source
    wavelength = params.get("wavelength")
    bandwidth = params.get("bandwidth")
    bandpass = (wavelength - bandwidth / 2, wavelength + bandwidth / 2)
    source = dlT.AlphaCen(
        n_wavels = params.get("n_wavelengths"),
        x_position = params.get("x_position"),
        y_position = params.get("y_position"),
        separation = params.get("separation"),
        position_angle = params.get("position_angle"),
        log_flux = params.get("log_flux"),
        contrast = params.get("contrast"),
        bandpass = bandpass
    )

    # Initialize the detector (no jitter for now)
    detector = dl.LayeredDetector(
        layers=[("downsample", dl.Downsample(model_optics.oversample))]
    )

    # Combine into a full telescope model
    model = dl.Telescope(
        source=source,
        optics=model_optics,
        detector=detector,
    )

    # Generate the PSF
    psf = model.model()

    # Return the PSF and optionally the full model object
    if return_model:
        return psf, model
    else:
        return psf


class SheraThreePlane_Model(dl.Telescope):
    """
    Builds a SheraThreePlane Optical Model using the given params.
    Outputs the telescope model.

    Parameters
    ----------
    params : SheraThreePlaneParams (optional)
        The optical system parameters, including telescope design, sampling,
        source properties, and aberrations. Default values are used if unspecified

    Returns
    -------
    model : dl.Telescope
        The full Telescope model object, with source, optics, and detector.
    """

    def __init__(self, params=None):
        if params is None:
            params = SheraThreePlaneParams()

        # Initialize the optical system given input params
        model_optics = self._initialize_optics(params)

        # Initialize the source
        source = self._initialize_source(params)

        # Initialize the detector (no jitter for now)
        detector = dl.LayeredDetector(
            layers=[("downsample", dl.Downsample(model_optics.oversample))]
        )

        # Initialize the parent Telescope class
        super().__init__(source=source, optics=model_optics, detector=detector)

    def _initialize_optics(self, params):
        """
        Initialize the optical system.
        """
        optics = SheraThreePlaneSystem(
            wf_npixels = params.get("pupil_npix"),
            psf_npixels = params.get("psf_npix"),
            oversample = 1,
            detector_pixel_pitch = params.get("pixel_size"),
            m1_noll_ind = params.get("m1_zernike_noll"),
            m2_noll_ind = params.get("m2_zernike_noll"),
            p1_diameter = params.get("p1_diameter"),
            p2_diameter = params.get("p2_diameter"),
            m1_focal_length = params.get("m1_focal_length"),
            m2_focal_length = params.get("m2_focal_length"),
            plane_separation = params.get("plane_separation")
        )

        # Normalize the Zernike basis so that the coefficients are scaled to units of nm
        if params.get("m1_zernike_noll") is not None:
            optics = optics.multiply('m1_aperture.basis', 1e-9)
            optics = optics.set('m1_aperture.coefficients', params.get("m1_zernike_amp"))
        if params.get("m2_zernike_noll") is not None:
            optics = optics.multiply('m2_aperture.basis', 1e-9)
            optics = optics.set('m2_aperture.coefficients', params.get("m2_zernike_amp"))

        # Initialize the Calibrated 1/f WFE
        # First for the Primary
        rng_key, subkey = jr.split(jr.PRNGKey(params.get("rng_seed")))
        m1_cal_wfe = oneoverf_noise_2D(optics.wf_npixels, params.get("m1_calibrated_power_law"), key=subkey)
        m1_cal_wfe = remove_PTT(m1_cal_wfe, optics.m1_aperture.transmission.astype(bool))  # Remove PTT from aperture
        m1_cal_wfe = m1_cal_wfe * (params.get("m1_calibrated_amplitude") / nanrms(
            m1_cal_wfe[optics.m1_aperture.transmission.astype(bool)]))  # Scale the 1/f noise map over the aperture
        m1_cal_layer = dll.AberratedLayer(m1_cal_wfe)
        optics = optics.insert_layer(('m1_calibration', m1_cal_layer), 3, 0)

        # Now for the Secondary
        rng_key, subkey = jr.split(rng_key)
        m2_cal_wfe = oneoverf_noise_2D(optics.wf_npixels, params.get("m2_calibrated_power_law"), key=subkey)
        m2_cal_wfe = remove_PTT(m2_cal_wfe, optics.m2_aperture.transmission.astype(bool))  # Remove PTT from aperture
        m2_cal_wfe = m2_cal_wfe * (params.get("m2_calibrated_amplitude") / nanrms(
            m2_cal_wfe[optics.m2_aperture.transmission.astype(bool)]))  # Scale the 1/f noise map over the aperture
        m2_cal_layer = dll.AberratedLayer(m2_cal_wfe)
        optics = optics.insert_layer(('m2_calibration', m2_cal_layer), 3, 1)

        # Initialize the Uncalibrated 1/f WFE
        # First for the Primary
        rng_key, subkey = jr.split(rng_key)
        m1_uncal_wfe = oneoverf_noise_2D(optics.wf_npixels, params.get("m1_uncalibrated_power_law"), key=subkey)
        m1_uncal_wfe = remove_PTT(m1_uncal_wfe, optics.m1_aperture.transmission.astype(bool))  # Remove PTT from aperture
        m1_uncal_wfe = m1_uncal_wfe * (params.get("m1_uncalibrated_amplitude") / nanrms(
            m1_uncal_wfe[optics.m1_aperture.transmission.astype(bool)]))  # Scale the 1/f noise map over the aperture
        m1_uncal_layer = dll.AberratedLayer(m1_uncal_wfe)
        optics = optics.insert_layer(('m1_wfe', m1_uncal_layer), 4, 0)

        # Now for the Secondary
        rng_key, subkey = jr.split(rng_key)
        m2_uncal_wfe = oneoverf_noise_2D(optics.wf_npixels, params.get("m2_uncalibrated_power_law"), key=subkey)
        m2_uncal_wfe = remove_PTT(m2_uncal_wfe, optics.m2_aperture.transmission.astype(bool))  # Remove PTT from aperture
        m2_uncal_wfe = m2_uncal_wfe * (params.get("m2_uncalibrated_amplitude") / nanrms(
            m2_uncal_wfe[optics.m2_aperture.transmission.astype(bool)]))  # Scale the 1/f noise map over the aperture
        m2_uncal_layer = dll.AberratedLayer(m2_uncal_wfe)
        optics = optics.insert_layer(('m2_wfe', m2_uncal_layer), 4, 1)

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
            "m1_zernike_amp": "m1_aperture.coefficients",
            "m2_zernike_amp": "m2_aperture.coefficients",
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

        # Initialize a new SheraThreePlaneParams object
        extracted_params = SheraThreePlaneParams(point_design=pd)

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

        optics = JNEXTOpticalSystem(
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

