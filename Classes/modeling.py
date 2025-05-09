import dLux as dl
import dLuxToliman as dlT
import jax.numpy as np
from Classes.optical_systems import SheraThreePlaneSystem

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
        m1_diameter = params.get("m1_diameter"),
        m2_diameter = params.get("m2_diameter"),
        m1_focal_length = params.get("m1_focal"),
        m2_focal_length = params.get("m2_focal"),
        m1_m2_separation = params.get("plane_separation")
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
        position_angle = params.get("angle"),
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


def SheraThreePlane_Model(params):
    """
    Builds a SheraThreePlane Optical Model using the given params,
    Outputs the telescope model

    Parameters
    ----------
    params : SheraThreePlaneParams
        The optical system parameters, including telescope design, sampling,
        source properties, and aberrations.

    Returns
    -------
    model : dl.Telescope
        The full Telescope model object, with source, optics, and detector
    """

    # Initialize the optical system given input params
    model_optics = SheraThreePlaneSystem(
        wf_npixels = params.get("pupil_npix"),
        psf_npixels = params.get("psf_npix"),
        oversample = 3,
        detector_pixel_pitch = params.get("pixel_size"),
        noll_indices = params.get("m1_zernike_noll"),
        m1_diameter = params.get("m1_diameter"),
        m2_diameter = params.get("m2_diameter"),
        m1_focal_length = params.get("m1_focal"),
        m2_focal_length = params.get("m2_focal"),
        m1_m2_separation = params.get("plane_separation")
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
        position_angle = params.get("angle"),
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

    # Return the full model object
    return model
