import dLux as dl
import dLuxToliman as dlT
import jax.numpy as np
from Classes.optical_systems import SheraThreePlaneSystem

__all__ = [
    "SheraThreePlane_ForwardModel"
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

    # Extract key parameters
    m1_diameter = params.get("m1_diameter")
    m2_diameter = params.get("m2_diameter")
    m1_focal_length = params.get("m1_focal")
    m2_focal_length = params.get("m2_focal")
    plane_separation = params.get("plane_separation")
    pixel_size = params.get("pixel_size")

    pupil_npix = params.get("pupil_npix")
    psf_npix = params.get("psf_npix")

    # Source parameters
    x_position = params.get("x_position")
    y_position = params.get("y_position")
    separation = params.get("separation")
    angle = params.get("angle")
    contrast = params.get("contrast")
    wavelength = params.get("wavelength")
    bandwidth = params.get("bandwidth")
    n_wavelengths = params.get("n_wavelengths")
    exposure_time = params.get("exposure_time")
    # frame_rate = params.get("frame_rate")

    # M1 aberrations
    m1_zernike_noll = params.get("m1_zernike_noll")
    m1_zernike_amp = params.get("m1_zernike_amp")

    # M2 aberrations
    # m2_zernike_noll = params.get("m2_zernike_noll")
    m2_zernike_amp = params.get("m2_zernike_amp")

    # Initialize the optical system
    model_optics = SheraThreePlaneSystem(
        wf_npixels = pupil_npix,
        psf_npixels = psf_npix,
        oversample = 3,
        detector_pixel_pitch = pixel_size,
        noll_indices = m1_zernike_noll,
        m1_diameter = m1_diameter,
        m2_diameter = m2_diameter,
        m1_focal_length = m1_focal_length,
        m2_focal_length = m2_focal_length,
        m1_m2_separation = plane_separation
    )

    # Normalise the zernike basis to be in units of nm
    model_optics = model_optics.multiply('m1_aperture.basis', 1e-9)
    model_optics = model_optics.multiply('m2_aperture.basis', 1e-9)

    # Set Zernike coefficients (units of nm)
    model_optics = model_optics.set('m1_aperture.coefficients', m1_zernike_amp)
    model_optics = model_optics.set('m2_aperture.coefficients', m2_zernike_amp)

    # Scale the source flux from exposure time
    # Default fluxes taken from Toliman Master spreadsheet
    starA_default_flux = 1.267e11  # photons / second of exposure / square meter of aperture / micron of band
    starB_default_flux = 4.557e10  # photons / second of exposure / square meter of aperture / micron of band
    default_total_flux = starA_default_flux + starB_default_flux
    aperture_area = np.pi * (
                model_optics.p1_diameter / 2) ** 2  # square meters of aperture (doesn't include M2 obscuration)
    bandpass = (wavelength - bandwidth / 2, wavelength + bandwidth / 2)
    total_flux = default_total_flux * exposure_time * aperture_area * (bandwidth / 1000)  # photons
    total_log_flux = np.log10(total_flux)

    # Initialize the source
    source = dlT.AlphaCen(
        n_wavels = n_wavelengths,
        x_position = x_position,
        y_position = y_position,
        separation = separation,
        position_angle = angle,
        log_flux = total_log_flux,
        contrast = contrast,
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
