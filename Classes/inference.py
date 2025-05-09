import jax.numpy as np
import numpyro as npy
import numpyro.distributions as dist
from Classes.modeling import SheraThreePlane_ForwardModel


def SheraThreePlane_NumpyroModel(data, params):
    """
    Numpyro model for recovering optical system parameters from noisy PSF data.

    Parameters
    ----------
    data : np.ndarray
        The observed PSF data.
    params : SheraThreePlaneParams
        The parameter set for the optical system.

    Returns
    -------
    None
        Defines the Numpyro model and sets up the likelihood for inference.
    """
    # Get the list of free parameters
    free_params = params.get_free_params()

    # Define the paths within the SheraThreePlaneParams object
    parameter_paths = {
        "x_position": "x_position",
        "y_position": "y_position",
        "separation": "separation",
        "position_angle": "position_angle",
        "contrast": "contrast",
        "log_flux": "log_flux",
        "m1_zernike_amp": "m1_aperture.coefficients",
        "m2_zernike_amp": "m2_aperture.coefficients"
    }

    # Sample free parameters and inject them into the params object
    sampled_params = {}
    for param_name, path in parameter_paths.items():
        if path in free_params:
            # Use the current parameter value as the mean
            current_value = params.get(path)

            if param_name in ["x_position", "y_position"]:
                sampled_params[path] = npy.sample(param_name, dist.Normal(current_value, 0.1))
            elif param_name == "separation":
                sampled_params[path] = npy.sample(param_name, dist.Normal(current_value, 0.001))
            elif param_name == "position_angle":
                sampled_params[path] = npy.sample(param_name, dist.Uniform(current_value - 5, current_value + 5))
            elif param_name == "contrast":
                sampled_params[path] = npy.sample(param_name, dist.LogNormal(np.log(current_value), 0.1))
            elif param_name == "log_flux":
                sampled_params[path] = npy.sample(param_name, dist.Normal(current_value, 0.5))
            elif param_name in ["m1_zernike_amp", "m2_zernike_amp"]:
                # Use the current coefficient array as the mean
                sampled_params[path] = npy.sample(param_name, dist.Normal(current_value, 10).expand([len(current_value)]).to_event(1))

    # Update the params object with the sampled values
    params.replace(sampled_params)

    # Generate the PSF using the forward model
    psf = SheraThreePlane_ForwardModel(params)

    # Define the likelihood using a plate for efficient vectorization
    with npy.plate("pixels", psf.size):
        npy.sample("obs", dist.Poisson(psf.flatten()), obs=data.flatten())