import jax.numpy as np
import numpyro as npy
import numpyro.distributions as dist

__all__ = ["SheraThreePlane_NumpyroModel"]

def SheraThreePlane_NumpyroModel(data, model):
    """
    Numpyro model for recovering optical system parameters from noisy PSF data.

    Parameters
    ----------
    data : np.ndarray
        The observed PSF data.
    model : SheraThreePlane_Model
        The initialized dl.Telescope model object, with source, optics, and detector.

    Returns
    -------
    None
        Defines the Numpyro model and sets up the likelihood for inference.
    """
    # Define the parameter paths within the model
    parameters = [
        "x_position",
        "y_position",
        "separation",
        "position_angle",
        "contrast",
        "log_flux",
        "m1_aperture.coefficients",
        "m2_aperture.coefficients"
    ]

    # Sample free parameters
    values = []
    for path in parameters:
        # Extract the current value from the model
        current_value = model.get(path)

        # Define the sampling distributions
        if path in ["x_position", "y_position"]:
            sample_value = npy.sample(path, dist.Normal(current_value, 0.1))
        elif path == "separation":
            sample_value = npy.sample(path, dist.Normal(current_value, 0.001))
        elif path == "position_angle":
            sample_value = npy.sample(path, dist.Uniform(current_value - 5, current_value + 5))
        elif path == "contrast":
            sample_value = npy.sample(path, dist.LogNormal(np.log(current_value), 0.1))
        elif path == "log_flux":
            sample_value = npy.sample(path, dist.Normal(current_value, 0.5))
        elif path in ["m1_aperture.coefficients", "m2_aperture.coefficients"]:
            n_coeffs = len(current_value)
            sample_value = npy.sample(path, dist.Normal(current_value, 1).expand([n_coeffs]).to_event(1))

        # Append the sampled value
        values.append(sample_value)

    # Updated for consistency with HMC example
    with npy.plate("data", len(data.flatten())):
        poisson_model = dist.Poisson(
            model.set(parameters, values).model().flatten())
        return npy.sample("psf", poisson_model, obs=data.flatten())


