# src/dluxshera/inference/inference.py
from __future__ import annotations

from typing import Sequence, Optional, Dict, Any, Tuple

import jax.numpy as jnp
import numpyro as npy
import numpyro.distributions as dist

from ..optics.config import SheraThreePlaneConfig, SHERA_TESTBED_CONFIG
from ..params.spec import ParamSpec, ParamKey, build_inference_spec_basic
from ..params.store import ParameterStore
from .optimization import run_image_gd, NoiseModel

__all__ = [
    "run_shera_image_gd_basic",
]


# ---------------------------------------------------------------------
# High-level convenience wrapper: image-based GD for SHERA
# ---------------------------------------------------------------------

def run_shera_image_gd_basic(
    data: jnp.ndarray,
    var: jnp.ndarray,
    *,
    cfg: SheraThreePlaneConfig = SHERA_TESTBED_CONFIG,
    infer_keys: Sequence[ParamKey] = (
        "binary.separation_as",
        "binary.x_position",
        "binary.y_position",
    ),
    init_overrides: Optional[Dict[ParamKey, Any]] = None,
    noise_model: NoiseModel = "gaussian",
    learning_rate: float = 1e-2,
    num_steps: int = 50,
) -> Tuple[jnp.ndarray, ParameterStore, Dict[str, jnp.ndarray]]:
    """
    Convenience front-end for running image-based gradient descent
    on a SHERA three-plane model.

    This wraps the lower-level `run_image_gd` function by:
      - constructing a default inference spec,
      - building a ParameterStore from that spec's defaults,
      - applying optional overrides for the initial store,
      - and running θ-space gradient descent using the Binder-based engine.

    Parameters
    ----------
    data :
        Observed image (PSF cutout) as a JAX array.
    var :
        Per-pixel variance map (same shape as `data`). For Poisson noise,
        this is ignored but kept for API compatibility.
    cfg :
        SheraThreePlaneConfig describing the optical design and sampling.
        Defaults to SHERA_TESTBED_CONFIG.
    infer_keys :
        Sequence of parameter keys (in the ParameterStore) to infer.
        Defaults to separation and centroid:
          ("binary.separation_as", "binary.x_position", "binary.y_position")
    init_overrides :
        Optional dict of {ParamKey: value} used to override the default
        ParameterStore initialisation (e.g. to start from a biased guess
        for separation or centroid).
    noise_model :
        "gaussian" (default) or "poisson" – selects which likelihood kernel
        to use inside the NLL.
    learning_rate :
        Step size for the internal Adam optimizer.
    num_steps :
        Number of gradient descent iterations.

    Returns
    -------
    theta_final :
        Final packed θ vector (1D array) corresponding to `infer_keys`.
    store_final :
        ParameterStore with the inferred keys updated to θ_final.
    history :
        Dict with at least:
          - "loss": 1D array of loss values per step.
    """
    # 1) Build a “basic” inference spec and default store
    inference_spec: ParamSpec = build_inference_spec_basic()
    store_init: ParameterStore = ParameterStore.from_spec_defaults(inference_spec)

    # 2) Apply user-provided initial overrides, if any
    if init_overrides is not None:
        store_init = store_init.replace(init_overrides)

    # 3) Delegate to the Binder-based θ-space GD engine
    theta_final, store_final, history = run_image_gd(
        cfg,
        inference_spec,
        store_init,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        learning_rate=learning_rate,
        num_steps=num_steps,
    )

    return theta_final, store_final, history


# ---------------------------------------------------------------------
# Legacy / experimental NumPyro model
#
# NOTE:
# - This is *not* wired into the new ParameterStore/Binder framework.
# - It remains here as a reference for how a NumPyro model might look.
# - In future, we’ll likely replace this with a Binder-based version
#   that consumes (cfg, spec, store) and uses the same forward model
#   as `run_image_gd`.
# ---------------------------------------------------------------------

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
            sample_value = npy.sample(path, dist.Uniform(current_value - 1, current_value + 1))
        elif path == "contrast":
            sample_value = npy.sample(path, dist.LogNormal(jnp.log(current_value), 0.05))
        elif path == "log_flux":
            sample_value = npy.sample(path, dist.LogNormal(jnp.log(current_value), 0.05))
        elif path in ["m1_aperture.coefficients", "m2_aperture.coefficients"]:
            n_coeffs = len(current_value)
            sample_value = npy.sample(path, dist.Normal(current_value, 0.1).expand([n_coeffs]).to_event(1))

        # Append the sampled value
        values.append(sample_value)

    # Updated for consistency with HMC example
    with npy.plate("data", len(data.flatten())):
        poisson_model = dist.Poisson(
            model.set(parameters, values).model().flatten())
        return npy.sample("psf", poisson_model, obs=data.flatten())


