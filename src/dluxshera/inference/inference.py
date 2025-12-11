# src/dluxshera/inference/inference.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any, Tuple

import jax
import jax.numpy as jnp
import numpyro as npy
import numpyro.distributions as dist
import optax

from ..optics.config import SheraThreePlaneConfig, SHERA_TESTBED_CONFIG
from ..params.spec import ParamSpec, ParamKey, build_forward_model_spec_from_config
from ..params.store import ParameterStore, refresh_derived
from ..params.transforms import DEFAULT_SYSTEM_ID, TRANSFORMS
from .optimization import (
    EigenThetaMap,
    fim_theta,
    make_binder_image_nll_fn,
    run_image_gd,
    run_simple_gd,
    NoiseModel,
)

__all__ = [
    "run_shera_image_gd_basic",
    "run_shera_image_gd_eigen",
    "EigenGdResults",
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
        "binary.x_position_as",
        "binary.y_position_as",
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
          ("binary.separation_as", "binary.x_position_as", "binary.y_position_as")
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
    # 1) Build a forward spec and default store (deriveds populated)
    forward_spec: ParamSpec = build_forward_model_spec_from_config(cfg)
    store_init: ParameterStore = ParameterStore.from_spec_defaults(forward_spec)

    # 2) Apply user-provided initial overrides, if any
    if init_overrides is not None:
        store_init = store_init.replace(init_overrides)

    store_init = refresh_derived(
        store_init, forward_spec, TRANSFORMS, system_id=DEFAULT_SYSTEM_ID
    )

    # 3) Delegate to the Binder-based θ-space GD engine
    theta_final, store_final, history = run_image_gd(
        cfg,
        forward_spec,
        store_init,
        infer_keys,
        data,
        var,
        noise_model=noise_model,
        learning_rate=learning_rate,
        num_steps=num_steps,
    )

    return theta_final, store_final, history


@dataclass
class EigenGdResults:
    """Container for eigenmode gradient-descent diagnostics."""

    eigen_map: EigenThetaMap
    z_history: jnp.ndarray
    theta_history: jnp.ndarray
    loss_history: jnp.ndarray
    theta_final: jnp.ndarray
    z_final: jnp.ndarray


def run_shera_image_gd_eigen(
    *,
    loss_fn: Optional[callable] = None,
    theta0: Optional[jnp.ndarray] = None,
    cfg: SheraThreePlaneConfig = SHERA_TESTBED_CONFIG,
    forward_spec: Optional[ParamSpec] = None,
    base_forward_store: Optional[ParameterStore] = None,
    infer_keys: Sequence[ParamKey] = (
        "binary.separation_as",
        "binary.x_position_as",
        "binary.y_position_as",
    ),
    data: Optional[jnp.ndarray] = None,
    var: Optional[jnp.ndarray] = None,
    noise_model: NoiseModel = "gaussian",
    num_steps: int = 50,
    learning_rate: Optional[float] = None,
    truncate: Optional[int] = None,
    whiten: bool = False,
    theta_ref: Optional[jnp.ndarray] = None,
    fim_kwargs: Optional[Dict[str, Any]] = None,
    use_system_graph: bool = False,
) -> EigenGdResults:
    """
    Run Shera image-based gradient descent in eigenmode coordinates.

    The flow mirrors ``run_shera_image_gd_basic`` but inserts an
    EigenThetaMap reparameterisation:

    1) Build or accept a θ-space loss ``loss_fn``.
    2) Compute a θ-space FIM/Hessian around ``theta_ref`` (defaults to
       ``theta0``) and build ``EigenThetaMap.from_fim`` with optional
       truncation/whitening.
    3) Map θ→z, run GD in z-space, then map back to θ for inspection.

    Parameters
    ----------
    loss_fn : callable, optional
        θ-space loss; if not provided, one is built via
        ``make_binder_image_nll_fn`` using the Shera config/spec/store
        arguments. Signature: ``loss_fn(theta) -> scalar``.
    theta0 : array-like, optional
        Initial θ vector. Required if ``loss_fn`` is provided directly;
        otherwise inferred from ``make_binder_image_nll_fn``.
    cfg / forward_spec / base_forward_store / infer_keys / data / var :
        Inputs passed through to ``make_binder_image_nll_fn`` when
        ``loss_fn`` is not provided. ``base_forward_store`` defaults to
        ``ParameterStore.from_spec_defaults`` and accepts the same
        ``infer_keys`` defaults as ``run_shera_image_gd_basic``.
    num_steps : int
        Number of gradient descent iterations in eigen (z) space.
    learning_rate : float, optional
        Adam step size in z-space. If None, uses a curvature-derived
        heuristic ``1/(max(eigvals)+1e-6)``.
    truncate : int, optional
        If provided, keep only the top-k eigenmodes when building
        ``EigenThetaMap``.
    whiten : bool
        If True, eigen coordinates are scaled by ``sqrt(eigvals)`` so a
        local quadratic loss becomes ≈ ½ ‖z‖² in the retained subspace.
    theta_ref : array-like, optional
        Reference θ for the FIM; defaults to ``theta0``.
    fim_kwargs : dict, optional
        Reserved for future extensions; currently unused but accepted to
        mirror other FIM helpers.
    use_system_graph : bool
        Whether to route Binder execution through SystemGraph when
        auto-building the loss function.

    Returns
    -------
    EigenGdResults
        Diagnostic container with eigen map, histories, and final θ/z.
    """

    fim_kwargs = fim_kwargs or {}

    # ------------------------------------------------------------------
    # 1) Build θ-space loss (Binder-based) if not provided
    # ------------------------------------------------------------------
    if loss_fn is None or theta0 is None:
        if data is None or var is None:
            raise ValueError("data and var must be provided when loss_fn is None")

        if forward_spec is None:
            forward_spec = build_forward_model_spec_from_config(cfg)
        if base_forward_store is None:
            base_forward_store = ParameterStore.from_spec_defaults(forward_spec)

        base_forward_store = refresh_derived(
            base_forward_store,
            forward_spec,
            TRANSFORMS,
            system_id=DEFAULT_SYSTEM_ID,
        )

        loss_fn, theta0 = make_binder_image_nll_fn(
            cfg,
            forward_spec,
            base_forward_store,
            infer_keys,
            data,
            var,
            noise_model=noise_model,
            reduce="sum",
            use_system_graph=use_system_graph,
        )
    else:
        if theta0 is None:
            raise ValueError("theta0 must accompany a custom loss_fn")

    theta0 = jnp.asarray(theta0)
    theta_ref = theta_ref if theta_ref is not None else theta0

    # ------------------------------------------------------------------
    # 2) Compute curvature / FIM in θ-space
    # ------------------------------------------------------------------
    if fim_kwargs:
        raise ValueError("fim_kwargs is reserved for future use and must be empty today")

    F = fim_theta(loss_fn, theta_ref)
    F = jnp.asarray(F)
    F = jnp.nan_to_num((F + F.T) / 2.0)
    F = F + 1e-8 * jnp.eye(F.shape[0], dtype=F.dtype)

    # ------------------------------------------------------------------
    # 3) Build EigenThetaMap (optionally truncate/whiten)
    # ------------------------------------------------------------------
    eigen_map = EigenThetaMap.from_fim(
        F,
        theta_ref,
        truncate=truncate,
        whiten=whiten,
    )

    # ------------------------------------------------------------------
    # 4) Map θ₀ → z₀ and choose learning rate if unspecified
    # ------------------------------------------------------------------
    z0 = eigen_map.to_eigen(theta0)
    if learning_rate is None:
        eigvals = eigen_map.eigvals if eigen_map.eigvals is not None else None
        if eigvals is None or eigvals.size == 0:
            learning_rate = 1e-2
        else:
            learning_rate = float(1.0 / (float(jnp.max(eigvals)) + 1e-6))

    # ------------------------------------------------------------------
    # 5) Define z-space loss = loss_theta(eigen_map.to_theta(z))
    # ------------------------------------------------------------------
    def loss_z(z):
        theta = eigen_map.to_theta(z)
        return loss_fn(theta)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(z0)

    @jax.jit
    def _step(z, opt_state):
        loss, g = jax.value_and_grad(loss_z)(z)
        updates, opt_state = optimizer.update(g, opt_state, z)
        z = optax.apply_updates(z, updates)
        z = jnp.nan_to_num(z)
        return z, opt_state, loss

    z_history = []
    theta_history = []
    loss_history = []

    z = z0
    for _ in range(num_steps):
        z, opt_state, loss_val = _step(z, opt_state)
        z_history.append(z)
        loss_history.append(loss_val)
        theta_history.append(eigen_map.to_theta(z))

    z_final = z
    theta_final = eigen_map.to_theta(z_final)

    return EigenGdResults(
        eigen_map=eigen_map,
        z_history=jnp.stack(z_history) if z_history else jnp.empty_like(z0),
        theta_history=jnp.stack(theta_history) if theta_history else jnp.empty_like(theta0),
        loss_history=jnp.stack(loss_history) if loss_history else jnp.empty((0,)),
        theta_final=theta_final,
        z_final=z_final,
    )


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


