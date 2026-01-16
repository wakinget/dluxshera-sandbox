"""Stubs for converting backend-agnostic prior specs into NumPyro distributions.

This is intentionally thin and not wired into the main inference flow yet. It is
meant to document how :class:`PriorSpec` could be adapted for NumPyro without
forcing a dependency at import time.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import jax

from .prior import PriorSpec
from ..params.spec import ParamKey


def numpyro_priors_from_spec(
    prior_spec: PriorSpec,
    keys: Optional[Iterable[ParamKey]] = None,
) -> Dict[ParamKey, object]:
    """Convert a :class:`PriorSpec` into a NumPyro-compatible prior mapping.

    Notes
    -----
    - NumPyro is intentionally not imported here to avoid hard dependencies in
      the base inference stack.
    """
    import jax.numpy as jnp
    import numpyro.distributions as dist

    selected_keys = tuple(prior_spec.fields.keys()) if keys is None else tuple(keys)
    priors: Dict[ParamKey, object] = {}
    for key in selected_keys:
        field = prior_spec.fields[key]
        field._assert_supported()
        mean = jnp.asarray(field.mean)
        sigma = jnp.asarray(field.sigma)
        if field.dist == "Normal":
            priors[key] = dist.Normal(loc=mean, scale=sigma)
        elif field.dist == "Uniform":
            priors[key] = dist.Uniform(low=mean - sigma, high=mean + sigma)
        else:
            priors[key] = dist.LogNormal(loc=jnp.log(mean), scale=sigma)
    return priors


def sample_numpyro_priors_from_spec(
    prior_spec: PriorSpec,
    rng_key,
    keys: Optional[Iterable[ParamKey]] = None,
) -> Dict[ParamKey, object]:
    """Sample NumPyro priors defined by a :class:`PriorSpec`.

    Returns
    -------
    dict
        Mapping of parameter keys to sampled values.
    """
    priors = numpyro_priors_from_spec(prior_spec, keys=keys)
    selected_keys = tuple(priors.keys())
    subkeys = jax.random.split(rng_key, len(selected_keys)) if selected_keys else ()
    return {key: priors[key].sample(subkey) for key, subkey in zip(selected_keys, subkeys)}
