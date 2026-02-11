"""Backend-agnostic prior abstractions used across inference utilities.

`PriorField` and `PriorSpec` capture independent priors on `ParamKey`s without
assuming a specific probabilistic programming backend. The goal is to support
lightweight MAP penalties, jittering initialisations near a reference store, and
future adapters (e.g., to NumPyro) without coupling the core abstractions to any
single library.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Optional

import jax
import jax.numpy as jnp

from ..params.spec import ParamKey
from ..params.store import ParameterStore

ArrayLike = jnp.ndarray | float | int
PriorInfo = Mapping[str, ArrayLike | str] | tuple[ArrayLike, str] | tuple[ArrayLike]


@dataclass(frozen=True)
class PriorField:
    """Description of a prior for a single parameter.

    Parameters
    ----------
    dist:
        Distribution family identifier. Supports "Normal", "Uniform", and
        "LogNormal".
    mean:
        Location parameter; shape-compatible with the associated parameter.
    sigma:
        Scale (standard deviation) parameter; can be a scalar or an array.
    """

    dist: str = "Normal"
    mean: ArrayLike | jnp.ndarray = 0.0
    sigma: ArrayLike | jnp.ndarray = 1.0

    def _assert_supported(self) -> None:
        if self.dist not in {"Normal", "Uniform", "LogNormal"}:
            raise ValueError(
                f"Unsupported distribution '{self.dist}'. Only 'Normal', 'Uniform', and 'LogNormal' are implemented."
            )


@dataclass(frozen=True)
class PriorSpec:
    """Mapping from ParamKey to :class:`PriorField`.

    This abstraction is backend-agnostic and keeps only the data and simple
    operations needed for MAP penalties and jittering near a reference store.
    """

    fields: Mapping[ParamKey, PriorField]

    @classmethod
    def from_sigmas(
        cls,
        center_store: ParameterStore,
        sigmas: Mapping[ParamKey, ArrayLike],
        dist: str = "Normal",
    ) -> "PriorSpec":
        """Build a :class:`PriorSpec` from per-key sigma values and a reference store.

        Parameters
        ----------
        center_store:
            Store providing the mean value for each parameter.
        sigmas:
            Mapping of parameter keys to scalar or array sigma values.
        dist:
            Distribution family identifier ("Normal", "Uniform", or "LogNormal").
        """

        fields: MutableMapping[ParamKey, PriorField] = {}
        for key, sigma in sigmas.items():
            fields[key] = PriorField(dist=dist, mean=center_store.get(key), sigma=sigma)
        return cls(fields=dict(fields))

    @classmethod
    def from_info(
        cls,
        center_store: ParameterStore,
        info: Mapping[ParamKey, PriorInfo],
        dist: str = "Normal",
    ) -> "PriorSpec":
        """Build a :class:`PriorSpec` from per-key prior info and a reference store.

        Parameters
        ----------
        center_store:
            Store providing the mean value for each parameter.
        info:
            Mapping of parameter keys to either ``{"sigma": ..., "dist": ...}``
            dictionaries or ``(sigma, dist)`` tuples. The distribution defaults
            to ``dist`` when omitted.
        dist:
            Default distribution family identifier ("Normal", "Uniform", or "LogNormal").
        """

        fields: MutableMapping[ParamKey, PriorField] = {}
        for key, entry in info.items():
            if isinstance(entry, tuple):
                if not entry:
                    raise ValueError("Prior tuple entries must include at least a sigma value.")
                sigma = entry[0]
                entry_dist = entry[1] if len(entry) > 1 else dist
            elif isinstance(entry, Mapping):
                if "sigma" not in entry:
                    raise ValueError("Prior mapping entries must include a 'sigma' value.")
                sigma = entry["sigma"]
                entry_dist = entry.get("dist", dist)
            else:
                raise TypeError("Prior info entries must be mappings or tuples.")
            fields[key] = PriorField(dist=str(entry_dist), mean=center_store.get(key), sigma=sigma)
        return cls(fields=dict(fields))

    def _select_keys(self, keys: Optional[Iterable[ParamKey]]) -> tuple[ParamKey, ...]:
        if keys is None:
            return tuple(self.fields.keys())
        return tuple(keys)

    def quadratic_penalty(
        self,
        store: ParameterStore,
        center_store: Optional[ParameterStore] = None,
        keys: Optional[Iterable[ParamKey]] = None,
    ) -> jnp.ndarray:
        """Compute a summed quadratic prior penalty over selected keys.

        Notes
        -----
        - Uses `(value - mean)^2 / (2 * sigma^2)` and sums over all elements.
          The interpretation of ``sigma`` depends on the distribution family.
        - ``center_store`` overrides the per-field mean when provided. This keeps
          the operation tied to an external reference (e.g., truth store) while
          allowing `PriorField.mean` to serve as a stored default.
        """

        penalty = jnp.array(0.0)
        for key in self._select_keys(keys):
            field = self.fields[key]
            field._assert_supported()
            value = jnp.asarray(store.get(key))
            mean = jnp.asarray(center_store.get(key) if center_store is not None else field.mean)
            sigma = jnp.asarray(field.sigma)
            penalty = penalty + jnp.sum((value - mean) ** 2 / (2.0 * sigma**2))
        return penalty

    def sample_near(
        self,
        center_store: ParameterStore,
        rng_key: jax.Array,
        keys: Optional[Iterable[ParamKey]] = None,
    ) -> ParameterStore:
        """Draw a new store by sampling independent priors around a reference store.

        Parameters
        ----------
        center_store:
            Store providing the mean value for each parameter.
        rng_key:
            JAX random key used to sample each prior.
        keys:
            Optional subset of parameter keys to sample. Defaults to all keys in the spec.
        """

        updates: MutableMapping[ParamKey, jnp.ndarray] = {}
        selected_keys = self._select_keys(keys)
        subkeys = jax.random.split(rng_key, len(selected_keys)) if selected_keys else ()
        for subkey, key in zip(subkeys, selected_keys):
            field = self.fields[key]
            field._assert_supported()
            mean = jnp.asarray(
                center_store.get(key) if center_store is not None else field.mean
            )
            sigma = jnp.asarray(field.sigma)
            sample_dtype = jnp.result_type(mean, sigma, jnp.float32)
            mean = mean.astype(sample_dtype)
            sigma = sigma.astype(sample_dtype)
            if field.dist == "Normal":
                noise = jax.random.normal(
                    subkey, shape=mean.shape, dtype=sample_dtype
                )
                updates[key] = mean + sigma * noise
            elif field.dist == "Uniform":
                updates[key] = jax.random.uniform(
                    subkey,
                    shape=mean.shape,
                    dtype=sample_dtype,
                    minval=mean - sigma,
                    maxval=mean + sigma,
                )
            elif field.dist == "LogNormal":
                updates[key] = mean * jax.random.lognormal(
                    subkey, sigma=sigma, shape=mean.shape, dtype=sample_dtype
                )
                # NOTE: Lognormal sampling — sigma is the std dev in *log-space*.
                # This matches NumPyro's LogNormal(loc, scale) with:
                #   JAX:     mean * jax.random.lognormal(key, sigma)
                #   NumPyro: dist.LogNormal(loc=np.log(mean), scale=sigma)
                # In this parameterization, `mean` is the *median* of the distribution.
            else: # Raise unrecognized distribution types
                raise ValueError(
                    f"Unsupported prior distribution '{field.dist}' for parameter '{key}'. "
                    "Supported distributions are: 'Normal', 'Uniform', 'LogNormal'."
                )
        return center_store.replace(updates)

    def sample(
        self,
        rng_key: jax.Array,
        keys: Optional[Iterable[ParamKey]] = None,
    ) -> ParameterStore:
        """Draw a new store by sampling independent priors from stored means.

        This method mirrors :meth:`sample_near` but uses the stored
        :class:`PriorField` mean values instead of requiring a reference store.

        Parameters
        ----------
        rng_key:
            JAX random key used to sample each prior.
        keys:
            Optional subset of parameter keys to sample. Defaults to all keys in the spec.
        """

        updates: MutableMapping[ParamKey, jnp.ndarray] = {}
        selected_keys = self._select_keys(keys)
        subkeys = jax.random.split(rng_key, len(selected_keys)) if selected_keys else ()
        for subkey, key in zip(subkeys, selected_keys):
            field = self.fields[key]
            field._assert_supported()
            mean = jnp.asarray(field.mean)
            sigma = jnp.asarray(field.sigma)
            sample_dtype = jnp.result_type(mean, sigma, jnp.float32)
            mean = mean.astype(sample_dtype)
            sigma = sigma.astype(sample_dtype)
            if field.dist == "Normal":
                noise = jax.random.normal(
                    subkey, shape=mean.shape, dtype=sample_dtype
                )
                updates[key] = mean + sigma * noise
            elif field.dist == "Uniform":
                updates[key] = jax.random.uniform(
                    subkey,
                    shape=mean.shape,
                    dtype=sample_dtype,
                    minval=mean - sigma,
                    maxval=mean + sigma,
                )
            elif field.dist == "LogNormal":
                updates[key] = mean * jax.random.lognormal(
                    subkey, sigma=sigma, shape=mean.shape, dtype=sample_dtype
                )
                # NOTE: Lognormal sampling — sigma is the std dev in *log-space*.
                # This matches NumPyro's LogNormal(loc, scale) with:
                #   JAX:     mean * jax.random.lognormal(key, sigma)
                #   NumPyro: dist.LogNormal(loc=np.log(mean), scale=sigma)
                # In this parameterization, `mean` is the *median* of the distribution.
            else: # Raise unrecognized distribution types
                raise ValueError(
                    f"Unsupported prior distribution '{field.dist}' for parameter '{key}'. "
                    "Supported distributions are: 'Normal', 'Uniform', 'LogNormal'."
                )
        return ParameterStore.from_dict(dict(updates))
