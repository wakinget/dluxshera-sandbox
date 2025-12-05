# src/dluxshera/core/binder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax.numpy as jnp
import dLux as dl

from ..optics.config import SheraThreePlaneConfig
from ..optics.builder import build_shera_threeplane_optics
from ..params.spec import ParamSpec, ParamKey
from ..params.store import ParameterStore
from .universe import build_alpha_cen_source


@dataclass
class SheraThreePlaneBinder:
    """
    Lightweight "binder" for the Shera three-plane system.

    Responsibilities
    ----------------
    - Hold a *static* optical system and detector built from a SheraThreePlaneConfig.
    - Track a "base" ParameterStore providing default values for all keys.
    - Given a (possibly partial) ParameterStore of inference parameters,
      merge it into the base store and produce a model image via dLux.

    This is intentionally *not* tied to the legacy SheraThreePlane_Model class.
    It works directly with:
      - SheraThreePlaneConfig  (geometry, bandpass, sampling)
      - ParameterStore         (numeric parameters)
      - AlphaCen source builder
      - SheraThreePlaneSystem  (via build_shera_threeplane_optics)
    """

    cfg: SheraThreePlaneConfig
    inference_spec: ParamSpec
    base_store: ParameterStore

    # Static components of the forward model
    optics: dl.OpticalSystem
    detector: dl.LayeredDetector

    def __init__(
        self,
        cfg: SheraThreePlaneConfig,
        inference_spec: ParamSpec,
        base_store: ParameterStore,
    ) -> None:
        # Store config + spec for later reference
        self.cfg = cfg
        self.inference_spec = inference_spec

        # Optional safety: ensure all keys in the store are known to the spec.
        # (You can drop this if it ever feels too strict.)
        base_store = base_store.validate_against(inference_spec)
        self.base_store = base_store

        # Build *static* optics once, using the base_store's Zernike coeffs
        # (or None) if present. This avoids re-constructing SheraThreePlaneSystem
        # inside jitted loss/gradient loops.
        optics = build_shera_threeplane_optics(cfg, store=base_store, spec=inference_spec)

        # Simple detector: same pattern as SheraThreePlane_Model
        # You can tweak this if your detector stack grows more complex later.
        detector = dl.LayeredDetector(
            layers=[("downsample", dl.Downsample(cfg.oversample))]
        )

        self.optics = optics
        self.detector = detector

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_store(self, store: ParameterStore) -> ParameterStore:
        """
        Merge a (possibly partial) store into the base store.

        Any keys present in `store` override the base; all other keys
        are inherited from `base_store`.
        """
        return self.base_store.replace(store.as_dict())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, store: ParameterStore) -> jnp.ndarray:
        """
        Compute a model image for the given parameter store.

        Parameters
        ----------
        store :
            ParameterStore containing updated values for some subset of
            inference keys (e.g. binary separation, centroid, flux, etc.).
            Any missing keys fall back to the values in `base_store`.

        Returns
        -------
        image : jnp.ndarray
            Model image array with the same shape as produced by the
            underlying dLux Telescope (PSF/image plane).
        """
        eff_store = self._merge_store(store)

        # Build a fresh Alpha Cen source for the updated parameters.
        # Bandpass / n_lambda come from the static cfg, astrometry/flux
        # come from eff_store.
        source = build_alpha_cen_source(eff_store, n_wavels=self.cfg.n_lambda)

        # Compose telescope and evaluate model
        telescope = dl.Telescope(
            source=source,
            optics=self.optics,
            detector=self.detector,
        )
        return telescope.model()
