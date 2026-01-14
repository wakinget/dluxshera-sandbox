# src/dluxshera/core/modeling.py

from __future__ import annotations

from dataclasses import dataclass
import dLuxToliman as dlT

from ..optics.builder import build_shera_threeplane_optics
from ..optics.config import SheraThreePlaneConfig
from ..optics.optical_systems import SheraThreePlaneOptics
from ..params.spec import ParamSpec
from ..params.store import ParameterStore
from .universe import build_alpha_cen_source


#------------------------------
#-  Parameter Refactor Code   -
#------------------------------

@dataclass(frozen=True)
class SheraThreePlaneComponents:
    """
    Lightweight container bundling together the core Shera three-plane
    objects for a single configuration + parameter state.

    This is intentionally minimal and does *not* embed any inference logic;
    it just packages:

      - the structural config (geometry, grids, bandpass, etc.),
      - the ParamSpec used to define the inference-level parameter schema,
      - the ParameterStore holding the current parameter values,
      - the three-plane optics object,
      - the AlphaCen source object.

    Higher-level code (e.g. inference routines, model wrappers) can consume
    this bundle and decide how to run forward models, attach likelihoods,
    etc., without having to know how to construct the pieces.
    """

    cfg: SheraThreePlaneConfig
    spec: ParamSpec
    store: ParameterStore

    optics: SheraThreePlaneOptics
    source: dlT.AlphaCen


def build_shera_threeplane_components(
    cfg: SheraThreePlaneConfig,
    spec: ParamSpec,
    store: ParameterStore,
) -> SheraThreePlaneComponents:
    """
    Construct the core Shera three-plane components (optics + source) from
    a config, spec, and ParameterStore.

    This is the P0 "end-to-end" builder that ties together:

      - structural configuration (SheraThreePlaneConfig),
      - parameter schema (ParamSpec),
      - parameter values (ParameterStore),

    and returns a single, immutable bundle that can be used by downstream
    modeling and inference code.

    Parameters
    ----------
    cfg :
        Structural configuration of the three-plane optics (geometry,
        grids, bandpass, Zernike basis structure, diffractive pupil, etc.).

    spec :
        ParamSpec that defines the valid keys and basic metadata for the
        inference-level parameters in the store.

    store :
        ParameterStore holding the current values for the inference-level
        parameters (binary separation/PA, centroid, flux, Zernike
        coefficients, etc.).

    Returns
    -------
    SheraThreePlaneComponents
        A dataclass bundling together `cfg`, `spec`, `store`, the
        `SheraThreePlaneOptics` optics object, and the `AlphaCen` source
        object.
    """
    # Ensure the store is consistent with the spec before using it.
    store = store.validate_against(spec)

    # Optics: three-plane system with Zernike coefficients injected from store.
    optics = build_shera_threeplane_optics(cfg, store=store, spec=spec)

    # Source: AlphaCen built from the same ParameterStore.
    source = build_alpha_cen_source(store, n_wavels=cfg.n_lambda)

    return SheraThreePlaneComponents(
        cfg=cfg,
        spec=spec,
        store=store,
        optics=optics,
        source=source,
    )
