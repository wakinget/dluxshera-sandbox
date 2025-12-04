# src/dluxshera/core/universe.py

from __future__ import annotations

from typing import Optional, Sequence

import jax.numpy as jnp

from dLuxToliman import AlphaCen  # external dependency

from ..params.store import ParameterStore


def build_alpha_cen_source(
    store: ParameterStore,
    n_wavels: int,
) -> AlphaCen:
    """
    Construct an AlphaCen source (from dLuxToliman) from a ParameterStore.

    Parameters
    ----------
    store :
        ParameterStore holding the *inference-level* source & binary
        parameters. For this P0 builder we expect the following keys:

        - 'binary.separation_as'        (float, arcseconds)
        - 'binary.position_angle_deg'   (float, degrees East of North)
        - 'binary.x_position'           (float, arcseconds)
        - 'binary.y_position'           (float, arcseconds)
        - 'source.log_flux_total'       (float, log10 photons)
        - 'binary.contrast'             (float, unitless flux ratio)

        These match the fields defined in `build_inference_spec_basic()`.

    n_wavels :
        Number of wavelength samples to pass to the AlphaCen constructor.
        In practice this should be kept consistent with the optical config
        (e.g. `SheraThreePlaneConfig.n_lambda`), but we keep the coupling
        explicit at the call site for now.

    Returns
    -------
    AlphaCen
        A dLuxToliman AlphaCen source object with the above parameters
        written into its internal state via the Zodiax `.set` interface.

    Notes
    -----
    - This is a *P0 convenience builder*: it assumes that the effective
      log-flux has already been computed (or chosen) and stored under
      'source.log_flux_total'. It does *not* try to derive that from
      physical quantities like exposure time, aperture area, etc.

    - Longer-term, we expect a dedicated “UniverseSpec” and parameter
      transforms to handle the “truth-level” flux bookkeeping. This builder
      will then simply read whatever effective quantities the spec provides.
    """
    # Required parameters – let KeyError surface if they’re missing.
    separation_as = store.get("binary.separation_as")
    position_angle_deg = store.get("binary.position_angle_deg")
    log_flux_total = store.get("binary.log_flux_total")
    contrast = store.get("binary.contrast")

    # Optional centre; default to (0, 0) if not present
    x_position = store.get("binary.x_position", default=0.0)
    y_position = store.get("binary.y_position", default=0.0)

    # Construct a “vanilla” AlphaCen and then overwrite its parameters via .set
    source = AlphaCen(n_wavels=n_wavels)

    # These paths are the canonical AlphaCen / dLuxToliman field names
    paths = [
        "x_position",
        "y_position",
        "separation",
        "position_angle",
        "log_flux",
        "contrast",
    ]
    values = [
        x_position,
        y_position,
        separation_as,       # arcsec
        position_angle_deg,  # degrees
        log_flux_total,      # log10 photons
        contrast,
    ]

    source = source.set(paths, values)

    return source
