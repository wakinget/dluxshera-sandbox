"""Two-plane Shera system definitions (binder, config, and presets)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import dLux as dl
import jax.numpy as jnp

from .base import BaseConfig, BaseSheraBinder
from ..core.universe import build_alpha_cen_source
from ..params.spec import ParamSpec
from ..params.store import ParameterStore


@dataclass(frozen=True)
class SheraTwoPlaneConfig(BaseConfig):
    """
    Structural configuration for the Shera two-plane optical system.

    This captures fixed geometry and sampling choices for the Toliman-like
    two-plane pupil→focal relay used by SheraTwoPlaneOptics. These values are
    separate from inference parameters (which live in ParameterStore/ParamSpec)
    and are intended to remain constant for a given instrument setup.
    """

    design_name: Optional[str] = None
    """Human-readable identifier for this optical design."""

    # ------------------------------------------------------------------
    # Pupil & PSF grids
    # ------------------------------------------------------------------
    pupil_npix: int = 256
    """Number of pixels across the pupil grid."""

    psf_npix: int = 256
    """Number of pixels across the detector/PSF cutout."""

    oversample: int = 3
    """PSF oversampling factor relative to the on-sky plate scale."""

    # ------------------------------------------------------------------
    # Wavelength sampling
    # ------------------------------------------------------------------
    wavelength_m: float = 550e-9
    """Central wavelength of the bandpass [meters]."""

    bandwidth_m: float = 110e-9
    """Width of the bandpass [meters]."""

    n_lambda: int = 3
    """Number of discrete wavelengths to sample across the bandpass."""

    # ------------------------------------------------------------------
    # System geometry (two-plane layout)
    # ------------------------------------------------------------------
    m1_diameter_m: float = 0.09
    """Primary mirror clear diameter [meters]."""

    central_obscuration_ratio: float = 0.0
    """
    Ratio of the central obscuration diameter to the primary diameter.

    Defaults to 0 (no obscuration) for the simplified two-plane relay.
    """

    n_struts: int = 4
    """Number of support struts in the primary aperture."""

    strut_width_m: float = 0.002
    """Width of the support struts [meters]."""

    strut_rotation_deg: float = -45.0
    """Rotation angle of the spider pattern [degrees]."""

    # ------------------------------------------------------------------
    # Fixed plate scale (primitive for the two-plane model)
    # ------------------------------------------------------------------
    plate_scale_as_per_pix: float = 0.3547
    """
    Plate scale in arcseconds per pixel.

    For the two-plane Shera system we treat plate scale as a primitive rather
    than deriving it from telescope geometry.
    """

    # ------------------------------------------------------------------
    # Zernike basis selection (structure, not coefficients)
    # ------------------------------------------------------------------
    primary_noll_indices: Tuple[int, ...] = ()
    """
    Noll indices defining the Zernike basis on the primary mirror.

    If this tuple is empty, the builder does not construct a Zernike BasisOptic
    for the primary. If non-empty, the forward spec will expect
    `primary.zernike_coeffs_nm` of matching length.
    """

    # ------------------------------------------------------------------
    # Diffractive pupil / fixed masks
    # ------------------------------------------------------------------
    diffractive_pupil_path: Optional[str] = None
    """Optional filesystem path to a diffractive pupil mask (e.g. .npy)."""

    dp_design_wavelength_m: Optional[float] = None
    """Design wavelength for the diffractive pupil mask [meters]."""


@dataclass
class SheraTwoPlaneBinder(BaseSheraBinder):
    """Generative model for the Shera two-plane system.

    Mirrors :class:`SheraThreePlaneBinder` semantics: mostly immutable, owns a
    forward-spec-validated base store, and exposes ``.model(store_delta)`` as the
    canonical evaluation path. ``.model`` fast-paths through the cached
    telescope when ``store_delta`` is omitted, and accepts non-structural
    overlays by default when an explicit delta is provided. Structural updates
    require ``allow_rebuild=True`` to rebuild the binder state. When
    ``.update_store()`` returns a new binder instance with the refreshed base
    store so the original binder remains unchanged.

    The ``with_store`` attribute is an alias of
    :meth:`BaseSheraBinder.with_store` to keep a stable public API. It preserves
    immutable-style semantics by always returning a new binder instance.
    """

    cfg: SheraTwoPlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    _detector: Optional[dl.LayeredDetector] = None

    def __init__(
        self,
        cfg: SheraTwoPlaneConfig,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
    ) -> None:
        """Construct a binder for the two-plane Shera configuration.

        Parameters
        ----------
        cfg : SheraTwoPlaneConfig
            Fully prepared Shera two-plane configuration. Derived config
            values expected by the optics/source builders should already be
            present.
        forward_spec : ParamSpec
            Parameter specification describing the forward store, including
            structural keys and derived entries.
        base_forward_store : ParameterStore
            Forward-style base store with derived values populated. The store
            is validated against ``forward_spec`` and used as the immutable
            baseline for evaluations.
        """
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
        )

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        """Evaluate the Shera two-plane model directly.

        Builds a fresh telescope using the two-plane optics and alpha Cen
        source with ``eff_store`` and returns the modeled PSF output.
        """
        return self._build_telescope(eff_store).model()

    def _build_optics(self, store: ParameterStore):
        """Build the Shera two-plane optics stack."""
        from ..optics.builder import build_shera_twoplane_optics

        return build_shera_twoplane_optics(self.cfg, store=store, spec=self.forward_spec)

    def _build_source(self, store: ParameterStore):
        """Build the Shera alpha Cen source for the two-plane system."""
        return build_alpha_cen_source(store, cfg=self.cfg)

    def _runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return the two-plane runtime bindings for non-structural keys."""
        from ..optics.builder import TWOPLANE_RUNTIME_BINDINGS

        return TWOPLANE_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        """Return the structural hash derived from the two-plane config."""
        from ..optics.builder import structural_hash_for_twoplane

        return structural_hash_for_twoplane(self.cfg)

    with_store = BaseSheraBinder.with_store


__all__ = [
    "SheraTwoPlaneConfig",
    "SheraTwoPlaneBinder",
]
