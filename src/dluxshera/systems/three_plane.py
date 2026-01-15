"""Three-plane Shera system definitions (binder, config, and presets)."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.resources as resources
from pathlib import Path
from typing import Optional, Tuple

import dLux as dl
import jax.numpy as jnp

from .base import BaseConfig, BaseSheraBinder
from ..core.universe import build_alpha_cen_source
from ..params.spec import ParamSpec
from ..params.store import ParameterStore


@dataclass(frozen=True)
class SheraThreePlaneConfig(BaseConfig):
    """
    Structural configuration for the Shera three-plane optical system.

    This dataclass collects *non-inferred* parameters that define the shape
    and layout of the optical model. These values:
      - control grid sizes, wavelength sampling, and geometry,
      - determine which Zernike bases exist on each surface,
      - are intended to remain fixed for a given run / instrument setup.

    They are separate from the ParameterStore/ParamSpec, which handle the
    *numeric state* of an inference run (binary separation, flux, Zernike
    coefficients, etc.).

    Note: Zernike bases are specified using Noll indices stored as immutable
    Python tuples of integers (see `primary_noll_indices` and
    `secondary_noll_indices`). Using tuples keeps this configuration layer
    backend-agnostic (no NumPy/JAX arrays), hashable, and safe as a frozen
    dataclass, while still being easy for the builder to consume when
    constructing Zernike `BasisOptic` layers.

    Structural hashing and caching
    ------------------------------
    The optics builder treats the following fields as *structural* when
    computing a hash for caching purposes:

      - Grid and sampling: `pupil_npix`, `psf_npix`, `oversample`.
      - Bandpass sampling (affects wavelength grid shapes elsewhere):
        `wavelength_m`, `bandwidth_m`, `n_lambda`.
      - Three-plane geometry: `m1_diameter_m`, `m2_diameter_m`,
        `m1_focal_length_m`, `m2_focal_length_m`, `m1_m2_separation_m`,
        `pixel_pitch_m`.
      - Aperture features: `n_struts`, `strut_width_m`, `strut_rotation_deg`.
      - Zernike basis selection: `primary_noll_indices`,
        `secondary_noll_indices` (coefficients live in the ParameterStore and
        are *not* part of the structural hash).
      - Diffractive pupil: `diffractive_pupil_path`,
        `dp_design_wavelength_m`.

    Metadata such as `design_name` is intentionally *not* included in the
    structural hash so that different labels can reuse the same cached optics
    structure.
    """

    design_name: Optional[str] = None
    """
    A human-readable identifier for this optical design.

    This has no effect on the optical model itself — it is purely metadata
    used for bookkeeping, logging, versioning, reproducibility, and selecting
    between different point designs (e.g., 'shera_testbed', 'shera_flight').
    """

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
    # System geometry (three-plane layout)
    # ------------------------------------------------------------------
    m1_diameter_m: float = 0.09
    """Primary mirror clear diameter [meters]."""

    m2_diameter_m: float = 0.025
    """Secondary mirror clear diameter [meters]."""

    m1_focal_length_m: float = 0.35796
    """Effective focal length of the primary mirror [meters]."""

    m2_focal_length_m: float = -0.041935
    """Effective focal length of the secondary mirror [meters]."""

    m1_m2_separation_m: float = 0.320
    """Axial separation between primary and secondary [meters]."""

    pixel_pitch_m: float = 6.5e-6
    """
    Physical pixel pitch of the detector [meters].

    In three-plane geometric transforms this, together with the effective
    focal length, determines the PSF plate scale.
    """

    # ------------------------------------------------------------------
    # Aperture features (spiders, etc.)
    # ------------------------------------------------------------------
    n_struts: int = 4
    """Number of support struts in the primary aperture."""

    strut_width_m: float = 0.002
    """
    Width of the support struts [meters].
    """

    strut_rotation_deg: float = 0.0
    """Rotation angle of the spider pattern [degrees]."""

    # ------------------------------------------------------------------
    # Zernike basis selection (structure, not coefficients)
    # ------------------------------------------------------------------
    primary_noll_indices: Tuple[int, ...] = ()
    """
    Noll indices defining the Zernike basis on the primary mirror.

    If this tuple is empty, the builder should *not* construct a Zernike
    BasisOptic for the primary. If non-empty, the builder will construct a
    basis using these indices and will expect `primary.zernike_coeffs_nm` in
    the ParameterStore to have matching length. Forward-model specs constructed
    from this config default the coefficients to a zero vector (no-aberration
    case) whenever a basis is present.
    """

    secondary_noll_indices: Tuple[int, ...] = ()
    """
    Noll indices defining the Zernike basis on the secondary mirror.

    Same convention as `primary_noll_indices`: an empty tuple means no
    Zernike basis is constructed for the secondary; a non-empty tuple means
    the builder constructs a BasisOptic and expects compatible coefficients
    under `secondary.zernike_coeffs_nm`. Forward-model specs built from this
    config seed those coefficients with a zero vector when the basis exists.
    """

    # ------------------------------------------------------------------
    # Diffractive pupil / fixed masks
    # ------------------------------------------------------------------
    diffractive_pupil_path: Optional[str] = None
    """
    Optional filesystem path to a diffractive pupil mask (e.g. a .npy file).

    If `diffractive_pupil_path` is None, no diffractive pupil is applied.

    Otherwise, interpretation depends on `dp_design_wavelength_m`:

    - If `dp_design_wavelength_m` is not None:
        The file is assumed to contain a dimensionless design pattern
        `P(x, y)` in [0, 1] representing a *normalized phase* over the
        interval [0, π] radians at the design wavelength. Specifically,

            P = 0 → 0 radians of phase
            P = 1 → π radians of phase

        at `dp_design_wavelength_m`. The builder will convert this to an
        OPD map in meters by computing the corresponding phase and mapping
        onto OPD via dlu.phase2opd. The resulting operation is:

            OPD_m = P * π * dp_design_wavelength_m / (2π).

        Equivalent to:

            OPD_m = P / 2 * dp_design_wavelength_m

        so P = 1 corresponds to half a wave of OPD at the design
        wavelength.

    - If `dp_design_wavelength_m` is None:
        The file is assumed to already encode an OPD map in meters on the
        same pupil grid as the primary and is applied directly without
        further scaling.

    """

    dp_design_wavelength_m: Optional[float] = None
    """
    Design wavelength for the provided diffractive pupil mask [meters].

    If this is not None, the diffractive pupil file is interpreted as a
    normalized phase pattern `P(x, y)` in [0, 1] spanning [0, π] radians
    at this wavelength:

        P = 0 → 0 radians of phase
        P = 1 → π radians of phase

    The builder converts this to an OPD map in meters by computing

        OPD_m = P * π * dp_design_wavelength_m / (2π)
              = 0.5 * P * dp_design_wavelength_m,

    so P = 1 corresponds to half a wave of OPD at `dp_design_wavelength_m`.

    If `dp_design_wavelength_m` is None, the diffractive pupil file is
    assumed to already encode an OPD map in meters on the primary pupil
    grid and is applied directly without further scaling.
    """


def default_diffractive_pupil_path() -> str:
    """Resolve the default diffractive pupil path as a string."""
    try:
        return str(resources.files("dluxshera.data") / "diffractive_pupil.npy")
    except Exception:
        return str(Path(__file__).resolve().parents[1] / "data" / "diffractive_pupil.npy")


# ---------------------------------------------------------------------
# Named point designs
# ---------------------------------------------------------------------

# Define the path to the default diffractive pupil file
DEFAULT_DP_PATH = default_diffractive_pupil_path()

SHERA_TESTBED_CONFIG = SheraThreePlaneConfig(
    design_name="shera_testbed",

    # --- system geometry ---
    m1_diameter_m=0.09,
    m2_diameter_m=0.025,
    m1_focal_length_m=0.35796,
    m2_focal_length_m=-0.041935,
    m1_m2_separation_m=0.320,
    pixel_pitch_m=6.5e-6,

    # --- grids & sampling ---
    pupil_npix=256,
    psf_npix=256,
    oversample=1,
    wavelength_m=550e-9,
    bandwidth_m=110e-9,
    n_lambda=3,

    # --- spiders / obscurations ---
    n_struts=4,
    strut_width_m=0.002,
    strut_rotation_deg=45.0,

    # --- Zernike basis structure ---
    # define Noll indices as an immutable Python tuple.
    primary_noll_indices=tuple(range(4, 12)),
    secondary_noll_indices=tuple(range(4, 12)),

    # --- diffractive pupil ---
    diffractive_pupil_path=DEFAULT_DP_PATH,
    dp_design_wavelength_m=550e-9,
)


SHERA_FLIGHT_CONFIG = SheraThreePlaneConfig(
    design_name="shera_flight",

    # --- system geometry ---
    m1_diameter_m=0.22,
    m2_diameter_m=0.025,
    m1_focal_length_m=0.604353,
    m2_focal_length_m=-0.0545,
    m1_m2_separation_m=0.55413,
    pixel_pitch_m=4.6e-6,

    # --- grids & sampling ---
    pupil_npix=256,
    psf_npix=256,
    oversample=1,
    wavelength_m=550e-9,
    bandwidth_m=41e-9,
    n_lambda=3,

    # --- spiders / obscurations ---
    n_struts=3,
    strut_width_m=0.002,
    strut_rotation_deg=-90.0,

    # --- Zernike basis structure ---
    # define Noll indices as an immutable Python tuple.
    primary_noll_indices=tuple(range(4, 12)),
    secondary_noll_indices=tuple(range(4, 12)),

    diffractive_pupil_path=DEFAULT_DP_PATH,
    dp_design_wavelength_m=550e-9,
)


@dataclass
class SheraThreePlaneBinder(BaseSheraBinder):
    """
    Canonical generative model for the Shera three-plane system.

    Binder is the successor to the legacy ``SheraThreePlane_Model`` facade and
    is intentionally treated as **mostly immutable**: instantiate it once for a
    given configuration + base forward store (with deriveds populated), then use
    ``.model(store_delta)`` to evaluate PSFs without mutating internal state.

    Key properties
    --------------
    - Holds the Shera config, forward ParamSpec, and a *forward-style* base
      ParameterStore (derived values already populated).
    - ``.model()`` is the primary API and is intentionally lightweight: with
      ``store_delta=None`` it fast-paths through the cached telescope. For
      non-structural overlays it merges ``store_delta`` onto the base store,
      then evaluates the direct builder path. Structural overrides require
      ``allow_rebuild=True`` and delegate to ``update_store()``.
    - ``.update_store()`` returns a new binder instance with the refreshed base
      store; the original binder remains unchanged.

    The ``with_store`` attribute is an alias of
    :meth:`BaseSheraBinder.with_store`, provided for parity with legacy APIs.
    It preserves the binder's immutable-style semantics by always returning a
    fresh binder instance rather than mutating in-place.
    """

    cfg: SheraThreePlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    # Internal detector references (prepared eagerly)
    _detector: Optional[dl.LayeredDetector] = None

    def __init__(
        self,
        cfg: SheraThreePlaneConfig,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
    ) -> None:
        """Construct a binder for the three-plane Shera configuration.

        Parameters
        ----------
        cfg : SheraThreePlaneConfig
            Fully prepared Shera three-plane configuration. Any derived
            configuration values needed by the optics/source builders should
            already be present on this object.
        forward_spec : ParamSpec
            Parameter specification describing the full forward store,
            including structural keys and derived entries.
        base_forward_store : ParameterStore
            Forward-style base store with derived values populated. The store
            is validated against ``forward_spec`` and treated as immutable
            baseline state for subsequent evaluations.
        """
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        """Evaluate the Shera three-plane model directly.

        Uses the three-plane optics and alpha Cen source builders to assemble
        a fresh telescope from ``eff_store`` and returns the PSF model output.
        This path is used when a non-structural store overlay is supplied.
        """
        return self._build_telescope(eff_store).model()

    def _build_optics(self, store: ParameterStore):
        """Build the Shera three-plane optics stack.

        Delegates to ``build_shera_threeplane_optics`` with the configured
        three-plane configuration, validated store, and forward specification.
        """
        from ..optics.builder import build_shera_threeplane_optics

        return build_shera_threeplane_optics(
            self.cfg, store=store, spec=self.forward_spec
        )

    def _build_source(self, store: ParameterStore):
        """Build the Shera alpha Cen source for the three-plane system."""
        return build_alpha_cen_source(store, cfg=self.cfg)

    def _runtime_bindings(self) -> tuple[tuple[str, str], ...]:
        """Return the three-plane runtime bindings for non-structural keys."""
        from ..optics.builder import THREEPLANE_RUNTIME_BINDINGS

        return THREEPLANE_RUNTIME_BINDINGS

    def _compute_structural_hash(self) -> Optional[str]:
        """Return the structural hash derived from the three-plane config."""
        from ..optics.builder import structural_hash_from_config

        return structural_hash_from_config(self.cfg)

    with_store = BaseSheraBinder.with_store


__all__ = [
    "SheraThreePlaneConfig",
    "SheraThreePlaneBinder",
    "SHERA_TESTBED_CONFIG",
    "SHERA_FLIGHT_CONFIG",
    "DEFAULT_DP_PATH",
    "default_diffractive_pupil_path",
]
