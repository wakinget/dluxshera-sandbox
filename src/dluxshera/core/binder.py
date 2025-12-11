# src/dluxshera/core/binder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import dLux as dl

from ..graph.system_graph import (
    build_shera_system_graph,
    build_shera_twoplane_system_graph,
)
from ..optics.config import SheraThreePlaneConfig, SheraTwoPlaneConfig
from ..optics.builder import build_shera_threeplane_optics, build_shera_twoplane_optics
from ..params.spec import ParamSpec
from ..params.store import ParameterStore
from .universe import build_alpha_cen_source


@dataclass
class SheraThreePlaneBinder:
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
    - Eagerly constructs and owns a SystemGraph when ``use_system_graph=True``;
      otherwise follows a direct optics + source builder path.
    - ``.model()`` is the primary API and is intentionally lightweight: merge
      ``store_delta`` onto the base store, then evaluate either the graph or the
      direct builder path.
    """

    cfg: SheraThreePlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    use_system_graph: bool = True

    # Internal graph/detector references (prepared eagerly)
    _graph: Optional[object] = None
    _detector: Optional[dl.LayeredDetector] = None

    def __init__(
        self,
        cfg: SheraThreePlaneConfig,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
        *,
        use_system_graph: bool = True,
    ) -> None:
        self.cfg = cfg
        self.forward_spec = forward_spec
        self.use_system_graph = bool(use_system_graph)

        # Validate and freeze the base forward store; derived values are allowed
        # because forward_spec includes them explicitly.
        self.base_forward_store = base_forward_store.validate_against(
            forward_spec, allow_derived=True
        )

        # Detector is static for this binder; reuse across evaluations and
        # share with the SystemGraph when enabled.
        self._detector = dl.LayeredDetector(
            layers=[("downsample", dl.Downsample(cfg.oversample))]
        )

        self._graph = None
        if self.use_system_graph:
            self._graph = build_shera_system_graph(
                cfg=self.cfg,
                forward_spec=self.forward_spec,
                base_forward_store=self.base_forward_store,
                detector=self._detector,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_store(self, store_delta: Optional[ParameterStore]) -> ParameterStore:
        """Merge a (possibly partial) store into the base forward store."""

        if store_delta is None:
            return self.base_forward_store

        store_delta = store_delta.validate_against(
            self.forward_spec,
            allow_missing=True,
            allow_extra=False,
            allow_derived=True,
        )
        return self.base_forward_store.replace(store_delta.as_dict())

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        optics = build_shera_threeplane_optics(
            self.cfg, store=eff_store, spec=self.forward_spec
        )
        source = build_alpha_cen_source(eff_store, n_wavels=self.cfg.n_lambda)
        telescope = dl.Telescope(
            source=source,
            optics=optics,
            detector=self._detector,
        )
        return telescope.model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def model(self, store_delta: Optional[ParameterStore] = None) -> jnp.ndarray:
        """
        Evaluate the Shera three-plane PSF for an optional store overlay.

        ``store_delta`` is merged onto the binder's base forward store; when no
        overlay is provided the base store is used directly. This method is the
        canonical public API and replaces the legacy ``.forward`` entry point.
        """

        eff_store = self._merge_store(store_delta)

        if self.use_system_graph and self._graph is not None:
            return self._graph.evaluate(eff_store, outputs=("psf",))

        return self._direct_model(eff_store)

    # ------------------------------------------------------------------
    # Mostly immutable helpers
    # ------------------------------------------------------------------

    def with_store(self, new_base_store: ParameterStore) -> "SheraThreePlaneBinder":
        """Return a new Binder sharing cfg/spec but with a different base store."""

        return SheraThreePlaneBinder(
            cfg=self.cfg,
            forward_spec=self.forward_spec,
            base_forward_store=new_base_store,
            use_system_graph=self.use_system_graph,
        )


@dataclass
class SheraTwoPlaneBinder:
    """Generative model for the Shera two-plane system.

    Mirrors :class:`SheraThreePlaneBinder` semantics: mostly immutable, owns a
    forward-spec-validated base store, and exposes ``.model(store_delta)`` as the
    canonical evaluation path. When ``use_system_graph`` is enabled, the binder
    delegates execution to a lightweight SystemGraph; otherwise a direct builder
    path is used.
    """

    cfg: SheraTwoPlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    use_system_graph: bool = True

    _graph: Optional[object] = None
    _detector: Optional[dl.LayeredDetector] = None

    def __init__(
        self,
        cfg: SheraTwoPlaneConfig,
        forward_spec: ParamSpec,
        base_forward_store: ParameterStore,
        *,
        use_system_graph: bool = True,
    ) -> None:
        self.cfg = cfg
        self.forward_spec = forward_spec
        self.use_system_graph = bool(use_system_graph)

        self.base_forward_store = base_forward_store.validate_against(
            forward_spec, allow_derived=True
        )

        self._detector = dl.LayeredDetector(
            layers=[("downsample", dl.Downsample(cfg.oversample))]
        )

        self._graph = None
        if self.use_system_graph:
            self._graph = build_shera_twoplane_system_graph(
                cfg=self.cfg,
                forward_spec=self.forward_spec,
                base_forward_store=self.base_forward_store,
                detector=self._detector,
            )

    def _merge_store(self, store_delta: Optional[ParameterStore]) -> ParameterStore:
        if store_delta is None:
            return self.base_forward_store

        store_delta = store_delta.validate_against(
            self.forward_spec,
            allow_missing=True,
            allow_derived=True,
            allow_extra=False,
        )
        return self.base_forward_store.replace(store_delta.as_dict())

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:
        optics = build_shera_twoplane_optics(
            self.cfg, store=eff_store, spec=self.forward_spec
        )
        source = build_alpha_cen_source(eff_store, n_wavels=self.cfg.n_lambda)
        telescope = dl.Telescope(
            source=source,
            optics=optics,
            detector=self._detector,
        )
        return telescope.model()

    def model(self, store_delta: Optional[ParameterStore] = None) -> jnp.ndarray:
        eff_store = self._merge_store(store_delta)

        if self.use_system_graph and self._graph is not None:
            return self._graph.evaluate(eff_store, outputs=("psf",))

        return self._direct_model(eff_store)

    def with_store(self, new_base_store: ParameterStore) -> "SheraTwoPlaneBinder":
        return SheraTwoPlaneBinder(
            cfg=self.cfg,
            forward_spec=self.forward_spec,
            base_forward_store=new_base_store,
            use_system_graph=self.use_system_graph,
        )
