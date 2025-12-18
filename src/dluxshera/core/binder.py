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
from ..params.store_namespace import StoreNamespace


BINDER_RESERVED_NAMES = {
    "cfg",
    "forward_spec",
    "base_forward_store",
    "get",
    "ns",
    "model",
    "with_store",
    "use_system_graph",
}
from .universe import build_alpha_cen_source


class BaseSheraBinder:
    """Shared backbone for Shera binder implementations.

    Encapsulates the common binder behaviour: storing config/spec/base-store,
    eager detector + optional SystemGraph construction, functional store merge,
    and the public ``.model`` / ``.with_store`` helpers. Concrete subclasses
    remain the public entry points and supply system-specific optics/graph
    builders via protected hooks.
    """

    def __init__(
        self,
        cfg,
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

        # Shared detector construction; subclasses can override if needed.
        self._detector = self._build_detector()

        self._graph = None
        if self.use_system_graph:
            self._graph = self._build_graph()

    def __getattr__(self, name):
        if name in BINDER_RESERVED_NAMES:
            raise AttributeError(name)

        if hasattr(self.cfg, name):
            return getattr(self.cfg, name)

        raise AttributeError(name)

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _build_detector(self) -> dl.LayeredDetector:
        return dl.LayeredDetector(layers=[("downsample", dl.Downsample(self.cfg.oversample))])

    def _build_graph(self):  # pragma: no cover - abstract hook
        raise NotImplementedError

    def _direct_model(self, eff_store: ParameterStore) -> jnp.ndarray:  # pragma: no cover - abstract hook
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, paths, default=None):
        """Retrieve values from the binder configuration or base store."""

        if isinstance(paths, (list, tuple)):
            return [self.get(path, default=default) for path in paths]

        path = paths
        if isinstance(path, str) and "." in path:
            if default is None:
                return self.base_forward_store.get(path)
            return self.base_forward_store.get(path, default)

        if hasattr(self.cfg, path):
            return getattr(self.cfg, path)

        if default is None:
            return self.base_forward_store.get(path)
        return self.base_forward_store.get(path, default)

    def ns(self, prefix: str) -> StoreNamespace:
        """Return a StoreNamespace proxy for a prefix in the base forward store."""

        if not isinstance(prefix, str) or not prefix.isidentifier():
            raise ValueError(f"Invalid namespace prefix: {prefix!r}")

        if prefix in BINDER_RESERVED_NAMES:
            raise ValueError(f"Namespace prefix {prefix!r} is reserved")

        has_prefix = any(
            key.startswith(f"{prefix}.") for key in self.base_forward_store.keys()
        )
        if not has_prefix:
            raise ValueError(f"No store keys found under prefix {prefix!r}")

        return StoreNamespace(self.base_forward_store, prefix)

    def model(self, store_delta: Optional[ParameterStore] = None) -> jnp.ndarray:
        """Evaluate the Shera PSF for an optional store overlay."""

        eff_store = self._merge_store(store_delta)

        if self.use_system_graph and self._graph is not None:
            return self._graph.evaluate(eff_store, outputs=("psf",))

        return self._direct_model(eff_store)

    # ------------------------------------------------------------------
    # Mostly immutable helpers
    # ------------------------------------------------------------------

    def with_store(self, new_base_store: ParameterStore):
        """Return a new Binder sharing cfg/spec but with a different base store."""

        return self.__class__(
            cfg=self.cfg,
            forward_spec=self.forward_spec,
            base_forward_store=new_base_store,
            use_system_graph=self.use_system_graph,
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
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
            use_system_graph=use_system_graph,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_graph(self):
        return build_shera_system_graph(
            cfg=self.cfg,
            forward_spec=self.forward_spec,
            base_forward_store=self.base_forward_store,
            detector=self._detector,
        )

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

    with_store = BaseSheraBinder.with_store


@dataclass
class SheraTwoPlaneBinder(BaseSheraBinder):
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
        super().__init__(
            cfg=cfg,
            forward_spec=forward_spec,
            base_forward_store=base_forward_store,
            use_system_graph=use_system_graph,
        )

    def _build_graph(self):
        return build_shera_twoplane_system_graph(
            cfg=self.cfg,
            forward_spec=self.forward_spec,
            base_forward_store=self.base_forward_store,
            detector=self._detector,
        )

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
    with_store = BaseSheraBinder.with_store
