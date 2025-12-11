"""Shera three-plane SystemGraph owned by :class:`SheraThreePlaneBinder`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import dLux as dl

from ..core.universe import build_alpha_cen_source
from ..optics.builder import build_shera_threeplane_optics, build_shera_twoplane_optics
from ..optics.config import SheraThreePlaneConfig, SheraTwoPlaneConfig
from ..params.spec import ParamSpec
from ..params.store import ParameterStore


Outputs = Tuple[str, ...]


@dataclass
class SystemGraph:
    """
    Minimal single-node execution graph for the Shera three-plane system.

    This graph owns a detector instance and reuses the cached optics builder to
    evaluate the PSF for a provided ParameterStore. It mirrors the binder’s
    mostly-immutable semantics: the base store is validated and stored, while
    per-call overrides are merged functionally.
    """

    cfg: SheraThreePlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    detector: dl.LayeredDetector

    def __post_init__(self) -> None:
        self.base_forward_store = self.base_forward_store.validate_against(
            self.forward_spec, allow_derived=True
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

    def evaluate(
        self,
        store_delta: Optional[ParameterStore] = None,
        *,
        outputs: Iterable[str] = ("psf",),
    ):
        """Evaluate the graph and return requested outputs (default: PSF)."""

        eff_store = self._merge_store(store_delta)

        optics = build_shera_threeplane_optics(
            self.cfg, store=eff_store, spec=self.forward_spec
        )
        source = build_alpha_cen_source(eff_store, n_wavels=self.cfg.n_lambda)
        telescope = dl.Telescope(source=source, optics=optics, detector=self.detector)

        psf = telescope.model()

        if outputs == ("psf",):
            return psf

        if isinstance(outputs, tuple):
            return {name: psf for name in outputs}

        return psf

    forward = evaluate
    run = evaluate


def build_shera_system_graph(
    cfg: SheraThreePlaneConfig,
    forward_spec: ParamSpec,
    base_forward_store: ParameterStore,
    detector: Optional[dl.LayeredDetector] = None,
) -> SystemGraph:
    """Factory for the default Shera three-plane SystemGraph."""

    detector = detector or dl.LayeredDetector(layers=[("downsample", dl.Downsample(cfg.oversample))])
    return SystemGraph(
        cfg=cfg,
        forward_spec=forward_spec,
        base_forward_store=base_forward_store,
        detector=detector,
    )


# Backwards-compatible alias for older call sites and tests.
build_threeplane_system_graph = build_shera_system_graph


@dataclass
class SheraTwoPlaneSystemGraph:
    """SystemGraph for the Shera two-plane binder path."""

    cfg: SheraTwoPlaneConfig
    forward_spec: ParamSpec
    base_forward_store: ParameterStore
    detector: dl.LayeredDetector

    def __post_init__(self) -> None:
        self.base_forward_store = self.base_forward_store.validate_against(
            self.forward_spec, allow_derived=True
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

    def evaluate(
        self,
        store_delta: Optional[ParameterStore] = None,
        *,
        outputs: Iterable[str] = ("psf",),
    ):
        eff_store = self._merge_store(store_delta)

        optics = build_shera_twoplane_optics(
            self.cfg, store=eff_store, spec=self.forward_spec
        )
        source = build_alpha_cen_source(eff_store, n_wavels=self.cfg.n_lambda)
        telescope = dl.Telescope(source=source, optics=optics, detector=self.detector)

        psf = telescope.model()

        if outputs == ("psf",):
            return psf

        if isinstance(outputs, tuple):
            return {name: psf for name in outputs}

        return psf

    forward = evaluate
    run = evaluate


def build_shera_twoplane_system_graph(
    cfg: SheraTwoPlaneConfig,
    forward_spec: ParamSpec,
    base_forward_store: ParameterStore,
    detector: Optional[dl.LayeredDetector] = None,
) -> SheraTwoPlaneSystemGraph:
    detector = detector or dl.LayeredDetector(
        layers=[("downsample", dl.Downsample(cfg.oversample))]
    )
    return SheraTwoPlaneSystemGraph(
        cfg=cfg,
        forward_spec=forward_spec,
        base_forward_store=base_forward_store,
        detector=detector,
    )
