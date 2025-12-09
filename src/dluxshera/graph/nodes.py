"""
Minimal node abstractions for the dLuxShera SystemGraph layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..optics.config import SheraThreePlaneConfig
from ..params.spec import ParamSpec
from ..params.store import ParameterStore
from ..core.builder import build_shera_threeplane_model


ModelBuilder = Callable[[SheraThreePlaneConfig, ParamSpec, ParameterStore], Any]


@dataclass
class DLuxSystemNode:
    """
    Thin wrapper turning (cfg, spec, store) into a model or PSF image.

    For P0 we intentionally keep this minimal: the node simply calls a
    `build_model_fn` with the provided configuration, ParamSpec, and an
    effective ParameterStore (base_store overlaid with any updates). When the
    returned object exposes a ``model`` method (as ``dLux.Telescope`` does),
    ``forward`` will evaluate it to yield a PSF image; otherwise the object is
    returned directly.
    """

    cfg: SheraThreePlaneConfig
    inference_spec: ParamSpec
    base_store: ParameterStore
    build_model_fn: ModelBuilder = build_shera_threeplane_model

    def __post_init__(self) -> None:
        # Validate and freeze the base store against the provided spec so we
        # have a consistent starting point for all forward calls.
        self.base_store = self.base_store.validate_against(self.inference_spec)

    def _merge_store(self, store: Optional[ParameterStore]) -> ParameterStore:
        """Overlay updates from ``store`` on top of ``base_store``."""
        if store is None:
            return self.base_store
        return self.base_store.replace(store.as_dict())

    def forward(self, store: Optional[ParameterStore] = None, *, return_model: bool = False):
        """Build and/or execute the system for the provided store."""
        eff_store = self._merge_store(store)
        model = self.build_model_fn(self.cfg, self.inference_spec, eff_store)

        if return_model:
            return model

        model_fn = getattr(model, "model", None)
        if callable(model_fn):
            return model_fn()
        return model
