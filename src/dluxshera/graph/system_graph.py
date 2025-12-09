"""
Minimal SystemGraph wrapper for Shera three-plane execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..optics.config import SheraThreePlaneConfig
from ..params.spec import ParamSpec
from ..params.store import ParameterStore
from ..core.builder import build_shera_threeplane_model
from .nodes import DLuxSystemNode


@dataclass
class SystemGraph:
    """
    Minimal single-node system graph.

    For now this is intentionally lightweight and assumes a single
    ``DLuxSystemNode`` representing the Shera three-plane system. Future
    variants can extend this to multiple nodes / DAG execution as needed.
    """

    node: DLuxSystemNode

    def forward(self, store: Optional[ParameterStore] = None):
        return self.node.forward(store)

    run = forward


def build_threeplane_system_graph(
    cfg: SheraThreePlaneConfig,
    inference_spec: ParamSpec,
    base_store: ParameterStore,
):
    """Factory for the default three-plane SystemGraph."""
    node = DLuxSystemNode(
        cfg=cfg,
        inference_spec=inference_spec,
        base_store=base_store,
        build_model_fn=build_shera_threeplane_model,
    )
    return SystemGraph(node=node)
