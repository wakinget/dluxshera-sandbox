# tests/test_modeling_components.py

import dLuxToliman as dlT

from dluxshera.params.spec import build_inference_spec_basic
from dluxshera.params.store import ParameterStore
from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.optics.optical_systems import SheraThreePlaneSystem
from dluxshera.core.modeling import (
    SheraThreePlaneComponents,
    build_shera_threeplane_components,
)


def test_build_shera_threeplane_components_smoke():
    cfg = SHERA_TESTBED_CONFIG
    spec = build_inference_spec_basic()
    store = ParameterStore.from_spec_defaults(spec)

    components = build_shera_threeplane_components(cfg, spec, store)

    # Basic type sanity
    assert isinstance(components, SheraThreePlaneComponents)
    assert isinstance(components.optics, SheraThreePlaneSystem)
    assert isinstance(components.source, dlT.AlphaCen)

    # The bundle should preserve the exact cfg/spec/store passed in
    assert components.cfg is cfg
    assert components.spec is spec
    # store gets validated (returns a new instance), so equality not identity
    assert components.store.as_dict() == store.as_dict()
