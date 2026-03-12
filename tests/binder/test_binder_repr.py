from dluxshera.systems import SheraBinder
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG
from tests.conftest import make_forward_store


def test_binder_repr_includes_components():
    forward_spec, forward_store = make_forward_store(SHERA_TESTBED_CONFIG)
    binder = SheraBinder(SHERA_TESTBED_CONFIG, forward_spec, forward_store)

    rendered = repr(binder)

    assert "SheraBinder(" in rendered
    assert "source=" in rendered
    assert "optics=" in rendered
    assert "detector=" in rendered

