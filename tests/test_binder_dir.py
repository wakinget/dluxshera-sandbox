from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.optics.config import SHERA_TESTBED_CONFIG

from .helpers import make_forward_store


def _make_binder():
    forward_spec, forward_store = make_forward_store(SHERA_TESTBED_CONFIG)
    return SheraThreePlaneBinder(
        SHERA_TESTBED_CONFIG,
        forward_spec,
        forward_store,
        use_system_graph=False,
    )


def test_dir_includes_cfg_fields_prefixes_and_unambiguous_leaves():
    binder = _make_binder()

    entries = dir(binder)

    assert "psf_npix" in entries  # cfg field
    assert "system" in entries  # store prefix
    assert "plate_scale_as_per_pix" in entries  # unambiguous leaf
    assert "zernike_coeffs_nm" not in entries  # ambiguous leaf omitted
