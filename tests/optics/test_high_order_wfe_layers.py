import jax.numpy as jnp

from dluxshera.components.optics import SheraTwoPlaneOptics, SheraThreePlaneOptics


def test_two_plane_always_has_high_order_layer():
    o = SheraTwoPlaneOptics(wf_npixels=32, psf_npixels=16)
    assert hasattr(o, "high_order_wfe")


def test_three_plane_always_has_high_order_layers():
    o = SheraThreePlaneOptics(wf_npixels=32, psf_npixels=16)
    assert "high_order_wfe" in o.p1_layers
    assert "high_order_wfe" in o.p2_layers


def test_units_nm_to_m_on_layer_payload():
    nm = 5.0
    arrm = jnp.ones((32, 32)) * nm * 1e-9
    o = SheraTwoPlaneOptics(wf_npixels=32, psf_npixels=16, high_order_wfe_opd_m=arrm)
    assert float(o.high_order_wfe.opd.mean()) == float(arrm.mean())
