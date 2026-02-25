from dluxshera.components.sources import build_alpha_cen_contract
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG


def test_alpha_cen_contract_keys_and_structural_flags():
    spec = build_alpha_cen_contract(SHERA_TESTBED_CONFIG)

    expected_structural = {
        "source.wavelength_m": True,
        "source.bandwidth_m": True,
        "source.n_lambda": True,
        "source.separation_as": False,
        "source.position_angle_deg": False,
        "source.x_position_as": False,
        "source.y_position_as": False,
        "source.log_flux_total": False,
        "source.contrast": False,
    }

    assert set(spec.keys()) == set(expected_structural.keys())
    for key, structural in expected_structural.items():
        field = spec.get(key)
        assert field.shape == ()
        assert field.structural is structural

