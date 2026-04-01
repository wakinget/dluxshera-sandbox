from dluxshera.components.sources import (
    build_alpha_cen_contract,
    build_binary_target_contract,
    get_target_spec,
)
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG


def test_alpha_cen_contract_keys_and_structural_flags():
    spec = build_alpha_cen_contract(SHERA_TESTBED_CONFIG)

    expected_structural = {
        "source.wavelength_m": True,
        "source.bandwidth_m": True,
        "source.n_lambda": True,
        "source.exposure_time_s": False,
        "source.target": False,
        "source.vmag_a": False,
        "source.vmag_b": False,
        "source.separation_as": False,
        "source.position_angle_deg": False,
        "source.x_position_as": False,
        "source.y_position_as": False,
        "source.log_flux_total": False,
        "source.contrast": False,
        "source.raw_fluxes": False,
    }

    assert set(spec.keys()) == set(expected_structural.keys())
    for key, structural in expected_structural.items():
        field = spec.get(key)
        expected_shape = (2,) if key == "source.raw_fluxes" else ()
        assert field.shape == expected_shape
        assert field.structural is structural


def test_binary_target_contract_uses_target_seed_defaults():
    spec = build_binary_target_contract(
        {
            "source": {
                "kind": "binary_target",
                "target": "ALPHA_CEN",
                "wavelength_m": 550e-9,
                "bandwidth_m": 100e-9,
                "n_lambda": 3,
            }
        }
    )

    alpha = get_target_spec("ALPHA_CEN")
    assert spec.get("source.separation_as").default == alpha.nominal_separation_as
    assert spec.get("source.position_angle_deg").default == alpha.nominal_position_angle_deg
    assert spec.get("source.target").default == "ALPHA_CEN"
    assert spec.get("source.vmag_a").default == alpha.vmag_a
    assert spec.get("source.vmag_b").default == alpha.vmag_b
    assert spec.get("source.contrast").default > 0.0
