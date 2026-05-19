from dluxshera.components.sources import (
    binary_component_fluxes_from_total_and_contrast,
    binary_mean_flux_from_total_and_contrast,
    build_alpha_cen_contract,
    build_binary_contract,
    build_binary_target_contract,
    build_single_star_contract,
    compute_source_flux_diagnostics,
    get_target_spec,
    linear_total_flux_from_log10,
)
from dluxshera.params.store import ParameterStore
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


def test_single_star_contract_omits_binary_only_keys():
    spec = build_single_star_contract(
        {
            "kind": "single_star",
            "wavelength_m": 650e-9,
            "bandwidth_m": 100e-9,
            "n_lambda": 11,
            "exposure_time_s": 0.05,
            "x_position_as": 0.0,
            "y_position_as": 0.0,
            "position_angle_deg": 0.0,
            "log_flux_total": 6.0,
        }
    )

    expected = {
        "source.wavelength_m",
        "source.bandwidth_m",
        "source.n_lambda",
        "source.exposure_time_s",
        "source.x_position_as",
        "source.y_position_as",
        "source.position_angle_deg",
        "source.log_flux_total",
    }
    assert set(spec.keys()) == expected
    assert "source.separation_as" not in spec
    assert "source.contrast" not in spec
    assert "source.raw_fluxes" not in spec
    assert spec.get("source.log_flux_total").kind == "primitive"


def test_generic_binary_contract_uses_public_log_total_flux():
    spec = build_binary_contract(
        {
            "kind": "binary",
            "wavelength_m": 650e-9,
            "bandwidth_m": 100e-9,
            "n_lambda": 11,
            "exposure_time_s": 0.05,
            "separation_as": 4.0,
            "position_angle_deg": 90.0,
            "log_flux_total": 6.0,
            "contrast": 1.5,
        }
    )

    assert "source.log_flux_total" in spec
    assert spec.get("source.log_flux_total").kind == "primitive"
    assert "source.separation_as" in spec
    assert "source.contrast" in spec
    assert "source.target" not in spec
    assert "source.raw_fluxes" not in spec


def test_flux_helpers_use_total_flux_convention():
    total = linear_total_flux_from_log10(6.0)
    primary, secondary = binary_component_fluxes_from_total_and_contrast(total, 3.0)

    assert float(total) == 1.0e6
    assert float(primary + secondary) == 1.0e6
    assert float(primary / secondary) == 3.0
    assert float(binary_mean_flux_from_total_and_contrast(total, 3.0)) == 5.0e5


def test_source_flux_diagnostics_are_source_kind_aware():
    single = ParameterStore.from_dict({"source.log_flux_total": 6.0})
    single_diag = compute_source_flux_diagnostics("single_star", single)
    assert float(single_diag["total_flux"]) == 1.0e6
    assert set(single_diag["component_fluxes"]) == {"star"}

    binary = ParameterStore.from_dict(
        {
            "source.log_flux_total": 6.0,
            "source.contrast": 3.0,
        }
    )
    binary_diag = compute_source_flux_diagnostics("binary_target", binary)
    primary = binary_diag["component_fluxes"]["primary"]
    secondary = binary_diag["component_fluxes"]["secondary"]
    assert float(primary + secondary) == 1.0e6
    assert float(primary / secondary) == 3.0
