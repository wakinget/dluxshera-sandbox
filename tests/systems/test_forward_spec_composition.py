from __future__ import annotations

import pytest

from dluxshera.components.optics import SheraThreePlaneOptics, SheraTwoPlaneOptics
from dluxshera.components.sources import build_alpha_cen_contract
from dluxshera.params.spec import ParamField, ParamSpec
from dluxshera.systems.base import compose_forward_spec
from dluxshera.systems.three_plane import SheraThreePlaneConfig, build_forward_spec_from_config
from dluxshera.builders.detector import build_detector_contract


def _system_cfg(optics_kind: str) -> dict:
    source_block = {
        "kind": "alpha_cen",
        "wavelength_m": 550e-9,
        "bandwidth_m": 100e-9,
        "n_lambda": 1,
        "exposure_time_s": 1.0,
        "target": "ALPHA_CEN",
        "separation_as": 10.0,
        "position_angle_deg": 0.0,
        "contrast": 3.0,
    }

    if optics_kind == "two_plane":
        optics_block = {
            "kind": "two_plane",
            "pupil_npix": 32,
            "psf_npix": 32,
            "oversample": 1,
            "m1_diameter_m": 0.1,
            "m2_diameter_m": 0.02,
            "n_struts": 3,
            "strut_width_m": 0.001,
            "strut_rotation_deg": 0.0,
            "throughput": 1.0,
            "dp_design_wavelength_m": 550e-9,
            "primary_noll_indices": [],
            "plate_scale_as_per_pix": 0.1,
        }
    else:
        optics_block = {
            "kind": "three_plane",
            "pupil_npix": 32,
            "psf_npix": 32,
            "oversample": 1,
            "pixel_pitch_m": 6.5e-6,
            "m1_diameter_m": 0.1,
            "m2_diameter_m": 0.02,
            "m1_focal_length_m": 1.0,
            "m2_focal_length_m": -0.5,
            "m1_m2_separation_m": 0.3,
            "n_struts": 3,
            "strut_width_m": 0.001,
            "strut_rotation_deg": 0.0,
            "throughput": 1.0,
            "dp_design_wavelength_m": 550e-9,
            "primary_noll_indices": [],
            "secondary_noll_indices": [],
        }

    detector_block = {"model": "GSENSE2020BSI", "layers": []}

    return {"system": {"source": source_block, "optics": optics_block, "detector": detector_block}}


@pytest.mark.parametrize(
    "system_cfg,optics_builder",
    [
        (_system_cfg("two_plane"), SheraTwoPlaneOptics.contract),
        (_system_cfg("three_plane"), SheraThreePlaneOptics.contract),
    ],
)
def test_compose_forward_spec_contains_union_of_component_contract_keys(system_cfg, optics_builder):
    source_contract = build_alpha_cen_contract(system_cfg["system"]["source"])
    optics_contract = optics_builder(system_cfg["system"]["optics"])
    detector_contract = build_detector_contract(system_cfg["system"]["detector"])

    composed = compose_forward_spec(system_cfg)

    expected = (
        set(source_contract.keys())
        | set(optics_contract.keys())
        | set(detector_contract.keys())
    )
    assert set(composed.keys()) == expected


@pytest.mark.parametrize("system_cfg", [_system_cfg("two_plane"), _system_cfg("three_plane")])
def test_compose_forward_spec_ordering_is_stable(system_cfg):
    source_contract = build_alpha_cen_contract(system_cfg["system"]["source"])

    composed_1 = compose_forward_spec(system_cfg)
    composed_2 = compose_forward_spec(system_cfg)

    keys = list(composed_1.keys())
    assert keys == list(composed_2.keys())

    source_keys = set(source_contract.keys())
    detector_keys = {k for k in composed_1.keys() if k.startswith("detector.")}

    source_last = max(i for i, key in enumerate(keys) if key in source_keys)
    detector_first = min(i for i, key in enumerate(keys) if key in detector_keys)
    optics_indices = [
        i
        for i, key in enumerate(keys)
        if key not in source_keys and key not in detector_keys
    ]

    assert source_last < min(optics_indices)
    assert max(optics_indices) < detector_first


def test_compose_forward_spec_raises_on_key_collisions(monkeypatch):
    system_cfg = _system_cfg("two_plane")

    colliding_detector_contract = ParamSpec(
        [
            ParamField(
                key="source.contrast",
                group="detector",
                kind="primitive",
                dtype=float,
                shape=(),
                default=1.0,
            )
        ]
    )

    def fake_build_detector_contract(_):
        return colliding_detector_contract

    import dluxshera.builders.detector as detector_mod

    monkeypatch.setattr(detector_mod, "build_detector_contract", fake_build_detector_contract)

    with pytest.raises(ValueError, match="key collision"):
        compose_forward_spec(system_cfg)


def test_compose_forward_spec_accepts_minimal_single_star_source():
    system_cfg = _system_cfg("two_plane")
    system_cfg["system"]["source"] = {
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

    spec = compose_forward_spec(system_cfg)

    assert "source.log_flux_total" in spec
    assert "source.x_position_as" in spec
    assert "source.y_position_as" in spec
    assert "source.position_angle_deg" in spec
    assert "source.separation_as" not in spec
    assert "source.contrast" not in spec


def test_compose_forward_spec_accepts_generic_binary_source():
    system_cfg = _system_cfg("two_plane")
    system_cfg["system"]["source"] = {
        "kind": "binary",
        "wavelength_m": 650e-9,
        "bandwidth_m": 100e-9,
        "n_lambda": 11,
        "exposure_time_s": 0.05,
        "x_position_as": 0.0,
        "y_position_as": 0.0,
        "separation_as": 4.0,
        "position_angle_deg": 90.0,
        "log_flux_total": 6.0,
        "contrast": 1.5,
    }

    spec = compose_forward_spec(system_cfg)

    assert "source.log_flux_total" in spec
    assert "source.separation_as" in spec
    assert "source.contrast" in spec
    assert "source.target" not in spec
    assert "source.raw_fluxes" not in spec


def test_legacy_wrapper_delegates_to_composed_forward_spec_three_plane():
    cfg = SheraThreePlaneConfig()
    system_cfg = _system_cfg("three_plane")["system"]
    system_cfg["detector"]["layers"] = cfg.detector_layers
    system_cfg["detector"]["model"] = cfg.detector_model

    legacy_spec = build_forward_spec_from_config(cfg)

    delegated_spec = compose_forward_spec({"system": system_cfg})

    assert list(legacy_spec.keys()) == list(delegated_spec.keys())
    assert {
        key: legacy_spec.get(key).shape for key in legacy_spec.keys()
    } == {
        key: delegated_spec.get(key).shape for key in delegated_spec.keys()
    }
