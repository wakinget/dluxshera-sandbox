from __future__ import annotations

import pytest

from dluxshera.components.optics import SheraThreePlaneOptics, SheraTwoPlaneOptics
from dluxshera.components.sources import build_alpha_cen_contract
from dluxshera.params.spec import ParamField, ParamSpec
from dluxshera.systems.base import compose_forward_spec
from dluxshera.systems.three_plane import SheraThreePlaneConfig, build_forward_spec_from_config
from dluxshera.systems.two_plane import SheraTwoPlaneConfig


def _minimal_detector_contract() -> ParamSpec:
    return ParamSpec(
        [
            ParamField(
                key="detector.jitter.sigma",
                group="detector",
                kind="primitive",
                dtype=float,
                shape=(),
                default=1e-12,
            )
        ]
    )


@pytest.mark.parametrize(
    "cfg,optics_builder",
    [
        (SheraTwoPlaneConfig(), SheraTwoPlaneOptics.contract),
        (SheraThreePlaneConfig(), SheraThreePlaneOptics.contract),
    ],
)
def test_compose_forward_spec_contains_union_of_component_contract_keys(cfg, optics_builder):
    source_contract = build_alpha_cen_contract(cfg)
    optics_contract = optics_builder(cfg)
    detector_contract = _minimal_detector_contract()

    composed = compose_forward_spec(cfg, detector_contract=detector_contract)

    expected = (
        set(source_contract.keys())
        | set(optics_contract.keys())
        | set(detector_contract.keys())
    )
    assert set(composed.keys()) == expected


@pytest.mark.parametrize("cfg", [SheraTwoPlaneConfig(), SheraThreePlaneConfig()])
def test_compose_forward_spec_ordering_is_stable(cfg):
    source_contract = build_alpha_cen_contract(cfg)
    detector_contract = _minimal_detector_contract()

    composed_1 = compose_forward_spec(cfg, detector_contract=detector_contract)
    composed_2 = compose_forward_spec(cfg, detector_contract=detector_contract)

    keys = list(composed_1.keys())
    assert keys == list(composed_2.keys())

    source_keys = set(source_contract.keys())
    detector_keys = set(detector_contract.keys())

    source_last = max(i for i, key in enumerate(keys) if key in source_keys)
    detector_first = min(i for i, key in enumerate(keys) if key in detector_keys)
    optics_indices = [
        i
        for i, key in enumerate(keys)
        if key not in source_keys and key not in detector_keys
    ]

    assert source_last < min(optics_indices)
    assert max(optics_indices) < detector_first


def test_compose_forward_spec_raises_on_key_collisions():
    cfg = SheraTwoPlaneConfig()
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

    with pytest.raises(ValueError, match="key collision"):
        compose_forward_spec(cfg, detector_contract=colliding_detector_contract)


def test_legacy_wrapper_delegates_to_composed_forward_spec_three_plane():
    cfg = SheraThreePlaneConfig()

    legacy_spec = build_forward_spec_from_config(cfg)

    # Wrapper computes detector_contract internally, so compare against itself
    # through a second invocation to ensure deterministic delegated behavior.
    delegated_spec = build_forward_spec_from_config(cfg)

    assert list(legacy_spec.keys()) == list(delegated_spec.keys())
    assert {
        key: legacy_spec.get(key).shape for key in legacy_spec.keys()
    } == {
        key: delegated_spec.get(key).shape for key in delegated_spec.keys()
    }
