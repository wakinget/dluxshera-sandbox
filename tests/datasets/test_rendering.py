from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from dluxshera.datasets import rendering
from dluxshera.params.spec import ParamField, ParamSpec
from dluxshera.params.store import ParameterStore
from work.experiments import generate_training_dataset_v3 as v3


@dataclass(frozen=True)
class Target:
    label: str
    base_key: str
    component_index: int | None
    nominal_value: float


def _spec() -> ParamSpec:
    return ParamSpec(
        [
            ParamField("source.x_position_as", "source", "primitive", default=1.0),
            ParamField("source.y_position_as", "source", "primitive", default=2.0),
            ParamField("source.position_angle_deg", "source", "primitive", default=30.0),
            ParamField(
                "optics.primary.zernike_coeffs_nm",
                "optics",
                "primitive",
                default=np.zeros(3),
                shape=(3,),
            ),
        ]
    )


def test_v3_private_helpers_delegate_to_shared_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"set": 0}

    def fake_set(store, param, value):
        calls["set"] += 1
        return store.replace({param.base_key: value})

    monkeypatch.setattr(rendering, "set_scalar_label", fake_set)
    store = ParameterStore.from_dict({"a": 0.0})
    param = v3.ScalarParameter(
        label="a",
        base_key="a",
        component_index=None,
        nominal_value=0.0,
        parameter_sigma=1.0,
        sweep_source_key="a",
        sweep_config=v3.SweepConfig(),
        min_abs_delta=1.0,
        max_abs_delta=2.0,
    )

    updated = v3._set_scalar_label(store, param, 5.0)

    assert calls["set"] == 1
    assert float(np.asarray(updated.get("a"))) == 5.0


def test_indexed_scalar_updates_preserve_component_placement() -> None:
    store = ParameterStore.from_dict({"z": np.array([1.0, 2.0, 3.0])})
    updated = rendering.set_scalar_label(store, Target("z[1]", "z", 1, 0.0), 9.0)

    np.testing.assert_allclose(np.asarray(updated.get("z")), [1.0, 9.0, 3.0])


def test_v3_nuisance_values_are_physical_deltas() -> None:
    spec = _spec()
    base = ParameterStore.from_spec_defaults(spec)
    target = Target("optics.primary.zernike_coeffs_nm[2]", "optics.primary.zernike_coeffs_nm", 2, 0.0)

    updated = rendering.apply_sample_deltas_to_store(
        base_store=base,
        theta_delta={"optics.primary.zernike_coeffs_nm[2]": 4.0},
        registration_nuisance_values={
            "source.x_position_as": 0.25,
            "source.y_position_as": -0.5,
            "source.position_angle_deg": 2.0,
        },
        parameters_by_label={target.label: target},
        forward_spec=spec,
        registration_nuisance_keys=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
    )

    assert float(np.asarray(updated.get("source.x_position_as"))) == pytest.approx(1.25)
    assert float(np.asarray(updated.get("source.y_position_as"))) == pytest.approx(1.5)
    assert float(np.asarray(updated.get("source.position_angle_deg"))) == pytest.approx(32.0)
    np.testing.assert_allclose(
        np.asarray(updated.get("optics.primary.zernike_coeffs_nm")),
        [0.0, 0.0, 4.0],
    )


def test_v4_absolute_science_then_delta_nuisance_semantics() -> None:
    spec = _spec()
    base = ParameterStore.from_spec_defaults(spec)

    updated = rendering.apply_absolute_vector_to_store(
        base_store=base,
        labels=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
            "optics.primary.zernike_coeffs_nm[1]",
        ),
        physical_values=(10.0, 20.0, 45.0, 7.0),
        forward_spec=spec,
        component_indices={"optics.primary.zernike_coeffs_nm[1]": 1},
        nuisance_labels=(
            "source.x_position_as",
            "source.y_position_as",
            "source.position_angle_deg",
        ),
        nuisance_physical_deltas=(0.5, -1.0, 3.0),
    )

    assert float(np.asarray(updated.get("source.x_position_as"))) == pytest.approx(10.5)
    assert float(np.asarray(updated.get("source.y_position_as"))) == pytest.approx(19.0)
    assert float(np.asarray(updated.get("source.position_angle_deg"))) == pytest.approx(48.0)
    np.testing.assert_allclose(
        np.asarray(updated.get("optics.primary.zernike_coeffs_nm")),
        [0.0, 7.0, 0.0],
    )
