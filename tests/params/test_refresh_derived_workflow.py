from __future__ import annotations

import pytest

from dluxshera.utils.source_photometry import (
    build_wavelength_grid_m,
    derive_source_photometry,
)
from dluxshera.params.spec import ParamField, ParamSpec
from dluxshera.params.store import ParameterStore, refresh_derived
from dluxshera.systems.three_plane import SHERA_TESTBED_CONFIG, build_forward_spec_from_config


def test_refresh_derived_lazy_registration_threeplane():
    """Derived transforms load automatically when refreshing a store."""

    spec = build_forward_spec_from_config(SHERA_TESTBED_CONFIG)
    store = ParameterStore.from_spec_defaults(spec).replace(
        {
            "source.target": None,
            "source.vmag_a": 1.5,
            "source.vmag_b": 2.2,
            "source.exposure_time_s": 2.0,
            "optics.throughput": 0.9,
        }
    )

    refreshed = store.refresh_derived(spec)

    assert "optics.plate_scale_as_per_pix" in refreshed
    assert "source.log_flux_total" in refreshed

    D = float(refreshed.get("optics.m1_diameter_m"))
    wavelength_m = float(refreshed.get("source.wavelength_m"))
    bandwidth_m = float(refreshed.get("source.bandwidth_m"))
    n_lambda = int(refreshed.get("source.n_lambda"))
    t_exp = float(refreshed.get("source.exposure_time_s"))
    throughput = float(refreshed.get("optics.throughput"))
    area_m2 = float(3.141592653589793 * (D / 2.0) ** 2)

    wavelength_grid_m = build_wavelength_grid_m(
        wavelength_m=wavelength_m,
        bandwidth_m=bandwidth_m,
        n_lambda=n_lambda,
    )
    expected = derive_source_photometry(
        wavelength_grid_m=wavelength_grid_m,
        bandwidth_m=bandwidth_m,
        collecting_area_m2=area_m2,
        exposure_time_s=t_exp,
        throughput=throughput,
        sed_a_path=None,
        sed_b_path=None,
        vmag_a=1.5,
        vmag_b=2.2,
    )

    assert refreshed.get("source.log_flux_total") == pytest.approx(expected.log_flux_total)


def test_refresh_derived_deterministic_ordering():
    calls: list[str] = []

    class RecordingResolver:
        def compute(self, key, store, system_id=None):
            calls.append(key)
            return {"alpha": 1.0, "beta": 2.0}[key]

    spec = ParamSpec(
        [
            ParamField("primitive", group="g", kind="primitive"),
            ParamField("beta", group="g", kind="derived"),
            ParamField("alpha", group="g", kind="derived"),
        ]
    )
    store = ParameterStore.from_dict({"primitive": 1})

    refreshed = refresh_derived(store, spec, resolver=RecordingResolver())

    assert calls == ["alpha", "beta"]
    assert refreshed.get("alpha") == 1.0
    assert refreshed.get("beta") == 2.0


def test_refresh_derived_recomputes_existing_values():
    class SimpleResolver:
        def compute(self, key, store, system_id=None):
            return 123

    spec = ParamSpec(
        [
            ParamField("primitive", group="g", kind="primitive"),
            ParamField("derived", group="g", kind="derived"),
        ]
    )
    store = ParameterStore.from_dict({"primitive": 1, "derived": 5})

    refreshed = refresh_derived(store, spec, resolver=SimpleResolver())

    assert refreshed.get("primitive") == 1
    assert refreshed.get("derived") == 123
