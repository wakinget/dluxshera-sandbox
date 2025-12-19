from __future__ import annotations

import math

import pytest

from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from dluxshera.params.spec import ParamField, ParamSpec, build_forward_model_spec_from_config
from dluxshera.params.store import ParameterStore, refresh_derived


def test_refresh_derived_lazy_registration_threeplane():
    """Derived transforms load automatically when refreshing a store."""

    spec = build_forward_model_spec_from_config(SHERA_TESTBED_CONFIG)
    store = ParameterStore.from_spec_defaults(spec).replace(
        {
            "imaging.exposure_time_s": 2.0,
            "imaging.throughput": 0.9,
            "binary.spectral_flux_density": 2.0e17,
        }
    )

    refreshed = store.refresh_derived(spec)

    assert "system.plate_scale_as_per_pix" in refreshed
    assert "binary.log_flux_total" in refreshed
    # Compare against a simple analytic log-flux to ensure transforms executed.
    D = float(refreshed.get("system.m1_diameter_m"))
    bandwidth_m = float(refreshed.get("band.bandwidth_m"))
    t_exp = float(refreshed.get("imaging.exposure_time_s"))
    throughput = float(refreshed.get("imaging.throughput"))
    flux_density = float(refreshed.get("binary.spectral_flux_density"))

    area = math.pi * (D / 2.0) ** 2
    expected_log_flux = math.log10(flux_density * bandwidth_m * area * t_exp * throughput)

    assert refreshed.get("binary.log_flux_total") == pytest.approx(expected_log_flux)


def test_refresh_derived_deterministic_ordering():
    calls: list[str] = []

    class RecordingResolver:
        def compute(self, key, store, system_id=None):
            calls.append(key)
            return f"value-{key}"

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
    assert refreshed.get("alpha") == "value-alpha"
    assert refreshed.get("beta") == "value-beta"


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
