from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "examples" / "recipes" / "prescribed_monte_carlo.py"
    spec = importlib.util.spec_from_file_location("prescribed_monte_carlo", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load prescribed_monte_carlo module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_knowledge_error_reproducible_gaussian():
    m = _load_module()
    from dluxshera.utils.noise import apply_knowledge_error

    base = jnp.ones((2, 2))
    cfg = {"model": "gaussian", "scale": 0.1}
    perturbed1, seed1 = apply_knowledge_error(base, knowledge_cfg=cfg, base_seed=123, token="dx")
    perturbed2, seed2 = apply_knowledge_error(base, knowledge_cfg=cfg, base_seed=123, token="dx")

    assert seed1 is not None and seed2 is not None
    assert jnp.allclose(perturbed1, perturbed2)


def test_apply_knowledge_error_uniform_reproducible():
    from dluxshera.utils.noise import apply_knowledge_error

    base = jnp.ones((2, 2))
    cfg = {"model": "uniform", "sigma": 0.05}
    perturbed1, _ = apply_knowledge_error(base, knowledge_cfg=cfg, base_seed=99, token="prf")
    perturbed2, _ = apply_knowledge_error(base, knowledge_cfg=cfg, base_seed=99, token="prf")

    assert jnp.allclose(perturbed1, perturbed2)


def test_apply_knowledge_error_noop_without_config():
    from dluxshera.utils.noise import apply_knowledge_error

    base = jnp.ones((2, 2))
    perturbed, seed = apply_knowledge_error(base, knowledge_cfg=None, base_seed=1, token="none")

    assert seed is None
    assert jnp.allclose(base, perturbed)


def test_plan_csv_resolution_relative(tmp_path):
    m = _load_module()
    prescription_path = tmp_path / "prescription.yaml"
    prescription_path.write_text("{}", encoding="utf-8")

    plan_rel = Path("plans/run_plan.csv")
    resolved = m._resolve_plan_csv_path(plan_rel, prescription_path=prescription_path)
    assert resolved == (prescription_path.parent / plan_rel).resolve()


def test_seed_detector_knowledge_errors_inserts_seed():
    m = _load_module()
    system_cfg = {
        "detector": {
            "layers": [
                {"name": "pixel_offsets", "knowledge_error": {"model": "gaussian", "scale": 1e-3}},
            ]
        }
    }
    seeded = m._seed_detector_knowledge_errors(system_cfg, base_seed=7, token_prefix="test")
    seeded_layers = seeded["detector"]["layers"]
    assert "seed" in seeded_layers[0]["knowledge_error"]


def test_get_pixel_offset_maps_handles_missing_layer():
    m = _load_module()
    class DummyDetector:
        def __init__(self):
            self.layers = {}

    class DummyBinder:
        def __init__(self):
            self.detector = DummyDetector()

    assert m._get_pixel_offset_maps(DummyBinder()) is None


def test_get_pixel_response_map_handles_missing_layer():
    m = _load_module()

    class DummyDetector:
        def __init__(self):
            self.layers = {}

    class DummyBinder:
        def __init__(self):
            self.detector = DummyDetector()

    assert m._get_pixel_response_map(DummyBinder()) is None


def test_refresh_preserving_derived_infer_keys_reapplies_derived_samples():
    m = _load_module()

    class DummyField:
        def __init__(self, kind: str):
            self.kind = kind

    class DummySpec:
        def __init__(self):
            self._fields = {
                "derived_key": DummyField("derived"),
                "primitive_key": DummyField("primitive"),
            }

        def __contains__(self, key):
            return key in self._fields

        def get(self, key):
            return self._fields[key]

    class DummyStore:
        def __init__(self, values, refreshed_values):
            self._values = dict(values)
            self._refreshed_values = dict(refreshed_values)

        def get(self, key):
            if key not in self._values:
                raise KeyError(key)
            return self._values[key]

        def refresh_derived(self, _spec):
            return DummyStore(self._refreshed_values, self._refreshed_values)

        def replace(self, updates):
            merged = dict(self._values)
            merged.update(updates)
            return DummyStore(merged, self._refreshed_values)

    spec = DummySpec()
    sampled = DummyStore(
        {"derived_key": 1.234, "primitive_key": 9.876},
        {"derived_key": 0.111, "primitive_key": 9.876},
    )

    refreshed = m._refresh_preserving_derived_infer_keys(
        sampled,
        infer_keys=("derived_key", "primitive_key"),
        spec=spec,
    )

    assert refreshed.get("derived_key") == 1.234
    assert refreshed.get("primitive_key") == 9.876


def test_trace_with_initial_point_prepends_theta_and_loss():
    m = _load_module()

    trace = {
        "theta": jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "loss": jnp.asarray([10.0, 5.0]),
    }
    theta0 = jnp.asarray([0.5, 1.5])
    loss0 = 12.0

    trace_with_init = m._trace_with_initial_point(trace, theta0=theta0, loss0=loss0)

    assert trace_with_init["theta"].shape == (3, 2)
    assert jnp.allclose(trace_with_init["theta"][0], theta0)
    assert trace_with_init["loss"].shape == (3,)
    assert float(trace_with_init["loss"][0]) == loss0


def test_trace_with_initial_point_does_not_duplicate_existing_iter0():
    m = _load_module()

    theta0 = jnp.asarray([0.5, 1.5])
    trace = {
        "theta": jnp.asarray([[0.5, 1.5], [1.0, 2.0]]),
        "loss": jnp.asarray([12.0, 10.0]),
    }

    trace_with_init = m._trace_with_initial_point(trace, theta0=theta0, loss0=12.0)

    assert trace_with_init["theta"].shape == (2, 2)
    assert jnp.allclose(trace_with_init["theta"][0], theta0)
    assert trace_with_init["loss"].shape == (2,)
