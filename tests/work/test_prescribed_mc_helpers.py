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


def test_monte_carlo_reuse_fim_populates_fim_defaults():
    m = _load_module()
    experiment_cfg = {"seed": 42}
    mc_cfg = {"reuse_fim": True}

    _, defaults = m._mc_defaults_from_experiment(experiment_cfg, mc_cfg)

    assert defaults["fim"]["reuse_fim"] is True


def test_detector_ke_policy_default_matches_fixed_per_experiment():
    m = _load_module()
    system_cfg = {
        "detector": {
            "layers": [
                {
                    "name": "pixel_offsets",
                    "knowledge_error": {"model": "gaussian", "scale": 1e-3},
                }
            ]
        }
    }

    seeded1, meta1 = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=123,
        run_seed=1,
        token_prefix="inference.detector",
    )
    seeded2, meta2 = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=123,
        run_seed=2,
        token_prefix="inference.detector",
    )

    seed1 = seeded1["detector"]["layers"][0]["knowledge_error"]["seed"]
    seed2 = seeded2["detector"]["layers"][0]["knowledge_error"]["seed"]
    assert seed1 == seed2
    assert meta1["layers"]["pixel_offsets"]["realization_policy"] == "fixed_per_experiment"
    assert meta2["layers"]["pixel_offsets"]["realization_policy"] == "fixed_per_experiment"


def test_detector_ke_policy_fixed_per_experiment_is_run_invariant():
    m = _load_module()
    system_cfg = {
        "detector": {
            "layers": [
                {
                    "name": "pixel_offsets",
                    "knowledge_error": {
                        "model": "gaussian",
                        "scale": 1e-3,
                        "realization_policy": "fixed_per_experiment",
                    },
                }
            ]
        }
    }

    seeded1, _ = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=55,
        run_seed=1001,
        token_prefix="inference.detector",
    )
    seeded2, _ = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=55,
        run_seed=2002,
        token_prefix="inference.detector",
    )

    seed1 = seeded1["detector"]["layers"][0]["knowledge_error"]["seed"]
    seed2 = seeded2["detector"]["layers"][0]["knowledge_error"]["seed"]
    assert seed1 == seed2


def test_detector_ke_policy_per_run_uses_run_seed_and_is_reproducible():
    m = _load_module()
    system_cfg = {
        "detector": {
            "layers": [
                {
                    "name": "pixel_offsets",
                    "knowledge_error": {
                        "model": "gaussian",
                        "scale": 1e-3,
                        "realization_policy": "per_run",
                    },
                }
            ]
        }
    }

    seeded1, _ = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=77,
        run_seed=1001,
        token_prefix="inference.detector",
    )
    seeded2, _ = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=77,
        run_seed=2002,
        token_prefix="inference.detector",
    )
    seeded3, _ = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=77,
        run_seed=1001,
        token_prefix="inference.detector",
    )

    seed1 = seeded1["detector"]["layers"][0]["knowledge_error"]["seed"]
    seed2 = seeded2["detector"]["layers"][0]["knowledge_error"]["seed"]
    seed3 = seeded3["detector"]["layers"][0]["knowledge_error"]["seed"]

    assert seed1 != seed2
    assert seed1 == seed3


def test_detector_ke_per_run_policy_inspection_keeps_base_config_pristine():
    m = _load_module()
    system_cfg = {
        "detector": {
            "layers": [
                {
                    "name": "pixel_offsets",
                    "knowledge_error": {
                        "model": "gaussian",
                        "scale": 1e-3,
                        "realization_policy": "per_run",
                    },
                }
            ]
        }
    }

    assert m._detector_ke_has_per_run_realization(system_cfg) is True
    assert "seed" not in system_cfg["detector"]["layers"][0]["knowledge_error"]

    seeded1, _ = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=77,
        run_seed=1001,
        token_prefix="inference.detector",
    )
    seeded2, _ = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=77,
        run_seed=2002,
        token_prefix="inference.detector",
    )

    seed1 = seeded1["detector"]["layers"][0]["knowledge_error"]["seed"]
    seed2 = seeded2["detector"]["layers"][0]["knowledge_error"]["seed"]
    assert seed1 != seed2
    assert "seed" not in system_cfg["detector"]["layers"][0]["knowledge_error"]


def test_detector_ke_policy_explicit_seed_wins_over_policy():
    m = _load_module()
    system_cfg = {
        "detector": {
            "layers": [
                {
                    "name": "pixel_offsets",
                    "knowledge_error": {
                        "model": "gaussian",
                        "scale": 1e-3,
                        "realization_policy": "per_run",
                        "seed": 999,
                    },
                }
            ]
        }
    }

    seeded, meta = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=77,
        run_seed=1001,
        token_prefix="inference.detector",
    )

    assert seeded["detector"]["layers"][0]["knowledge_error"]["seed"] == 999
    assert meta["layers"]["pixel_offsets"]["seed"] == 999
    assert meta["layers"]["pixel_offsets"]["seed_source"] == "explicit"


def test_detector_ke_metadata_includes_model_scale_policy_and_seed():
    m = _load_module()
    system_cfg = {
        "detector": {
            "layers": [
                {
                    "name": "pixel_offsets",
                    "knowledge_error": {
                        "model": "gaussian",
                        "scale": 2e-3,
                        "realization_policy": "per_run",
                    },
                }
            ]
        }
    }

    _, meta = m._seed_detector_knowledge_errors_with_policy(
        system_cfg,
        experiment_seed=42,
        run_seed=4242,
        token_prefix="inference.detector",
    )
    layer_meta = meta["layers"]["pixel_offsets"]

    assert layer_meta["model"] == "gaussian"
    assert layer_meta["scale"] == 2e-3
    assert layer_meta["realization_policy"] == "per_run"
    assert isinstance(layer_meta["seed"], int)
    assert layer_meta["seed_source"] == "run_seed"
    assert meta["has_per_run_realization"] is True


def test_fim_cache_key_payload_hash_changes_with_cfg_hash():
    m = _load_module()
    payload_a = m._build_fim_cache_key_payload(
        infer_keys=("source.separation_as",),
        system_label="SYS",
        cfg_hash="cfg_A",
        forward_spec_hash="spec_same",
        theta_true_hash="theta_same",
        loss_kind="nll",
    )
    payload_b = m._build_fim_cache_key_payload(
        infer_keys=("source.separation_as",),
        system_label="SYS",
        cfg_hash="cfg_B",
        forward_spec_hash="spec_same",
        theta_true_hash="theta_same",
        loss_kind="nll",
    )

    hash_a = m._stable_hash_payload(payload_a)
    hash_b = m._stable_hash_payload(payload_b)
    assert payload_a["cfg_hash"] == "cfg_A"
    assert payload_b["cfg_hash"] == "cfg_B"
    assert hash_a != hash_b


def test_detector_ke_policy_invalid_value_raises():
    m = _load_module()
    system_cfg = {
        "detector": {
            "layers": [
                {
                    "name": "pixel_offsets",
                    "knowledge_error": {
                        "model": "gaussian",
                        "scale": 1e-3,
                        "realization_policy": "not_a_policy",
                    },
                }
            ]
        }
    }

    try:
        m._seed_detector_knowledge_errors_with_policy(
            system_cfg,
            experiment_seed=1,
            run_seed=2,
            token_prefix="inference.detector",
        )
    except ValueError as exc:
        assert "realization_policy" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid realization_policy.")


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
