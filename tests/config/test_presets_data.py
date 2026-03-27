from __future__ import annotations

from pathlib import Path

from dluxshera.config import (
    load_experiment_preset,
    load_system_preset,
    resolve_config,
    resolve_experiment_config,
    resolve_system_config,
)


SYSTEM_PRESET_NAMES = [
    "SHERA_TESTBED_3P",
    "SHERA_FLIGHT_3P",
    "SHERA_TESTBED_2P",
    "SHERA_FLIGHT_2P",
]
EXPERIMENT_PRESET_NAMES = ["INFERENCE_CANONICAL", "CANONICAL_ASTROMETRY"]


def _system_presets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "dluxshera" / "data" / "system_presets"


def _experiment_presets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "dluxshera" / "data" / "experiment_presets"


def test_system_presets_load_via_loader():
    for name in SYSTEM_PRESET_NAMES:
        loaded = load_system_preset(name, presets_dir=_system_presets_dir())
        assert isinstance(loaded, dict)
        assert "system" in loaded
        assert "experiment" not in loaded


def test_experiment_presets_load_via_loader():
    for name in EXPERIMENT_PRESET_NAMES:
        loaded = load_experiment_preset(name, presets_dir=_experiment_presets_dir())
        assert isinstance(loaded, dict)
        assert "experiment" in loaded
        assert "system" not in loaded


def test_system_and_experiment_resolve_under_strict_schema_with_empty_override():
    for name in SYSTEM_PRESET_NAMES:
        resolved_system = resolve_system_config(
            {"preset": name},
            presets_dir=_system_presets_dir(),
        )
        resolved_experiment = resolve_experiment_config(
            {"preset": "INFERENCE_CANONICAL"},
            presets_dir=_experiment_presets_dir(),
        )
        resolved = resolve_config(
            {
                "system": {"preset": name},
                "experiment": {"preset": "INFERENCE_CANONICAL"},
            },
            system_presets_dir=_system_presets_dir(),
            experiment_presets_dir=_experiment_presets_dir(),
        )
        assert resolved_system["preset"] == name
        assert isinstance(resolved_system["detector"]["layers"], list)
        assert resolved_experiment["kind"] == "inference"
        assert resolved["system"]["preset"] == name


def test_optics_kind_matches_preset_suffix():
    for name in SYSTEM_PRESET_NAMES:
        loaded = load_system_preset(name, presets_dir=_system_presets_dir())
        kind = loaded["system"]["optics"]["kind"]
        expected = "two_plane" if name.endswith("_2P") else "three_plane"
        assert kind == expected


def test_detector_layers_have_name_and_kind_keys():
    for name in SYSTEM_PRESET_NAMES:
        loaded = load_system_preset(name, presets_dir=_system_presets_dir())
        layers = loaded["system"]["detector"]["layers"]
        assert isinstance(layers, list)
        assert layers
        for layer in layers:
            assert isinstance(layer, dict)
            assert "name" in layer
            assert "kind" in layer


def test_preset_files_are_block_scoped():
    for name in SYSTEM_PRESET_NAMES:
        loaded = load_system_preset(name, presets_dir=_system_presets_dir())
        assert "experiment" not in loaded

    for name in EXPERIMENT_PRESET_NAMES:
        loaded = load_experiment_preset(name, presets_dir=_experiment_presets_dir())
        assert "system" not in loaded
