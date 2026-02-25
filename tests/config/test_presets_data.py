from __future__ import annotations

from pathlib import Path

from dluxshera.config import load_preset, resolve_config


PRESET_NAMES = [
    "SHERA_TESTBED_3P",
    "SHERA_FLIGHT_3P",
    "SHERA_TESTBED_2P",
    "SHERA_FLIGHT_2P",
]


def _presets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "src" / "dluxshera" / "data" / "presets"


def test_presets_load_via_loader():
    for name in PRESET_NAMES:
        loaded = load_preset(name, presets_dir=_presets_dir())
        assert isinstance(loaded, dict)
        assert "system" in loaded
        assert "experiment" in loaded


def test_presets_resolve_under_strict_schema_with_empty_override():
    for name in PRESET_NAMES:
        resolved = resolve_config(
            {"system": {"preset": name}, "experiment": {}},
            presets_dir=_presets_dir(),
        )
        assert resolved["system"]["preset"] == name
        assert isinstance(resolved["system"]["detector"]["layers"], list)


def test_optics_kind_matches_preset_suffix():
    for name in PRESET_NAMES:
        loaded = load_preset(name, presets_dir=_presets_dir())
        kind = loaded["system"]["optics"]["kind"]
        expected = "two_plane" if name.endswith("_2P") else "three_plane"
        assert kind == expected


def test_detector_layers_have_name_keys():
    for name in PRESET_NAMES:
        loaded = load_preset(name, presets_dir=_presets_dir())
        layers = loaded["system"]["detector"]["layers"]
        assert isinstance(layers, list)
        assert layers
        for layer in layers:
            assert isinstance(layer, dict)
            assert "name" in layer
