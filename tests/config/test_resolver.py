from __future__ import annotations

import json
import warnings

import pytest

from dluxshera.config import deep_merge, resolve_config


@pytest.fixture
def preset_dir(tmp_path):
    good_preset = {
        "system": {
            "preset": "TEST_PRESET",
            "source": {
                "kind": "alpha_cen",
                "wavelength_m": 5.5e-7,
                "bandwidth_m": 1.1e-7,
                "n_lambda": 3,
            },
            "optics": {
                "kind": "three_plane",
                "psf_npix": 128,
                "oversample": 2,
            },
            "detector": {
                "model": "gsense2020",
                "layers": [{"name": "downsample", "factor": 2}],
            },
        },
        "experiment": {"kind": "inference", "n_steps": 100},
    }
    missing_layers = {
        "system": {
            "preset": "MISSING_LAYERS",
            "source": good_preset["system"]["source"],
            "optics": good_preset["system"]["optics"],
            "detector": {"model": "gsense2020"},
        },
        "experiment": {"kind": "inference"},
    }
    missing_optics_kind = {
        "system": {
            "preset": "MISSING_OPTICS_KIND",
            "source": good_preset["system"]["source"],
            "optics": {"psf_npix": 128, "oversample": 2},
            "detector": good_preset["system"]["detector"],
        },
        "experiment": {"kind": "inference"},
    }

    (tmp_path / "TEST_PRESET.json").write_text(json.dumps(good_preset), encoding="utf-8")
    (tmp_path / "MISSING_LAYERS.json").write_text(json.dumps(missing_layers), encoding="utf-8")
    (tmp_path / "MISSING_OPTICS_KIND.json").write_text(json.dumps(missing_optics_kind), encoding="utf-8")
    return tmp_path


def test_deep_merge_replaces_lists():
    base = {"a": {"b": [1, 2], "c": 1}}
    overrides = {"a": {"b": [3], "d": 2}}
    merged = deep_merge(base, overrides)

    assert merged == {"a": {"b": [3], "c": 1, "d": 2}}


def test_resolve_config_loads_preset_and_user_override_wins(preset_dir):
    user_cfg = {
        "system": {
            "preset": "TEST_PRESET",
            "source": {"n_lambda": 5},
            "optics": {"psf_npix": 256, "oversample": "4"},
            "detector": {"layers": [{"name": "jitter", "sigma": 1e-3}]},
        },
        "experiment": {"kind": "inference"},
    }

    resolved = resolve_config(user_cfg, presets_dir=preset_dir)

    assert resolved["system"]["source"]["n_lambda"] == 5
    assert resolved["system"]["optics"]["psf_npix"] == 256
    assert resolved["system"]["optics"]["oversample"] == 4
    assert resolved["system"]["detector"]["layers"] == [{"name": "jitter", "sigma": 1e-3}]


def test_missing_required_key_errors(preset_dir):
    with pytest.raises(ValueError, match="system.detector.layers"):
        resolve_config(
            {"system": {"preset": "MISSING_LAYERS"}, "experiment": {"kind": "inference"}},
            presets_dir=preset_dir,
        )

    with pytest.raises(ValueError, match="system.optics.kind"):
        resolve_config(
            {"system": {"preset": "MISSING_OPTICS_KIND"}, "experiment": {"kind": "inference"}},
            presets_dir=preset_dir,
        )


def test_unknown_key_warning_includes_dotted_path(preset_dir):
    user_cfg = {
        "system": {
            "preset": "TEST_PRESET",
            "optics": {"foo": 10},
        },
        "experiment": {"kind": "inference", "bar": True},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_config(user_cfg, presets_dir=preset_dir)

    messages = [str(w.message) for w in caught]
    assert any("system.optics.foo" in msg for msg in messages)
    assert any("experiment.bar" in msg for msg in messages)


def test_type_normalization(preset_dir):
    user_cfg = {
        "system": {
            "preset": "TEST_PRESET",
            "source": {"wavelength_m": "5.6e-7", "bandwidth_m": 1.0e-7, "n_lambda": "7"},
            "optics": {"psf_npix": "300", "oversample": 5},
        },
        "experiment": {"kind": "inference"},
    }

    resolved = resolve_config(user_cfg, presets_dir=preset_dir)

    assert isinstance(resolved["system"]["source"]["wavelength_m"], float)
    assert isinstance(resolved["system"]["source"]["bandwidth_m"], float)
    assert isinstance(resolved["system"]["source"]["n_lambda"], int)
    assert isinstance(resolved["system"]["optics"]["psf_npix"], int)
    assert isinstance(resolved["system"]["optics"]["oversample"], int)


def test_resolved_config_to_system_config(preset_dir):
    from dluxshera.config import resolved_config_to_system_config
    from dluxshera.systems.three_plane import SheraThreePlaneConfig

    resolved = resolve_config(
        {"system": {"preset": "TEST_PRESET"}, "experiment": {"kind": "inference"}},
        presets_dir=preset_dir,
    )
    cfg = resolved_config_to_system_config(resolved)

    assert isinstance(cfg, SheraThreePlaneConfig)
    assert cfg.psf_npix == 128
    assert cfg.oversample == 2
