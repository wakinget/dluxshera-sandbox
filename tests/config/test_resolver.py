from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import pytest

from dluxshera.config import (
    as_dict,
    deep_merge,
    resolve_config,
    resolve_experiment_config,
    resolve_system_config,
)


@pytest.fixture
def preset_dirs(tmp_path):
    system_dir = tmp_path / "system_presets"
    experiment_dir = tmp_path / "experiment_presets"
    system_dir.mkdir()
    experiment_dir.mkdir()

    good_system = {
        "system": {
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
                "layers": [{"name": "downsample", "kind": "Downsample", "factor": 2}],
            },
        }
    }
    missing_layers = {
        "system": {
            "source": good_system["system"]["source"],
            "optics": good_system["system"]["optics"],
            "detector": {"model": "gsense2020"},
        }
    }
    missing_optics_kind = {
        "system": {
            "source": good_system["system"]["source"],
            "optics": {"psf_npix": 128, "oversample": 2},
            "detector": good_system["system"]["detector"],
        }
    }
    experiment = {
        "experiment": {
            "kind": "inference",
            "seed": 1,
        }
    }

    (system_dir / "TEST_PRESET.json").write_text(json.dumps(good_system), encoding="utf-8")
    (system_dir / "MISSING_LAYERS.json").write_text(json.dumps(missing_layers), encoding="utf-8")
    (system_dir / "MISSING_OPTICS_KIND.json").write_text(json.dumps(missing_optics_kind), encoding="utf-8")
    (experiment_dir / "INFERENCE_CANONICAL.json").write_text(json.dumps(experiment), encoding="utf-8")
    return system_dir, experiment_dir


def test_deep_merge_replaces_lists():
    base = {"a": {"b": [1, 2], "c": 1}}
    overrides = {"a": {"b": [3], "d": 2}}
    merged = deep_merge(base, overrides)

    assert merged == {"a": {"b": [3], "c": 1, "d": 2}}


@dataclass
class _CfgDC:
    system: dict | None = None
    experiment: dict | None = None


class _CfgObj:
    def __init__(self, *, system=None, experiment=None):
        if system is not None:
            self.system = system
        if experiment is not None:
            self.experiment = experiment


def test_as_dict_accepts_system_or_experiment_only():
    only_system = as_dict({"system": {"preset": "TEST_PRESET"}})
    only_experiment = as_dict(_CfgDC(experiment={"preset": "INFERENCE_CANONICAL"}))
    both_from_obj = as_dict(_CfgObj(system={"preset": "TEST_PRESET"}, experiment={"preset": "X"}))

    assert set(only_system) == {"system"}
    assert "experiment" in only_experiment
    assert set(both_from_obj) == {"system", "experiment"}


def test_as_dict_rejects_missing_both_blocks():
    with pytest.raises(TypeError, match="at least one of 'system' or 'experiment'"):
        as_dict({"foo": 1})


def test_resolve_system_config_loads_preset_and_user_override_wins(preset_dirs):
    system_dir, _ = preset_dirs

    resolved = resolve_system_config(
        {
            "preset": "TEST_PRESET",
            "source": {"n_lambda": 5},
            "optics": {"psf_npix": 256, "oversample": "4"},
            "detector": {"layers": [{"name": "jitter", "kind": "ApplyJitter", "sigma": 1e-3}]},
        },
        presets_dir=system_dir,
    )

    assert resolved["source"]["n_lambda"] == 5
    assert resolved["optics"]["psf_npix"] == 256
    assert resolved["optics"]["oversample"] == 4
    assert resolved["detector"]["layers"] == [
        {"name": "jitter", "kind": "ApplyJitter", "sigma": 1e-3}
    ]


def test_resolve_experiment_config_loads_preset(preset_dirs):
    _, experiment_dir = preset_dirs
    resolved = resolve_experiment_config(
        {"preset": "INFERENCE_CANONICAL"},
        presets_dir=experiment_dir,
    )

    assert resolved["kind"] == "inference"


def test_resolve_config_is_permissive_top_level(preset_dirs):
    system_dir, experiment_dir = preset_dirs

    only_system = resolve_config(
        {"system": {"preset": "TEST_PRESET"}},
        system_presets_dir=system_dir,
        experiment_presets_dir=experiment_dir,
    )
    only_experiment = resolve_config(
        {"experiment": {"preset": "INFERENCE_CANONICAL"}},
        system_presets_dir=system_dir,
        experiment_presets_dir=experiment_dir,
    )
    both = resolve_config(
        {
            "system": {"preset": "TEST_PRESET"},
            "experiment": {"preset": "INFERENCE_CANONICAL"},
        },
        system_presets_dir=system_dir,
        experiment_presets_dir=experiment_dir,
    )

    assert set(only_system.keys()) == {"system"}
    assert set(only_experiment.keys()) == {"experiment"}
    assert set(both.keys()) == {"system", "experiment"}


def test_missing_required_key_errors(preset_dirs):
    system_dir, experiment_dir = preset_dirs
    with pytest.raises(ValueError, match="system.detector.layers"):
        resolve_system_config(
            {"preset": "MISSING_LAYERS"},
            presets_dir=system_dir,
        )

    with pytest.raises(ValueError, match="system.optics.kind"):
        resolve_system_config(
            {"preset": "MISSING_OPTICS_KIND"},
            presets_dir=system_dir,
        )

    with pytest.raises(ValueError, match="experiment.kind"):
        resolve_experiment_config(
            {"preset": "INFERENCE_CANONICAL", "kind": ""},
            presets_dir=experiment_dir,
        )


def test_unknown_key_warning_includes_dotted_path(preset_dirs):
    system_dir, experiment_dir = preset_dirs

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_config(
            {
                "system": {
                    "preset": "TEST_PRESET",
                    "optics": {"foo": 10},
                },
                "experiment": {
                    "preset": "INFERENCE_CANONICAL",
                    "bar": True,
                },
            },
            system_presets_dir=system_dir,
            experiment_presets_dir=experiment_dir,
        )

    messages = [str(w.message) for w in caught]
    assert any("system.optics.foo" in msg for msg in messages)
    assert any("experiment.bar" in msg for msg in messages)


def test_type_normalization(preset_dirs):
    system_dir, _ = preset_dirs

    resolved = resolve_system_config(
        {
            "preset": "TEST_PRESET",
            "source": {"wavelength_m": "5.6e-7", "bandwidth_m": 1.0e-7, "n_lambda": "7"},
            "optics": {"psf_npix": "300", "oversample": 5},
        },
        presets_dir=system_dir,
    )

    assert isinstance(resolved["source"]["wavelength_m"], float)
    assert isinstance(resolved["source"]["bandwidth_m"], float)
    assert isinstance(resolved["source"]["n_lambda"], int)
    assert isinstance(resolved["optics"]["psf_npix"], int)
    assert isinstance(resolved["optics"]["oversample"], int)


def test_resolved_config_to_system_config(preset_dirs):
    from dluxshera.config import resolved_config_to_system_config
    from dluxshera.systems.three_plane import SheraThreePlaneConfig

    system_dir, experiment_dir = preset_dirs
    resolved = resolve_config(
        {
            "system": {"preset": "TEST_PRESET"},
            "experiment": {"preset": "INFERENCE_CANONICAL"},
        },
        system_presets_dir=system_dir,
        experiment_presets_dir=experiment_dir,
    )
    with pytest.deprecated_call(match="deprecated"):
        cfg = resolved_config_to_system_config(resolved)

    assert isinstance(cfg, SheraThreePlaneConfig)
    assert cfg.psf_npix == 128
    assert cfg.oversample == 2


def test_resolve_system_config_rejects_detector_layer_missing_kind(preset_dirs):
    system_dir, _ = preset_dirs

    with pytest.raises(ValueError, match=r"system\.detector\.layers\[0\]\.kind"):
        resolve_system_config(
            {
                "preset": "TEST_PRESET",
                "detector": {"layers": [{"name": "downsample", "kernel_size": 2}]},
            },
            presets_dir=system_dir,
        )


def test_resolve_system_config_rejects_duplicate_detector_layer_names(preset_dirs):
    system_dir, _ = preset_dirs

    with pytest.raises(ValueError, match="Duplicate detector layer name 'shared'"):
        resolve_system_config(
            {
                "preset": "TEST_PRESET",
                "detector": {
                    "layers": [
                        {"name": "shared", "kind": "Downsample", "kernel_size": 2},
                        {"name": "shared", "kind": "ApplyJitter", "sigma": 1e-3},
                    ]
                },
            },
            presets_dir=system_dir,
        )
