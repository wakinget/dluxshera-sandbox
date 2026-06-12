from __future__ import annotations

import pytest

from dluxshera.config.resolver import resolve_config
from dluxshera.utils.detector_layer_overrides import (
    apply_detector_layer_overrides,
    detector_layer_stack,
    get_detector_layer,
)


def _conv_system() -> dict:
    return dict(resolve_config({"system": {"preset": "SHERA_FLIGHT_3P_CONV"}})["system"])


def test_conv_preset_contains_expected_named_layers() -> None:
    names = [row["name"] for row in detector_layer_stack(_conv_system())]
    assert names == ["pixel_mtf", "diffusion", "pixel_offsets", "pixel_response", "jitter", "smear"]


def test_detector_override_updates_jitter_by_name() -> None:
    system, provenance = apply_detector_layer_overrides(
        _conv_system(),
        {"layers": {"jitter": {"action": "update", "kernel": {"sigma_x": 0.001, "sigma_y": 0.001}}}},
        context="test",
    )
    jitter = get_detector_layer(system, "jitter")
    assert jitter is not None
    assert jitter["kernel"]["sigma_x"] == 0.001
    assert jitter["kernel"]["sigma_y"] == 0.001
    assert provenance["applied"][0]["status"] == "updated"


def test_detector_override_removes_jitter_and_smear_by_name() -> None:
    system, _ = apply_detector_layer_overrides(
        _conv_system(),
        {"layers": {"jitter": {"action": "remove"}, "smear": {"action": "remove"}}},
    )
    names = [row["name"] for row in detector_layer_stack(system)]
    assert "jitter" not in names
    assert "smear" not in names


def test_missing_layer_override_fails_clearly() -> None:
    with pytest.raises(ValueError, match="missing layer 'not_a_layer'"):
        apply_detector_layer_overrides(
            _conv_system(),
            {"layers": {"not_a_layer": {"action": "remove"}}},
            context="test_context",
        )

