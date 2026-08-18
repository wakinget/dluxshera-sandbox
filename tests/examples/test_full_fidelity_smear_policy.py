from __future__ import annotations

import pytest

from dluxshera.config.resolver import resolve_config
from dluxshera.utils.detector_layer_overrides import (
    detector_layer_stack,
    get_detector_layer,
    patch_smear_layer_for_policy,
    validate_no_accidental_default_smear,
)


def _conv_system() -> dict:
    return dict(resolve_config({"system": {"preset": "SHERA_FLIGHT_3P_CONV"}})["system"])


def test_smear_disabled_removes_named_smear_layer() -> None:
    patched, provenance = patch_smear_layer_for_policy(
        _conv_system(),
        {"enabled": False, "render": {"mode": "disabled", "target_layer": "smear"}},
    )
    assert "smear" not in [row["name"] for row in detector_layer_stack(patched)]
    assert provenance["mode"] == "disabled"


def test_metadata_only_removes_active_default_smear_layer() -> None:
    patched, _ = patch_smear_layer_for_policy(
        _conv_system(),
        {"enabled": True, "render": {"mode": "metadata_only", "target_layer": "smear"}},
    )
    assert get_detector_layer(patched, "smear") is None


def test_subblock_constant_patches_existing_smear_layer() -> None:
    patched, provenance = patch_smear_layer_for_policy(
        _conv_system(),
        {
            "enabled": True,
            "render": {
                "mode": "subblock_constant_layer",
                "target_layer": "smear",
                "require_existing_layer": True,
                "allow_layer_injection": False,
            },
        },
        representative_kernel={
            "kind": "line",
            "length": 1.25,
            "theta_deg": 17.0,
            "sigma_perp": 0.1,
            "kernel_size": 11,
            "units": "detector_pix",
        },
    )
    smear = get_detector_layer(patched, "smear")
    assert smear is not None
    assert smear["kernel"]["length"] == 1.25
    assert smear["kernel"]["theta_deg"] == 17.0
    assert provenance["representative_kernel"]["kernel_size"] == 11


def test_subblock_constant_requires_existing_smear_layer() -> None:
    system, _ = patch_smear_layer_for_policy(_conv_system(), {"enabled": False, "render": {"mode": "disabled"}})
    with pytest.raises(ValueError, match="requires existing detector layer 'smear'"):
        patch_smear_layer_for_policy(
            system,
            {
                "enabled": True,
                "render": {
                    "mode": "subblock_constant_layer",
                    "target_layer": "smear",
                    "require_existing_layer": True,
                    "allow_layer_injection": False,
                },
            },
        )


def test_strict_validation_catches_accidental_default_smear() -> None:
    with pytest.raises(ValueError, match="default nonzero smear layer"):
        validate_no_accidental_default_smear(
            _conv_system(),
            system_preset="SHERA_FLIGHT_3P_CONV",
            smear_cfg={"enabled": True, "render": {"mode": "metadata_only", "target_layer": "smear"}},
            strict=True,
        )
