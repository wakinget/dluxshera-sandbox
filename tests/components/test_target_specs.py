from importlib import resources

import pytest

from dluxshera.components.sources import TARGET_SPECS, get_target_spec


def test_get_target_spec_known_target():
    spec = get_target_spec("alpha_cen")
    assert spec.key == "ALPHA_CEN"
    assert spec.display_name == "Alpha Centauri"


def test_get_target_spec_supports_61_cyg():
    spec = get_target_spec("61_cyg")
    assert spec.key == "61_CYG"
    assert spec.display_name == "61 Cygni"


def test_target_spec_sed_files_exist_in_package_data():
    sed_root = resources.files("dluxshera").joinpath("data", "target_seds")
    for spec in TARGET_SPECS.values():
        for sed_file in (spec.sed_a_file, spec.sed_b_file):
            if sed_file is None:
                continue
            assert sed_root.joinpath(sed_file).is_file(), (
                f"Missing curated SED file {sed_file!r} for target {spec.key}."
            )


def test_get_target_spec_unknown_target_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown target"):
        get_target_spec("NOT_A_TARGET")
