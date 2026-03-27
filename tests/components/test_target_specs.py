import pytest

from dluxshera.components.sources import get_target_spec


def test_get_target_spec_known_target():
    spec = get_target_spec("alpha_cen")
    assert spec.key == "ALPHA_CEN"
    assert spec.display_name == "Alpha Centauri"


def test_get_target_spec_unknown_target_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown target"):
        get_target_spec("NOT_A_TARGET")
