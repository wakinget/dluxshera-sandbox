from __future__ import annotations

import dataclasses
import importlib
import json
import pathlib
import sys

# Ensure editable-src imports when running this diagnostic directly.
ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dluxshera.core.binder import SheraThreePlaneBinder
from dluxshera.optics.config import SHERA_TESTBED_CONFIG
from tests.conftest import make_forward_store


def _check_instance_of(module_name: str, class_name: str, instance) -> bool:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return False

    cls = getattr(module, class_name, None)
    return cls is not None and isinstance(instance, cls)


def test_binder_introspection_snapshot(capsys):
    cfg = SHERA_TESTBED_CONFIG
    forward_spec, forward_store = make_forward_store(cfg)
    binder = SheraThreePlaneBinder(cfg, forward_spec, forward_store)
    base_store = binder.base_forward_store

    dataclass_params = getattr(type(binder), "__dataclass_params__", None)
    dataclass_slots = getattr(dataclass_params, "slots", None)
    if dataclass_slots is None:
        # __dataclass_params__ is undocumented and may not track newer flags (slots, etc.)
        dataclass_slots = hasattr(type(binder), "__slots__")

    diagnostics = {
        "is_dataclass": dataclasses.is_dataclass(binder),
        "dataclass_frozen": getattr(dataclass_params, "frozen", None),
        "dataclass_slots": bool(dataclass_slots),
        "has_slots_attr": hasattr(type(binder), "__slots__"),
        "is_eqx_module": _check_instance_of("equinox", "Module", binder),
        "is_zdx_base": _check_instance_of("zodiax", "Base", binder),
        "base_store_type": type(base_store).__name__,
        "base_store_methods": [
            name
            for name in (
                "get",
                "keys",
                "items",
                "values",
                "as_dict",
                "replace",
                "validate_against",
            )
            if hasattr(base_store, name)
        ],
        "derived_key_presence": {
            "system.plate_scale_as_per_pix": "system.plate_scale_as_per_pix" in base_store,
        },
        "base_store_len": len(base_store),
    }

    print(json.dumps(diagnostics, indent=2, sort_keys=True))
    captured = capsys.readouterr()

    # The prints are for quick diagnostics; the asserts guard against regressions
    # in Binder mutability/typing expectations.
    assert diagnostics["is_dataclass"] is True
    assert diagnostics["dataclass_frozen"] is False
    assert diagnostics["dataclass_slots"] is False
    assert diagnostics["has_slots_attr"] is False
    assert diagnostics["is_eqx_module"] is False
    assert diagnostics["is_zdx_base"] is False
    assert diagnostics["base_store_type"] == "ParameterStore"
    assert set(diagnostics["base_store_methods"]) >= {"get", "keys", "items", "as_dict"}
    assert diagnostics["derived_key_presence"]["system.plate_scale_as_per_pix"] is True
    assert captured.out.strip(), "Expected diagnostics to be printed"
