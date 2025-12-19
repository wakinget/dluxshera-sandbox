import importlib


def test_package_imports_smoke():
    pkg = importlib.import_module("dluxshera")
    plotting = importlib.import_module("dluxshera.plot.plotting")

    assert pkg is not None
    assert hasattr(plotting, "choose_subplot_grid")
    assert plotting.choose_subplot_grid(3) == (3, 1)
