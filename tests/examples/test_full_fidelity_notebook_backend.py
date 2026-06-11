from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "examples/notebooks/full_fidelity_resolved_system_review.ipynb"
HELPER_PATHS = (
    REPO_ROOT / "src/dluxshera/utils/full_fidelity_review.py",
    REPO_ROOT / "src/dluxshera/plot/plotting.py",
    REPO_ROOT / "src/dluxshera/plot/obs_subblock.py",
)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def test_full_fidelity_notebook_is_valid_nbformat() -> None:
    nbformat = pytest.importorskip("nbformat")

    with NOTEBOOK_PATH.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    nbformat.validate(notebook)


def test_full_fidelity_notebook_configures_backend_before_pyplot() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    sources = [_cell_source(cell) for cell in code_cells]

    setup_index = next(i for i, source in enumerate(sources) if "PLOT_BACKEND = \"auto\"" in source)
    pyplot_index = next(i for i, source in enumerate(sources) if "import matplotlib.pyplot as plt" in source)
    diagnostics_index = next(i for i, source in enumerate(sources) if "Matplotlib backend:" in source)

    assert setup_index < pyplot_index
    assert pyplot_index < diagnostics_index
    assert "ACTIVE_MATPLOTLIB_BACKEND" in sources[setup_index]
    assert "plt.ion()" in sources[pyplot_index]
    assert "plt.isinteractive()" in sources[diagnostics_index]


def test_full_fidelity_notebook_plot_cells_show_figures() -> None:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    plot_sources = [
        _cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "plt.subplots" in _cell_source(cell)
    ]

    assert plot_sources
    assert all("plt.show()" in source for source in plot_sources)


def test_review_helpers_do_not_force_agg_on_import() -> None:
    forbidden = (
        'matplotlib.use("Agg", force=True)',
        "matplotlib.use('Agg', force=True)",
        'matplotlib.use("Agg")',
        "matplotlib.use('Agg')",
        "plt.ioff()",
    )

    for path in HELPER_PATHS:
        source = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern not in source, f"{path} contains import-time backend forcing: {pattern}"

