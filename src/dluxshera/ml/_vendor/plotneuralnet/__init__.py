"""Vendored PlotNeuralNet TikZ layer resources.

Only the LaTeX/TikZ layer definitions needed by
``dluxshera.ml.visualization`` are vendored here.  dLuxShera owns the Python
rendering wrapper and does not expose upstream PlotNeuralNet internals.
"""

from __future__ import annotations

from importlib import resources

__all__ = ["layers_path"]


def layers_path() -> str:
    """Return the installed PlotNeuralNet layer resource path for LaTeX."""
    return str(resources.files(__package__).joinpath("layers"))
