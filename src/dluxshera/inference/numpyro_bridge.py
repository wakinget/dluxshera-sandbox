"""Stubs for converting backend-agnostic prior specs into NumPyro distributions.

This is intentionally thin and not wired into the main inference flow yet. It is
meant to document how :class:`PriorSpec` could be adapted for NumPyro without
forcing a dependency at import time.
"""
from __future__ import annotations

from typing import Dict

from .prior import PriorSpec
from ..params.spec import ParamKey, ParamSpec


def numpyro_priors_from_spec(prior_spec: PriorSpec, inference_spec: ParamSpec) -> Dict[ParamKey, object]:
    """Convert a :class:`PriorSpec` into a NumPyro-compatible prior mapping.

    Notes
    -----
    - NumPyro is intentionally not imported here to avoid hard dependencies in
      the base inference stack.
    - This function is a placeholder for future integration work; callers should
      expect ``NotImplementedError`` until the NumPyro bridge is fleshed out.
    """

    raise NotImplementedError(
        "NumPyro adapter is not yet implemented. This stub documents the intended "
        "conversion point from PriorSpec to backend distributions."
    )
