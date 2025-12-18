"""Lightweight proxy for accessing prefixed store keys via attributes."""

from __future__ import annotations

from typing import Any


class StoreNamespace:
    """Expose dotted store keys under a shared prefix as attributes."""

    def __init__(self, store: Any, prefix: str) -> None:
        self._store = store
        self._prefix = prefix

    def __getattr__(self, leaf: str) -> Any:
        key = f"{self._prefix}.{leaf}"
        try:
            return self._store.get(key)
        except KeyError:
            raise AttributeError(leaf) from None

    def get(self, leaf: str, default: Any = None) -> Any:
        """Return the value for ``prefix.leaf`` if present, else ``default``."""

        key = f"{self._prefix}.{leaf}"
        return self._store.get(key, default)
