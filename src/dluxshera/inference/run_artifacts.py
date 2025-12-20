"""
Run artifact I/O utilities.

This module implements the Phase A scaffold for saving and loading per-run
artifacts as described in the optimization artifacts doc. It intentionally
keeps a small surface: trace/meta/summary are always written, while
additional diagnostics remain opt-in.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional

import numpy as np

from dluxshera.params.spec import ParamSpec
from dluxshera.params.store import ParameterStore


ArrayMapping = Mapping[str, object]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_arrays(mapping: ArrayMapping) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in mapping.items()}


def _save_npz(path: Path, data: ArrayMapping) -> None:
    arrays = _normalize_arrays(data)
    np.savez_compressed(path, **arrays)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {k: np.asarray(v) for k, v in npz.items()}


def save_run(
    run_dir: Path | str,
    trace: ArrayMapping,
    meta: Mapping,
    summary: Mapping,
    *,
    signals: Optional[ArrayMapping] = None,
    grads: Optional[ArrayMapping] = None,
    curvature: Optional[ArrayMapping] = None,
    precond: Optional[ArrayMapping] = None,
    checkpoints: Optional[Mapping[str, ArrayMapping]] = None,
    diag_steps: Optional[Iterable[Mapping]] = None,
) -> None:
    """
    Save optimizer run artifacts to ``run_dir``.

    Always writes ``trace.npz``, ``meta.json``, and ``summary.json``. Optional
    artifacts are only written when their corresponding argument is provided.
    """

    run_path = Path(run_dir)
    _ensure_dir(run_path)

    _save_npz(run_path / "trace.npz", trace)

    with (run_path / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with (run_path / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if signals is not None:
        _save_npz(run_path / "signals.npz", signals)

    if grads is not None:
        _save_npz(run_path / "grads.npz", grads)

    if curvature is not None:
        _save_npz(run_path / "curvature.npz", curvature)

    if precond is not None:
        _save_npz(run_path / "precond.npz", precond)

    if checkpoints is not None:
        for name, payload in checkpoints.items():
            checkpoint_path = run_path / f"checkpoint_{name}.npz"
            _save_npz(checkpoint_path, payload)

    if diag_steps is not None:
        diag_path = run_path / "diag_steps.jsonl"
        with diag_path.open("w", encoding="utf-8") as f:
            for record in diag_steps:
                json.dump(record, f)
                f.write("\n")


def load_trace(run_dir: Path | str) -> dict[str, np.ndarray]:
    return _load_npz(Path(run_dir) / "trace.npz")


def load_meta(run_dir: Path | str):
    with (Path(run_dir) / "meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(run_dir: Path | str):
    with (Path(run_dir) / "summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(run_dir: Path | str, which: str = "best") -> dict[str, np.ndarray]:
    path = Path(run_dir) / f"checkpoint_{which}.npz"
    return _load_npz(path)


def _compute_layout_hash(entries: list[dict[str, object]]) -> str:
    payload = [(entry["name"], entry["shape"]) for entry in entries]
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_index_map(
    spec_subset: ParamSpec,
    store: ParameterStore,
    *,
    theta=None,
    block_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """Build a serializable IndexMap aligned with parameter packing order."""

    entries: list[dict[str, object]] = []
    offset = 0

    for key in spec_subset.keys():
        value = store.get(key)
        if value is None:
            raise ValueError(
                f"IndexMap requires concrete values; got None for key {key!r}."
            )
        arr = np.asarray(value)
        size = int(arr.size)
        shape = list(arr.shape)
        start = offset
        stop = offset + size
        block = block_fn(key) if block_fn is not None else key

        entries.append(
            {
                "name": key,
                "start": start,
                "stop": stop,
                "shape": shape,
                "block": block,
            }
        )

        offset = stop

    if theta is not None:
        theta_size = int(np.asarray(theta).size)
        if theta_size != offset:
            raise ValueError(
                "IndexMap size mismatch: packed size from spec/store does not "
                f"match theta.size ({offset} vs {theta_size})."
            )

    index_map = {
        "entries": entries,
        "layout_hash": _compute_layout_hash(entries),
    }

    return index_map
