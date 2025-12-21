"""Sweep utilities for aggregating run artifacts."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .run_artifacts import load_meta, load_summary


DEFAULT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "created_at",
    "git.commit",
    "status",
    "theta.dim",
    "theta.theta_space",
    "optimizer.name",
    "optimizer.num_steps",
    "optimizer.base_lr",
    "preconditioning.enabled",
    "preconditioning.method",
    "loss_init",
    "loss_final",
    "loss_best",
    "best_step",
    "runtime_total_s",
    "has_checkpoint_best",
    "has_checkpoint_final",
    "has_signals",
    "has_precond",
    "has_curvature",
)


def _safe_load_meta(run_dir: Path) -> Mapping[str, Any]:
    try:
        return load_meta(run_dir)
    except FileNotFoundError:
        return {}


def _safe_load_summary(run_dir: Path) -> Mapping[str, Any]:
    try:
        return load_summary(run_dir)
    except FileNotFoundError:
        return {}


def _get_nested(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for key in path.split("."):
        if not isinstance(current, Mapping):
            return None
        if key not in current:
            return None
        current = current[key]
    return current


def collect_runs(runs_dir: Path) -> list[Path]:
    """Return run directories containing both ``summary.json`` and ``meta.json``.

    The search is shallow: the provided directory is checked directly, and its
    immediate children are considered run directories when they contain the
    required files. Non-directories are ignored.
    """

    runs_dir = Path(runs_dir)
    run_dirs: list[Path] = []

    def _maybe_add(path: Path) -> None:
        if (
            path.is_dir()
            and (path / "summary.json").exists()
            and (path / "meta.json").exists()
        ):
            run_dirs.append(path)

    _maybe_add(runs_dir)
    for child in runs_dir.iterdir():
        _maybe_add(child)

    return sorted(run_dirs)


def load_run_row(run_dir: Path) -> dict[str, Any]:
    """Load a flattened row of metadata/summary fields for a single run.

    Missing fields are represented as ``None`` so downstream consumers (e.g.,
    CSV writers) can remain robust to partial artifacts.
    """

    run_dir = Path(run_dir)
    meta = _safe_load_meta(run_dir)
    summary = _safe_load_summary(run_dir)

    precond_meta = _get_nested(meta, "optimizer.preconditioning") or {}

    row: dict[str, Any] = {
        "run_id": summary.get("run_id") or meta.get("run_id") or run_dir.name,
        "created_at": summary.get("created_at") or meta.get("created_at"),
        "git.commit": _get_nested(meta, "git.commit"),
        "status": summary.get("status"),
        "theta.dim": _get_nested(meta, "theta.dim"),
        "theta.theta_space": _get_nested(meta, "theta.theta_space"),
        "optimizer.name": _get_nested(meta, "optimizer.name"),
        "optimizer.num_steps": _get_nested(meta, "optimizer.num_steps"),
        "optimizer.base_lr": _get_nested(meta, "optimizer.base_lr"),
        "preconditioning.enabled": bool(precond_meta)
        if precond_meta is not None
        else False,
        "preconditioning.method": _get_nested(precond_meta, "method")
        if isinstance(precond_meta, Mapping)
        else None,
        "loss_init": summary.get("loss_init"),
        "loss_final": summary.get("loss_final"),
        "loss_best": summary.get("loss_best"),
        "best_step": summary.get("best_step"),
        "runtime_total_s": summary.get("runtime_total_s"),
        "has_checkpoint_best": summary.get("has_checkpoint_best"),
        "has_checkpoint_final": summary.get("has_checkpoint_final"),
        "has_signals": summary.get("has_signals"),
        "has_precond": summary.get("has_precond"),
        "has_curvature": summary.get("has_curvature"),
    }

    return row


def _extract_extra_meta(meta: Mapping[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key in keys:
        if key in extras:
            continue
        extras[key] = _get_nested(meta, key)
    return extras


def write_sweep_csv(
    runs_dir: Path,
    out_csv: Path,
    *,
    include_meta_fields: Sequence[str] | None = None,
) -> int:
    """Collect runs under ``runs_dir`` and write a summary CSV.

    Parameters
    ----------
    runs_dir : Path
        Directory containing run subdirectories.
    out_csv : Path
        Output CSV path. Parent directories are created if needed.
    include_meta_fields : Sequence[str] | None
        Optional dotted meta keys to append as extra columns.

    Returns
    -------
    int
        Number of run rows written.
    """

    runs_dir = Path(runs_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    run_dirs = collect_runs(runs_dir)
    extra_fields = list(include_meta_fields) if include_meta_fields is not None else []

    columns: list[str] = list(DEFAULT_COLUMNS)
    for key in extra_fields:
        if key not in columns:
            columns.append(key)

    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        meta = _safe_load_meta(run_dir)
        row = load_run_row(run_dir)
        row.update(_extract_extra_meta(meta, extra_fields))
        rows.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows)
