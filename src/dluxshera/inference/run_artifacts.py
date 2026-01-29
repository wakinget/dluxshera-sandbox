"""
Run artifact I/O utilities.

This module implements the Phase A scaffold for saving and loading per-run
artifacts as described in the optimization artifacts doc. It intentionally
keeps a small surface: trace/meta/summary are always written, while
additional diagnostics remain opt-in.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Literal, Mapping, Optional, TypedDict

import numpy as np

ArrayMapping = Mapping[str, object]


class ArtifactPayload(TypedDict, total=False):
    kind: Literal["npz", "json", "jsonl"]
    content: object
    filename: str
    description: str


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_arrays(mapping: ArrayMapping) -> dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in mapping.items()}


def _save_npz(path: Path, data: ArrayMapping) -> None:
    arrays = _normalize_arrays(data)
    np.savez_compressed(path, **arrays)


def _write_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _write_jsonl(path: Path, records: Iterable[object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            if isinstance(record, str):
                line = record
            else:
                line = json.dumps(record)
            f.write(line)
            f.write("\n")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {k: np.asarray(v) for k, v in npz.items()}


def _get_git_info(run_path: Path) -> dict[str, object] | None:
    try:
        commit = subprocess.run(
            ["git", "-C", str(run_path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    if commit.returncode != 0:
        return None

    commit_sha = commit.stdout.strip()
    if not commit_sha:
        return None

    dirty_result = subprocess.run(
        ["git", "-C", str(run_path), "diff", "--quiet"],
        check=False,
        capture_output=True,
        text=True,
    )
    dirty = dirty_result.returncode != 0

    return {"commit": commit_sha, "dirty": dirty}


def save_run(
    run_dir: Path | str,
    trace: ArrayMapping,
    meta: Mapping,
    summary: Mapping,
    *,
    artifacts: Optional[Mapping[str, ArtifactPayload]] = None,
) -> None:
    """
    Save optimizer run artifacts to ``run_dir``.

    Always writes ``trace.npz``, ``meta.json``, and ``summary.json``. Optional
    artifacts are only written when provided via the ``artifacts`` mapping.
    """
    # ---------------------------------------------------------------------
    # Artifact manifest + payload spec (Phase 0)
    #
    # Required artifacts (always written):
    #   - trace.npz   (kind="npz", name="trace")
    #   - meta.json   (kind="json", name="meta")
    #   - summary.json (kind="json", name="summary")
    #
    # Optional artifacts: provided explicitly by caller; only saved if present.
    #
    # Artifact kinds (minimum viable):
    #   - "npz"  : dense numeric arrays (dict-of-arrays)
    #   - "json" : JSON-serializable dicts/metadata
    #   - "jsonl": line-delimited logs (iterable of dicts/strings)
    #
    # Naming + filenames:
    #   - Artifact name defaults to filename stem (e.g., "signals").
    #   - Default filename is "{name}.{ext}" derived from kind.
    #   - Explicit filename override is allowed when needed
    #     (e.g., "checkpoint_best.npz").
    #
    # Manifest placement:
    #   - meta["manifest"]: mapping keyed by artifact name
    #     {name: {"name", "filename", "kind", "description"?}}
    #   - summary["manifest"]: compact mapping keyed by name
    #     {name: {"name", "filename", "kind"}}
    #
    # Overwrite behavior:
    #   - save_run may be called multiple times with the same run_dir.
    #     Overwriting meta.json/summary.json is acceptable when explicitly
    #     targeting a fixed run_dir. (runs_dir callers should use distinct
    #     subdirectories per run.)
    #
    # Git provenance (best-effort, written into meta.json):
    #   - meta["git"] = {"commit": "<sha>", "dirty": bool} when available.
    # ---------------------------------------------------------------------

    run_path = Path(run_dir)
    _ensure_dir(run_path)

    meta_out = dict(meta)
    summary_out = dict(summary)

    manifest_meta: dict[str, dict[str, object]] = {}
    manifest_summary: dict[str, dict[str, object]] = {}

    def register(name: str, filename: str, kind: str, description: Optional[str] = None) -> None:
        entry = {"name": name, "filename": filename, "kind": kind}
        meta_entry = dict(entry)
        if description:
            meta_entry["description"] = description
        manifest_meta[name] = meta_entry
        manifest_summary[name] = dict(entry)

    _save_npz(run_path / "trace.npz", trace)
    register("trace", "trace.npz", "npz")

    git_info = _get_git_info(run_path)
    if git_info is not None:
        meta_out["git"] = git_info

    register("meta", "meta.json", "json")
    register("summary", "summary.json", "json")

    if artifacts:
        for name, payload in artifacts.items():
            kind = payload.get("kind")
            if kind not in {"npz", "json", "jsonl"}:
                raise ValueError(f"Unknown artifact kind '{kind}' for '{name}'")
            filename = payload.get("filename") or f"{name}.{kind}"
            description = payload.get("description")
            target = run_path / filename
            content = payload.get("content")
            if kind == "npz":
                if not isinstance(content, Mapping):
                    raise TypeError(f"NPZ artifact '{name}' content must be a mapping")
                _save_npz(target, content)
            elif kind == "json":
                _write_json(target, content)
            else:
                if content is None:
                    raise TypeError(f"JSONL artifact '{name}' requires iterable content")
                _write_jsonl(target, content)
            register(name, filename, kind, description)

    meta_out["manifest"] = manifest_meta
    summary_out["manifest"] = manifest_summary

    _write_json(run_path / "meta.json", meta_out)
    _write_json(run_path / "summary.json", summary_out)


def load_trace(run_dir: Path | str) -> dict[str, np.ndarray]:
    return _load_npz(Path(run_dir) / "trace.npz")


def load_meta(run_dir: Path | str):
    with (Path(run_dir) / "meta.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_summary(run_dir: Path | str):
    with (Path(run_dir) / "summary.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_manifest(run_dir: Path | str) -> Mapping[str, Mapping[str, object]] | None:
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    if summary_path.exists():
        summary = load_summary(run_path)
        manifest = summary.get("manifest")
        if isinstance(manifest, Mapping):
            return manifest
    meta_path = run_path / "meta.json"
    if meta_path.exists():
        meta = load_meta(run_path)
        manifest = meta.get("manifest")
        if isinstance(manifest, Mapping):
            return manifest
    return None


def load_checkpoint(run_dir: Path | str, which: str = "best") -> dict[str, np.ndarray]:
    name = f"checkpoint_{which}"
    manifest = _load_manifest(run_dir)
    if manifest and name in manifest:
        entry = manifest[name]
        filename = entry.get("filename")
        kind = entry.get("kind")
        if kind != "npz":
            raise ValueError(f"Checkpoint artifact '{name}' has non-npz kind '{kind}'")
        if not filename:
            raise ValueError(f"Checkpoint artifact '{name}' is missing a filename")
        path = Path(run_dir) / str(filename)
        return _load_npz(path)
    path = Path(run_dir) / f"{name}.npz"
    return _load_npz(path)
